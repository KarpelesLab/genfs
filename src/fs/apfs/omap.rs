//! Object Map (`omap_phys_t`) — translates virtual object IDs into physical
//! block addresses.
//!
//! Reference: *Apple File System Reference*, section "Object Map".
//!
//! ```text
//! omap_phys_t (header for the omap itself)
//!    0   om_o                obj_phys_t (32 B)
//!   32   om_flags            u32
//!   36   om_snap_count       u32
//!   40   om_tree_type         u32
//!   44   om_snapshot_tree_type u32
//!   48   om_tree_oid         u64  ← physical block of the omap B-tree root
//!   56   om_snapshot_tree_oid u64
//!   64   om_most_recent_snap  u64
//!   72   om_pending_revert_min u64
//!   80   om_pending_revert_max u64
//! ```
//!
//! ```text
//! omap_key_t (16 B):
//!   0   ok_oid  u64
//!   8   ok_xid  u64
//! omap_val_t (16 B):
//!   0   ov_flags  u32
//!   4   ov_size   u32
//!   8   ov_paddr  u64
//! ```
//!
//! The omap is a fixed-KV B-tree (16/16). Lookup of `(oid, xid)` walks the
//! tree as usual; we pick the entry with the largest `xid <= target_xid`.

use super::btree::{BTreeNode, NodeCache};
use super::obj::{OBJECT_TYPE_OMAP, ObjPhys};

/// Decoded `omap_phys_t`.
#[derive(Debug, Clone)]
pub struct OmapPhys {
    pub obj: ObjPhys,
    pub flags: u32,
    pub snap_count: u32,
    pub tree_type: u32,
    pub tree_oid: u64,
}

impl OmapPhys {
    pub const MIN_SIZE: usize = 88;
    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < Self::MIN_SIZE {
            return Err(crate::Error::InvalidImage(
                "apfs: omap_phys buffer too short".into(),
            ));
        }
        let obj = ObjPhys::decode(buf)?;
        if obj.obj_type() != OBJECT_TYPE_OMAP {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: o_type {:#x} is not OMAP",
                obj.obj_type()
            )));
        }
        let flags = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        let snap_count = u32::from_le_bytes(buf[36..40].try_into().unwrap());
        let tree_type = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        let tree_oid = u64::from_le_bytes(buf[48..56].try_into().unwrap());
        Ok(Self {
            obj,
            flags,
            snap_count,
            tree_type,
            tree_oid,
        })
    }
}

/// A decoded omap key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OmapKey {
    pub oid: u64,
    pub xid: u64,
}

impl OmapKey {
    pub const SIZE: usize = 16;
    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(crate::Error::InvalidImage(
                "apfs: omap_key too short".into(),
            ));
        }
        Ok(Self {
            oid: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            xid: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        })
    }
}

/// A decoded omap value.
#[derive(Debug, Clone, Copy)]
pub struct OmapVal {
    pub flags: u32,
    pub size: u32,
    pub paddr: u64,
}

impl OmapVal {
    pub const SIZE: usize = 16;
    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(crate::Error::InvalidImage(
                "apfs: omap_val too short".into(),
            ));
        }
        Ok(Self {
            flags: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            size: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            paddr: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        })
    }
}

/// Compare two omap keys per the spec's ordering rule: ascending `oid`
/// first, then ascending `xid`.
fn cmp_omap_key(a: &OmapKey, b: &OmapKey) -> std::cmp::Ordering {
    a.oid.cmp(&b.oid).then(a.xid.cmp(&b.xid))
}

/// Search an omap B-tree of arbitrary depth for the entry matching
/// `target_oid` with the largest `xid <= target_xid`. Returns the
/// resolved `OmapVal` or `None` when no such entry exists.
///
/// `read_block` is supplied by the caller; given a physical block
/// number it must fill the passed buffer (length = block size) with
/// that block's contents. The function makes no assumption about the
/// underlying device.
///
/// Internal nodes are descended by binary-searching for the smallest
/// key that compares **strictly greater** than the target `(oid,
/// target_xid)`, then stepping back one entry — the standard "largest
/// key ≤ target" rule, except that here every internal-node separator
/// is the *smallest* key in its child subtree (APFS uses this
/// convention).
///
/// Multi-level walks consult `cache` to avoid re-fetching blocks.
/// Callers that don't want caching can pass `NodeCache::new(0)`.
pub fn lookup<F>(
    root_block: &[u8],
    target_oid: u64,
    target_xid: u64,
    read_block: &mut F,
) -> crate::Result<Option<OmapVal>>
where
    F: FnMut(u64, &mut [u8]) -> crate::Result<()>,
{
    let mut cache = NodeCache::new(NodeCache::DEFAULT_CAP);
    lookup_with_cache(root_block, target_oid, target_xid, read_block, &mut cache)
}

/// Same as [`lookup`] but with caller-supplied cache so repeated lookups
/// in the same omap reuse loaded internal nodes.
pub fn lookup_with_cache<F>(
    root_block: &[u8],
    target_oid: u64,
    target_xid: u64,
    read_block: &mut F,
    cache: &mut NodeCache,
) -> crate::Result<Option<OmapVal>>
where
    F: FnMut(u64, &mut [u8]) -> crate::Result<()>,
{
    let target = OmapKey {
        oid: target_oid,
        xid: target_xid,
    };
    let block_size = root_block.len();
    let root = BTreeNode::decode(root_block)?;
    let (klen, vlen) = root
        .fixed_kv_size()
        .ok_or_else(|| crate::Error::InvalidImage("apfs: omap root missing btree_info".into()))?;
    if klen != OmapKey::SIZE || vlen != OmapVal::SIZE {
        return Err(crate::Error::InvalidImage(format!(
            "apfs: omap fixed kv size ({klen}, {vlen}) != (16, 16)"
        )));
    }

    if root.is_leaf() {
        return scan_leaf_for_best(&root, klen, target_oid, target_xid);
    }

    // Descend by re-binding to an owned child block each iteration.
    // APFS B-tree levels strictly decrease from root to leaf, so we bound
    // the descent by the root's level (capped) and require each step to
    // strictly decrease `node.level`. This turns a corrupted/cyclic tree
    // from a malicious image into a clean error instead of an infinite loop.
    let max_depth = (root.level as usize).min(MAX_BTREE_DEPTH);
    let mut prev_level = root.level;
    let child_idx = find_child_for_key(&root, klen, &target)?;
    let (_, mut child_paddr) = root.child_entry_at(child_idx, klen)?;
    let mut cur = fetch_block(child_paddr, block_size, read_block, cache)?;
    for _ in 0..=max_depth {
        let node = BTreeNode::decode(&cur)?;
        if node.level >= prev_level {
            return Err(crate::Error::InvalidImage(
                "apfs: omap btree level did not strictly decrease on descent".into(),
            ));
        }
        prev_level = node.level;
        if node.is_leaf() {
            return scan_leaf_for_best(&node, klen, target_oid, target_xid);
        }
        let idx = find_child_for_key(&node, klen, &target)?;
        let (_, paddr) = node.child_entry_at(idx, klen)?;
        child_paddr = paddr;
        // Replace cur with the next child block (this drops the previous
        // `node` borrow since `cur` is rebound).
        cur = fetch_block(child_paddr, block_size, read_block, cache)?;
    }
    Err(crate::Error::InvalidImage(
        "apfs: omap btree descent exceeded maximum depth".into(),
    ))
}

/// Hard cap on B-tree descent depth. APFS levels are `u16`, but a valid
/// tree is never anywhere near this deep; the cap defends against a
/// malicious image whose root advertises an absurd `btn_level`.
const MAX_BTREE_DEPTH: usize = 64;

/// Locate the child of an internal omap node whose subtree may contain
/// `target`. Per the spec, internal keys are the *first* key in their
/// subtree, so we want the largest entry whose key ≤ target.
fn find_child_for_key(node: &BTreeNode<'_>, klen: usize, target: &OmapKey) -> crate::Result<u32> {
    if node.nkeys == 0 {
        return Err(crate::Error::InvalidImage(
            "apfs: empty omap internal node".into(),
        ));
    }
    // Binary search for the largest index whose key ≤ target.
    let mut lo: i64 = 0;
    let mut hi: i64 = node.nkeys as i64 - 1;
    let mut best: i64 = 0; // default: leftmost child
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let (kb, _) = node.child_entry_at(mid as u32, klen)?;
        let k = OmapKey::decode(kb)?;
        match cmp_omap_key(&k, target) {
            std::cmp::Ordering::Less | std::cmp::Ordering::Equal => {
                best = mid;
                lo = mid + 1;
            }
            std::cmp::Ordering::Greater => {
                hi = mid - 1;
            }
        }
    }
    Ok(best as u32)
}

/// Scan a leaf node for the largest-xid entry matching `target_oid`
/// with `xid <= target_xid`.
fn scan_leaf_for_best(
    node: &BTreeNode<'_>,
    klen: usize,
    target_oid: u64,
    target_xid: u64,
) -> crate::Result<Option<OmapVal>> {
    let vlen = OmapVal::SIZE;
    let mut best: Option<OmapVal> = None;
    let mut best_xid: u64 = 0;
    for i in 0..node.nkeys {
        let (kb, vb) = node.entry_at(i, klen, vlen)?;
        let k = OmapKey::decode(kb)?;
        if k.oid != target_oid {
            continue;
        }
        if k.xid > target_xid {
            continue;
        }
        if best.is_none() || k.xid > best_xid {
            best_xid = k.xid;
            best = Some(OmapVal::decode(vb)?);
        }
    }
    Ok(best)
}

fn fetch_block<F>(
    paddr: u64,
    block_size: usize,
    read_block: &mut F,
    cache: &mut NodeCache,
) -> crate::Result<Vec<u8>>
where
    F: FnMut(u64, &mut [u8]) -> crate::Result<()>,
{
    if let Some(b) = cache.get(paddr) {
        return Ok(b.to_vec());
    }
    let mut buf = vec![0u8; block_size];
    read_block(paddr, &mut buf)?;
    cache.put(paddr, buf.clone());
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::super::btree::{BTNODE_FIXED_KV_SIZE, BTNODE_LEAF, BTNODE_ROOT, BTREE_INFO_SIZE};
    use super::super::obj::OBJECT_TYPE_BTREE_NODE;
    use super::*;

    /// Build a leaf omap node with a list of `(oid, xid, paddr)` entries
    /// in the order given. Caller is responsible for keeping the list
    /// sorted by `(oid, xid)`.
    fn build_leaf_omap_node(
        block_size: usize,
        entries: &[(u64, u64, u64)],
        is_root: bool,
    ) -> Vec<u8> {
        let mut block = vec![0u8; block_size];
        block[24..28].copy_from_slice(&OBJECT_TYPE_BTREE_NODE.to_le_bytes());
        let mut flags = BTNODE_LEAF | BTNODE_FIXED_KV_SIZE;
        if is_root {
            flags |= BTNODE_ROOT;
        }
        block[32..34].copy_from_slice(&flags.to_le_bytes());
        block[34..36].copy_from_slice(&0u16.to_le_bytes()); // level
        block[36..40].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        let toc_len = entries.len() * 4;
        block[40..42].copy_from_slice(&0u16.to_le_bytes()); // toc_off
        block[42..44].copy_from_slice(&(toc_len as u16).to_le_bytes());

        let toc_base = 56;
        let keys_start = toc_base + toc_len;
        let vals_end = if is_root {
            block_size - BTREE_INFO_SIZE
        } else {
            block_size
        };

        for (i, &(oid, xid, paddr)) in entries.iter().enumerate() {
            // ToC: kvoff with k.off = i*16, v.off = (i+1)*16
            let k_off = (i * 16) as u16;
            let v_off = ((i + 1) * 16) as u16;
            block[toc_base + i * 4..toc_base + i * 4 + 2].copy_from_slice(&k_off.to_le_bytes());
            block[toc_base + i * 4 + 2..toc_base + i * 4 + 4].copy_from_slice(&v_off.to_le_bytes());

            // Key bytes at keys_start + k_off
            let ks = keys_start + k_off as usize;
            block[ks..ks + 8].copy_from_slice(&oid.to_le_bytes());
            block[ks + 8..ks + 16].copy_from_slice(&xid.to_le_bytes());

            // Value bytes at vals_end - v_off (16 B) — flags=0, size=0, paddr
            let vs = vals_end - v_off as usize;
            block[vs..vs + 4].copy_from_slice(&0u32.to_le_bytes());
            block[vs + 4..vs + 8].copy_from_slice(&0u32.to_le_bytes());
            block[vs + 8..vs + 16].copy_from_slice(&paddr.to_le_bytes());
        }

        if is_root {
            let info_off = block_size - BTREE_INFO_SIZE;
            block[info_off + 8..info_off + 12].copy_from_slice(&16u32.to_le_bytes());
            block[info_off + 12..info_off + 16].copy_from_slice(&16u32.to_le_bytes());
        }
        block
    }

    /// Build an internal omap node: keys are `(oid, xid)` separators
    /// and the values are 8-byte child paddrs (despite bt_val_size = 16
    /// for the leaves — internal nodes always store 8-byte child ptrs).
    fn build_internal_omap_node(
        block_size: usize,
        entries: &[(u64, u64, u64)],
        is_root: bool,
    ) -> Vec<u8> {
        let mut block = vec![0u8; block_size];
        block[24..28].copy_from_slice(&OBJECT_TYPE_BTREE_NODE.to_le_bytes());
        let mut flags = BTNODE_FIXED_KV_SIZE;
        if is_root {
            flags |= BTNODE_ROOT;
        }
        block[32..34].copy_from_slice(&flags.to_le_bytes());
        block[34..36].copy_from_slice(&1u16.to_le_bytes()); // level = 1
        block[36..40].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        let toc_len = entries.len() * 4;
        block[40..42].copy_from_slice(&0u16.to_le_bytes());
        block[42..44].copy_from_slice(&(toc_len as u16).to_le_bytes());

        let toc_base = 56;
        let keys_start = toc_base + toc_len;
        let vals_end = if is_root {
            block_size - BTREE_INFO_SIZE
        } else {
            block_size
        };

        for (i, &(oid, xid, child_paddr)) in entries.iter().enumerate() {
            let k_off = (i * 16) as u16;
            // Internal node values are 8 bytes — offsets must reflect
            // that. Use 8B per slot.
            let v_off = ((i + 1) * 8) as u16;
            block[toc_base + i * 4..toc_base + i * 4 + 2].copy_from_slice(&k_off.to_le_bytes());
            block[toc_base + i * 4 + 2..toc_base + i * 4 + 4].copy_from_slice(&v_off.to_le_bytes());

            let ks = keys_start + k_off as usize;
            block[ks..ks + 8].copy_from_slice(&oid.to_le_bytes());
            block[ks + 8..ks + 16].copy_from_slice(&xid.to_le_bytes());

            let vs = vals_end - v_off as usize;
            block[vs..vs + 8].copy_from_slice(&child_paddr.to_le_bytes());
        }

        if is_root {
            // bt_key_size = 16, bt_val_size = 16 (leaf payload size).
            let info_off = block_size - BTREE_INFO_SIZE;
            block[info_off + 8..info_off + 12].copy_from_slice(&16u32.to_le_bytes());
            block[info_off + 12..info_off + 16].copy_from_slice(&16u32.to_le_bytes());
        }
        block
    }

    /// Single-leaf omap: lookup picks the largest xid ≤ target.
    #[test]
    fn lookup_single_leaf_picks_latest_xid_below_target() {
        let block_size = 512usize;
        // Entries sorted (oid, xid): (5,1)→0x100, (5,3)→0x200, (5,7)→0x300
        let root = build_leaf_omap_node(
            block_size,
            &[(5, 1, 0x100), (5, 3, 0x200), (5, 7, 0x300)],
            true,
        );
        let mut no_read =
            |_p: u64, _b: &mut [u8]| panic!("single-leaf lookup must not read from device");
        let mut cache = NodeCache::new(NodeCache::DEFAULT_CAP);
        let v = lookup_with_cache(&root, 5, 5, &mut no_read, &mut cache).unwrap();
        assert_eq!(v.unwrap().paddr, 0x200);
        let v = lookup_with_cache(&root, 5, 10, &mut no_read, &mut cache).unwrap();
        assert_eq!(v.unwrap().paddr, 0x300);
        let v = lookup_with_cache(&root, 5, 0, &mut no_read, &mut cache).unwrap();
        assert!(v.is_none());
        let v = lookup_with_cache(&root, 99, 5, &mut no_read, &mut cache).unwrap();
        assert!(v.is_none());
    }

    /// Two-level omap: descend through an internal root to a leaf
    /// child. We register the leaf at paddr 10 in a fake device.
    #[test]
    fn lookup_two_level_descends_to_correct_leaf() {
        let block_size = 512usize;
        // Build two leaves:
        //  leaf_a (paddr 10) holds oid=5 entries
        //  leaf_b (paddr 11) holds oid=9 entries
        let leaf_a = build_leaf_omap_node(block_size, &[(5, 1, 0x100), (5, 3, 0x200)], false);
        let leaf_b = build_leaf_omap_node(block_size, &[(9, 2, 0x900)], false);

        // Build an internal root: first entry (5, 0) → child paddr 10,
        // second entry (9, 0) → child paddr 11.
        let root = build_internal_omap_node(block_size, &[(5, 0, 10), (9, 0, 11)], true);

        let mut device: std::collections::HashMap<u64, Vec<u8>> = Default::default();
        device.insert(10, leaf_a);
        device.insert(11, leaf_b);

        let mut read = |paddr: u64, buf: &mut [u8]| -> crate::Result<()> {
            let b = device
                .get(&paddr)
                .ok_or_else(|| crate::Error::InvalidImage(format!("no block at {paddr}")))?;
            buf.copy_from_slice(b);
            Ok(())
        };
        let mut cache = NodeCache::new(NodeCache::DEFAULT_CAP);

        let v = lookup_with_cache(&root, 5, 3, &mut read, &mut cache).unwrap();
        assert_eq!(v.unwrap().paddr, 0x200);

        let v = lookup_with_cache(&root, 9, 10, &mut read, &mut cache).unwrap();
        assert_eq!(v.unwrap().paddr, 0x900);

        // Missing oid: descends to the appropriate leaf, finds nothing.
        let v = lookup_with_cache(&root, 7, 10, &mut read, &mut cache).unwrap();
        assert!(v.is_none());
    }

    /// A malicious omap whose internal node points to a same-level
    /// (cyclic) child must yield a clean `InvalidImage` error rather than
    /// looping forever. The root is level 1; its child at paddr 10 is
    /// another level-1 internal node, so the strictly-decreasing guard
    /// fires immediately.
    #[test]
    fn lookup_rejects_non_decreasing_levels() {
        let block_size = 512usize;
        // Non-root internal node (level 1) whose own child points to
        // paddr 10 — i.e. back to itself.
        let cyclic_child = build_internal_omap_node(block_size, &[(5, 0, 10)], false);

        // Internal root (level 1) whose only child is paddr 10.
        let root = build_internal_omap_node(block_size, &[(5, 0, 10)], true);

        let mut device: std::collections::HashMap<u64, Vec<u8>> = Default::default();
        device.insert(10, cyclic_child);

        let mut read = |paddr: u64, buf: &mut [u8]| -> crate::Result<()> {
            let b = device
                .get(&paddr)
                .ok_or_else(|| crate::Error::InvalidImage(format!("no block at {paddr}")))?;
            buf.copy_from_slice(b);
            Ok(())
        };
        let mut cache = NodeCache::new(NodeCache::DEFAULT_CAP);

        let err = lookup_with_cache(&root, 5, 5, &mut read, &mut cache);
        assert!(matches!(err, Err(crate::Error::InvalidImage(_))));
    }

    #[test]
    fn decode_omap_phys() {
        let mut buf = vec![0u8; 4096];
        buf[24..28].copy_from_slice(&OBJECT_TYPE_OMAP.to_le_bytes());
        buf[32..36].copy_from_slice(&1u32.to_le_bytes()); // flags
        buf[40..44].copy_from_slice(&0xeu32.to_le_bytes()); // tree_type
        buf[48..56].copy_from_slice(&0x1234u64.to_le_bytes()); // tree_oid
        let o = OmapPhys::decode(&buf).unwrap();
        assert_eq!(o.tree_oid, 0x1234);
        assert_eq!(o.flags, 1);
    }

    #[test]
    fn omap_key_val_decode() {
        let mut kb = [0u8; 16];
        kb[0..8].copy_from_slice(&7u64.to_le_bytes());
        kb[8..16].copy_from_slice(&9u64.to_le_bytes());
        let k = OmapKey::decode(&kb).unwrap();
        assert_eq!(k.oid, 7);
        assert_eq!(k.xid, 9);

        let mut vb = [0u8; 16];
        vb[0..4].copy_from_slice(&2u32.to_le_bytes());
        vb[4..8].copy_from_slice(&1u32.to_le_bytes());
        vb[8..16].copy_from_slice(&0x4000u64.to_le_bytes());
        let v = OmapVal::decode(&vb).unwrap();
        assert_eq!(v.flags, 2);
        assert_eq!(v.size, 1);
        assert_eq!(v.paddr, 0x4000);
    }
}
