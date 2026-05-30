//! Multi-level walker / range scanner for the per-volume fs-tree.
//!
//! Reference: *Apple File System Reference*, sections "Filesystem
//! Objects" and "B-Trees".
//!
//! The fs-tree is a *virtual* B-tree: its root and child links are
//! virtual object identifiers (`oid_t`) that must be resolved through
//! the volume's object map (omap) before they can be loaded from disk.
//!
//! All fs-tree records share a `j_key_t` prefix:
//!
//! ```text
//! j_key_t (8 bytes)
//!   0  obj_id_and_type  u64   high 4 bits = record kind, low 60 = oid
//! ```
//!
//! The tree is ordered, ascending, by:
//!
//! 1. The lower 60 bits of `obj_id_and_type` (the object id).
//! 2. The high 4 bits (the record type code).
//! 3. A type-specific tail of the key. For our purposes:
//!    - `APFS_TYPE_INODE` and `APFS_TYPE_DSTREAM_ID`: no tail.
//!    - `APFS_TYPE_DIR_REC` (hashed layout): the next 4 bytes encode
//!      `(name_hash << 10) | name_length`; we sort by the *hash* first
//!      (per spec) then by name bytes. The plain layout has no hash;
//!      we fall back to lexicographic name compare for it.
//!    - `APFS_TYPE_FILE_EXTENT`: the next 8 bytes are `logical_addr`.
//!    - `APFS_TYPE_XATTR`: the next 2 bytes are name_length, then the
//!      name bytes; we sort lexicographically by name.

use std::cmp::Ordering;

use super::btree::{BTreeNode, NodeCache};
use super::jrec::{
    APFS_TYPE_DIR_REC, APFS_TYPE_FILE_EXTENT, APFS_TYPE_XATTR, J_DREC_LEN_MASK, split_obj_id,
};
use super::omap::lookup_with_cache as omap_lookup;

/// Layout of the directory-record key. The fs-tree only ever stores one
/// flavour per volume; the choice depends on the volume's
/// `APFS_INCOMPAT_NORMALIZATION_INSENSITIVE` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrecKeyLayout {
    /// `j_drec_hashed_key_t` — used when the volume sets
    /// `APFS_INCOMPAT_NORMALIZATION_INSENSITIVE` (the common case on
    /// modern macOS / iOS volumes).
    Hashed,
    /// `j_drec_key_t` — used on case-sensitive volumes without
    /// normalization-insensitive lookup.
    Plain,
}

/// A parsed fs-tree key, retaining enough state to compare it against a
/// target according to the APFS ordering rules.
#[derive(Debug, Clone)]
pub struct FsKey<'a> {
    pub oid: u64,
    pub kind: u8,
    /// Raw key bytes (including the 8-byte j_key_t header). We keep
    /// this for type-specific tail comparison without re-parsing.
    pub raw: &'a [u8],
}

impl<'a> FsKey<'a> {
    pub fn decode(raw: &'a [u8]) -> crate::Result<Self> {
        if raw.len() < 8 {
            return Err(crate::Error::InvalidImage(
                "apfs: fs-tree key shorter than j_key_t".into(),
            ));
        }
        let hdr = u64::from_le_bytes(raw[0..8].try_into().unwrap());
        let (kind, oid) = split_obj_id(hdr);
        Ok(Self { oid, kind, raw })
    }
}

/// A "target" describing what we're searching for. For point lookups
/// this carries enough type-specific data to disambiguate; for range
/// scans we only need the `(oid, kind)` prefix and supply `tail = &[]`.
#[derive(Debug, Clone, Copy)]
pub struct FsKeyTarget<'a> {
    pub oid: u64,
    pub kind: u8,
    /// Type-specific tail. Caller must lay this out exactly as APFS
    /// stores it after the j_key_t header.
    pub tail: &'a [u8],
    /// Drec layout in use for the volume; consulted when `kind ==
    /// APFS_TYPE_DIR_REC`.
    pub drec_layout: DrecKeyLayout,
}

/// Compare a stored fs-tree key against a target. Returns the standard
/// `Ordering` for `key cmp target`.
pub fn cmp_fs_key(key: &FsKey<'_>, target: &FsKeyTarget<'_>) -> Ordering {
    match key.oid.cmp(&target.oid) {
        Ordering::Equal => {}
        other => return other,
    }
    match key.kind.cmp(&target.kind) {
        Ordering::Equal => {}
        other => return other,
    }
    cmp_tail(key, target)
}

fn cmp_tail(key: &FsKey<'_>, target: &FsKeyTarget<'_>) -> Ordering {
    let kt = if key.raw.len() > 8 {
        &key.raw[8..]
    } else {
        &[][..]
    };
    match key.kind {
        APFS_TYPE_DIR_REC => match target.drec_layout {
            DrecKeyLayout::Hashed => cmp_drec_hashed_tail(kt, target.tail),
            DrecKeyLayout::Plain => cmp_drec_plain_tail(kt, target.tail),
        },
        APFS_TYPE_FILE_EXTENT => cmp_u64_le_tail(kt, target.tail),
        APFS_TYPE_XATTR => cmp_xattr_tail(kt, target.tail),
        _ => kt.cmp(target.tail),
    }
}

/// Drec hashed-key tail compare. The stored bytes are
/// `[u32 name_len_and_hash][u8 name[len]...]`. APFS sorts hashed drecs
/// by the hash field, then by name. When the target tail is empty
/// (range-scan start), every stored key sorts strictly greater.
fn cmp_drec_hashed_tail(stored: &[u8], target: &[u8]) -> Ordering {
    if target.is_empty() {
        // Real drec records always carry a non-empty tail; treat an
        // empty stored tail as Equal (only happens in synthetic test
        // inputs and end-of-prefix scan terminators).
        return if stored.is_empty() {
            Ordering::Equal
        } else {
            Ordering::Greater
        };
    }
    if stored.len() < 4 || target.len() < 4 {
        return stored.cmp(target);
    }
    let s_nh = u32::from_le_bytes(stored[0..4].try_into().unwrap());
    let t_nh = u32::from_le_bytes(target[0..4].try_into().unwrap());
    let s_hash = s_nh & !J_DREC_LEN_MASK;
    let t_hash = t_nh & !J_DREC_LEN_MASK;
    match s_hash.cmp(&t_hash) {
        Ordering::Equal => {}
        o => return o,
    }
    let s_len = (s_nh & J_DREC_LEN_MASK) as usize;
    let t_len = (t_nh & J_DREC_LEN_MASK) as usize;
    let s_name = &stored[4..4 + s_len.min(stored.len().saturating_sub(4))];
    let t_name = &target[4..4 + t_len.min(target.len().saturating_sub(4))];
    s_name.cmp(t_name)
}

/// Drec plain-key tail compare: `[u16 name_len][u8 name[len]]`.
fn cmp_drec_plain_tail(stored: &[u8], target: &[u8]) -> Ordering {
    if target.is_empty() {
        return if stored.is_empty() {
            Ordering::Equal
        } else {
            Ordering::Greater
        };
    }
    if stored.len() < 2 || target.len() < 2 {
        return stored.cmp(target);
    }
    let s_len = u16::from_le_bytes(stored[0..2].try_into().unwrap()) as usize;
    let t_len = u16::from_le_bytes(target[0..2].try_into().unwrap()) as usize;
    let s_name = &stored[2..2 + s_len.min(stored.len().saturating_sub(2))];
    let t_name = &target[2..2 + t_len.min(target.len().saturating_sub(2))];
    s_name.cmp(t_name)
}

fn cmp_u64_le_tail(stored: &[u8], target: &[u8]) -> Ordering {
    if target.is_empty() {
        return if stored.is_empty() {
            Ordering::Equal
        } else {
            Ordering::Greater
        };
    }
    if stored.len() < 8 || target.len() < 8 {
        return stored.cmp(target);
    }
    let s = u64::from_le_bytes(stored[0..8].try_into().unwrap());
    let t = u64::from_le_bytes(target[0..8].try_into().unwrap());
    s.cmp(&t)
}

fn cmp_xattr_tail(stored: &[u8], target: &[u8]) -> Ordering {
    if target.is_empty() {
        return if stored.is_empty() {
            Ordering::Equal
        } else {
            Ordering::Greater
        };
    }
    if stored.len() < 2 || target.len() < 2 {
        return stored.cmp(target);
    }
    let s_len = u16::from_le_bytes(stored[0..2].try_into().unwrap()) as usize;
    let t_len = u16::from_le_bytes(target[0..2].try_into().unwrap()) as usize;
    let s_name = &stored[2..2 + s_len.min(stored.len().saturating_sub(2))];
    let t_name = &target[2..2 + t_len.min(target.len().saturating_sub(2))];
    s_name.cmp(t_name)
}

/// All state needed to descend the fs-tree: the volume omap root block,
/// the target xid for omap lookups, and a pair of small LRU caches
/// (one for omap nodes, one for fs-tree nodes). The caller owns the
/// underlying `BlockDevice` and supplies a closure that reads physical
/// blocks.
///
/// We deliberately store the omap_root bytes inline so the context is
/// self-contained — no lifetime gymnastics for callers, and the cost
/// is one block of memory per open volume which is trivial.
pub struct FsTreeCtx {
    /// Volume omap root block, kept around because every internal-node
    /// descent re-walks it to resolve a child's virtual oid → paddr.
    pub omap_root: Vec<u8>,
    /// XID used for omap lookups (typically the volume superblock's xid).
    pub target_xid: u64,
    /// Block size in bytes.
    pub block_size: usize,
    /// LRU cache for omap internal nodes (multi-level omap walking).
    pub omap_cache: NodeCache,
    /// LRU cache for fs-tree nodes encountered during a walk / scan.
    pub fs_cache: NodeCache,
}

impl FsTreeCtx {
    pub fn new(omap_root: Vec<u8>, target_xid: u64, block_size: usize) -> Self {
        Self {
            omap_root,
            target_xid,
            block_size,
            omap_cache: NodeCache::new(NodeCache::DEFAULT_CAP),
            fs_cache: NodeCache::new(NodeCache::DEFAULT_CAP),
        }
    }

    /// Resolve a virtual oid to a physical block address by walking the
    /// volume omap.
    pub(super) fn resolve_vid<F>(&mut self, vid: u64, read_block: &mut F) -> crate::Result<u64>
    where
        F: FnMut(u64, &mut [u8]) -> crate::Result<()>,
    {
        let val = omap_lookup(
            &self.omap_root,
            vid,
            self.target_xid,
            read_block,
            &mut self.omap_cache,
        )?;
        let v = v_or_invalid(val, vid)?;
        Ok(v.paddr)
    }

    /// Load an fs-tree node (by physical address) into a fresh owned
    /// `Vec`, going through the LRU cache.
    fn fetch_fs_block<F>(&mut self, paddr: u64, read_block: &mut F) -> crate::Result<Vec<u8>>
    where
        F: FnMut(u64, &mut [u8]) -> crate::Result<()>,
    {
        if let Some(b) = self.fs_cache.get(paddr) {
            return Ok(b.to_vec());
        }
        let mut buf = vec![0u8; self.block_size];
        read_block(paddr, &mut buf)?;
        self.fs_cache.put(paddr, buf.clone());
        Ok(buf)
    }
}

fn v_or_invalid(
    val: Option<super::omap::OmapVal>,
    vid: u64,
) -> crate::Result<super::omap::OmapVal> {
    val.ok_or_else(|| {
        crate::Error::InvalidImage(format!("apfs: volume omap has no entry for vid {vid:#x}"))
    })
}

/// Look up a single record in the fs-tree by exact `(oid, kind, tail)`
/// match. Returns the value bytes on hit, `None` when not present.
///
/// The root block is the *physical* block containing the fs-tree root;
/// internal-node children are virtual oids resolved through the omap.
pub fn lookup<F>(
    fsroot_block: &[u8],
    target: &FsKeyTarget<'_>,
    ctx: &mut FsTreeCtx,
    read_block: &mut F,
) -> crate::Result<Option<Vec<u8>>>
where
    F: FnMut(u64, &mut [u8]) -> crate::Result<()>,
{
    let (fixed_klen, _fixed_kv) = read_root_kv_sizes(fsroot_block);
    let mut cur: Vec<u8> = fsroot_block.to_vec();
    // APFS B-tree levels strictly decrease root→leaf. Bound the descent so a
    // malicious image with a cyclic/corrupted fs-tree errors out instead of
    // looping forever.
    let mut prev_level: Option<u16> = None;
    for _ in 0..=MAX_BTREE_DEPTH {
        let node = BTreeNode::decode(&cur)?;
        if let Some(pl) = prev_level
            && node.level >= pl
        {
            return Err(crate::Error::InvalidImage(
                "apfs: fs-tree btree level did not strictly decrease on descent".into(),
            ));
        }
        prev_level = Some(node.level);
        if node.is_leaf() {
            return scan_leaf_for_exact(&node, fixed_klen, target);
        }
        let idx = find_child_idx(&node, fixed_klen, target)?;
        let (_, child_vid) = node.child_entry_at(idx, fixed_klen)?;
        let paddr = ctx.resolve_vid(child_vid, read_block)?;
        cur = ctx.fetch_fs_block(paddr, read_block)?;
    }
    Err(crate::Error::InvalidImage(
        "apfs: fs-tree btree descent exceeded maximum depth".into(),
    ))
}

/// Hard cap on B-tree descent depth, mirroring the omap guard. A valid
/// fs-tree is never anywhere near this deep; the cap defends against a
/// malicious image whose nodes never reach a leaf.
const MAX_BTREE_DEPTH: usize = 64;

/// A range-scan iterator over fs-tree leaf records whose key shares an
/// `(oid, kind)` prefix with the start target. Owns its descent stack
/// and node caches via the supplied `FsTreeCtx`.
///
/// Callers must repeatedly invoke [`RangeScan::next`], passing the
/// device read closure each time. The closure is taken as a parameter
/// per-call so it doesn't need to outlive the scan struct.
pub struct RangeScan {
    /// Stack of (block_bytes, current-cursor-into-node). For the leaf
    /// (top of stack) the cursor is the next *entry index* to yield;
    /// for internal nodes it's the index of the child we descended
    /// into last (we bump and re-descend on pop).
    stack: Vec<(Vec<u8>, u32)>,
    fixed_klen: usize,
    /// Stop predicate: the `(oid, kind)` to keep matching.
    stop_oid: u64,
    stop_kind: u8,
}

impl RangeScan {
    /// Build a scan positioned at the first key ≥ `start` matching its
    /// `(oid, kind)` prefix.
    pub fn start<F>(
        fsroot_block: &[u8],
        start: &FsKeyTarget<'_>,
        ctx: &mut FsTreeCtx,
        read_block: &mut F,
    ) -> crate::Result<Self>
    where
        F: FnMut(u64, &mut [u8]) -> crate::Result<()>,
    {
        let (fixed_klen, _) = read_root_kv_sizes(fsroot_block);
        let mut stack: Vec<(Vec<u8>, u32)> = Vec::new();
        let mut cur = fsroot_block.to_vec();
        // Bound the initial descent: levels must strictly decrease and the
        // depth is capped, so a malicious cyclic tree errors instead of
        // looping / growing the stack without bound.
        let mut prev_level: Option<u16> = None;
        loop {
            if stack.len() > MAX_BTREE_DEPTH {
                return Err(crate::Error::InvalidImage(
                    "apfs: fs-tree range scan descent exceeded maximum depth".into(),
                ));
            }
            let node = BTreeNode::decode(&cur)?;
            if let Some(pl) = prev_level
                && node.level >= pl
            {
                return Err(crate::Error::InvalidImage(
                    "apfs: fs-tree btree level did not strictly decrease on descent".into(),
                ));
            }
            prev_level = Some(node.level);
            if node.is_leaf() {
                // Find first key ≥ start.
                let mut first = node.nkeys;
                for i in 0..node.nkeys {
                    let (kb, _) = node.entry_at(i, fixed_klen, 0)?;
                    let key = FsKey::decode(kb)?;
                    if cmp_fs_key(&key, start) != Ordering::Less {
                        first = i;
                        break;
                    }
                }
                stack.push((cur, first));
                break;
            }
            let idx = find_child_idx(&node, fixed_klen, start)?;
            let (_, child_vid) = node.child_entry_at(idx, fixed_klen)?;
            stack.push((cur, idx));
            let paddr = ctx.resolve_vid(child_vid, read_block)?;
            cur = ctx.fetch_fs_block(paddr, read_block)?;
        }
        Ok(Self {
            stack,
            fixed_klen,
            stop_oid: start.oid,
            stop_kind: start.kind,
        })
    }

    /// Yield the next matching `(key_bytes, value_bytes)` pair, or
    /// `None` when the prefix stops matching or the tree is exhausted.
    pub fn next<F>(
        &mut self,
        ctx: &mut FsTreeCtx,
        read_block: &mut F,
    ) -> crate::Result<Option<(Vec<u8>, Vec<u8>)>>
    where
        F: FnMut(u64, &mut [u8]) -> crate::Result<()>,
    {
        loop {
            if self.stack.is_empty() {
                return Ok(None);
            }
            let leaf_idx = self.stack.len() - 1;
            // Pull values out without holding a borrow on the stack
            // across the `entry_at` call.
            let (leaf_bytes, cursor) = {
                let (lb, c) = &self.stack[leaf_idx];
                (lb.clone(), *c)
            };
            let node = BTreeNode::decode(&leaf_bytes)?;
            if !node.is_leaf() {
                return Err(crate::Error::InvalidImage(
                    "apfs: range scan: top of stack is not a leaf".into(),
                ));
            }
            if cursor < node.nkeys {
                let (kb, vb) = node.entry_at(cursor, self.fixed_klen, 0)?;
                let key = FsKey::decode(kb)?;
                if key.oid != self.stop_oid || key.kind != self.stop_kind {
                    return Ok(None);
                }
                let kb_owned = kb.to_vec();
                let vb_owned = vb.to_vec();
                self.stack[leaf_idx].1 += 1;
                return Ok(Some((kb_owned, vb_owned)));
            }
            // Leaf exhausted — pop and walk back up.
            self.stack.pop();
            if self.stack.is_empty() {
                return Ok(None);
            }
            // Re-descend: bump the parent's child index and walk down
            // again until the next leaf.
            self.descend_next(ctx, read_block)?;
            if self.stack.is_empty() {
                return Ok(None);
            }
        }
    }

    /// After popping a leaf, advance the topmost parent's cursor and
    /// descend through fresh children until we land on a new leaf. If
    /// all ancestors are exhausted, the stack ends empty.
    fn descend_next<F>(&mut self, ctx: &mut FsTreeCtx, read_block: &mut F) -> crate::Result<()>
    where
        F: FnMut(u64, &mut [u8]) -> crate::Result<()>,
    {
        loop {
            let parent_idx = self.stack.len() - 1;
            let (parent_bytes, cursor) = {
                let (pb, c) = &self.stack[parent_idx];
                (pb.clone(), *c)
            };
            let parent = BTreeNode::decode(&parent_bytes)?;
            // The cursor on the parent was the *previously* descended
            // child. Bump it.
            let next_idx = cursor + 1;
            if next_idx >= parent.nkeys {
                // This parent is also done; pop it.
                self.stack.pop();
                if self.stack.is_empty() {
                    return Ok(());
                }
                continue;
            }
            // Update parent's cursor in place.
            self.stack[parent_idx].1 = next_idx;
            let (_, child_vid) = parent.child_entry_at(next_idx, self.fixed_klen)?;
            let paddr = ctx.resolve_vid(child_vid, read_block)?;
            let mut child = ctx.fetch_fs_block(paddr, read_block)?;
            // Push the child and keep descending if it's still internal.
            // Bound the descent: levels must strictly decrease below the
            // parent and the stack depth is capped, so a malicious cyclic
            // tree errors instead of looping / growing the stack forever.
            let mut prev_level = parent.level;
            loop {
                if self.stack.len() > MAX_BTREE_DEPTH {
                    return Err(crate::Error::InvalidImage(
                        "apfs: fs-tree range scan descent exceeded maximum depth".into(),
                    ));
                }
                let cn = BTreeNode::decode(&child)?;
                if cn.level >= prev_level {
                    return Err(crate::Error::InvalidImage(
                        "apfs: fs-tree btree level did not strictly decrease on descent".into(),
                    ));
                }
                prev_level = cn.level;
                if cn.is_leaf() {
                    self.stack.push((child, 0));
                    return Ok(());
                }
                // First child of this internal node.
                let (_, vid) = cn.child_entry_at(0, self.fixed_klen)?;
                self.stack.push((child, 0));
                let p = ctx.resolve_vid(vid, read_block)?;
                child = ctx.fetch_fs_block(p, read_block)?;
            }
        }
    }
}

fn read_root_kv_sizes(root_block: &[u8]) -> (usize, bool) {
    if let Ok(root) = BTreeNode::decode(root_block) {
        let fixed_kv = root.fixed_kv;
        let klen = root.fixed_kv_size().map(|(k, _)| k).unwrap_or(0);
        (klen, fixed_kv)
    } else {
        (0, false)
    }
}

fn find_child_idx(
    node: &BTreeNode<'_>,
    fixed_klen: usize,
    target: &FsKeyTarget<'_>,
) -> crate::Result<u32> {
    if node.nkeys == 0 {
        return Err(crate::Error::InvalidImage(
            "apfs: empty fs-tree internal node".into(),
        ));
    }
    let mut lo: i64 = 0;
    let mut hi: i64 = node.nkeys as i64 - 1;
    let mut best: i64 = 0;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let (kb, _) = node.child_entry_at(mid as u32, fixed_klen)?;
        let key = FsKey::decode(kb)?;
        match cmp_fs_key(&key, target) {
            Ordering::Less | Ordering::Equal => {
                best = mid;
                lo = mid + 1;
            }
            Ordering::Greater => {
                hi = mid - 1;
            }
        }
    }
    Ok(best as u32)
}

fn scan_leaf_for_exact(
    node: &BTreeNode<'_>,
    fixed_klen: usize,
    target: &FsKeyTarget<'_>,
) -> crate::Result<Option<Vec<u8>>> {
    for i in 0..node.nkeys {
        let (kb, vb) = node.entry_at(i, fixed_klen, 0)?;
        let key = FsKey::decode(kb)?;
        if cmp_fs_key(&key, target) == Ordering::Equal {
            return Ok(Some(vb.to_vec()));
        }
    }
    Ok(None)
}

/// Build the type-specific tail bytes for a drec target. We can't
/// compute the real APFS hash here (the algorithm is normalization-
/// aware and the spec doesn't fully specify it for arbitrary input);
/// the comparator below falls back to lexicographic name compare
/// inside a hash bucket, so the resulting target is only safe to use
/// for **range scans** that iterate by prefix and filter in caller
/// code by exact name. Direct point lookup against a hashed drec is
/// not supported by this helper.
pub fn make_drec_target_tail(name: &str, layout: DrecKeyLayout) -> Vec<u8> {
    match layout {
        DrecKeyLayout::Plain => {
            let bytes = name.as_bytes();
            let mut out = Vec::with_capacity(2 + bytes.len() + 1);
            let nlen = (bytes.len() + 1) as u16;
            out.extend_from_slice(&nlen.to_le_bytes());
            out.extend_from_slice(bytes);
            out.push(0);
            out
        }
        DrecKeyLayout::Hashed => {
            let bytes = name.as_bytes();
            let nlen = ((bytes.len() + 1) as u32) & J_DREC_LEN_MASK;
            let mut out = Vec::with_capacity(4 + bytes.len() + 1);
            out.extend_from_slice(&nlen.to_le_bytes());
            out.extend_from_slice(bytes);
            out.push(0);
            out
        }
    }
}

/// Tail for a file-extent target with the given `logical_addr`.
pub fn make_file_extent_tail(logical_addr: u64) -> [u8; 8] {
    logical_addr.to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::super::jrec::{OBJ_ID_MASK, OBJ_TYPE_SHIFT};
    use super::*;

    #[test]
    fn cmp_fs_key_orders_by_oid_first() {
        let mut raw1 = [0u8; 8];
        let mut raw2 = [0u8; 8];
        let h1 = ((APFS_TYPE_DIR_REC as u64) << OBJ_TYPE_SHIFT) | (5 & OBJ_ID_MASK);
        let h2 = ((APFS_TYPE_DIR_REC as u64) << OBJ_TYPE_SHIFT) | (6 & OBJ_ID_MASK);
        raw1.copy_from_slice(&h1.to_le_bytes());
        raw2.copy_from_slice(&h2.to_le_bytes());
        let k1 = FsKey::decode(&raw1).unwrap();
        let k2 = FsKey::decode(&raw2).unwrap();
        let t = FsKeyTarget {
            oid: 6,
            kind: APFS_TYPE_DIR_REC,
            tail: &[],
            drec_layout: DrecKeyLayout::Hashed,
        };
        assert!(cmp_fs_key(&k1, &t) == Ordering::Less);
        // k2 has same oid+kind+empty tail (with stored kt empty too)
        // since raw2 is exactly 8 bytes — no tail. cmp_drec_hashed_tail
        // sees stored empty and target empty → falls into `stored.cmp(target)`
        // which is Equal, so cmp_fs_key returns Equal.
        assert_eq!(cmp_fs_key(&k2, &t), Ordering::Equal);
    }

    #[test]
    fn cmp_fs_key_kind_breaks_oid_tie() {
        let mut raw1 = [0u8; 8];
        let mut raw2 = [0u8; 8];
        let h1 = ((APFS_TYPE_DIR_REC as u64) << OBJ_TYPE_SHIFT) | 5;
        let h2 = ((APFS_TYPE_FILE_EXTENT as u64) << OBJ_TYPE_SHIFT) | 5;
        raw1.copy_from_slice(&h1.to_le_bytes());
        raw2.copy_from_slice(&h2.to_le_bytes());
        let k1 = FsKey::decode(&raw1).unwrap();
        let k2 = FsKey::decode(&raw2).unwrap();
        let t = FsKeyTarget {
            oid: 5,
            kind: APFS_TYPE_DIR_REC,
            tail: &[],
            drec_layout: DrecKeyLayout::Hashed,
        };
        // DIR_REC = 9 stored against a DIR_REC target → tail-compare path
        // (both empty → Equal here).
        assert_eq!(cmp_fs_key(&k1, &t), Ordering::Equal);
        // FILE_EXTENT = 8 < DIR_REC = 9 in the kind tier.
        assert_eq!(cmp_fs_key(&k2, &t), Ordering::Less);
    }

    #[test]
    fn drec_hashed_target_tail_layout() {
        let t = make_drec_target_tail("foo", DrecKeyLayout::Hashed);
        // 4 bytes header + 3 name + 1 NUL = 8
        assert_eq!(t.len(), 8);
        let nh = u32::from_le_bytes(t[0..4].try_into().unwrap());
        assert_eq!(nh & J_DREC_LEN_MASK, 4); // "foo\0" = 4 bytes
    }

    #[test]
    fn file_extent_tail_is_le_u64() {
        let t = make_file_extent_tail(0x1234_5678_9abc_def0);
        assert_eq!(u64::from_le_bytes(t), 0x1234_5678_9abc_def0);
    }

    // ---- multi-level walker integration tests ----

    use super::super::btree::{BTNODE_FIXED_KV_SIZE, BTNODE_LEAF, BTNODE_ROOT, BTREE_INFO_SIZE};
    use super::super::jrec::{APFS_TYPE_DIR_REC, APFS_TYPE_INODE};
    use super::super::obj::OBJECT_TYPE_BTREE_NODE;

    /// Build a variable-KV leaf fs-tree node holding `(key_bytes,
    /// val_bytes)` entries in the order given. The kv lengths come from
    /// the ToC, so the entries can be heterogeneous.
    fn build_var_leaf(block_size: usize, entries: &[(Vec<u8>, Vec<u8>)], is_root: bool) -> Vec<u8> {
        let mut block = vec![0u8; block_size];
        block[24..28].copy_from_slice(&OBJECT_TYPE_BTREE_NODE.to_le_bytes());
        let mut flags = BTNODE_LEAF;
        if is_root {
            flags |= BTNODE_ROOT;
        }
        block[32..34].copy_from_slice(&flags.to_le_bytes());
        block[34..36].copy_from_slice(&0u16.to_le_bytes()); // level
        block[36..40].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        let toc_len = entries.len() * 8;
        block[40..42].copy_from_slice(&0u16.to_le_bytes());
        block[42..44].copy_from_slice(&(toc_len as u16).to_le_bytes());

        let toc_base = 56;
        let keys_start = toc_base + toc_len;
        let vals_end = if is_root {
            block_size - BTREE_INFO_SIZE
        } else {
            block_size
        };

        // Pack keys ascending forward, values descending backward.
        let mut k_cursor = 0usize;
        let mut v_cursor_back = 0usize;
        for (i, (kb, vb)) in entries.iter().enumerate() {
            let k_off = k_cursor as u16;
            let k_len = kb.len() as u16;
            v_cursor_back += vb.len();
            let v_off = v_cursor_back as u16;
            let v_len = vb.len() as u16;
            block[toc_base + i * 8..toc_base + i * 8 + 2].copy_from_slice(&k_off.to_le_bytes());
            block[toc_base + i * 8 + 2..toc_base + i * 8 + 4].copy_from_slice(&k_len.to_le_bytes());
            block[toc_base + i * 8 + 4..toc_base + i * 8 + 6].copy_from_slice(&v_off.to_le_bytes());
            block[toc_base + i * 8 + 6..toc_base + i * 8 + 8].copy_from_slice(&v_len.to_le_bytes());

            let ks = keys_start + k_off as usize;
            block[ks..ks + kb.len()].copy_from_slice(kb);
            let vs = vals_end - v_off as usize;
            block[vs..vs + vb.len()].copy_from_slice(vb);

            k_cursor += kb.len();
        }
        block
    }

    /// Drec hashed key bytes for a fixture record under `parent_oid`
    /// with the given name. We synthesize a fake hash (any value works
    /// for our walker since it iterates by prefix and the caller filters
    /// by name).
    fn drec_hashed_key(parent_oid: u64, name: &str, fake_hash: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(12 + name.len() + 1);
        let hdr = ((APFS_TYPE_DIR_REC as u64) << OBJ_TYPE_SHIFT) | (parent_oid & OBJ_ID_MASK);
        out.extend_from_slice(&hdr.to_le_bytes());
        let nh = (fake_hash << 10) | ((name.len() as u32 + 1) & J_DREC_LEN_MASK);
        out.extend_from_slice(&nh.to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.push(0);
        out
    }

    fn drec_val(file_id: u64, dtype: u16) -> Vec<u8> {
        let mut out = vec![0u8; 18];
        out[0..8].copy_from_slice(&file_id.to_le_bytes());
        out[16..18].copy_from_slice(&dtype.to_le_bytes());
        out
    }

    /// Range scan over a single-leaf fs-tree yields all drecs under a
    /// parent oid and stops at the prefix boundary.
    #[test]
    fn range_scan_single_leaf_yields_all_drecs_under_parent() {
        let block_size = 1024usize;
        // Three entries: oid=2 has "foo", "bar"; oid=3 has "qux"
        // Drec hashed keys must sort by (oid, kind, hash); we pick
        // hashes 0x100, 0x200 to keep them sorted under oid=2.
        let entries = vec![
            (
                drec_hashed_key(2, "bar", 0x100),
                drec_val(0x11, super::super::jrec::DT_REG),
            ),
            (
                drec_hashed_key(2, "foo", 0x200),
                drec_val(0x12, super::super::jrec::DT_DIR),
            ),
            (
                drec_hashed_key(3, "qux", 0x100),
                drec_val(0x13, super::super::jrec::DT_REG),
            ),
        ];
        let root = build_var_leaf(block_size, &entries, true);

        // Omap can be a degenerate single-leaf with no entries — we
        // never call it because the root is a leaf.
        let omap_root = build_empty_omap_root(block_size);

        let mut ctx = FsTreeCtx::new(omap_root, 0, block_size);
        let target = FsKeyTarget {
            oid: 2,
            kind: APFS_TYPE_DIR_REC,
            tail: &[],
            drec_layout: DrecKeyLayout::Hashed,
        };
        let mut never = |_p: u64, _b: &mut [u8]| -> crate::Result<()> {
            panic!("single-leaf walk must not hit the device")
        };
        let mut scan = RangeScan::start(&root, &target, &mut ctx, &mut never).unwrap();
        let a = scan.next(&mut ctx, &mut never).unwrap().unwrap();
        let b = scan.next(&mut ctx, &mut never).unwrap().unwrap();
        let c = scan.next(&mut ctx, &mut never).unwrap();
        assert!(c.is_none(), "third call should stop at prefix boundary");

        // Verify we got "bar" then "foo".
        let k_a = super::super::jrec::DrecKey::decode_hashed(&a.0).unwrap();
        let k_b = super::super::jrec::DrecKey::decode_hashed(&b.0).unwrap();
        assert_eq!(k_a.name, "bar");
        assert_eq!(k_b.name, "foo");
    }

    /// Build a minimal omap root block that contains zero entries. Used
    /// when tests don't exercise the omap path.
    fn build_empty_omap_root(block_size: usize) -> Vec<u8> {
        let mut block = vec![0u8; block_size];
        block[24..28].copy_from_slice(&OBJECT_TYPE_BTREE_NODE.to_le_bytes());
        let flags = BTNODE_ROOT | BTNODE_LEAF | BTNODE_FIXED_KV_SIZE;
        block[32..34].copy_from_slice(&flags.to_le_bytes());
        block[36..40].copy_from_slice(&0u32.to_le_bytes()); // nkeys
        block[40..42].copy_from_slice(&0u16.to_le_bytes());
        block[42..44].copy_from_slice(&0u16.to_le_bytes());
        let info_off = block_size - BTREE_INFO_SIZE;
        block[info_off + 8..info_off + 12].copy_from_slice(&16u32.to_le_bytes());
        block[info_off + 12..info_off + 16].copy_from_slice(&16u32.to_le_bytes());
        block
    }

    /// Range scan over a *two-level* fs-tree descends via virtual-oid
    /// resolution. We build:
    ///   omap (1 leaf):  vid=100 (xid 1) → paddr 0x10
    ///                  vid=101 (xid 1) → paddr 0x11
    ///   fs root (internal): key (5, INODE) → child vid=100
    ///                       key (5, DIR_REC) → child vid=101
    ///   leaf at paddr 0x10: an inode record for oid=5
    ///   leaf at paddr 0x11: two drec records under oid=5
    #[test]
    fn range_scan_two_level_descends_via_omap() {
        let block_size = 1024usize;
        // Build inode-leaf.
        let inode_key = {
            let mut k = vec![0u8; 8];
            let hdr = ((APFS_TYPE_INODE as u64) << OBJ_TYPE_SHIFT) | (5 & OBJ_ID_MASK);
            k.copy_from_slice(&hdr.to_le_bytes());
            k
        };
        let inode_val = vec![0u8; super::super::jrec::J_INODE_VAL_FIXED_SIZE];
        let inode_leaf = build_var_leaf(block_size, &[(inode_key, inode_val)], false);

        // Build drec leaf with two entries under oid=5.
        let drec_entries = vec![
            (
                drec_hashed_key(5, "a", 0x100),
                drec_val(0x21, super::super::jrec::DT_REG),
            ),
            (
                drec_hashed_key(5, "b", 0x200),
                drec_val(0x22, super::super::jrec::DT_REG),
            ),
        ];
        let drec_leaf = build_var_leaf(block_size, &drec_entries, false);

        // Build the omap leaf at root: vid 100 → 0x10, vid 101 → 0x11.
        let omap_root = build_omap_leaf_root(block_size, &[(100, 1, 0x10), (101, 1, 0x11)]);

        // Build the fs-tree internal root. Its keys are (5, INODE) and
        // (5, DIR_REC); its values are 8-byte virtual oids 100 and 101.
        let fs_root = build_var_internal_fsroot(
            block_size,
            &[
                ((APFS_TYPE_INODE, 5), 100u64),
                ((APFS_TYPE_DIR_REC, 5), 101u64),
            ],
        );

        let mut device: std::collections::HashMap<u64, Vec<u8>> = Default::default();
        device.insert(0x10, inode_leaf);
        device.insert(0x11, drec_leaf);

        let mut ctx = FsTreeCtx::new(omap_root, 5, block_size);

        // ---- Range-scan the drecs ----
        let target = FsKeyTarget {
            oid: 5,
            kind: APFS_TYPE_DIR_REC,
            tail: &[],
            drec_layout: DrecKeyLayout::Hashed,
        };
        let mut read = |paddr: u64, buf: &mut [u8]| -> crate::Result<()> {
            let b = device
                .get(&paddr)
                .ok_or_else(|| crate::Error::InvalidImage(format!("no block at paddr {paddr}")))?;
            buf.copy_from_slice(b);
            Ok(())
        };
        let mut scan = RangeScan::start(&fs_root, &target, &mut ctx, &mut read).unwrap();
        let a = scan.next(&mut ctx, &mut read).unwrap().unwrap();
        let b = scan.next(&mut ctx, &mut read).unwrap().unwrap();
        let c = scan.next(&mut ctx, &mut read).unwrap();
        assert!(c.is_none());

        let k_a = super::super::jrec::DrecKey::decode_hashed(&a.0).unwrap();
        let k_b = super::super::jrec::DrecKey::decode_hashed(&b.0).unwrap();
        assert_eq!((k_a.name.as_str(), k_b.name.as_str()), ("a", "b"));
    }

    /// Tiny omap leaf root constructor (fixed-KV 16/16). Used by the
    /// fs-tree tests.
    fn build_omap_leaf_root(block_size: usize, entries: &[(u64, u64, u64)]) -> Vec<u8> {
        let mut block = vec![0u8; block_size];
        block[24..28].copy_from_slice(&OBJECT_TYPE_BTREE_NODE.to_le_bytes());
        let flags = BTNODE_ROOT | BTNODE_LEAF | BTNODE_FIXED_KV_SIZE;
        block[32..34].copy_from_slice(&flags.to_le_bytes());
        block[36..40].copy_from_slice(&(entries.len() as u32).to_le_bytes());
        let toc_len = entries.len() * 4;
        block[40..42].copy_from_slice(&0u16.to_le_bytes());
        block[42..44].copy_from_slice(&(toc_len as u16).to_le_bytes());

        let toc_base = 56;
        let keys_start = toc_base + toc_len;
        let vals_end = block_size - BTREE_INFO_SIZE;

        for (i, &(oid, xid, paddr)) in entries.iter().enumerate() {
            let k_off = (i * 16) as u16;
            let v_off = ((i + 1) * 16) as u16;
            block[toc_base + i * 4..toc_base + i * 4 + 2].copy_from_slice(&k_off.to_le_bytes());
            block[toc_base + i * 4 + 2..toc_base + i * 4 + 4].copy_from_slice(&v_off.to_le_bytes());

            let ks = keys_start + k_off as usize;
            block[ks..ks + 8].copy_from_slice(&oid.to_le_bytes());
            block[ks + 8..ks + 16].copy_from_slice(&xid.to_le_bytes());

            let vs = vals_end - v_off as usize;
            block[vs + 8..vs + 16].copy_from_slice(&paddr.to_le_bytes());
        }

        let info_off = block_size - BTREE_INFO_SIZE;
        block[info_off + 8..info_off + 12].copy_from_slice(&16u32.to_le_bytes());
        block[info_off + 12..info_off + 16].copy_from_slice(&16u32.to_le_bytes());
        block
    }

    /// Build a variable-KV internal fs-tree root whose keys are
    /// `(kind, oid)` (laid out as a j_key_t header) and whose values
    /// are 8-byte virtual oids pointing at child nodes.
    fn build_var_internal_fsroot(block_size: usize, entries: &[((u8, u64), u64)]) -> Vec<u8> {
        // Convert (kind, oid) into 8-byte keys.
        let mapped: Vec<(Vec<u8>, Vec<u8>)> = entries
            .iter()
            .map(|((kind, oid), vid)| {
                let hdr = ((*kind as u64) << OBJ_TYPE_SHIFT) | (oid & OBJ_ID_MASK);
                (hdr.to_le_bytes().to_vec(), vid.to_le_bytes().to_vec())
            })
            .collect();

        let mut block = vec![0u8; block_size];
        block[24..28].copy_from_slice(&OBJECT_TYPE_BTREE_NODE.to_le_bytes());
        let flags = BTNODE_ROOT; // not leaf, var-KV
        block[32..34].copy_from_slice(&flags.to_le_bytes());
        block[34..36].copy_from_slice(&1u16.to_le_bytes()); // level
        block[36..40].copy_from_slice(&(mapped.len() as u32).to_le_bytes());
        let toc_len = mapped.len() * 8;
        block[40..42].copy_from_slice(&0u16.to_le_bytes());
        block[42..44].copy_from_slice(&(toc_len as u16).to_le_bytes());

        let toc_base = 56;
        let keys_start = toc_base + toc_len;
        let vals_end = block_size - BTREE_INFO_SIZE;

        let mut k_cursor = 0usize;
        let mut v_cursor_back = 0usize;
        for (i, (kb, vb)) in mapped.iter().enumerate() {
            let k_off = k_cursor as u16;
            let k_len = kb.len() as u16;
            v_cursor_back += vb.len();
            let v_off = v_cursor_back as u16;
            let v_len = vb.len() as u16;
            block[toc_base + i * 8..toc_base + i * 8 + 2].copy_from_slice(&k_off.to_le_bytes());
            block[toc_base + i * 8 + 2..toc_base + i * 8 + 4].copy_from_slice(&k_len.to_le_bytes());
            block[toc_base + i * 8 + 4..toc_base + i * 8 + 6].copy_from_slice(&v_off.to_le_bytes());
            block[toc_base + i * 8 + 6..toc_base + i * 8 + 8].copy_from_slice(&v_len.to_le_bytes());

            let ks = keys_start + k_off as usize;
            block[ks..ks + kb.len()].copy_from_slice(kb);
            let vs = vals_end - v_off as usize;
            block[vs..vs + vb.len()].copy_from_slice(vb);

            k_cursor += kb.len();
        }
        // For variable-KV trees the trailing btree_info_t still records
        // key/val sizes — set them to 0 since they're not used.
        block
    }
}
