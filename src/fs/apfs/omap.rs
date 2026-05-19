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

use super::btree::BTreeNode;
use super::obj::{ObjPhys, OBJECT_TYPE_OMAP};

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
            return Err(crate::Error::InvalidImage("apfs: omap_key too short".into()));
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
            return Err(crate::Error::InvalidImage("apfs: omap_val too short".into()));
        }
        Ok(Self {
            flags: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            size: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            paddr: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        })
    }
}

/// Search an omap B-tree (a leaf-only root node, or a multi-level tree
/// rooted at `root_block`) for the entry matching `target_oid` with the
/// largest `xid <= target_xid`. Returns the resolved `OmapVal` or `None`
/// when the oid is absent.
///
/// `read_block` is supplied by the caller; it's given a physical block
/// number and must fill the passed buffer (length = block size) with that
/// block's contents. We use this trampoline so the function is fully
/// reusable: it makes no assumption about the underlying device.
///
/// **Limitation:** this routine only walks a *single-level* omap (root
/// node is also a leaf). Multi-level omaps return
/// `Err(Unsupported)` for now — a full container needs the multi-level
/// walker, which we punt for v1.
pub fn lookup<F>(
    root_block: &[u8],
    target_oid: u64,
    target_xid: u64,
    _read_block: &mut F,
) -> crate::Result<Option<OmapVal>>
where
    F: FnMut(u64, &mut [u8]) -> crate::Result<()>,
{
    let node = BTreeNode::decode(root_block)?;
    if !node.is_leaf() {
        return Err(crate::Error::Unsupported(
            "apfs: multi-level omap walking not yet implemented".into(),
        ));
    }
    let (klen, vlen) = node
        .fixed_kv_size()
        .ok_or_else(|| crate::Error::InvalidImage("apfs: omap root missing btree_info".into()))?;
    if klen != OmapKey::SIZE || vlen != OmapVal::SIZE {
        return Err(crate::Error::InvalidImage(format!(
            "apfs: omap fixed kv size ({klen}, {vlen}) != (16, 16)"
        )));
    }

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
            let v = OmapVal::decode(vb)?;
            best_xid = k.xid;
            best = Some(v);
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

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
