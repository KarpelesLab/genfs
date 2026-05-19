//! `btree_node_phys_t` parsing.
//!
//! Reference: *Apple File System Reference*, section "B-Trees".
//!
//! ```text
//! btree_node_phys_t (header = 56 bytes)
//!    0   btn_o              obj_phys_t (32 B)
//!   32   btn_flags          u16
//!   34   btn_level          u16     0 = leaf
//!   36   btn_nkeys          u32
//!   40   btn_table_space    nloc_t  (u16 off, u16 len) — ToC area location
//!   44   btn_free_space     nloc_t
//!   48   btn_key_free_list  nloc_t
//!   52   btn_val_free_list  nloc_t
//!   56   btn_data[]                  — variable-size payload
//! ```
//!
//! Inside `btn_data[]`:
//! - The table of contents starts at offset `btn_table_space.off` (relative
//!   to the start of `btn_data[]`) and is `btn_table_space.len` bytes long.
//! - Each ToC entry is either a `kvloc_t` (8 B) or a `kvoff_t` (4 B),
//!   selected by the `BTNODE_FIXED_KV_SIZE` flag.
//! - Keys live in the area right after the ToC. Key offsets in ToC entries
//!   are *relative to the start of that key area* (i.e. growing forward).
//! - Values live at the high end of `btn_data[]`. Value offsets are
//!   *relative to the end of the data area* (growing backward). For a root
//!   node, the data area ends 40 bytes before the end of the block (the
//!   trailing `btree_info_t` lives there).
//!
//! A leaf value of 0xFFFF (in a `BTREE_ALLOW_GHOSTS` tree) means "no value".

use super::obj::{OBJECT_TYPE_BTREE_NODE, ObjPhys};

/// `BTNODE_ROOT` — set on the root node of a tree. Root nodes have a
/// trailing `btree_info_t` in the last 40 bytes of the block.
pub const BTNODE_ROOT: u16 = 0x0001;
/// `BTNODE_LEAF` — set on leaf nodes.
pub const BTNODE_LEAF: u16 = 0x0002;
/// `BTNODE_FIXED_KV_SIZE` — set when every key and every value have the
/// same size (ToC uses 4-byte `kvoff_t` instead of 8-byte `kvloc_t`).
pub const BTNODE_FIXED_KV_SIZE: u16 = 0x0004;

/// Trailing `btree_info_t` is 40 bytes on every root node.
pub const BTREE_INFO_SIZE: usize = 40;

/// Sentinel value used in the value offset field when a key has no value
/// (only valid in `BTREE_ALLOW_GHOSTS` trees).
pub const BTOFF_INVALID: u16 = 0xFFFF;

/// A decoded `btree_node_phys_t` (only the header + ToC; key/value bytes
/// are looked up on demand via [`BTreeNode::entry_at`]).
#[derive(Debug)]
pub struct BTreeNode<'a> {
    /// Backing block bytes (length = container block size).
    block: &'a [u8],
    pub obj: ObjPhys,
    pub flags: u16,
    pub level: u16,
    pub nkeys: u32,
    /// Offset of the ToC area inside `btn_data[]`.
    pub toc_off: u16,
    pub toc_len: u16,
    /// True when every key/value is the same fixed size (uses `kvoff_t`).
    pub fixed_kv: bool,
    /// True when this is the root of the tree (trailing btree_info_t).
    pub is_root: bool,
    /// Absolute offset (inside `block`) of the start of `btn_data[]`.
    data_start: usize,
    /// Absolute offset (inside `block`) of the start of the key area
    /// (right after the ToC).
    pub keys_start: usize,
    /// Absolute offset (inside `block`) of the *end* of the value area.
    /// Values grow downward from here.
    pub vals_end: usize,
}

impl<'a> BTreeNode<'a> {
    /// Parse a B-tree node from `block` (a full container block). The
    /// caller is responsible for slicing `block` to the right length
    /// (usually 4096 bytes — see `nx_block_size`).
    pub fn decode(block: &'a [u8]) -> crate::Result<Self> {
        if block.len() < 56 {
            return Err(crate::Error::InvalidImage(
                "apfs: btree node block too small".into(),
            ));
        }
        let obj = ObjPhys::decode(block)?;
        if obj.obj_type() != OBJECT_TYPE_BTREE_NODE
            && obj.obj_type() != super::obj::OBJECT_TYPE_BTREE
        {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: o_type {:#x} is not BTREE_NODE / BTREE",
                obj.obj_type()
            )));
        }
        let flags = u16::from_le_bytes(block[32..34].try_into().unwrap());
        let level = u16::from_le_bytes(block[34..36].try_into().unwrap());
        let nkeys = u32::from_le_bytes(block[36..40].try_into().unwrap());
        let toc_off = u16::from_le_bytes(block[40..42].try_into().unwrap());
        let toc_len = u16::from_le_bytes(block[42..44].try_into().unwrap());

        let data_start: usize = 56;
        let is_root = flags & BTNODE_ROOT != 0;
        let fixed_kv = flags & BTNODE_FIXED_KV_SIZE != 0;
        let keys_start = data_start
            .checked_add(toc_off as usize)
            .and_then(|v| v.checked_add(toc_len as usize))
            .ok_or_else(|| crate::Error::InvalidImage("apfs: btree ToC overflow".into()))?;
        let vals_end_raw = block.len();
        let vals_end = if is_root {
            vals_end_raw
                .checked_sub(BTREE_INFO_SIZE)
                .ok_or_else(|| crate::Error::InvalidImage("apfs: btree root too small".into()))?
        } else {
            vals_end_raw
        };
        if keys_start > vals_end {
            return Err(crate::Error::InvalidImage(
                "apfs: btree node key area collides with value area".into(),
            ));
        }

        Ok(Self {
            block,
            obj,
            flags,
            level,
            nkeys,
            toc_off,
            toc_len,
            fixed_kv,
            is_root,
            data_start,
            keys_start,
            vals_end,
        })
    }

    /// True iff this node is a leaf (records are key→value, not key→child).
    pub fn is_leaf(&self) -> bool {
        self.flags & BTNODE_LEAF != 0
    }

    /// Read the ToC entry at `index`, returning `(key_off, key_len,
    /// val_off, val_len)` where:
    /// - `key_off` is the offset *from `keys_start`* (i.e. add to it),
    /// - `val_off` is the offset *from `vals_end`* (i.e. subtract from it).
    ///
    /// For a non-fixed-KV node, the lengths come straight from the ToC.
    /// For a fixed-KV node, the caller must supply `(fixed_klen,
    /// fixed_vlen)` via [`BTreeNode::fixed_kv_size`] from the trailing
    /// `btree_info_t` — but for our read-only path we treat them as zero
    /// (we slice via the next entry's offsets or read at fixed sizes
    /// known to the caller). See [`BTreeNode::entry_at`] for the
    /// fully-resolved view used by record decoders.
    fn raw_entry(&self, index: u32) -> crate::Result<RawEntry> {
        if index >= self.nkeys {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: btree index {index} >= nkeys {}",
                self.nkeys
            )));
        }
        let toc_base = self.data_start + self.toc_off as usize;
        if self.fixed_kv {
            let off = toc_base + (index as usize) * 4;
            if off + 4 > self.block.len() {
                return Err(crate::Error::InvalidImage(
                    "apfs: btree fixed-kv ToC out of bounds".into(),
                ));
            }
            let k = u16::from_le_bytes(self.block[off..off + 2].try_into().unwrap());
            let v = u16::from_le_bytes(self.block[off + 2..off + 4].try_into().unwrap());
            Ok(RawEntry {
                key_off: k,
                key_len: None,
                val_off: v,
                val_len: None,
            })
        } else {
            let off = toc_base + (index as usize) * 8;
            if off + 8 > self.block.len() {
                return Err(crate::Error::InvalidImage(
                    "apfs: btree var-kv ToC out of bounds".into(),
                ));
            }
            let ko = u16::from_le_bytes(self.block[off..off + 2].try_into().unwrap());
            let kl = u16::from_le_bytes(self.block[off + 2..off + 4].try_into().unwrap());
            let vo = u16::from_le_bytes(self.block[off + 4..off + 6].try_into().unwrap());
            let vl = u16::from_le_bytes(self.block[off + 6..off + 8].try_into().unwrap());
            Ok(RawEntry {
                key_off: ko,
                key_len: Some(kl),
                val_off: vo,
                val_len: Some(vl),
            })
        }
    }

    /// Iterate over `(key_bytes, value_bytes)` slices for every entry in
    /// this node. For fixed-KV nodes we expect callers to know the fixed
    /// key and value sizes (passed as `(klen, vlen)`); the trailing
    /// `btree_info_t` carries them at offsets 32 and 36 respectively. For
    /// variable-KV nodes the lengths come from the ToC, so we pass dummy
    /// zeros.
    ///
    /// Ghost entries (val_off == 0xFFFF) yield an empty value slice.
    pub fn entries(
        &self,
        fixed_klen: usize,
        fixed_vlen: usize,
    ) -> impl Iterator<Item = crate::Result<(&'a [u8], &'a [u8])>> + '_ {
        (0..self.nkeys).map(move |i| self.entry_at(i, fixed_klen, fixed_vlen))
    }

    /// Resolve the i-th entry into key / value byte slices over the
    /// underlying block. See [`BTreeNode::entries`] for `fixed_*` semantics.
    pub fn entry_at(
        &self,
        index: u32,
        fixed_klen: usize,
        fixed_vlen: usize,
    ) -> crate::Result<(&'a [u8], &'a [u8])> {
        let e = self.raw_entry(index)?;
        let klen = e.key_len.map(usize::from).unwrap_or(fixed_klen);
        let kstart = self
            .keys_start
            .checked_add(e.key_off as usize)
            .ok_or_else(|| crate::Error::InvalidImage("apfs: btree key offset overflow".into()))?;
        let kend = kstart
            .checked_add(klen)
            .ok_or_else(|| crate::Error::InvalidImage("apfs: btree key length overflow".into()))?;
        if kend > self.vals_end {
            return Err(crate::Error::InvalidImage(
                "apfs: btree key extends past value area".into(),
            ));
        }
        let key = &self.block[kstart..kend];

        let val: &[u8] = if e.val_off == BTOFF_INVALID {
            &[]
        } else {
            let vlen = e.val_len.map(usize::from).unwrap_or(fixed_vlen);
            // Per spec: `v.off` is the value's offset *measured backward
            // from `vals_end`*, and the value extends forward from there
            // for `v.len` bytes. So vstart = vals_end - v.off and
            // vend = vstart + v.len.
            let vstart = self
                .vals_end
                .checked_sub(e.val_off as usize)
                .ok_or_else(|| {
                    crate::Error::InvalidImage("apfs: btree value offset underflow".into())
                })?;
            let vend = vstart.checked_add(vlen).ok_or_else(|| {
                crate::Error::InvalidImage("apfs: btree value length overflow".into())
            })?;
            if vstart < self.keys_start || vend > self.vals_end {
                return Err(crate::Error::InvalidImage(
                    "apfs: btree value extends outside data area".into(),
                ));
            }
            &self.block[vstart..vend]
        };
        Ok((key, val))
    }

    /// Read the trailing `btree_info_t.bt_fixed.bt_key_size` and
    /// `bt_val_size` fields. Only meaningful on a root node with fixed KVs.
    ///
    /// ```text
    /// btree_info_fixed_t (32 B) at block_end - 40
    ///    0  bt_flags         u32
    ///    4  bt_node_size     u32
    ///    8  bt_key_size      u32
    ///   12  bt_val_size      u32
    /// btree_info_t (40 B) at block_end - 40
    ///    0  bt_fixed         btree_info_fixed_t (16 B)
    ///   16  bt_longest_key   u32
    ///   20  bt_longest_val   u32
    ///   24  bt_key_count     u64
    ///   32  bt_node_count    u64
    /// ```
    pub fn fixed_kv_size(&self) -> Option<(usize, usize)> {
        if !self.is_root {
            return None;
        }
        let off = self.block.len().checked_sub(BTREE_INFO_SIZE)?;
        if off + 16 > self.block.len() {
            return None;
        }
        let key_size = u32::from_le_bytes(self.block[off + 8..off + 12].try_into().unwrap());
        let val_size = u32::from_le_bytes(self.block[off + 12..off + 16].try_into().unwrap());
        Some((key_size as usize, val_size as usize))
    }
}

/// A raw ToC entry. `key_len` / `val_len` are `None` for fixed-KV nodes.
#[derive(Debug, Clone, Copy)]
struct RawEntry {
    key_off: u16,
    key_len: Option<u16>,
    val_off: u16,
    val_len: Option<u16>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::apfs::obj::OBJECT_TYPE_BTREE_NODE;

    /// Hand-craft a 256-byte non-root, leaf, variable-KV B-tree node with
    /// two entries and confirm we can read both back.
    #[test]
    fn decode_variable_kv_leaf() {
        let block_size = 256usize;
        let mut block = vec![0u8; block_size];
        // obj_phys: cksum zero, oid 1, type BTREE_NODE.
        block[24..28].copy_from_slice(&OBJECT_TYPE_BTREE_NODE.to_le_bytes());
        // header
        block[32..34].copy_from_slice(&BTNODE_LEAF.to_le_bytes()); // flags = leaf only
        block[34..36].copy_from_slice(&0u16.to_le_bytes()); // level
        block[36..40].copy_from_slice(&2u32.to_le_bytes()); // nkeys = 2

        // ToC at btn_data[0..] = block[56..]. Two kvloc_t entries = 16 B.
        // Keys area starts at block[56+16] = block[72].
        // First key: "AA" (2 B) at key_off 0, len 2.
        // Second key: "BB" (2 B) at key_off 2, len 2.
        // Values are at the end of the block; first value bytes "vv" (2 B)
        // at the very end (val_off = 2 measuring backward), second value
        // "VV" (2 B) at val_off 4.
        block[40..42].copy_from_slice(&0u16.to_le_bytes()); // table_space.off
        block[42..44].copy_from_slice(&16u16.to_le_bytes()); // table_space.len

        // ToC entry 0: kvloc { k.off=0, k.len=2, v.off=2, v.len=2 }
        let toc_base = 56;
        block[toc_base..toc_base + 2].copy_from_slice(&0u16.to_le_bytes()); // k.off
        block[toc_base + 2..toc_base + 4].copy_from_slice(&2u16.to_le_bytes()); // k.len
        block[toc_base + 4..toc_base + 6].copy_from_slice(&2u16.to_le_bytes()); // v.off
        block[toc_base + 6..toc_base + 8].copy_from_slice(&2u16.to_le_bytes()); // v.len
        // ToC entry 1: kvloc { k.off=2, k.len=2, v.off=4, v.len=2 }
        block[toc_base + 8..toc_base + 10].copy_from_slice(&2u16.to_le_bytes());
        block[toc_base + 10..toc_base + 12].copy_from_slice(&2u16.to_le_bytes());
        block[toc_base + 12..toc_base + 14].copy_from_slice(&4u16.to_le_bytes());
        block[toc_base + 14..toc_base + 16].copy_from_slice(&2u16.to_le_bytes());

        // Keys.
        let keys_start = toc_base + 16; // 72
        block[keys_start..keys_start + 2].copy_from_slice(b"AA");
        block[keys_start + 2..keys_start + 4].copy_from_slice(b"BB");

        // Values, growing from end of block.
        let vend = block_size;
        block[vend - 2..vend].copy_from_slice(b"vv");
        block[vend - 4..vend - 2].copy_from_slice(b"VV");

        let node = BTreeNode::decode(&block).unwrap();
        assert!(node.is_leaf());
        assert!(!node.fixed_kv);
        assert!(!node.is_root);
        assert_eq!(node.nkeys, 2);
        let (k0, v0) = node.entry_at(0, 0, 0).unwrap();
        let (k1, v1) = node.entry_at(1, 0, 0).unwrap();
        assert_eq!(k0, b"AA");
        assert_eq!(v0, b"vv");
        assert_eq!(k1, b"BB");
        assert_eq!(v1, b"VV");
    }

    /// Hand-craft a fixed-KV root node (e.g. an omap node) with one entry.
    /// Keys are 16 bytes, values 16 bytes — matching real omap layouts.
    #[test]
    fn decode_fixed_kv_root() {
        let block_size = 512usize;
        let mut block = vec![0u8; block_size];
        block[24..28].copy_from_slice(&OBJECT_TYPE_BTREE_NODE.to_le_bytes());
        let flags = BTNODE_ROOT | BTNODE_LEAF | BTNODE_FIXED_KV_SIZE;
        block[32..34].copy_from_slice(&flags.to_le_bytes());
        block[36..40].copy_from_slice(&1u32.to_le_bytes()); // nkeys
        block[40..42].copy_from_slice(&0u16.to_le_bytes()); // table_space.off
        block[42..44].copy_from_slice(&4u16.to_le_bytes()); // table_space.len (one kvoff_t)

        // Single kvoff_t: k=0, v=16
        let toc_base = 56;
        block[toc_base..toc_base + 2].copy_from_slice(&0u16.to_le_bytes());
        block[toc_base + 2..toc_base + 4].copy_from_slice(&16u16.to_le_bytes());

        // Keys area starts at 56+4 = 60.
        let keys_start = 60;
        for (i, b) in block.iter_mut().enumerate().skip(keys_start).take(16) {
            *b = (i as u8).wrapping_add(0x10);
        }
        // Value area ends 40 bytes before block end (root has trailing
        // btree_info_t). Value bytes occupy [vals_end-16, vals_end).
        let vals_end = block_size - BTREE_INFO_SIZE;
        for (i, b) in block[vals_end - 16..vals_end].iter_mut().enumerate() {
            *b = (i as u8).wrapping_add(0xa0);
        }

        // Populate btree_info.bt_fixed.bt_key_size = 16, bt_val_size = 16.
        let info_off = block_size - BTREE_INFO_SIZE;
        block[info_off + 8..info_off + 12].copy_from_slice(&16u32.to_le_bytes());
        block[info_off + 12..info_off + 16].copy_from_slice(&16u32.to_le_bytes());

        let node = BTreeNode::decode(&block).unwrap();
        assert!(node.fixed_kv);
        assert!(node.is_root);
        let (klen, vlen) = node.fixed_kv_size().unwrap();
        assert_eq!((klen, vlen), (16, 16));
        let (k, v) = node.entry_at(0, klen, vlen).unwrap();
        assert_eq!(k.len(), 16);
        assert_eq!(v.len(), 16);
        assert_eq!(k[0], 0x10 + keys_start as u8);
        assert_eq!(v[0], 0xa0);
    }
}
