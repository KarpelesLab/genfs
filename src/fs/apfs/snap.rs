//! Snapshot-metadata record decoders.
//!
//! Reference: *Apple File System Reference*, section "Snapshot Metadata".
//!
//! The snapshot-metadata tree is a physical B-tree referenced from the
//! volume's `apfs_snap_meta_tree_oid`. It carries two kinds of records,
//! both keyed by the standard 8-byte `j_key_t` prefix:
//!
//! ```text
//! j_snap_metadata_key_t (= j_key_t)
//!   0  obj_id_and_type  u64    high 4 bits = APFS_TYPE_SNAP_METADATA,
//!                              low 60 = snapshot xid
//!
//! j_snap_metadata_val_t (variable-length, fixed prefix = 48 bytes)
//!   0  extentref_tree_oid    u64
//!   8  sblock_oid            u64    physical block of the snapshot's APSB
//!  16  create_time           u64
//!  24  change_time           u64
//!  32  inum                  u64
//!  40  extentref_tree_type   u32
//!  44  flags                 u32
//!  48  name_len              u16
//!  50  name[name_len]        u8     NUL-terminated UTF-8
//!
//! j_snap_name_key_t
//!   0  obj_id_and_type  u64    high 4 bits = APFS_TYPE_SNAP_NAME,
//!                              low 60 = ~0 (per spec; we don't filter)
//!   8  name_len         u16
//!  10  name[name_len]   u8
//!
//! j_snap_name_val_t
//!   0  snap_xid         u64
//! ```
//!
//! The tree is ordered ascending by the 8-byte key prefix; SNAP_METADATA
//! and SNAP_NAME records co-exist with disjoint kind codes, so callers
//! filter by `kind` in the key.

use super::jrec::split_obj_id;

/// Decode the `j_key_t` prefix of a snap-meta-tree key and return
/// `(record_kind, value_in_low_60_bits)`. The interpretation of the low
/// 60 bits depends on the kind: for `APFS_TYPE_SNAP_METADATA` it's the
/// snapshot xid; for `APFS_TYPE_SNAP_NAME` it's typically ~0 and you
/// should look at the name in the tail bytes instead.
pub fn decode_snap_meta_key(buf: &[u8]) -> crate::Result<(u8, u64)> {
    if buf.len() < 8 {
        return Err(crate::Error::InvalidImage(
            "apfs: snap-meta key shorter than j_key_t".into(),
        ));
    }
    let hdr = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let (kind, lo) = split_obj_id(hdr);
    Ok((kind, lo))
}

/// Decoded `j_snap_metadata_val_t`. Only the fields we actually surface
/// are retained.
#[derive(Debug, Clone)]
pub struct SnapMetaVal {
    pub extentref_tree_oid: u64,
    pub sblock_oid: u64,
    pub create_time: u64,
    pub change_time: u64,
    pub inum: u64,
    pub extentref_tree_type: u32,
    pub flags: u32,
    pub name: String,
}

impl SnapMetaVal {
    /// Fixed-prefix size in bytes (everything before `name[]`).
    pub const FIXED_PREFIX: usize = 50;

    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < Self::FIXED_PREFIX {
            return Err(crate::Error::InvalidImage(
                "apfs: j_snap_metadata_val too short".into(),
            ));
        }
        let extentref_tree_oid = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let sblock_oid = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let create_time = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let change_time = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        let inum = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        let extentref_tree_type = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        let flags = u32::from_le_bytes(buf[44..48].try_into().unwrap());
        let name_len = u16::from_le_bytes(buf[48..50].try_into().unwrap()) as usize;
        let end = (50 + name_len).min(buf.len());
        let raw = &buf[50..end];
        let nul = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let name = String::from_utf8_lossy(&raw[..nul]).into_owned();
        Ok(Self {
            extentref_tree_oid,
            sblock_oid,
            create_time,
            change_time,
            inum,
            extentref_tree_type,
            flags,
            name,
        })
    }
}

/// Decoded `j_snap_name_val_t` — just the snapshot xid.
#[derive(Debug, Clone, Copy)]
pub struct SnapNameVal {
    pub snap_xid: u64,
}

impl SnapNameVal {
    pub const SIZE: usize = 8;
    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(crate::Error::InvalidImage(
                "apfs: j_snap_name_val too short".into(),
            ));
        }
        Ok(Self {
            snap_xid: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::jrec::{APFS_TYPE_SNAP_METADATA, OBJ_TYPE_SHIFT};
    use super::*;

    #[test]
    fn decode_snap_meta_key_basic() {
        let mut k = [0u8; 8];
        let hdr = ((APFS_TYPE_SNAP_METADATA as u64) << OBJ_TYPE_SHIFT) | 42u64;
        k.copy_from_slice(&hdr.to_le_bytes());
        let (kind, xid) = decode_snap_meta_key(&k).unwrap();
        assert_eq!(kind, APFS_TYPE_SNAP_METADATA);
        assert_eq!(xid, 42);
    }

    #[test]
    fn decode_snap_meta_val_with_name() {
        let mut v = vec![0u8; SnapMetaVal::FIXED_PREFIX + 8];
        v[8..16].copy_from_slice(&0x1234u64.to_le_bytes()); // sblock_oid
        v[16..24].copy_from_slice(&100u64.to_le_bytes()); // create_time
        v[48..50].copy_from_slice(&5u16.to_le_bytes()); // name_len = 5 ("hi\0")
        v[50..55].copy_from_slice(b"hi\0\0\0");
        let m = SnapMetaVal::decode(&v).unwrap();
        assert_eq!(m.sblock_oid, 0x1234);
        assert_eq!(m.create_time, 100);
        assert_eq!(m.name, "hi");
    }

    #[test]
    fn decode_snap_name_val() {
        let mut v = [0u8; 8];
        v.copy_from_slice(&77u64.to_le_bytes());
        let n = SnapNameVal::decode(&v).unwrap();
        assert_eq!(n.snap_xid, 77);
    }
}
