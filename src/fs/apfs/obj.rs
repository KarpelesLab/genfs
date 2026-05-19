//! `obj_phys_t` — the 32-byte header on every persistent APFS on-disk object.
//!
//! Reference: *Apple File System Reference*, section "Object Identifiers and
//! Checksums" / "Generic Object Header".
//!
//! ```text
//! obj_phys_t (32 bytes, all little-endian)
//!   0  o_cksum   u8[8]   Fletcher-64 over the rest of the object
//!   8  o_oid     u64     object identifier (virtual / ephemeral / physical)
//!  16  o_xid     u64     transaction identifier
//!  24  o_type    u32     low 16 bits = type, high 16 = flags
//!  28  o_subtype u32     subtype (e.g. for B-tree contents)
//! ```

/// 32-byte header that precedes every APFS on-disk object.
#[derive(Debug, Clone, Copy)]
pub struct ObjPhys {
    pub cksum: u64,
    pub oid: u64,
    pub xid: u64,
    pub type_and_flags: u32,
    pub subtype: u32,
}

impl ObjPhys {
    /// Size of the on-disk header in bytes.
    pub const SIZE: usize = 32;

    /// Decode the first 32 bytes of `buf` as an `obj_phys_t`. Returns an
    /// error if the buffer is too short.
    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(crate::Error::InvalidImage(
                "apfs: obj_phys_t buffer too short".into(),
            ));
        }
        Ok(Self {
            cksum: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            oid: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            xid: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            type_and_flags: u32::from_le_bytes(buf[24..28].try_into().unwrap()),
            subtype: u32::from_le_bytes(buf[28..32].try_into().unwrap()),
        })
    }

    /// Object type — the low 16 bits of `o_type` (one of the
    /// `OBJECT_TYPE_*` constants).
    pub fn obj_type(&self) -> u32 {
        self.type_and_flags & OBJECT_TYPE_MASK
    }

    /// Object flags — the high 16 bits of `o_type`.
    pub fn obj_flags(&self) -> u32 {
        self.type_and_flags & OBJECT_TYPE_FLAGS_MASK
    }
}

/// Mask isolating the type bits of `o_type`.
pub const OBJECT_TYPE_MASK: u32 = 0x0000_ffff;
/// Mask isolating the flag bits of `o_type`.
pub const OBJECT_TYPE_FLAGS_MASK: u32 = 0xffff_0000;

// Public-spec object type constants — only those we actually consume.
pub const OBJECT_TYPE_NX_SUPERBLOCK: u32 = 0x0000_0001;
pub const OBJECT_TYPE_BTREE: u32 = 0x0000_0002;
pub const OBJECT_TYPE_BTREE_NODE: u32 = 0x0000_0003;
pub const OBJECT_TYPE_OMAP: u32 = 0x0000_000b;
pub const OBJECT_TYPE_CHECKPOINT_MAP: u32 = 0x0000_000c;
pub const OBJECT_TYPE_FS: u32 = 0x0000_000d;
pub const OBJECT_TYPE_FSTREE: u32 = 0x0000_000e;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_simple_obj_phys() {
        let mut buf = [0u8; 32];
        buf[0..8].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
        buf[8..16].copy_from_slice(&0x42u64.to_le_bytes());
        buf[16..24].copy_from_slice(&0xa5u64.to_le_bytes());
        // type = NX_SUPERBLOCK, with a flag in the high half.
        buf[24..28].copy_from_slice(&(OBJECT_TYPE_NX_SUPERBLOCK | 0x8000_0000).to_le_bytes());
        buf[28..32].copy_from_slice(&7u32.to_le_bytes());

        let o = ObjPhys::decode(&buf).unwrap();
        assert_eq!(o.cksum, 0x1122_3344_5566_7788);
        assert_eq!(o.oid, 0x42);
        assert_eq!(o.xid, 0xa5);
        assert_eq!(o.obj_type(), OBJECT_TYPE_NX_SUPERBLOCK);
        assert_eq!(o.obj_flags(), 0x8000_0000);
        assert_eq!(o.subtype, 7);
    }

    #[test]
    fn decode_rejects_short_buffer() {
        let buf = [0u8; 16];
        assert!(ObjPhys::decode(&buf).is_err());
    }
}
