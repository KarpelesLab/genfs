//! Filesystem-tree record decoders: `j_key_t`, `j_inode_val_t`,
//! `j_drec_hashed_key_t`, `j_drec_val_t`, `j_file_extent_key_t`,
//! `j_file_extent_val_t`, and the extended-field walker
//! (`xf_blob_t` / `x_field_t`).
//!
//! Reference: *Apple File System Reference*, section "Filesystem
//! Objects". All multi-byte fields are little-endian.

/// Mask isolating the object id in `j_key.obj_id_and_type` (low 60 bits).
pub const OBJ_ID_MASK: u64 = 0x0fff_ffff_ffff_ffff;
/// Mask isolating the record type (high 4 bits).
pub const OBJ_TYPE_SHIFT: u32 = 60;

// Record-type values stored in the top 4 bits of obj_id_and_type.
pub const APFS_TYPE_SNAP_METADATA: u8 = 1;
pub const APFS_TYPE_EXTENT: u8 = 2;
pub const APFS_TYPE_INODE: u8 = 3;
pub const APFS_TYPE_XATTR: u8 = 4;
pub const APFS_TYPE_SIBLING_LINK: u8 = 5;
pub const APFS_TYPE_DSTREAM_ID: u8 = 6;
pub const APFS_TYPE_CRYPTO_STATE: u8 = 7;
pub const APFS_TYPE_FILE_EXTENT: u8 = 8;
pub const APFS_TYPE_DIR_REC: u8 = 9;
pub const APFS_TYPE_DIR_STATS: u8 = 10;
pub const APFS_TYPE_SNAP_NAME: u8 = 11;
pub const APFS_TYPE_SIBLING_MAP: u8 = 12;

// Extended-field type identifiers (`x_type` in `x_field_t`).
pub const INO_EXT_TYPE_SNAP_XID: u8 = 1;
pub const INO_EXT_TYPE_DELTA_TREE_OID: u8 = 2;
pub const INO_EXT_TYPE_DOCUMENT_ID: u8 = 3;
pub const INO_EXT_TYPE_NAME: u8 = 4;
pub const INO_EXT_TYPE_PREV_FSIZE: u8 = 5;
pub const INO_EXT_TYPE_FINDER_INFO: u8 = 7;
pub const INO_EXT_TYPE_DSTREAM: u8 = 8;
pub const INO_EXT_TYPE_DIR_STATS_KEY: u8 = 10;
pub const INO_EXT_TYPE_FS_UUID: u8 = 11;
pub const INO_EXT_TYPE_SPARSE_BYTES: u8 = 13;
pub const INO_EXT_TYPE_RDEV: u8 = 14;

// File extent length/flag masks.
pub const J_FILE_EXTENT_LEN_MASK: u64 = 0x00ff_ffff_ffff_ffff;

// Directory record flag mask: low 8 bits encode the DT_* file type.
pub const DREC_TYPE_MASK: u16 = 0x000f;
pub const DT_UNKNOWN: u16 = 0;
pub const DT_FIFO: u16 = 1;
pub const DT_CHR: u16 = 2;
pub const DT_DIR: u16 = 4;
pub const DT_BLK: u16 = 6;
pub const DT_REG: u16 = 8;
pub const DT_LNK: u16 = 10;
pub const DT_SOCK: u16 = 12;
pub const DT_WHT: u16 = 14;

/// Split `obj_id_and_type` from a `j_key_t`-prefixed record key into
/// `(record_type, object_id)`.
pub fn split_obj_id(raw: u64) -> (u8, u64) {
    let kind = (raw >> OBJ_TYPE_SHIFT) as u8;
    let oid = raw & OBJ_ID_MASK;
    (kind, oid)
}

/// j_inode_val_t fixed portion (size = 0x5C / 92 bytes); trailing
/// `xfields[]` follows for variable-length attributes.
///
/// ```text
///   0  parent_id                       u64
///   8  private_id                      u64    object id of default dstream
///  16  create_time                     u64    nanoseconds since 1970-01-01
///  24  mod_time                        u64
///  32  change_time                     u64
///  40  access_time                     u64
///  48  internal_flags                  u64
///  56  nchildren / nlink (union)       i32
///  60  default_protection_class        u32
///  64  write_generation_counter        u32
///  68  bsd_flags                       u32
///  72  owner                           u32
///  76  group                           u32
///  80  mode                            u16
///  82  pad1                            u16
///  84  uncompressed_size               u64
///  92  xfields[]
/// ```
pub const J_INODE_VAL_FIXED_SIZE: usize = 92;

#[derive(Debug, Clone)]
pub struct InodeVal {
    pub parent_id: u64,
    pub private_id: u64,
    pub create_time: u64,
    pub mod_time: u64,
    pub change_time: u64,
    pub access_time: u64,
    pub internal_flags: u64,
    pub nchildren_or_nlink: i32,
    pub owner: u32,
    pub group: u32,
    pub mode: u16,
    pub uncompressed_size: u64,
    /// Optional name (filename) recorded as INO_EXT_TYPE_NAME. Trimmed of
    /// trailing NUL.
    pub name: Option<String>,
    /// Optional default-data-stream metadata recorded as
    /// INO_EXT_TYPE_DSTREAM.
    pub dstream: Option<DstreamXf>,
}

impl InodeVal {
    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < J_INODE_VAL_FIXED_SIZE {
            return Err(crate::Error::InvalidImage(
                "apfs: j_inode_val too short".into(),
            ));
        }
        let parent_id = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let private_id = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let create_time = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let mod_time = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        let change_time = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        let access_time = u64::from_le_bytes(buf[40..48].try_into().unwrap());
        let internal_flags = u64::from_le_bytes(buf[48..56].try_into().unwrap());
        let nchildren_or_nlink = i32::from_le_bytes(buf[56..60].try_into().unwrap());
        let owner = u32::from_le_bytes(buf[72..76].try_into().unwrap());
        let group = u32::from_le_bytes(buf[76..80].try_into().unwrap());
        let mode = u16::from_le_bytes(buf[80..82].try_into().unwrap());
        let uncompressed_size = u64::from_le_bytes(buf[84..92].try_into().unwrap());

        // Parse trailing extended fields if any.
        let (name, dstream) = if buf.len() > J_INODE_VAL_FIXED_SIZE {
            parse_inode_xfields(&buf[J_INODE_VAL_FIXED_SIZE..])
        } else {
            (None, None)
        };

        Ok(Self {
            parent_id,
            private_id,
            create_time,
            mod_time,
            change_time,
            access_time,
            internal_flags,
            nchildren_or_nlink,
            owner,
            group,
            mode,
            uncompressed_size,
            name,
            dstream,
        })
    }
}

/// j_dstream_t stored in INO_EXT_TYPE_DSTREAM xfield (0x28 / 40 bytes).
///
/// ```text
///   0  size                    u64
///   8  alloced_size            u64
///  16  default_crypto_id       u64
///  24  total_bytes_written     u64
///  32  total_bytes_read        u64
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DstreamXf {
    pub size: u64,
    pub alloced_size: u64,
}

impl DstreamXf {
    pub const SIZE: usize = 40;
    fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 16 {
            return None;
        }
        Some(Self {
            size: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            alloced_size: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        })
    }
}

/// Walk the inode's xfields blob and pull out NAME and DSTREAM if present.
///
/// `xf_blob_t` layout:
/// ```text
///   0  xf_num_exts   u16
///   2  xf_used_data  u16
///   4  array of x_field_t[xf_num_exts] (each 4 bytes)
///   then the values themselves, 8-byte aligned, in declared order
/// ```
fn parse_inode_xfields(buf: &[u8]) -> (Option<String>, Option<DstreamXf>) {
    if buf.len() < 4 {
        return (None, None);
    }
    let num_exts = u16::from_le_bytes(buf[0..2].try_into().unwrap()) as usize;
    let _used = u16::from_le_bytes(buf[2..4].try_into().unwrap());
    let desc_off = 4;
    let val_area = desc_off + num_exts * 4;
    if val_area > buf.len() {
        return (None, None);
    }
    // Offset *within* the value area. Each value is stored at the next
    // 8-byte-aligned offset *relative to the value area's start*. The
    // padding goes after each value (i.e. we advance, then round up).
    let mut rel: usize = 0;
    let mut name = None;
    let mut dstream = None;
    for i in 0..num_exts {
        let d = desc_off + i * 4;
        if d + 4 > buf.len() {
            break;
        }
        let xtype = buf[d];
        let _xflags = buf[d + 1];
        let xsize = u16::from_le_bytes(buf[d + 2..d + 4].try_into().unwrap()) as usize;
        let abs = val_area + rel;
        if abs + xsize > buf.len() {
            break;
        }
        let val = &buf[abs..abs + xsize];
        match xtype {
            INO_EXT_TYPE_NAME => {
                let end = val.iter().position(|&b| b == 0).unwrap_or(val.len());
                name = Some(String::from_utf8_lossy(&val[..end]).into_owned());
            }
            INO_EXT_TYPE_DSTREAM => {
                dstream = DstreamXf::decode(val);
            }
            _ => {}
        }
        rel += xsize;
        rel = (rel + 7) & !7;
    }
    (name, dstream)
}

/// j_drec_hashed_key_t layout:
/// ```text
///   0  hdr (j_key_t)         u64
///   8  name_len_and_hash     u32 (low 10 bits = name length incl. NUL,
///                                  high 22 bits = name hash)
///  12  name[name_len]        u8     UTF-8, NUL-terminated
/// ```
pub const J_DREC_LEN_MASK: u32 = 0x0000_03ff;

#[derive(Debug, Clone)]
pub struct DrecKey {
    pub parent_oid: u64,
    pub name: String,
}

impl DrecKey {
    pub fn decode_hashed(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < 12 {
            return Err(crate::Error::InvalidImage(
                "apfs: drec hashed key too short".into(),
            ));
        }
        let raw = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let (kind, oid) = split_obj_id(raw);
        if kind != APFS_TYPE_DIR_REC {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: drec key kind {kind} != DIR_REC"
            )));
        }
        let nh = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let nlen = (nh & J_DREC_LEN_MASK) as usize;
        if 12 + nlen > buf.len() || nlen == 0 {
            return Err(crate::Error::InvalidImage(
                "apfs: drec name length oob".into(),
            ));
        }
        let raw_name = &buf[12..12 + nlen];
        let end = raw_name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(raw_name.len());
        let name = String::from_utf8_lossy(&raw_name[..end]).into_owned();
        Ok(Self {
            parent_oid: oid,
            name,
        })
    }

    /// Decode an *unhashed* drec key (used by case-sensitive volumes that
    /// don't carry a hash). Layout:
    /// ```text
    ///   0  hdr (j_key_t)   u64
    ///   8  name_len        u16
    ///  10  name[name_len]  u8
    /// ```
    pub fn decode_plain(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < 10 {
            return Err(crate::Error::InvalidImage(
                "apfs: drec plain key too short".into(),
            ));
        }
        let raw = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let (kind, oid) = split_obj_id(raw);
        if kind != APFS_TYPE_DIR_REC {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: drec key kind {kind} != DIR_REC"
            )));
        }
        let nlen = u16::from_le_bytes(buf[8..10].try_into().unwrap()) as usize;
        if 10 + nlen > buf.len() || nlen == 0 {
            return Err(crate::Error::InvalidImage(
                "apfs: drec name length oob".into(),
            ));
        }
        let raw_name = &buf[10..10 + nlen];
        let end = raw_name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(raw_name.len());
        let name = String::from_utf8_lossy(&raw_name[..end]).into_owned();
        Ok(Self {
            parent_oid: oid,
            name,
        })
    }
}

/// j_drec_val_t fixed portion (size = 0x12 / 18 bytes).
/// ```text
///   0  file_id                  u64     object id of the target inode
///   8  date_added               u64
///  16  flags                    u16     low 8 bits = DT_* type
///  18  xfields[]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DrecVal {
    pub file_id: u64,
    pub flags: u16,
}

impl DrecVal {
    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < 18 {
            return Err(crate::Error::InvalidImage(
                "apfs: drec_val too short".into(),
            ));
        }
        Ok(Self {
            file_id: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            flags: u16::from_le_bytes(buf[16..18].try_into().unwrap()),
        })
    }

    /// Extract the DT_* file-type code.
    pub fn dtype(&self) -> u16 {
        self.flags & DREC_TYPE_MASK
    }
}

/// j_file_extent_key_t layout:
/// ```text
///   0  hdr (j_key_t)            u64
///   8  logical_addr             u64
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FileExtentKey {
    pub oid: u64,
    pub logical_addr: u64,
}

impl FileExtentKey {
    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < 16 {
            return Err(crate::Error::InvalidImage(
                "apfs: file_extent key too short".into(),
            ));
        }
        let raw = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let (kind, oid) = split_obj_id(raw);
        if kind != APFS_TYPE_FILE_EXTENT {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: file_extent key kind {kind} != FILE_EXTENT"
            )));
        }
        let logical_addr = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        Ok(Self { oid, logical_addr })
    }
}

/// j_file_extent_val_t layout (24 bytes).
/// ```text
///   0  len_and_flags    u64    low 56 bits = length, high 8 = flags
///   8  phys_block_num   u64    zero ⇒ sparse extent (all zero bytes)
///  16  crypto_id        u64
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FileExtentVal {
    pub length: u64,
    pub phys_block_num: u64,
}

impl FileExtentVal {
    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < 24 {
            return Err(crate::Error::InvalidImage(
                "apfs: file_extent val too short".into(),
            ));
        }
        let lf = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let pbn = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        Ok(Self {
            length: lf & J_FILE_EXTENT_LEN_MASK,
            phys_block_num: pbn,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_obj_id_separates_kind_and_oid() {
        let raw = (3u64 << 60) | 0x42;
        assert_eq!(split_obj_id(raw), (3, 0x42));
        let raw = (8u64 << 60) | 0x0fff_ffff_ffff_ffff;
        assert_eq!(split_obj_id(raw), (8, 0x0fff_ffff_ffff_ffff));
    }

    #[test]
    fn decode_drec_hashed_key_with_name() {
        let mut buf = [0u8; 64];
        let raw = ((APFS_TYPE_DIR_REC as u64) << 60) | 0x123;
        buf[0..8].copy_from_slice(&raw.to_le_bytes());
        let name = b"hello\0";
        let nh = name.len() as u32; // low bits are length incl. NUL
        buf[8..12].copy_from_slice(&nh.to_le_bytes());
        buf[12..12 + name.len()].copy_from_slice(name);
        // Truncate to actual size so we don't decode past the name.
        let key = DrecKey::decode_hashed(&buf[..12 + name.len()]).unwrap();
        assert_eq!(key.parent_oid, 0x123);
        assert_eq!(key.name, "hello");
    }

    #[test]
    fn decode_drec_val_dtype() {
        let mut buf = vec![0u8; 18];
        buf[0..8].copy_from_slice(&0xfeedu64.to_le_bytes());
        buf[16..18].copy_from_slice(&(DT_REG | 0x0200).to_le_bytes());
        let v = DrecVal::decode(&buf).unwrap();
        assert_eq!(v.file_id, 0xfeed);
        assert_eq!(v.dtype(), DT_REG);
    }

    #[test]
    fn decode_file_extent_key_val() {
        let mut kb = vec![0u8; 16];
        let raw = ((APFS_TYPE_FILE_EXTENT as u64) << 60) | 0x100;
        kb[0..8].copy_from_slice(&raw.to_le_bytes());
        kb[8..16].copy_from_slice(&4096u64.to_le_bytes());
        let k = FileExtentKey::decode(&kb).unwrap();
        assert_eq!(k.oid, 0x100);
        assert_eq!(k.logical_addr, 4096);

        let mut vb = vec![0u8; 24];
        // length = 8192, flags = 0x10
        let lf = 8192u64 | (0x10u64 << 56);
        vb[0..8].copy_from_slice(&lf.to_le_bytes());
        vb[8..16].copy_from_slice(&0x5000u64.to_le_bytes());
        let v = FileExtentVal::decode(&vb).unwrap();
        assert_eq!(v.length, 8192);
        assert_eq!(v.phys_block_num, 0x5000);
    }

    /// Build an inode value with two xfields (NAME + DSTREAM) and confirm
    /// they're extracted correctly. xfields blob starts at byte 92.
    #[test]
    fn decode_inode_with_xfields() {
        let mut buf = vec![0u8; J_INODE_VAL_FIXED_SIZE + 64];
        // Fill fixed fields.
        buf[0..8].copy_from_slice(&1u64.to_le_bytes()); // parent_id
        buf[8..16].copy_from_slice(&7u64.to_le_bytes()); // private_id
        buf[80..82].copy_from_slice(&0o0100644u16.to_le_bytes()); // mode (S_IFREG | 0644)

        let xoff = J_INODE_VAL_FIXED_SIZE;
        // xf_blob_t: num_exts = 2, used_data = 8 (name) + 40 (dstream) = 48
        buf[xoff..xoff + 2].copy_from_slice(&2u16.to_le_bytes());
        buf[xoff + 2..xoff + 4].copy_from_slice(&48u16.to_le_bytes());
        // x_field[0] = NAME, size 6 ("abc\0\0\0" padded to 6 bytes)
        buf[xoff + 4] = INO_EXT_TYPE_NAME;
        buf[xoff + 5] = 0;
        buf[xoff + 6..xoff + 8].copy_from_slice(&6u16.to_le_bytes());
        // x_field[1] = DSTREAM, size 40
        buf[xoff + 8] = INO_EXT_TYPE_DSTREAM;
        buf[xoff + 9] = 0;
        buf[xoff + 10..xoff + 12].copy_from_slice(&40u16.to_le_bytes());
        // Values start at xoff + 12.
        let voff = xoff + 12;
        // NAME bytes "abc\0" + 2 padding to fill 6 bytes.
        buf[voff..voff + 4].copy_from_slice(b"abc\0");
        // After NAME (6 bytes), align to 8 — cursor goes to voff+8.
        // DSTREAM at voff + 8: size=1000, alloced_size=4096
        let dstream_off = voff + 8;
        buf[dstream_off..dstream_off + 8].copy_from_slice(&1000u64.to_le_bytes());
        buf[dstream_off + 8..dstream_off + 16].copy_from_slice(&4096u64.to_le_bytes());

        let ino = InodeVal::decode(&buf).unwrap();
        assert_eq!(ino.parent_id, 1);
        assert_eq!(ino.private_id, 7);
        assert_eq!(ino.mode, 0o0100644);
        assert_eq!(ino.name.as_deref(), Some("abc"));
        let d = ino.dstream.expect("dstream xf should be present");
        assert_eq!(d.size, 1000);
        assert_eq!(d.alloced_size, 4096);
    }
}
