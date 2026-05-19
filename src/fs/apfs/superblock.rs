//! Container (`nx_superblock_t`) and volume (`apfs_superblock_t`) superblock
//! decoders.
//!
//! Reference: *Apple File System Reference*, sections "Container" and
//! "Volumes". We only decode the fields we actually consume; the structures
//! are large and most fields are uninteresting for read-only listing.
//!
//! ```text
//! nx_superblock_t (truncated to fields we use)
//!    0   nx_o                  obj_phys_t (32 B)
//!   32   nx_magic              u32  = 'NXSB' (0x4253584E LE)
//!   36   nx_block_size         u32
//!   40   nx_block_count        u64
//!   48   nx_features           u64
//!   56   nx_readonly_compatible_features u64
//!   64   nx_incompatible_features u64
//!   72   nx_uuid               u8[16]
//!   88   nx_next_oid           u64
//!   96   nx_next_xid           u64
//!  104   nx_xp_desc_blocks     u32
//!  108   nx_xp_data_blocks     u32
//!  112   nx_xp_desc_base       u64
//!  120   nx_xp_data_base       u64
//!  128   nx_xp_desc_next       u32
//!  132   nx_xp_data_next       u32
//!  136   nx_xp_desc_index      u32
//!  140   nx_xp_desc_len        u32
//!  144   nx_xp_data_index      u32
//!  148   nx_xp_data_len        u32
//!  152   nx_spaceman_oid       u64
//!  160   nx_omap_oid           u64
//!  168   nx_reaper_oid         u64
//!  176   nx_test_type          u32
//!  180   nx_max_file_systems   u32
//!  184   nx_fs_oid[NX_MAX_FILE_SYSTEMS=100] u64  → ends at 184+800 = 984
//! ```

use super::obj::{ObjPhys, OBJECT_TYPE_NX_SUPERBLOCK};

/// Magic value of `nx_magic` ("NXSB" read little-endian).
pub const NX_MAGIC: u32 = 0x4253_584e;
/// Magic value of `apfs_magic` ("APSB" read little-endian).
pub const APFS_MAGIC: u32 = 0x4253_5041;

/// Maximum number of volumes a single container can hold.
pub const NX_MAX_FILE_SYSTEMS: usize = 100;

/// Container superblock — the root object of an APFS container.
#[derive(Debug, Clone)]
pub struct NxSuperblock {
    pub obj: ObjPhys,
    pub magic: u32,
    pub block_size: u32,
    pub block_count: u64,
    pub features: u64,
    pub readonly_compatible_features: u64,
    pub incompatible_features: u64,
    pub uuid: [u8; 16],
    pub next_oid: u64,
    pub next_xid: u64,
    pub xp_desc_blocks: u32,
    pub xp_desc_base: u64,
    pub xp_desc_index: u32,
    pub xp_desc_len: u32,
    pub omap_oid: u64,
    pub max_file_systems: u32,
    /// First N slots of `nx_fs_oid[]`, copied out. APFS only uses the
    /// active slots (typically just the first one for a single-volume
    /// container); zeros mean "no volume". We keep the full 100-entry
    /// array because the spec says callers must scan all slots.
    pub fs_oid: [u64; NX_MAX_FILE_SYSTEMS],
}

impl NxSuperblock {
    /// Minimum on-disk size we need to decode the fields above.
    pub const MIN_SIZE: usize = 984;

    /// Decode a container superblock from `buf`. Validates the magic and the
    /// `o_type` field. Does NOT validate the Fletcher-64 checksum; callers
    /// may verify with [`super::checksum::verify`] if they care.
    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < Self::MIN_SIZE {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: nx_superblock buffer too short ({} < {})",
                buf.len(),
                Self::MIN_SIZE
            )));
        }
        let obj = ObjPhys::decode(buf)?;
        if obj.obj_type() != OBJECT_TYPE_NX_SUPERBLOCK {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: o_type {:#x} is not NX_SUPERBLOCK",
                obj.obj_type()
            )));
        }
        let magic = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        if magic != NX_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: nx_magic {magic:#010x} != 'NXSB'"
            )));
        }
        let block_size = u32::from_le_bytes(buf[36..40].try_into().unwrap());
        let block_count = u64::from_le_bytes(buf[40..48].try_into().unwrap());
        let features = u64::from_le_bytes(buf[48..56].try_into().unwrap());
        let ro_compat = u64::from_le_bytes(buf[56..64].try_into().unwrap());
        let incompat = u64::from_le_bytes(buf[64..72].try_into().unwrap());
        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(&buf[72..88]);
        let next_oid = u64::from_le_bytes(buf[88..96].try_into().unwrap());
        let next_xid = u64::from_le_bytes(buf[96..104].try_into().unwrap());
        let xp_desc_blocks = u32::from_le_bytes(buf[104..108].try_into().unwrap());
        let xp_desc_base = u64::from_le_bytes(buf[112..120].try_into().unwrap());
        let xp_desc_index = u32::from_le_bytes(buf[136..140].try_into().unwrap());
        let xp_desc_len = u32::from_le_bytes(buf[140..144].try_into().unwrap());
        let omap_oid = u64::from_le_bytes(buf[160..168].try_into().unwrap());
        let max_file_systems = u32::from_le_bytes(buf[180..184].try_into().unwrap());

        let mut fs_oid = [0u64; NX_MAX_FILE_SYSTEMS];
        for (i, slot) in fs_oid.iter_mut().enumerate() {
            let off = 184 + i * 8;
            *slot = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        }

        Ok(Self {
            obj,
            magic,
            block_size,
            block_count,
            features,
            readonly_compatible_features: ro_compat,
            incompatible_features: incompat,
            uuid,
            next_oid,
            next_xid,
            xp_desc_blocks,
            xp_desc_base,
            xp_desc_index,
            xp_desc_len,
            omap_oid,
            max_file_systems,
            fs_oid,
        })
    }
}

/// Volume superblock — root of one APFS volume inside a container.
///
/// ```text
/// apfs_superblock_t (truncated to fields we use)
///    0   apfs_o                obj_phys_t
///   32   apfs_magic            u32 = 'APSB' (0x42535041 LE)
///   36   apfs_fs_index         u32
///   40   apfs_features         u64
///   48   apfs_readonly_compatible_features u64
///   56   apfs_incompatible_features u64
///   64   apfs_unmount_time     u64
///   72   apfs_fs_reserve_block_count u64
///   80   apfs_fs_quota_block_count u64
///   88   apfs_fs_alloc_count   u64
///   96   apfs_meta_crypto      wrapped_meta_crypto_state_t (20 bytes)
///  116   apfs_root_tree_type   u32
///  120   apfs_extentref_tree_type u32
///  124   apfs_snap_meta_tree_type u32
///  128   apfs_omap_oid         oid_t (u64)
///  136   apfs_root_tree_oid    oid_t (u64)
///  144   apfs_extentref_tree_oid oid_t (u64)
///  152   apfs_snap_meta_tree_oid oid_t (u64)
///  160   apfs_revert_to_xid    xid_t (u64)
///  168   apfs_revert_to_sblock_oid oid_t (u64)
///  176   apfs_next_obj_id      u64
///  184   apfs_num_files        u64
///  192   apfs_num_directories  u64
///  200   apfs_num_symlinks     u64
///  208   apfs_num_other_fsobjects u64
///  216   apfs_num_snapshots    u64
///  224   apfs_total_blocks_alloced u64
///  232   apfs_total_blocks_freed u64
///  240   apfs_vol_uuid         u8[16]
///  256   apfs_last_mod_time    u64
///  264   apfs_fs_flags         u64
///  272   apfs_formatted_by     apfs_modified_by_t (48 bytes: u8[32] id + u64 timestamp + u64 last_xid)
///  320   apfs_modified_by      apfs_modified_by_t[8] (48 * 8 = 384 bytes)
///  704   apfs_volname          u8[256]   NUL-terminated UTF-8
///  960   apfs_next_doc_id      u32
///  964   apfs_role             u16
/// ```
#[derive(Debug, Clone)]
pub struct ApfsSuperblock {
    pub obj: ObjPhys,
    pub magic: u32,
    pub fs_index: u32,
    pub features: u64,
    pub readonly_compatible_features: u64,
    pub incompatible_features: u64,
    pub root_tree_type: u32,
    pub extentref_tree_type: u32,
    pub snap_meta_tree_type: u32,
    pub omap_oid: u64,
    pub root_tree_oid: u64,
    pub extentref_tree_oid: u64,
    pub snap_meta_tree_oid: u64,
    pub num_files: u64,
    pub num_directories: u64,
    pub num_symlinks: u64,
    pub vol_uuid: [u8; 16],
    pub fs_flags: u64,
    /// Volume name, UTF-8, NUL-terminated. We trim at the first NUL.
    pub volname: String,
}

impl ApfsSuperblock {
    /// Minimum on-disk size we need.
    pub const MIN_SIZE: usize = 960;

    pub fn decode(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < Self::MIN_SIZE {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: apfs_superblock buffer too short ({} < {})",
                buf.len(),
                Self::MIN_SIZE
            )));
        }
        let obj = ObjPhys::decode(buf)?;
        // apfs_superblock typically has o_type = OBJECT_TYPE_FS | OBJ_VIRTUAL,
        // so we check via magic instead — more robust.
        let magic = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        if magic != APFS_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: apfs_magic {magic:#010x} != 'APSB'"
            )));
        }
        let fs_index = u32::from_le_bytes(buf[36..40].try_into().unwrap());
        let features = u64::from_le_bytes(buf[40..48].try_into().unwrap());
        let ro_compat = u64::from_le_bytes(buf[48..56].try_into().unwrap());
        let incompat = u64::from_le_bytes(buf[56..64].try_into().unwrap());
        let root_tree_type = u32::from_le_bytes(buf[116..120].try_into().unwrap());
        let extentref_tree_type = u32::from_le_bytes(buf[120..124].try_into().unwrap());
        let snap_meta_tree_type = u32::from_le_bytes(buf[124..128].try_into().unwrap());
        let omap_oid = u64::from_le_bytes(buf[128..136].try_into().unwrap());
        let root_tree_oid = u64::from_le_bytes(buf[136..144].try_into().unwrap());
        let extentref_tree_oid = u64::from_le_bytes(buf[144..152].try_into().unwrap());
        let snap_meta_tree_oid = u64::from_le_bytes(buf[152..160].try_into().unwrap());
        let num_files = u64::from_le_bytes(buf[184..192].try_into().unwrap());
        let num_directories = u64::from_le_bytes(buf[192..200].try_into().unwrap());
        let num_symlinks = u64::from_le_bytes(buf[200..208].try_into().unwrap());
        let mut vol_uuid = [0u8; 16];
        vol_uuid.copy_from_slice(&buf[240..256]);
        let fs_flags = u64::from_le_bytes(buf[264..272].try_into().unwrap());

        // volname is up to 256 bytes; we trim at the first NUL and lossy-
        // convert (it's already UTF-8 on disk per spec, but tolerate dirty
        // images by replacing bad sequences instead of failing).
        let raw = &buf[704..704 + 256.min(buf.len() - 704)];
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let volname = String::from_utf8_lossy(&raw[..end]).into_owned();

        Ok(Self {
            obj,
            magic,
            fs_index,
            features,
            readonly_compatible_features: ro_compat,
            incompatible_features: incompat,
            root_tree_type,
            extentref_tree_type,
            snap_meta_tree_type,
            omap_oid,
            root_tree_oid,
            extentref_tree_oid,
            snap_meta_tree_oid,
            num_files,
            num_directories,
            num_symlinks,
            vol_uuid,
            fs_flags,
            volname,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal NXSB buffer and verify the decoder picks out the
    /// fields we read. We don't bother with the checksum here — that's
    /// covered in the obj/checksum tests.
    #[test]
    fn decode_nxsb_minimal() {
        let mut buf = vec![0u8; 4096];
        // obj_phys
        buf[24..28].copy_from_slice(&OBJECT_TYPE_NX_SUPERBLOCK.to_le_bytes());
        // nx_magic
        buf[32..36].copy_from_slice(&NX_MAGIC.to_le_bytes());
        buf[36..40].copy_from_slice(&4096u32.to_le_bytes()); // block size
        buf[40..48].copy_from_slice(&1234u64.to_le_bytes()); // block count
        buf[104..108].copy_from_slice(&8u32.to_le_bytes()); // xp_desc_blocks
        buf[112..120].copy_from_slice(&1u64.to_le_bytes()); // xp_desc_base
        buf[136..140].copy_from_slice(&2u32.to_le_bytes()); // xp_desc_index
        buf[140..144].copy_from_slice(&3u32.to_le_bytes()); // xp_desc_len
        buf[160..168].copy_from_slice(&0xfeu64.to_le_bytes()); // omap_oid
        buf[180..184].copy_from_slice(&1u32.to_le_bytes()); // max_file_systems
        buf[184..192].copy_from_slice(&0x4242u64.to_le_bytes()); // fs_oid[0]

        let sb = NxSuperblock::decode(&buf).unwrap();
        assert_eq!(sb.magic, NX_MAGIC);
        assert_eq!(sb.block_size, 4096);
        assert_eq!(sb.block_count, 1234);
        assert_eq!(sb.xp_desc_blocks, 8);
        assert_eq!(sb.xp_desc_base, 1);
        assert_eq!(sb.xp_desc_index, 2);
        assert_eq!(sb.xp_desc_len, 3);
        assert_eq!(sb.omap_oid, 0xfe);
        assert_eq!(sb.max_file_systems, 1);
        assert_eq!(sb.fs_oid[0], 0x4242);
        assert_eq!(sb.fs_oid[1], 0);
    }

    #[test]
    fn decode_nxsb_rejects_bad_magic() {
        let mut buf = vec![0u8; 4096];
        buf[24..28].copy_from_slice(&OBJECT_TYPE_NX_SUPERBLOCK.to_le_bytes());
        buf[32..36].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        assert!(NxSuperblock::decode(&buf).is_err());
    }

    #[test]
    fn decode_apsb_minimal() {
        let mut buf = vec![0u8; 4096];
        buf[32..36].copy_from_slice(&APFS_MAGIC.to_le_bytes());
        buf[128..136].copy_from_slice(&0x10u64.to_le_bytes()); // omap_oid
        buf[136..144].copy_from_slice(&0x20u64.to_le_bytes()); // root_tree_oid
        // volname "Macintosh HD"
        let name = b"Macintosh HD";
        buf[704..704 + name.len()].copy_from_slice(name);
        let sb = ApfsSuperblock::decode(&buf).unwrap();
        assert_eq!(sb.magic, APFS_MAGIC);
        assert_eq!(sb.omap_oid, 0x10);
        assert_eq!(sb.root_tree_oid, 0x20);
        assert_eq!(sb.volname, "Macintosh HD");
    }
}
