//! ext2/3/4 superblock — typed representation + encode/decode.
//!
//! The on-disk superblock is 1024 bytes regardless of `s_inode_size`. We
//! cover the classic ext2 fields plus the dynamic-rev extensions; ext3/4
//! adds more fields (most of the second half of the structure), but the
//! v1 ext2 writer only touches what's documented here. Anything we don't
//! write is left zero, which matches a freshly-formatted ext2 from
//! `mke2fs -t ext2`.

use super::constants::{
    EXT2_MAGIC, FIRST_INO_DYNAMIC, FS_VALID, INODE_SIZE_DYNAMIC, OS_LINUX, REV_DYNAMIC,
    SUPERBLOCK_SIZE,
};
use crate::Result;

/// Typed superblock. All fields are in host order; encode/decode handle the
/// little-endian on-disk representation.
#[derive(Debug, Clone)]
pub struct Superblock {
    pub inodes_count: u32,
    pub blocks_count: u32,
    pub r_blocks_count: u32,
    pub free_blocks_count: u32,
    pub free_inodes_count: u32,
    pub first_data_block: u32,
    pub log_block_size: u32, // 0 → 1 KiB, 1 → 2 KiB, 2 → 4 KiB
    pub log_frag_size: u32,
    pub blocks_per_group: u32,
    pub frags_per_group: u32,
    pub inodes_per_group: u32,
    pub mtime: u32,
    pub wtime: u32,
    pub mnt_count: u16,
    pub max_mnt_count: u16,
    pub magic: u16,
    pub state: u16,
    pub errors: u16,
    pub minor_rev_level: u16,
    pub lastcheck: u32,
    pub checkinterval: u32,
    pub creator_os: u32,
    pub rev_level: u32,
    pub def_resuid: u16,
    pub def_resgid: u16,
    // DYNAMIC_REV extensions (only meaningful when rev_level == REV_DYNAMIC):
    pub first_ino: u32,
    pub inode_size: u16,
    pub block_group_nr: u16,
    pub feature_compat: u32,
    pub feature_incompat: u32,
    pub feature_ro_compat: u32,
    pub uuid: [u8; 16],
    pub volume_name: [u8; 16],
    pub last_mounted: [u8; 64],
    pub algorithm_usage_bitmap: u32,
    /// Inode number of the journal (only meaningful with the HAS_JOURNAL
    /// compat feature). 0 for ext2.
    pub journal_inum: u32,
    /// On-disk size of each group descriptor. 0 means the classic 32-byte
    /// form; when `INCOMPAT_64BIT` is set this is 64 (or larger).
    pub desc_size: u16,
    /// `s_log_groups_per_flex` at offset 0x174. Base-2 log of the number
    /// of groups packed into one flex unit when `INCOMPAT_FLEX_BG` is
    /// active. 0 means the classic one-group-at-a-time metadata layout.
    pub log_groups_per_flex: u8,
}

impl Superblock {
    /// Effective group-descriptor size in bytes: `desc_size` if non-zero,
    /// otherwise the classic 32.
    pub fn group_desc_size(&self) -> usize {
        if self.desc_size == 0 {
            32
        } else {
            self.desc_size as usize
        }
    }
}

impl Superblock {
    /// Build a default superblock suitable for ext2 (no features). Caller
    /// must then fill in counts and sizes.
    pub fn ext2_default() -> Self {
        Self {
            inodes_count: 0,
            blocks_count: 0,
            r_blocks_count: 0,
            free_blocks_count: 0,
            free_inodes_count: 0,
            first_data_block: 0,
            log_block_size: 0,
            log_frag_size: 0,
            blocks_per_group: 0,
            frags_per_group: 0,
            inodes_per_group: 0,
            mtime: 0,
            wtime: 0,
            mnt_count: 0,
            max_mnt_count: 20,
            magic: EXT2_MAGIC,
            state: FS_VALID,
            // genext2fs sets s_errors to 0 ("undefined" in dumpe2fs); the
            // kernel treats it as the default behaviour (continue). We
            // match for byte-exact compatibility.
            errors: 0,
            minor_rev_level: 0,
            lastcheck: 0,
            checkinterval: 0,
            creator_os: OS_LINUX,
            rev_level: REV_DYNAMIC,
            def_resuid: 0,
            def_resgid: 0,
            first_ino: FIRST_INO_DYNAMIC,
            inode_size: INODE_SIZE_DYNAMIC,
            block_group_nr: 0,
            feature_compat: 0,
            feature_incompat: 0,
            feature_ro_compat: 0,
            uuid: [0; 16],
            volume_name: [0; 16],
            last_mounted: [0; 64],
            algorithm_usage_bitmap: 0,
            journal_inum: 0,
            desc_size: 0,
            log_groups_per_flex: 0,
        }
    }

    /// Block size in bytes derived from `log_block_size`.
    pub fn block_size(&self) -> u32 {
        1024u32 << self.log_block_size
    }

    /// Number of block groups: ceil(blocks_count / blocks_per_group).
    pub fn group_count(&self) -> u32 {
        self.blocks_count.div_ceil(self.blocks_per_group)
    }

    /// Encode into the 1024-byte on-disk representation.
    pub fn encode(&self) -> [u8; SUPERBLOCK_SIZE] {
        let mut buf = [0u8; SUPERBLOCK_SIZE];
        let p = &mut buf;
        write_u32(p, 0, self.inodes_count);
        write_u32(p, 4, self.blocks_count);
        write_u32(p, 8, self.r_blocks_count);
        write_u32(p, 12, self.free_blocks_count);
        write_u32(p, 16, self.free_inodes_count);
        write_u32(p, 20, self.first_data_block);
        write_u32(p, 24, self.log_block_size);
        write_u32(p, 28, self.log_frag_size);
        write_u32(p, 32, self.blocks_per_group);
        write_u32(p, 36, self.frags_per_group);
        write_u32(p, 40, self.inodes_per_group);
        write_u32(p, 44, self.mtime);
        write_u32(p, 48, self.wtime);
        write_u16(p, 52, self.mnt_count);
        write_u16(p, 54, self.max_mnt_count);
        write_u16(p, 56, self.magic);
        write_u16(p, 58, self.state);
        write_u16(p, 60, self.errors);
        write_u16(p, 62, self.minor_rev_level);
        write_u32(p, 64, self.lastcheck);
        write_u32(p, 68, self.checkinterval);
        write_u32(p, 72, self.creator_os);
        write_u32(p, 76, self.rev_level);
        write_u16(p, 80, self.def_resuid);
        write_u16(p, 82, self.def_resgid);
        write_u32(p, 84, self.first_ino);
        write_u16(p, 88, self.inode_size);
        write_u16(p, 90, self.block_group_nr);
        write_u32(p, 92, self.feature_compat);
        write_u32(p, 96, self.feature_incompat);
        write_u32(p, 100, self.feature_ro_compat);
        p[104..120].copy_from_slice(&self.uuid);
        p[120..136].copy_from_slice(&self.volume_name);
        p[136..200].copy_from_slice(&self.last_mounted);
        write_u32(p, 200, self.algorithm_usage_bitmap);
        // 204: s_prealloc_blocks (u8) — left zero
        // 205: s_prealloc_dir_blocks (u8) — left zero
        // 206..208: s_padding1 (u16) — left zero
        // 208..224: s_journal_uuid — left zero
        write_u32(p, 224, self.journal_inum);
        // 228..252: s_journal_dev, s_last_orphan, s_hash_seed — left zero
        // 252: s_def_hash_version (u8), 253: s_jnl_backup_type (u8) — zero
        write_u16(p, 254, self.desc_size);
        // 256..0x174: s_default_mount_opts, s_first_meta_bg, s_mkfs_time,
        // s_jnl_blocks — left zero.
        // 0x174: s_log_groups_per_flex (u8).
        p[0x174] = self.log_groups_per_flex;
        // 0x175..: s_checksum_type, padding, ... — left zero. (The
        // metadata-checksum path in mod.rs sets 0x175 directly when
        // needed.)
        buf
    }

    /// Decode from a 1024-byte on-disk representation. Validates the magic.
    pub fn decode(buf: &[u8; SUPERBLOCK_SIZE]) -> Result<Self> {
        let magic = read_u16(buf, 56);
        if magic != EXT2_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "ext: bad superblock magic {magic:#06x}, expected {EXT2_MAGIC:#06x}"
            )));
        }
        let rev_level = read_u32(buf, 76);
        let (first_ino, inode_size) = if rev_level == 0 {
            // good_old rev: these fields have implicit values.
            (11, 128)
        } else {
            (read_u32(buf, 84), read_u16(buf, 88))
        };
        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(&buf[104..120]);
        let mut volume_name = [0u8; 16];
        volume_name.copy_from_slice(&buf[120..136]);
        let mut last_mounted = [0u8; 64];
        last_mounted.copy_from_slice(&buf[136..200]);
        Ok(Self {
            inodes_count: read_u32(buf, 0),
            blocks_count: read_u32(buf, 4),
            r_blocks_count: read_u32(buf, 8),
            free_blocks_count: read_u32(buf, 12),
            free_inodes_count: read_u32(buf, 16),
            first_data_block: read_u32(buf, 20),
            log_block_size: read_u32(buf, 24),
            log_frag_size: read_u32(buf, 28),
            blocks_per_group: read_u32(buf, 32),
            frags_per_group: read_u32(buf, 36),
            inodes_per_group: read_u32(buf, 40),
            mtime: read_u32(buf, 44),
            wtime: read_u32(buf, 48),
            mnt_count: read_u16(buf, 52),
            max_mnt_count: read_u16(buf, 54),
            magic,
            state: read_u16(buf, 58),
            errors: read_u16(buf, 60),
            minor_rev_level: read_u16(buf, 62),
            lastcheck: read_u32(buf, 64),
            checkinterval: read_u32(buf, 68),
            creator_os: read_u32(buf, 72),
            rev_level,
            def_resuid: read_u16(buf, 80),
            def_resgid: read_u16(buf, 82),
            first_ino,
            inode_size,
            block_group_nr: read_u16(buf, 90),
            feature_compat: read_u32(buf, 92),
            feature_incompat: read_u32(buf, 96),
            feature_ro_compat: read_u32(buf, 100),
            uuid,
            volume_name,
            last_mounted,
            algorithm_usage_bitmap: read_u32(buf, 200),
            journal_inum: read_u32(buf, 224),
            desc_size: read_u16(buf, 254),
            log_groups_per_flex: buf[0x174],
        })
    }
}

#[inline]
fn write_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

#[inline]
fn write_u16(buf: &mut [u8], off: usize, v: u16) {
    buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

#[inline]
fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

#[inline]
fn read_u16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(buf[off..off + 2].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_default() {
        let mut sb = Superblock::ext2_default();
        sb.inodes_count = 1024;
        sb.blocks_count = 8192;
        sb.free_blocks_count = 7000;
        sb.free_inodes_count = 1013;
        sb.log_block_size = 0;
        sb.blocks_per_group = 8192;
        sb.frags_per_group = 8192;
        sb.inodes_per_group = 1024;
        sb.first_data_block = 1;
        sb.uuid = [0x42; 16];
        let buf = sb.encode();
        let decoded = Superblock::decode(&buf).unwrap();
        assert_eq!(decoded.inodes_count, sb.inodes_count);
        assert_eq!(decoded.blocks_count, sb.blocks_count);
        assert_eq!(decoded.uuid, sb.uuid);
        assert_eq!(decoded.magic, EXT2_MAGIC);
        assert_eq!(decoded.block_size(), 1024);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = [0u8; SUPERBLOCK_SIZE];
        // No magic written
        let err = Superblock::decode(&buf).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidImage(_)));
        // Write wrong magic
        buf[56..58].copy_from_slice(&0x1234u16.to_le_bytes());
        let err = Superblock::decode(&buf).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidImage(_)));
    }
}
