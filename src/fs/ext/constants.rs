//! Constants drawn from the ext2/3/4 on-disk format specification.
//!
//! References:
//! - linux/fs/ext4/ext4.h (canonical superblock + inode + dirent layout)
//! - linux/fs/ext2/ext2.h (the ext2 subset)
//! - e2fsprogs lib/ext2fs/ext2_fs.h
//!
//! Only the constants we actually use are defined. Add more as the writer
//! grows.

/// Superblock magic number at offset 0x38 of the superblock.
pub const EXT2_MAGIC: u16 = 0xEF53;

/// Byte offset of the primary superblock from the start of the device.
/// (Inside block 0 if `block_size >= 1024`, or in block 1 if 1 KiB blocks.)
pub const SUPERBLOCK_OFFSET: u64 = 1024;

/// Fixed superblock size — even if `s_inode_size` is larger than 128 the
/// superblock itself is always 1024 bytes.
pub const SUPERBLOCK_SIZE: usize = 1024;

/// Size of a group descriptor in classic ext2/3 (32 bytes). When the
/// `INCOMPAT_64BIT` feature is set, descriptors are 64 bytes and the
/// superblock's `s_desc_size` field records the actual size.
pub const GROUP_DESC_SIZE: usize = 32;

/// Size of a 64-bit (`INCOMPAT_64BIT`) group descriptor.
pub const GROUP_DESC_SIZE_64: usize = 64;

/// Default and minimum size of an inode in ext2 (the "good old" rev).
pub const INODE_SIZE_GOOD_OLD: u16 = 128;

/// Inode size used in DYNAMIC_REV (rev 1). 128 keeps us format-compatible
/// with ext2 tooling.
pub const INODE_SIZE_DYNAMIC: u16 = 128;

/// Revision levels.
pub const REV_GOOD_OLD: u32 = 0;
pub const REV_DYNAMIC: u32 = 1;

/// First non-reserved inode in dynamic rev. ext2 reserves inodes 1..=10.
pub const FIRST_INO_DYNAMIC: u32 = 11;

/// Reserved inode numbers.
pub const INO_BAD_BLOCKS: u32 = 1;
pub const INO_ROOT_DIR: u32 = 2;
pub const INO_USER_QUOTA: u32 = 3;
pub const INO_GROUP_QUOTA: u32 = 4;
pub const INO_BOOT_LOADER: u32 = 5;
pub const INO_UNDELETE_DIR: u32 = 6;
pub const INO_RESIZE: u32 = 7;
pub const INO_JOURNAL: u32 = 8;

/// Volume state bits.
pub const FS_VALID: u16 = 0x0001;

/// On-error behaviour.
pub const ERRORS_CONTINUE: u16 = 1;

/// Creator OS values.
pub const OS_LINUX: u32 = 0;

/// File mode bits (S_IFMT / S_IF*).
pub const S_IFMT: u16 = 0o170000;
pub const S_IFSOCK: u16 = 0o140000;
pub const S_IFLNK: u16 = 0o120000;
pub const S_IFREG: u16 = 0o100000;
pub const S_IFBLK: u16 = 0o060000;
pub const S_IFDIR: u16 = 0o040000;
pub const S_IFCHR: u16 = 0o020000;
pub const S_IFIFO: u16 = 0o010000;

/// Directory-entry filetype bytes (used only when the FILETYPE incompat
/// feature is enabled — not in our default ext2 build, but defined so we can
/// turn it on for ext4 later).
pub const DENT_UNKNOWN: u8 = 0;
pub const DENT_REG: u8 = 1;
pub const DENT_DIR: u8 = 2;
pub const DENT_CHR: u8 = 3;
pub const DENT_BLK: u8 = 4;
pub const DENT_FIFO: u8 = 5;
pub const DENT_SOCK: u8 = 6;
pub const DENT_LNK: u8 = 7;

/// Feature flag groups (defined for future ext3/ext4 wiring).
pub mod feature {
    // compat
    pub const COMPAT_DIR_PREALLOC: u32 = 0x0001;
    pub const COMPAT_IMAGIC_INODES: u32 = 0x0002;
    pub const COMPAT_HAS_JOURNAL: u32 = 0x0004;
    pub const COMPAT_EXT_ATTR: u32 = 0x0008;
    pub const COMPAT_RESIZE_INODE: u32 = 0x0010;
    pub const COMPAT_DIR_INDEX: u32 = 0x0020;

    // incompat
    pub const INCOMPAT_COMPRESSION: u32 = 0x0001;
    pub const INCOMPAT_FILETYPE: u32 = 0x0002;
    pub const INCOMPAT_RECOVER: u32 = 0x0004;
    pub const INCOMPAT_JOURNAL_DEV: u32 = 0x0008;
    pub const INCOMPAT_META_BG: u32 = 0x0010;
    pub const INCOMPAT_EXTENTS: u32 = 0x0040;
    pub const INCOMPAT_64BIT: u32 = 0x0080;
    pub const INCOMPAT_FLEX_BG: u32 = 0x0200;

    // ro_compat
    pub const RO_COMPAT_SPARSE_SUPER: u32 = 0x0001;
    pub const RO_COMPAT_LARGE_FILE: u32 = 0x0002;
    pub const RO_COMPAT_BTREE_DIR: u32 = 0x0004;
    pub const RO_COMPAT_HUGE_FILE: u32 = 0x0008;
    pub const RO_COMPAT_GDT_CSUM: u32 = 0x0010;
    pub const RO_COMPAT_DIR_NLINK: u32 = 0x0020;
    pub const RO_COMPAT_EXTRA_ISIZE: u32 = 0x0040;
    pub const RO_COMPAT_METADATA_CSUM: u32 = 0x0400;
}

/// Inode flag `EXT4_EXTENTS_FL` — set on inodes whose `i_block` array
/// holds an ext4 extent tree rather than direct/indirect block pointers.
pub const EXT4_EXTENTS_FL: u32 = 0x0008_0000;

/// Number of direct block pointers in an inode (`i_block[0..12]`).
pub const N_DIRECT: usize = 12;
/// Index of the single-indirect block in `i_block`.
pub const IDX_INDIRECT: usize = 12;
/// Index of the double-indirect block in `i_block`.
pub const IDX_DOUBLE_INDIRECT: usize = 13;
/// Index of the triple-indirect block in `i_block`.
pub const IDX_TRIPLE_INDIRECT: usize = 14;
/// Total slots in `i_block`.
pub const N_BLOCKS: usize = 15;
