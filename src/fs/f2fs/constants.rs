//! On-disk constants shared across the F2FS reader modules.
//!
//! Sources: kernel docs (`docs.kernel.org/filesystems/f2fs.html`) and the
//! FAST '15 paper. We deliberately do not import any kernel header.

/// F2FS block size in bytes. Fixed at 4 KiB.
pub const F2FS_BLKSIZE: usize = 4096;

/// Offset (within a 4 KiB CP / NAT / SIT / dnode block) of the trailing
/// CRC32 footer that protects the block. For CP blocks this is what
/// mkfs.f2fs writes into the `checksum_offset` field.
pub const F2FS_BLK_CSUM_OFFSET: usize = F2FS_BLKSIZE - 4;

/// F2FS magic; doubles as the initial CRC seed.
pub const F2FS_SUPER_MAGIC: u32 = 0xF2F5_2010;

/// Compute the F2FS CRC32 over `buf`. F2FS uses Linux's raw `crc32_le`
/// — IEEE 802.3 polynomial 0xEDB88320 but with NO initial XOR and NO
/// final XOR, seeded with `F2FS_SUPER_MAGIC` instead of the usual
/// 0xFFFFFFFF. The `crc32fast` crate's standard `hash()` is the
/// final-XOR variant (Ethernet); using it here would produce values
/// that fsck.f2fs / mkfs.f2fs reject, so we hand-roll the right one.
pub fn f2fs_crc32(buf: &[u8]) -> u32 {
    let mut crc = F2FS_SUPER_MAGIC;
    for &b in buf {
        crc = (crc >> 8) ^ F2FS_CRC32_TABLE[((crc ^ u32::from(b)) & 0xFF) as usize];
    }
    crc
}

/// Reflected IEEE 802.3 CRC32 table (polynomial 0xEDB88320). Built at
/// compile time so the codec stays zero-cost.
const F2FS_CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut c = i;
        let mut j = 0;
        while j < 8 {
            c = (c >> 1) ^ ((c & 1).wrapping_neg() & 0xEDB8_8320);
            j += 1;
        }
        table[i as usize] = c;
        i += 1;
    }
    table
};

/// Inode block — number of direct data-block pointers.
///
/// FAST '15 / kernel docs both quote 923; this is `i_addr` after the
/// 360-byte inode header and the inline xattr reservation. The on-disk
/// `i_inline` bitfield can shrink the array (inline data / inline dentry)
/// but those modes are detected at parse time, not here.
pub const ADDRS_PER_INODE: usize = 923;

/// Number of node-id slots in an inode block (2 direct, 2 indirect, 1
/// triple-indirect).
pub const NIDS_PER_INODE: usize = 5;

/// Indices into `i_nid[5]`.
pub const NID_DIRECT_1: usize = 0;
pub const NID_DIRECT_2: usize = 1;
pub const NID_INDIRECT_1: usize = 2;
pub const NID_INDIRECT_2: usize = 3;
pub const NID_TRIPLE_INDIRECT: usize = 4;

/// Direct / indirect node blocks each hold 1018 u32 entries.
pub const ADDRS_PER_BLOCK: usize = 1018;
pub const NIDS_PER_BLOCK: usize = 1018;

/// NAT entry size on disk: version(1) + ino(4) + block_addr(4) = 9 bytes.
pub const NAT_ENTRY_SIZE: usize = 9;
/// NAT entries per NAT page: floor(4096 / 9) = 455.
pub const NAT_ENTRY_PER_BLOCK: usize = F2FS_BLKSIZE / NAT_ENTRY_SIZE;

/// Dentry block layout — 27-byte bitmap + 3 reserved + 11 × 214 dentries +
/// 8 × 214 filenames = 4096 bytes total.
pub const NR_DENTRY_IN_BLOCK: usize = 214;
pub const SIZE_OF_DIR_ENTRY: usize = 11;
pub const SIZE_OF_DENTRY_BITMAP: usize = 27;
pub const SIZE_OF_RESERVED: usize = 3;
pub const F2FS_SLOT_LEN: usize = 8;

/// Bits in `F2fsInode::i_inline`.
pub const F2FS_INLINE_DATA: u8 = 0x02;
pub const F2FS_INLINE_DENTRY: u8 = 0x04;
pub const F2FS_INLINE_XATTR: u8 = 0x01;
pub const F2FS_DATA_EXIST: u8 = 0x08;

/// Checkpoint flag bits we care about (others ignored).
pub const CP_COMPACT_SUM_FLAG: u32 = 0x0001;
pub const CP_ORPHAN_PRESENT_FLAG: u32 = 0x0002;
pub const CP_UMOUNT_FLAG: u32 = 0x0004;
pub const CP_FASTBOOT_FLAG: u32 = 0x0008;
pub const CP_CRC_RECOVERY_FLAG: u32 = 0x0010;

/// Reserved/special block addresses. Anything `>= 1` and `< NEW_ADDR`
/// is a real allocation; the two specials below mark "not yet on disk"
/// (data is in the inode's inline area) and "explicit hole".
pub const NULL_ADDR: u32 = 0;
pub const NEW_ADDR: u32 = u32::MAX - 1;

/// Standard POSIX type bits in `i_mode`.
pub const S_IFMT: u16 = 0xF000;
pub const S_IFREG: u16 = 0x8000;
pub const S_IFDIR: u16 = 0x4000;
pub const S_IFLNK: u16 = 0xA000;
pub const S_IFCHR: u16 = 0x2000;
pub const S_IFBLK: u16 = 0x6000;
pub const S_IFIFO: u16 = 0x1000;
pub const S_IFSOCK: u16 = 0xC000;

/// On-disk file types stored in the 1-byte `file_type` field of a
/// `f2fs_dir_entry`. Numbering matches the standard d_type set so it
/// reads naturally.
pub const F2FS_FT_UNKNOWN: u8 = 0;
pub const F2FS_FT_REG_FILE: u8 = 1;
pub const F2FS_FT_DIR: u8 = 2;
pub const F2FS_FT_CHRDEV: u8 = 3;
pub const F2FS_FT_BLKDEV: u8 = 4;
pub const F2FS_FT_FIFO: u8 = 5;
pub const F2FS_FT_SOCK: u8 = 6;
pub const F2FS_FT_SYMLINK: u8 = 7;
