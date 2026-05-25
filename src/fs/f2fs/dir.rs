//! F2FS directory entry decoding.
//!
//! A directory's body is a series of 4 KiB *dentry blocks*. Each block
//! holds, in order:
//!
//! ```text
//!   27 bytes  : slot bitmap (1 bit per slot, 214 slots → 27 B + padding)
//!    3 bytes  : reserved
//!  11 × 214 B : dentries     (hash u32 | ino u32 | name_len u16 | type u8)
//!   8 × 214 B : filename slots (each entry consumes ceil(name_len/8) slots)
//! ```
//!
//! A small directory can avoid spending a whole block by setting
//! `F2FS_INLINE_DENTRY` on the inode and packing the same structure into
//! the inode's literal area (with smaller arrays sized to fit). For v1
//! we honour the inline-dentry path with the inline geometry that
//! actually fits the 3712-byte literal region.
//!
//! Reference: kernel docs §"Directory Structure".

use super::constants::{
    F2FS_FT_BLKDEV, F2FS_FT_CHRDEV, F2FS_FT_DIR, F2FS_FT_FIFO, F2FS_FT_REG_FILE, F2FS_FT_SOCK,
    F2FS_FT_SYMLINK, F2FS_SLOT_LEN, NR_DENTRY_IN_BLOCK, SIZE_OF_DENTRY_BITMAP, SIZE_OF_DIR_ENTRY,
    SIZE_OF_RESERVED,
};
use super::inode::F2fsInode;
use crate::fs::EntryKind;

/// In-memory dentry, hostname not yet UTF-8 validated.
#[derive(Debug, Clone)]
pub struct RawDentry {
    pub hash: u32,
    pub ino: u32,
    pub file_type: u8,
    pub name: Vec<u8>,
}

impl RawDentry {
    /// Convert to the crate's neutral [`crate::fs::DirEntry`] view.
    pub fn into_dir_entry(self) -> crate::fs::DirEntry {
        crate::fs::DirEntry {
            name: String::from_utf8_lossy(&self.name).into_owned(),
            inode: self.ino,
            kind: type_to_kind(self.file_type),
            // RawDentry doesn't carry the file size (it lives on the
            // inode block, not the dentry). Callers that need a size
            // should look it up via the FS's read API.
            size: 0,
        }
    }
}

pub fn type_to_kind(file_type: u8) -> EntryKind {
    match file_type {
        F2FS_FT_REG_FILE => EntryKind::Regular,
        F2FS_FT_DIR => EntryKind::Dir,
        F2FS_FT_SYMLINK => EntryKind::Symlink,
        F2FS_FT_CHRDEV => EntryKind::Char,
        F2FS_FT_BLKDEV => EntryKind::Block,
        F2FS_FT_FIFO => EntryKind::Fifo,
        F2FS_FT_SOCK => EntryKind::Socket,
        _ => EntryKind::Unknown,
    }
}

/// Decode every populated dentry in a 4 KiB dentry block.
pub fn decode_dentry_block(buf: &[u8]) -> crate::Result<Vec<RawDentry>> {
    decode_dentry_region(
        buf,
        NR_DENTRY_IN_BLOCK,
        0,
        SIZE_OF_DENTRY_BITMAP + SIZE_OF_RESERVED,
        SIZE_OF_DENTRY_BITMAP + SIZE_OF_RESERVED + NR_DENTRY_IN_BLOCK * SIZE_OF_DIR_ENTRY,
    )
}

/// Decode dentries from an inline-dentry region inside an inode block.
///
/// We pick the largest geometry that fits the 3712-byte literal area at
/// `I_ADDR_OFFSET`: 182 slots → 23 B bitmap + 1 reserved + 182 × 11 + 182 × 8
/// = 3489 B (well inside the budget).
pub fn decode_inline_dentries(
    inode: &F2fsInode,
    inode_block: &[u8],
) -> crate::Result<Vec<RawDentry>> {
    if !inode.is_inline_dentry() {
        return Ok(Vec::new());
    }
    let payload = inode.inline_payload(inode_block);
    let nr = INLINE_DENTRY_NR;
    let bitmap_bytes = nr.div_ceil(8);
    // fsck.f2fs's INLINE_RESERVED_SIZE = MAX_INLINE_DATA - (nr*19 + bitmap_size)
    // = 3488 - (182*19 + 23) = 7 bytes.
    let reserved = 7usize;
    let dentries_off = bitmap_bytes + reserved;
    let names_off = dentries_off + nr * SIZE_OF_DIR_ENTRY;
    decode_dentry_region(payload, nr, 0, dentries_off, names_off)
}

/// Inline dentry slot count. Chosen so the block fits inside
/// `i_addr` + `i_nid` (3712 B): 182 × (11 + 8) = 3458 + bitmap (23) +
/// reserved (1) = 3482 B used. Standard F2FS picks the same value
/// derivation: maximize subject to `bitmap + reserved + nr*(de+slot) ≤ space`.
pub const INLINE_DENTRY_NR: usize = 182;

/// Shared decoder: walk a (`bitmap | dentries | names`) region and emit
/// each set entry. `bitmap_off`, `dentries_off`, `names_off` are byte
/// offsets within `region`; `nr` is the slot count.
fn decode_dentry_region(
    region: &[u8],
    nr: usize,
    bitmap_off: usize,
    dentries_off: usize,
    names_off: usize,
) -> crate::Result<Vec<RawDentry>> {
    let bitmap_bytes = nr.div_ceil(8);
    let end = names_off + nr * F2FS_SLOT_LEN;
    if region.len() < end {
        return Err(crate::Error::InvalidImage(
            "f2fs: dentry region truncated".into(),
        ));
    }
    let bitmap = &region[bitmap_off..bitmap_off + bitmap_bytes];

    let mut out = Vec::new();
    let mut slot = 0;
    while slot < nr {
        if !get_bit(bitmap, slot) {
            slot += 1;
            continue;
        }
        let de_off = dentries_off + slot * SIZE_OF_DIR_ENTRY;
        let hash = u32::from_le_bytes(region[de_off..de_off + 4].try_into().unwrap());
        let ino = u32::from_le_bytes(region[de_off + 4..de_off + 8].try_into().unwrap());
        let name_len =
            u16::from_le_bytes(region[de_off + 8..de_off + 10].try_into().unwrap()) as usize;
        let file_type = region[de_off + 10];

        // How many 8-byte slots does this name consume?
        let need_slots = name_len.div_ceil(F2FS_SLOT_LEN).max(1);
        if slot + need_slots > nr {
            return Err(crate::Error::InvalidImage(format!(
                "f2fs: dentry@slot {slot} runs past end (name_len={name_len})"
            )));
        }
        let name_start = names_off + slot * F2FS_SLOT_LEN;
        let name_end = (name_start + name_len).min(region.len());
        let name = region[name_start..name_end].to_vec();
        out.push(RawDentry {
            hash,
            ino,
            file_type,
            name,
        });
        slot += need_slots;
    }
    Ok(out)
}

#[inline]
fn get_bit(bitmap: &[u8], i: usize) -> bool {
    let byte = i / 8;
    let bit = i % 8;
    byte < bitmap.len() && (bitmap[byte] & (1 << bit)) != 0
}

#[cfg(test)]
#[inline]
fn set_bit(bitmap: &mut [u8], i: usize) {
    let byte = i / 8;
    let bit = i % 8;
    if byte < bitmap.len() {
        bitmap[byte] |= 1 << bit;
    }
}

// ---------------------------------------------------------------------
// Directory hashing + multi-level bucket layout (kernel fs/f2fs/{hash,dir}.c)
// ---------------------------------------------------------------------

/// Maximum directory hash depth (`MAX_DIR_HASH_DEPTH`).
pub const MAX_DIR_HASH_DEPTH: u32 = 63;
/// Maximum buckets at the deepest levels (`MAX_DIR_BUCKETS`).
pub const MAX_DIR_BUCKETS: u32 = 1 << ((MAX_DIR_HASH_DEPTH / 2) - 1);

const HASH_DELTA: u32 = 0x9E37_79B9;

fn tea_transform(buf: &mut [u32; 4], input: &[u32; 4]) {
    let mut sum: u32 = 0;
    let (mut b0, mut b1) = (buf[0], buf[1]);
    let (a, b, c, d) = (input[0], input[1], input[2], input[3]);
    for _ in 0..16 {
        sum = sum.wrapping_add(HASH_DELTA);
        b0 = b0.wrapping_add(
            ((b1 << 4).wrapping_add(a)) ^ (b1.wrapping_add(sum)) ^ ((b1 >> 5).wrapping_add(b)),
        );
        b1 = b1.wrapping_add(
            ((b0 << 4).wrapping_add(c)) ^ (b0.wrapping_add(sum)) ^ ((b0 >> 5).wrapping_add(d)),
        );
    }
    buf[0] = buf[0].wrapping_add(b0);
    buf[1] = buf[1].wrapping_add(b1);
}

/// Pack up to 16 bytes of `msg` into four 32-bit words (kernel
/// `str2hashbuf`). `full_len` is the *remaining* name length (used for the
/// pad value); only `min(full_len, 16)` bytes are consumed.
fn str2hashbuf(msg: &[u8], full_len: usize, out: &mut [u32; 4]) {
    let pad = {
        let l = full_len as u32;
        let mut p = l | (l << 8);
        p |= p << 16;
        p
    };
    let mut val = pad;
    let len = full_len.min(16);
    let mut num: i32 = 4;
    let mut idx = 0usize;
    for (i, &byte) in msg.iter().take(len).enumerate() {
        if i % 4 == 0 {
            val = pad;
        }
        val = (byte as u32).wrapping_add(val << 8);
        if i % 4 == 3 {
            out[idx] = val;
            idx += 1;
            val = pad;
            num -= 1;
        }
    }
    num -= 1;
    if num >= 0 {
        out[idx] = val;
        idx += 1;
    }
    loop {
        num -= 1;
        if num < 0 {
            break;
        }
        out[idx] = pad;
        idx += 1;
    }
}

/// F2FS directory-entry hash (`f2fs_dentry_hash` / `TEA_hash_name`). `.`
/// and `..` hash to 0 (matching `name_is_dot_dotdot`). The kernel clears
/// `F2FS_HASH_COL_BIT` (bit 63) which is a no-op on the 32-bit result.
pub fn f2fs_dentry_hash(name: &[u8]) -> u32 {
    if name == b"." || name == b".." {
        return 0;
    }
    let mut buf = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let mut off = 0usize;
    loop {
        let remaining = name.len() - off;
        let mut input = [0u32; 4];
        str2hashbuf(&name[off..], remaining, &mut input);
        tea_transform(&mut buf, &input);
        off += 16;
        if remaining <= 16 {
            break;
        }
    }
    buf[0]
}

/// Number of hash buckets at `level` (`dir_buckets`).
pub fn dir_buckets(level: u32, dir_level: u8) -> u32 {
    let s = level + dir_level as u32;
    if s < MAX_DIR_HASH_DEPTH / 2 {
        1u32 << s
    } else {
        MAX_DIR_BUCKETS
    }
}

/// Number of dentry blocks per bucket at `level` (`bucket_blocks`).
pub fn bucket_blocks(level: u32) -> u32 {
    if level < MAX_DIR_HASH_DEPTH / 2 { 2 } else { 4 }
}

/// Logical dentry-block index of bucket `bucket` at `level`
/// (`dir_block_index`): the count of all blocks in levels below, plus the
/// bucket's offset within this level.
pub fn dir_block_index(level: u32, dir_level: u8, bucket: u32) -> u64 {
    let mut bidx: u64 = 0;
    for i in 0..level {
        bidx += dir_buckets(i, dir_level) as u64 * bucket_blocks(i) as u64;
    }
    bidx + bucket as u64 * bucket_blocks(level) as u64
}

/// Compute the file-type byte for a POSIX mode.
pub fn file_type_from_mode(mode: u16) -> u8 {
    use super::constants::{
        S_IFBLK, S_IFCHR, S_IFDIR, S_IFIFO, S_IFLNK, S_IFMT, S_IFREG, S_IFSOCK,
    };
    match mode & S_IFMT {
        S_IFREG => F2FS_FT_REG_FILE,
        S_IFDIR => F2FS_FT_DIR,
        S_IFLNK => F2FS_FT_SYMLINK,
        S_IFCHR => F2FS_FT_CHRDEV,
        S_IFBLK => F2FS_FT_BLKDEV,
        S_IFIFO => F2FS_FT_FIFO,
        S_IFSOCK => F2FS_FT_SOCK,
        _ => 0,
    }
}

/// Test helper: produce a 4 KiB dentry block holding the given entries.
/// Names that exceed F2FS_SLOT_LEN consume multiple slots, like on-disk.
#[cfg(test)]
pub(crate) fn encode_dentry_block(entries: &[RawDentry]) -> Vec<u8> {
    encode_dentry_region(
        entries,
        NR_DENTRY_IN_BLOCK,
        SIZE_OF_DENTRY_BITMAP,
        SIZE_OF_RESERVED,
        super::constants::F2FS_BLKSIZE,
    )
}

#[cfg(test)]
pub(crate) fn encode_inline_dentries_payload(entries: &[RawDentry]) -> Vec<u8> {
    encode_dentry_region(
        entries,
        INLINE_DENTRY_NR,
        INLINE_DENTRY_NR.div_ceil(8),
        7,
        // size: bitmap + reserved + nr*(de + slot)
        INLINE_DENTRY_NR.div_ceil(8) + 7 + INLINE_DENTRY_NR * (SIZE_OF_DIR_ENTRY + F2FS_SLOT_LEN),
    )
}

#[cfg(test)]
fn encode_dentry_region(
    entries: &[RawDentry],
    nr: usize,
    bitmap_bytes: usize,
    reserved: usize,
    region_size: usize,
) -> Vec<u8> {
    let dentries_off = bitmap_bytes + reserved;
    let names_off = dentries_off + nr * SIZE_OF_DIR_ENTRY;
    let mut buf = vec![0u8; region_size];
    let mut slot = 0;
    for e in entries {
        let name_slots = e.name.len().div_ceil(F2FS_SLOT_LEN).max(1);
        if slot + name_slots > nr {
            break;
        }
        // Bitmap: only mark the *head* slot — the on-disk convention.
        set_bit(&mut buf[0..bitmap_bytes], slot);
        let de_off = dentries_off + slot * SIZE_OF_DIR_ENTRY;
        buf[de_off..de_off + 4].copy_from_slice(&e.hash.to_le_bytes());
        buf[de_off + 4..de_off + 8].copy_from_slice(&e.ino.to_le_bytes());
        buf[de_off + 8..de_off + 10].copy_from_slice(&(e.name.len() as u16).to_le_bytes());
        buf[de_off + 10] = e.file_type;
        let name_start = names_off + slot * F2FS_SLOT_LEN;
        let name_end = name_start + e.name.len();
        buf[name_start..name_end].copy_from_slice(&e.name);
        slot += name_slots;
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dentry_hash_dots_are_zero() {
        assert_eq!(f2fs_dentry_hash(b"."), 0);
        assert_eq!(f2fs_dentry_hash(b".."), 0);
    }

    #[test]
    fn dentry_hash_matches_kernel_tea() {
        // Reference values from an independent replica of the kernel
        // fs/f2fs/hash.c TEA_hash_name (and ultimately cross-checked by
        // fsck.f2fs in CI, which recomputes the hash for every dentry).
        assert_eq!(f2fs_dentry_hash(b"a"), 0x6d0e_a4c1);
        assert_eq!(f2fs_dentry_hash(b"file00000"), 0x4bc4_becc);
        assert_eq!(f2fs_dentry_hash(b"file09999"), 0xa554_9af2);
        assert_eq!(f2fs_dentry_hash(b"abcdefghijklmnop"), 0xf4ac_8cb5);
        assert_eq!(f2fs_dentry_hash(b"hello"), 0x6f5b_b1a8);
    }

    #[test]
    fn bucket_layout_math() {
        assert_eq!(dir_buckets(0, 0), 1);
        assert_eq!(dir_buckets(1, 0), 2);
        assert_eq!(dir_buckets(2, 0), 4);
        assert_eq!(dir_buckets(5, 0), 32);
        assert_eq!(bucket_blocks(0), 2);
        assert_eq!(bucket_blocks(30), 2);
        assert_eq!(bucket_blocks(31), 4);
        // Level 0: bucket 0 → block 0.
        assert_eq!(dir_block_index(0, 0, 0), 0);
        // Level 1: 1 bucket × 2 blocks below, then bucket offset.
        assert_eq!(dir_block_index(1, 0, 0), 2);
        assert_eq!(dir_block_index(1, 0, 1), 4);
        // Level 2: (1×2 + 2×2) = 6 blocks below.
        assert_eq!(dir_block_index(2, 0, 0), 6);
        assert_eq!(dir_block_index(2, 0, 3), 12);
    }
}
