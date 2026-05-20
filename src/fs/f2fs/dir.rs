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
