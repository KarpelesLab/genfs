//! Linear directory-entry encoding for ext2.
//!
//! Without the `INCOMPAT_FILETYPE` feature (our default for genext2fs
//! compatibility), the on-disk entry layout is:
//!
//! ```text
//!   offset  size  field
//!     0     4     inode number (u32 LE)
//!     4     2     rec_len (u16 LE) — byte length of this entry incl. padding
//!     6     2     name_len (u16 LE)
//!     8     N     name (N = name_len)
//! ```
//!
//! Each entry's `rec_len` is rounded up to 4 bytes so the next entry is
//! 4-byte aligned. The *last* entry in a directory data block extends its
//! `rec_len` to fill the rest of the block (so a reader sweeping by rec_len
//! always lands on the next block boundary cleanly).
//!
//! With `INCOMPAT_FILETYPE` enabled, the high byte of `name_len` becomes a
//! file_type field; we keep the no-FILETYPE layout for v1 ext2.

use super::constants::{DENT_BLK, DENT_CHR, DENT_DIR, DENT_FIFO, DENT_LNK, DENT_REG, DENT_SOCK};

/// Size of the fixed prefix of a dir entry (inode + rec_len + name_len).
pub const DIRENT_HEADER_LEN: usize = 8;

/// Round a value up to the next multiple of 4.
#[inline]
pub fn align4(n: usize) -> usize {
    (n + 3) & !3
}

/// Minimum on-disk length of a dir entry for a name of `name_len` bytes,
/// 4-byte aligned.
pub fn min_rec_len(name_len: usize) -> usize {
    align4(DIRENT_HEADER_LEN + name_len)
}

/// Append a single entry to `out` with the given `rec_len`. Caller is
/// responsible for choosing `rec_len` — for non-final entries it should be
/// [`min_rec_len(name.len())`]; the final entry in a block must absorb the
/// trailing slack so it reaches the block boundary.
///
/// The `file_type` argument is the dirent type byte (`DENT_REG`, `DENT_DIR`,
/// ...) which is only used when the FILETYPE incompat feature is enabled —
/// the writer threads it through so the same function can serve both
/// modes. With FILETYPE off (our default), it is ignored.
pub fn encode_entry(
    out: &mut Vec<u8>,
    inode: u32,
    name: &[u8],
    rec_len: u16,
    file_type: u8,
    with_filetype: bool,
) {
    assert!(
        name.len() <= 255,
        "ext2 name_len is u8 (or u16 without FILETYPE)"
    );
    let start = out.len();
    out.extend_from_slice(&inode.to_le_bytes());
    out.extend_from_slice(&rec_len.to_le_bytes());
    if with_filetype {
        out.push(name.len() as u8);
        out.push(file_type);
    } else {
        // 16-bit name_len when FILETYPE is off.
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    }
    out.extend_from_slice(name);
    let padded_len = rec_len as usize;
    let written = out.len() - start;
    assert!(
        written <= padded_len,
        "dir entry overflows declared rec_len: wrote {written}, declared {padded_len}"
    );
    out.resize(start + padded_len, 0);
}

/// Decode one entry starting at `b[..]`. Returns the entry plus the byte
/// length consumed (its `rec_len`).
pub fn decode_entry(b: &[u8], with_filetype: bool) -> Option<DecodedEntry<'_>> {
    if b.len() < DIRENT_HEADER_LEN {
        return None;
    }
    let inode = u32::from_le_bytes(b[0..4].try_into().unwrap());
    let rec_len = u16::from_le_bytes(b[4..6].try_into().unwrap()) as usize;
    let (name_len, file_type) = if with_filetype {
        (b[6] as usize, b[7])
    } else {
        let nl = u16::from_le_bytes(b[6..8].try_into().unwrap()) as usize;
        (nl, 0)
    };
    if rec_len < DIRENT_HEADER_LEN || DIRENT_HEADER_LEN + name_len > rec_len {
        return None;
    }
    if b.len() < rec_len {
        return None;
    }
    Some(DecodedEntry {
        inode,
        name: &b[DIRENT_HEADER_LEN..DIRENT_HEADER_LEN + name_len],
        rec_len,
        file_type,
    })
}

/// One decoded directory entry, borrowing its name from the source buffer.
#[derive(Debug)]
pub struct DecodedEntry<'a> {
    pub inode: u32,
    pub name: &'a [u8],
    pub rec_len: usize,
    pub file_type: u8,
}

/// Map a [`crate::fs::EntryKind`] to the dirent file_type byte.
pub fn file_type_byte(k: crate::fs::EntryKind) -> u8 {
    use crate::fs::EntryKind::*;
    match k {
        Regular => DENT_REG,
        Dir => DENT_DIR,
        Symlink => DENT_LNK,
        Char => DENT_CHR,
        Block => DENT_BLK,
        Fifo => DENT_FIFO,
        Socket => DENT_SOCK,
        Unknown => 0,
    }
}

/// Build an "empty placeholder" directory data block: a single entry with
/// inode=0, name_len=0, rec_len=block_size. e2fsck accepts this as a
/// well-formed (but empty) dir block. genext2fs uses this to pad
/// pre-allocated dirs like lost+found.
pub fn make_empty_dir_block(block_size: u32) -> Vec<u8> {
    let mut buf = vec![0u8; block_size as usize];
    // inode (4 bytes) is already zero.
    // rec_len at offset 4..6 = block_size.
    buf[4..6].copy_from_slice(&(block_size as u16).to_le_bytes());
    // name_len at offset 6..8 = 0 (already zero).
    buf
}

/// Build the directory data block for a fresh directory: just "." and "..".
/// The "." entry points at `self_inode`, ".." at `parent_inode`. `block_size`
/// is the FS block size in bytes; the ".." entry's `rec_len` is extended so
/// it fills the whole block.
pub fn make_initial_dir_block(
    self_inode: u32,
    parent_inode: u32,
    block_size: u32,
    with_filetype: bool,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(block_size as usize);

    // "." entry
    let dot_rec = min_rec_len(1) as u16;
    encode_entry(&mut buf, self_inode, b".", dot_rec, DENT_DIR, with_filetype);

    // ".." entry fills the rest of the block
    let dotdot_rec = (block_size as usize - buf.len()) as u16;
    encode_entry(
        &mut buf,
        parent_inode,
        b"..",
        dotdot_rec,
        DENT_DIR,
        with_filetype,
    );

    debug_assert_eq!(buf.len(), block_size as usize);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align4_examples() {
        assert_eq!(align4(0), 0);
        assert_eq!(align4(1), 4);
        assert_eq!(align4(8), 8);
        assert_eq!(align4(9), 12);
        assert_eq!(align4(11), 12);
        assert_eq!(align4(12), 12);
    }

    #[test]
    fn min_rec_len_for_short_name() {
        // "x" -> header(8) + 1 -> 12 (aligned)
        assert_eq!(min_rec_len(1), 12);
        assert_eq!(min_rec_len(4), 12);
        assert_eq!(min_rec_len(5), 16);
    }

    #[test]
    fn initial_dir_block_round_trips() {
        let buf = make_initial_dir_block(2, 2, 1024, false);
        assert_eq!(buf.len(), 1024);
        let first = decode_entry(&buf, false).unwrap();
        assert_eq!(first.inode, 2);
        assert_eq!(first.name, b".");
        let second = decode_entry(&buf[first.rec_len..], false).unwrap();
        assert_eq!(second.inode, 2);
        assert_eq!(second.name, b"..");
        assert_eq!(first.rec_len + second.rec_len, 1024);
    }

    #[test]
    fn many_entries_round_trip() {
        let mut buf = Vec::new();
        let names: &[&[u8]] = &[b"first", b"second", b"three", b"four"];
        // Each entry's rec_len = min for its name; build a packed buffer.
        for (inode, name) in (11u32..).zip(names.iter()) {
            let rl = min_rec_len(name.len()) as u16;
            encode_entry(&mut buf, inode, name, rl, DENT_REG, false);
        }
        // Now decode them back.
        let mut off = 0;
        let mut got_names = Vec::new();
        while off < buf.len() {
            let e = decode_entry(&buf[off..], false).unwrap();
            got_names.push(e.name.to_vec());
            off += e.rec_len;
        }
        let got: Vec<&[u8]> = got_names.iter().map(|v| v.as_slice()).collect();
        assert_eq!(got, names);
    }
}
