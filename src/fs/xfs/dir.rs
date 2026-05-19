//! XFS directory parsing.
//!
//! XFS has three on-disk directory formats. The on-disk choice is recorded
//! in the inode's `di_format` for `local` (shortform) vs `extents` /
//! `btree` (block, leaf, node, btree). This module implements the
//! **shortform** layout, which is what fits in the inode's literal area
//! and covers tiny directories — including the root of small images.
//!
//! Shortform header (v5 / v3 inodes):
//!
//! ```text
//!   off  len  field
//!     0    1  count            number of entries (excluding ".")
//!     1    1  i8count          (zero when no entries use 8-byte parent ino)
//!     2  4|8  parent           inode number of parent dir
//! ```
//!
//! `parent` is 4 bytes when `i8count == 0`, 8 bytes otherwise (the kernel
//! upgrades the whole directory if any inode in it can't fit in 32 bits).
//!
//! Each entry that follows the header is:
//!
//! ```text
//!   off  len  field
//!     0    1  namelen
//!     1    2  offset           dir-block "tag" (NOT a file position)
//!     3    namelen   name      bytes (no NUL)
//!     +    1  ftype            (v5/v3 only — XFS_DIR3_FT_*; 0 = unknown)
//!     +  4|8  inumber          file inode number; size matches header.i8count
//! ```
//!
//! On a v3 inode the per-entry `ftype` byte is present after the name. On a
//! v2 inode (no CRC) it is absent. We detect by `inode_version` passed in.
//!
//! File types (`XFS_DIR3_FT_*`) follow the same numbering as the kernel's
//! per-entry ftype byte (1=REG, 2=DIR, 3=CHR, 4=BLK, 5=FIFO, 6=SOCK, 7=LNK).
//!
//! ## What this module does NOT implement
//!
//! - **Block** directories (data fork holds one or more directory blocks
//!   on disk, inode format = `extents`, total size ≤ one directory block).
//! - **Leaf+ / Node / B-tree** directories (larger trees).
//!
//! Callers that encounter those formats should return
//! `Error::Unsupported("xfs: block/leaf directories not implemented")`.

use crate::Result;
use crate::fs::{DirEntry, EntryKind};

/// XFS_DIR3_FT_ — per-entry filetype byte (v5 directories).
pub const XFS_DIR3_FT_UNKNOWN: u8 = 0;
pub const XFS_DIR3_FT_REG_FILE: u8 = 1;
pub const XFS_DIR3_FT_DIR: u8 = 2;
pub const XFS_DIR3_FT_CHRDEV: u8 = 3;
pub const XFS_DIR3_FT_BLKDEV: u8 = 4;
pub const XFS_DIR3_FT_FIFO: u8 = 5;
pub const XFS_DIR3_FT_SOCK: u8 = 6;
pub const XFS_DIR3_FT_SYMLINK: u8 = 7;

/// Map an XFS ftype byte to the generic `EntryKind`. `Unknown` is mapped to
/// `EntryKind::Unknown` so the caller can still display the entry.
pub fn ftype_to_kind(ft: u8) -> EntryKind {
    match ft {
        XFS_DIR3_FT_REG_FILE => EntryKind::Regular,
        XFS_DIR3_FT_DIR => EntryKind::Dir,
        XFS_DIR3_FT_CHRDEV => EntryKind::Char,
        XFS_DIR3_FT_BLKDEV => EntryKind::Block,
        XFS_DIR3_FT_FIFO => EntryKind::Fifo,
        XFS_DIR3_FT_SOCK => EntryKind::Socket,
        XFS_DIR3_FT_SYMLINK => EntryKind::Symlink,
        _ => EntryKind::Unknown,
    }
}

/// Decoded shortform-directory entry. `inumber` is the resolved inode
/// number of the target.
#[derive(Debug, Clone)]
pub struct ShortformEntry {
    pub name: String,
    pub inumber: u64,
    pub ftype: u8,
}

/// Decode a shortform directory's literal area. `has_ftype` MUST be true on
/// v3 inodes (v5 filesystem) and false on v2 (legacy v4 / no-CRC).
pub fn decode_shortform(lit: &[u8], has_ftype: bool) -> Result<(u64, Vec<ShortformEntry>)> {
    if lit.len() < 2 {
        return Err(crate::Error::InvalidImage(
            "xfs: shortform dir header truncated".into(),
        ));
    }
    let count = lit[0] as usize;
    let i8count = lit[1];
    let parent_len = if i8count == 0 { 4 } else { 8 };
    if lit.len() < 2 + parent_len {
        return Err(crate::Error::InvalidImage(
            "xfs: shortform dir parent truncated".into(),
        ));
    }
    let parent = if parent_len == 4 {
        u32::from_be_bytes(lit[2..6].try_into().unwrap()) as u64
    } else {
        u64::from_be_bytes(lit[2..10].try_into().unwrap())
    };
    let inum_len = parent_len;

    let mut pos = 2 + parent_len;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if pos + 3 > lit.len() {
            return Err(crate::Error::InvalidImage(
                "xfs: shortform dir entry truncated".into(),
            ));
        }
        let namelen = lit[pos] as usize;
        // bytes [pos+1..pos+3] = offset (skip).
        let name_start = pos + 3;
        let name_end = name_start + namelen;
        if name_end > lit.len() {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: shortform dir entry name truncated (need {name_end}, have {})",
                lit.len()
            )));
        }
        let name = std::str::from_utf8(&lit[name_start..name_end])
            .map_err(|_| {
                crate::Error::InvalidImage("xfs: non-UTF-8 shortform dir name".into())
            })?
            .to_string();
        let mut cur = name_end;
        let ftype = if has_ftype {
            if cur >= lit.len() {
                return Err(crate::Error::InvalidImage(
                    "xfs: shortform dir entry missing ftype byte".into(),
                ));
            }
            let f = lit[cur];
            cur += 1;
            f
        } else {
            XFS_DIR3_FT_UNKNOWN
        };
        if cur + inum_len > lit.len() {
            return Err(crate::Error::InvalidImage(
                "xfs: shortform dir entry missing inum".into(),
            ));
        }
        let inumber = if inum_len == 4 {
            u32::from_be_bytes(lit[cur..cur + 4].try_into().unwrap()) as u64
        } else {
            u64::from_be_bytes(lit[cur..cur + 8].try_into().unwrap())
        };
        cur += inum_len;
        pos = cur;
        entries.push(ShortformEntry {
            name,
            inumber,
            ftype,
        });
    }

    Ok((parent, entries))
}

/// Convert shortform entries into the generic `DirEntry` shape. The caller
/// must inject any `.` / `..` entries it wants surfaced; we do not.
pub fn shortform_to_generic(entries: &[ShortformEntry]) -> Vec<DirEntry> {
    entries
        .iter()
        .map(|e| DirEntry {
            name: e.name.clone(),
            // XFS inode numbers are 64-bit; the `DirEntry.inode` field is
            // u32. We pass the low 32 bits — collisions are unlikely on the
            // small test images we use and the field is documented as
            // "stable per-entry id, not used for resolution".
            inode: e.inumber as u32,
            kind: ftype_to_kind(e.ftype),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic shortform area: count, i8count, parent (4B),
    /// then `(name, inumber, ftype)` triples each encoded as namelen, 2B
    /// offset placeholder, name bytes, ftype, 4B inumber.
    fn synth(parent: u32, entries: &[(&str, u32, u8)], has_ftype: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(entries.len() as u8);
        buf.push(0); // i8count
        buf.extend_from_slice(&parent.to_be_bytes());
        for (name, ino, ft) in entries {
            buf.push(name.len() as u8);
            buf.extend_from_slice(&[0, 0]); // offset placeholder
            buf.extend_from_slice(name.as_bytes());
            if has_ftype {
                buf.push(*ft);
            }
            buf.extend_from_slice(&ino.to_be_bytes());
        }
        buf
    }

    #[test]
    fn decode_two_entries_with_ftype() {
        let buf = synth(
            128,
            &[("a", 200, XFS_DIR3_FT_REG_FILE), ("dir", 201, XFS_DIR3_FT_DIR)],
            true,
        );
        let (parent, entries) = decode_shortform(&buf, true).unwrap();
        assert_eq!(parent, 128);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "a");
        assert_eq!(entries[0].inumber, 200);
        assert_eq!(entries[0].ftype, XFS_DIR3_FT_REG_FILE);
        assert_eq!(entries[1].name, "dir");
        assert_eq!(entries[1].inumber, 201);
        assert_eq!(entries[1].ftype, XFS_DIR3_FT_DIR);
    }

    #[test]
    fn decode_empty() {
        let buf = synth(128, &[], true);
        let (parent, entries) = decode_shortform(&buf, true).unwrap();
        assert_eq!(parent, 128);
        assert!(entries.is_empty());
    }

    #[test]
    fn decode_without_ftype_v2() {
        let buf = synth(128, &[("x", 7, 0)], false);
        let (parent, entries) = decode_shortform(&buf, false).unwrap();
        assert_eq!(parent, 128);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "x");
        assert_eq!(entries[0].inumber, 7);
        assert_eq!(entries[0].ftype, XFS_DIR3_FT_UNKNOWN);
    }

    #[test]
    fn decode_rejects_truncation() {
        let mut buf = synth(128, &[("name", 1, XFS_DIR3_FT_REG_FILE)], true);
        // Drop the trailing inumber.
        buf.truncate(buf.len() - 2);
        assert!(decode_shortform(&buf, true).is_err());
    }

    #[test]
    fn ftype_mapping() {
        assert_eq!(ftype_to_kind(XFS_DIR3_FT_REG_FILE), EntryKind::Regular);
        assert_eq!(ftype_to_kind(XFS_DIR3_FT_DIR), EntryKind::Dir);
        assert_eq!(ftype_to_kind(XFS_DIR3_FT_SYMLINK), EntryKind::Symlink);
        assert_eq!(ftype_to_kind(XFS_DIR3_FT_UNKNOWN), EntryKind::Unknown);
    }

    #[test]
    fn to_generic_preserves_data() {
        let entries = vec![
            ShortformEntry {
                name: "x".into(),
                inumber: 7,
                ftype: XFS_DIR3_FT_REG_FILE,
            },
        ];
        let g = shortform_to_generic(&entries);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].name, "x");
        assert_eq!(g[0].inode, 7);
        assert_eq!(g[0].kind, EntryKind::Regular);
    }
}
