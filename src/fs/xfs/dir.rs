//! XFS directory parsing.
//!
//! XFS has four on-disk directory formats. The on-disk choice is recorded
//! in the inode's `di_format`:
//!
//! - `local`  (shortform)   — entries packed in the inode literal area
//! - `extents` (block / leaf / node) — entries stored in directory blocks on
//!   disk, addressed through the inode's extent list
//! - `btree`  — a B-tree-of-extents form for very large directories
//!
//! This module implements **shortform**, **block**, and **leaf** formats.
//! Node format is best-effort: the leaf-index variant of node directories is
//! parsed (we read every data block) but full node-btree traversal is not
//! required for correctness because all data blocks have logical addresses
//! < `XFS_DIR2_LEAF_OFFSET` and we can scan them in order. B-tree (`btree`
//! di_format) directories return `Error::Unsupported`.
//!
//! ## Shortform header (v5 / v3 inodes)
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
//! ## Shortform entry
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
//! ## Data-block layout (block / leaf / node directories)
//!
//! Each "directory block" (size = `1 << sb_dirblklog` FS blocks) starts with
//! a header, then a sequence of variable-length records. Each record is
//! either a used `data_entry` or a free `data_unused` region.
//!
//! Headers:
//!
//! - v4 data block: magic `"XD2D"`, then `bestfree[3]` (3 × 4 bytes). 16 B.
//! - v4 block dir: magic `"XD2B"`, then `bestfree[3]`. 16 B header, plus
//!   leaf array + `block_tail{count,stale}` at the tail.
//! - v5 data block: magic `"XDD3"`, 48 B v5-header + `bestfree[3]` + 4 B
//!   pad = 64 B.
//! - v5 block dir: magic `"XDB3"`, same 64 B header, plus tail (as v4).
//!
//! Per-entry:
//!
//! ```text
//!   data_entry:    __be64 inumber, u8 namelen, u8 name[namelen],
//!                  [u8 ftype (v5)], <pad>, __be16 tag — total padded to 8 B.
//!   data_unused:   __be16 freetag = 0xffff, __be16 length, <pad>, __be16 tag.
//! ```
//!
//! File types (`XFS_DIR3_FT_*`) follow the kernel numbering (1=REG, 2=DIR,
//! 3=CHR, 4=BLK, 5=FIFO, 6=SOCK, 7=LNK).

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

/// Magic numbers for the various directory block flavours.
pub const XFS_DIR2_BLOCK_MAGIC: u32 = 0x5844_3242; // "XD2B"
pub const XFS_DIR2_DATA_MAGIC: u32 = 0x5844_3244; // "XD2D"
pub const XFS_DIR3_BLOCK_MAGIC: u32 = 0x5844_4233; // "XDB3"
pub const XFS_DIR3_DATA_MAGIC: u32 = 0x5844_4433; // "XDD3"

// Note: leaf/node/free-index block magics (XD2F/XD2L/XDLF/XDD2/etc.) are
// recognised implicitly by checking that a block's magic does NOT match
// either of the data magics above. We do not parse those blocks here —
// the data-block scan in `Xfs::read_extent_dir_entries` is sufficient to
// enumerate every entry without going through the hashed leaf index.

/// Freetag stored at the start of a `data_unused` region.
pub const XFS_DIR2_DATA_FREE_TAG: u16 = 0xFFFF;

/// Logical block index (in dir-block units) where the leaf block lives in a
/// leaf-format directory. Defined as `32 GiB / dirblksize`.
pub const XFS_DIR2_LEAF_FIRSTDB_BYTES: u64 = 32 * 1024 * 1024 * 1024; // 32 GiB

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

/// A decoded entry from a block / leaf / node directory data block.
#[derive(Debug, Clone)]
pub struct DataEntry {
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
            .map_err(|_| crate::Error::InvalidImage("xfs: non-UTF-8 shortform dir name".into()))?
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

/// Convert block/leaf/node data entries to the generic `DirEntry` shape.
pub fn data_entries_to_generic(entries: &[DataEntry]) -> Vec<DirEntry> {
    entries
        .iter()
        .map(|e| DirEntry {
            name: e.name.clone(),
            inode: e.inumber as u32,
            kind: ftype_to_kind(e.ftype),
        })
        .collect()
}

/// Parse a single directory **data** or **block** block. Walks the
/// per-entry stream between `entries_start` and either the end of the buffer
/// or the start of the trailing leaf array (for block-format directories).
///
/// `has_ftype` is true on v5 filesystems (per-entry filetype byte present).
/// `entries_end` is the byte offset where parsing should stop — for plain
/// data blocks this is the buffer length; for block-format directories it
/// is the start of the leaf-entry array (== `header.bestfree.offsets` range
/// after the block tail accounting).
fn parse_data_records(
    block: &[u8],
    entries_start: usize,
    entries_end: usize,
    has_ftype: bool,
) -> Result<Vec<DataEntry>> {
    if entries_start > entries_end || entries_end > block.len() {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: dir data record range out of bounds: [{entries_start}..{entries_end}] vs buf {}",
            block.len()
        )));
    }
    let mut out = Vec::new();
    let mut pos = entries_start;
    while pos + 4 <= entries_end {
        // Detect freetag (data_unused): first 2 bytes == 0xFFFF.
        let head = u16::from_be_bytes(block[pos..pos + 2].try_into().unwrap());
        if head == XFS_DIR2_DATA_FREE_TAG {
            // length field is bytes [pos+2..pos+4].
            let length = u16::from_be_bytes(block[pos + 2..pos + 4].try_into().unwrap()) as usize;
            if length == 0 {
                return Err(crate::Error::InvalidImage(
                    "xfs: dir data_unused with length 0".into(),
                ));
            }
            if pos + length > entries_end {
                return Err(crate::Error::InvalidImage(format!(
                    "xfs: dir data_unused length {length} overshoots end at {entries_end}"
                )));
            }
            pos += length;
            continue;
        }
        // Otherwise it's a data_entry: inumber(8) namelen(1) name[] [ftype(1)] tag(2) pad-to-8.
        if pos + 8 + 1 > entries_end {
            return Err(crate::Error::InvalidImage(
                "xfs: dir data_entry header truncated".into(),
            ));
        }
        let inumber = u64::from_be_bytes(block[pos..pos + 8].try_into().unwrap());
        let namelen = block[pos + 8] as usize;
        if namelen == 0 {
            return Err(crate::Error::InvalidImage(
                "xfs: dir data_entry has namelen=0".into(),
            ));
        }
        let name_start = pos + 9;
        let name_end = name_start + namelen;
        if name_end > entries_end {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: dir data_entry name truncated (need {name_end}, have {entries_end})"
            )));
        }
        let name = std::str::from_utf8(&block[name_start..name_end])
            .map_err(|_| {
                crate::Error::InvalidImage("xfs: non-UTF-8 dir name in data block".into())
            })?
            .to_string();
        let mut after_name = name_end;
        let ftype = if has_ftype {
            if after_name >= entries_end {
                return Err(crate::Error::InvalidImage(
                    "xfs: dir data_entry missing ftype byte".into(),
                ));
            }
            let f = block[after_name];
            after_name += 1;
            f
        } else {
            XFS_DIR3_FT_UNKNOWN
        };
        // The tag (__be16) is at the last 2 bytes of the padded record. The
        // padded record size is: 8 (inumber) + 1 (namelen) + namelen + (1 if
        // has_ftype) + 2 (tag), rounded up to 8 bytes.
        let raw_len = 8 + 1 + namelen + (if has_ftype { 1 } else { 0 }) + 2;
        let padded_len = (raw_len + 7) & !7;
        if pos + padded_len > entries_end {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: dir data_entry padded length {padded_len} overshoots end {entries_end}"
            )));
        }
        // Sanity: skip past the entry. We don't validate the tag value.
        let _ = after_name;
        pos += padded_len;
        out.push(DataEntry {
            name,
            inumber,
            ftype,
        });
    }
    Ok(out)
}

/// Decode the header of a directory data or block block. Returns
/// `(entries_start, magic)`.
///
/// `is_v5` controls which header layout is expected. The magic must match
/// one of the data/block magics for the requested version. Block-format
/// directories share the same header but additionally have a leaf-entry
/// array + tail at the END of the block (see [`block_dir_entries_end`]).
fn decode_data_header(block: &[u8], is_v5: bool) -> Result<(usize, u32)> {
    if block.len() < 4 {
        return Err(crate::Error::InvalidImage(
            "xfs: dir data block too small".into(),
        ));
    }
    let magic = u32::from_be_bytes(block[0..4].try_into().unwrap());
    if is_v5 {
        match magic {
            XFS_DIR3_BLOCK_MAGIC | XFS_DIR3_DATA_MAGIC => {}
            _ => {
                return Err(crate::Error::InvalidImage(format!(
                    "xfs: bad v5 dir block magic {magic:#010x}"
                )));
            }
        }
        // v5 header is 48 B + bestfree[3](12) + pad(4) = 64 B.
        if block.len() < 64 {
            return Err(crate::Error::InvalidImage(
                "xfs: v5 dir block shorter than header".into(),
            ));
        }
        Ok((64, magic))
    } else {
        match magic {
            XFS_DIR2_BLOCK_MAGIC | XFS_DIR2_DATA_MAGIC => {}
            _ => {
                return Err(crate::Error::InvalidImage(format!(
                    "xfs: bad v4 dir block magic {magic:#010x}"
                )));
            }
        }
        // v4 header is magic(4) + bestfree[3](12) = 16 B.
        if block.len() < 16 {
            return Err(crate::Error::InvalidImage(
                "xfs: v4 dir block shorter than header".into(),
            ));
        }
        Ok((16, magic))
    }
}

/// For a **block-format** directory, the byte offset where data records end
/// and the trailing leaf-entry array begins. Layout (from end of block):
///
/// ```text
///   ... data entries ...
///   leaf[count] :: 8 bytes each (hashval, address)
///   __be32 count
///   __be32 stale
/// ```
///
/// Returns `entries_end`, i.e. the start of the leaf array.
fn block_dir_entries_end(block: &[u8]) -> Result<usize> {
    let blen = block.len();
    if blen < 8 {
        return Err(crate::Error::InvalidImage(
            "xfs: block dir too short for tail".into(),
        ));
    }
    let count = u32::from_be_bytes(block[blen - 8..blen - 4].try_into().unwrap()) as usize;
    // We don't actually need the leaf entries to walk records, but we DO
    // need to know how much of the block is leaf-entry area so we stop
    // parsing data records before it. Each leaf entry is 8 bytes.
    let leaf_bytes = count
        .checked_mul(8)
        .ok_or_else(|| crate::Error::InvalidImage("xfs: block dir leaf count overflows".into()))?;
    let tail = 8 + leaf_bytes;
    if tail > blen {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: block dir tail (8 + 8*{count}) > block size {blen}"
        )));
    }
    Ok(blen - tail)
}

/// Decode a single **block-format** directory block. Used when the inode is
/// `EXTENTS` and the directory data fits in one directory block.
pub fn decode_block_dir(block: &[u8], is_v5: bool) -> Result<Vec<DataEntry>> {
    let (entries_start, magic) = decode_data_header(block, is_v5)?;
    let want_magic = if is_v5 {
        XFS_DIR3_BLOCK_MAGIC
    } else {
        XFS_DIR2_BLOCK_MAGIC
    };
    if magic != want_magic {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: block dir expected magic {want_magic:#010x}, got {magic:#010x}"
        )));
    }
    let entries_end = block_dir_entries_end(block)?;
    parse_data_records(block, entries_start, entries_end, is_v5)
}

/// Decode a **data-only** directory block (one of many in a leaf-format
/// directory). Skips the header and walks all records to the end of the
/// buffer.
pub fn decode_data_block(block: &[u8], is_v5: bool) -> Result<Vec<DataEntry>> {
    let (entries_start, magic) = decode_data_header(block, is_v5)?;
    let want_magic = if is_v5 {
        XFS_DIR3_DATA_MAGIC
    } else {
        XFS_DIR2_DATA_MAGIC
    };
    if magic != want_magic {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: data dir expected magic {want_magic:#010x}, got {magic:#010x}"
        )));
    }
    parse_data_records(block, entries_start, block.len(), is_v5)
}

/// CRC offset inside a v5 directory-block header (XDB3 / XDD3 / XDL3 /
/// XDF3 / XDN3). The v5 block header is
/// `magic(4) crc(4) blkno(8) lsn(8) uuid(16) owner(8) = 48 B`, followed
/// by per-magic specialised fields.
pub const V5_DIR_CRC_OFFSET: usize = 4;

/// Bytes of v5 directory data-block header (XDD3 / XDB3) up to the start
/// of variable-length records.
pub const V5_DATA_HDR_SIZE: usize = 64;

/// Stamp the v5 directory-block CRC32C in place. The CRC is taken over
/// the entire block with the 4-byte `crc` field at offset 4 zeroed,
/// then stored as little-endian.
pub fn stamp_v5_dir_block_crc(block: &mut [u8]) {
    block[V5_DIR_CRC_OFFSET..V5_DIR_CRC_OFFSET + 4].copy_from_slice(&[0u8; 4]);
    let crc = crc32c::crc32c(block);
    block[V5_DIR_CRC_OFFSET..V5_DIR_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
}

/// Encode a v5 block-format (XDB3) directory holding `entries` plus the
/// synthetic `.` and `..` records. `owner` is the directory inode
/// number; `parent` is the parent directory's inode (== `owner` for the
/// root); `uuid` is the volume meta UUID stamped into the v5 header.
///
/// Block layout (bytes):
/// ```text
///   0     XDB3 v5 data header (64 B):
///         magic crc blkno lsn uuid owner | bestfree[3] | pad
///  64     packed data_entry records (incl. "." and ".."), in order
///         ...
///         data_unused freetag region filling slack
/// blen-8 - 8*count   start of leaf entries
/// blen-8 .. blen-4   __be32 count
/// blen-4 .. blen     __be32 stale
/// ```
pub fn encode_v5_block_dir(
    dir_block_size: usize,
    owner: u64,
    parent: u64,
    entries: &[(String, u64, u8)],
    uuid: &[u8; 16],
    block_basic_blkno: u64,
) -> Result<Vec<u8>> {
    if dir_block_size < V5_DATA_HDR_SIZE + 32 {
        return Err(crate::Error::InvalidArgument(format!(
            "xfs: dir block size {dir_block_size} too small for v5 header"
        )));
    }
    let mut block = vec![0u8; dir_block_size];
    block[0..4].copy_from_slice(&XFS_DIR3_BLOCK_MAGIC.to_be_bytes());
    // crc at [4..8] is stamped at the end.
    block[8..16].copy_from_slice(&block_basic_blkno.to_be_bytes());
    // lsn at [16..24] zero
    block[24..40].copy_from_slice(uuid);
    block[40..48].copy_from_slice(&owner.to_be_bytes());
    // bestfree[3] (48..60) filled below; pad (60..64) zero

    let mut all: Vec<(String, u64, u8)> = Vec::with_capacity(entries.len() + 2);
    all.push((".".to_string(), owner, XFS_DIR3_FT_DIR));
    all.push(("..".to_string(), parent, XFS_DIR3_FT_DIR));
    all.extend(entries.iter().cloned());
    if all.len() > u32::MAX as usize {
        return Err(crate::Error::InvalidArgument(
            "xfs: block dir has too many entries".into(),
        ));
    }
    let leaf_count = all.len() as u32;
    let leaf_bytes = (leaf_count as usize) * 8;
    let tail_off = dir_block_size - 8 - leaf_bytes;

    let mut pos = V5_DATA_HDR_SIZE;
    let mut leaf_pairs: Vec<(u32, u32)> = Vec::with_capacity(all.len());
    for (name, inum, ft) in &all {
        let namelen = name.len();
        let raw_len = 8 + 1 + namelen + 1 + 2;
        let padded = (raw_len + 7) & !7;
        if pos + padded > tail_off {
            return Err(crate::Error::InvalidArgument(
                "xfs: block dir overflowed available space".into(),
            ));
        }
        block[pos..pos + 8].copy_from_slice(&inum.to_be_bytes());
        block[pos + 8] = namelen as u8;
        block[pos + 9..pos + 9 + namelen].copy_from_slice(name.as_bytes());
        block[pos + 9 + namelen] = *ft;
        let tag = (pos as u16).to_be_bytes();
        block[pos + padded - 2..pos + padded].copy_from_slice(&tag);

        let hashval = dahashname(name.as_bytes());
        let address = (pos / 8) as u32;
        leaf_pairs.push((hashval, address));
        pos += padded;
    }

    if tail_off > pos {
        let slack = tail_off - pos;
        if slack < 8 {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: block dir slack {slack} < 8 bytes"
            )));
        }
        block[pos..pos + 2].copy_from_slice(&XFS_DIR2_DATA_FREE_TAG.to_be_bytes());
        block[pos + 2..pos + 4].copy_from_slice(&(slack as u16).to_be_bytes());
        let tag_off = pos + slack - 2;
        block[tag_off..tag_off + 2].copy_from_slice(&(pos as u16).to_be_bytes());

        // bestfree[0] = this slack region.
        block[48..50].copy_from_slice(&(pos as u16).to_be_bytes());
        block[50..52].copy_from_slice(&(slack as u16).to_be_bytes());
    }

    leaf_pairs.sort_by_key(|p| p.0);
    for (i, (h, a)) in leaf_pairs.iter().enumerate() {
        let off = tail_off + i * 8;
        block[off..off + 4].copy_from_slice(&h.to_be_bytes());
        block[off + 4..off + 8].copy_from_slice(&a.to_be_bytes());
    }
    block[dir_block_size - 8..dir_block_size - 4].copy_from_slice(&leaf_count.to_be_bytes());
    block[dir_block_size - 4..dir_block_size].copy_from_slice(&0u32.to_be_bytes());

    stamp_v5_dir_block_crc(&mut block);
    Ok(block)
}

/// XFS directory-name hash (`xfs_da_hashname`). Used as the leaf-key in
/// every dabtree-indexed directory / attribute block.
pub fn dahashname(name: &[u8]) -> u32 {
    let mut hash: u32 = 0;
    let mut i = 0;
    while i + 4 <= name.len() {
        let n0 = name[i] as u32;
        let n1 = name[i + 1] as u32;
        let n2 = name[i + 2] as u32;
        let n3 = name[i + 3] as u32;
        hash = (n0 << 21) ^ (n1 << 14) ^ (n2 << 7) ^ n3 ^ hash.rotate_left(28);
        i += 4;
    }
    let remaining = name.len() - i;
    match remaining {
        3 => {
            let n0 = name[i] as u32;
            let n1 = name[i + 1] as u32;
            let n2 = name[i + 2] as u32;
            (n0 << 14) ^ (n1 << 7) ^ n2 ^ hash.rotate_left(21)
        }
        2 => {
            let n0 = name[i] as u32;
            let n1 = name[i + 1] as u32;
            (n0 << 7) ^ n1 ^ hash.rotate_left(14)
        }
        1 => {
            let n0 = name[i] as u32;
            n0 ^ hash.rotate_left(7)
        }
        _ => hash,
    }
}

/// Distinguish between block-format and leaf-format directories by peeking
/// at the magic of the first directory block. Returns `true` for block
/// format, `false` for data format (caller should then walk all data
/// blocks).
pub fn is_block_format(block: &[u8]) -> Result<bool> {
    if block.len() < 4 {
        return Err(crate::Error::InvalidImage(
            "xfs: dir first block too small".into(),
        ));
    }
    let magic = u32::from_be_bytes(block[0..4].try_into().unwrap());
    Ok(matches!(magic, XFS_DIR2_BLOCK_MAGIC | XFS_DIR3_BLOCK_MAGIC))
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
            &[
                ("a", 200, XFS_DIR3_FT_REG_FILE),
                ("dir", 201, XFS_DIR3_FT_DIR),
            ],
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
        let entries = vec![ShortformEntry {
            name: "x".into(),
            inumber: 7,
            ftype: XFS_DIR3_FT_REG_FILE,
        }];
        let g = shortform_to_generic(&entries);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].name, "x");
        assert_eq!(g[0].inode, 7);
        assert_eq!(g[0].kind, EntryKind::Regular);
    }

    /// Build a v5 data/block "data entry" record: inumber(8), namelen(1),
    /// name[], ftype(1), tag(2), padded to 8 bytes.
    fn build_v5_entry(inumber: u64, name: &str, ftype: u8) -> Vec<u8> {
        let raw_len = 8 + 1 + name.len() + 1 + 2;
        let padded = (raw_len + 7) & !7;
        let mut e = vec![0u8; padded];
        e[0..8].copy_from_slice(&inumber.to_be_bytes());
        e[8] = name.len() as u8;
        e[9..9 + name.len()].copy_from_slice(name.as_bytes());
        e[9 + name.len()] = ftype;
        // tag at last two bytes — we leave it zero, decoder doesn't validate.
        e
    }

    /// Build a v4 data entry record (no ftype byte): inumber(8), namelen(1),
    /// name[], tag(2), padded to 8 bytes.
    fn build_v4_entry(inumber: u64, name: &str) -> Vec<u8> {
        let raw_len = 8 + 1 + name.len() + 2;
        let padded = (raw_len + 7) & !7;
        let mut e = vec![0u8; padded];
        e[0..8].copy_from_slice(&inumber.to_be_bytes());
        e[8] = name.len() as u8;
        e[9..9 + name.len()].copy_from_slice(name.as_bytes());
        e
    }

    /// Build a synthetic v5 block-format directory: 64B header + entries +
    /// leaf array + tail. Inserts a `data_unused` freetag region between the
    /// last entry and the leaf-array reservation, matching what XFS writes
    /// on disk.
    fn build_v5_block_dir(entries: &[(u64, &str, u8)], dirsize: usize) -> Vec<u8> {
        let mut block = vec![0u8; dirsize];
        block[0..4].copy_from_slice(&XFS_DIR3_BLOCK_MAGIC.to_be_bytes());
        let mut pos = 64usize;
        for (ino, name, ft) in entries {
            let rec = build_v5_entry(*ino, name, *ft);
            block[pos..pos + rec.len()].copy_from_slice(&rec);
            pos += rec.len();
        }
        // Tail: count (4B BE) + stale (4B BE) at the end.
        let count = entries.len() as u32;
        block[dirsize - 8..dirsize - 4].copy_from_slice(&count.to_be_bytes());
        // Leaf-array reservation: count × 8 bytes immediately before tail.
        let entries_end = dirsize - 8 - (count as usize) * 8;
        // Fill the slack [pos..entries_end] with a data_unused region.
        if entries_end > pos {
            let slack = entries_end - pos;
            block[pos..pos + 2].copy_from_slice(&XFS_DIR2_DATA_FREE_TAG.to_be_bytes());
            block[pos + 2..pos + 4].copy_from_slice(&(slack as u16).to_be_bytes());
        }
        block
    }

    #[test]
    fn decode_v5_block_dir_smoke() {
        let block = build_v5_block_dir(
            &[
                (200, "hello", XFS_DIR3_FT_REG_FILE),
                (300, "subdir", XFS_DIR3_FT_DIR),
            ],
            4096,
        );
        let entries = decode_block_dir(&block, true).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "hello");
        assert_eq!(entries[0].inumber, 200);
        assert_eq!(entries[0].ftype, XFS_DIR3_FT_REG_FILE);
        assert_eq!(entries[1].name, "subdir");
        assert_eq!(entries[1].inumber, 300);
        assert_eq!(entries[1].ftype, XFS_DIR3_FT_DIR);
    }

    #[test]
    fn block_dir_handles_unused_region() {
        // Build a block with a freetag region between two entries.
        let dirsize = 4096usize;
        let mut block = vec![0u8; dirsize];
        block[0..4].copy_from_slice(&XFS_DIR3_BLOCK_MAGIC.to_be_bytes());
        let mut pos = 64;
        let rec1 = build_v5_entry(100, "a", XFS_DIR3_FT_REG_FILE);
        block[pos..pos + rec1.len()].copy_from_slice(&rec1);
        pos += rec1.len();
        // Insert a 16-byte unused region.
        block[pos..pos + 2].copy_from_slice(&XFS_DIR2_DATA_FREE_TAG.to_be_bytes());
        block[pos + 2..pos + 4].copy_from_slice(&16u16.to_be_bytes());
        pos += 16;
        let rec2 = build_v5_entry(101, "bee", XFS_DIR3_FT_DIR);
        block[pos..pos + rec2.len()].copy_from_slice(&rec2);
        pos += rec2.len();
        // Tail: count = 2.
        let count = 2u32;
        block[dirsize - 8..dirsize - 4].copy_from_slice(&count.to_be_bytes());
        // Fill the rest with a freetag block up to entries_end.
        let entries_end = dirsize - 8 - (count as usize) * 8;
        if entries_end > pos {
            let slack = entries_end - pos;
            block[pos..pos + 2].copy_from_slice(&XFS_DIR2_DATA_FREE_TAG.to_be_bytes());
            block[pos + 2..pos + 4].copy_from_slice(&(slack as u16).to_be_bytes());
        }
        let entries = decode_block_dir(&block, true).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "a");
        assert_eq!(entries[1].name, "bee");
    }

    #[test]
    fn decode_v5_data_block_smoke() {
        let mut block = vec![0u8; 4096];
        block[0..4].copy_from_slice(&XFS_DIR3_DATA_MAGIC.to_be_bytes());
        let mut pos = 64;
        for (ino, name, ft) in &[
            (10u64, "f1", XFS_DIR3_FT_REG_FILE),
            (11, "f2", XFS_DIR3_FT_REG_FILE),
            (12, "f3", XFS_DIR3_FT_REG_FILE),
        ] {
            let rec = build_v5_entry(*ino, name, *ft);
            block[pos..pos + rec.len()].copy_from_slice(&rec);
            pos += rec.len();
        }
        // Trailing freetag-region to indicate "end of records, free space
        // until end of block".
        block[pos..pos + 2].copy_from_slice(&XFS_DIR2_DATA_FREE_TAG.to_be_bytes());
        block[pos + 2..pos + 4].copy_from_slice(&((4096 - pos) as u16).to_be_bytes());
        let entries = decode_data_block(&block, true).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, "f1");
        assert_eq!(entries[2].name, "f3");
    }

    #[test]
    fn decode_v4_block_dir_no_ftype() {
        let dirsize = 4096usize;
        let mut block = vec![0u8; dirsize];
        block[0..4].copy_from_slice(&XFS_DIR2_BLOCK_MAGIC.to_be_bytes());
        let mut pos = 16; // v4 header is 16 bytes
        let rec = build_v4_entry(42, "abc");
        block[pos..pos + rec.len()].copy_from_slice(&rec);
        pos += rec.len();
        let count = 1u32;
        block[dirsize - 8..dirsize - 4].copy_from_slice(&count.to_be_bytes());
        // Insert freetag for the slack [pos .. entries_end).
        let entries_end = dirsize - 8 - (count as usize) * 8;
        if entries_end > pos {
            let slack = entries_end - pos;
            block[pos..pos + 2].copy_from_slice(&XFS_DIR2_DATA_FREE_TAG.to_be_bytes());
            block[pos + 2..pos + 4].copy_from_slice(&(slack as u16).to_be_bytes());
        }
        let entries = decode_block_dir(&block, false).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "abc");
        assert_eq!(entries[0].inumber, 42);
        assert_eq!(entries[0].ftype, XFS_DIR3_FT_UNKNOWN);
    }

    #[test]
    fn rejects_bad_magic() {
        let block = vec![0u8; 4096];
        let r = decode_block_dir(&block, true);
        assert!(matches!(r, Err(crate::Error::InvalidImage(_))));
    }

    #[test]
    fn is_block_format_detects() {
        let mut block = vec![0u8; 4096];
        block[0..4].copy_from_slice(&XFS_DIR3_BLOCK_MAGIC.to_be_bytes());
        assert!(is_block_format(&block).unwrap());
        block[0..4].copy_from_slice(&XFS_DIR3_DATA_MAGIC.to_be_bytes());
        assert!(!is_block_format(&block).unwrap());
        block[0..4].copy_from_slice(&XFS_DIR2_BLOCK_MAGIC.to_be_bytes());
        assert!(is_block_format(&block).unwrap());
    }
}
