//! SquashFS inode decoding.
//!
//! Every inode begins with a 16-byte common header. The remaining payload
//! depends on the type field. We decode the variants we need to walk the
//! filesystem (directories, regular files, symlinks) and produce a compact
//! summary for the rarer device / fifo / socket types.
//!
//! A few struct fields and helper methods on [`Inode`] aren't consumed by
//! this module yet — they exist because the integrator will surface header
//! metadata (mtime, perms, parent_inode, etc.) through `AnyFs`.
#![allow(dead_code)]

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::EntryKind;
use crate::fs::squashfs::metablock::MetadataReader;

// Inode type IDs from the SquashFS 4.0 binary format.
pub const INODE_BASIC_DIR: u16 = 1;
pub const INODE_BASIC_FILE: u16 = 2;
pub const INODE_BASIC_SYMLINK: u16 = 3;
pub const INODE_BASIC_BLOCK: u16 = 4;
pub const INODE_BASIC_CHAR: u16 = 5;
pub const INODE_BASIC_FIFO: u16 = 6;
pub const INODE_BASIC_SOCKET: u16 = 7;
pub const INODE_EXT_DIR: u16 = 8;
pub const INODE_EXT_FILE: u16 = 9;
pub const INODE_EXT_SYMLINK: u16 = 10;
pub const INODE_EXT_BLOCK: u16 = 11;
pub const INODE_EXT_CHAR: u16 = 12;
pub const INODE_EXT_FIFO: u16 = 13;
pub const INODE_EXT_SOCKET: u16 = 14;

/// Common inode header carried by every inode.
#[derive(Debug, Clone, Copy)]
pub struct InodeHeader {
    pub kind: u16,
    pub permissions: u16,
    pub uid_idx: u16,
    pub gid_idx: u16,
    pub mtime: u32,
    pub inode_number: u32,
}

impl InodeHeader {
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < 16 {
            return Err(crate::Error::InvalidImage(
                "squashfs: short inode header".into(),
            ));
        }
        Ok(Self {
            kind: u16::from_le_bytes(buf[0..2].try_into().unwrap()),
            permissions: u16::from_le_bytes(buf[2..4].try_into().unwrap()),
            uid_idx: u16::from_le_bytes(buf[4..6].try_into().unwrap()),
            gid_idx: u16::from_le_bytes(buf[6..8].try_into().unwrap()),
            mtime: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            inode_number: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
        })
    }
}

/// Directory inode (covers both basic and extended forms). Holds the
/// pointer into the directory table (`block_index`, `block_offset`) and
/// the total `file_size` of the listing data.
#[derive(Debug, Clone)]
pub struct DirInode {
    pub header: InodeHeader,
    /// Relative offset of the metadata block in the directory table that
    /// contains this directory's listing.
    pub block_index: u32,
    /// Uncompressed offset within that block where the first directory
    /// header sits.
    pub block_offset: u16,
    /// Total bytes of listing data, including headers. For basic
    /// directories the on-disk field is stored size+3; we normalise to
    /// the true size here. `0` means an empty directory.
    pub file_size: u32,
    pub parent_inode: u32,
    /// Extended directories carry an xattr index; basic directories store
    /// `u32::MAX`.
    pub xattr_index: u32,
}

/// Regular file inode (basic + extended forms). Sizes are widened to
/// `u64` so we can decode `LREG` images > 4 GiB without changing the
/// downstream code paths.
#[derive(Debug, Clone)]
pub struct FileInode {
    pub header: InodeHeader,
    pub blocks_start: u64,
    pub file_size: u64,
    /// `0xFFFFFFFF` ⇒ no fragment tail.
    pub fragment_index: u32,
    /// Uncompressed offset of the file's tail within the fragment block.
    pub fragment_offset: u32,
    /// One entry per full data block.
    pub block_sizes: Vec<u32>,
    /// Extended files carry an xattr index; basic files store `u32::MAX`.
    pub xattr_index: u32,
}

impl FileInode {
    pub fn has_fragment(&self) -> bool {
        self.fragment_index != 0xFFFF_FFFF
    }
}

/// Symlink inode (basic + extended forms). Extended symlinks carry an
/// `xattr_index`; basic symlinks store `u32::MAX` here.
#[derive(Debug, Clone)]
pub struct SymlinkInode {
    pub header: InodeHeader,
    pub target: String,
    pub xattr_index: u32,
}

/// A decoded inode of any type. For inode kinds we don't fully decode
/// (block/char/fifo/socket) we still surface the header so listings show
/// the right name + kind.
#[derive(Debug, Clone)]
pub enum Inode {
    Dir(DirInode),
    File(FileInode),
    Symlink(SymlinkInode),
    Other {
        header: InodeHeader,
        kind: EntryKind,
    },
}

impl Inode {
    pub fn header(&self) -> &InodeHeader {
        match self {
            Inode::Dir(d) => &d.header,
            Inode::File(f) => &f.header,
            Inode::Symlink(s) => &s.header,
            Inode::Other { header, .. } => header,
        }
    }

    pub fn entry_kind(&self) -> EntryKind {
        match self {
            Inode::Dir(_) => EntryKind::Dir,
            Inode::File(_) => EntryKind::Regular,
            Inode::Symlink(_) => EntryKind::Symlink,
            Inode::Other { kind, .. } => *kind,
        }
    }

    /// Xattr table index, or `u32::MAX` if the inode has no xattrs.
    pub fn xattr_index(&self) -> u32 {
        match self {
            Inode::Dir(d) => d.xattr_index,
            Inode::File(f) => f.xattr_index,
            Inode::Symlink(s) => s.xattr_index,
            Inode::Other { .. } => u32::MAX,
        }
    }
}

/// Convert a 48-bit inode reference (as stored in the superblock's
/// `root_inode` or in directory headers' `start`) into `(block, offset)`.
/// Per the spec: the lower 16 bits are the uncompressed offset, the next
/// 32 bits are the metadata-block disk offset relative to
/// `inode_table_start`; the top 16 bits are unused.
pub fn inode_ref(reference: u64) -> (u64, u16) {
    let offset = (reference & 0xFFFF) as u16;
    let block = (reference >> 16) & 0xFFFF_FFFF;
    (block, offset)
}

/// Read one inode at the given reference. Returns the decoded inode plus
/// the file_size hint for directories (caller uses this for listing
/// length).
pub fn read_inode(
    dev: &mut dyn BlockDevice,
    mr: &mut MetadataReader,
    block_rel: u64,
    offset: u16,
    block_size: u32,
) -> Result<Inode> {
    let header_bytes = read_bytes(dev, mr, block_rel, offset as usize, 16)?;
    let header = InodeHeader::decode(&header_bytes.0)?;

    match header.kind {
        INODE_BASIC_DIR => decode_basic_dir(dev, mr, header_bytes.1, header_bytes.2, header),
        INODE_EXT_DIR => decode_ext_dir(dev, mr, header_bytes.1, header_bytes.2, header),
        INODE_BASIC_FILE => {
            decode_basic_file(dev, mr, header_bytes.1, header_bytes.2, header, block_size)
        }
        INODE_EXT_FILE => {
            decode_ext_file(dev, mr, header_bytes.1, header_bytes.2, header, block_size)
        }
        INODE_BASIC_SYMLINK | INODE_EXT_SYMLINK => {
            decode_symlink(dev, mr, header_bytes.1, header_bytes.2, header)
        }
        INODE_BASIC_BLOCK | INODE_EXT_BLOCK => Ok(Inode::Other {
            header,
            kind: EntryKind::Block,
        }),
        INODE_BASIC_CHAR | INODE_EXT_CHAR => Ok(Inode::Other {
            header,
            kind: EntryKind::Char,
        }),
        INODE_BASIC_FIFO | INODE_EXT_FIFO => Ok(Inode::Other {
            header,
            kind: EntryKind::Fifo,
        }),
        INODE_BASIC_SOCKET | INODE_EXT_SOCKET => Ok(Inode::Other {
            header,
            kind: EntryKind::Socket,
        }),
        other => Err(crate::Error::InvalidImage(format!(
            "squashfs: unknown inode type {other}"
        ))),
    }
}

fn decode_basic_dir(
    dev: &mut dyn BlockDevice,
    mr: &mut MetadataReader,
    block_rel: u64,
    offset: usize,
    header: InodeHeader,
) -> Result<Inode> {
    // 16 bytes after the header.
    let (buf, _, _) = read_bytes(dev, mr, block_rel, offset, 16)?;
    let block_index = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let _link_count = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let raw_file_size = u16::from_le_bytes(buf[8..10].try_into().unwrap());
    let block_offset = u16::from_le_bytes(buf[10..12].try_into().unwrap());
    let parent_inode = u32::from_le_bytes(buf[12..16].try_into().unwrap());
    let file_size = if raw_file_size <= 3 {
        0
    } else {
        (raw_file_size as u32) - 3
    };
    Ok(Inode::Dir(DirInode {
        header,
        block_index,
        block_offset,
        file_size,
        parent_inode,
        xattr_index: u32::MAX,
    }))
}

fn decode_ext_dir(
    dev: &mut dyn BlockDevice,
    mr: &mut MetadataReader,
    block_rel: u64,
    offset: usize,
    header: InodeHeader,
) -> Result<Inode> {
    // 24 bytes after the header (we skip the index list that follows).
    let (buf, _, _) = read_bytes(dev, mr, block_rel, offset, 24)?;
    let _link_count = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let raw_file_size = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let block_index = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let parent_inode = u32::from_le_bytes(buf[12..16].try_into().unwrap());
    let _index_count = u16::from_le_bytes(buf[16..18].try_into().unwrap());
    let block_offset = u16::from_le_bytes(buf[18..20].try_into().unwrap());
    let xattr_index = u32::from_le_bytes(buf[20..24].try_into().unwrap());
    // The extended directory's file_size is stored as size+3 too, per the
    // dr-emann reference implementation; subtract to recover the real size.
    let file_size = raw_file_size.saturating_sub(3);
    Ok(Inode::Dir(DirInode {
        header,
        block_index,
        block_offset,
        file_size,
        parent_inode,
        xattr_index,
    }))
}

fn decode_basic_file(
    dev: &mut dyn BlockDevice,
    mr: &mut MetadataReader,
    block_rel: u64,
    offset: usize,
    header: InodeHeader,
    block_size: u32,
) -> Result<Inode> {
    let (buf, b2, o2) = read_bytes(dev, mr, block_rel, offset, 16)?;
    let blocks_start = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as u64;
    let fragment_index = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let fragment_offset = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let file_size = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as u64;
    let num_full_blocks =
        full_block_count(file_size, block_size as u64, fragment_index != 0xFFFF_FFFF);
    let (block_sizes, _, _) = read_block_sizes(dev, mr, b2, o2, num_full_blocks)?;
    Ok(Inode::File(FileInode {
        header,
        blocks_start,
        file_size,
        fragment_index,
        fragment_offset,
        block_sizes,
        xattr_index: u32::MAX,
    }))
}

fn decode_ext_file(
    dev: &mut dyn BlockDevice,
    mr: &mut MetadataReader,
    block_rel: u64,
    offset: usize,
    header: InodeHeader,
    block_size: u32,
) -> Result<Inode> {
    let (buf, b2, o2) = read_bytes(dev, mr, block_rel, offset, 40)?;
    let blocks_start = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let file_size = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    let _sparse = u64::from_le_bytes(buf[16..24].try_into().unwrap());
    let _link_count = u32::from_le_bytes(buf[24..28].try_into().unwrap());
    let fragment_index = u32::from_le_bytes(buf[28..32].try_into().unwrap());
    let fragment_offset = u32::from_le_bytes(buf[32..36].try_into().unwrap());
    let xattr_index = u32::from_le_bytes(buf[36..40].try_into().unwrap());
    let num_full_blocks =
        full_block_count(file_size, block_size as u64, fragment_index != 0xFFFF_FFFF);
    let (block_sizes, _, _) = read_block_sizes(dev, mr, b2, o2, num_full_blocks)?;
    Ok(Inode::File(FileInode {
        header,
        blocks_start,
        file_size,
        fragment_index,
        fragment_offset,
        block_sizes,
        xattr_index,
    }))
}

fn decode_symlink(
    dev: &mut dyn BlockDevice,
    mr: &mut MetadataReader,
    block_rel: u64,
    offset: usize,
    header: InodeHeader,
) -> Result<Inode> {
    let (head, b2, o2) = read_bytes(dev, mr, block_rel, offset, 8)?;
    let _link_count = u32::from_le_bytes(head[0..4].try_into().unwrap());
    let target_size = u32::from_le_bytes(head[4..8].try_into().unwrap());
    // SquashFS imposes 65535 as a hard upper bound; reject anything more.
    if target_size > 65535 {
        return Err(crate::Error::InvalidImage(format!(
            "squashfs: symlink target size {target_size} exceeds 65535"
        )));
    }
    let (raw, b3, o3) = read_bytes(dev, mr, b2, o2, target_size as usize)?;
    let target = String::from_utf8(raw).map_err(|e| {
        crate::Error::InvalidImage(format!("squashfs: symlink target is not utf-8: {e}"))
    })?;
    let xattr_index = if header.kind == INODE_EXT_SYMLINK {
        let (xbuf, _, _) = read_bytes(dev, mr, b3, o3, 4)?;
        u32::from_le_bytes(xbuf[0..4].try_into().unwrap())
    } else {
        u32::MAX
    };
    Ok(Inode::Symlink(SymlinkInode {
        header,
        target,
        xattr_index,
    }))
}

/// Number of full data blocks for a file given its size, the FS block
/// size, and whether the tail is packed into a fragment block.
fn full_block_count(file_size: u64, block_size: u64, has_fragment: bool) -> usize {
    if block_size == 0 {
        return 0;
    }
    if has_fragment {
        (file_size / block_size) as usize
    } else {
        file_size.div_ceil(block_size) as usize
    }
}

fn read_block_sizes(
    dev: &mut dyn BlockDevice,
    mr: &mut MetadataReader,
    block_rel: u64,
    offset: usize,
    count: usize,
) -> Result<(Vec<u32>, u64, usize)> {
    let need = count * 4;
    let (raw, nb, no) = read_bytes(dev, mr, block_rel, offset, need)?;
    let mut sizes = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * 4;
        sizes.push(u32::from_le_bytes(raw[off..off + 4].try_into().unwrap()));
    }
    Ok((sizes, nb, no))
}

/// Thin wrapper around `MetadataReader::read` that returns the new cursor
/// as `(block_rel, offset)` directly.
fn read_bytes(
    dev: &mut dyn BlockDevice,
    mr: &mut MetadataReader,
    block_rel: u64,
    offset: usize,
    n: usize,
) -> Result<(Vec<u8>, u64, usize)> {
    mr.read(dev, block_rel, offset, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_block_count_with_fragment() {
        assert_eq!(full_block_count(0, 4096, true), 0);
        assert_eq!(full_block_count(4095, 4096, true), 0);
        assert_eq!(full_block_count(4096, 4096, true), 1);
        assert_eq!(full_block_count(4097, 4096, true), 1);
    }

    #[test]
    fn full_block_count_without_fragment() {
        assert_eq!(full_block_count(0, 4096, false), 0);
        assert_eq!(full_block_count(1, 4096, false), 1);
        assert_eq!(full_block_count(4096, 4096, false), 1);
        assert_eq!(full_block_count(4097, 4096, false), 2);
    }

    #[test]
    fn inode_ref_splits_block_and_offset() {
        // block = 0x1234, offset = 0xABCD
        let r = (0x1234u64 << 16) | 0xABCDu64;
        let (b, o) = inode_ref(r);
        assert_eq!(b, 0x1234);
        assert_eq!(o, 0xABCD);
    }
}
