//! SquashFS directory table walking.
//!
//! A directory's listing is a sequence of `(header, entry…)` runs stored
//! in the directory table. The header advertises how many entries share
//! the same inode metablock and a base inode number; each entry then
//! carries the uncompressed offset within that metablock plus a signed
//! difference from the base inode number.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::EntryKind;
use crate::fs::squashfs::Compression;
use crate::fs::squashfs::inode::{self, Inode};
use crate::fs::squashfs::metablock::MetadataReader;

const DIR_HEADER_SIZE: usize = 12;
const DIR_ENTRY_FIXED: usize = 8;

/// One raw directory entry as it lives on disk, paired with the inode
/// reference needed to fetch its inode.
#[derive(Debug, Clone)]
pub struct RawDirEntry {
    pub name: String,
    /// Metablock disk offset (relative to `inode_table_start`) containing
    /// the inode.
    pub inode_block: u32,
    /// Uncompressed offset within that metablock.
    pub inode_offset: u16,
    /// Type as stored in the entry — always the *basic* form even for
    /// extended inodes.
    pub inode_type: u16,
    /// Inode number reconstructed from the header's base.
    pub inode_number: u32,
}

/// Read the full set of entries for a directory.
pub fn read_directory_entries(
    dev: &mut dyn BlockDevice,
    dir_table_start: u64,
    compression: Compression,
    block_index: u32,
    block_offset: u16,
    listing_size: u32,
) -> Result<Vec<RawDirEntry>> {
    let mut mr = MetadataReader::new(dir_table_start, compression);
    let mut block: u64 = block_index as u64;
    let mut offset = block_offset as usize;
    let mut remaining = listing_size as usize;
    let mut out = Vec::new();

    while remaining >= DIR_HEADER_SIZE {
        let (hdr, nb, no) = mr.read(dev, block, offset, DIR_HEADER_SIZE)?;
        let count = u32::from_le_bytes(hdr[0..4].try_into().unwrap()).saturating_add(1);
        let start_block = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        let base_inode = u32::from_le_bytes(hdr[8..12].try_into().unwrap());
        block = nb;
        offset = no;
        remaining = remaining.saturating_sub(DIR_HEADER_SIZE);
        for _ in 0..count {
            if remaining < DIR_ENTRY_FIXED {
                return Err(crate::Error::InvalidImage(
                    "squashfs: directory listing truncated before entry fixed part".into(),
                ));
            }
            let (fixed, nb2, no2) = mr.read(dev, block, offset, DIR_ENTRY_FIXED)?;
            let entry_offset = u16::from_le_bytes(fixed[0..2].try_into().unwrap());
            let inode_offset_signed = i16::from_le_bytes(fixed[2..4].try_into().unwrap());
            let inode_type = u16::from_le_bytes(fixed[4..6].try_into().unwrap());
            let name_size = u16::from_le_bytes(fixed[6..8].try_into().unwrap()) as usize + 1;
            block = nb2;
            offset = no2;
            remaining = remaining.saturating_sub(DIR_ENTRY_FIXED);
            if remaining < name_size {
                return Err(crate::Error::InvalidImage(
                    "squashfs: directory entry name truncated".into(),
                ));
            }
            let (name_bytes, nb3, no3) = mr.read(dev, block, offset, name_size)?;
            block = nb3;
            offset = no3;
            remaining = remaining.saturating_sub(name_size);
            let name = String::from_utf8(name_bytes).map_err(|e| {
                crate::Error::InvalidImage(format!("squashfs: directory name not utf-8: {e}"))
            })?;
            let inode_number = (base_inode as i64 + inode_offset_signed as i64) as u32;
            out.push(RawDirEntry {
                name,
                inode_block: start_block,
                inode_offset: entry_offset,
                inode_type,
                inode_number,
            });
        }
    }
    Ok(out)
}

/// Convert the entry's basic-type field into our public [`EntryKind`].
pub fn entry_kind_from_type(t: u16) -> EntryKind {
    match t {
        inode::INODE_BASIC_DIR | inode::INODE_EXT_DIR => EntryKind::Dir,
        inode::INODE_BASIC_FILE | inode::INODE_EXT_FILE => EntryKind::Regular,
        inode::INODE_BASIC_SYMLINK | inode::INODE_EXT_SYMLINK => EntryKind::Symlink,
        inode::INODE_BASIC_BLOCK | inode::INODE_EXT_BLOCK => EntryKind::Block,
        inode::INODE_BASIC_CHAR | inode::INODE_EXT_CHAR => EntryKind::Char,
        inode::INODE_BASIC_FIFO | inode::INODE_EXT_FIFO => EntryKind::Fifo,
        inode::INODE_BASIC_SOCKET | inode::INODE_EXT_SOCKET => EntryKind::Socket,
        _ => EntryKind::Unknown,
    }
}

/// Split a path into trimmed components; "/", "" and "." all become an
/// empty slice (= the root directory).
pub fn split_path(path: &str) -> Vec<&str> {
    path.split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect()
}

/// Walk from the root inode down to `path`, returning the inode at the
/// final component. The root inode is loaded once and decoded; each path
/// component triggers one directory listing.
pub fn resolve_path(
    dev: &mut dyn BlockDevice,
    inode_table_start: u64,
    dir_table_start: u64,
    compression: Compression,
    root_ref: u64,
    block_size: u32,
    path: &str,
) -> Result<Inode> {
    let mut mr = MetadataReader::new(inode_table_start, compression);
    let (mut block, mut offset) = inode::inode_ref(root_ref);
    let mut current = inode::read_inode(dev, &mut mr, block, offset, block_size)?;
    for part in split_path(path) {
        let dir = match &current {
            Inode::Dir(d) => d.clone(),
            _ => {
                return Err(crate::Error::InvalidArgument(format!(
                    "squashfs: component {part:?} is not a directory"
                )));
            }
        };
        let entries = read_directory_entries(
            dev,
            dir_table_start,
            compression,
            dir.block_index,
            dir.block_offset,
            dir.file_size,
        )?;
        let entry = entries
            .into_iter()
            .find(|e| e.name == part)
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!("squashfs: no such entry {part:?}"))
            })?;
        block = entry.inode_block as u64;
        offset = entry.inode_offset;
        current = inode::read_inode(dev, &mut mr, block, offset, block_size)?;
    }
    Ok(current)
}
