//! SquashFS fragment table reader.
//!
//! The fragment table is two layers of indirection:
//!
//! 1. `fragment_table_start` in the superblock points to a contiguous,
//!    *uncompressed* array of `u64` little-endian values, one per metablock
//!    holding fragment entries.
//! 2. Each of those metablocks (8 KiB uncompressed each → up to 512
//!    16-byte entries) stores [`FragmentEntry`] records in order.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::squashfs::Compression;
use crate::fs::squashfs::metablock::MetadataReader;

/// On-disk fragment table entry. The unused trailing `u32` is ignored.
#[derive(Debug, Clone, Copy)]
pub struct FragmentEntry {
    /// Absolute archive offset of the fragment block.
    pub start: u64,
    /// Encoded size word. Bit 24 (`1 << 24`) set ⇒ stored uncompressed.
    pub size_word: u32,
}

impl FragmentEntry {
    pub fn is_uncompressed(&self) -> bool {
        self.size_word & 0x0100_0000 != 0
    }
    pub fn on_disk_size(&self) -> u32 {
        self.size_word & 0x00FF_FFFF
    }
}

/// Look up fragment index `idx` in the on-disk fragment table.
///
/// `fragment_count` is taken from the superblock and bounds-checked.
/// `fragment_table_start` points at the L1 array of metablock locations.
pub fn read_fragment(
    dev: &mut dyn BlockDevice,
    fragment_table_start: u64,
    fragment_count: u32,
    compression: Compression,
    idx: u32,
) -> Result<FragmentEntry> {
    if idx >= fragment_count {
        return Err(crate::Error::InvalidImage(format!(
            "squashfs: fragment index {idx} >= fragment_count {fragment_count}"
        )));
    }
    // 512 entries per 8 KiB metablock.
    let metablock_idx = (idx / 512) as u64;
    let entry_idx_in_mb = (idx % 512) as usize;
    // Read the L1 pointer (u64) at fragment_table_start + 8*metablock_idx.
    let mut ptr_buf = [0u8; 8];
    dev.read_at(fragment_table_start + 8 * metablock_idx, &mut ptr_buf)?;
    let mb_disk = u64::from_le_bytes(ptr_buf);
    // Now read entry_idx_in_mb * 16 bytes into the metablock.
    let mut mr = MetadataReader::new(mb_disk, compression);
    let (bytes, _, _) = mr.read(dev, 0, entry_idx_in_mb * 16, 16)?;
    let start = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let size_word = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    Ok(FragmentEntry { start, size_word })
}
