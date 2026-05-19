//! HFS+ volume header — TN1150 "Volume Header".
//!
//! The volume header lives at byte offset 1024 from the start of the
//! volume (sector 2 at 512-byte sectors) and is followed by an
//! alternate copy near the end of the volume. We only consume the
//! primary copy.
//!
//! All multi-byte integers in HFS+ are big-endian.
//!
//! ## On-disk layout (TN1150, "HFSPlusVolumeHeader")
//!
//! ```text
//! offset  size  field
//! 0x000   2     signature       ("H+" = 0x482B, "HX" = 0x4858)
//! 0x002   2     version         (4 for H+, 5 for HX)
//! 0x004   4     attributes
//! 0x008   4     lastMountedVersion
//! 0x00C   4     journalInfoBlock
//! 0x010   4     createDate
//! 0x014   4     modifyDate
//! 0x018   4     backupDate
//! 0x01C   4     checkedDate
//! 0x020   4     fileCount
//! 0x024   4     folderCount
//! 0x028   4     blockSize
//! 0x02C   4     totalBlocks
//! 0x030   4     freeBlocks
//! 0x034   4     nextAllocation
//! 0x038   4     rsrcClumpSize
//! 0x03C   4     dataClumpSize
//! 0x040   4     nextCatalogID
//! 0x044   4     writeCount
//! 0x048   8     encodingsBitmap
//! 0x050   32    finderInfo[8]   (8 u32 values)
//! 0x070   80    allocationFile  (HFSPlusForkData, 80 bytes)
//! 0x0C0   80    extentsFile
//! 0x110   80    catalogFile
//! 0x160   80    attributesFile
//! 0x1B0   80    startupFile
//! ```
//!
//! `HFSPlusForkData` (80 bytes, TN1150 "HFSPlusForkData"):
//!
//! ```text
//! offset  size  field
//! 0x00    8     logicalSize     (file length in bytes)
//! 0x08    4     clumpSize
//! 0x0C    4     totalBlocks     (in this fork)
//! 0x10    64    extents[8]      (HFSPlusExtentDescriptor[8])
//! ```
//!
//! `HFSPlusExtentDescriptor` is `{ u32 startBlock, u32 blockCount }` (8 bytes).

use crate::Result;
use crate::block::BlockDevice;

/// Byte offset of the primary HFS+ volume header from the start of the volume.
pub const VOLUME_HEADER_OFFSET: u64 = 1024;

/// Encoded size of an HFSPlusForkData record.
pub const FORK_DATA_SIZE: usize = 80;

/// Number of extents stored inline in a fork data record.
pub const FORK_EXTENT_COUNT: usize = 8;

/// `"H+"` (HFS+) signature, big-endian.
pub const SIG_HFS_PLUS: [u8; 2] = *b"H+";

/// `"HX"` (HFSX, case-sensitive variant) signature, big-endian.
pub const SIG_HFSX: [u8; 2] = *b"HX";

/// One `(startBlock, blockCount)` allocation-block extent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExtentDescriptor {
    /// First allocation block of this run.
    pub start_block: u32,
    /// Number of contiguous allocation blocks.
    pub block_count: u32,
}

impl ExtentDescriptor {
    /// Decode an 8-byte extent descriptor (big-endian).
    pub fn decode(buf: &[u8; 8]) -> Self {
        Self {
            start_block: u32::from_be_bytes(buf[0..4].try_into().unwrap()),
            block_count: u32::from_be_bytes(buf[4..8].try_into().unwrap()),
        }
    }
}

/// HFSPlusForkData — the per-fork allocation map embedded in the
/// volume header (for the special files) and in catalog file records
/// (for user files).
#[derive(Debug, Clone, Copy)]
pub struct ForkData {
    /// File length in bytes.
    pub logical_size: u64,
    /// Allocation hint; we ignore it on read.
    pub clump_size: u32,
    /// Total number of allocation blocks this fork uses.
    pub total_blocks: u32,
    /// First eight extents inline; the rest (if any) live in the
    /// extents-overflow B-tree.
    pub extents: [ExtentDescriptor; FORK_EXTENT_COUNT],
}

impl ForkData {
    /// Decode an 80-byte HFSPlusForkData record.
    pub fn decode(buf: &[u8; FORK_DATA_SIZE]) -> Self {
        let logical_size = u64::from_be_bytes(buf[0..8].try_into().unwrap());
        let clump_size = u32::from_be_bytes(buf[8..12].try_into().unwrap());
        let total_blocks = u32::from_be_bytes(buf[12..16].try_into().unwrap());
        let mut extents = [ExtentDescriptor::default(); FORK_EXTENT_COUNT];
        for (i, slot) in extents.iter_mut().enumerate() {
            let off = 16 + i * 8;
            let mut e = [0u8; 8];
            e.copy_from_slice(&buf[off..off + 8]);
            *slot = ExtentDescriptor::decode(&e);
        }
        Self {
            logical_size,
            clump_size,
            total_blocks,
            extents,
        }
    }

    /// Sum of `block_count` across the inline extents — used to
    /// determine whether overflow extents would be required.
    pub fn inline_blocks(&self) -> u64 {
        self.extents.iter().map(|e| u64::from(e.block_count)).sum()
    }
}

/// Decoded volume header. Only the fields the rest of this module
/// actually consumes are kept; the remainder is dropped.
#[derive(Debug, Clone)]
pub struct VolumeHeader {
    /// Either `SIG_HFS_PLUS` or `SIG_HFSX`.
    pub signature: [u8; 2],
    /// Format version (4 for H+, 5 for HX).
    pub version: u16,
    /// Attribute flags.
    pub attributes: u32,
    /// Allocation block size, in bytes.
    pub block_size: u32,
    /// Total number of allocation blocks in the volume.
    pub total_blocks: u32,
    /// Number of free allocation blocks.
    pub free_blocks: u32,
    /// CNID to use for the next new catalog node.
    pub next_catalog_id: u32,
    /// Fork data for the allocation bitmap file.
    pub allocation_file: ForkData,
    /// Fork data for the extents-overflow B-tree file.
    pub extents_file: ForkData,
    /// Fork data for the catalog B-tree file.
    pub catalog_file: ForkData,
    /// Fork data for the attributes B-tree file.
    pub attributes_file: ForkData,
    /// Fork data for the startup file (rarely used).
    pub startup_file: ForkData,
}

impl VolumeHeader {
    /// Total encoded size of the volume header proper.
    pub const ENCODED_SIZE: usize = 512;

    /// Decode a 512-byte volume-header buffer.
    pub fn decode(buf: &[u8; Self::ENCODED_SIZE]) -> Result<Self> {
        let signature = [buf[0], buf[1]];
        if signature != SIG_HFS_PLUS && signature != SIG_HFSX {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: bad volume signature {:02x}{:02x} (expected 'H+' or 'HX')",
                signature[0], signature[1]
            )));
        }
        let version = u16::from_be_bytes(buf[2..4].try_into().unwrap());
        let attributes = u32::from_be_bytes(buf[4..8].try_into().unwrap());
        let block_size = u32::from_be_bytes(buf[40..44].try_into().unwrap());
        let total_blocks = u32::from_be_bytes(buf[44..48].try_into().unwrap());
        let free_blocks = u32::from_be_bytes(buf[48..52].try_into().unwrap());
        let next_catalog_id = u32::from_be_bytes(buf[64..68].try_into().unwrap());

        // Five HFSPlusForkData blocks at offsets 0x70, 0xC0, 0x110,
        // 0x160, 0x1B0.
        let fd = |off: usize| -> ForkData {
            let mut tmp = [0u8; FORK_DATA_SIZE];
            tmp.copy_from_slice(&buf[off..off + FORK_DATA_SIZE]);
            ForkData::decode(&tmp)
        };
        let allocation_file = fd(0x070);
        let extents_file = fd(0x0C0);
        let catalog_file = fd(0x110);
        let attributes_file = fd(0x160);
        let startup_file = fd(0x1B0);

        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: block_size {block_size} is not a positive power of two"
            )));
        }

        Ok(Self {
            signature,
            version,
            attributes,
            block_size,
            total_blocks,
            free_blocks,
            next_catalog_id,
            allocation_file,
            extents_file,
            catalog_file,
            attributes_file,
            startup_file,
        })
    }

    /// Whether this volume is the case-sensitive HFSX variant.
    pub fn is_hfsx(&self) -> bool {
        self.signature == SIG_HFSX
    }
}

/// Read and decode the primary volume header at offset 1024.
pub fn read_volume_header(dev: &mut dyn BlockDevice) -> Result<VolumeHeader> {
    let size = dev.total_size();
    if size < VOLUME_HEADER_OFFSET + VolumeHeader::ENCODED_SIZE as u64 {
        return Err(crate::Error::InvalidImage(format!(
            "hfs+: device size {size} too small to hold a volume header"
        )));
    }
    let mut buf = [0u8; VolumeHeader::ENCODED_SIZE];
    dev.read_at(VOLUME_HEADER_OFFSET, &mut buf)?;
    VolumeHeader::decode(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 512-byte volume-header buffer with the given block size
    /// and one inline extent for the catalog fork.
    fn synth_header(block_size: u32, total_blocks: u32) -> [u8; VolumeHeader::ENCODED_SIZE] {
        let mut b = [0u8; VolumeHeader::ENCODED_SIZE];
        b[0..2].copy_from_slice(&SIG_HFS_PLUS);
        b[2..4].copy_from_slice(&4u16.to_be_bytes());
        b[4..8].copy_from_slice(&0u32.to_be_bytes()); // attributes
        b[40..44].copy_from_slice(&block_size.to_be_bytes());
        b[44..48].copy_from_slice(&total_blocks.to_be_bytes());
        b[48..52].copy_from_slice(&0u32.to_be_bytes()); // free
        b[64..68].copy_from_slice(&100u32.to_be_bytes()); // next CNID

        // Catalog fork at 0x110: logicalSize 8192, clump 0, totalBlocks 2,
        // one extent (start_block=5, block_count=2).
        let cat = 0x110usize;
        b[cat..cat + 8].copy_from_slice(&8192u64.to_be_bytes());
        b[cat + 8..cat + 12].copy_from_slice(&0u32.to_be_bytes());
        b[cat + 12..cat + 16].copy_from_slice(&2u32.to_be_bytes());
        b[cat + 16..cat + 20].copy_from_slice(&5u32.to_be_bytes());
        b[cat + 20..cat + 24].copy_from_slice(&2u32.to_be_bytes());
        b
    }

    #[test]
    fn decode_valid_header() {
        let buf = synth_header(4096, 1024);
        let vh = VolumeHeader::decode(&buf).unwrap();
        assert_eq!(vh.signature, *b"H+");
        assert_eq!(vh.version, 4);
        assert_eq!(vh.block_size, 4096);
        assert_eq!(vh.total_blocks, 1024);
        assert_eq!(vh.next_catalog_id, 100);
        assert_eq!(vh.catalog_file.logical_size, 8192);
        assert_eq!(vh.catalog_file.total_blocks, 2);
        assert_eq!(vh.catalog_file.extents[0].start_block, 5);
        assert_eq!(vh.catalog_file.extents[0].block_count, 2);
        assert!(!vh.is_hfsx());
    }

    #[test]
    fn rejects_bad_signature() {
        let mut buf = synth_header(4096, 1024);
        buf[0] = b'X';
        buf[1] = b'X';
        assert!(VolumeHeader::decode(&buf).is_err());
    }

    #[test]
    fn rejects_non_power_of_two_block_size() {
        let buf = synth_header(3000, 1024);
        assert!(VolumeHeader::decode(&buf).is_err());
    }

    #[test]
    fn fork_data_decodes_inline_blocks() {
        let mut buf = [0u8; FORK_DATA_SIZE];
        buf[0..8].copy_from_slice(&1_048_576u64.to_be_bytes()); // logical_size
        buf[12..16].copy_from_slice(&3u32.to_be_bytes()); // total_blocks
        // extent 0: 100 .. +1
        buf[16..20].copy_from_slice(&100u32.to_be_bytes());
        buf[20..24].copy_from_slice(&1u32.to_be_bytes());
        // extent 1: 200 .. +2
        buf[24..28].copy_from_slice(&200u32.to_be_bytes());
        buf[28..32].copy_from_slice(&2u32.to_be_bytes());

        let fd = ForkData::decode(&buf);
        assert_eq!(fd.logical_size, 1_048_576);
        assert_eq!(fd.total_blocks, 3);
        assert_eq!(fd.extents[0].start_block, 100);
        assert_eq!(fd.extents[0].block_count, 1);
        assert_eq!(fd.extents[1].start_block, 200);
        assert_eq!(fd.extents[1].block_count, 2);
        assert_eq!(fd.inline_blocks(), 3);
    }
}
