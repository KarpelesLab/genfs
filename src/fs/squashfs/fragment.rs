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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;
    use crate::fs::squashfs::metablock::encode_uncompressed;

    /// Helper: write `entries` as a single metablock at `mb_disk`, then write
    /// the one-element L1 location array at `l1_off`. Returns the populated
    /// memory backend.
    fn build_image(entries: &[FragmentEntry], mb_disk: u64, l1_off: u64) -> MemoryBackend {
        let mut payload = Vec::with_capacity(entries.len() * 16);
        for e in entries {
            payload.extend_from_slice(&e.start.to_le_bytes());
            payload.extend_from_slice(&e.size_word.to_le_bytes());
            // 4 reserved trailing bytes per entry (ignored by reader).
            payload.extend_from_slice(&[0u8; 4]);
        }
        let mb = encode_uncompressed(&payload);
        let total = (l1_off + 8).max(mb_disk + mb.len() as u64) + 16;
        let mut dev = MemoryBackend::new(total);
        dev.write_at(mb_disk, &mb).unwrap();
        dev.write_at(l1_off, &mb_disk.to_le_bytes()).unwrap();
        dev
    }

    #[test]
    fn size_word_flags() {
        // Bit 24 set => uncompressed; lower 24 bits => on-disk size.
        let e = FragmentEntry {
            start: 0,
            size_word: 0x0100_1234,
        };
        assert!(e.is_uncompressed());
        assert_eq!(e.on_disk_size(), 0x0000_1234);
        let e = FragmentEntry {
            start: 0,
            size_word: 0x0000_1234,
        };
        assert!(!e.is_uncompressed());
        assert_eq!(e.on_disk_size(), 0x0000_1234);
        // Upper byte above bit 24 must not leak into on_disk_size().
        let e = FragmentEntry {
            start: 0,
            size_word: 0xFF00_0000,
        };
        assert_eq!(e.on_disk_size(), 0);
    }

    #[test]
    fn reads_single_entry() {
        let entries = [FragmentEntry {
            start: 0xDEAD_BEEF,
            size_word: 0x0100_2000, // uncompressed, 8 KiB
        }];
        let mb_disk = 256u64;
        let l1_off = 4096u64;
        let mut dev = build_image(&entries, mb_disk, l1_off);
        let got = read_fragment(&mut dev, l1_off, 1, Compression::Gzip, 0).unwrap();
        assert_eq!(got.start, 0xDEAD_BEEF);
        assert_eq!(got.size_word, 0x0100_2000);
        assert!(got.is_uncompressed());
        assert_eq!(got.on_disk_size(), 0x2000);
    }

    #[test]
    fn out_of_range_index_errors() {
        let entries = [FragmentEntry {
            start: 0,
            size_word: 0,
        }];
        let mut dev = build_image(&entries, 256, 4096);
        let err = read_fragment(&mut dev, 4096, 1, Compression::Gzip, 1).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("fragment index"), "got: {msg}");
    }

    #[test]
    fn second_entry_in_block_is_at_offset_16() {
        // Two entries: confirm the reader picks up the second one at byte
        // offset 16 inside the metablock (and not, say, 12 or 20).
        let entries = [
            FragmentEntry {
                start: 0x1111,
                size_word: 0x0000_0100,
            },
            FragmentEntry {
                start: 0x2222,
                size_word: 0x0100_0200,
            },
        ];
        let mut dev = build_image(&entries, 128, 1024);
        let e0 = read_fragment(&mut dev, 1024, 2, Compression::Gzip, 0).unwrap();
        let e1 = read_fragment(&mut dev, 1024, 2, Compression::Gzip, 1).unwrap();
        assert_eq!(e0.start, 0x1111);
        assert_eq!(e0.size_word, 0x0000_0100);
        assert_eq!(e1.start, 0x2222);
        assert_eq!(e1.size_word, 0x0100_0200);
    }

    #[test]
    fn second_metablock_lookup_uses_second_l1_pointer() {
        // Build two metablocks: first holds 512 dummy entries (8 KiB),
        // second holds one real entry. Index 512 must dereference the
        // *second* L1 pointer and read at offset 0 inside that block.
        const ENTRIES_PER_BLOCK: usize = 512;
        let mut payload0 = Vec::with_capacity(ENTRIES_PER_BLOCK * 16);
        for _ in 0..ENTRIES_PER_BLOCK {
            payload0.extend_from_slice(&[0u8; 16]);
        }
        let mb0 = encode_uncompressed(&payload0);

        let target = FragmentEntry {
            start: 0xCAFE_F00D,
            size_word: 0x0100_4000,
        };
        let mut payload1 = Vec::with_capacity(16);
        payload1.extend_from_slice(&target.start.to_le_bytes());
        payload1.extend_from_slice(&target.size_word.to_le_bytes());
        payload1.extend_from_slice(&[0u8; 4]);
        let mb1 = encode_uncompressed(&payload1);

        let mb0_disk = 256u64;
        let mb1_disk = mb0_disk + mb0.len() as u64;
        let l1_off = mb1_disk + mb1.len() as u64 + 64;

        let mut dev = MemoryBackend::new(l1_off + 32);
        dev.write_at(mb0_disk, &mb0).unwrap();
        dev.write_at(mb1_disk, &mb1).unwrap();
        dev.write_at(l1_off, &mb0_disk.to_le_bytes()).unwrap();
        dev.write_at(l1_off + 8, &mb1_disk.to_le_bytes()).unwrap();

        let fragment_count = (ENTRIES_PER_BLOCK as u32) + 1;
        let got = read_fragment(&mut dev, l1_off, fragment_count, Compression::Gzip, 512).unwrap();
        assert_eq!(got.start, 0xCAFE_F00D);
        assert_eq!(got.size_word, 0x0100_4000);
        assert!(got.is_uncompressed());
        assert_eq!(got.on_disk_size(), 0x4000);
    }
}
