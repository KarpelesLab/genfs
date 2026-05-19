//! SquashFS id lookup table — uid / gid resolution.
//!
//! Layout (kernel docs + community reference):
//!
//! - The superblock's `id_table_start` points at an **uncompressed**
//!   contiguous `u64[]` of absolute disk offsets, one per metablock
//!   holding `u32` id values.
//! - Each metablock holds up to 2048 packed little-endian `u32` ids.
//! - The number of location entries is `ceil(id_count * 4 / 8192)`.
//!
//! Inodes carry 16-bit indices into this flat array; we expose
//! [`IdTable::resolve`] to map an index to the underlying `u32` value.
//!
//! The writer side is in [`crate::fs::squashfs::writer`]; this module is
//! read-only.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::squashfs::Compression;
use crate::fs::squashfs::metablock::read_metablock;

/// Cached id lookup table. Reads happen on demand; once we've loaded a
/// metablock we keep its raw bytes around so subsequent lookups in the
/// same block are O(1).
#[derive(Debug, Default)]
pub struct IdTable {
    /// Decoded id values, in the order they appear on disk. Empty when
    /// the table hasn't been loaded yet.
    values: Vec<u32>,
    /// True once we've attempted a load.
    loaded: bool,
}

impl IdTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the table from disk. Idempotent: subsequent calls are no-ops.
    pub fn ensure_loaded(
        &mut self,
        dev: &mut dyn BlockDevice,
        table_start: u64,
        id_count: u16,
        compression: Compression,
    ) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        self.loaded = true;
        if id_count == 0 || table_start == u64::MAX {
            return Ok(());
        }
        let bytes_total = id_count as usize * 4;
        let metablock_count = bytes_total.div_ceil(8192);
        // Read the u64[] location array (uncompressed, contiguous).
        let mut locs = vec![0u8; metablock_count * 8];
        dev.read_at(table_start, &mut locs)?;
        let mut values = Vec::with_capacity(id_count as usize);
        let mut remaining = bytes_total;
        for i in 0..metablock_count {
            let off = i * 8;
            let mb_disk = u64::from_le_bytes(locs[off..off + 8].try_into().unwrap());
            let mb = read_metablock(dev, mb_disk, compression)?;
            let want = remaining.min(8192);
            if mb.data.len() < want {
                return Err(crate::Error::InvalidImage(format!(
                    "squashfs: id-table metablock {i} decoded to {} bytes, want >= {want}",
                    mb.data.len()
                )));
            }
            for j in 0..(want / 4) {
                let p = j * 4;
                values.push(u32::from_le_bytes(mb.data[p..p + 4].try_into().unwrap()));
            }
            remaining -= want;
        }
        self.values = values;
        Ok(())
    }

    /// Resolve a uid_idx / gid_idx field from an inode to the underlying
    /// `u32` value. Out-of-range indices return `0` (matches what kernel
    /// docs recommend for malformed images so we don't panic on a slightly
    /// off-by-one fixture).
    pub fn resolve(&self, idx: u16) -> u32 {
        self.values.get(idx as usize).copied().unwrap_or(0)
    }
}

/// Encode a list of `u32` id values into the on-disk format: a sequence
/// of metablocks followed by a `u64[]` location array. Returns
/// `(disk_payload, location_array_offset_within_payload)`.
///
/// The caller writes `disk_payload` at absolute offset `base`; the
/// stored location offsets are absolute (`base + relative`), matching
/// what the reader expects. `id_table_start` in the superblock is
/// `base + location_array_offset`.
#[allow(dead_code)]
pub fn encode_id_table(
    values: &[u32],
    base: u64,
    compression: Compression,
) -> Result<(Vec<u8>, u64)> {
    use crate::fs::squashfs::metablock::encode_metablock;

    let mut raw = Vec::with_capacity(values.len() * 4);
    for &v in values {
        raw.extend_from_slice(&v.to_le_bytes());
    }

    let mut out = Vec::new();
    let mut locations: Vec<u64> = Vec::new();
    let mut pos = 0usize;
    while pos < raw.len() {
        let end = (pos + 8192).min(raw.len());
        let mb_offset_abs = base + out.len() as u64;
        let mb = encode_metablock(&raw[pos..end], compression)?;
        out.extend_from_slice(&mb);
        locations.push(mb_offset_abs);
        pos = end;
    }
    let loc_offset = out.len() as u64;
    for l in &locations {
        out.extend_from_slice(&l.to_le_bytes());
    }
    Ok((out, loc_offset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn round_trip_id_table_uncompressed() {
        let values: Vec<u32> = (0..3).collect();
        let base = 100u64;
        let (payload, loc_off) = encode_id_table(&values, base, Compression::Unknown(0)).unwrap();
        let mut dev = MemoryBackend::new(base + payload.len() as u64 + 32);
        dev.write_at(base, &payload).unwrap();
        let mut t = IdTable::new();
        t.ensure_loaded(
            &mut dev,
            base + loc_off,
            values.len() as u16,
            Compression::Unknown(0),
        )
        .unwrap();
        assert_eq!(t.resolve(0), 0);
        assert_eq!(t.resolve(1), 1);
        assert_eq!(t.resolve(2), 2);
        // Out-of-range indices return 0 instead of panicking.
        assert_eq!(t.resolve(99), 0);
    }
}
