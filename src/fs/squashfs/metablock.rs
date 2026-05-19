//! SquashFS metablock reader.
//!
//! Metablocks are the unit of metadata storage in SquashFS: each is at most
//! 8 KiB uncompressed and is prefixed on disk by a 2-byte little-endian
//! header. The header's lower 15 bits hold the on-disk length, and the high
//! bit (`0x8000`) is set when the payload is stored uncompressed.
//!
//! This implementation only handles uncompressed metablocks. When a
//! compressed block is encountered we return [`crate::Error::Unsupported`]
//! naming the algorithm, so callers can surface a clear message.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::squashfs::Compression;

/// Maximum uncompressed size of a metablock.
pub const METABLOCK_SIZE: usize = 8192;

/// High bit of the metablock header signalling "stored uncompressed".
pub const METABLOCK_UNCOMPRESSED: u16 = 0x8000;

/// A single decoded metablock along with the on-disk size of the block
/// (header + payload) so callers can compute the next block's location.
#[derive(Debug)]
pub struct Metablock {
    /// Uncompressed payload bytes.
    pub data: Vec<u8>,
    /// Total bytes consumed on disk, including the 2-byte header.
    pub on_disk_size: u32,
}

/// Read a single metablock at `disk_offset` on `dev`. The block-list bit
/// encoding is *not* the same as a metablock header — see `data_block_size`.
pub fn read_metablock(
    dev: &mut dyn BlockDevice,
    disk_offset: u64,
    compression: Compression,
) -> Result<Metablock> {
    let mut header = [0u8; 2];
    dev.read_at(disk_offset, &mut header)?;
    let raw = u16::from_le_bytes(header);
    let uncompressed = raw & METABLOCK_UNCOMPRESSED != 0;
    let on_disk_len = (raw & 0x7FFF) as usize;
    if on_disk_len == 0 || on_disk_len > METABLOCK_SIZE {
        return Err(crate::Error::InvalidImage(format!(
            "squashfs: metablock at {disk_offset} has invalid on-disk size {on_disk_len}"
        )));
    }
    let mut payload = vec![0u8; on_disk_len];
    dev.read_at(disk_offset + 2, &mut payload)?;
    let data = if uncompressed {
        payload
    } else {
        let algo = compression_to_algo(compression).ok_or_else(|| {
            crate::Error::InvalidImage(format!(
                "squashfs: unknown compressor id {}",
                compression_label(compression)
            ))
        })?;
        crate::compression::decompress(algo, &payload, METABLOCK_SIZE)?
    };
    Ok(Metablock {
        data,
        on_disk_size: (on_disk_len + 2) as u32,
    })
}

pub(crate) fn compression_to_algo(c: Compression) -> Option<crate::compression::Algo> {
    use crate::compression::Algo;
    Some(match c {
        Compression::Gzip => Algo::Gzip,
        Compression::Lzma => Algo::Lzma,
        Compression::Lzo => Algo::Lzo,
        Compression::Xz => Algo::Xz,
        Compression::Lz4 => Algo::Lz4,
        Compression::Zstd => Algo::Zstd,
        Compression::Unknown(_) => return None,
    })
}

/// A logical cursor into a stream of metablocks, anchored at a table-start
/// disk offset. Used to read inodes (anchored at `inode_table_start`) and
/// directory contents (anchored at `directory_table_start`). The reader
/// caches one decoded metablock at a time so successive reads at nearby
/// offsets within the same block don't re-fetch from disk.
pub struct MetadataReader {
    /// Disk offset of the first metablock in this table.
    base: u64,
    /// Compression algorithm advertised by the superblock — only used to
    /// produce a precise error string when a compressed block is hit.
    compression: Compression,
    /// Cached metablock and the byte offset of its header relative to `base`.
    cached: Option<CachedBlock>,
}

struct CachedBlock {
    /// Offset (relative to `base`) of the metablock header on disk.
    disk_rel: u64,
    /// Decoded payload.
    data: Vec<u8>,
    /// On-disk size of header + payload, so we know where the next block
    /// header begins.
    on_disk_size: u32,
}

impl MetadataReader {
    pub fn new(base: u64, compression: Compression) -> Self {
        Self {
            base,
            compression,
            cached: None,
        }
    }

    /// Ensure the metablock starting at `disk_rel` (a relative offset into
    /// the metablock stream, *not* an uncompressed offset) is cached.
    fn ensure(&mut self, dev: &mut dyn BlockDevice, disk_rel: u64) -> Result<()> {
        if let Some(c) = &self.cached
            && c.disk_rel == disk_rel
        {
            return Ok(());
        }
        let mb = read_metablock(dev, self.base + disk_rel, self.compression)?;
        self.cached = Some(CachedBlock {
            disk_rel,
            data: mb.data,
            on_disk_size: mb.on_disk_size,
        });
        Ok(())
    }

    /// Read `n` bytes starting at an uncompressed `(block_disk_rel, offset)`
    /// reference. Returns the bytes plus the new `(block_disk_rel, offset)`
    /// cursor pointing just past the last byte read, with the metablock
    /// boundary already crossed if `offset` would have ended up past 8 KiB.
    pub fn read(
        &mut self,
        dev: &mut dyn BlockDevice,
        mut block_disk_rel: u64,
        mut offset: usize,
        n: usize,
    ) -> Result<(Vec<u8>, u64, usize)> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            self.ensure(dev, block_disk_rel)?;
            let cached = self.cached.as_ref().expect("cached after ensure");
            if offset > cached.data.len() {
                return Err(crate::Error::InvalidImage(format!(
                    "squashfs: metablock offset {offset} past uncompressed length {}",
                    cached.data.len()
                )));
            }
            let want = n - out.len();
            let avail = cached.data.len() - offset;
            let take = want.min(avail);
            out.extend_from_slice(&cached.data[offset..offset + take]);
            offset += take;
            if out.len() < n {
                // Hopped past end of this metablock — advance to next.
                block_disk_rel = block_disk_rel
                    .checked_add(cached.on_disk_size as u64)
                    .ok_or_else(|| {
                        crate::Error::InvalidImage(
                            "squashfs: metablock chain overflowed u64".into(),
                        )
                    })?;
                offset = 0;
            }
        }
        // Normalise: if we ended exactly at the boundary, advance to the
        // next block so callers see (next, 0) — this matches how SquashFS
        // inode references encode positions.
        if let Some(cached) = &self.cached
            && offset == cached.data.len()
        {
            block_disk_rel = block_disk_rel
                .checked_add(cached.on_disk_size as u64)
                .ok_or_else(|| {
                    crate::Error::InvalidImage("squashfs: metablock chain overflowed u64".into())
                })?;
            offset = 0;
        }
        Ok((out, block_disk_rel, offset))
    }
}

fn compression_label(c: Compression) -> &'static str {
    match c {
        Compression::Gzip => "gzip",
        Compression::Lzma => "lzma",
        Compression::Lzo => "lzo",
        Compression::Xz => "xz",
        Compression::Lz4 => "lz4",
        Compression::Zstd => "zstd",
        Compression::Unknown(_) => "unknown",
    }
}

/// Build a metablock prefix (2-byte header + payload) for the given payload,
/// marking it as uncompressed. Test-only helper, exposed so other modules in
/// this crate can hand-craft tiny fixtures.
#[cfg(test)]
pub fn encode_uncompressed(payload: &[u8]) -> Vec<u8> {
    assert!(payload.len() <= METABLOCK_SIZE);
    let mut out = Vec::with_capacity(payload.len() + 2);
    let header = (payload.len() as u16) | METABLOCK_UNCOMPRESSED;
    out.extend_from_slice(&header.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn reads_uncompressed_metablock() {
        let payload: Vec<u8> = (0u8..10).collect();
        let encoded = encode_uncompressed(&payload);
        let mut dev = MemoryBackend::new(64);
        dev.write_at(0, &encoded).unwrap();
        let mb = read_metablock(&mut dev, 0, Compression::Gzip).unwrap();
        assert_eq!(mb.data, payload);
        assert_eq!(mb.on_disk_size as usize, encoded.len());
    }

    #[test]
    fn compressed_metablock_returns_unsupported() {
        let mut dev = MemoryBackend::new(64);
        // 4-byte "compressed" payload, no uncompressed bit set.
        dev.write_at(0, &4u16.to_le_bytes()).unwrap();
        dev.write_at(2, &[0xAA, 0xBB, 0xCC, 0xDD]).unwrap();
        let err = read_metablock(&mut dev, 0, Compression::Zstd).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("zstd"), "expected zstd in message: {msg}");
    }

    #[test]
    fn metadata_reader_crosses_block_boundary() {
        // Build two consecutive uncompressed metablocks: first has [0..5),
        // second has [5..10).
        let mut image = Vec::new();
        image.extend_from_slice(&encode_uncompressed(&[0, 1, 2, 3, 4]));
        image.extend_from_slice(&encode_uncompressed(&[5, 6, 7, 8, 9]));
        let mut dev = MemoryBackend::new(image.len() as u64 + 16);
        dev.write_at(0, &image).unwrap();
        let mut mr = MetadataReader::new(0, Compression::Gzip);
        let (bytes, next_block, next_off) = mr.read(&mut dev, 0, 3, 5).unwrap();
        assert_eq!(bytes, vec![3, 4, 5, 6, 7]);
        // Should now sit inside the second block at offset 3.
        assert_eq!(next_off, 3);
        assert_eq!(next_block, 7); // first block was 2+5 = 7 bytes on disk
    }

    #[test]
    fn metadata_reader_advances_at_exact_boundary() {
        let mut image = Vec::new();
        image.extend_from_slice(&encode_uncompressed(&[0, 1, 2, 3, 4]));
        image.extend_from_slice(&encode_uncompressed(&[5, 6, 7, 8, 9]));
        let mut dev = MemoryBackend::new(image.len() as u64 + 16);
        dev.write_at(0, &image).unwrap();
        let mut mr = MetadataReader::new(0, Compression::Gzip);
        let (bytes, next_block, next_off) = mr.read(&mut dev, 0, 0, 5).unwrap();
        assert_eq!(bytes, vec![0, 1, 2, 3, 4]);
        // After consuming all of block 0, we should jump to block 1, offset 0.
        assert_eq!(next_off, 0);
        assert_eq!(next_block, 7);
    }
}
