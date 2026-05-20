//! Mish block (`BLKX`) parsing — the per-partition chunk table of a UDIF
//! disk image.
//!
//! ## On-disk layout (big-endian)
//!
//! ```text
//! offset  size  field
//! 0x000   4     signature ("mish" / 0x6D697368)
//! 0x004   4     version  (1)
//! 0x008   8     sector_number   ← first sector this BLKX covers on the
//!                                  virtual disk (relative to the BLKX,
//!                                  not absolute on the image)
//! 0x010   8     sector_count
//! 0x018   8     data_offset     ← byte offset of this BLKX's compressed
//!                                  payload inside the data fork
//! 0x020   4     buffers_needed
//! 0x024   4     block_descriptor
//! 0x028   24    reserved[6]
//! 0x040   136   checksum  (4 type + 4 size + 128 data)
//! 0x0C8   4     number_of_blocks  (n)
//! 0x0CC   n×40  chunks[]
//! ```
//!
//! Each chunk is 40 bytes:
//!
//! ```text
//! offset  size  field
//! 0x00    4     entry_type
//! 0x04    4     comment
//! 0x08    8     sector_number       ← virtual-disk sector, **relative
//!                                      to the BLKX's sector_number**
//! 0x10    8     sector_count
//! 0x18    8     compressed_offset   ← byte offset into the data fork
//!                                      (absolute; we add koly's
//!                                      data_fork_offset on read)
//! 0x20    8     compressed_length
//! ```
//!
//! Entry types we handle:
//!
//! - `0x00000000` — zero fill
//! - `0x00000001` — raw / uncompressed
//! - `0x00000002` — "ignored" / free-space sparse (read as zero)
//! - `0x80000005` — zlib (`Deflate` with a zlib wrapper)
//! - `0x7FFFFFFE` — comment (skip)
//! - `0xFFFFFFFF` — terminator (end of array)
//!
//! ADC (`0x80000004`), bzip2 (`0x80000006`), LZFSE (`0x80000007`) and
//! LZMA (`0x80000008`) are decoded by [`crate::block::dmg::codec`].

use crate::Result;

/// Mish block magic: `"mish"`.
pub const MISH_MAGIC: u32 = 0x6D69_7368;

/// Length of the fixed mish header before the variable-length chunk array.
pub const MISH_HEADER_LEN: usize = 0x0CC + 4;

/// Length of a single chunk descriptor.
pub const CHUNK_LEN: usize = 40;

/// Recognised chunk-entry types. The numeric values are part of the
/// UDIF format and must not be renumbered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    /// 0x00000000 — fill with zeros, no source bytes consumed.
    Zero,
    /// 0x00000001 — raw bytes, length = sector_count * 512.
    Raw,
    /// 0x00000002 — free space / "ignored". Read as zeros; same as
    /// `Zero` from the reader's perspective.
    Ignored,
    /// 0x80000005 — zlib (RFC 1950) compressed.
    Zlib,
    /// 0x80000004 — Apple Data Compression.
    Adc,
    /// 0x80000006 — bzip2.
    Bz2,
    /// 0x80000007 — LZFSE.
    Lzfse,
    /// 0x80000008 — LZMA.
    Lzma,
    /// 0x7FFFFFFE — comment. Skip; not actual data.
    Comment,
    /// 0xFFFFFFFF — array terminator.
    Terminator,
}

impl ChunkType {
    /// Map an on-disk u32 to a [`ChunkType`]. Unknown codes return an
    /// error rather than silently being mistaken for zero-fill.
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0x0000_0000 => Self::Zero,
            0x0000_0001 => Self::Raw,
            0x0000_0002 => Self::Ignored,
            0x8000_0004 => Self::Adc,
            0x8000_0005 => Self::Zlib,
            0x8000_0006 => Self::Bz2,
            0x8000_0007 => Self::Lzfse,
            0x8000_0008 => Self::Lzma,
            0x7FFF_FFFE => Self::Comment,
            0xFFFF_FFFF => Self::Terminator,
            _ => {
                return Err(crate::Error::InvalidImage(format!(
                    "dmg: unknown chunk entry_type {v:#010x}"
                )));
            }
        })
    }
}

/// One chunk inside a mish block. All sector / byte counts come straight
/// from the on-disk fields; the offsets below have already been resolved
/// into the absolute coordinate system of a single-segment image:
///
/// - `virtual_sector_start` is the **absolute** sector on the virtual
///   disk where this chunk begins (mish.sector_number + chunk.sector_number).
/// - `compressed_offset_in_fork` is the **absolute** byte offset of the
///   compressed payload inside the data fork.
#[derive(Debug, Clone, Copy)]
pub struct Chunk {
    pub kind: ChunkType,
    pub virtual_sector_start: u64,
    pub sector_count: u64,
    pub compressed_offset_in_fork: u64,
    pub compressed_length: u64,
}

/// Decoded mish block — header fields the chunk router cares about,
/// plus the resolved chunks. Comment / Terminator entries are filtered
/// out by `decode_mish`; they exist on disk but have no bytes to map.
#[derive(Debug, Clone)]
pub struct Mish {
    pub first_sector: u64,
    pub sector_count: u64,
    /// Absolute byte offset into the data fork where this BLKX's
    /// payload starts. Kept for diagnostics; chunk lookups use
    /// [`Chunk::compressed_offset_in_fork`] directly.
    pub data_offset: u64,
    pub chunks: Vec<Chunk>,
}

/// Parse a serialised mish block.
///
/// Returns `Err(InvalidImage)` on bad magic, truncated input or an
/// unknown chunk type. The chunk array is read until the on-disk
/// `number_of_blocks` count is exhausted or a `Terminator` entry is
/// encountered, whichever comes first — both forms appear in real
/// images.
pub fn decode_mish(buf: &[u8]) -> Result<Mish> {
    if buf.len() < MISH_HEADER_LEN {
        return Err(crate::Error::InvalidImage(format!(
            "dmg: mish block shorter than {MISH_HEADER_LEN} bytes (got {})",
            buf.len()
        )));
    }
    let sig = u32::from_be_bytes(buf[0x000..0x004].try_into().unwrap());
    if sig != MISH_MAGIC {
        return Err(crate::Error::InvalidImage(format!(
            "dmg: mish magic mismatch (got {sig:#010x})"
        )));
    }
    let _version = u32::from_be_bytes(buf[0x004..0x008].try_into().unwrap());
    let first_sector = u64::from_be_bytes(buf[0x008..0x010].try_into().unwrap());
    let sector_count = u64::from_be_bytes(buf[0x010..0x018].try_into().unwrap());
    let data_offset = u64::from_be_bytes(buf[0x018..0x020].try_into().unwrap());
    let number_of_blocks = u32::from_be_bytes(buf[0x0C8..0x0CC].try_into().unwrap()) as usize;

    let chunk_array_start = MISH_HEADER_LEN;
    let need =
        chunk_array_start
            .checked_add(number_of_blocks.checked_mul(CHUNK_LEN).ok_or_else(|| {
                crate::Error::InvalidImage("dmg: mish chunk count overflow".into())
            })?)
            .ok_or_else(|| crate::Error::InvalidImage("dmg: mish chunk extent overflow".into()))?;
    if buf.len() < need {
        return Err(crate::Error::InvalidImage(format!(
            "dmg: mish block truncated (need {need} bytes, got {})",
            buf.len()
        )));
    }

    let mut chunks = Vec::with_capacity(number_of_blocks);
    for i in 0..number_of_blocks {
        let off = chunk_array_start + i * CHUNK_LEN;
        let entry_type = u32::from_be_bytes(buf[off..off + 4].try_into().unwrap());
        let kind = ChunkType::from_u32(entry_type)?;
        let sector_number_rel = u64::from_be_bytes(buf[off + 0x08..off + 0x10].try_into().unwrap());
        let chunk_sector_count =
            u64::from_be_bytes(buf[off + 0x10..off + 0x18].try_into().unwrap());
        let comp_off = u64::from_be_bytes(buf[off + 0x18..off + 0x20].try_into().unwrap());
        let comp_len = u64::from_be_bytes(buf[off + 0x20..off + 0x28].try_into().unwrap());

        match kind {
            ChunkType::Terminator => break,
            ChunkType::Comment => continue,
            _ => {}
        }

        chunks.push(Chunk {
            kind,
            virtual_sector_start: first_sector.saturating_add(sector_number_rel),
            sector_count: chunk_sector_count,
            compressed_offset_in_fork: comp_off,
            compressed_length: comp_len,
        });
    }

    Ok(Mish {
        first_sector,
        sector_count,
        data_offset,
        chunks,
    })
}

/// Encode a mish block from a header summary + an explicit chunk list.
/// Used only by tests; the live reader never writes mish blocks.
///
/// Test helper rather than public surface — kept behind `cfg(test)` so
/// the writer-side bytes aren't part of the released API.
#[cfg(test)]
pub fn encode_mish_for_tests(
    first_sector: u64,
    sector_count: u64,
    data_offset: u64,
    chunks: &[Chunk],
) -> Vec<u8> {
    let mut buf = vec![0u8; MISH_HEADER_LEN + chunks.len() * CHUNK_LEN];
    buf[0x000..0x004].copy_from_slice(&MISH_MAGIC.to_be_bytes());
    buf[0x004..0x008].copy_from_slice(&1u32.to_be_bytes()); // version
    buf[0x008..0x010].copy_from_slice(&first_sector.to_be_bytes());
    buf[0x010..0x018].copy_from_slice(&sector_count.to_be_bytes());
    buf[0x018..0x020].copy_from_slice(&data_offset.to_be_bytes());
    // buffers_needed / block_descriptor / reserved / checksum stay zero.
    buf[0x0C8..0x0CC].copy_from_slice(&(chunks.len() as u32).to_be_bytes());
    for (i, c) in chunks.iter().enumerate() {
        let off = MISH_HEADER_LEN + i * CHUNK_LEN;
        let etype: u32 = match c.kind {
            ChunkType::Zero => 0x0000_0000,
            ChunkType::Raw => 0x0000_0001,
            ChunkType::Ignored => 0x0000_0002,
            ChunkType::Adc => 0x8000_0004,
            ChunkType::Zlib => 0x8000_0005,
            ChunkType::Bz2 => 0x8000_0006,
            ChunkType::Lzfse => 0x8000_0007,
            ChunkType::Lzma => 0x8000_0008,
            ChunkType::Comment => 0x7FFF_FFFE,
            ChunkType::Terminator => 0xFFFF_FFFF,
        };
        buf[off..off + 4].copy_from_slice(&etype.to_be_bytes());
        // comment field stays zero
        // sector_number is encoded **relative** to first_sector — the
        // reader adds first_sector back in.
        let rel_sec = c.virtual_sector_start.saturating_sub(first_sector);
        buf[off + 0x08..off + 0x10].copy_from_slice(&rel_sec.to_be_bytes());
        buf[off + 0x10..off + 0x18].copy_from_slice(&c.sector_count.to_be_bytes());
        buf[off + 0x18..off + 0x20].copy_from_slice(&c.compressed_offset_in_fork.to_be_bytes());
        buf[off + 0x20..off + 0x28].copy_from_slice(&c.compressed_length.to_be_bytes());
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_zero_raw_zlib() {
        let chunks = vec![
            Chunk {
                kind: ChunkType::Zero,
                virtual_sector_start: 0,
                sector_count: 2,
                compressed_offset_in_fork: 0,
                compressed_length: 0,
            },
            Chunk {
                kind: ChunkType::Raw,
                virtual_sector_start: 2,
                sector_count: 1,
                compressed_offset_in_fork: 0,
                compressed_length: 512,
            },
            Chunk {
                kind: ChunkType::Zlib,
                virtual_sector_start: 3,
                sector_count: 1,
                compressed_offset_in_fork: 512,
                compressed_length: 17,
            },
        ];
        let buf = encode_mish_for_tests(0, 4, 0, &chunks);
        let m = decode_mish(&buf).unwrap();
        assert_eq!(m.first_sector, 0);
        assert_eq!(m.sector_count, 4);
        assert_eq!(m.chunks.len(), 3);
        assert_eq!(m.chunks[0].kind, ChunkType::Zero);
        assert_eq!(m.chunks[1].kind, ChunkType::Raw);
        assert_eq!(m.chunks[2].kind, ChunkType::Zlib);
        assert_eq!(m.chunks[1].virtual_sector_start, 2);
        assert_eq!(m.chunks[2].compressed_offset_in_fork, 512);
    }

    #[test]
    fn stops_at_terminator() {
        // Two real chunks + one terminator. The decoder should keep
        // only the two real ones.
        let chunks = vec![
            Chunk {
                kind: ChunkType::Zero,
                virtual_sector_start: 0,
                sector_count: 1,
                compressed_offset_in_fork: 0,
                compressed_length: 0,
            },
            Chunk {
                kind: ChunkType::Raw,
                virtual_sector_start: 1,
                sector_count: 1,
                compressed_offset_in_fork: 0,
                compressed_length: 512,
            },
            Chunk {
                kind: ChunkType::Terminator,
                virtual_sector_start: 2,
                sector_count: 0,
                compressed_offset_in_fork: 0,
                compressed_length: 0,
            },
        ];
        let buf = encode_mish_for_tests(0, 2, 0, &chunks);
        let m = decode_mish(&buf).unwrap();
        assert_eq!(m.chunks.len(), 2);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut buf = vec![0u8; MISH_HEADER_LEN];
        buf[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        assert!(decode_mish(&buf).is_err());
    }
}
