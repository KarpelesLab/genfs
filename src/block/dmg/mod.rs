//! Apple Disk Image (DMG) — read-only chunk-decoding backend.
//!
//! ## Status
//!
//! - Detection + 512-byte koly trailer parse.
//! - XML resource-fork plist + per-partition `mish` block decode.
//! - Per-chunk reader for every compressed type real `.dmg` images
//!   use in the wild: zero-fill, raw / uncompressed, zlib, bzip2,
//!   LZFSE, LZMA, and ADC. The four compressed codecs are split out
//!   into [`codec`] so each one's wiring (and feature gate, when it
//!   has one) sits in a single place.
//!
//! [`DmgBackend`] satisfies the [`BlockDevice`] trait, so dropping it
//! into the existing [`crate::inspect::detect_fs`] / [`crate::inspect::AnyFs`]
//! pipeline transparently exposes whatever filesystem lives inside the
//! image (HFS+, APFS, ISO 9660, …). Writers are out of scope for v1.
//!
//! ## File layout (recap)
//!
//! ```text
//!   [ data-fork bytes (compressed chunks) ]
//!   [ XML / binary-plist resource fork    ]   ← table of `mish` blocks
//!   [ 512-byte koly trailer at file end   ]   ← magic, offsets, sector count
//! ```
//!
//! All multi-byte fields in the koly trailer are **big-endian**. We
//! anchor detection on the `koly` magic at `file_size - 512`. Older
//! NDIF (`disk`-magic) images are explicitly out of scope.
//!
//! ## References
//!
//! Apple has never published a formal UDIF spec. The layout in this
//! module follows two public reverse-engineering write-ups:
//!
//! - Jonathan Levin, *DMG file structure* (newosxbook.com).
//! - The `libdmg-hfsplus` project's `dmg.h`, particularly the
//!   `UDIFResourceFile` struct.
//!
//! All offsets and field meanings are public spec; no Apple
//! source / SDK was consulted.

mod base64;
mod codec;
mod mish;
mod plist;

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::Result;
use crate::block::BlockDevice;

pub use mish::{Chunk, ChunkType, Mish};

/// `koly` magic — first four bytes of the 512-byte UDIF trailer.
pub const KOLY_MAGIC: u32 = 0x6B6F_6C79; // "koly"

/// Total size of the trailer (one sector).
pub const KOLY_SIZE: u64 = 512;

/// Fields from the 512-byte UDIF trailer. Multi-byte values are
/// big-endian on disk; this struct holds them in host order after
/// `decode`. Only the fields the reader actually needs are surfaced;
/// the trailer's per-segment / checksum payloads are decoded into raw
/// byte arrays so a caller that wants them can still inspect them.
#[derive(Debug, Clone)]
pub struct KolyTrailer {
    /// `udif_signature` — must equal [`KOLY_MAGIC`].
    pub signature: u32,
    /// `udif_version` — 4 for current images.
    pub version: u32,
    /// `header_size` — typically 512 (the trailer is its own header).
    pub header_size: u32,
    /// `flags` — implementation-defined; carry through verbatim.
    pub flags: u32,
    /// `running_data_fork_offset` — start of the data fork inside this
    /// segment. Zero for single-segment images.
    pub running_data_fork_offset: u64,
    /// `data_fork_offset` — absolute byte offset of the compressed
    /// chunk data within the file.
    pub data_fork_offset: u64,
    /// `data_fork_length` — bytes of compressed data.
    pub data_fork_length: u64,
    /// `rsrc_fork_offset` — absolute byte offset of the legacy NDIF
    /// resource fork. Usually 0 on UDIF.
    pub rsrc_fork_offset: u64,
    /// `rsrc_fork_length`.
    pub rsrc_fork_length: u64,
    /// 1-based segment index of this file in a multi-segment image.
    pub segment_number: u32,
    /// Total segment count.
    pub segment_count: u32,
    /// Stable 16-byte segment identifier shared across all segments of
    /// a multi-segment image.
    pub segment_id: [u8; 16],
    /// CRC over the data fork: `data_checksum_type` (1 = CRC32) +
    /// `data_checksum_size` + `data_checksum[]`.
    pub data_checksum_type: u32,
    pub data_checksum_size: u32,
    pub data_checksum: [u8; 128],
    /// Byte offset + length of the XML / binary-plist resource that
    /// carries the per-chunk `mish` table.
    pub xml_offset: u64,
    pub xml_length: u64,
    /// Master image checksum: same shape as the data checksum.
    pub master_checksum_type: u32,
    pub master_checksum_size: u32,
    pub master_checksum: [u8; 128],
    /// `image_variant` — implementation-defined.
    pub image_variant: u32,
    /// `sector_count` — virtual size in 512-byte sectors. Total virtual
    /// bytes are `sector_count * 512`.
    pub sector_count: u64,
}

impl KolyTrailer {
    /// Decode the 512-byte trailer from a buffer whose first 512 bytes
    /// are the koly block. Returns `Err` if the magic doesn't match or
    /// the version isn't 4 (older NDIF / future versions out of scope).
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < KOLY_SIZE as usize {
            return Err(crate::Error::InvalidImage(
                "dmg: trailer slice shorter than 512 bytes".into(),
            ));
        }
        let signature = u32::from_be_bytes(buf[0x000..0x004].try_into().unwrap());
        if signature != KOLY_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "dmg: koly magic mismatch (got {signature:#010x})"
            )));
        }
        let version = u32::from_be_bytes(buf[0x004..0x008].try_into().unwrap());
        if version != 4 {
            return Err(crate::Error::Unsupported(format!(
                "dmg: koly version {version} not supported (only v4)"
            )));
        }
        let mut segment_id = [0u8; 16];
        segment_id.copy_from_slice(&buf[0x040..0x050]);
        let mut data_checksum = [0u8; 128];
        data_checksum.copy_from_slice(&buf[0x058..0x0D8]);
        let mut master_checksum = [0u8; 128];
        master_checksum.copy_from_slice(&buf[0x168..0x1E8]);
        Ok(Self {
            signature,
            version,
            header_size: u32::from_be_bytes(buf[0x008..0x00C].try_into().unwrap()),
            flags: u32::from_be_bytes(buf[0x00C..0x010].try_into().unwrap()),
            running_data_fork_offset: u64::from_be_bytes(buf[0x010..0x018].try_into().unwrap()),
            data_fork_offset: u64::from_be_bytes(buf[0x018..0x020].try_into().unwrap()),
            data_fork_length: u64::from_be_bytes(buf[0x020..0x028].try_into().unwrap()),
            rsrc_fork_offset: u64::from_be_bytes(buf[0x028..0x030].try_into().unwrap()),
            rsrc_fork_length: u64::from_be_bytes(buf[0x030..0x038].try_into().unwrap()),
            segment_number: u32::from_be_bytes(buf[0x038..0x03C].try_into().unwrap()),
            segment_count: u32::from_be_bytes(buf[0x03C..0x040].try_into().unwrap()),
            segment_id,
            data_checksum_type: u32::from_be_bytes(buf[0x050..0x054].try_into().unwrap()),
            data_checksum_size: u32::from_be_bytes(buf[0x054..0x058].try_into().unwrap()),
            data_checksum,
            xml_offset: u64::from_be_bytes(buf[0x0D8..0x0E0].try_into().unwrap()),
            xml_length: u64::from_be_bytes(buf[0x0E0..0x0E8].try_into().unwrap()),
            master_checksum_type: u32::from_be_bytes(buf[0x160..0x164].try_into().unwrap()),
            master_checksum_size: u32::from_be_bytes(buf[0x164..0x168].try_into().unwrap()),
            master_checksum,
            image_variant: u32::from_be_bytes(buf[0x1E8..0x1EC].try_into().unwrap()),
            sector_count: u64::from_be_bytes(buf[0x1EC..0x1F4].try_into().unwrap()),
        })
    }
}

/// Cheap detector: probe a file's last 512 bytes for the koly trailer.
/// Returns `Ok(true)` only when the magic matches; any I/O failure or
/// size-too-small condition returns `Ok(false)` so callers can fall
/// through to other backends.
pub fn probe(path: &Path) -> Result<bool> {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return Ok(false),
    };
    if meta.len() < KOLY_SIZE {
        return Ok(false);
    }
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };
    let trailer_offset = meta.len() - KOLY_SIZE;
    f.seek(SeekFrom::Start(trailer_offset))?;
    let mut head = [0u8; 4];
    if f.read_exact(&mut head).is_err() {
        return Ok(false);
    }
    Ok(u32::from_be_bytes(head) == KOLY_MAGIC)
}

/// Read-only DMG backend.
///
/// Holds the source file, the decoded koly trailer, and all per-partition
/// mish blocks recovered from the resource-fork plist. [`read_at`]
/// walks the virtual byte range, locates each chunk it intersects in a
/// flat list sorted by virtual-sector start, and emits zero / raw /
/// zlib-decompressed bytes. The other compressed chunk kinds are
/// recognised but error out as [`crate::Error::Unsupported`] — this
/// matches the README's "Limitations" entry.
///
/// [`read_at`]: BlockDevice::read_at
#[derive(Debug)]
pub struct DmgBackend {
    file: File,
    trailer: KolyTrailer,
    /// Cached virtual size in bytes (`sector_count * 512`).
    virtual_size: u64,
    /// Position of the implicit `Seek` cursor — kept so the `Seek`
    /// impl works the way callers expect from a `BlockDevice`.
    cursor: u64,
    /// All chunks from every mish block, flattened and sorted by
    /// `virtual_sector_start`. The sort is what makes a chunk lookup
    /// a binary search; on a 100 GB image with thousands of chunks
    /// the linear cost would otherwise dominate every read.
    chunks: Vec<Chunk>,
}

impl DmgBackend {
    /// Open a DMG file. Validates the koly trailer, the version field
    /// (must be 4), and loads the resource-fork plist + every mish
    /// block it references. Errors out for multi-segment images
    /// (segment_count > 1) and for plists that carry no `blkx` array.
    pub fn open(path: &Path) -> Result<Self> {
        let meta = std::fs::metadata(path)?;
        if meta.len() < KOLY_SIZE {
            return Err(crate::Error::InvalidImage(
                "dmg: file smaller than the 512-byte koly trailer".into(),
            ));
        }
        let mut file = File::open(path)?;
        let trailer_offset = meta.len() - KOLY_SIZE;
        file.seek(SeekFrom::Start(trailer_offset))?;
        let mut buf = vec![0u8; KOLY_SIZE as usize];
        file.read_exact(&mut buf)?;
        let trailer = KolyTrailer::decode(&buf)?;
        if trailer.segment_count > 1 {
            return Err(crate::Error::Unsupported(format!(
                "dmg: multi-segment images not supported (segment_count = {})",
                trailer.segment_count
            )));
        }
        let virtual_size = trailer
            .sector_count
            .checked_mul(512)
            .ok_or_else(|| crate::Error::InvalidImage("dmg: sector_count overflows u64".into()))?;

        // Pull the XML plist out of the resource-fork window.
        if trailer.xml_length == 0 {
            return Err(crate::Error::InvalidImage(
                "dmg: koly trailer has empty XML resource fork (xml_length = 0)".into(),
            ));
        }
        // The plist can be up to ~megabytes for huge images; that's
        // still fine to hold in RAM (it's a tiny fraction of the
        // image and we need random access to all of it to parse).
        // Cap it at 128 MiB to keep hostile inputs from OOMing.
        const MAX_PLIST_BYTES: u64 = 128 * 1024 * 1024;
        if trailer.xml_length > MAX_PLIST_BYTES {
            return Err(crate::Error::InvalidImage(format!(
                "dmg: resource-fork plist is implausibly large ({} bytes)",
                trailer.xml_length
            )));
        }
        file.seek(SeekFrom::Start(trailer.xml_offset))?;
        let mut plist_bytes = vec![0u8; trailer.xml_length as usize];
        file.read_exact(&mut plist_bytes)?;
        let plist_str = std::str::from_utf8(&plist_bytes).map_err(|e| {
            crate::Error::InvalidImage(format!("dmg: resource-fork plist isn't UTF-8: {e}"))
        })?;

        let data_entries = plist::extract_blkx_data_entries(plist_str)?;
        if data_entries.is_empty() {
            return Err(crate::Error::InvalidImage(
                "dmg: blkx array is empty — no chunks to map".into(),
            ));
        }

        let mut chunks: Vec<Chunk> = Vec::new();
        for entry in &data_entries {
            let raw = base64::decode(entry)?;
            let m = mish::decode_mish(&raw)?;
            chunks.extend(m.chunks);
        }
        chunks.sort_by_key(|c| c.virtual_sector_start);

        // Sanity: chunks shouldn't overlap. We don't fail loud on this
        // (some images emit zero-byte stub chunks at partition
        // boundaries that look like duplicates), but adjacent chunks
        // should be monotonic.
        for w in chunks.windows(2) {
            let prev_end = w[0].virtual_sector_start.saturating_add(w[0].sector_count);
            if w[1].virtual_sector_start < prev_end {
                log::warn!(
                    "dmg: chunk overlap detected: chunk ending at sector {} > next chunk start {}",
                    prev_end,
                    w[1].virtual_sector_start
                );
            }
        }

        Ok(Self {
            file,
            trailer,
            virtual_size,
            cursor: 0,
            chunks,
        })
    }

    /// Borrow the decoded trailer for diagnostics.
    pub fn trailer(&self) -> &KolyTrailer {
        &self.trailer
    }

    /// Number of chunks discovered across all mish blocks. Useful
    /// for diagnostics and tests.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Read the raw on-disk bytes of a single chunk's compressed
    /// payload. Splits out of [`read_chunk_into`] so the latter can
    /// hand the buffer to a codec without re-reading.
    fn read_compressed_payload(&mut self, chunk: &Chunk) -> Result<Vec<u8>> {
        let abs_offset = self
            .trailer
            .data_fork_offset
            .checked_add(chunk.compressed_offset_in_fork)
            .ok_or_else(|| {
                crate::Error::InvalidImage(
                    "dmg: chunk absolute offset overflows the data fork".into(),
                )
            })?;
        let len = chunk.compressed_length as usize;
        self.file.seek(SeekFrom::Start(abs_offset))?;
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Decode a single chunk into a full-sized byte buffer (the
    /// uncompressed length is `sector_count * 512`).
    ///
    /// We materialise the whole chunk because chunks are intentionally
    /// small (typically 2 MiB / 4096 sectors at most) — peeking into
    /// arbitrary byte ranges of a deflate stream isn't possible without
    /// running the inflate state machine to that point anyway.
    fn decode_chunk(&mut self, chunk: &Chunk) -> Result<Vec<u8>> {
        let plain_len = (chunk.sector_count as usize)
            .checked_mul(512)
            .ok_or_else(|| {
                crate::Error::InvalidImage("dmg: chunk plain length overflows usize".into())
            })?;

        match chunk.kind {
            ChunkType::Zero | ChunkType::Ignored => Ok(vec![0u8; plain_len]),
            ChunkType::Raw => {
                let buf = self.read_compressed_payload(chunk)?;
                if buf.len() != plain_len {
                    return Err(crate::Error::InvalidImage(format!(
                        "dmg: raw chunk has compressed_length {} but sector_count*512 = {}",
                        buf.len(),
                        plain_len
                    )));
                }
                Ok(buf)
            }
            ChunkType::Zlib => decode_zlib(&self.read_compressed_payload(chunk)?, plain_len),
            ChunkType::Adc => {
                let buf = self.read_compressed_payload(chunk)?;
                codec::decode_adc_chunk(&buf, plain_len)
            }
            ChunkType::Bz2 => {
                let buf = self.read_compressed_payload(chunk)?;
                codec::decode_bzip2(&buf, plain_len)
            }
            ChunkType::Lzfse => {
                let buf = self.read_compressed_payload(chunk)?;
                codec::decode_lzfse(&buf, plain_len)
            }
            ChunkType::Lzma => {
                let buf = self.read_compressed_payload(chunk)?;
                codec::decode_lzma(&buf, plain_len)
            }
            // Comment / Terminator are filtered during mish parsing.
            ChunkType::Comment | ChunkType::Terminator => Ok(vec![0u8; plain_len]),
        }
    }

    /// Locate the chunk whose virtual-sector span contains `sector`.
    /// Binary-searches the sorted list. Returns `Err(OutOfBounds)`
    /// when the sector lies in an unmapped gap — which shouldn't
    /// happen on a well-formed image but is worth catching.
    fn find_chunk_idx(&self, sector: u64) -> Result<usize> {
        // `partition_point` returns the first index whose start sector
        // is **strictly greater** than `sector`; subtract one to land
        // on the candidate chunk.
        let after = self
            .chunks
            .partition_point(|c| c.virtual_sector_start <= sector);
        if after == 0 {
            return Err(crate::Error::OutOfBounds {
                offset: sector * 512,
                len: 0,
                size: self.virtual_size,
            });
        }
        let idx = after - 1;
        let c = &self.chunks[idx];
        let end = c.virtual_sector_start + c.sector_count;
        if sector >= end {
            return Err(crate::Error::OutOfBounds {
                offset: sector * 512,
                len: 0,
                size: self.virtual_size,
            });
        }
        Ok(idx)
    }
}

/// Inflate `src` as a zlib (RFC 1950) stream into a buffer of exactly
/// `plain_len` bytes. Wraps the `flate2` reader API behind the `gzip`
/// feature flag — the same flag that gates SquashFS gzip / zlib reads,
/// so any build that can open a SquashFS can also open a zlib DMG.
#[cfg(feature = "gzip")]
fn decode_zlib(src: &[u8], plain_len: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut dec = flate2::read::ZlibDecoder::new(src);
    let mut out = Vec::with_capacity(plain_len);
    dec.read_to_end(&mut out)
        .map_err(|e| crate::Error::InvalidImage(format!("dmg: zlib chunk inflate failed: {e}")))?;
    if out.len() != plain_len {
        return Err(crate::Error::InvalidImage(format!(
            "dmg: zlib chunk inflated to {} bytes but sector_count*512 = {}",
            out.len(),
            plain_len
        )));
    }
    Ok(out)
}

#[cfg(not(feature = "gzip"))]
fn decode_zlib(_src: &[u8], _plain_len: usize) -> Result<Vec<u8>> {
    Err(crate::Error::Unsupported(
        "dmg: zlib chunks require the `gzip` Cargo feature (the same one that gates SquashFS gzip)"
            .into(),
    ))
}

impl BlockDevice for DmgBackend {
    fn block_size(&self) -> u32 {
        512
    }

    fn total_size(&self) -> u64 {
        self.virtual_size
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        // Bounds check, matching the trait contract.
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size: self.virtual_size,
            })?;
        if end > self.virtual_size {
            return Err(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size: self.virtual_size,
            });
        }
        if buf.is_empty() {
            return Ok(());
        }

        // Walk one chunk at a time, filling whatever portion of `buf`
        // intersects the chunk's virtual-sector span.
        let mut filled = 0usize;
        let mut cursor = offset;
        while filled < buf.len() {
            let sector = cursor / 512;
            let idx = self.find_chunk_idx(sector)?;
            let chunk = self.chunks[idx];

            let chunk_byte_start = chunk.virtual_sector_start * 512;
            let chunk_byte_end = chunk_byte_start + chunk.sector_count * 512;

            // Decode the chunk once; we may take a partial slice on
            // either end. A future optimisation is to LRU-cache the
            // most recently decoded chunk so sequential reads don't
            // re-inflate the same buffer for every 4 KiB request.
            let plain = self.decode_chunk(&chunk)?;
            debug_assert_eq!(plain.len() as u64, chunk.sector_count * 512);

            let local_start = (cursor - chunk_byte_start) as usize;
            let available = (chunk_byte_end - cursor) as usize;
            let want = (buf.len() - filled).min(available);
            buf[filled..filled + want].copy_from_slice(&plain[local_start..local_start + want]);

            filled += want;
            cursor += want as u64;
        }
        Ok(())
    }

    fn write_at(&mut self, _offset: u64, _buf: &[u8]) -> Result<()> {
        Err(crate::Error::Unsupported(
            "dmg: read-only container; writes are out of scope".into(),
        ))
    }
}

impl Read for DmgBackend {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.cursor >= self.virtual_size {
            return Ok(0);
        }
        let remaining = self.virtual_size - self.cursor;
        let take = (buf.len() as u64).min(remaining) as usize;
        if take == 0 {
            return Ok(0);
        }
        self.read_at(self.cursor, &mut buf[..take])
            .map_err(|e| io::Error::other(format!("{e}")))?;
        self.cursor += take as u64;
        Ok(take)
    }
}

impl Write for DmgBackend {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("dmg: read-only container"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for DmgBackend {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let total = self.virtual_size;
        let new = match pos {
            SeekFrom::Start(o) => o,
            SeekFrom::Current(d) => (self.cursor as i64).saturating_add(d).max(0) as u64,
            SeekFrom::End(d) => (total as i64).saturating_add(d).max(0) as u64,
        };
        self.cursor = new;
        Ok(new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::dmg::mish::{Chunk, ChunkType, encode_mish_for_tests};

    /// Build a minimal koly trailer with the given fields. Big-endian
    /// fields; everything not explicitly set stays zero.
    fn fake_trailer(
        sector_count: u64,
        version: u32,
        data_fork_offset: u64,
        data_fork_length: u64,
        xml_offset: u64,
        xml_length: u64,
    ) -> Vec<u8> {
        let mut v = vec![0u8; KOLY_SIZE as usize];
        v[0x000..0x004].copy_from_slice(&KOLY_MAGIC.to_be_bytes());
        v[0x004..0x008].copy_from_slice(&version.to_be_bytes());
        v[0x008..0x00C].copy_from_slice(&512u32.to_be_bytes());
        v[0x018..0x020].copy_from_slice(&data_fork_offset.to_be_bytes());
        v[0x020..0x028].copy_from_slice(&data_fork_length.to_be_bytes());
        v[0x0D8..0x0E0].copy_from_slice(&xml_offset.to_be_bytes());
        v[0x0E0..0x0E8].copy_from_slice(&xml_length.to_be_bytes());
        v[0x1EC..0x1F4].copy_from_slice(&sector_count.to_be_bytes());
        v
    }

    /// Encode `bytes` as standard-alphabet base64 with `=` padding.
    /// Used only inside this test module to wrap a mish block before
    /// embedding it in the XML plist.
    fn b64_encode(bytes: &[u8]) -> String {
        const ALPHA: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
        for chunk in bytes.chunks(3) {
            let (a, b, c, len) = match chunk.len() {
                3 => (chunk[0], chunk[1], chunk[2], 3),
                2 => (chunk[0], chunk[1], 0, 2),
                1 => (chunk[0], 0, 0, 1),
                _ => unreachable!(),
            };
            let v = ((a as u32) << 16) | ((b as u32) << 8) | (c as u32);
            out.push(ALPHA[((v >> 18) & 0x3F) as usize] as char);
            out.push(ALPHA[((v >> 12) & 0x3F) as usize] as char);
            if len >= 2 {
                out.push(ALPHA[((v >> 6) & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
            if len == 3 {
                out.push(ALPHA[(v & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
        }
        out
    }

    /// Build a complete, openable DMG file in `dir` with a single
    /// mish block holding the given chunks + payload. Returns the
    /// path to the synthesised image.
    fn build_test_dmg(
        dir: &std::path::Path,
        sector_count: u64,
        chunks: &[Chunk],
        data_payload: &[u8],
    ) -> std::path::PathBuf {
        // Layout: [data fork][XML plist][koly trailer].
        let data_offset = 0u64;
        let xml_offset = data_payload.len() as u64;

        let mish_buf = encode_mish_for_tests(0, sector_count, 0, chunks);
        let b64 = b64_encode(&mish_buf);
        let xml = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
  <key>resource-fork</key>
  <dict>
    <key>blkx</key>
    <array>
      <dict>
        <key>Data</key>
        <data>{b64}</data>
      </dict>
    </array>
  </dict>
</dict>
</plist>"#
        );
        let xml_bytes = xml.into_bytes();
        let xml_length = xml_bytes.len() as u64;

        let trailer = fake_trailer(
            sector_count,
            4,
            data_offset,
            data_payload.len() as u64,
            xml_offset,
            xml_length,
        );

        let p = dir.join("img.dmg");
        let mut buf = Vec::new();
        buf.extend_from_slice(data_payload);
        buf.extend_from_slice(&xml_bytes);
        buf.extend_from_slice(&trailer);
        std::fs::write(&p, &buf).unwrap();
        p
    }

    #[test]
    fn decode_recognises_valid_trailer() {
        let buf = fake_trailer(2048, 4, 0, 0, 0, 0);
        let t = KolyTrailer::decode(&buf).unwrap();
        assert_eq!(t.signature, KOLY_MAGIC);
        assert_eq!(t.version, 4);
        assert_eq!(t.header_size, 512);
        assert_eq!(t.sector_count, 2048);
    }

    #[test]
    fn decode_rejects_wrong_magic() {
        let mut buf = fake_trailer(0, 4, 0, 0, 0, 0);
        buf[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        let err = KolyTrailer::decode(&buf).unwrap_err();
        match err {
            crate::Error::InvalidImage(_) => {}
            _ => panic!("expected InvalidImage, got {err:?}"),
        }
    }

    #[test]
    fn decode_rejects_unknown_version() {
        let buf = fake_trailer(0, 3, 0, 0, 0, 0);
        let err = KolyTrailer::decode(&buf).unwrap_err();
        match err {
            crate::Error::Unsupported(_) => {}
            _ => panic!("expected Unsupported, got {err:?}"),
        }
    }

    #[test]
    fn probe_matches_trailing_koly() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("img.dmg");
        let mut content = vec![0u8; 8192];
        content.extend_from_slice(&fake_trailer(16, 4, 0, 0, 0, 0));
        std::fs::write(&p, &content).unwrap();
        assert!(probe(&p).unwrap());
    }

    #[test]
    fn probe_misses_when_no_trailer() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("not.dmg");
        std::fs::write(&p, vec![0u8; 8192]).unwrap();
        assert!(!probe(&p).unwrap());
    }

    #[test]
    fn open_rejects_multi_segment() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("img.dmg");
        let mut t = fake_trailer(0, 4, 0, 0, 0, 0);
        // segment_count at 0x03C
        t[0x03C..0x040].copy_from_slice(&3u32.to_be_bytes());
        let mut content = vec![0u8; 8192];
        content.extend_from_slice(&t);
        std::fs::write(&p, &content).unwrap();
        let err = DmgBackend::open(&p).unwrap_err();
        match err {
            crate::Error::Unsupported(_) => {}
            _ => panic!("expected Unsupported, got {err:?}"),
        }
    }

    /// End-to-end: build a one-chunk raw DMG, open it, read the bytes
    /// back. Pins the data fork + XML plist + koly layout integration.
    #[test]
    fn round_trip_raw_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let mut payload = vec![0u8; 512];
        for (i, b) in payload.iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
        let chunks = vec![Chunk {
            kind: ChunkType::Raw,
            virtual_sector_start: 0,
            sector_count: 1,
            compressed_offset_in_fork: 0,
            compressed_length: 512,
        }];
        let p = build_test_dmg(dir.path(), 1, &chunks, &payload);

        let mut dmg = DmgBackend::open(&p).unwrap();
        assert_eq!(dmg.total_size(), 512);
        assert_eq!(dmg.chunk_count(), 1);

        let mut out = vec![0u8; 512];
        dmg.read_at(0, &mut out).unwrap();
        assert_eq!(out, payload);

        // Partial read in the middle.
        let mut out2 = vec![0u8; 16];
        dmg.read_at(100, &mut out2).unwrap();
        assert_eq!(out2, &payload[100..116]);
    }

    /// Zero-fill chunks must produce zero bytes without referencing
    /// the data fork at all.
    #[test]
    fn round_trip_zero_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let chunks = vec![Chunk {
            kind: ChunkType::Zero,
            virtual_sector_start: 0,
            sector_count: 4,
            compressed_offset_in_fork: 0,
            compressed_length: 0,
        }];
        let p = build_test_dmg(dir.path(), 4, &chunks, &[]);

        let mut dmg = DmgBackend::open(&p).unwrap();
        assert_eq!(dmg.total_size(), 4 * 512);
        let mut out = vec![0xAAu8; 1024];
        dmg.read_at(256, &mut out).unwrap();
        assert!(out.iter().all(|&b| b == 0));
    }

    /// Zlib chunk: deflate a known payload, embed it, read it back.
    /// Cross-checks the chunk router + the flate2 wiring.
    #[cfg(feature = "gzip")]
    #[test]
    fn round_trip_zlib_chunk() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        // 1024 bytes of varied data so deflate produces a meaningful
        // payload, not just a stored block.
        let mut plain = vec![0u8; 1024];
        for (i, b) in plain.iter_mut().enumerate() {
            *b = ((i * 31 + 7) & 0xFF) as u8;
        }
        let mut compressed = Vec::new();
        {
            let mut enc =
                flate2::write::ZlibEncoder::new(&mut compressed, flate2::Compression::default());
            enc.write_all(&plain).unwrap();
            enc.finish().unwrap();
        }
        let chunks = vec![Chunk {
            kind: ChunkType::Zlib,
            virtual_sector_start: 0,
            sector_count: 2,
            compressed_offset_in_fork: 0,
            compressed_length: compressed.len() as u64,
        }];
        let p = build_test_dmg(dir.path(), 2, &chunks, &compressed);

        let mut dmg = DmgBackend::open(&p).unwrap();
        let mut out = vec![0u8; 1024];
        dmg.read_at(0, &mut out).unwrap();
        assert_eq!(out, plain);

        // Cross-chunk-boundary read isn't exercised here (only one
        // chunk), but partial-range reads inside the chunk are.
        let mut out2 = vec![0u8; 200];
        dmg.read_at(700, &mut out2).unwrap();
        assert_eq!(out2, &plain[700..900]);
    }

    /// bzip2 chunk end-to-end: pre-generated `bzip2 -z` of a 512-byte
    /// pattern, embedded as a single bz2 chunk, read back through
    /// `read_at`. Pins the codec dispatch + the bzip2-rs reader.
    #[cfg(feature = "dmg-bzip2")]
    #[test]
    fn round_trip_bzip2_chunk() {
        let dir = tempfile::tempdir().unwrap();
        // A 512-byte plaintext we can name (`AB.AB.AB...`). bzip2 of
        // this is generated below at test time by piping into the
        // bzip2-rs companion encoder. Since bzip2-rs is decode-only,
        // we use a pre-baked compressed vector instead.
        // Vector below was generated by `bzip2 -z` on a 512-byte
        // buffer of bytes (i*7+3) & 0xFF, then captured as inline
        // hex. Pure-decode test — no encoder dep required.
        let mut plain = vec![0u8; 512];
        for (i, b) in plain.iter_mut().enumerate() {
            *b = ((i * 7 + 3) & 0xFF) as u8;
        }
        // Generated with bzip2(1) on the same `plain` buffer.
        let compressed: &[u8] = include_bytes!("testdata/pattern_512_i7p3.bz2");

        let chunks = vec![Chunk {
            kind: ChunkType::Bz2,
            virtual_sector_start: 0,
            sector_count: 1,
            compressed_offset_in_fork: 0,
            compressed_length: compressed.len() as u64,
        }];
        let p = build_test_dmg(dir.path(), 1, &chunks, compressed);
        let mut dmg = DmgBackend::open(&p).unwrap();
        let mut out = vec![0u8; 512];
        dmg.read_at(0, &mut out).unwrap();
        assert_eq!(out, plain);
    }

    /// LZMA chunk end-to-end: compress on the fly with lzma-rs, embed
    /// the raw-LZMA1 frame as a single chunk, decode through `read_at`.
    #[cfg(feature = "lzma")]
    #[test]
    fn round_trip_lzma_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let mut plain = vec![0u8; 1024];
        for (i, b) in plain.iter_mut().enumerate() {
            *b = ((i ^ (i >> 3)) & 0xFF) as u8;
        }
        let mut compressed = Vec::new();
        {
            let mut input = std::io::Cursor::new(&plain);
            lzma_rs::lzma_compress(&mut input, &mut compressed).unwrap();
        }
        let chunks = vec![Chunk {
            kind: ChunkType::Lzma,
            virtual_sector_start: 0,
            sector_count: 2,
            compressed_offset_in_fork: 0,
            compressed_length: compressed.len() as u64,
        }];
        let p = build_test_dmg(dir.path(), 2, &chunks, &compressed);
        let mut dmg = DmgBackend::open(&p).unwrap();
        let mut out = vec![0u8; 1024];
        dmg.read_at(0, &mut out).unwrap();
        assert_eq!(out, plain);
    }

    /// LZFSE chunk end-to-end: compress with `lzfse_rust`, embed,
    /// read back. Pins the codec dispatch + the lzfse_rust decoder.
    #[cfg(feature = "dmg-lzfse")]
    #[test]
    fn round_trip_lzfse_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let mut plain = vec![0u8; 4096];
        for (i, b) in plain.iter_mut().enumerate() {
            *b = ((i * 13 + 5) & 0xFF) as u8;
        }
        let mut compressed = Vec::new();
        lzfse_rust::encode_bytes(&plain, &mut compressed).unwrap();
        let chunks = vec![Chunk {
            kind: ChunkType::Lzfse,
            virtual_sector_start: 0,
            sector_count: 8,
            compressed_offset_in_fork: 0,
            compressed_length: compressed.len() as u64,
        }];
        let p = build_test_dmg(dir.path(), 8, &chunks, &compressed);
        let mut dmg = DmgBackend::open(&p).unwrap();
        let mut out = vec![0u8; 4096];
        dmg.read_at(0, &mut out).unwrap();
        assert_eq!(out, plain);
    }

    /// ADC chunk end-to-end: build a tiny ADC stream by hand using
    /// known opcode forms and feed it through the full pipeline.
    /// Plaintext is 512 bytes of `0xAA` — an overlap-run that ADC
    /// encodes as `[literal 0xAA] [short-ref dist=1 len=...]`.
    #[test]
    fn round_trip_adc_chunk() {
        let dir = tempfile::tempdir().unwrap();
        // 512 bytes of 0xAA = one literal byte + many short-ref runs
        // of length 18 each (the max a short-ref encodes).
        // Build the stream as: literal(0xAA) followed by N short-refs
        // of dist=1, length=18, then a final short-ref for the
        // remainder.
        let mut stream: Vec<u8> = Vec::new();
        // Literal: opcode 0x00 (length 1), data byte 0xAA.
        stream.push(0x00);
        stream.push(0xAA);
        let mut emitted: usize = 1;
        while emitted < 512 {
            let want = (512 - emitted).min(18);
            // short-ref: op = 0x80 | ((want-3) << 2) | hi_of_dist0
            //          = 0x80 | ((want-3) << 2)  (dist 1 ⇒ dist-1 = 0)
            // trailing byte = lo_of_dist0 = 0
            let op = 0x80u8 | (((want - 3) as u8) << 2);
            stream.push(op);
            stream.push(0x00);
            emitted += want;
        }

        let chunks = vec![Chunk {
            kind: ChunkType::Adc,
            virtual_sector_start: 0,
            sector_count: 1,
            compressed_offset_in_fork: 0,
            compressed_length: stream.len() as u64,
        }];
        let p = build_test_dmg(dir.path(), 1, &chunks, &stream);
        let mut dmg = DmgBackend::open(&p).unwrap();
        let mut out = vec![0u8; 512];
        dmg.read_at(0, &mut out).unwrap();
        assert!(out.iter().all(|&b| b == 0xAA));
    }

    /// Mixed-chunk image: zero-fill followed by raw followed by zlib.
    /// Exercises the binary-search router on more than one chunk and
    /// confirms the boundary arithmetic for cross-chunk reads.
    #[cfg(feature = "gzip")]
    #[test]
    fn round_trip_mixed_chunks() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();

        // Data layout, in sectors:
        //  0..2  zero fill
        //  2..4  raw       (1024 bytes of pattern A)
        //  4..6  zlib      (1024 bytes of pattern B)
        let mut raw_payload = vec![0u8; 1024];
        for (i, b) in raw_payload.iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
        let mut zlib_plain = vec![0u8; 1024];
        for (i, b) in zlib_plain.iter_mut().enumerate() {
            *b = ((255 - (i & 0xFF)) & 0xFF) as u8;
        }
        let mut zlib_payload = Vec::new();
        {
            let mut enc =
                flate2::write::ZlibEncoder::new(&mut zlib_payload, flate2::Compression::default());
            enc.write_all(&zlib_plain).unwrap();
            enc.finish().unwrap();
        }

        // Data fork = raw_payload || zlib_payload.
        let mut fork = Vec::new();
        let raw_off = fork.len() as u64;
        fork.extend_from_slice(&raw_payload);
        let zlib_off = fork.len() as u64;
        fork.extend_from_slice(&zlib_payload);

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
                sector_count: 2,
                compressed_offset_in_fork: raw_off,
                compressed_length: 1024,
            },
            Chunk {
                kind: ChunkType::Zlib,
                virtual_sector_start: 4,
                sector_count: 2,
                compressed_offset_in_fork: zlib_off,
                compressed_length: zlib_payload.len() as u64,
            },
        ];
        let p = build_test_dmg(dir.path(), 6, &chunks, &fork);

        let mut dmg = DmgBackend::open(&p).unwrap();
        assert_eq!(dmg.chunk_count(), 3);
        assert_eq!(dmg.total_size(), 6 * 512);

        // Read the whole virtual disk in one shot — exercises the
        // cross-chunk walk in `read_at`.
        let mut out = vec![0xAAu8; 6 * 512];
        dmg.read_at(0, &mut out).unwrap();
        assert!(out[..1024].iter().all(|&b| b == 0), "zero region");
        assert_eq!(&out[1024..2048], raw_payload.as_slice(), "raw region");
        assert_eq!(&out[2048..3072], zlib_plain.as_slice(), "zlib region");

        // Read straddling the raw → zlib boundary.
        let mut out2 = vec![0u8; 32];
        dmg.read_at(2048 - 16, &mut out2).unwrap();
        assert_eq!(&out2[..16], &raw_payload[1008..1024]);
        assert_eq!(&out2[16..], &zlib_plain[..16]);
    }

    /// Out-of-bounds reads must return `OutOfBounds` rather than
    /// silently returning zero or panicking.
    #[test]
    fn read_at_rejects_out_of_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let chunks = vec![Chunk {
            kind: ChunkType::Zero,
            virtual_sector_start: 0,
            sector_count: 1,
            compressed_offset_in_fork: 0,
            compressed_length: 0,
        }];
        let p = build_test_dmg(dir.path(), 1, &chunks, &[]);
        let mut dmg = DmgBackend::open(&p).unwrap();
        let mut buf = [0u8; 8];
        let err = dmg.read_at(512, &mut buf).unwrap_err();
        match err {
            crate::Error::OutOfBounds { .. } => {}
            _ => panic!("expected OutOfBounds, got {err:?}"),
        }
    }
}
