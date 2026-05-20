//! HFSCompression (`com.apple.decmpfs`) — transparent file compression.
//!
//! When a file's BSD `ownerFlags` carries the `UF_COMPRESSED` bit
//! (`0x20`), its data fork is empty and the logical content lives in
//! the `com.apple.decmpfs` extended attribute (a 16-byte header
//! optionally followed by inline compressed data). For larger files
//! the header points at the file's *resource fork*, which holds a
//! block table + per-block compressed chunks of ≤ 64 KiB each.
//!
//! ## Header layout (`decmpfs_disk_header`, 16 bytes)
//!
//! ```text
//! offset  size  field                 endianness
//! 0       4     compression_magic     BE ("fpmc" = 0x66 0x70 0x6D 0x63)
//! 4       4     compression_type      LE
//! 8       8     uncompressed_size     LE
//! 16      ...   optional inline data
//! ```
//!
//! The magic spells "cmpf" big-endian; the rest of the struct is
//! little-endian. Type field identifies the codec and where the data
//! lives:
//!
//! | type | codec  | location           |
//! |-----:|--------|--------------------|
//! | 1    | none   | xattr (uncompressed)|
//! | 3    | zlib   | xattr (inline)     |
//! | 4    | zlib   | resource fork      |
//! | 7    | LZVN   | xattr (inline)     |
//! | 8    | LZVN   | resource fork      |
//! | 11   | LZFSE  | xattr (inline)     |
//! | 12   | LZFSE  | resource fork      |
//!
//! Only zlib (types 3 and 4) is implemented here. The other codecs
//! return [`crate::Error::Unsupported`] when encountered; that's the
//! pragmatic choice given that most real images we care about (Mac OS
//! installer DMGs, Time Machine backups) use zlib for the vast
//! majority of compressed files.
//!
//! ## Resource fork layout (compression_type = 4)
//!
//! Standard Mac OS resource-fork framing wraps a single "block table"
//! record that lists `(offset, length)` pairs into the resource data
//! area. The data area starts at the `dataOffset` from the 16-byte
//! resource-fork header (offsets 0..16):
//!
//! ```text
//! resource header
//! offset  size  field
//! 0       4     dataOffset   (BE, byte offset to the resource data)
//! 4       4     mapOffset    (BE)
//! 8       4     dataLength   (BE)
//! 12      4     mapLength    (BE)
//!
//! at dataOffset:
//! offset  size  field
//! 0       4     blockCount   (LE)
//! 4       8*N   blockTable   (N × (u32 LE offset, u32 LE length))
//! ```
//!
//! Each block's `offset` is relative to the *block table start*
//! (i.e. the first byte after the resource header's `dataOffset`),
//! and `length` is the on-disk compressed size of that block. Each
//! block decompresses to at most 64 KiB; the final block carries the
//! tail and may be shorter.

use crate::Result;

/// "fpmc" — big-endian "cmpf" magic at the head of the decmpfs header.
pub const DECMPFS_MAGIC: u32 = 0x636d_7066;

/// The `UF_COMPRESSED` bit in `HFSPlusBSDInfo.ownerFlags`, indicating
/// that the file's payload lives in `com.apple.decmpfs` rather than
/// the data fork. Documented in Darwin's `sys/stat.h`.
pub const UF_COMPRESSED: u8 = 0x20;

/// HFSCompression block size — every compressed block decompresses
/// to at most this many bytes.
pub const HFSCOMPRESS_BLOCK_SIZE: usize = 65_536;

/// Compression types we recognise. Only `ZlibAttr` and `ZlibResource`
/// are decoded; the rest are returned as `Unsupported`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    /// Type 1 — inline, uncompressed.
    Uncompressed,
    /// Type 3 — inline, zlib.
    ZlibAttr,
    /// Type 4 — resource fork, zlib.
    ZlibResource,
    /// Type 7 — inline, LZVN.
    LzvnAttr,
    /// Type 8 — resource fork, LZVN.
    LzvnResource,
    /// Type 11 — inline, LZFSE.
    LzfseAttr,
    /// Type 12 — resource fork, LZFSE.
    LzfseResource,
    /// Anything else — record the numeric tag so the error message is
    /// informative.
    Unknown(u32),
}

impl CompressionType {
    fn from_u32(t: u32) -> Self {
        match t {
            1 => Self::Uncompressed,
            3 => Self::ZlibAttr,
            4 => Self::ZlibResource,
            7 => Self::LzvnAttr,
            8 => Self::LzvnResource,
            11 => Self::LzfseAttr,
            12 => Self::LzfseResource,
            other => Self::Unknown(other),
        }
    }

    /// Whether this codec's data lives in the resource fork rather
    /// than inline in the attribute record.
    pub fn is_resource_fork(self) -> bool {
        matches!(
            self,
            Self::ZlibResource | Self::LzvnResource | Self::LzfseResource
        )
    }
}

/// Decoded decmpfs header.
#[derive(Debug, Clone, Copy)]
pub struct DecmpfsHeader {
    /// Compression codec & data location.
    pub compression_type: CompressionType,
    /// Total uncompressed file size in bytes.
    pub uncompressed_size: u64,
}

impl DecmpfsHeader {
    /// Encoded size of the on-disk header (the inline data — if any —
    /// follows this).
    pub const SIZE: usize = 16;

    /// Decode a `decmpfs_disk_header` from `buf`.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(crate::Error::InvalidImage(
                "hfs+: decmpfs xattr shorter than 16-byte header".into(),
            ));
        }
        let magic = u32::from_be_bytes(buf[0..4].try_into().unwrap());
        if magic != DECMPFS_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: decmpfs bad magic {magic:#010x} (expected {DECMPFS_MAGIC:#010x})"
            )));
        }
        let compression_type = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let uncompressed_size = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        Ok(Self {
            compression_type: CompressionType::from_u32(compression_type),
            uncompressed_size,
        })
    }
}

/// Decompress the inline payload following a decmpfs header. `tail`
/// is the xattr bytes after the 16-byte header. `expected_len` is the
/// uncompressed_size carried by the header — the decoded output MUST
/// match it byte-for-byte.
///
/// Handles only types 3 (zlib inline) and 1 (uncompressed inline);
/// other codecs return `Unsupported`.
pub fn decompress_inline(
    compression_type: CompressionType,
    tail: &[u8],
    expected_len: u64,
) -> Result<Vec<u8>> {
    match compression_type {
        CompressionType::Uncompressed => {
            if tail.len() as u64 != expected_len {
                return Err(crate::Error::InvalidImage(format!(
                    "hfs+: decmpfs type 1 payload is {} bytes but header says {expected_len}",
                    tail.len()
                )));
            }
            Ok(tail.to_vec())
        }
        CompressionType::ZlibAttr => decompress_zlib_block(tail, expected_len as usize),
        CompressionType::Unknown(t) => Err(crate::Error::Unsupported(format!(
            "hfs+: decmpfs unknown compression type {t}"
        ))),
        other => Err(crate::Error::Unsupported(format!(
            "hfs+: decmpfs compression type {other:?} not yet implemented"
        ))),
    }
}

/// Decompress a resource-fork payload (compression type 4).
///
/// `resource_bytes` is the entire resource fork, exactly as returned
/// by a fork reader. `expected_len` is the uncompressed_size from the
/// decmpfs header.
pub fn decompress_resource_fork(
    compression_type: CompressionType,
    resource_bytes: &[u8],
    expected_len: u64,
) -> Result<Vec<u8>> {
    match compression_type {
        CompressionType::ZlibResource => decompress_resource_zlib(resource_bytes, expected_len),
        CompressionType::LzvnResource | CompressionType::LzfseResource => {
            Err(crate::Error::Unsupported(format!(
                "hfs+: decmpfs compression type {compression_type:?} not yet implemented"
            )))
        }
        other => Err(crate::Error::InvalidImage(format!(
            "hfs+: decmpfs type {other:?} should not point at the resource fork"
        ))),
    }
}

/// Walk the resource-fork header + block table and inflate each
/// per-block zlib stream into a single contiguous buffer of
/// `expected_len` bytes.
fn decompress_resource_zlib(rf: &[u8], expected_len: u64) -> Result<Vec<u8>> {
    if rf.len() < 16 {
        return Err(crate::Error::InvalidImage(
            "hfs+: decmpfs resource fork shorter than 16-byte header".into(),
        ));
    }
    let data_offset = u32::from_be_bytes(rf[0..4].try_into().unwrap()) as usize;
    let data_length = u32::from_be_bytes(rf[8..12].try_into().unwrap()) as usize;
    if data_offset.saturating_add(data_length) > rf.len() {
        return Err(crate::Error::InvalidImage(format!(
            "hfs+: decmpfs resource data {data_offset}..+{data_length} exceeds fork ({} bytes)",
            rf.len()
        )));
    }
    if data_length < 4 {
        return Err(crate::Error::InvalidImage(
            "hfs+: decmpfs resource data too short for blockCount header".into(),
        ));
    }
    // The block table is *inside* the resource data area, immediately
    // after the 4-byte data-length prefix that wraps a single resource
    // record. The decmpfs convention is: data_offset points at a u32
    // BE wrapper, but the block table actually starts 4 bytes in —
    // i.e. data_offset + 4. Block offsets are themselves relative to
    // that same starting point (data_offset + 4).
    //
    // Concretely the layout at `rf[data_offset..]` is:
    //   data_offset+0  4 bytes  unknown-or-record-length (BE)
    //   data_offset+4  4 bytes  blockCount               (LE)
    //   data_offset+8  8*N      blockTable               (N × (u32 LE off, u32 LE len))
    //   ...           ...       block streams
    // Block offsets are added to `data_offset + 4` to reach the start
    // of each zlib block within `rf`.
    let table_base = data_offset
        .checked_add(4)
        .ok_or_else(|| crate::Error::InvalidImage("hfs+: decmpfs data_offset overflow".into()))?;
    if table_base + 4 > rf.len() {
        return Err(crate::Error::InvalidImage(
            "hfs+: decmpfs resource block-count past end of fork".into(),
        ));
    }
    let block_count =
        u32::from_le_bytes(rf[table_base..table_base + 4].try_into().unwrap()) as usize;
    if block_count == 0 {
        // No blocks — file must be empty per the header.
        if expected_len != 0 {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: decmpfs resource has 0 blocks but header says {expected_len} bytes"
            )));
        }
        return Ok(Vec::new());
    }
    let table_off = table_base + 4;
    let table_end = table_off
        .checked_add(block_count.checked_mul(8).ok_or_else(|| {
            crate::Error::InvalidImage("hfs+: decmpfs block count overflow".into())
        })?)
        .ok_or_else(|| {
            crate::Error::InvalidImage("hfs+: decmpfs block table end overflow".into())
        })?;
    if table_end > rf.len() {
        return Err(crate::Error::InvalidImage(format!(
            "hfs+: decmpfs block table extends past resource fork (need {table_end}, have {})",
            rf.len()
        )));
    }
    // Inflate each block.
    let expected = usize::try_from(expected_len).map_err(|_| {
        crate::Error::InvalidImage(format!(
            "hfs+: decmpfs uncompressed size {expected_len} exceeds usize"
        ))
    })?;
    let mut out = Vec::with_capacity(expected);
    for i in 0..block_count {
        let entry_off = table_off + 8 * i;
        let blk_off = u32::from_le_bytes(rf[entry_off..entry_off + 4].try_into().unwrap()) as usize;
        let blk_len =
            u32::from_le_bytes(rf[entry_off + 4..entry_off + 8].try_into().unwrap()) as usize;
        let blk_start = table_base.checked_add(blk_off).ok_or_else(|| {
            crate::Error::InvalidImage("hfs+: decmpfs block offset overflow".into())
        })?;
        let blk_end = blk_start.checked_add(blk_len).ok_or_else(|| {
            crate::Error::InvalidImage("hfs+: decmpfs block length overflow".into())
        })?;
        if blk_end > rf.len() {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: decmpfs block {i} ({blk_off}..+{blk_len}) extends past resource fork"
            )));
        }
        let remaining = expected.saturating_sub(out.len());
        let block_target = remaining.min(HFSCOMPRESS_BLOCK_SIZE);
        let chunk = &rf[blk_start..blk_end];
        // Apple's decmpfs marks a block as "stored verbatim" when its
        // first byte is 0xFF — the rest of the block is the raw
        // uncompressed payload, no zlib wrapper. This shows up on
        // incompressible data and is required for byte-exact reads.
        if !chunk.is_empty() && chunk[0] == 0xFF {
            let payload = &chunk[1..];
            if payload.len() != block_target {
                return Err(crate::Error::InvalidImage(format!(
                    "hfs+: decmpfs raw block {i} is {} bytes but expected {block_target}",
                    payload.len()
                )));
            }
            out.extend_from_slice(payload);
            continue;
        }
        let decoded = decompress_zlib_block(chunk, block_target)?;
        out.extend_from_slice(&decoded);
    }
    if out.len() as u64 != expected_len {
        return Err(crate::Error::InvalidImage(format!(
            "hfs+: decmpfs decoded {} bytes but header says {expected_len}",
            out.len()
        )));
    }
    Ok(out)
}

/// Inflate a zlib (RFC 1950) stream that should produce exactly
/// `target_len` bytes. The decmpfs convention is identical to the
/// DMG zlib path — we gate on the `gzip` feature for the same reason
/// (`flate2`).
#[cfg(feature = "gzip")]
fn decompress_zlib_block(src: &[u8], target_len: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut dec = flate2::read::ZlibDecoder::new(src);
    let mut out = Vec::with_capacity(target_len);
    dec.read_to_end(&mut out).map_err(|e| {
        crate::Error::InvalidImage(format!("hfs+: decmpfs zlib block inflate failed: {e}"))
    })?;
    if out.len() != target_len {
        return Err(crate::Error::InvalidImage(format!(
            "hfs+: decmpfs zlib block inflated to {} bytes but expected {target_len}",
            out.len()
        )));
    }
    Ok(out)
}

#[cfg(not(feature = "gzip"))]
fn decompress_zlib_block(_src: &[u8], _target_len: usize) -> Result<Vec<u8>> {
    Err(crate::Error::Unsupported(
        "hfs+: decmpfs zlib decompression requires the `gzip` Cargo feature".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_decode_type_3() {
        // Magic big-endian "cmpf" = 0x636d7066 -> bytes 0x63 6D 70 66
        // when read big-endian. The header carries the magic at offset
        // 0 in big-endian, followed by type+size little-endian.
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&DECMPFS_MAGIC.to_be_bytes());
        buf[4..8].copy_from_slice(&3u32.to_le_bytes());
        buf[8..16].copy_from_slice(&1234u64.to_le_bytes());
        let h = DecmpfsHeader::decode(&buf).unwrap();
        assert_eq!(h.compression_type, CompressionType::ZlibAttr);
        assert_eq!(h.uncompressed_size, 1234);
    }

    #[test]
    fn header_decode_rejects_bad_magic() {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&0xdeadbeefu32.to_be_bytes());
        assert!(DecmpfsHeader::decode(&buf).is_err());
    }

    #[test]
    fn header_decode_too_short() {
        assert!(DecmpfsHeader::decode(&[0u8; 8]).is_err());
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn inline_zlib_round_trip() {
        use flate2::{Compression, write::ZlibEncoder};
        use std::io::Write;
        let plain = b"hello hfsplus decmpfs world!".repeat(8);
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&plain).unwrap();
        let compressed = enc.finish().unwrap();
        let out =
            decompress_inline(CompressionType::ZlibAttr, &compressed, plain.len() as u64).unwrap();
        assert_eq!(out, plain);
    }

    #[test]
    fn inline_uncompressed_pass_through() {
        let plain = b"plain text".to_vec();
        let out =
            decompress_inline(CompressionType::Uncompressed, &plain, plain.len() as u64).unwrap();
        assert_eq!(out, plain);
    }

    #[test]
    fn inline_lzvn_returns_unsupported() {
        // LZVN inline is not yet implemented; we expose an Unsupported
        // error so callers can degrade gracefully.
        let r = decompress_inline(CompressionType::LzvnAttr, &[0u8; 4], 0);
        assert!(matches!(r, Err(crate::Error::Unsupported(_))));
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn resource_fork_zlib_round_trip() {
        use flate2::{Compression, write::ZlibEncoder};
        use std::io::Write;

        // Build a synthesized resource fork with two zlib blocks
        // (forcing the block-table path). First block is 64 KiB,
        // second is the tail.
        let mut plain = Vec::new();
        plain.extend(std::iter::repeat_n(0xABu8, HFSCOMPRESS_BLOCK_SIZE));
        plain.extend_from_slice(b"second block tail of hfscompression test data");
        let block1 = &plain[..HFSCOMPRESS_BLOCK_SIZE];
        let block2 = &plain[HFSCOMPRESS_BLOCK_SIZE..];

        let compress = |data: &[u8]| -> Vec<u8> {
            let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
            e.write_all(data).unwrap();
            e.finish().unwrap()
        };
        let c1 = compress(block1);
        let c2 = compress(block2);

        // Block-table layout (relative to table_base = data_offset + 4):
        //   table_off  = 0   -> block_count u32 LE
        //   table_off  = 4   -> N × (u32 off, u32 len), offsets relative to table_base
        //   ...               -> block bytes
        let block_count: u32 = 2;
        let table_size = 4 + 8 * block_count as usize;
        let blk1_off = table_size as u32; // first byte after the table
        let blk2_off = blk1_off + c1.len() as u32;

        let mut rdata = Vec::new();
        // 4-byte wrapper length (some readers ignore this; we set it
        // to the size of the rest of the resource record).
        let inner_size: u32 = (table_size + c1.len() + c2.len()) as u32;
        rdata.extend_from_slice(&inner_size.to_be_bytes());
        // block_count + table
        rdata.extend_from_slice(&block_count.to_le_bytes());
        rdata.extend_from_slice(&blk1_off.to_le_bytes());
        rdata.extend_from_slice(&(c1.len() as u32).to_le_bytes());
        rdata.extend_from_slice(&blk2_off.to_le_bytes());
        rdata.extend_from_slice(&(c2.len() as u32).to_le_bytes());
        rdata.extend_from_slice(&c1);
        rdata.extend_from_slice(&c2);

        // Resource-fork header (16 bytes BE).
        let data_offset: u32 = 256; // Apple's resource-fork header reserves 256 bytes
        let data_length: u32 = rdata.len() as u32;
        let mut rf = Vec::new();
        rf.extend_from_slice(&data_offset.to_be_bytes());
        rf.extend_from_slice(&(data_offset + data_length).to_be_bytes()); // map_offset (we don't use it)
        rf.extend_from_slice(&data_length.to_be_bytes());
        rf.extend_from_slice(&0u32.to_be_bytes()); // map_length
        // Pad up to data_offset
        rf.resize(data_offset as usize, 0);
        rf.extend_from_slice(&rdata);

        let out = decompress_resource_fork(CompressionType::ZlibResource, &rf, plain.len() as u64)
            .unwrap();
        assert_eq!(out.len(), plain.len());
        assert_eq!(out, plain);
    }
}
