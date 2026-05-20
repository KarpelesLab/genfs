//! DMG chunk decoders for the compressed entry types that aren't
//! zero / raw / zlib.
//!
//! Each function takes the raw compressed payload as recorded in the
//! data fork and writes exactly `plain_len` decoded bytes back. The
//! callers in [`crate::block::dmg`] derive `plain_len` from
//! `chunk.sector_count * 512`, which is the contract every BLKX entry
//! advertises.
//!
//! ## Codec choices
//!
//! - **bzip2**: `bzip2-rs`, pure-Rust decoder.
//! - **LZFSE**: `lzfse_rust`, pure-Rust en/decoder.
//! - **LZMA**: `lzma-rs::lzma_decompress` over the raw LZMA1 frame
//!   (DMG never uses the XZ container).
//! - **ADC**: implemented in this file from the public spec described
//!   in Apple's `imageformat.h`; no external dep.
//!
//! When a build is configured without one of the optional codec deps
//! the corresponding `decode_*` function returns
//! [`crate::Error::Unsupported`] so a slim binary can still open zlib
//! / raw / zero-only images.

use crate::Result;

/// Decode an ADC (Apple Data Compression) stream into `out`.
///
/// ADC is a tiny LZ77 variant. The encoder emits one of three opcodes:
///
/// - `0x00..=0x7F`: literal run. The opcode byte's value + 1 (1..=128)
///   bytes follow directly in the stream and are copied verbatim.
/// - `0x80..=0xBF`: short back-reference. One additional byte follows.
///   - length = ((opcode >> 2) & 0x0F) + 3        →  3..=18
///   - distance = (((opcode & 0x03) << 8) | next) + 1  →  1..=1024
/// - `0xC0..=0xFF`: long back-reference. Two additional bytes follow
///   (big-endian).
///   - length = (opcode & 0x3F) + 4               →  4..=67
///   - distance = ((hi << 8) | lo) + 1            →  1..=65536
///
/// The back-reference distance is measured from the **current write
/// position**, looking backward into already-decoded bytes. Length and
/// distance can overlap (`distance < length`); when they do, the copy
/// must propagate forward byte-by-byte so each freshly written byte is
/// visible to the rest of the run. That's how ADC encodes simple runs
/// like `aaaa` (distance 1, length 4).
///
/// The function writes into the prefix of `out`, returning the number
/// of bytes produced. Callers should size `out` to the chunk's expected
/// plain length and treat any short / long result as a malformed image.
pub fn decode_adc(src: &[u8], out: &mut [u8]) -> Result<usize> {
    let mut sp = 0usize;
    let mut dp = 0usize;
    while sp < src.len() {
        let op = src[sp];
        sp += 1;

        if op < 0x80 {
            // Literal run of (op + 1) bytes.
            let len = op as usize + 1;
            if sp + len > src.len() {
                return Err(crate::Error::InvalidImage(format!(
                    "dmg/adc: literal run of {len} bytes runs past source ({} of {} consumed)",
                    sp,
                    src.len()
                )));
            }
            if dp + len > out.len() {
                return Err(crate::Error::InvalidImage(format!(
                    "dmg/adc: literal run of {len} bytes overflows output ({}+{}>{})",
                    dp,
                    len,
                    out.len()
                )));
            }
            out[dp..dp + len].copy_from_slice(&src[sp..sp + len]);
            sp += len;
            dp += len;
        } else if op < 0xC0 {
            // Short reference: op + one trailing byte.
            if sp >= src.len() {
                return Err(crate::Error::InvalidImage(
                    "dmg/adc: short reference truncated".into(),
                ));
            }
            let b = src[sp];
            sp += 1;
            let len = (((op >> 2) & 0x0F) as usize) + 3;
            let dist = ((((op & 0x03) as usize) << 8) | b as usize) + 1;
            adc_copy(out, &mut dp, dist, len)?;
        } else {
            // Long reference: op + two trailing big-endian bytes.
            if sp + 2 > src.len() {
                return Err(crate::Error::InvalidImage(
                    "dmg/adc: long reference truncated".into(),
                ));
            }
            let hi = src[sp] as usize;
            let lo = src[sp + 1] as usize;
            sp += 2;
            let len = ((op & 0x3F) as usize) + 4;
            let dist = ((hi << 8) | lo) + 1;
            adc_copy(out, &mut dp, dist, len)?;
        }
    }
    Ok(dp)
}

/// Inner copy helper for ADC back-references. Propagates byte-by-byte
/// so overlapping copies (distance < length) replicate the source run
/// the way ADC encoders rely on.
fn adc_copy(out: &mut [u8], dp: &mut usize, dist: usize, len: usize) -> Result<()> {
    if dist == 0 || dist > *dp {
        return Err(crate::Error::InvalidImage(format!(
            "dmg/adc: back-reference distance {dist} out of range (dp = {dp})",
            dp = *dp
        )));
    }
    if *dp + len > out.len() {
        return Err(crate::Error::InvalidImage(format!(
            "dmg/adc: back-reference of {len} bytes overflows output ({}+{}>{})",
            *dp,
            len,
            out.len()
        )));
    }
    for i in 0..len {
        out[*dp + i] = out[*dp + i - dist];
    }
    *dp += len;
    Ok(())
}

/// Decode a bzip2 chunk payload into a vector of exactly `plain_len`
/// bytes. DMG bz2 chunks are standalone bzip2 streams (BZh magic,
/// no additional framing).
#[cfg(feature = "dmg-bzip2")]
pub fn decode_bzip2(src: &[u8], plain_len: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut rdr = bzip2_rs::DecoderReader::new(src);
    let mut out = Vec::with_capacity(plain_len);
    rdr.read_to_end(&mut out)
        .map_err(|e| crate::Error::InvalidImage(format!("dmg: bzip2 chunk decode failed: {e}")))?;
    if out.len() != plain_len {
        return Err(crate::Error::InvalidImage(format!(
            "dmg: bzip2 chunk inflated to {} bytes but sector_count*512 = {}",
            out.len(),
            plain_len
        )));
    }
    Ok(out)
}

/// Stub when the bzip2 dep was compiled out.
#[cfg(not(feature = "dmg-bzip2"))]
pub fn decode_bzip2(_src: &[u8], _plain_len: usize) -> Result<Vec<u8>> {
    Err(crate::Error::Unsupported(
        "dmg: bzip2 chunks require the `dmg-bzip2` Cargo feature".into(),
    ))
}

/// Decode an LZFSE chunk payload into a vector of exactly `plain_len`
/// bytes. DMG LZFSE chunks are standalone bvxN/bvx2/bvx1/bvx- framed
/// blocks the `lzfse_rust` crate's `decode_bytes` understands.
#[cfg(feature = "dmg-lzfse")]
pub fn decode_lzfse(src: &[u8], plain_len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(plain_len);
    lzfse_rust::decode_bytes(src, &mut out)
        .map_err(|e| crate::Error::InvalidImage(format!("dmg: lzfse chunk decode failed: {e}")))?;
    if out.len() != plain_len {
        return Err(crate::Error::InvalidImage(format!(
            "dmg: lzfse chunk inflated to {} bytes but sector_count*512 = {}",
            out.len(),
            plain_len
        )));
    }
    Ok(out)
}

/// Stub when the LZFSE dep was compiled out.
#[cfg(not(feature = "dmg-lzfse"))]
pub fn decode_lzfse(_src: &[u8], _plain_len: usize) -> Result<Vec<u8>> {
    Err(crate::Error::Unsupported(
        "dmg: LZFSE chunks require the `dmg-lzfse` Cargo feature".into(),
    ))
}

/// Decode an LZMA chunk payload into a vector of exactly `plain_len`
/// bytes. DMG uses the raw LZMA1 frame (the legacy `.lzma` shape with
/// a 13-byte header carrying properties + dictionary size +
/// uncompressed length), not the XZ container.
#[cfg(feature = "lzma")]
pub fn decode_lzma(src: &[u8], plain_len: usize) -> Result<Vec<u8>> {
    let mut input = std::io::BufReader::new(src);
    let mut out = Vec::with_capacity(plain_len);
    lzma_rs::lzma_decompress(&mut input, &mut out)
        .map_err(|e| crate::Error::InvalidImage(format!("dmg: lzma chunk decode failed: {e}")))?;
    if out.len() != plain_len {
        return Err(crate::Error::InvalidImage(format!(
            "dmg: lzma chunk inflated to {} bytes but sector_count*512 = {}",
            out.len(),
            plain_len
        )));
    }
    Ok(out)
}

/// Stub when the LZMA dep was compiled out.
#[cfg(not(feature = "lzma"))]
pub fn decode_lzma(_src: &[u8], _plain_len: usize) -> Result<Vec<u8>> {
    Err(crate::Error::Unsupported(
        "dmg: LZMA chunks require the `lzma` Cargo feature".into(),
    ))
}

/// Decode an ADC chunk into a vector of exactly `plain_len` bytes.
/// Wraps [`decode_adc`] so the call site mirrors the other codec
/// helpers.
pub fn decode_adc_chunk(src: &[u8], plain_len: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; plain_len];
    let n = decode_adc(src, &mut out)?;
    if n != plain_len {
        return Err(crate::Error::InvalidImage(format!(
            "dmg: ADC chunk produced {n} bytes but sector_count*512 = {plain_len}"
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode `data` as a single ADC literal run. Caller must keep
    /// `data.len() <= 128`. Used by the tests to build minimal vectors.
    fn adc_literal(data: &[u8]) -> Vec<u8> {
        assert!(!data.is_empty() && data.len() <= 128);
        let mut v = Vec::with_capacity(1 + data.len());
        v.push((data.len() - 1) as u8);
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn adc_literal_only() {
        let payload = b"hello world";
        let stream = adc_literal(payload);
        let mut out = vec![0u8; payload.len()];
        let n = decode_adc(&stream, &mut out).unwrap();
        assert_eq!(n, payload.len());
        assert_eq!(&out[..], payload);
    }

    #[test]
    fn adc_long_literal_run() {
        // 128-byte literal run, the maximum a single opcode can carry.
        let payload: Vec<u8> = (0..128u8).collect();
        let stream = adc_literal(&payload);
        let mut out = vec![0u8; 128];
        let n = decode_adc(&stream, &mut out).unwrap();
        assert_eq!(n, 128);
        assert_eq!(out, payload);
    }

    #[test]
    fn adc_short_reference() {
        // First emit "ABCD" as a literal, then a short reference of
        // length 4 / distance 4 — should reproduce "ABCD" again.
        // op = 0x80 | ((len-3) << 2) | ((dist-1) >> 8)
        // len = 4 → (len-3) = 1
        // dist = 4 → (dist-1) = 3 → hi 0, lo 3
        // op = 0x80 | (1 << 2) | 0 = 0x84, trailing byte = 3.
        let mut stream = adc_literal(b"ABCD");
        stream.push(0x84);
        stream.push(0x03);
        let mut out = vec![0u8; 8];
        let n = decode_adc(&stream, &mut out).unwrap();
        assert_eq!(n, 8);
        assert_eq!(&out[..], b"ABCDABCD");
    }

    #[test]
    fn adc_short_reference_overlap_run() {
        // Encode "a" then back-reference dist=1 len=3 — should give
        // "aaaa". op = 0x80 | ((3-3)<<2) | 0 = 0x80, trailing = 0.
        let mut stream = adc_literal(b"a");
        stream.push(0x80);
        stream.push(0x00);
        let mut out = vec![0u8; 4];
        let n = decode_adc(&stream, &mut out).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&out[..], b"aaaa");
    }

    #[test]
    fn adc_long_reference() {
        // Build a 200-byte literal preamble of 0xAA, then a long
        // reference len=67 dist=200 reproducing the first 67 bytes.
        let mut stream = Vec::new();
        // Two 100-byte literal runs (literal opcode max length 128).
        stream.push(99u8); // run of 100 bytes
        stream.extend(std::iter::repeat_n(0xAA, 100));
        stream.push(99u8);
        stream.extend(std::iter::repeat_n(0xBB, 100));

        // Long reference: op = 0xC0 | (len-4) = 0xC0 | 63 = 0xFF.
        // dist - 1 = 199 → 0x00C7.
        stream.push(0xC0 | 63); // len = 67
        stream.push(0x00);
        stream.push(0xC7);

        let mut out = vec![0u8; 267];
        let n = decode_adc(&stream, &mut out).unwrap();
        assert_eq!(n, 267);
        // First 100 bytes 0xAA, next 100 bytes 0xBB, last 67 bytes
        // are the AAs from distance-200 = the start of the buffer.
        assert!(out[0..100].iter().all(|&b| b == 0xAA));
        assert!(out[100..200].iter().all(|&b| b == 0xBB));
        assert!(out[200..267].iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn adc_rejects_distance_zero() {
        // Short reference with zero in the trailing byte AND zero in
        // the opcode low bits ⇒ distance = 0 + 1 = 1, fine. So we have
        // to construct an actual zero-distance case. That requires
        // dp = 0 and any reference — pick a short ref at start of
        // stream which must fail since there's nothing to reference.
        let stream = vec![0x80, 0x00];
        let mut out = vec![0u8; 4];
        let err = decode_adc(&stream, &mut out).unwrap_err();
        match err {
            crate::Error::InvalidImage(_) => {}
            _ => panic!("expected InvalidImage, got {err:?}"),
        }
    }

    #[test]
    fn adc_rejects_truncated_short_reference() {
        // Opcode 0x80 marks a short reference but the trailing byte is
        // missing.
        let stream = vec![0x80];
        let mut out = vec![0u8; 4];
        assert!(decode_adc(&stream, &mut out).is_err());
    }

    #[test]
    fn adc_rejects_truncated_long_reference() {
        // Long reference opcode with only one trailing byte.
        let stream = vec![0xC0, 0x00];
        let mut out = vec![0u8; 4];
        assert!(decode_adc(&stream, &mut out).is_err());
    }

    #[test]
    fn adc_rejects_literal_overrun() {
        // Literal opcode promises 5 bytes but only 2 are present.
        let stream = vec![0x04, 0x01, 0x02];
        let mut out = vec![0u8; 8];
        assert!(decode_adc(&stream, &mut out).is_err());
    }

    #[cfg(feature = "dmg-bzip2")]
    #[test]
    fn bzip2_roundtrip_via_libbz2_independent_vector() {
        // Pre-computed bzip2 stream of the ASCII text "hello world\n"
        // produced by the bzip2 reference CLI. Inline so the test
        // doesn't depend on a write-side encoder.
        let compressed: &[u8] = &[
            0x42, 0x5A, 0x68, 0x39, 0x31, 0x41, 0x59, 0x26, 0x53, 0x59, 0x4E, 0xEC, 0xE8, 0x36,
            0x00, 0x00, 0x02, 0x51, 0x80, 0x00, 0x10, 0x40, 0x00, 0x06, 0x44, 0x90, 0x80, 0x20,
            0x00, 0x31, 0x06, 0x4C, 0x41, 0x01, 0xA7, 0xA9, 0xA5, 0x80, 0xBB, 0x94, 0x31, 0xF8,
            0xBB, 0x92, 0x29, 0xC2, 0x84, 0x82, 0x77, 0x67, 0x41, 0xB0,
        ];
        let plain = decode_bzip2(compressed, b"hello world\n".len()).unwrap();
        assert_eq!(plain, b"hello world\n");
    }

    #[cfg(feature = "lzma")]
    #[test]
    fn lzma_roundtrip_via_lzma_rs() {
        // Build a raw-LZMA1 stream on the fly with lzma-rs (the same
        // crate the decoder uses) and check we round-trip through
        // decode_lzma.
        let plain = b"the quick brown fox jumps over the lazy dog".repeat(8);
        let mut compressed = Vec::new();
        {
            let mut input = std::io::Cursor::new(&plain);
            lzma_rs::lzma_compress(&mut input, &mut compressed).unwrap();
        }
        let out = decode_lzma(&compressed, plain.len()).unwrap();
        assert_eq!(out, plain);
    }
}
