//! LZNT1 decompression for NTFS compressed `$DATA` streams.
//!
//! NTFS divides a compressed non-resident attribute into "compression units"
//! (CU). The CU size in clusters is `1 << compression_unit` (the value of the
//! non-resident header field at offset 0x22). For the canonical 4 KiB cluster
//! / `compression_unit == 4` case this is 16 clusters == 64 KiB raw.
//!
//! Each CU is stored on disk as one of three shapes, detectable by run-list
//! shape rather than by an in-band marker:
//!
//! * **All-zero (sparse)** — the run for this CU is sparse (no LCN). The CU
//!   yields 64 KiB of zeroes.
//! * **Uncompressed** — the run carries the full CU verbatim (16 clusters of
//!   real data, no LZNT1 framing).
//! * **Compressed** — the run carries fewer than 16 clusters of LZNT1-encoded
//!   chunks; the remainder of the CU's VCN range is sparse (NTFS leaves the
//!   tail "deallocated"). The decoder produces the full CU size of output.
//!
//! The on-disk LZNT1 frame is a sequence of "chunks", each starting with a
//! 16-bit little-endian header:
//!
//! ```text
//!     bit 15:    1 = compressed chunk, 0 = literal copy of next length bytes
//!     bits 14..12: signature, always 0b011
//!     bits 11..0:  chunk length minus 1 (so payload is `len + 1` bytes)
//! ```
//!
//! A header value of 0x0000 marks the end of stream. Compressed chunks
//! contain a stream of 8-token groups: one flag byte (LSB first) followed by
//! 8 tokens. A 0 flag bit means "copy 1 literal byte". A 1 flag bit means
//! "back-reference"; the next 16 little-endian bits split into a length and
//! offset whose bit-widths depend on how many bytes have been emitted so far
//! within the chunk (Microsoft's "sliding bit allocator"):
//!
//! ```text
//!     U = ceil(log2(emitted_bytes_in_chunk))    // saturated to [4, 12]
//!     length bits  = 16 - U     length minimum = 3
//!     offset bits  = U          (high bits of the 16-bit word)
//! ```
//!
//! Specifically, given the 16-bit token `t`:
//!
//! ```text
//!     length_mask  = (1 << (16 - U)) - 1
//!     length       = (t & length_mask) + 3
//!     offset       = (t >> (16 - U)) + 1
//! ```
//!
//! The back-reference copies `length` bytes starting `offset` bytes before
//! the current write cursor. Self-overlap is legal (a length > offset run
//! produces RLE-like repetition).
//!
//! Reference: Microsoft "[MS-XCA]" §2.5 ("LZNT1 Algorithm Details") and the
//! Russon & Fledel "NTFS Documentation" community PDF.

/// Decompress one or more LZNT1 chunks from `src` into `dst`.
///
/// The decoder stops when:
///   * `dst` is full (`emitted == dst.len()`), or
///   * the source ends (no more chunks), or
///   * a chunk header equal to 0 is seen (explicit end-of-stream marker).
///
/// Any bytes left in `dst` past the last decoded byte are filled with zero so
/// the caller always sees a CU-sized buffer with valid contents at every
/// offset.
///
/// Returns the number of bytes that came from real chunks (i.e. emitted
/// before the zero-fill tail). Errors are returned for malformed framing
/// (truncated chunks, back-reference past the start of the chunk, etc.).
pub fn decompress_unit(src: &[u8], dst: &mut [u8]) -> crate::Result<usize> {
    let mut src_cursor = 0usize;
    let mut dst_cursor = 0usize;

    while src_cursor + 2 <= src.len() && dst_cursor < dst.len() {
        let header = u16::from_le_bytes([src[src_cursor], src[src_cursor + 1]]);
        if header == 0 {
            // Explicit end-of-stream — caller may pass us a longer buffer
            // (e.g. cluster-padded) and rely on the zero terminator to
            // stop us; we don't need to advance the cursor any further.
            break;
        }
        src_cursor += 2;
        let is_compressed = header & 0x8000 != 0;
        let chunk_len = (header & 0x0FFF) as usize + 1;
        // Signature bits 14..12 should be 0b011 but old encoders sometimes
        // emit other patterns. We don't enforce this — we just need to know
        // how many on-disk bytes the chunk occupies.

        if src_cursor + chunk_len > src.len() {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: LZNT1 chunk length {chunk_len} oversteps source ({} remaining)",
                src.len() - src_cursor
            )));
        }
        let chunk = &src[src_cursor..src_cursor + chunk_len];
        src_cursor += chunk_len;

        if !is_compressed {
            // Literal copy: chunk is raw output.
            let take = chunk_len.min(dst.len() - dst_cursor);
            dst[dst_cursor..dst_cursor + take].copy_from_slice(&chunk[..take]);
            dst_cursor += take;
            continue;
        }

        // Decompressed chunks are written into a per-chunk window of up to
        // 4 KiB. The bit allocator is based on the chunk-local emitted
        // count, NOT the global emitted count.
        let chunk_start_in_dst = dst_cursor;
        let mut chunk_in = 0usize;
        while chunk_in < chunk.len() && dst_cursor < dst.len() {
            let flags = chunk[chunk_in];
            chunk_in += 1;
            for bit in 0..8u8 {
                if chunk_in >= chunk.len() || dst_cursor >= dst.len() {
                    break;
                }
                let is_back_ref = flags & (1 << bit) != 0;
                if !is_back_ref {
                    // Single literal byte.
                    dst[dst_cursor] = chunk[chunk_in];
                    chunk_in += 1;
                    dst_cursor += 1;
                } else {
                    if chunk_in + 2 > chunk.len() {
                        return Err(crate::Error::InvalidImage(
                            "ntfs: LZNT1 back-ref token truncated".into(),
                        ));
                    }
                    let tok = u16::from_le_bytes([chunk[chunk_in], chunk[chunk_in + 1]]);
                    chunk_in += 2;

                    // Determine U from how many bytes we've written within
                    // this chunk so far.
                    let emitted_in_chunk = (dst_cursor - chunk_start_in_dst) as u32;
                    let u = bit_allocator_u(emitted_in_chunk);
                    let length_bits = 16 - u;
                    let length_mask = (1u16 << length_bits) - 1;
                    let length = (tok & length_mask) as usize + 3;
                    let offset = (tok >> length_bits) as usize + 1;

                    if offset > dst_cursor - chunk_start_in_dst {
                        return Err(crate::Error::InvalidImage(format!(
                            "ntfs: LZNT1 back-ref offset {offset} predates chunk start"
                        )));
                    }
                    let src_pos_start = dst_cursor - offset;
                    let take = length.min(dst.len() - dst_cursor);
                    // Self-overlapping copy (offset < length permitted).
                    for i in 0..take {
                        dst[dst_cursor + i] = dst[src_pos_start + i];
                    }
                    dst_cursor += take;
                }
            }
        }
    }

    // Zero-fill the rest of the CU (NTFS holds the tail "deallocated").
    let emitted = dst_cursor;
    for b in &mut dst[emitted..] {
        *b = 0;
    }
    Ok(emitted)
}

/// Microsoft's bit allocator. `emitted` is the number of bytes already
/// decompressed within the current chunk. The return value is the number of
/// high bits the 16-bit back-reference token devotes to the offset field;
/// the remaining `16 - U` bits encode (length − 3).
fn bit_allocator_u(emitted: u32) -> u32 {
    // U starts at 4 and grows by one each time the emitted size doubles past
    // 16. The classical formulation is:
    //   U = ceil(log2(emitted))
    //   clamped to the [4, 12] range
    // We compute that directly: how many bits are needed to represent the
    // largest possible offset (which is the current chunk-local position).
    let mut u = 4u32;
    let mut threshold = 1u32 << 4; // 16
    while emitted >= threshold && u < 12 {
        u += 1;
        threshold <<= 1;
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompresses_literal_chunk() {
        // header: not compressed, length-1 = 3 → payload of 4 bytes "ABCD".
        // Header value: length-1 == 0x0003 with bit 15 = 0.
        let mut src = vec![0x03, 0x30]; // 0x3003 — sig bits 0b011, len-1=3, bit15=0
        src.extend_from_slice(b"ABCD");
        // Followed by end-of-stream marker.
        src.extend_from_slice(&[0u8, 0u8]);
        let mut dst = vec![0u8; 16];
        let n = decompress_unit(&src, &mut dst).unwrap();
        assert_eq!(n, 4);
        assert_eq!(&dst[..4], b"ABCD");
        assert_eq!(dst[4..], [0u8; 12]);
    }

    #[test]
    fn decompresses_all_literal_compressed_chunk() {
        // A "compressed" chunk where the flag byte is 0 (all literals).
        // chunk payload: flags(0x00), b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H'
        // length-1 = 8, bit 15 = 1, signature 0b011.
        let chunk_payload = vec![0x00u8, b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H'];
        let chunk_len_minus_1 = chunk_payload.len() as u16 - 1;
        let header = 0xB000u16 | chunk_len_minus_1;
        let mut src = header.to_le_bytes().to_vec();
        src.extend_from_slice(&chunk_payload);
        src.extend_from_slice(&[0u8, 0u8]);
        let mut dst = vec![0u8; 16];
        let n = decompress_unit(&src, &mut dst).unwrap();
        assert_eq!(n, 8);
        assert_eq!(&dst[..8], b"ABCDEFGH");
    }

    #[test]
    fn decompresses_back_reference() {
        // We want to emit "ABCABC".
        // 'A','B','C' are literals (3 literal bytes, flag bits 0,1,2 = 0).
        // Then a back-ref token: emitted_in_chunk = 3, so U=4. offset_bits=4,
        // length_bits=12. length=3 → length-3 = 0. offset=3 → offset-1 = 2.
        // Token bits: (offset-1) in top 4 bits, length-3 in low 12 bits.
        // token = (2 << 12) | 0 = 0x2000.
        // Flags byte: bit3 set (the 4th token is the back-ref), others zero
        // → 0b0000_1000 = 0x08.
        // chunk payload: flag(0x08), b'A', b'B', b'C', tok_lo, tok_hi
        let chunk_payload = vec![0x08u8, b'A', b'B', b'C', 0x00, 0x20];
        let chunk_len_minus_1 = chunk_payload.len() as u16 - 1;
        let header = 0xB000u16 | chunk_len_minus_1;
        let mut src = header.to_le_bytes().to_vec();
        src.extend_from_slice(&chunk_payload);
        src.extend_from_slice(&[0u8, 0u8]);
        let mut dst = vec![0u8; 16];
        let n = decompress_unit(&src, &mut dst).unwrap();
        assert_eq!(n, 6);
        assert_eq!(&dst[..6], b"ABCABC");
    }

    #[test]
    fn back_reference_self_overlap() {
        // Emit a single 'X', then a back-ref with offset=1 length=5 → "XXXXXX".
        // After 1 emitted byte, U=4, length_bits=12, offset_bits=4.
        // token: (0 << 12) | (5 - 3) = 2; offset-1 = 0.
        // flag byte: bit1 set (1st literal then back-ref) = 0b10 = 0x02.
        // payload: flag(0x02), b'X', tok_lo(0x02), tok_hi(0x00)
        let chunk_payload = vec![0x02u8, b'X', 0x02, 0x00];
        let chunk_len_minus_1 = chunk_payload.len() as u16 - 1;
        let header = 0xB000u16 | chunk_len_minus_1;
        let mut src = header.to_le_bytes().to_vec();
        src.extend_from_slice(&chunk_payload);
        src.extend_from_slice(&[0u8, 0u8]);
        let mut dst = vec![0u8; 16];
        let n = decompress_unit(&src, &mut dst).unwrap();
        assert_eq!(n, 6);
        assert_eq!(&dst[..6], b"XXXXXX");
    }

    #[test]
    fn bit_allocator_clamps() {
        assert_eq!(bit_allocator_u(0), 4);
        assert_eq!(bit_allocator_u(15), 4);
        assert_eq!(bit_allocator_u(16), 5);
        assert_eq!(bit_allocator_u(31), 5);
        assert_eq!(bit_allocator_u(32), 6);
        assert_eq!(bit_allocator_u(4095), 12);
        assert_eq!(bit_allocator_u(8192), 12); // clamped
    }
}
