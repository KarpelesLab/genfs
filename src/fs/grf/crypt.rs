//! Fixed-key permutation cipher used by GRF v0x102 / v0x103.
//!
//! Despite the libgrf naming (`decode_des_etc`, `GRF_FLAG_DES`), this
//! is NOT real DES — it's a custom permutation network using fixed
//! lookup tables, no secret key. The transformation is asymmetric:
//! `decode_des_etc` only goes one way (decrypt), and libgrf never
//! writes new files encrypted with it — on repack the writer clears
//! the `MIXCRYPT` / `DES` flags and stores plain `GRF_FLAG_FILE`
//! entries. We mirror that: only the decode direction exists here.
//!
//! Tables and algorithm ported verbatim from `libgrf/src/grf.c`
//! lines 40–204 (used with the original author's permission).
//!
//! Two distinct decoders sit on top of the bit primitives:
//!
//! - [`decode_filename`] — used on v0x102/0x103 file-table entries
//!   where each filename byte buffer is encrypted in 8-byte chunks
//!   with a fixed three-pass transform (nibble swap → BitConvert1
//!   → BitConvert4 → BitConvert2).
//! - [`decode_des_etc`] — used on file *bodies* carrying the
//!   `GRF_FLAG_MIXCRYPT` or `GRF_FLAG_DES` flag. The per-block rule
//!   depends on `flag_type` and a `cycle` counter derived from the
//!   file's compressed length.

const BIT_MASK_TABLE: [u8; 8] = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];

const BIT_SWAP_TABLE_1: [u8; 64] = [
    58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4, 62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8, 57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35, 27, 19, 11, 3, 61,
    53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7,
];

const BIT_SWAP_TABLE_2: [u8; 64] = [
    40, 8, 48, 16, 56, 24, 64, 32, 39, 7, 47, 15, 55, 23, 63, 31, 38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29, 36, 4, 44, 12, 52, 20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26, 33, 1, 41, 9, 49, 17, 57, 25,
];

const BIT_SWAP_TABLE_3: [u8; 32] = [
    16, 7, 20, 21, 29, 12, 28, 17, 1, 15, 23, 26, 5, 18, 31, 10, 2, 8, 24, 14, 32, 27, 3, 9, 19,
    13, 30, 6, 22, 11, 4, 25,
];

const NIBBLE_DATA: [[u8; 64]; 4] = [
    [
        0xef, 0x03, 0x41, 0xfd, 0xd8, 0x74, 0x1e, 0x47, 0x26, 0xef, 0xfb, 0x22, 0xb3, 0xd8, 0x84,
        0x1e, 0x39, 0xac, 0xa7, 0x60, 0x62, 0xc1, 0xcd, 0xba, 0x5c, 0x96, 0x90, 0x59, 0x05, 0x3b,
        0x7a, 0x85, 0x40, 0xfd, 0x1e, 0xc8, 0xe7, 0x8a, 0x8b, 0x21, 0xda, 0x43, 0x64, 0x9f, 0x2d,
        0x14, 0xb1, 0x72, 0xf5, 0x5b, 0xc8, 0xb6, 0x9c, 0x37, 0x76, 0xec, 0x39, 0xa0, 0xa3, 0x05,
        0x52, 0x6e, 0x0f, 0xd9,
    ],
    [
        0xa7, 0xdd, 0x0d, 0x78, 0x9e, 0x0b, 0xe3, 0x95, 0x60, 0x36, 0x36, 0x4f, 0xf9, 0x60, 0x5a,
        0xa3, 0x11, 0x24, 0xd2, 0x87, 0xc8, 0x52, 0x75, 0xec, 0xbb, 0xc1, 0x4c, 0xba, 0x24, 0xfe,
        0x8f, 0x19, 0xda, 0x13, 0x66, 0xaf, 0x49, 0xd0, 0x90, 0x06, 0x8c, 0x6a, 0xfb, 0x91, 0x37,
        0x8d, 0x0d, 0x78, 0xbf, 0x49, 0x11, 0xf4, 0x23, 0xe5, 0xce, 0x3b, 0x55, 0xbc, 0xa2, 0x57,
        0xe8, 0x22, 0x74, 0xce,
    ],
    [
        0x2c, 0xea, 0xc1, 0xbf, 0x4a, 0x24, 0x1f, 0xc2, 0x79, 0x47, 0xa2, 0x7c, 0xb6, 0xd9, 0x68,
        0x15, 0x80, 0x56, 0x5d, 0x01, 0x33, 0xfd, 0xf4, 0xae, 0xde, 0x30, 0x07, 0x9b, 0xe5, 0x83,
        0x9b, 0x68, 0x49, 0xb4, 0x2e, 0x83, 0x1f, 0xc2, 0xb5, 0x7c, 0xa2, 0x19, 0xd8, 0xe5, 0x7c,
        0x2f, 0x83, 0xda, 0xf7, 0x6b, 0x90, 0xfe, 0xc4, 0x01, 0x5a, 0x97, 0x61, 0xa6, 0x3d, 0x40,
        0x0b, 0x58, 0xe6, 0x3d,
    ],
    [
        0x4d, 0xd1, 0xb2, 0x0f, 0x28, 0xbd, 0xe4, 0x78, 0xf6, 0x4a, 0x0f, 0x93, 0x8b, 0x17, 0xd1,
        0xa4, 0x3a, 0xec, 0xc9, 0x35, 0x93, 0x56, 0x7e, 0xcb, 0x55, 0x20, 0xa0, 0xfe, 0x6c, 0x89,
        0x17, 0x62, 0x17, 0x62, 0x4b, 0xb1, 0xb4, 0xde, 0xd1, 0x87, 0xc9, 0x14, 0x3c, 0x4a, 0x7e,
        0xa8, 0xe2, 0x7d, 0xa0, 0x9f, 0xf6, 0x5c, 0x6a, 0x09, 0x8d, 0xf0, 0x0f, 0xe3, 0x53, 0x25,
        0x95, 0x36, 0x28, 0xcb,
    ],
];

/// Swap the high and low nibble of every byte in `buf`. Used as the
/// first pass of the filename transform.
fn nibble_swap(buf: &mut [u8]) {
    for b in buf.iter_mut() {
        *b = (*b).rotate_left(4);
    }
}

/// 64-bit permutation through `table`. The C version treated the
/// 8-byte block as two `DWORD`s and wrote them back with a single
/// store; we just zero the output buffer and OR bits in. Same shape.
fn bit_convert(src: &mut [u8; 8], table: &[u8; 64]) {
    let mut tmp = [0u8; 8];
    for lop in 0..64 {
        let prm = (table[lop] - 1) as usize;
        if src[(prm >> 3) & 7] & BIT_MASK_TABLE[prm & 7] != 0 {
            tmp[(lop >> 3) & 7] |= BIT_MASK_TABLE[lop & 7];
        }
    }
    *src = tmp;
}

/// Second pass: S-box substitution (via [`NIBBLE_DATA`]) followed by
/// a 32-bit permutation through [`BIT_SWAP_TABLE_3`], then XOR'd
/// back into the high half of `src`.
fn bit_convert4(src: &mut [u8; 8]) {
    let mut tmp = [0u8; 8];

    // First 8 bytes of `tmp` carry 6-bit S-box indices packed out of
    // the high nibble of `src[4..8]`. Bit layout follows the
    // original C code line-for-line.
    tmp[0] = ((src[7] << 5) | (src[4] >> 3)) & 0x3f;
    tmp[1] = ((src[4] << 1) | (src[5] >> 7)) & 0x3f;
    tmp[2] = ((src[4] << 5) | (src[5] >> 3)) & 0x3f;
    tmp[3] = ((src[5] << 1) | (src[6] >> 7)) & 0x3f;
    tmp[4] = ((src[5] << 5) | (src[6] >> 3)) & 0x3f;
    tmp[5] = ((src[6] << 1) | (src[7] >> 7)) & 0x3f;
    tmp[6] = ((src[6] << 5) | (src[7] >> 3)) & 0x3f;
    tmp[7] = ((src[7] << 1) | (src[4] >> 7)) & 0x3f;

    // S-box: take high nibble from box[i][tmp[2i]] and low nibble
    // from box[i][tmp[2i+1]]. Result overwrites tmp[0..4].
    for lop in 0..4 {
        tmp[lop] = (NIBBLE_DATA[lop][tmp[lop * 2] as usize] & 0xf0)
            | (NIBBLE_DATA[lop][tmp[lop * 2 + 1] as usize] & 0x0f);
    }

    // Zero the high half, then permute the 32 result bits through
    // table 3 into it.
    tmp[4] = 0;
    tmp[5] = 0;
    tmp[6] = 0;
    tmp[7] = 0;
    for lop in 0..32 {
        let prm = (BIT_SWAP_TABLE_3[lop] - 1) as usize;
        if tmp[prm >> 3] & BIT_MASK_TABLE[prm & 7] != 0 {
            tmp[(lop >> 3) + 4] |= BIT_MASK_TABLE[lop & 7];
        }
    }

    // XOR the permuted high half back into src[0..4]. The C code
    // does this as a 32-bit XOR; we do it byte-by-byte.
    for i in 0..4 {
        src[i] ^= tmp[i + 4];
    }
}

/// Decode v0x102/0x103 table filenames in-place. The buffer is
/// processed in 8-byte chunks; bytes past the last full chunk are
/// left untouched (the caller pads/truncates as appropriate).
pub fn decode_filename(buf: &mut [u8]) {
    let mut off = 0;
    while off + 8 <= buf.len() {
        let chunk: &mut [u8; 8] = (&mut buf[off..off + 8]).try_into().unwrap();
        nibble_swap(chunk);
        bit_convert(chunk, &BIT_SWAP_TABLE_1);
        bit_convert4(chunk);
        bit_convert(chunk, &BIT_SWAP_TABLE_2);
        off += 8;
    }
}

/// Decode a file body encrypted with the per-entry `MIXCRYPT` /
/// `DES` flags. `flag_type` is 0 for MIXCRYPT, 1 for DES; `cycle`
/// is the cycle counter derived by the table parser from
/// `compressed_len`. The first 20 blocks always get the full
/// 3-pass transform; later blocks are gated on the cycle / type
/// combination, with a sneaky byte-swap-and-substitute fallback
/// for the MIXCRYPT path.
pub fn decode_des_etc(buf: &mut [u8], flag_type: u8, mut cycle: i32) {
    // Cycle is bumped up to keep the decryption sparser as files
    // grow. Values < 3 are clamped; the rest matches the curve in
    // grf.c line 145-148.
    cycle = if cycle < 3 {
        3
    } else if cycle < 5 {
        cycle + 1
    } else if cycle < 7 {
        cycle + 9
    } else {
        cycle + 15
    };

    let mut cnt = 0i32;
    let mut lop = 0i32;
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let chunk: &mut [u8; 8] = (&mut buf[off..off + 8]).try_into().unwrap();

        if lop < 20 || (flag_type == 0 && lop % cycle == 0) {
            bit_convert(chunk, &BIT_SWAP_TABLE_1);
            bit_convert4(chunk);
            bit_convert(chunk, &BIT_SWAP_TABLE_2);
        } else if cnt == 7 && flag_type == 0 {
            let tmp = *chunk;
            cnt = 0;
            chunk[0] = tmp[3];
            chunk[1] = tmp[4];
            chunk[2] = tmp[6];
            chunk[3] = tmp[0];
            chunk[4] = tmp[1];
            chunk[5] = tmp[2];
            chunk[6] = tmp[5];
            // Byte 7 goes through a small self-inverse substitution.
            let a = tmp[7];
            chunk[7] = match a {
                0x00 => 0x2b,
                0x2b => 0x00,
                0x01 => 0x68,
                0x68 => 0x01,
                0x48 => 0x77,
                0x77 => 0x48,
                0x60 => 0xff,
                0xff => 0x60,
                0x6c => 0x80,
                0x80 => 0x6c,
                0xb9 => 0xc0,
                0xc0 => 0xb9,
                0xeb => 0xfe,
                0xfe => 0xeb,
                other => other,
            };
        } else {
            cnt += 1;
        }

        lop += 1;
        off += 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nibble_swap_is_self_inverse() {
        let mut buf = *b"abcdefgh";
        nibble_swap(&mut buf);
        nibble_swap(&mut buf);
        assert_eq!(&buf, b"abcdefgh");
    }

    /// `bit_convert` followed by its inverse should return the
    /// input. We don't have a separate inverse table on hand, so
    /// this test just confirms the function is deterministic by
    /// running it twice and checking the second pass equals the
    /// first.
    #[test]
    fn bit_convert_is_deterministic() {
        let mut a = [0xde, 0xad, 0xbe, 0xef, 0xfe, 0xed, 0xfa, 0xce];
        let mut b = a;
        bit_convert(&mut a, &BIT_SWAP_TABLE_1);
        bit_convert(&mut b, &BIT_SWAP_TABLE_1);
        assert_eq!(a, b);
    }

    /// The full filename transform applied twice to a hand-checked
    /// 8-byte buffer is reversible only with a separate inverse —
    /// here we just confirm the transform doesn't blow up and
    /// produces a deterministic non-identity output.
    #[test]
    fn decode_filename_changes_input() {
        let mut a = *b"data.txt";
        let original = a;
        decode_filename(&mut a);
        assert_ne!(a, original, "transform should not be identity");
        let mut b = original;
        decode_filename(&mut b);
        assert_eq!(a, b, "transform should be deterministic");
    }

    #[test]
    fn decode_des_etc_runs_on_short_buffer() {
        // Smaller than 8 bytes — should leave the buffer alone.
        let mut buf = *b"short";
        let original = buf;
        decode_des_etc(&mut buf, 0, 1);
        assert_eq!(buf, original);
    }

    #[test]
    fn decode_des_etc_first_block_uses_full_transform() {
        // One block, cycle=1, flag_type=0. lop=0 enters the DES
        // branch (lop < 20), so the output should differ from input.
        let mut buf = [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11];
        let original = buf;
        decode_des_etc(&mut buf, 0, 1);
        assert_ne!(buf, original);
    }
}
