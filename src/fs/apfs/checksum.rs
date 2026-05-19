//! Fletcher-64 checksum used in APFS object headers (`obj_phys_t.o_cksum`).
//!
//! APFS computes a modified Fletcher-64 over each on-disk object. The first
//! eight bytes of every object are the stored checksum itself; the algorithm
//! treats those eight bytes as zero while running, and the final checksum is
//! adjusted so that re-running the algorithm over the *whole* block —
//! including the stored cksum — leaves both running sums at zero. This makes
//! verification a single pass over the bytes.
//!
//! Reference: *Apple File System Reference*, section "Object Identifiers and
//! Checksums".
//!
//! ```text
//! sum1, sum2 : u32 mod (2^32 - 1)
//! for each 4-byte LE word w in data:
//!     sum1 = (sum1 + w)      mod (2^32 - 1)
//!     sum2 = (sum2 + sum1)   mod (2^32 - 1)
//! c1 = (2^32 - 1) - ((sum1 + sum2) mod (2^32 - 1))
//! c2 = (2^32 - 1) - ((sum1 + c1)   mod (2^32 - 1))
//! cksum = (c2 << 32) | c1
//! ```

const MOD: u64 = 0xFFFF_FFFF; // 2^32 - 1

/// Compute the APFS Fletcher-64 checksum over `data`. `data.len()` must be a
/// multiple of 4 (every APFS object is block-aligned and a multiple of 4
/// bytes, so this is always true in practice). Bytes beyond the last
/// 4-byte boundary are silently ignored.
///
/// The returned 64-bit value is what would be stored in
/// `obj_phys_t.o_cksum` for `data` whose first 8 bytes are zero. Pass the
/// whole object with `o_cksum` zeroed to compute the value to store; pass
/// the whole object with `o_cksum` intact to verify (the result must equal
/// zero in both sums — see [`verify`]).
pub fn fletcher64(data: &[u8]) -> u64 {
    let mut sum1: u64 = 0;
    let mut sum2: u64 = 0;
    let n = data.len() / 4;
    for i in 0..n {
        let off = i * 4;
        let w = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            as u64;
        sum1 = (sum1 + w) % MOD;
        sum2 = (sum2 + sum1) % MOD;
    }
    let c1 = MOD - ((sum1 + sum2) % MOD);
    let c2 = MOD - ((sum1 + c1) % MOD);
    (c2 << 32) | c1
}

/// Verify the Fletcher-64 stored in the first 8 bytes of `block`. Returns
/// `true` when the block's checksum is self-consistent. The check works by
/// running Fletcher-64 over the entire block (cksum included): both running
/// sums end at zero on a valid block.
pub fn verify(block: &[u8]) -> bool {
    if block.len() < 8 || block.len() % 4 != 0 {
        return false;
    }
    let mut sum1: u64 = 0;
    let mut sum2: u64 = 0;
    let n = block.len() / 4;
    for i in 0..n {
        let off = i * 4;
        let w = u32::from_le_bytes([
            block[off],
            block[off + 1],
            block[off + 2],
            block[off + 3],
        ]) as u64;
        sum1 = (sum1 + w) % MOD;
        sum2 = (sum2 + sum1) % MOD;
    }
    sum1 == 0 && sum2 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A round-trip property: zeroing the first 8 bytes of a buffer,
    /// computing fletcher64, writing the result back, then [`verify`]ing
    /// over the whole buffer must succeed.
    #[test]
    fn roundtrip_zero_payload() {
        let mut block = [0u8; 64];
        // Fill payload with a recognisable pattern beyond the cksum slot.
        for (i, b) in block.iter_mut().enumerate().skip(8) {
            *b = (i as u8).wrapping_mul(7);
        }
        let cksum = fletcher64(&block);
        block[..8].copy_from_slice(&cksum.to_le_bytes());
        assert!(verify(&block));
    }

    #[test]
    fn flipping_a_byte_breaks_verification() {
        let mut block = [0u8; 64];
        for (i, b) in block.iter_mut().enumerate().skip(8) {
            *b = i as u8;
        }
        let cksum = fletcher64(&block);
        block[..8].copy_from_slice(&cksum.to_le_bytes());
        assert!(verify(&block));
        block[20] ^= 0x01;
        assert!(!verify(&block));
    }

    /// Self-consistency: the computed cksum, when stored back at offset 0,
    /// yields a block on which both running sums are zero (the defining
    /// property of the algorithm).
    #[test]
    fn known_pattern_is_self_consistent() {
        // 16 bytes (4 words), first 8 reserved for cksum.
        let mut block = [0u8; 16];
        block[8..12].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        block[12..16].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        let cksum = fletcher64(&block);
        block[..8].copy_from_slice(&cksum.to_le_bytes());
        assert!(verify(&block));
    }

    #[test]
    fn empty_and_short_inputs() {
        // Empty input: sum1 = sum2 = 0, so c1 = MOD, c2 = 0, result = MOD.
        assert_eq!(fletcher64(&[]), MOD);
        // Verify rejects bad sizes.
        assert!(!verify(&[]));
        assert!(!verify(&[0u8; 7]));
    }
}
