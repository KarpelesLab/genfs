//! Fletcher-64 checksum used in APFS object headers (`obj_phys_t.o_cksum`).
//!
//! APFS uses a slightly unusual variant of Fletcher-64. For every on-disk
//! object the first 8 bytes are the stored checksum; computing or
//! verifying it proceeds in two passes over the data:
//!
//! 1. Run Fletcher-64 over the *payload* (everything after the 8-byte
//!    `o_cksum` field), accumulating `(sum1, sum2)` modulo 2^32 - 1.
//! 2. Run Fletcher-64 over the 8-byte `o_cksum` field, continuing from
//!    the running state of step 1. On a valid block, both running sums
//!    end at exactly zero.
//!
//! The stored cksum is chosen so that step 2 drives the sums to zero —
//! see the derivation below.
//!
//! Reference: *Apple File System Reference*, section "Object Identifiers
//! and Checksums". Algorithmic description is also derivable from the
//! formula directly; only field names are taken from the spec.
//!
//! ```text
//! Compute cksum for an object whose payload is data[8..]:
//!   sum1, sum2 = fletcher64(data[8..])         # modulo MOD
//!   c1 = MOD - ((sum1 + sum2) mod MOD)
//!   c2 = MOD - ((sum1 + c1)   mod MOD)
//!   store (c1 || c2) little-endian into data[0..8]
//! ```

const MOD: u64 = 0xFFFF_FFFF; // 2^32 - 1

/// Run the raw Fletcher-64 accumulator over `data`, returning
/// `(sum1, sum2)` *modulo* `MOD`. `data.len()` MUST be a multiple of 4.
fn accumulate(data: &[u8], mut sum1: u64, mut sum2: u64) -> (u64, u64) {
    let n = data.len() / 4;
    for i in 0..n {
        let off = i * 4;
        let w = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as u64;
        sum1 = (sum1 + w) % MOD;
        sum2 = (sum2 + sum1) % MOD;
    }
    (sum1, sum2)
}

/// Compute the APFS Fletcher-64 checksum for an object whose payload
/// (everything after the 8-byte `o_cksum` field) is `payload`. The
/// returned `u64` is what gets stored at `data[0..8]` little-endian
/// (low 32 = c1, high 32 = c2).
///
/// `payload.len()` must be a multiple of 4. APFS objects are always
/// block-aligned, so this is satisfied in practice.
pub fn fletcher64_payload(payload: &[u8]) -> u64 {
    let (sum1, sum2) = accumulate(payload, 0, 0);
    let c1 = MOD - ((sum1 + sum2) % MOD);
    let c2 = MOD - ((sum1 + c1) % MOD);
    (c2 << 32) | c1
}

/// Compute the APFS Fletcher-64 checksum for a full block whose first 8
/// bytes are the (about-to-be-written) cksum field. The contents of the
/// first 8 bytes are ignored; only `block[8..]` is fed through the
/// accumulator. Equivalent to [`fletcher64_payload`] called on
/// `block[8..]`.
pub fn fletcher64(block: &[u8]) -> u64 {
    if block.len() < 8 {
        // Edge case — match the empty-payload behaviour.
        return fletcher64_payload(&[]);
    }
    fletcher64_payload(&block[8..])
}

/// Verify the Fletcher-64 stored in the first 8 bytes of `block`. Runs
/// the two-pass procedure: payload first, then the cksum bytes; on a
/// valid block both running sums are zero at the end.
pub fn verify(block: &[u8]) -> bool {
    if block.len() < 8 || !block.len().is_multiple_of(4) {
        return false;
    }
    // Reject sentinel cksum values (matches the reference implementation;
    // both all-zero and all-ones are treated as "not yet checksummed").
    let cksum = u64::from_le_bytes(block[0..8].try_into().unwrap());
    if cksum == 0 || cksum == u64::MAX {
        return false;
    }
    let (sum1, sum2) = accumulate(&block[8..], 0, 0);
    let (sum1, sum2) = accumulate(&block[..8], sum1, sum2);
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
        // Empty payload: sum1 = sum2 = 0 ⇒ c1 = MOD, c2 = MOD ⇒ result =
        // u64::MAX. (`verify` then rejects this as a sentinel.)
        assert_eq!(fletcher64_payload(&[]), u64::MAX);
        assert!(!verify(&[]));
        assert!(!verify(&[0u8; 7]));
        // Block of exactly 8 bytes with cksum = u64::MAX: verify rejects
        // sentinel cksum, even though it would otherwise be a valid
        // 0-byte-payload block.
        let mut b = [0u8; 8];
        b.copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(!verify(&b));
    }
}
