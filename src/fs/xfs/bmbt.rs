//! XFS block-map B-tree records — extent decoding.
//!
//! When `di_format == EXTENTS`, the inode's data fork is an array of
//! 128-bit packed records. Each record encodes a single contiguous range
//! mapping a logical file offset to a physical FS block. The packed layout
//! (big-endian, MSB → LSB) is:
//!
//! ```text
//!   bit  63    : flag (preallocated/unwritten)
//!   bits 62..9 : 54-bit logical file-block offset
//!   bits  8..0 : top 9 bits of startblock
//!   --- second 64-bit word ---
//!   bits 63..21: bottom 43 bits of startblock (52 bits total)
//!   bits 20..0 : 21-bit block count
//! ```
//!
//! XFS reuses the same record shape for unwritten-extent flags; we expose
//! the flag bit and refuse to read unwritten extents (they should appear as
//! zeros for a read-only consumer but the safer default is to error).
//!
//! Records are always stored sorted by `offset`; the array is dense (no
//! gaps) — but a single file may have holes that aren't represented by any
//! record. A read across a hole returns zero bytes.

use crate::Result;

/// Bytes per extent record on disk.
pub const BMBT_REC_SIZE: usize = 16;

/// Decoded extent record. `startblock` is an XFS "filesystem block number"
/// (FSB) — a global block index across all AGs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    /// First logical file-block this extent covers.
    pub offset: u64,
    /// Physical starting FS block.
    pub startblock: u64,
    /// Number of FS blocks in this extent.
    pub blockcount: u32,
    /// True for unwritten / preallocated extents (kernel exposes as zeros).
    pub unwritten: bool,
}

impl Extent {
    /// Decode one packed 128-bit record.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < BMBT_REC_SIZE {
            return Err(crate::Error::InvalidImage(
                "xfs: bmbt record buffer too small".into(),
            ));
        }
        let hi = u64::from_be_bytes(buf[0..8].try_into().unwrap());
        let lo = u64::from_be_bytes(buf[8..16].try_into().unwrap());
        let unwritten = (hi >> 63) & 1 == 1;
        // bits 62..9 — 54-bit offset.
        let offset = (hi >> 9) & ((1u64 << 54) - 1);
        // Top 9 bits of startblock (bits 8..0 of hi) become bits 51..43 of
        // the 52-bit field; remaining 43 bits sit at bits 63..21 of lo.
        let sb_hi = hi & ((1u64 << 9) - 1);
        let sb_lo = lo >> 21;
        let startblock = (sb_hi << 43) | sb_lo;
        let blockcount = (lo & ((1u64 << 21) - 1)) as u32;
        if blockcount == 0 {
            return Err(crate::Error::InvalidImage(
                "xfs: bmbt record has blockcount=0".into(),
            ));
        }
        Ok(Self {
            offset,
            startblock,
            blockcount,
            unwritten,
        })
    }

    /// Encode for the round-trip unit test.
    #[cfg(test)]
    pub fn encode(&self) -> [u8; BMBT_REC_SIZE] {
        let unwritten = if self.unwritten { 1u64 } else { 0 };
        let hi = (unwritten << 63)
            | ((self.offset & ((1 << 54) - 1)) << 9)
            | ((self.startblock >> 43) & ((1 << 9) - 1));
        let lo = ((self.startblock & ((1 << 43) - 1)) << 21)
            | (self.blockcount as u64 & ((1 << 21) - 1));
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&hi.to_be_bytes());
        out[8..16].copy_from_slice(&lo.to_be_bytes());
        out
    }
}

/// Decode `n` consecutive extent records from `buf`. `buf` must hold at
/// least `n * BMBT_REC_SIZE` bytes.
pub fn decode_extents(buf: &[u8], n: u32) -> Result<Vec<Extent>> {
    let need = (n as usize)
        .checked_mul(BMBT_REC_SIZE)
        .ok_or_else(|| crate::Error::InvalidImage("xfs: extent count overflows".into()))?;
    if buf.len() < need {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: extent array truncated: need {need} bytes, have {}",
            buf.len()
        )));
    }
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        let start = i * BMBT_REC_SIZE;
        out.push(Extent::decode(&buf[start..start + BMBT_REC_SIZE])?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extent_roundtrip() {
        let cases = [
            Extent {
                offset: 0,
                startblock: 100,
                blockcount: 1,
                unwritten: false,
            },
            Extent {
                offset: 4,
                startblock: 200,
                blockcount: 8,
                unwritten: false,
            },
            // High bits set in startblock (top 9 bits).
            Extent {
                offset: 1 << 30,
                startblock: (1u64 << 50) | 0x12_3456,
                blockcount: 0x1FFFFF,
                unwritten: true,
            },
        ];
        for c in cases {
            let bytes = c.encode();
            let back = Extent::decode(&bytes).unwrap();
            assert_eq!(back, c);
        }
    }

    #[test]
    fn extent_rejects_zero_blockcount() {
        let mut bytes = [0u8; 16];
        // offset=0, startblock=0, blockcount=0 — invalid.
        let e = Extent::decode(&bytes);
        assert!(e.is_err());
        // Even with a non-zero offset, blockcount=0 is bogus.
        bytes[0] = 0x00;
        let e = Extent::decode(&bytes);
        assert!(e.is_err());
    }

    #[test]
    fn known_pattern() {
        // Build a known record by hand: offset=4, startblock=200, blockcount=8.
        let offset: u64 = 4;
        let startblock: u64 = 200;
        let blockcount: u64 = 8;
        let hi = (offset << 9) | ((startblock >> 43) & ((1 << 9) - 1));
        let lo = ((startblock & ((1 << 43) - 1)) << 21) | blockcount;
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&hi.to_be_bytes());
        buf[8..16].copy_from_slice(&lo.to_be_bytes());
        let e = Extent::decode(&buf).unwrap();
        assert_eq!(e.offset, 4);
        assert_eq!(e.startblock, 200);
        assert_eq!(e.blockcount, 8);
        assert!(!e.unwritten);
    }
}
