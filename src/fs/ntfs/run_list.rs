//! NTFS data-run list decoder.
//!
//! A non-resident attribute's data extents live as a compressed sequence
//! of (length, offset) pairs called a "run list". Each entry starts with
//! a 1-byte header where the low nibble encodes the byte length of the
//! run-length field, and the high nibble encodes the byte length of the
//! run-offset field. A header of 0x00 terminates the list.
//!
//! - The length field is unsigned LE, in clusters.
//! - The offset field is a signed LE relative LCN delta from the previous
//!   run. If the offset field length is 0, this is a sparse run (no LCN).
//!
//! Layout per "NTFS Documentation" (Russon & Fledel).

use crate::Result;

/// One extent of a non-resident attribute's data, in clusters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    /// Logical cluster number, or `None` for a sparse extent (reads as zero).
    pub lcn: Option<u64>,
    /// Number of clusters this extent covers.
    pub length: u64,
}

/// Decode a run list from `buf`. Stops on the terminating 0x00 header or at
/// the end of `buf`. Returns the parsed extents.
pub fn decode(buf: &[u8]) -> Result<Vec<Extent>> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    let mut prev_lcn: i64 = 0;
    while cursor < buf.len() {
        let header = buf[cursor];
        if header == 0 {
            break;
        }
        cursor += 1;
        let len_size = (header & 0x0F) as usize;
        let off_size = ((header >> 4) & 0x0F) as usize;
        if len_size == 0 || len_size > 8 || off_size > 8 {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: bad run-list header 0x{header:02x} at offset {cursor}"
            )));
        }
        if cursor + len_size + off_size > buf.len() {
            return Err(crate::Error::InvalidImage(
                "ntfs: run-list truncated".into(),
            ));
        }
        let length = read_unsigned_le(&buf[cursor..cursor + len_size]);
        cursor += len_size;
        let lcn = if off_size == 0 {
            None
        } else {
            let delta = read_signed_le(&buf[cursor..cursor + off_size]);
            cursor += off_size;
            prev_lcn = prev_lcn
                .checked_add(delta)
                .ok_or_else(|| crate::Error::InvalidImage("ntfs: run-list LCN overflow".into()))?;
            if prev_lcn < 0 {
                return Err(crate::Error::InvalidImage(format!(
                    "ntfs: run-list produced negative LCN {prev_lcn}"
                )));
            }
            Some(prev_lcn as u64)
        };
        out.push(Extent { lcn, length });
    }
    Ok(out)
}

fn read_unsigned_le(b: &[u8]) -> u64 {
    let mut v = 0u64;
    for (i, &byte) in b.iter().enumerate() {
        v |= (byte as u64) << (8 * i);
    }
    v
}

fn read_signed_le(b: &[u8]) -> i64 {
    let n = b.len();
    if n == 0 {
        return 0;
    }
    let mut v = 0i64;
    for (i, &byte) in b.iter().enumerate() {
        v |= (byte as i64) << (8 * i);
    }
    // Sign-extend from the high byte's MSB.
    let sign_bit = 1i64 << (8 * n - 1);
    if v & sign_bit != 0 {
        let mask = !((1i64 << (8 * n)).wrapping_sub(1));
        v |= mask;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_single_run() {
        // 0x21 0x18 0x34 0x12: length=1 byte (0x18=24), offset=2 bytes LE (0x1234=4660)
        let runs = decode(&[0x21, 0x18, 0x34, 0x12, 0x00]).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].length, 24);
        assert_eq!(runs[0].lcn, Some(4660));
    }

    #[test]
    fn decode_sparse_run() {
        // 0x01 0x08: length=1 byte (8 clusters), no offset → sparse.
        let runs = decode(&[0x01, 0x08, 0x00]).unwrap();
        assert_eq!(runs[0].lcn, None);
        assert_eq!(runs[0].length, 8);
    }

    #[test]
    fn decode_two_runs_relative() {
        // 0x21 0x10 0x00 0x01 -> length=16, lcn=256
        // 0x21 0x08 0x00 0x01 -> length=8, delta=+256 → lcn=512
        // 0x00 terminator
        let runs = decode(&[0x21, 0x10, 0x00, 0x01, 0x21, 0x08, 0x00, 0x01, 0x00]).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].lcn, Some(256));
        assert_eq!(runs[1].lcn, Some(512));
    }

    #[test]
    fn decode_negative_delta() {
        // Second run delta is negative (FF = -1 in signed 1-byte).
        let runs = decode(&[0x11, 0x04, 0x10, 0x11, 0x04, 0xFF, 0x00]).unwrap();
        assert_eq!(runs[0].lcn, Some(0x10));
        assert_eq!(runs[1].lcn, Some(0x0F));
    }
}
