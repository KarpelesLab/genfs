//! exFAT Up-case Table — maps every Unicode BMP code unit to its
//! case-folded form for case-insensitive name comparison.
//!
//! On-disk encoding is an array of little-endian u16 values, optionally
//! run-length compressed: the value 0xFFFF followed by a count N means
//! "the next N code units map to themselves" (i.e. identity for that
//! range). The table covers indices 0..N where N is encoded in the
//! UpcaseTable directory entry's DataLength (in bytes) divided by 2.
//!
//! ## Format
//!
//! ```text
//!     decompressed[i]      = next u16 (i increments by 1)
//!     0xFFFF, count        = decompressed[i..i+count] = i..i+count
//! ```
//!
//! ## Validation
//!
//! The full table for a "standard" exFAT volume has a fixed table-
//! checksum value, but volumes are free to use a custom table — so we do
//! not enforce a specific checksum. We just compute the rolling 32-bit
//! checksum over the raw on-disk bytes and expose it for callers that
//! want to compare against the UpcaseTable directory entry's
//! `TableChecksum`.

/// A decoded up-case table. Indexable by u16 code unit; entries beyond
/// `table.len()` map to themselves.
#[derive(Debug, Clone, Default)]
pub struct Upcase {
    table: Vec<u16>,
}

impl Upcase {
    /// Build an ASCII-only up-case table: a..z → A..Z, everything else
    /// identity. Fallback used when reading the real on-disk table fails.
    pub fn ascii() -> Self {
        let mut table = vec![0u16; 0x80];
        for (i, slot) in table.iter_mut().enumerate() {
            let c = i as u8;
            *slot = if c.is_ascii_lowercase() {
                (c - b'a' + b'A') as u16
            } else {
                c as u16
            };
        }
        Self { table }
    }

    /// Decode the on-disk up-case table. `bytes` is the raw cluster
    /// data; `data_length` is the table's `DataLength` from its directory
    /// entry (in bytes). The table is a u16 stream with optional 0xFFFF
    /// run-length escapes for identity runs.
    pub fn decode(bytes: &[u8], data_length: u64) -> crate::Result<Self> {
        let data_length = data_length as usize;
        if data_length > bytes.len() {
            return Err(crate::Error::InvalidImage(format!(
                "exfat: upcase table DataLength {data_length} exceeds cluster bytes {}",
                bytes.len()
            )));
        }
        if !data_length.is_multiple_of(2) {
            return Err(crate::Error::InvalidImage(
                "exfat: upcase table DataLength is not a multiple of 2".into(),
            ));
        }
        let slice = &bytes[..data_length];
        let mut table: Vec<u16> = Vec::new();
        let mut i = 0usize;
        while i + 2 <= slice.len() {
            let v = u16::from_le_bytes(slice[i..i + 2].try_into().unwrap());
            i += 2;
            if v == 0xFFFF {
                // Expect a count u16; identity-fill that many slots.
                if i + 2 > slice.len() {
                    return Err(crate::Error::InvalidImage(
                        "exfat: upcase table truncated after 0xFFFF escape".into(),
                    ));
                }
                let count = u16::from_le_bytes(slice[i..i + 2].try_into().unwrap()) as usize;
                i += 2;
                let base = table.len();
                // count==0 is technically legal as "fill zero identities" —
                // benign, but unusual. Cap the total table size to a sane
                // limit so a malformed image can't allocate forever.
                if base + count > 0x11_0000 {
                    return Err(crate::Error::InvalidImage(
                        "exfat: upcase table exceeds 0x110000 entries".into(),
                    ));
                }
                for k in 0..count {
                    table.push((base + k) as u16);
                }
            } else {
                table.push(v);
            }
        }
        Ok(Self { table })
    }

    /// Up-case a single u16 code unit.
    pub fn up(&self, ch: u16) -> u16 {
        self.table.get(ch as usize).copied().unwrap_or(ch)
    }

    /// Up-case a sequence of u16 code units (UTF-16) into a new vector.
    pub fn up_slice(&self, units: &[u16]) -> Vec<u16> {
        units.iter().map(|&u| self.up(u)).collect()
    }

    /// Up-case a UTF-16 sequence of a Rust `&str`.
    pub fn up_str(&self, s: &str) -> Vec<u16> {
        let units: Vec<u16> = s.encode_utf16().collect();
        self.up_slice(&units)
    }

    /// Compare two UTF-16 sequences for case-insensitive equality using
    /// this up-case table.
    pub fn eq_ignore_case(&self, a: &[u16], b: &[u16]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter()
            .zip(b.iter())
            .all(|(&x, &y)| self.up(x) == self.up(y))
    }

    /// Number of slots actually populated; beyond this, characters map to
    /// themselves.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether the table holds no explicit mappings.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

/// Rolling 32-bit checksum used by exFAT for the up-case table and the
/// entry-set checksum (with different skip semantics). Each byte rotates
/// the accumulator right by one bit and adds the byte.
pub fn table_checksum(bytes: &[u8]) -> u32 {
    let mut sum: u32 = 0;
    for &b in bytes {
        sum = sum.rotate_right(1).wrapping_add(b as u32);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_upcase_basic() {
        let uc = Upcase::ascii();
        assert_eq!(uc.up(b'a' as u16), b'A' as u16);
        assert_eq!(uc.up(b'z' as u16), b'Z' as u16);
        assert_eq!(uc.up(b'A' as u16), b'A' as u16);
        assert_eq!(uc.up(b'0' as u16), b'0' as u16);
        // Outside the populated range maps to itself.
        assert_eq!(uc.up(0x1234), 0x1234);
    }

    #[test]
    fn ascii_upcase_str() {
        let uc = Upcase::ascii();
        let folded = uc.up_str("Hello.txt");
        let expected: Vec<u16> = "HELLO.TXT".encode_utf16().collect();
        assert_eq!(folded, expected);
    }

    #[test]
    fn ascii_eq_ignore_case() {
        let uc = Upcase::ascii();
        let a: Vec<u16> = "ReadMe".encode_utf16().collect();
        let b: Vec<u16> = "README".encode_utf16().collect();
        assert!(uc.eq_ignore_case(&a, &b));
        let c: Vec<u16> = "README!".encode_utf16().collect();
        assert!(!uc.eq_ignore_case(&a, &c));
    }

    #[test]
    fn decode_simple_uncompressed() {
        // Three explicit entries: 'a' -> 'A', 'b' -> 'B', 'c' -> 'C'.
        let mut bytes = Vec::new();
        for v in [b'A' as u16, b'B' as u16, b'C' as u16] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let uc = Upcase::decode(&bytes, bytes.len() as u64).unwrap();
        assert_eq!(uc.len(), 3);
        assert_eq!(uc.up(0), b'A' as u16);
        assert_eq!(uc.up(1), b'B' as u16);
        assert_eq!(uc.up(2), b'C' as u16);
    }

    #[test]
    fn decode_with_identity_run() {
        // Build: 'A','B', then identity-run of 5 (slots 2..7), then 0xABCD.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(b'A' as u16).to_le_bytes());
        bytes.extend_from_slice(&(b'B' as u16).to_le_bytes());
        bytes.extend_from_slice(&0xFFFFu16.to_le_bytes());
        bytes.extend_from_slice(&5u16.to_le_bytes());
        bytes.extend_from_slice(&0xABCDu16.to_le_bytes());
        let uc = Upcase::decode(&bytes, bytes.len() as u64).unwrap();
        assert_eq!(uc.len(), 8);
        assert_eq!(uc.up(0), b'A' as u16);
        assert_eq!(uc.up(1), b'B' as u16);
        // Identity slots 2..7 map to themselves.
        assert_eq!(uc.up(2), 2);
        assert_eq!(uc.up(6), 6);
        // Slot 7 is the explicit 0xABCD.
        assert_eq!(uc.up(7), 0xABCD);
    }

    #[test]
    fn checksum_known() {
        // Empty input → 0.
        assert_eq!(table_checksum(&[]), 0);
        // Single byte 0x01 → rotate_right(1) of 0 is 0, plus 1 = 1.
        assert_eq!(table_checksum(&[0x01]), 1);
        // Two bytes 0x01, 0x02 →
        //   after first: 1
        //   after second: rotate_right(1) of 1 is 0x80000000, plus 2 = 0x80000002
        assert_eq!(table_checksum(&[0x01, 0x02]), 0x8000_0002);
    }
}
