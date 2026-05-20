//! GRF on-disk header (offset 0, 46 bytes).
//!
//! Layout (little-endian, no inter-field padding):
//!
//! | offset | size | field            | notes                                                    |
//! |--------|------|------------------|----------------------------------------------------------|
//! | 0      | 16   | `magic`          | ASCII `"Master of Magic"` + 0x00                         |
//! | 16     | 14   | `header_key`     | all-zero = no header crypt; 0x01..0x0e = legacy crypt    |
//! | 30     | 4    | `offset`         | byte offset of the file table, **from end of header**    |
//! | 34     | 4    | `seed`           | filecount obfuscation seed (ignored on write today)      |
//! | 38     | 4    | `filecount_enc`  | actual filecount = `filecount_enc - seed - 7`            |
//! | 42     | 4    | `version`        | 0x102 / 0x103 / 0x200                                    |

use crate::Result;

pub(crate) const HEADER_SIZE: usize = 0x2e;
pub(crate) const MAGIC: &[u8; 16] = b"Master of Magic\0";

/// Decoded view of a 46-byte GRF header.
#[derive(Debug, Clone)]
pub struct Header {
    /// `true` when the 14-byte `header_key` block is non-zero (the
    /// legacy "01 02 03 … 0e" pattern). Only the v0x102/0x103 era
    /// ever set this; v0x200 GRFs never do.
    pub encrypted_header: bool,
    /// Byte offset of the file table, counted from the end of the
    /// header (i.e. add 46 — the header size — for an absolute file
    /// offset).
    pub table_offset: u32,
    /// Seed used to obfuscate `filecount_enc`. Modern writers leave
    /// this at zero.
    pub seed: u32,
    /// Number of real file entries in the archive.
    pub filecount: u32,
    /// GRF version word (0x102 / 0x103 / 0x200).
    pub version: u32,
}

impl Header {
    /// Decode a 46-byte buffer into a [`Header`]. Errors on wrong
    /// magic or an unrecognised version.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(crate::Error::InvalidImage(format!(
                "grf: header buffer is {} bytes, need {}",
                buf.len(),
                HEADER_SIZE
            )));
        }
        if &buf[0..16] != MAGIC {
            return Err(crate::Error::InvalidImage(
                "grf: bad header magic (expected \"Master of Magic\")".into(),
            ));
        }
        let key = &buf[16..30];
        let encrypted_header = !key.iter().all(|&b| b == 0);
        let table_offset = u32::from_le_bytes(buf[30..34].try_into().unwrap());
        let seed = u32::from_le_bytes(buf[34..38].try_into().unwrap());
        let filecount_enc = u32::from_le_bytes(buf[38..42].try_into().unwrap());
        let version = u32::from_le_bytes(buf[42..46].try_into().unwrap());

        let filecount = filecount_enc.wrapping_sub(seed).wrapping_sub(7);

        match version {
            0x102 | 0x103 | 0x200 => {}
            other => {
                return Err(crate::Error::Unsupported(format!(
                    "grf: unsupported version {other:#x}"
                )));
            }
        }

        Ok(Self {
            encrypted_header,
            table_offset,
            seed,
            filecount,
            version,
        })
    }

    /// Encode the header back to its 46-byte on-disk form.
    pub fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..16].copy_from_slice(MAGIC);
        if self.encrypted_header {
            // Legacy header crypt marker: bytes 1..=14 in the key
            // slot. The values are positional; they're not actually
            // a key, just a sentinel.
            for (i, b) in buf[16..30].iter_mut().enumerate() {
                *b = (i + 1) as u8;
            }
        }
        buf[30..34].copy_from_slice(&self.table_offset.to_le_bytes());
        buf[34..38].copy_from_slice(&self.seed.to_le_bytes());
        let filecount_enc = self.filecount.wrapping_add(self.seed).wrapping_add(7);
        buf[38..42].copy_from_slice(&filecount_enc.to_le_bytes());
        buf[42..46].copy_from_slice(&self.version.to_le_bytes());
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let h = Header {
            encrypted_header: false,
            table_offset: 0xdead_beef,
            seed: 0,
            filecount: 42,
            version: 0x200,
        };
        let bytes = h.encode();
        let back = Header::decode(&bytes).unwrap();
        assert_eq!(back.table_offset, 0xdead_beef);
        assert_eq!(back.seed, 0);
        assert_eq!(back.filecount, 42);
        assert_eq!(back.version, 0x200);
        assert!(!back.encrypted_header);
    }

    #[test]
    fn encrypted_header_marker() {
        let h = Header {
            encrypted_header: true,
            table_offset: 0,
            seed: 0,
            filecount: 0,
            version: 0x103,
        };
        let bytes = h.encode();
        assert_eq!(
            &bytes[16..30],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        );
        let back = Header::decode(&bytes).unwrap();
        assert!(back.encrypted_header);
    }

    #[test]
    fn filecount_with_seed() {
        let h = Header {
            encrypted_header: false,
            table_offset: 0,
            seed: 100,
            filecount: 50,
            version: 0x200,
        };
        let bytes = h.encode();
        // On disk, the stored filecount is filecount + seed + 7 = 157.
        assert_eq!(u32::from_le_bytes(bytes[38..42].try_into().unwrap()), 157);
        let back = Header::decode(&bytes).unwrap();
        assert_eq!(back.filecount, 50);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[42..46].copy_from_slice(&0x200u32.to_le_bytes());
        let err = Header::decode(&bytes).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidImage(_)));
    }

    #[test]
    fn rejects_unknown_version() {
        let h = Header {
            encrypted_header: false,
            table_offset: 0,
            seed: 0,
            filecount: 0,
            version: 0x100,
        };
        let bytes = h.encode();
        let err = Header::decode(&bytes).unwrap_err();
        assert!(matches!(err, crate::Error::Unsupported(_)));
    }
}
