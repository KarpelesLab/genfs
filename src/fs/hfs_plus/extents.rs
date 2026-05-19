//! HFS+ extents-overflow file — TN1150 "Extents Overflow File".
//!
//! The extents-overflow B-tree records the ninth and later extents of
//! any fork (data or resource) that needs more than the eight inline
//! extents kept in the catalog. Its key is:
//!
//! ```text
//! offset  size  field
//! 0       2     keyLength (= 10)
//! 2       1     forkType  (0 = data, 0xFF = resource)
//! 3       1     pad
//! 4       4     fileID    (CNID)
//! 8       4     startBlock (the fork-block where this record's run begins)
//! ```
//!
//! Each leaf record's data is an array of 8 `HFSPlusExtentDescriptor`
//! entries (`{ u32 startBlock, u32 blockCount }`).
//!
//! For v1, fstool only consumes the eight inline extents stored in
//! the catalog record. If a file requires the overflow tree, the
//! caller surfaces an `Unsupported` error rather than walking the
//! tree here. This module retains the on-disk constants and a key
//! decoder for diagnostic use and to ease later expansion.

use crate::Result;

/// Fork-type tag for the data fork.
pub const FORK_DATA: u8 = 0x00;
/// Fork-type tag for the resource fork.
pub const FORK_RESOURCE: u8 = 0xFF;

/// Fixed length (10 bytes) of the keyLength-bearing portion of an
/// HFSPlusExtentKey on disk.
pub const EXTENT_KEY_PAYLOAD_LEN: usize = 10;

/// Decoded HFSPlusExtentKey.
#[derive(Debug, Clone, Copy)]
pub struct ExtentKey {
    pub fork_type: u8,
    pub file_id: u32,
    pub start_block: u32,
}

impl ExtentKey {
    /// Decode an extent-overflow key from `buf`. The leading u16
    /// `keyLength` is consumed.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < 2 + EXTENT_KEY_PAYLOAD_LEN {
            return Err(crate::Error::InvalidImage(
                "hfs+: short extents-overflow key".into(),
            ));
        }
        let key_length = u16::from_be_bytes([buf[0], buf[1]]);
        if key_length as usize != EXTENT_KEY_PAYLOAD_LEN {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: extents-overflow key_length {key_length} != {EXTENT_KEY_PAYLOAD_LEN}"
            )));
        }
        Ok(Self {
            fork_type: buf[2],
            // buf[3] is the pad byte
            file_id: u32::from_be_bytes(buf[4..8].try_into().unwrap()),
            start_block: u32::from_be_bytes(buf[8..12].try_into().unwrap()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extent_key_decodes() {
        // keyLength = 10, fork = data, pad = 0, fileID = 42, startBlock = 9
        let mut buf = Vec::new();
        buf.extend_from_slice(&(EXTENT_KEY_PAYLOAD_LEN as u16).to_be_bytes());
        buf.push(FORK_DATA);
        buf.push(0);
        buf.extend_from_slice(&42u32.to_be_bytes());
        buf.extend_from_slice(&9u32.to_be_bytes());
        let k = ExtentKey::decode(&buf).unwrap();
        assert_eq!(k.fork_type, FORK_DATA);
        assert_eq!(k.file_id, 42);
        assert_eq!(k.start_block, 9);
    }

    #[test]
    fn extent_key_rejects_wrong_length() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&8u16.to_be_bytes()); // wrong key length
        buf.extend(std::iter::repeat_n(0u8, EXTENT_KEY_PAYLOAD_LEN));
        assert!(ExtentKey::decode(&buf).is_err());
    }
}
