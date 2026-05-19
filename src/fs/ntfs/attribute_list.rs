//! `$ATTRIBUTE_LIST` (type 0x20) decoder.
//!
//! When a file's attributes don't fit in a single MFT entry, NTFS spills
//! the overflow into one or more extension records and lists their location
//! in the `$ATTRIBUTE_LIST` attribute of the *base* record. Each entry in
//! the list describes one attribute by:
//!
//! * type code (which `$DATA`, `$INDEX_ALLOCATION`, …),
//! * the lowest VCN covered (non-zero only for split non-resident attrs),
//! * the MFT reference of the segment that actually holds the attribute,
//! * the attribute id,
//! * and an optional UTF-16LE name (for named streams / `$I30`).
//!
//! Entry layout (offsets within the entry, all values little-endian):
//!
//! ```text
//!   0x00  u32   attribute type code
//!   0x04  u16   entry length (covers padding to 8 bytes)
//!   0x06  u8    name length (in u16 codepoints, 0 if unnamed)
//!   0x07  u8    name offset (from entry start)
//!   0x08  u64   starting VCN
//!   0x10  u64   MFT reference of segment holding the attribute
//!   0x18  u16   attribute id
//!   0x1A  ...   UTF-16LE attribute name (when name_len > 0)
//! ```
//!
//! Reference: Microsoft "[MS-FSCC]" §2.4.3 and the Russon & Fledel
//! "NTFS Documentation" community PDF, §Attribute List.

use crate::Result;

use super::attribute::decode_utf16le;

/// One row of the `$ATTRIBUTE_LIST`.
#[derive(Debug, Clone)]
pub struct AttributeListEntry {
    /// Attribute type code (`TYPE_*` from `super::attribute`).
    pub type_code: u32,
    /// Starting VCN for split non-resident attributes; 0 for whole attrs.
    pub starting_vcn: u64,
    /// MFT reference (record number in low 48 bits, sequence in high 16).
    pub mft_reference: u64,
    /// Attribute id (matches the id stamped into the segment's attribute
    /// header — used to disambiguate when type + name collide).
    pub attribute_id: u16,
    /// Attribute name (UTF-16LE decoded). Empty if unnamed.
    pub name: String,
}

impl AttributeListEntry {
    /// MFT record number portion of `mft_reference` (low 48 bits).
    pub fn record_number(&self) -> u64 {
        self.mft_reference & 0x0000_FFFF_FFFF_FFFF
    }
}

/// Decode the `$ATTRIBUTE_LIST` value (whether sourced from a resident
/// attribute or read out of a non-resident attribute via the streaming
/// reader). Stops at the end of `buf` or at an obviously-invalid entry
/// length.
pub fn decode(buf: &[u8]) -> Result<Vec<AttributeListEntry>> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor + 0x1A <= buf.len() {
        let type_code = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().unwrap());
        if type_code == 0xFFFF_FFFF {
            // End marker (some encoders use 0xFFFFFFFF here).
            break;
        }
        let entry_len =
            u16::from_le_bytes(buf[cursor + 4..cursor + 6].try_into().unwrap()) as usize;
        if entry_len < 0x1A {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: $ATTRIBUTE_LIST entry length {entry_len} too small"
            )));
        }
        if cursor + entry_len > buf.len() {
            return Err(crate::Error::InvalidImage(
                "ntfs: $ATTRIBUTE_LIST entry oversteps buffer".into(),
            ));
        }
        let name_len = buf[cursor + 6] as usize;
        let name_off = buf[cursor + 7] as usize;
        let starting_vcn = u64::from_le_bytes(buf[cursor + 8..cursor + 0x10].try_into().unwrap());
        let mft_reference =
            u64::from_le_bytes(buf[cursor + 0x10..cursor + 0x18].try_into().unwrap());
        let attribute_id =
            u16::from_le_bytes(buf[cursor + 0x18..cursor + 0x1A].try_into().unwrap());
        let name = if name_len == 0 {
            String::new()
        } else {
            let name_start = cursor + name_off;
            let name_end = name_start + name_len * 2;
            if name_end > cursor + entry_len {
                return Err(crate::Error::InvalidImage(
                    "ntfs: $ATTRIBUTE_LIST name oversteps entry".into(),
                ));
            }
            decode_utf16le(&buf[name_start..name_end])
        };
        out.push(AttributeListEntry {
            type_code,
            starting_vcn,
            mft_reference,
            attribute_id,
            name,
        });
        cursor += entry_len;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_two_entries() {
        // Two entries; first $DATA, second $INDEX_ALLOCATION, both unnamed.
        let mut buf = Vec::new();
        // Entry 0
        let entry_len: u16 = 0x20; // 32 bytes, aligned
        buf.extend_from_slice(&0x80u32.to_le_bytes()); // type
        buf.extend_from_slice(&entry_len.to_le_bytes()); // entry_len
        buf.push(0); // name_len
        buf.push(0x1A); // name_off (immaterial — name_len=0)
        buf.extend_from_slice(&0u64.to_le_bytes()); // starting_vcn
        buf.extend_from_slice(&((42u64) | (1u64 << 48)).to_le_bytes()); // mft ref
        buf.extend_from_slice(&7u16.to_le_bytes()); // attr id
        buf.extend(std::iter::repeat_n(0u8, 6)); // pad to 32
        // Entry 1
        buf.extend_from_slice(&0xA0u32.to_le_bytes());
        buf.extend_from_slice(&entry_len.to_le_bytes());
        buf.push(0);
        buf.push(0x1A);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&43u64.to_le_bytes());
        buf.extend_from_slice(&8u16.to_le_bytes());
        buf.extend(std::iter::repeat_n(0u8, 6));

        let entries = decode(&buf).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].type_code, 0x80);
        assert_eq!(entries[0].record_number(), 42);
        assert_eq!(entries[0].attribute_id, 7);
        assert_eq!(entries[1].type_code, 0xA0);
        assert_eq!(entries[1].record_number(), 43);
    }
}
