//! NTFS attribute decode.
//!
//! Attributes share a 16-byte common header: type code, total length,
//! non-resident flag, attribute-name length and offset, flags, and
//! attribute id. Resident attributes embed their value inline; non-resident
//! attributes carry a "mapping pairs" / run-list block instead.

use crate::Result;

use super::run_list::{self, Extent};

pub const TYPE_STANDARD_INFORMATION: u32 = 0x10;
pub const TYPE_ATTRIBUTE_LIST: u32 = 0x20;
pub const TYPE_FILE_NAME: u32 = 0x30;
pub const TYPE_OBJECT_ID: u32 = 0x40;
pub const TYPE_SECURITY_DESCRIPTOR: u32 = 0x50;
pub const TYPE_VOLUME_NAME: u32 = 0x60;
pub const TYPE_DATA: u32 = 0x80;
pub const TYPE_INDEX_ROOT: u32 = 0x90;
pub const TYPE_INDEX_ALLOCATION: u32 = 0xA0;
pub const TYPE_BITMAP: u32 = 0xB0;
pub const TYPE_REPARSE_POINT: u32 = 0xC0;
pub const TYPE_END: u32 = 0xFFFF_FFFF;

pub const ATTR_FLAG_COMPRESSED: u16 = 0x0001;
pub const ATTR_FLAG_ENCRYPTED: u16 = 0x4000;
pub const ATTR_FLAG_SPARSE: u16 = 0x8000;

/// Decoded view of one attribute. Lifetime borrows the underlying MFT
/// record buffer.
#[derive(Debug)]
pub struct Attribute<'a> {
    /// Offset of this attribute's start within the record.
    pub offset: usize,
    /// Total bytes spanned by this attribute (header + value).
    pub length: u32,
    pub type_code: u32,
    pub flags: u16,
    pub attribute_id: u16,
    /// Attribute name (UTF-16LE decoded to a String) — empty if unnamed.
    pub name: String,
    pub kind: AttributeKind<'a>,
}

#[derive(Debug)]
pub enum AttributeKind<'a> {
    Resident {
        value: &'a [u8],
        /// `RESIDENT_FORM_INDEXED` flag (0x01). Only meaningful for the
        /// view that decides which $FILE_NAME goes into the index.
        indexed_flag: u8,
    },
    NonResident {
        starting_vcn: u64,
        last_vcn: u64,
        allocated_size: u64,
        real_size: u64,
        initialized_size: u64,
        /// Compression unit size as a power of two (0 means uncompressed).
        compression_unit: u8,
        /// Pre-decoded run list.
        runs: Vec<Extent>,
    },
}

impl<'a> Attribute<'a> {
    pub fn is_compressed(&self) -> bool {
        self.flags & ATTR_FLAG_COMPRESSED != 0
    }
    pub fn is_encrypted(&self) -> bool {
        self.flags & ATTR_FLAG_ENCRYPTED != 0
    }
    pub fn is_sparse(&self) -> bool {
        self.flags & ATTR_FLAG_SPARSE != 0
    }
}

/// Iterator-style walker over the attributes in a single MFT record.
pub struct AttributeIter<'a> {
    record: &'a [u8],
    cursor: usize,
    finished: bool,
}

impl<'a> AttributeIter<'a> {
    pub fn new(record: &'a [u8], first_attr_offset: usize) -> Self {
        Self {
            record,
            cursor: first_attr_offset,
            finished: false,
        }
    }
}

impl<'a> Iterator for AttributeIter<'a> {
    type Item = Result<Attribute<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if self.cursor + 4 > self.record.len() {
            self.finished = true;
            return None;
        }
        let type_code = u32::from_le_bytes(
            self.record[self.cursor..self.cursor + 4]
                .try_into()
                .unwrap(),
        );
        if type_code == TYPE_END {
            self.finished = true;
            return None;
        }
        if self.cursor + 16 > self.record.len() {
            self.finished = true;
            return Some(Err(crate::Error::InvalidImage(
                "ntfs: attribute header truncated".into(),
            )));
        }
        let length = u32::from_le_bytes(
            self.record[self.cursor + 4..self.cursor + 8]
                .try_into()
                .unwrap(),
        );
        if length < 16 || self.cursor + length as usize > self.record.len() {
            self.finished = true;
            return Some(Err(crate::Error::InvalidImage(format!(
                "ntfs: attribute length {length} oversteps record"
            ))));
        }
        let non_resident = self.record[self.cursor + 8] != 0;
        let name_len = self.record[self.cursor + 9] as usize;
        let name_off = u16::from_le_bytes(
            self.record[self.cursor + 10..self.cursor + 12]
                .try_into()
                .unwrap(),
        ) as usize;
        let flags = u16::from_le_bytes(
            self.record[self.cursor + 12..self.cursor + 14]
                .try_into()
                .unwrap(),
        );
        let attribute_id = u16::from_le_bytes(
            self.record[self.cursor + 14..self.cursor + 16]
                .try_into()
                .unwrap(),
        );

        let name = if name_len == 0 {
            String::new()
        } else {
            let name_bytes_start = self.cursor + name_off;
            let name_bytes_end = name_bytes_start + name_len * 2;
            if name_bytes_end > self.record.len() {
                self.finished = true;
                return Some(Err(crate::Error::InvalidImage(
                    "ntfs: attribute name oversteps record".into(),
                )));
            }
            decode_utf16le(&self.record[name_bytes_start..name_bytes_end])
        };

        let kind = if !non_resident {
            // Resident header (offsets relative to attribute start):
            // 0x10: value_length (u32)
            // 0x14: value_offset (u16)
            // 0x16: indexed_flag (u8)
            let value_len = u32::from_le_bytes(
                self.record[self.cursor + 0x10..self.cursor + 0x14]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let value_off = u16::from_le_bytes(
                self.record[self.cursor + 0x14..self.cursor + 0x16]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let indexed_flag = self.record[self.cursor + 0x16];
            let val_start = self.cursor + value_off;
            let val_end = val_start + value_len;
            if val_end > self.cursor + length as usize {
                self.finished = true;
                return Some(Err(crate::Error::InvalidImage(
                    "ntfs: resident attribute value oversteps attribute".into(),
                )));
            }
            AttributeKind::Resident {
                value: &self.record[val_start..val_end],
                indexed_flag,
            }
        } else {
            // Non-resident header (offsets relative to attribute start):
            // 0x10: starting VCN (u64)
            // 0x18: last VCN     (u64)
            // 0x20: run-list off (u16)
            // 0x22: compression unit (u16, low byte is power of two)
            // 0x28: allocated size  (u64)
            // 0x30: real size       (u64)
            // 0x38: initialized     (u64)
            // Run list begins at attribute_start + run-list-offset.
            if length < 0x40 {
                self.finished = true;
                return Some(Err(crate::Error::InvalidImage(
                    "ntfs: non-resident attribute header too short".into(),
                )));
            }
            let starting_vcn = u64::from_le_bytes(
                self.record[self.cursor + 0x10..self.cursor + 0x18]
                    .try_into()
                    .unwrap(),
            );
            let last_vcn = u64::from_le_bytes(
                self.record[self.cursor + 0x18..self.cursor + 0x20]
                    .try_into()
                    .unwrap(),
            );
            let runs_off = u16::from_le_bytes(
                self.record[self.cursor + 0x20..self.cursor + 0x22]
                    .try_into()
                    .unwrap(),
            ) as usize;
            let compression_unit = u16::from_le_bytes(
                self.record[self.cursor + 0x22..self.cursor + 0x24]
                    .try_into()
                    .unwrap(),
            ) as u8;
            let allocated_size = u64::from_le_bytes(
                self.record[self.cursor + 0x28..self.cursor + 0x30]
                    .try_into()
                    .unwrap(),
            );
            let real_size = u64::from_le_bytes(
                self.record[self.cursor + 0x30..self.cursor + 0x38]
                    .try_into()
                    .unwrap(),
            );
            let initialized_size = u64::from_le_bytes(
                self.record[self.cursor + 0x38..self.cursor + 0x40]
                    .try_into()
                    .unwrap(),
            );
            let runs_start = self.cursor + runs_off;
            let runs_end = self.cursor + length as usize;
            if runs_off < 0x40 || runs_start > runs_end {
                self.finished = true;
                return Some(Err(crate::Error::InvalidImage(
                    "ntfs: non-resident run-list offset invalid".into(),
                )));
            }
            let runs = match run_list::decode(&self.record[runs_start..runs_end]) {
                Ok(r) => r,
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            };
            AttributeKind::NonResident {
                starting_vcn,
                last_vcn,
                allocated_size,
                real_size,
                initialized_size,
                compression_unit,
                runs,
            }
        };

        let attr = Attribute {
            offset: self.cursor,
            length,
            type_code,
            flags,
            attribute_id,
            name,
            kind,
        };
        self.cursor += length as usize;
        Some(Ok(attr))
    }
}

/// Decode a UTF-16LE little-endian byte slice to a String, replacing
/// unpaired surrogates with U+FFFD.
pub fn decode_utf16le(b: &[u8]) -> String {
    let units: Vec<u16> = b
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    char::decode_utf16(units.iter().copied())
        .map(|r| r.unwrap_or('\u{FFFD}'))
        .collect()
}

/// Decoded $STANDARD_INFORMATION attribute. Times are NT-FILETIME (100ns
/// units since 1601-01-01 UTC).
#[derive(Debug, Clone)]
pub struct StandardInformation {
    pub creation_time: u64,
    pub modified_time: u64,
    pub mft_changed_time: u64,
    pub accessed_time: u64,
    /// DOS-style file attributes (READONLY/HIDDEN/SYSTEM/ARCHIVE/...)
    pub file_attributes: u32,
}

impl StandardInformation {
    pub fn parse(value: &[u8]) -> Result<Self> {
        if value.len() < 48 {
            return Err(crate::Error::InvalidImage(
                "ntfs: $STANDARD_INFORMATION too short".into(),
            ));
        }
        Ok(Self {
            creation_time: u64::from_le_bytes(value[0..8].try_into().unwrap()),
            modified_time: u64::from_le_bytes(value[8..16].try_into().unwrap()),
            mft_changed_time: u64::from_le_bytes(value[16..24].try_into().unwrap()),
            accessed_time: u64::from_le_bytes(value[24..32].try_into().unwrap()),
            file_attributes: u32::from_le_bytes(value[32..36].try_into().unwrap()),
        })
    }

    /// Pack the four FILETIMEs into a 32-byte raw blob (create, modify,
    /// change, access) for the `user.ntfs.times.raw` xattr.
    pub fn times_raw(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0..8].copy_from_slice(&self.creation_time.to_le_bytes());
        out[8..16].copy_from_slice(&self.modified_time.to_le_bytes());
        out[16..24].copy_from_slice(&self.mft_changed_time.to_le_bytes());
        out[24..32].copy_from_slice(&self.accessed_time.to_le_bytes());
        out
    }
}

/// Decoded $FILE_NAME attribute.
#[derive(Debug, Clone)]
pub struct FileName {
    pub parent_mft_ref: u64,
    pub creation_time: u64,
    pub modified_time: u64,
    pub mft_changed_time: u64,
    pub accessed_time: u64,
    pub allocated_size: u64,
    pub real_size: u64,
    pub flags: u32,
    pub namespace: u8,
    pub name: String,
}

impl FileName {
    pub const NAMESPACE_POSIX: u8 = 0;
    pub const NAMESPACE_WIN32: u8 = 1;
    pub const NAMESPACE_DOS: u8 = 2;
    pub const NAMESPACE_WIN32_DOS: u8 = 3;

    /// File attribute bit: directory.
    pub const FLAG_DIRECTORY: u32 = 0x1000_0000;

    pub fn parse(value: &[u8]) -> Result<Self> {
        if value.len() < 66 {
            return Err(crate::Error::InvalidImage(
                "ntfs: $FILE_NAME too short".into(),
            ));
        }
        let parent_mft_ref = u64::from_le_bytes(value[0..8].try_into().unwrap());
        let creation_time = u64::from_le_bytes(value[8..16].try_into().unwrap());
        let modified_time = u64::from_le_bytes(value[16..24].try_into().unwrap());
        let mft_changed_time = u64::from_le_bytes(value[24..32].try_into().unwrap());
        let accessed_time = u64::from_le_bytes(value[32..40].try_into().unwrap());
        let allocated_size = u64::from_le_bytes(value[40..48].try_into().unwrap());
        let real_size = u64::from_le_bytes(value[48..56].try_into().unwrap());
        let flags = u32::from_le_bytes(value[56..60].try_into().unwrap());
        // 60..64: reparse value (we ignore here).
        let name_len = value[64] as usize;
        let namespace = value[65];
        let name_bytes_end = 66 + name_len * 2;
        if name_bytes_end > value.len() {
            return Err(crate::Error::InvalidImage(
                "ntfs: $FILE_NAME name oversteps attribute".into(),
            ));
        }
        let name = decode_utf16le(&value[66..name_bytes_end]);
        Ok(Self {
            parent_mft_ref,
            creation_time,
            modified_time,
            mft_changed_time,
            accessed_time,
            allocated_size,
            real_size,
            flags,
            namespace,
            name,
        })
    }

    /// Extract the 48-bit MFT record number from a packed MFT reference.
    pub fn parent_record_number(&self) -> u64 {
        self.parent_mft_ref & 0x0000_FFFF_FFFF_FFFF
    }

    pub fn is_directory(&self) -> bool {
        self.flags & Self::FLAG_DIRECTORY != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_resident_data() {
        // Build a fake record with a single resident $DATA attribute:
        let mut rec = vec![0u8; 256];
        // Attribute header
        let off = 0;
        rec[off..off + 4].copy_from_slice(&TYPE_DATA.to_le_bytes()); // type
        let val_off = 0x18u16;
        let val_len = 4u32;
        let attr_len = (val_off as u32) + val_len; // 0x1C
        rec[off + 4..off + 8].copy_from_slice(&attr_len.to_le_bytes()); // length
        rec[off + 8] = 0; // resident
        rec[off + 9] = 0; // name_len
        rec[off + 10..off + 12].copy_from_slice(&0u16.to_le_bytes());
        rec[off + 12..off + 14].copy_from_slice(&0u16.to_le_bytes()); // flags
        rec[off + 14..off + 16].copy_from_slice(&0u16.to_le_bytes()); // attr id
        rec[off + 0x10..off + 0x14].copy_from_slice(&val_len.to_le_bytes());
        rec[off + 0x14..off + 0x16].copy_from_slice(&val_off.to_le_bytes());
        rec[off + 0x16] = 0;
        rec[off + 0x18..off + 0x1C].copy_from_slice(b"DATA");

        // Then terminator
        let term = off + attr_len as usize;
        rec[term..term + 4].copy_from_slice(&TYPE_END.to_le_bytes());

        let mut iter = AttributeIter::new(&rec, 0);
        let a = iter.next().unwrap().unwrap();
        assert_eq!(a.type_code, TYPE_DATA);
        match a.kind {
            AttributeKind::Resident { value, .. } => assert_eq!(value, b"DATA"),
            _ => panic!("expected resident"),
        }
        assert!(iter.next().is_none());
    }

    #[test]
    fn standard_information_parse() {
        let mut v = vec![0u8; 48];
        v[0..8].copy_from_slice(&0x1122334455667788u64.to_le_bytes()); // creation
        v[32..36].copy_from_slice(&0x1234_5678u32.to_le_bytes()); // attrs
        let si = StandardInformation::parse(&v).unwrap();
        assert_eq!(si.creation_time, 0x1122334455667788);
        assert_eq!(si.file_attributes, 0x1234_5678);
        assert_eq!(si.times_raw()[0], 0x88);
    }

    #[test]
    fn file_name_parse() {
        let name_utf16: Vec<u8> = "hi".encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
        let mut v = vec![0u8; 66 + name_utf16.len()];
        v[64] = 2; // name_len in u16 codepoints
        v[65] = FileName::NAMESPACE_WIN32;
        v[56..60].copy_from_slice(&FileName::FLAG_DIRECTORY.to_le_bytes());
        v[66..66 + name_utf16.len()].copy_from_slice(&name_utf16);
        let fname = FileName::parse(&v).unwrap();
        assert_eq!(fname.name, "hi");
        assert_eq!(fname.namespace, FileName::NAMESPACE_WIN32);
        assert!(fname.is_directory());
    }
}
