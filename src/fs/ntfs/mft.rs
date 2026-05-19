//! NTFS MFT record decode.
//!
//! Each MFT entry is a `FILE` record (magic `b"FILE"`). The header carries
//! the update-sequence array (USA) used to detect torn sector writes:
//! the last 2 bytes of every 512-byte sector inside the record are
//! displaced by the USA, and the original value lives in the USA. We
//! validate the USN and restore the sector tails before decoding.
//!
//! The fixed header is followed by attributes. Each attribute starts with
//! a 16-byte common header (type, length, non-resident flag, name, flags,
//! attribute id) and then either a resident value or a non-resident
//! mapping-pairs / run-list block.

use crate::Result;

pub(crate) const FILE_RECORD_MAGIC: &[u8; 4] = b"FILE";
pub(crate) const BAAD_RECORD_MAGIC: &[u8; 4] = b"BAAD";

/// Apply the NTFS update-sequence-array fixup to an in-place record buffer.
///
/// The record header at offset 0 has:
///   - `usa_offset` at offset 4 (u16)
///   - `usa_size`   at offset 6 (u16; number of u16 entries = 1 USN + N
///     sector tails, so total sectors covered = usa_size - 1)
///
/// The first entry is the USN. The remaining entries hold the original
/// values that were displaced into the last two bytes of each
/// `sector_size`-byte sector. We:
///   1. Read the USN.
///   2. Verify each sector's tail equals the USN.
///   3. Restore the original byte pair from the USA.
pub fn apply_fixup(buf: &mut [u8], sector_size: usize) -> Result<()> {
    if buf.len() < 8 {
        return Err(crate::Error::InvalidImage(
            "ntfs: record too small for fixup header".into(),
        ));
    }
    let usa_offset = u16::from_le_bytes([buf[4], buf[5]]) as usize;
    let usa_size = u16::from_le_bytes([buf[6], buf[7]]) as usize;
    if usa_size < 2 {
        return Err(crate::Error::InvalidImage(
            "ntfs: USA size < 2 (no sectors)".into(),
        ));
    }
    let usa_bytes = usa_size * 2;
    if usa_offset + usa_bytes > buf.len() {
        return Err(crate::Error::InvalidImage(
            "ntfs: USA extends past record".into(),
        ));
    }
    let sectors = usa_size - 1;
    if buf.len() < sectors * sector_size {
        return Err(crate::Error::InvalidImage(
            "ntfs: record shorter than USA-covered sectors".into(),
        ));
    }
    let usn = [buf[usa_offset], buf[usa_offset + 1]];
    for i in 0..sectors {
        let tail_off = (i + 1) * sector_size - 2;
        if buf[tail_off] != usn[0] || buf[tail_off + 1] != usn[1] {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: USA mismatch on sector {i} (torn write?)"
            )));
        }
        let orig = [buf[usa_offset + 2 + i * 2], buf[usa_offset + 2 + i * 2 + 1]];
        buf[tail_off] = orig[0];
        buf[tail_off + 1] = orig[1];
    }
    Ok(())
}

/// Apply the inverse fixup transform. Used both by test fixtures and by
/// the writer when emitting fresh MFT / INDX records: place the original
/// last-two-bytes of every sector into the USA, then stamp the USN into
/// those tails so subsequent `apply_fixup` calls round-trip.
pub fn install_fixup(buf: &mut [u8], sector_size: usize, usn: u16) {
    let usa_offset = u16::from_le_bytes([buf[4], buf[5]]) as usize;
    let usa_size = u16::from_le_bytes([buf[6], buf[7]]) as usize;
    let sectors = usa_size - 1;
    buf[usa_offset] = usn as u8;
    buf[usa_offset + 1] = (usn >> 8) as u8;
    for i in 0..sectors {
        let tail_off = (i + 1) * sector_size - 2;
        let orig = [buf[tail_off], buf[tail_off + 1]];
        buf[usa_offset + 2 + i * 2] = orig[0];
        buf[usa_offset + 2 + i * 2 + 1] = orig[1];
        buf[tail_off] = usn as u8;
        buf[tail_off + 1] = (usn >> 8) as u8;
    }
}

/// Fields read out of an MFT record's fixed header (after fixup).
#[derive(Debug, Clone)]
pub struct RecordHeader {
    /// Offset of the first attribute relative to the record start.
    pub first_attribute_offset: u16,
    /// `MFT_RECORD_IN_USE` (bit 0) and `MFT_RECORD_IS_DIRECTORY` (bit 1).
    pub flags: u16,
    /// Size of the data actually in use (header + attributes + 0xffff_ffff
    /// terminator). May be less than the on-disk record size.
    pub bytes_in_use: u32,
    /// Total allocated record size (matches BPB-derived mft_record_size).
    pub bytes_allocated: u32,
    /// MFT reference of the base record (0 if this IS the base record).
    pub base_record_ref: u64,
}

impl RecordHeader {
    pub const FLAG_IN_USE: u16 = 0x0001;
    pub const FLAG_DIRECTORY: u16 = 0x0002;

    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < 56 {
            return Err(crate::Error::InvalidImage(
                "ntfs: record buffer too small for header".into(),
            ));
        }
        if &buf[0..4] == BAAD_RECORD_MAGIC {
            return Err(crate::Error::InvalidImage(
                "ntfs: encountered BAAD record".into(),
            ));
        }
        if &buf[0..4] != FILE_RECORD_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: bad record magic {:02x?}",
                &buf[0..4]
            )));
        }
        let first_attribute_offset = u16::from_le_bytes([buf[0x14], buf[0x15]]);
        let flags = u16::from_le_bytes([buf[0x16], buf[0x17]]);
        let bytes_in_use = u32::from_le_bytes(buf[0x18..0x1C].try_into().unwrap());
        let bytes_allocated = u32::from_le_bytes(buf[0x1C..0x20].try_into().unwrap());
        let base_record_ref = u64::from_le_bytes(buf[0x20..0x28].try_into().unwrap());
        Ok(Self {
            first_attribute_offset,
            flags,
            bytes_in_use,
            bytes_allocated,
            base_record_ref,
        })
    }

    pub fn is_in_use(&self) -> bool {
        self.flags & Self::FLAG_IN_USE != 0
    }

    pub fn is_directory(&self) -> bool {
        self.flags & Self::FLAG_DIRECTORY != 0
    }
}
