//! Volume Descriptors per ECMA-119 §8 and Joliet §3.
//!
//! Every descriptor occupies one logical sector (2048 bytes) and starts
//! with a one-byte type code, the 5-byte standard identifier "CD001",
//! and a one-byte version. We parse the four kinds that matter for a
//! reader: Primary (1), Supplementary (2), Boot Record (0), and Volume
//! Descriptor Set Terminator (255). Partition Descriptors (3) are
//! ignored — none of the modern source ISOs we want to read use them.

use crate::Result;

use super::directory::DirRecord;

/// Decoded view of one volume descriptor sector.
#[derive(Debug, Clone)]
pub enum VolumeDescriptor {
    Primary(PrimaryVolumeDescriptor),
    Supplementary(SupplementaryVolumeDescriptor),
    Boot { catalog_lba: u32 },
    Partition,
    Terminator,
}

impl VolumeDescriptor {
    /// Decode a single 2048-byte sector into a typed descriptor.
    pub fn probe(sector: &[u8]) -> Result<Self> {
        if sector.len() < super::SECTOR_SIZE as usize {
            return Err(crate::Error::InvalidImage(
                "iso9660: short volume descriptor sector".into(),
            ));
        }
        let kind = sector[0];
        if &sector[1..6] != super::ISO_MAGIC {
            return Err(crate::Error::InvalidImage(
                "iso9660: CD001 magic missing in descriptor".into(),
            ));
        }
        match kind {
            0 => {
                // Boot Record: bytes 7..39 = boot system identifier,
                // 39..71 = boot identifier. The Catalog LBA for El
                // Torito sits at offset 0x47..0x4B as a little-endian
                // u32. (Catalog LBA is what we care about.)
                let catalog_lba = u32::from_le_bytes(sector[0x47..0x4B].try_into().unwrap());
                Ok(Self::Boot { catalog_lba })
            }
            1 => Ok(Self::Primary(PrimaryVolumeDescriptor::decode(sector)?)),
            2 => Ok(Self::Supplementary(SupplementaryVolumeDescriptor::decode(
                sector,
            )?)),
            3 => Ok(Self::Partition),
            255 => Ok(Self::Terminator),
            other => Err(crate::Error::InvalidImage(format!(
                "iso9660: unknown volume descriptor type {other}"
            ))),
        }
    }
}

/// Primary Volume Descriptor — ECMA-119 §8.4. The fields we surface are
/// what the rest of the reader / `info` need; everything else (creation
/// date, application id, copyright file identifier, ...) is parsed only
/// far enough to skip the right number of bytes.
#[derive(Debug, Clone)]
pub struct PrimaryVolumeDescriptor {
    pub system_id: String,
    pub volume_id: String,
    pub volume_space_size: u32,
    pub logical_block_size: u16,
    pub path_table_size: u32,
    pub l_path_table_lba: u32,
    pub m_path_table_lba: u32,
    /// Root directory record, parsed from bytes 156..190 of the PVD.
    pub root: DirRecord,
}

impl PrimaryVolumeDescriptor {
    fn decode(sector: &[u8]) -> Result<Self> {
        // ECMA-119 §8.4 — A1 = a-characters, D1 = d-characters; both
        // surface as ASCII-or-Unicode-compatible bytes here. We trim.
        let system_id = trim_strd(&sector[8..40]);
        let volume_id = trim_strd(&sector[40..72]);
        let volume_space_size = decode_both_endian_u32(&sector[80..88], "volume_space_size")?;
        let logical_block_size = decode_both_endian_u16(&sector[128..132], "logical_block_size")?;
        let path_table_size = decode_both_endian_u32(&sector[132..140], "path_table_size")?;
        let l_path_table_lba = u32::from_le_bytes(sector[140..144].try_into().unwrap());
        let m_path_table_lba = u32::from_be_bytes(sector[148..152].try_into().unwrap());

        // Root directory record is exactly 34 bytes at offset 156.
        let root = DirRecord::decode(&sector[156..156 + 34])?;
        Ok(Self {
            system_id,
            volume_id,
            volume_space_size,
            logical_block_size,
            path_table_size,
            l_path_table_lba,
            m_path_table_lba,
            root,
        })
    }
}

/// Supplementary Volume Descriptor — same shape as the PVD plus the
/// 32-byte escape sequence at offset 88. Joliet is signalled by one of
/// `%/@` / `%/C` / `%/E` (UCS-2 levels 1/2/3 respectively).
#[derive(Debug, Clone)]
pub struct SupplementaryVolumeDescriptor {
    pub escape_sequences: [u8; 32],
    pub volume_id: String,
    pub volume_space_size: u32,
    pub logical_block_size: u16,
    pub path_table_size: u32,
    pub l_path_table_lba: u32,
    pub m_path_table_lba: u32,
    pub root: DirRecord,
}

impl SupplementaryVolumeDescriptor {
    fn decode(sector: &[u8]) -> Result<Self> {
        // Volume identifier in an SVD is UCS-2 big-endian — Joliet
        // §3.4.2.2. We convert here.
        let volume_id = super::joliet::ucs2_be_to_string(&sector[40..72]);
        let mut escape_sequences = [0u8; 32];
        escape_sequences.copy_from_slice(&sector[88..120]);
        let volume_space_size = decode_both_endian_u32(&sector[80..88], "volume_space_size")?;
        let logical_block_size = decode_both_endian_u16(&sector[128..132], "logical_block_size")?;
        let path_table_size = decode_both_endian_u32(&sector[132..140], "path_table_size")?;
        let l_path_table_lba = u32::from_le_bytes(sector[140..144].try_into().unwrap());
        let m_path_table_lba = u32::from_be_bytes(sector[148..152].try_into().unwrap());
        let root = DirRecord::decode(&sector[156..156 + 34])?;
        Ok(Self {
            escape_sequences,
            volume_id,
            volume_space_size,
            logical_block_size,
            path_table_size,
            l_path_table_lba,
            m_path_table_lba,
            root,
        })
    }

    /// `true` when the escape sequences identify a Joliet supplement.
    pub fn is_joliet(&self) -> bool {
        let seq = &self.escape_sequences;
        // Joliet UCS-2 levels 1/2/3 are "%/@", "%/C", "%/E".
        seq.starts_with(b"%/@") || seq.starts_with(b"%/C") || seq.starts_with(b"%/E")
    }
}

/// ECMA-119 §7.3.1 — both-endian u32: little-endian u32 followed by the
/// same number big-endian. The two halves must agree.
pub(crate) fn decode_both_endian_u32(buf: &[u8], label: &str) -> Result<u32> {
    if buf.len() < 8 {
        return Err(crate::Error::InvalidImage(format!(
            "iso9660: short both-endian u32 for {label}"
        )));
    }
    let le = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let be = u32::from_be_bytes(buf[4..8].try_into().unwrap());
    if le != be {
        return Err(crate::Error::InvalidImage(format!(
            "iso9660: both-endian {label} mismatch ({le} != {be})"
        )));
    }
    Ok(le)
}

pub(crate) fn decode_both_endian_u16(buf: &[u8], label: &str) -> Result<u16> {
    if buf.len() < 4 {
        return Err(crate::Error::InvalidImage(format!(
            "iso9660: short both-endian u16 for {label}"
        )));
    }
    let le = u16::from_le_bytes(buf[0..2].try_into().unwrap());
    let be = u16::from_be_bytes(buf[2..4].try_into().unwrap());
    if le != be {
        return Err(crate::Error::InvalidImage(format!(
            "iso9660: both-endian {label} mismatch ({le} != {be})"
        )));
    }
    Ok(le)
}

/// Strip trailing space padding (ISO 9660 strD/strA padding). Also
/// drops any trailing NULs that mkisofs sometimes leaves behind.
fn trim_strd(buf: &[u8]) -> String {
    let end = buf
        .iter()
        .rposition(|&b| b != b' ' && b != 0)
        .map(|p| p + 1)
        .unwrap_or(0);
    String::from_utf8_lossy(&buf[..end]).into_owned()
}
