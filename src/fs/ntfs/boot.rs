//! NTFS boot sector / BIOS Parameter Block (BPB) decode.
//!
//! The boot sector lives at LBA 0 and combines a DOS-style jump+BPB header
//! with NTFS-specific extended fields. The crucial values for the driver
//! are the cluster geometry, the location of the MFT, and the way MFT /
//! index records are sized (negative `clusters_per_*_record` means "raise
//! 2 to the |value|" rather than counting clusters).

/// First 8 bytes of an NTFS boot sector's OEM ID. The 3-byte jump at
/// offset 0 (`0xEB 0x52 0x90`) is shared with FAT/exFAT, so we anchor on
/// the OEM string at offset 3.
pub(crate) const NTFS_OEM: &[u8; 8] = b"NTFS    ";

/// Decoded NTFS boot-sector / BPB fields. Multi-byte fields are LE.
#[derive(Debug, Clone)]
pub struct BootSector {
    pub bytes_per_sector: u16,
    pub sectors_per_cluster: u8,
    /// `total_sectors` from BPB: total sectors in volume - 1.
    pub total_sectors: u64,
    /// Logical cluster number of the MFT's first record.
    pub mft_lcn: u64,
    /// Logical cluster number of the MFT mirror's first record.
    pub mft_mirr_lcn: u64,
    /// `clusters_per_mft_record`. Positive: clusters per record. Negative
    /// (interpreted as signed i8): `record_size = 1 << -value` bytes
    /// (used when records are smaller than a cluster, e.g. 1024 bytes).
    pub clusters_per_mft_record: i8,
    pub clusters_per_index_record: i8,
    pub volume_serial: u64,
}

impl BootSector {
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 80 || &buf[3..11] != NTFS_OEM {
            return None;
        }
        let bytes_per_sector = u16::from_le_bytes(buf[11..13].try_into().ok()?);
        let sectors_per_cluster = buf[13];
        let total_sectors = u64::from_le_bytes(buf[0x28..0x30].try_into().ok()?);
        let mft_lcn = u64::from_le_bytes(buf[0x30..0x38].try_into().ok()?);
        let mft_mirr_lcn = u64::from_le_bytes(buf[0x38..0x40].try_into().ok()?);
        let clusters_per_mft_record = buf[0x40] as i8;
        let clusters_per_index_record = buf[0x44] as i8;
        let volume_serial = u64::from_le_bytes(buf[0x48..0x50].try_into().ok()?);
        Some(Self {
            bytes_per_sector,
            sectors_per_cluster,
            total_sectors,
            mft_lcn,
            mft_mirr_lcn,
            clusters_per_mft_record,
            clusters_per_index_record,
            volume_serial,
        })
    }

    /// Resolve the MFT record size in bytes from the BPB field. Either
    /// `clusters_per_mft_record * cluster_size` (positive) or
    /// `1 << -value` bytes (negative; the canonical 1024-byte case).
    pub fn mft_record_size(&self) -> u32 {
        Self::record_size(
            self.clusters_per_mft_record,
            u32::from(self.bytes_per_sector) * u32::from(self.sectors_per_cluster),
        )
    }

    /// Resolve the index record size in bytes from the BPB field.
    pub fn index_record_size(&self) -> u32 {
        Self::record_size(
            self.clusters_per_index_record,
            u32::from(self.bytes_per_sector) * u32::from(self.sectors_per_cluster),
        )
    }

    fn record_size(field: i8, cluster_size: u32) -> u32 {
        if field >= 0 {
            (field as u32) * cluster_size
        } else {
            1u32 << (-(field as i32))
        }
    }

    /// Cluster size = sector size * sectors-per-cluster.
    pub fn cluster_size(&self) -> u32 {
        u32::from(self.bytes_per_sector) * u32::from(self.sectors_per_cluster)
    }
}
