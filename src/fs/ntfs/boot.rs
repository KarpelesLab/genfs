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

/// Largest cluster size we accept (bytes). Real NTFS tops out at 2 MiB
/// clusters; we use that as the ceiling for `bytes_per_sector *
/// sectors_per_cluster` so a malicious BPB cannot drive multi-MiB-per-cluster
/// allocations.
pub(crate) const MAX_CLUSTER_SIZE: u32 = 2 * 1024 * 1024;

/// Minimum / maximum MFT and index record sizes (bytes). NTFS records are
/// almost always 1024 bytes; we allow the full 256 B .. 1 MiB range that the
/// on-disk encoding can express and reject anything outside it.
pub(crate) const MIN_RECORD_SIZE: u32 = 256;
pub(crate) const MAX_RECORD_SIZE: u32 = 1024 * 1024;

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

        // Validate geometry up front so every downstream consumer (cluster /
        // record sizing, divide-by, shifts) operates on sane values. A
        // malicious image can put anything here, so reject rather than panic.
        //
        // `bytes_per_sector`: power of two in 256..=4096.
        if !(256..=4096).contains(&bytes_per_sector) || !bytes_per_sector.is_power_of_two() {
            return None;
        }
        // `sectors_per_cluster`: power of two >= 1. (Values >= 0x80 encode
        // huge clusters via 2^(256-value) on real NTFS, but we keep the
        // simple linear interpretation and bound the product below.)
        if sectors_per_cluster == 0 || !sectors_per_cluster.is_power_of_two() {
            return None;
        }
        // Resulting cluster size must not overflow u32 and must stay within a
        // sane ceiling.
        let cluster_size =
            u32::from(bytes_per_sector).checked_mul(u32::from(sectors_per_cluster))?;
        if cluster_size > MAX_CLUSTER_SIZE {
            return None;
        }
        // MFT / index record sizes must resolve to a sane byte range.
        let mft_rec = Self::record_size(clusters_per_mft_record, cluster_size)?;
        let idx_rec = Self::record_size(clusters_per_index_record, cluster_size)?;
        if !(MIN_RECORD_SIZE..=MAX_RECORD_SIZE).contains(&mft_rec)
            || !(MIN_RECORD_SIZE..=MAX_RECORD_SIZE).contains(&idx_rec)
        {
            return None;
        }

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
        // Geometry is validated in `decode`, so this always resolves.
        Self::record_size(self.clusters_per_mft_record, self.cluster_size())
            .expect("mft_record_size validated in BootSector::decode")
    }

    /// Resolve the index record size in bytes from the BPB field.
    pub fn index_record_size(&self) -> u32 {
        Self::record_size(self.clusters_per_index_record, self.cluster_size())
            .expect("index_record_size validated in BootSector::decode")
    }

    /// Compute a record size in bytes from a `clusters_per_*_record` field.
    /// Returns `None` on arithmetic overflow (negative field shifting past
    /// 31, or positive field times `cluster_size` overflowing u32).
    fn record_size(field: i8, cluster_size: u32) -> Option<u32> {
        if field >= 0 {
            (field as u32).checked_mul(cluster_size)
        } else {
            // Negative field: record_size = 1 << -field. `-field` is in
            // 1..=128; `checked_shl` rejects shifts >= 32.
            1u32.checked_shl((-(field as i32)) as u32)
        }
    }

    /// Cluster size = sector size * sectors-per-cluster.
    pub fn cluster_size(&self) -> u32 {
        u32::from(self.bytes_per_sector) * u32::from(self.sectors_per_cluster)
    }
}
