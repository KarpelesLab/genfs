//! FAT32 boot sector + BIOS Parameter Block (BPB).
//!
//! The boot sector is sector 0 (512 bytes). FAT32 also keeps a backup copy
//! at sector 6. Layout (offsets in bytes, all multi-byte fields little-
//! endian) per the public Microsoft FAT specification:
//!
//! ```text
//!     0   3   jump instruction (EB 58 90)
//!     3   8   OEM name
//!    11   2   bytes_per_sector
//!    13   1   sectors_per_cluster
//!    14   2   reserved_sector_count   (32 for FAT32)
//!    16   1   num_fats                (2)
//!    17   2   root_entry_count        (0 for FAT32)
//!    19   2   total_sectors_16        (0 for FAT32 — see total_sectors_32)
//!    21   1   media                   (0xF8)
//!    22   2   fat_size_16             (0 for FAT32 — see fat_size_32)
//!    24   2   sectors_per_track
//!    26   2   num_heads
//!    28   4   hidden_sectors
//!    32   4   total_sectors_32
//!    36   4   fat_size_32             (sectors per FAT)
//!    40   2   ext_flags
//!    42   2   fs_version              (0)
//!    44   4   root_cluster            (usually 2)
//!    48   2   fs_info_sector          (1)
//!    50   2   backup_boot_sector      (6)
//!    52  12   reserved
//!    64   1   drive_number
//!    65   1   reserved1
//!    66   1   boot_signature          (0x29)
//!    67   4   volume_id
//!    71  11   volume_label
//!    82   8   fs_type                 ("FAT32   ")
//!   510   2   0x55 0xAA
//! ```

/// Bytes in a boot sector.
pub const BOOT_SECTOR_SIZE: usize = 512;

/// Fields of a FAT32 boot sector. Only the values genfs needs to set or
/// read are modelled; boot code is left zero (we produce data images, not
/// bootable media).
#[derive(Debug, Clone)]
pub struct BootSector {
    pub bytes_per_sector: u16,
    pub sectors_per_cluster: u8,
    pub reserved_sector_count: u16,
    pub num_fats: u8,
    pub media: u8,
    pub sectors_per_track: u16,
    pub num_heads: u16,
    pub hidden_sectors: u32,
    pub total_sectors: u32,
    pub fat_size: u32,
    pub root_cluster: u32,
    pub fs_info_sector: u16,
    pub backup_boot_sector: u16,
    pub drive_number: u8,
    pub volume_id: u32,
    pub volume_label: [u8; 11],
}

impl BootSector {
    /// A BootSector with FAT32-conventional fixed fields and the rest zero.
    /// Caller fills `sectors_per_cluster`, `total_sectors`, `fat_size`.
    pub fn fat32_default() -> Self {
        Self {
            bytes_per_sector: 512,
            sectors_per_cluster: 1,
            reserved_sector_count: 32,
            num_fats: 2,
            media: 0xF8,
            sectors_per_track: 32,
            num_heads: 8,
            hidden_sectors: 0,
            total_sectors: 0,
            fat_size: 0,
            root_cluster: 2,
            fs_info_sector: 1,
            backup_boot_sector: 6,
            drive_number: 0x80,
            volume_id: 0,
            volume_label: *b"NO NAME    ",
        }
    }

    /// First data sector — where cluster 2 begins. Clusters are numbered
    /// from 2, so cluster `n` starts at `data_start + (n-2)*spc`.
    pub fn data_start_sector(&self) -> u32 {
        self.reserved_sector_count as u32 + self.num_fats as u32 * self.fat_size
    }

    /// Total number of data clusters in the volume.
    pub fn cluster_count(&self) -> u32 {
        let data_sectors = self.total_sectors - self.data_start_sector();
        data_sectors / self.sectors_per_cluster as u32
    }

    /// Encode into the 512-byte on-disk boot sector.
    pub fn encode(&self) -> [u8; BOOT_SECTOR_SIZE] {
        let mut b = [0u8; BOOT_SECTOR_SIZE];
        // Jump instruction + OEM name.
        b[0..3].copy_from_slice(&[0xEB, 0x58, 0x90]);
        b[3..11].copy_from_slice(b"fstool  ");
        b[11..13].copy_from_slice(&self.bytes_per_sector.to_le_bytes());
        b[13] = self.sectors_per_cluster;
        b[14..16].copy_from_slice(&self.reserved_sector_count.to_le_bytes());
        b[16] = self.num_fats;
        // 17..19 root_entry_count = 0; 19..21 total_sectors_16 = 0.
        b[21] = self.media;
        // 22..24 fat_size_16 = 0.
        b[24..26].copy_from_slice(&self.sectors_per_track.to_le_bytes());
        b[26..28].copy_from_slice(&self.num_heads.to_le_bytes());
        b[28..32].copy_from_slice(&self.hidden_sectors.to_le_bytes());
        b[32..36].copy_from_slice(&self.total_sectors.to_le_bytes());
        b[36..40].copy_from_slice(&self.fat_size.to_le_bytes());
        // 40..42 ext_flags = 0; 42..44 fs_version = 0.
        b[44..48].copy_from_slice(&self.root_cluster.to_le_bytes());
        b[48..50].copy_from_slice(&self.fs_info_sector.to_le_bytes());
        b[50..52].copy_from_slice(&self.backup_boot_sector.to_le_bytes());
        b[64] = self.drive_number;
        b[66] = 0x29; // extended boot signature
        b[67..71].copy_from_slice(&self.volume_id.to_le_bytes());
        b[71..82].copy_from_slice(&self.volume_label);
        b[82..90].copy_from_slice(b"FAT32   ");
        b[510] = 0x55;
        b[511] = 0xAA;
        b
    }

    /// Decode from a 512-byte boot sector. Validates the 0x55AA signature
    /// and the FAT32 `fs_type` string.
    pub fn decode(b: &[u8; BOOT_SECTOR_SIZE]) -> crate::Result<Self> {
        if b[510] != 0x55 || b[511] != 0xAA {
            return Err(crate::Error::InvalidImage(
                "fat32: missing 0x55AA boot-sector signature".into(),
            ));
        }
        if &b[82..87] != b"FAT32" {
            return Err(crate::Error::InvalidImage(
                "fat32: fs_type is not \"FAT32\"".into(),
            ));
        }
        let mut volume_label = [0u8; 11];
        volume_label.copy_from_slice(&b[71..82]);
        Ok(Self {
            bytes_per_sector: u16::from_le_bytes(b[11..13].try_into().unwrap()),
            sectors_per_cluster: b[13],
            reserved_sector_count: u16::from_le_bytes(b[14..16].try_into().unwrap()),
            num_fats: b[16],
            media: b[21],
            sectors_per_track: u16::from_le_bytes(b[24..26].try_into().unwrap()),
            num_heads: u16::from_le_bytes(b[26..28].try_into().unwrap()),
            hidden_sectors: u32::from_le_bytes(b[28..32].try_into().unwrap()),
            total_sectors: u32::from_le_bytes(b[32..36].try_into().unwrap()),
            fat_size: u32::from_le_bytes(b[36..40].try_into().unwrap()),
            root_cluster: u32::from_le_bytes(b[44..48].try_into().unwrap()),
            fs_info_sector: u16::from_le_bytes(b[48..50].try_into().unwrap()),
            backup_boot_sector: u16::from_le_bytes(b[50..52].try_into().unwrap()),
            drive_number: b[64],
            volume_id: u32::from_le_bytes(b[67..71].try_into().unwrap()),
            volume_label,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut bs = BootSector::fat32_default();
        bs.total_sectors = 131072;
        bs.fat_size = 1009;
        bs.volume_id = 0x1234_5678;
        bs.volume_label = *b"REFVOL     ";
        let enc = bs.encode();
        let dec = BootSector::decode(&enc).unwrap();
        assert_eq!(dec.total_sectors, 131072);
        assert_eq!(dec.fat_size, 1009);
        assert_eq!(dec.root_cluster, 2);
        assert_eq!(dec.reserved_sector_count, 32);
        assert_eq!(dec.num_fats, 2);
        assert_eq!(dec.volume_id, 0x1234_5678);
        assert_eq!(&dec.volume_label, b"REFVOL     ");
    }

    #[test]
    fn data_start_and_cluster_count() {
        let mut bs = BootSector::fat32_default();
        bs.total_sectors = 131072;
        bs.fat_size = 1009;
        // 32 reserved + 2 * 1009 = 2050.
        assert_eq!(bs.data_start_sector(), 2050);
        // (131072 - 2050) / 1 = 129022 clusters.
        assert_eq!(bs.cluster_count(), 129022);
    }

    #[test]
    fn bad_signature_rejected() {
        let buf = [0u8; BOOT_SECTOR_SIZE];
        assert!(BootSector::decode(&buf).is_err());
    }
}
