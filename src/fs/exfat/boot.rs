//! exFAT main boot sector (a.k.a. Main Boot Sector / MBS).
//!
//! Layout per Microsoft's published exFAT specification (2019). Every
//! multi-byte field is little-endian.
//!
//! ```text
//!   off  size  name
//!     0     3  JumpBoot                  (EB 76 90)
//!     3     8  FileSystemName            ("EXFAT   ")
//!    11    53  MustBeZero
//!    64     8  PartitionOffset           (sectors; advisory)
//!    72     8  VolumeLength              (in sectors)
//!    80     4  FatOffset                 (sectors from volume start)
//!    84     4  FatLength                 (sectors per FAT)
//!    88     4  ClusterHeapOffset         (sectors from volume start)
//!    92     4  ClusterCount
//!    96     4  FirstClusterOfRootDirectory
//!   100     4  VolumeSerialNumber
//!   104     2  FileSystemRevision        (high = major, low = minor)
//!   106     2  VolumeFlags
//!   108     1  BytesPerSectorShift       (power of 2: 9..=12 → 512..4096)
//!   109     1  SectorsPerClusterShift    (power of 2: 0..=25-BPSshift)
//!   110     1  NumberOfFats              (1 or 2; TexFAT uses 2)
//!   111     1  DriveSelect
//!   112     1  PercentInUse
//!   113     7  Reserved
//!   120   390  BootCode
//!   510     2  BootSignature             (0x55 0xAA)
//! ```
//!
//! exFAT actually uses a 12-sector "boot region" (main + 8 extension +
//! OEM + reserved + checksum), with an identical backup at sectors 12..23.
//! We only need the first sector for read-only support.

/// On-disk size of the main boot sector in this implementation. exFAT
/// permits 512..=4096 byte sectors; the boot sector is always exactly one
/// sector wide and parsing reads only the first 512 bytes (every field we
/// care about lives in the first 120 bytes).
pub const BOOT_SECTOR_PARSE_SIZE: usize = 512;

/// Parsed exFAT boot sector. Field names follow the Microsoft spec.
#[derive(Debug, Clone)]
pub struct BootSector {
    pub partition_offset: u64,
    pub volume_length: u64,
    pub fat_offset: u32,
    pub fat_length: u32,
    pub cluster_heap_offset: u32,
    pub cluster_count: u32,
    pub first_cluster_of_root_directory: u32,
    pub volume_serial_number: u32,
    pub fs_revision_major: u8,
    pub fs_revision_minor: u8,
    pub volume_flags: u16,
    pub bytes_per_sector_shift: u8,
    pub sectors_per_cluster_shift: u8,
    pub number_of_fats: u8,
    pub drive_select: u8,
    pub percent_in_use: u8,
}

impl BootSector {
    /// Bytes per sector — 2^BytesPerSectorShift.
    pub fn bytes_per_sector(&self) -> u32 {
        1u32 << self.bytes_per_sector_shift
    }

    /// Sectors per cluster — 2^SectorsPerClusterShift.
    pub fn sectors_per_cluster(&self) -> u32 {
        1u32 << self.sectors_per_cluster_shift
    }

    /// Bytes per cluster.
    pub fn bytes_per_cluster(&self) -> u32 {
        self.bytes_per_sector() << self.sectors_per_cluster_shift
    }

    /// Byte offset of the first FAT.
    pub fn fat_byte_offset(&self) -> u64 {
        self.fat_offset as u64 * self.bytes_per_sector() as u64
    }

    /// Byte size of one FAT.
    pub fn fat_byte_length(&self) -> u64 {
        self.fat_length as u64 * self.bytes_per_sector() as u64
    }

    /// Byte offset of the cluster heap (data area; cluster 2 starts here).
    pub fn cluster_heap_byte_offset(&self) -> u64 {
        self.cluster_heap_offset as u64 * self.bytes_per_sector() as u64
    }

    /// Byte offset of cluster `n` (n >= 2).
    pub fn cluster_byte_offset(&self, cluster: u32) -> u64 {
        self.cluster_heap_byte_offset() + (cluster as u64 - 2) * self.bytes_per_cluster() as u64
    }

    /// Decode from the first 512 bytes of LBA 0.
    pub fn decode(b: &[u8; BOOT_SECTOR_PARSE_SIZE]) -> crate::Result<Self> {
        if &b[3..11] != b"EXFAT   " {
            return Err(crate::Error::InvalidImage(
                "exfat: missing \"EXFAT   \" signature at offset 3".into(),
            ));
        }
        // MustBeZero (offset 11..64) must be all zeros.
        if b[11..64].iter().any(|&x| x != 0) {
            return Err(crate::Error::InvalidImage(
                "exfat: MustBeZero region is non-zero".into(),
            ));
        }
        if b[510] != 0x55 || b[511] != 0xAA {
            return Err(crate::Error::InvalidImage(
                "exfat: missing 0x55AA boot-sector signature".into(),
            ));
        }
        let bytes_per_sector_shift = b[108];
        let sectors_per_cluster_shift = b[109];
        if !(9..=12).contains(&bytes_per_sector_shift) {
            return Err(crate::Error::InvalidImage(format!(
                "exfat: invalid BytesPerSectorShift {bytes_per_sector_shift} (must be 9..=12)"
            )));
        }
        if (bytes_per_sector_shift as u32 + sectors_per_cluster_shift as u32) > 25 {
            return Err(crate::Error::InvalidImage(format!(
                "exfat: BytesPerSectorShift + SectorsPerClusterShift = {} exceeds 25",
                bytes_per_sector_shift as u32 + sectors_per_cluster_shift as u32
            )));
        }
        let number_of_fats = b[110];
        if number_of_fats != 1 && number_of_fats != 2 {
            return Err(crate::Error::InvalidImage(format!(
                "exfat: NumberOfFats {number_of_fats} (must be 1 or 2)"
            )));
        }
        let fs_revision = u16::from_le_bytes(b[104..106].try_into().unwrap());
        Ok(Self {
            partition_offset: u64::from_le_bytes(b[64..72].try_into().unwrap()),
            volume_length: u64::from_le_bytes(b[72..80].try_into().unwrap()),
            fat_offset: u32::from_le_bytes(b[80..84].try_into().unwrap()),
            fat_length: u32::from_le_bytes(b[84..88].try_into().unwrap()),
            cluster_heap_offset: u32::from_le_bytes(b[88..92].try_into().unwrap()),
            cluster_count: u32::from_le_bytes(b[92..96].try_into().unwrap()),
            first_cluster_of_root_directory: u32::from_le_bytes(b[96..100].try_into().unwrap()),
            volume_serial_number: u32::from_le_bytes(b[100..104].try_into().unwrap()),
            fs_revision_major: (fs_revision >> 8) as u8,
            fs_revision_minor: (fs_revision & 0xff) as u8,
            volume_flags: u16::from_le_bytes(b[106..108].try_into().unwrap()),
            bytes_per_sector_shift,
            sectors_per_cluster_shift,
            number_of_fats,
            drive_select: b[111],
            percent_in_use: b[112],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-craft a 512-byte boot sector for round-trip testing. Mirrors a
    /// 64 MiB volume with 512-byte sectors and 4 KiB clusters.
    fn make_boot_sector() -> [u8; BOOT_SECTOR_PARSE_SIZE] {
        let mut b = [0u8; BOOT_SECTOR_PARSE_SIZE];
        b[0..3].copy_from_slice(&[0xEB, 0x76, 0x90]);
        b[3..11].copy_from_slice(b"EXFAT   ");
        // 11..64 is MustBeZero — leave as zero.
        b[64..72].copy_from_slice(&0u64.to_le_bytes()); // partition_offset
        b[72..80].copy_from_slice(&131072u64.to_le_bytes()); // volume_length (64 MiB / 512)
        b[80..84].copy_from_slice(&2048u32.to_le_bytes()); // fat_offset
        b[84..88].copy_from_slice(&128u32.to_le_bytes()); // fat_length
        b[88..92].copy_from_slice(&2304u32.to_le_bytes()); // cluster_heap_offset
        b[92..96].copy_from_slice(&16_096u32.to_le_bytes()); // cluster_count
        b[96..100].copy_from_slice(&5u32.to_le_bytes()); // first_cluster_of_root_directory
        b[100..104].copy_from_slice(&0xCAFE_F00Du32.to_le_bytes()); // volume_serial
        b[104..106].copy_from_slice(&0x0100u16.to_le_bytes()); // revision = 1.0
        b[106..108].copy_from_slice(&0u16.to_le_bytes()); // volume_flags
        b[108] = 9; // BytesPerSectorShift = 9 → 512
        b[109] = 3; // SectorsPerClusterShift = 3 → 8 sectors/cluster = 4 KiB
        b[110] = 1; // NumberOfFats
        b[111] = 0x80; // DriveSelect
        b[112] = 0; // PercentInUse
        b[510] = 0x55;
        b[511] = 0xAA;
        b
    }

    #[test]
    fn decode_handcrafted() {
        let b = make_boot_sector();
        let bs = BootSector::decode(&b).unwrap();
        assert_eq!(bs.bytes_per_sector(), 512);
        assert_eq!(bs.sectors_per_cluster(), 8);
        assert_eq!(bs.bytes_per_cluster(), 4096);
        assert_eq!(bs.fat_offset, 2048);
        assert_eq!(bs.fat_length, 128);
        assert_eq!(bs.cluster_heap_offset, 2304);
        assert_eq!(bs.cluster_count, 16_096);
        assert_eq!(bs.first_cluster_of_root_directory, 5);
        assert_eq!(bs.volume_serial_number, 0xCAFE_F00D);
        assert_eq!(bs.fs_revision_major, 1);
        assert_eq!(bs.fs_revision_minor, 0);
        assert_eq!(bs.number_of_fats, 1);
        // Cluster 2 begins at cluster_heap_offset.
        assert_eq!(bs.cluster_byte_offset(2), 2304 * 512);
        assert_eq!(bs.cluster_byte_offset(3), 2304 * 512 + 4096);
    }

    #[test]
    fn rejects_missing_magic() {
        let mut b = make_boot_sector();
        b[3] = b'X';
        assert!(BootSector::decode(&b).is_err());
    }

    #[test]
    fn rejects_bad_signature() {
        let mut b = make_boot_sector();
        b[510] = 0;
        assert!(BootSector::decode(&b).is_err());
    }

    #[test]
    fn rejects_nonzero_must_be_zero() {
        let mut b = make_boot_sector();
        b[20] = 0xFF;
        assert!(BootSector::decode(&b).is_err());
    }

    #[test]
    fn rejects_invalid_bps_shift() {
        let mut b = make_boot_sector();
        b[108] = 13; // > 12
        assert!(BootSector::decode(&b).is_err());
    }
}
