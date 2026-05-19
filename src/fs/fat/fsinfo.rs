//! FAT32 FSInfo sector.
//!
//! The FSInfo sector (sector 1 by convention) caches the free-cluster
//! count and a "next free cluster" search hint, so a driver doesn't have
//! to scan the whole FAT on mount. The values are advisory — a clean FAT32
//! must still have them consistent, which fsck.vfat checks.
//!
//! ```text
//!     0   4   lead_signature   = 0x41615252  ("RRaA")
//!     4 480   reserved (zero)
//!   484   4   struct_signature = 0x61417272  ("rrAa")
//!   488   4   free_count
//!   492   4   next_free
//!   496  12   reserved (zero)
//!   508   4   trail_signature  = 0xAA550000
//! ```

/// Size of the FSInfo sector.
pub const FSINFO_SIZE: usize = 512;

const LEAD_SIG: u32 = 0x4161_5252;
const STRUCT_SIG: u32 = 0x6141_7272;
const TRAIL_SIG: u32 = 0xAA55_0000;

/// The two cached values in an FSInfo sector.
#[derive(Debug, Clone, Copy)]
pub struct FsInfo {
    /// Number of free data clusters.
    pub free_count: u32,
    /// Cluster number to start the next free-cluster search from.
    pub next_free: u32,
}

impl FsInfo {
    /// Encode into the 512-byte on-disk FSInfo sector.
    pub fn encode(&self) -> [u8; FSINFO_SIZE] {
        let mut b = [0u8; FSINFO_SIZE];
        b[0..4].copy_from_slice(&LEAD_SIG.to_le_bytes());
        b[484..488].copy_from_slice(&STRUCT_SIG.to_le_bytes());
        b[488..492].copy_from_slice(&self.free_count.to_le_bytes());
        b[492..496].copy_from_slice(&self.next_free.to_le_bytes());
        b[508..512].copy_from_slice(&TRAIL_SIG.to_le_bytes());
        b
    }

    /// Decode from a 512-byte FSInfo sector, validating the three signatures.
    pub fn decode(b: &[u8; FSINFO_SIZE]) -> crate::Result<Self> {
        let lead = u32::from_le_bytes(b[0..4].try_into().unwrap());
        let strukt = u32::from_le_bytes(b[484..488].try_into().unwrap());
        let trail = u32::from_le_bytes(b[508..512].try_into().unwrap());
        if lead != LEAD_SIG || strukt != STRUCT_SIG || trail != TRAIL_SIG {
            return Err(crate::Error::InvalidImage(
                "fat32: bad FSInfo signature".into(),
            ));
        }
        Ok(Self {
            free_count: u32::from_le_bytes(b[488..492].try_into().unwrap()),
            next_free: u32::from_le_bytes(b[492..496].try_into().unwrap()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let fi = FsInfo {
            free_count: 129021,
            next_free: 3,
        };
        let dec = FsInfo::decode(&fi.encode()).unwrap();
        assert_eq!(dec.free_count, 129021);
        assert_eq!(dec.next_free, 3);
    }

    #[test]
    fn signatures_checked() {
        assert!(FsInfo::decode(&[0u8; FSINFO_SIZE]).is_err());
    }
}
