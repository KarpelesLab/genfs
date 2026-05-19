//! The FAT itself — the cluster allocation table.
//!
//! A FAT32 FAT is a flat array of 32-bit entries indexed by cluster
//! number. Each entry holds the *next* cluster in that file's chain, or an
//! end-of-chain marker, or 0 for a free cluster. Only the low 28 bits are
//! meaningful; the top 4 are reserved.
//!
//! Entries 0 and 1 are reserved: entry 0 holds the media byte in its low
//! 8 bits (0x0FFFFFF8), entry 1 is the end-of-chain value and also carries
//! the "volume clean" / "no hard error" status bits — a freshly formatted
//! volume uses 0x0FFFFFFF.

/// Minimum value that counts as an end-of-chain marker.
pub const EOC_MIN: u32 = 0x0FFF_FFF8;
/// The end-of-chain value genfs writes.
pub const EOC: u32 = 0x0FFF_FFFF;
/// A free cluster.
pub const FREE: u32 = 0;
/// Low 28 bits — the meaningful part of a FAT32 entry.
pub const ENTRY_MASK: u32 = 0x0FFF_FFFF;

/// An in-memory FAT. `entries.len()` is the table's full entry capacity
/// (`fat_size_sectors * bytes_per_sector / 4`); only indices
/// `0..cluster_count + 2` correspond to real clusters.
#[derive(Debug, Clone)]
pub struct Fat {
    entries: Vec<u32>,
}

impl Fat {
    /// A fresh FAT with `capacity` entries, all free, with the two
    /// reserved entries (0 and 1) initialised for `media`.
    pub fn new(capacity: usize, media: u8) -> Self {
        let mut entries = vec![FREE; capacity];
        entries[0] = 0x0FFF_FF00 | media as u32;
        entries[1] = EOC;
        Self { entries }
    }

    /// Total entry capacity.
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Read the entry for `cluster` (low 28 bits).
    pub fn get(&self, cluster: u32) -> u32 {
        self.entries[cluster as usize] & ENTRY_MASK
    }

    /// Set the entry for `cluster`. Only the low 28 bits are stored; the
    /// reserved top 4 bits are kept zero.
    pub fn set(&mut self, cluster: u32, value: u32) {
        self.entries[cluster as usize] = value & ENTRY_MASK;
    }

    /// Whether `value` marks the end of a cluster chain.
    pub fn is_eoc(value: u32) -> bool {
        value >= EOC_MIN
    }

    /// Encode into the on-disk byte image of one FAT copy.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.entries.len() * 4];
        for (i, &e) in self.entries.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&e.to_le_bytes());
        }
        out
    }

    /// Decode from the on-disk byte image of one FAT copy.
    pub fn decode(bytes: &[u8]) -> Self {
        let entries = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Self { entries }
    }

    /// Follow the cluster chain starting at `start`, returning every
    /// cluster in order. Stops at an end-of-chain marker; returns an error
    /// on a free/zero entry mid-chain or an obvious loop.
    pub fn chain(&self, start: u32) -> crate::Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut cur = start;
        while !Self::is_eoc(cur) {
            if cur < 2 || cur as usize >= self.entries.len() {
                return Err(crate::Error::InvalidImage(format!(
                    "fat32: cluster {cur} out of range while walking a chain"
                )));
            }
            if out.len() > self.entries.len() {
                return Err(crate::Error::InvalidImage(
                    "fat32: cluster chain loops".into(),
                ));
            }
            out.push(cur);
            cur = self.get(cur);
            if cur == FREE {
                return Err(crate::Error::InvalidImage(
                    "fat32: cluster chain hits a free cluster".into(),
                ));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_entries() {
        let fat = Fat::new(256, 0xF8);
        assert_eq!(fat.get(0), 0x0FFF_FFF8);
        assert_eq!(fat.get(1), EOC);
        assert_eq!(fat.get(2), FREE);
    }

    #[test]
    fn set_get_roundtrip_via_bytes() {
        let mut fat = Fat::new(64, 0xF8);
        // A 3-cluster chain: 2 -> 3 -> 4 -> EOC.
        fat.set(2, 3);
        fat.set(3, 4);
        fat.set(4, EOC);
        let decoded = Fat::decode(&fat.encode());
        assert_eq!(decoded.chain(2).unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn eoc_classification() {
        assert!(Fat::is_eoc(EOC));
        assert!(Fat::is_eoc(0x0FFF_FFF8));
        assert!(!Fat::is_eoc(0x0FFF_FFF7));
        assert!(!Fat::is_eoc(5));
    }

    #[test]
    fn chain_detects_free_break() {
        let mut fat = Fat::new(64, 0xF8);
        fat.set(2, 3); // 3 is still FREE
        assert!(fat.chain(2).is_err());
    }
}
