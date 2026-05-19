//! exFAT File Allocation Table — 32-bit entries.
//!
//! Each entry maps a cluster number to either the next cluster in its
//! chain or to one of three sentinels:
//!
//! ```text
//!     0x00000000   FREE         cluster is unallocated
//!     0xFFFFFFF7   BAD          cluster is marked defective
//!     0xFFFFFFFF   EOC          end-of-chain marker
//!     other        NEXT(value)  follow to cluster `value`
//! ```
//!
//! Entries 0 and 1 are reserved: entry 0 holds 0xFFFFFFF8 | media (commonly
//! 0xFFFFFFF8), entry 1 is the end-of-chain value 0xFFFFFFFF. The "no FAT
//! chain" bit in a StreamExtension entry means a file's clusters are stored
//! contiguously and the FAT entries are not required to be valid — the
//! reader must follow the contiguous run instead of the FAT.

/// Free cluster.
pub const FREE: u32 = 0x0000_0000;
/// Bad-cluster marker.
pub const BAD: u32 = 0xFFFF_FFF7;
/// End-of-chain marker.
pub const EOC: u32 = 0xFFFF_FFFF;

/// Classification of one FAT entry's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FatEntry {
    Free,
    Bad,
    Eoc,
    /// The next cluster in this chain.
    Next(u32),
}

/// Classify a raw 32-bit FAT entry value.
pub fn classify(value: u32) -> FatEntry {
    match value {
        FREE => FatEntry::Free,
        BAD => FatEntry::Bad,
        EOC => FatEntry::Eoc,
        // Anything in 0xFFFFFFF8..=0xFFFFFFFE is reserved per the spec but
        // commonly treated as "end of chain" by implementations. Per the
        // Microsoft spec only 0xFFFFFFFF is EOC; we honour that strictly
        // and treat the reserved values as Next() to surface broken
        // images rather than silently following them.
        n => FatEntry::Next(n),
    }
}

/// An in-memory copy of (the part of) the FAT we need to walk chains.
///
/// `entries.len()` may exceed `cluster_count + 2` because the FAT is
/// allocated in whole sectors; only indices `[0, cluster_count + 2)` are
/// meaningful (the rest are typically zero).
#[derive(Debug, Clone)]
pub struct Fat {
    entries: Vec<u32>,
}

impl Fat {
    /// Decode a byte image of one FAT copy.
    pub fn decode(bytes: &[u8]) -> Self {
        let entries = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Self { entries }
    }

    /// Build a fresh FAT with `capacity` 32-bit entries. Entry 0 is set to
    /// the media descriptor (0xFFFFFFF8) and entry 1 to EOC, per the spec.
    /// All other entries are FREE.
    pub fn new_blank(capacity: usize) -> Self {
        assert!(capacity >= 2, "FAT must hold at least two reserved entries");
        let mut entries = vec![FREE; capacity];
        entries[0] = 0xFFFF_FFF8;
        entries[1] = EOC;
        Self { entries }
    }

    /// Serialise the FAT to a byte vector exactly `entries.len() * 4` bytes long.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.entries.len() * 4);
        for e in &self.entries {
            out.extend_from_slice(&e.to_le_bytes());
        }
        out
    }

    /// Raw 32-bit entry for `cluster`.
    pub fn raw(&self, cluster: u32) -> Option<u32> {
        self.entries.get(cluster as usize).copied()
    }

    /// Set the raw 32-bit value of `cluster`.
    pub fn set_raw(&mut self, cluster: u32, value: u32) {
        if let Some(slot) = self.entries.get_mut(cluster as usize) {
            *slot = value;
        }
    }

    /// Classified entry for `cluster`.
    pub fn get(&self, cluster: u32) -> Option<FatEntry> {
        self.raw(cluster).map(classify)
    }

    /// Number of entries the in-memory FAT holds (including the two
    /// reserved low entries).
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Walk the cluster chain starting at `start` and return every cluster
    /// in order. Stops at an EOC marker. Errors on out-of-range, FREE-in-
    /// chain, BAD-in-chain, or a chain that exceeds the FAT's capacity
    /// (loop guard).
    pub fn chain(&self, start: u32) -> crate::Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut cur = start;
        loop {
            if cur < 2 || (cur as usize) >= self.entries.len() {
                return Err(crate::Error::InvalidImage(format!(
                    "exfat: cluster {cur} out of range while walking a chain"
                )));
            }
            if out.len() > self.entries.len() {
                return Err(crate::Error::InvalidImage(
                    "exfat: cluster chain loops".into(),
                ));
            }
            out.push(cur);
            match classify(self.entries[cur as usize]) {
                FatEntry::Eoc => return Ok(out),
                FatEntry::Free => {
                    return Err(crate::Error::InvalidImage(
                        "exfat: cluster chain hits a FREE entry".into(),
                    ));
                }
                FatEntry::Bad => {
                    return Err(crate::Error::InvalidImage(
                        "exfat: cluster chain hits a BAD entry".into(),
                    ));
                }
                FatEntry::Next(n) => cur = n,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_known_values() {
        assert_eq!(classify(0), FatEntry::Free);
        assert_eq!(classify(0xFFFF_FFF7), FatEntry::Bad);
        assert_eq!(classify(0xFFFF_FFFF), FatEntry::Eoc);
        assert_eq!(classify(7), FatEntry::Next(7));
    }

    /// Build an in-memory FAT for `entries` 32-bit values.
    fn build(entries: &[u32]) -> Fat {
        let mut bytes = Vec::with_capacity(entries.len() * 4);
        for e in entries {
            bytes.extend_from_slice(&e.to_le_bytes());
        }
        Fat::decode(&bytes)
    }

    #[test]
    fn chain_walks_to_eoc() {
        // [0]=media, [1]=EOC, [2]→3, [3]→5, [4]=FREE, [5]=EOC.
        let fat = build(&[0xFFFF_FFF8, 0xFFFF_FFFF, 3, 5, FREE, EOC]);
        assert_eq!(fat.chain(2).unwrap(), vec![2, 3, 5]);
    }

    #[test]
    fn chain_rejects_free_break() {
        let fat = build(&[0xFFFF_FFF8, EOC, 3, FREE]);
        assert!(fat.chain(2).is_err());
    }

    #[test]
    fn chain_rejects_bad_in_chain() {
        let fat = build(&[0xFFFF_FFF8, EOC, 3, BAD]);
        assert!(fat.chain(2).is_err());
    }

    #[test]
    fn chain_rejects_out_of_range_start() {
        let fat = build(&[0xFFFF_FFF8, EOC, EOC]);
        // cluster 5 is past capacity.
        assert!(fat.chain(5).is_err());
        // cluster 1 is reserved and not a valid start.
        assert!(fat.chain(1).is_err());
    }
}
