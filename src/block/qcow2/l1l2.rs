//! L1 + L2 cluster mapping tables.
//!
//! qcow2's L1/L2 indirection is a two-level page table for the virtual
//! → physical cluster mapping:
//!
//! ```text
//!   cluster_idx = virtual_offset >> cluster_bits
//!   l2_entries  = cluster_size / 8           // u64 entries per L2 cluster
//!   l1_idx      = cluster_idx / l2_entries
//!   l2_idx      = cluster_idx % l2_entries
//! ```
//!
//! - L1 is small (one entry per L2 cluster), loaded in full at open.
//! - L2 is per-table-cluster; we cache the ones we've touched.
//!
//! Entry bit layout (same for L1 and L2):
//!
//! ```text
//!   63       COPIED   refcount == 1, no COW needed
//!   62       COMPRESSED (L2 only; we reject)
//!   9..55    cluster offset
//!   else     reserved (must be 0)
//! ```

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::Result;

use super::header::Header;

/// Set when refcount == 1 — we own the cluster outright.
pub const COPIED: u64 = 1u64 << 63;
/// L2-only: compressed cluster. We reject these.
pub const COMPRESSED: u64 = 1u64 << 62;
/// Mask isolating the cluster-aligned physical byte offset.
pub const OFFSET_MASK: u64 = 0x00FF_FFFF_FFFF_FE00;

/// In-memory L1 + L2 mapping state.
pub struct L1L2 {
    pub cluster_size: u64,
    pub cluster_bits: u32,
    /// Number of u64 entries that fit in one L2 cluster.
    pub l2_entries: usize,
    /// In-memory copy of the L1 table.
    pub l1: Vec<u64>,
    /// Byte offset of the L1 table on disk.
    pub l1_table_offset: u64,
    /// Cached L2 tables, keyed by physical L2 cluster offset.
    pub l2_cache: HashMap<u64, L2Entry>,
    /// L2 cache size cap (number of cached L2 clusters). Old entries
    /// are dropped on insert. Set high enough that linear-scan workloads
    /// don't thrash.
    pub l2_cache_cap: usize,
}

pub struct L2Entry {
    pub entries: Vec<u64>,
    pub dirty: bool,
}

impl L1L2 {
    /// Load the L1 table from disk. The caller is responsible for
    /// seeking — this method reads `header.l1_size` u64 BE entries
    /// starting at `header.l1_table_offset`.
    pub fn load<F: Read + Seek>(file: &mut F, header: &Header) -> Result<Self> {
        let cluster_size = header.cluster_size();
        let l2_entries = (cluster_size / 8) as usize;
        let l1_bytes = header.l1_size as usize * 8;
        file.seek(SeekFrom::Start(header.l1_table_offset))?;
        let mut raw = vec![0u8; l1_bytes];
        file.read_exact(&mut raw)?;
        let mut l1 = Vec::with_capacity(header.l1_size as usize);
        for chunk in raw.chunks_exact(8) {
            l1.push(u64::from_be_bytes(chunk.try_into().unwrap()));
        }
        Ok(Self {
            cluster_size,
            cluster_bits: header.cluster_bits,
            l2_entries,
            l1,
            l1_table_offset: header.l1_table_offset,
            l2_cache: HashMap::new(),
            l2_cache_cap: 32,
        })
    }

    /// Split a virtual byte offset into the L1 index, the L2 index, and
    /// the byte offset within the cluster.
    pub fn split_addr(&self, vaddr: u64) -> (usize, usize, u64) {
        let cluster_idx = vaddr >> self.cluster_bits;
        let l1_idx = (cluster_idx as usize) / self.l2_entries;
        let l2_idx = (cluster_idx as usize) % self.l2_entries;
        let in_cluster = vaddr & (self.cluster_size - 1);
        (l1_idx, l2_idx, in_cluster)
    }

    /// Look up the physical byte offset of the cluster containing `vaddr`.
    /// Returns `Ok(None)` when the cluster is unallocated (read should
    /// return zeros) and `Err(Unsupported)` when the L2 entry has the
    /// COMPRESSED bit set.
    pub fn lookup<F: Read + Seek>(&mut self, file: &mut F, vaddr: u64) -> Result<Option<u64>> {
        let (l1_idx, l2_idx, _) = self.split_addr(vaddr);
        if l1_idx >= self.l1.len() {
            return Ok(None);
        }
        let l1_entry = self.l1[l1_idx];
        let l2_cluster_off = l1_entry & OFFSET_MASK;
        if l2_cluster_off == 0 {
            return Ok(None);
        }
        let l2 = self.load_l2(file, l2_cluster_off)?;
        let l2_entry = l2.entries[l2_idx];
        if l2_entry & COMPRESSED != 0 {
            return Err(crate::Error::Unsupported(
                "qcow2: compressed clusters are not supported".into(),
            ));
        }
        let phys = l2_entry & OFFSET_MASK;
        if phys == 0 {
            return Ok(None);
        }
        Ok(Some(phys))
    }

    fn load_l2<F: Read + Seek>(&mut self, file: &mut F, l2_off: u64) -> Result<&L2Entry> {
        if !self.l2_cache.contains_key(&l2_off) {
            file.seek(SeekFrom::Start(l2_off))?;
            let mut raw = vec![0u8; self.cluster_size as usize];
            file.read_exact(&mut raw)?;
            let entries: Vec<u64> = raw
                .chunks_exact(8)
                .map(|c| u64::from_be_bytes(c.try_into().unwrap()))
                .collect();
            if self.l2_cache.len() >= self.l2_cache_cap {
                // Drop one entry — pick the first non-dirty to evict.
                // (Simple policy; if everything's dirty we don't evict.)
                let victim = self
                    .l2_cache
                    .iter()
                    .find(|(_, v)| !v.dirty)
                    .map(|(k, _)| *k);
                if let Some(k) = victim {
                    self.l2_cache.remove(&k);
                }
            }
            self.l2_cache.insert(
                l2_off,
                L2Entry {
                    entries,
                    dirty: false,
                },
            );
        }
        Ok(self.l2_cache.get(&l2_off).unwrap())
    }

    /// Install a mapping for the cluster containing `vaddr` to point at
    /// `physical_offset`. Marks the affected L2 dirty. The caller must
    /// have already allocated the data cluster and (if needed) the L2
    /// cluster, and updated the L1 entry via [`Self::set_l1`].
    pub fn set_l2_entry(&mut self, l2_cluster_off: u64, l2_idx: usize, value: u64) -> Result<()> {
        let entry = self.l2_cache.get_mut(&l2_cluster_off).ok_or_else(|| {
            crate::Error::InvalidImage(format!(
                "qcow2: L2 cluster {l2_cluster_off:#x} not in cache; load it first"
            ))
        })?;
        entry.entries[l2_idx] = value;
        entry.dirty = true;
        Ok(())
    }

    /// Set L1[l1_idx] = value and mark the L1 table for flush.
    /// (Phase A doesn't write; this is for Phase B's allocator.)
    pub fn set_l1(&mut self, l1_idx: usize, value: u64) {
        self.l1[l1_idx] = value;
    }

    /// Write every dirty L2 cluster back to disk, then the L1 table.
    pub fn flush<F: Write + Seek>(&mut self, file: &mut F) -> Result<()> {
        for (&off, entry) in self.l2_cache.iter_mut() {
            if !entry.dirty {
                continue;
            }
            let mut raw = vec![0u8; self.cluster_size as usize];
            for (i, &e) in entry.entries.iter().enumerate() {
                raw[i * 8..i * 8 + 8].copy_from_slice(&e.to_be_bytes());
            }
            file.seek(SeekFrom::Start(off))?;
            file.write_all(&raw)?;
            entry.dirty = false;
        }
        // Re-emit the L1 table. Always — small + cheap.
        let mut raw = vec![0u8; self.l1.len() * 8];
        for (i, &e) in self.l1.iter().enumerate() {
            raw[i * 8..i * 8 + 8].copy_from_slice(&e.to_be_bytes());
        }
        file.seek(SeekFrom::Start(self.l1_table_offset))?;
        file.write_all(&raw)?;
        Ok(())
    }

    /// Insert a freshly-allocated L2 cluster into the cache (all zeros).
    /// Caller must update the L1 entry and persist via [`Self::flush`].
    pub fn insert_empty_l2(&mut self, l2_cluster_off: u64) {
        self.l2_cache.insert(
            l2_cluster_off,
            L2Entry {
                entries: vec![0u64; self.l2_entries],
                dirty: true,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_addr_math() {
        let l = L1L2 {
            cluster_size: 65536,
            cluster_bits: 16,
            l2_entries: 8192,
            l1: vec![0; 4],
            l1_table_offset: 0,
            l2_cache: HashMap::new(),
            l2_cache_cap: 32,
        };
        // Cluster 0 → L1[0], L2[0], offset 0.
        assert_eq!(l.split_addr(0), (0, 0, 0));
        // Cluster 1 → L1[0], L2[1], offset 0.
        assert_eq!(l.split_addr(65536), (0, 1, 0));
        // Middle of cluster 1.
        assert_eq!(l.split_addr(65536 + 1024), (0, 1, 1024));
        // Crossing into the second L1 entry: cluster 8192.
        assert_eq!(l.split_addr(8192u64 * 65536), (1, 0, 0));
    }

    #[test]
    fn offset_mask_drops_flags() {
        let entry = COPIED | 0x0001_0000;
        assert_eq!(entry & OFFSET_MASK, 0x0001_0000);
    }
}
