//! qcow2 refcount table + refcount blocks.
//!
//! The refcount table is an array of u64 BE pointers to refcount
//! *blocks*. Each refcount block carries an array of refcount entries
//! — one per physical cluster in the backing file. fstool only supports
//! `refcount_order = 4`, i.e. 16-bit refcounts (`cluster_size / 2`
//! entries per block).
//!
//! Allocation strategy (modify path):
//!
//! 1. Scan existing refcount blocks for a free entry (`refcount == 0`)
//!    starting at a hint cursor.
//! 2. If nothing free, grow the file: pick the next cluster past EOF,
//!    set its refcount to 1, and return it. If the refcount block that
//!    would track the new cluster doesn't exist yet, allocate THE
//!    refcount block at that same range (it tracks itself).
//!
//! The strategy assumes the refcount table itself never needs to grow.
//! That holds for any realistic image size: 64 KiB clusters give
//! 8192 table entries × 32768 cluster-refcounts per block = 256 M
//! clusters = 16 PiB of addressable storage.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::Result;

use super::header::Header;

/// Bytes per refcount entry (we only support `refcount_order=4` → u16).
pub const REFCOUNT_ENTRY_BYTES: u64 = 2;

/// In-memory refcount management state.
pub struct Refcount {
    pub cluster_size: u64,
    pub table_offset: u64,
    pub table_clusters: u32,
    /// Pointers to refcount blocks (BE u64 on disk, native u64 here).
    /// `table.len() = table_clusters * cluster_size / 8`.
    pub table: Vec<u64>,
    /// Cached refcount blocks, keyed by physical block offset.
    pub block_cache: HashMap<u64, RefcountBlock>,
    pub block_cache_cap: usize,
    /// Hint: start the free-cluster scan from this cluster index.
    pub next_free_hint: u64,
    pub table_dirty: bool,
}

pub struct RefcountBlock {
    pub entries: Vec<u16>,
    pub dirty: bool,
}

impl Refcount {
    /// Entries per refcount block (`cluster_size / 2` for 16-bit refs).
    pub fn entries_per_block(&self) -> u64 {
        self.cluster_size / REFCOUNT_ENTRY_BYTES
    }

    /// Total table capacity (`table_clusters × cluster_size / 8`).
    pub fn table_capacity(&self) -> u64 {
        self.table_clusters as u64 * self.cluster_size / 8
    }

    /// Load the refcount table from disk. The refcount blocks themselves
    /// are loaded lazily on demand.
    pub fn load<F: Read + Seek>(file: &mut F, header: &Header) -> Result<Self> {
        let cluster_size = header.cluster_size();

        // `refcount_table_clusters` is attacker-controlled and the table is
        // allocated up front (`table_clusters * cluster_size` bytes) before any
        // bounds-checked read. Cap it before allocating: the table can never
        // legitimately be larger than the file that holds it.
        let table_bytes = (header.refcount_table_clusters as u64)
            .checked_mul(cluster_size)
            .ok_or_else(|| {
                crate::Error::InvalidImage(
                    "qcow2: refcount_table_clusters * cluster_size overflows".into(),
                )
            })?;
        let file_len = file.seek(SeekFrom::End(0))?;
        if table_bytes > file_len {
            return Err(crate::Error::InvalidImage(format!(
                "qcow2: refcount table ({table_bytes} bytes) exceeds file length {file_len}"
            )));
        }
        let table_entries = (table_bytes / 8) as usize;
        let mut raw = vec![0u8; table_entries * 8];
        file.seek(SeekFrom::Start(header.refcount_table_offset))?;
        file.read_exact(&mut raw)?;
        let table: Vec<u64> = raw
            .chunks_exact(8)
            .map(|c| u64::from_be_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Self {
            cluster_size,
            table_offset: header.refcount_table_offset,
            table_clusters: header.refcount_table_clusters,
            table,
            block_cache: HashMap::new(),
            block_cache_cap: 32,
            next_free_hint: 0,
            table_dirty: false,
        })
    }

    /// Build a fresh in-memory refcount state for a brand-new image.
    /// The table and the (single, empty) refcount block exist only in
    /// memory; the caller writes them to disk at format time.
    ///
    /// `initial_clusters` is the set of physical cluster indices that
    /// should start with refcount=1 — i.e. the header / refcount table /
    /// refcount block / L1 clusters that the new image's layout uses.
    pub fn new_fresh(
        cluster_size: u64,
        table_offset: u64,
        first_refcount_block_offset: u64,
        initial_clusters: &[u64],
    ) -> Self {
        let table_capacity = cluster_size / 8;
        let mut table = vec![0u64; table_capacity as usize];
        table[0] = first_refcount_block_offset;
        let entries_per_block = cluster_size / REFCOUNT_ENTRY_BYTES;
        let mut entries = vec![0u16; entries_per_block as usize];
        for &c in initial_clusters {
            entries[c as usize] = 1;
        }
        let mut block_cache = HashMap::new();
        block_cache.insert(
            first_refcount_block_offset,
            RefcountBlock {
                entries,
                dirty: true,
            },
        );
        Self {
            cluster_size,
            table_offset,
            table_clusters: 1,
            table,
            block_cache,
            block_cache_cap: 32,
            next_free_hint: initial_clusters.iter().copied().max().unwrap_or(0) + 1,
            table_dirty: true,
        }
    }

    fn load_block<F: Read + Seek>(
        &mut self,
        file: &mut F,
        block_off: u64,
    ) -> Result<&mut RefcountBlock> {
        if !self.block_cache.contains_key(&block_off) {
            file.seek(SeekFrom::Start(block_off))?;
            let mut raw = vec![0u8; self.cluster_size as usize];
            file.read_exact(&mut raw)?;
            let entries: Vec<u16> = raw
                .chunks_exact(2)
                .map(|c| u16::from_be_bytes(c.try_into().unwrap()))
                .collect();
            // Drop a non-dirty cache entry if we're at the cap.
            if self.block_cache.len() >= self.block_cache_cap
                && let Some(k) = self
                    .block_cache
                    .iter()
                    .find(|(_, v)| !v.dirty)
                    .map(|(k, _)| *k)
            {
                self.block_cache.remove(&k);
            }
            self.block_cache.insert(
                block_off,
                RefcountBlock {
                    entries,
                    dirty: false,
                },
            );
        }
        Ok(self.block_cache.get_mut(&block_off).unwrap())
    }

    /// Allocate one physical cluster: find a free entry in an existing
    /// refcount block, bump it to 1, and return the cluster index. If no
    /// existing block has space, extend the file by allocating the next
    /// cluster past EOF (and a new refcount block for it if needed).
    /// Updates `file_len` to reflect any growth.
    pub fn alloc_cluster<F: Read + Write + Seek>(
        &mut self,
        file: &mut F,
        file_len: &mut u64,
    ) -> Result<u64> {
        // Pass 1: scan existing refcount blocks for refcount==0.
        let hint_block = self.next_free_hint / self.entries_per_block();
        let table_len = self.table.len();
        for sweep in 0..2 {
            // sweep 0: from hint to end; sweep 1: from 0 to hint.
            let (lo, hi) = if sweep == 0 {
                (hint_block as usize, table_len)
            } else {
                (0, hint_block as usize)
            };
            for block_idx in lo..hi {
                let block_off = self.table[block_idx];
                if block_off == 0 {
                    continue;
                }
                let entries_per_block = self.entries_per_block();
                let _block = self.load_block(file, block_off)?;
                // Scan starting from the hint within this block.
                let start_in_block = if sweep == 0 && block_idx == hint_block as usize {
                    (self.next_free_hint % entries_per_block) as usize
                } else {
                    0
                };
                let block = self.block_cache.get_mut(&block_off).unwrap();
                for (i, slot) in block.entries.iter_mut().enumerate().skip(start_in_block) {
                    if *slot == 0 {
                        *slot = 1;
                        block.dirty = true;
                        let cluster = block_idx as u64 * entries_per_block + i as u64;
                        self.next_free_hint = cluster + 1;
                        return Ok(cluster);
                    }
                }
            }
        }
        // Pass 2: grow the file. Pick the cluster right past current EOF.
        let new_cluster = *file_len / self.cluster_size;
        let entries_per_block = self.entries_per_block();
        let block_idx = (new_cluster / entries_per_block) as usize;
        if block_idx >= table_len {
            return Err(crate::Error::Unsupported(
                "qcow2: refcount table is full (image would exceed 16 PiB at default cluster size)"
                    .into(),
            ));
        }
        let block_off = self.table[block_idx];
        if block_off == 0 {
            // Need to allocate a new refcount block. It lives at the
            // next cluster past EOF (`new_cluster`), tracks itself, and
            // also tracks the data cluster we're handing out
            // (`new_cluster + 1`).
            let rcb_cluster = new_cluster;
            let data_cluster = new_cluster + 1;
            let next_block_idx = (data_cluster / entries_per_block) as usize;
            if next_block_idx != block_idx {
                // The data cluster falls into the next refcount block —
                // an unlikely edge at 2 GiB boundaries that we don't
                // handle in v1.
                return Err(crate::Error::Unsupported(
                    "qcow2: allocation across refcount-block boundary not implemented".into(),
                ));
            }
            let rcb_off = rcb_cluster * self.cluster_size;
            let data_off = data_cluster * self.cluster_size;
            let mut entries = vec![0u16; entries_per_block as usize];
            entries[(rcb_cluster % entries_per_block) as usize] = 1;
            entries[(data_cluster % entries_per_block) as usize] = 1;
            self.block_cache.insert(
                rcb_off,
                RefcountBlock {
                    entries,
                    dirty: true,
                },
            );
            self.table[block_idx] = rcb_off;
            self.table_dirty = true;
            *file_len = data_off + self.cluster_size;
            self.next_free_hint = data_cluster + 1;
            return Ok(data_cluster);
        }
        // Existing refcount block; just bump the new cluster's entry.
        let _block = self.load_block(file, block_off)?;
        let block = self.block_cache.get_mut(&block_off).unwrap();
        let idx = (new_cluster % entries_per_block) as usize;
        block.entries[idx] = 1;
        block.dirty = true;
        *file_len = (new_cluster + 1) * self.cluster_size;
        self.next_free_hint = new_cluster + 1;
        Ok(new_cluster)
    }

    /// Write every dirty refcount block back, then the refcount table.
    pub fn flush<F: Write + Seek>(&mut self, file: &mut F) -> Result<()> {
        for (&off, block) in self.block_cache.iter_mut() {
            if !block.dirty {
                continue;
            }
            let mut raw = vec![0u8; self.cluster_size as usize];
            for (i, &e) in block.entries.iter().enumerate() {
                raw[i * 2..i * 2 + 2].copy_from_slice(&e.to_be_bytes());
            }
            file.seek(SeekFrom::Start(off))?;
            file.write_all(&raw)?;
            block.dirty = false;
        }
        if self.table_dirty {
            let mut raw = vec![0u8; self.table.len() * 8];
            for (i, &e) in self.table.iter().enumerate() {
                raw[i * 8..i * 8 + 8].copy_from_slice(&e.to_be_bytes());
            }
            file.seek(SeekFrom::Start(self.table_offset))?;
            file.write_all(&raw)?;
            self.table_dirty = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn fresh_state_marks_initial_clusters() {
        // Pretend cluster_size=512 to keep the test tiny; we just check
        // the bookkeeping. Initial clusters [0,1,2,3] are reserved.
        let r = Refcount::new_fresh(512, 512, 1024, &[0, 1, 2, 3]);
        assert_eq!(r.table[0], 1024);
        let block = r.block_cache.get(&1024).unwrap();
        assert_eq!(block.entries[0], 1);
        assert_eq!(block.entries[1], 1);
        assert_eq!(block.entries[2], 1);
        assert_eq!(block.entries[3], 1);
        assert_eq!(block.entries[4], 0);
        assert_eq!(r.next_free_hint, 4);
    }

    #[test]
    fn alloc_picks_first_free_in_existing_block() {
        let mut r = Refcount::new_fresh(512, 512, 1024, &[0, 1, 2, 3]);
        let mut file_len = 4 * 512;
        // No real I/O needed since the block is already in cache.
        let mut buf: Vec<u8> = Vec::new();
        let mut cur = Cursor::new(&mut buf);
        let c = r.alloc_cluster(&mut cur, &mut file_len).unwrap();
        assert_eq!(c, 4);
        let block = r.block_cache.get(&1024).unwrap();
        assert_eq!(block.entries[4], 1);
    }
}
