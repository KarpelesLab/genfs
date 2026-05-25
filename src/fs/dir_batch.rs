//! A small capacity-bounded cache of pending directory-entry batches,
//! shared by the write-through filesystem backends (NTFS, XFS, FAT32,
//! exFAT).
//!
//! Several on-disk directory formats serialize a whole directory (sort
//! its entries, rewrite its block / index / cluster chain) every time a
//! single child is added. When a caller — a repack, say — writes many
//! files into one directory in a row, that is O(N²) work for what could
//! be one pass.
//!
//! [`DirBatch`] lets a backend defer that work: each `create_*` stages
//! the new entry here instead of touching the disk. A directory's batch
//! is serialized **once**, when either
//!
//! * a *new* directory is staged and the cache is already at capacity —
//!   the oldest directory is evicted and handed back to the backend to
//!   serialize (the "as other directories enter the cache" case), or
//! * the filesystem is flushed — [`drain_all`](DirBatch::drain_all)
//!   returns every remaining batch, or
//! * something reads the directory back before flush —
//!   [`take`](DirBatch::take) pulls one batch out so the backend can
//!   serialize it on demand and keep reads consistent.
//!
//! The cache is deliberately dumb: it never touches the block device or
//! any backend state. The backend owns `dev` and does the actual
//! serialization, so there is no borrow entanglement and the same cache
//! works for every backend regardless of its on-disk layout. `K` is the
//! backend's directory identity (an inode / MFT record / cluster) and
//! `E` is whatever the backend wants to replay (a built index entry, a
//! `(name, child)` pair, …).
//!
//! Eviction is FIFO by first-insertion order, which matches the typical
//! "finish one directory, move to the next" write pattern: by the time a
//! directory is evicted, the caller has almost always moved on from it.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

/// Default number of directories kept resident before the oldest is
/// evicted and serialized.
pub const DEFAULT_CAPACITY: usize = 64;

/// Capacity-bounded, FIFO directory-entry batch cache. See the module
/// docs for the contract.
#[derive(Debug)]
pub struct DirBatch<K: Eq + Hash + Clone, E> {
    map: HashMap<K, Vec<E>>,
    /// Directory keys in first-insertion order; front == oldest.
    order: VecDeque<K>,
    capacity: usize,
}

impl<K: Eq + Hash + Clone, E> DirBatch<K, E> {
    /// Create a cache holding up to `capacity` directories (clamped to at
    /// least 1).
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    /// Stage `entry` under directory `key`. If staging a *new* directory
    /// pushes the cache past its capacity, the oldest directory is
    /// evicted and returned as `(key, entries)` for the caller to
    /// serialize immediately; otherwise `None`.
    pub fn stage(&mut self, key: K, entry: E) -> Option<(K, Vec<E>)> {
        if let Some(batch) = self.map.get_mut(&key) {
            batch.push(entry);
            return None;
        }
        // New directory. Evict the oldest first if we are full.
        let victim = if self.map.len() >= self.capacity {
            self.order
                .pop_front()
                .and_then(|old| self.map.remove(&old).map(|entries| (old, entries)))
        } else {
            None
        };
        self.order.push_back(key.clone());
        self.map.insert(key, vec![entry]);
        victim
    }

    /// Remove and return one directory's pending batch, if any. Used to
    /// flush a single directory on demand before it is read back.
    pub fn take(&mut self, key: &K) -> Option<Vec<E>> {
        let entries = self.map.remove(key)?;
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        Some(entries)
    }

    /// Remove and return every pending batch, emptying the cache. Used at
    /// final flush. Batches are independent, so order does not matter.
    pub fn drain_all(&mut self) -> Vec<(K, Vec<E>)> {
        self.order.clear();
        self.map.drain().collect()
    }

    /// `true` when no directory currently has pending entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_without_eviction_under_capacity() {
        let mut b: DirBatch<u32, &str> = DirBatch::new(4);
        assert!(b.stage(1, "a").is_none());
        assert!(b.stage(1, "b").is_none());
        assert!(b.stage(2, "c").is_none());
        // Same-directory restage never evicts, even at capacity.
        let mut full: DirBatch<u32, u32> = DirBatch::new(1);
        assert!(full.stage(7, 0).is_none());
        assert!(full.stage(7, 1).is_none());
        assert_eq!(full.take(&7), Some(vec![0, 1]));
    }

    #[test]
    fn evicts_oldest_when_a_new_dir_overflows() {
        let mut b: DirBatch<u32, u32> = DirBatch::new(2);
        assert!(b.stage(1, 10).is_none());
        assert!(b.stage(2, 20).is_none());
        // Third *new* directory evicts the oldest (1) with its batch.
        let victim = b.stage(3, 30);
        assert_eq!(victim, Some((1, vec![10])));
        // The survivors are still retrievable.
        assert_eq!(b.take(&2), Some(vec![20]));
        assert_eq!(b.take(&3), Some(vec![30]));
        assert!(b.is_empty());
    }

    #[test]
    fn drain_all_returns_everything() {
        let mut b: DirBatch<u32, u32> = DirBatch::new(8);
        b.stage(1, 10);
        b.stage(1, 11);
        b.stage(2, 20);
        let mut all = b.drain_all();
        all.sort();
        assert_eq!(all, vec![(1, vec![10, 11]), (2, vec![20])]);
        assert!(b.is_empty());
    }
}
