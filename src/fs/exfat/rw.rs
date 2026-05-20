//! `Filesystem::open_file_rw` for exFAT — eager-write file handle.
//!
//! The handle owns mutable borrows of both the [`super::Exfat`] state and
//! the [`BlockDevice`] for its lifetime. Each [`std::io::Write::write`]
//! call writes through to disk immediately:
//!
//! - bytes that land in existing clusters patch them in place;
//! - bytes past EOF (or past the current allocation) allocate fresh
//!   clusters via the bitmap, link them into the file's FAT chain, and
//!   update the on-disk StreamExtension entry (DataLength /
//!   ValidDataLength / FirstCluster / secondary flags).
//!
//! Truncation walks the tail of the chain, frees those clusters in the
//! bitmap + FAT, and rewrites the entry-set in memory; the on-disk
//! entry-set bytes are flushed on [`super::ExfatFileHandle::sync`] (or
//! when the handle is dropped).
//!
//! exFAT has no journal — partial writes are safe at single-sector
//! granularity (each FAT, bitmap, and entry-set update is itself a
//! small atomic store). A crash mid-grow leaves at most an orphan
//! cluster, recoverable by `fsck.exfat`.

use std::io::{Read, Seek, SeekFrom, Write};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::{FileHandle, FileReadHandle};

use super::dir::{self, ENTRY_SIZE, FileEntrySet, SECFLAG_ALLOC_POSSIBLE, SECFLAG_NO_FAT_CHAIN};
use super::fat;
use super::{Exfat, set_bitmap_bit, split_path};

/// An open exFAT file with read + write access. Created by
/// [`super::Exfat`]'s `Filesystem::open_file_rw` adapter and returned as a
/// `Box<dyn FileHandle>`.
pub struct ExfatFileHandle<'a> {
    pub(super) fs: &'a mut Exfat,
    pub(super) dev: &'a mut dyn BlockDevice,
    /// First cluster of the parent directory.
    pub(super) parent_cluster: u32,
    /// Byte offset of the file's entry set within the parent directory's
    /// logical bytes (all clusters concatenated).
    pub(super) entry_pos: u64,
    /// Total bytes of the entry set on disk (== (1 + secondary_count) * 32).
    pub(super) entry_total: usize,
    /// In-memory copy of the entry set; mutations stay here until `sync`.
    pub(super) entry_bytes: Vec<u8>,
    /// File's cluster chain (in order).
    pub(super) chain: Vec<u32>,
    /// Whether the on-disk stream extension marks the file as NoFatChain.
    pub(super) no_fat_chain: bool,
    /// Logical length (ValidDataLength).
    pub(super) len: u64,
    /// Current read/write cursor.
    pub(super) pos: u64,
    /// True when `entry_bytes` differs from what's on disk.
    pub(super) entry_dirty: bool,
}

impl<'a> ExfatFileHandle<'a> {
    fn cluster_size(&self) -> u64 {
        self.fs.boot.bytes_per_cluster() as u64
    }

    fn cluster_disk_offset(&self, cluster: u32) -> u64 {
        self.fs.boot.cluster_byte_offset(cluster)
    }

    /// Patch fields of the in-memory entry set so that on `sync` we write
    /// back consistent DataLength / ValidDataLength / FirstCluster /
    /// secondary flags.
    fn refresh_entry_bytes(&mut self) {
        // Stream extension is the second 32-byte slot.
        let stream_off = ENTRY_SIZE;
        // GeneralSecondaryFlags.
        let mut flags = self.entry_bytes[stream_off + 1];
        if self.chain.is_empty() {
            // Empty file: clear AllocationPossible (per spec).
            flags &= !SECFLAG_ALLOC_POSSIBLE;
            flags &= !SECFLAG_NO_FAT_CHAIN;
        } else {
            flags |= SECFLAG_ALLOC_POSSIBLE;
            if self.no_fat_chain {
                flags |= SECFLAG_NO_FAT_CHAIN;
            } else {
                flags &= !SECFLAG_NO_FAT_CHAIN;
            }
        }
        self.entry_bytes[stream_off + 1] = flags;

        // ValidDataLength (8 bytes at offset 8).
        self.entry_bytes[stream_off + 8..stream_off + 16]
            .copy_from_slice(&self.len.to_le_bytes());
        // FirstCluster (4 bytes at offset 20).
        let first_cluster = if self.chain.is_empty() {
            0
        } else {
            self.chain[0]
        };
        self.entry_bytes[stream_off + 20..stream_off + 24]
            .copy_from_slice(&first_cluster.to_le_bytes());
        // DataLength (8 bytes at offset 24). exFAT requires
        // DataLength >= ValidDataLength; we keep them equal here.
        self.entry_bytes[stream_off + 24..stream_off + 32]
            .copy_from_slice(&self.len.to_le_bytes());

        // Recompute SetChecksum over the whole set (skipping primary[2..4]).
        let csum = dir::set_checksum(&self.entry_bytes);
        self.entry_bytes[2..4].copy_from_slice(&csum.to_le_bytes());
        self.entry_dirty = true;
    }

    /// Write the in-memory entry set back to its slots in the parent
    /// directory. Used by `sync`.
    fn write_entry_set(&mut self) -> Result<()> {
        if !self.entry_dirty {
            return Ok(());
        }
        let cb = self.cluster_size();
        let chain = self.fs.dir_chain(self.parent_cluster)?;
        let n_slots = self.entry_total / ENTRY_SIZE;
        for k in 0..n_slots {
            let p = self.entry_pos + (k as u64) * ENTRY_SIZE as u64;
            let cluster_idx = (p / cb) as usize;
            let cluster_off = p % cb;
            if cluster_idx >= chain.len() {
                return Err(crate::Error::InvalidImage(
                    "exfat: entry-set position past parent directory chain".into(),
                ));
            }
            let cluster = chain[cluster_idx];
            let disk_off = self.cluster_disk_offset(cluster) + cluster_off;
            let src_off = k * ENTRY_SIZE;
            self.dev
                .write_at(disk_off, &self.entry_bytes[src_off..src_off + ENTRY_SIZE])?;
        }
        self.entry_dirty = false;
        Ok(())
    }

    /// Convert a NoFatChain run into an explicit FAT chain so we can
    /// extend it with non-contiguous clusters. Writes FAT entries linking
    /// `chain[i] -> chain[i+1]` for every adjacent pair and EOC for the
    /// tail. Idempotent — clears the `no_fat_chain` flag.
    fn materialise_fat_chain(&mut self) {
        if !self.no_fat_chain {
            return;
        }
        for i in 0..self.chain.len() {
            let cur = self.chain[i];
            let next = if i + 1 < self.chain.len() {
                self.chain[i + 1]
            } else {
                fat::EOC
            };
            self.fs.fat.set_raw(cur, next);
        }
        self.fs.fat_dirty = true;
        self.no_fat_chain = false;
    }

    /// Allocate `n` additional clusters and append them to this file's
    /// chain. The first appended cluster is linked from the previous
    /// tail's FAT entry (when there is a previous tail).
    ///
    /// On NoFatChain runs: first promotes the chain to a real FAT
    /// chain (so we can grow non-contiguously), then appends.
    fn grow_chain(&mut self, n: u32) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.materialise_fat_chain();
        let prev_tail = self.chain.last().copied();
        for i in 0..n {
            let c = self.fs.alloc_cluster()?;
            if i == 0 {
                if let Some(prev) = prev_tail {
                    self.fs.fat.set_raw(prev, c);
                }
            } else {
                let prev = *self.chain.last().unwrap();
                self.fs.fat.set_raw(prev, c);
            }
            // The new tail stays as EOC (alloc_cluster already wrote EOC).
            self.chain.push(c);
        }
        self.fs.fat_dirty = true;
        Ok(())
    }

    /// Free clusters from the tail until the chain holds exactly
    /// `keep` clusters. No-op when `keep >= chain.len()`.
    fn shrink_chain(&mut self, keep: usize) -> Result<()> {
        if keep >= self.chain.len() {
            return Ok(());
        }
        // Free all clusters from index `keep` onwards directly via the
        // bitmap + FAT (we already have the cluster list — no need to
        // walk).
        for &c in &self.chain[keep..] {
            if c >= 2 {
                self.fs.fat.set_raw(c, fat::FREE);
                set_bitmap_bit(&mut self.fs.bitmap, c, false);
                if self.fs.next_free_hint > c {
                    self.fs.next_free_hint = c;
                }
            }
        }
        self.fs.fat_dirty = true;
        self.fs.bitmap_dirty = true;
        self.chain.truncate(keep);
        // Patch the new tail to EOC if any clusters remain and we are
        // using a real FAT chain.
        if !self.chain.is_empty() && !self.no_fat_chain {
            let tail = *self.chain.last().unwrap();
            self.fs.fat.set_raw(tail, fat::EOC);
        }
        Ok(())
    }

    /// Ensure the chain is large enough to cover at least `target_len`
    /// bytes. Grows the chain (allocating new clusters) when it isn't.
    fn ensure_capacity(&mut self, target_len: u64) -> Result<()> {
        let cb = self.cluster_size();
        let have = self.chain.len() as u64 * cb;
        if target_len <= have {
            return Ok(());
        }
        let need_clusters = target_len.div_ceil(cb) as u32;
        let extra = need_clusters - self.chain.len() as u32;
        self.grow_chain(extra)
    }

    /// Zero the byte range `[from, to)` on disk by writing through to the
    /// relevant clusters. Both offsets must be within the current
    /// allocation (caller is responsible for growing first).
    fn zero_range(&mut self, from: u64, to: u64) -> Result<()> {
        if from >= to {
            return Ok(());
        }
        let cb = self.cluster_size();
        let mut pos = from;
        // Reusable zero scratch buffer, capped at one cluster.
        let zero = vec![0u8; cb.min(64 * 1024) as usize];
        while pos < to {
            let cluster_idx = (pos / cb) as usize;
            let cluster_off = pos % cb;
            if cluster_idx >= self.chain.len() {
                return Err(crate::Error::InvalidImage(
                    "exfat: zero_range past chain".into(),
                ));
            }
            let cluster = self.chain[cluster_idx];
            let in_cluster = (cb - cluster_off).min(to - pos);
            let mut written: u64 = 0;
            while written < in_cluster {
                let n = (in_cluster - written).min(zero.len() as u64) as usize;
                let disk_off =
                    self.cluster_disk_offset(cluster) + cluster_off + written;
                self.dev.write_at(disk_off, &zero[..n])?;
                written += n as u64;
            }
            pos += in_cluster;
        }
        Ok(())
    }

    /// Read `buf.len()` bytes (or fewer at EOF) starting at the current
    /// position. Helper for `Read::read`.
    fn read_internal(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.len {
            return Ok(0);
        }
        let cb = self.cluster_size();
        let avail = self.len - self.pos;
        let cluster_idx = (self.pos / cb) as usize;
        let cluster_off = self.pos % cb;
        if cluster_idx >= self.chain.len() {
            return Ok(0);
        }
        let cluster = self.chain[cluster_idx];
        let in_cluster = cb - cluster_off;
        let want = (buf.len() as u64).min(in_cluster).min(avail) as usize;
        let disk_off = self.cluster_disk_offset(cluster) + cluster_off;
        self.dev
            .read_at(disk_off, &mut buf[..want])
            .map_err(std::io::Error::other)?;
        self.pos += want as u64;
        Ok(want)
    }

    /// Write `buf` at the current position, growing the file as needed.
    /// Helper for `Write::write` — returns a [`crate::Error`] on failure
    /// which the wrapping `Write` impl converts to `io::Error`.
    fn write_internal(&mut self, buf: &[u8]) -> Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        // Phase 1: if `pos > len`, the range [len, pos) becomes a sparse
        // gap. Allocate clusters covering it and zero those bytes.
        if self.pos > self.len {
            self.ensure_capacity(self.pos)?;
            let gap_lo = self.len;
            let gap_hi = self.pos;
            self.zero_range(gap_lo, gap_hi)?;
            self.len = self.pos;
            self.refresh_entry_bytes();
        }

        // Phase 2: ensure capacity for the new bytes.
        let new_end = self.pos + buf.len() as u64;
        self.ensure_capacity(new_end)?;

        // Phase 3: write the buffer cluster-by-cluster.
        let cb = self.cluster_size();
        let mut written: usize = 0;
        while written < buf.len() {
            let p = self.pos + written as u64;
            let cluster_idx = (p / cb) as usize;
            let cluster_off = p % cb;
            let cluster = self.chain[cluster_idx];
            let in_cluster = (cb - cluster_off).min((buf.len() - written) as u64) as usize;
            let disk_off = self.cluster_disk_offset(cluster) + cluster_off;
            self.dev
                .write_at(disk_off, &buf[written..written + in_cluster])?;
            written += in_cluster;
        }

        // Phase 4: update length + entry bytes if we extended past EOF.
        self.pos += buf.len() as u64;
        if self.pos > self.len {
            self.len = self.pos;
        }
        self.refresh_entry_bytes();
        Ok(written)
    }
}

impl<'a> Read for ExfatFileHandle<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.read_internal(buf)
    }
}

impl<'a> Write for ExfatFileHandle<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_internal(buf)
            .map_err(std::io::Error::other)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // Defer the heavyweight FAT/bitmap persistence to `sync` —
        // `flush` per std::io is a no-op for streams whose `write`
        // already reached the device.
        Ok(())
    }
}

impl<'a> Seek for ExfatFileHandle<'a> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::Current(n) => self.pos as i128 + n as i128,
            SeekFrom::End(n) => self.len as i128 + n as i128,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "exfat: seek to negative offset",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

impl<'a> FileHandle for ExfatFileHandle<'a> {
    fn len(&self) -> u64 {
        self.len
    }

    fn set_len(&mut self, new_len: u64) -> Result<()> {
        let cb = self.cluster_size();
        if new_len < self.len {
            let keep_clusters = new_len.div_ceil(cb) as usize;
            self.shrink_chain(keep_clusters)?;
            self.len = new_len;
            // If pos is past new end, leave it — the next read will
            // return 0; the next write will reopen the sparse-fill path.
        } else if new_len > self.len {
            self.ensure_capacity(new_len)?;
            // Zero the freshly-exposed bytes (everything from old `len`
            // up to `new_len`).
            let old_len = self.len;
            self.zero_range(old_len, new_len)?;
            self.len = new_len;
        } else {
            return Ok(());
        }
        self.refresh_entry_bytes();
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        // Persist entry-set first so a subsequent flush picks up any
        // FAT/bitmap state for the new chain.
        self.write_entry_set()?;
        // Then push FAT + bitmap.
        self.fs.flush(self.dev)?;
        Ok(())
    }
}

impl<'a> Drop for ExfatFileHandle<'a> {
    fn drop(&mut self) {
        // Best-effort persistence on drop. Errors are swallowed —
        // callers who care about durability call `sync` (or
        // `Filesystem::flush`) explicitly.
        let _ = self.write_entry_set();
    }
}

// ---------------------------------------------------------------------------
// Filesystem::open_file_rw adapter — lives on Exfat itself so the trait impl
// in mod.rs can delegate to it.
// ---------------------------------------------------------------------------

impl Exfat {
    /// Open `path` for in-place reads + writes. Implements
    /// [`crate::fs::Filesystem::open_file_rw`].
    pub(super) fn open_rw<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
        flags: crate::fs::OpenFlags,
        meta: Option<crate::fs::FileMeta>,
    ) -> Result<Box<dyn FileHandle + 'a>> {
        // Resolve parent directory + leaf name.
        let parts = split_path(path);
        if parts.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "exfat: cannot open root as a file".into(),
            ));
        }
        let (leaf, prefix) = parts.split_last().unwrap();
        let mut parent_cluster = self.boot.first_cluster_of_root_directory;
        for part in prefix {
            let bytes = self.read_dir_bytes(dev, parent_cluster)?;
            let next = super::iter_file_sets(&bytes)?
                .into_iter()
                .find(|e| self.name_matches(&e.name_utf16, part))
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "exfat: no such entry {part:?} under {path:?}"
                    ))
                })?;
            if !next.is_directory {
                return Err(crate::Error::InvalidArgument(format!(
                    "exfat: {part:?} is not a directory"
                )));
            }
            parent_cluster = next.first_cluster;
        }

        // Look up the leaf in the parent directory.
        let found = self.find_entry_in_dir(dev, parent_cluster, leaf)?;

        match found {
            Some((pos, set, total)) => {
                if set.is_directory {
                    return Err(crate::Error::InvalidArgument(format!(
                        "exfat: {path:?} is a directory"
                    )));
                }
                // Build the handle on the existing entry.
                let chain = self.build_data_chain(
                    set.first_cluster,
                    set.no_fat_chain(),
                    set.data_length,
                )?;
                let mut handle = self.handle_from_existing(
                    dev,
                    parent_cluster,
                    pos,
                    total,
                    set,
                    chain,
                )?;
                if flags.truncate {
                    handle.set_len(0)?;
                }
                if flags.append {
                    handle.pos = handle.len;
                }
                Ok(Box::new(handle))
            }
            None => {
                if !flags.create {
                    return Err(crate::Error::InvalidArgument(format!(
                        "exfat: no such file {path:?}"
                    )));
                }
                let m = meta.ok_or_else(|| {
                    crate::Error::InvalidArgument(
                        "exfat: open_file_rw with create=true requires meta".into(),
                    )
                })?;
                let ts = super::unix_to_exfat_timestamp(m.mtime);
                // Create empty file via existing path.
                let _ = self.create_file_in(
                    dev,
                    parent_cluster,
                    leaf,
                    &mut std::io::empty(),
                    0,
                    ts,
                )?;
                // Re-find the entry we just wrote — its position is the
                // first free slot in the parent, but it's simplest to
                // look it up by name.
                let (pos, set, total) = self
                    .find_entry_in_dir(dev, parent_cluster, leaf)?
                    .ok_or_else(|| {
                        crate::Error::InvalidImage(
                            "exfat: created file vanished from parent directory".into(),
                        )
                    })?;
                let chain = self.build_data_chain(
                    set.first_cluster,
                    set.no_fat_chain(),
                    set.data_length,
                )?;
                let mut handle = self.handle_from_existing(
                    dev,
                    parent_cluster,
                    pos,
                    total,
                    set,
                    chain,
                )?;
                if flags.append {
                    handle.pos = handle.len;
                }
                Ok(Box::new(handle))
            }
        }
    }

    /// Open `path` for read-only access. Implements
    /// [`crate::fs::Filesystem::open_file_ro`].
    pub(super) fn open_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<Box<dyn FileReadHandle + 'a>> {
        let parts = split_path(path);
        if parts.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "exfat: cannot open root as a file".into(),
            ));
        }
        let (leaf, prefix) = parts.split_last().unwrap();
        let mut parent_cluster = self.boot.first_cluster_of_root_directory;
        for part in prefix {
            let bytes = self.read_dir_bytes(dev, parent_cluster)?;
            let next = super::iter_file_sets(&bytes)?
                .into_iter()
                .find(|e| self.name_matches(&e.name_utf16, part))
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "exfat: no such entry {part:?} under {path:?}"
                    ))
                })?;
            if !next.is_directory {
                return Err(crate::Error::InvalidArgument(format!(
                    "exfat: {part:?} is not a directory"
                )));
            }
            parent_cluster = next.first_cluster;
        }
        let (pos, set, total) = self
            .find_entry_in_dir(dev, parent_cluster, leaf)?
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!("exfat: no such file {path:?}"))
            })?;
        if set.is_directory {
            return Err(crate::Error::InvalidArgument(format!(
                "exfat: {path:?} is a directory"
            )));
        }
        let chain =
            self.build_data_chain(set.first_cluster, set.no_fat_chain(), set.data_length)?;
        let inner = self.handle_from_existing(dev, parent_cluster, pos, total, set, chain)?;
        Ok(Box::new(ReadOnlyExfatHandle { inner }))
    }

    /// Construct an [`ExfatFileHandle`] for an entry we already resolved.
    /// Reads the on-disk entry-set bytes so the handle can mutate +
    /// rewrite them on sync.
    #[allow(clippy::too_many_arguments)]
    fn handle_from_existing<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        parent_cluster: u32,
        entry_pos: u64,
        entry_total: usize,
        set: FileEntrySet,
        chain: Vec<u32>,
    ) -> Result<ExfatFileHandle<'a>> {
        // Pull the entry-set bytes off disk so we can mutate them.
        let cb = self.boot.bytes_per_cluster() as u64;
        let dir_chain = self.dir_chain(parent_cluster)?;
        let mut entry_bytes = vec![0u8; entry_total];
        let n_slots = entry_total / ENTRY_SIZE;
        for k in 0..n_slots {
            let p = entry_pos + (k as u64) * ENTRY_SIZE as u64;
            let cluster_idx = (p / cb) as usize;
            let cluster_off = p % cb;
            if cluster_idx >= dir_chain.len() {
                return Err(crate::Error::InvalidImage(
                    "exfat: entry-set position past parent directory chain".into(),
                ));
            }
            let cluster = dir_chain[cluster_idx];
            let disk_off = self.boot.cluster_byte_offset(cluster) + cluster_off;
            dev.read_at(
                disk_off,
                &mut entry_bytes[k * ENTRY_SIZE..(k + 1) * ENTRY_SIZE],
            )?;
        }
        let no_fat_chain = set.no_fat_chain();
        let len = set.valid_data_length;
        Ok(ExfatFileHandle {
            fs: self,
            dev,
            parent_cluster,
            entry_pos,
            entry_total,
            entry_bytes,
            chain,
            no_fat_chain,
            len,
            pos: 0,
            entry_dirty: false,
        })
    }
}

/// Read-only adapter over [`ExfatFileHandle`]. Forwards `Read` + `Seek`
/// to the inner handle; never invokes any mutating method, so the
/// underlying handle stays clean — its `Drop` is a no-op because
/// `entry_dirty` was never set.
pub struct ReadOnlyExfatHandle<'a> {
    inner: ExfatFileHandle<'a>,
}

impl<'a> Read for ReadOnlyExfatHandle<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl<'a> Seek for ReadOnlyExfatHandle<'a> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl<'a> FileReadHandle for ReadOnlyExfatHandle<'a> {
    fn len(&self) -> u64 {
        FileHandle::len(&self.inner)
    }
}
