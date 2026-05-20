//! FAT32 in-place read/write file handle.
//!
//! Backs [`crate::fs::Filesystem::open_file_rw`]. Every `Write::write` is
//! eager: the relevant cluster(s) are read, patched, written back, and the
//! directory entry's `file_size` is updated when bytes land past the
//! previous EOF. There is no in-memory buffer of file data on the handle —
//! crash safety reduces to "the FAT and the 8.3 entry were persisted by
//! the most recent `sync()` (or `Drop`)".
//!
//! ## Lifetime
//!
//! The handle holds `&'a mut Fat32` and `&'a mut dyn BlockDevice` for its
//! full lifetime, matching the trait signature.

use std::io::{self, Read, Seek, SeekFrom, Write};

use super::{Fat32, SECTOR, dir, table};
use crate::Result;
use crate::block::BlockDevice;
use crate::fs::FileHandle;

/// A FAT32 file handle: eager byte-granular reads + writes, with on-write
/// directory-entry updates.
pub struct FatFileHandle<'a> {
    fs: &'a mut Fat32,
    dev: &'a mut dyn BlockDevice,
    /// Cluster chain backing the file's data. Empty for zero-length files.
    chain: Vec<u32>,
    /// Logical file size in bytes.
    file_size: u64,
    /// Read/write cursor.
    pos: u64,
    /// Cluster chain of the parent directory.
    dir_chain: Vec<u32>,
    /// Byte offset of the 8.3 directory entry within the parent's flat
    /// directory buffer (= `cluster_index * cluster_bytes + in_cluster_off`).
    entry_pos: usize,
    /// Attribute byte for the entry — preserved across rewrites.
    entry_attr: u8,
    /// Raw 8.3 name field — preserved across rewrites.
    entry_name_83: [u8; 11],
    /// True once writes (or set_len) have made the in-memory FAT / entry
    /// diverge from disk. Cleared on `sync()` and `Drop`.
    dirty: bool,
}

impl<'a> FatFileHandle<'a> {
    /// Build a handle for the file whose 8.3 entry lives in `parent_chain`
    /// at byte offset `entry_pos`. Decodes the entry to recover the cluster
    /// chain and length; subsequent writes go through `self`.
    pub(super) fn open_existing(
        fs: &'a mut Fat32,
        dev: &'a mut dyn BlockDevice,
        parent_chain: Vec<u32>,
        entry_pos: usize,
        entry: dir::DirEntry,
    ) -> Result<Self> {
        let chain = if entry.first_cluster < 2 {
            Vec::new()
        } else {
            fs.chain_of(entry.first_cluster)?
        };
        Ok(Self {
            fs,
            dev,
            chain,
            file_size: u64::from(entry.file_size),
            pos: 0,
            dir_chain: parent_chain,
            entry_pos,
            entry_attr: entry.attr,
            entry_name_83: entry.name_83,
            dirty: false,
        })
    }

    /// Cluster size in bytes.
    fn cb(&self) -> u64 {
        self.fs.boot_sector().sectors_per_cluster as u64 * SECTOR as u64
    }

    /// Absolute byte offset of a data cluster's first sector.
    fn cluster_offset(&self, cluster: u32) -> u64 {
        let boot = self.fs.boot_sector();
        let sector = boot.data_start_sector() + (cluster - 2) * boot.sectors_per_cluster as u32;
        sector as u64 * SECTOR as u64
    }

    /// Read `buf.len()` bytes starting at the current cursor — up to EOF.
    /// Returns the number of bytes actually read.
    fn read_inner(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.file_size || buf.is_empty() {
            return Ok(0);
        }
        let cb = self.cb();
        let remaining_in_file = self.file_size - self.pos;
        let want = (buf.len() as u64).min(remaining_in_file);
        let mut written: u64 = 0;
        while written < want {
            let pos = self.pos + written;
            let cluster_idx = (pos / cb) as usize;
            let in_cluster = pos % cb;
            let chunk = (cb - in_cluster).min(want - written);
            if cluster_idx >= self.chain.len() {
                break;
            }
            let cluster = self.chain[cluster_idx];
            let off = self.cluster_offset(cluster) + in_cluster;
            let dst_start = written as usize;
            let dst_end = dst_start + chunk as usize;
            self.dev
                .read_at(off, &mut buf[dst_start..dst_end])
                .map_err(io::Error::other)?;
            written += chunk;
        }
        self.pos += written;
        Ok(written as usize)
    }

    /// Extend the file's cluster chain so it covers at least `needed_clusters`.
    /// New clusters are appended; the FAT links are updated in memory.
    fn ensure_chain_clusters(&mut self, needed: u32) -> Result<()> {
        if self.chain.len() as u32 >= needed {
            return Ok(());
        }
        let extra = needed - self.chain.len() as u32;
        let new_clusters = self.fs.alloc_free_clusters(extra)?;
        if let Some(&last) = self.chain.last() {
            // The previous tail is no longer EOC — link it to the new head.
            self.fs.fat_mut().set(last, new_clusters[0]);
        }
        // Zero the newly-allocated clusters so reads of holes see zeros.
        let cb = self.cb();
        let zero = vec![0u8; cb as usize];
        for &c in &new_clusters {
            self.dev.write_at(self.cluster_offset(c), &zero)?;
        }
        self.chain.extend_from_slice(&new_clusters);
        Ok(())
    }

    /// Truncate the cluster chain to at most `keep` clusters, freeing the rest.
    /// Updates the new tail (if any) to EOC.
    fn truncate_chain(&mut self, keep: u32) -> Result<()> {
        if self.chain.len() as u32 <= keep {
            return Ok(());
        }
        let drained: Vec<u32> = self.chain.drain(keep as usize..).collect();
        for c in &drained {
            self.fs.fat_mut().set(*c, table::FREE);
        }
        if let Some(&last) = self.chain.last() {
            self.fs.fat_mut().set(last, table::EOC);
        }
        // Allow the allocator to revisit these.
        if let Some(&first_freed) = drained.first() {
            self.fs.hint_next_free(first_freed);
        }
        Ok(())
    }

    /// Write `data` into the cluster chain starting at byte offset `off`. The
    /// chain must already be large enough to cover `off + data.len()` bytes.
    fn write_into_chain(&mut self, off: u64, data: &[u8]) -> io::Result<()> {
        let cb = self.cb();
        let mut written: u64 = 0;
        let total = data.len() as u64;
        while written < total {
            let pos = off + written;
            let cluster_idx = (pos / cb) as usize;
            let in_cluster = pos % cb;
            let chunk = (cb - in_cluster).min(total - written);
            let cluster = self.chain[cluster_idx];
            let dst = self.cluster_offset(cluster) + in_cluster;
            let src_start = written as usize;
            let src_end = src_start + chunk as usize;
            self.dev
                .write_at(dst, &data[src_start..src_end])
                .map_err(io::Error::other)?;
            written += chunk;
        }
        Ok(())
    }

    /// Persist the in-memory directory entry: re-encode the 8.3 record with
    /// the current `first_cluster` and `file_size`, then write it back to
    /// disk at its original slot.
    fn flush_dir_entry(&mut self) -> Result<()> {
        let first_cluster = self.chain.first().copied().unwrap_or(0);
        let entry = dir::DirEntry {
            name_83: self.entry_name_83,
            attr: self.entry_attr,
            first_cluster,
            file_size: self.file_size as u32,
        };
        let enc = entry.encode();
        let cb = self.cb() as usize;
        let cluster_idx = self.entry_pos / cb;
        let in_cluster = self.entry_pos % cb;
        let cluster = self.dir_chain[cluster_idx];
        let off = self.cluster_offset(cluster) + in_cluster as u64;
        self.dev.write_at(off, &enc)?;
        Ok(())
    }

    /// Write zeros into the file across the byte range `[start, end)`. The
    /// chain must already cover that range. Used to fill the gap when the
    /// user `set_len`-grows past EOF.
    fn zero_range(&mut self, start: u64, end: u64) -> io::Result<()> {
        if end <= start {
            return Ok(());
        }
        let cb = self.cb();
        let zero = vec![0u8; cb as usize];
        let mut pos = start;
        while pos < end {
            let cluster_idx = (pos / cb) as usize;
            let in_cluster = pos % cb;
            let chunk = (cb - in_cluster).min(end - pos);
            let cluster = self.chain[cluster_idx];
            let dst = self.cluster_offset(cluster) + in_cluster;
            self.dev
                .write_at(dst, &zero[..chunk as usize])
                .map_err(io::Error::other)?;
            pos += chunk;
        }
        Ok(())
    }

    /// Internal write path: extends the chain as needed, writes the bytes,
    /// updates `file_size`, and marks the handle dirty.
    fn write_inner(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let cb = self.cb();
        let new_end = self.pos + buf.len() as u64;
        let needed_clusters = new_end.div_ceil(cb) as u32;
        // If we're writing past EOF and EOF isn't on a cluster boundary, the
        // remainder of the EOF cluster is already zeroed (we always zero new
        // clusters on allocation). If EOF *is* mid-cluster and we leave a
        // gap (seek + write), the gap clusters and the gap bytes inside the
        // current EOF cluster also need to be zeroed — handled below.
        let gap_start = self.file_size;
        let gap_end = self.pos.min(new_end);
        self.ensure_chain_clusters(needed_clusters)
            .map_err(|e| io::Error::other(e.to_string()))?;
        // Zero the gap (between old EOF and the cursor) so unwritten bytes
        // read back as zero. Newly-allocated clusters are already zeroed,
        // but bytes inside the old EOF cluster between old_size and
        // start-of-new-cluster aren't.
        if gap_end > gap_start {
            self.zero_range(gap_start, gap_end)?;
        }
        self.write_into_chain(self.pos, buf)?;
        self.pos += buf.len() as u64;
        if self.pos > self.file_size {
            self.file_size = self.pos;
        }
        self.dirty = true;
        Ok(buf.len())
    }

    /// Internal set_len: grow or shrink the chain + size, mark dirty.
    fn set_len_inner(&mut self, new_len: u64) -> Result<()> {
        let cb = self.cb();
        let needed_clusters = new_len.div_ceil(cb) as u32;
        if new_len > self.file_size {
            // Grow.
            self.ensure_chain_clusters(needed_clusters)?;
            // Fill the gap between old EOF and new EOF with zeros.
            let old_len = self.file_size;
            self.file_size = new_len;
            self.zero_range(old_len, new_len)
                .map_err(crate::Error::Io)?;
        } else if new_len < self.file_size {
            // Shrink.
            self.truncate_chain(needed_clusters)?;
            self.file_size = new_len;
            // Clamp the cursor if it now points past EOF.
            if self.pos > self.file_size {
                self.pos = self.file_size;
            }
        }
        self.dirty = true;
        Ok(())
    }
}

impl<'a> Read for FatFileHandle<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_inner(buf)
    }
}

impl<'a> Write for FatFileHandle<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write_inner(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        self.flush_dir_entry()
            .map_err(|e| io::Error::other(e.to_string()))?;
        self.fs
            .flush(self.dev)
            .map_err(|e| io::Error::other(e.to_string()))?;
        self.dirty = false;
        Ok(())
    }
}

impl<'a> Seek for FatFileHandle<'a> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos: i128 = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(d) => self.file_size as i128 + d as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
        };
        if new_pos < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "fat32: seek to negative offset",
            ));
        }
        // Cap at u32::MAX since FAT file sizes are 32-bit on disk.
        if new_pos > u32::MAX as i128 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "fat32: seek past 4 GiB file-size limit",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

impl<'a> FileHandle for FatFileHandle<'a> {
    fn len(&self) -> u64 {
        self.file_size
    }

    fn set_len(&mut self, new_len: u64) -> Result<()> {
        if new_len > u32::MAX as u64 {
            return Err(crate::Error::InvalidArgument(
                "fat32: files cannot exceed 4 GiB".into(),
            ));
        }
        self.set_len_inner(new_len)
    }

    fn sync(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        self.flush_dir_entry()?;
        self.fs.flush(self.dev)?;
        self.dirty = false;
        Ok(())
    }
}

impl<'a> Drop for FatFileHandle<'a> {
    fn drop(&mut self) {
        // Best-effort: persist on drop so the file is durable even when the
        // caller forgets to `sync`. Errors are swallowed because Drop can't
        // return them; tests should call sync() explicitly to surface I/O
        // failures.
        if self.dirty {
            let _ = self.flush_dir_entry();
            let _ = self.fs.flush(self.dev);
            self.dirty = false;
        }
    }
}
