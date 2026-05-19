//! qcow2 — QEMU's copy-on-write disk format, as a [`BlockDevice`].
//!
//! Supports reading v2 and v3 images; writes Phase B (refcount + create)
//! and unsupported features (compression, encryption, snapshots, backing
//! files, external data files, extended L2 entries) error with
//! [`crate::Error::Unsupported`] on open.
//!
//! ## Layout
//!
//! A qcow2 file is a sequence of fixed-size clusters (typically 64 KiB).
//! The first cluster carries the header. From there, three sets of
//! metadata clusters live alongside data clusters:
//!
//! - **Refcount table** (one or more clusters, pointed at by
//!   `refcount_table_offset`): array of u64 entries pointing to refcount
//!   *blocks*.
//! - **Refcount blocks**: array of u16 refcounts, one per data/metadata
//!   cluster. Used to find free clusters when allocating.
//! - **L1 table** (`l1_table_offset`): array of u64 entries pointing to
//!   **L2 tables**, which in turn point to data clusters.
//!
//! `total_size()` returns the virtual size from the header; the backing
//! file is allocate-on-write, so a freshly-created 100 GiB image is only
//! a few clusters on disk until you write to it.
//!
//! ## Concurrency
//!
//! qcow2 is not safe to share between writers. `Qcow2Backend` holds the
//! file open `O_RDWR` without an exclusive lock — the caller is expected
//! to not have another writer pointed at the same file.

pub mod header;
pub mod l1l2;
pub mod refcount;

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use header::Header;
use l1l2::{COPIED, L1L2};
use refcount::Refcount;

use super::BlockDevice;
use crate::Result;

/// A [`BlockDevice`] backed by a qcow2 image.
pub struct Qcow2Backend {
    file: File,
    header: Header,
    cluster_size: u64,
    l1l2: L1L2,
    refcount: Refcount,
    /// Current backing-file size in bytes; grows when allocate-on-write
    /// extends the file past the previous EOF.
    file_len: u64,
    /// Virtual cursor for the `Read`/`Write`/`Seek` impls.
    cursor: u64,
}

impl std::fmt::Debug for Qcow2Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qcow2Backend")
            .field("version", &self.header.version)
            .field("cluster_size", &self.cluster_size)
            .field("virtual_size", &self.header.size)
            .field("l1_size", &self.header.l1_size)
            .finish()
    }
}

impl Qcow2Backend {
    /// Open an existing qcow2 file read+write. Errors with `Unsupported`
    /// if the image uses features fstool doesn't implement.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())?;
        let mut buf = [0u8; header::V3_HEADER_LEN];
        file.read_exact(&mut buf)?;
        let header = Header::decode(&buf)?;
        let cluster_size = header.cluster_size();
        let l1l2 = L1L2::load(&mut file, &header)?;
        let refcount = Refcount::load(&mut file, &header)?;
        let file_len = file.metadata()?.len();
        Ok(Self {
            file,
            header,
            cluster_size,
            l1l2,
            refcount,
            file_len,
            cursor: 0,
        })
    }

    /// Format a fresh qcow2 v3 image at `path`. The file is created
    /// (truncating any existing one) and seeded with the header, an
    /// empty refcount table + refcount block, and an L1 table. All
    /// data clusters are allocate-on-write.
    pub fn create<P: AsRef<Path>>(path: P, virtual_size: u64, cluster_size: u32) -> Result<Self> {
        if !cluster_size.is_power_of_two() || cluster_size < 512 {
            return Err(crate::Error::InvalidArgument(format!(
                "qcow2: cluster_size {cluster_size} must be a power of two ≥ 512"
            )));
        }
        let cs = cluster_size as u64;
        let cluster_bits = cs.trailing_zeros();

        // Compute L1 size: one L2 cluster covers (cs/8) clusters, which
        // covers (cs/8) * cs virtual bytes. l1 entries needed:
        let l2_coverage = (cs / 8) * cs;
        let l1_size = virtual_size.div_ceil(l2_coverage) as u32;
        // L1 size must be a power of two? No — but it does need to fit
        // in some number of clusters. Round up `l1_size` to a multiple
        // of (cs / 8) so the L1 table is a whole number of clusters.
        let l1_per_cluster = (cs / 8) as u32;
        let l1_clusters = l1_size.div_ceil(l1_per_cluster);
        let l1_size = l1_clusters * l1_per_cluster;

        // Layout (in clusters):
        //   0:                header
        //   1:                refcount table (1 cluster)
        //   2:                refcount block 0
        //   3..3+l1_clusters: L1 table
        let refcount_table_cluster = 1u64;
        let refcount_block_cluster = 2u64;
        let l1_first_cluster = 3u64;
        let next_free_cluster = l1_first_cluster + l1_clusters as u64;
        let file_len = next_free_cluster * cs;

        // The clusters we just laid out must all have refcount=1.
        let initial: Vec<u64> = {
            let mut v = Vec::new();
            v.push(0); // header
            v.push(refcount_table_cluster);
            v.push(refcount_block_cluster);
            for i in 0..l1_clusters as u64 {
                v.push(l1_first_cluster + i);
            }
            v
        };

        // Build the header.
        let header = Header {
            version: header::VERSION_V3,
            backing_file_offset: 0,
            backing_file_size: 0,
            cluster_bits,
            size: virtual_size,
            crypt_method: 0,
            l1_size,
            l1_table_offset: l1_first_cluster * cs,
            refcount_table_offset: refcount_table_cluster * cs,
            refcount_table_clusters: 1,
            nb_snapshots: 0,
            snapshots_offset: 0,
            incompatible_features: 0,
            compatible_features: 0,
            autoclear_features: 0,
            refcount_order: 4,
            header_length: header::V3_HEADER_LEN as u32,
        };

        // Create the backing file at exactly `file_len` bytes,
        // zero-filled by `set_len` (sparse).
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())?;
        file.set_len(file_len)?;

        // Write the header at byte 0, padded to the cluster.
        file.seek(SeekFrom::Start(0))?;
        let mut cluster0 = vec![0u8; cs as usize];
        cluster0[..header::V3_HEADER_LEN].copy_from_slice(&header.encode_v3());
        file.write_all(&cluster0)?;

        // L1 table starts all-zero — set_len already zero-filled it.
        let mut l1l2 = L1L2 {
            cluster_size: cs,
            cluster_bits,
            l2_entries: (cs / 8) as usize,
            l1: vec![0u64; l1_size as usize],
            l1_table_offset: l1_first_cluster * cs,
            l2_cache: std::collections::HashMap::new(),
            l2_cache_cap: 32,
        };

        // Refcount table + initial refcount block live in memory; flush
        // them so the on-disk view matches.
        let mut refcount = Refcount::new_fresh(
            cs,
            refcount_table_cluster * cs,
            refcount_block_cluster * cs,
            &initial,
        );
        refcount.flush(&mut file)?;
        l1l2.flush(&mut file)?;
        file.sync_data()?;

        Ok(Self {
            file,
            header,
            cluster_size: cs,
            l1l2,
            refcount,
            file_len,
            cursor: 0,
        })
    }

    /// Read-only convenience: open and confirm this is a qcow2 image.
    pub fn probe<P: AsRef<Path>>(path: P) -> Result<bool> {
        let mut file = File::open(path.as_ref())?;
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_err() {
            return Ok(false);
        }
        Ok(magic == header::MAGIC)
    }

    /// The decoded header — exposed for diagnostics.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Write `buf` to virtual offset `offset`, allocating physical
    /// clusters and L2 tables on demand.
    fn write_virtual(&mut self, mut offset: u64, mut buf: &[u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size: self.header.size,
            })?;
        if end > self.header.size {
            return Err(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size: self.header.size,
            });
        }
        let cs = self.cluster_size;
        while !buf.is_empty() {
            let in_cluster = offset & (cs - 1);
            let take = ((cs - in_cluster) as usize).min(buf.len());
            let (chunk, rest) = buf.split_at(take);
            let cluster_start = offset - in_cluster;
            let phys = self.ensure_mapping(cluster_start)?;
            self.file.seek(SeekFrom::Start(phys + in_cluster))?;
            self.file.write_all(chunk)?;
            offset += take as u64;
            buf = rest;
        }
        Ok(())
    }

    /// Make sure the cluster covering virtual offset `vaddr_cluster_aligned`
    /// has a physical mapping, allocating one if not. Returns the
    /// physical byte offset of the cluster.
    fn ensure_mapping(&mut self, vaddr: u64) -> Result<u64> {
        let (l1_idx, l2_idx, _) = self.l1l2.split_addr(vaddr);
        let l1_entry = self.l1l2.l1[l1_idx];
        let l2_off = l1_entry & l1l2::OFFSET_MASK;
        let (l2_off, _) = if l2_off == 0 {
            // Allocate an L2 cluster.
            let cluster_idx = self
                .refcount
                .alloc_cluster(&mut self.file, &mut self.file_len)?;
            // Make sure the file is long enough to hold the new L2 cluster.
            let new_end = (cluster_idx + 1) * self.cluster_size;
            if new_end > self.file_len {
                self.file_len = new_end;
            }
            self.file.set_len(self.file_len)?;
            let new_l2_off = cluster_idx * self.cluster_size;
            self.l1l2.insert_empty_l2(new_l2_off);
            self.l1l2.set_l1(l1_idx, new_l2_off | COPIED);
            (new_l2_off, l2_idx)
        } else {
            // Cache-load the L2 if it isn't already in cache.
            let _ = self.l1l2.lookup(&mut self.file, vaddr)?;
            (l2_off, l2_idx)
        };

        let l2_entry = self
            .l1l2
            .l2_cache
            .get(&l2_off)
            .expect("L2 just loaded/created")
            .entries[l2_idx];
        let data_off = l2_entry & l1l2::OFFSET_MASK;
        if data_off != 0 {
            return Ok(data_off);
        }
        // Allocate a data cluster.
        let data_cluster = self
            .refcount
            .alloc_cluster(&mut self.file, &mut self.file_len)?;
        let new_data_off = data_cluster * self.cluster_size;
        let new_end = new_data_off + self.cluster_size;
        if new_end > self.file_len {
            self.file_len = new_end;
        }
        self.file.set_len(self.file_len)?;
        self.l1l2
            .set_l2_entry(l2_off, l2_idx, new_data_off | COPIED)?;
        Ok(new_data_off)
    }

    /// Read `buf.len()` bytes starting at virtual offset `offset`. Walks
    /// the L1/L2 mapping cluster-by-cluster; unallocated clusters return
    /// zeroes.
    fn read_virtual(&mut self, mut offset: u64, mut buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size: self.header.size,
            })?;
        if end > self.header.size {
            return Err(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size: self.header.size,
            });
        }
        let cs = self.cluster_size;
        while !buf.is_empty() {
            let in_cluster = offset & (cs - 1);
            let take = ((cs - in_cluster) as usize).min(buf.len());
            let (chunk, rest) = buf.split_at_mut(take);
            let cluster_start = offset - in_cluster;
            match self.l1l2.lookup(&mut self.file, cluster_start)? {
                Some(phys) => {
                    self.file.seek(SeekFrom::Start(phys + in_cluster))?;
                    self.file.read_exact(chunk)?;
                }
                None => {
                    chunk.fill(0);
                }
            }
            offset += take as u64;
            buf = rest;
        }
        Ok(())
    }
}

impl Read for Qcow2Backend {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let remaining = self.header.size.saturating_sub(self.cursor);
        let n = (buf.len() as u64).min(remaining) as usize;
        if n == 0 {
            return Ok(0);
        }
        self.read_virtual(self.cursor, &mut buf[..n])
            .map_err(io::Error::other)?;
        self.cursor += n as u64;
        Ok(n)
    }
}

impl Write for Qcow2Backend {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let remaining = self.header.size.saturating_sub(self.cursor);
        let n = (buf.len() as u64).min(remaining) as usize;
        if n == 0 {
            return Ok(0);
        }
        self.write_virtual(self.cursor, &buf[..n])
            .map_err(io::Error::other)?;
        self.cursor += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        // The qcow2 layer flushes its metadata on `sync`; the std
        // `Write::flush` contract just says "drain buffered data", and
        // we have no internal buffer.
        Ok(())
    }
}

impl Seek for Qcow2Backend {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let size = self.header.size;
        let new = match pos {
            SeekFrom::Start(n) => n,
            SeekFrom::End(n) => size
                .checked_add_signed(n)
                .ok_or_else(|| io::Error::other("qcow2: seek past i64 bounds"))?,
            SeekFrom::Current(n) => self
                .cursor
                .checked_add_signed(n)
                .ok_or_else(|| io::Error::other("qcow2: seek past i64 bounds"))?,
        };
        self.cursor = new;
        Ok(self.cursor)
    }
}

impl BlockDevice for Qcow2Backend {
    fn block_size(&self) -> u32 {
        512
    }

    fn total_size(&self) -> u64 {
        self.header.size
    }

    fn sync(&mut self) -> Result<()> {
        self.l1l2.flush(&mut self.file)?;
        self.refcount.flush(&mut self.file)?;
        self.file.sync_data()?;
        Ok(())
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.read_virtual(offset, buf)
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        self.write_virtual(offset, buf)
    }

    fn zero_range(&mut self, offset: u64, len: u64) -> Result<()> {
        // We don't implement discard/punch here — just write zeros
        // through the allocator. That's allocation-heavy for big ranges
        // but produces a correct image and matches the BlockDevice
        // trait's default behaviour.
        if len == 0 {
            return Ok(());
        }
        let zero = vec![0u8; 4096];
        let mut written = 0u64;
        while written < len {
            let n = (len - written).min(zero.len() as u64) as usize;
            self.write_virtual(offset + written, &zero[..n])?;
            written += n as u64;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test using a hand-rolled minimal qcow2 image: header, an
    /// empty L1 entry, and a small refcount table. The reader should
    /// return zeros for every offset (everything unallocated).
    #[test]
    fn read_returns_zeros_on_fresh_image() {
        // Generate a minimal v3 image in a tempfile and read it back.
        // Cluster size 64 KiB, virtual size 64 MiB, one L1 entry pointing
        // at nothing (everything unallocated).
        use std::io::Write;
        use tempfile::NamedTempFile;

        let tmp = NamedTempFile::new().unwrap();
        let cluster_size = 65536u64;
        let virtual_size = 64u64 * 1024 * 1024;
        let h = Header {
            version: header::VERSION_V3,
            backing_file_offset: 0,
            backing_file_size: 0,
            cluster_bits: 16,
            size: virtual_size,
            crypt_method: 0,
            // virtual_size / cluster_size = 1024 clusters; one L2 cluster
            // (8192 entries) covers 8192 clusters, so l1_size = 1.
            l1_size: 1,
            l1_table_offset: 3 * cluster_size,
            refcount_table_offset: cluster_size,
            refcount_table_clusters: 1,
            nb_snapshots: 0,
            snapshots_offset: 0,
            incompatible_features: 0,
            compatible_features: 0,
            autoclear_features: 0,
            refcount_order: 4,
            header_length: header::V3_HEADER_LEN as u32,
        };
        let mut f = std::fs::File::create(tmp.path()).unwrap();
        // Cluster 0: header padded to a cluster.
        let mut c0 = vec![0u8; cluster_size as usize];
        c0[..header::V3_HEADER_LEN].copy_from_slice(&h.encode_v3());
        f.write_all(&c0).unwrap();
        // Cluster 1: refcount table, all-zero (we don't read it on the
        // pure read path).
        f.write_all(&vec![0u8; cluster_size as usize]).unwrap();
        // Cluster 2: refcount block, all-zero.
        f.write_all(&vec![0u8; cluster_size as usize]).unwrap();
        // Cluster 3: L1 table, one entry == 0 (unallocated).
        f.write_all(&vec![0u8; cluster_size as usize]).unwrap();
        f.sync_all().unwrap();
        drop(f);

        let mut back = Qcow2Backend::open(tmp.path()).unwrap();
        assert_eq!(back.total_size(), virtual_size);
        assert_eq!(back.header.cluster_size(), cluster_size);

        // Reading from anywhere returns zeros.
        let mut buf = [0xffu8; 4096];
        back.read_at(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));

        let mut buf2 = [0xffu8; 8192];
        back.read_at(virtual_size - 8192, &mut buf2).unwrap();
        assert!(buf2.iter().all(|&b| b == 0));

        // OOB rejection.
        let mut tail = [0u8; 16];
        let err = back.read_at(virtual_size, &mut tail).unwrap_err();
        assert!(matches!(err, crate::Error::OutOfBounds { .. }));

        // Read trait works via cursor.
        back.seek(SeekFrom::Start(0)).unwrap();
        let mut chunk = [0u8; 1024];
        let n = back.read(&mut chunk).unwrap();
        assert_eq!(n, 1024);
        assert!(chunk.iter().all(|&b| b == 0));
    }
}
