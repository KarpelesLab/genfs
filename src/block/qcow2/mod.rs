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

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use header::Header;
use l1l2::L1L2;

use super::BlockDevice;
use crate::Result;

/// A [`BlockDevice`] backed by a qcow2 image.
pub struct Qcow2Backend {
    file: File,
    header: Header,
    cluster_size: u64,
    l1l2: L1L2,
    /// Virtual cursor for the `Read`/`Write`/`Seek` impls. The file's
    /// own cursor is repositioned per-cluster on the read side.
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
        Ok(Self {
            file,
            header,
            cluster_size,
            l1l2,
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
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        // Phase A is read-only; writes land in Phase B (refcount +
        // allocate-on-write). Erroring here keeps the BlockDevice trait
        // contract honest without pretending the write succeeded.
        Err(io::Error::other(
            "qcow2: write path not implemented (Phase B)",
        ))
    }

    fn flush(&mut self) -> io::Result<()> {
        // No buffered writes to flush.
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
        self.file.sync_data()?;
        Ok(())
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.read_virtual(offset, buf)
    }

    fn write_at(&mut self, _offset: u64, _buf: &[u8]) -> Result<()> {
        Err(crate::Error::Unsupported(
            "qcow2: write path not implemented (Phase B)".into(),
        ))
    }

    fn zero_range(&mut self, _offset: u64, _len: u64) -> Result<()> {
        Err(crate::Error::Unsupported(
            "qcow2: write path not implemented (Phase B)".into(),
        ))
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

        // Write path errors with Unsupported in Phase A.
        let err = back.write_at(0, &[0u8; 16]).unwrap_err();
        assert!(matches!(err, crate::Error::Unsupported(_)));

        // Read trait works via cursor.
        back.seek(SeekFrom::Start(0)).unwrap();
        let mut chunk = [0u8; 1024];
        let n = back.read(&mut chunk).unwrap();
        assert_eq!(n, 1024);
        assert!(chunk.iter().all(|&b| b == 0));
    }
}
