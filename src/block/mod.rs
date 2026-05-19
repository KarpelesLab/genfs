//! Block-device abstraction — the bottom layer of the fstool stack.
//!
//! A [`BlockDevice`] is a seekable byte-addressable store. Every higher layer
//! (partition table, filesystem) reads and writes through this trait, which
//! makes it trivial to substitute an on-disk file with an in-memory buffer in
//! tests or with a sub-range view when carving partitions.
//!
//! ## Invariants
//!
//! - `total_size()` is the logical capacity in bytes; reads and writes outside
//!   `[0, total_size())` MUST be rejected (the trait returns a short read /
//!   short write at the boundary via the standard `Read`/`Write` contract, and
//!   fstool's explicit positional helpers return [`crate::Error::OutOfBounds`]).
//! - Implementations are free to back themselves with sparse storage. Bytes
//!   that have never been written MUST read as zero.
//! - `block_size()` reports the *logical* sector size — usually 512 — and is
//!   purely advisory; it does not constrain the alignment of reads or writes.
//!
//! ## Streaming guarantee
//!
//! The whole point of this trait is to support multi-gigabyte images without
//! buffering them in RAM. Backends MUST NOT pull the full device into memory.
//! [`MemoryBackend`] is the only intentionally in-RAM backend and exists for
//! tests; it carries a soft cap to prevent accidental use on huge images.

use std::io::{Read, Seek, Write};

use crate::Result;

pub mod file;
pub mod memory;
pub mod qcow2;
pub mod sliced;

pub use file::FileBackend;
pub use memory::MemoryBackend;
pub use qcow2::Qcow2Backend;
pub use sliced::SlicedBackend;

use std::path::Path;

/// Open `path` as a [`BlockDevice`], picking the backend automatically.
/// qcow2 files (detected by magic `"QFI\xfb"`) get a [`Qcow2Backend`];
/// everything else — regular files, block devices — gets a
/// [`FileBackend`]. The detection reads just the first four bytes.
pub fn open_image(path: &Path) -> crate::Result<Box<dyn BlockDevice>> {
    if Qcow2Backend::probe(path)? {
        Ok(Box::new(Qcow2Backend::open(path)?))
    } else {
        Ok(Box::new(FileBackend::open(path)?))
    }
}

/// Options for [`create_image`].
#[derive(Debug, Clone, Copy)]
pub struct CreateOpts {
    /// qcow2 cluster size in bytes (power of two, ≥ 512). Default 64 KiB,
    /// matching qemu-img. Ignored when creating a raw image.
    pub cluster_size: u32,
}

impl Default for CreateOpts {
    fn default() -> Self {
        Self {
            cluster_size: 65_536,
        }
    }
}

/// Create a new image at `path` of capacity `virtual_size` bytes. The
/// backend is chosen by the path's extension: `.qcow2` (or `.qcow` /
/// `.q2`) → [`Qcow2Backend`], everything else → [`FileBackend`] (sparse
/// raw file or block device).
pub fn create_image(
    path: &Path,
    virtual_size: u64,
    opts: &CreateOpts,
) -> crate::Result<Box<dyn BlockDevice>> {
    if is_qcow2_extension(path) {
        Ok(Box::new(Qcow2Backend::create(
            path,
            virtual_size,
            opts.cluster_size,
        )?))
    } else {
        Ok(Box::new(FileBackend::create(path, virtual_size)?))
    }
}

fn is_qcow2_extension(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
        return false;
    };
    matches!(ext.to_ascii_lowercase().as_str(), "qcow2" | "qcow" | "q2")
}

/// A seekable byte-addressable store of fixed capacity.
///
/// Implementors compose `Read + Write + Seek` so the standard library's
/// streaming APIs work directly. The extra trait methods expose information
/// that higher layers need (advisory sector size, total capacity, sparse-zero
/// hint, durability flush).
pub trait BlockDevice: Read + Write + Seek + Send {
    /// Advisory logical sector size, in bytes. Usually 512. Higher layers may
    /// use this for alignment hints; it does not constrain valid I/O offsets.
    fn block_size(&self) -> u32;

    /// Total capacity of the device in bytes.
    fn total_size(&self) -> u64;

    /// Hint that the range `[offset, offset+len)` should read as zero. The
    /// default implementation actually writes zero bytes; backends with sparse
    /// support (file with `set_len`, memory) may override to do nothing when
    /// the underlying storage is already zero-initialised.
    fn zero_range(&mut self, offset: u64, len: u64) -> Result<()> {
        let size = self.total_size();
        if offset.checked_add(len).is_none_or(|end| end > size) {
            return Err(crate::Error::OutOfBounds { offset, len, size });
        }
        if len == 0 {
            return Ok(());
        }
        self.seek(std::io::SeekFrom::Start(offset))?;
        let zero = [0u8; 4096];
        let mut remaining = len;
        while remaining > 0 {
            let n = remaining.min(zero.len() as u64) as usize;
            self.write_all(&zero[..n])?;
            remaining -= n as u64;
        }
        Ok(())
    }

    /// Persist outstanding writes. For [`FileBackend`] this is `fsync`; for
    /// [`MemoryBackend`] it is a no-op.
    fn sync(&mut self) -> Result<()>;

    /// Positional read — fills `buf` from `offset` without moving the
    /// implicit stream cursor across calls (the cursor IS seeked, but callers
    /// should not rely on its position after this method returns).
    ///
    /// Returns [`crate::Error::OutOfBounds`] if `offset + buf.len()` exceeds
    /// [`total_size`](Self::total_size). Implementations that can do a true
    /// `pread` (positional read without modifying the cursor) are encouraged
    /// to override this.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let size = self.total_size();
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size,
            })?;
        if end > size {
            return Err(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size,
            });
        }
        self.seek(std::io::SeekFrom::Start(offset))?;
        self.read_exact(buf)?;
        Ok(())
    }

    /// Positional write — writes `buf` at `offset`. Mirrors
    /// [`read_at`](Self::read_at)'s semantics.
    fn write_at(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        let size = self.total_size();
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size,
            })?;
        if end > size {
            return Err(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size,
            });
        }
        self.seek(std::io::SeekFrom::Start(offset))?;
        self.write_all(buf)?;
        Ok(())
    }
}
