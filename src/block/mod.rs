//! Block-device abstraction — the bottom layer of the genfs stack.
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
//!   genfs's explicit positional helpers return [`Error::OutOfBounds`]).
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
pub mod sliced;

pub use file::FileBackend;
pub use memory::MemoryBackend;
pub use sliced::SlicedBackend;

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
    /// Returns [`Error::OutOfBounds`] if `offset + buf.len()` exceeds
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

    /// Positional write — writes `buf` at `offset`. Mirrors [`read_at`]'s
    /// semantics.
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
