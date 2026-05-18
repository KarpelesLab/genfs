//! Sub-range view over a parent [`BlockDevice`].
//!
//! A `SlicedBackend` borrows a parent device and exposes only the byte range
//! `[start, start + len)` as its own device. Every read and write is bounds-
//! checked against the slice; the parent never sees a byte outside the slice
//! regardless of seek-cursor confusion in the parent.
//!
//! This is how partitions get their isolated view: the partition-table layer
//! constructs one `SlicedBackend` per partition, and the filesystem layer
//! formats / mounts that slice as if it were the whole device.

use std::io::{self, Read, Seek, SeekFrom, Write};

use super::BlockDevice;
use crate::Result;

/// A bounded view into a region of `parent` starting at byte `start`, of
/// length `len`. Holds an exclusive mutable borrow of the parent for the
/// lifetime `'a` so the parent cannot be seeked or written behind our back.
#[derive(Debug)]
pub struct SlicedBackend<'a, B: BlockDevice + ?Sized> {
    parent: &'a mut B,
    start: u64,
    len: u64,
    /// Position *within* the slice, in [0, len].
    cursor: u64,
    /// Cached parent block size at construction time; the parent's value is
    /// fixed in practice but we snapshot it to avoid relying on that.
    block_size: u32,
}

impl<'a, B: BlockDevice + ?Sized> SlicedBackend<'a, B> {
    /// Construct a slice covering `[start, start + len)` of `parent`.
    ///
    /// Returns [`crate::Error::OutOfBounds`] if the slice extends past the parent's
    /// `total_size()` or if the arithmetic overflows.
    pub fn new(parent: &'a mut B, start: u64, len: u64) -> Result<Self> {
        let parent_size = parent.total_size();
        let end = start.checked_add(len).ok_or(crate::Error::OutOfBounds {
            offset: start,
            len,
            size: parent_size,
        })?;
        if end > parent_size {
            return Err(crate::Error::OutOfBounds {
                offset: start,
                len,
                size: parent_size,
            });
        }
        let block_size = parent.block_size();
        Ok(Self {
            parent,
            start,
            len,
            cursor: 0,
            block_size,
        })
    }

    /// Byte offset of this slice within its parent.
    pub fn parent_offset(&self) -> u64 {
        self.start
    }

    /// Map an in-slice offset to a parent offset, returning `OutOfBounds`
    /// if the in-slice range overflows the slice.
    fn translate(&self, offset: u64, len: u64) -> Result<u64> {
        let end = offset.checked_add(len).ok_or(crate::Error::OutOfBounds {
            offset,
            len,
            size: self.len,
        })?;
        if end > self.len {
            return Err(crate::Error::OutOfBounds {
                offset,
                len,
                size: self.len,
            });
        }
        Ok(self.start + offset)
    }
}

impl<'a, B: BlockDevice + ?Sized> Read for SlicedBackend<'a, B> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.cursor >= self.len {
            return Ok(0);
        }
        let available = self.len - self.cursor;
        let n = available.min(buf.len() as u64) as usize;
        self.parent
            .seek(SeekFrom::Start(self.start + self.cursor))?;
        let read = self.parent.read(&mut buf[..n])?;
        self.cursor += read as u64;
        Ok(read)
    }
}

impl<'a, B: BlockDevice + ?Sized> Write for SlicedBackend<'a, B> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.cursor >= self.len {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "write past end of SlicedBackend",
            ));
        }
        let available = self.len - self.cursor;
        let n = available.min(buf.len() as u64) as usize;
        self.parent
            .seek(SeekFrom::Start(self.start + self.cursor))?;
        let written = self.parent.write(&buf[..n])?;
        self.cursor += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.parent.flush()
    }
}

impl<'a, B: BlockDevice + ?Sized> Seek for SlicedBackend<'a, B> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(d) => self.len as i128 + d as i128,
            SeekFrom::Current(d) => self.cursor as i128 + d as i128,
        };
        if new < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek before start of slice",
            ));
        }
        // We allow seeking up to `len` (one past the last byte), matching the
        // standard library's tolerance for end-of-stream seeks. Writes from
        // there will short-write.
        self.cursor = new as u64;
        Ok(self.cursor)
    }
}

impl<'a, B: BlockDevice + ?Sized> BlockDevice for SlicedBackend<'a, B> {
    fn block_size(&self) -> u32 {
        self.block_size
    }

    fn total_size(&self) -> u64 {
        self.len
    }

    fn zero_range(&mut self, offset: u64, len: u64) -> Result<()> {
        let parent_off = self.translate(offset, len)?;
        self.parent.zero_range(parent_off, len)
    }

    fn sync(&mut self) -> Result<()> {
        self.parent.sync()
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let parent_off = self.translate(offset, buf.len() as u64)?;
        self.parent.read_at(parent_off, buf)
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        let parent_off = self.translate(offset, buf.len() as u64)?;
        self.parent.write_at(parent_off, buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn slice_covers_subrange() {
        let mut parent = MemoryBackend::new(1024);
        let mut slice = SlicedBackend::new(&mut parent, 256, 512).unwrap();
        assert_eq!(slice.total_size(), 512);
        slice.write_at(0, &[0xab; 16]).unwrap();
        let mut got = [0u8; 16];
        slice.read_at(0, &mut got).unwrap();
        assert!(got.iter().all(|&b| b == 0xab));
    }

    #[test]
    fn slice_isolated_from_parent_bytes_outside() {
        let mut parent = MemoryBackend::new(1024);
        // Plant a sentinel before the slice and another after.
        parent.write_at(0, &[0x11; 256]).unwrap();
        parent.write_at(768, &[0x22; 256]).unwrap();
        // Slice covers [256, 768). Fill it entirely.
        {
            let mut slice = SlicedBackend::new(&mut parent, 256, 512).unwrap();
            slice.write_at(0, &[0x33; 512]).unwrap();
        }
        let mut leading = [0u8; 256];
        let mut middle = [0u8; 512];
        let mut trailing = [0u8; 256];
        parent.read_at(0, &mut leading).unwrap();
        parent.read_at(256, &mut middle).unwrap();
        parent.read_at(768, &mut trailing).unwrap();
        assert!(
            leading.iter().all(|&b| b == 0x11),
            "slice leaked before its start"
        );
        assert!(middle.iter().all(|&b| b == 0x33));
        assert!(
            trailing.iter().all(|&b| b == 0x22),
            "slice leaked past its end"
        );
    }

    #[test]
    fn slice_rejects_out_of_parent() {
        let mut parent = MemoryBackend::new(1024);
        let err = SlicedBackend::new(&mut parent, 800, 500).unwrap_err();
        assert!(matches!(err, crate::Error::OutOfBounds { .. }));
    }

    #[test]
    fn slice_rejects_write_past_end() {
        let mut parent = MemoryBackend::new(1024);
        let mut slice = SlicedBackend::new(&mut parent, 0, 64).unwrap();
        let err = slice.write_at(50, &[0u8; 32]).unwrap_err();
        assert!(matches!(err, crate::Error::OutOfBounds { .. }));
    }

    #[test]
    fn slice_rejects_overflow_offset() {
        let mut parent = MemoryBackend::new(1024);
        let mut slice = SlicedBackend::new(&mut parent, 0, 64).unwrap();
        let err = slice.write_at(u64::MAX - 10, &[0u8; 32]).unwrap_err();
        assert!(matches!(err, crate::Error::OutOfBounds { .. }));
    }

    #[test]
    fn streaming_read_short_reads_at_boundary() {
        let mut parent = MemoryBackend::new(1024);
        let mut slice = SlicedBackend::new(&mut parent, 100, 50).unwrap();
        // Streaming read past the end should short-read, matching std::io::Read.
        slice.seek(SeekFrom::Start(40)).unwrap();
        let mut buf = [0u8; 100];
        let n = slice.read(&mut buf).unwrap();
        assert_eq!(n, 10);
    }
}
