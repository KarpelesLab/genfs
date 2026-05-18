//! In-memory [`BlockDevice`] backed by a `Vec<u8>`. Test fixture only.

use std::io::{self, Read, Seek, SeekFrom, Write};

use super::BlockDevice;
use crate::Result;

/// Soft upper bound on `MemoryBackend` capacity: 256 MiB. Anything larger
/// almost certainly indicates a bug (the caller meant to use a file backend).
const SOFT_MAX_CAPACITY: u64 = 256 * 1024 * 1024;

/// A [`BlockDevice`] backed by a fixed-capacity `Vec<u8>`. Bytes default to
/// zero. Intended for unit tests; emit a `log::warn!` if larger than
/// [`SOFT_MAX_CAPACITY`].
#[derive(Debug, Clone)]
pub struct MemoryBackend {
    buf: Vec<u8>,
    cursor: u64,
    block_size: u32,
}

impl MemoryBackend {
    /// Create a backend of exactly `size` bytes, all zero.
    pub fn new(size: u64) -> Self {
        Self::with_block_size(size, 512)
    }

    /// Create a backend of exactly `size` bytes with a custom advisory
    /// sector size.
    pub fn with_block_size(size: u64, block_size: u32) -> Self {
        assert!(
            block_size.is_power_of_two(),
            "block_size must be a power of two"
        );
        if size > SOFT_MAX_CAPACITY {
            log::warn!(
                "MemoryBackend created with size {size} bytes (> {SOFT_MAX_CAPACITY} soft cap); \
                 prefer FileBackend for large images"
            );
        }
        Self {
            buf: vec![0; size as usize],
            cursor: 0,
            block_size,
        }
    }

    /// Borrow the underlying byte buffer.
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }
}

impl Read for MemoryBackend {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.cursor >= self.buf.len() as u64 {
            return Ok(0);
        }
        let start = self.cursor as usize;
        let available = self.buf.len() - start;
        let n = available.min(out.len());
        out[..n].copy_from_slice(&self.buf[start..start + n]);
        self.cursor += n as u64;
        Ok(n)
    }
}

impl Write for MemoryBackend {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if self.cursor >= self.buf.len() as u64 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "write past end of MemoryBackend",
            ));
        }
        let start = self.cursor as usize;
        let available = self.buf.len() - start;
        let n = available.min(data.len());
        self.buf[start..start + n].copy_from_slice(&data[..n]);
        self.cursor += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for MemoryBackend {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let total = self.buf.len() as u64;
        let new = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(d) => total as i128 + d as i128,
            SeekFrom::Current(d) => self.cursor as i128 + d as i128,
        };
        if new < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek before start",
            ));
        }
        self.cursor = new as u64;
        Ok(self.cursor)
    }
}

impl BlockDevice for MemoryBackend {
    fn block_size(&self) -> u32 {
        self.block_size
    }

    fn total_size(&self) -> u64 {
        self.buf.len() as u64
    }

    fn zero_range(&mut self, offset: u64, len: u64) -> Result<()> {
        let size = self.total_size();
        if offset.checked_add(len).is_none_or(|e| e > size) {
            return Err(crate::Error::OutOfBounds { offset, len, size });
        }
        let start = offset as usize;
        let end = (offset + len) as usize;
        self.buf[start..end].fill(0);
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_zero_initialised() {
        let mut dev = MemoryBackend::new(64);
        let mut buf = [0xffu8; 32];
        dev.read_at(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn write_then_read_at_roundtrip() {
        let mut dev = MemoryBackend::new(1024);
        let payload: Vec<u8> = (0..256u16).map(|i| i as u8).collect();
        dev.write_at(100, &payload).unwrap();
        let mut got = vec![0u8; 256];
        dev.read_at(100, &mut got).unwrap();
        assert_eq!(payload, got);
    }

    #[test]
    fn write_at_past_end_rejected() {
        let mut dev = MemoryBackend::new(64);
        let err = dev.write_at(50, &[0u8; 32]).unwrap_err();
        match err {
            crate::Error::OutOfBounds { offset, len, size } => {
                assert_eq!((offset, len, size), (50, 32, 64));
            }
            _ => panic!("expected OutOfBounds, got {err:?}"),
        }
    }

    #[test]
    fn zero_range_clears_existing_data() {
        let mut dev = MemoryBackend::new(128);
        dev.write_at(0, &[0xaa; 128]).unwrap();
        dev.zero_range(32, 32).unwrap();
        let mut buf = [0u8; 128];
        dev.read_at(0, &mut buf).unwrap();
        assert!(buf[..32].iter().all(|&b| b == 0xaa));
        assert!(buf[32..64].iter().all(|&b| b == 0x00));
        assert!(buf[64..].iter().all(|&b| b == 0xaa));
    }

    #[test]
    fn seek_modes_consistent() {
        let mut dev = MemoryBackend::new(100);
        assert_eq!(dev.seek(SeekFrom::Start(10)).unwrap(), 10);
        assert_eq!(dev.seek(SeekFrom::Current(5)).unwrap(), 15);
        assert_eq!(dev.seek(SeekFrom::End(-1)).unwrap(), 99);
        assert!(dev.seek(SeekFrom::End(-101)).is_err());
    }
}
