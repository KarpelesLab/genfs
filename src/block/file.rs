//! On-disk [`BlockDevice`] backed by a regular file.
//!
//! Newly created backends call `set_len` so the file appears at the right
//! capacity without touching any data blocks — on filesystems that support
//! sparse files (every modern Linux filesystem) the unwritten regions cost
//! zero on-disk bytes until written.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::BlockDevice;
use crate::Result;

/// Default advisory sector size for new file-backed images.
pub const DEFAULT_SECTOR: u32 = 512;

/// A [`BlockDevice`] backed by `std::fs::File` opened read/write.
#[derive(Debug)]
pub struct FileBackend {
    file: File,
    size: u64,
    block_size: u32,
}

impl FileBackend {
    /// Create a fresh image file of exactly `size` bytes, sparsely allocated.
    /// Truncates any existing file at the path.
    pub fn create<P: AsRef<Path>>(path: P, size: u64) -> Result<Self> {
        Self::create_with_block_size(path, size, DEFAULT_SECTOR)
    }

    /// Create with an explicit advisory sector size.
    pub fn create_with_block_size<P: AsRef<Path>>(
        path: P,
        size: u64,
        block_size: u32,
    ) -> Result<Self> {
        assert!(block_size.is_power_of_two(), "block_size must be a power of two");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.as_ref())?;
        file.set_len(size)?;
        Ok(Self {
            file,
            size,
            block_size,
        })
    }

    /// Open an existing image file. The file's current length is taken as the
    /// device capacity.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_block_size(path, DEFAULT_SECTOR)
    }

    /// Open with an explicit advisory sector size.
    pub fn open_with_block_size<P: AsRef<Path>>(path: P, block_size: u32) -> Result<Self> {
        assert!(block_size.is_power_of_two(), "block_size must be a power of two");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())?;
        let size = file.metadata()?.len();
        Ok(Self {
            file,
            size,
            block_size,
        })
    }
}

impl Read for FileBackend {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.file.read(buf)
    }
}

impl Write for FileBackend {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let pos = self.file.stream_position()?;
        let remaining = self.size.saturating_sub(pos);
        if remaining == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "write past end of FileBackend",
            ));
        }
        let n = remaining.min(buf.len() as u64) as usize;
        self.file.write(&buf[..n])
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

impl Seek for FileBackend {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.file.seek(pos)
    }
}

impl BlockDevice for FileBackend {
    fn block_size(&self) -> u32 {
        self.block_size
    }

    fn total_size(&self) -> u64 {
        self.size
    }

    fn zero_range(&mut self, offset: u64, len: u64) -> Result<()> {
        let size = self.total_size();
        if offset.checked_add(len).is_none_or(|e| e > size) {
            return Err(crate::Error::OutOfBounds { offset, len, size });
        }
        if len == 0 {
            return Ok(());
        }
        // On Linux we could use fallocate(FALLOC_FL_PUNCH_HOLE) for a true
        // sparse hole; for portability v1 just writes zeros. A future
        // optimisation can detect Linux and punch instead. The result is
        // semantically identical: bytes read as zero.
        self.seek(SeekFrom::Start(offset))?;
        let zero = [0u8; 4096];
        let mut remaining = len;
        while remaining > 0 {
            let n = remaining.min(zero.len() as u64) as usize;
            self.write_all(&zero[..n])?;
            remaining -= n as u64;
        }
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.file.sync_data()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn temp_path() -> NamedTempFile {
        NamedTempFile::new().expect("tempfile")
    }

    #[test]
    fn create_sets_length() {
        let tmp = temp_path();
        let dev = FileBackend::create(tmp.path(), 1024).unwrap();
        assert_eq!(dev.total_size(), 1024);
        assert_eq!(std::fs::metadata(tmp.path()).unwrap().len(), 1024);
    }

    #[test]
    fn create_reads_back_as_zero() {
        let tmp = temp_path();
        let mut dev = FileBackend::create(tmp.path(), 4096).unwrap();
        let mut buf = [0xffu8; 64];
        dev.read_at(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn write_then_read_roundtrip() {
        let tmp = temp_path();
        let mut dev = FileBackend::create(tmp.path(), 8192).unwrap();
        let payload: Vec<u8> = (0..512u16).map(|i| (i & 0xff) as u8).collect();
        dev.write_at(1024, &payload).unwrap();
        let mut got = vec![0u8; 512];
        dev.read_at(1024, &mut got).unwrap();
        assert_eq!(payload, got);
    }

    #[test]
    fn write_at_past_end_rejected() {
        let tmp = temp_path();
        let mut dev = FileBackend::create(tmp.path(), 128).unwrap();
        let err = dev.write_at(100, &[0u8; 64]).unwrap_err();
        assert!(matches!(err, crate::Error::OutOfBounds { .. }));
    }

    #[test]
    fn reopen_preserves_size_and_content() {
        let tmp = temp_path();
        {
            let mut dev = FileBackend::create(tmp.path(), 4096).unwrap();
            dev.write_at(2000, b"hello, genfs").unwrap();
            dev.sync().unwrap();
        }
        let mut dev = FileBackend::open(tmp.path()).unwrap();
        assert_eq!(dev.total_size(), 4096);
        let mut buf = [0u8; 12];
        dev.read_at(2000, &mut buf).unwrap();
        assert_eq!(&buf, b"hello, genfs");
    }
}
