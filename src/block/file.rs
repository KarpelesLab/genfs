//! On-disk [`BlockDevice`] backed by a regular file *or* a real block
//! device (e.g. `/dev/sdb`, `/dev/nvme0n1`).
//!
//! For regular files: newly created backends call `set_len` so the file
//! appears at the right capacity without touching any data blocks — on
//! filesystems that support sparse files (every modern Linux filesystem)
//! the unwritten regions cost zero on-disk bytes until written.
//!
//! For block devices: `set_len` is skipped (you can't truncate a block
//! device), the capacity is queried via the kernel ioctl
//! (`BLKGETSIZE64` on Linux, `DKIOCGETBLOCKCOUNT`×`DKIOCGETBLOCKSIZE` on
//! macOS), and the file is opened with `O_EXCL` so the kernel refuses
//! the open if any of the device's partitions is mounted.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::BlockDevice;
use crate::Result;

/// True when `path` refers to a block device. False everywhere on Windows
/// (block-device paths there look like `\\.\PhysicalDriveN` and need a
/// completely different code path that v1 doesn't ship).
pub fn is_block_device(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileTypeExt;
        std::fs::metadata(path)
            .map(|m| m.file_type().is_block_device())
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        false
    }
}

/// Total byte size of an opened block device. Errors on platforms without
/// an ioctl implementation (currently anything other than Linux/macOS).
#[cfg(unix)]
fn block_device_size(file: &File) -> io::Result<u64> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    #[cfg(target_os = "linux")]
    {
        // `BLKGETSIZE64`: kernel writes the device size in bytes into a u64
        // out-parameter. The constant is the same across Linux archs
        // (0x80081272) — the macro uses sizeof(size_t) = 8 by convention
        // here, regardless of the running arch's actual size_t width.
        const BLKGETSIZE64: libc::c_ulong = 0x8008_1272;
        let mut size: u64 = 0;
        let r = unsafe { libc::ioctl(fd, BLKGETSIZE64, &mut size as *mut u64) };
        if r < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(size)
    }
    #[cfg(target_os = "macos")]
    {
        // macOS doesn't expose a single "byte size" ioctl — we multiply
        // block count by block size. Constants from <sys/disk.h>.
        const DKIOCGETBLOCKCOUNT: libc::c_ulong = 0x4008_6419;
        const DKIOCGETBLOCKSIZE: libc::c_ulong = 0x4004_6418;
        let mut count: u64 = 0;
        let mut bs: u32 = 0;
        let r1 = unsafe { libc::ioctl(fd, DKIOCGETBLOCKCOUNT, &mut count) };
        if r1 < 0 {
            return Err(io::Error::last_os_error());
        }
        let r2 = unsafe { libc::ioctl(fd, DKIOCGETBLOCKSIZE, &mut bs) };
        if r2 < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(count.saturating_mul(bs as u64))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = fd;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "fstool: block-device size query not implemented on this Unix",
        ))
    }
}

#[cfg(not(unix))]
fn block_device_size(_file: &File) -> io::Result<u64> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "fstool: block devices are only supported on Unix in v1",
    ))
}

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
    ///
    /// If the path resolves to a block device (`/dev/sdX`, etc.), the
    /// `size` argument is treated as a *minimum* — the actual capacity is
    /// read from the kernel and used as-is, the file is opened without
    /// truncating, and `O_EXCL` makes the open fail if any partition is
    /// mounted. The device must be at least `size` bytes.
    pub fn create<P: AsRef<Path>>(path: P, size: u64) -> Result<Self> {
        Self::create_with_block_size(path, size, DEFAULT_SECTOR)
    }

    /// Create with an explicit advisory sector size.
    pub fn create_with_block_size<P: AsRef<Path>>(
        path: P,
        size: u64,
        block_size: u32,
    ) -> Result<Self> {
        assert!(
            block_size.is_power_of_two(),
            "block_size must be a power of two"
        );
        let p = path.as_ref();
        if p.exists() && is_block_device(p) {
            let file = open_existing_for_write(p, /* exclusive = */ true)?;
            let actual = block_device_size(&file).map_err(crate::Error::from)?;
            if actual < size {
                return Err(crate::Error::InvalidArgument(format!(
                    "fstool: block device {} is {} bytes, need at least {}",
                    p.display(),
                    actual,
                    size
                )));
            }
            return Ok(Self {
                file,
                size: actual,
                block_size,
            });
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(p)?;
        file.set_len(size)?;
        Ok(Self {
            file,
            size,
            block_size,
        })
    }

    /// Open an existing image file or block device. For regular files the
    /// capacity is the file's current length; for block devices it's the
    /// device's real size queried from the kernel.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_block_size(path, DEFAULT_SECTOR)
    }

    /// Open with an explicit advisory sector size.
    pub fn open_with_block_size<P: AsRef<Path>>(path: P, block_size: u32) -> Result<Self> {
        assert!(
            block_size.is_power_of_two(),
            "block_size must be a power of two"
        );
        let p = path.as_ref();
        let is_block = is_block_device(p);
        let file = open_existing_for_write(p, /* exclusive = */ is_block)?;
        let size = if is_block {
            block_device_size(&file).map_err(crate::Error::from)?
        } else {
            file.metadata()?.len()
        };
        Ok(Self {
            file,
            size,
            block_size,
        })
    }
}

/// Open an existing path read+write, optionally with `O_EXCL` (used for
/// block devices so the kernel refuses an open while any partition is
/// mounted). `O_EXCL` on a regular file would prevent re-opening, so it's
/// only set for block devices.
fn open_existing_for_write(path: &Path, exclusive: bool) -> io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.read(true).write(true);
    #[cfg(unix)]
    if exclusive {
        use std::os::unix::fs::OpenOptionsExt;
        opts.custom_flags(libc::O_EXCL);
    }
    #[cfg(not(unix))]
    {
        let _ = exclusive;
    }
    opts.open(path)
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
            dev.write_at(2000, b"hello, fstool").unwrap();
            dev.sync().unwrap();
        }
        let mut dev = FileBackend::open(tmp.path()).unwrap();
        assert_eq!(dev.total_size(), 4096);
        let mut buf = [0u8; 13];
        dev.read_at(2000, &mut buf).unwrap();
        assert_eq!(&buf, b"hello, fstool");
    }

    #[cfg(unix)]
    #[test]
    fn is_block_device_discriminates() {
        use std::path::Path;
        // A tempfile is a regular file, not a block device.
        let tmp = temp_path();
        assert!(!is_block_device(tmp.path()));
        // /dev/null is a CHARACTER device; the predicate must distinguish.
        let null = Path::new("/dev/null");
        if null.exists() {
            assert!(!is_block_device(null));
        }
        // A non-existent path must not panic and must report false.
        assert!(!is_block_device(Path::new(
            "/nonexistent/fstool-blkdev-probe"
        )));
    }
}
