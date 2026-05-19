//! Filesystem layer — the [`Filesystem`] trait and shared types.
//!
//! Each filesystem implementation lives in its own submodule and implements
//! [`Filesystem`]. v1 ships [`ext::Ext`] (covers ext2 in its initial form;
//! ext3/ext4 follow in P4).
//!
//! ## Streaming invariant
//!
//! [`FileSource::HostPath`] is the canonical way to push large files into a
//! filesystem: implementations open the path, seek + read in a fixed buffer
//! (default 64 KiB), and write blocks directly to the underlying device. A
//! multi-gigabyte file MUST NOT be loaded into memory.
//!
//! [`FileSource::Reader`] is the generic streaming variant for anything that
//! can produce its bytes through [`std::io::Read`] + [`std::io::Seek`] with a
//! known total length.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub mod ext;
pub mod fat;
pub mod rootdevs;

pub use rootdevs::{DeviceEntry, RootDevs};

/// Permissions + ownership + timestamps for a new filesystem entry. All
/// fields have sensible defaults via [`Default`] so callers only need to set
/// what they care about.
#[derive(Debug, Clone, Copy)]
pub struct FileMeta {
    /// POSIX permission bits (excludes file-type bits — those come from the
    /// `create_*` method called).
    pub mode: u16,
    /// Owning user id.
    pub uid: u32,
    /// Owning group id.
    pub gid: u32,
    /// Modification time (seconds since the Unix epoch).
    pub mtime: u32,
    /// Access time. Defaults to `mtime`.
    pub atime: u32,
    /// Creation/change time. Defaults to `mtime`.
    pub ctime: u32,
}

impl Default for FileMeta {
    fn default() -> Self {
        Self {
            mode: 0o644,
            uid: 0,
            gid: 0,
            mtime: 0,
            atime: 0,
            ctime: 0,
        }
    }
}

impl FileMeta {
    /// Convenience: file metadata with the given mode and zero everything else.
    pub fn with_mode(mode: u16) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }
}

/// Source of a regular file's data. Built once per file by the caller; the
/// filesystem implementation drives the read.
pub enum FileSource {
    /// Stream from a path on the host filesystem. Length is taken from
    /// `metadata().len()` at the time the source is constructed.
    HostPath(PathBuf),
    /// Stream from an arbitrary seekable reader with a known length.
    Reader {
        /// The underlying reader.
        reader: Box<dyn ReadSeek + Send>,
        /// Total number of bytes the reader will produce.
        len: u64,
    },
    /// Zero-length placeholder of the given size. Useful for sparse files —
    /// the filesystem may either allocate zero blocks (true hole) or allocate
    /// data blocks and leave them zero, depending on its feature flags.
    Zero(u64),
}

impl FileSource {
    /// Length the source will produce.
    pub fn len(&self) -> io::Result<u64> {
        match self {
            FileSource::HostPath(p) => Ok(std::fs::metadata(p)?.len()),
            FileSource::Reader { len, .. } => Ok(*len),
            FileSource::Zero(n) => Ok(*n),
        }
    }

    /// Whether the source produces no bytes. `false` for `Zero(_)` because
    /// the filesystem still needs to record the size on the inode.
    pub fn is_empty(&self) -> io::Result<bool> {
        self.len().map(|n| n == 0)
    }

    /// Open the source for reading. Returns a boxed `Read` together with the
    /// total length; callers stream bytes through a fixed buffer rather than
    /// reading to end.
    pub fn open(self) -> io::Result<(Box<dyn ReadSeek + Send>, u64)> {
        match self {
            FileSource::HostPath(p) => {
                let f = File::open(&p)?;
                let len = f.metadata()?.len();
                Ok((Box::new(f), len))
            }
            FileSource::Reader { reader, len } => Ok((reader, len)),
            FileSource::Zero(n) => Ok((Box::new(ZeroReader { remaining: n }), n)),
        }
    }
}

/// Combined Read + Seek trait used by [`FileSource::Reader`].
pub trait ReadSeek: Read + Seek {}
impl<T: Read + Seek> ReadSeek for T {}

/// Special-file class for [`Filesystem::create_device`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// Character device (mode S_IFCHR).
    Char,
    /// Block device (mode S_IFBLK).
    Block,
    /// FIFO / named pipe (mode S_IFIFO).
    Fifo,
    /// Unix-domain socket (mode S_IFSOCK).
    Socket,
}

/// A single directory entry returned by [`Filesystem::list`].
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub inode: u32,
    pub kind: EntryKind,
}

/// File-type bucket exposed by [`DirEntry`]. Mirrors POSIX `d_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    Regular,
    Dir,
    Symlink,
    Char,
    Block,
    Fifo,
    Socket,
    Unknown,
}

/// Top-level API every filesystem implements.
///
/// The two-pass nature of writers is encoded externally: callers first
/// [`format`](Self::format) (which determines the FS geometry from the
/// device's size) and then call the `create_*` methods. Filesystems whose
/// geometry depends on the *total content size* (e.g. ext, which sizes its
/// inode table at mkfs time) provide a `format_for_plan` constructor in
/// their own module so the planner can pre-compute counts.
pub trait Filesystem: Sized {
    /// Format options understood by this filesystem. Each implementation
    /// exposes its own type (e.g. [`ext::FormatOpts`]).
    type FormatOpts;

    /// Format a fresh filesystem on `dev`. Overwrites whatever was there.
    fn format(
        dev: &mut dyn crate::block::BlockDevice,
        opts: &Self::FormatOpts,
    ) -> crate::Result<Self>;

    /// Open an existing filesystem from `dev`. Returns
    /// [`Error::InvalidImage`](crate::Error::InvalidImage) if the on-disk
    /// metadata is malformed.
    fn open(dev: &mut dyn crate::block::BlockDevice) -> crate::Result<Self>;

    /// Create a regular file at `path` populated from `src` with metadata `meta`.
    fn create_file(
        &mut self,
        dev: &mut dyn crate::block::BlockDevice,
        path: &Path,
        src: FileSource,
        meta: FileMeta,
    ) -> crate::Result<()>;

    /// Create a directory at `path`.
    fn create_dir(
        &mut self,
        dev: &mut dyn crate::block::BlockDevice,
        path: &Path,
        meta: FileMeta,
    ) -> crate::Result<()>;

    /// Create a symbolic link at `path` pointing at `target`.
    fn create_symlink(
        &mut self,
        dev: &mut dyn crate::block::BlockDevice,
        path: &Path,
        target: &Path,
        meta: FileMeta,
    ) -> crate::Result<()>;

    /// Create a device node / FIFO / socket.
    fn create_device(
        &mut self,
        dev: &mut dyn crate::block::BlockDevice,
        path: &Path,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: FileMeta,
    ) -> crate::Result<()>;

    /// Remove a file, directory, or special entry. Returns
    /// [`Error::InvalidArgument`](crate::Error::InvalidArgument) for a
    /// non-empty directory.
    fn remove(&mut self, dev: &mut dyn crate::block::BlockDevice, path: &Path)
    -> crate::Result<()>;

    /// List the entries of a directory.
    fn list(
        &mut self,
        dev: &mut dyn crate::block::BlockDevice,
        path: &Path,
    ) -> crate::Result<Vec<DirEntry>>;

    /// Open a regular file for reading. Returns a boxed streaming reader
    /// that borrows both `self` (for filesystem metadata) and `dev` (for
    /// actual block reads), so it must outlive both.
    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn crate::block::BlockDevice,
        path: &Path,
    ) -> crate::Result<Box<dyn Read + 'a>>;

    /// Persist outstanding dirty state to the device.
    fn flush(&mut self, dev: &mut dyn crate::block::BlockDevice) -> crate::Result<()>;
}

/// A `Read + Seek` that produces `remaining` zero bytes and then EOF.
/// Internal helper for [`FileSource::Zero`].
struct ZeroReader {
    remaining: u64,
}

impl Read for ZeroReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let n = (buf.len() as u64).min(self.remaining) as usize;
        buf[..n].fill(0);
        self.remaining -= n as u64;
        Ok(n)
    }
}

impl Seek for ZeroReader {
    fn seek(&mut self, _pos: SeekFrom) -> io::Result<u64> {
        // We don't currently need to support seek on Zero(); ext writer reads
        // straight through. Return an error so a misuse is loud.
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "ZeroReader does not support seeking",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_source_streams_in_chunks() {
        let src = FileSource::Zero(10_000);
        let (mut reader, len) = src.open().unwrap();
        assert_eq!(len, 10_000);
        let mut total = 0;
        let mut buf = [0u8; 4096];
        loop {
            let n = reader.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            assert!(buf[..n].iter().all(|&b| b == 0));
            total += n;
        }
        assert_eq!(total, 10_000);
    }

    #[test]
    fn host_path_source_length_matches_file() {
        use tempfile::NamedTempFile;
        let mut f = NamedTempFile::new().unwrap();
        std::io::Write::write_all(f.as_file_mut(), b"hello world").unwrap();
        let src = FileSource::HostPath(f.path().to_path_buf());
        assert_eq!(src.len().unwrap(), 11);
    }
}
