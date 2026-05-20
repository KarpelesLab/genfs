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
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub mod apfs;
pub mod exfat;
pub mod ext;
pub mod f2fs;
pub mod fat;
pub mod grf;
pub mod hfs_plus;
pub mod iso9660;
pub mod ntfs;
pub mod rootdevs;
pub mod squashfs;
pub mod tar;
pub mod xfs;

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

/// Flags controlling how [`Filesystem::open_file_rw`] opens a file.
///
/// Defaults to "open existing file for read+write at offset 0." Each
/// flag toggles a single Unix-style behaviour. `truncate` and
/// `append` are mutually meaningful at open-time only — once the
/// handle exists, the user can `seek` to any position freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenFlags {
    /// Create the file if it does not already exist. Requires the
    /// `meta` argument to [`Filesystem::open_file_rw`] to be `Some(_)`
    /// (otherwise the implementation returns `InvalidArgument`).
    pub create: bool,
    /// Truncate to zero length on open.
    pub truncate: bool,
    /// Position the initial cursor at end-of-file (so the first
    /// `write` appends). Equivalent to seeking to `len()` after open.
    pub append: bool,
}

/// A handle into a regular file opened for in-place reads and writes
/// via [`Filesystem::open_file_rw`]. Implementations are `Read + Write
/// + Seek`; dropping the handle persists any pending writes (each
/// implementation chooses whether `Write::write` is eager or buffered
/// — `sync` forces a flush either way).
///
/// The handle borrows both the filesystem state and the block device
/// for its full lifetime. A subsequent call to
/// [`Filesystem::flush`] persists handle-side changes that haven't
/// already been written, so consumers don't need to call `sync`
/// explicitly when they're going to `flush` the whole FS anyway.
pub trait FileHandle: Read + Write + Seek {
    /// Logical length of the file (after any pending writes).
    fn len(&self) -> u64;

    /// Whether the file is currently zero bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resize the file to `new_len`. Growing fills with zeroes;
    /// shrinking discards trailing bytes (and frees underlying
    /// blocks). May return an error if the filesystem can't allocate
    /// enough space.
    fn set_len(&mut self, new_len: u64) -> crate::Result<()>;

    /// Persist this handle's writes to disk. Implementations should
    /// also push any associated metadata changes (size, mtime, block
    /// pointers). After `sync` returns Ok, the new bytes are durable
    /// without needing a separate [`Filesystem::flush`] for the
    /// file itself — though the FS as a whole may still need a
    /// flush for unrelated dirty state.
    fn sync(&mut self) -> crate::Result<()>;
}

/// A handle into a regular file opened **read-only** via
/// [`Filesystem::open_file_ro`]. Implementations are `Read + Seek`
/// with a known total `len`; no writes, no allocation, no journal
/// interaction. Every backend (including the immutable ones —
/// ISO 9660 / SquashFS / tar — and the streaming ones if the format
/// permits backward seeks) can implement this.
///
/// The handle borrows both the filesystem state and the block device
/// for its full lifetime, mirroring [`FileHandle`].
pub trait FileReadHandle: Read + Seek {
    /// Total length of the file in bytes.
    fn len(&self) -> u64;

    /// Whether the file is zero bytes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

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
    /// Size in bytes when this is a regular file. `0` for directories,
    /// symlinks, devices, etc. Filesystems that can't surface size
    /// cheaply during a listing may also return `0` — callers that
    /// need an authoritative figure should open the file and seek.
    pub size: u64,
}

/// How — and whether — a filesystem can be mutated after `flush()`.
/// Returned by [`Filesystem::mutation_capability`].
///
/// The variants are ordered from most-capable to least-capable; each
/// implies the abilities of the ones below it (`Mutable` can do what
/// `WholeFileOnly` can, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationCapability {
    /// Full in-place edits: open an existing image, `create_*` /
    /// `remove` files, and (when a partial-write API exists) patch
    /// arbitrary byte ranges within an existing file. ext, FAT32,
    /// F2FS today.
    Mutable,
    /// `create_file` (whole-file replacement) and `remove` work, but
    /// partial in-place writes to an existing file's contents do
    /// not. Existing files can be removed and rewritten from scratch;
    /// they cannot be patched in place. No backend reports this
    /// today — reserved for future formats (write-once archives with
    /// free-list reclamation, content-addressed stores, append-only
    /// records) where add/remove is structurally possible but
    /// offset-write isn't.
    WholeFileOnly,
    /// Writer is sequential / streaming — once bytes are emitted you
    /// can't seek backward to patch. The only way to "modify" is to
    /// produce a new image from scratch. Tar today.
    Streaming,
    /// On-disk layout is laid down at format time and the format has
    /// no in-place mutation hooks (no free-list, no journal). The
    /// writer can seek, but the image isn't re-openable as writable.
    /// ISO 9660 and SquashFS today.
    Immutable,
}

impl MutationCapability {
    /// Whether the filesystem can satisfy a `create_file` / `remove`
    /// request — i.e. add or delete a whole file. True for `Mutable`
    /// and `WholeFileOnly`; false for `Streaming` and `Immutable`.
    pub fn supports_add_remove(self) -> bool {
        matches!(self, Self::Mutable | Self::WholeFileOnly)
    }

    /// Whether the filesystem can patch bytes inside an existing
    /// file without removing-and-recreating it. True only for
    /// `Mutable`. Future partial-write APIs gate on this.
    pub fn supports_partial_writes(self) -> bool {
        matches!(self, Self::Mutable)
    }
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

/// Top-level dyn-compatible API every filesystem implements. The
/// `format` / `open` factory methods live on the sibling
/// [`FilesystemFactory`] trait so this one stays object-safe — the
/// generic walker in [`crate::repack`] can hold a `&mut dyn
/// Filesystem` and drive any of `Ext` / `Fat32` / `HfsPlus` / `Ntfs`
/// / `F2fs` / `Squashfs` / `Xfs` through the same `create_*` /
/// `remove` / `list` / `read_file` / `flush` entry points.
pub trait Filesystem {
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

    /// Open a regular file for **random-access reads** with no
    /// writes. The returned handle is `Read + Seek` and reports the
    /// file's total length via `len()`. Every backend that can
    /// surface file contents should implement this — including the
    /// immutable formats (ISO 9660 / SquashFS / tar / GRF) where
    /// `open_file_rw` is unsupported but seeking inside a file is
    /// still meaningful.
    ///
    /// Default: returns `Unsupported`. Implementations override —
    /// most can do so by reusing the same extent / runlist walker
    /// that powers [`Self::read_file`].
    fn open_file_ro<'a>(
        &'a mut self,
        _dev: &'a mut dyn crate::block::BlockDevice,
        _path: &Path,
    ) -> crate::Result<Box<dyn FileReadHandle + 'a>> {
        Err(crate::Error::Unsupported(
            "this filesystem does not yet implement open_file_ro".into(),
        ))
    }

    /// Open a regular file for **in-place reads + writes** at byte
    /// granularity. The returned handle is `Read + Write + Seek`;
    /// dropping it persists any pending bytes (each implementation
    /// chooses whether `Write::write` is eager or buffered).
    ///
    /// Filesystems whose on-disk format requires journaling to be
    /// safe across crash boundaries should refuse this method until
    /// their journal is wired — partial writes that bypass a journal
    /// leave the FS in a "needs `fsck`" state on next mount, which
    /// is a worse default than a clear `Unsupported` error.
    ///
    /// Default: returns `Unsupported`. Implementations override only
    /// when they can produce a result that survives a clean unmount
    /// without external repair.
    fn open_file_rw<'a>(
        &'a mut self,
        _dev: &'a mut dyn crate::block::BlockDevice,
        _path: &Path,
        _flags: OpenFlags,
        _meta: Option<FileMeta>,
    ) -> crate::Result<Box<dyn FileHandle + 'a>> {
        Err(crate::Error::Unsupported(
            "this filesystem does not yet implement open_file_rw".into(),
        ))
    }

    /// Persist outstanding dirty state to the device.
    fn flush(&mut self, dev: &mut dyn crate::block::BlockDevice) -> crate::Result<()>;

    /// Capability of this filesystem with respect to mutating an
    /// already-flushed image. Three cases:
    ///
    /// - [`MutationCapability::Mutable`]: full in-place edits via
    ///   `create_*` / `remove` (ext, FAT32, F2FS).
    /// - [`MutationCapability::Streaming`]: writer is sequential
    ///   only — adding to an existing image means producing a new
    ///   one from scratch (tar).
    /// - [`MutationCapability::Immutable`]: writer can seek but the
    ///   on-disk format has no in-place mutation hooks (ISO 9660,
    ///   SquashFS). `repack` rebuilds.
    ///
    /// Default: `Mutable`. Override on backends that aren't.
    fn mutation_capability(&self) -> MutationCapability {
        MutationCapability::Mutable
    }

    /// Convenience shortcut: can this filesystem satisfy
    /// `create_file` / `remove`? Equivalent to
    /// `mutation_capability().supports_add_remove()`. Returns true
    /// for both [`MutationCapability::Mutable`] and
    /// [`MutationCapability::WholeFileOnly`] — callers that need
    /// finer detail (e.g. "can I patch byte N?") should query
    /// [`Self::mutation_capability`] directly.
    fn supports_mutation(&self) -> bool {
        self.mutation_capability().supports_add_remove()
    }

    /// Read a symbolic link's target. Default returns `Unsupported`
    /// — filesystems that have symlinks (ext, tar, xfs, hfs+, ntfs,
    /// squashfs, iso 9660 via Rock Ridge) override.
    fn read_symlink(
        &mut self,
        _dev: &mut dyn crate::block::BlockDevice,
        _path: &Path,
    ) -> crate::Result<std::path::PathBuf> {
        Err(crate::Error::Unsupported(
            "this filesystem does not implement read_symlink".into(),
        ))
    }

    /// Recursive sum of all regular-file sizes in the filesystem.
    /// Uses the `size` field on [`DirEntry`] returned by [`Self::list`]
    /// — filesystems that don't surface size from a listing return 0
    /// for those entries, in which case the total is best-effort.
    ///
    /// Skips the special names `"."`, `".."`, and `"lost+found"` so
    /// the walk doesn't loop / double-count ext's reserved tree.
    fn total_file_bytes(&mut self, dev: &mut dyn crate::block::BlockDevice) -> crate::Result<u64> {
        let mut total = 0u64;
        let mut stack: Vec<std::path::PathBuf> = vec![std::path::PathBuf::from("/")];
        while let Some(dir) = stack.pop() {
            let entries = self.list(dev, &dir)?;
            for e in entries {
                if e.name == "." || e.name == ".." || e.name == "lost+found" {
                    continue;
                }
                let child = dir.join(&e.name);
                match e.kind {
                    EntryKind::Regular => total = total.saturating_add(e.size),
                    EntryKind::Dir => stack.push(child),
                    _ => {}
                }
            }
        }
        Ok(total)
    }
}

/// Companion trait for filesystems that can be created from scratch
/// (`format`) or opened from an existing image (`open`). Kept separate
/// from [`Filesystem`] so the latter remains object-safe.
pub trait FilesystemFactory: Filesystem + Sized {
    /// Format options understood by this filesystem. Each implementation
    /// exposes its own type (e.g. [`ext::FormatOpts`]).
    type FormatOpts;

    /// Format a fresh filesystem on `dev`. Overwrites whatever was there.
    fn format(
        dev: &mut dyn crate::block::BlockDevice,
        opts: &Self::FormatOpts,
    ) -> crate::Result<Self>;

    /// Open an existing filesystem from `dev`.
    fn open(dev: &mut dyn crate::block::BlockDevice) -> crate::Result<Self>;
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
