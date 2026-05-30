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
pub mod archive;
pub(crate) mod dir_batch;
pub mod exfat;
pub mod ext;
pub mod f2fs;
pub mod fat;
pub mod grf;
pub mod hfs;
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
    /// Stream from an owned temporary file. The handle lives inside the
    /// source, so deferred-write backends (SquashFS / ISO 9660 / GRF,
    /// which keep the `FileSource` and read it at `flush`) keep the
    /// backing bytes until they're consumed — then the temp file is
    /// deleted on drop. Used by the default
    /// [`Filesystem::create_file_streaming`] to bridge a borrowed reader
    /// into the `FileSource` API without buffering the whole file in RAM.
    TempFile(tempfile::NamedTempFile),
}

impl FileSource {
    /// Length the source will produce.
    pub fn len(&self) -> io::Result<u64> {
        match self {
            FileSource::HostPath(p) => Ok(std::fs::metadata(p)?.len()),
            FileSource::Reader { len, .. } => Ok(*len),
            FileSource::Zero(n) => Ok(*n),
            FileSource::TempFile(t) => Ok(t.as_file().metadata()?.len()),
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
            FileSource::TempFile(t) => {
                // Re-open the temp file by path for an independent cursor;
                // the `NamedTempFile` is dropped here, but on a deferred
                // backend the source was opened at flush time so the file
                // still exists. (The backend owns the source until then.)
                let len = t.as_file().metadata()?.len();
                let f = File::open(t.path())?;
                Ok((Box::new(f), len))
            }
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

/// Full attributes for a single path, returned by [`Filesystem::getattr`].
///
/// Mirrors what FUSE's `getattr`/`lookup` callbacks need to populate
/// `FileAttr`. The default impl on [`Filesystem`] synthesises this from
/// [`Filesystem::list`] of the parent — that delivers `kind`, `size`,
/// and an inode number, with the rest defaulted (mode `0o644`/`0o755`,
/// uid/gid 0, all times 0). Backends that carry per-file metadata
/// (ext, ntfs, hfs+, …) should override `getattr` to surface the real
/// values.
#[derive(Debug, Clone, Copy)]
pub struct FileAttrs {
    pub kind: EntryKind,
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    /// Block count in 512-byte units (POSIX `st_blocks`).
    pub blocks: u64,
    pub nlink: u32,
    pub atime: u32,
    pub mtime: u32,
    pub ctime: u32,
    /// Device number for char/block special files; `0` otherwise.
    pub rdev: u32,
    /// Inode number this path resolves to. `0` when the backend has
    /// no per-path identity (the FUSE adapter assigns its own ids in
    /// that case).
    pub inode: u32,
}

impl FileAttrs {
    /// Permissive defaults for backends that can't surface real values
    /// — used by the trait-level fallback `getattr`. `mode` is set by
    /// the caller based on `kind` (0o755 for dirs, 0o644 for files).
    fn defaults_for(kind: EntryKind, size: u64, inode: u32) -> Self {
        let mode = match kind {
            EntryKind::Dir => 0o755,
            EntryKind::Symlink => 0o777,
            _ => 0o644,
        };
        let nlink = match kind {
            EntryKind::Dir => 2,
            _ => 1,
        };
        Self {
            kind,
            mode,
            uid: 0,
            gid: 0,
            size,
            blocks: size.div_ceil(512),
            nlink,
            atime: 0,
            mtime: 0,
            ctime: 0,
            rdev: 0,
            inode,
        }
    }
}

/// Mutation request for [`Filesystem::set_attrs`]. Every field is
/// optional — `None` means "leave as-is." Mirrors the shape of FUSE's
/// `setattr` so the adapter can pass a single packed struct in.
#[derive(Debug, Default, Clone, Copy)]
pub struct SetAttrs {
    pub mode: Option<u16>,
    pub uid: Option<u32>,
    pub gid: Option<u32>,
    pub atime: Option<u32>,
    pub mtime: Option<u32>,
    pub ctime: Option<u32>,
}

/// Filesystem-level capacity stats returned by [`Filesystem::statfs`].
/// All `u64` so backends with huge counts don't overflow. `name_max`
/// is the longest filename the FS will accept.
#[derive(Debug, Clone, Copy)]
pub struct StatFs {
    pub block_size: u32,
    pub blocks: u64,
    pub blocks_free: u64,
    pub blocks_avail: u64,
    pub inodes: u64,
    pub inodes_free: u64,
    pub name_max: u32,
}

impl Default for StatFs {
    fn default() -> Self {
        // 4 KiB block, no quota, generous name budget — the same
        // numbers the kernel hands out for tmpfs in a fresh mount.
        Self {
            block_size: 4096,
            blocks: 0,
            blocks_free: 0,
            blocks_avail: 0,
            inodes: 0,
            inodes_free: 0,
            name_max: 255,
        }
    }
}

/// A single extended attribute, returned by [`Filesystem::list_xattrs`].
#[derive(Debug, Clone)]
pub struct XattrPair {
    pub name: String,
    pub value: Vec<u8>,
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
    /// exFAT, XFS, HFS+, NTFS today — and a freshly-`format`ted F2FS
    /// handle (a *reopened* F2FS handle is `Immutable`; see below).
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
    /// ISO 9660 and SquashFS today, plus a *reopened* F2FS handle
    /// (build-once: its writer serializes the whole FS from in-memory
    /// state at flush and isn't reconstructed on `open`).
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

/// What kind of zero-copy extent sharing — *reflinks* — the backend
/// can express. Returned by [`Filesystem::clone_capability`].
///
/// Reflinks let a destination file (or a range of it) point at the
/// same physical extents as a source, with copy-on-write semantics on
/// the next write to either side. The on-disk encoding differs per
/// backend (XFS refcount-btree, APFS clone records, Btrfs shared
/// extents), so the trait surface offers two operations and a
/// capability gate, with sensible byte-copy fallbacks for backends
/// that can't share extents natively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloneCapability {
    /// No extent sharing. `clone_file` falls back to a stream-copy
    /// (semantically equivalent, just not zero-copy); `clone_range`
    /// returns [`crate::Error::Unsupported`].
    None,
    /// Whole-file clone only — the backend can share *every* extent
    /// of the source into the destination as one atomic operation
    /// (e.g. APFS file-clone records). Sub-file ranges aren't
    /// individually shareable, so `clone_range` still errors
    /// `Unsupported` unless the range exactly covers the whole file.
    WholeFile,
    /// Arbitrary range clone — `clone_range` works for any allocation-
    /// unit-aligned `(src_off, dst_off, len)` triple. XFS reflink,
    /// Btrfs `BTRFS_IOC_CLONE_RANGE`, and `FICLONERANGE` in general.
    Range,
}

impl CloneCapability {
    /// True for [`CloneCapability::WholeFile`] or [`CloneCapability::Range`].
    /// `clone_file` is guaranteed not to byte-copy in this case.
    pub fn shares_extents(self) -> bool {
        !matches!(self, Self::None)
    }

    /// True only for [`CloneCapability::Range`] — i.e. sub-file
    /// `clone_range` calls will succeed (assuming alignment etc.).
    pub fn supports_range(self) -> bool {
        matches!(self, Self::Range)
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

    /// Create a regular file at `path` streaming exactly `len` bytes from
    /// `body`. Unlike [`create_file`](Self::create_file)'s
    /// [`FileSource::Reader`] (which needs an owned `ReadSeek + Send`),
    /// `body` is a plain borrowed [`Read`] — so a body borrowed from
    /// another open filesystem can be piped straight through without an
    /// intermediate tempfile or a `Seek`/`Send` bound. Implementations
    /// MUST NOT retain `body` past the call.
    ///
    /// Default: spool `body` into a tempfile and delegate to
    /// `create_file(FileSource::HostPath(..))` — correct everywhere, so
    /// no backend regresses. Backends whose writer already consumes a
    /// `&mut dyn Read` (ext, FAT32, the archive core, …) override this
    /// for true zero-copy streaming.
    /// Whether [`create_file`](Self::create_file) consumes its
    /// `FileSource` synchronously (`true`) rather than storing it to read
    /// later at `flush` (`false` — e.g. SquashFS / ISO 9660 / GRF, which
    /// keep every source until they serialise). Immediate backends let
    /// [`create_file_streaming`](Self::create_file_streaming) buffer small
    /// files in memory instead of spilling each one to a temp file.
    fn streams_immediately(&self) -> bool {
        false
    }

    fn create_file_streaming(
        &mut self,
        dev: &mut dyn crate::block::BlockDevice,
        path: &Path,
        body: &mut dyn Read,
        len: u64,
        meta: FileMeta,
    ) -> crate::Result<()> {
        // For backends that consume the source now, buffer small files in
        // memory: this avoids creating, copying into, fsync-ing and
        // reading back a temp file *per file* — the dominant per-file cost
        // when repacking many small files. Larger files (and deferred
        // backends, which must keep the bytes until `flush`) spill to a
        // temp file — but we never fsync it: it's read back in this same
        // process from the page cache, so durability is irrelevant.
        const MEM_CAP: u64 = 8 * 1024 * 1024;
        if self.streams_immediately() && len <= MEM_CAP {
            let mut buf = Vec::with_capacity(len as usize);
            body.take(len).read_to_end(&mut buf)?;
            let actual = buf.len() as u64;
            return self.create_file(
                dev,
                path,
                FileSource::Reader {
                    reader: Box::new(io::Cursor::new(buf)),
                    len: actual,
                },
                meta,
            );
        }
        let mut tmp = tempfile::NamedTempFile::new()?;
        let mut limited = body.take(len);
        io::copy(&mut limited, tmp.as_file_mut())?;
        // Deferred backends (SquashFS / ISO 9660 / GRF) store the
        // `FileSource` and read it at `flush`; `FileSource::TempFile`
        // keeps the bytes alive until then.
        self.create_file(dev, path, FileSource::TempFile(tmp), meta)
    }

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

    /// For archive / streaming writers backed by a pre-sized device:
    /// the exact number of bytes the output occupies after [`flush`].
    /// The backing file is provisioned generously (and sparsely), so
    /// the caller truncates it to this length — an archive must be
    /// exactly its own size, not padded with a zero tail.
    ///
    /// Default `None`: filesystem images keep their provisioned size.
    /// Only the archive backends (zip/cpio/ar) override this.
    ///
    /// [`flush`]: Self::flush
    fn image_len(&self) -> Option<u64> {
        None
    }

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

    /// Reflink / clone capability of this filesystem — does it natively
    /// share extents, and at what granularity?
    ///
    /// Default: [`CloneCapability::None`] — `clone_file` will byte-copy
    /// and `clone_range` will return `Unsupported`. Reflink-capable
    /// backends (XFS once the REFLINK feature is on, APFS clones,
    /// Btrfs) override to surface the right variant.
    fn clone_capability(&self) -> CloneCapability {
        CloneCapability::None
    }

    /// Clone the file at `src` into a new file at `dst`. Reflink-capable
    /// backends share extents (zero data copy, refcount-btree updates);
    /// everything else falls back to the default byte-copy.
    ///
    /// **Default behaviour.** Spools `src` to a host tempfile so the
    /// read-borrow on `self` is dropped, then runs `create_file(dst,
    /// FileSource::TempFile, ...)`. Best-effort metadata via
    /// [`Self::getattr`] (mode / uid / gid / mtime); falls back to
    /// [`FileMeta::default`] when the source backend has no `getattr`
    /// implementation.
    ///
    /// **Contracts.**
    /// * `src` must exist; `dst` must not already exist.
    /// * `dst`'s parent directory must exist.
    /// * The result is observable through `read_file` / `getattr`
    ///   regardless of whether extents were shared — callers needn't
    ///   inspect `clone_capability` to use this method.
    ///
    /// Returns `Err(Unsupported)` only when neither sharing nor the
    /// default fallback can satisfy the call (e.g. an immutable
    /// backend can't create files at all).
    fn clone_file(
        &mut self,
        dev: &mut dyn crate::block::BlockDevice,
        src: &Path,
        dst: &Path,
    ) -> crate::Result<()> {
        // Best-effort metadata snapshot before the read borrow.
        let (meta, size) = match self.getattr(dev, src) {
            Ok(a) => (
                FileMeta {
                    mode: a.mode,
                    uid: a.uid,
                    gid: a.gid,
                    mtime: a.mtime,
                    atime: a.atime,
                    ctime: a.ctime,
                },
                a.size,
            ),
            // Unknown size → take the conservative temp-file path below.
            Err(_) => (FileMeta::default(), u64::MAX),
        };
        // Buffer the source so the read borrow ends before `create_file`.
        // Small files go through memory (no temp file); larger ones spool
        // to a temp file — never fsync'd, as it's read back in-process.
        const MEM_CAP: u64 = 8 * 1024 * 1024;
        if size <= MEM_CAP {
            let mut buf = Vec::with_capacity(size as usize);
            {
                let mut reader = self.read_file(dev, src)?;
                reader.read_to_end(&mut buf).map_err(crate::Error::from)?;
            }
            let actual = buf.len() as u64;
            return self.create_file(
                dev,
                dst,
                FileSource::Reader {
                    reader: Box::new(io::Cursor::new(buf)),
                    len: actual,
                },
                meta,
            );
        }
        let mut tmp = tempfile::NamedTempFile::new().map_err(crate::Error::from)?;
        {
            let mut reader = self.read_file(dev, src)?;
            io::copy(&mut reader, &mut tmp).map_err(crate::Error::from)?;
        }
        self.create_file(dev, dst, FileSource::TempFile(tmp), meta)
    }

    /// Clone an arbitrary byte range `src[src_off..src_off+len]` into
    /// `dst[dst_off..dst_off+len]`. Reflink-capable backends share the
    /// underlying extents (`BTRFS_IOC_CLONE_RANGE` / `FICLONERANGE`
    /// semantics: writes through either side trigger COW).
    ///
    /// **Default:** [`crate::Error::Unsupported`]. Sub-file extent
    /// sharing is fundamentally a refcount-btree operation; backends
    /// without that machinery cannot satisfy it (a byte-copy would
    /// have different semantics — writes through `src` *after* the
    /// "clone" would NOT propagate, defeating the point).
    ///
    /// **Contracts** (when supported):
    /// * `src` and `dst` must both exist (`dst` may equal `src`).
    /// * `src_off + len` must not exceed `src`'s size; `dst_off` may
    ///   extend `dst` (the backend grows it).
    /// * Offsets and length must be aligned to the backend's
    ///   allocation unit (typically the cluster / block size); the
    ///   backend's docs specify the exact rule.
    fn clone_range(
        &mut self,
        _dev: &mut dyn crate::block::BlockDevice,
        _src: &Path,
        _src_off: u64,
        _dst: &Path,
        _dst_off: u64,
        _len: u64,
    ) -> crate::Result<()> {
        Err(crate::Error::Unsupported(
            "this filesystem does not implement clone_range (no extent sharing)".into(),
        ))
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

    /// Full attributes for `path`. Used by the FUSE adapter to populate
    /// `getattr` and `lookup` replies; also handy for any consumer that
    /// wants a complete stat-like result without juggling
    /// [`Self::list`] + the file handle.
    ///
    /// Default: best-effort. We `list` the parent and find the entry
    /// — that gives `kind`, `size`, and `inode`. The rest (mode, uid,
    /// gid, times) are defaulted by `FileAttrs::defaults_for` (mode
    /// `0o755` for dirs, `0o644` for files, `0o777` for symlinks; all
    /// times 0). Backends with per-file metadata (ext, hfs+, ntfs,
    /// xfs, …) should override.
    fn getattr(
        &mut self,
        dev: &mut dyn crate::block::BlockDevice,
        path: &Path,
    ) -> crate::Result<FileAttrs> {
        if path == Path::new("/") || path.as_os_str().is_empty() {
            return Ok(FileAttrs::defaults_for(EntryKind::Dir, 0, 1));
        }
        let parent = path.parent().unwrap_or(Path::new("/"));
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| crate::Error::InvalidArgument("getattr: bad path".into()))?;
        let entries = self.list(dev, parent)?;
        let entry = entries
            .into_iter()
            .find(|e| e.name == name)
            .ok_or_else(|| crate::Error::InvalidArgument(format!("getattr: {name} not found")))?;
        Ok(FileAttrs::defaults_for(entry.kind, entry.size, entry.inode))
    }

    /// Update attributes on `path`. Fields set to `None` in `attrs`
    /// are left unchanged. Backends that can't change a given field
    /// silently ignore it (FAT, for example, has no uid/gid concept).
    ///
    /// Default: returns `Unsupported`. Read-only and metadata-poor
    /// backends keep this default; mutable backends override.
    fn set_attrs(
        &mut self,
        _dev: &mut dyn crate::block::BlockDevice,
        _path: &Path,
        _attrs: SetAttrs,
    ) -> crate::Result<()> {
        Err(crate::Error::Unsupported(
            "this filesystem does not implement set_attrs".into(),
        ))
    }

    /// Resize `path` to `new_size` bytes. Equivalent to
    /// [`FileHandle::set_len`] reached through a path. Growing fills
    /// with zeros; shrinking discards trailing bytes and frees blocks.
    ///
    /// Default: returns `Unsupported`. Mutable backends override.
    fn truncate(
        &mut self,
        _dev: &mut dyn crate::block::BlockDevice,
        _path: &Path,
        _new_size: u64,
    ) -> crate::Result<()> {
        Err(crate::Error::Unsupported(
            "this filesystem does not implement truncate".into(),
        ))
    }

    /// Rename `old_path` to `new_path`. Cross-directory moves and
    /// directory renames are both in scope — the operation must
    /// preserve the target inode (so hardlinks survive).
    ///
    /// Default: returns `Unsupported`. Mutable backends override.
    fn rename(
        &mut self,
        _dev: &mut dyn crate::block::BlockDevice,
        _old_path: &Path,
        _new_path: &Path,
    ) -> crate::Result<()> {
        Err(crate::Error::Unsupported(
            "this filesystem does not implement rename".into(),
        ))
    }

    /// Add a new directory entry at `new_path` that points at the
    /// existing inode at `target_path` — a POSIX hard link.
    ///
    /// Default: returns `Unsupported`. Only ext implements this today.
    fn hardlink(
        &mut self,
        _dev: &mut dyn crate::block::BlockDevice,
        _target_path: &Path,
        _new_path: &Path,
    ) -> crate::Result<()> {
        Err(crate::Error::Unsupported(
            "this filesystem does not implement hardlink".into(),
        ))
    }

    /// List the extended attributes attached to `path`, with both
    /// names and values. The FUSE adapter splits this into
    /// `listxattr` / `getxattr` itself; we surface both at once so
    /// backends don't need two parallel walkers.
    ///
    /// Default: empty vec. Backends with xattr storage (ext, ntfs,
    /// hfs+) override.
    fn list_xattrs(
        &mut self,
        _dev: &mut dyn crate::block::BlockDevice,
        _path: &Path,
    ) -> crate::Result<Vec<XattrPair>> {
        Ok(Vec::new())
    }

    /// Write or replace the xattr `name` on `path`. Returns
    /// `Unsupported` when the backend can't store xattrs.
    fn set_xattr(
        &mut self,
        _dev: &mut dyn crate::block::BlockDevice,
        _path: &Path,
        _name: &str,
        _value: &[u8],
    ) -> crate::Result<()> {
        Err(crate::Error::Unsupported(
            "this filesystem does not implement set_xattr".into(),
        ))
    }

    /// Write a whole set of xattrs onto `path` at once, replacing any
    /// existing set. Backends that store xattrs in a single on-disk
    /// structure (ext's external attribute block) override this to write
    /// them atomically — applying them one at a time via [`set_xattr`]
    /// would orphan the previous block on each call. The default applies
    /// them individually.
    ///
    /// [`set_xattr`]: Self::set_xattr
    fn set_xattrs(
        &mut self,
        dev: &mut dyn crate::block::BlockDevice,
        path: &Path,
        xattrs: &[XattrPair],
    ) -> crate::Result<()> {
        for x in xattrs {
            self.set_xattr(dev, path, &x.name, &x.value)?;
        }
        Ok(())
    }

    /// Remove the xattr `name` from `path`. Returns `Unsupported`
    /// when the backend can't store xattrs.
    fn remove_xattr(
        &mut self,
        _dev: &mut dyn crate::block::BlockDevice,
        _path: &Path,
        _name: &str,
    ) -> crate::Result<()> {
        Err(crate::Error::Unsupported(
            "this filesystem does not implement remove_xattr".into(),
        ))
    }

    /// Filesystem-level capacity stats. The FUSE adapter calls this
    /// to answer `statfs`; the CLI's `info` command could too.
    ///
    /// Default: [`StatFs::default`] — 4 KiB block size, zero counts,
    /// `name_max = 255`. Backends with real superblock data override.
    fn statfs(&mut self, _dev: &mut dyn crate::block::BlockDevice) -> crate::Result<StatFs> {
        Ok(StatFs::default())
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

    #[test]
    fn file_attrs_defaults_set_kind_appropriate_mode() {
        // Directories default to 0o755 + nlink 2; regular files to
        // 0o644 + nlink 1; symlinks to 0o777. These are the values
        // the FUSE adapter surfaces for backends that don't override
        // `getattr`, so they need to look like a sensible mount.
        let d = FileAttrs::defaults_for(EntryKind::Dir, 0, 42);
        assert_eq!(d.mode, 0o755);
        assert_eq!(d.nlink, 2);
        assert_eq!(d.inode, 42);

        let f = FileAttrs::defaults_for(EntryKind::Regular, 1024, 7);
        assert_eq!(f.mode, 0o644);
        assert_eq!(f.nlink, 1);
        assert_eq!(f.size, 1024);
        // 1024 bytes = 2 × 512-byte blocks.
        assert_eq!(f.blocks, 2);

        let s = FileAttrs::defaults_for(EntryKind::Symlink, 16, 9);
        assert_eq!(s.mode, 0o777);
    }

    #[test]
    fn default_getattr_walks_parent_listing() {
        // Trait-level fallback: a backend that only implements the
        // five required methods still gets a working `getattr` via
        // `list` of the parent. We assert that here against a tiny
        // hand-rolled `Filesystem` to lock in the contract for any
        // backend that doesn't override.
        use crate::block::BlockDevice;
        struct Tiny {
            entries: Vec<DirEntry>,
        }
        impl Filesystem for Tiny {
            fn create_file(
                &mut self,
                _: &mut dyn BlockDevice,
                _: &Path,
                _: FileSource,
                _: FileMeta,
            ) -> crate::Result<()> {
                unimplemented!()
            }
            fn create_dir(
                &mut self,
                _: &mut dyn BlockDevice,
                _: &Path,
                _: FileMeta,
            ) -> crate::Result<()> {
                unimplemented!()
            }
            fn create_symlink(
                &mut self,
                _: &mut dyn BlockDevice,
                _: &Path,
                _: &Path,
                _: FileMeta,
            ) -> crate::Result<()> {
                unimplemented!()
            }
            fn create_device(
                &mut self,
                _: &mut dyn BlockDevice,
                _: &Path,
                _: DeviceKind,
                _: u32,
                _: u32,
                _: FileMeta,
            ) -> crate::Result<()> {
                unimplemented!()
            }
            fn remove(&mut self, _: &mut dyn BlockDevice, _: &Path) -> crate::Result<()> {
                unimplemented!()
            }
            fn list(&mut self, _: &mut dyn BlockDevice, _: &Path) -> crate::Result<Vec<DirEntry>> {
                Ok(self.entries.clone())
            }
            fn read_file<'a>(
                &'a mut self,
                _: &'a mut dyn BlockDevice,
                _: &Path,
            ) -> crate::Result<Box<dyn Read + 'a>> {
                unimplemented!()
            }
            fn flush(&mut self, _: &mut dyn BlockDevice) -> crate::Result<()> {
                Ok(())
            }
        }
        let mut t = Tiny {
            entries: vec![DirEntry {
                name: "hello.txt".into(),
                inode: 17,
                kind: EntryKind::Regular,
                size: 11,
            }],
        };
        // Need *some* BlockDevice; MemoryBackend works.
        let mut dev = crate::block::MemoryBackend::new(4096);
        let attrs = t.getattr(&mut dev, Path::new("/hello.txt")).unwrap();
        assert_eq!(attrs.kind, EntryKind::Regular);
        assert_eq!(attrs.size, 11);
        assert_eq!(attrs.inode, 17);
        assert_eq!(attrs.mode, 0o644);
    }
}
