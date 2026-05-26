//! Shared archive core — the machinery every archive-format backend
//! plugs into.
//!
//! Archives (zip, cpio, ar, …) aren't real filesystems, but — like
//! [`tar`](crate::fs::tar) — they carry the same per-entry metadata
//! (path, mode, uid/gid, mtime, symlink target, device numbers), so
//! they slot into the [`Filesystem`](crate::fs::Filesystem) trait and
//! flow through `info` / `ls` / `cat` / `repack` on the same codepath
//! as ext4/NTFS/etc.
//!
//! Rather than copy tar's trait plumbing once per format, the formats
//! share this core:
//!
//! - [`ArchiveIndex`] — a flat list of [`ArchiveEntry`] plus a
//!   synthesised directory tree (`children`), built once by a format's
//!   *scanner* from a single pass over the device.
//! - [`ArchiveFs`] — one struct that implements the whole
//!   [`Filesystem`](crate::fs::Filesystem) surface over an
//!   `ArchiveIndex`. Reads resolve from the index and decode the
//!   located byte range through [`reader::open`]; writes (when the
//!   handle was built via `format`) flow through a format-specific
//!   [`ArchiveBuilder`].
//!
//! A concrete format therefore supplies only (a) a scanner that
//! populates an [`ArchiveIndex`] and (b) — if writable — an
//! [`ArchiveBuilder`]. The `impl_archive_fs_filesystem` macro
//! generates the one-line `Filesystem` forwarding for each format's
//! newtype so none of it is hand-written ten times.

pub mod reader;
pub mod tree;
pub mod writer;

// Fully-implemented formats (read + repack).
pub mod ar;
pub mod cpio;
pub mod zip;

// Detection + scaffold formats: recognised by `detect_fs`, but their
// readers/writers are not implemented yet and return a clean
// `Unsupported` naming the format. Filled in later behind pure-Rust,
// feature-gated decoder crates.
pub mod arc;
pub mod cab;
pub mod lha;
pub mod lzx;
pub mod rar;
pub mod sevenz;
pub mod sit;

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::{
    DeviceKind, DirEntry, FileAttrs, FileMeta, FileReadHandle, FileSource, MutationCapability,
    SetAttrs,
};

/// File-type bucket for an archive entry. Superset of
/// [`crate::fs::EntryKind`] — adds `HardLink`, which archives express
/// but the FUSE-facing enum folds into `Regular`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    Regular,
    Dir,
    Symlink,
    HardLink,
    Char,
    Block,
    Fifo,
    Socket,
}

impl EntryKind {
    /// Map to the FUSE-facing [`crate::fs::EntryKind`]. Hard links
    /// surface as regular files (their target's content is resolved at
    /// scan time).
    pub fn to_fs(self) -> crate::fs::EntryKind {
        match self {
            Self::Regular | Self::HardLink => crate::fs::EntryKind::Regular,
            Self::Dir => crate::fs::EntryKind::Dir,
            Self::Symlink => crate::fs::EntryKind::Symlink,
            Self::Char => crate::fs::EntryKind::Char,
            Self::Block => crate::fs::EntryKind::Block,
            Self::Fifo => crate::fs::EntryKind::Fifo,
            Self::Socket => crate::fs::EntryKind::Socket,
        }
    }
}

/// How one entry's body is encoded in the archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// No compression — the byte range is the file content verbatim.
    Stored,
    /// Raw DEFLATE (zip method 8). Decoded via `flate2` directly (the
    /// shared codec layer only exposes gzip/zlib *framing*).
    Deflate,
    /// One of the streaming codecs in [`crate::compression`].
    Codec(crate::compression::Algo),
    /// A method id we can name but not decode. Indexing still succeeds;
    /// reading the body returns `Unsupported`.
    Unsupported(u16),
}

/// Where an entry's bytes live on the device and how to decode them.
#[derive(Debug, Clone)]
pub struct DataLocator {
    /// Absolute byte offset of the (possibly compressed) body in the device.
    pub offset: u64,
    /// Bytes to read from the device (== `uncompressed_len` for `Stored`).
    pub compressed_len: u64,
    /// Logical file size — the value `list` / `getattr` report.
    pub uncompressed_len: u64,
    /// Codec used for the body.
    pub method: Method,
}

/// One archived entry, fully resolved.
#[derive(Debug, Clone)]
pub struct ArchiveEntry {
    /// Normalised path: starts with `/`, no trailing `/`.
    pub path: String,
    pub kind: EntryKind,
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub mtime: u64,
    /// Symlink / hard-link target.
    pub link_target: Option<String>,
    pub device_major: u32,
    pub device_minor: u32,
    /// `Some` for regular files (and resolved hard links); `None` for
    /// dirs / symlinks / devices.
    pub data: Option<DataLocator>,
}

impl ArchiveEntry {
    /// A bare regular-file entry with default metadata; callers set the
    /// fields they carry.
    pub fn regular(path: impl Into<String>, data: DataLocator) -> Self {
        Self {
            path: path.into(),
            kind: EntryKind::Regular,
            mode: 0o644,
            uid: 0,
            gid: 0,
            mtime: 0,
            link_target: None,
            device_major: 0,
            device_minor: 0,
            data: Some(data),
        }
    }

    /// A directory entry with default metadata.
    pub fn dir(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            kind: EntryKind::Dir,
            mode: 0o755,
            uid: 0,
            gid: 0,
            mtime: 0,
            link_target: None,
            device_major: 0,
            device_minor: 0,
            data: None,
        }
    }

    fn logical_size(&self) -> u64 {
        match (&self.data, self.kind) {
            (Some(loc), EntryKind::Regular | EntryKind::HardLink) => loc.uncompressed_len,
            _ => 0,
        }
    }
}

/// A scanned archive resolved into memory: the flat entry list plus a
/// synthesised parent → children map so `list` works even for formats
/// (ar) that store no directory records at all.
pub struct ArchiveIndex {
    /// Short identifier (`"zip"`, `"cpio"`, …) used in error messages.
    pub kind: &'static str,
    entries: Vec<ArchiveEntry>,
    by_path: HashMap<String, usize>,
    children: HashMap<String, Vec<String>>,
}

impl ArchiveIndex {
    /// An empty index rooted at `/`.
    pub fn new(kind: &'static str) -> Self {
        let mut children = HashMap::new();
        children.insert("/".to_string(), Vec::new());
        Self {
            kind,
            entries: Vec::new(),
            by_path: HashMap::new(),
            children,
        }
    }

    /// Insert an entry, normalising its path and registering every
    /// ancestor directory (so `list("/a")` works when only
    /// `/a/b/c.txt` was stored). A later entry for the same path
    /// replaces an earlier one (last-write-wins, matching how layered
    /// archives resolve duplicates).
    pub fn push(&mut self, mut e: ArchiveEntry) {
        e.path = tree::normalise_path(&e.path);
        if e.path == "/" {
            // Root is implicit; ignore an explicit root record.
            return;
        }

        // Register each path component as a child of its parent, and
        // give every intermediate component an empty children bucket.
        let comps: Vec<&str> = e.path.trim_start_matches('/').split('/').collect();
        let mut parent = String::from("/");
        for (i, comp) in comps.iter().enumerate() {
            let child_path = if parent == "/" {
                format!("/{comp}")
            } else {
                format!("{parent}/{comp}")
            };
            let kids = self.children.entry(parent.clone()).or_default();
            if !kids.iter().any(|k| k == comp) {
                kids.push((*comp).to_string());
            }
            let is_leaf = i + 1 == comps.len();
            if !is_leaf {
                self.children.entry(child_path.clone()).or_default();
            }
            parent = child_path;
        }
        if matches!(e.kind, EntryKind::Dir) {
            self.children.entry(e.path.clone()).or_default();
        }

        if let Some(&existing) = self.by_path.get(&e.path) {
            self.entries[existing] = e;
        } else {
            let idx = self.entries.len();
            self.by_path.insert(e.path.clone(), idx);
            self.entries.push(e);
        }
    }

    /// All entries in insertion order.
    pub fn entries(&self) -> &[ArchiveEntry] {
        &self.entries
    }

    /// Look up an entry by path (normalised).
    pub fn lookup(&self, path: &str) -> Option<&ArchiveEntry> {
        let p = tree::normalise_path(path);
        self.by_path.get(&p).map(|&i| &self.entries[i])
    }

    /// List a directory's children as [`DirEntry`]s. Intermediate
    /// directories with no stored record list as `Dir`.
    pub fn list(&self, path: &str) -> Result<Vec<DirEntry>> {
        let p = tree::normalise_path(path);
        let names = self.children.get(&p).ok_or_else(|| {
            crate::Error::InvalidArgument(format!("{}: no such directory {p:?}", self.kind))
        })?;
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let child = if p == "/" {
                format!("/{name}")
            } else {
                format!("{p}/{name}")
            };
            let (kind, size, inode) = match self.by_path.get(&child) {
                Some(&i) => {
                    let e = &self.entries[i];
                    (e.kind.to_fs(), e.logical_size(), i as u32 + 1)
                }
                // Synthesised intermediate directory.
                None => (crate::fs::EntryKind::Dir, 0, 0),
            };
            out.push(DirEntry {
                name: name.clone(),
                inode,
                kind,
                size,
            });
        }
        Ok(out)
    }
}

/// Format-specific write side. A freshly-`format`ted [`ArchiveFs`]
/// holds one of these; each `create_*` call appends to the output
/// device and `finish` lays down the trailer / central directory.
///
/// Implementors hold only a cursor + capacity + in-memory bookkeeping
/// (never a borrow of the device) — the device is passed in on every
/// call, mirroring the [`Filesystem`](crate::fs::Filesystem) methods.
pub trait ArchiveBuilder: Send {
    fn add_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        src: FileSource,
        meta: FileMeta,
    ) -> Result<()>;

    fn add_dir(&mut self, dev: &mut dyn BlockDevice, path: &str, meta: FileMeta) -> Result<()>;

    fn add_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        target: &str,
        meta: FileMeta,
    ) -> Result<()>;

    fn add_device(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: FileMeta,
    ) -> Result<()>;

    /// Finalise the archive (trailer / central directory) and `sync`.
    fn finish(&mut self, dev: &mut dyn BlockDevice) -> Result<()>;

    /// Bytes written to the device so far (the write cursor). Read after
    /// [`finish`](Self::finish) to truncate the over-provisioned backing
    /// file to the archive's true length.
    fn position(&self) -> u64;
}

/// Generic archive filesystem: one implementation of the read side of
/// [`Filesystem`](crate::fs::Filesystem) over an [`ArchiveIndex`], plus
/// an optional [`ArchiveBuilder`] for the write (repack) path.
pub struct ArchiveFs {
    index: ArchiveIndex,
    /// Present only when built via a format's `format()` (write mode).
    builder: Option<Box<dyn ArchiveBuilder>>,
    cap: MutationCapability,
    /// `true` for detection-only scaffolds: reads return a clean
    /// `Unsupported` naming the format instead of an empty tree.
    scaffold: bool,
    /// Archive byte length recorded after `flush` finalises the writer.
    flushed_len: Option<u64>,
}

impl ArchiveFs {
    /// Read-only handle over an already-scanned index.
    pub fn from_index(index: ArchiveIndex) -> Self {
        Self {
            index,
            builder: None,
            cap: MutationCapability::Streaming,
            scaffold: false,
            flushed_len: None,
        }
    }

    /// Write handle: an empty index plus a format-specific builder.
    /// `kind` is the format's short id (for errors).
    pub fn writer(kind: &'static str, builder: Box<dyn ArchiveBuilder>) -> Self {
        Self {
            index: ArchiveIndex::new(kind),
            builder: Some(builder),
            cap: MutationCapability::Streaming,
            scaffold: false,
            flushed_len: None,
        }
    }

    /// Detection-only handle for a format whose reader isn't built yet.
    pub fn scaffold(kind: &'static str) -> Self {
        Self {
            index: ArchiveIndex::new(kind),
            builder: None,
            cap: MutationCapability::Immutable,
            scaffold: true,
            flushed_len: None,
        }
    }

    /// The format's short identifier.
    pub fn kind(&self) -> &'static str {
        self.index.kind
    }

    fn guard_scaffold(&self, op: &str) -> Result<()> {
        if self.scaffold {
            return Err(crate::Error::Unsupported(format!(
                "{}: {op} not implemented yet — this format is detection-only",
                self.index.kind
            )));
        }
        Ok(())
    }

    fn write_refused(&self, op: &'static str) -> crate::Error {
        match self.cap {
            MutationCapability::Immutable => crate::Error::Immutable {
                kind: self.index.kind,
                op,
            },
            _ => crate::Error::Streaming {
                kind: self.index.kind,
                op,
            },
        }
    }

    fn path_str<'p>(&self, path: &'p Path) -> Result<&'p str> {
        path.to_str().ok_or_else(|| {
            crate::Error::InvalidArgument(format!("{}: non-UTF-8 path", self.index.kind))
        })
    }
}

impl crate::fs::Filesystem for ArchiveFs {
    fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &Path,
        src: FileSource,
        meta: FileMeta,
    ) -> Result<()> {
        let s = self.path_str(path)?.to_string();
        match self.builder.as_mut() {
            Some(b) => b.add_file(dev, &s, src, meta),
            None => Err(self.write_refused("write")),
        }
    }

    /// The builder writes each entry's header + body straight to the
    /// device cursor as `create_file` is called, so the streaming repack
    /// path can hand us a body without spooling it to a temp file first
    /// (small bodies buffer in RAM; larger ones spill, same as every other
    /// stream-through backend).
    fn streams_immediately(&self) -> bool {
        true
    }

    fn create_dir(&mut self, dev: &mut dyn BlockDevice, path: &Path, meta: FileMeta) -> Result<()> {
        let s = self.path_str(path)?.to_string();
        match self.builder.as_mut() {
            Some(b) => b.add_dir(dev, &s, meta),
            None => Err(self.write_refused("write")),
        }
    }

    fn create_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &Path,
        target: &Path,
        meta: FileMeta,
    ) -> Result<()> {
        let s = self.path_str(path)?.to_string();
        let t = target.to_str().ok_or_else(|| {
            crate::Error::InvalidArgument(format!("{}: non-UTF-8 symlink target", self.index.kind))
        })?;
        match self.builder.as_mut() {
            Some(b) => b.add_symlink(dev, &s, t, meta),
            None => Err(self.write_refused("write")),
        }
    }

    fn create_device(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &Path,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: FileMeta,
    ) -> Result<()> {
        let s = self.path_str(path)?.to_string();
        match self.builder.as_mut() {
            Some(b) => b.add_device(dev, &s, kind, major, minor, meta),
            None => Err(self.write_refused("write")),
        }
    }

    fn remove(&mut self, _dev: &mut dyn BlockDevice, _path: &Path) -> Result<()> {
        Err(self.write_refused("rm"))
    }

    fn list(&mut self, _dev: &mut dyn BlockDevice, path: &Path) -> Result<Vec<DirEntry>> {
        self.guard_scaffold("list")?;
        let s = self.path_str(path)?;
        self.index.list(s)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &Path,
    ) -> Result<Box<dyn Read + 'a>> {
        self.guard_scaffold("read")?;
        let s = self.path_str(path)?;
        let e = self.index.lookup(s).ok_or_else(|| {
            crate::Error::InvalidArgument(format!("{}: no entry at {s:?}", self.index.kind))
        })?;
        // Clone the locator out of the index so the returned reader
        // borrows only `dev`, not `self` (the tar trick).
        let loc = match (e.kind, &e.data) {
            (EntryKind::Regular | EntryKind::HardLink, Some(loc)) => loc.clone(),
            _ => {
                return Err(crate::Error::InvalidArgument(format!(
                    "{}: {s:?} is not a regular file",
                    self.index.kind
                )));
            }
        };
        reader::open(dev, loc)
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &Path,
    ) -> Result<Box<dyn FileReadHandle + 'a>> {
        self.guard_scaffold("read")?;
        let s = self.path_str(path)?;
        let e = self.index.lookup(s).ok_or_else(|| {
            crate::Error::InvalidArgument(format!("{}: no entry at {s:?}", self.index.kind))
        })?;
        let loc = match (e.kind, &e.data) {
            (EntryKind::Regular | EntryKind::HardLink, Some(loc)) => loc.clone(),
            _ => {
                return Err(crate::Error::InvalidArgument(format!(
                    "{}: {s:?} is not a regular file",
                    self.index.kind
                )));
            }
        };
        reader::open_ro(dev, loc)
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        if let Some(b) = self.builder.as_mut() {
            b.finish(dev)?;
            self.flushed_len = Some(b.position());
        }
        Ok(())
    }

    fn image_len(&self) -> Option<u64> {
        self.flushed_len
    }

    fn mutation_capability(&self) -> MutationCapability {
        self.cap
    }

    fn read_symlink(&mut self, _dev: &mut dyn BlockDevice, path: &Path) -> Result<PathBuf> {
        self.guard_scaffold("read")?;
        let s = self.path_str(path)?;
        let e = self.index.lookup(s).ok_or_else(|| {
            crate::Error::InvalidArgument(format!("{}: no entry at {s:?}", self.index.kind))
        })?;
        if !matches!(e.kind, EntryKind::Symlink) {
            return Err(crate::Error::InvalidArgument(format!(
                "{}: {s:?} is not a symlink",
                self.index.kind
            )));
        }
        e.link_target.clone().map(PathBuf::from).ok_or_else(|| {
            crate::Error::InvalidArgument(format!(
                "{}: symlink {s:?} has no target",
                self.index.kind
            ))
        })
    }

    fn getattr(&mut self, dev: &mut dyn BlockDevice, path: &Path) -> Result<FileAttrs> {
        self.guard_scaffold("getattr").or_else(|e| {
            // Allow stat of the root even on a scaffold so `info` has a
            // heading to print; deeper paths surface the Unsupported.
            if path == Path::new("/") || path.as_os_str().is_empty() {
                Ok(())
            } else {
                Err(e)
            }
        })?;
        if path == Path::new("/") || path.as_os_str().is_empty() {
            return Ok(dir_attrs(1));
        }
        let s = self.path_str(path)?;
        let (e, inode) = {
            let idx = self.index.by_path.get(s).copied();
            match idx {
                Some(i) => (self.index.entries[i].clone(), i as u32 + 1),
                None => {
                    // Could be a synthesised intermediate dir.
                    if self.index.children.contains_key(s) {
                        return Ok(dir_attrs(0));
                    }
                    // Fall back to the trait default (list the parent).
                    return default_getattr(self, dev, path);
                }
            }
        };
        let size = e.logical_size();
        Ok(FileAttrs {
            kind: e.kind.to_fs(),
            mode: e.mode,
            uid: e.uid,
            gid: e.gid,
            size,
            blocks: size.div_ceil(512),
            nlink: 1,
            atime: e.mtime as u32,
            mtime: e.mtime as u32,
            ctime: e.mtime as u32,
            rdev: 0,
            inode,
        })
    }

    fn set_attrs(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &Path,
        _attrs: SetAttrs,
    ) -> Result<()> {
        Err(self.write_refused("set_attrs"))
    }
}

/// A directory [`FileAttrs`] with the given inode.
fn dir_attrs(inode: u32) -> FileAttrs {
    FileAttrs {
        kind: crate::fs::EntryKind::Dir,
        mode: 0o755,
        uid: 0,
        gid: 0,
        size: 0,
        blocks: 0,
        nlink: 2,
        atime: 0,
        mtime: 0,
        ctime: 0,
        rdev: 0,
        inode,
    }
}

/// Reproduce the trait-default `getattr` (list the parent, find the
/// entry) for paths that aren't directly in `by_path`.
fn default_getattr(
    fs: &mut ArchiveFs,
    dev: &mut dyn BlockDevice,
    path: &Path,
) -> Result<FileAttrs> {
    let parent = path.parent().unwrap_or(Path::new("/"));
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| crate::Error::InvalidArgument("getattr: bad path".into()))?;
    let entries = crate::fs::Filesystem::list(fs, dev, parent)?;
    let entry = entries
        .into_iter()
        .find(|e| e.name == name)
        .ok_or_else(|| crate::Error::InvalidArgument(format!("getattr: {name} not found")))?;
    let mode = match entry.kind {
        crate::fs::EntryKind::Dir => 0o755,
        crate::fs::EntryKind::Symlink => 0o777,
        _ => 0o644,
    };
    Ok(FileAttrs {
        kind: entry.kind,
        mode,
        uid: 0,
        gid: 0,
        size: entry.size,
        blocks: entry.size.div_ceil(512),
        nlink: 1,
        atime: 0,
        mtime: 0,
        ctime: 0,
        rdev: 0,
        inode: entry.inode,
    })
}

/// Generate the `Filesystem` impl for a format newtype `struct Foo(ArchiveFs)`
/// by forwarding every method to the inner [`ArchiveFs`].
#[macro_export]
macro_rules! impl_archive_fs_filesystem {
    ($t:ty) => {
        impl $crate::fs::Filesystem for $t {
            fn create_file(
                &mut self,
                dev: &mut dyn $crate::block::BlockDevice,
                path: &std::path::Path,
                src: $crate::fs::FileSource,
                meta: $crate::fs::FileMeta,
            ) -> $crate::Result<()> {
                self.0.create_file(dev, path, src, meta)
            }
            fn create_dir(
                &mut self,
                dev: &mut dyn $crate::block::BlockDevice,
                path: &std::path::Path,
                meta: $crate::fs::FileMeta,
            ) -> $crate::Result<()> {
                self.0.create_dir(dev, path, meta)
            }
            fn create_symlink(
                &mut self,
                dev: &mut dyn $crate::block::BlockDevice,
                path: &std::path::Path,
                target: &std::path::Path,
                meta: $crate::fs::FileMeta,
            ) -> $crate::Result<()> {
                self.0.create_symlink(dev, path, target, meta)
            }
            fn create_device(
                &mut self,
                dev: &mut dyn $crate::block::BlockDevice,
                path: &std::path::Path,
                kind: $crate::fs::DeviceKind,
                major: u32,
                minor: u32,
                meta: $crate::fs::FileMeta,
            ) -> $crate::Result<()> {
                self.0.create_device(dev, path, kind, major, minor, meta)
            }
            fn remove(
                &mut self,
                dev: &mut dyn $crate::block::BlockDevice,
                path: &std::path::Path,
            ) -> $crate::Result<()> {
                self.0.remove(dev, path)
            }
            fn list(
                &mut self,
                dev: &mut dyn $crate::block::BlockDevice,
                path: &std::path::Path,
            ) -> $crate::Result<Vec<$crate::fs::DirEntry>> {
                self.0.list(dev, path)
            }
            fn read_file<'a>(
                &'a mut self,
                dev: &'a mut dyn $crate::block::BlockDevice,
                path: &std::path::Path,
            ) -> $crate::Result<Box<dyn std::io::Read + 'a>> {
                self.0.read_file(dev, path)
            }
            fn open_file_ro<'a>(
                &'a mut self,
                dev: &'a mut dyn $crate::block::BlockDevice,
                path: &std::path::Path,
            ) -> $crate::Result<Box<dyn $crate::fs::FileReadHandle + 'a>> {
                self.0.open_file_ro(dev, path)
            }
            fn flush(&mut self, dev: &mut dyn $crate::block::BlockDevice) -> $crate::Result<()> {
                self.0.flush(dev)
            }
            fn streams_immediately(&self) -> bool {
                self.0.streams_immediately()
            }
            fn image_len(&self) -> Option<u64> {
                self.0.image_len()
            }
            fn mutation_capability(&self) -> $crate::fs::MutationCapability {
                self.0.mutation_capability()
            }
            fn read_symlink(
                &mut self,
                dev: &mut dyn $crate::block::BlockDevice,
                path: &std::path::Path,
            ) -> $crate::Result<std::path::PathBuf> {
                self.0.read_symlink(dev, path)
            }
            fn getattr(
                &mut self,
                dev: &mut dyn $crate::block::BlockDevice,
                path: &std::path::Path,
            ) -> $crate::Result<$crate::fs::FileAttrs> {
                self.0.getattr(dev, path)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(len: u64) -> DataLocator {
        DataLocator {
            offset: 0,
            compressed_len: len,
            uncompressed_len: len,
            method: Method::Stored,
        }
    }

    #[test]
    fn synthesises_intermediate_dirs() {
        // An archive that stores only a deep file must still list every
        // intermediate directory.
        let mut idx = ArchiveIndex::new("test");
        idx.push(ArchiveEntry::regular("a/b/c.txt", loc(10)));

        let root = idx.list("/").unwrap();
        assert_eq!(root.len(), 1);
        assert_eq!(root[0].name, "a");
        assert_eq!(root[0].kind, crate::fs::EntryKind::Dir);

        let a = idx.list("/a").unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].name, "b");

        let b = idx.list("/a/b").unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].name, "c.txt");
        assert_eq!(b[0].kind, crate::fs::EntryKind::Regular);
        assert_eq!(b[0].size, 10);
    }

    #[test]
    fn last_write_wins_on_duplicate_path() {
        let mut idx = ArchiveIndex::new("test");
        idx.push(ArchiveEntry::regular("dup", loc(1)));
        idx.push(ArchiveEntry::regular("dup", loc(99)));
        assert_eq!(idx.lookup("/dup").unwrap().logical_size(), 99);
        // Not double-listed.
        assert_eq!(idx.list("/").unwrap().len(), 1);
    }

    #[test]
    fn explicit_dir_record_merges_with_synthesised() {
        let mut idx = ArchiveIndex::new("test");
        idx.push(ArchiveEntry::regular("d/f", loc(1)));
        idx.push(ArchiveEntry::dir("d"));
        // Listing root still shows a single `d` (Dir).
        let root = idx.list("/").unwrap();
        assert_eq!(root.len(), 1);
        assert_eq!(root[0].kind, crate::fs::EntryKind::Dir);
        assert_eq!(idx.list("/d").unwrap()[0].name, "f");
    }
}
