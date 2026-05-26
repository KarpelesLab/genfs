//! SquashFS — read-only compressed archive filesystem.
//!
//! ## Status
//!
//! Read-only support for **uncompressed** SquashFS v4 images:
//!
//! - Listing any directory by absolute path.
//! - Streaming any regular file by absolute path.
//! - Reading symlink targets.
//!
//! Compressed metablocks, data blocks, and fragment blocks return
//! [`crate::Error::Unsupported`] with the algorithm name. The integrator
//! gates real decompressors (gzip / xz / lz4 / zstd / lzo / lzma) behind
//! optional Cargo features so this module stays dependency-free.
//!
//! ## Reference
//!
//! - <https://docs.kernel.org/filesystems/squashfs.html> — kernel docs.
//! - <https://dr-emann.github.io/squashfs/squashfs.html> — community
//!   binary-format reference (cross-checked field offsets only).
//!
//! ## Versioning
//!
//! Only major version 4 is accepted. Earlier images open with an
//! [`crate::Error::Unsupported`] error naming the version.

use std::cell::RefCell;
use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::DeviceKind;
use crate::fs::DirEntry;
use crate::fs::FileSource;

mod directory;
mod file;
mod fragment;
mod idtable;
mod inode;
mod metablock;
mod writer;
mod xattr;

pub use file::FileReader;
pub use writer::{DEFAULT_BLOCK_SIZE, EntryMeta};
pub use xattr::Xattr;

/// SquashFS magic, little-endian: `hsqs` reversed on disk = `0x73717368`.
const SQUASHFS_MAGIC: u32 = 0x7371_7368;

/// Compression scheme advertised in the superblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    Gzip,
    Lzma,
    Lzo,
    Xz,
    Lz4,
    Zstd,
    Unknown(u16),
}

impl Compression {
    fn from_id(id: u16) -> Self {
        match id {
            1 => Self::Gzip,
            2 => Self::Lzma,
            3 => Self::Lzo,
            4 => Self::Xz,
            5 => Self::Lz4,
            6 => Self::Zstd,
            other => Self::Unknown(other),
        }
    }
}

/// Decoded SquashFS superblock. The on-disk layout is 96 bytes total; we
/// surface everything later layers need to walk the metadata tables.
#[derive(Debug, Clone)]
pub struct Superblock {
    pub magic: u32,
    pub inode_count: u32,
    /// Last modification time in Unix seconds.
    pub mkfs_time: u32,
    /// Block size in bytes (power-of-two, 4 KiB … 1 MiB).
    pub block_size: u32,
    /// Number of fragment entries in the fragment table.
    pub fragment_count: u32,
    pub compression: Compression,
    /// log2(block_size), redundant with `block_size`.
    pub block_log: u16,
    pub flags: u16,
    /// IDs in the id lookup table (uid+gid entries).
    pub id_count: u16,
    pub major: u16,
    pub minor: u16,
    /// Inode reference (block,offset) for the root directory inode.
    pub root_inode: u64,
    pub bytes_used: u64,
    pub id_table_start: u64,
    pub xattr_id_table_start: u64,
    pub inode_table_start: u64,
    pub directory_table_start: u64,
    pub fragment_table_start: u64,
    pub export_table_start: u64,
}

impl Superblock {
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 96 {
            return None;
        }
        let magic = u32::from_le_bytes(buf[0..4].try_into().ok()?);
        if magic != SQUASHFS_MAGIC {
            return None;
        }
        let inode_count = u32::from_le_bytes(buf[4..8].try_into().ok()?);
        let mkfs_time = u32::from_le_bytes(buf[8..12].try_into().ok()?);
        let block_size = u32::from_le_bytes(buf[12..16].try_into().ok()?);
        let fragment_count = u32::from_le_bytes(buf[16..20].try_into().ok()?);
        let compression = Compression::from_id(u16::from_le_bytes(buf[20..22].try_into().ok()?));
        let block_log = u16::from_le_bytes(buf[22..24].try_into().ok()?);
        let flags = u16::from_le_bytes(buf[24..26].try_into().ok()?);
        let id_count = u16::from_le_bytes(buf[26..28].try_into().ok()?);
        let major = u16::from_le_bytes(buf[28..30].try_into().ok()?);
        let minor = u16::from_le_bytes(buf[30..32].try_into().ok()?);
        let root_inode = u64::from_le_bytes(buf[32..40].try_into().ok()?);
        let bytes_used = u64::from_le_bytes(buf[40..48].try_into().ok()?);
        let id_table_start = u64::from_le_bytes(buf[48..56].try_into().ok()?);
        let xattr_id_table_start = u64::from_le_bytes(buf[56..64].try_into().ok()?);
        let inode_table_start = u64::from_le_bytes(buf[64..72].try_into().ok()?);
        let directory_table_start = u64::from_le_bytes(buf[72..80].try_into().ok()?);
        let fragment_table_start = u64::from_le_bytes(buf[80..88].try_into().ok()?);
        let export_table_start = u64::from_le_bytes(buf[88..96].try_into().ok()?);
        Some(Self {
            magic,
            inode_count,
            mkfs_time,
            block_size,
            fragment_count,
            compression,
            block_log,
            flags,
            id_count,
            major,
            minor,
            root_inode,
            bytes_used,
            id_table_start,
            xattr_id_table_start,
            inode_table_start,
            directory_table_start,
            fragment_table_start,
            export_table_start,
        })
    }
}

/// Quick detection — reads only the first four bytes.
pub fn probe(dev: &mut dyn BlockDevice) -> Result<bool> {
    if dev.total_size() < 4 {
        return Ok(false);
    }
    let mut head = [0u8; 4];
    dev.read_at(0, &mut head)?;
    Ok(u32::from_le_bytes(head) == SQUASHFS_MAGIC)
}

/// Format-time options for [`Squashfs::format`].
#[derive(Debug, Clone)]
pub struct FormatOpts {
    /// Data block size, must be a power of two between 4 KiB and 1 MiB.
    /// Defaults to 128 KiB.
    pub block_size: u32,
    /// Compression algorithm used for both metadata + data. Defaults to
    /// [`Compression::Zstd`] when the `zstd` feature is enabled, else
    /// [`Compression::Gzip`].
    pub compression: Compression,
}

impl Default for FormatOpts {
    fn default() -> Self {
        let compression = if cfg!(feature = "zstd") {
            Compression::Zstd
        } else if cfg!(feature = "gzip") {
            Compression::Gzip
        } else {
            Compression::Unknown(0)
        };
        Self {
            block_size: writer::DEFAULT_BLOCK_SIZE,
            compression,
        }
    }
}

impl FormatOpts {
    /// Apply a generic option-bag (CLI `-O key=val` / TOML
    /// `[filesystem.options]`) on top of these opts. Unknown keys are
    /// left in the map for the caller to flag.
    pub fn apply_options(&mut self, map: &mut crate::format_opts::OptionMap) -> crate::Result<()> {
        if let Some(sz) = map.take_size("block_size")? {
            self.block_size = sz as u32;
        }
        if let Some(s) = map.take_str("compression") {
            self.compression = match s.to_ascii_lowercase().as_str() {
                "gzip" | "zlib" | "deflate" => Compression::Gzip,
                "lzma" => Compression::Lzma,
                "lzo" => Compression::Lzo,
                "xz" => Compression::Xz,
                "lz4" => Compression::Lz4,
                "zstd" | "zstandard" => Compression::Zstd,
                other => {
                    return Err(crate::Error::InvalidImage(format!(
                        "unknown squashfs compression `{other}`"
                    )));
                }
            };
        }
        Ok(())
    }
}

/// Resolved metadata for a single inode, returned by
/// [`Squashfs::inode_meta`].
#[derive(Debug, Clone)]
pub struct InodeMeta {
    pub inode_number: u32,
    pub kind: crate::fs::EntryKind,
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub mtime: u32,
    pub file_size: u64,
    pub xattr_index: u32,
}

/// Open handle on a SquashFS image. Owns the on-disk superblock plus
/// lazily-populated caches for the id and xattr tables. Read APIs may
/// populate those caches on first use. After [`Squashfs::format`] the
/// instance is in *write* mode; call [`Squashfs::flush`] to materialise
/// the image, then it switches back to read mode.
pub struct Squashfs {
    sb: Superblock,
    id_table: RefCell<idtable::IdTable>,
    xattr_reader: RefCell<xattr::XattrReader>,
    write_state: Option<writer::WriteState>,
}

impl std::fmt::Debug for Squashfs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Squashfs")
            .field("sb", &self.sb)
            .field("write_mode", &self.write_state.is_some())
            .finish()
    }
}

impl Squashfs {
    /// Open an existing SquashFS image. Validates magic + version; the
    /// metadata tables are not touched until a `list_path` /
    /// `open_file_reader` / `read_symlink` call walks them.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        if dev.total_size() < 96 {
            return Err(crate::Error::InvalidImage(
                "squashfs: device too small to hold a superblock".into(),
            ));
        }
        let mut buf = [0u8; 96];
        dev.read_at(0, &mut buf)?;
        let sb = Superblock::decode(&buf).ok_or_else(|| {
            crate::Error::InvalidImage("squashfs: superblock magic mismatch".into())
        })?;
        if sb.major != 4 {
            return Err(crate::Error::Unsupported(format!(
                "squashfs: only version 4.x is supported (got {}.{})",
                sb.major, sb.minor
            )));
        }
        Ok(Self {
            sb,
            id_table: RefCell::new(idtable::IdTable::new()),
            xattr_reader: RefCell::new(xattr::XattrReader::new()),
            write_state: None,
        })
    }

    /// Begin building a fresh SquashFS image on `dev`. The returned
    /// handle is in *write* mode: call [`create_dir`](Self::create_dir),
    /// [`create_file`](Self::create_file),
    /// [`create_symlink`](Self::create_symlink) to populate the tree,
    /// then [`flush`](Self::flush) to materialise it.
    pub fn format(_dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<Self> {
        if !opts.block_size.is_power_of_two()
            || opts.block_size < 4096
            || opts.block_size > 1_048_576
        {
            return Err(crate::Error::InvalidArgument(format!(
                "squashfs: block_size {} must be a power of two between 4 KiB and 1 MiB",
                opts.block_size
            )));
        }
        let sb = Superblock {
            magic: SQUASHFS_MAGIC,
            inode_count: 0,
            mkfs_time: 0,
            block_size: opts.block_size,
            fragment_count: 0,
            compression: opts.compression,
            block_log: opts.block_size.trailing_zeros() as u16,
            flags: 0,
            id_count: 0,
            major: 4,
            minor: 0,
            root_inode: 0,
            bytes_used: 0,
            id_table_start: u64::MAX,
            xattr_id_table_start: u64::MAX,
            inode_table_start: u64::MAX,
            directory_table_start: u64::MAX,
            fragment_table_start: u64::MAX,
            export_table_start: u64::MAX,
        };
        Ok(Self {
            sb,
            id_table: RefCell::new(idtable::IdTable::new()),
            xattr_reader: RefCell::new(xattr::XattrReader::new()),
            write_state: Some(writer::WriteState::new(opts.block_size, opts.compression)),
        })
    }

    /// Buffer a new directory in the writer's in-memory tree. Parents
    /// are created implicitly with permissive defaults.
    pub fn create_dir(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &str,
        meta: EntryMeta,
        xattrs: Vec<Xattr>,
    ) -> Result<()> {
        let s = self
            .write_state
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: not in write mode".into()))?;
        s.create_dir(path, meta, xattrs)
    }

    /// Buffer a new regular file. Bytes are streamed from `src` to the
    /// data area immediately, so large [`FileSource`]s are never loaded
    /// entirely into memory nor spilled to a temp file.
    pub fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        src: FileSource,
        meta: EntryMeta,
        xattrs: Vec<Xattr>,
    ) -> Result<()> {
        let s = self
            .write_state
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: not in write mode".into()))?;
        s.create_file(dev, path, src, meta, xattrs)
    }

    /// Buffer a new symbolic link.
    pub fn create_symlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &str,
        target: &str,
        meta: EntryMeta,
        xattrs: Vec<Xattr>,
    ) -> Result<()> {
        let s = self
            .write_state
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: not in write mode".into()))?;
        s.create_symlink(path, target, meta, xattrs)
    }

    /// Add a second directory entry at `dst_path` aliasing the inode of
    /// `src_path`. Bumps the source inode's `link_count`. Rejected if the
    /// source resolves to a directory (POSIX-style — SquashFS doesn't
    /// support directory hard links).
    pub fn create_hardlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        src_path: &str,
        dst_path: &str,
    ) -> Result<()> {
        let s = self
            .write_state
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: not in write mode".into()))?;
        s.create_hardlink(src_path, dst_path)
    }

    /// Buffer a device node, FIFO, or Unix-domain socket at `path`. The
    /// `major`/`minor` pair is encoded with the Linux `MKDEV` formula
    /// (top 12 bits major, bottom 20 bits minor) into the device word that
    /// lives on disk for block / char inodes. FIFO / socket inodes have no
    /// device word — `major`/`minor` are accepted but ignored.
    #[allow(clippy::too_many_arguments)]
    pub fn create_device(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &str,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: EntryMeta,
        xattrs: Vec<Xattr>,
    ) -> Result<()> {
        let s = self
            .write_state
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: not in write mode".into()))?;
        s.create_device(path, kind, major, minor, meta, xattrs)
    }

    /// Lay every buffered entry out on the device. After flushing the
    /// writer is consumed; read APIs become available.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        let Some(mut state) = self.write_state.take() else {
            return Ok(());
        };
        let new_sb = state.flush(dev)?;
        self.sb = new_sb;
        self.id_table = RefCell::new(idtable::IdTable::new());
        self.xattr_reader = RefCell::new(xattr::XattrReader::new());
        Ok(())
    }

    /// Resolve a path to its inode metadata. uid/gid are looked up
    /// through the id table; uid_idx / gid_idx out of range fall back
    /// to 0.
    pub fn inode_meta(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<InodeMeta> {
        if self.write_state.is_some() {
            return Err(crate::Error::InvalidArgument(
                "squashfs: inode_meta() unavailable until flush()".into(),
            ));
        }
        let resolved = directory::resolve_path(
            dev,
            self.sb.inode_table_start,
            self.sb.directory_table_start,
            self.sb.compression,
            self.sb.root_inode,
            self.sb.block_size,
            path,
        )?;
        self.id_table.borrow_mut().ensure_loaded(
            dev,
            self.sb.id_table_start,
            self.sb.id_count,
            self.sb.compression,
        )?;
        let id_table = self.id_table.borrow();
        let hdr = resolved.header();
        let kind = resolved.entry_kind();
        let file_size = match &resolved {
            inode::Inode::File(f) => f.file_size,
            inode::Inode::Dir(d) => d.file_size as u64,
            inode::Inode::Symlink(s) => s.target.len() as u64,
            inode::Inode::Other { .. } => 0,
        };
        Ok(InodeMeta {
            inode_number: hdr.inode_number,
            kind,
            mode: hdr.permissions,
            uid: id_table.resolve(hdr.uid_idx),
            gid: id_table.resolve(hdr.gid_idx),
            mtime: hdr.mtime,
            file_size,
            xattr_index: resolved.xattr_index(),
        })
    }

    /// Read the extended attributes of the inode at `path`. Returns
    /// an empty vector when the inode has no xattrs or the image was
    /// built without an xattr table.
    pub fn read_xattrs(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<Vec<Xattr>> {
        if self.write_state.is_some() {
            return Err(crate::Error::InvalidArgument(
                "squashfs: read_xattrs() unavailable until flush()".into(),
            ));
        }
        let resolved = directory::resolve_path(
            dev,
            self.sb.inode_table_start,
            self.sb.directory_table_start,
            self.sb.compression,
            self.sb.root_inode,
            self.sb.block_size,
            path,
        )?;
        let idx = resolved.xattr_index();
        if idx == u32::MAX || self.sb.xattr_id_table_start == u64::MAX {
            return Ok(Vec::new());
        }
        self.xattr_reader.borrow_mut().ensure_loaded(
            dev,
            self.sb.xattr_id_table_start,
            self.sb.compression,
        )?;
        let reader = self.xattr_reader.borrow();
        reader.fetch(dev, idx, self.sb.compression)
    }

    pub fn total_bytes(&self) -> u64 {
        self.sb.bytes_used
    }

    pub fn block_size(&self) -> u32 {
        self.sb.block_size
    }

    pub fn compression(&self) -> Compression {
        self.sb.compression
    }

    pub fn superblock(&self) -> &Superblock {
        &self.sb
    }

    /// List a directory by absolute path. `/`, `""`, `.` all resolve to
    /// the root. Non-directory paths return [`crate::Error::InvalidArgument`].
    pub fn list_path(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<Vec<DirEntry>> {
        let resolved = directory::resolve_path(
            dev,
            self.sb.inode_table_start,
            self.sb.directory_table_start,
            self.sb.compression,
            self.sb.root_inode,
            self.sb.block_size,
            path,
        )?;
        let dir = match resolved {
            inode::Inode::Dir(d) => d,
            _ => {
                return Err(crate::Error::InvalidArgument(format!(
                    "squashfs: {path:?} is not a directory"
                )));
            }
        };
        let raw_entries = directory::read_directory_entries(
            dev,
            self.sb.directory_table_start,
            self.sb.compression,
            dir.block_index,
            dir.block_offset,
            dir.file_size,
        )?;
        Ok(raw_entries
            .into_iter()
            .map(|e| DirEntry {
                name: e.name,
                inode: e.inode_number,
                kind: directory::entry_kind_from_type(e.inode_type),
                // squashfs directory entries don't embed file size; it
                // lives on the inode block referenced by `inode_number`.
                size: 0,
            })
            .collect())
    }

    /// Open a streaming reader for the regular file at `path`. Returns
    /// [`crate::Error::InvalidArgument`] if `path` is a directory or
    /// missing, and [`crate::Error::Unsupported`] if the file's data is
    /// stored compressed.
    pub fn open_file_reader<'a>(
        &self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<Box<dyn Read + 'a>> {
        let resolved = directory::resolve_path(
            dev,
            self.sb.inode_table_start,
            self.sb.directory_table_start,
            self.sb.compression,
            self.sb.root_inode,
            self.sb.block_size,
            path,
        )?;
        let file_inode = match resolved {
            inode::Inode::File(f) => f,
            _ => {
                return Err(crate::Error::InvalidArgument(format!(
                    "squashfs: {path:?} is not a regular file"
                )));
            }
        };
        Ok(Box::new(FileReader::new(
            dev,
            &file_inode,
            self.sb.compression,
            self.sb.fragment_table_start,
            self.sb.fragment_count,
            self.sb.block_size,
        )))
    }

    /// Open a regular file for random-access reads — same backend as
    /// [`Self::open_file_reader`] but returning a `Read + Seek + len`
    /// handle that caches the last decompressed block (cheap
    /// re-reads after small seeks, single-block cost on a cross-block
    /// jump).
    pub fn open_file_read_handle<'a>(
        &self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<file::SquashfsFileReadHandle<'a>> {
        let resolved = directory::resolve_path(
            dev,
            self.sb.inode_table_start,
            self.sb.directory_table_start,
            self.sb.compression,
            self.sb.root_inode,
            self.sb.block_size,
            path,
        )?;
        let file_inode = match resolved {
            inode::Inode::File(f) => f,
            _ => {
                return Err(crate::Error::InvalidArgument(format!(
                    "squashfs: {path:?} is not a regular file"
                )));
            }
        };
        Ok(file::SquashfsFileReadHandle::new(
            dev,
            &file_inode,
            self.sb.compression,
            self.sb.fragment_table_start,
            self.sb.fragment_count,
            self.sb.block_size,
        ))
    }

    /// Read a symbolic link's target. The target lives inline in the
    /// inode, so no data-block decompression is involved.
    pub fn read_symlink(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<String> {
        let resolved = directory::resolve_path(
            dev,
            self.sb.inode_table_start,
            self.sb.directory_table_start,
            self.sb.compression,
            self.sb.root_inode,
            self.sb.block_size,
            path,
        )?;
        match resolved {
            inode::Inode::Symlink(s) => Ok(s.target),
            _ => Err(crate::Error::InvalidArgument(format!(
                "squashfs: {path:?} is not a symlink"
            ))),
        }
    }
}

/// Convert the public [`crate::fs::FileMeta`] to the squashfs-private
/// [`writer::EntryMeta`]. Drops `atime` / `ctime` since SquashFS only
/// stores `mtime`.
fn entry_meta_from(meta: crate::fs::FileMeta) -> writer::EntryMeta {
    writer::EntryMeta {
        mode: meta.mode,
        uid: meta.uid,
        gid: meta.gid,
        mtime: meta.mtime,
    }
}

// ----------------------------------------------------------------------
// `crate::fs::Filesystem` trait impl — bridges Squashfs into the
// generic walker. SquashFS is a write-once archive: mutators only work
// after `format()`, and `remove()` is unsupported.
// ----------------------------------------------------------------------

impl crate::fs::FilesystemFactory for Squashfs {
    type FormatOpts = FormatOpts;

    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }

    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl crate::fs::Filesystem for Squashfs {
    fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        src: crate::fs::FileSource,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: non-UTF-8 path".into()))?;
        self.create_file(dev, s, src, entry_meta_from(meta), Vec::new())
    }

    fn create_file_streaming(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        body: &mut dyn std::io::Read,
        len: u64,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        // SquashFS streams payloads to the data area as each file arrives,
        // so we consume `body` directly — no temp file, no in-RAM spool.
        // This bypasses the trait default entirely (so `streams_immediately`
        // is irrelevant here).
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: non-UTF-8 path".into()))?;
        let st = self
            .write_state
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: not in write mode".into()))?;
        st.create_file_streaming(dev, s, body, len, entry_meta_from(meta), Vec::new())
    }

    fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: non-UTF-8 path".into()))?;
        self.create_dir(dev, s, entry_meta_from(meta), Vec::new())
    }

    fn create_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        target: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: non-UTF-8 path".into()))?;
        let t = target.to_str().ok_or_else(|| {
            crate::Error::InvalidArgument("squashfs: non-UTF-8 symlink target".into())
        })?;
        self.create_symlink(dev, s, t, entry_meta_from(meta), Vec::new())
    }

    fn create_device(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        kind: crate::fs::DeviceKind,
        major: u32,
        minor: u32,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: non-UTF-8 path".into()))?;
        self.create_device(
            dev,
            s,
            kind,
            major,
            minor,
            entry_meta_from(meta),
            Vec::new(),
        )
    }

    fn remove(&mut self, _dev: &mut dyn BlockDevice, _path: &std::path::Path) -> Result<()> {
        Err(crate::Error::Immutable {
            kind: "squashfs",
            op: "rm",
        })
    }

    fn list(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: non-UTF-8 path".into()))?;
        self.list_path(dev, s)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn std::io::Read + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: non-UTF-8 path".into()))?;
        let r = self.open_file_reader(dev, s)?;
        Ok(Box::new(r))
    }

    fn getattr(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<crate::fs::FileAttrs> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: non-UTF-8 path".into()))?;
        let m = self.inode_meta(dev, s)?;
        // SquashFS stores only mtime (unix seconds); atime/ctime mirror it.
        Ok(crate::fs::FileAttrs {
            kind: m.kind,
            mode: m.mode & 0o7777,
            uid: m.uid,
            gid: m.gid,
            size: m.file_size,
            blocks: m.file_size.div_ceil(512),
            nlink: 1,
            atime: m.mtime,
            mtime: m.mtime,
            ctime: m.mtime,
            rdev: 0,
            inode: m.inode_number,
        })
    }

    fn list_xattrs(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::XattrPair>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: non-UTF-8 path".into()))?;
        Ok(self
            .read_xattrs(dev, s)?
            .into_iter()
            .map(|x| crate::fs::XattrPair {
                name: x.key,
                value: x.value,
            })
            .collect())
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn crate::fs::FileReadHandle + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: non-UTF-8 path".into()))?;
        let h = self.open_file_read_handle(dev, s)?;
        Ok(Box::new(h))
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        Self::flush(self, dev)
    }

    /// The superblock's `bytes_used` is the exact image length, so a
    /// repack can truncate the over-provisioned backing file to fit.
    fn image_len(&self) -> Option<u64> {
        (self.sb.bytes_used > 0).then_some(self.sb.bytes_used)
    }

    fn mutation_capability(&self) -> crate::fs::MutationCapability {
        crate::fs::MutationCapability::Immutable
    }

    fn read_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<std::path::PathBuf> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("squashfs: non-UTF-8 path".into()))?;
        Ok(std::path::PathBuf::from(Squashfs::read_symlink(
            self, dev, s,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;
    use crate::fs::EntryKind;

    /// Build a 96-byte uncompressed-v4 superblock with caller-controlled
    /// table offsets. `comp` is the compressor id (1=gzip, etc.).
    #[allow(clippy::too_many_arguments)]
    fn fake_sb_v4(
        comp: u16,
        block_size: u32,
        fragment_count: u32,
        root_inode: u64,
        bytes_used: u64,
        inode_table_start: u64,
        directory_table_start: u64,
        fragment_table_start: u64,
    ) -> Vec<u8> {
        let mut v = vec![0u8; 96];
        v[0..4].copy_from_slice(&SQUASHFS_MAGIC.to_le_bytes());
        v[4..8].copy_from_slice(&8u32.to_le_bytes());
        v[8..12].copy_from_slice(&0u32.to_le_bytes());
        v[12..16].copy_from_slice(&block_size.to_le_bytes());
        v[16..20].copy_from_slice(&fragment_count.to_le_bytes());
        v[20..22].copy_from_slice(&comp.to_le_bytes());
        let block_log = block_size.trailing_zeros() as u16;
        v[22..24].copy_from_slice(&block_log.to_le_bytes());
        v[24..26].copy_from_slice(&0u16.to_le_bytes());
        v[26..28].copy_from_slice(&0u16.to_le_bytes());
        v[28..30].copy_from_slice(&4u16.to_le_bytes());
        v[30..32].copy_from_slice(&0u16.to_le_bytes());
        v[32..40].copy_from_slice(&root_inode.to_le_bytes());
        v[40..48].copy_from_slice(&bytes_used.to_le_bytes());
        v[48..56].copy_from_slice(&u64::MAX.to_le_bytes()); // id_table_start
        v[56..64].copy_from_slice(&u64::MAX.to_le_bytes()); // xattr_id_table_start
        v[64..72].copy_from_slice(&inode_table_start.to_le_bytes());
        v[72..80].copy_from_slice(&directory_table_start.to_le_bytes());
        v[80..88].copy_from_slice(&fragment_table_start.to_le_bytes());
        v[88..96].copy_from_slice(&u64::MAX.to_le_bytes()); // export_table_start
        v
    }

    fn fake_sb(major: u16, comp: u16) -> Vec<u8> {
        let mut v = fake_sb_v4(comp, 131072, 0, 0, 512, u64::MAX, u64::MAX, u64::MAX);
        v[28..30].copy_from_slice(&major.to_le_bytes());
        v
    }

    #[test]
    fn decode_recognises_zstd() {
        let v = fake_sb(4, 6);
        let sb = Superblock::decode(&v).unwrap();
        assert_eq!(sb.compression, Compression::Zstd);
        assert_eq!(sb.block_size, 131072);
    }

    #[test]
    fn open_rejects_v3() {
        let mut dev = MemoryBackend::new(4096);
        dev.write_at(0, &fake_sb(3, 1)).unwrap();
        let err = Squashfs::open(&mut dev).unwrap_err();
        match err {
            crate::Error::Unsupported(_) => {}
            _ => panic!("expected Unsupported, got {err:?}"),
        }
    }

    #[test]
    fn open_accepts_v4() {
        let mut dev = MemoryBackend::new(4096);
        dev.write_at(0, &fake_sb(4, 6)).unwrap();
        let s = Squashfs::open(&mut dev).unwrap();
        assert_eq!(s.compression(), Compression::Zstd);
    }

    #[test]
    fn probe_matches_magic() {
        let mut dev = MemoryBackend::new(4096);
        dev.write_at(0, &SQUASHFS_MAGIC.to_le_bytes()).unwrap();
        assert!(probe(&mut dev).unwrap());
    }

    // ----- end-to-end fixture: hand-crafted uncompressed image -----------
    //
    // Layout we build for the integration test:
    //
    //   [0..96)        superblock
    //   [96..]         data blocks for "hi.txt" (5 bytes "hello")
    //   [..]           inode table (one metablock):
    //                    [0]  root dir inode  (BasicDir)  16+16 bytes
    //                    [32] regular file    (BasicFile) 16+16+0 bytes
    //                    [64] symlink inode   (BasicSymlink) 16+8+4 bytes
    //   [..]           directory table (one metablock):
    //                    header + 2 entries (hi.txt, lnk)
    //   [..]           — no fragment table —
    //
    // All metablocks are uncompressed (high bit set).

    use super::metablock::encode_uncompressed;

    struct Built {
        image: Vec<u8>,
        root_inode_ref: u64,
        inode_table_start: u64,
        directory_table_start: u64,
        data_offset: u64,
    }

    fn build_fixture() -> Built {
        // ----- Data block for "hi.txt" -----
        let file_payload = b"hello";
        let data_offset = 96u64;
        let data_block_size = file_payload.len() as u32 | 0x0100_0000; // uncompressed

        // ----- Inode table metablock contents (uncompressed payload) -----
        let mut inodes: Vec<u8> = Vec::new();

        // Root directory (BasicDir): offsets [0..32) within the metablock.
        // header (16 bytes):
        inodes.extend_from_slice(&1u16.to_le_bytes()); // type = BasicDir
        inodes.extend_from_slice(&0o755u16.to_le_bytes()); // perms
        inodes.extend_from_slice(&0u16.to_le_bytes()); // uid_idx
        inodes.extend_from_slice(&0u16.to_le_bytes()); // gid_idx
        inodes.extend_from_slice(&0u32.to_le_bytes()); // mtime
        inodes.extend_from_slice(&1u32.to_le_bytes()); // inode_number
        // basic_dir payload (16 bytes):
        inodes.extend_from_slice(&0u32.to_le_bytes()); // block_index = 0
        inodes.extend_from_slice(&3u32.to_le_bytes()); // link_count
        // file_size: stored size+3. Real listing size to be filled in below
        // — patch later once we know it; placeholder for now.
        let dir_size_patch_offset = inodes.len();
        inodes.extend_from_slice(&0u16.to_le_bytes()); // file_size placeholder
        inodes.extend_from_slice(&0u16.to_le_bytes()); // block_offset = 0
        inodes.extend_from_slice(&0u32.to_le_bytes()); // parent_inode

        // BasicFile inode (16 + 16 + 0 bytes), starting at offset 32.
        let file_inode_offset = inodes.len() as u16;
        inodes.extend_from_slice(&2u16.to_le_bytes()); // type = BasicFile
        inodes.extend_from_slice(&0o644u16.to_le_bytes());
        inodes.extend_from_slice(&0u16.to_le_bytes());
        inodes.extend_from_slice(&0u16.to_le_bytes());
        inodes.extend_from_slice(&0u32.to_le_bytes()); // mtime
        inodes.extend_from_slice(&2u32.to_le_bytes()); // inode_number
        // BasicFile payload:
        inodes.extend_from_slice(&(data_offset as u32).to_le_bytes()); // blocks_start
        inodes.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // frag_index = none
        inodes.extend_from_slice(&0u32.to_le_bytes()); // frag_offset
        inodes.extend_from_slice(&(file_payload.len() as u32).to_le_bytes()); // file_size
        // 1 block size word:
        inodes.extend_from_slice(&data_block_size.to_le_bytes());

        // BasicSymlink inode at offset = current length.
        let symlink_inode_offset = inodes.len() as u16;
        inodes.extend_from_slice(&3u16.to_le_bytes()); // type = BasicSymlink
        inodes.extend_from_slice(&0o777u16.to_le_bytes());
        inodes.extend_from_slice(&0u16.to_le_bytes());
        inodes.extend_from_slice(&0u16.to_le_bytes());
        inodes.extend_from_slice(&0u32.to_le_bytes());
        inodes.extend_from_slice(&3u32.to_le_bytes()); // inode_number
        // symlink payload:
        let target = b"hi.txt";
        inodes.extend_from_slice(&1u32.to_le_bytes()); // link_count
        inodes.extend_from_slice(&(target.len() as u32).to_le_bytes()); // target_size
        inodes.extend_from_slice(target);

        // ----- Directory listing payload -----
        // One header + two entries (hi.txt @ file_inode_offset, lnk @ symlink_inode_offset).
        let mut dirs: Vec<u8> = Vec::new();
        // header (12 bytes):
        dirs.extend_from_slice(&1u32.to_le_bytes()); // count = entries-1 = 1
        dirs.extend_from_slice(&0u32.to_le_bytes()); // start_block (rel to inode table)
        dirs.extend_from_slice(&2u32.to_le_bytes()); // inode_number base = 2 (file is 2)
        // entry hi.txt:
        dirs.extend_from_slice(&file_inode_offset.to_le_bytes());
        dirs.extend_from_slice(&0i16.to_le_bytes()); // inode_offset = 0 -> inode_number 2
        dirs.extend_from_slice(&2u16.to_le_bytes()); // type = BasicFile
        dirs.extend_from_slice(&((b"hi.txt".len() - 1) as u16).to_le_bytes()); // name_size (off by one)
        dirs.extend_from_slice(b"hi.txt");
        // entry lnk:
        dirs.extend_from_slice(&symlink_inode_offset.to_le_bytes());
        dirs.extend_from_slice(&1i16.to_le_bytes()); // base+1 = 3
        dirs.extend_from_slice(&3u16.to_le_bytes()); // BasicSymlink
        dirs.extend_from_slice(&((b"lnk".len() - 1) as u16).to_le_bytes());
        dirs.extend_from_slice(b"lnk");

        // Patch the root dir's file_size now that we know the listing size.
        let dir_size_real = dirs.len() as u16 + 3; // stored as size+3
        let patch = dir_size_real.to_le_bytes();
        inodes[dir_size_patch_offset..dir_size_patch_offset + 2].copy_from_slice(&patch);

        // ----- Stitch it together -----
        let mut image = vec![0u8; data_offset as usize + file_payload.len()];
        image[data_offset as usize..data_offset as usize + file_payload.len()]
            .copy_from_slice(file_payload);

        let inode_table_start = image.len() as u64;
        image.extend_from_slice(&encode_uncompressed(&inodes));
        let directory_table_start = image.len() as u64;
        image.extend_from_slice(&encode_uncompressed(&dirs));

        // Root inode reference: block 0 (offset within inode table), offset 0.
        let root_inode_ref: u64 = 0;

        let bytes_used = image.len() as u64;
        let mut sb = fake_sb_v4(
            1, // gzip; doesn't matter — payload is uncompressed
            4096,
            0,
            root_inode_ref,
            bytes_used,
            inode_table_start,
            directory_table_start,
            u64::MAX,
        );
        // Splice superblock into the head.
        image[..96].copy_from_slice(&sb[..]);
        // (also keep the un-spliced `sb` around for type, but discard).
        let _ = &mut sb;
        Built {
            image,
            root_inode_ref,
            inode_table_start,
            directory_table_start,
            data_offset,
        }
    }

    #[test]
    fn end_to_end_list_read_symlink() {
        let built = build_fixture();
        assert_eq!(built.root_inode_ref, 0);
        assert!(built.inode_table_start > 0);
        assert!(built.directory_table_start > built.inode_table_start);
        assert!(built.data_offset < built.inode_table_start);

        let mut dev = MemoryBackend::new(built.image.len() as u64 + 64);
        dev.write_at(0, &built.image).unwrap();
        let s = Squashfs::open(&mut dev).unwrap();

        // List root.
        let entries = s.list_path(&mut dev, "/").unwrap();
        assert_eq!(entries.len(), 2);
        let by_name: std::collections::HashMap<_, _> = entries
            .iter()
            .map(|e| (e.name.as_str(), (e.inode, e.kind)))
            .collect();
        assert_eq!(by_name["hi.txt"].1, EntryKind::Regular);
        assert_eq!(by_name["lnk"].1, EntryKind::Symlink);
        assert_eq!(by_name["hi.txt"].0, 2);
        assert_eq!(by_name["lnk"].0, 3);

        // Read the file by path.
        let mut r = s.open_file_reader(&mut dev, "/hi.txt").unwrap();
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut out).unwrap();
        drop(r);
        assert_eq!(out, b"hello");

        // Read the symlink target.
        let tgt = s.read_symlink(&mut dev, "/lnk").unwrap();
        assert_eq!(tgt, "hi.txt");
    }

    #[test]
    fn list_path_on_missing_entry_errors() {
        let built = build_fixture();
        let mut dev = MemoryBackend::new(built.image.len() as u64 + 64);
        dev.write_at(0, &built.image).unwrap();
        let s = Squashfs::open(&mut dev).unwrap();
        let err = s.list_path(&mut dev, "/nope").unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    #[test]
    fn compressed_data_block_surfaces_unsupported() {
        // Build a fixture but mark the file's data block as compressed
        // (clear bit 24). Behaviour depends on build features:
        // - With `gzip` enabled (default), the decompressor attempts to
        //   inflate the raw "hello" bytes and fails with a gzip-decode
        //   error (the bytes aren't valid gzip).
        // - With `gzip` disabled, we get an `Unsupported` error from the
        //   compression module saying the feature is off.
        // Either way the message mentions "gzip".
        let mut built = build_fixture();
        // Open + intercept: re-decode superblock, walk to file inode, patch
        // its block_size word. Easier: locate the bytes "hello" data was
        // at, and patch the metablock containing the file inode directly.
        //
        // The file's block size word sits inside the inode metablock at
        // a known offset. Rebuild that block deterministically:
        //   - dir inode at offset 0..32
        //   - file inode at offset 32..64 (+ 4-byte block size at 64..68)
        //   - block size word is at byte 64 of the payload, i.e.
        //     metablock_header(2) + 64 = 66 inside the *metablock on disk*.
        let sb_buf = &built.image[0..96];
        let sb = Superblock::decode(sb_buf).unwrap();
        let off = sb.inode_table_start as usize + 2 + 64; // header + 64
        // Clear the uncompressed bit (bit 24).
        let mut word_bytes = [0u8; 4];
        word_bytes.copy_from_slice(&built.image[off..off + 4]);
        let mut word = u32::from_le_bytes(word_bytes);
        word &= !0x0100_0000; // clear uncompressed bit
        built.image[off..off + 4].copy_from_slice(&word.to_le_bytes());

        let mut dev = MemoryBackend::new(built.image.len() as u64 + 64);
        dev.write_at(0, &built.image).unwrap();
        let s = Squashfs::open(&mut dev).unwrap();
        let mut r = s.open_file_reader(&mut dev, "/hi.txt").unwrap();
        let mut sink = Vec::new();
        let res = std::io::Read::read_to_end(&mut r, &mut sink);
        let err = res.unwrap_err();
        let msg = format!("{err}");
        // Either "gzip" (codec disabled) or "zlib" (codec enabled and
        // tries to decode the garbage we wrote — SquashFS calls this
        // compressor "gzip" but the on-wire framing is zlib).
        assert!(
            msg.contains("zlib") || msg.contains("gzip"),
            "unexpected message: {msg}"
        );
    }

    /// End-to-end writer test: format an image into a memory backend,
    /// add a directory, file, and symlink with non-zero uid/gid and an
    /// xattr, flush, then re-open and validate the read path covers id
    /// resolution and xattr fetch.
    #[test]
    fn writer_round_trip_with_id_and_xattr() {
        let mut dev = crate::block::MemoryBackend::new(2 * 1024 * 1024);
        // Use Unknown(0) → uncompressed metablocks, so this test stays
        // codec-agnostic. (Codec round-trips happen in their own modules.)
        let mut s = Squashfs::format(
            &mut dev,
            &FormatOpts {
                block_size: 4096,
                compression: Compression::Unknown(0),
            },
        )
        .unwrap();
        s.create_dir(
            &mut dev,
            "/etc",
            EntryMeta {
                mode: 0o755,
                uid: 0,
                gid: 0,
                mtime: 100,
            },
            Vec::new(),
        )
        .unwrap();
        s.create_file(
            &mut dev,
            "/etc/greeting",
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"hi there\n".to_vec())),
                len: 9,
            },
            EntryMeta {
                mode: 0o644,
                uid: 1234,
                gid: 5678,
                mtime: 200,
            },
            vec![
                Xattr {
                    key: "user.color".into(),
                    value: b"orange".to_vec(),
                },
                Xattr {
                    key: "security.selinux".into(),
                    value: b"unconfined_u:object_r:default_t:s0".to_vec(),
                },
            ],
        )
        .unwrap();
        s.create_symlink(
            &mut dev,
            "/lnk",
            "etc/greeting",
            EntryMeta {
                mode: 0o777,
                uid: 0,
                gid: 0,
                mtime: 0,
            },
            Vec::new(),
        )
        .unwrap();
        s.flush(&mut dev).unwrap();

        // Re-open from the same device.
        let s = Squashfs::open(&mut dev).unwrap();
        let root = s.list_path(&mut dev, "/").unwrap();
        let names: Vec<&str> = root.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"etc"));
        assert!(names.contains(&"lnk"));

        // File contents.
        let mut r = s.open_file_reader(&mut dev, "/etc/greeting").unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut buf).unwrap();
        drop(r);
        assert_eq!(buf, b"hi there\n");

        // Symlink target.
        let tgt = s.read_symlink(&mut dev, "/lnk").unwrap();
        assert_eq!(tgt, "etc/greeting");

        // Id resolution.
        let meta = s.inode_meta(&mut dev, "/etc/greeting").unwrap();
        assert_eq!(meta.uid, 1234);
        assert_eq!(meta.gid, 5678);
        assert_eq!(meta.mode, 0o644);
        assert_eq!(meta.mtime, 200);
        assert_eq!(meta.file_size, 9);

        // Xattrs.
        let xs = s.read_xattrs(&mut dev, "/etc/greeting").unwrap();
        assert_eq!(xs.len(), 2);
        let mut by_key: std::collections::HashMap<_, _> =
            xs.into_iter().map(|x| (x.key, x.value)).collect();
        assert_eq!(by_key.remove("user.color").unwrap(), b"orange");
        assert!(
            by_key
                .remove("security.selinux")
                .unwrap()
                .starts_with(b"unconfined_u:object_r:")
        );

        // An entry without xattrs returns an empty vec.
        let xs2 = s.read_xattrs(&mut dev, "/lnk").unwrap();
        assert_eq!(xs2.len(), 0);
    }

    /// Full-coverage round-trip: build an image containing a file, a
    /// nested directory, a symlink, a hardlink, every device-node kind,
    /// and xattrs on a few entries. Flush, re-open via [`Squashfs::open`],
    /// then validate every entry via [`Squashfs::list_path`],
    /// [`Squashfs::read_xattrs`], [`Squashfs::inode_meta`],
    /// [`Squashfs::open_file_reader`], and [`Squashfs::read_symlink`].
    #[test]
    fn cross_validation_round_trip_full() {
        let mut dev = crate::block::MemoryBackend::new(2 * 1024 * 1024);
        let mut s = Squashfs::format(
            &mut dev,
            &FormatOpts {
                block_size: 4096,
                compression: Compression::Unknown(0),
            },
        )
        .unwrap();

        // /etc dir, /etc/hosts file, /etc/link (hardlink to hosts).
        s.create_dir(
            &mut dev,
            "/etc",
            EntryMeta {
                mode: 0o755,
                uid: 0,
                gid: 0,
                mtime: 100,
            },
            Vec::new(),
        )
        .unwrap();
        s.create_file(
            &mut dev,
            "/etc/hosts",
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"127.0.0.1 localhost\n".to_vec())),
                len: 20,
            },
            EntryMeta {
                mode: 0o644,
                uid: 7,
                gid: 8,
                mtime: 200,
            },
            vec![Xattr {
                key: "user.kind".into(),
                value: b"file".to_vec(),
            }],
        )
        .unwrap();
        s.create_hardlink(&mut dev, "/etc/hosts", "/etc/link")
            .unwrap();

        // /sym -> etc/hosts symlink.
        s.create_symlink(
            &mut dev,
            "/sym",
            "etc/hosts",
            EntryMeta {
                mode: 0o777,
                uid: 0,
                gid: 0,
                mtime: 300,
            },
            Vec::new(),
        )
        .unwrap();

        // Device nodes — one of each.
        s.create_device(
            &mut dev,
            "/dev/null",
            crate::fs::DeviceKind::Char,
            1,
            3,
            EntryMeta {
                mode: 0o666,
                uid: 0,
                gid: 0,
                mtime: 400,
            },
            Vec::new(),
        )
        .unwrap();
        s.create_device(
            &mut dev,
            "/dev/sda",
            crate::fs::DeviceKind::Block,
            8,
            0,
            EntryMeta {
                mode: 0o600,
                uid: 0,
                gid: 0,
                mtime: 500,
            },
            Vec::new(),
        )
        .unwrap();
        s.create_device(
            &mut dev,
            "/run/fifo",
            crate::fs::DeviceKind::Fifo,
            0,
            0,
            EntryMeta {
                mode: 0o600,
                uid: 0,
                gid: 0,
                mtime: 600,
            },
            Vec::new(),
        )
        .unwrap();
        s.create_device(
            &mut dev,
            "/run/sock",
            crate::fs::DeviceKind::Socket,
            0,
            0,
            EntryMeta {
                mode: 0o600,
                uid: 0,
                gid: 0,
                mtime: 700,
            },
            vec![Xattr {
                key: "security.selinux".into(),
                value: b"system_u:object_r:tmp_t:s0".to_vec(),
            }],
        )
        .unwrap();

        s.flush(&mut dev).unwrap();

        // Re-open and inspect.
        let s = Squashfs::open(&mut dev).unwrap();

        // Root listing.
        let mut root_names: Vec<String> = s
            .list_path(&mut dev, "/")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        root_names.sort();
        assert_eq!(root_names, vec!["dev", "etc", "run", "sym"]);

        // /etc listing contains both hosts and link, link points at hosts' inode.
        let etc = s.list_path(&mut dev, "/etc").unwrap();
        let by_name: std::collections::HashMap<_, _> =
            etc.iter().map(|e| (e.name.clone(), e.clone())).collect();
        assert!(by_name.contains_key("hosts"));
        assert!(by_name.contains_key("link"));
        assert_eq!(by_name["hosts"].inode, by_name["link"].inode);
        assert_eq!(by_name["hosts"].kind, crate::fs::EntryKind::Regular);
        assert_eq!(by_name["link"].kind, crate::fs::EntryKind::Regular);

        // File contents readable via both names.
        for path in ["/etc/hosts", "/etc/link"] {
            let mut r = s.open_file_reader(&mut dev, path).unwrap();
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut r, &mut buf).unwrap();
            drop(r);
            assert_eq!(buf, b"127.0.0.1 localhost\n", "via {path}");
        }

        // Symlink target.
        let tgt = s.read_symlink(&mut dev, "/sym").unwrap();
        assert_eq!(tgt, "etc/hosts");

        // Device-node listings.
        let dev_dir = s.list_path(&mut dev, "/dev").unwrap();
        let dev_by: std::collections::HashMap<_, _> =
            dev_dir.iter().map(|e| (e.name.clone(), e.kind)).collect();
        assert_eq!(dev_by["null"], crate::fs::EntryKind::Char);
        assert_eq!(dev_by["sda"], crate::fs::EntryKind::Block);
        let run_dir = s.list_path(&mut dev, "/run").unwrap();
        let run_by: std::collections::HashMap<_, _> =
            run_dir.iter().map(|e| (e.name.clone(), e.kind)).collect();
        assert_eq!(run_by["fifo"], crate::fs::EntryKind::Fifo);
        assert_eq!(run_by["sock"], crate::fs::EntryKind::Socket);

        // Inode metadata.
        let m = s.inode_meta(&mut dev, "/etc/hosts").unwrap();
        assert_eq!(m.uid, 7);
        assert_eq!(m.gid, 8);
        assert_eq!(m.mtime, 200);
        assert_eq!(m.file_size, 20);
        let m_null = s.inode_meta(&mut dev, "/dev/null").unwrap();
        assert_eq!(m_null.kind, crate::fs::EntryKind::Char);
        assert_eq!(m_null.mode, 0o666);

        // Xattrs on the file.
        let xs = s.read_xattrs(&mut dev, "/etc/hosts").unwrap();
        assert_eq!(xs.len(), 1);
        assert_eq!(xs[0].key, "user.kind");
        assert_eq!(xs[0].value, b"file");
        // Xattrs on the socket.
        let xs = s.read_xattrs(&mut dev, "/run/sock").unwrap();
        assert_eq!(xs.len(), 1);
        assert_eq!(xs[0].key, "security.selinux");
    }

    /// Multi-fragment-block spill: build a tree with many small files
    /// whose tails collectively exceed one block_size's worth of fragment
    /// data, forcing the writer to emit multiple fragment-table entries.
    /// The reader fetches the right entry per file.
    #[test]
    fn writer_spills_to_multiple_fragment_blocks() {
        let mut dev = crate::block::MemoryBackend::new(8 * 1024 * 1024);
        // 4 KiB block; each file is ~1500 bytes, so 3 files saturate one
        // fragment block. Five files = at least two fragment blocks.
        let mut s = Squashfs::format(
            &mut dev,
            &FormatOpts {
                block_size: 4096,
                compression: Compression::Unknown(0),
            },
        )
        .unwrap();
        let payloads: Vec<Vec<u8>> = (0..5)
            .map(|i| (0..1500).map(|j| ((i * 1500 + j) % 251) as u8).collect())
            .collect();
        for (i, p) in payloads.iter().enumerate() {
            s.create_file(
                &mut dev,
                &format!("/f{i}.bin"),
                FileSource::Reader {
                    reader: Box::new(std::io::Cursor::new(p.clone())),
                    len: p.len() as u64,
                },
                EntryMeta::default(),
                Vec::new(),
            )
            .unwrap();
        }
        s.flush(&mut dev).unwrap();
        // The superblock should advertise >1 fragment.
        let s = Squashfs::open(&mut dev).unwrap();
        assert!(
            s.superblock().fragment_count >= 2,
            "expected multi-fragment image, got {}",
            s.superblock().fragment_count
        );
        // Each file reads back exactly.
        for (i, p) in payloads.iter().enumerate() {
            let mut r = s.open_file_reader(&mut dev, &format!("/f{i}.bin")).unwrap();
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut r, &mut buf).unwrap();
            drop(r);
            assert_eq!(&buf, p, "file f{i}.bin contents differ");
        }
    }

    /// Large directory spill: build a directory whose listing exceeds the
    /// basic-dir's 65532-byte limit, forcing the writer to promote it to
    /// the extended-dir inode form. The reader handles both, so list_path
    /// should still return the full set.
    #[test]
    fn writer_spills_to_extended_dir_inode() {
        let mut dev = crate::block::MemoryBackend::new(8 * 1024 * 1024);
        let mut s = Squashfs::format(
            &mut dev,
            &FormatOpts {
                block_size: 4096,
                compression: Compression::Unknown(0),
            },
        )
        .unwrap();
        // 4000 entries, each named ~16 bytes — upper-bound estimate
        // exceeds 65532, forcing ext-dir form.
        for i in 0..4000 {
            s.create_symlink(
                &mut dev,
                &format!("/big/sym_{i:08}"),
                "target",
                EntryMeta::default(),
                Vec::new(),
            )
            .unwrap();
        }
        s.flush(&mut dev).unwrap();
        let s = Squashfs::open(&mut dev).unwrap();
        let listing = s.list_path(&mut dev, "/big").unwrap();
        assert_eq!(listing.len(), 4000);
        // Spot-check one to confirm the entry kind decoded correctly.
        assert!(listing.iter().any(|e| e.name == "sym_00000000"));
        assert!(listing.iter().any(|e| e.name == "sym_00003999"));
    }

    /// `open_file_ro` returns a Read+Seek+len handle backed by the
    /// same block walker as `open_file_reader`, but with a single
    /// decompressed-block cache so backward seeks within a block
    /// reuse the work. The seek+read round-trip lands exact bytes.
    #[test]
    fn open_file_ro_random_seek_round_trip() {
        use crate::fs::Filesystem;
        use std::io::{Read, Seek, SeekFrom};
        let body: Vec<u8> = (0..4096u32).map(|i| (i & 0xff) as u8).collect();
        let mut dev = MemoryBackend::new(2 * 1024 * 1024);
        let mut s = Squashfs::format(&mut dev, &FormatOpts::default()).unwrap();
        s.create_file(
            &mut dev,
            "/data.bin",
            crate::fs::FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(body.clone())),
                len: body.len() as u64,
            },
            EntryMeta::default(),
            Vec::new(),
        )
        .unwrap();
        s.flush(&mut dev).unwrap();

        let mut s = Squashfs::open(&mut dev).unwrap();
        let mut h = s
            .open_file_ro(&mut dev, std::path::Path::new("/data.bin"))
            .unwrap();
        assert_eq!(h.len(), body.len() as u64);
        // Read from start.
        let mut chunk = [0u8; 64];
        h.read_exact(&mut chunk).unwrap();
        assert_eq!(&chunk[..], &body[..64]);
        // Seek mid-file, read.
        h.seek(SeekFrom::Start(1000)).unwrap();
        let mut chunk = [0u8; 32];
        h.read_exact(&mut chunk).unwrap();
        assert_eq!(&chunk[..], &body[1000..1032]);
        // Backward seek + reread (exercises cache).
        h.seek(SeekFrom::Current(-32)).unwrap();
        h.read_exact(&mut chunk).unwrap();
        assert_eq!(&chunk[..], &body[1000..1032]);
        // Seek past end caps at len.
        let where_ = h.seek(SeekFrom::End(100)).unwrap();
        assert_eq!(where_, body.len() as u64);
        let n = h.read(&mut chunk).unwrap();
        assert_eq!(n, 0);
    }
}
