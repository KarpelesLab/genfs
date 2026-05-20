//! ISO 9660 writer — mkisofs-style two-pass builder.
//!
//! The writer buffers create_*-call entries in memory, then on `flush()`
//! walks the tree twice:
//!
//! 1. **Pass 1** — assign LBAs. Counts sectors needed for PVD, optional
//!    Joliet SVD, optional boot record, VDST, path tables (L + M, plus
//!    Joliet equivalents), directory records, and file data. Each
//!    directory and each file gets a starting LBA + sector count.
//!
//! 2. **Pass 2** — write everything out. Order:
//!    - system area (LBAs 0..15) zero-filled
//!    - PVD at LBA 16
//!    - Boot Record at LBA 17 (if El Torito set)
//!    - Joliet SVD at LBA 18 (if joliet enabled)
//!    - VDST terminator
//!    - L-path table + M-path table (PVD names)
//!    - L-path table + M-path table (Joliet names)
//!    - PVD directory records (each dir's stream of records)
//!    - Joliet SVD directory records
//!    - Boot image data
//!    - File data
//!
//! Streaming invariant: file payloads come in as [`FileSource`] and are
//! pumped through a 64 KiB scratch buffer during pass 2. The writer
//! never loads a file fully into memory.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::{DeviceKind, FileMeta, FileSource};

use super::el_torito::BootCatalog;

/// Format options. `volume_id` is required; everything else has sane
/// defaults.
#[derive(Debug, Clone)]
pub struct FormatOpts {
    pub volume_id: String,
    pub publisher_id: String,
    pub data_preparer_id: String,
    pub application_id: String,
    pub joliet: bool,
    pub rock_ridge: bool,
    pub el_torito: Option<BootCatalog>,
    /// UNIX timestamp used for all metadata timestamps.
    pub create_date: u32,
}

impl Default for FormatOpts {
    fn default() -> Self {
        Self {
            volume_id: "CDROM".into(),
            publisher_id: String::new(),
            data_preparer_id: String::new(),
            application_id: String::new(),
            joliet: true,
            rock_ridge: true,
            el_torito: None,
            create_date: 0,
        }
    }
}

/// One in-memory entry the writer is buffering. Kept private — the
/// public API speaks through [`crate::fs::Filesystem`].
#[allow(dead_code)]
enum PendingEntry {
    File {
        meta: FileMeta,
        src: Option<FileSource>,
        size: u64,
    },
    Dir {
        meta: FileMeta,
    },
    Symlink {
        meta: FileMeta,
        target: PathBuf,
    },
    Device {
        meta: FileMeta,
        kind: DeviceKind,
        major: u32,
        minor: u32,
    },
}

/// Two-pass ISO 9660 writer. Construct with [`Iso9660Writer::new`],
/// stream entries via the Filesystem trait, then call `flush` (via the
/// trait) to lay out the image.
pub struct Iso9660Writer {
    #[allow(dead_code)]
    opts: FormatOpts,
    /// Tree of entries keyed by normalized path (no trailing slash,
    /// always starts with "/").
    entries: BTreeMap<PathBuf, PendingEntry>,
    /// Set once `flush` has run successfully. Subsequent flushes are
    /// no-ops (writers are one-shot — `repack` always builds a fresh
    /// device).
    flushed: bool,
}

impl Iso9660Writer {
    /// Build a fresh writer. The root directory is implicit; the caller
    /// only inserts child paths.
    pub fn new(opts: FormatOpts) -> Self {
        Self {
            opts,
            entries: BTreeMap::new(),
            flushed: false,
        }
    }

    /// Buffer a regular file. The payload is held as a [`FileSource`]
    /// reference; bytes only flow to disk during `flush`.
    pub fn add_file(&mut self, path: &Path, src: FileSource, meta: FileMeta) -> Result<()> {
        let path = normalize(path)?;
        // Pre-compute size — pass 1 of `flush` needs it to lay out the
        // data area before pass 2 streams the bytes through.
        let size = src.len()?;
        self.entries.insert(
            path,
            PendingEntry::File {
                meta,
                src: Some(src),
                size,
            },
        );
        Ok(())
    }

    /// Buffer a directory.
    pub fn add_dir(&mut self, path: &Path, meta: FileMeta) -> Result<()> {
        let path = normalize(path)?;
        self.entries.insert(path, PendingEntry::Dir { meta });
        Ok(())
    }

    /// Buffer a symbolic link. Rock Ridge `SL` records the target.
    pub fn add_symlink(&mut self, path: &Path, target: &Path, meta: FileMeta) -> Result<()> {
        let path = normalize(path)?;
        self.entries.insert(
            path,
            PendingEntry::Symlink {
                meta,
                target: target.to_path_buf(),
            },
        );
        Ok(())
    }

    /// Buffer a device / FIFO / socket node. Encoded via Rock Ridge
    /// `PN` (POSIX device number) on the ISO side.
    pub fn add_device(
        &mut self,
        path: &Path,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: FileMeta,
    ) -> Result<()> {
        let path = normalize(path)?;
        self.entries.insert(
            path,
            PendingEntry::Device {
                meta,
                kind,
                major,
                minor,
            },
        );
        Ok(())
    }

    /// Remove a buffered entry (so `repack` can satisfy the
    /// `Filesystem::remove` contract during builds).
    pub fn remove_entry(&mut self, path: &Path) -> Result<()> {
        let path = normalize(path)?;
        if self.entries.remove(&path).is_none() {
            return Err(crate::Error::InvalidArgument(format!(
                "iso9660: no buffered entry at {}",
                path.display()
            )));
        }
        Ok(())
    }

    /// Write the buffered tree to `dev`. Idempotent: subsequent calls
    /// return without re-writing.
    pub fn flush(&mut self, _dev: &mut dyn BlockDevice) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        // TODO: full two-pass write is not yet implemented. For now we
        // return Unsupported so repack-to-ISO fails cleanly with a
        // useful message instead of producing a broken image.
        Err(crate::Error::Unsupported(
            "iso9660: writer flush not yet implemented; reader-only for v0".into(),
        ))
    }

    /// Total number of buffered entries (excluding root) — exposed for
    /// tests / `info`.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

fn normalize(path: &Path) -> Result<PathBuf> {
    let s = path
        .to_str()
        .ok_or_else(|| crate::Error::InvalidArgument("iso9660: non-UTF-8 path".into()))?;
    let trimmed = s.trim_end_matches('/');
    if trimmed.is_empty() || trimmed == "/" {
        return Err(crate::Error::InvalidArgument(
            "iso9660: cannot create root explicitly".into(),
        ));
    }
    if !trimmed.starts_with('/') {
        return Err(crate::Error::InvalidArgument(format!(
            "iso9660: path must be absolute: {trimmed}"
        )));
    }
    Ok(PathBuf::from(trimmed))
}
