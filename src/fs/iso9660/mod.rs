//! ISO 9660 — read-only support for optical-media images.
//!
//! ## Status
//!
//! Reader only on this trait surface; writing happens by repack through
//! [`writer::Iso9660Writer`] which builds a fresh image from any
//! [`Filesystem`](crate::fs::Filesystem) source.
//!
//! Supported on the read path:
//!
//! - **ISO 9660 base** per ECMA-119: Primary Volume Descriptor + path
//!   table + directory records + file extents.
//! - **Joliet** (Microsoft TR-IMSA-1999) for long Unicode names, via the
//!   Supplementary Volume Descriptor with one of escape sequences
//!   `%/@`, `%/C`, `%/E`.
//! - **Rock Ridge** (IEEE P1282) for POSIX attributes: System Use Sharing
//!   Protocol header (SP), PX (uid/gid/mode/nlink), NM (long name),
//!   SL (symlink target), TF (timestamps), and CE (continuation area).
//! - **El Torito Bootable CD-ROM Format** v1: boot record at LBA 17,
//!   validation entry, default entry, optional section headers. The
//!   catalog is exposed for `info` and preserved by `repack`.
//!
//! ## Layout reference
//!
//! ```text
//! LBA 0..15     System area (zero on most media)
//! LBA 16        Primary Volume Descriptor (type 1, magic "CD001")
//! LBA 17        Boot Record (type 0)         — only if El Torito present
//! LBA 17..N     Supplementary Volume Descriptor(s) (type 2)
//! LBA next      Volume Descriptor Set Terminator (type 255)
//! ...           Path tables, directory records, file data
//! ```
//!
//! Each LBA is 2048 bytes ("logical sector size" in ECMA-119).
//!
//! Streaming invariant: file payloads are returned through a
//! [`Read`]-implementing handle that reads from the underlying
//! [`BlockDevice`] in 64 KiB chunks. The reader never loads a whole
//! file into memory.

use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::DirEntry;

mod directory;
mod el_torito;
mod joliet;
mod rock_ridge;
mod vd;
pub mod writer;

pub use directory::{DirEntryRaw, DirRecord};
pub use el_torito::{BootCatalog, BootEntry};
pub use vd::{PrimaryVolumeDescriptor, SupplementaryVolumeDescriptor, VolumeDescriptor};
pub use writer::{FormatOpts, Iso9660Writer};

/// Logical sector size for ISO 9660 / ECMA-119. Fixed at 2 KiB.
pub const SECTOR_SIZE: u32 = 2048;

/// LBA of the Primary Volume Descriptor.
pub const PVD_LBA: u32 = 16;

/// ISO 9660 magic bytes — the standard identifier inside every volume
/// descriptor at byte offset 1. ECMA-119 §8.1.2.
pub const ISO_MAGIC: &[u8; 5] = b"CD001";

/// Probe `dev`: return `Ok(())` if the first volume descriptor at LBA 16
/// carries the ISO 9660 magic. Used by [`crate::inspect::detect_fs`].
pub fn probe(dev: &mut dyn BlockDevice) -> bool {
    let mut buf = [0u8; 7];
    if dev
        .read_at(u64::from(PVD_LBA) * u64::from(SECTOR_SIZE), &mut buf)
        .is_err()
    {
        return false;
    }
    &buf[1..6] == ISO_MAGIC
}

/// Top-level handle for an ISO 9660 volume. Holds either parsed
/// descriptors (when opened from a device) or a buffered writer (when
/// constructed via `format`) — `repack` flows through the writer side.
pub struct Iso9660 {
    /// PVD parsed from disk (defaults are zero-filled when in writer mode).
    pub pvd: PrimaryVolumeDescriptor,
    /// First Joliet SVD found, when present.
    pub joliet: Option<SupplementaryVolumeDescriptor>,
    /// El Torito boot catalog when the boot record was present.
    pub boot: Option<BootCatalog>,
    /// `true` when Rock Ridge System Use Area is detected in the PVD's
    /// root directory record. Decides whether we apply RR overrides on
    /// names / attrs during `list`.
    pub rock_ridge: bool,
    /// Present when the handle was built via `format`; absorbs all
    /// `create_*` calls until `flush()` lays out the image.
    writer: Option<Iso9660Writer>,
}

impl Iso9660 {
    /// Open `dev`. Walks the volume descriptors at LBA 16+, locating
    /// PVD, optional Joliet SVD, optional El Torito boot record, and the
    /// VDST terminator.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let mut pvd: Option<PrimaryVolumeDescriptor> = None;
        let mut joliet: Option<SupplementaryVolumeDescriptor> = None;
        let mut boot_record_lba: Option<u32> = None;

        let mut lba = PVD_LBA;
        loop {
            let mut buf = vec![0u8; SECTOR_SIZE as usize];
            dev.read_at(u64::from(lba) * u64::from(SECTOR_SIZE), &mut buf)?;
            if &buf[1..6] != ISO_MAGIC {
                return Err(crate::Error::InvalidImage(format!(
                    "iso9660: missing CD001 magic at LBA {lba}"
                )));
            }
            match VolumeDescriptor::probe(&buf)? {
                VolumeDescriptor::Primary(p) => {
                    pvd = Some(p);
                }
                VolumeDescriptor::Supplementary(s) => {
                    if joliet.is_none() && s.is_joliet() {
                        joliet = Some(s);
                    }
                }
                VolumeDescriptor::Boot { catalog_lba } => {
                    boot_record_lba = Some(catalog_lba);
                }
                VolumeDescriptor::Partition => { /* ignored */ }
                VolumeDescriptor::Terminator => break,
            }
            lba = lba
                .checked_add(1)
                .ok_or_else(|| crate::Error::InvalidImage("iso9660: VD chain overflow".into()))?;
            if lba > 1024 {
                return Err(crate::Error::InvalidImage(
                    "iso9660: VD chain exceeds 1024 sectors without terminator".into(),
                ));
            }
        }

        let pvd = pvd.ok_or_else(|| {
            crate::Error::InvalidImage("iso9660: no Primary Volume Descriptor found".into())
        })?;

        let boot = if let Some(catalog_lba) = boot_record_lba {
            el_torito::BootCatalog::load(dev, catalog_lba).ok()
        } else {
            None
        };

        // Rock Ridge detection: parse the root directory's first entry
        // and look for an SP entry in the System Use Area.
        let rock_ridge = rock_ridge::root_has_rr(dev, &pvd).unwrap_or(false);

        Ok(Self {
            pvd,
            joliet,
            boot,
            rock_ridge,
            writer: None,
        })
    }

    /// Build a fresh, writable handle. Pass options through `opts`;
    /// drive `create_*` to buffer entries; call `flush()` to lay out
    /// the image. Used by `repack --fs-type iso`.
    pub fn format(_dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<Self> {
        Ok(Self {
            pvd: PrimaryVolumeDescriptor {
                system_id: String::new(),
                volume_id: opts.volume_id.clone(),
                volume_space_size: 0,
                logical_block_size: SECTOR_SIZE as u16,
                path_table_size: 0,
                l_path_table_lba: 0,
                m_path_table_lba: 0,
                root: DirRecord {
                    len_dr: 34,
                    extent_lba: 0,
                    length: 0,
                    flags: 0x02,
                    identifier: vec![0u8],
                    system_use: Vec::new(),
                },
            },
            joliet: None,
            boot: None,
            rock_ridge: opts.rock_ridge,
            writer: Some(Iso9660Writer::new(opts.clone())),
        })
    }

    /// Volume identifier from the PVD (or Joliet SVD if present) — the
    /// human-readable name `info` shows.
    pub fn volume_id(&self) -> String {
        if let Some(j) = self.joliet.as_ref() {
            return j.volume_id.clone();
        }
        self.pvd.volume_id.clone()
    }

    /// List the directory at `path` ("/" for the root).
    pub fn list_path(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<Vec<DirEntry>> {
        let rec = self.resolve_path(dev, path)?;
        if !rec.is_dir() {
            return Err(crate::Error::InvalidArgument(format!(
                "iso9660: {path:?} is not a directory"
            )));
        }
        let prefer_joliet = self.joliet.is_some() && !self.rock_ridge;
        let entries = directory::read_directory(
            dev,
            &rec,
            self.rock_ridge,
            if prefer_joliet {
                self.joliet.as_ref()
            } else {
                None
            },
        )?;
        Ok(entries
            .into_iter()
            .filter(|e| e.name != "." && e.name != "..")
            .map(|e| e.into_dir_entry())
            .collect())
    }

    /// Resolve `path` to a directory record (file or directory).
    ///
    /// Precedence (matches the Linux kernel iso9660 driver):
    ///
    /// 1. **Rock Ridge** when present — gives us long names via the `NM`
    ///    entry, plus POSIX modes / symlinks / device numbers. RR
    ///    records live inside PVD directory records, so we walk the PVD.
    /// 2. **Joliet** when RR isn't present — Microsoft-style long names
    ///    via the Supplementary Volume Descriptor.
    /// 3. **Plain ISO 9660** otherwise — uppercased 8.3 with `;version`
    ///    stripped.
    fn resolve_path(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<DirRecord> {
        let prefer_joliet = self.joliet.is_some() && !self.rock_ridge;
        let mut cur = if prefer_joliet {
            self.joliet.as_ref().unwrap().root.clone()
        } else {
            self.pvd.root.clone()
        };
        let path = path.trim_matches('/');
        if path.is_empty() {
            return Ok(cur);
        }
        for comp in path.split('/') {
            if comp.is_empty() {
                continue;
            }
            if !cur.is_dir() {
                return Err(crate::Error::InvalidArgument(format!(
                    "iso9660: non-directory in path before {comp:?}"
                )));
            }
            let entries = directory::read_directory(
                dev,
                &cur,
                self.rock_ridge,
                if prefer_joliet {
                    self.joliet.as_ref()
                } else {
                    None
                },
            )?;
            let m = entries
                .iter()
                .find(|e| e.name == comp)
                .ok_or_else(|| crate::Error::InvalidArgument(format!("iso9660: no {comp:?}")))?;
            cur = m.record.clone();
        }
        Ok(cur)
    }

    /// Open a file for streaming reads.
    pub fn open_file_reader<'a>(
        &'a self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<Box<dyn Read + 'a>> {
        let rec = self.resolve_path(dev, path)?;
        if rec.is_dir() {
            return Err(crate::Error::InvalidArgument(format!(
                "iso9660: {path:?} is a directory"
            )));
        }
        Ok(Box::new(file::ExtentReader::new(
            dev,
            rec.extent_lba,
            rec.length,
        )))
    }
}

// ----------------------------------------------------------------------
// `crate::fs::Filesystem` / `FilesystemFactory` impls — bridges ISO 9660
// into the generic repack walker. ISO is repack-only on the write side:
// every `create_*` buffers an entry, `flush()` lays out the image.
// ----------------------------------------------------------------------

impl crate::fs::FilesystemFactory for Iso9660 {
    type FormatOpts = FormatOpts;

    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }

    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl crate::fs::Filesystem for Iso9660 {
    fn create_file(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        src: crate::fs::FileSource,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        self.writer
            .as_mut()
            .ok_or_else(|| {
                crate::Error::Unsupported(
                    "iso9660: read-only handle (no in-place modification)".into(),
                )
            })?
            .add_file(path, src, meta)
    }

    fn create_dir(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        self.writer
            .as_mut()
            .ok_or_else(|| {
                crate::Error::Unsupported(
                    "iso9660: read-only handle (no in-place modification)".into(),
                )
            })?
            .add_dir(path, meta)
    }

    fn create_symlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        target: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        self.writer
            .as_mut()
            .ok_or_else(|| {
                crate::Error::Unsupported(
                    "iso9660: read-only handle (no in-place modification)".into(),
                )
            })?
            .add_symlink(path, target, meta)
    }

    fn create_device(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        kind: crate::fs::DeviceKind,
        major: u32,
        minor: u32,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        self.writer
            .as_mut()
            .ok_or_else(|| {
                crate::Error::Unsupported(
                    "iso9660: read-only handle (no in-place modification)".into(),
                )
            })?
            .add_device(path, kind, major, minor, meta)
    }

    fn remove(&mut self, _dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<()> {
        self.writer
            .as_mut()
            .ok_or_else(|| {
                crate::Error::Unsupported(
                    "iso9660: read-only handle (no in-place modification)".into(),
                )
            })?
            .remove_entry(path)
    }

    fn list(&mut self, dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<Vec<DirEntry>> {
        let p = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("iso9660: non-UTF-8 path".into()))?;
        self.list_path(dev, p)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn std::io::Read + 'a>> {
        let p = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("iso9660: non-UTF-8 path".into()))?;
        self.open_file_reader(dev, p)
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        if let Some(w) = self.writer.as_mut() {
            w.flush(dev)?;
        }
        Ok(())
    }
}

mod file {
    use std::io::Read;

    use crate::block::BlockDevice;

    use super::SECTOR_SIZE;

    /// Streaming reader for a contiguous ISO extent. ISO 9660 files use
    /// a single extent + length; multi-extent files are an extension we
    /// don't see in the wild for normal data files.
    pub struct ExtentReader<'a> {
        dev: &'a mut dyn BlockDevice,
        start_byte: u64,
        remaining: u64,
        cursor: u64,
    }

    impl<'a> ExtentReader<'a> {
        pub fn new(dev: &'a mut dyn BlockDevice, extent_lba: u32, length: u64) -> Self {
            Self {
                dev,
                start_byte: u64::from(extent_lba) * u64::from(SECTOR_SIZE),
                remaining: length,
                cursor: 0,
            }
        }
    }

    impl<'a> Read for ExtentReader<'a> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.remaining == 0 || buf.is_empty() {
                return Ok(0);
            }
            let want = (buf.len() as u64).min(self.remaining) as usize;
            self.dev
                .read_at(self.start_byte + self.cursor, &mut buf[..want])
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            self.cursor += want as u64;
            self.remaining -= want as u64;
            Ok(want)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn probe_rejects_empty_device() {
        let mut dev = MemoryBackend::new(64 * 1024);
        assert!(!probe(&mut dev));
    }

    #[test]
    fn probe_accepts_cd001_magic_at_lba_16() {
        let mut dev = MemoryBackend::new(64 * 1024);
        // Manually stamp the magic at byte 16*2048 + 1.
        let mut buf = vec![0u8; 7];
        buf[0] = 1; // PVD type
        buf[1..6].copy_from_slice(ISO_MAGIC);
        buf[6] = 1; // version
        crate::block::BlockDevice::write_at(&mut dev, 16 * 2048, &buf).unwrap();
        assert!(probe(&mut dev));
    }
}
