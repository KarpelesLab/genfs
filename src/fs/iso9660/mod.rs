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
            .ok_or(crate::Error::Immutable {
                kind: "iso9660",
                op: "write",
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
            .ok_or(crate::Error::Immutable {
                kind: "iso9660",
                op: "write",
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
            .ok_or(crate::Error::Immutable {
                kind: "iso9660",
                op: "write",
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
            .ok_or(crate::Error::Immutable {
                kind: "iso9660",
                op: "write",
            })?
            .add_device(path, kind, major, minor, meta)
    }

    fn remove(&mut self, _dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<()> {
        self.writer
            .as_mut()
            .ok_or(crate::Error::Immutable {
                kind: "iso9660",
                op: "write",
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

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn crate::fs::FileReadHandle + 'a>> {
        let p = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("iso9660: non-UTF-8 path".into()))?;
        let rec = self.resolve_path(dev, p)?;
        if rec.is_dir() {
            return Err(crate::Error::InvalidArgument(format!(
                "iso9660: {p:?} is a directory"
            )));
        }
        Ok(Box::new(file::Iso9660FileReadHandle::new(
            dev,
            rec.extent_lba,
            rec.length,
        )))
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        if let Some(w) = self.writer.as_mut() {
            w.flush(dev)?;
        }
        Ok(())
    }

    fn mutation_capability(&self) -> crate::fs::MutationCapability {
        crate::fs::MutationCapability::Immutable
    }
}

mod file {
    use std::io::{Read, Seek, SeekFrom};

    use crate::block::BlockDevice;
    use crate::fs::FileReadHandle;

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

    /// Random-access (`Read + Seek`) handle over a contiguous ISO extent.
    /// Backs [`crate::fs::Filesystem::open_file_ro`] for ISO 9660:
    /// reads come straight from `extent_lba * 2048 + position` for the
    /// remaining range; `Seek` shifts the in-extent cursor with the
    /// usual `SeekFrom` semantics, and seeking past `len` clamps the
    /// cursor at `len` so subsequent reads return EOF rather than read
    /// random sectors after the extent.
    pub struct Iso9660FileReadHandle<'a> {
        dev: &'a mut dyn BlockDevice,
        start_byte: u64,
        len: u64,
        pos: u64,
    }

    impl<'a> Iso9660FileReadHandle<'a> {
        pub fn new(dev: &'a mut dyn BlockDevice, extent_lba: u32, length: u64) -> Self {
            Self {
                dev,
                start_byte: u64::from(extent_lba) * u64::from(SECTOR_SIZE),
                len: length,
                pos: 0,
            }
        }
    }

    impl<'a> Read for Iso9660FileReadHandle<'a> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if buf.is_empty() || self.pos >= self.len {
                return Ok(0);
            }
            let remaining = self.len - self.pos;
            let want = (buf.len() as u64).min(remaining) as usize;
            self.dev
                .read_at(self.start_byte + self.pos, &mut buf[..want])
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            self.pos += want as u64;
            Ok(want)
        }
    }

    impl<'a> Seek for Iso9660FileReadHandle<'a> {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            // Resolve to a signed offset against the chosen anchor, then
            // clamp into `[0, len]` with `SeekFrom::End(positive)`
            // capping at len so a subsequent `read` returns EOF.
            let (anchor, offset) = match pos {
                SeekFrom::Start(n) => {
                    let new_pos = n.min(self.len);
                    self.pos = new_pos;
                    return Ok(self.pos);
                }
                SeekFrom::Current(d) => (self.pos as i128, d as i128),
                SeekFrom::End(d) => (self.len as i128, d as i128),
            };
            let target = anchor + offset;
            if target < 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "iso9660: seek to negative offset",
                ));
            }
            let target = target as u128;
            // Cap at len (matches the brief: SeekFrom::End(positive)
            // caps at len, reads return EOF). Same clamp applied
            // uniformly to Current so seeking past the end via either
            // anchor behaves identically.
            self.pos = u64::try_from(target.min(self.len as u128)).unwrap_or(self.len);
            Ok(self.pos)
        }
    }

    impl<'a> FileReadHandle for Iso9660FileReadHandle<'a> {
        fn len(&self) -> u64 {
            self.len
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

    /// Verify `open_file_ro` returns a Read+Seek+len handle over an
    /// ISO file: build a small image via the writer, reopen, then
    /// open a file at random offset and read back exact bytes.
    #[test]
    fn open_file_ro_random_seek_round_trip() {
        use crate::fs::Filesystem;
        use std::io::{Read, Seek, SeekFrom};

        let body: Vec<u8> = (0..4096u32).map(|i| (i & 0xff) as u8).collect();
        let mut dev = MemoryBackend::new(4 * 1024 * 1024);
        let mut iso = Iso9660::format(&mut dev, &FormatOpts::default()).unwrap();
        iso.create_file(
            &mut dev,
            std::path::Path::new("/data.bin"),
            crate::fs::FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(body.clone())),
                len: body.len() as u64,
            },
            crate::fs::FileMeta::default(),
        )
        .unwrap();
        iso.flush(&mut dev).unwrap();

        let mut iso = Iso9660::open(&mut dev).unwrap();
        // Look up the actual name from the root listing — varies
        // depending on whether Joliet / Rock Ridge stuck the long
        // name through. Both reader paths converge in the trait
        // `list` impl above.
        let entries = iso
            .list(&mut dev, std::path::Path::new("/"))
            .expect("list root");
        let name = entries
            .iter()
            .find(|e| matches!(e.kind, crate::fs::EntryKind::Regular))
            .expect("at least one regular file in root")
            .name
            .clone();
        let path = format!("/{name}");
        let mut h = iso
            .open_file_ro(&mut dev, std::path::Path::new(&path))
            .expect("open_file_ro should succeed");

        assert_eq!(h.len(), body.len() as u64);
        // Read first 64 bytes from start.
        let mut chunk = [0u8; 64];
        h.read_exact(&mut chunk).unwrap();
        assert_eq!(&chunk[..], &body[..64]);
        // Seek to offset 1000 and read 32 bytes.
        h.seek(SeekFrom::Start(1000)).unwrap();
        let mut chunk = [0u8; 32];
        h.read_exact(&mut chunk).unwrap();
        assert_eq!(&chunk[..], &body[1000..1032]);
        // Seek relative; verify reads track.
        h.seek(SeekFrom::Current(-32)).unwrap();
        h.read_exact(&mut chunk).unwrap();
        assert_eq!(&chunk[..], &body[1000..1032]);
    }

    #[test]
    fn open_file_ro_seek_past_end_is_capped() {
        use crate::fs::Filesystem;
        use std::io::{Read, Seek, SeekFrom};

        let body = b"short".to_vec();
        let mut dev = MemoryBackend::new(2 * 1024 * 1024);
        let mut iso = Iso9660::format(&mut dev, &FormatOpts::default()).unwrap();
        iso.create_file(
            &mut dev,
            std::path::Path::new("/x"),
            crate::fs::FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(body.clone())),
                len: body.len() as u64,
            },
            crate::fs::FileMeta::default(),
        )
        .unwrap();
        iso.flush(&mut dev).unwrap();

        let mut iso = Iso9660::open(&mut dev).unwrap();
        let entries = iso
            .list(&mut dev, std::path::Path::new("/"))
            .expect("list root");
        let name = entries
            .iter()
            .find(|e| matches!(e.kind, crate::fs::EntryKind::Regular))
            .expect("at least one regular file in root")
            .name
            .clone();
        let path = format!("/{name}");
        let mut h = iso
            .open_file_ro(&mut dev, std::path::Path::new(&path))
            .expect("open_file_ro should resolve");

        // SeekFrom::End(100) — past the end — caps at len; reads
        // immediately return 0 (EOF), not random sector data.
        let where_ = h.seek(SeekFrom::End(100)).unwrap();
        assert_eq!(where_, body.len() as u64);
        let mut chunk = [0u8; 32];
        let n = h.read(&mut chunk).unwrap();
        assert_eq!(n, 0);
    }
}
