//! Unified read-side API: probe an image, identify the filesystem on it,
//! and expose a small inspection surface (list / cat / info) that the CLI
//! can drive without knowing which filesystem it's talking to.
//!
//! The probe is deliberately minimal — it reads a couple of well-known
//! offsets and matches magic numbers. It is *not* a full mountability
//! check; opening the image with the chosen backend is still where actual
//! validation happens.

use std::io::Read;
use std::path::Path;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::DirEntry;
use crate::fs::ext::Ext;
use crate::fs::fat::Fat32;

/// Which filesystem an image carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsKind {
    /// ext2 / ext3 / ext4 — distinguished further by feature flags.
    Ext,
    /// FAT32.
    Fat32,
}

/// Probe `dev` to decide which filesystem it carries. Reads only sector 0
/// and the ext superblock at byte 1024; no mutation, no full open.
pub fn detect_fs(dev: &mut dyn BlockDevice) -> Result<FsKind> {
    // FAT32 first: cheap, only 90 bytes of sector 0, signature is very
    // specific ("FAT32" at +82 and 0x55AA at +510). An ext superblock
    // could in principle live on a disk that also has a sector-0 boot
    // sector, but ext images start with all-zero (mke2fs leaves the first
    // 1024 bytes for boot code), so a real FAT32 signature here is decisive.
    let mut bs = [0u8; 512];
    dev.read_at(0, &mut bs)?;
    if bs[510] == 0x55 && bs[511] == 0xAA && &bs[82..87] == b"FAT32" {
        return Ok(FsKind::Fat32);
    }

    // ext superblock starts at byte 1024; s_magic (0xEF53) is at offset 56.
    let mut sb_magic = [0u8; 2];
    dev.read_at(1024 + 56, &mut sb_magic)?;
    if sb_magic == [0x53, 0xEF] {
        return Ok(FsKind::Ext);
    }

    Err(crate::Error::InvalidImage(
        "inspect: no recognised filesystem (ext2/3/4 or FAT32) on this image".into(),
    ))
}

/// A unified read-side handle. Hides whether the underlying filesystem
/// is ext or FAT32 — the CLI calls `list` / `read_file` / `summary` and
/// the right backend dispatches.
///
/// The ext backend's state is much larger than FAT32's (group descriptors,
/// bitmaps, inode table cache), so it's boxed to keep the enum compact.
pub enum AnyFs {
    Ext(Box<Ext>),
    Fat32(Box<Fat32>),
}

impl AnyFs {
    /// Open `dev`, picking the backend automatically.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        match detect_fs(dev)? {
            FsKind::Ext => Ok(Self::Ext(Box::new(Ext::open(dev)?))),
            FsKind::Fat32 => Ok(Self::Fat32(Box::new(Fat32::open(dev)?))),
        }
    }

    /// Which filesystem this handle is talking to.
    pub fn kind(&self) -> FsKind {
        match self {
            Self::Ext(_) => FsKind::Ext,
            Self::Fat32(_) => FsKind::Fat32,
        }
    }

    /// List the entries of a directory by absolute path.
    pub fn list(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<Vec<DirEntry>> {
        match self {
            Self::Ext(ext) => {
                let ino = ext.path_to_inode(dev, path)?;
                ext.list_inode(dev, ino)
            }
            Self::Fat32(fat) => fat.list_path(dev, path),
        }
    }

    /// Stream a regular file's bytes into `out`. The file is read in
    /// 64 KiB chunks; nothing larger than that buffer is ever resident.
    pub fn copy_file_to(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
        out: &mut dyn std::io::Write,
    ) -> Result<u64> {
        let mut buf = [0u8; 64 * 1024];
        let mut total = 0u64;
        match self {
            Self::Ext(ext) => {
                let ino = ext.path_to_inode(dev, path)?;
                let mut reader = ext.open_file_reader(dev, ino)?;
                loop {
                    let n = reader.read(&mut buf).map_err(crate::Error::from)?;
                    if n == 0 {
                        break;
                    }
                    out.write_all(&buf[..n]).map_err(crate::Error::from)?;
                    total += n as u64;
                }
            }
            Self::Fat32(fat) => {
                let mut reader = fat.open_file_reader(dev, path)?;
                loop {
                    let n = reader.read(&mut buf).map_err(crate::Error::from)?;
                    if n == 0 {
                        break;
                    }
                    out.write_all(&buf[..n]).map_err(crate::Error::from)?;
                    total += n as u64;
                }
            }
        }
        Ok(total)
    }

    /// One-line FS summary, used by `fstool info`'s heading.
    pub fn kind_string(&self) -> &'static str {
        match self {
            Self::Ext(ext) => match ext.kind {
                crate::fs::ext::FsKind::Ext2 => "ext2",
                crate::fs::ext::FsKind::Ext3 => "ext3",
                crate::fs::ext::FsKind::Ext4 => "ext4",
            },
            Self::Fat32(_) => "fat32",
        }
    }
}

/// One-shot helper: open `path` as a file-backed device, identify the FS,
/// and return the handle. Useful for CLI subcommands.
pub fn open_image_file(path: &Path) -> Result<(crate::block::FileBackend, AnyFs)> {
    let mut dev = crate::block::FileBackend::open(path)?;
    let fs = AnyFs::open(&mut dev)?;
    Ok((dev, fs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{FileBackend, MemoryBackend};
    use crate::fs::ext::{Ext, FormatOpts};

    #[test]
    fn detects_ext2_in_memory() {
        let opts = FormatOpts::default();
        let mut dev = MemoryBackend::new(opts.blocks_count as u64 * opts.block_size as u64);
        Ext::format_with(&mut dev, &opts).unwrap();
        assert_eq!(detect_fs(&mut dev).unwrap(), FsKind::Ext);
    }

    #[test]
    fn detects_fat32_in_memory() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = crate::fs::fat::FatFormatOpts {
            total_sectors: 64 * 1024 * 1024 / 512,
            volume_id: 0xCAFE_F00D,
            volume_label: *b"DETECTVOL  ",
        };
        crate::fs::fat::Fat32::format(&mut dev, &opts).unwrap();
        assert_eq!(detect_fs(&mut dev).unwrap(), FsKind::Fat32);
    }

    #[test]
    fn rejects_random_garbage() {
        let mut dev = MemoryBackend::new(64 * 1024);
        // First write a byte to make the device non-pristine.
        dev.write_at(0, b"not a filesystem").unwrap();
        assert!(detect_fs(&mut dev).is_err());
    }

    #[test]
    fn anyfs_lists_an_ext_image() {
        use tempfile::NamedTempFile;
        let opts = FormatOpts::default();
        let size = opts.blocks_count as u64 * opts.block_size as u64;
        let tmp = NamedTempFile::new().unwrap();
        let mut dev = FileBackend::create(tmp.path(), size).unwrap();
        let mut ext = Ext::format_with(&mut dev, &opts).unwrap();
        ext.flush(&mut dev).unwrap();
        dev.sync().unwrap();
        drop(dev);

        let (mut dev, fs) = open_image_file(tmp.path()).unwrap();
        assert_eq!(fs.kind(), FsKind::Ext);
        let entries = fs.list(&mut dev, "/").unwrap();
        // Default ext format includes lost+found.
        assert!(entries.iter().any(|e| e.name == "lost+found"));
    }
}
