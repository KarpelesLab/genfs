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
use crate::fs::ext::Ext;
use crate::fs::fat::Fat32;
use crate::fs::{DirEntry, Filesystem};

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

    /// Add a regular file at `dest_path`, populated from a host file.
    /// Parent directories must already exist. For ext, the destination's
    /// mode is taken from the host file's permission bits; FAT has no
    /// Unix permissions and ignores them.
    pub fn add_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        dest_path: &str,
        host_src: &Path,
    ) -> Result<()> {
        match self {
            Self::Ext(ext) => {
                use std::os::unix::fs::PermissionsExt;
                let meta = std::fs::symlink_metadata(host_src)?;
                let fmeta = crate::fs::FileMeta {
                    mode: (meta.permissions().mode() & 0o7777) as u16,
                    ..crate::fs::FileMeta::default()
                };
                let dest = std::path::Path::new(dest_path);
                use crate::fs::FileSource;
                ext.create_file(
                    dev,
                    dest,
                    FileSource::HostPath(host_src.to_path_buf()),
                    fmeta,
                )
            }
            Self::Fat32(fat) => fat.add_file(dev, dest_path, host_src),
        }
    }

    /// Recursively add a host directory tree at `dest_path`. The
    /// destination's parent must exist; the leaf is created. Symlinks in
    /// the source are skipped on FAT.
    pub fn add_dir_tree(
        &mut self,
        dev: &mut dyn BlockDevice,
        dest_path: &str,
        host_src: &Path,
    ) -> Result<()> {
        match self {
            Self::Ext(ext) => {
                use std::os::unix::fs::PermissionsExt;
                let meta = std::fs::symlink_metadata(host_src)?;
                let fmeta = crate::fs::FileMeta {
                    mode: (meta.permissions().mode() & 0o7777) as u16,
                    ..crate::fs::FileMeta::default()
                };
                let dest = std::path::Path::new(dest_path);
                ext.create_dir(dev, dest, fmeta)?;
                let dir_ino = ext.path_to_inode(dev, dest_path)?;
                ext.populate_from_host_dir(dev, dir_ino, host_src)?;
                Ok(())
            }
            Self::Fat32(fat) => {
                fat.add_dir(dev, dest_path)?;
                add_host_tree_into_fat32(fat, dev, dest_path, host_src)
            }
        }
    }

    /// Create an empty directory at `path`. The parent must already
    /// exist. For ext the mode is 0o755 (umask 022 over 0o777); FAT has
    /// no Unix permissions.
    pub fn mkdir(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<()> {
        match self {
            Self::Ext(ext) => {
                let meta = crate::fs::FileMeta {
                    mode: 0o755,
                    ..crate::fs::FileMeta::default()
                };
                ext.create_dir(dev, std::path::Path::new(path), meta)
            }
            Self::Fat32(fat) => fat.add_dir(dev, path),
        }
    }

    /// Remove an entry at `path` — a file, symlink, device, or empty
    /// directory. Non-empty directories are rejected.
    pub fn remove(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<()> {
        match self {
            Self::Ext(ext) => ext.remove_path(dev, path),
            Self::Fat32(fat) => fat.remove(dev, path),
        }
    }

    /// Persist any in-memory metadata changes to the device.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        match self {
            Self::Ext(ext) => ext.flush(dev),
            Self::Fat32(fat) => fat.flush(dev),
        }
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

/// Recursively copy a host directory tree into a pre-existing FAT32
/// directory at `dest_path`. Used by [`AnyFs::add_dir_tree`] for the
/// FAT32 backend; symlinks in the source are skipped, regular files are
/// streamed through `add_file`, and subdirectories recurse.
fn add_host_tree_into_fat32(
    fat: &mut Fat32,
    dev: &mut dyn BlockDevice,
    dest_path: &str,
    host_src: &Path,
) -> Result<()> {
    let mut entries: Vec<std::fs::DirEntry> =
        std::fs::read_dir(host_src)?.collect::<std::result::Result<_, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        let meta = entry.metadata()?;
        let name = entry
            .file_name()
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("fat32: non-UTF-8 file name".into()))?
            .to_string();
        let child_dest = if dest_path.ends_with('/') {
            format!("{dest_path}{name}")
        } else {
            format!("{dest_path}/{name}")
        };
        let ft = meta.file_type();
        if ft.is_symlink() {
            continue; // FAT has no symlinks
        }
        if ft.is_file() {
            fat.add_file(dev, &child_dest, &path)?;
        } else if ft.is_dir() {
            fat.add_dir(dev, &child_dest)?;
            add_host_tree_into_fat32(fat, dev, &child_dest, &path)?;
        }
    }
    Ok(())
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
