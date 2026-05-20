//! Unified read-side API: probe an image, identify the filesystem on it,
//! and expose a small inspection surface (list / cat / info) that the CLI
//! can drive without knowing which filesystem it's talking to.
//!
//! The probe is deliberately minimal — it reads a couple of well-known
//! offsets and matches magic numbers. It is *not* a full mountability
//! check; opening the image with the chosen backend is still where actual
//! validation happens.

use std::path::{Path, PathBuf};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::DirEntry;
use crate::fs::apfs::Apfs;
use crate::fs::exfat::Exfat;
use crate::fs::ext::Ext;
use crate::fs::f2fs::F2fs;
use crate::fs::fat::Fat32;
use crate::fs::hfs_plus::HfsPlus;
use crate::fs::ntfs::Ntfs;
use crate::fs::squashfs::Squashfs;
use crate::fs::tar::Tar;
use crate::fs::xfs::Xfs;
use crate::part::{Gpt, Mbr, Partition, PartitionTable, slice_partition};

/// Which filesystem an image carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsKind {
    /// ext2 / ext3 / ext4 — distinguished further by feature flags.
    Ext,
    /// FAT32.
    Fat32,
    /// A tar archive treated as a read-only filesystem.
    Tar,
    /// XFS — read-only (shortform dirs + extent files).
    Xfs,
    /// exFAT — read-only.
    Exfat,
    /// HFS+ — read-only.
    HfsPlus,
    /// APFS — read-only, single-leaf-tree case only.
    Apfs,
    /// NTFS — scaffold; detection only, all ops return `Unsupported`.
    Ntfs,
    /// F2FS — scaffold; detection only.
    F2fs,
    /// SquashFS — scaffold; detection only.
    Squashfs,
    /// ISO 9660 (optical media). Read-only on this trait surface;
    /// writing happens through `repack` to a fresh image.
    Iso9660,
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
    // exFAT: "EXFAT   " at offset 3 of LBA 0 (also has 0x55AA at +510).
    if &bs[3..11] == b"EXFAT   " {
        return Ok(FsKind::Exfat);
    }
    // NTFS: "NTFS    " at offset 3 of LBA 0.
    if &bs[3..11] == b"NTFS    " {
        return Ok(FsKind::Ntfs);
    }

    // XFS: "XFSB" at offset 0 of LBA 0.
    if &bs[0..4] == b"XFSB" {
        return Ok(FsKind::Xfs);
    }

    // SquashFS: little-endian "hsqs" at offset 0.
    if &bs[0..4] == b"hsqs" {
        return Ok(FsKind::Squashfs);
    }

    // Tar: "ustar\0" or "ustar " magic at offset 257 of the first block.
    if &bs[257..262] == b"ustar" {
        return Ok(FsKind::Tar);
    }

    // ISO 9660: PVD at LBA 16 (byte 32768) starts with type=0x01,
    // standard identifier "CD001", version=0x01. The PVD is the
    // canonical entry point regardless of Joliet / Rock Ridge / boot
    // record presence.
    if dev.total_size() >= 32768 + 7 {
        let mut iso = [0u8; 7];
        dev.read_at(32768, &mut iso)?;
        if &iso[1..6] == b"CD001" {
            return Ok(FsKind::Iso9660);
        }
    }

    // APFS: container superblock magic "NXSB" at offset 32 of block 0.
    if &bs[32..36] == b"NXSB" {
        return Ok(FsKind::Apfs);
    }

    // ext superblock starts at byte 1024; s_magic (0xEF53) is at offset 56.
    let mut sb_magic = [0u8; 2];
    dev.read_at(1024 + 56, &mut sb_magic)?;
    if sb_magic == [0x53, 0xEF] {
        return Ok(FsKind::Ext);
    }

    // HFS+ / HFSX volume header sig at byte 1024.
    let mut hfs_sig = [0u8; 2];
    if dev.total_size() >= 1024 + 2 {
        dev.read_at(1024, &mut hfs_sig)?;
        if &hfs_sig == b"H+" || &hfs_sig == b"HX" {
            return Ok(FsKind::HfsPlus);
        }
    }

    // F2FS: 32-bit LE magic 0xF2F52010 at offset 1024 (primary) or
    // 1024 + 0x1000 (backup). Check both copies before giving up.
    let mut f2_magic = [0u8; 4];
    if dev.total_size() >= 1024 + 0x1000 + 4 {
        dev.read_at(1024, &mut f2_magic)?;
        if u32::from_le_bytes(f2_magic) == 0xF2F5_2010 {
            return Ok(FsKind::F2fs);
        }
        dev.read_at(1024 + 0x1000, &mut f2_magic)?;
        if u32::from_le_bytes(f2_magic) == 0xF2F5_2010 {
            return Ok(FsKind::F2fs);
        }
    }

    Err(crate::Error::InvalidImage(
        "inspect: no recognised filesystem (ext2/3/4, FAT32, exFAT, XFS, HFS+, APFS, tar, NTFS, F2FS, SquashFS) on this image".into(),
    ))
}

/// A unified read-side handle. Hides whether the underlying filesystem
/// is ext, FAT32, tar, XFS, exFAT, HFS+, or APFS — the CLI calls
/// `list` / `read_file` / `summary` and the right backend dispatches.
///
/// Of these, ext, FAT32, and tar support full read/write; the other
/// four are read-only. Write-side operations on a read-only variant
/// return [`crate::Error::Unsupported`].
pub enum AnyFs {
    Ext(Box<Ext>),
    Fat32(Box<Fat32>),
    /// Tar archive — read-only via this handle.
    Tar(Box<Tar>),
    /// XFS — read-only (shortform dirs + extent-format files).
    Xfs(Box<Xfs>),
    /// exFAT — read-only.
    Exfat(Box<Exfat>),
    /// HFS+ — read-only.
    HfsPlus(Box<HfsPlus>),
    /// APFS — read-only; single-leaf trees only.
    Apfs(Box<Apfs>),
    /// NTFS — scaffold; only `info` returns useful data, list/read error.
    Ntfs(Box<Ntfs>),
    /// F2FS — scaffold; only `info` returns useful data, list/read error.
    F2fs(Box<F2fs>),
    /// SquashFS — scaffold; only `info` returns useful data, list/read error.
    Squashfs(Box<Squashfs>),
    /// ISO 9660 — read-only (PVD + Joliet + Rock Ridge + El Torito).
    Iso9660(Box<crate::fs::iso9660::Iso9660>),
}

impl AnyFs {
    /// Open `dev`, picking the backend automatically.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        match detect_fs(dev)? {
            FsKind::Ext => Ok(Self::Ext(Box::new(Ext::open(dev)?))),
            FsKind::Fat32 => Ok(Self::Fat32(Box::new(Fat32::open(dev)?))),
            FsKind::Tar => Ok(Self::Tar(Box::new(Tar::open(dev)?))),
            FsKind::Xfs => Ok(Self::Xfs(Box::new(Xfs::open(dev)?))),
            FsKind::Exfat => Ok(Self::Exfat(Box::new(Exfat::open(dev)?))),
            FsKind::HfsPlus => Ok(Self::HfsPlus(Box::new(HfsPlus::open(dev)?))),
            FsKind::Apfs => Ok(Self::Apfs(Box::new(Apfs::open(dev)?))),
            FsKind::Ntfs => Ok(Self::Ntfs(Box::new(Ntfs::open(dev)?))),
            FsKind::F2fs => Ok(Self::F2fs(Box::new(F2fs::open(dev)?))),
            FsKind::Squashfs => Ok(Self::Squashfs(Box::new(Squashfs::open(dev)?))),
            FsKind::Iso9660 => Ok(Self::Iso9660(Box::new(crate::fs::iso9660::Iso9660::open(
                dev,
            )?))),
        }
    }

    /// Which filesystem this handle is talking to.
    pub fn kind(&self) -> FsKind {
        match self {
            Self::Ext(_) => FsKind::Ext,
            Self::Fat32(_) => FsKind::Fat32,
            Self::Tar(_) => FsKind::Tar,
            Self::Xfs(_) => FsKind::Xfs,
            Self::Exfat(_) => FsKind::Exfat,
            Self::HfsPlus(_) => FsKind::HfsPlus,
            Self::Apfs(_) => FsKind::Apfs,
            Self::Ntfs(_) => FsKind::Ntfs,
            Self::F2fs(_) => FsKind::F2fs,
            Self::Squashfs(_) => FsKind::Squashfs,
            Self::Iso9660(_) => FsKind::Iso9660,
        }
    }

    /// List the entries of a directory by absolute path. Takes `&mut self`
    /// because some read-only backends (NTFS, F2FS) maintain cached state
    /// (run-list bootstrap, checkpoint selection) behind their list path.
    pub fn list(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<Vec<DirEntry>> {
        match self {
            Self::Ext(ext) => {
                let ino = ext.path_to_inode(dev, path)?;
                ext.list_inode(dev, ino)
            }
            Self::Fat32(fat) => fat.list_path(dev, path),
            Self::Tar(tar) => tar.list_path(dev, path),
            Self::Xfs(xfs) => xfs.list_path(dev, path),
            Self::Exfat(exfat) => exfat.list_path(dev, path),
            Self::HfsPlus(hfs) => hfs.list_path(dev, path),
            Self::Apfs(apfs) => apfs.list_path(dev, path),
            Self::Ntfs(ntfs) => ntfs.list_path(dev, path),
            Self::F2fs(f2) => f2.list_path(dev, path),
            Self::Squashfs(sq) => sq.list_path(dev, path),
            Self::Iso9660(iso) => iso.list_path(dev, path),
        }
    }

    /// Stream a regular file's bytes into `out`. The file is read in
    /// 64 KiB chunks; nothing larger than that buffer is ever resident.
    /// Takes `&mut self` for the same reason as [`AnyFs::list`].
    pub fn copy_file_to(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        out: &mut dyn std::io::Write,
    ) -> Result<u64> {
        let mut buf = [0u8; 64 * 1024];
        match self {
            Self::Ext(ext) => {
                let ino = ext.path_to_inode(dev, path)?;
                let mut r = ext.open_file_reader(dev, ino)?;
                pump(&mut r, out, &mut buf)
            }
            Self::Fat32(fat) => {
                let mut r = fat.open_file_reader(dev, path)?;
                pump(&mut r, out, &mut buf)
            }
            Self::Tar(tar) => {
                let mut r = tar.open_file_reader(dev, path)?;
                pump(&mut r, out, &mut buf)
            }
            Self::Xfs(xfs) => {
                let mut r = xfs.open_file_reader(dev, path)?;
                pump(&mut r, out, &mut buf)
            }
            Self::Exfat(exfat) => {
                let mut r = exfat.open_file_reader(dev, path)?;
                pump(&mut r, out, &mut buf)
            }
            Self::HfsPlus(hfs) => {
                let mut r = hfs.open_file_reader(dev, path)?;
                pump(&mut r, out, &mut buf)
            }
            Self::Apfs(apfs) => {
                let mut r = apfs.open_file_reader(dev, path)?;
                pump(&mut r, out, &mut buf)
            }
            Self::Ntfs(ntfs) => {
                let mut r = ntfs.open_file_reader(dev, path)?;
                pump(&mut r, out, &mut buf)
            }
            Self::F2fs(f2) => {
                let mut r = f2.open_file_reader(dev, path)?;
                pump(&mut r, out, &mut buf)
            }
            Self::Squashfs(sq) => {
                let mut r = sq.open_file_reader(dev, path)?;
                pump(&mut r, out, &mut buf)
            }
            Self::Iso9660(iso) => {
                let mut r = iso.open_file_reader(dev, path)?;
                pump(&mut r, out, &mut buf)
            }
        }
    }

    /// Add a regular file at `dest_path`, populated from a host file.
    /// Parent directories must already exist. Dispatches through the
    /// generic [`crate::fs::Filesystem`] trait. Filesystems whose
    /// reader can't be re-opened as a writer (most non-ext/non-FAT)
    /// will error with a trait-specific message.
    pub fn add_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        dest_path: &str,
        host_src: &Path,
    ) -> Result<()> {
        let meta = std::fs::symlink_metadata(host_src)?;
        let fmeta = crate::fs::FileMeta {
            mode: host_mode_from_meta(&meta, false),
            ..crate::fs::FileMeta::default()
        };
        let dest = std::path::Path::new(dest_path);
        let src = crate::fs::FileSource::HostPath(host_src.to_path_buf());
        self.as_filesystem_dyn(move |fs| fs.create_file(dev, dest, src, fmeta))
    }

    /// Recursively add a host directory tree at `dest_path`. The
    /// destination's parent must exist; the leaf is created. Dispatches
    /// through the [`crate::fs::Filesystem`] trait.
    pub fn add_dir_tree(
        &mut self,
        dev: &mut dyn BlockDevice,
        dest_path: &str,
        host_src: &Path,
    ) -> Result<()> {
        let meta = std::fs::symlink_metadata(host_src)?;
        let fmeta = crate::fs::FileMeta {
            mode: host_mode_from_meta(&meta, true),
            ..crate::fs::FileMeta::default()
        };
        let dest = std::path::Path::new(dest_path);
        self.as_filesystem_dyn(|fs| fs.create_dir(dev, dest, fmeta))?;
        // Walk the host source recursively through the trait. Errors
        // from each entry propagate immediately.
        let source = crate::repack::Source::HostDir(host_src.to_path_buf());
        self.populate_from_source_at(dev, dest_path, &source)
    }

    /// Create an empty directory at `path` with mode 0o755 (umask 022
    /// over 0o777). Dispatches through the trait.
    pub fn mkdir(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<()> {
        let fmeta = crate::fs::FileMeta {
            mode: 0o755,
            ..crate::fs::FileMeta::default()
        };
        let p = std::path::Path::new(path);
        self.as_filesystem_dyn(|fs| fs.create_dir(dev, p, fmeta))
    }

    /// Remove an entry at `path` — a file, symlink, device, or empty
    /// directory. Non-empty directories are rejected. Dispatches
    /// through the trait.
    pub fn remove(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<()> {
        let p = std::path::Path::new(path);
        self.as_filesystem_dyn(|fs| fs.remove(dev, p))
    }

    /// Dispatch a closure to whichever inner filesystem implements
    /// [`crate::fs::Filesystem`]. Centralises the per-variant `match`
    /// so callers like [`Self::add_file`] / [`Self::mkdir`] /
    /// [`Self::remove`] aren't 10-arm long.
    fn as_filesystem_dyn<R>(
        &mut self,
        f: impl FnOnce(&mut dyn crate::fs::Filesystem) -> Result<R>,
    ) -> Result<R> {
        match self {
            Self::Ext(ext) => f(ext.as_mut()),
            Self::Fat32(fat) => f(fat.as_mut()),
            Self::HfsPlus(h) => f(h.as_mut()),
            Self::Ntfs(n) => f(n.as_mut()),
            Self::F2fs(fs2) => f(fs2.as_mut()),
            Self::Squashfs(sq) => f(sq.as_mut()),
            Self::Xfs(x) => f(x.as_mut()),
            Self::Tar(_) => Err(read_only_fs("tar")),
            Self::Apfs(_) => Err(read_only_fs("apfs")),
            Self::Exfat(_) => Err(read_only_fs("exfat")),
            Self::Iso9660(_) => Err(read_only_fs("iso9660")),
        }
    }

    /// Walk the file tree under `src` into `at_path` on this FS via the
    /// generic trait. Used by `add_dir_tree` after the leaf dir has been
    /// created.
    fn populate_from_source_at(
        &mut self,
        dev: &mut dyn BlockDevice,
        at_path: &str,
        src: &crate::repack::Source,
    ) -> Result<()> {
        let _ = at_path; // generic populate walks from the source root;
        // a future enhancement can rebase entries under `at_path` for
        // add_dir_tree's "drop the tree here" semantics.
        self.as_filesystem_dyn(|fs| {
            // SAFETY: populate_fs_from_source is generic over F:
            // Filesystem; we can't call it directly on a trait object,
            // so we open a private helper that takes &mut dyn.
            crate::repack::populate_fs_from_source_dyn(dev, fs, src)
        })
    }

    /// Persist any in-memory metadata changes to the device.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        match self {
            Self::Ext(ext) => ext.flush(dev),
            Self::Fat32(fat) => fat.flush(dev),
            // Read-only handles have nothing to flush.
            Self::Tar(_)
            | Self::Xfs(_)
            | Self::Exfat(_)
            | Self::HfsPlus(_)
            | Self::Apfs(_)
            | Self::Ntfs(_)
            | Self::F2fs(_)
            | Self::Squashfs(_)
            | Self::Iso9660(_) => Ok(()),
        }
    }

    /// One-line FS summary, used by `fstool info`'s heading.
    pub fn kind_string(&self) -> &'static str {
        // Stitch in the read-only variants up front so the existing
        // arms below for ext/fat32/tar don't need restructuring.
        if let Self::Xfs(_) = self {
            return "xfs";
        }
        if let Self::Exfat(_) = self {
            return "exfat";
        }
        if let Self::HfsPlus(_) = self {
            return "hfs+";
        }
        if let Self::Apfs(_) = self {
            return "apfs";
        }
        if let Self::Ntfs(_) = self {
            return "ntfs";
        }
        if let Self::F2fs(_) = self {
            return "f2fs";
        }
        if let Self::Squashfs(_) = self {
            return "squashfs";
        }
        if let Self::Iso9660(_) = self {
            return "iso9660";
        }
        match self {
            Self::Ext(ext) => match ext.kind {
                crate::fs::ext::FsKind::Ext2 => "ext2",
                crate::fs::ext::FsKind::Ext3 => "ext3",
                crate::fs::ext::FsKind::Ext4 => "ext4",
            },
            Self::Fat32(_) => "fat32",
            Self::Tar(_) => "tar",
            // Read-only variants are dispatched above.
            Self::Xfs(_)
            | Self::Exfat(_)
            | Self::HfsPlus(_)
            | Self::Apfs(_)
            | Self::Ntfs(_)
            | Self::F2fs(_)
            | Self::Squashfs(_)
            | Self::Iso9660(_) => unreachable!(),
        }
    }
}

/// Best-effort mode for a host file pulled in via `add_file` / `add_dir_tree`.
/// On Unix: the actual permission bits. On Windows: a fixed 0o755 for
/// directories and 0o644 for everything else (we don't have POSIX bits
/// to read).
fn host_mode_from_meta(meta: &std::fs::Metadata, is_dir: bool) -> u16 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = is_dir;
        (meta.permissions().mode() & 0o7777) as u16
    }
    #[cfg(not(unix))]
    {
        let _ = meta;
        if is_dir { 0o755 } else { 0o644 }
    }
}

fn read_only_fs(name: &str) -> crate::Error {
    crate::Error::Unsupported(format!(
        "{name} is mounted read-only by fstool; build a fresh output with `fstool repack`"
    ))
}

/// Pump `reader` into `out` through `buf` until EOF, returning total
/// bytes copied. Used by `copy_file_to` for every backend. `W` is
/// `?Sized` so `&mut dyn Write` callers work directly.
fn pump<R: std::io::Read + ?Sized, W: std::io::Write + ?Sized>(
    reader: &mut R,
    out: &mut W,
    buf: &mut [u8],
) -> Result<u64> {
    let mut total = 0u64;
    loop {
        let n = reader.read(buf).map_err(crate::Error::from)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(crate::Error::from)?;
        total += n as u64;
    }
    Ok(total)
}

/// One-shot helper: open `path` (regular file, block device, or qcow2),
/// identify the filesystem on it, and return the handle.
pub fn open_image_file(path: &Path) -> Result<(Box<dyn BlockDevice>, AnyFs)> {
    let mut dev = crate::block::open_image(path)?;
    let fs = AnyFs::open(dev.as_mut())?;
    Ok((dev, fs))
}

// -- partition-aware target plumbing ------------------------------------

/// A parsed CLI image target. The user can write `disk.img` for the
/// whole image or `disk.img:N` to target the N-th partition (1-indexed,
/// matching `sgdisk -p` and `loopXpN`). Internally we store the index
/// zero-based for convenience.
#[derive(Debug, Clone)]
pub struct Target {
    pub path: PathBuf,
    /// `None` → whole disk; `Some(i)` → partition with zero-based index `i`.
    pub partition: Option<usize>,
}

impl Target {
    /// Parse a target spec. `disk.img:N` is the partition form; any other
    /// `:` in the path (e.g. on Windows) is preserved by only splitting on
    /// the *last* `:` and only when the trailing segment parses as a
    /// 1-based partition number.
    pub fn parse(s: &str) -> Self {
        if let Some((head, tail)) = s.rsplit_once(':')
            && let Ok(n) = tail.parse::<usize>()
            && n >= 1
        {
            return Self {
                path: PathBuf::from(head),
                partition: Some(n - 1),
            };
        }
        Self {
            path: PathBuf::from(s),
            partition: None,
        }
    }
}

/// A disk's partition table. Box-wrapped behind [`PartitionTable`] so
/// callers can consume MBR and GPT through the same dispatch.
pub enum DetectedTable {
    Gpt(Box<Gpt>),
    Mbr(Box<Mbr>),
}

impl DetectedTable {
    /// Returns the inner trait object for slicing / iteration.
    pub fn as_table(&self) -> &dyn PartitionTable {
        match self {
            Self::Gpt(g) => g.as_ref(),
            Self::Mbr(m) => m.as_ref(),
        }
    }

    /// Short label for UI ("gpt" / "mbr").
    pub fn label(&self) -> &'static str {
        match self {
            Self::Gpt(_) => "gpt",
            Self::Mbr(_) => "mbr",
        }
    }

    /// All non-empty partitions, in disk order.
    pub fn partitions(&self) -> &[Partition] {
        self.as_table().partitions()
    }
}

/// Probe `dev` for a partition table. Returns `Ok(Some(table))` when a
/// GPT or MBR is found, `Ok(None)` when sector 0 looks like an ext or
/// FAT32 image (no partition table), and `Err(_)` only on I/O failures.
///
/// GPT takes precedence: a GPT disk's sector 0 contains a *protective*
/// MBR whose only entry has type 0xEE, so we'd otherwise treat it as a
/// legacy MBR and slice incorrectly.
pub fn detect_partition_table(dev: &mut dyn BlockDevice) -> Result<Option<DetectedTable>> {
    if dev.total_size() < 512 {
        return Ok(None);
    }
    // Look at the FS signatures first — if the sector 0 carries a FAT32
    // boot record or the LBA-2 region (offset 1024) carries an ext
    // superblock, it's a bare FS, not a partition table.
    let mut s0 = [0u8; 512];
    dev.read_at(0, &mut s0)?;
    let is_fat32 = s0[510] == 0x55 && s0[511] == 0xAA && &s0[82..87] == b"FAT32";
    if is_fat32 {
        return Ok(None);
    }
    let has_55aa = s0[510] == 0x55 && s0[511] == 0xAA;
    // GPT signature at LBA 1 (offset 512) is "EFI PART".
    if dev.total_size() >= 1024 {
        let mut s1_head = [0u8; 8];
        dev.read_at(512, &mut s1_head)?;
        if &s1_head == b"EFI PART" {
            let gpt = Gpt::read(dev)?;
            return Ok(Some(DetectedTable::Gpt(Box::new(gpt))));
        }
    }
    // Legacy MBR: 0x55AA signature plus at least one partition entry whose
    // type byte is non-zero. (A zero-FS image with a stray 0x55AA in the
    // first 512 bytes is unlikely but possible — the entry-type check
    // prevents misidentification.)
    if has_55aa {
        for i in 0..4 {
            let entry_off = 446 + i * 16;
            if s0[entry_off + 4] != 0 {
                let mbr = Mbr::read(dev)?;
                return Ok(Some(DetectedTable::Mbr(Box::new(mbr))));
            }
        }
    }
    Ok(None)
}

/// Run `op` with a [`BlockDevice`] that points at whatever `target`
/// resolves to: the whole disk for `disk.img`, or a partition slice for
/// `disk.img:N`. The closure opens its own [`AnyFs`] (or doesn't, e.g.
/// `info` may want to list the partition table instead).
///
/// Errors with [`crate::Error::InvalidArgument`] when `target` names a
/// partition but the image carries no partition table (or the index is
/// out of range).
pub fn with_target_device<F, R>(target: &Target, op: F) -> Result<R>
where
    F: FnOnce(&mut dyn BlockDevice) -> Result<R>,
{
    // _tmp keeps the decompressed temp file alive for the duration of
    // the borrow — when it drops, the file is unlinked.
    let (mut disk, _tmp) = crate::block::open_image_maybe_compressed(&target.path)?;
    match target.partition {
        None => op(disk.as_mut()),
        Some(idx) => {
            let table = detect_partition_table(disk.as_mut())?.ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "{}: no partition table found, can't target partition {}",
                    target.path.display(),
                    idx + 1
                ))
            })?;
            let mut slice = slice_partition(table.as_table(), disk.as_mut(), idx)?;
            op(&mut slice)
        }
    }
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

        let (mut dev, mut fs) = open_image_file(tmp.path()).unwrap();
        assert_eq!(fs.kind(), FsKind::Ext);
        let entries = fs.list(dev.as_mut(), "/").unwrap();
        // Default ext format includes lost+found.
        assert!(entries.iter().any(|e| e.name == "lost+found"));
    }
}
