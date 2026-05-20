//! Repack — copy file trees from one source into a freshly-formatted
//! destination filesystem.
//!
//! Three kinds of sources, exposed as [`Source`]:
//!
//! * A **host directory** (`Source::HostDir`) — the original
//!   `ext-build` / `fat-build` flow, walks a directory tree.
//! * A **tar archive** on disk (`Source::TarArchive`), with optional
//!   compression codec. Compressed archives go through a two-pass
//!   stream-index → replay flow; plain `.tar` falls through to the
//!   `Image` path (the regular tar reader sits on top of a
//!   [`BlockDevice`](crate::block::BlockDevice)).
//! * An **existing image** (`Source::Image`) — a raw or qcow2 file,
//!   optionally with a `:N` partition selector. Walks the source FS
//!   through [`AnyFs`](crate::inspect::AnyFs) and copies entries
//!   straight through without host-filesystem intermediation.
//!
//! The two main entry points — [`populate_ext_from_source`] and
//! [`populate_fat32_from_source`] — take an already-formatted
//! destination filesystem and stream the chosen source's contents
//! into it. Auto-sizing helpers ([`ext_build_plan_for_source`],
//! [`fat32_min_bytes_for_source`]) let callers right-size the
//! destination geometry up-front.
//!
//! Used by both the `fstool repack` CLI command and by the
//! [`spec`](crate::spec) layer when a TOML `source = "..."` value
//! points at a tar / image instead of a directory.

use std::path::{Path, PathBuf};

use crate::Result;
use crate::compression::Algo;
use crate::fs::ext::{Ext, FsKind};

/// Where to draw a filesystem's contents from when building or
/// populating it. See module docs.
#[derive(Debug, Clone)]
pub enum Source {
    /// A directory on the host filesystem, walked recursively.
    HostDir(PathBuf),
    /// A tar archive on disk, plain or compressed.
    TarArchive { path: PathBuf, codec: Option<Algo> },
    /// An existing image, optionally with a `:N` partition selector.
    Image(crate::inspect::Target),
}

impl Source {
    /// Auto-detect what kind of source `spec` points at.
    ///
    /// * An existing directory path → `HostDir`.
    /// * A recognised tar extension (`.tar`, `.tar.gz`, `.tgz`,
    ///   `.tar.xz`, `.txz`, `.tar.zst`, `.tar.lz4`, `.tar.lzma`,
    ///   `.tar.lzo`) → `TarArchive`.
    /// * Anything else, including a `path:N` partition selector
    ///   → `Image`. Parsed by [`crate::inspect::Target::parse`].
    pub fn detect(spec: &str) -> Result<Self> {
        let bare = spec.split(':').next().unwrap_or(spec);
        let bare_path = Path::new(bare);
        if bare == spec
            && let Ok(meta) = std::fs::metadata(bare_path)
            && meta.is_dir()
        {
            return Ok(Self::HostDir(bare_path.to_path_buf()));
        }
        if let Some(codec) = tar_input_codec(spec) {
            return Ok(Self::TarArchive {
                path: bare_path.to_path_buf(),
                codec: Some(codec),
            });
        }
        if has_plain_tar_extension(bare_path) {
            return Ok(Self::TarArchive {
                path: bare_path.to_path_buf(),
                codec: None,
            });
        }
        Ok(Self::Image(crate::inspect::Target::parse(spec)))
    }
}

fn has_plain_tar_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("tar"))
}

/// Open `target` as a block device, treat its contents as an
/// arbitrary readable FS, and copy every entry into `dst` (a
/// freshly-formatted ext{2,3,4}).
fn copy_image_into_ext(
    target: &crate::inspect::Target,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut Ext,
) -> Result<()> {
    crate::inspect::with_target_device(target, |src_dev| {
        let src_fs = crate::inspect::AnyFs::open(src_dev)?;
        copy_into_ext(src_dev, &src_fs, dst_dev, dst)
    })
}

/// FAT32 sibling of [`copy_image_into_ext`].
fn copy_image_into_fat32(
    target: &crate::inspect::Target,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::fat::Fat32,
) -> Result<()> {
    crate::inspect::with_target_device(target, |src_dev| {
        let src_fs = crate::inspect::AnyFs::open(src_dev)?;
        copy_into_fat32(src_dev, &src_fs, dst_dev, dst)
    })
}

/// Populate `dst` (a freshly formatted ext{2,3,4}) with the contents
/// of `source`. The destination is assumed to already exist.
pub fn populate_ext_from_source(
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut Ext,
    source: &Source,
) -> Result<()> {
    match source {
        Source::HostDir(p) => dst.populate_from_host_dir(dst_dev, 2, p),
        Source::TarArchive {
            path,
            codec: Some(algo),
        } => {
            let spec = path.to_string_lossy().into_owned();
            let index = build_tar_stream_index(&spec, *algo)?;
            replay_tar_index_into_ext(&spec, *algo, &index, dst_dev, dst)
        }
        Source::TarArchive { path, codec: None } => {
            let target = crate::inspect::Target::parse(&path.to_string_lossy());
            copy_image_into_ext(&target, dst_dev, dst)
        }
        Source::Image(target) => copy_image_into_ext(target, dst_dev, dst),
    }
}

/// Populate `dst` (a freshly formatted FAT32) with the contents of
/// `source`. The destination is assumed to already exist.
pub fn populate_fat32_from_source(
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::fat::Fat32,
    source: &Source,
) -> Result<()> {
    match source {
        Source::HostDir(p) => dst.populate_from_host_dir(dst_dev, p),
        Source::TarArchive {
            path,
            codec: Some(algo),
        } => {
            let spec = path.to_string_lossy().into_owned();
            let index = build_tar_stream_index(&spec, *algo)?;
            replay_tar_index_into_fat(&spec, *algo, &index, dst_dev, dst)
        }
        Source::TarArchive { path, codec: None } => {
            let target = crate::inspect::Target::parse(&path.to_string_lossy());
            copy_image_into_fat32(&target, dst_dev, dst)
        }
        Source::Image(target) => copy_image_into_fat32(target, dst_dev, dst),
    }
}

/// Build a [`BuildPlan`](crate::fs::ext::BuildPlan) sized for the
/// source. Walks the source once and feeds entry counts + byte totals
/// into the plan; the resulting `to_format_opts()` is ready to drive
/// `Ext::format_with`.
pub fn ext_build_plan_for_source(
    source: &Source,
    block_size: u32,
    kind: FsKind,
) -> Result<crate::fs::ext::BuildPlan> {
    let mut plan = crate::fs::ext::BuildPlan::new(block_size, kind);
    match source {
        Source::HostDir(p) => plan.scan_host_path(p)?,
        Source::TarArchive {
            path,
            codec: Some(algo),
        } => {
            let spec = path.to_string_lossy().into_owned();
            let index = build_tar_stream_index(&spec, *algo)?;
            walk_tar_index_for_plan(&index, &mut plan);
        }
        Source::TarArchive { path, codec: None } => {
            let target = crate::inspect::Target::parse(&path.to_string_lossy());
            crate::inspect::with_target_device(&target, |src_dev| {
                let src_fs = crate::inspect::AnyFs::open(src_dev)?;
                build_ext_plan_inner(src_dev, &src_fs, &mut plan)
            })?;
        }
        Source::Image(target) => {
            crate::inspect::with_target_device(target, |src_dev| {
                let src_fs = crate::inspect::AnyFs::open(src_dev)?;
                build_ext_plan_inner(src_dev, &src_fs, &mut plan)
            })?;
        }
    }
    Ok(plan)
}

/// Populate any [`crate::fs::Filesystem`] from `source`, dispatching
/// through the trait for every entry create. Works for HFS+, NTFS,
/// F2FS, SquashFS, XFS — whatever implements the trait — though
/// ext/FAT32 callers should prefer [`populate_ext_from_source`] /
/// [`populate_fat32_from_source`] which preserve xattrs and use the
/// per-FS fast paths.
///
/// Tar-archive and existing-image sources route through an internal
/// helper that opens the source via
/// [`crate::inspect::AnyFs`] and replays entries through trait
/// methods.
pub fn populate_fs_from_source<F: crate::fs::Filesystem>(
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut F,
    source: &Source,
) -> Result<()> {
    populate_fs_from_source_dyn(dst_dev, dst, source)
}

/// Trait-object form of [`populate_fs_from_source`]. Used by code
/// paths (e.g. [`crate::inspect::AnyFs`] dispatch helpers) that have
/// a `&mut dyn Filesystem` rather than a known concrete type.
pub fn populate_fs_from_source_dyn(
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut dyn crate::fs::Filesystem,
    source: &Source,
) -> Result<()> {
    match source {
        Source::HostDir(p) => populate_host_dir_via_trait(dst_dev, dst, p),
        Source::TarArchive {
            path,
            codec: Some(_algo),
        } => {
            // Compressed tar: open as image via AnyFs::Tar isn't easy
            // (we'd need to decompress first). For now, the generic
            // path supports plain tar via the image walker below.
            let target = crate::inspect::Target::parse(&path.to_string_lossy());
            populate_image_via_trait(&target, dst_dev, dst)
        }
        Source::TarArchive { path, codec: None } => {
            let target = crate::inspect::Target::parse(&path.to_string_lossy());
            populate_image_via_trait(&target, dst_dev, dst)
        }
        Source::Image(target) => populate_image_via_trait(target, dst_dev, dst),
    }
}

/// Recursively copy a host directory tree into `dst` via the trait.
fn populate_host_dir_via_trait(
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut dyn crate::fs::Filesystem,
    root: &Path,
) -> Result<()> {
    // Stack of (host dir, fs path).
    let mut stack: Vec<(PathBuf, String)> = vec![(root.to_path_buf(), "/".to_string())];
    while let Some((dir, fs_dir)) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_str().ok_or_else(|| {
                crate::Error::InvalidArgument(format!("repack: non-UTF-8 host filename {:?}", name))
            })?;
            let dest = join_fs_path(&fs_dir, name_str);
            let dest_path = std::path::Path::new(&dest);
            let meta = entry.metadata()?;
            let ft = meta.file_type();
            let fmeta = host_meta_to_fs(&meta);
            if ft.is_dir() {
                dst.create_dir(dst_dev, dest_path, fmeta)?;
                stack.push((entry.path(), dest));
            } else if ft.is_symlink() {
                let target = std::fs::read_link(entry.path())?;
                dst.create_symlink(dst_dev, dest_path, &target, fmeta)?;
            } else if ft.is_file() {
                let len = meta.len();
                let src = crate::fs::FileSource::HostPath(entry.path());
                dst.create_file(dst_dev, dest_path, src, fmeta)?;
                let _ = len;
            }
        }
    }
    Ok(())
}

/// Open `target` as an image, walk its filesystem, replay entries
/// into `dst` via trait methods.
fn populate_image_via_trait(
    target: &crate::inspect::Target,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut dyn crate::fs::Filesystem,
) -> Result<()> {
    crate::inspect::with_target_device(target, |src_dev| {
        let mut src_fs = crate::inspect::AnyFs::open(src_dev)?;
        let mut stack: Vec<String> = vec!["/".to_string()];
        while let Some(dir) = stack.pop() {
            let entries = src_fs.list(src_dev, &dir)?;
            for e in entries {
                if e.name == "." || e.name == ".." || e.name == "lost+found" {
                    continue;
                }
                let child = join_fs_path(&dir, &e.name);
                let child_path = std::path::Path::new(&child);
                let fmeta = crate::fs::FileMeta::default();
                match e.kind {
                    crate::fs::EntryKind::Dir => {
                        dst.create_dir(dst_dev, child_path, fmeta)?;
                        stack.push(child);
                    }
                    crate::fs::EntryKind::Regular => {
                        // Stream source body through a tempfile so the
                        // destination's create_file can take a HostPath
                        // (works around the trait's source-Read lifetime
                        // borrowing src_dev mutably).
                        let tmp = staging_tempfile(src_dev, &mut src_fs, &child)?;
                        let path = tmp.path().to_path_buf();
                        // Keep tmp alive for the duration of create_file
                        // by binding it; the source path is consumed
                        // before tmp drops.
                        dst.create_file(
                            dst_dev,
                            child_path,
                            crate::fs::FileSource::HostPath(path),
                            fmeta,
                        )?;
                        drop(tmp);
                    }
                    crate::fs::EntryKind::Symlink => {
                        let target_str = read_symlink_via_anyfs(src_dev, &mut src_fs, &child)?;
                        dst.create_symlink(
                            dst_dev,
                            child_path,
                            std::path::Path::new(&target_str),
                            fmeta,
                        )?;
                    }
                    crate::fs::EntryKind::Char
                    | crate::fs::EntryKind::Block
                    | crate::fs::EntryKind::Fifo
                    | crate::fs::EntryKind::Socket => {
                        // Major/minor extraction isn't on AnyFs yet; fall
                        // back to (0, 0) which is enough for FIFO/socket.
                        let dk = match e.kind {
                            crate::fs::EntryKind::Char => crate::fs::DeviceKind::Char,
                            crate::fs::EntryKind::Block => crate::fs::DeviceKind::Block,
                            crate::fs::EntryKind::Fifo => crate::fs::DeviceKind::Fifo,
                            crate::fs::EntryKind::Socket => crate::fs::DeviceKind::Socket,
                            _ => unreachable!(),
                        };
                        let _ = dst.create_device(dst_dev, child_path, dk, 0, 0, fmeta);
                    }
                    crate::fs::EntryKind::Unknown => {
                        eprintln!("repack: skipping unknown entry {child:?}");
                    }
                }
            }
        }
        Ok(())
    })
}

/// Stream a file's body from the source FS into a fresh tempfile and
/// return the handle. Lets the trait-based copier feed
/// `FileSource::HostPath` to the destination without re-borrowing the
/// source device.
fn staging_tempfile(
    src_dev: &mut dyn crate::block::BlockDevice,
    src_fs: &mut crate::inspect::AnyFs,
    fs_path: &str,
) -> Result<tempfile::NamedTempFile> {
    let mut tmp = tempfile::NamedTempFile::new()?;
    src_fs.copy_file_to(src_dev, fs_path, &mut tmp)?;
    tmp.as_file_mut().sync_all()?;
    Ok(tmp)
}

/// Read the target of a symbolic link via the best per-FS reader on
/// `AnyFs`. Different FSes name the reader differently (some take a
/// path, some an inode); we centralise the dispatch here. Sources
/// that don't (yet) expose a symlink reader return `Unsupported`.
fn read_symlink_via_anyfs(
    src_dev: &mut dyn crate::block::BlockDevice,
    src_fs: &mut crate::inspect::AnyFs,
    path: &str,
) -> Result<String> {
    use crate::inspect::AnyFs;
    match src_fs {
        AnyFs::Ext(_) => Err(crate::Error::Unsupported(
            "repack: ext symlinks via the generic walker are not yet wired".into(),
        )),
        AnyFs::Fat32(_) => Err(crate::Error::Unsupported(
            "repack: FAT32 source has no symlinks".into(),
        )),
        AnyFs::Tar(t) => t
            .lookup(path)
            .and_then(|e| e.link_target.clone())
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "repack: tar source has no symlink at {path:?}"
                ))
            }),
        AnyFs::Xfs(x) => x.read_symlink(src_dev, path),
        AnyFs::HfsPlus(h) => h.read_symlink_target_path(src_dev, path),
        AnyFs::Apfs(_) => Err(crate::Error::Unsupported(
            "repack: APFS symlink reading via repack not yet wired".into(),
        )),
        AnyFs::Ntfs(_) => Err(crate::Error::Unsupported(
            "repack: NTFS symlink reading via repack not yet wired".into(),
        )),
        AnyFs::F2fs(_) => Err(crate::Error::Unsupported(
            "repack: F2FS symlink reading via repack not yet wired".into(),
        )),
        AnyFs::Squashfs(s) => s.read_symlink(src_dev, path),
        AnyFs::Exfat(_) => Err(crate::Error::Unsupported(
            "repack: exFAT source has no symlinks".into(),
        )),
    }
}

/// Convert host `Metadata` into a public [`crate::fs::FileMeta`].
fn host_meta_to_fs(meta: &std::fs::Metadata) -> crate::fs::FileMeta {
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    #[cfg(unix)]
    {
        crate::fs::FileMeta {
            mode: (meta.mode() & 0o7777) as u16,
            uid: meta.uid(),
            gid: meta.gid(),
            mtime: meta.mtime() as u32,
            atime: meta.atime() as u32,
            ctime: meta.ctime() as u32,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = meta;
        crate::fs::FileMeta::default()
    }
}

/// Compute the minimum FAT32 byte capacity needed to fit `source`.
/// Bumps to the FAT32 cluster-count minimum + rounds up to a 512-byte
/// sector boundary.
pub fn fat32_min_bytes_for_source(source: &Source) -> Result<u64> {
    let bytes = match source {
        Source::HostDir(p) => sum_host_dir_bytes(p)?,
        Source::TarArchive {
            path,
            codec: Some(algo),
        } => {
            let spec = path.to_string_lossy().into_owned();
            let index = build_tar_stream_index(&spec, *algo)?;
            let (sz, _, _, _, _, _) = size_from_tar_index(&index, "fat32")?;
            return Ok(sz);
        }
        Source::TarArchive { path, codec: None } => {
            let target = crate::inspect::Target::parse(&path.to_string_lossy());
            let mut sum = 0u64;
            crate::inspect::with_target_device(&target, |src_dev| {
                let src_fs = crate::inspect::AnyFs::open(src_dev)?;
                sum = sum_source_file_bytes(src_dev, &src_fs)?;
                Ok(())
            })?;
            sum
        }
        Source::Image(target) => {
            let mut sum = 0u64;
            crate::inspect::with_target_device(target, |src_dev| {
                let src_fs = crate::inspect::AnyFs::open(src_dev)?;
                sum = sum_source_file_bytes(src_dev, &src_fs)?;
                Ok(())
            })?;
            sum
        }
    };
    let needed = bytes
        .saturating_mul(2)
        .max(crate::fs::fat::MIN_FAT32_CLUSTERS as u64 * 1024);
    Ok(needed.div_ceil(512) * 512)
}

fn sum_host_dir_bytes(root: &Path) -> Result<u64> {
    let mut total = 0u64;
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_dir() {
                stack.push(entry.path());
            } else if meta.is_file() {
                total = total.saturating_add(meta.len());
            }
        }
    }
    Ok(total)
}

/// Drive [`walk_ext_for_plan`] / [`walk_fat_for_plan`] /
/// [`walk_tar_for_plan`] based on the source FS type.
fn build_ext_plan_inner(
    src_dev: &mut dyn crate::block::BlockDevice,
    src_fs: &crate::inspect::AnyFs,
    plan: &mut crate::fs::ext::BuildPlan,
) -> Result<()> {
    use crate::inspect::AnyFs;
    match src_fs {
        AnyFs::Ext(src_ext) => walk_ext_for_plan(src_dev, src_ext, 2, plan),
        AnyFs::Fat32(src_fat) => walk_fat_for_plan(src_dev, src_fat, "/", plan),
        AnyFs::Tar(src_tar) => {
            walk_tar_for_plan(src_tar, plan);
            Ok(())
        }
        _ => Err(unsupported_repack_src(src_fs)),
    }
}

/// TarStreamIndex variant of [`walk_tar_for_plan`] — adds one entry
/// of each kind to the build plan for every record in the index.
fn walk_tar_index_for_plan(
    index: &crate::fs::tar::TarStreamIndex,
    plan: &mut crate::fs::ext::BuildPlan,
) {
    use crate::fs::tar::EntryKind as TarKind;
    for ix in index.entries() {
        match ix.entry.kind {
            TarKind::Regular | TarKind::HardLink => plan.add_file(ix.entry.size),
            TarKind::Dir => plan.add_dir(),
            TarKind::Symlink => plan.add_symlink(
                ix.entry
                    .link_target
                    .as_deref()
                    .map(|s| s.len())
                    .unwrap_or(0),
            ),
            TarKind::CharDev | TarKind::BlockDev | TarKind::Fifo => plan.add_device(),
        }
    }
}

// ----------------------------------------------------------------------
// Internal helpers (moved verbatim from src/bin/fstool/main.rs)
// ----------------------------------------------------------------------

/// Walk the source filesystem and recreate every entry inside the
/// destination ext. Preserves mode, uid/gid, mtime, atime, ctime; copies
/// symlinks and device nodes verbatim when the source is ext (FAT
/// source has none of those).
pub(crate) fn copy_into_ext(
    src_dev: &mut dyn crate::block::BlockDevice,
    src_fs: &crate::inspect::AnyFs,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::ext::Ext,
) -> crate::Result<()> {
    use crate::fs::FileMeta;
    use crate::inspect::AnyFs;
    match src_fs {
        AnyFs::Ext(src_ext) => copy_ext_dir(src_dev, src_ext, 2, dst_dev, dst, 2),
        AnyFs::Fat32(src_fat) => {
            copy_fat_dir_into_ext(src_dev, src_fat, "/", dst_dev, dst, 2, &FileMeta::default())
        }
        AnyFs::Tar(src_tar) => copy_tar_into_ext(src_dev, src_tar, dst_dev, dst),
        _ => Err(unsupported_repack_src(src_fs)),
    }
}

/// Walk the source filesystem and recreate every entry inside the
/// destination FAT32. FAT can't represent symlinks / device nodes /
/// per-file permissions — those are dropped (with a stderr note when
/// the source had them).
pub(crate) fn copy_into_fat32(
    src_dev: &mut dyn crate::block::BlockDevice,
    src_fs: &crate::inspect::AnyFs,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::fat::Fat32,
) -> crate::Result<()> {
    use crate::inspect::AnyFs;
    match src_fs {
        AnyFs::Ext(src_ext) => copy_ext_dir_into_fat(src_dev, src_ext, 2, "/", dst_dev, dst),
        AnyFs::Fat32(src_fat) => copy_fat_dir_into_fat(src_dev, src_fat, "/", dst_dev, dst),
        AnyFs::Tar(src_tar) => copy_tar_into_fat(src_dev, src_tar, dst_dev, dst),
        _ => Err(unsupported_repack_src(src_fs)),
    }
}

/// Repack-source error for the four read-only FSes (xfs/exfat/hfs+/apfs)
/// — they're inspectable via ls/cat/info but not yet wired into the
/// FS-to-FS copy walkers.
/// If `path` looks like a compressed tar (`.tar.gz`, `.tar.zst`,
/// `.tar.xz`, `.tgz`, `.txz`, `.tar.lz4`, `.tar.lzma`, `.tar.lzo`),
/// return the codec to use; otherwise `None`. Used by repack to pick a
/// streaming compressor for the output file.
pub(crate) fn tar_output_codec(path: &std::path::Path) -> Option<crate::compression::Algo> {
    let s = path.to_string_lossy().to_ascii_lowercase();
    if s.ends_with(".tgz") {
        return Some(crate::compression::Algo::Gzip);
    }
    if s.ends_with(".txz") {
        return Some(crate::compression::Algo::Xz);
    }
    if !s.contains(".tar.") {
        // Bare `.gz` / `.zst` etc. without a `.tar.` prefix isn't tar.
        return None;
    }
    crate::compression::Algo::from_extension(path)
}

/// `Some(algo)` when `path` points at a compressed tar archive that
/// should be stream-walked rather than decompressed-to-tempfile.
/// `None` for plain `.tar` (the regular BlockDevice path handles it
/// fine) and for non-tar files.
pub(crate) fn tar_input_codec(path: &str) -> Option<crate::compression::Algo> {
    // Strip any `:N` partition selector — tar archives don't have
    // partitions, but the parsing helper allows the form.
    let p = std::path::Path::new(path.split(':').next().unwrap_or(path));
    tar_output_codec(p)
}

/// Open `src` as a freshly-decoded `Read` positioned at the
/// decompressed stream's byte 0. Boxed so it composes with the
/// existing helpers; callers feed this to [`TarStreamIndex::open_body`]
/// to seek to a specific entry's body offset.
pub(crate) fn open_decoded_stream(
    src: &str,
    algo: crate::compression::Algo,
) -> crate::Result<Box<dyn std::io::Read>> {
    let p = std::path::Path::new(src.split(':').next().unwrap_or(src));
    let file = std::fs::File::open(p)?;
    let buffered: Box<dyn std::io::Read> =
        Box::new(std::io::BufReader::with_capacity(64 * 1024, file));
    crate::compression::make_reader(algo, buffered)
}

/// Single-pass walk that builds a [`TarStreamIndex`] for a compressed
/// tar source. Bodies are NOT consumed: the underlying reader skips
/// past each body's bytes during `next_entry`, so the only buffered
/// data is the per-entry metadata.
pub(crate) fn build_tar_stream_index(
    src: &str,
    algo: crate::compression::Algo,
) -> crate::Result<crate::fs::tar::TarStreamIndex> {
    let reader = open_tar_stream_reader(src, Some(algo))?;
    crate::fs::tar::TarStreamIndex::build_from(reader)
}

/// Aggregate the size-relevant counters from a built [`TarStreamIndex`]
/// and return `(size_estimate, files, dirs, symlinks, devices, bytes)`.
/// `target_lower` tunes the size estimate per destination FS.
pub(crate) fn size_from_tar_index(
    index: &crate::fs::tar::TarStreamIndex,
    target_lower: &str,
) -> crate::Result<(u64, u64, u64, u64, u64, u64)> {
    use crate::fs::tar::EntryKind as TarKind;
    let mut files = 0u64;
    let mut dirs = 0u64;
    let mut symlinks = 0u64;
    let mut devices = 0u64;
    let mut bytes = 0u64;
    for ix in index.entries() {
        match ix.entry.kind {
            TarKind::Regular => {
                files += 1;
                bytes += ix.entry.size;
            }
            TarKind::HardLink => {
                files += 1;
                bytes += ix.entry.size;
            }
            TarKind::Dir => dirs += 1,
            TarKind::Symlink => symlinks += 1,
            TarKind::CharDev | TarKind::BlockDev | TarKind::Fifo => devices += 1,
        }
    }
    let size_estimate = match target_lower {
        "ext2" | "ext3" | "ext4" => {
            // Conservative ext sizing: file bytes + dir/inode overhead.
            // We give 4 KiB per inode + 1 MiB structural pad; min 8 MiB.
            let inodes = files + dirs + symlinks + devices + 16;
            let raw = bytes + inodes * 4096 + 1024 * 1024;
            raw.max(8 * 1024 * 1024).div_ceil(4096) * 4096
        }
        "fat32" | "vfat" => {
            // FAT32 needs at least MIN_FAT32_CLUSTERS clusters of 1 KiB
            // overhead per cluster. Double the byte total to leave room
            // for cluster fragmentation + FAT tables + dir entries.
            let needed = bytes
                .saturating_mul(2)
                .max(crate::fs::fat::MIN_FAT32_CLUSTERS as u64 * 1024);
            needed.div_ceil(512) * 512
        }
        _ => bytes + 16 * 1024 * 1024,
    };
    Ok((size_estimate, files, dirs, symlinks, devices, bytes))
}

/// Pass 2 (ext): replay the indexed entries into a freshly-formatted
/// ext destination, re-decompressing per regular file via
/// `TarStreamIndex::open_body`.
pub(crate) fn replay_tar_index_into_ext(
    src: &str,
    algo: crate::compression::Algo,
    index: &crate::fs::tar::TarStreamIndex,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::ext::Ext,
) -> crate::Result<()> {
    use crate::fs::tar::EntryKind as TarKind;
    use crate::fs::{DeviceKind, FileMeta};
    let mut path_to_ino: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    path_to_ino.insert("/".into(), 2);

    for ix in index.entries() {
        let e = &ix.entry;
        let parent_path = parent_of(&e.path);
        let parent_ino = ensure_ext_dir(dst_dev, dst, &mut path_to_ino, &parent_path)?;
        let leaf = leaf_of(&e.path);
        let meta = FileMeta {
            mode: e.mode & 0o7777,
            uid: e.uid,
            gid: e.gid,
            mtime: e.mtime as u32,
            atime: e.mtime as u32,
            ctime: e.mtime as u32,
        };
        let new_ino = match e.kind {
            TarKind::Regular => {
                let mut body = index.open_body(&e.path, || open_decoded_stream(src, algo))?;
                dst.add_file_to_streaming(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    &mut body,
                    e.size,
                    meta,
                )?
            }
            TarKind::HardLink => {
                // Materialise the linked target's content. Preserving
                // ext hard-link semantics across FS types is out of
                // scope; we copy the bytes instead.
                let mut body = index.open_body(&e.path, || open_decoded_stream(src, algo))?;
                let len = body.remaining();
                dst.add_file_to_streaming(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    &mut body,
                    len,
                    meta,
                )?
            }
            TarKind::Dir => ensure_ext_dir(dst_dev, dst, &mut path_to_ino, &e.path)?,
            TarKind::Symlink => {
                let target = e.link_target.as_deref().unwrap_or("");
                dst.add_symlink_to(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    target.as_bytes(),
                    meta,
                )?
            }
            TarKind::CharDev => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Char,
                e.device_major,
                e.device_minor,
                meta,
            )?,
            TarKind::BlockDev => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Block,
                e.device_major,
                e.device_minor,
                meta,
            )?,
            TarKind::Fifo => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Fifo,
                0,
                0,
                meta,
            )?,
        };
        if matches!(e.kind, TarKind::Dir) {
            path_to_ino.insert(e.path.clone(), new_ino);
        }
        if !e.xattrs.is_empty() {
            dst.set_xattrs(dst_dev, new_ino, &e.xattrs)?;
        }
    }
    Ok(())
}

/// Pass 2 (FAT32): same as the ext replay, minus the metadata FAT
/// can't carry. Entries that aren't regular / dir / hard-link are
/// dropped with a stderr note.
pub(crate) fn replay_tar_index_into_fat(
    src: &str,
    algo: crate::compression::Algo,
    index: &crate::fs::tar::TarStreamIndex,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::fat::Fat32,
) -> crate::Result<()> {
    use crate::fs::tar::EntryKind as TarKind;
    let mut made_dirs: std::collections::HashSet<String> =
        std::collections::HashSet::from(["/".into()]);
    for ix in index.entries() {
        let e = &ix.entry;
        let parent = parent_of(&e.path);
        ensure_fat_dir(dst_dev, dst, &mut made_dirs, &parent)?;
        match e.kind {
            TarKind::Regular => {
                let mut body = index.open_body(&e.path, || open_decoded_stream(src, algo))?;
                dst.add_file_from_reader(dst_dev, &e.path, &mut body, e.size)?;
            }
            TarKind::HardLink => {
                let mut body = index.open_body(&e.path, || open_decoded_stream(src, algo))?;
                let len = body.remaining();
                dst.add_file_from_reader(dst_dev, &e.path, &mut body, len)?;
            }
            TarKind::Dir => {
                ensure_fat_dir(dst_dev, dst, &mut made_dirs, &e.path)?;
            }
            _ => {
                eprintln!(
                    "repack: dropping {:?} — FAT32 can't represent {:?}",
                    e.path, e.kind
                );
            }
        }
    }
    Ok(())
}

/// Open a (possibly codec-wrapped) tar archive as a streaming reader.
pub(crate) fn open_tar_stream_reader(
    path: &str,
    algo: Option<crate::compression::Algo>,
) -> crate::Result<crate::fs::tar::TarStreamReader<Box<dyn std::io::Read>>> {
    let p = std::path::Path::new(path.split(':').next().unwrap_or(path));
    let file = std::fs::File::open(p)?;
    let buffered: Box<dyn std::io::Read> =
        Box::new(std::io::BufReader::with_capacity(64 * 1024, file));
    let inner: Box<dyn std::io::Read> = match algo {
        Some(a) => crate::compression::make_reader(a, buffered)?,
        None => buffered,
    };
    Ok(crate::fs::tar::TarStreamReader::new(inner))
}

/// Open the tar source (optionally codec-wrapped) and build a
/// random-access index over it. Shared entry point for the
/// streaming-tar inspector commands.
pub fn open_tar_stream_index(
    image: &str,
    algo: Option<crate::compression::Algo>,
) -> crate::Result<crate::fs::tar::TarStreamIndex> {
    let reader = open_tar_stream_reader(image, algo)?;
    crate::fs::tar::TarStreamIndex::build_from(reader)
}

pub(crate) fn unsupported_repack_src(src_fs: &crate::inspect::AnyFs) -> crate::Error {
    crate::Error::Unsupported(format!(
        "repack: {} source is not yet wired into the FS-to-FS copy path (it's inspectable via `ls`/`cat`/`info` but can't yet be a repack source)",
        src_fs.kind_string()
    ))
}

// ─── ext → ext (full metadata preservation) ─────────────────────────────

pub(crate) fn copy_ext_dir(
    src_dev: &mut dyn crate::block::BlockDevice,
    src: &crate::fs::ext::Ext,
    src_ino: u32,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::ext::Ext,
    dst_ino: u32,
) -> crate::Result<()> {
    use crate::fs::ext::inode::decode_devnum;
    use crate::fs::{DeviceKind, FileMeta};

    let entries = src.list_inode(src_dev, src_ino)?;
    for e in entries {
        if e.name == "." || e.name == ".." || (src_ino == 2 && e.name == "lost+found") {
            continue;
        }
        let inode = src.read_inode(src_dev, e.inode)?;
        let meta = FileMeta {
            mode: inode.mode & 0o7777,
            uid: inode.uid as u32,
            gid: inode.gid as u32,
            mtime: inode.mtime,
            atime: inode.atime,
            ctime: inode.ctime,
        };
        let name = e.name.as_bytes();
        let mode_type = inode.mode & crate::fs::ext::constants::S_IFMT;
        // Read source xattrs once per entry; preserve them across the
        // create + (optional) recursion.
        let xattrs = src.read_xattrs(src_dev, e.inode)?;
        let new_ino = match mode_type {
            t if t == crate::fs::ext::constants::S_IFREG => {
                let mut reader = src.open_file_reader(src_dev, e.inode)?;
                dst.add_file_to_streaming(
                    dst_dev,
                    dst_ino,
                    name,
                    &mut reader,
                    inode.size as u64,
                    meta,
                )?
            }
            t if t == crate::fs::ext::constants::S_IFDIR => {
                let child_ino = dst.add_dir_to(dst_dev, dst_ino, name, meta)?;
                copy_ext_dir(src_dev, src, e.inode, dst_dev, dst, child_ino)?;
                child_ino
            }
            t if t == crate::fs::ext::constants::S_IFLNK => {
                let target = src.read_symlink_target(src_dev, e.inode)?;
                dst.add_symlink_to(dst_dev, dst_ino, name, target.as_bytes(), meta)?
            }
            t if t == crate::fs::ext::constants::S_IFCHR => {
                let (major, minor) = decode_devnum(inode.block[0]);
                dst.add_device_to(dst_dev, dst_ino, name, DeviceKind::Char, major, minor, meta)?
            }
            t if t == crate::fs::ext::constants::S_IFBLK => {
                let (major, minor) = decode_devnum(inode.block[0]);
                dst.add_device_to(
                    dst_dev,
                    dst_ino,
                    name,
                    DeviceKind::Block,
                    major,
                    minor,
                    meta,
                )?
            }
            t if t == crate::fs::ext::constants::S_IFIFO => {
                dst.add_device_to(dst_dev, dst_ino, name, DeviceKind::Fifo, 0, 0, meta)?
            }
            t if t == crate::fs::ext::constants::S_IFSOCK => {
                dst.add_device_to(dst_dev, dst_ino, name, DeviceKind::Socket, 0, 0, meta)?
            }
            _ => {
                eprintln!(
                    "repack: skipping inode {} ({:?}) — unknown mode {:#o}",
                    e.inode, e.name, inode.mode
                );
                continue;
            }
        };
        if !xattrs.is_empty() {
            dst.set_xattrs(dst_dev, new_ino, &xattrs)?;
        }
    }
    Ok(())
}

// ─── FAT32 → FAT32 ──────────────────────────────────────────────────────

pub(crate) fn copy_fat_dir_into_fat(
    src_dev: &mut dyn crate::block::BlockDevice,
    src: &crate::fs::fat::Fat32,
    src_path: &str,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::fat::Fat32,
) -> crate::Result<()> {
    use crate::fs::EntryKind;
    let entries = src.list_path(src_dev, src_path)?;
    for e in entries {
        let child = join_fs_path(src_path, &e.name);
        match e.kind {
            EntryKind::Dir => {
                dst.add_dir(dst_dev, &child)?;
                copy_fat_dir_into_fat(src_dev, src, &child, dst_dev, dst)?;
            }
            EntryKind::Regular => {
                // Resolve the source entry to get its actual file_size.
                let (entry, _) = src.resolve_entry(src_dev, &child)?;
                let mut reader = src.open_file_reader(src_dev, &child)?;
                dst.add_file_from_reader(dst_dev, &child, &mut reader, entry.file_size as u64)?;
            }
            _ => {} // FAT can't carry anything else
        }
    }
    Ok(())
}

// ─── ext → FAT32 (drops metadata FAT can't store) ───────────────────────

pub(crate) fn copy_ext_dir_into_fat(
    src_dev: &mut dyn crate::block::BlockDevice,
    src: &crate::fs::ext::Ext,
    src_ino: u32,
    cur_path: &str,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::fat::Fat32,
) -> crate::Result<()> {
    let entries = src.list_inode(src_dev, src_ino)?;
    for e in entries {
        if e.name == "." || e.name == ".." || (src_ino == 2 && e.name == "lost+found") {
            continue;
        }
        let inode = src.read_inode(src_dev, e.inode)?;
        let mode_type = inode.mode & crate::fs::ext::constants::S_IFMT;
        let child = join_fs_path(cur_path, &e.name);
        match mode_type {
            t if t == crate::fs::ext::constants::S_IFREG => {
                let mut reader = src.open_file_reader(src_dev, e.inode)?;
                dst.add_file_from_reader(dst_dev, &child, &mut reader, inode.size as u64)?;
            }
            t if t == crate::fs::ext::constants::S_IFDIR => {
                dst.add_dir(dst_dev, &child)?;
                copy_ext_dir_into_fat(src_dev, src, e.inode, &child, dst_dev, dst)?;
            }
            t if t == crate::fs::ext::constants::S_IFLNK
                || t == crate::fs::ext::constants::S_IFCHR
                || t == crate::fs::ext::constants::S_IFBLK
                || t == crate::fs::ext::constants::S_IFIFO
                || t == crate::fs::ext::constants::S_IFSOCK =>
            {
                eprintln!(
                    "repack: dropping {child:?} ({:?}) — FAT32 can't represent it",
                    fstool_mode_kind(mode_type)
                );
            }
            _ => {}
        }
    }
    Ok(())
}

// ─── FAT32 → ext ────────────────────────────────────────────────────────

pub(crate) fn copy_fat_dir_into_ext(
    src_dev: &mut dyn crate::block::BlockDevice,
    src: &crate::fs::fat::Fat32,
    src_path: &str,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::ext::Ext,
    dst_ino: u32,
    meta: &crate::fs::FileMeta,
) -> crate::Result<()> {
    use crate::fs::EntryKind;
    let entries = src.list_path(src_dev, src_path)?;
    for e in entries {
        let child = join_fs_path(src_path, &e.name);
        match e.kind {
            EntryKind::Dir => {
                let new_ino = dst.add_dir_to(dst_dev, dst_ino, e.name.as_bytes(), *meta)?;
                copy_fat_dir_into_ext(src_dev, src, &child, dst_dev, dst, new_ino, meta)?;
            }
            EntryKind::Regular => {
                let (entry, _) = src.resolve_entry(src_dev, &child)?;
                let mut reader = src.open_file_reader(src_dev, &child)?;
                dst.add_file_to_streaming(
                    dst_dev,
                    dst_ino,
                    e.name.as_bytes(),
                    &mut reader,
                    entry.file_size as u64,
                    *meta,
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

// ─── Tar → ext ──────────────────────────────────────────────────────────

/// Replay a tar archive's entries into a fresh ext destination.
/// Preserves mode, uid/gid, mtime, symlinks, device nodes, and xattrs.
pub(crate) fn copy_tar_into_ext(
    src_dev: &mut dyn crate::block::BlockDevice,
    tar: &crate::fs::tar::Tar,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::ext::Ext,
) -> crate::Result<()> {
    use crate::fs::tar::EntryKind;
    use crate::fs::{DeviceKind, FileMeta};
    // Map every absolute path in the tar to its destination inode,
    // creating ancestor dirs on demand so an entry can land before its
    // parent dir appears in the archive.
    let mut path_to_ino: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    path_to_ino.insert("/".into(), 2);

    let entries: Vec<crate::fs::tar::Entry> = tar.entries().to_vec();
    for e in entries {
        let parent_path = parent_of(&e.path);
        let parent_ino = ensure_ext_dir(dst_dev, dst, &mut path_to_ino, &parent_path)?;
        let leaf = leaf_of(&e.path);
        let meta = FileMeta {
            mode: e.mode & 0o7777,
            uid: e.uid,
            gid: e.gid,
            mtime: e.mtime as u32,
            atime: e.mtime as u32,
            ctime: e.mtime as u32,
        };
        let new_ino = match e.kind {
            EntryKind::Regular => {
                let mut reader = tar.open_file_reader(src_dev, &e.path)?;
                dst.add_file_to_streaming(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    &mut reader,
                    e.size,
                    meta,
                )?
            }
            EntryKind::Dir => {
                // ensure_ext_dir already creates it if missing; we just
                // need its inode.
                ensure_ext_dir(dst_dev, dst, &mut path_to_ino, &e.path)?
            }
            EntryKind::Symlink => {
                let target = e.link_target.as_deref().unwrap_or("");
                dst.add_symlink_to(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    target.as_bytes(),
                    meta,
                )?
            }
            EntryKind::HardLink => {
                // Materialise the link target's content again. Preserves
                // file content across the conversion at the cost of a
                // copy; preserving the link itself across FS types is
                // out of scope.
                let target = e.link_target.as_deref().unwrap_or("");
                let abs_target = if target.starts_with('/') {
                    target.to_string()
                } else {
                    format!("/{target}")
                };
                let target_entry = tar.lookup(&abs_target).ok_or_else(|| {
                    crate::Error::InvalidImage(format!(
                        "tar: hard link {:?} → {abs_target:?} (target missing)",
                        e.path
                    ))
                })?;
                let mut reader = tar.open_file_reader(src_dev, &abs_target)?;
                dst.add_file_to_streaming(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    &mut reader,
                    target_entry.size,
                    meta,
                )?
            }
            EntryKind::CharDev => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Char,
                e.device_major,
                e.device_minor,
                meta,
            )?,
            EntryKind::BlockDev => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Block,
                e.device_major,
                e.device_minor,
                meta,
            )?,
            EntryKind::Fifo => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Fifo,
                0,
                0,
                meta,
            )?,
        };
        if matches!(e.kind, EntryKind::Dir) {
            path_to_ino.insert(e.path.clone(), new_ino);
        }
        if !e.xattrs.is_empty() {
            dst.set_xattrs(dst_dev, new_ino, &e.xattrs)?;
        }
    }
    Ok(())
}

pub(crate) fn ensure_ext_dir(
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::ext::Ext,
    path_to_ino: &mut std::collections::HashMap<String, u32>,
    path: &str,
) -> crate::Result<u32> {
    use crate::fs::FileMeta;
    if let Some(&ino) = path_to_ino.get(path) {
        return Ok(ino);
    }
    let parent = parent_of(path);
    let parent_ino = ensure_ext_dir(dst_dev, dst, path_to_ino, &parent)?;
    let leaf = leaf_of(path);
    let meta = FileMeta {
        mode: 0o755,
        ..FileMeta::default()
    };
    let ino = dst.add_dir_to(dst_dev, parent_ino, leaf.as_bytes(), meta)?;
    path_to_ino.insert(path.to_string(), ino);
    Ok(ino)
}

// ─── Tar → FAT32 ────────────────────────────────────────────────────────

pub(crate) fn copy_tar_into_fat(
    src_dev: &mut dyn crate::block::BlockDevice,
    tar: &crate::fs::tar::Tar,
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::fat::Fat32,
) -> crate::Result<()> {
    use crate::fs::tar::EntryKind;
    let mut made_dirs: std::collections::HashSet<String> =
        std::collections::HashSet::from(["/".into()]);
    let entries: Vec<crate::fs::tar::Entry> = tar.entries().to_vec();
    for e in entries {
        let parent = parent_of(&e.path);
        ensure_fat_dir(dst_dev, dst, &mut made_dirs, &parent)?;
        match e.kind {
            EntryKind::Regular => {
                let mut reader = tar.open_file_reader(src_dev, &e.path)?;
                dst.add_file_from_reader(dst_dev, &e.path, &mut reader, e.size)?;
            }
            EntryKind::Dir => {
                ensure_fat_dir(dst_dev, dst, &mut made_dirs, &e.path)?;
            }
            EntryKind::HardLink => {
                let target = e.link_target.as_deref().unwrap_or("");
                let abs_target = if target.starts_with('/') {
                    target.to_string()
                } else {
                    format!("/{target}")
                };
                if let Some(target_entry) = tar.lookup(&abs_target) {
                    let mut reader = tar.open_file_reader(src_dev, &abs_target)?;
                    dst.add_file_from_reader(dst_dev, &e.path, &mut reader, target_entry.size)?;
                }
            }
            _ => {
                eprintln!(
                    "repack: dropping {:?} — FAT32 can't represent {:?}",
                    e.path, e.kind
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn ensure_fat_dir(
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::fat::Fat32,
    made: &mut std::collections::HashSet<String>,
    path: &str,
) -> crate::Result<()> {
    if made.contains(path) {
        return Ok(());
    }
    let parent = parent_of(path);
    ensure_fat_dir(dst_dev, dst, made, &parent)?;
    dst.add_dir(dst_dev, path)?;
    made.insert(path.to_string());
    Ok(())
}

pub(crate) fn parent_of(path: &str) -> String {
    let p = path.trim_end_matches('/');
    match p.rfind('/') {
        Some(0) | None => "/".into(),
        Some(i) => p[..i].into(),
    }
}

pub(crate) fn leaf_of(path: &str) -> &str {
    let p = path.trim_end_matches('/');
    p.rsplit('/').next().unwrap_or(p)
}

pub(crate) fn join_fs_path(parent: &str, leaf: &str) -> String {
    if parent.ends_with('/') {
        format!("{parent}{leaf}")
    } else {
        format!("{parent}/{leaf}")
    }
}

pub(crate) fn fstool_mode_kind(mode_type: u16) -> &'static str {
    use crate::fs::ext::constants::*;
    match mode_type {
        t if t == S_IFLNK => "symlink",
        t if t == S_IFCHR => "char-device",
        t if t == S_IFBLK => "block-device",
        t if t == S_IFIFO => "fifo",
        t if t == S_IFSOCK => "socket",
        _ => "other",
    }
}

// ─── shrink sizing ───────────────────────────────────────────────────────

pub(crate) fn walk_ext_for_plan(
    src_dev: &mut dyn crate::block::BlockDevice,
    src: &crate::fs::ext::Ext,
    src_ino: u32,
    plan: &mut crate::fs::ext::BuildPlan,
) -> crate::Result<()> {
    let entries = src.list_inode(src_dev, src_ino)?;
    for e in entries {
        if e.name == "." || e.name == ".." || (src_ino == 2 && e.name == "lost+found") {
            continue;
        }
        let inode = src.read_inode(src_dev, e.inode)?;
        let mode_type = inode.mode & crate::fs::ext::constants::S_IFMT;
        match mode_type {
            t if t == crate::fs::ext::constants::S_IFREG => plan.add_file(inode.size as u64),
            t if t == crate::fs::ext::constants::S_IFDIR => {
                plan.add_dir();
                walk_ext_for_plan(src_dev, src, e.inode, plan)?;
            }
            t if t == crate::fs::ext::constants::S_IFLNK => plan.add_symlink(inode.size as usize),
            t if t
                == crate::fs::ext::constants::S_IFCHR
                    | crate::fs::ext::constants::S_IFBLK
                    | crate::fs::ext::constants::S_IFIFO
                    | crate::fs::ext::constants::S_IFSOCK =>
            {
                plan.add_device()
            }
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn walk_fat_for_plan(
    src_dev: &mut dyn crate::block::BlockDevice,
    src: &crate::fs::fat::Fat32,
    src_path: &str,
    plan: &mut crate::fs::ext::BuildPlan,
) -> crate::Result<()> {
    use crate::fs::EntryKind;
    let entries = src.list_path(src_dev, src_path)?;
    for e in entries {
        let child = join_fs_path(src_path, &e.name);
        match e.kind {
            EntryKind::Dir => {
                plan.add_dir();
                walk_fat_for_plan(src_dev, src, &child, plan)?;
            }
            EntryKind::Regular => {
                let (entry, _) = src.resolve_entry(src_dev, &child)?;
                plan.add_file(entry.file_size as u64);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Sum the size of every regular file in the source filesystem — used
/// by FAT32 shrink sizing.
pub(crate) fn sum_source_file_bytes(
    src_dev: &mut dyn crate::block::BlockDevice,
    src_fs: &crate::inspect::AnyFs,
) -> crate::Result<u64> {
    use crate::inspect::AnyFs;
    match src_fs {
        AnyFs::Ext(src_ext) => sum_ext_file_bytes(src_dev, src_ext, 2),
        AnyFs::Fat32(src_fat) => sum_fat_file_bytes(src_dev, src_fat, "/"),
        AnyFs::Tar(src_tar) => Ok(src_tar
            .entries()
            .iter()
            .filter(|e| matches!(e.kind, crate::fs::tar::EntryKind::Regular))
            .map(|e| e.size)
            .sum()),
        _ => Err(unsupported_repack_src(src_fs)),
    }
}

/// Fold a tar archive's entries into a BuildPlan suitable for sizing an
/// ext destination.
pub(crate) fn walk_tar_for_plan(tar: &crate::fs::tar::Tar, plan: &mut crate::fs::ext::BuildPlan) {
    use crate::fs::tar::EntryKind;
    for e in tar.entries() {
        match e.kind {
            EntryKind::Regular | EntryKind::HardLink => plan.add_file(e.size),
            EntryKind::Dir => plan.add_dir(),
            EntryKind::Symlink => {
                plan.add_symlink(e.link_target.as_deref().map(|s| s.len()).unwrap_or(0))
            }
            EntryKind::CharDev | EntryKind::BlockDev | EntryKind::Fifo => plan.add_device(),
        }
    }
}

pub(crate) fn sum_ext_file_bytes(
    src_dev: &mut dyn crate::block::BlockDevice,
    src: &crate::fs::ext::Ext,
    src_ino: u32,
) -> crate::Result<u64> {
    let mut total = 0u64;
    for e in src.list_inode(src_dev, src_ino)? {
        if e.name == "." || e.name == ".." || (src_ino == 2 && e.name == "lost+found") {
            continue;
        }
        let inode = src.read_inode(src_dev, e.inode)?;
        let mode_type = inode.mode & crate::fs::ext::constants::S_IFMT;
        if mode_type == crate::fs::ext::constants::S_IFREG {
            total += inode.size as u64;
        } else if mode_type == crate::fs::ext::constants::S_IFDIR {
            total += sum_ext_file_bytes(src_dev, src, e.inode)?;
        }
    }
    Ok(total)
}

pub(crate) fn sum_fat_file_bytes(
    src_dev: &mut dyn crate::block::BlockDevice,
    src: &crate::fs::fat::Fat32,
    src_path: &str,
) -> crate::Result<u64> {
    use crate::fs::EntryKind;
    let mut total = 0u64;
    for e in src.list_path(src_dev, src_path)? {
        let child = join_fs_path(src_path, &e.name);
        match e.kind {
            EntryKind::Regular => {
                let (entry, _) = src.resolve_entry(src_dev, &child)?;
                total += entry.file_size as u64;
            }
            EntryKind::Dir => total += sum_fat_file_bytes(src_dev, src, &child)?,
            _ => {}
        }
    }
    Ok(total)
}
