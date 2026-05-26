//! Generic source analysis: walk a filesystem image, tar archive, host
//! directory, or layered source once and report the aggregate metrics
//! that determine how big a destination image needs to be — file / dir /
//! symlink / device counts, total file bytes, and an ext inode/block
//! estimate.
//!
//! This is the single home of the "how big should the destination be"
//! logic that `repack`/`convert` use for `--shrink` sizing
//! ([`Analysis::recommended_size`]), and it backs the `fstool analyze`
//! CLI command. It reuses the existing walk machinery wholesale
//! ([`crate::repack::walk_source_into_sink`] / [`walk_anyfs`]) via a
//! single accumulating [`RepackSink`].
//!
//! [`walk_anyfs`]: crate::repack::walk_anyfs

use std::collections::BTreeMap;
use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::ext::{BuildPlan, FsKind};
use crate::fs::{DeviceKind, XattrPair};
use crate::inspect::AnyFs;
use crate::repack::{RepackMeta, RepackSink, Source, walk_anyfs, walk_source_into_sink};

/// Filesystem types whose destination size [`Analysis::recommended_size`]
/// can compute — the fixed-size block filesystems the `--shrink` path
/// formats to fit. Streamed / archive outputs (tar, zip, cpio, ar, grf,
/// iso) grow-then-truncate and need no content-fit size, so they are
/// deliberately absent.
pub const SIZED_FS_TYPES: &[&str] = &["ext2", "ext3", "ext4", "fat32"];

/// Aggregate result of walking a source. Counts are of directory
/// entries: a hard link is counted under [`hardlinks`](Self::hardlinks),
/// not [`files`](Self::files). The embedded ext [`BuildPlan`] (built with
/// the `block_size` passed to the analyzer) powers the ext inode/block
/// estimates; it is not part of the public surface.
#[derive(Debug, Clone)]
pub struct Analysis {
    pub files: u64,
    pub dirs: u64,
    pub symlinks: u64,
    pub devices: u64,
    /// Hard links to already-seen inodes (extra directory entries beyond
    /// the first link). The walkers dedup these.
    pub hardlinks: u64,
    /// Sum of every regular file's size in bytes.
    pub total_file_bytes: u64,
    pub(crate) plan: BuildPlan,
}

impl Analysis {
    /// Recommended ext-style inode count for the content (reserved +
    /// every entry, rounded to the bitmap alignment). Block-size
    /// independent.
    pub fn inode_count(&self) -> u32 {
        self.plan.inodes_count()
    }

    /// ext block size the analysis was computed with (affects the ext
    /// size estimate, not the counts).
    pub fn block_size(&self) -> u32 {
        self.plan.block_size
    }

    /// Recommended destination image size in bytes for `fs_type`, or
    /// `None` when that destination doesn't take a content-fit size.
    ///
    /// `Some` only for the fixed-size block filesystems in
    /// [`SIZED_FS_TYPES`]: `ext{2,3,4}` (from the [`BuildPlan`]) and
    /// `fat32`/`vfat` (`2× file bytes`, floored at the FAT32 minimum,
    /// rounded to a sector). `None` for streamed / archive outputs
    /// (`tar`, `zip`, `cpio`, `ar`, `grf`, `iso`) and for self-sizing
    /// filesystems not wired into `--shrink`. These formulas mirror the
    /// previous inline `repack` sizing exactly.
    pub fn recommended_size(&self, fs_type: &str) -> Option<u64> {
        let lower = fs_type.to_ascii_lowercase();
        match lower.as_str() {
            "ext2" | "ext3" | "ext4" => {
                let mut p = self.plan.clone();
                p.kind = match lower.as_str() {
                    "ext2" => FsKind::Ext2,
                    "ext3" => FsKind::Ext3,
                    _ => FsKind::Ext4,
                };
                Some(p.blocks_count() as u64 * p.block_size as u64)
            }
            "fat32" | "vfat" => {
                let needed = self
                    .total_file_bytes
                    .saturating_mul(2)
                    .max(crate::fs::fat::MIN_FAT32_CLUSTERS as u64 * 1024);
                Some(needed.div_ceil(512) * 512)
            }
            _ => None,
        }
    }

    /// ext [`FormatOpts`](crate::fs::ext::FormatOpts) sized for this
    /// content as ext flavour `kind` — what the `repack`/`convert` ext
    /// write path formats the destination with.
    pub fn ext_format_opts(&self, kind: FsKind) -> crate::fs::ext::FormatOpts {
        let mut p = self.plan.clone();
        p.kind = kind;
        p.to_format_opts()
    }

    /// Build a serialisable report, including recommended sizes for each
    /// of `fs_types` that has one.
    pub fn report(&self, fs_types: &[&str]) -> AnalysisReport {
        let mut recommended_size = BTreeMap::new();
        for &t in fs_types {
            if let Some(sz) = self.recommended_size(t) {
                recommended_size.insert(t.to_string(), sz);
            }
        }
        AnalysisReport {
            files: self.files,
            dirs: self.dirs,
            symlinks: self.symlinks,
            devices: self.devices,
            hardlinks: self.hardlinks,
            total_file_bytes: self.total_file_bytes,
            inode_count: self.inode_count(),
            block_size: self.block_size(),
            recommended_size,
        }
    }
}

/// Flat, serialisable view of an [`Analysis`] — what `fstool analyze
/// --json` emits and the human formatter consumes.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AnalysisReport {
    pub files: u64,
    pub dirs: u64,
    pub symlinks: u64,
    pub devices: u64,
    pub hardlinks: u64,
    pub total_file_bytes: u64,
    pub inode_count: u32,
    pub block_size: u32,
    /// `fs_type` → recommended destination size in bytes. Only contains
    /// the types from the requested set that yield a size.
    pub recommended_size: BTreeMap<String, u64>,
}

/// A [`RepackSink`] that produces no output and instead accumulates an
/// [`Analysis`]. Supersedes the former `PlanSink` + `ByteSumSink`:
/// `put_*` feed both the entry counts and the ext [`BuildPlan`], so one
/// walk yields every metric the sizing path needs. Bodies are never
/// read (the walkers skip the unread payload), so analyzing a source is
/// metadata-only.
struct AnalysisSink {
    a: Analysis,
}

impl AnalysisSink {
    fn new(block_size: u32) -> Self {
        Self {
            a: Analysis {
                files: 0,
                dirs: 0,
                symlinks: 0,
                devices: 0,
                hardlinks: 0,
                total_file_bytes: 0,
                // Seed kind is neutral: `recommended_size` overrides it
                // per ext flavour, and the counts are kind-independent.
                plan: BuildPlan::new(block_size, FsKind::Ext4),
            },
        }
    }
}

impl RepackSink for AnalysisSink {
    fn put_dir(&mut self, _path: &str, _meta: RepackMeta, _xattrs: &[XattrPair]) -> Result<()> {
        self.a.dirs += 1;
        self.a.plan.add_dir();
        Ok(())
    }
    fn put_file(
        &mut self,
        _path: &str,
        _body: &mut dyn Read,
        len: u64,
        _meta: RepackMeta,
        _xattrs: &[XattrPair],
    ) -> Result<()> {
        self.a.files += 1;
        self.a.total_file_bytes = self.a.total_file_bytes.saturating_add(len);
        self.a.plan.add_file(len);
        Ok(())
    }
    fn put_symlink(
        &mut self,
        _path: &str,
        target: &str,
        _meta: RepackMeta,
        _xattrs: &[XattrPair],
    ) -> Result<()> {
        self.a.symlinks += 1;
        self.a.plan.add_symlink(target.len());
        Ok(())
    }
    fn put_device(
        &mut self,
        _path: &str,
        _kind: DeviceKind,
        _major: u32,
        _minor: u32,
        _meta: RepackMeta,
        _xattrs: &[XattrPair],
    ) -> Result<()> {
        self.a.devices += 1;
        self.a.plan.add_device();
        Ok(())
    }
    fn put_hardlink(
        &mut self,
        _path: &str,
        _target: &str,
        _meta: RepackMeta,
        _xattrs: &[XattrPair],
    ) -> Result<bool> {
        // Over-reserve one inode (matches the former PlanSink upper
        // bound — the write pass shares the target's inode).
        self.a.hardlinks += 1;
        self.a.plan.add_file(0);
        Ok(true)
    }
    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Analyze any [`Source`] — a filesystem image (optionally a `path:N`
/// partition), a plain or compressed tar (streamed, no tempfile), a host
/// directory, or a layered stack. `block_size` parameterises the ext
/// size estimate only.
pub fn analyze_source(source: &Source, block_size: u32) -> Result<Analysis> {
    let mut sink = AnalysisSink::new(block_size);
    walk_source_into_sink(source, &mut sink)?;
    Ok(sink.a)
}

/// Analyze an already-open filesystem (the form `repack`/`convert` use
/// when they have the source mounted for the copy).
pub fn analyze_fs(fs: &mut AnyFs, dev: &mut dyn BlockDevice, block_size: u32) -> Result<Analysis> {
    let mut sink = AnalysisSink::new(block_size);
    walk_anyfs(fs, dev, &mut sink)?;
    Ok(sink.a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repack::RepackMeta;

    fn meta() -> RepackMeta {
        RepackMeta {
            mode: 0o644,
            uid: 0,
            gid: 0,
            mtime: 0,
            atime: 0,
            ctime: 0,
        }
    }

    /// Drive the sink directly with a known set of entries and check the
    /// counts, byte total, inode count, and per-fs recommended sizes.
    #[test]
    fn sink_accumulates_counts_and_sizes() {
        let mut sink = AnalysisSink::new(4096);
        sink.put_dir("/d", meta(), &[]).unwrap();
        sink.put_file("/d/a", &mut std::io::empty(), 4096, meta(), &[])
            .unwrap();
        sink.put_file("/d/b", &mut std::io::empty(), 100, meta(), &[])
            .unwrap();
        sink.put_symlink("/d/l", "a", meta(), &[]).unwrap();
        sink.put_device("/d/dev", DeviceKind::Char, 1, 3, meta(), &[])
            .unwrap();
        sink.put_hardlink("/d/h", "/d/a", meta(), &[]).unwrap();
        let a = sink.a;

        assert_eq!(a.files, 2);
        assert_eq!(a.dirs, 1);
        assert_eq!(a.symlinks, 1);
        assert_eq!(a.devices, 1);
        assert_eq!(a.hardlinks, 1);
        assert_eq!(a.total_file_bytes, 4196);

        // inodes: 10 reserved + 1 lost+found + (2 files + 1 hardlink-inode
        // over-reserve + 1 dir + 1 symlink + 1 device) = 17 → round to 24.
        assert_eq!(a.inode_count(), 24);

        // ext4 size = blocks_count * block_size (matches the plan formula).
        let mut p = a.plan.clone();
        p.kind = FsKind::Ext4;
        assert_eq!(
            a.recommended_size("ext4"),
            Some(p.blocks_count() as u64 * 4096)
        );
        // fat32 = max(2*bytes, MIN*1024) rounded to 512.
        let want_fat = (a.total_file_bytes * 2)
            .max(crate::fs::fat::MIN_FAT32_CLUSTERS as u64 * 1024)
            .div_ceil(512)
            * 512;
        assert_eq!(a.recommended_size("fat32"), Some(want_fat));
    }

    /// Streamed / archive outputs take no content-fit size.
    #[test]
    fn recommended_size_none_for_streamed_outputs() {
        let a = AnalysisSink::new(1024).a;
        for t in ["tar", "zip", "cpio", "ar", "grf", "iso", "iso9660"] {
            assert_eq!(a.recommended_size(t), None, "{t} should be None");
        }
        for t in ["ext2", "ext3", "ext4", "fat32", "vfat"] {
            assert!(a.recommended_size(t).is_some(), "{t} should be Some");
        }
        // ext flavours differ (ext3/4 reserve a journal).
        assert!(a.recommended_size("ext4").unwrap() > a.recommended_size("ext2").unwrap());
    }
}
