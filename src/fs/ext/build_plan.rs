//! `BuildPlan` — pre-scans an input tree and computes recommended sizing
//! (`inodes_count`, `blocks_count`) so the resulting image is exactly large
//! enough to hold the contents plus metadata overhead. Equivalent to
//! `genext2fs -d <dir>` invoked without `-b` / `-N`.
//!
//! The scan reads only metadata (sizes, types, target lengths). File
//! contents are not loaded.

use std::fs;
use std::os::unix::fs::FileTypeExt;
use std::path::Path;

use super::FsKind;
use super::constants::{GROUP_DESC_SIZE, INODE_SIZE_DYNAMIC, N_DIRECT};
use crate::Result;

/// Accumulates the resource needs of an ext filesystem build. After
/// scanning, [`inodes_count`](Self::inodes_count) and
/// [`blocks_count`](Self::blocks_count) return numbers suitable for
/// [`super::FormatOpts`].
#[derive(Debug, Clone)]
pub struct BuildPlan {
    pub block_size: u32,
    pub kind: FsKind,
    pub create_lost_found: bool,
    /// Journal size in blocks (0 → 1024-block default when `kind.has_journal()`).
    pub journal_blocks: u32,

    n_files: u32,
    n_dirs: u32,
    n_symlinks: u32,
    n_devices: u32,
    /// Total data blocks for file contents (rounded up to block size per file).
    data_blocks_files: u64,
    /// Approximate indirect / double-indirect overhead.
    indirect_overhead: u64,
    /// Symlinks whose target is too long for inline storage and need a data block.
    long_symlinks: u32,
}

impl BuildPlan {
    /// Create an empty plan. `kind` controls whether a journal is reserved
    /// (ext3 / ext4) and which feature flags downstream `FormatOpts` carries.
    pub fn new(block_size: u32, kind: FsKind) -> Self {
        Self {
            block_size,
            kind,
            create_lost_found: true,
            journal_blocks: 0,
            n_files: 0,
            n_dirs: 0,
            n_symlinks: 0,
            n_devices: 0,
            data_blocks_files: 0,
            indirect_overhead: 0,
            long_symlinks: 0,
        }
    }

    /// Record a regular file of the given size.
    pub fn add_file(&mut self, size: u64) {
        self.n_files += 1;
        if size == 0 {
            return;
        }
        let bs = self.block_size as u64;
        let blocks = size.div_ceil(bs);
        self.data_blocks_files += blocks;
        self.indirect_overhead += indirect_blocks_for(blocks, bs);
    }

    /// Record a directory entry. Assumes the directory's contents fit in a
    /// single data block (v1 limitation).
    pub fn add_dir(&mut self) {
        self.n_dirs += 1;
    }

    /// Record a symbolic link with a target of `target_len` bytes. Targets
    /// ≤ 60 bytes use the inline "fast symlink" path (no data block); longer
    /// ones get one block.
    pub fn add_symlink(&mut self, target_len: usize) {
        self.n_symlinks += 1;
        if target_len > 60 {
            self.long_symlinks += 1;
        }
    }

    /// Record a special file (char / block / FIFO / socket). No data blocks.
    pub fn add_device(&mut self) {
        self.n_devices += 1;
    }

    /// Walk a host directory recursively, calling the appropriate `add_*`
    /// method for each entry. Follows file metadata but never opens the
    /// file's bytes.
    pub fn scan_host_path<P: AsRef<Path>>(&mut self, src: P) -> Result<()> {
        self.scan_inner(src.as_ref())
    }

    fn scan_inner(&mut self, src: &Path) -> Result<()> {
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            let ft = meta.file_type();
            if ft.is_dir() {
                self.add_dir();
                self.scan_inner(&entry.path())?;
            } else if ft.is_file() {
                self.add_file(meta.len());
            } else if ft.is_symlink() {
                let target = fs::read_link(entry.path())?;
                let tl = target.to_string_lossy().len();
                self.add_symlink(tl);
            } else if ft.is_block_device() || ft.is_char_device() || ft.is_fifo() || ft.is_socket()
            {
                self.add_device();
            } else {
                // Skip unknown types.
            }
        }
        Ok(())
    }

    /// Recommended `inodes_count` for this plan, rounded up to a multiple
    /// of 8 (the bitmap byte alignment).
    pub fn inodes_count(&self) -> u32 {
        // Reserved inodes 1..=10 plus optional lost+found (inode 11) plus
        // every user entry.
        let user = self.n_files + self.n_dirs + self.n_symlinks + self.n_devices;
        let lf = if self.create_lost_found { 1 } else { 0 };
        let needed = 10 + lf + user;
        needed.div_ceil(8) * 8
    }

    /// Recommended `blocks_count` for this plan. Conservative — biased
    /// slightly larger than the strict minimum so a margin of error doesn't
    /// produce a "too small" failure.
    pub fn blocks_count(&self) -> u32 {
        let bs = self.block_size as u64;
        let bs_u32 = self.block_size;

        // Dir data blocks: one per dir, plus lost+found.
        let dir_data = self.n_dirs as u64;
        let lf_data = if self.create_lost_found {
            16384u64.div_ceil(bs)
        } else {
            0
        };
        let lf_indirect = if self.create_lost_found {
            indirect_blocks_for(lf_data, bs)
        } else {
            0
        };

        // Symlink data blocks.
        let symlink_data = self.long_symlinks as u64;

        // Journal blocks + indirection.
        let journal = if self.kind.has_journal() {
            if self.journal_blocks == 0 {
                1024
            } else {
                self.journal_blocks
            }
        } else {
            0
        };
        let journal_indirect = indirect_blocks_for(journal as u64, bs);

        let data = self.data_blocks_files
            + self.indirect_overhead
            + dir_data
            + lf_data
            + lf_indirect
            + symlink_data
            + journal as u64
            + journal_indirect;

        let inodes = self.inodes_count();
        let inode_table = (inodes as u64 * INODE_SIZE_DYNAMIC as u64).div_ceil(bs);

        // Single-group estimate: 1 SB + GDT + 2 bitmaps + inode table + data.
        // GDT is at minimum 1 block.
        let gdt_blocks = (GROUP_DESC_SIZE as u64).div_ceil(bs); // 1 group
        let per_group_meta = 1 + gdt_blocks + 2 + inode_table;

        let first_block = if bs_u32 == 1024 { 1 } else { 0 };
        // The boot block (for 1 KiB blocks) is outside the bitmap-tracked
        // region but still counts toward blocks_count, hence `first_block`.
        let total = first_block as u64 + per_group_meta + data;

        // Add 10% slack, floor at 64 blocks. For a single-group FS
        // blocks_per_group == blocks_count, and that must be a multiple of
        // 8 (the block bitmap is checked byte-aligned per group) — so round
        // the final count up to a multiple of 8.
        let with_slack = (total as f64 * 1.10).ceil() as u64;
        let floored = with_slack.max(64);
        (floored.div_ceil(8) * 8) as u32
    }

    /// Build a [`super::FormatOpts`] populated with the recommended counts.
    /// Other fields take their defaults; the caller can override them
    /// before passing to [`super::Ext::format_with`].
    pub fn to_format_opts(&self) -> super::FormatOpts {
        super::FormatOpts {
            kind: self.kind,
            block_size: self.block_size,
            blocks_count: self.blocks_count(),
            inodes_count: self.inodes_count(),
            create_lost_found: self.create_lost_found,
            journal_blocks: self.journal_blocks,
            ..super::FormatOpts::default()
        }
    }

    // ─── inspection helpers ────────────────────────────────────────────
    pub fn n_files(&self) -> u32 {
        self.n_files
    }
    pub fn n_dirs(&self) -> u32 {
        self.n_dirs
    }
    pub fn n_symlinks(&self) -> u32 {
        self.n_symlinks
    }
    pub fn n_devices(&self) -> u32 {
        self.n_devices
    }
}

/// Count the indirection metadata blocks needed to address `n_data` data
/// blocks with the classic direct/single/double indirection scheme.
/// `bs` is the FS block size in bytes; `bs/4` pointers fit per block.
fn indirect_blocks_for(n_data: u64, bs: u64) -> u64 {
    let direct = N_DIRECT as u64;
    if n_data <= direct {
        return 0;
    }
    let ptrs = bs / 4;
    let mut meta = 1; // single-indirect block
    let after_single = n_data - direct;
    if after_single <= ptrs {
        return meta;
    }
    // Double-indirect
    meta += 1; // DIND block
    let remaining = after_single - ptrs;
    meta += remaining.div_ceil(ptrs); // IND blocks under DIND
    meta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_plan_has_floor() {
        let plan = BuildPlan::new(1024, FsKind::Ext2);
        assert!(plan.blocks_count() >= 64);
        // 10 reserved + 1 lost+found + 0 user = 11 → round up to 16
        assert_eq!(plan.inodes_count(), 16);
    }

    #[test]
    fn ext2_with_a_few_files() {
        let mut plan = BuildPlan::new(1024, FsKind::Ext2);
        plan.add_file(5); // 1 block
        plan.add_file(2048); // 2 blocks
        plan.add_dir();
        plan.add_symlink(3); // fast symlink
        plan.add_symlink(120); // slow symlink, 1 data block
        plan.add_device();
        let inodes = plan.inodes_count();
        // 10 reserved + 1 LF + 2 files + 1 dir + 2 symlinks + 1 device = 17 → 24
        assert_eq!(inodes, 24);
        let _ = plan.blocks_count(); // sanity — no precise expected value
    }

    #[test]
    fn ext3_includes_journal() {
        let mut plan = BuildPlan::new(1024, FsKind::Ext3);
        plan.add_file(100);
        let no_journal = {
            let mut p = plan.clone();
            p.kind = FsKind::Ext2;
            p.blocks_count()
        };
        let with_journal = plan.blocks_count();
        // ext3 needs ~1024 extra blocks for the journal
        assert!(with_journal > no_journal + 900);
    }

    #[test]
    fn indirect_blocks_match_layout() {
        // N_DIRECT (12) blocks → no overhead
        assert_eq!(indirect_blocks_for(12, 1024), 0);
        // 13 blocks → 1 IND
        assert_eq!(indirect_blocks_for(13, 1024), 1);
        // 12 + 256 = 268 blocks → 1 IND
        assert_eq!(indirect_blocks_for(268, 1024), 1);
        // 269 blocks → 1 IND + 1 DIND + 1 IND under DIND
        assert_eq!(indirect_blocks_for(269, 1024), 3);
        // 12 + 256 + 256 = 524 blocks → 1 IND + 1 DIND + 1 IND under DIND
        assert_eq!(indirect_blocks_for(524, 1024), 3);
        // 12 + 256 + 257 = 525 blocks → 1 IND + 1 DIND + 2 IND under DIND
        assert_eq!(indirect_blocks_for(525, 1024), 4);
    }
}
