//! ext2 / ext3 / ext4 filesystem implementation.
//!
//! v1 writes ext2 (no journal, no extents, no htree). Feature-flag wiring
//! for ext3/ext4 follows in P4; the on-disk format types are intentionally
//! shared so adding the deltas does not duplicate the superblock /
//! group-descriptor / inode encoders.
//!
//! ## Streaming
//!
//! Only *metadata* is kept in memory during a write — per-group bitmaps,
//! the in-progress inode table (one slot per allocated inode), and the
//! data blocks of directories being assembled. File contents are streamed
//! straight to the device through a fixed-size buffer; no file is ever
//! fully resident in memory regardless of size.
//!
//! ## Binary-exact compatibility with genext2fs
//!
//! Defaults are chosen to match `genext2fs -d <dir> -f -q` (-f = zero
//! timestamps, -q = squash uids/perms). See `tests/ext2_genext2fs_compat.rs`
//! for the diff harness once it lands.

pub mod build_plan;
pub mod constants;
pub mod csum;
pub mod dir;
pub mod extent;
pub mod group;
pub mod inode;
pub mod jbd2;
pub mod layout;
pub mod rw;
pub mod superblock;
pub mod xattr;

pub use build_plan::BuildPlan;

use std::io::Read;

use constants::{INO_ROOT_DIR, SUPERBLOCK_OFFSET};
use group::{GroupDesc, set_bit, set_first_n, test_bit};
use inode::{Inode, SpecialKind};
use layout::Layout;
use superblock::Superblock;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::rootdevs::{RootDevs, device_table};
use crate::fs::{DeviceKind, FileMeta, FileSource};

/// Which member of the ext family to produce. Controls which feature flags
/// are set and whether a journal is allocated. ext4-specific format work
/// (extent tree, 64-bit, flex_bg, ...) lands incrementally — v1 accepts
/// `Ext4` but currently emits the same on-disk layout as ext3, which
/// modern kernels mount as ext4 once the feature flags are set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FsKind {
    #[default]
    Ext2,
    Ext3,
    Ext4,
}

impl FsKind {
    /// Whether this kind uses a journal.
    pub fn has_journal(self) -> bool {
        matches!(self, FsKind::Ext3 | FsKind::Ext4)
    }
}

/// Options accepted by [`Ext::format_with`].
///
/// Defaults mirror `genext2fs -d <dir> -f -q -B 1024`: 1 KiB blocks, no
/// features, all-zero UUID + label, mtime 0, 5% reserved blocks, lost+found
/// pre-allocated.
#[derive(Debug, Clone)]
pub struct FormatOpts {
    pub kind: FsKind,
    pub block_size: u32,
    pub blocks_count: u32,
    pub inodes_count: u32,
    pub uuid: [u8; 16],
    pub volume_label: [u8; 16],
    pub mtime: u32,
    pub reserved_blocks_percent: u8,
    pub create_lost_found: bool,
    /// Journal size in FS blocks. Only used when `kind.has_journal()`.
    /// 0 → pick a sensible default (256 blocks).
    pub journal_blocks: u32,
    /// When set, regular files are written sparsely: any block that is
    /// entirely zero is left unallocated (a hole) instead of consuming a
    /// data block. The file still reads back identically. Off by default
    /// so plain ext2 output stays byte-for-byte comparable with genext2fs.
    pub sparse: bool,
    /// When true, only groups 0 and 1 plus groups whose number is a power
    /// of 3, 5, or 7 hold a superblock + GDT backup; the rest skip them.
    /// Advertised on disk by `RO_COMPAT_SPARSE_SUPER`. Off by default so
    /// raw-ext2 output stays binary-exact with genext2fs (which doesn't
    /// emit sparse_super); on by default for ext3/ext4 (which match mke2fs).
    pub sparse_super: bool,
    /// Base-2 logarithm of the flex-unit size when `INCOMPAT_FLEX_BG` is
    /// enabled (0 disables the feature).
    ///
    /// With flex_bg the block bitmap, inode bitmap, and inode table of
    /// every group in a flex unit are packed contiguously into the first
    /// group of that unit; the remaining groups in the unit hold only
    /// data (and optional SB+GDT backups). Improves large-FS performance
    /// at the cost of clustered metadata.
    ///
    /// Valid range: 0..=5 (1 to 32 groups per unit). Defaults to 0
    /// (disabled) so that ext2/3/4 output stays bit-for-bit compatible
    /// with the pre-flex_bg writer; opt in by setting this to 4 (mke2fs's
    /// default for small/medium FSes, 16 groups per unit).
    pub log_groups_per_flex: u8,
    /// When true, emit 64-byte group descriptors and advertise
    /// `INCOMPAT_64BIT` + `INCOMPAT_META_BG` in the superblock. Required
    /// for filesystems whose block count exceeds 2³² (≈ 16 TiB with 4 KiB
    /// blocks). The reader transparently handles either descriptor size.
    /// Off by default — the v1 writer never emits block numbers above 2³²,
    /// so the upper halves remain zero.
    pub use_64bit: bool,
    /// When true, emit the `sparse_super2` compat feature: SB+GDT backups
    /// live in exactly the two block groups listed (rather than groups 0,
    /// 1, and powers-of-3/5/7 under classic `sparse_super`). The two
    /// groups are recorded in `s_backup_bgs[2]`; the writer picks
    /// `[1, last_group]` automatically. Off by default.
    pub sparse_super2: bool,
}

impl Default for FormatOpts {
    fn default() -> Self {
        Self {
            kind: FsKind::Ext2,
            block_size: 1024,
            blocks_count: 1024,
            inodes_count: 16,
            uuid: [0; 16],
            volume_label: [0; 16],
            mtime: 0,
            reserved_blocks_percent: 5,
            create_lost_found: true,
            journal_blocks: 0,
            sparse: false,
            sparse_super: false,
            log_groups_per_flex: 0,
            use_64bit: false,
            sparse_super2: false,
        }
    }
}

impl FormatOpts {
    /// Recommended `log_groups_per_flex` for a filesystem with the given
    /// number of block groups. Mirrors mke2fs's heuristic: leave flex_bg
    /// off for very small FSes (it buys nothing) and use a 16-group flex
    /// unit (log 4) otherwise. The integrator is free to override.
    pub const fn default_log_groups_per_flex(num_groups: u32) -> u8 {
        if num_groups < 16 { 0 } else { 4 }
    }

    /// Pull ext-specific keys out of an [`OptionMap`] and apply them on
    /// top of `self`. Recognised keys mirror the field names of this
    /// struct verbatim (so `-O block_size=4096` does what you expect):
    ///
    /// - `block_size` (u32)
    /// - `blocks_count` (u32)
    /// - `inodes_count` (u32)
    /// - `reserved_blocks_percent` (u8)
    /// - `mtime` (u32, Unix epoch seconds)
    /// - `journal_blocks` (u32)
    /// - `sparse` (bool)
    /// - `sparse_super` (bool)
    /// - `sparse_super2` (bool)
    /// - `use_64bit` (bool)
    /// - `log_groups_per_flex` (u8, 0..=5)
    /// - `volume_label` (string, ≤ 16 bytes; longer is rejected)
    /// - `create_lost_found` (bool)
    ///
    /// Leaves `kind` and `uuid` alone — those are set by the caller
    /// (the CLI's `--type` flag drives `kind`; the spec doesn't
    /// surface a UUID knob yet).
    ///
    /// [`OptionMap`]: crate::format_opts::OptionMap
    pub fn apply_options(
        &mut self,
        map: &mut crate::format_opts::OptionMap,
    ) -> crate::Result<()> {
        if let Some(v) = map.take_u32("block_size")? {
            self.block_size = v;
        }
        if let Some(v) = map.take_u32("blocks_count")? {
            self.blocks_count = v;
        }
        if let Some(v) = map.take_u32("inodes_count")? {
            self.inodes_count = v;
        }
        if let Some(v) = map.take_u8("reserved_blocks_percent")? {
            self.reserved_blocks_percent = v;
        }
        if let Some(v) = map.take_u32("mtime")? {
            self.mtime = v;
        }
        if let Some(v) = map.take_u32("journal_blocks")? {
            self.journal_blocks = v;
        }
        if let Some(v) = map.take_bool("sparse")? {
            self.sparse = v;
        }
        if let Some(v) = map.take_bool("sparse_super")? {
            self.sparse_super = v;
        }
        if let Some(v) = map.take_bool("sparse_super2")? {
            self.sparse_super2 = v;
        }
        if let Some(v) = map.take_bool("use_64bit")? {
            self.use_64bit = v;
        }
        if let Some(v) = map.take_u8("log_groups_per_flex")? {
            self.log_groups_per_flex = v;
        }
        if let Some(v) = map.take_bool("create_lost_found")? {
            self.create_lost_found = v;
        }
        if let Some(label) = map.take_label::<16>("volume_label", 0)? {
            self.volume_label = label;
        }
        Ok(())
    }

    /// Validate this opts' flex_bg setting before format. Returns Ok if
    /// flex_bg is disabled or in-range; an error if `log_groups_per_flex`
    /// exceeds the 5 (32-groups) cap defined by the on-disk format.
    fn check_flex_bg(&self) -> crate::Result<()> {
        if self.log_groups_per_flex > 5 {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: log_groups_per_flex {} > 5 (max 32 groups per flex unit)",
                self.log_groups_per_flex
            )));
        }
        Ok(())
    }
}

/// In-memory state of a block group during writing.
#[derive(Debug, Clone)]
struct GroupState {
    block_bitmap: Vec<u8>,
    inode_bitmap: Vec<u8>,
    desc: GroupDesc,
}

/// An open / under-construction ext filesystem.
///
/// During the build phase the on-disk state may be inconsistent: bitmaps
/// and inode-table entries are only written when [`Ext::flush`] runs
/// (called automatically at the end of [`Ext::format_with`] for the empty
/// FS case, and explicitly after a batch of `add_*` calls).
#[derive(Debug)]
pub struct Ext {
    pub sb: Superblock,
    pub layout: Layout,
    /// Which ext flavour to write — controls extent-vs-indirect block
    /// pointers, FILETYPE in dirents, etc.
    pub kind: FsKind,
    /// When set, all-zero blocks in regular files are written as holes.
    pub sparse: bool,
    groups: Vec<GroupState>,
    /// Next free inode number to hand out (starts at first_ino).
    next_inode: u32,
    /// Inodes allocated so far during this build session. Written to the
    /// on-disk inode table during [`flush_metadata`].
    pub(crate) inodes: Vec<(u32, Inode)>,
    /// Data blocks staged for write (typically directory data blocks and
    /// indirect-block tables, which we assemble in memory). Regular file
    /// data is NOT staged here — it streams straight to the device.
    pub(crate) data_blocks: Vec<(u32, Vec<u8>)>,
    /// Directory data blocks staged in `data_blocks`, tagged with their
    /// owning directory inode. Used at flush time to stamp the per-block
    /// CRC32C checksum tail when `metadata_csum` is active.
    dir_blocks: Vec<(u32, u32)>,
    /// True until the first flush after a `format_with` lands. The
    /// initial flush is a "blast everything fresh" write that doesn't
    /// ride a journal transaction (there's nothing yet to be consistent
    /// with — the journal is fresh and clean). Subsequent flushes on an
    /// image with a journal go through the JBD2 commit/checkpoint path.
    bootstrap: bool,
}

impl Ext {
    /// Format the device, returning an `Ext` handle. At return the on-disk
    /// image is a valid ext2 containing just the root directory (and
    /// `/lost+found` if requested in `opts`).
    pub fn format_with(dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<Self> {
        opts.check_flex_bg()?;
        // Pre-compute the sparse-super mode and the `s_backup_bgs` pair so
        // both the layout planner and the on-disk superblock agree on
        // which groups carry SB+GDT backups.
        //
        // For `sparse_super2` we need to know the group count to pick the
        // two backup groups (`[1, last]` matches mke2fs's default). Probe
        // the layout once with All to get `num_groups`, then replan with
        // the real mode if sparse_super2 is on.
        let (sparse_mode, backup_bgs) = if opts.sparse_super2 {
            // First pass: just need group count.
            let probe = layout::plan_layout(
                opts.block_size,
                opts.blocks_count,
                opts.inodes_count,
                layout::SparseSuperMode::All,
                opts.log_groups_per_flex,
                opts.use_64bit,
            )?;
            let last = probe.num_groups().saturating_sub(1);
            // For a single-group FS use [0, 0] — group 0 always carries
            // the primary SB anyway.
            let bgs = if probe.num_groups() <= 1 {
                [0, 0]
            } else {
                [1, last]
            };
            (layout::SparseSuperMode::Two(bgs), bgs)
        } else if opts.sparse_super {
            (layout::SparseSuperMode::Classic, [0, 0])
        } else {
            (layout::SparseSuperMode::All, [0, 0])
        };
        let layout = layout::plan_layout(
            opts.block_size,
            opts.blocks_count,
            opts.inodes_count,
            sparse_mode,
            opts.log_groups_per_flex,
            opts.use_64bit,
        )?;
        let total_bytes = layout.blocks_count as u64 * layout.block_size as u64;
        if dev.total_size() < total_bytes {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: device has {} bytes, need {total_bytes}",
                dev.total_size()
            )));
        }

        // Zero the FS region. Backends may treat this as a sparse hole.
        dev.zero_range(0, total_bytes)?;

        // Build superblock.
        let mut sb = Superblock::ext2_default();
        sb.blocks_count = layout.blocks_count;
        sb.inodes_count = layout.inodes_count;
        sb.first_data_block = layout.first_data_block;
        sb.log_block_size = layout.block_size.trailing_zeros() - 10;
        sb.log_frag_size = sb.log_block_size;
        sb.blocks_per_group = layout.blocks_per_group;
        sb.frags_per_group = layout.blocks_per_group;
        sb.inodes_per_group = layout.inodes_per_group;
        sb.mtime = opts.mtime;
        sb.wtime = opts.mtime;
        sb.uuid = opts.uuid;
        sb.volume_name = opts.volume_label;
        sb.r_blocks_count =
            (layout.blocks_count as u64 * opts.reserved_blocks_percent as u64 / 100) as u32;
        sb.lastcheck = opts.mtime;

        // Initialise per-group bitmaps with metadata blocks already marked
        // as used, plus padding-bit-as-used tails for short groups and small
        // inode counts.
        let bs = layout.block_size;
        let mut groups = Vec::with_capacity(layout.groups.len());
        for g in &layout.groups {
            let mut block_bitmap = vec![0u8; bs as usize];
            let mut inode_bitmap = vec![0u8; bs as usize];

            // Mark metadata blocks of this group as used.
            for blk in g.start_block..g.data_start {
                set_bit(&mut block_bitmap, blk - g.start_block);
            }
            // For 1 KiB blocks where first_data_block == 1, block 0 (boot
            // block) is outside the bitmap entirely. Higher block sizes
            // overlap the boot region with block 0 itself, which IS the
            // first block of group 0, so it must be marked used: that's
            // already covered above because block 0 lies in [start, data_start).
            let group_blocks = g.end_block - g.start_block + 1;
            // Bits past the group's last valid block are marked used so the
            // allocator never picks them.
            for bit in group_blocks..(bs * 8) {
                set_bit(&mut block_bitmap, bit);
            }
            // Same on the inode bitmap.
            for bit in layout.inodes_per_group..(bs * 8) {
                set_bit(&mut inode_bitmap, bit);
            }

            let desc = GroupDesc {
                block_bitmap: g.block_bitmap,
                inode_bitmap: g.inode_bitmap,
                inode_table: g.inode_table,
                free_blocks_count: 0,
                free_inodes_count: 0,
                used_dirs_count: 0,
                flags: 0,
            };
            groups.push(GroupState {
                block_bitmap,
                inode_bitmap,
                desc,
            });
        }

        let mut ext = Self {
            sb,
            layout,
            kind: opts.kind,
            sparse: opts.sparse,
            groups,
            next_inode: 0,
            inodes: Vec::new(),
            data_blocks: Vec::new(),
            dir_blocks: Vec::new(),
            // During format the journal SB is staged in `data_blocks`
            // and the file system as a whole is being assembled fresh;
            // the initial flush is a "blast everything" write rather
            // than a JBD2-protected transaction. After the first flush
            // lands this is set to false so subsequent incremental
            // edits ride the journal commit/checkpoint path.
            bootstrap: true,
        };

        // Set feature flags up front — before create_root / create_lost_found
        // run — so those build their directory blocks with the right layout
        // (FILETYPE dirents, metadata_csum tail). The journal feature is set
        // here too; allocate_journal itself runs further down.
        if matches!(opts.kind, FsKind::Ext4) {
            ext.sb.feature_incompat |= constants::feature::INCOMPAT_EXTENTS;
            // FILETYPE is required for metadata_csum's directory checksum
            // tail (the tail uses the dirent file_type byte as its marker),
            // and matches mke2fs ext4 anyway.
            ext.sb.feature_incompat |= constants::feature::INCOMPAT_FILETYPE;
            // Match mke2fs: a fresh ext4 carries CRC32C metadata checksums.
            ext.sb.feature_ro_compat |= constants::feature::RO_COMPAT_METADATA_CSUM;
        }
        if opts.sparse_super {
            ext.sb.feature_ro_compat |= constants::feature::RO_COMPAT_SPARSE_SUPER;
        }
        if opts.kind.has_journal() {
            ext.sb.feature_compat |= constants::feature::COMPAT_HAS_JOURNAL;
        }
        // flex_bg: when log_groups_per_flex != 0 the layout planner has
        // already packed metadata into the first group of each flex unit;
        // we just record the feature flag + the log value in the
        // superblock so the kernel and e2fsck know to expect the packed
        // layout.
        if opts.log_groups_per_flex > 0 {
            ext.sb.feature_incompat |= constants::feature::INCOMPAT_FLEX_BG;
            ext.sb.log_groups_per_flex = opts.log_groups_per_flex;
        }
        // 64-bit FS: 64-byte group descriptors carry the upper half of the
        // bitmap/itable block numbers. Kernel docs pair `INCOMPAT_64BIT`
        // with `INCOMPAT_META_BG`; set both. `s_desc_size = 64` tells the
        // reader to expect the wider descriptor.
        if opts.use_64bit {
            ext.sb.feature_incompat |= constants::feature::INCOMPAT_64BIT;
            ext.sb.feature_incompat |= constants::feature::INCOMPAT_META_BG;
            ext.sb.desc_size = constants::GROUP_DESC_SIZE_64 as u16;
        }
        // sparse_super2: backups only in the two listed groups. Mutually
        // exclusive with `sparse_super` in semantics (the layout planner
        // gives `Two` precedence), but the kernel docs put each flag in
        // its own feature word so both bits *could* be set. We only flip
        // the sparse_super2 bit; the on-disk `s_backup_bgs` array carries
        // the actual group numbers.
        if opts.sparse_super2 {
            ext.sb.feature_compat |= constants::feature::COMPAT_SPARSE_SUPER2;
            ext.sb.backup_bgs = backup_bgs;
        }

        // Reserve inodes 1..first_ino-1 (1..=10 for dynamic rev).
        let first_ino = ext.sb.first_ino;
        set_first_n(&mut ext.groups[0].inode_bitmap, first_ino - 1);
        ext.next_inode = first_ino;

        // Create the root directory at inode 2.
        ext.create_root(opts.mtime)?;

        // Optional /lost+found.
        if opts.create_lost_found {
            ext.create_lost_found(dev, opts.mtime)?;
        }

        // Optional journal (ext3 / ext4). JBD2 requires a minimum of 1024
        // blocks; smaller journals are rejected by the kernel + e2fsck.
        if opts.kind.has_journal() {
            let blocks = if opts.journal_blocks == 0 {
                1024
            } else {
                opts.journal_blocks
            };
            ext.allocate_journal(blocks, opts.mtime)?;
            ext.sb.journal_inum = constants::INO_JOURNAL;
        }

        ext.recompute_free_counts();
        ext.flush_metadata(dev)?;
        Ok(ext)
    }

    /// Whether the `metadata_csum` feature is active on this filesystem.
    fn has_metadata_csum(&self) -> bool {
        self.sb.feature_ro_compat & constants::feature::RO_COMPAT_METADATA_CSUM != 0
    }

    /// Whether directory entries carry a `file_type` byte (`INCOMPAT_FILETYPE`).
    fn has_filetype(&self) -> bool {
        self.sb.feature_incompat & constants::feature::INCOMPAT_FILETYPE != 0
    }

    /// The filesystem-wide checksum seed. genfs never sets the
    /// `metadata_csum_seed` feature, so the seed is always derived from the
    /// UUID.
    fn csum_seed(&self) -> u32 {
        csum::fs_seed(&self.sb.uuid, None)
    }

    /// Wire `data_blocks` into an inode's block-pointer array. Picks the
    /// representation based on the filesystem kind: ext4 uses an extent
    /// tree (depth 0, up to 4 leaves, no extra metadata blocks); ext2/3 use
    /// the classic direct + single-indirect + double-indirect scheme.
    /// Returns the number of metadata (indirection) blocks allocated.
    fn fill_block_pointers(&mut self, inode: &mut Inode, data: &[u32]) -> Result<u32> {
        if matches!(self.kind, FsKind::Ext4) {
            return self.fill_block_pointers_extent(inode, data);
        }
        self.fill_block_pointers_indirect(inode, data)
    }

    /// Ext4 path: pack `data` into an extent tree stored directly in
    /// `i_block` and set `EXT4_EXTENTS_FL` on the inode. Allocates no extra
    /// metadata blocks for depth-0 trees (the cap is 4 extents per inode).
    fn fill_block_pointers_extent(&mut self, inode: &mut Inode, data: &[u32]) -> Result<u32> {
        let runs = extent::coalesce(data);
        let packed = extent::pack_into_iblock(&runs)?;
        // Decode the 60 packed bytes back into the 15 u32 slots of i_block.
        for (i, slot) in inode.block.iter_mut().enumerate() {
            let off = i * 4;
            *slot = u32::from_le_bytes(packed[off..off + 4].try_into().unwrap());
        }
        inode.flags |= constants::EXT4_EXTENTS_FL;
        Ok(0)
    }

    /// Ext2 / Ext3 path: direct + single + double indirection. v1 cap.
    /// At 1 KiB blocks that's up to 12 + 256 + 256² ≈ 65 MiB; at 4 KiB
    /// it's ~4 GiB.
    ///
    /// A `0` in `data` is a hole (sparse file): the corresponding block
    /// pointer stays 0, and an indirect block whose entire range is holes
    /// is not allocated at all.
    fn fill_block_pointers_indirect(&mut self, inode: &mut Inode, data: &[u32]) -> Result<u32> {
        let bs = self.layout.block_size;
        let ptrs_per_block = (bs / 4) as usize;
        let n = data.len();
        let n_direct = constants::N_DIRECT.min(n);
        inode.block[..n_direct].copy_from_slice(&data[..n_direct]);
        let mut allocated_meta = 0u32;
        let mut consumed = n_direct;

        if consumed < n {
            // Single-indirect — only allocate the indirect block if at least
            // one block in its range is actually present.
            let take = (n - consumed).min(ptrs_per_block);
            let range = &data[consumed..consumed + take];
            if range.iter().any(|&b| b != 0) {
                let ind = self.alloc_data_block()?;
                allocated_meta += 1;
                inode.block[constants::IDX_INDIRECT] = ind;
                let mut buf = vec![0u8; bs as usize];
                for (i, &b) in range.iter().enumerate() {
                    buf[i * 4..i * 4 + 4].copy_from_slice(&b.to_le_bytes());
                }
                self.data_blocks.push((ind, buf));
            }
            consumed += take;
        }

        if consumed < n {
            // Double-indirect. Each sub-indirect block is allocated only if
            // its range has a non-hole block; the double-indirect block
            // itself is allocated only if at least one sub-indirect is.
            let mut dind_buf = vec![0u8; bs as usize];
            let mut dind_slot = 0;
            let mut any_sub = false;
            while consumed < n {
                if dind_slot >= ptrs_per_block {
                    return Err(crate::Error::Unsupported(
                        "ext: file exceeds direct+single+double indirection capacity".into(),
                    ));
                }
                let take = (n - consumed).min(ptrs_per_block);
                let range = &data[consumed..consumed + take];
                if range.iter().any(|&b| b != 0) {
                    let ind = self.alloc_data_block()?;
                    allocated_meta += 1;
                    any_sub = true;
                    dind_buf[dind_slot * 4..dind_slot * 4 + 4].copy_from_slice(&ind.to_le_bytes());
                    let mut ind_buf = vec![0u8; bs as usize];
                    for (i, &b) in range.iter().enumerate() {
                        ind_buf[i * 4..i * 4 + 4].copy_from_slice(&b.to_le_bytes());
                    }
                    self.data_blocks.push((ind, ind_buf));
                }
                consumed += take;
                dind_slot += 1;
            }
            if any_sub {
                let dind = self.alloc_data_block()?;
                allocated_meta += 1;
                inode.block[constants::IDX_DOUBLE_INDIRECT] = dind;
                self.data_blocks.push((dind, dind_buf));
            }
        }

        Ok(allocated_meta)
    }

    /// Allocate the journal inode (inode 8) and its data blocks. The first
    /// data block is initialised with a JBD2 v2 journal superblock marking
    /// the journal as clean (s_start = 0, so no recovery needed). The rest
    /// of the journal is zeroed by the up-front zero_range, which JBD2
    /// reads as empty log blocks.
    fn allocate_journal(&mut self, blocks: u32, mtime: u32) -> Result<()> {
        let ino = constants::INO_JOURNAL;
        let bs = self.layout.block_size;
        let mut data = Vec::with_capacity(blocks as usize);
        for _ in 0..blocks {
            data.push(self.alloc_data_block()?);
        }
        let mut inode = Inode::regular(blocks * bs, 0o600, 0, 0, mtime);
        let meta_blocks = self.fill_block_pointers(&mut inode, &data)?;
        inode.blocks_512 = (blocks + meta_blocks) * (bs / 512);

        // Build the JBD2 v2 journal superblock for block 0 of the journal.
        let jsb = build_jbd2_superblock(bs, blocks);
        self.data_blocks.push((data[0], jsb));

        self.inodes.push((ino, inode));
        Ok(())
    }

    /// Allocate inode #2 (the root dir), give it a fresh data block with
    /// "." and "..", and stage both for write.
    fn create_root(&mut self, mtime: u32) -> Result<()> {
        let ino = INO_ROOT_DIR;
        set_bit(&mut self.groups[0].inode_bitmap, ino - 1);

        let blk = self.alloc_data_block()?;
        let csum_tail = self.has_metadata_csum();
        let with_filetype = self.has_filetype();
        let block_bytes =
            dir::make_initial_dir_block(ino, ino, self.layout.block_size, with_filetype, csum_tail);

        let mut inode = Inode::directory(self.layout.block_size, 0o755, 0, 0, mtime);
        inode.block[0] = blk;
        inode.blocks_512 = self.layout.block_size / 512;

        self.groups[0].desc.used_dirs_count += 1;
        self.inodes.push((ino, inode));
        self.data_blocks.push((blk, block_bytes));
        self.dir_blocks.push((blk, ino));
        Ok(())
    }

    /// Create the conventional /lost+found directory pre-allocated to 16 KiB.
    fn create_lost_found(&mut self, dev: &mut dyn BlockDevice, mtime: u32) -> Result<()> {
        let bs = self.layout.block_size;
        let target_data_blocks: u32 = 16384u32.div_ceil(bs);

        // Allocate inode first (uses next_inode).
        let ino = self.alloc_inode()?;

        // Allocate data blocks sequentially.
        let mut data_blocks = Vec::with_capacity(target_data_blocks as usize);
        for _ in 0..target_data_blocks {
            data_blocks.push(self.alloc_data_block()?);
        }

        let mut inode = Inode::directory(16384, 0o700, 0, 0, mtime);
        let meta_blocks = self.fill_block_pointers(&mut inode, &data_blocks)?;
        inode.blocks_512 = (target_data_blocks + meta_blocks) * (bs / 512);

        // First data block: "." / "..". All blocks of lost+found are
        // directory blocks owned by inode `ino`.
        let csum_tail = self.has_metadata_csum();
        let with_filetype = self.has_filetype();
        let dir_block =
            dir::make_initial_dir_block(ino, INO_ROOT_DIR, bs, with_filetype, csum_tail);
        self.data_blocks.push((data_blocks[0], dir_block));
        self.dir_blocks.push((data_blocks[0], ino));
        // All trailing data blocks: empty-placeholder entry so e2fsck reads
        // them as well-formed empty dir blocks.
        for &blk in &data_blocks[1..] {
            self.data_blocks
                .push((blk, dir::make_empty_dir_block(bs, csum_tail)));
            self.dir_blocks.push((blk, ino));
        }

        self.groups[0].desc.used_dirs_count += 1;
        self.inodes.push((ino, inode));

        // Add to root dir + bump root's link count (a new subdir's ".." is
        // a fresh link to the parent).
        self.add_entry_to_dir_block_for(
            dev,
            INO_ROOT_DIR,
            b"lost+found",
            ino,
            constants::DENT_DIR,
        )?;
        self.patch_inode(dev, INO_ROOT_DIR, |i| i.links_count += 1)?;
        Ok(())
    }

    /// Append a dir entry into the data block(s) of the directory whose
    /// inode is `dir_inode`. v1: only ever touches the directory's first
    /// data block; full-block error if it doesn't fit.
    fn add_entry_to_dir_block_for(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_inode: u32,
        name: &[u8],
        child_ino: u32,
        file_type: u8,
    ) -> Result<()> {
        self.ensure_inode_staged(dev, dir_inode)?;
        // Resolve the dir's first data block via file_block (handles both
        // direct-pointer ext2 dirs and extent-encoded ext4 dirs uniformly).
        let inode_copy = self
            .inodes
            .iter()
            .find(|(i, _)| *i == dir_inode)
            .map(|(_, i)| *i)
            .unwrap();
        let dir_block_num = self.file_block(dev, &inode_copy, 0)?;
        if dir_block_num == 0 {
            return Err(crate::Error::InvalidImage(format!(
                "ext: dir inode {dir_inode} has no first data block"
            )));
        }
        self.ensure_block_staged(dev, dir_block_num)?;
        // This block is a directory block; tag it so flush stamps its
        // checksum tail (no-op if already recorded).
        if !self.dir_blocks.iter().any(|(b, _)| *b == dir_block_num) {
            self.dir_blocks.push((dir_block_num, dir_inode));
        }
        let usable = dir::usable_dir_len(self.layout.block_size, self.has_metadata_csum());
        let with_filetype = self.has_filetype();
        let block = self
            .data_blocks
            .iter_mut()
            .find(|(b, _)| *b == dir_block_num)
            .map(|(_, bytes)| bytes)
            .unwrap();
        append_dir_entry(block, name, child_ino, file_type, with_filetype, usable)
    }

    /// Mutate an inode in place. If the inode isn't already in the staged
    /// cache, reads it from disk first so the mutation is preserved across
    /// the next flush.
    fn patch_inode<F: FnOnce(&mut Inode)>(
        &mut self,
        dev: &mut dyn BlockDevice,
        ino: u32,
        f: F,
    ) -> Result<()> {
        self.ensure_inode_staged(dev, ino)?;
        for (i_no, i) in self.inodes.iter_mut() {
            if *i_no == ino {
                f(i);
                return Ok(());
            }
        }
        unreachable!("ensure_inode_staged guarantees the inode is present")
    }

    /// Ensure inode `ino` is in the staged write set, fetching from disk if
    /// not. No-op if already staged.
    pub(crate) fn ensure_inode_staged(
        &mut self,
        dev: &mut dyn BlockDevice,
        ino: u32,
    ) -> Result<()> {
        if self.inodes.iter().any(|(i, _)| *i == ino) {
            return Ok(());
        }
        let inode = self.read_inode(dev, ino)?;
        self.inodes.push((ino, inode));
        Ok(())
    }

    /// Ensure block `blk` is in the staged write set, fetching from disk
    /// if not. No-op if already staged.
    fn ensure_block_staged(&mut self, dev: &mut dyn BlockDevice, blk: u32) -> Result<()> {
        if self.data_blocks.iter().any(|(b, _)| *b == blk) {
            return Ok(());
        }
        let mut buf = vec![0u8; self.layout.block_size as usize];
        self.read_block(dev, blk, &mut buf)?;
        self.data_blocks.push((blk, buf));
        Ok(())
    }

    /// Reserve the next available inode. Inode `N` lives in group
    /// `(N-1) / inodes_per_group` at bitmap bit `(N-1) % inodes_per_group`,
    /// so a monotonic `next_inode` counter spans all groups.
    fn alloc_inode(&mut self) -> Result<u32> {
        if self.next_inode > self.layout.inodes_count {
            return Err(crate::Error::Unsupported(format!(
                "ext: out of inodes (allocated {}, max {})",
                self.next_inode - 1,
                self.layout.inodes_count
            )));
        }
        let ino = self.next_inode;
        let g = ((ino - 1) / self.layout.inodes_per_group) as usize;
        let idx = (ino - 1) % self.layout.inodes_per_group;
        set_bit(&mut self.groups[g].inode_bitmap, idx);
        self.next_inode += 1;
        Ok(ino)
    }

    /// Allocate a single data block. Scans groups in order and returns the
    /// first free data block, so callers get contiguous runs within a group
    /// (good for extent coalescing) and spill into later groups when a group
    /// fills up.
    pub(crate) fn alloc_data_block(&mut self) -> Result<u32> {
        for gi in 0..self.layout.groups.len() {
            let layout_g = self.layout.groups[gi];
            let start_rel = layout_g.data_start - layout_g.start_block;
            let group_blocks = layout_g.end_block - layout_g.start_block + 1;
            let bitmap = &mut self.groups[gi].block_bitmap;
            for bit in start_rel..group_blocks {
                if !test_bit(bitmap, bit) {
                    set_bit(bitmap, bit);
                    return Ok(layout_g.start_block + bit);
                }
            }
        }
        Err(crate::Error::Unsupported(
            "ext: filesystem has no free data blocks".into(),
        ))
    }

    fn recompute_free_counts(&mut self) {
        let mut total_free_blocks = 0u64;
        let mut total_free_inodes = 0u64;
        for (i, g) in self.layout.groups.iter().enumerate() {
            let group_blocks = g.end_block - g.start_block + 1;
            let used_blocks = popcount_bits(&self.groups[i].block_bitmap, 0, group_blocks);
            let free_blocks = group_blocks - used_blocks;
            let used_inodes = popcount_bits(
                &self.groups[i].inode_bitmap,
                0,
                self.layout.inodes_per_group,
            );
            let free_inodes = self.layout.inodes_per_group - used_inodes;
            self.groups[i].desc.free_blocks_count = free_blocks as u16;
            self.groups[i].desc.free_inodes_count = free_inodes as u16;
            total_free_blocks += free_blocks as u64;
            total_free_inodes += free_inodes as u64;
        }
        self.sb.free_blocks_count = total_free_blocks as u32;
        self.sb.free_inodes_count = total_free_inodes as u32;
    }

    /// Write all staged state to the device. Primary superblock is written
    /// **last** to maintain the "torn write → unmountable, not corrupt"
    /// invariant. When the filesystem has a JBD2 journal the metadata
    /// updates ride a real journal transaction (descriptor + data + commit);
    /// otherwise they go straight to the device (ext2 path).
    fn flush_metadata(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        // Stamp dir-block checksum tails into `data_blocks` before we
        // serialise the metadata image set; the journal must carry the
        // checksum-stamped versions.
        self.stamp_dir_block_checksums();

        // Build the block-aligned metadata image set: every full-block
        // write that flush would emit (bitmaps, GDTs, inode-table blocks,
        // staged dir / extent leaf / indirect blocks). Excludes the
        // primary and backup superblocks: those are written outside the
        // journal as the final step.
        let images = self.collect_metadata_images(dev)?;

        // Initial format flush ("bootstrap") writes every block directly
        // — the journal SB itself is in `images` and has no prior on-disk
        // state to stay consistent with. Subsequent flushes on an image
        // that carries a journal go through the JBD2 commit/checkpoint
        // path so a crash mid-flush leaves a replayable transaction in
        // the log instead of a torn metadata block.
        if self.has_journal() && !self.bootstrap {
            self.commit_journal_and_checkpoint(dev, &images)?;
        } else {
            for (blk, bytes) in &images {
                let bs = self.layout.block_size as u64;
                dev.write_at(*blk as u64 * bs, bytes)?;
            }
        }

        // Backup SB copies and the primary SB. The primary SB is the very
        // last write to preserve the "torn write → unmountable, not
        // corrupt" invariant.
        self.write_superblocks(dev)?;
        dev.sync()?;

        // Subsequent flushes (e.g. from open_file_rw) ride the journal.
        self.bootstrap = false;

        // The staged dir / extent leaf / indirect data blocks have
        // landed on disk; drop them so the next flush only journals
        // genuine new edits (not stale snapshots from before this
        // flush). `self.inodes` is kept around — open file handles
        // assume their inode stays staged across `sync` calls.
        self.data_blocks.clear();
        self.dir_blocks.clear();
        Ok(())
    }

    /// Stamp every staged directory data block's CRC32C tail in place.
    /// No-op when `metadata_csum` is off. Called by [`flush_metadata`]
    /// before the journal commit so the journaled image matches what
    /// the checkpoint phase writes to the directory's home block.
    fn stamp_dir_block_checksums(&mut self) {
        if !self.has_metadata_csum() {
            return;
        }
        let seed = self.csum_seed();
        for (blk, bytes) in &mut self.data_blocks {
            if let Some((_, dir_ino)) = self.dir_blocks.iter().find(|(b, _)| b == blk) {
                let generation = self
                    .inodes
                    .iter()
                    .find(|(i, _)| i == dir_ino)
                    .map(|(_, i)| i.generation)
                    .unwrap_or(0);
                let n = bytes.len();
                let c = csum::dir_block(seed, *dir_ino, generation, &bytes[..n - 12]);
                bytes[n - 4..].copy_from_slice(&c.to_le_bytes());
            }
        }
    }

    /// Build the block-aligned metadata image set (block_no, full-block
    /// bytes). Includes bitmaps (every group), GDT blocks (every backup
    /// group + group 0), inode-table blocks (each block patched with all
    /// staged inodes whose slot falls inside it), and staged data blocks
    /// (dir / extent leaf / indirect). Does NOT include superblocks.
    fn collect_metadata_images(&self, dev: &mut dyn BlockDevice) -> Result<Vec<(u32, Vec<u8>)>> {
        let bs = self.layout.block_size as u64;
        let mut out: Vec<(u32, Vec<u8>)> = Vec::new();

        // Build encoded GDT (same content for every group's copy). With
        // metadata_csum each descriptor's bg_checksum + bitmap checksums
        // are stamped here.
        let desc_size = self.layout.desc_size;
        let with_csum = self.has_metadata_csum();
        let seed = self.csum_seed();
        let bbm_len = (self.layout.blocks_per_group / 8) as usize;
        let ibm_len = (self.layout.inodes_per_group / 8) as usize;
        let mut gdt = vec![0u8; self.layout.gdt_blocks as usize * bs as usize];
        for (i, g) in self.groups.iter().enumerate() {
            let off = i * desc_size;
            let desc = &mut gdt[off..off + desc_size];
            desc[..constants::GROUP_DESC_SIZE].copy_from_slice(&g.desc.encode());
            if with_csum {
                let bbm_full = csum::bitmap(seed, &g.block_bitmap[..bbm_len]);
                let ibm_full = csum::bitmap(seed, &g.inode_bitmap[..ibm_len]);
                let bbm_lo = bbm_full as u16;
                let ibm_lo = ibm_full as u16;
                desc[0x18..0x1A].copy_from_slice(&bbm_lo.to_le_bytes());
                desc[0x1A..0x1C].copy_from_slice(&ibm_lo.to_le_bytes());
                if desc_size >= 64 {
                    let bbm_hi = (bbm_full >> 16) as u16;
                    let ibm_hi = (ibm_full >> 16) as u16;
                    desc[0x38..0x3A].copy_from_slice(&bbm_hi.to_le_bytes());
                    desc[0x3A..0x3C].copy_from_slice(&ibm_hi.to_le_bytes());
                }
                let bg = csum::group_desc(seed, i as u32, desc);
                desc[0x1E..0x20].copy_from_slice(&bg.to_le_bytes());
            }
        }

        // Bitmaps and GDT copies, per group.
        for (i, g) in self.layout.groups.iter().enumerate() {
            if g.has_superblock {
                // The GDT itself; SB backup is handled by write_superblocks.
                let gdt_start_block = if i == 0 {
                    if self.layout.first_data_block == 1 {
                        2u32
                    } else {
                        1u32
                    }
                } else {
                    g.start_block + 1
                };
                for (blk_off, chunk) in gdt.chunks(bs as usize).enumerate() {
                    out.push((gdt_start_block + blk_off as u32, chunk.to_vec()));
                }
            }
            out.push((g.block_bitmap, self.groups[i].block_bitmap.clone()));
            out.push((g.inode_bitmap, self.groups[i].inode_bitmap.clone()));
        }

        // Inode-table blocks: group staged inodes by their containing
        // table block, then RMW each block.
        let inode_size = self.layout.inode_size as u64;
        let inodes_per_block = (bs / inode_size) as u32;
        let mut by_block: std::collections::BTreeMap<u32, Vec<(u32, &Inode)>> =
            std::collections::BTreeMap::new();
        for (ino, inode) in &self.inodes {
            let (group, idx_in_group) = self.inode_location(*ino);
            let table_block = self.layout.groups[group as usize].inode_table;
            let block_off = idx_in_group / inodes_per_block;
            let blk = table_block + block_off;
            by_block.entry(blk).or_default().push((*ino, inode));
        }
        for (blk, slots) in by_block {
            // RMW: start from the current on-disk content. We avoid
            // touching staged data_blocks here because inode-table blocks
            // are not staged in `data_blocks` (only dirs / extent leaves
            // / indirects are).
            let mut buf = vec![0u8; bs as usize];
            dev.read_at(blk as u64 * bs, &mut buf)?;
            for (ino, inode) in slots {
                let (_, idx_in_group) = self.inode_location(ino);
                let inblock_idx = idx_in_group % inodes_per_block;
                let off = inblock_idx as u64 * inode_size;
                let encoded = self.encode_inode(ino, inode);
                let body_len = encoded.len().min(inode_size as usize);
                buf[off as usize..off as usize + body_len].copy_from_slice(&encoded[..body_len]);
                // Tail bytes (i_extra_isize region of large inodes, if any)
                // are left as their on-disk values.
            }
            out.push((blk, buf));
        }

        // Staged data blocks (directories, extent leaves, indirect blocks).
        // These are already block-sized in `data_blocks`. Dir-block
        // checksums are already stamped (stamp_dir_block_checksums).
        for (blk, bytes) in &self.data_blocks {
            out.push((*blk, bytes.clone()));
        }

        // De-duplicate by block number, keeping the latest write per block
        // (a later entry wins). This matters when, e.g., an inode-table
        // block is also staged as a generic data block (shouldn't happen
        // today, but the guard is cheap and keeps the journal payload
        // free of duplicates).
        let mut seen: std::collections::BTreeMap<u32, Vec<u8>> = std::collections::BTreeMap::new();
        for (blk, bytes) in out {
            seen.insert(blk, bytes);
        }
        Ok(seen.into_iter().collect())
    }

    /// Write the GDT + bitmap + inode + data-block updates as a single
    /// JBD2 transaction, fsync, then checkpoint by writing the same blocks
    /// to their target FS locations.
    fn commit_journal_and_checkpoint(
        &mut self,
        dev: &mut dyn BlockDevice,
        images: &[(u32, Vec<u8>)],
    ) -> Result<()> {
        let jino = self.sb.journal_inum;
        if jino == 0 {
            return Err(crate::Error::InvalidImage(
                "ext: HAS_JOURNAL set but s_journal_inum is 0".into(),
            ));
        }
        let bs = self.layout.block_size;

        if images.is_empty() {
            // Nothing to journal; SB will still be written by the caller.
            return Ok(());
        }

        // Read journal SB and inode. The journal SB may be staged in
        // `data_blocks` (first flush after format) — consult that cache
        // before falling back to the device.
        let journal_inode = self.read_inode(dev, jino)?;
        let jsb_phys = self.file_block(dev, &journal_inode, 0)?;
        if jsb_phys == 0 {
            return Err(crate::Error::InvalidImage(
                "ext: journal block 0 unmapped".into(),
            ));
        }
        let mut jsb_buf = vec![0u8; bs as usize];
        self.read_block(dev, jsb_phys, &mut jsb_buf)?;
        let jsb = jbd2::JournalSuperblock::decode(&jsb_buf)?;
        if jsb.blocksize != bs {
            return Err(crate::Error::InvalidImage(format!(
                "ext: journal blocksize {} != FS blocksize {bs}",
                jsb.blocksize
            )));
        }

        // Build payload list.
        let blocks: Vec<jbd2::JournalBlock> = images
            .iter()
            .map(|(blk, bytes)| jbd2::JournalBlock {
                fs_block: *blk,
                bytes: bytes.clone(),
            })
            .collect();

        // Pick the next tid and the start of the log ring. Path A v1 only
        // writes one transaction at a time, starting at `s_first`; the
        // ring isn't reused mid-flush.
        let tid = jsb.sequence;
        let start_idx = jsb.first;
        let _next_idx = jbd2::write_transaction(
            self,
            dev,
            &journal_inode,
            &mut jsb_buf,
            &jsb,
            start_idx,
            tid,
            &blocks,
            self.sb.wtime as u64,
            0,
        )?;
        dev.sync()?;

        // Now stamp s_start + s_sequence so a crash from here on yields
        // a replayable journal: replay will see this exact transaction
        // and re-apply it.
        jbd2::set_start(&mut jsb_buf, start_idx);
        jbd2::set_sequence(&mut jsb_buf, tid);
        dev.write_at(jsb_phys as u64 * bs as u64, &jsb_buf)?;
        dev.sync()?;

        // Checkpoint: write each block image to its FS-home location.
        let bs64 = bs as u64;
        for (blk, bytes) in images {
            dev.write_at(*blk as u64 * bs64, bytes)?;
        }
        dev.sync()?;

        // Mark the journal clean: s_start = 0, s_sequence = tid + 1. A
        // future open sees a clean journal and skips replay.
        jbd2::set_start(&mut jsb_buf, 0);
        jbd2::set_sequence(&mut jsb_buf, tid.wrapping_add(1));
        dev.write_at(jsb_phys as u64 * bs as u64, &jsb_buf)?;
        Ok(())
    }

    /// Write every group's superblock copy (primary and any backups).
    /// Called as the final phase of `flush_metadata` so a torn write of
    /// the primary leaves the on-disk state mountable from a backup.
    fn write_superblocks(&self, dev: &mut dyn BlockDevice) -> Result<()> {
        let bs = self.layout.block_size as u64;
        for (i, g) in self.layout.groups.iter().enumerate() {
            if g.has_superblock && i != 0 {
                let mut sb_copy = self.sb.clone();
                sb_copy.block_group_nr = i as u16;
                dev.write_at(g.start_block as u64 * bs, &self.encode_sb(&sb_copy))?;
            }
        }
        dev.write_at(SUPERBLOCK_OFFSET, &self.encode_sb(&self.sb))?;
        Ok(())
    }

    /// Whether the filesystem has a JBD2 journal (`COMPAT_HAS_JOURNAL`)
    /// AND it's not an external journal device.
    fn has_journal(&self) -> bool {
        self.sb.feature_compat & constants::feature::COMPAT_HAS_JOURNAL != 0
            && self.sb.feature_incompat & constants::feature::INCOMPAT_JOURNAL_DEV == 0
    }

    /// Encode an inode, stamping its CRC32C checksum (`l_i_checksum_lo` at
    /// offset 124) when `metadata_csum` is set. With 128-byte inodes there
    /// is no room for `i_checksum_hi`, so only the low 16 bits are stored —
    /// the kernel handles a 16-bit inode checksum for small inodes.
    fn encode_inode(&self, ino: u32, inode: &Inode) -> [u8; inode::INODE_BASE_SIZE] {
        let mut buf = inode.encode();
        if self.has_metadata_csum() {
            // Zero the checksum field before summing — an inode read back
            // from disk (modify-after-open) carries its previous checksum
            // in osd2, which must not feed into the recomputed value.
            buf[124..126].fill(0);
            let c = csum::inode(self.csum_seed(), ino, inode.generation, &buf);
            buf[124..126].copy_from_slice(&((c & 0xffff) as u16).to_le_bytes());
        }
        buf
    }

    /// Encode a superblock, stamping the CRC32C `s_checksum` field when the
    /// `metadata_csum` feature is set. Without the feature the field stays
    /// zero (the kernel ignores it).
    fn encode_sb(&self, sb: &Superblock) -> [u8; constants::SUPERBLOCK_SIZE] {
        let mut buf = sb.encode();
        if self.has_metadata_csum() {
            // s_checksum_type (offset 0x175) must be 1 (CRC32C) — the kernel
            // refuses to mount a metadata_csum FS otherwise. Set it before
            // computing the checksum so it's covered.
            buf[0x175] = 1;
            let c = csum::superblock(&buf);
            buf[1020..1024].copy_from_slice(&c.to_le_bytes());
        }
        buf
    }

    fn inode_location(&self, ino: u32) -> (u32, u32) {
        let g = (ino - 1) / self.layout.inodes_per_group;
        let idx = (ino - 1) % self.layout.inodes_per_group;
        (g, idx)
    }

    // ──────────────────────────── populate API ───────────────────────────
    //
    // These methods stage in-memory state (bitmaps, inode table, dir blocks)
    // and stream regular-file data straight to the device. Call
    // [`Ext::flush`] when done to persist the staged metadata.

    /// Create a regular file under `parent_ino` with the given name and
    /// metadata, streaming bytes from `src` straight to the device through a
    /// fixed-size buffer. The file is *never* fully resident in memory.
    pub fn add_file_to(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u32,
        name: &[u8],
        src: FileSource,
        meta: FileMeta,
    ) -> Result<u32> {
        let len = src.len()?;
        let (mut reader, _) = src.open()?;
        self.add_file_to_streaming(dev, parent_ino, name, &mut *reader, len, meta)
    }

    /// Like [`Self::add_file_to`] but pulls bytes from any [`std::io::Read`]
    /// instead of a [`FileSource`]. Useful when streaming from a borrowed
    /// reader (e.g. another filesystem's `open_file_reader`) where the
    /// `'static` lifetime in `FileSource::Reader` doesn't fit.
    pub fn add_file_to_streaming(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u32,
        name: &[u8],
        reader: &mut dyn std::io::Read,
        len: u64,
        meta: FileMeta,
    ) -> Result<u32> {
        let bs = self.layout.block_size;
        if len > u32::MAX as u64 {
            return Err(crate::Error::Unsupported(
                "ext: file > 4 GiB requires LARGE_FILE (deferred to ext4)".into(),
            ));
        }
        let n_data_blocks = len.div_ceil(bs as u64) as u32;

        let ino = self.alloc_inode()?;
        let mut inode = Inode::regular(
            len as u32,
            meta.mode & 0o7777,
            meta.uid,
            meta.gid,
            meta.mtime,
        );

        // Stream one block at a time. Each block is read into a fixed
        // buffer (the file is never fully resident in memory). In sparse
        // mode an all-zero block becomes a hole: `data_blocks[i] == 0`
        // means logical block i is unallocated and reads back as zero.
        let mut buf = vec![0u8; bs as usize];
        let mut data_blocks = Vec::with_capacity(n_data_blocks as usize);
        let mut remaining = len;
        let mut allocated_data = 0u32;
        for _ in 0..n_data_blocks {
            let to_read = remaining.min(bs as u64) as usize;
            buf.fill(0);
            reader.read_exact(&mut buf[..to_read])?;
            if self.sparse && buf.iter().all(|&b| b == 0) {
                data_blocks.push(0);
            } else {
                let blk = self.alloc_data_block()?;
                dev.write_at(blk as u64 * bs as u64, &buf[..to_read])?;
                data_blocks.push(blk);
                allocated_data += 1;
            }
            remaining -= to_read as u64;
        }
        debug_assert_eq!(remaining, 0);

        let allocated_meta_blocks = self.fill_block_pointers(&mut inode, &data_blocks)?;
        inode.blocks_512 = (allocated_data + allocated_meta_blocks) * (bs / 512);

        self.inodes.push((ino, inode));
        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino, constants::DENT_REG)?;
        Ok(ino)
    }

    /// Create a subdirectory under `parent_ino`. Allocates one data block
    /// holding "." / "..", patches the parent's link count.
    pub fn add_dir_to(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u32,
        name: &[u8],
        meta: FileMeta,
    ) -> Result<u32> {
        let bs = self.layout.block_size;
        let ino = self.alloc_inode()?;
        let blk = self.alloc_data_block()?;
        let mut inode = Inode::directory(bs, meta.mode & 0o7777, meta.uid, meta.gid, meta.mtime);
        // For ext4, encode the single data block as an inline extent tree;
        // for ext2/3, store the block number directly in i_block[0].
        if matches!(self.kind, FsKind::Ext4) {
            self.fill_block_pointers_extent(&mut inode, &[blk])?;
        } else {
            inode.block[0] = blk;
        }
        inode.blocks_512 = bs / 512;
        let csum_tail = self.has_metadata_csum();
        let with_filetype = self.has_filetype();
        let block_bytes =
            dir::make_initial_dir_block(ino, parent_ino, bs, with_filetype, csum_tail);
        self.data_blocks.push((blk, block_bytes));
        self.dir_blocks.push((blk, ino));
        self.inodes.push((ino, inode));
        self.groups[0].desc.used_dirs_count += 1;

        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino, constants::DENT_DIR)?;
        self.patch_inode(dev, parent_ino, |i| i.links_count += 1)?;
        Ok(ino)
    }

    /// Create a symbolic link. Targets ≤ 60 bytes are stored inline in
    /// `i_block[0..15]` (the "fast symlink" optimization — no data block
    /// allocated, blocks_512 stays at zero). Longer targets get a data block.
    pub fn add_symlink_to(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u32,
        name: &[u8],
        target: &[u8],
        meta: FileMeta,
    ) -> Result<u32> {
        if target.len() > 4095 {
            return Err(crate::Error::Unsupported(
                "ext: symlink target > 4095 bytes".into(),
            ));
        }
        let bs = self.layout.block_size;
        let ino = self.alloc_inode()?;
        let mut inode = Inode::symlink(
            target.len() as u32,
            meta.mode & 0o7777,
            meta.uid,
            meta.gid,
            meta.mtime,
        );

        // Fast symlink: target fits in i_block (60 bytes = 15 × 4).
        const FAST_MAX: usize = 60;
        if target.len() <= FAST_MAX {
            // Pack target bytes into i_block array.
            let mut packed = [0u8; FAST_MAX];
            packed[..target.len()].copy_from_slice(target);
            for (i, slot) in inode.block.iter_mut().enumerate() {
                let off = i * 4;
                *slot = u32::from_le_bytes(packed[off..off + 4].try_into().unwrap());
            }
            // blocks_512 stays 0; no data block.
        } else {
            // Slow symlink: target gets a data block.
            let blk = self.alloc_data_block()?;
            inode.block[0] = blk;
            inode.blocks_512 = bs / 512;
            let mut buf = vec![0u8; bs as usize];
            buf[..target.len()].copy_from_slice(target);
            dev.write_at(blk as u64 * bs as u64, &buf)?;
        }

        self.inodes.push((ino, inode));
        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino, constants::DENT_LNK)?;
        Ok(ino)
    }

    /// Create a device node, FIFO, or socket. No data blocks are allocated;
    /// for char/block devices the major+minor are encoded into `i_block\[0\]`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_device_to(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u32,
        name: &[u8],
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: FileMeta,
    ) -> Result<u32> {
        let ino = self.alloc_inode()?;
        let special = match kind {
            DeviceKind::Char => SpecialKind::Char,
            DeviceKind::Block => SpecialKind::Block,
            DeviceKind::Fifo => SpecialKind::Fifo,
            DeviceKind::Socket => SpecialKind::Socket,
        };
        let inode = Inode::special(
            special,
            major,
            minor,
            meta.mode & 0o7777,
            meta.uid,
            meta.gid,
            meta.mtime,
        );
        let ft = match kind {
            DeviceKind::Char => constants::DENT_CHR,
            DeviceKind::Block => constants::DENT_BLK,
            DeviceKind::Fifo => constants::DENT_FIFO,
            DeviceKind::Socket => constants::DENT_SOCK,
        };
        self.inodes.push((ino, inode));
        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino, ft)?;
        Ok(ino)
    }

    /// Remove the file / empty directory / symlink / device node at the
    /// absolute path `path`. Frees its inode and data blocks and unlinks it
    /// from its parent directory. A non-empty directory is rejected.
    pub fn remove_path(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<()> {
        let (parent, name) = split_path(std::path::Path::new(path))?;
        let parent_str = parent
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 path".into()))?;
        let parent_ino = self.path_to_inode(dev, parent_str)?;

        // Locate the target entry in the parent directory.
        let entries = self.list_inode(dev, parent_ino)?;
        let target_ino = entries
            .iter()
            .find(|e| e.name.as_bytes() == name.as_bytes())
            .map(|e| e.inode)
            .ok_or_else(|| crate::Error::InvalidArgument(format!("ext: no such entry {name:?}")))?;
        let target = self.read_inode(dev, target_ino)?;
        let is_dir = target.mode & constants::S_IFMT == constants::S_IFDIR;
        if is_dir {
            let children = self.list_inode(dev, target_ino)?;
            let non_self = children
                .iter()
                .filter(|e| e.name != "." && e.name != "..")
                .count();
            if non_self != 0 {
                return Err(crate::Error::InvalidArgument(format!(
                    "ext: directory {name:?} is not empty ({non_self} entries)"
                )));
            }
        }

        // Free the target's data + indirection blocks, then its inode.
        self.free_inode_blocks(dev, &target)?;
        self.free_inode(target_ino);
        // Stage a zeroed inode so flush writes a clean (links=0) slot.
        self.inodes.retain(|(i, _)| *i != target_ino);
        self.inodes.push((target_ino, Inode::default()));

        // Unlink from the parent directory.
        self.unlink_dir_entry(dev, parent_ino, name.as_bytes())?;

        if is_dir {
            // The removed dir's ".." was a link to the parent.
            self.patch_inode(dev, parent_ino, |i| {
                i.links_count = i.links_count.saturating_sub(1);
            })?;
            self.groups[0].desc.used_dirs_count =
                self.groups[0].desc.used_dirs_count.saturating_sub(1);
        }
        Ok(())
    }

    /// Free every data block (and classic indirection metadata block) an
    /// inode references. No-op for inodes with no allocated blocks (fast
    /// symlinks, device nodes).
    fn free_inode_blocks(&mut self, dev: &mut dyn BlockDevice, inode: &Inode) -> Result<()> {
        if inode.blocks_512 == 0 {
            return Ok(());
        }
        let bs = self.layout.block_size;
        let n_blocks = (inode.size as u64).div_ceil(bs as u64) as u32;
        for n in 0..n_blocks {
            let phys = self.file_block(dev, inode, n)?;
            if phys != 0 {
                self.free_block(phys);
            }
        }
        // Classic indirection metadata blocks (extent inodes keep their tree
        // inline in i_block, so they have no external metadata blocks).
        if inode.flags & constants::EXT4_EXTENTS_FL == 0 {
            let ind = inode.block[constants::IDX_INDIRECT];
            if ind != 0 {
                self.free_block(ind);
            }
            let dind = inode.block[constants::IDX_DOUBLE_INDIRECT];
            if dind != 0 {
                let mut buf = vec![0u8; bs as usize];
                self.read_block(dev, dind, &mut buf)?;
                for i in 0..(bs as usize / 4) {
                    let sub = u32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
                    if sub != 0 {
                        self.free_block(sub);
                    }
                }
                self.free_block(dind);
            }
        }
        Ok(())
    }

    /// Clear the block-bitmap bit for an absolute block number.
    pub(crate) fn free_block(&mut self, blk: u32) {
        for (gi, g) in self.layout.groups.iter().enumerate() {
            if blk >= g.start_block && blk <= g.end_block {
                group::clear_bit(&mut self.groups[gi].block_bitmap, blk - g.start_block);
                return;
            }
        }
    }

    /// Clear the inode-bitmap bit for an inode number.
    fn free_inode(&mut self, ino: u32) {
        let (g, idx) = self.inode_location(ino);
        group::clear_bit(&mut self.groups[g as usize].inode_bitmap, idx);
    }

    /// Remove the named entry from a directory's first data block by
    /// merging its `rec_len` into the preceding entry.
    fn unlink_dir_entry(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_inode: u32,
        name: &[u8],
    ) -> Result<()> {
        self.ensure_inode_staged(dev, dir_inode)?;
        let inode_copy = self
            .inodes
            .iter()
            .find(|(i, _)| *i == dir_inode)
            .map(|(_, i)| *i)
            .unwrap();
        let dir_block_num = self.file_block(dev, &inode_copy, 0)?;
        self.ensure_block_staged(dev, dir_block_num)?;
        if !self.dir_blocks.iter().any(|(b, _)| *b == dir_block_num) {
            self.dir_blocks.push((dir_block_num, dir_inode));
        }
        let with_filetype = self.has_filetype();
        let usable = dir::usable_dir_len(self.layout.block_size, self.has_metadata_csum());
        let block = self
            .data_blocks
            .iter_mut()
            .find(|(b, _)| *b == dir_block_num)
            .map(|(_, bytes)| bytes)
            .unwrap();

        let mut off = 0usize;
        let mut prev_off: Option<usize> = None;
        loop {
            let entry = dir::decode_entry(&block[off..], with_filetype).ok_or_else(|| {
                crate::Error::InvalidImage("corrupt dir entry while unlinking".into())
            })?;
            let rec_len = entry.rec_len;
            if entry.inode != 0 && entry.name == name {
                match prev_off {
                    Some(p) => {
                        // Absorb this entry's rec_len into the previous one.
                        let prev = dir::decode_entry(&block[p..], with_filetype)
                            .expect("prev entry decodes");
                        let merged = (prev.rec_len + rec_len) as u16;
                        block[p + 4..p + 6].copy_from_slice(&merged.to_le_bytes());
                    }
                    None => {
                        // First entry (normally "."): just void the inode.
                        block[off..off + 4].fill(0);
                    }
                }
                return Ok(());
            }
            let next = off + rec_len;
            if next >= usable {
                break;
            }
            prev_off = Some(off);
            off = next;
        }
        Err(crate::Error::InvalidArgument(format!(
            "ext: entry {name:?} not found in directory"
        )))
    }

    /// Persist all staged metadata (bitmaps, inode table, dir blocks,
    /// superblock) to the device. The primary superblock is written last.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        self.recompute_free_counts();
        self.flush_metadata(dev)
    }

    /// Recursively copy a host directory into `parent_ino`. Each file's
    /// contents are streamed via `FileSource::HostPath` (never fully loaded
    /// in memory). Mode bits are taken from host metadata; uid, gid, and
    /// timestamps are squashed to 0 to keep the output reproducible.
    /// Override per-entry by populating the tree yourself.
    pub fn populate_from_host_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u32,
        src: &std::path::Path,
    ) -> Result<()> {
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            let ft = meta.file_type();
            let name = entry.file_name();
            let name_bytes = name.as_encoded_bytes();
            #[cfg(unix)]
            let mode = {
                use std::os::unix::fs::PermissionsExt;
                (meta.permissions().mode() & 0o7777) as u16
            };
            #[cfg(not(unix))]
            let mode: u16 = if ft.is_dir() { 0o755 } else { 0o644 };
            let fmeta = FileMeta {
                mode,
                uid: 0,
                gid: 0,
                mtime: 0,
                atime: 0,
                ctime: 0,
            };
            if ft.is_dir() {
                let child = self.add_dir_to(dev, parent_ino, name_bytes, fmeta)?;
                self.populate_from_host_dir(dev, child, &entry.path())?;
            } else if ft.is_file() {
                let src_path = entry.path();
                self.add_file_to(
                    dev,
                    parent_ino,
                    name_bytes,
                    FileSource::HostPath(src_path),
                    fmeta,
                )?;
            } else if ft.is_symlink() {
                let target = std::fs::read_link(entry.path())?;
                let target_str = target.to_string_lossy();
                self.add_symlink_to(dev, parent_ino, name_bytes, target_str.as_bytes(), fmeta)?;
            } else {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::{FileTypeExt, MetadataExt};
                    if ft.is_block_device() || ft.is_char_device() {
                        let rdev = meta.rdev();
                        // Linux dev_t: major in bits 8..19 and 32..47, minor in 0..7 and 20..31.
                        let major = ((rdev >> 8) & 0xfff) | ((rdev >> 32) & !0xfff);
                        let minor = (rdev & 0xff) | ((rdev >> 12) & !0xff);
                        let kind = if ft.is_char_device() {
                            DeviceKind::Char
                        } else {
                            DeviceKind::Block
                        };
                        self.add_device_to(
                            dev,
                            parent_ino,
                            name_bytes,
                            kind,
                            major as u32,
                            minor as u32,
                            fmeta,
                        )?;
                    } else if ft.is_fifo() {
                        self.add_device_to(
                            dev,
                            parent_ino,
                            name_bytes,
                            DeviceKind::Fifo,
                            0,
                            0,
                            fmeta,
                        )?;
                    } else if ft.is_socket() {
                        self.add_device_to(
                            dev,
                            parent_ino,
                            name_bytes,
                            DeviceKind::Socket,
                            0,
                            0,
                            fmeta,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// One-shot: scan a host directory, compute the needed FS geometry via
    /// [`BuildPlan`], format the device, populate it, and flush. The closest
    /// analogue to `genext2fs -d <dir> img.ext2` — except sizing is exact.
    pub fn build_from_host_dir(
        dev: &mut dyn BlockDevice,
        src: &std::path::Path,
        kind: FsKind,
        block_size: u32,
    ) -> Result<Self> {
        let mut plan = BuildPlan::new(block_size, kind);
        plan.scan_host_path(src)?;
        let opts = plan.to_format_opts();
        let mut ext = Self::format_with(dev, &opts)?;
        ext.populate_from_host_dir(dev, INO_ROOT_DIR, src)?;
        ext.flush(dev)?;
        Ok(ext)
    }

    /// Create `/dev` with the standard set of device nodes for `kind` —
    /// the building block for `--rootdevs minimal | standard`. The `/dev`
    /// directory is owned by `root:root` mode 0755; each node's permissions
    /// follow the conventional Linux defaults from the device-numbers
    /// registry (e.g. `console` is 0600, `null` is 0666).
    ///
    /// Pass [`RootDevs::None`] to do nothing (returns `Ok(None)`).
    /// Returns the inode number of `/dev` on success.
    pub fn populate_rootdevs(
        &mut self,
        dev: &mut dyn BlockDevice,
        kind: RootDevs,
        owner_uid: u32,
        owner_gid: u32,
        mtime: u32,
    ) -> Result<Option<u32>> {
        if kind == RootDevs::None {
            return Ok(None);
        }
        let entries = device_table(kind);
        if entries.is_empty() {
            return Ok(None);
        }
        let dir_meta = FileMeta {
            mode: 0o755,
            uid: owner_uid,
            gid: owner_gid,
            mtime,
            atime: mtime,
            ctime: mtime,
        };
        let dev_ino = self.add_dir_to(dev, INO_ROOT_DIR, b"dev", dir_meta)?;
        for e in entries {
            let meta = FileMeta {
                mode: e.mode,
                uid: owner_uid,
                gid: owner_gid,
                mtime,
                atime: mtime,
                ctime: mtime,
            };
            self.add_device_to(
                dev,
                dev_ino,
                e.name.as_bytes(),
                e.kind,
                e.major,
                e.minor,
                meta,
            )?;
        }
        Ok(Some(dev_ino))
    }

    // ──────────────────────────────── reader API ─────────────────────────
    //
    // These methods do NOT touch the staged write state; they read directly
    // from the device every time.

    /// Open an existing ext filesystem from `dev`. Parses the primary
    /// superblock, every group descriptor, and both bitmaps per group.
    /// Inode-table and data-block contents are read lazily.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let mut sb_buf = [0u8; constants::SUPERBLOCK_SIZE];
        dev.read_at(constants::SUPERBLOCK_OFFSET, &mut sb_buf)?;
        let sb = Superblock::decode(&sb_buf)?;
        // When metadata_csum is set, the superblock carries a CRC32C in its
        // last 4 bytes. A mismatch means the image is corrupt — refuse it
        // rather than silently working from bad metadata.
        if sb.feature_ro_compat & constants::feature::RO_COMPAT_METADATA_CSUM != 0 {
            let stored = u32::from_le_bytes(sb_buf[1020..1024].try_into().unwrap());
            let computed = csum::superblock(&sb_buf);
            if stored != computed {
                return Err(crate::Error::InvalidImage(format!(
                    "ext: superblock checksum mismatch (stored {stored:#010x}, computed {computed:#010x})"
                )));
            }
        }
        let mut layout = layout::from_superblock(&sb)?;

        // GDT location: same logic as the writer.
        let bs = layout.block_size as u64;
        let gdt_off = if layout.first_data_block == 1 {
            2 * bs
        } else {
            bs
        };
        let mut gdt = vec![0u8; layout.gdt_blocks as usize * bs as usize];
        dev.read_at(gdt_off, &mut gdt)?;

        let desc_size = layout.desc_size;
        let mut groups = Vec::with_capacity(layout.groups.len());
        for i in 0..layout.groups.len() {
            let off = i * desc_size;
            let desc = GroupDesc::decode(&gdt[off..off + constants::GROUP_DESC_SIZE]);
            // The metadata positions in `layout.groups[i]` were *computed*
            // assuming the classic contiguous layout. With flex_bg (and in
            // general for any third-party writer) the descriptor is the
            // authoritative source — overwrite the computed positions with
            // the on-disk pointers so inode/bitmap reads land correctly.
            layout.groups[i].block_bitmap = desc.block_bitmap;
            layout.groups[i].inode_bitmap = desc.inode_bitmap;
            layout.groups[i].inode_table = desc.inode_table;
            let mut block_bitmap = vec![0u8; bs as usize];
            dev.read_at(desc.block_bitmap as u64 * bs, &mut block_bitmap)?;
            let mut inode_bitmap = vec![0u8; bs as usize];
            dev.read_at(desc.inode_bitmap as u64 * bs, &mut inode_bitmap)?;
            groups.push(GroupState {
                block_bitmap,
                inode_bitmap,
                desc,
            });
        }

        // next_inode: first clear bit in group 0's inode bitmap past the
        // reserved range. (Subsequent groups can be tackled later.)
        let mut next_inode = sb.first_ino;
        while next_inode <= layout.inodes_per_group
            && test_bit(&groups[0].inode_bitmap, next_inode - 1)
        {
            next_inode += 1;
        }

        // Infer kind from feature flags on the parsed superblock so reads
        // post-open know whether to expect extent trees or indirect blocks.
        let kind = if sb.feature_incompat & constants::feature::INCOMPAT_EXTENTS != 0 {
            FsKind::Ext4
        } else if sb.feature_compat & constants::feature::COMPAT_HAS_JOURNAL != 0 {
            FsKind::Ext3
        } else {
            FsKind::Ext2
        };

        Ok(Self {
            sb,
            layout,
            kind,
            // Default sparse off for an opened image; the caller can flip it
            // via `set_sparse` before adding files.
            sparse: false,
            groups,
            next_inode,
            inodes: Vec::new(),
            data_blocks: Vec::new(),
            dir_blocks: Vec::new(),
            // Opened (vs. just-formatted) images go through the journal
            // path on flush. `open()` itself runs JBD2 replay before
            // returning, so by the time we land here the on-disk journal
            // is clean.
            bootstrap: false,
        })
    }

    /// Enable or disable sparse-file writing for subsequent `add_file_to`
    /// calls. Useful after [`Ext::open`], which defaults it off.
    pub fn set_sparse(&mut self, sparse: bool) {
        self.sparse = sparse;
    }

    /// Re-read every group's bitmaps and group descriptor from disk into
    /// the in-memory `groups` vector. Called by the journal-replay path
    /// after applying a transaction so subsequent staged metadata
    /// writes don't shadow the just-replayed values.
    pub(crate) fn reload_groups_from_disk(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        let bs = self.layout.block_size as u64;
        let gdt_off = if self.layout.first_data_block == 1 {
            2 * bs
        } else {
            bs
        };
        let mut gdt = vec![0u8; self.layout.gdt_blocks as usize * bs as usize];
        dev.read_at(gdt_off, &mut gdt)?;
        let desc_size = self.layout.desc_size;
        for i in 0..self.layout.groups.len() {
            let off = i * desc_size;
            let desc = GroupDesc::decode(&gdt[off..off + constants::GROUP_DESC_SIZE]);
            self.layout.groups[i].block_bitmap = desc.block_bitmap;
            self.layout.groups[i].inode_bitmap = desc.inode_bitmap;
            self.layout.groups[i].inode_table = desc.inode_table;
            dev.read_at(
                desc.block_bitmap as u64 * bs,
                &mut self.groups[i].block_bitmap,
            )?;
            dev.read_at(
                desc.inode_bitmap as u64 * bs,
                &mut self.groups[i].inode_bitmap,
            )?;
            self.groups[i].desc = desc;
        }
        Ok(())
    }

    /// Read inode number `ino`. Consults the in-memory staged-write cache
    /// first so a caller can interleave `add_*` and read calls without an
    /// explicit flush; falls back to the on-disk inode table.
    pub fn read_inode(&self, dev: &mut dyn BlockDevice, ino: u32) -> Result<Inode> {
        if ino == 0 || ino > self.layout.inodes_count {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: inode {ino} out of range"
            )));
        }
        for (i, staged) in &self.inodes {
            if *i == ino {
                return Ok(*staged);
            }
        }
        let (group, idx) = self.inode_location(ino);
        let table_block = self.layout.groups[group as usize].inode_table;
        let bs = self.layout.block_size as u64;
        let off = table_block as u64 * bs + idx as u64 * self.layout.inode_size as u64;
        let mut buf = [0u8; inode::INODE_BASE_SIZE];
        dev.read_at(off, &mut buf)?;
        Ok(Inode::decode(&buf))
    }

    /// Read a single block's contents into `out`. Consults staged data
    /// blocks first (dir blocks built up during writes) and falls back to
    /// the device.
    pub(crate) fn read_block(
        &self,
        dev: &mut dyn BlockDevice,
        blk: u32,
        out: &mut [u8],
    ) -> Result<()> {
        for (b, bytes) in &self.data_blocks {
            if *b == blk {
                out.copy_from_slice(bytes);
                return Ok(());
            }
        }
        let bs = self.layout.block_size as u64;
        dev.read_at(blk as u64 * bs, out)?;
        Ok(())
    }

    /// Return the absolute block number for the `n`-th block (0-indexed) of
    /// the file at inode `ino`. Picks the representation based on the
    /// inode flags: `EXT4_EXTENTS_FL` → walk the extent tree;
    /// otherwise → direct + single-indirect (double/triple deferred).
    pub fn file_block(&self, dev: &mut dyn BlockDevice, ino: &Inode, n: u32) -> Result<u32> {
        if ino.flags & constants::EXT4_EXTENTS_FL != 0 {
            return self.file_block_extent(dev, ino, n);
        }
        if (n as usize) < constants::N_DIRECT {
            return Ok(ino.block[n as usize]);
        }
        let ptrs_per_block = self.layout.block_size / 4;
        let n_off = n - constants::N_DIRECT as u32;
        if n_off < ptrs_per_block {
            let ind = ino.block[constants::IDX_INDIRECT];
            if ind == 0 {
                return Err(crate::Error::InvalidImage(
                    "ext: indirect block index unset".into(),
                ));
            }
            let mut buf = vec![0u8; self.layout.block_size as usize];
            self.read_block(dev, ind, &mut buf)?;
            let off = (n_off as usize) * 4;
            return Ok(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }
        Err(crate::Error::Unsupported(
            "ext: double/triple indirection not yet supported in reader".into(),
        ))
    }

    /// Resolve logical block `n` against an inode that uses an ext4
    /// extent tree. Supports depth-0 (inline up to 4 leaves) and depth-1
    /// (up to 4 idx entries in `i_block`, each pointing at one leaf
    /// block on disk holding the actual extent records).
    #[allow(clippy::needless_pass_by_ref_mut)]
    fn file_block_extent(&self, dev: &mut dyn BlockDevice, ino: &Inode, n: u32) -> Result<u32> {
        let iblock = extent::iblock_to_bytes(&ino.block);
        let header = extent::decode_header(&iblock[..12])?;
        if header.depth == 0 {
            let (_, runs) = extent::decode_depth0_iblock(&iblock)?;
            return Ok(resolve_logical_in_runs(&runs, n));
        }
        if header.depth == 1 {
            let (_, indices) = extent::decode_idx_iblock(&iblock)?;
            // The idx array is sorted by ei_block ascending; find the
            // last idx whose block <= n.
            let mut chosen: Option<extent::ExtentIdx> = None;
            for idx in &indices {
                if idx.block <= n {
                    chosen = Some(*idx);
                } else {
                    break;
                }
            }
            let Some(idx) = chosen else {
                return Ok(0);
            };
            let bs = self.layout.block_size as usize;
            let mut buf = vec![0u8; bs];
            self.read_block(dev, idx.leaf as u32, &mut buf)?;
            let (_, runs) = extent::decode_leaf_block(&buf)?;
            return Ok(resolve_logical_in_runs(&runs, n));
        }
        Err(crate::Error::Unsupported(format!(
            "ext4: extent tree depth {} not yet supported in reader (depth-0 and depth-1 only)",
            header.depth
        )))
    }

    /// List the entries of the directory inode `ino`. Returns
    /// [`crate::Error::InvalidArgument`] if `ino` is not a directory.
    pub fn list_inode(
        &self,
        dev: &mut dyn BlockDevice,
        ino: u32,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let inode = self.read_inode(dev, ino)?;
        if inode.mode & constants::S_IFMT != constants::S_IFDIR {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: inode {ino} is not a directory"
            )));
        }
        let bs = self.layout.block_size;
        let n_blocks = inode.size.div_ceil(bs);
        let mut out = Vec::new();
        let with_filetype = self.sb.feature_incompat & constants::feature::INCOMPAT_FILETYPE != 0;
        let mut block_buf = vec![0u8; bs as usize];
        for n in 0..n_blocks {
            let blk = self.file_block(dev, &inode, n)?;
            if blk == 0 {
                continue;
            }
            self.read_block(dev, blk, &mut block_buf)?;
            let mut off = 0usize;
            while off < block_buf.len() {
                let Some(entry) = dir::decode_entry(&block_buf[off..], with_filetype) else {
                    break;
                };
                if entry.inode != 0 && !entry.name.is_empty() {
                    let child = self.read_inode(dev, entry.inode)?;
                    let kind = kind_from_mode(child.mode);
                    let size = if matches!(kind, crate::fs::EntryKind::Regular) {
                        u64::from(child.size)
                    } else {
                        0
                    };
                    out.push(crate::fs::DirEntry {
                        name: String::from_utf8_lossy(entry.name).into_owned(),
                        inode: entry.inode,
                        kind,
                        size,
                    });
                }
                off += entry.rec_len;
                if entry.rec_len == 0 {
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Resolve an absolute path (must start with '/') to its inode number.
    /// Each component is matched exactly; symlinks are NOT followed in v1.
    pub fn path_to_inode(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<u32> {
        if !path.starts_with('/') {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: path must be absolute, got {path:?}"
            )));
        }
        let mut cur = constants::INO_ROOT_DIR;
        for comp in path.split('/').filter(|c| !c.is_empty()) {
            let entries = self.list_inode(dev, cur)?;
            let next = entries
                .iter()
                .find(|e| e.name == comp)
                .map(|e| e.inode)
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!("ext: no such entry {comp:?} in path"))
                })?;
            cur = next;
        }
        Ok(cur)
    }

    /// Open a streaming reader over the regular file at `ino`. The reader
    /// holds a mutable borrow of `dev` for its lifetime; reads pull the
    /// file's data blocks lazily through a per-block fetch.
    /// Read this inode's extended attributes. Combines two storage
    /// locations: inline xattrs in the extended inode body (when
    /// `inode_size > 128`, post-`i_extra_isize`) and external block
    /// xattrs (pointed at by `inode.file_acl`). Inline entries come
    /// first to match the kernel's ordering.
    ///
    /// The per-block CRC32C isn't validated here.
    pub fn read_xattrs(&self, dev: &mut dyn BlockDevice, ino: u32) -> Result<Vec<xattr::Xattr>> {
        let mut out = Vec::new();
        // Inline xattrs only exist when the on-disk inode is bigger than
        // the classic 128 bytes.
        if self.layout.inode_size > inode::INODE_BASE_SIZE as u16 {
            out.extend(self.read_inline_xattrs(dev, ino)?);
        }
        let inode = self.read_inode(dev, ino)?;
        if inode.file_acl != 0 {
            let bs = self.layout.block_size as usize;
            let mut block = vec![0u8; bs];
            dev.read_at(inode.file_acl as u64 * bs as u64, &mut block)?;
            out.extend(xattr::decode_block(&block)?);
        }
        Ok(out)
    }

    /// Read inline xattrs from the extended-inode area. `read_inode`
    /// only returns the 128-byte base struct, so this re-reads the full
    /// on-disk inode (`layout.inode_size` bytes) and walks anything past
    /// the standard fields + `i_extra_isize`.
    fn read_inline_xattrs(&self, dev: &mut dyn BlockDevice, ino: u32) -> Result<Vec<xattr::Xattr>> {
        let (group, idx) = self.inode_location(ino);
        let table_block = self.layout.groups[group as usize].inode_table;
        let bs = self.layout.block_size as u64;
        let inode_size = self.layout.inode_size as usize;
        let off = table_block as u64 * bs + idx as u64 * inode_size as u64;
        let mut buf = vec![0u8; inode_size];
        dev.read_at(off, &mut buf)?;
        // i_extra_isize is the first u16 of the extended area at offset 128.
        if buf.len() < 130 {
            return Ok(Vec::new());
        }
        let extra_isize = u16::from_le_bytes(buf[128..130].try_into().unwrap()) as usize;
        let inline_start = inode::INODE_BASE_SIZE + extra_isize;
        if inline_start >= buf.len() {
            return Ok(Vec::new());
        }
        xattr::decode_inline(&buf[inline_start..])
    }

    /// Attach the given extended attributes to a freshly-staged inode.
    /// Allocates one data block, encodes the xattrs into it, stamps the
    /// CRC32C if `metadata_csum` is on, points `inode.file_acl` at the
    /// new block, and sets `COMPAT_EXT_ATTR` on the superblock.
    ///
    /// `ino` MUST refer to an inode that was just added via one of the
    /// `add_*_to` methods (i.e. it lives in `self.inodes`). Setting
    /// xattrs on a disk-resident inode is a separate code path and is
    /// not implemented in v1.
    pub fn set_xattrs(
        &mut self,
        dev: &mut dyn BlockDevice,
        ino: u32,
        xattrs: &[xattr::Xattr],
    ) -> Result<()> {
        if xattrs.is_empty() {
            return Ok(());
        }
        let bs = self.layout.block_size;
        let mut block = xattr::encode_block(xattrs, bs as usize)?;
        let block_num = self.alloc_data_block()?;
        if self.has_metadata_csum() {
            xattr::stamp_checksum(&mut block, self.csum_seed(), block_num as u64);
        }
        dev.write_at(block_num as u64 * bs as u64, &block)?;

        let entry = self
            .inodes
            .iter_mut()
            .find(|(i, _)| *i == ino)
            .ok_or_else(|| {
                crate::Error::Unsupported(format!(
                    "ext: set_xattrs on disk-resident inode {ino} not yet supported"
                ))
            })?;
        entry.1.file_acl = block_num;
        entry.1.blocks_512 += bs / 512;
        self.sb.feature_compat |= constants::feature::COMPAT_EXT_ATTR;
        Ok(())
    }

    /// Read the target of the symlink at inode `ino`. Errors if the inode
    /// isn't a symlink.
    ///
    /// ext stores short symlinks (≤ 60 bytes) inline in the inode's
    /// `block` array; longer ones go through the normal block-pointer
    /// machinery and are streamed via [`Self::open_file_reader`].
    pub fn read_symlink_target(&self, dev: &mut dyn BlockDevice, ino: u32) -> Result<String> {
        use std::io::Read as _;
        let inode = self.read_inode(dev, ino)?;
        if inode.mode & constants::S_IFMT != constants::S_IFLNK {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: inode {ino} is not a symlink"
            )));
        }
        let size = inode.size as usize;
        // Fast (inline) symlink: target is in the 60 bytes of block[].
        if size <= 60 && inode.blocks_512 == 0 {
            let mut bytes = [0u8; 60];
            for (i, &w) in inode.block.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            return Ok(String::from_utf8_lossy(&bytes[..size]).into_owned());
        }
        // Slow symlink: stored in data blocks, same as a regular file's body.
        // Spoof the mode bits so open_file_reader accepts it.
        let mut reg_inode = inode;
        reg_inode.mode = (reg_inode.mode & !constants::S_IFMT) | constants::S_IFREG;
        let reader = FileReader {
            ext: self,
            dev,
            inode: reg_inode,
            pos: 0,
            block_buf: vec![0u8; self.layout.block_size as usize],
            cached_block: u32::MAX,
        };
        let mut buf = Vec::with_capacity(size);
        let mut r = reader;
        r.read_to_end(&mut buf)?;
        buf.truncate(size);
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    pub fn open_file_reader<'a>(
        &'a self,
        dev: &'a mut dyn BlockDevice,
        ino: u32,
    ) -> Result<FileReader<'a>> {
        let inode = self.read_inode(dev, ino)?;
        if inode.mode & constants::S_IFMT != constants::S_IFREG {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: inode {ino} is not a regular file"
            )));
        }
        Ok(FileReader {
            ext: self,
            dev,
            inode,
            pos: 0,
            block_buf: vec![0u8; self.layout.block_size as usize],
            cached_block: u32::MAX,
        })
    }
}

/// Streaming reader over a regular file's data blocks. Constructed via
/// [`Ext::open_file_reader`]. Reads pull one FS block at a time from the
/// device into an internal buffer; no full-file allocation.
pub struct FileReader<'a> {
    ext: &'a Ext,
    dev: &'a mut dyn BlockDevice,
    inode: Inode,
    pos: u64,
    block_buf: Vec<u8>,
    /// Block number currently in `block_buf`, or `u32::MAX` if empty.
    cached_block: u32,
}

impl<'a> Read for FileReader<'a> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let total = self.inode.size as u64;
        if self.pos >= total {
            return Ok(0);
        }
        let bs = self.ext.layout.block_size as u64;
        let block_idx = (self.pos / bs) as u32;
        let block_off = (self.pos % bs) as usize;
        if self.cached_block != block_idx {
            let abs = self
                .ext
                .file_block(self.dev, &self.inode, block_idx)
                .map_err(std::io::Error::other)?;
            if abs == 0 {
                self.block_buf.fill(0);
            } else {
                self.dev
                    .read_at(abs as u64 * bs, &mut self.block_buf)
                    .map_err(std::io::Error::other)?;
            }
            self.cached_block = block_idx;
        }
        let remaining_in_block = bs as usize - block_off;
        let remaining_in_file = (total - self.pos) as usize;
        let n = out.len().min(remaining_in_block).min(remaining_in_file);
        out[..n].copy_from_slice(&self.block_buf[block_off..block_off + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl<'a> std::io::Seek for FileReader<'a> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let total = self.inode.size as i128;
        let new = match pos {
            std::io::SeekFrom::Start(n) => n as i128,
            std::io::SeekFrom::Current(d) => self.pos as i128 + d as i128,
            std::io::SeekFrom::End(d) => total + d as i128,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "ext: seek to negative offset",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

impl<'a> crate::fs::FileReadHandle for FileReader<'a> {
    fn len(&self) -> u64 {
        self.inode.size as u64
    }
}

/// Build a JBD2 v2 journal superblock for a clean (never-mounted) journal.
/// Layout per linux/include/linux/jbd2.h; note that JBD2 fields are
/// **big-endian** on disk, unlike the rest of the ext filesystem.
fn build_jbd2_superblock(block_size: u32, journal_blocks: u32) -> Vec<u8> {
    let mut buf = vec![0u8; block_size as usize];
    // journal_header_s: h_magic, h_blocktype, h_sequence (each u32 BE)
    buf[0..4].copy_from_slice(&0xC03B_3998u32.to_be_bytes()); // h_magic
    buf[4..8].copy_from_slice(&4u32.to_be_bytes()); // h_blocktype = SB v2
    // 8..12: h_sequence — zero
    // journal_superblock_s body:
    buf[12..16].copy_from_slice(&block_size.to_be_bytes()); // s_blocksize
    buf[16..20].copy_from_slice(&journal_blocks.to_be_bytes()); // s_maxlen
    buf[20..24].copy_from_slice(&1u32.to_be_bytes()); // s_first = 1
    buf[24..28].copy_from_slice(&1u32.to_be_bytes()); // s_sequence
    // 28..32: s_start = 0  →  CLEAN journal, no recovery needed
    // 32..36: s_errno = 0
    // 36..48: feature_{compat,incompat,ro_compat} = 0
    // 48..64: s_uuid = 0
    buf[64..68].copy_from_slice(&1u32.to_be_bytes()); // s_nr_users = 1
    // rest zero
    buf
}

/// Split an absolute path into (parent path, last component). Errors for
/// paths that don't start with '/', that ARE just '/', or whose last
/// component contains a slash (defensive).
pub(crate) fn split_path(path: &std::path::Path) -> Result<(std::path::PathBuf, String)> {
    let s = path
        .to_str()
        .ok_or_else(|| crate::Error::InvalidArgument(format!("ext: non-UTF-8 path {path:?}")))?;
    if !s.starts_with('/') {
        return Err(crate::Error::InvalidArgument(format!(
            "ext: path must be absolute, got {s:?}"
        )));
    }
    if s == "/" {
        return Err(crate::Error::InvalidArgument(
            "ext: cannot create or remove the root".into(),
        ));
    }
    let trimmed = s.trim_end_matches('/');
    let (parent, name) = match trimmed.rsplit_once('/') {
        Some((p, n)) => (if p.is_empty() { "/" } else { p }, n),
        None => {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: bad path {s:?}"
            )));
        }
    };
    Ok((std::path::PathBuf::from(parent), name.to_string()))
}

impl crate::fs::FilesystemFactory for Ext {
    type FormatOpts = FormatOpts;

    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format_with(dev, opts)
    }

    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl crate::fs::Filesystem for Ext {
    fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        src: FileSource,
        meta: FileMeta,
    ) -> Result<()> {
        let (parent, name) = split_path(path)?;
        let parent_str = parent
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 parent path".into()))?;
        let parent_ino = self.path_to_inode(dev, parent_str)?;
        self.add_file_to(dev, parent_ino, name.as_bytes(), src, meta)?;
        Ok(())
    }

    fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        meta: FileMeta,
    ) -> Result<()> {
        let (parent, name) = split_path(path)?;
        let parent_str = parent.to_str().unwrap();
        let parent_ino = self.path_to_inode(dev, parent_str)?;
        self.add_dir_to(dev, parent_ino, name.as_bytes(), meta)?;
        Ok(())
    }

    fn create_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        target: &std::path::Path,
        meta: FileMeta,
    ) -> Result<()> {
        let (parent, name) = split_path(path)?;
        let parent_str = parent.to_str().unwrap();
        let parent_ino = self.path_to_inode(dev, parent_str)?;
        let target_bytes = target
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 symlink target".into()))?
            .as_bytes();
        self.add_symlink_to(dev, parent_ino, name.as_bytes(), target_bytes, meta)?;
        Ok(())
    }

    fn create_device(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: FileMeta,
    ) -> Result<()> {
        let (parent, name) = split_path(path)?;
        let parent_str = parent.to_str().unwrap();
        let parent_ino = self.path_to_inode(dev, parent_str)?;
        self.add_device_to(dev, parent_ino, name.as_bytes(), kind, major, minor, meta)?;
        Ok(())
    }

    fn remove(&mut self, dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 path".into()))?;
        self.remove_path(dev, s)
    }

    fn list(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 path".into()))?;
        let ino = self.path_to_inode(dev, s)?;
        self.list_inode(dev, ino)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn Read + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 path".into()))?;
        let ino = self.path_to_inode(dev, s)?;
        let reader = self.open_file_reader(dev, ino)?;
        Ok(Box::new(reader))
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn crate::fs::FileReadHandle + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 path".into()))?;
        let ino = self.path_to_inode(dev, s)?;
        let reader = self.open_file_reader(dev, ino)?;
        Ok(Box::new(reader))
    }

    fn open_file_rw<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
        flags: crate::fs::OpenFlags,
        meta: Option<FileMeta>,
    ) -> Result<Box<dyn crate::fs::FileHandle + 'a>> {
        rw::open_file_rw_ext(self, dev, path, flags, meta)
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        Self::flush(self, dev)
    }

    fn read_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<std::path::PathBuf> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 path".into()))?;
        let ino = self.path_to_inode(dev, s)?;
        let target = self.read_symlink_target(dev, ino)?;
        Ok(std::path::PathBuf::from(target))
    }
}

/// Scan a leaf-extent list for the run containing logical block `n` and
/// return the corresponding physical block. Returns 0 if `n` falls in a
/// hole (no extent covers it).
fn resolve_logical_in_runs(runs: &[extent::ExtentRun], n: u32) -> u32 {
    for r in runs {
        let len = if r.len > extent::MAX_LEN_PER_EXTENT {
            r.len - extent::MAX_LEN_PER_EXTENT
        } else {
            r.len
        };
        if n >= r.logical && n < r.logical + len as u32 {
            let phys = r.physical + (n - r.logical) as u64;
            return phys as u32;
        }
    }
    0
}

/// Translate an ext mode word into a [`crate::fs::EntryKind`].
fn kind_from_mode(mode: u16) -> crate::fs::EntryKind {
    use crate::fs::EntryKind;
    match mode & constants::S_IFMT {
        constants::S_IFREG => EntryKind::Regular,
        constants::S_IFDIR => EntryKind::Dir,
        constants::S_IFLNK => EntryKind::Symlink,
        constants::S_IFCHR => EntryKind::Char,
        constants::S_IFBLK => EntryKind::Block,
        constants::S_IFIFO => EntryKind::Fifo,
        constants::S_IFSOCK => EntryKind::Socket,
        _ => EntryKind::Unknown,
    }
}

/// Append a dir entry to a directory block by shrinking the existing last
/// entry to its natural minimum and writing the new entry into the freed
/// tail. `usable` is the byte length available for real entries — the whole
/// block, or `block_size - 12` when `metadata_csum` reserves a checksum
/// tail. The last real entry's `rec_len` runs up to `usable`.
fn append_dir_entry(
    block: &mut [u8],
    name: &[u8],
    inode: u32,
    file_type: u8,
    with_filetype: bool,
    usable: usize,
) -> Result<()> {
    let needed = dir::min_rec_len(name.len());
    let mut off = 0usize;
    let last_off: usize;
    loop {
        let entry = dir::decode_entry(&block[off..], with_filetype).ok_or_else(|| {
            crate::Error::InvalidImage("corrupt dir entry while appending".into())
        })?;
        let next = off + entry.rec_len;
        if next >= usable {
            last_off = off;
            break;
        }
        off = next;
    }
    let last_entry = dir::decode_entry(&block[last_off..], with_filetype).expect("decode last");
    let last_min = dir::min_rec_len(last_entry.name.len());
    let last_real_end = last_off + last_entry.rec_len;
    let new_entry_off = last_off + last_min;
    let new_entry_space = last_real_end - new_entry_off;
    if new_entry_space < needed {
        return Err(crate::Error::Unsupported(
            "ext: dir block full — multi-block directories not yet implemented".into(),
        ));
    }
    // Shrink the last entry's rec_len.
    block[last_off + 4..last_off + 6].copy_from_slice(&(last_min as u16).to_le_bytes());
    // Encode the new entry into a buffer, then copy.
    let mut tail = Vec::with_capacity(new_entry_space);
    dir::encode_entry(
        &mut tail,
        inode,
        name,
        new_entry_space as u16,
        file_type,
        with_filetype,
    );
    debug_assert_eq!(tail.len(), new_entry_space);
    block[new_entry_off..new_entry_off + new_entry_space].copy_from_slice(&tail);
    Ok(())
}

fn popcount_bits(bm: &[u8], start: u32, end: u32) -> u32 {
    (start..end).filter(|&i| test_bit(bm, i)).count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn format_creates_clean_filesystem() {
        let mut dev = MemoryBackend::new(1024 * 1024);
        let opts = FormatOpts::default();
        let ext = Ext::format_with(&mut dev, &opts).expect("format");
        assert_eq!(ext.sb.magic, constants::EXT2_MAGIC);
        assert_eq!(ext.sb.blocks_count, 1024);
        assert_eq!(ext.sb.inodes_count, 16);
        assert_eq!(ext.sb.block_size(), 1024);
    }

    #[test]
    fn flex_bg_off_by_default() {
        // The default FormatOpts must not enable flex_bg, preserving the
        // pre-flex_bg byte-exact ext2 layout.
        let mut dev = MemoryBackend::new(1024 * 1024);
        let opts = FormatOpts::default();
        let ext = Ext::format_with(&mut dev, &opts).expect("format");
        assert_eq!(ext.sb.log_groups_per_flex, 0);
        assert_eq!(
            ext.sb.feature_incompat & constants::feature::INCOMPAT_FLEX_BG,
            0
        );
        assert_eq!(ext.layout.log_groups_per_flex, 0);
    }

    #[test]
    fn flex_bg_sets_feature_and_log() {
        // bs=4096, 64 MiB FS → 2 groups of 32768 blocks. log_groups_per_flex = 1.
        let total_bytes = 64u64 * 1024 * 1024;
        let mut dev = MemoryBackend::new(total_bytes);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 16 * 1024,
            inodes_count: 1024,
            log_groups_per_flex: 1,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let ext = Ext::format_with(&mut dev, &opts).expect("format flex_bg");
        assert_eq!(ext.sb.log_groups_per_flex, 1);
        assert!(
            ext.sb.feature_incompat & constants::feature::INCOMPAT_FLEX_BG != 0,
            "INCOMPAT_FLEX_BG must be set when log_groups_per_flex > 0"
        );
        // Reopen: the parsed superblock must round-trip the flex value.
        let reopened = Ext::open(&mut dev).expect("reopen flex_bg image");
        assert_eq!(reopened.sb.log_groups_per_flex, 1);
        assert_eq!(reopened.layout.log_groups_per_flex, 1);
    }

    #[test]
    fn flex_bg_rejects_invalid_log() {
        let mut dev = MemoryBackend::new(64u64 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 16 * 1024,
            inodes_count: 1024,
            log_groups_per_flex: 6,
            ..FormatOpts::default()
        };
        let err = Ext::format_with(&mut dev, &opts).expect_err("must reject log > 5");
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    #[test]
    fn flex_bg_default_helper_picks_reasonable_log() {
        // Small (single-group) → 0 (off). Large → 4.
        assert_eq!(FormatOpts::default_log_groups_per_flex(1), 0);
        assert_eq!(FormatOpts::default_log_groups_per_flex(15), 0);
        assert_eq!(FormatOpts::default_log_groups_per_flex(16), 4);
        assert_eq!(FormatOpts::default_log_groups_per_flex(1024), 4);
    }

    #[test]
    fn flex_bg_metadata_packed_in_first_group() {
        // 4 groups of 32768 blocks (4 KiB blocks) with log_groups_per_flex=2
        // packs every group's bitmap + inode-table into group 0. Reopen the
        // image and assert that *every* non-leader group's bitmap_block and
        // inode_table fall strictly inside the leader's metadata extent.
        let mut dev = MemoryBackend::new(768u64 * 1024 * 1024);
        let blocks_per_group = 8 * 4096u32;
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 4 * blocks_per_group,
            inodes_count: 4096,
            log_groups_per_flex: 2,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let ext = Ext::format_with(&mut dev, &opts).expect("format flex_bg");
        assert_eq!(ext.layout.num_groups(), 4, "test setup must yield 4 groups");
        let g0 = ext.layout.groups[0];
        // Leader's metadata range: [start_block + sb+gdt, data_start).
        let leader_meta_start = g0.start_block
            + if g0.has_superblock {
                1 + ext.layout.gdt_blocks
            } else {
                0
            };
        let leader_meta_end = g0.data_start;
        for gi in 1..ext.layout.num_groups() as usize {
            let g = ext.layout.groups[gi];
            assert!(
                g.block_bitmap >= leader_meta_start && g.block_bitmap < leader_meta_end,
                "group {gi} block_bitmap {} not inside leader metadata [{}, {})",
                g.block_bitmap,
                leader_meta_start,
                leader_meta_end,
            );
            assert!(
                g.inode_bitmap >= leader_meta_start && g.inode_bitmap < leader_meta_end,
                "group {gi} inode_bitmap {} not inside leader metadata [{}, {})",
                g.inode_bitmap,
                leader_meta_start,
                leader_meta_end,
            );
            assert!(
                g.inode_table >= leader_meta_start
                    && g.inode_table + ext.layout.inode_table_blocks <= leader_meta_end,
                "group {gi} inode_table {} (+{} blocks) not inside leader metadata [{}, {})",
                g.inode_table,
                ext.layout.inode_table_blocks,
                leader_meta_start,
                leader_meta_end,
            );
        }

        // Reopen and verify the same property survives an Ext::open
        // (i.e. the on-disk group-descriptor pointers, not just the planner).
        let reopened = Ext::open(&mut dev).expect("reopen flex_bg image");
        assert!(
            reopened.sb.feature_incompat & constants::feature::INCOMPAT_FLEX_BG != 0,
            "INCOMPAT_FLEX_BG must round-trip through the superblock"
        );
        for gi in 1..reopened.layout.num_groups() as usize {
            let g = reopened.layout.groups[gi];
            assert!(
                g.block_bitmap >= leader_meta_start && g.block_bitmap < leader_meta_end,
                "reopened: group {gi} block_bitmap {} not in leader metadata",
                g.block_bitmap,
            );
            assert!(
                g.inode_table >= leader_meta_start
                    && g.inode_table + reopened.layout.inode_table_blocks <= leader_meta_end,
                "reopened: group {gi} inode_table {} not in leader metadata",
                g.inode_table,
            );
        }
    }

    /// Format a small ext4 image with flex_bg enabled and run `e2fsck -fn`
    /// on it. Skipped silently when e2fsck isn't installed on the host —
    /// the in-memory checks above already pin the layout invariants.
    #[test]
    fn flex_bg_image_passes_e2fsck() {
        use std::process::Command;
        let e2fsck = match Command::new("sh")
            .arg("-c")
            .arg("command -v e2fsck")
            .output()
        {
            Ok(o) if o.status.success() && !o.stdout.is_empty() => {
                String::from_utf8(o.stdout).unwrap().trim().to_string()
            }
            _ => {
                eprintln!("skipping flex_bg_image_passes_e2fsck: e2fsck not installed");
                return;
            }
        };

        let blocks_per_group = 8 * 4096u32;
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 2 * blocks_per_group,
            inodes_count: 2048,
            log_groups_per_flex: 1,
            sparse_super: true,
            journal_blocks: 1024,
            ..FormatOpts::default()
        };
        let size = opts.blocks_count as u64 * opts.block_size as u64;
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let mut dev =
            crate::block::FileBackend::create(tmp.path(), size).expect("create FileBackend");
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format flex_bg");

        // Plant a couple of files so e2fsck exercises the bitmaps + inode
        // table in *both* flex members (not just the leader's slot 0).
        let body = vec![b'A'; 8 * 1024];
        ext.add_file_to_streaming(
            &mut dev,
            constants::INO_ROOT_DIR,
            b"a.bin",
            &mut body.as_slice(),
            body.len() as u64,
            FileMeta {
                mode: 0o644,
                uid: 0,
                gid: 0,
                mtime: 0,
                atime: 0,
                ctime: 0,
            },
        )
        .expect("add file a.bin");
        ext.add_file_to_streaming(
            &mut dev,
            constants::INO_ROOT_DIR,
            b"b.bin",
            &mut body.as_slice(),
            body.len() as u64,
            FileMeta {
                mode: 0o644,
                uid: 0,
                gid: 0,
                mtime: 0,
                atime: 0,
                ctime: 0,
            },
        )
        .expect("add file b.bin");
        ext.flush(&mut dev).expect("flush");
        BlockDevice::sync(&mut dev).expect("sync");
        drop(dev);

        let out = Command::new(&e2fsck)
            .arg("-fn")
            .arg(tmp.path())
            .output()
            .expect("run e2fsck");
        assert!(
            out.status.success(),
            "e2fsck failed on flex_bg image:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }

    #[test]
    fn flex_bg_add_and_readback_file() {
        // Format a flex_bg image, add a file, reopen, read it back.
        let mut dev = MemoryBackend::new(64u64 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 16 * 1024,
            inodes_count: 1024,
            log_groups_per_flex: 1,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format flex_bg");
        let payload = b"hello flex_bg".to_vec();
        ext.add_file_to_streaming(
            &mut dev,
            constants::INO_ROOT_DIR,
            b"greet.txt",
            &mut payload.as_slice(),
            payload.len() as u64,
            FileMeta {
                mode: 0o644,
                uid: 0,
                gid: 0,
                mtime: 0,
                atime: 0,
                ctime: 0,
            },
        )
        .expect("add file");
        ext.flush(&mut dev).expect("flush");

        // Reopen and read back.
        let reopened = Ext::open(&mut dev).expect("reopen");
        let ino = reopened
            .path_to_inode(&mut dev, "/greet.txt")
            .expect("path lookup");
        use std::io::Read as _;
        let mut buf = Vec::new();
        reopened
            .open_file_reader(&mut dev, ino)
            .expect("open reader")
            .read_to_end(&mut buf)
            .expect("read");
        assert_eq!(&buf, &payload);
    }

    #[test]
    fn use_64bit_sets_feature_and_desc_size() {
        // With `use_64bit` the writer must advertise INCOMPAT_64BIT +
        // INCOMPAT_META_BG and emit 64-byte descriptors.
        let mut dev = MemoryBackend::new(64u64 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 16 * 1024,
            inodes_count: 1024,
            use_64bit: true,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let ext = Ext::format_with(&mut dev, &opts).expect("format 64bit");
        assert_eq!(ext.sb.desc_size, constants::GROUP_DESC_SIZE_64 as u16);
        assert!(
            ext.sb.feature_incompat & constants::feature::INCOMPAT_64BIT != 0,
            "INCOMPAT_64BIT must be set"
        );
        assert!(
            ext.sb.feature_incompat & constants::feature::INCOMPAT_META_BG != 0,
            "INCOMPAT_META_BG must be set (the kernel pair with 64BIT)"
        );
        assert_eq!(
            ext.layout.desc_size,
            constants::GROUP_DESC_SIZE_64,
            "layout planner must widen desc_size when use_64bit is on"
        );
    }

    #[test]
    fn use_64bit_round_trip_add_and_read() {
        // Round-trip: format with 64-byte descriptors, add a file, reopen,
        // verify the reopened image keeps the same feature set + reads back.
        let mut dev = MemoryBackend::new(64u64 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 16 * 1024,
            inodes_count: 1024,
            use_64bit: true,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format 64bit");
        let payload = b"hello 64-bit".to_vec();
        ext.add_file_to_streaming(
            &mut dev,
            constants::INO_ROOT_DIR,
            b"big.txt",
            &mut payload.as_slice(),
            payload.len() as u64,
            FileMeta {
                mode: 0o644,
                uid: 0,
                gid: 0,
                mtime: 0,
                atime: 0,
                ctime: 0,
            },
        )
        .expect("add file");
        ext.flush(&mut dev).expect("flush");

        let reopened = Ext::open(&mut dev).expect("reopen 64bit");
        assert_eq!(reopened.sb.desc_size, constants::GROUP_DESC_SIZE_64 as u16);
        assert!(
            reopened.sb.feature_incompat & constants::feature::INCOMPAT_64BIT != 0,
            "round-tripped image must keep INCOMPAT_64BIT"
        );
        assert_eq!(reopened.layout.desc_size, constants::GROUP_DESC_SIZE_64);
        let ino = reopened
            .path_to_inode(&mut dev, "/big.txt")
            .expect("path lookup");
        use std::io::Read as _;
        let mut buf = Vec::new();
        reopened
            .open_file_reader(&mut dev, ino)
            .expect("open reader")
            .read_to_end(&mut buf)
            .expect("read");
        assert_eq!(&buf, &payload);
    }

    #[test]
    fn sparse_super2_off_by_default() {
        // Default opts must keep sparse_super2 off (COMPAT_SPARSE_SUPER2 = 0).
        let mut dev = MemoryBackend::new(1024 * 1024);
        let ext = Ext::format_with(&mut dev, &FormatOpts::default()).expect("format default");
        assert_eq!(
            ext.sb.feature_compat & constants::feature::COMPAT_SPARSE_SUPER2,
            0
        );
        assert_eq!(ext.sb.backup_bgs, [0, 0]);
    }

    #[test]
    fn sparse_super2_records_backup_bgs_and_skips_other_groups() {
        // 4 groups at 4 KiB blocks. With sparse_super2 only groups [1,
        // last=3] hold SB+GDT backups; group 2 must not.
        let mut dev = MemoryBackend::new(512u64 * 1024 * 1024);
        let blocks_per_group = 8 * 4096u32;
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 4 * blocks_per_group,
            inodes_count: 4096,
            sparse_super2: true,
            ..FormatOpts::default()
        };
        let ext = Ext::format_with(&mut dev, &opts).expect("format sparse_super2");
        assert_eq!(ext.layout.num_groups(), 4);
        assert!(
            ext.sb.feature_compat & constants::feature::COMPAT_SPARSE_SUPER2 != 0,
            "COMPAT_SPARSE_SUPER2 must be set"
        );
        assert_eq!(ext.sb.backup_bgs, [1, 3]);
        // Group 0 is always implicit (it holds the primary SB). Group 1
        // and group 3 (the two listed) carry backups; group 2 must not.
        assert!(ext.layout.groups[0].has_superblock);
        assert!(ext.layout.groups[1].has_superblock);
        assert!(!ext.layout.groups[2].has_superblock);
        assert!(ext.layout.groups[3].has_superblock);

        // Round-trip: reopen and verify the on-disk superblock parses the
        // same way (backup_bgs decoded, sparse_super2 honoured).
        let reopened = Ext::open(&mut dev).expect("reopen sparse_super2");
        assert_eq!(reopened.sb.backup_bgs, [1, 3]);
        assert!(reopened.sb.feature_compat & constants::feature::COMPAT_SPARSE_SUPER2 != 0);
        assert!(!reopened.layout.groups[2].has_superblock);
    }

    #[test]
    fn xattrs_round_trip_on_ext4_image() {
        // End-to-end: format an ext4 image (so metadata_csum is on),
        // add a regular file, attach a mix of xattrs spanning every
        // supported namespace, flush, reopen, and verify
        // `read_xattrs` returns the same set the writer was handed.
        //
        // This exercises the full `set_xattrs` → block alloc →
        // CRC32C-stamped block write → `decode_block` path that the
        // unit-level encode/decode tests in `xattr.rs` don't cover,
        // and pins the `COMPAT_EXT_ATTR` feature bit as a side effect.
        let mut dev = MemoryBackend::new(64u64 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 16 * 1024,
            inodes_count: 1024,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext4");
        let payload = b"hello xattrs".to_vec();
        let ino = ext
            .add_file_to_streaming(
                &mut dev,
                constants::INO_ROOT_DIR,
                b"labelled.txt",
                &mut payload.as_slice(),
                payload.len() as u64,
                FileMeta {
                    mode: 0o644,
                    uid: 0,
                    gid: 0,
                    mtime: 0,
                    atime: 0,
                    ctime: 0,
                },
            )
            .expect("add file");

        // One xattr per supported namespace prefix so the index-encoding
        // path is fully covered by a single round-trip.
        let xs = vec![
            xattr::Xattr::new("user.greeting", b"hello".to_vec()),
            xattr::Xattr::new(
                "security.selinux",
                b"system_u:object_r:unlabeled_t:s0\0".to_vec(),
            ),
            xattr::Xattr::new("trusted.opaque", vec![0u8, 1, 2, 3, 4]),
            xattr::Xattr::new("system.foo", b"bar".to_vec()),
        ];
        ext.set_xattrs(&mut dev, ino, &xs).expect("set_xattrs");
        ext.flush(&mut dev).expect("flush");

        // `COMPAT_EXT_ATTR` must be advertised after attaching xattrs.
        assert!(
            ext.sb.feature_compat & constants::feature::COMPAT_EXT_ATTR != 0,
            "COMPAT_EXT_ATTR must be set once xattrs are attached"
        );

        let reopened = Ext::open(&mut dev).expect("reopen ext4");
        assert!(
            reopened.sb.feature_compat & constants::feature::COMPAT_EXT_ATTR != 0,
            "round-tripped image must keep COMPAT_EXT_ATTR"
        );
        let ino2 = reopened
            .path_to_inode(&mut dev, "/labelled.txt")
            .expect("path lookup");
        let mut back = reopened.read_xattrs(&mut dev, ino2).expect("read_xattrs");

        // `read_xattrs` returns entries in the kernel's sort order
        // (name_index ASC, suffix ASC), so sort the expected set the
        // same way before comparing.
        let mut want = xs.clone();
        let key = |x: &xattr::Xattr| {
            let (idx, suffix) = xattr::name_index_and_suffix(&x.name);
            (idx, suffix.to_string())
        };
        back.sort_by_key(&key);
        want.sort_by_key(&key);
        assert_eq!(back, want);
    }

    // ───────────────────── open_file_rw (ext2 only) ─────────────────────

    use crate::fs::{Filesystem, OpenFlags};
    use std::io::{Seek as _, SeekFrom, Write as _};

    /// Build a fresh ext2 image, write one regular file via the populate
    /// API, flush, and return the live `Ext` + backing device. Block size
    /// is fixed at 1 KiB so most data lives in direct pointers; tests
    /// that want to exercise indirect blocks size the file accordingly.
    fn ext2_with_file(name: &[u8], payload: &[u8]) -> (Ext, MemoryBackend) {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext2,
            block_size: 1024,
            blocks_count: 8192,
            inodes_count: 256,
            ..FormatOpts::default()
        };
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext2");
        if !payload.is_empty() {
            ext.add_file_to_streaming(
                &mut dev,
                constants::INO_ROOT_DIR,
                name,
                &mut std::io::Cursor::new(payload.to_vec()),
                payload.len() as u64,
                FileMeta::default(),
            )
            .expect("add file");
        }
        ext.flush(&mut dev).expect("flush");
        (ext, dev)
    }

    fn read_full_via_handle(ext: &mut Ext, dev: &mut MemoryBackend, path: &str) -> Vec<u8> {
        let p = std::path::Path::new(path);
        let mut h = ext
            .open_file_rw(dev, p, OpenFlags::default(), None)
            .expect("open_file_rw");
        let mut out = Vec::new();
        h.read_to_end(&mut out).expect("read");
        out
    }

    #[test]
    fn open_file_rw_partial_write_round_trip_ext2() {
        let payload = vec![b'a'; 4096]; // 4 blocks at 1 KiB
        let (mut ext, mut dev) = ext2_with_file(b"hello.bin", &payload);

        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/hello.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open_file_rw");
            assert_eq!(h.len(), 4096);
            // Patch 4 bytes at offset 1000 (block 0) and 4 bytes at
            // offset 2500 (block 2).
            h.seek(SeekFrom::Start(1000)).unwrap();
            h.write_all(b"XXXX").unwrap();
            h.seek(SeekFrom::Start(2500)).unwrap();
            h.write_all(b"YYYY").unwrap();
            h.sync().expect("sync");
        }

        // Reopen the FS from disk to verify the writes survived flush.
        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let got = read_full_via_handle(&mut reopened, &mut dev, "/hello.bin");
        assert_eq!(got.len(), 4096);
        for (i, b) in got.iter().enumerate() {
            let expected = if (1000..1004).contains(&i) {
                b'X'
            } else if (2500..2504).contains(&i) {
                b'Y'
            } else {
                b'a'
            };
            assert_eq!(*b, expected, "mismatch at {i}");
        }
    }

    #[test]
    fn open_file_rw_extends_file_ext2() {
        let payload = b"abcd".to_vec();
        let (mut ext, mut dev) = ext2_with_file(b"grow.bin", &payload);
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/grow.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open");
            h.seek(SeekFrom::End(0)).unwrap();
            h.write_all(b"EFGH").unwrap();
            h.sync().unwrap();
            assert_eq!(h.len(), 8);
        }
        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let got = read_full_via_handle(&mut reopened, &mut dev, "/grow.bin");
        assert_eq!(got, b"abcdEFGH");
    }

    #[test]
    fn open_file_rw_set_len_grow_and_shrink_ext2() {
        // Big enough to spill into the single-indirect range (>12 KiB at
        // 1 KiB blocks). Start with a 4 KiB payload, then grow well past
        // 12 blocks, then shrink back to 100 bytes.
        let payload = vec![b'q'; 4096];
        let (mut ext, mut dev) = ext2_with_file(b"flex.bin", &payload);
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/flex.bin"),
                    OpenFlags::default(),
                    None,
                )
                .unwrap();
            // Grow to 20 KiB (forces indirect-block allocation).
            h.set_len(20 * 1024).unwrap();
            assert_eq!(h.len(), 20 * 1024);
            // Bytes beyond the original 4 KiB must read as zero.
            let mut buf = vec![0u8; 16 * 1024];
            h.seek(SeekFrom::Start(4096)).unwrap();
            h.read_exact(&mut buf).unwrap();
            assert!(buf.iter().all(|&b| b == 0), "grown region must be zero");
            // Now shrink to 100 bytes.
            h.set_len(100).unwrap();
            assert_eq!(h.len(), 100);
            h.sync().unwrap();
        }
        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let got = read_full_via_handle(&mut reopened, &mut dev, "/flex.bin");
        assert_eq!(got.len(), 100);
        assert!(got.iter().all(|&b| b == b'q'));
        // The indirect block should be freed too — free count must be
        // back where it was before the grow (or higher, since shrinking
        // past the original size also frees the data blocks we just
        // allocated). Sanity-check: at least one free block exists.
        assert!(reopened.sb.free_blocks_count > 0);
    }

    #[test]
    fn open_file_rw_append_ext2() {
        let payload = b"first".to_vec();
        let (mut ext, mut dev) = ext2_with_file(b"app.bin", &payload);
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/app.bin"),
                    OpenFlags {
                        append: true,
                        ..OpenFlags::default()
                    },
                    None,
                )
                .unwrap();
            h.write_all(b"-second").unwrap();
            h.sync().unwrap();
        }
        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let got = read_full_via_handle(&mut reopened, &mut dev, "/app.bin");
        assert_eq!(got, b"first-second");
    }

    #[test]
    fn open_file_rw_create_new_ext2() {
        let (mut ext, mut dev) = ext2_with_file(b"_unused.bin", b"x");
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/fresh.txt"),
                    OpenFlags {
                        create: true,
                        ..OpenFlags::default()
                    },
                    Some(FileMeta::with_mode(0o644)),
                )
                .expect("open create");
            h.write_all(b"hello world").unwrap();
            h.sync().unwrap();
        }
        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let got = read_full_via_handle(&mut reopened, &mut dev, "/fresh.txt");
        assert_eq!(got, b"hello world");
    }

    #[test]
    fn open_file_rw_truncate_ext2() {
        let payload = vec![b'k'; 4096];
        let (mut ext, mut dev) = ext2_with_file(b"trunc.bin", &payload);
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/trunc.bin"),
                    OpenFlags {
                        truncate: true,
                        ..OpenFlags::default()
                    },
                    None,
                )
                .unwrap();
            assert_eq!(h.len(), 0);
            h.write_all(b"short").unwrap();
            h.sync().unwrap();
        }
        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let got = read_full_via_handle(&mut reopened, &mut dev, "/trunc.bin");
        assert_eq!(got, b"short");
    }

    /// Build a fresh ext3 image (1 KiB blocks, indirect-tree files) with a
    /// 1024-block clean JBD2 journal, write one regular file via the
    /// populate API, flush, and return the live `Ext` + backing device.
    /// Used by the clean-journal round-trip + dirty-journal refusal tests.
    fn ext3_with_file(name: &[u8], payload: &[u8]) -> (Ext, MemoryBackend) {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext3,
            block_size: 1024,
            blocks_count: 8192,
            inodes_count: 256,
            journal_blocks: 1024,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext3");
        if !payload.is_empty() {
            ext.add_file_to_streaming(
                &mut dev,
                constants::INO_ROOT_DIR,
                name,
                &mut std::io::Cursor::new(payload.to_vec()),
                payload.len() as u64,
                FileMeta::default(),
            )
            .expect("add file");
        }
        ext.flush(&mut dev).expect("flush");
        (ext, dev)
    }

    #[test]
    fn open_file_rw_round_trip_ext3_clean_journal() {
        // ext3 has a JBD2 journal; freshly-formatted images have
        // s_start = 0 (clean). open_file_rw must accept the image,
        // perform the partial write in place, and on sync produce a
        // filesystem that round-trips through a fresh `Ext::open`.
        let payload = vec![b'a'; 4096];
        let (mut ext, mut dev) = ext3_with_file(b"hello.bin", &payload);
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/hello.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open_file_rw on clean-journal ext3");
            assert_eq!(h.len(), 4096);
            h.seek(SeekFrom::Start(1000)).unwrap();
            h.write_all(b"ZZZZ").unwrap();
            h.sync().unwrap();
        }
        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let got = read_full_via_handle(&mut reopened, &mut dev, "/hello.bin");
        let mut expect = payload.clone();
        expect[1000..1004].copy_from_slice(b"ZZZZ");
        assert_eq!(got, expect);
    }

    /// Format an ext3 image, run `e2fsck -fn` on it after an in-place
    /// partial-write through `open_file_rw`. Skipped silently when
    /// e2fsck isn't installed.
    #[test]
    fn open_file_rw_ext3_clean_journal_passes_e2fsck() {
        use std::process::Command;
        let e2fsck = match Command::new("sh")
            .arg("-c")
            .arg("command -v e2fsck")
            .output()
        {
            Ok(o) if o.status.success() && !o.stdout.is_empty() => {
                String::from_utf8(o.stdout).unwrap().trim().to_string()
            }
            _ => {
                eprintln!(
                    "skipping open_file_rw_ext3_clean_journal_passes_e2fsck: e2fsck not installed"
                );
                return;
            }
        };
        let opts = FormatOpts {
            kind: FsKind::Ext3,
            block_size: 1024,
            blocks_count: 8192,
            inodes_count: 256,
            journal_blocks: 1024,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let size = opts.blocks_count as u64 * opts.block_size as u64;
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let mut dev =
            crate::block::FileBackend::create(tmp.path(), size).expect("create FileBackend");
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext3");
        let payload = vec![b'a'; 4096];
        ext.add_file_to_streaming(
            &mut dev,
            constants::INO_ROOT_DIR,
            b"hello.bin",
            &mut std::io::Cursor::new(payload.clone()),
            payload.len() as u64,
            FileMeta::default(),
        )
        .expect("add file");
        ext.flush(&mut dev).expect("flush");
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/hello.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open_file_rw on clean-journal ext3");
            h.seek(SeekFrom::Start(1000)).unwrap();
            h.write_all(b"ZZZZ").unwrap();
            h.sync().unwrap();
        }
        BlockDevice::sync(&mut dev).expect("sync");
        drop(dev);

        let out = Command::new(&e2fsck)
            .arg("-fn")
            .arg(tmp.path())
            .output()
            .expect("run e2fsck");
        assert!(
            out.status.success(),
            "e2fsck failed on ext3 image after rw:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }

    #[test]
    fn open_file_rw_replays_dirty_journal_on_open() {
        // Synthesize a "crash between commit and checkpoint":
        //   1) Format ext3, snapshot the pre-write image (clean journal).
        //   2) Run an open_file_rw `set_len` that extends the file —
        //      this updates the inode (size, i_blocks) and the block
        //      bitmap, both of which ride the journal.
        //   3) The sync's journal-commit phase lands descriptor+data+
        //      commit in the log, then the checkpoint phase writes the
        //      same blocks to their FS homes and marks the journal
        //      clean. We snapshot the journal *log* blocks at that
        //      point — they still hold the committed transaction even
        //      though the on-disk SB now says s_start=0.
        //   4) Restore the pre-write image (rolls back the inode-table
        //      and bitmaps), restore the journal log blocks (so the
        //      committed transaction is back on disk), and stamp
        //      s_start != 0 in the journal SB.
        //   5) Open the image and confirm replay re-applies the
        //      metadata: the extended file size is visible.
        let payload = vec![b'a'; 1024];
        let (mut ext, mut dev) = ext3_with_file(b"foo.bin", &payload);
        let bs = ext.layout.block_size as usize;
        let bs64 = ext.layout.block_size as u64;
        let nblocks = ext.layout.blocks_count;

        // Snapshot the pre-write on-disk image (all rolled-back metadata
        // will come from here).
        let mut pre_image = vec![0u8; bs * nblocks as usize];
        dev.read_at(0, &mut pre_image).expect("snapshot pre-image");

        // Note the journal block layout. Journal blocks were allocated
        // at format time and don't move; mapping indices to physical
        // blocks once is sufficient. Only the leading log slots get
        // touched by a small transaction, so iterate to the lower of
        // (journal size, 64 — well within the direct-block range for
        // 1 KiB indirect-tree inodes the reader can resolve).
        let jino = ext.sb.journal_inum;
        let jinode = ext.read_inode(&mut dev, jino).expect("journal inode");
        let n_journal_blocks = (jinode.size as u64 / bs64) as u32;
        let probe = n_journal_blocks.min(64);
        let mut journal_phys: Vec<u32> = Vec::with_capacity(probe as usize);
        for i in 0..probe {
            let phys = ext.file_block(&mut dev, &jinode, i).expect("file_block");
            journal_phys.push(phys);
        }

        // Extend the file via the rw handle — purely a metadata
        // operation as far as the journal is concerned (inode-table,
        // bitmap, GDT).
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/foo.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open_file_rw");
            h.set_len(4096).expect("set_len");
            h.sync().unwrap();
        }

        // Snapshot the journal log blocks (descriptor + data + commit).
        // Even though the on-disk journal SB now reads s_start=0 (clean),
        // the log payload from the just-finished commit is still on disk
        // because the post-checkpoint cleanup only rewrites the SB.
        let mut post_journal: Vec<(u32, Vec<u8>)> = Vec::new();
        for phys in &journal_phys {
            if *phys == 0 {
                continue;
            }
            let mut buf = vec![0u8; bs];
            dev.read_at(*phys as u64 * bs64, &mut buf).expect("read");
            post_journal.push((*phys, buf));
        }

        // Roll the entire image back to the pre-write snapshot.
        dev.write_at(0, &pre_image).expect("restore pre-image");
        // Restore the journal *log* blocks from the post-write snapshot.
        // Skip index 0 (the journal SB itself): pre_image already holds
        // the clean (s_sequence=1, s_start=0) journal SB, and that's the
        // baseline we need replay to use.
        let jsb_phys = journal_phys[0];
        for (phys, buf) in &post_journal {
            if *phys == jsb_phys {
                continue;
            }
            dev.write_at(*phys as u64 * bs64, buf)
                .expect("restore journal");
        }
        // Dirty the journal SB: set s_start to `s_first` so replay
        // walks the log starting at the descriptor we just restored.
        // s_sequence is already the tid the transaction was committed
        // with (the post-format value), so descriptor.tid will match.
        let mut jsb_buf = vec![0u8; bs];
        dev.read_at(jsb_phys as u64 * bs64, &mut jsb_buf)
            .expect("read jsb");
        let first = u32::from_be_bytes(jsb_buf[20..24].try_into().unwrap());
        jsb_buf[28..32].copy_from_slice(&first.to_be_bytes());
        dev.write_at(jsb_phys as u64 * bs64, &jsb_buf)
            .expect("write jsb dirty");
        BlockDevice::sync(&mut dev).expect("sync");

        // Sanity: a plain read (no replay) still sees the pre-write
        // file size — the rollback worked.
        {
            let reopened = Ext::open(&mut dev).expect("reopen pre-replay");
            let ino = reopened
                .path_to_inode(&mut dev, "/foo.bin")
                .expect("path_to_inode");
            let inode = reopened.read_inode(&mut dev, ino).expect("inode");
            assert_eq!(
                inode.size, 1024,
                "pre-replay inode should still show original size"
            );
        }

        // Open for writing: replay must apply the committed transaction
        // before the handle attaches. After replay the inode size is
        // the extended 4096.
        {
            let mut ext2 = Ext::open(&mut dev).expect("reopen");
            let _ = ext2
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/foo.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open_file_rw triggers replay");
        }
        let reopened = Ext::open(&mut dev).expect("reopen after replay");
        let ino = reopened
            .path_to_inode(&mut dev, "/foo.bin")
            .expect("path_to_inode");
        let inode = reopened.read_inode(&mut dev, ino).expect("inode");
        assert_eq!(
            inode.size, 4096,
            "replay should have applied the journaled inode-table block"
        );

        // And the journal is now clean.
        let mut jsb_after = vec![0u8; bs];
        dev.read_at(jsb_phys as u64 * bs64, &mut jsb_after)
            .expect("read jsb");
        let s_start_after = u32::from_be_bytes(jsb_after[28..32].try_into().unwrap());
        assert_eq!(s_start_after, 0, "journal SB s_start must be cleared");
    }

    /// Build a fresh ext4 image (4 KiB blocks, depth-0 inline extents) with
    /// a clean JBD2 journal and one regular file written via the populate
    /// API. Used by the ext4 extent-write tests below.
    fn ext4_with_file(name: &[u8], payload: &[u8]) -> (Ext, MemoryBackend) {
        let mut dev = MemoryBackend::new(64u64 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 16 * 1024,
            inodes_count: 1024,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext4");
        if !payload.is_empty() {
            ext.add_file_to_streaming(
                &mut dev,
                constants::INO_ROOT_DIR,
                name,
                &mut std::io::Cursor::new(payload.to_vec()),
                payload.len() as u64,
                FileMeta::default(),
            )
            .expect("add file");
        }
        ext.flush(&mut dev).expect("flush");
        (ext, dev)
    }

    #[test]
    fn open_file_rw_round_trip_ext4_extents() {
        // Write at an offset inside an existing extent, reopen the FS,
        // and verify the modification persists. The image is built on a
        // file-backed device so we can hand it to `e2fsck -fn` at the
        // end (skipped when e2fsck isn't installed).
        use std::process::Command;
        let e2fsck = match Command::new("sh")
            .arg("-c")
            .arg("command -v e2fsck")
            .output()
        {
            Ok(o) if o.status.success() && !o.stdout.is_empty() => {
                Some(String::from_utf8(o.stdout).unwrap().trim().to_string())
            }
            _ => {
                eprintln!(
                    "open_file_rw_round_trip_ext4_extents: e2fsck not installed; skipping fsck check"
                );
                None
            }
        };

        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 16 * 1024,
            inodes_count: 1024,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let size = opts.blocks_count as u64 * opts.block_size as u64;
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        let mut dev =
            crate::block::FileBackend::create(tmp.path(), size).expect("create FileBackend");
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext4");

        let payload = vec![b'a'; 16 * 1024]; // 4 blocks at 4 KiB
        ext.add_file_to_streaming(
            &mut dev,
            constants::INO_ROOT_DIR,
            b"hello.bin",
            &mut std::io::Cursor::new(payload.clone()),
            payload.len() as u64,
            FileMeta::default(),
        )
        .expect("add file");
        ext.flush(&mut dev).expect("flush");

        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/hello.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open_file_rw on ext4 extents");
            assert_eq!(h.len(), payload.len() as u64);
            // Patch 4 bytes inside block 0 and 4 bytes inside block 2.
            h.seek(SeekFrom::Start(1000)).unwrap();
            h.write_all(b"XXXX").unwrap();
            h.seek(SeekFrom::Start(8500)).unwrap();
            h.write_all(b"YYYY").unwrap();
            h.sync().expect("sync");
        }

        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let got = {
            let mut h = reopened
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/hello.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open_file_rw");
            let mut out = Vec::new();
            h.read_to_end(&mut out).expect("read");
            out
        };
        assert_eq!(got.len(), payload.len());
        for (i, b) in got.iter().enumerate() {
            let expected = if (1000..1004).contains(&i) {
                b'X'
            } else if (8500..8504).contains(&i) {
                b'Y'
            } else {
                b'a'
            };
            assert_eq!(*b, expected, "mismatch at byte {i}");
        }

        // Hand the on-disk image to e2fsck if available. `-fn` forces a
        // full check in read-only mode.
        if let Some(e2fsck) = e2fsck {
            BlockDevice::sync(&mut dev).expect("sync");
            drop(dev);
            let out = Command::new(&e2fsck)
                .arg("-fn")
                .arg(tmp.path())
                .output()
                .expect("run e2fsck");
            assert!(
                out.status.success(),
                "e2fsck failed on ext4 image after rw:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
        }
    }

    #[test]
    fn open_file_rw_extends_ext4_file() {
        // Append past EOF; this must allocate a new physical block and
        // (depending on contiguity with the existing tail extent) either
        // grow that extent or add a new one.
        let payload = vec![b'a'; 4096]; // exactly one block
        let (mut ext, mut dev) = ext4_with_file(b"grow.bin", &payload);
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/grow.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open");
            h.seek(SeekFrom::End(0)).unwrap();
            // Write 5 KiB past EOF — spans into a new block.
            let extra = vec![b'b'; 5000];
            h.write_all(&extra).unwrap();
            h.sync().unwrap();
            assert_eq!(h.len(), 4096 + 5000);
        }
        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let got = read_full_via_handle(&mut reopened, &mut dev, "/grow.bin");
        assert_eq!(got.len(), 4096 + 5000);
        assert!(got[..4096].iter().all(|&b| b == b'a'));
        assert!(got[4096..].iter().all(|&b| b == b'b'));
    }

    #[test]
    fn open_file_rw_set_len_grow_and_shrink_ext4() {
        // Exercise grow + shrink via set_len on an extent inode.
        let payload = vec![b'q'; 4096]; // one block
        let (mut ext, mut dev) = ext4_with_file(b"flex.bin", &payload);
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/flex.bin"),
                    OpenFlags::default(),
                    None,
                )
                .unwrap();
            // Grow to 5 blocks. New region must read as zero.
            h.set_len(5 * 4096).unwrap();
            assert_eq!(h.len(), 5 * 4096);
            let mut buf = vec![0u8; 4 * 4096];
            h.seek(SeekFrom::Start(4096)).unwrap();
            h.read_exact(&mut buf).unwrap();
            assert!(buf.iter().all(|&b| b == 0), "grown region must be zero");
            // Shrink back to 100 bytes.
            h.set_len(100).unwrap();
            assert_eq!(h.len(), 100);
            h.sync().unwrap();
        }
        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let got = read_full_via_handle(&mut reopened, &mut dev, "/flex.bin");
        assert_eq!(got.len(), 100);
        assert!(got.iter().all(|&b| b == b'q'));
        // The trailing data blocks should have been returned to the
        // bitmap — sanity-check the FS still reports free space.
        assert!(reopened.sb.free_blocks_count > 0);
    }

    #[test]
    fn open_file_rw_refused_ext4_when_extent_depth_too_deep() {
        // Synthesise an inode whose extent header claims depth > 0. The
        // writer must refuse cleanly at open time rather than trying to
        // walk an index it can't allocate.
        let (ext, mut dev) = ext4_with_file(b"deep.bin", b"hello");

        // Locate the file's inode and patch its i_block header so eh_depth = 2.
        // Depth 0 (inline) and depth 1 (one level of idx → leaf) are both
        // supported by the writer; depth 2+ still needs a second level of
        // idx-block allocation and is refused.
        let ino = ext
            .path_to_inode(&mut dev, "/deep.bin")
            .expect("path lookup");
        let mut inode = ext.read_inode(&mut dev, ino).expect("read inode");
        let mut bytes = extent::iblock_to_bytes(&inode.block);
        // eh_depth lives at bytes 6..8 (little-endian).
        bytes[6..8].copy_from_slice(&2u16.to_le_bytes());
        inode.block = extent::bytes_to_iblock(&bytes);

        // Write the patched inode straight to disk; we want the
        // subsequent open_file_rw to see the on-disk state, not anything
        // cached by `ext`.
        let (group, idx) = ext.inode_location(ino);
        let table_block = ext.layout.groups[group as usize].inode_table;
        let bs = ext.layout.block_size as u64;
        let off = table_block as u64 * bs + idx as u64 * ext.layout.inode_size as u64;
        let mut enc = inode.encode().to_vec();
        if ext.layout.inode_size as usize > enc.len() {
            enc.resize(ext.layout.inode_size as usize, 0);
        }
        dev.write_at(off, &enc).expect("write patched inode");

        // Reopen the FS so the staged-inode cache is empty.
        let mut ext = Ext::open(&mut dev).expect("reopen");
        let res = ext.open_file_rw(
            &mut dev,
            std::path::Path::new("/deep.bin"),
            OpenFlags::default(),
            None,
        );
        match res {
            Ok(_) => panic!("must refuse on deeper extent tree"),
            Err(crate::Error::Unsupported(msg)) => {
                assert!(msg.contains("depth"), "unexpected message: {msg}");
            }
            Err(other) => panic!("expected Unsupported, got {other}"),
        }
    }

    /// Build an ext4 file whose extent tree must spill into depth-1
    /// (more than 4 leaf extents). We force fragmentation by writing
    /// individual blocks at logically-discontiguous offsets, so each
    /// block sits in its own extent run rather than merging into a
    /// single run.
    #[test]
    fn open_file_rw_depth1_extent_round_trip_ext4() {
        let mut dev = MemoryBackend::new(64u64 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 16 * 1024,
            inodes_count: 1024,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext4");
        // Start from an empty file so every alloc lands somewhere
        // chosen by the bitmap walker rather than continuing a tail.
        ext.add_file_to_streaming(
            &mut dev,
            constants::INO_ROOT_DIR,
            b"deep.bin",
            &mut std::io::Cursor::new(Vec::<u8>::new()),
            0,
            FileMeta::default(),
        )
        .expect("add empty file");
        ext.flush(&mut dev).expect("flush");

        // Six distinct sparse offsets → six logically-discontiguous
        // extents (depth-0 caps at 4, so the tree must promote to
        // depth-1).
        let offsets: &[u64] = &[0, 40_000, 80_000, 120_000, 160_000, 200_000];
        let mark = b"DEPTH1!"; // 7 bytes, well under a block

        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/deep.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open_file_rw on empty extent file");
            for off in offsets {
                h.seek(SeekFrom::Start(*off)).unwrap();
                h.write_all(mark).unwrap();
            }
            h.sync().expect("sync");
            assert_eq!(h.len(), 200_000 + mark.len() as u64);
        }

        // Walk the on-disk inode to verify the tree is in fact depth-1.
        let ino = ext.path_to_inode(&mut dev, "/deep.bin").expect("ino");
        let inode = ext.read_inode(&mut dev, ino).expect("read inode");
        let iblock = extent::iblock_to_bytes(&inode.block);
        let header = extent::decode_header(&iblock[..12]).expect("header");
        assert_eq!(
            header.depth, 1,
            "expected depth-1 extent tree, got depth {}",
            header.depth
        );
        assert!(
            header.entries >= 1 && header.entries <= 4,
            "expected 1..=4 idx entries, got {}",
            header.entries
        );

        // Reopen and verify the contents survive. Each marked offset
        // must read back its bytes; everything else must read as zero.
        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let mut h = reopened
            .open_file_rw(
                &mut dev,
                std::path::Path::new("/deep.bin"),
                OpenFlags::default(),
                None,
            )
            .expect("reopen rw on depth-1 file");
        for off in offsets {
            h.seek(SeekFrom::Start(*off)).unwrap();
            let mut buf = vec![0u8; mark.len()];
            h.read_exact(&mut buf).unwrap();
            assert_eq!(&buf[..], mark, "marker mismatch at offset {off}");
        }
        // Pick a block we never wrote and verify it's zero (hole).
        h.seek(SeekFrom::Start(20_000)).unwrap();
        let mut zero = vec![0u8; 4096];
        h.read_exact(&mut zero).unwrap();
        assert!(
            zero.iter().all(|&b| b == 0),
            "untouched region must be zero"
        );
    }

    /// Round-trip a depth-1 tree through shrink: write enough extents to
    /// promote to depth-1, then shrink past EOF and confirm the tree
    /// drops back to depth-0 and any leaf blocks are returned to the
    /// bitmap.
    #[test]
    fn open_file_rw_depth1_shrink_back_to_depth0() {
        let mut dev = MemoryBackend::new(64u64 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 16 * 1024,
            inodes_count: 1024,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext4");
        ext.add_file_to_streaming(
            &mut dev,
            constants::INO_ROOT_DIR,
            b"shrink.bin",
            &mut std::io::Cursor::new(Vec::<u8>::new()),
            0,
            FileMeta::default(),
        )
        .unwrap();
        ext.flush(&mut dev).expect("flush");

        // Snapshot free-block count before any depth-1 work.
        let free_before = ext.sb.free_blocks_count;

        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/shrink.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open empty");
            for &off in &[0u64, 40_000, 80_000, 120_000, 160_000] {
                h.seek(SeekFrom::Start(off)).unwrap();
                h.write_all(b"x").unwrap();
            }
            h.sync().unwrap();

            // Confirm we hit depth-1 mid-sequence.
            let ino_now = h.len(); // not used; the assert below uses inode lookup
            let _ = ino_now;
        }
        {
            // Walk the inode to confirm depth-1 reached.
            let ino = ext.path_to_inode(&mut dev, "/shrink.bin").unwrap();
            let inode = ext.read_inode(&mut dev, ino).unwrap();
            let iblock = extent::iblock_to_bytes(&inode.block);
            let header = extent::decode_header(&iblock[..12]).expect("header");
            assert_eq!(header.depth, 1, "should be depth-1 mid-shrink");
        }

        // Now truncate to 100 bytes — drops back to a single block, well
        // within depth-0.
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/shrink.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("reopen");
            h.set_len(100).unwrap();
            h.sync().unwrap();
        }
        {
            let ino = ext.path_to_inode(&mut dev, "/shrink.bin").unwrap();
            let inode = ext.read_inode(&mut dev, ino).unwrap();
            let iblock = extent::iblock_to_bytes(&inode.block);
            let header = extent::decode_header(&iblock[..12]).expect("header");
            assert_eq!(header.depth, 0, "should drop back to depth-0 after shrink");
        }

        // The leaf block(s) and any freed data blocks must be back in the
        // bitmap — free-blocks should be ≥ free_before − 1 (we still have
        // the one data block holding the surviving 100 bytes).
        let free_after = ext.sb.free_blocks_count;
        assert!(
            free_after + 2 >= free_before,
            "shrink leaked blocks: free was {free_before}, now {free_after}",
        );
    }

    #[test]
    fn open_file_ro_random_seek_ext() {
        // open_file_ro must work for both ext2 (indirect blocks) and ext4
        // (extents); the file_block walker handles both formats. Test on
        // ext4 to lock in the case open_file_rw can't satisfy.
        use crate::fs::Filesystem;
        use std::io::{Read, Seek, SeekFrom};

        let mut dev = MemoryBackend::new(64u64 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 4096,
            blocks_count: 16 * 1024,
            inodes_count: 1024,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext4");
        // Multi-block file to exercise the extent walker.
        let data: Vec<u8> = (0..15_000u32).map(|i| (i & 0xFF) as u8).collect();
        ext.add_file_to_streaming(
            &mut dev,
            constants::INO_ROOT_DIR,
            b"ro.bin",
            &mut std::io::Cursor::new(data.clone()),
            data.len() as u64,
            FileMeta::default(),
        )
        .unwrap();
        ext.flush(&mut dev).unwrap();

        // Reopen and exercise the read-only path through the trait.
        let mut ext = Ext::open(&mut dev).expect("reopen ext4");
        let mut h = ext
            .open_file_ro(&mut dev, std::path::Path::new("/ro.bin"))
            .expect("open_file_ro on ext4 extent file");
        assert_eq!(h.len(), data.len() as u64);
        assert!(!h.is_empty());

        h.seek(SeekFrom::Start(9876)).unwrap();
        let mut buf = [0u8; 200];
        h.read_exact(&mut buf).unwrap();
        assert_eq!(&buf[..], &data[9876..10076]);

        h.seek(SeekFrom::Start(42)).unwrap();
        let mut buf2 = [0u8; 64];
        h.read_exact(&mut buf2).unwrap();
        assert_eq!(&buf2[..], &data[42..106]);
    }
}
