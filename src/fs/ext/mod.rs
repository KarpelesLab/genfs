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
pub mod htree;
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
    /// When true, advertise `INCOMPAT_INLINE_DATA` and store small
    /// regular files (≤ 60 bytes) directly in the inode's `i_block`
    /// array instead of allocating a data block. Saves one block per
    /// small file. Off by default — flipping it changes the on-disk
    /// shape of the FS so kernel/e2fsck installations older than ~3.8
    /// would refuse to mount the result.
    pub inline_data: bool,
    /// When true, the caller guarantees the device's first
    /// `blocks_count * block_size` bytes already read back as zero
    /// (e.g. a freshly-`set_len`'d sparse file, a fresh `Qcow2Backend`,
    /// or a freshly-allocated `MemoryBackend`). The formatter skips its
    /// upfront full-device `zero_range`, which on an 8 GiB raw image is
    /// 8 GiB of writes, and on a sparse qcow2 is a no-op walk over
    /// every cluster but still pure overhead.
    ///
    /// Off by default so the formatter remains correct on a device with
    /// arbitrary prior contents — leaving stale data behind in the
    /// inode-table tail of empty groups, or in the journal data area
    /// past block 0, would produce a filesystem that e2fsck and the
    /// kernel reject. Flip it on only when you know the destination is
    /// zero.
    pub prezeroed: bool,
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
            inline_data: false,
            prezeroed: false,
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
    pub fn apply_options(&mut self, map: &mut crate::format_opts::OptionMap) -> crate::Result<()> {
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
    /// First bit index *possibly* free in this group's block bitmap.
    /// Advanced by [`Ext::alloc_data_block`] so it doesn't rescan the
    /// known-allocated prefix on every call — the original linear scan
    /// from bit 0 was the dominant cost (35 % of total instructions) for
    /// bulk-insert workloads, since allocations are append-only and each
    /// call walked the full set prefix before finding the next free bit.
    next_free_block_bit: u32,
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
    /// Extent-tree leaf blocks staged in `data_blocks`, tagged with their
    /// owning inode. Used at flush time to stamp the 4-byte
    /// `ext4_extent_tail` CRC32C when `metadata_csum` is active.
    extent_leaf_blocks: Vec<(u32, u32)>,
    /// HTree dx_root blocks staged in `data_blocks`. Tuple is
    /// (block, owning_inode, dx_entry_count). The CRC at flush time
    /// covers only the in-use prefix (count_offset + count * 8 bytes)
    /// plus a 4-byte dt_reserved + a 4-byte dt_checksum placeholder —
    /// distinct from both the dir-block tail csum and the extent_tail
    /// csum.
    dx_root_blocks: Vec<(u32, u32, u16)>,
    /// HTree dx_node intermediate blocks staged in `data_blocks`,
    /// used only when a directory's index has `indirect_levels = 1`.
    /// Same csum scheme as dx_root but with the smaller 12-byte
    /// fake-dirent prefix (no `.` / `..` / dx_root_info overhead).
    dx_node_blocks: Vec<(u32, u32, u16)>,
    /// `inode number -> index in `inodes`` so per-file lookups are O(1)
    /// instead of a linear scan of every staged inode (which made a
    /// many-files build O(n²)). Maintained at every `inodes` mutation;
    /// `inodes` is never cleared at flush, so neither is this.
    inode_idx: std::collections::HashMap<u32, usize>,
    /// `block number -> index in `data_blocks``, same rationale. Cleared
    /// with `data_blocks` at each flush.
    data_block_idx: std::collections::HashMap<u32, usize>,
    /// Set of block numbers present in `dir_blocks`, for O(1) membership
    /// checks. Cleared with `dir_blocks` at each flush.
    dir_block_set: std::collections::HashSet<u32>,
    /// True until the first flush after a `format_with` lands. The
    /// initial flush is a "blast everything fresh" write that doesn't
    /// ride a journal transaction (there's nothing yet to be consistent
    /// with — the journal is fresh and clean). Subsequent flushes on an
    /// image with a journal go through the JBD2 commit/checkpoint path.
    bootstrap: bool,
}

/// Outcome of appending one extent into a subtree (see
/// [`Ext::append_into_node`]). Either the insert was absorbed locally, or
/// the subtree split and produced a new sibling node the parent must link.
enum NodeAppend {
    /// Insert absorbed; payload is the count of metadata blocks allocated.
    Done(u32),
    /// The visited node was full; `idx` points at a freshly-allocated
    /// sibling node (at the same depth) that the parent must add. `meta` is
    /// the total metadata blocks allocated handling this insert.
    NewSibling { idx: extent::ExtentIdx, meta: u32 },
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

        // Zero the FS region so anything we don't explicitly write back
        // — the tail of every inode-table block in a group with no
        // staged inodes, journal data blocks past the JBD2 superblock,
        // bitmap-tracked-but-untouched regions — reads back as zero.
        // Backends may treat this as a sparse hole. Callers that just
        // created the device (`block::create_image`, `MemoryBackend`)
        // set `prezeroed` to skip the pass: those devices already read
        // as zero, and an 8 GiB `zero_range` on a freshly-`set_len`'d
        // raw file is 8 GiB of pointless writes.
        if !opts.prezeroed {
            dev.zero_range(0, total_bytes)?;
        }

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
                next_free_block_bit: 0,
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
            extent_leaf_blocks: Vec::new(),
            dx_root_blocks: Vec::new(),
            dx_node_blocks: Vec::new(),
            inode_idx: std::collections::HashMap::new(),
            data_block_idx: std::collections::HashMap::new(),
            dir_block_set: std::collections::HashSet::new(),
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
            // Advertise DIR_INDEX (HTree) capability. Setting the bit
            // doesn't oblige every dir to be indexed — un-indexed dirs
            // are still valid; we set EXT4_INDEX_FL per-inode on the
            // ones we actually emit as HTree.
            ext.sb.feature_compat |= constants::feature::COMPAT_DIR_INDEX;
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
        if opts.inline_data {
            ext.sb.feature_incompat |= constants::feature::INCOMPAT_INLINE_DATA;
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
    pub(crate) fn has_metadata_csum(&self) -> bool {
        self.sb.feature_ro_compat & constants::feature::RO_COMPAT_METADATA_CSUM != 0
    }

    /// Whether the `inline_data` feature is active. When true, small
    /// regular files (≤ 60 bytes) get stored in `i_block` directly
    /// instead of allocating a data block; readers honour the
    /// `EXT4_INLINE_DATA_FL` flag on the inode to decode them.
    pub(crate) fn has_inline_data(&self) -> bool {
        self.sb.feature_incompat & constants::feature::INCOMPAT_INLINE_DATA != 0
    }

    /// If the filesystem carries a JBD2 journal with committed-but-not-
    /// checkpointed transactions, replay them onto the device. Mirrors
    /// what the Linux kernel does on first mount of an unclean ext{3,4}
    /// filesystem: walk the journal log starting at `s_start`, apply
    /// each transaction's data blocks to their target FS locations,
    /// then mark the journal clean (`s_start = 0`).
    ///
    /// No-op when the journal is clean, absent, or the FS isn't a
    /// journalled flavour.
    ///
    /// Returns `true` if any work was replayed. After a successful
    /// replay the in-memory bitmaps and group descriptors are
    /// re-read from disk (replay rewrote them) and `INCOMPAT_RECOVER`
    /// is cleared from the in-memory superblock — recovery has been
    /// completed, even if we haven't yet flushed that back to the
    /// on-disk SB (callers writing back to the source should issue a
    /// fresh flush; read-only consumers won't notice the difference).
    pub fn replay_pending_journal(&mut self, dev: &mut dyn BlockDevice) -> Result<bool> {
        if self.sb.feature_compat & constants::feature::COMPAT_HAS_JOURNAL == 0 {
            return Ok(false);
        }
        let replayed = jbd2::replay_journal(self, dev)?;
        if replayed {
            self.reload_groups_from_disk(dev)?;
            self.sb.feature_incompat &= !constants::feature::INCOMPAT_RECOVER;
        }
        Ok(replayed)
    }

    /// Public accessor for `Self::has_metadata_csum`, exposed for the
    /// repack layer (which needs to mirror destination metadata_csum
    /// state when pre-sizing directories).
    pub fn has_metadata_csum_pub(&self) -> bool {
        self.has_metadata_csum()
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
    fn fill_block_pointers(&mut self, ino: u32, inode: &mut Inode, data: &[u32]) -> Result<u32> {
        if matches!(self.kind, FsKind::Ext4) {
            return self.fill_block_pointers_extent(ino, inode, data);
        }
        self.fill_block_pointers_indirect(inode, data)
    }

    /// Ext4 path: pack `data` into an extent tree stored in `i_block`
    /// and set `EXT4_EXTENTS_FL`. A run list that fits the 4-extent
    /// inline budget stays depth-0 (no metadata blocks). Beyond that
    /// [`extent::pack_extent_tree`] builds a tree of whatever depth the
    /// run count needs — leaves, then nested index levels until the top
    /// fits the 4 inline `i_block` idx slots — so the one-shot build
    /// (used by `repack`) handles arbitrarily fragmented files. Returns
    /// the count of metadata (index + leaf) blocks allocated.
    ///
    /// `ino` is needed so every staged tree block can be tracked for
    /// `ext4_extent_tail` CRC stamping at flush when `metadata_csum`
    /// is on.
    fn fill_block_pointers_extent(
        &mut self,
        ino: u32,
        inode: &mut Inode,
        data: &[u32],
    ) -> Result<u32> {
        let runs = extent::coalesce(data);
        inode.flags |= constants::EXT4_EXTENTS_FL;

        if runs.len() <= extent::MAX_EXTENTS_IN_INODE {
            let packed = extent::pack_into_iblock(&runs)?;
            inode.block = extent::bytes_to_iblock(&packed);
            return Ok(0);
        }

        // Build an extent tree of whatever depth the run count needs
        // (depth ≥ 2 for heavily fragmented files / large directories).
        let bs = self.layout.block_size;
        let csum_tail = self.has_metadata_csum();
        let (i_block_bytes, tree_blocks) = {
            let mut alloc = || self.alloc_data_block();
            extent::pack_extent_tree(&runs, bs, csum_tail, &mut alloc)?
        };
        let meta = tree_blocks.len();
        for tb in tree_blocks {
            // Stage each tree block (leaf + internal index nodes); track
            // it so the metadata_csum tail is stamped with this inode's
            // seed at flush time.
            self.push_data_block(tb.phys, tb.image);
            self.track_extent_leaf_block(tb.phys, ino);
        }
        inode.block = extent::bytes_to_iblock(&i_block_bytes);
        Ok(meta as u32)
    }

    /// Ext2 / Ext3 path: direct + single + double + triple indirection.
    /// At 4 KiB blocks that's up to 12 + 1024 + 1024² + 1024³ ≈ 4 TiB —
    /// the actual ext2 single-file cap on most setups (the `LARGE_FILE`
    /// RO-compat feature must be set when any file uses i_size_high,
    /// which the caller does in [`Self::add_file_to_streaming`]).
    ///
    /// A `0` in `data` is a hole (sparse file): the corresponding block
    /// pointer stays 0, and any indirect block whose entire range is
    /// holes is not allocated at all.
    fn fill_block_pointers_indirect(&mut self, inode: &mut Inode, data: &[u32]) -> Result<u32> {
        let bs = self.layout.block_size;
        let ptrs = (bs / 4) as usize;
        let n = data.len();
        let n_direct = constants::N_DIRECT.min(n);
        inode.block[..n_direct].copy_from_slice(&data[..n_direct]);
        let mut allocated_meta = 0u32;
        let mut consumed = n_direct;

        if consumed < n {
            let ind = self.build_indirect_l1(data, &mut consumed, bs, ptrs, &mut allocated_meta)?;
            inode.block[constants::IDX_INDIRECT] = ind;
        }

        if consumed < n {
            let dind =
                self.build_indirect_l2(data, &mut consumed, bs, ptrs, &mut allocated_meta)?;
            inode.block[constants::IDX_DOUBLE_INDIRECT] = dind;
        }

        if consumed < n {
            let tind =
                self.build_indirect_l3(data, &mut consumed, bs, ptrs, &mut allocated_meta)?;
            inode.block[constants::IDX_TRIPLE_INDIRECT] = tind;
        }

        if consumed < n {
            return Err(crate::Error::Unsupported(
                "ext: file exceeds direct+single+double+triple indirection capacity".into(),
            ));
        }

        Ok(allocated_meta)
    }

    /// Build a single-indirect (level-1) block: up to `ptrs` data-block
    /// pointers laid out in a freshly-allocated block. Returns the
    /// block number, or `0` when every pointer in the range is a hole
    /// (in which case nothing is allocated). Increments `meta` by the
    /// indirect block this call allocated.
    fn build_indirect_l1(
        &mut self,
        data: &[u32],
        consumed: &mut usize,
        bs: u32,
        ptrs: usize,
        meta: &mut u32,
    ) -> Result<u32> {
        let take = (data.len() - *consumed).min(ptrs);
        let range = &data[*consumed..*consumed + take];
        *consumed += take;
        if range.iter().all(|&b| b == 0) {
            return Ok(0);
        }
        let ind = self.alloc_data_block()?;
        *meta += 1;
        let mut buf = vec![0u8; bs as usize];
        for (i, &b) in range.iter().enumerate() {
            buf[i * 4..i * 4 + 4].copy_from_slice(&b.to_le_bytes());
        }
        self.push_data_block(ind, buf);
        Ok(ind)
    }

    /// Build a double-indirect (level-2) block: up to `ptrs` slots, each
    /// pointing at a single-indirect block built by `build_indirect_l1`.
    /// Allocates the double-indirect block only when at least one
    /// sub-block is non-empty. Returns `0` when the entire double-
    /// indirect range is holes.
    fn build_indirect_l2(
        &mut self,
        data: &[u32],
        consumed: &mut usize,
        bs: u32,
        ptrs: usize,
        meta: &mut u32,
    ) -> Result<u32> {
        let mut dind_buf = vec![0u8; bs as usize];
        let mut any_sub = false;
        for slot in 0..ptrs {
            if *consumed >= data.len() {
                break;
            }
            let ind = self.build_indirect_l1(data, consumed, bs, ptrs, meta)?;
            if ind != 0 {
                dind_buf[slot * 4..slot * 4 + 4].copy_from_slice(&ind.to_le_bytes());
                any_sub = true;
            }
        }
        if !any_sub {
            return Ok(0);
        }
        let dind = self.alloc_data_block()?;
        *meta += 1;
        self.push_data_block(dind, dind_buf);
        Ok(dind)
    }

    /// Build a triple-indirect (level-3) block: up to `ptrs` slots, each
    /// pointing at a double-indirect block built by `build_indirect_l2`.
    /// Same all-holes elision rule as the lower levels.
    fn build_indirect_l3(
        &mut self,
        data: &[u32],
        consumed: &mut usize,
        bs: u32,
        ptrs: usize,
        meta: &mut u32,
    ) -> Result<u32> {
        let mut tind_buf = vec![0u8; bs as usize];
        let mut any_dind = false;
        for slot in 0..ptrs {
            if *consumed >= data.len() {
                break;
            }
            let dind = self.build_indirect_l2(data, consumed, bs, ptrs, meta)?;
            if dind != 0 {
                tind_buf[slot * 4..slot * 4 + 4].copy_from_slice(&dind.to_le_bytes());
                any_dind = true;
            }
        }
        if !any_dind {
            return Ok(0);
        }
        let tind = self.alloc_data_block()?;
        *meta += 1;
        self.push_data_block(tind, tind_buf);
        Ok(tind)
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
        let meta_blocks = self.fill_block_pointers(ino, &mut inode, &data)?;
        inode.blocks_512 = (blocks + meta_blocks) * (bs / 512);

        // Build the JBD2 v2 journal superblock for block 0 of the journal.
        let jsb = build_jbd2_superblock(bs, blocks);
        self.push_data_block(data[0], jsb);

        self.push_inode(ino, inode);
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
        self.push_inode(ino, inode);
        self.push_data_block(blk, block_bytes);
        self.track_dir_block(blk, ino);
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
        let meta_blocks = self.fill_block_pointers(ino, &mut inode, &data_blocks)?;
        inode.blocks_512 = (target_data_blocks + meta_blocks) * (bs / 512);

        // First data block: "." / "..". All blocks of lost+found are
        // directory blocks owned by inode `ino`.
        let csum_tail = self.has_metadata_csum();
        let with_filetype = self.has_filetype();
        let dir_block =
            dir::make_initial_dir_block(ino, INO_ROOT_DIR, bs, with_filetype, csum_tail);
        self.push_data_block(data_blocks[0], dir_block);
        self.track_dir_block(data_blocks[0], ino);
        // All trailing data blocks: empty-placeholder entry so e2fsck reads
        // them as well-formed empty dir blocks.
        for &blk in &data_blocks[1..] {
            self.data_blocks
                .push((blk, dir::make_empty_dir_block(bs, csum_tail)));
            self.track_dir_block(blk, ino);
        }

        self.groups[0].desc.used_dirs_count += 1;
        self.push_inode(ino, inode);

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
    /// inode is `dir_inode`. Walks the existing data blocks (last-to-first,
    /// since appends cluster at the tail) and writes into the first one
    /// with room; if every block is full, allocates a new data block,
    /// extends the inode's block-pointer storage, and writes into the
    /// fresh block.
    /// Push a staged inode and keep [`inode_idx`](Self::inode_idx) in
    /// sync. All `inodes` growth must go through here.
    fn push_inode(&mut self, ino: u32, inode: Inode) {
        self.inode_idx.insert(ino, self.inodes.len());
        self.inodes.push((ino, inode));
    }

    /// Rebuild `inode_idx` from scratch after a structural edit (e.g. a
    /// `retain` in the remove path) that invalidates positions.
    fn rebuild_inode_idx(&mut self) {
        self.inode_idx = self
            .inodes
            .iter()
            .enumerate()
            .map(|(i, (no, _))| (*no, i))
            .collect();
    }

    /// O(1) staged-inode position by inode number.
    fn inode_pos(&self, ino: u32) -> Option<usize> {
        self.inode_idx.get(&ino).copied()
    }

    /// Push a staged data block and keep `data_block_idx` in sync. All
    /// `data_blocks` growth must go through here.
    fn push_data_block(&mut self, blk: u32, bytes: Vec<u8>) {
        self.data_block_idx.insert(blk, self.data_blocks.len());
        self.data_blocks.push((blk, bytes));
    }

    /// O(1) staged-data-block position by block number.
    fn data_block_pos(&self, blk: u32) -> Option<usize> {
        self.data_block_idx.get(&blk).copied()
    }

    /// Tag `blk` as a directory data block owned by `ino` (idempotent),
    /// keeping `dir_block_set` in sync for O(1) membership checks.
    fn track_dir_block(&mut self, blk: u32, ino: u32) {
        if self.dir_block_set.insert(blk) {
            self.dir_blocks.push((blk, ino));
        }
    }

    fn add_entry_to_dir_block_for(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_inode: u32,
        name: &[u8],
        child_ino: u32,
        file_type: u8,
    ) -> Result<()> {
        self.ensure_inode_staged(dev, dir_inode)?;
        let bs = self.layout.block_size;
        let usable = dir::usable_dir_len(bs, self.has_metadata_csum());
        let with_filetype = self.has_filetype();

        // HTree-indexed dir? Route by hash to the right leaf instead
        // of scanning linearly. The leaf is always non-block-0 (block 0
        // is the dx_root and never holds real entries).
        let inode_copy = self.inodes[self.inode_pos(dir_inode).unwrap()].1;
        if inode_copy.flags & constants::EXT4_INDEX_FL != 0 {
            let logical_leaf = self.dx_route_logical_leaf(dev, dir_inode, name)?;
            let phys = self.file_block(dev, &inode_copy, logical_leaf)?;
            self.ensure_block_staged(dev, phys)?;
            self.track_dir_block(phys, dir_inode);
            let pos = self.data_block_pos(phys).unwrap();
            let block = &mut self.data_blocks[pos].1;
            if !try_append_dir_entry(block, name, child_ino, file_type, with_filetype, usable)? {
                return Err(crate::Error::Unsupported(format!(
                    "ext: HTree leaf {phys} for dir {dir_inode} is full — bucket-split not implemented"
                )));
            }
            return Ok(());
        }

        let n_blocks = inode_copy.size.div_ceil(bs);

        // Try existing blocks last-to-first: the tail block is the only
        // candidate with room under a build-only workload; falling back
        // to earlier blocks covers the cold path where deletions opened
        // slack.
        for logical in (0..n_blocks).rev() {
            let blk = self.file_block(dev, &inode_copy, logical)?;
            if blk == 0 {
                // Sparse gap inside a directory — skip.
                continue;
            }
            self.ensure_block_staged(dev, blk)?;
            self.track_dir_block(blk, dir_inode);
            let pos = self.data_block_pos(blk).unwrap();
            let block = &mut self.data_blocks[pos].1;
            if try_append_dir_entry(block, name, child_ino, file_type, with_filetype, usable)? {
                return Ok(());
            }
        }

        // Every existing block is full (or there are none): grow the dir
        // by one block and write the entry into it.
        self.grow_dir_block_and_append(dev, dir_inode, name, child_ino, file_type)
    }

    /// Allocate one new data block for directory `dir_inode`, append it to
    /// the inode's block-pointer storage (extents for ext4, direct +
    /// single-indirect for ext2/3), initialise it as an empty dir block,
    /// and write the new entry into it.
    fn grow_dir_block_and_append(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_inode: u32,
        name: &[u8],
        child_ino: u32,
        file_type: u8,
    ) -> Result<()> {
        let bs = self.layout.block_size;
        let csum_tail = self.has_metadata_csum();
        let with_filetype = self.has_filetype();
        let usable = dir::usable_dir_len(bs, csum_tail);

        let new_blk = self.alloc_data_block()?;

        // Compute the next logical block index from the current inode size.
        let inode_copy = self
            .inodes
            .iter()
            .find(|(i, _)| *i == dir_inode)
            .map(|(_, i)| *i)
            .unwrap();
        let new_logical = inode_copy.size.div_ceil(bs);

        // Wire the new block into the inode's block-pointer storage.
        let meta_blocks_added =
            self.append_data_block_to_inode(dev, dir_inode, new_logical, new_blk)?;

        // Grow i_size by one block and account for the new data block plus
        // any indirection metadata that the append required.
        let sectors_per_block = bs / 512;
        self.patch_inode(dev, dir_inode, |i| {
            i.size += bs;
            i.blocks_512 += sectors_per_block * (1 + meta_blocks_added);
        })?;

        // Initialise the new dir block in memory (one empty placeholder
        // entry spanning the usable region; csum tail stamped at flush).
        let mut new_buf = dir::make_empty_dir_block(bs, csum_tail);
        // Write the actual entry into the placeholder.
        if !try_append_dir_entry(
            &mut new_buf,
            name,
            child_ino,
            file_type,
            with_filetype,
            usable,
        )? {
            return Err(crate::Error::Unsupported(format!(
                "ext: dir entry for {:?} doesn't fit in a fresh {bs}-byte block",
                String::from_utf8_lossy(name)
            )));
        }
        self.push_data_block(new_blk, new_buf);
        self.track_dir_block(new_blk, dir_inode);
        Ok(())
    }

    /// Append a single new data block to the block-pointer storage of
    /// inode `inode_no`. Dispatches on the extent-tree flag: extents for
    /// ext4 inodes, classic direct + single-indirect for ext2/3 inodes.
    ///
    /// Returns the number of newly allocated *metadata* blocks (indirect
    /// blocks for ext2/3, extent-tree leaves for ext4) so the caller can
    /// fold them into `blocks_512`. The data block itself is allocated by
    /// the caller and is not counted here.
    fn append_data_block_to_inode(
        &mut self,
        dev: &mut dyn BlockDevice,
        inode_no: u32,
        new_logical: u32,
        new_phys: u32,
    ) -> Result<u32> {
        let uses_extents = self
            .inodes
            .iter()
            .find(|(i, _)| *i == inode_no)
            .map(|(_, i)| i.flags & constants::EXT4_EXTENTS_FL != 0)
            .unwrap();
        if uses_extents {
            self.append_extent_inline(dev, inode_no, new_logical, new_phys)
        } else {
            self.append_indirect_block(dev, inode_no, new_logical, new_phys)
        }
    }

    /// Read a staged metadata block's bytes (staging it from disk first
    /// if needed). Used by the incremental extent-tree append path.
    fn staged_block_bytes(&mut self, dev: &mut dyn BlockDevice, phys: u32) -> Result<Vec<u8>> {
        self.ensure_block_staged(dev, phys)?;
        let pos = self.data_block_pos(phys).expect("just staged");
        Ok(self.data_blocks[pos].1.clone())
    }

    /// Replace (or insert) a staged metadata block's image.
    fn update_staged_block(&mut self, phys: u32, image: Vec<u8>) {
        if let Some(pos) = self.data_block_pos(phys) {
            self.data_blocks[pos].1 = image;
        } else {
            self.push_data_block(phys, image);
        }
    }

    /// Append `new_phys` (at logical block `new_logical`) to an
    /// inline-extent-tree inode. Tries to extend the last extent first
    /// (zero allocation, best for the typical contiguous case); otherwise
    /// adds a new extent. Promotes depth-0 → depth-1 when more than 4 leaf
    /// extents are needed, and deeper still (depth-1 → 2 → 3 → …) as the
    /// rightmost spine fills, via [`Self::append_extent_deep`].
    ///
    /// Returns the number of newly allocated metadata blocks (extent
    /// leaves) so the caller can fold them into `blocks_512`.
    fn append_extent_inline(
        &mut self,
        dev: &mut dyn BlockDevice,
        inode_no: u32,
        new_logical: u32,
        new_phys: u32,
    ) -> Result<u32> {
        let inode_copy = self
            .inodes
            .iter()
            .find(|(i, _)| *i == inode_no)
            .map(|(_, i)| *i)
            .unwrap();
        let iblock = extent::iblock_to_bytes(&inode_copy.block);
        let header = extent::decode_header(&iblock[..12])?;
        match header.depth {
            0 => self.append_extent_depth0(dev, inode_no, &iblock, new_logical, new_phys),
            1 => self.append_extent_depth1(dev, inode_no, new_logical, new_phys),
            _ => self.append_extent_deep(dev, inode_no, new_logical, new_phys),
        }
    }

    /// Depth-0 append: try last-extent extension, fall back to adding a
    /// new extent. If that would exceed the 4-extent inline cap, promote
    /// the tree to depth-1.
    fn append_extent_depth0(
        &mut self,
        dev: &mut dyn BlockDevice,
        inode_no: u32,
        iblock: &[u8; 60],
        new_logical: u32,
        new_phys: u32,
    ) -> Result<u32> {
        let (_, mut runs) = extent::decode_depth0_iblock(iblock)?;
        let extended = if let Some(last) = runs.last_mut() {
            last.physical + last.len as u64 == new_phys as u64
                && (last.logical + last.len as u32) == new_logical
                && last.len < extent::MAX_LEN_PER_EXTENT
                && {
                    last.len += 1;
                    true
                }
        } else {
            false
        };
        if !extended {
            runs.push(extent::ExtentRun {
                logical: new_logical,
                len: 1,
                physical: new_phys as u64,
            });
        }
        if runs.len() <= extent::MAX_EXTENTS_IN_INODE {
            let packed = extent::pack_into_iblock(&runs)?;
            self.patch_inode(dev, inode_no, |i| {
                i.block = extent::bytes_to_iblock(&packed);
            })?;
            return Ok(0);
        }
        // Promote depth-0 → depth-1. With locality-favouring sequential
        // allocation, runs.len() usually stays small; we still need this
        // path when files are interleaved with directory growth.
        self.promote_extent_tree_to_depth1(dev, inode_no, runs)
    }

    /// Rebuild the inode's extent tree as a depth-1 layout: one idx node
    /// inline in `i_block`, plus N leaf blocks (each capped to `per_leaf`
    /// extents). Allocates the leaf blocks fresh, stages them in
    /// `data_blocks`, and tracks them for CRC stamping when
    /// `metadata_csum` is on.
    fn promote_extent_tree_to_depth1(
        &mut self,
        dev: &mut dyn BlockDevice,
        inode_no: u32,
        runs: Vec<extent::ExtentRun>,
    ) -> Result<u32> {
        let bs = self.layout.block_size;
        let csum_tail = self.has_metadata_csum();
        let per_leaf = extent::entries_per_leaf_block_capped(bs, csum_tail);
        let need_leaves = runs.len().div_ceil(per_leaf);
        if need_leaves > extent::MAX_INDICES_IN_INODE {
            return Err(crate::Error::Unsupported(format!(
                "ext4: depth-1 tree needs {need_leaves} leaf blocks, max {} inline (depth>1 not supported)",
                extent::MAX_INDICES_IN_INODE
            )));
        }
        let mut leaf_phys = Vec::with_capacity(need_leaves);
        for _ in 0..need_leaves {
            leaf_phys.push(self.alloc_data_block()?);
        }
        let (i_block_bytes, leaf_images) = extent::pack_depth1(&runs, bs, csum_tail, &leaf_phys)?;
        for (phys, image) in leaf_phys.iter().zip(leaf_images) {
            // Stage in data_blocks; track for CRC stamping at flush.
            if let Some(slot) = self.data_blocks.iter_mut().find(|(b, _)| b == phys) {
                slot.1 = image;
            } else {
                self.push_data_block(*phys, image);
            }
            self.track_extent_leaf_block(*phys, inode_no);
        }
        self.patch_inode(dev, inode_no, |i| {
            i.block = extent::bytes_to_iblock(&i_block_bytes);
        })?;
        Ok(need_leaves as u32)
    }

    /// Depth-1 append: walk the idx array, load the last leaf block,
    /// extend its last extent or add a new one; if that leaf is full,
    /// allocate another leaf and add an idx entry pointing at it (capped
    /// at 4 inline idx entries — beyond that we'd need depth-2).
    fn append_extent_depth1(
        &mut self,
        dev: &mut dyn BlockDevice,
        inode_no: u32,
        new_logical: u32,
        new_phys: u32,
    ) -> Result<u32> {
        let bs = self.layout.block_size;
        let csum_tail = self.has_metadata_csum();
        let per_leaf = extent::entries_per_leaf_block_capped(bs, csum_tail);

        let inode_copy = self
            .inodes
            .iter()
            .find(|(i, _)| *i == inode_no)
            .map(|(_, i)| *i)
            .unwrap();
        let iblock = extent::iblock_to_bytes(&inode_copy.block);
        let (_, mut indices) = extent::decode_idx_iblock(&iblock)?;
        if indices.is_empty() {
            return Err(crate::Error::InvalidImage(
                "ext4: depth-1 extent tree with zero idx entries".into(),
            ));
        }
        // Append to the LAST leaf block (the only candidate with room
        // under the streaming-build workload).
        let last_idx = indices.last().copied().unwrap();
        let last_leaf_phys = last_idx.leaf as u32;
        self.ensure_block_staged(dev, last_leaf_phys)?;
        self.track_extent_leaf_block(last_leaf_phys, inode_no);

        let leaf_bytes = self
            .data_blocks
            .iter()
            .find(|(b, _)| *b == last_leaf_phys)
            .map(|(_, bytes)| bytes.clone())
            .unwrap();
        let (leaf_header, mut leaf_runs) = extent::decode_leaf_block(&leaf_bytes[..bs as usize])?;
        let _ = leaf_header;

        // Try extending the last extent on the last leaf.
        let extended = if let Some(last) = leaf_runs.last_mut() {
            last.physical + last.len as u64 == new_phys as u64
                && (last.logical + last.len as u32) == new_logical
                && last.len < extent::MAX_LEN_PER_EXTENT
                && {
                    last.len += 1;
                    true
                }
        } else {
            false
        };
        let mut allocated_meta = 0u32;
        if extended {
            // Re-encode the last leaf in place.
            let new_image = extent::encode_leaf_block(&leaf_runs, bs, csum_tail)?;
            if let Some(slot) = self
                .data_blocks
                .iter_mut()
                .find(|(b, _)| *b == last_leaf_phys)
            {
                slot.1 = new_image;
            }
            return Ok(allocated_meta);
        }
        // Try adding a new extent into the last leaf.
        if leaf_runs.len() < per_leaf {
            leaf_runs.push(extent::ExtentRun {
                logical: new_logical,
                len: 1,
                physical: new_phys as u64,
            });
            let new_image = extent::encode_leaf_block(&leaf_runs, bs, csum_tail)?;
            if let Some(slot) = self
                .data_blocks
                .iter_mut()
                .find(|(b, _)| *b == last_leaf_phys)
            {
                slot.1 = new_image;
            }
            return Ok(allocated_meta);
        }
        // Last leaf is full. Allocate a new leaf with the single new
        // extent, and add an idx entry pointing at it.
        let new_run = extent::ExtentRun {
            logical: new_logical,
            len: 1,
            physical: new_phys as u64,
        };
        let new_leaf_phys = self.alloc_data_block()?;
        allocated_meta += 1;
        let new_leaf_image = extent::encode_leaf_block(&[new_run], bs, csum_tail)?;
        self.push_data_block(new_leaf_phys, new_leaf_image);
        self.track_extent_leaf_block(new_leaf_phys, inode_no);

        if indices.len() < extent::MAX_INDICES_IN_INODE {
            // Still room for another idx inline: stay depth-1.
            indices.push(extent::ExtentIdx {
                block: new_logical,
                leaf: new_leaf_phys as u64,
            });
            let packed = extent::encode_idx_iblock(&indices, 1);
            self.patch_inode(dev, inode_no, |i| {
                i.block = extent::bytes_to_iblock(&packed);
            })?;
            return Ok(allocated_meta);
        }

        // Inline idx slots are full → promote to depth-2: the existing
        // leaves move into one internal index block, joined by the new
        // leaf, and `i_block` holds a single idx pointing at it.
        let mut ib_indices = indices;
        ib_indices.push(extent::ExtentIdx {
            block: new_logical,
            leaf: new_leaf_phys as u64,
        });
        let ib_phys = self.alloc_data_block()?;
        allocated_meta += 1;
        let ib_image = extent::encode_idx_block(&ib_indices, bs, csum_tail, 1)?;
        self.push_data_block(ib_phys, ib_image);
        self.track_extent_leaf_block(ib_phys, inode_no);
        let top = [extent::ExtentIdx {
            block: ib_indices[0].block,
            leaf: ib_phys as u64,
        }];
        let packed = extent::encode_idx_iblock(&top, 2);
        self.patch_inode(dev, inode_no, |i| {
            i.block = extent::bytes_to_iblock(&packed);
        })?;
        Ok(allocated_meta)
    }

    /// Incrementally append one data block to an extent tree of depth ≥ 2.
    /// Walks the rightmost spine to the deepest leaf and inserts there,
    /// then propagates node splits back up: a full leaf spawns a new leaf,
    /// a full index node spawns a new sibling, and a full inline root is
    /// promoted one level deeper. Because the streaming writer only ever
    /// appends at the logical tail, only the rightmost path is touched, so
    /// each call is O(depth). Returns the count of newly allocated metadata
    /// (index + leaf) blocks for `blocks_512`.
    fn append_extent_deep(
        &mut self,
        dev: &mut dyn BlockDevice,
        inode_no: u32,
        new_logical: u32,
        new_phys: u32,
    ) -> Result<u32> {
        let bs = self.layout.block_size;
        let csum_tail = self.has_metadata_csum();

        let inode_copy = self.inodes[self.inode_pos(inode_no).unwrap()].1;
        let iblock = extent::iblock_to_bytes(&inode_copy.block);
        let header = extent::decode_header(&iblock[..12])?;
        let depth = header.depth;
        let (_, mut top) = extent::decode_idx_iblock(&iblock)?;
        let rightmost = top.last().expect("depth ≥ 2 root has ≥ 1 idx").leaf as u32;

        // Recurse into the rightmost subtree (an on-disk index block).
        match self.append_into_node(dev, inode_no, rightmost, depth - 1, new_logical, new_phys)? {
            NodeAppend::Done(meta) => Ok(meta),
            NodeAppend::NewSibling { idx, meta } => {
                // The rightmost subtree split and handed us a fresh node at
                // depth `depth - 1` to slot into the inline root.
                if top.len() < extent::MAX_INDICES_IN_INODE {
                    top.push(idx);
                    let packed = extent::encode_idx_iblock(&top, depth);
                    self.patch_inode(dev, inode_no, |i| {
                        i.block = extent::bytes_to_iblock(&packed);
                    })?;
                    return Ok(meta);
                }
                // Root is full → promote one level. The existing entries plus
                // the new sibling (all depth-1 nodes relative to the root)
                // move into a single fresh index block at `depth`, and the
                // inline root becomes a one-entry `depth + 1` node pointing
                // at it.
                top.push(idx);
                let combined_phys = self.alloc_data_block()?;
                let combined_img = extent::encode_idx_block(&top, bs, csum_tail, depth)?;
                self.push_data_block(combined_phys, combined_img);
                self.track_extent_leaf_block(combined_phys, inode_no);
                let new_top = [extent::ExtentIdx {
                    block: top[0].block,
                    leaf: combined_phys as u64,
                }];
                let packed = extent::encode_idx_iblock(&new_top, depth + 1);
                self.patch_inode(dev, inode_no, |i| {
                    i.block = extent::bytes_to_iblock(&packed);
                })?;
                Ok(meta + 1)
            }
        }
    }

    /// Recursive helper for [`Self::append_extent_deep`]: append into the
    /// subtree rooted at on-disk index block `node_phys` (header depth
    /// `node_depth` ≥ 1), always following the rightmost child. Returns
    /// either [`NodeAppend::Done`] when the insert was absorbed without
    /// growing this node's parent, or [`NodeAppend::NewSibling`] carrying a
    /// freshly-allocated sibling node (at `node_depth`) that the caller must
    /// link into the level above.
    fn append_into_node(
        &mut self,
        dev: &mut dyn BlockDevice,
        inode_no: u32,
        node_phys: u32,
        node_depth: u16,
        new_logical: u32,
        new_phys: u32,
    ) -> Result<NodeAppend> {
        let bs = self.layout.block_size;
        let csum_tail = self.has_metadata_csum();
        let per = extent::entries_per_leaf_block_capped(bs, csum_tail);

        let node_bytes = self.staged_block_bytes(dev, node_phys)?;
        let node_hdr = extent::decode_header(&node_bytes[..12])?;
        let mut indices: Vec<extent::ExtentIdx> = (0..node_hdr.entries as usize)
            .map(|i| extent::decode_idx(&node_bytes[12 + i * 12..24 + i * 12]))
            .collect();
        let rightmost = indices.last().expect("index node has ≥ 1 entry").leaf as u32;

        // The new child to splice into this node (a fresh leaf when this is
        // a depth-1 node, or a fresh sibling bubbled up from below) along
        // with how many metadata blocks its creation cost.
        let (new_child, child_meta) = if node_depth == 1 {
            // Children are leaves. Try the rightmost leaf in place first.
            self.track_extent_leaf_block(rightmost, inode_no);
            let leaf_bytes = self.staged_block_bytes(dev, rightmost)?;
            let (_, mut leaf_runs) = extent::decode_leaf_block(&leaf_bytes[..bs as usize])?;
            let extended = if let Some(last) = leaf_runs.last_mut() {
                last.physical + last.len as u64 == new_phys as u64
                    && (last.logical + last.len as u32) == new_logical
                    && last.len < extent::MAX_LEN_PER_EXTENT
                    && {
                        last.len += 1;
                        true
                    }
            } else {
                false
            };
            if extended || leaf_runs.len() < per {
                if !extended {
                    leaf_runs.push(extent::ExtentRun {
                        logical: new_logical,
                        len: 1,
                        physical: new_phys as u64,
                    });
                }
                let img = extent::encode_leaf_block(&leaf_runs, bs, csum_tail)?;
                self.update_staged_block(rightmost, img);
                return Ok(NodeAppend::Done(0));
            }
            // Rightmost leaf is full → spawn a new leaf for the extent.
            let new_leaf_phys = self.alloc_data_block()?;
            let new_run = extent::ExtentRun {
                logical: new_logical,
                len: 1,
                physical: new_phys as u64,
            };
            let img = extent::encode_leaf_block(&[new_run], bs, csum_tail)?;
            self.push_data_block(new_leaf_phys, img);
            self.track_extent_leaf_block(new_leaf_phys, inode_no);
            (
                extent::ExtentIdx {
                    block: new_logical,
                    leaf: new_leaf_phys as u64,
                },
                1,
            )
        } else {
            // Children are index blocks → recurse into the rightmost one.
            match self.append_into_node(
                dev,
                inode_no,
                rightmost,
                node_depth - 1,
                new_logical,
                new_phys,
            )? {
                NodeAppend::Done(meta) => return Ok(NodeAppend::Done(meta)),
                NodeAppend::NewSibling { idx, meta } => (idx, meta),
            }
        };

        // Splice `new_child` into this node, or split it when full.
        if indices.len() < per {
            indices.push(new_child);
            let img = extent::encode_idx_block(&indices, bs, csum_tail, node_depth)?;
            self.update_staged_block(node_phys, img);
            self.track_extent_leaf_block(node_phys, inode_no);
            return Ok(NodeAppend::Done(child_meta));
        }
        // This node is full → allocate a sibling at the same depth holding
        // just the new child, and hand it up for the parent to link in.
        let sib_phys = self.alloc_data_block()?;
        let sib_img = extent::encode_idx_block(&[new_child], bs, csum_tail, node_depth)?;
        self.push_data_block(sib_phys, sib_img);
        self.track_extent_leaf_block(sib_phys, inode_no);
        Ok(NodeAppend::NewSibling {
            idx: extent::ExtentIdx {
                block: new_child.block,
                leaf: sib_phys as u64,
            },
            meta: child_meta + 1,
        })
    }

    /// Append `new_phys` (at logical block `new_logical`) to an ext2/3
    /// inode using direct + single-indirect block pointers. Allocates an
    /// indirect block on demand (returning 1 in that case so the caller
    /// folds it into `blocks_512`).
    fn append_indirect_block(
        &mut self,
        dev: &mut dyn BlockDevice,
        inode_no: u32,
        new_logical: u32,
        new_phys: u32,
    ) -> Result<u32> {
        let bs = self.layout.block_size;
        let ptrs_per_block = bs / 4;
        let n_direct = constants::N_DIRECT as u32;
        if new_logical < n_direct {
            self.patch_inode(dev, inode_no, |i| {
                i.block[new_logical as usize] = new_phys;
            })?;
            return Ok(0);
        }
        let single_off = new_logical - n_direct;
        if single_off >= ptrs_per_block {
            return Err(crate::Error::Unsupported(format!(
                "ext: directory grew past single-indirect capacity at logical block {new_logical}"
            )));
        }
        // Locate (or allocate) the single-indirect block.
        let inode_copy = self
            .inodes
            .iter()
            .find(|(i, _)| *i == inode_no)
            .map(|(_, i)| *i)
            .unwrap();
        let (ind_blk, meta_added) = match inode_copy.block[constants::IDX_INDIRECT] {
            0 => {
                let blk = self.alloc_data_block()?;
                self.patch_inode(dev, inode_no, |i| {
                    i.block[constants::IDX_INDIRECT] = blk;
                })?;
                // Initialise the indirect block to all zeros (the holes
                // pattern); the slot we're about to write is the only
                // non-zero entry initially.
                self.push_data_block(blk, vec![0u8; bs as usize]);
                (blk, 1u32)
            }
            existing => {
                self.ensure_block_staged(dev, existing)?;
                (existing, 0u32)
            }
        };
        let ind_buf = self
            .data_blocks
            .iter_mut()
            .find(|(b, _)| *b == ind_blk)
            .map(|(_, bytes)| bytes)
            .unwrap();
        let off = single_off as usize * 4;
        ind_buf[off..off + 4].copy_from_slice(&new_phys.to_le_bytes());
        Ok(meta_added)
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
        if self.inode_idx.contains_key(&ino) {
            return Ok(());
        }
        let inode = self.read_inode(dev, ino)?;
        self.push_inode(ino, inode);
        Ok(())
    }

    /// Ensure block `blk` is in the staged write set, fetching from disk
    /// if not. No-op if already staged.
    fn ensure_block_staged(&mut self, dev: &mut dyn BlockDevice, blk: u32) -> Result<()> {
        if self.data_block_idx.contains_key(&blk) {
            return Ok(());
        }
        let mut buf = vec![0u8; self.layout.block_size as usize];
        self.read_block(dev, blk, &mut buf)?;
        self.push_data_block(blk, buf);
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
            // Resume from the per-group cursor (clamped to the data area)
            // instead of rescanning from bit 0 every call — the original
            // linear scan was O(allocated) per call and dominated the
            // 100 k-file repack profile (~35 % of total Ir).
            let cursor = self.groups[gi].next_free_block_bit.max(start_rel);
            if cursor >= group_blocks {
                continue;
            }
            let bitmap = &mut self.groups[gi].block_bitmap;
            for bit in cursor..group_blocks {
                if !test_bit(bitmap, bit) {
                    set_bit(bitmap, bit);
                    self.groups[gi].next_free_block_bit = bit + 1;
                    return Ok(layout_g.start_block + bit);
                }
            }
            // No free bit at-or-past the cursor — mark the group exhausted
            // so the next call skips straight to the next group.
            self.groups[gi].next_free_block_bit = group_blocks;
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
        self.data_block_idx.clear();
        self.dir_blocks.clear();
        self.dir_block_set.clear();
        self.extent_leaf_blocks.clear();
        self.dx_root_blocks.clear();
        self.dx_node_blocks.clear();
        Ok(())
    }

    /// Stamp every staged metadata block's CRC32C tail in place: regular
    /// dir blocks (12-byte trailing dirent + CRC), extent-tree leaf
    /// blocks (4-byte `ext4_extent_tail`), and HTree dx_root blocks
    /// (8-byte `dx_tail` covering only the in-use prefix). No-op when
    /// `metadata_csum` is off. Called by [`flush_metadata`] before the
    /// journal commit so the journaled image matches what the checkpoint
    /// phase writes to each block's home location.
    fn stamp_dir_block_checksums(&mut self) {
        if !self.has_metadata_csum() {
            return;
        }
        let seed = self.csum_seed();
        for (blk, bytes) in &mut self.data_blocks {
            // dx_root: distinct csum layout (dt_tail at the very end,
            // covers only `count_offset + count * 8` bytes from the
            // start). Check this BEFORE the generic dir_blocks lookup
            // because dx_root blocks aren't tagged in dir_blocks.
            if let Some((_, owner_ino, count)) = self
                .dx_root_blocks
                .iter()
                .find(|(b, _, _)| b == blk)
                .copied()
            {
                let generation = self
                    .inodes
                    .iter()
                    .find(|(i, _)| *i == owner_ino)
                    .map(|(_, i)| i.generation)
                    .unwrap_or(0);
                let c = htree::compute_dx_csum(
                    csum::raw_update,
                    seed,
                    owner_ino,
                    generation,
                    bytes,
                    htree::DX_ROOT_HEADER_LEN,
                    count as usize,
                );
                htree::stamp_dx_csum(bytes, c);
                continue;
            }
            // dx_node: same csum scheme but smaller header (12 bytes
            // for the fake dirent vs 32 for dx_root's `.`/`..`/info).
            if let Some((_, owner_ino, count)) = self
                .dx_node_blocks
                .iter()
                .find(|(b, _, _)| b == blk)
                .copied()
            {
                let generation = self
                    .inodes
                    .iter()
                    .find(|(i, _)| *i == owner_ino)
                    .map(|(_, i)| i.generation)
                    .unwrap_or(0);
                let c = htree::compute_dx_csum(
                    csum::raw_update,
                    seed,
                    owner_ino,
                    generation,
                    bytes,
                    htree::DX_NODE_HEADER_LEN,
                    count as usize,
                );
                htree::stamp_dx_csum(bytes, c);
                continue;
            }
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
                continue;
            }
            if let Some((_, owner_ino)) = self.extent_leaf_blocks.iter().find(|(b, _)| b == blk) {
                let generation = self
                    .inodes
                    .iter()
                    .find(|(i, _)| i == owner_ino)
                    .map(|(_, i)| i.generation)
                    .unwrap_or(0);
                let n = bytes.len();
                let c = csum::extent_tail(seed, *owner_ino, generation, &bytes[..n - 4]);
                bytes[n - 4..].copy_from_slice(&c.to_le_bytes());
            }
        }
    }

    /// Register `blk` as an extent-tree leaf block owned by `inode_no`.
    /// At flush time the per-block `ext4_extent_tail` CRC32C is stamped
    /// against this inode's number + generation. Idempotent.
    pub(crate) fn track_extent_leaf_block(&mut self, blk: u32, inode_no: u32) {
        if !self.extent_leaf_blocks.iter().any(|(b, _)| *b == blk) {
            self.extent_leaf_blocks.push((blk, inode_no));
        }
    }

    /// Reverse of [`track_extent_leaf_block`]: drop the record so a freed
    /// leaf block is no longer stamped on flush.
    pub(crate) fn untrack_extent_leaf_block(&mut self, blk: u32) {
        self.extent_leaf_blocks.retain(|(b, _)| *b != blk);
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

        // A single transaction can never exceed the journal ring. When the
        // metadata set is larger than the journal can hold — the bulk
        // build / populate case (e.g. a directory with 100k entries dirties
        // thousands of inode-table + directory blocks at once) — there is
        // no way to journal it, and no concurrent mount to stay consistent
        // with, so write the metadata directly and leave the journal clean.
        // mke2fs / genext2fs produce exactly this (empty journal) for a
        // freshly-populated image.
        let avail = jsb.maxlen.saturating_sub(jsb.first) as usize;
        let first_cap = jbd2::descriptor_tag_capacity(bs, true);
        let next_cap = jbd2::descriptor_tag_capacity(bs, false);
        let n_descs = if images.len() <= first_cap {
            1
        } else {
            1 + (images.len() - first_cap).div_ceil(next_cap)
        };
        let need = n_descs + images.len() + 1;
        if need > avail {
            for (blk, bytes) in images {
                dev.write_at(*blk as u64 * bs as u64, bytes)?;
            }
            return Ok(());
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
        // Cap at u32 logical blocks — this is well beyond the
        // triple-indirect addressing range at every block size we
        // emit (≈ 4 TiB at 4 KiB blocks), so a request past this is
        // already going to fail at `fill_block_pointers` anyway.
        // Catch it early with a clearer message.
        let n_data_blocks_u64 = len.div_ceil(bs as u64);
        if n_data_blocks_u64 > u32::MAX as u64 {
            return Err(crate::Error::Unsupported(format!(
                "ext: file of {len} bytes needs {n_data_blocks_u64} blocks (> u32::MAX)"
            )));
        }
        // Files > 4 GiB ride the RO_COMPAT_LARGE_FILE layout: the
        // upper 32 bits of size live in i_size_high (the
        // `size_hi_or_dir_acl` field for regular files), and the
        // feature bit must be set so older readers refuse the FS
        // instead of silently truncating to the low 32 bits.
        if len > u32::MAX as u64 {
            self.sb.feature_ro_compat |= constants::feature::RO_COMPAT_LARGE_FILE;
        }

        // Inline-data fast path: files that fit in the inode's
        // i_block array don't need a data block at all. Reduces
        // small-file overhead from "1 inode + 1 data block" to "1
        // inode" — a 10-byte file goes from 4 KiB on disk to ~128.
        //
        // The on-disk contract: when `EXT4_INLINE_DATA_FL` is set
        // every such inode MUST carry a `system.data` xattr too (the
        // kernel uses its presence as the "this inode is inline-data"
        // probe; e2fsck enforces it). The xattr's value holds any
        // overflow beyond i_block's 60 bytes — for files ≤ 60 bytes
        // we stamp an empty value to satisfy the invariant.
        const INLINE_CAP: u64 = 60;
        if self.has_inline_data() && len <= INLINE_CAP {
            let mut payload = [0u8; INLINE_CAP as usize];
            reader.read_exact(&mut payload[..len as usize])?;
            let ino = self.alloc_inode()?;
            let mut inode = Inode::regular(
                len as u32,
                meta.mode & 0o7777,
                meta.uid,
                meta.gid,
                meta.mtime,
            );
            inode.flags |= constants::EXT4_INLINE_DATA_FL;
            inode.blocks_512 = 0;
            // Pack the data into i_block (60 bytes = 15 × 4-byte slots).
            for (i, slot) in inode.block.iter_mut().enumerate() {
                let off = i * 4;
                *slot = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
            }
            self.push_inode(ino, inode);
            self.add_entry_to_dir_block_for(dev, parent_ino, name, ino, constants::DENT_REG)?;
            // Stamp the marker xattr. Value is empty for files that
            // fit entirely in i_block; for > 60 bytes (deferred — see
            // the cap above) it would hold bytes 60..end.
            let marker = xattr::Xattr::new("system.data", Vec::<u8>::new());
            self.set_xattrs(dev, ino, &[marker])?;
            return Ok(ino);
        }

        let n_data_blocks = n_data_blocks_u64 as u32;

        let ino = self.alloc_inode()?;
        let mut inode = Inode::regular(
            // Low 32 bits; size_hi is stamped below for > 4 GiB files.
            len as u32,
            meta.mode & 0o7777,
            meta.uid,
            meta.gid,
            meta.mtime,
        );
        inode.set_file_size(len);

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

        let allocated_meta_blocks = self.fill_block_pointers(ino, &mut inode, &data_blocks)?;
        inode.blocks_512 = (allocated_data + allocated_meta_blocks) * (bs / 512);

        self.push_inode(ino, inode);
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
            self.fill_block_pointers_extent(ino, &mut inode, &[blk])?;
        } else {
            inode.block[0] = blk;
        }
        inode.blocks_512 = bs / 512;
        let csum_tail = self.has_metadata_csum();
        let with_filetype = self.has_filetype();
        let block_bytes =
            dir::make_initial_dir_block(ino, parent_ino, bs, with_filetype, csum_tail);
        self.push_data_block(blk, block_bytes);
        self.track_dir_block(blk, ino);
        self.push_inode(ino, inode);
        self.groups[0].desc.used_dirs_count += 1;

        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino, constants::DENT_DIR)?;
        self.patch_inode(dev, parent_ino, |i| i.links_count += 1)?;
        Ok(ino)
    }

    /// Like [`Self::add_dir_to`] but pre-allocates `n_blocks` data blocks
    /// for the new directory's body, wired into the inode in one shot.
    /// Use this when the caller knows the destination directory's child
    /// count up front (e.g. repack walking a source directory): a
    /// contiguous run from the sequential allocator coalesces into a
    /// single extent, side-stepping the per-grow extent-tree mutations
    /// and keeping the dir at depth-0 even with many entries.
    ///
    /// `n_blocks` must be at least 1. If the caller under-estimates,
    /// the streaming `add_entry_to_dir_block_for` growth path takes over
    /// naturally — this is a hint, not a hard limit.
    pub fn add_dir_to_with_blocks(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u32,
        name: &[u8],
        meta: FileMeta,
        n_blocks: u32,
    ) -> Result<u32> {
        let n_blocks = n_blocks.max(1);
        let bs = self.layout.block_size;
        let csum_tail = self.has_metadata_csum();
        let with_filetype = self.has_filetype();
        let ino = self.alloc_inode()?;
        // Allocate the dir's blocks back-to-back. With the sequential
        // bitmap allocator this is a contiguous run within a group, so
        // `coalesce` produces a single extent.
        let mut blocks = Vec::with_capacity(n_blocks as usize);
        for _ in 0..n_blocks {
            blocks.push(self.alloc_data_block()?);
        }
        let mut inode = Inode::directory(
            bs * n_blocks,
            meta.mode & 0o7777,
            meta.uid,
            meta.gid,
            meta.mtime,
        );
        // ext4: extent tree (depth-0 for any contiguous run that fits in
        // 4 leaves, depth-1 otherwise via the streaming-grow promote
        // path — but with sequential alloc we should always stay
        // depth-0). ext2/3: direct + indirect chain.
        let allocated_meta = if matches!(self.kind, FsKind::Ext4) {
            self.fill_block_pointers_extent(ino, &mut inode, &blocks)?
        } else {
            self.fill_block_pointers_indirect(&mut inode, &blocks)?
        };
        inode.blocks_512 = (n_blocks + allocated_meta) * (bs / 512);

        // First block: "." / ".."; all trailing blocks: empty placeholder
        // so the linear-scan reader sees well-formed dir blocks.
        let head = dir::make_initial_dir_block(ino, parent_ino, bs, with_filetype, csum_tail);
        self.push_data_block(blocks[0], head);
        self.track_dir_block(blocks[0], ino);
        for &blk in &blocks[1..] {
            self.data_blocks
                .push((blk, dir::make_empty_dir_block(bs, csum_tail)));
            self.track_dir_block(blk, ino);
        }
        self.push_inode(ino, inode);
        self.groups[0].desc.used_dirs_count += 1;

        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino, constants::DENT_DIR)?;
        self.patch_inode(dev, parent_ino, |i| i.links_count += 1)?;
        Ok(ino)
    }

    /// Like [`Self::add_dir_to_with_blocks`] but builds an HTree-indexed
    /// directory: block 0 becomes a `dx_root` pointing at K leaf
    /// blocks, the new inode carries `EXT4_INDEX_FL`, and later
    /// `add_entry_to_dir_block_for` calls hash each name and route to
    /// the matching leaf (preserving lookup-by-hash order).
    ///
    /// `expected_names` is the full list of names the caller will add
    /// under this directory — needed up front so we can hash each one
    /// and partition them into leaves. Names that don't end up being
    /// added still cost an unused tail-slack byte or two in their
    /// leaf, but the writer doesn't care; conversely, names added that
    /// weren't predicted still fit because every entry is hash-routed
    /// to a leaf with available room.
    ///
    /// Ext2/Ext3 don't support HTree; calling this on a non-ext4
    /// instance returns `Unsupported`.
    pub fn add_dir_indexed(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u32,
        name: &[u8],
        meta: FileMeta,
        expected_names: &[&[u8]],
    ) -> Result<u32> {
        if !matches!(self.kind, FsKind::Ext4) {
            return Err(crate::Error::Unsupported(
                "ext: HTree (DIR_INDEX) requires ext4".into(),
            ));
        }
        let bs = self.layout.block_size;
        let csum_tail = self.has_metadata_csum();
        let with_filetype = self.has_filetype();
        let usable = dir::usable_dir_len(bs, csum_tail);

        // Hash every expected name, sort, then bucket by byte budget
        // (87.5% target fill so post-creation appends have slack).
        let mut hashes: Vec<(u32, usize)> = expected_names
            .iter()
            .enumerate()
            .map(|(i, n)| (htree::half_md4_hash(n).0, i))
            .collect();
        hashes.sort_by_key(|(h, _)| *h);

        let mut leaves: Vec<Vec<usize>> = vec![Vec::new()];
        let mut current_bytes: usize = 0;
        let cap = usable.saturating_sub(usable / 8);
        for &(_, idx) in &hashes {
            let need = dir::min_rec_len(expected_names[idx].len());
            if current_bytes + need > cap && !leaves.last().unwrap().is_empty() {
                leaves.push(Vec::new());
                current_bytes = 0;
            }
            leaves.last_mut().unwrap().push(idx);
            current_bytes += need;
        }
        let n_leaves = leaves.len();
        // First hash of each leaf — the dx_entry boundary key for the
        // routing layers. Leaf 0's boundary is 0 (implicit leftmost).
        let leaf_first_hash: Vec<u32> = leaves
            .iter()
            .map(|leaf| {
                let idx = leaf[0];
                htree::half_md4_hash(expected_names[idx]).0
            })
            .collect();

        let root_limit = htree::dx_root_limit(bs, csum_tail);
        let node_limit = htree::dx_node_limit(bs, csum_tail);

        // Decide tree shape:
        //   indirect_levels = 0 — dx_root entries point at leaves.
        //                         Cap: root_limit leaves.
        //   indirect_levels = 1 — dx_root entries point at dx_nodes,
        //                         dx_node entries point at leaves.
        //                         Cap: root_limit * node_limit leaves.
        let (indirect_levels, n_nodes) = if n_leaves <= root_limit {
            (0u8, 0usize)
        } else {
            // Partition leaves across as few dx_nodes as possible,
            // each holding up to `node_limit` leaves.
            let need_nodes = n_leaves.div_ceil(node_limit);
            if need_nodes > root_limit {
                return Err(crate::Error::Unsupported(format!(
                    "ext: HTree dir {:?} needs {n_leaves} leaves, exceeds the depth-1 cap \
                     (root_limit={root_limit} * node_limit={node_limit} = {}); depth >= 2 \
                     not implemented",
                    String::from_utf8_lossy(name),
                    root_limit * node_limit
                )));
            }
            (1u8, need_nodes)
        };

        let total_blocks = 1 + n_nodes + n_leaves; // dx_root + dx_nodes + leaves
        let ino = self.alloc_inode()?;
        let mut blocks = Vec::with_capacity(total_blocks);
        for _ in 0..total_blocks {
            blocks.push(self.alloc_data_block()?);
        }
        let mut inode = Inode::directory(
            bs * total_blocks as u32,
            meta.mode & 0o7777,
            meta.uid,
            meta.gid,
            meta.mtime,
        );
        let allocated_meta = self.fill_block_pointers_extent(ino, &mut inode, &blocks)?;
        inode.blocks_512 = (total_blocks as u32 + allocated_meta) * (bs / 512);
        inode.flags |= constants::EXT4_INDEX_FL;

        // Logical-block layout (for the inode's view of the dir body):
        //   [0]               dx_root
        //   [1 .. 1+n_nodes]  dx_nodes (only when indirect_levels = 1)
        //   [1+n_nodes ..]    leaves
        let nodes_start_logical: u32 = 1;
        let leaves_start_logical: u32 = (1 + n_nodes) as u32;

        if indirect_levels == 0 {
            // Single-level: dx_root entries map directly to leaves.
            // Slot 0 is the countlimit (with leftmost leaf in its
            // block field); slots 1..n carry (hash, leaf_logical_blk).
            let mut entries: Vec<htree::DxEntry> = Vec::with_capacity(n_leaves);
            entries.push(htree::DxEntry {
                hash: htree::pack_countlimit(root_limit as u16, n_leaves as u16),
                block: leaves_start_logical,
            });
            for (i, &h) in leaf_first_hash.iter().enumerate().skip(1) {
                entries.push(htree::DxEntry {
                    hash: h,
                    block: leaves_start_logical + i as u32,
                });
            }
            let dx_root_buf = htree::make_dx_root_block(
                ino,
                parent_ino,
                bs,
                htree::DX_HASH_HALF_MD4,
                0,
                &entries,
                with_filetype,
                csum_tail,
            );
            self.push_data_block(blocks[0], dx_root_buf);
            self.dx_root_blocks.push((blocks[0], ino, n_leaves as u16));
        } else {
            // Two-level: dx_root entries map to dx_nodes; dx_node
            // entries map to leaves. We chunk leaves into groups of
            // `node_limit`, one group per dx_node.
            let mut leaf_idx = 0usize;
            let mut node_first_hashes: Vec<u32> = Vec::with_capacity(n_nodes);
            for node_i in 0..n_nodes {
                let chunk_start = leaf_idx;
                let chunk_end = (chunk_start + node_limit).min(n_leaves);
                let chunk_len = chunk_end - chunk_start;
                node_first_hashes.push(leaf_first_hash[chunk_start]);

                // dx_node entries: countlimit + (chunk_len - 1) real slots.
                let mut node_entries: Vec<htree::DxEntry> = Vec::with_capacity(chunk_len);
                node_entries.push(htree::DxEntry {
                    hash: htree::pack_countlimit(node_limit as u16, chunk_len as u16),
                    block: leaves_start_logical + chunk_start as u32,
                });
                for off in 1..chunk_len {
                    node_entries.push(htree::DxEntry {
                        hash: leaf_first_hash[chunk_start + off],
                        block: leaves_start_logical + (chunk_start + off) as u32,
                    });
                }
                let node_buf = htree::make_dx_node_block(bs, &node_entries, csum_tail);
                let phys = blocks[(nodes_start_logical as usize) + node_i];
                self.push_data_block(phys, node_buf);
                self.dx_node_blocks.push((phys, ino, chunk_len as u16));

                leaf_idx = chunk_end;
            }

            // dx_root: slot 0 is countlimit pointing at the first
            // dx_node; slots 1..n point at subsequent dx_nodes.
            let mut root_entries: Vec<htree::DxEntry> = Vec::with_capacity(n_nodes);
            root_entries.push(htree::DxEntry {
                hash: htree::pack_countlimit(root_limit as u16, n_nodes as u16),
                block: nodes_start_logical,
            });
            for (i, &h) in node_first_hashes.iter().enumerate().skip(1) {
                root_entries.push(htree::DxEntry {
                    hash: h,
                    block: nodes_start_logical + i as u32,
                });
            }
            let dx_root_buf = htree::make_dx_root_block(
                ino,
                parent_ino,
                bs,
                htree::DX_HASH_HALF_MD4,
                1,
                &root_entries,
                with_filetype,
                csum_tail,
            );
            self.push_data_block(blocks[0], dx_root_buf);
            self.dx_root_blocks.push((blocks[0], ino, n_nodes as u16));
        }

        // Leaves start empty; the router fills them as entries arrive.
        for i in 0..n_leaves {
            let blk = blocks[(leaves_start_logical as usize) + i];
            self.data_blocks
                .push((blk, dir::make_empty_dir_block(bs, csum_tail)));
            self.track_dir_block(blk, ino);
        }

        self.push_inode(ino, inode);
        self.groups[0].desc.used_dirs_count += 1;

        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino, constants::DENT_DIR)?;
        self.patch_inode(dev, parent_ino, |i| i.links_count += 1)?;
        Ok(ino)
    }

    /// Resolve a name to the logical block index of its HTree leaf.
    /// Walks dx_root → (dx_node)* → leaf, picking the rightmost
    /// dx_entry whose hash is ≤ the target hash at each level (with
    /// the countlimit slot serving as the implicit "leftmost"
    /// catch-all for hashes that precede the first real boundary).
    fn dx_route_logical_leaf(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_inode: u32,
        name: &[u8],
    ) -> Result<u32> {
        let inode_copy = self
            .inodes
            .iter()
            .find(|(i, _)| *i == dir_inode)
            .map(|(_, i)| *i)
            .unwrap();
        // Read dx_root from logical block 0.
        let dx_root_blk = self.file_block(dev, &inode_copy, 0)?;
        self.ensure_block_staged(dev, dx_root_blk)?;
        let root_buf = self
            .data_blocks
            .iter()
            .find(|(b, _)| *b == dx_root_blk)
            .map(|(_, bytes)| bytes.clone())
            .unwrap();
        // dx_root_info.indirect_levels lives at offset 30.
        let indirect_levels = root_buf[30];
        let (hash, _minor) = htree::half_md4_hash(name);

        // Walk dx_root's dx_entry table to pick the child (leaf or
        // dx_node, depending on indirect_levels).
        let next_logical = dx_lookup_logical(&root_buf, htree::DX_ROOT_HEADER_LEN, hash);

        if indirect_levels == 0 {
            return Ok(next_logical);
        }
        if indirect_levels != 1 {
            return Err(crate::Error::Unsupported(format!(
                "ext4: HTree indirect_levels={indirect_levels} not supported (writer caps at 1)"
            )));
        }
        // Depth-1: next_logical is the logical block of a dx_node.
        // Read it and walk its dx_entry table to find the leaf.
        let dx_node_phys = self.file_block(dev, &inode_copy, next_logical)?;
        self.ensure_block_staged(dev, dx_node_phys)?;
        let node_buf = self
            .data_blocks
            .iter()
            .find(|(b, _)| *b == dx_node_phys)
            .map(|(_, bytes)| bytes.clone())
            .unwrap();
        let leaf_logical = dx_lookup_logical(&node_buf, htree::DX_NODE_HEADER_LEN, hash);
        Ok(leaf_logical)
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

        self.push_inode(ino, inode);
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
        self.push_inode(ino, inode);
        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino, ft)?;
        Ok(ino)
    }

    /// Create a hard link to an existing inode: add `name` under
    /// `parent_ino` as a dirent pointing at `target_ino`, and bump
    /// `target_ino`'s `links_count`. No inode is allocated and no data
    /// is copied.
    ///
    /// Refuses to link a directory inode — POSIX disallows it and
    /// e2fsck would flag the result. Refuses targets whose mode is
    /// unset (already-freed or never-initialised inodes).
    pub fn add_link_to(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u32,
        name: &[u8],
        target_ino: u32,
    ) -> Result<()> {
        self.ensure_inode_staged(dev, target_ino)?;
        let target = self
            .inodes
            .iter()
            .find(|(i, _)| *i == target_ino)
            .map(|(_, i)| *i)
            .unwrap();
        let mode_type = target.mode & constants::S_IFMT;
        if mode_type == 0 {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: cannot hardlink to uninitialised inode {target_ino}"
            )));
        }
        if mode_type == constants::S_IFDIR {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: cannot hardlink to directory inode {target_ino} (POSIX disallows)"
            )));
        }
        let file_type = match mode_type {
            constants::S_IFREG => constants::DENT_REG,
            constants::S_IFLNK => constants::DENT_LNK,
            constants::S_IFCHR => constants::DENT_CHR,
            constants::S_IFBLK => constants::DENT_BLK,
            constants::S_IFIFO => constants::DENT_FIFO,
            constants::S_IFSOCK => constants::DENT_SOCK,
            _ => 0,
        };
        self.patch_inode(dev, target_ino, |i| {
            i.links_count = i.links_count.saturating_add(1);
        })?;
        self.add_entry_to_dir_block_for(dev, parent_ino, name, target_ino, file_type)?;
        Ok(())
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

        // Unlink the dirent from the parent first; this is the
        // operation that's visible to other observers. Inode-side
        // cleanup follows.
        self.unlink_dir_entry(dev, parent_ino, name.as_bytes())?;

        if is_dir {
            // Removing a directory always frees its inode (POSIX dirs
            // can't be hardlinked outside ./.. so links_count is
            // always exactly 2 here). The parent's links_count drops
            // by 1 because the gone dir's ".." was a link back.
            self.free_inode_blocks(dev, &target)?;
            self.free_inode(target_ino);
            self.inodes.retain(|(i, _)| *i != target_ino);
            // `retain` shifted positions; rebuild before re-staging.
            self.rebuild_inode_idx();
            self.push_inode(target_ino, Inode::default());
            self.patch_inode(dev, parent_ino, |i| {
                i.links_count = i.links_count.saturating_sub(1);
            })?;
            self.groups[0].desc.used_dirs_count =
                self.groups[0].desc.used_dirs_count.saturating_sub(1);
            return Ok(());
        }

        // Non-dir: hardlink-aware. Decrement links_count; only free
        // the inode and its data blocks when the last link is gone.
        if target.links_count > 1 {
            self.patch_inode(dev, target_ino, |i| {
                i.links_count = i.links_count.saturating_sub(1);
            })?;
        } else {
            self.free_inode_blocks(dev, &target)?;
            self.free_inode(target_ino);
            self.inodes.retain(|(i, _)| *i != target_ino);
            // `retain` shifted positions; rebuild before re-staging.
            self.rebuild_inode_idx();
            self.push_inode(target_ino, Inode::default());
        }
        Ok(())
    }

    /// Change the permission bits (low 12 bits of `i_mode`) of an
    /// existing inode. Preserves the file-type bits (`S_IFMT`).
    /// POSIX `chmod`.
    pub fn chmod(&mut self, dev: &mut dyn BlockDevice, ino: u32, mode_perms: u16) -> Result<()> {
        let new_perms = mode_perms & 0o7777;
        self.patch_inode(dev, ino, |i| {
            i.mode = (i.mode & constants::S_IFMT) | new_perms;
        })
    }

    /// Change the ownership (uid/gid) of an existing inode. POSIX
    /// `chown`. Values are truncated to 16 bits — the high halves
    /// would live in `osd2.l_i_uid_high` / `osd2.l_i_gid_high` but
    /// the v1 inode encoder doesn't surface them yet.
    pub fn chown(&mut self, dev: &mut dyn BlockDevice, ino: u32, uid: u32, gid: u32) -> Result<()> {
        self.patch_inode(dev, ino, |i| {
            i.uid = (uid & 0xffff) as u16;
            i.gid = (gid & 0xffff) as u16;
        })
    }

    /// Stamp atime / mtime / ctime on an existing inode. POSIX
    /// `utimensat`. Each argument is a UNIX timestamp in seconds;
    /// passing `None` leaves that field unchanged. We don't yet
    /// store the nanosecond extension (`i_atime_extra` etc.).
    pub fn set_times(
        &mut self,
        dev: &mut dyn BlockDevice,
        ino: u32,
        atime: Option<u32>,
        mtime: Option<u32>,
        ctime: Option<u32>,
    ) -> Result<()> {
        self.patch_inode(dev, ino, |i| {
            if let Some(a) = atime {
                i.atime = a;
            }
            if let Some(m) = mtime {
                i.mtime = m;
            }
            if let Some(c) = ctime {
                i.ctime = c;
            }
        })
    }

    /// Truncate a regular file to `new_size` bytes. Grow: leaves a
    /// hole — no blocks are allocated until the file is actually
    /// written. Shrink: frees any data block past the new end and
    /// shrinks the inode's block list / extent tree to match.
    /// Only operates on regular files; returns `InvalidArgument` for
    /// dirs, symlinks, devices.
    pub fn truncate(&mut self, dev: &mut dyn BlockDevice, ino: u32, new_size: u64) -> Result<()> {
        let bs = self.layout.block_size;
        let new_blocks_u64 = new_size.div_ceil(bs as u64);
        if new_blocks_u64 > u32::MAX as u64 {
            return Err(crate::Error::Unsupported(format!(
                "ext: truncate target {new_size} bytes needs {new_blocks_u64} blocks (> u32::MAX)"
            )));
        }
        if new_size > u32::MAX as u64 {
            // Truncating an inode past 4 GiB requires LARGE_FILE; stamp
            // the feature on the SB so the result is well-formed.
            self.sb.feature_ro_compat |= constants::feature::RO_COMPAT_LARGE_FILE;
        }
        self.ensure_inode_staged(dev, ino)?;
        let inode = self
            .inodes
            .iter()
            .find(|(i, _)| *i == ino)
            .map(|(_, i)| *i)
            .unwrap();
        let mode_type = inode.mode & constants::S_IFMT;
        if mode_type != constants::S_IFREG {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: truncate target inode {ino} is not a regular file (mode={:#o})",
                inode.mode
            )));
        }
        let old_blocks = inode.file_size().div_ceil(bs as u64) as u32;
        let new_blocks = new_blocks_u64 as u32;
        // Shrink path: free everything past the new end.
        if new_blocks < old_blocks {
            for n in new_blocks..old_blocks {
                let phys = self.file_block(dev, &inode, n)?;
                if phys != 0 {
                    self.free_block(phys);
                }
            }
            // The simplest reliable way to rebuild the block-pointer
            // structure is to gather the surviving block list and
            // re-pack it. For ext4 extent trees this stays inline
            // (≤ 4 leaves typical for small files); for ext2/3 it
            // re-establishes a fresh indirect chain.
            let surviving: Vec<u32> = (0..new_blocks)
                .map(|n| self.file_block(dev, &inode, n).unwrap_or(0))
                .collect();
            // Clear the old block pointers so fill_block_pointers
            // starts from a known state. Preserve EXTENTS_FL — we'll
            // re-stamp it inside fill_block_pointers_extent.
            self.patch_inode(dev, ino, |i| {
                i.block = [0u32; constants::N_BLOCKS];
                i.flags &= !constants::EXT4_EXTENTS_FL;
            })?;
            let mut staged = self
                .inodes
                .iter()
                .find(|(i, _)| *i == ino)
                .map(|(_, i)| *i)
                .unwrap();
            let allocated_meta = if matches!(self.kind, FsKind::Ext4) {
                self.fill_block_pointers_extent(ino, &mut staged, &surviving)?
            } else {
                self.fill_block_pointers_indirect(&mut staged, &surviving)?
            };
            let sectors_per_block = bs / 512;
            let real_blocks: u32 = surviving.iter().filter(|&&b| b != 0).count() as u32;
            self.patch_inode(dev, ino, |i| {
                i.block = staged.block;
                i.flags = staged.flags;
                i.set_file_size(new_size);
                i.blocks_512 = (real_blocks + allocated_meta) * sectors_per_block;
            })?;
        } else {
            // Grow path (or no-op): leave block list alone, just bump
            // size. Subsequent writes will allocate as needed.
            self.patch_inode(dev, ino, |i| {
                i.set_file_size(new_size);
            })?;
        }
        Ok(())
    }

    /// Rename a single entry: remove `old_name` from `old_parent_ino`
    /// and re-add it under `new_name` in `new_parent_ino`, preserving
    /// the target inode (so all hardlinks survive). Cross-directory
    /// moves correctly update the parent's `links_count` when the
    /// target is a directory (its `..` link transfers).
    ///
    /// `new_name` must not already exist in `new_parent_ino`. Posix
    /// `rename` overwrites; we leave that to the caller
    /// (probe-then-remove-then-rename) until we're ready to make
    /// atomic-overwrite work end to end.
    pub fn rename(
        &mut self,
        dev: &mut dyn BlockDevice,
        old_parent_ino: u32,
        old_name: &[u8],
        new_parent_ino: u32,
        new_name: &[u8],
    ) -> Result<()> {
        // Look up the source.
        let entries = self.list_inode(dev, old_parent_ino)?;
        let target = entries
            .iter()
            .find(|e| e.name.as_bytes() == old_name)
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "ext: rename source {:?} not found in dir {old_parent_ino}",
                    String::from_utf8_lossy(old_name)
                ))
            })?;
        let target_ino = target.inode;
        let target_inode = self.read_inode(dev, target_ino)?;
        let is_dir = target_inode.mode & constants::S_IFMT == constants::S_IFDIR;

        // Ensure new_name doesn't already exist in new_parent_ino.
        let dest_entries = self.list_inode(dev, new_parent_ino)?;
        if dest_entries.iter().any(|e| e.name.as_bytes() == new_name) {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: rename target {:?} already exists in dir {new_parent_ino}",
                String::from_utf8_lossy(new_name)
            )));
        }

        let file_type = match target_inode.mode & constants::S_IFMT {
            constants::S_IFREG => constants::DENT_REG,
            constants::S_IFDIR => constants::DENT_DIR,
            constants::S_IFLNK => constants::DENT_LNK,
            constants::S_IFCHR => constants::DENT_CHR,
            constants::S_IFBLK => constants::DENT_BLK,
            constants::S_IFIFO => constants::DENT_FIFO,
            constants::S_IFSOCK => constants::DENT_SOCK,
            _ => 0,
        };

        // Add the new dirent first so a partial-success crash leaves
        // the file findable under SOME name (matches kernel rename
        // semantics — better to have a duplicate than to lose the
        // file). Then drop the old dirent.
        self.add_entry_to_dir_block_for(dev, new_parent_ino, new_name, target_ino, file_type)?;
        self.unlink_dir_entry(dev, old_parent_ino, old_name)?;

        // Cross-directory move of a directory: the target's `..` now
        // points at a different parent. Update old/new parents'
        // links_count and rewrite the moved dir's `..` dirent.
        if is_dir && old_parent_ino != new_parent_ino {
            self.patch_inode(dev, old_parent_ino, |i| {
                i.links_count = i.links_count.saturating_sub(1);
            })?;
            self.patch_inode(dev, new_parent_ino, |i| {
                i.links_count = i.links_count.saturating_add(1);
            })?;
            self.repoint_dotdot(dev, target_ino, new_parent_ino)?;
        }
        Ok(())
    }

    /// Rewrite the `..` dirent of `dir_ino` to point at `new_parent`.
    /// Called by `rename` on a cross-directory move of a directory.
    fn repoint_dotdot(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_ino: u32,
        new_parent: u32,
    ) -> Result<()> {
        self.ensure_inode_staged(dev, dir_ino)?;
        let inode_copy = self
            .inodes
            .iter()
            .find(|(i, _)| *i == dir_ino)
            .map(|(_, i)| *i)
            .unwrap();
        let blk = self.file_block(dev, &inode_copy, 0)?;
        if blk == 0 {
            return Err(crate::Error::InvalidImage(format!(
                "ext: dir inode {dir_ino} has no first data block"
            )));
        }
        self.ensure_block_staged(dev, blk)?;
        if !self.dir_blocks.iter().any(|(b, _)| *b == blk) {
            self.track_dir_block(blk, dir_ino);
        }
        let block = self
            .data_blocks
            .iter_mut()
            .find(|(b, _)| *b == blk)
            .map(|(_, bytes)| bytes)
            .unwrap();
        // "." at offset 0 (rec_len 12). ".." at offset 12: inode in
        // the first 4 bytes.
        block[12..16].copy_from_slice(&new_parent.to_le_bytes());
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
                let bit = blk - g.start_block;
                group::clear_bit(&mut self.groups[gi].block_bitmap, bit);
                // Rewind the per-group allocation cursor so the next
                // `alloc_data_block` can reuse this freed bit. Without
                // this, append-only workloads stay O(1) but anything
                // that frees and re-allocates (the fragmentation tests,
                // mutate paths) skips the holes entirely.
                if bit < self.groups[gi].next_free_block_bit {
                    self.groups[gi].next_free_block_bit = bit;
                }
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
            self.track_dir_block(dir_block_num, dir_inode);
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
                next_free_block_bit: 0,
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
            extent_leaf_blocks: Vec::new(),
            dx_root_blocks: Vec::new(),
            dx_node_blocks: Vec::new(),
            inode_idx: std::collections::HashMap::new(),
            data_block_idx: std::collections::HashMap::new(),
            dir_block_set: std::collections::HashSet::new(),
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
    /// otherwise → direct + single + double + triple indirect, matching
    /// the writer's `fill_block_pointers_indirect`. Returns `0` for a
    /// hole at any level (`mke2fs`-style sparse files).
    pub fn file_block(&self, dev: &mut dyn BlockDevice, ino: &Inode, n: u32) -> Result<u32> {
        if ino.flags & constants::EXT4_EXTENTS_FL != 0 {
            return self.file_block_extent(dev, ino, n);
        }
        if (n as usize) < constants::N_DIRECT {
            return Ok(ino.block[n as usize]);
        }
        let ptrs = self.layout.block_size / 4;
        let mut n_off = n - constants::N_DIRECT as u32;
        let bs = self.layout.block_size as usize;
        let mut buf = vec![0u8; bs];
        let read_ptr = |this: &Self,
                        dev: &mut dyn BlockDevice,
                        blk: u32,
                        idx: u32,
                        buf: &mut [u8]|
         -> Result<u32> {
            this.read_block(dev, blk, buf)?;
            let off = (idx as usize) * 4;
            Ok(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()))
        };
        // Single-indirect.
        if n_off < ptrs {
            let ind = ino.block[constants::IDX_INDIRECT];
            if ind == 0 {
                return Ok(0);
            }
            return read_ptr(self, dev, ind, n_off, &mut buf);
        }
        n_off -= ptrs;
        // Double-indirect.
        if n_off < ptrs * ptrs {
            let dind = ino.block[constants::IDX_DOUBLE_INDIRECT];
            if dind == 0 {
                return Ok(0);
            }
            let sub = read_ptr(self, dev, dind, n_off / ptrs, &mut buf)?;
            if sub == 0 {
                return Ok(0);
            }
            return read_ptr(self, dev, sub, n_off % ptrs, &mut buf);
        }
        n_off -= ptrs * ptrs;
        // Triple-indirect.
        if n_off < ptrs * ptrs * ptrs {
            let tind = ino.block[constants::IDX_TRIPLE_INDIRECT];
            if tind == 0 {
                return Ok(0);
            }
            let dind = read_ptr(self, dev, tind, n_off / (ptrs * ptrs), &mut buf)?;
            if dind == 0 {
                return Ok(0);
            }
            let rem = n_off % (ptrs * ptrs);
            let sub = read_ptr(self, dev, dind, rem / ptrs, &mut buf)?;
            if sub == 0 {
                return Ok(0);
            }
            return read_ptr(self, dev, sub, rem % ptrs, &mut buf);
        }
        Err(crate::Error::InvalidImage(format!(
            "ext: logical block {n} exceeds triple-indirect range"
        )))
    }

    /// Resolve logical block `n` against an inode that uses an ext4
    /// extent tree of any depth. Depth-0 reads the inline leaf extents;
    /// deeper trees descend index level by index level — at each level
    /// picking the last idx entry whose `ei_block <= n` — until a depth-0
    /// leaf block is reached, then resolves `n` within its extents.
    #[allow(clippy::needless_pass_by_ref_mut)]
    fn file_block_extent(&self, dev: &mut dyn BlockDevice, ino: &Inode, n: u32) -> Result<u32> {
        let iblock = extent::iblock_to_bytes(&ino.block);
        let header = extent::decode_header(&iblock[..12])?;
        if header.depth == 0 {
            let (_, runs) = extent::decode_depth0_iblock(&iblock)?;
            return Ok(resolve_logical_in_runs(&runs, n));
        }
        // Pick the child subtree from the inline (`i_block`) index node.
        let (_, indices) = extent::decode_idx_iblock(&iblock)?;
        let Some(mut child) = pick_idx_for_logical(&indices, n) else {
            return Ok(0);
        };

        // Descend through any further index levels, then the leaf.
        let bs = self.layout.block_size as usize;
        let mut buf = vec![0u8; bs];
        loop {
            self.read_block(dev, child, &mut buf)?;
            let h = extent::decode_header(&buf[..12])?;
            if h.depth == 0 {
                let (_, runs) = extent::decode_leaf_block(&buf)?;
                return Ok(resolve_logical_in_runs(&runs, n));
            }
            // Internal index block: parse its idx entries and pick the
            // subtree covering `n`.
            let mut chosen: Option<u32> = None;
            for i in 0..h.entries as usize {
                let off = 12 + i * 12;
                let idx = extent::decode_idx(&buf[off..off + 12]);
                if idx.block <= n {
                    chosen = Some(idx.leaf as u32);
                } else {
                    break;
                }
            }
            match chosen {
                Some(c) => child = c,
                None => return Ok(0),
            }
        }
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
        let total = self.inode.file_size();
        if self.pos >= total {
            return Ok(0);
        }
        // Inline-data fast path: the file's body lives inside i_block
        // (the 60-byte block-pointer array). No data block to walk.
        if self.inode.flags & constants::EXT4_INLINE_DATA_FL != 0 {
            let inline_bytes = extent::iblock_to_bytes(&self.inode.block);
            let remaining_in_file = (total - self.pos) as usize;
            let n = out.len().min(remaining_in_file);
            out[..n].copy_from_slice(&inline_bytes[self.pos as usize..self.pos as usize + n]);
            self.pos += n as u64;
            return Ok(n);
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
        let total = self.inode.file_size() as i128;
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

    fn create_file_streaming(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        body: &mut dyn std::io::Read,
        len: u64,
        meta: FileMeta,
    ) -> Result<()> {
        let (parent, name) = split_path(path)?;
        let parent_str = parent
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 parent path".into()))?;
        let parent_ino = self.path_to_inode(dev, parent_str)?;
        self.add_file_to_streaming(dev, parent_ino, name.as_bytes(), body, len, meta)?;
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

    fn getattr(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<crate::fs::FileAttrs> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 path".into()))?;
        let ino = self.path_to_inode(dev, s)?;
        let inode = self.read_inode(dev, ino)?;
        let kind = kind_from_mode(inode.mode);
        // Device numbers live in i_block[0] when the inode is a
        // char/block device (encoded by `add_device_to`).
        let rdev = if matches!(
            kind,
            crate::fs::EntryKind::Char | crate::fs::EntryKind::Block
        ) {
            inode.block[0]
        } else {
            0
        };
        // For regular files past 4 GiB the upper 32 bits of size live
        // in i_size_high (`size_hi_or_dir_acl` for regular files);
        // `file_size()` is a no-op for the others (their size_hi is 0
        // or unused as `i_dir_acl`).
        let size = if matches!(kind, crate::fs::EntryKind::Regular) {
            inode.file_size()
        } else {
            inode.size as u64
        };
        Ok(crate::fs::FileAttrs {
            kind,
            mode: inode.mode & 0o7777,
            uid: inode.uid as u32,
            gid: inode.gid as u32,
            size,
            blocks: inode.blocks_512 as u64,
            nlink: inode.links_count as u32,
            atime: inode.atime,
            mtime: inode.mtime,
            ctime: inode.ctime,
            rdev,
            inode: ino,
        })
    }

    fn set_attrs(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        attrs: crate::fs::SetAttrs,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 path".into()))?;
        let ino = self.path_to_inode(dev, s)?;
        if let Some(m) = attrs.mode {
            self.chmod(dev, ino, m)?;
        }
        if attrs.uid.is_some() || attrs.gid.is_some() {
            let cur = self.read_inode(dev, ino)?;
            let new_uid = attrs.uid.unwrap_or(cur.uid as u32);
            let new_gid = attrs.gid.unwrap_or(cur.gid as u32);
            self.chown(dev, ino, new_uid, new_gid)?;
        }
        if attrs.atime.is_some() || attrs.mtime.is_some() || attrs.ctime.is_some() {
            self.set_times(dev, ino, attrs.atime, attrs.mtime, attrs.ctime)?;
        }
        Ok(())
    }

    fn truncate(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        new_size: u64,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 path".into()))?;
        let ino = self.path_to_inode(dev, s)?;
        Self::truncate(self, dev, ino, new_size)
    }

    fn rename(
        &mut self,
        dev: &mut dyn BlockDevice,
        old_path: &std::path::Path,
        new_path: &std::path::Path,
    ) -> Result<()> {
        let (op, on) = split_path(old_path)?;
        let (np, nn) = split_path(new_path)?;
        let op_s = op
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 old parent".into()))?;
        let np_s = np
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 new parent".into()))?;
        let op_ino = self.path_to_inode(dev, op_s)?;
        let np_ino = self.path_to_inode(dev, np_s)?;
        Self::rename(self, dev, op_ino, on.as_bytes(), np_ino, nn.as_bytes())
    }

    fn hardlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        target_path: &std::path::Path,
        new_path: &std::path::Path,
    ) -> Result<()> {
        let target_s = target_path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 target path".into()))?;
        let target_ino = self.path_to_inode(dev, target_s)?;
        let (np, nn) = split_path(new_path)?;
        let np_s = np
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 new parent".into()))?;
        let np_ino = self.path_to_inode(dev, np_s)?;
        self.add_link_to(dev, np_ino, nn.as_bytes(), target_ino)
    }

    fn list_xattrs(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::XattrPair>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 path".into()))?;
        let ino = self.path_to_inode(dev, s)?;
        let xattrs = self.read_xattrs(dev, ino)?;
        Ok(xattrs
            .into_iter()
            .map(|x| crate::fs::XattrPair {
                name: x.name,
                value: x.value,
            })
            .collect())
    }

    fn set_xattrs(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        xattrs: &[crate::fs::XattrPair],
    ) -> Result<()> {
        if xattrs.is_empty() {
            return Ok(());
        }
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ext: non-UTF-8 path".into()))?;
        let ino = self.path_to_inode(dev, s)?;
        let converted: Vec<xattr::Xattr> = xattrs
            .iter()
            .map(|x| xattr::Xattr {
                name: x.name.clone(),
                value: x.value.clone(),
            })
            .collect();
        // Inherent batch writer (one external block) — resolved over the
        // trait method by argument types.
        self.set_xattrs(dev, ino, &converted)
    }

    fn statfs(&mut self, _dev: &mut dyn BlockDevice) -> Result<crate::fs::StatFs> {
        let sb = &self.sb;
        Ok(crate::fs::StatFs {
            block_size: self.layout.block_size,
            blocks: sb.blocks_count as u64,
            blocks_free: sb.free_blocks_count as u64,
            blocks_avail: sb.free_blocks_count as u64,
            inodes: sb.inodes_count as u64,
            inodes_free: sb.free_inodes_count as u64,
            name_max: 255,
        })
    }
}

/// Scan a leaf-extent list for the run containing logical block `n` and
/// return the corresponding physical block. Returns 0 if `n` falls in a
/// hole (no extent covers it).
/// Pick the child physical block for logical block `n` from a sorted
/// (ascending `ei_block`) index array: the last entry whose `block <= n`.
/// `None` when `n` precedes the first entry (a hole before any extent).
fn pick_idx_for_logical(indices: &[extent::ExtentIdx], n: u32) -> Option<u32> {
    let mut chosen = None;
    for idx in indices {
        if idx.block <= n {
            chosen = Some(idx.leaf as u32);
        } else {
            break;
        }
    }
    chosen
}

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
/// tail.
///
/// `usable` is the byte length available for real entries — the whole
/// block, or `block_size - 12` when `metadata_csum` reserves a checksum
/// tail. The last real entry's `rec_len` always runs up to `usable`.
///
/// Returns `Ok(true)` when the entry fits and is written, `Ok(false)`
/// when the block has insufficient trailing slack (caller should grow the
/// directory by one block and retry), and `Err` only on a corrupt block.
fn try_append_dir_entry(
    block: &mut [u8],
    name: &[u8],
    inode: u32,
    file_type: u8,
    with_filetype: bool,
    usable: usize,
) -> Result<bool> {
    let needed = dir::min_rec_len(name.len());

    // Fresh-block fast path: when the sole entry is the empty placeholder
    // produced by `make_empty_dir_block` (inode=0, name_len=0, rec_len
    // spanning the usable region), overwrite it entirely with the new
    // entry rather than leaving an 8-byte zero stub at offset 0. e2fsck
    // accepts blocks that are *entirely* empty (single placeholder) and
    // blocks whose first entry is a real one, but flags "placeholder stub
    // followed by real entries" as a corrupted block.
    if let Some(first) = dir::decode_entry(block, with_filetype) {
        if first.inode == 0 && first.name.is_empty() && first.rec_len >= usable && needed <= usable
        {
            // Wipe the usable region (the csum tail past `usable` is
            // untouched), then encode the single new entry to span it.
            for b in block[..usable].iter_mut() {
                *b = 0;
            }
            let mut tail = Vec::with_capacity(usable);
            dir::encode_entry(
                &mut tail,
                inode,
                name,
                usable as u16,
                file_type,
                with_filetype,
            );
            debug_assert_eq!(tail.len(), usable);
            block[..usable].copy_from_slice(&tail);
            return Ok(true);
        }
    }

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
        return Ok(false);
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
    Ok(true)
}

fn popcount_bits(bm: &[u8], start: u32, end: u32) -> u32 {
    (start..end).filter(|&i| test_bit(bm, i)).count() as u32
}

/// Walk a dx_root or dx_node's dx_entry table to find the child block
/// whose hash range covers `target`. `header_len` is the byte offset
/// where the dx_entry table starts (32 for dx_root, 12 for dx_node).
/// Slot 0 is the countlimit (high half of its hash field is `count`,
/// low half is `limit`); slots 1..count carry real `(hash, block)`
/// rows sorted by ascending hash, and the rightmost slot whose hash
/// ≤ target wins. The countlimit slot's `block` field is the
/// catch-all for hashes preceding any real boundary.
fn dx_lookup_logical(buf: &[u8], header_len: usize, target: u32) -> u32 {
    let cl_hash = u32::from_le_bytes(buf[header_len..header_len + 4].try_into().unwrap());
    let count = (cl_hash >> 16) as usize;
    let mut chosen_slot = 0usize;
    for slot in 1..count {
        let off = header_len + slot * 8;
        let slot_hash = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        if slot_hash <= target {
            chosen_slot = slot;
        } else {
            break;
        }
    }
    let block_off = header_len + chosen_slot * 8 + 4;
    u32::from_le_bytes(buf[block_off..block_off + 4].try_into().unwrap())
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

    /// FUSE-style write: each kernel WRITE issues a fresh
    /// `open_file_rw`, seeks to the kernel-supplied offset, writes
    /// the chunk, drops the handle. A single user-space
    /// `write_all(1952 bytes)` can fragment into multiple
    /// chunks — verify that the file ends up with the full 1952
    /// bytes across re-opens.
    #[test]
    fn open_file_rw_multi_open_write_extends_ext2() {
        let (mut ext, mut dev) = ext2_with_file(b"_unused.bin", b"x");
        // Create empty file via add_file_to_streaming (mirrors
        // adapter::create's FileSource::Zero(0) path).
        ext.add_file_to_streaming(
            &mut dev,
            constants::INO_ROOT_DIR,
            b"chunked.bin",
            &mut std::io::Cursor::new(Vec::<u8>::new()),
            0,
            FileMeta::with_mode(0o644),
        )
        .expect("create empty");

        // Two separate open_file_rw cycles writing 1024 + 928 bytes.
        let chunk_a = vec![b'A'; 1024];
        let chunk_b = vec![b'B'; 928];
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/chunked.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open 1");
            h.seek(SeekFrom::Start(0)).unwrap();
            h.write_all(&chunk_a).unwrap();
            // No sync — drop the handle while the inode update is
            // only staged, like the FUSE adapter does between
            // FUSE_WRITE calls.
        }
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/chunked.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open 2");
            h.seek(SeekFrom::Start(1024)).unwrap();
            h.write_all(&chunk_b).unwrap();
        }

        // Read through the SAME Ext instance (no reopen) — this
        // is the path adapter::read takes after adapter::write.
        let mut expected = Vec::with_capacity(1952);
        expected.extend_from_slice(&chunk_a);
        expected.extend_from_slice(&chunk_b);
        let got = read_full_via_handle(&mut ext, &mut dev, "/chunked.bin");
        assert_eq!(
            got.len(),
            1952,
            "size after two open_file_rw cycles should be 1952, got {}",
            got.len()
        );
        assert_eq!(got, expected, "content mismatch after two-stage write");
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

    /// Build an ext4 file whose extent tree spills all the way into
    /// depth-2 through the in-place `open_file_rw` path, then reopen and
    /// confirm every marked offset round-trips. With 1 KiB blocks a leaf
    /// holds ~84 extents, so depth-1 caps at 4 × 84 = 336; ~400 sparse
    /// single-block writes overflows that and forces a second idx level.
    #[test]
    fn open_file_rw_depth2_extent_round_trip_ext4() {
        let mut dev = MemoryBackend::new(64u64 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 1024,
            blocks_count: 32 * 1024,
            inodes_count: 1024,
            sparse_super: true,
            ..FormatOpts::default()
        };
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext4");
        ext.add_file_to_streaming(
            &mut dev,
            constants::INO_ROOT_DIR,
            b"deep2.bin",
            &mut std::io::Cursor::new(Vec::<u8>::new()),
            0,
            FileMeta::default(),
        )
        .expect("add empty file");
        ext.flush(&mut dev).expect("flush");

        // 400 logically-discontiguous single-block writes. An 8-block gap
        // between each keeps the runs from coalescing, so the leaf count
        // crosses the depth-1 ceiling.
        let n = 400u64;
        let gap = 1024 * 8;
        let mark = b"D2!";
        {
            let mut h = ext
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/deep2.bin"),
                    OpenFlags::default(),
                    None,
                )
                .expect("open_file_rw on empty extent file");
            for i in 0..n {
                h.seek(SeekFrom::Start(i * gap)).unwrap();
                h.write_all(mark).unwrap();
            }
            h.sync().expect("sync");
        }

        // The on-disk inode must now carry a depth-2 extent tree.
        let ino = ext.path_to_inode(&mut dev, "/deep2.bin").expect("ino");
        let inode = ext.read_inode(&mut dev, ino).expect("read inode");
        let iblock = extent::iblock_to_bytes(&inode.block);
        let header = extent::decode_header(&iblock[..12]).expect("header");
        assert_eq!(
            header.depth, 2,
            "expected depth-2 extent tree, got depth {}",
            header.depth
        );

        // Reopen from scratch and verify every marker survives, with an
        // untouched block reading back as a hole.
        let mut reopened = Ext::open(&mut dev).expect("reopen");
        let mut h = reopened
            .open_file_rw(
                &mut dev,
                std::path::Path::new("/deep2.bin"),
                OpenFlags::default(),
                None,
            )
            .expect("reopen rw on depth-2 file");
        for i in 0..n {
            h.seek(SeekFrom::Start(i * gap)).unwrap();
            let mut buf = vec![0u8; mark.len()];
            h.read_exact(&mut buf).unwrap();
            assert_eq!(&buf[..], mark, "marker mismatch at index {i}");
        }
        h.seek(SeekFrom::Start(gap - 1024)).unwrap();
        let mut zero = vec![0u8; 1024];
        h.read_exact(&mut zero).unwrap();
        assert!(
            zero.iter().all(|&b| b == 0),
            "untouched region must be zero"
        );
    }

    /// Drive the *streaming* incremental-append path
    /// ([`Ext::append_data_block_to_inode`], the route directory growth
    /// takes) hard enough to promote an extent tree all the way to depth-3,
    /// exercising every new branch in [`Ext::append_extent_deep`]: leaf
    /// split, index-node split (a new sibling bubbling up), and inline-root
    /// promotion. With 1 KiB blocks a node holds ~84 entries, so depth-2
    /// saturates at 4 × 84 × 84 = 28 224 extents; appending past that forces
    /// depth-3. We then walk the tree back and confirm every extent — in
    /// logical order — survived the rebuilds.
    #[test]
    fn append_extent_deep_streaming_promotes_to_depth3() {
        let mut dev = MemoryBackend::new(96u64 * 1024 * 1024);
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 1024,
            blocks_count: 80 * 1024,
            inodes_count: 1024,
            ..FormatOpts::default()
        };
        let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext4");
        // A directory is the only inode that grows via the incremental
        // append path; it starts depth-0 with one block (logical 0).
        let dir = ext
            .add_dir_to(
                &mut dev,
                constants::INO_ROOT_DIR,
                b"d",
                FileMeta::with_mode(0o755),
            )
            .expect("mkdir");

        // Append sparse logical blocks so none coalesce; 28 225+ extents
        // (plus the initial one) tip the tree into depth-3.
        let n: u32 = 29_000;
        let mut expect: Vec<(u32, u64)> = Vec::with_capacity(n as usize);
        for i in 0..n {
            let logical = 2 + i * 2; // gap of 1 between each → distinct runs
            let phys = ext.alloc_data_block().expect("alloc data block");
            ext.append_data_block_to_inode(&mut dev, dir, logical, phys)
                .expect("append data block");
            expect.push((logical, phys as u64));
        }

        // Inline root must now be depth-3.
        let inode = ext
            .inodes
            .iter()
            .find(|(i, _)| *i == dir)
            .map(|(_, i)| *i)
            .expect("dir staged");
        let iblock = extent::iblock_to_bytes(&inode.block);
        let header = extent::decode_header(&iblock[..12]).expect("header");
        assert_eq!(
            header.depth, 3,
            "expected depth-3 extent tree, got depth {}",
            header.depth
        );

        // Walk the whole tree from the staged blocks and collect leaf runs.
        fn walk(
            ext: &mut Ext,
            dev: &mut dyn BlockDevice,
            phys: u32,
            out: &mut Vec<extent::ExtentRun>,
        ) {
            let buf = ext.staged_block_bytes(dev, phys).expect("staged block");
            let bs = ext.layout.block_size as usize;
            let hdr = extent::decode_header(&buf[..12]).expect("node header");
            if hdr.depth == 0 {
                let (_, mut runs) = extent::decode_leaf_block(&buf[..bs]).expect("leaf");
                out.append(&mut runs);
            } else {
                for i in 0..hdr.entries as usize {
                    let off = 12 + i * 12;
                    let idx = extent::decode_idx(&buf[off..off + 12]);
                    walk(ext, dev, idx.leaf as u32, out);
                }
            }
        }
        let (_, top) = extent::decode_idx_iblock(&iblock).expect("root idx");
        let mut runs = Vec::new();
        for idx in &top {
            walk(&mut ext, &mut dev, idx.leaf as u32, &mut runs);
        }

        // The initial dir block (logical 0) plus our n appends, all in
        // ascending logical order, every physical block intact.
        assert_eq!(runs.len(), n as usize + 1, "extent count mismatch");
        assert_eq!(runs[0].logical, 0, "initial dir block missing");
        for (run, (logical, phys)) in runs[1..].iter().zip(expect.iter()) {
            assert_eq!(run.logical, *logical, "logical mismatch");
            assert_eq!(run.len, 1, "each sparse append is a single block");
            assert_eq!(run.physical, *phys, "physical mismatch at {logical}");
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
