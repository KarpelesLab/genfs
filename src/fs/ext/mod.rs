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
pub mod dir;
pub mod extent;
pub mod group;
pub mod inode;
pub mod layout;
pub mod superblock;

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
        }
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
    groups: Vec<GroupState>,
    /// Next free inode number to hand out (starts at first_ino).
    next_inode: u32,
    /// Inodes allocated so far during this build session. Written to the
    /// on-disk inode table during [`flush_metadata`].
    inodes: Vec<(u32, Inode)>,
    /// Data blocks staged for write (typically directory data blocks and
    /// indirect-block tables, which we assemble in memory). Regular file
    /// data is NOT staged here — it streams straight to the device.
    data_blocks: Vec<(u32, Vec<u8>)>,
}

impl Ext {
    /// Format the device, returning an `Ext` handle. At return the on-disk
    /// image is a valid ext2 containing just the root directory (and
    /// `/lost+found` if requested in `opts`).
    pub fn format_with(dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<Self> {
        let layout = layout::plan(opts.block_size, opts.blocks_count, opts.inodes_count)?;
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
            groups,
            next_inode: 0,
            inodes: Vec::new(),
            data_blocks: Vec::new(),
        };

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
            // Set feature flag + journal_inum on the superblock.
            ext.sb.feature_compat |= constants::feature::COMPAT_HAS_JOURNAL;
            ext.sb.journal_inum = constants::INO_JOURNAL;
        }
        // ext4-only feature flags.
        if matches!(opts.kind, FsKind::Ext4) {
            ext.sb.feature_incompat |= constants::feature::INCOMPAT_EXTENTS;
        }

        ext.recompute_free_counts();
        ext.flush_metadata(dev)?;
        Ok(ext)
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
    fn fill_block_pointers_indirect(&mut self, inode: &mut Inode, data: &[u32]) -> Result<u32> {
        let bs = self.layout.block_size;
        let ptrs_per_block = (bs / 4) as usize;
        let n = data.len();
        let n_direct = constants::N_DIRECT.min(n);
        inode.block[..n_direct].copy_from_slice(&data[..n_direct]);
        let mut allocated_meta = 0u32;
        let mut consumed = n_direct;

        if consumed < n {
            // Single-indirect.
            let ind = self.alloc_data_block(0)?;
            allocated_meta += 1;
            inode.block[constants::IDX_INDIRECT] = ind;
            let take = (n - consumed).min(ptrs_per_block);
            let mut buf = vec![0u8; bs as usize];
            for (i, &b) in data[consumed..consumed + take].iter().enumerate() {
                let off = i * 4;
                buf[off..off + 4].copy_from_slice(&b.to_le_bytes());
            }
            self.data_blocks.push((ind, buf));
            consumed += take;
        }

        if consumed < n {
            // Double-indirect.
            let dind = self.alloc_data_block(0)?;
            allocated_meta += 1;
            inode.block[constants::IDX_DOUBLE_INDIRECT] = dind;
            let mut dind_buf = vec![0u8; bs as usize];
            let mut dind_slot = 0;
            while consumed < n {
                if dind_slot >= ptrs_per_block {
                    return Err(crate::Error::Unsupported(
                        "ext: file exceeds direct+single+double indirection capacity".into(),
                    ));
                }
                let ind = self.alloc_data_block(0)?;
                allocated_meta += 1;
                let off = dind_slot * 4;
                dind_buf[off..off + 4].copy_from_slice(&ind.to_le_bytes());
                let take = (n - consumed).min(ptrs_per_block);
                let mut ind_buf = vec![0u8; bs as usize];
                for (i, &b) in data[consumed..consumed + take].iter().enumerate() {
                    let off = i * 4;
                    ind_buf[off..off + 4].copy_from_slice(&b.to_le_bytes());
                }
                self.data_blocks.push((ind, ind_buf));
                consumed += take;
                dind_slot += 1;
            }
            self.data_blocks.push((dind, dind_buf));
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
            data.push(self.alloc_data_block(0)?);
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

        let blk = self.alloc_data_block(0)?;
        let block_bytes = dir::make_initial_dir_block(ino, ino, self.layout.block_size, false);

        let mut inode = Inode::directory(self.layout.block_size, 0o755, 0, 0, mtime);
        inode.block[0] = blk;
        inode.blocks_512 = self.layout.block_size / 512;

        self.groups[0].desc.used_dirs_count += 1;
        self.inodes.push((ino, inode));
        self.data_blocks.push((blk, block_bytes));
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
            data_blocks.push(self.alloc_data_block(0)?);
        }

        let mut inode = Inode::directory(16384, 0o700, 0, 0, mtime);
        let meta_blocks = self.fill_block_pointers(&mut inode, &data_blocks)?;
        inode.blocks_512 = (target_data_blocks + meta_blocks) * (bs / 512);

        // First data block: "." / "..".
        let dir_block = dir::make_initial_dir_block(ino, INO_ROOT_DIR, bs, false);
        self.data_blocks.push((data_blocks[0], dir_block));
        // All trailing data blocks: empty-placeholder entry so e2fsck reads
        // them as well-formed empty dir blocks.
        for &blk in &data_blocks[1..] {
            self.data_blocks.push((blk, dir::make_empty_dir_block(bs)));
        }

        self.groups[0].desc.used_dirs_count += 1;
        self.inodes.push((ino, inode));

        // Add to root dir + bump root's link count (a new subdir's ".." is
        // a fresh link to the parent).
        self.add_entry_to_dir_block_for(dev, INO_ROOT_DIR, b"lost+found", ino)?;
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
        let block = self
            .data_blocks
            .iter_mut()
            .find(|(b, _)| *b == dir_block_num)
            .map(|(_, bytes)| bytes)
            .unwrap();
        append_dir_entry(block, name, child_ino, constants::DENT_DIR, false)
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
    fn ensure_inode_staged(&mut self, dev: &mut dyn BlockDevice, ino: u32) -> Result<()> {
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

    /// Reserve the next available inode. Currently only allocates from
    /// group 0; multi-group allocation lands once we have tests for it.
    fn alloc_inode(&mut self) -> Result<u32> {
        if self.next_inode > self.layout.inodes_count {
            return Err(crate::Error::Unsupported(format!(
                "ext: out of inodes (allocated {}, max {})",
                self.next_inode - 1,
                self.layout.inodes_count
            )));
        }
        if self.next_inode > self.layout.inodes_per_group {
            return Err(crate::Error::Unsupported(
                "ext: inode allocation past group 0 not yet implemented".into(),
            ));
        }
        let ino = self.next_inode;
        set_bit(&mut self.groups[0].inode_bitmap, ino - 1);
        self.next_inode += 1;
        Ok(ino)
    }

    /// Allocate a single data block in `group`. Returns its absolute block
    /// number.
    fn alloc_data_block(&mut self, group: usize) -> Result<u32> {
        let layout_g = &self.layout.groups[group];
        let state = &mut self.groups[group];
        let start_rel = layout_g.data_start - layout_g.start_block;
        let group_blocks = layout_g.end_block - layout_g.start_block + 1;
        for bit in start_rel..group_blocks {
            if !test_bit(&state.block_bitmap, bit) {
                set_bit(&mut state.block_bitmap, bit);
                return Ok(layout_g.start_block + bit);
            }
        }
        Err(crate::Error::Unsupported(format!(
            "ext: group {group} has no free data blocks"
        )))
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
    /// invariant.
    fn flush_metadata(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        let bs = self.layout.block_size as u64;

        // Build encoded GDT (same for every group copy).
        let mut gdt = vec![0u8; self.layout.gdt_blocks as usize * bs as usize];
        for (i, g) in self.groups.iter().enumerate() {
            let off = i * constants::GROUP_DESC_SIZE;
            gdt[off..off + constants::GROUP_DESC_SIZE].copy_from_slice(&g.desc.encode());
        }

        // For each group with a superblock copy, write SB (skip primary for
        // last step), GDT, bitmaps.
        for (i, g) in self.layout.groups.iter().enumerate() {
            if !g.has_superblock {
                continue;
            }
            if i != 0 {
                let mut sb_copy = self.sb.clone();
                sb_copy.block_group_nr = i as u16;
                dev.write_at(g.start_block as u64 * bs, &sb_copy.encode())?;
            }
            // GDT block(s):
            // - group 0 (first_data_block=1): GDT at block 2 (after SB at 1)
            // - group 0 (first_data_block=0): GDT at block 1 (SB shares block 0)
            // - other groups: GDT at start_block + 1 (after the SB copy)
            let gdt_off = if i == 0 {
                if self.layout.first_data_block == 1 {
                    2 * bs
                } else {
                    bs
                }
            } else {
                (g.start_block as u64 + 1) * bs
            };
            dev.write_at(gdt_off, &gdt)?;

            dev.write_at(g.block_bitmap as u64 * bs, &self.groups[i].block_bitmap)?;
            dev.write_at(g.inode_bitmap as u64 * bs, &self.groups[i].inode_bitmap)?;
        }

        // Write all staged inodes into their slots in the inode table.
        for (ino, inode) in &self.inodes {
            let (group, idx_in_group) = self.inode_location(*ino);
            let table_block = self.layout.groups[group as usize].inode_table;
            let off = table_block as u64 * bs + idx_in_group as u64 * self.layout.inode_size as u64;
            dev.write_at(off, &inode.encode())?;
        }

        // Write staged data blocks.
        for (blk, bytes) in &self.data_blocks {
            dev.write_at(*blk as u64 * bs, bytes)?;
        }

        // Primary SB last.
        dev.write_at(SUPERBLOCK_OFFSET, &self.sb.encode())?;
        dev.sync()?;
        Ok(())
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
        let bs = self.layout.block_size;
        let len = src.len()?;
        if len > u32::MAX as u64 {
            return Err(crate::Error::Unsupported(
                "ext: file > 4 GiB requires LARGE_FILE (deferred to ext4)".into(),
            ));
        }
        let n_data_blocks = len.div_ceil(bs as u64) as u32;
        let mut data_blocks = Vec::with_capacity(n_data_blocks as usize);
        for _ in 0..n_data_blocks {
            data_blocks.push(self.alloc_data_block(0)?);
        }

        let ino = self.alloc_inode()?;
        let mut inode = Inode::regular(
            len as u32,
            meta.mode & 0o7777,
            meta.uid,
            meta.gid,
            meta.mtime,
        );
        let allocated_meta_blocks = self.fill_block_pointers(&mut inode, &data_blocks)?;
        inode.blocks_512 = (n_data_blocks + allocated_meta_blocks) * (bs / 512);

        // Stream data straight to device.
        let (mut reader, _) = src.open()?;
        let mut buf = vec![0u8; bs as usize];
        let mut remaining = len;
        for &blk in &data_blocks {
            let to_read = remaining.min(bs as u64) as usize;
            reader.read_exact(&mut buf[..to_read])?;
            dev.write_at(blk as u64 * bs as u64, &buf[..to_read])?;
            remaining -= to_read as u64;
        }
        debug_assert_eq!(remaining, 0);

        self.inodes.push((ino, inode));
        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino)?;
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
        let blk = self.alloc_data_block(0)?;
        let mut inode = Inode::directory(bs, meta.mode & 0o7777, meta.uid, meta.gid, meta.mtime);
        // For ext4, encode the single data block as an inline extent tree;
        // for ext2/3, store the block number directly in i_block[0].
        if matches!(self.kind, FsKind::Ext4) {
            self.fill_block_pointers_extent(&mut inode, &[blk])?;
        } else {
            inode.block[0] = blk;
        }
        inode.blocks_512 = bs / 512;
        let block_bytes = dir::make_initial_dir_block(ino, parent_ino, bs, false);
        self.data_blocks.push((blk, block_bytes));
        self.inodes.push((ino, inode));
        self.groups[0].desc.used_dirs_count += 1;

        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino)?;
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
            let blk = self.alloc_data_block(0)?;
            inode.block[0] = blk;
            inode.blocks_512 = bs / 512;
            let mut buf = vec![0u8; bs as usize];
            buf[..target.len()].copy_from_slice(target);
            dev.write_at(blk as u64 * bs as u64, &buf)?;
        }

        self.inodes.push((ino, inode));
        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino)?;
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
        self.inodes.push((ino, inode));
        self.add_entry_to_dir_block_for(dev, parent_ino, name, ino)?;
        Ok(ino)
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
        use std::os::unix::fs::{FileTypeExt, PermissionsExt};
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            let ft = meta.file_type();
            let name = entry.file_name();
            let name_bytes = name.as_encoded_bytes();
            let mode = (meta.permissions().mode() & 0o7777) as u16;
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
            } else if ft.is_block_device() || ft.is_char_device() {
                use std::os::unix::fs::MetadataExt;
                let rdev = meta.rdev();
                // Linux dev_t: major in bits 8..19 and 32..47, minor in 0..7 and 20..31.
                // For typical small device numbers the canonical formulas:
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
                self.add_device_to(dev, parent_ino, name_bytes, DeviceKind::Fifo, 0, 0, fmeta)?;
            } else if ft.is_socket() {
                self.add_device_to(dev, parent_ino, name_bytes, DeviceKind::Socket, 0, 0, fmeta)?;
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
        let layout = layout::from_superblock(&sb)?;

        // GDT location: same logic as the writer.
        let bs = layout.block_size as u64;
        let gdt_off = if layout.first_data_block == 1 {
            2 * bs
        } else {
            bs
        };
        let mut gdt = vec![0u8; layout.gdt_blocks as usize * bs as usize];
        dev.read_at(gdt_off, &mut gdt)?;

        let mut groups = Vec::with_capacity(layout.groups.len());
        for i in 0..layout.groups.len() {
            let off = i * constants::GROUP_DESC_SIZE;
            let desc = GroupDesc::decode(
                gdt[off..off + constants::GROUP_DESC_SIZE]
                    .try_into()
                    .unwrap(),
            );
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
            groups,
            next_inode,
            inodes: Vec::new(),
            data_blocks: Vec::new(),
        })
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
    fn read_block(&self, dev: &mut dyn BlockDevice, blk: u32, out: &mut [u8]) -> Result<()> {
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
            return self.file_block_extent(ino, n);
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

    /// Resolve logical block `n` against an inode that uses an inline
    /// (depth-0) ext4 extent tree.
    fn file_block_extent(&self, ino: &Inode, n: u32) -> Result<u32> {
        // The 15-slot u32 i_block array overlays as a 60-byte extent tree.
        let mut buf = [0u8; 60];
        for (i, slot) in ino.block.iter().enumerate() {
            let off = i * 4;
            buf[off..off + 4].copy_from_slice(&slot.to_le_bytes());
        }
        let magic = u16::from_le_bytes(buf[0..2].try_into().unwrap());
        if magic != extent::EXT4_EXT_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "ext4: extent header magic {magic:#06x} != {:#06x}",
                extent::EXT4_EXT_MAGIC
            )));
        }
        let entries = u16::from_le_bytes(buf[2..4].try_into().unwrap()) as usize;
        let depth = u16::from_le_bytes(buf[6..8].try_into().unwrap());
        if depth != 0 {
            return Err(crate::Error::Unsupported(
                "ext4: multi-level extent trees not yet supported in reader".into(),
            ));
        }
        for i in 0..entries {
            let off = 12 + i * 12;
            let ee_block = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            let ee_len = u16::from_le_bytes(buf[off + 4..off + 6].try_into().unwrap());
            let ee_start_hi = u16::from_le_bytes(buf[off + 6..off + 8].try_into().unwrap()) as u64;
            let ee_start_lo = u32::from_le_bytes(buf[off + 8..off + 12].try_into().unwrap()) as u64;
            // Uninitialized extents have ee_len in the [32768, 65535] range
            // and represent zero blocks; we don't emit those but tolerate
            // them on read.
            let len = if ee_len > extent::MAX_LEN_PER_EXTENT {
                ee_len - extent::MAX_LEN_PER_EXTENT
            } else {
                ee_len
            };
            if n >= ee_block && n < ee_block + len as u32 {
                let phys = (ee_start_hi << 32) | ee_start_lo;
                return Ok((phys + (n - ee_block) as u64) as u32);
            }
        }
        // Logical block not covered by any extent → sparse hole, reads as zero.
        Ok(0)
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
                    out.push(crate::fs::DirEntry {
                        name: String::from_utf8_lossy(entry.name).into_owned(),
                        inode: entry.inode,
                        kind: kind_from_mode(child.mode),
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
fn split_path(path: &std::path::Path) -> Result<(std::path::PathBuf, String)> {
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

impl crate::fs::Filesystem for Ext {
    type FormatOpts = FormatOpts;

    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format_with(dev, opts)
    }

    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }

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

    fn remove(&mut self, _dev: &mut dyn BlockDevice, _path: &std::path::Path) -> Result<()> {
        Err(crate::Error::Unsupported(
            "ext: remove() not yet implemented".into(),
        ))
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

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        Self::flush(self, dev)
    }
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

/// Append a dir entry to a 4 KiB-aligned directory block by shrinking the
/// existing last entry to its natural minimum and writing the new entry
/// into the freed tail.
fn append_dir_entry(
    block: &mut [u8],
    name: &[u8],
    inode: u32,
    file_type: u8,
    with_filetype: bool,
) -> Result<()> {
    let needed = dir::min_rec_len(name.len());
    let mut off = 0usize;
    let last_off: usize;
    loop {
        let entry = dir::decode_entry(&block[off..], with_filetype).ok_or_else(|| {
            crate::Error::InvalidImage("corrupt dir entry while appending".into())
        })?;
        let next = off + entry.rec_len;
        if next >= block.len() {
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
}
