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

pub mod constants;
pub mod dir;
pub mod group;
pub mod inode;
pub mod layout;
pub mod superblock;

use std::io::Read;

use constants::{INO_ROOT_DIR, SUPERBLOCK_OFFSET};
use group::{GroupDesc, set_bit, set_first_n, test_bit};
use inode::{Inode, SpecialKind};
use layout::Layout;
use superblock::Superblock;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::{DeviceKind, FileMeta, FileSource};

/// Options accepted by [`Ext::format_with`].
///
/// Defaults mirror `genext2fs -d <dir> -f -q -B 1024`: 1 KiB blocks, no
/// features, all-zero UUID + label, mtime 0, 5% reserved blocks, lost+found
/// pre-allocated.
#[derive(Debug, Clone)]
pub struct FormatOpts {
    pub block_size: u32,
    pub blocks_count: u32,
    pub inodes_count: u32,
    pub uuid: [u8; 16],
    pub volume_label: [u8; 16],
    pub mtime: u32,
    pub reserved_blocks_percent: u8,
    pub create_lost_found: bool,
}

impl Default for FormatOpts {
    fn default() -> Self {
        Self {
            block_size: 1024,
            blocks_count: 1024,
            inodes_count: 16,
            uuid: [0; 16],
            volume_label: [0; 16],
            mtime: 0,
            reserved_blocks_percent: 5,
            create_lost_found: true,
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
            ext.create_lost_found(opts.mtime)?;
        }

        ext.recompute_free_counts();
        ext.flush_metadata(dev)?;
        Ok(ext)
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
    fn create_lost_found(&mut self, mtime: u32) -> Result<()> {
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
        let n_direct = (constants::N_DIRECT as u32).min(target_data_blocks) as usize;
        inode.block[..n_direct].copy_from_slice(&data_blocks[..n_direct]);
        if target_data_blocks > constants::N_DIRECT as u32 {
            let ind_block = self.alloc_data_block(0)?;
            inode.block[constants::IDX_INDIRECT] = ind_block;
            let mut ind = vec![0u8; bs as usize];
            for (i, &b) in data_blocks[constants::N_DIRECT..].iter().enumerate() {
                let off = i * 4;
                ind[off..off + 4].copy_from_slice(&b.to_le_bytes());
            }
            self.data_blocks.push((ind_block, ind));
            inode.blocks_512 = (target_data_blocks + 1) * (bs / 512);
        } else {
            inode.blocks_512 = target_data_blocks * (bs / 512);
        }

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
        self.add_entry_to_dir_block_for(INO_ROOT_DIR, b"lost+found", ino)?;
        self.patch_inode(INO_ROOT_DIR, |i| i.links_count += 1);
        Ok(())
    }

    /// Append a dir entry into the data block(s) of the directory whose
    /// inode is `dir_inode`. v1: only ever touches the directory's first
    /// data block; full-block error if it doesn't fit.
    fn add_entry_to_dir_block_for(
        &mut self,
        dir_inode: u32,
        name: &[u8],
        child_ino: u32,
    ) -> Result<()> {
        let dir_block_num = self
            .inodes
            .iter()
            .find(|(i, _)| *i == dir_inode)
            .map(|(_, inode)| inode.block[0])
            .ok_or_else(|| crate::Error::InvalidArgument("dir inode not staged".into()))?;
        let block = self
            .data_blocks
            .iter_mut()
            .find(|(b, _)| *b == dir_block_num)
            .map(|(_, bytes)| bytes)
            .ok_or_else(|| crate::Error::InvalidImage("dir data block missing".into()))?;
        append_dir_entry(block, name, child_ino, constants::DENT_DIR, false)
    }

    /// Mutate a staged inode entry in place. Panics if absent.
    fn patch_inode<F: FnOnce(&mut Inode)>(&mut self, ino: u32, f: F) {
        for (i_no, i) in self.inodes.iter_mut() {
            if *i_no == ino {
                f(i);
                return;
            }
        }
        panic!("inode {ino} not staged");
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
        // v1 supports direct + single-indirect blocks only. Capacity:
        //   N_DIRECT + (block_size / 4)  blocks
        let max_blocks = constants::N_DIRECT as u64 + bs as u64 / 4;
        let max_bytes = max_blocks * bs as u64;
        if len > max_bytes {
            return Err(crate::Error::Unsupported(format!(
                "ext: file too large in v1 (max {max_bytes} bytes with single-indirect)"
            )));
        }
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

        // Block pointers.
        let n_direct = (constants::N_DIRECT as u32).min(n_data_blocks) as usize;
        inode.block[..n_direct].copy_from_slice(&data_blocks[..n_direct]);
        let mut allocated_meta_blocks = 0u32;
        if n_data_blocks as usize > constants::N_DIRECT {
            let ind = self.alloc_data_block(0)?;
            allocated_meta_blocks += 1;
            inode.block[constants::IDX_INDIRECT] = ind;
            let mut ind_buf = vec![0u8; bs as usize];
            for (i, &b) in data_blocks[constants::N_DIRECT..].iter().enumerate() {
                let off = i * 4;
                ind_buf[off..off + 4].copy_from_slice(&b.to_le_bytes());
            }
            self.data_blocks.push((ind, ind_buf));
        }
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
        self.add_entry_to_dir_block_for(parent_ino, name, ino)?;
        Ok(ino)
    }

    /// Create a subdirectory under `parent_ino`. Allocates one data block
    /// holding "." / "..", patches the parent's link count.
    pub fn add_dir_to(
        &mut self,
        _dev: &mut dyn BlockDevice,
        parent_ino: u32,
        name: &[u8],
        meta: FileMeta,
    ) -> Result<u32> {
        let bs = self.layout.block_size;
        let ino = self.alloc_inode()?;
        let blk = self.alloc_data_block(0)?;
        let mut inode = Inode::directory(bs, meta.mode & 0o7777, meta.uid, meta.gid, meta.mtime);
        inode.block[0] = blk;
        inode.blocks_512 = bs / 512;
        let block_bytes = dir::make_initial_dir_block(ino, parent_ino, bs, false);
        self.data_blocks.push((blk, block_bytes));
        self.inodes.push((ino, inode));
        self.groups[0].desc.used_dirs_count += 1;

        self.add_entry_to_dir_block_for(parent_ino, name, ino)?;
        self.patch_inode(parent_ino, |i| i.links_count += 1);
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
        self.add_entry_to_dir_block_for(parent_ino, name, ino)?;
        Ok(ino)
    }

    /// Create a device node, FIFO, or socket. No data blocks are allocated;
    /// for char/block devices the major+minor are encoded into `i_block\[0\]`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_device_to(
        &mut self,
        _dev: &mut dyn BlockDevice,
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
        self.add_entry_to_dir_block_for(parent_ino, name, ino)?;
        Ok(ino)
    }

    /// Persist all staged metadata (bitmaps, inode table, dir blocks,
    /// superblock) to the device. The primary superblock is written last.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        self.recompute_free_counts();
        self.flush_metadata(dev)
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
