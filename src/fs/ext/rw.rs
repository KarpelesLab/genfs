//! In-place read/write file handle for ext2/3/4 images.
//!
//! Backs [`crate::fs::Filesystem::open_file_rw`] for any ext family image
//! whose journal (if present) is either clean or carries only committed
//! work that this open replays. File data blocks stream straight to the
//! device; metadata updates (inode, bitmaps, group descriptors,
//! extent leaf blocks) ride the existing staging vectors and become
//! durable on [`FileHandle::sync`] / [`Ext::flush`].
//!
//! ## Journal handling (Path A: real JBD2 transactions)
//!
//! ext3 and ext4 carry a JBD2 journal (`COMPAT_HAS_JOURNAL`). On
//! [`FileHandle::sync`] the flush path emits one JBD2 transaction
//! covering every metadata block image that would otherwise be written
//! directly: bitmaps, GDT, inode-table blocks containing patched inodes,
//! and staged dir / extent-leaf / indirect blocks. The transaction
//! ordering is descriptor → data payload → commit → journal-SB
//! `s_start` update → checkpoint (write blocks to their FS-home
//! locations) → journal-SB `s_start = 0`. A crash after the commit
//! block lands but before checkpoint completes is caught at the next
//! open: `jbd2::replay_journal` re-applies the committed transaction.
//!
//! On open, if `s_start != 0` we replay the log before attaching the
//! handle; a clean journal opens with no replay work. The primary
//! superblock is written outside the journal (the kernel does this too
//! for many of its SB updates); free-block / free-inode counters in the
//! SB are recomputed from the on-disk bitmaps each [`Ext::flush`].
//!
//! ## Extent-tree support
//!
//! Inodes with `EXT4_EXTENTS_FL` are read/written through a depth-0
//! inline extent tree (up to 4 leaves stored directly in `i_block`) or a
//! depth-1 tree (up to 4 idx entries in `i_block`, each pointing at a
//! single leaf block holding the actual extent records). The writer
//! flattens the on-disk representation into a single list of runs on
//! read, mutates that list in memory, and partitions it back out across
//! leaf blocks on writeback — promoting from depth-0 to depth-1
//! automatically when more than 4 leaves are needed. Depth-2 and deeper
//! trees are still refused at open time; the writer rejects requests
//! that would need a second level of index blocks.

use std::io::{self, Read, Seek, SeekFrom, Write};

use super::Ext;
use super::constants::{
    self, EXT4_EXTENTS_FL, IDX_DOUBLE_INDIRECT, IDX_INDIRECT, IDX_TRIPLE_INDIRECT, N_DIRECT,
    S_IFMT, S_IFREG,
};
use super::extent::{self, ExtentRun, MAX_LEN_PER_EXTENT};
use super::jbd2;
use crate::Result;
use crate::block::BlockDevice;
use crate::fs::FileHandle;

/// Snapshot of the inode's extent tree: the flat list of leaf extents,
/// plus the physical block numbers of every on-disk tree block (internal
/// index nodes at any depth *and* leaf blocks) the inode currently
/// references. The inline root lives in `i_block`, so it is never listed.
/// The `meta_blocks` list is empty for a depth-0 inline tree.
struct ExtentTreeState {
    runs: Vec<ExtentRun>,
    meta_blocks: Vec<u32>,
}

/// Read/write file handle on an ext family image. Drives the
/// indirect-block tree (no extents). Implements [`FileHandle`] via
/// [`Read`] + [`Write`] + [`Seek`].
pub struct Ext2FileHandle<'a> {
    ext: &'a mut Ext,
    dev: &'a mut dyn BlockDevice,
    ino: u32,
    pos: u64,
    /// Logical file length tracked locally so successive writes / seeks
    /// see the latest size without re-reading the inode.
    len: u64,
}

impl<'a> Ext2FileHandle<'a> {
    /// Construct from an existing regular-file inode. Caller is
    /// responsible for `truncate` / `append` handling.
    pub(crate) fn new(
        ext: &'a mut Ext,
        dev: &'a mut dyn BlockDevice,
        ino: u32,
        len: u64,
    ) -> Result<Self> {
        // Make sure the inode is staged so subsequent mutations bypass
        // the read-only on-disk slot. This is what `patch_inode` does
        // internally for the small mutations elsewhere in the writer.
        ext.ensure_inode_staged(dev, ino)?;
        Ok(Self {
            ext,
            dev,
            ino,
            pos: 0,
            len,
        })
    }

    /// Move the local file length and patch the staged inode's
    /// `i_size` / `i_size_high`. `blocks_512` stays in sync via
    /// [`Self::recompute_blocks_512`]. For files > 4 GiB the
    /// `RO_COMPAT_LARGE_FILE` feature is stamped on the superblock so
    /// the high half of the size is honoured by readers.
    fn set_inode_size(&mut self, new_len: u64) -> Result<()> {
        let bs = self.ext.layout.block_size;
        let blocks_needed = new_len.div_ceil(bs as u64);
        if blocks_needed > u32::MAX as u64 {
            return Err(crate::Error::Unsupported(format!(
                "ext: file of {new_len} bytes needs {blocks_needed} blocks (> u32::MAX)"
            )));
        }
        if new_len > u32::MAX as u64 {
            self.ext.sb.feature_ro_compat |=
                crate::fs::ext::constants::feature::RO_COMPAT_LARGE_FILE;
        }
        self.len = new_len;
        let inode = self.staged_inode_mut();
        inode.set_file_size(new_len);
        Ok(())
    }

    /// True if the underlying inode uses an ext4 extent tree.
    fn is_extent(&self) -> bool {
        self.staged_inode().flags & EXT4_EXTENTS_FL != 0
    }

    /// Recompute `i_blocks_512` from the current set of allocated blocks
    /// (direct + indirect tree). Walks the tree, counts every populated
    /// pointer (data) and every indirection-metadata block, multiplies by
    /// `block_size / 512`.
    fn recompute_blocks_512(&mut self) -> Result<()> {
        if self.is_extent() {
            return self.recompute_blocks_512_extent();
        }
        let bs = self.ext.layout.block_size;
        let ptrs = (bs / 4) as u64;
        let inode = *self.staged_inode();

        let mut data = 0u64;
        let mut meta = 0u64;

        // Direct.
        for b in &inode.block[..N_DIRECT] {
            if *b != 0 {
                data += 1;
            }
        }
        // Single-indirect.
        let ind = inode.block[IDX_INDIRECT];
        if ind != 0 {
            meta += 1;
            let buf = self.read_indirect_block(ind)?;
            for i in 0..ptrs as usize {
                let p = read_u32_le(&buf, i * 4);
                if p != 0 {
                    data += 1;
                }
            }
        }
        // Double-indirect.
        let dind = inode.block[IDX_DOUBLE_INDIRECT];
        if dind != 0 {
            meta += 1;
            let outer = self.read_indirect_block(dind)?;
            for i in 0..ptrs as usize {
                let sub = read_u32_le(&outer, i * 4);
                if sub != 0 {
                    meta += 1;
                    let inner = self.read_indirect_block(sub)?;
                    for j in 0..ptrs as usize {
                        let p = read_u32_le(&inner, j * 4);
                        if p != 0 {
                            data += 1;
                        }
                    }
                }
            }
        }
        // Triple-indirect (rarely used, but support for completeness).
        let tind = inode.block[IDX_TRIPLE_INDIRECT];
        if tind != 0 {
            meta += 1;
            let l1 = self.read_indirect_block(tind)?;
            for i in 0..ptrs as usize {
                let dind2 = read_u32_le(&l1, i * 4);
                if dind2 != 0 {
                    meta += 1;
                    let l2 = self.read_indirect_block(dind2)?;
                    for j in 0..ptrs as usize {
                        let sub = read_u32_le(&l2, j * 4);
                        if sub != 0 {
                            meta += 1;
                            let inner = self.read_indirect_block(sub)?;
                            for k in 0..ptrs as usize {
                                let p = read_u32_le(&inner, k * 4);
                                if p != 0 {
                                    data += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        let total_sectors = (data + meta) * (bs as u64 / 512);
        self.staged_inode_mut().blocks_512 = total_sectors as u32;
        Ok(())
    }

    /// Snapshot of the inode's current extent tree layout. Used by
    /// [`Self::write_extent_runs_with_state`] to figure out which old tree
    /// blocks need to be freed when the tree is re-packed. Walks trees of
    /// any depth: the inline root is decoded here, then every subtree is
    /// gathered recursively by [`Self::collect_extent_tree`].
    fn read_extent_tree_state(&mut self) -> Result<ExtentTreeState> {
        let inode = *self.staged_inode();
        let bytes = extent::iblock_to_bytes(&inode.block);
        let header = extent::decode_header(&bytes[..12])?;
        if header.depth == 0 {
            let (_, runs) = extent::decode_depth0_iblock(&bytes)?;
            return Ok(ExtentTreeState {
                runs,
                meta_blocks: Vec::new(),
            });
        }
        // Depth ≥ 1: the inline root holds idx entries pointing at on-disk
        // subtrees (internal index blocks for depth ≥ 2, leaf blocks for
        // depth 1). Recurse into each, collecting runs in logical order
        // and every on-disk tree block.
        let (_, indices) = extent::decode_idx_iblock(&bytes)?;
        let mut runs = Vec::new();
        let mut meta_blocks = Vec::with_capacity(indices.len());
        for idx in &indices {
            self.collect_extent_tree(idx.leaf as u32, &mut runs, &mut meta_blocks)?;
        }
        Ok(ExtentTreeState { runs, meta_blocks })
    }

    /// Recursively gather the subtree rooted at on-disk block `blk`:
    /// append its leaf extents to `runs` and record every block it spans
    /// (itself plus any descendants) in `meta_blocks`. A block whose
    /// header claims depth 0 is a leaf; otherwise it is an internal index
    /// node and its idx entries are followed.
    fn collect_extent_tree(
        &mut self,
        blk: u32,
        runs: &mut Vec<ExtentRun>,
        meta_blocks: &mut Vec<u32>,
    ) -> Result<()> {
        let bs = self.ext.layout.block_size as usize;
        let buf = self.read_indirect_block(blk)?;
        let header = extent::decode_header(&buf[..12])?;
        meta_blocks.push(blk);
        if header.depth == 0 {
            let (_, mut leaf_runs) = extent::decode_leaf_block(&buf[..bs])?;
            runs.append(&mut leaf_runs);
        } else {
            for i in 0..header.entries as usize {
                let off = 12 + i * 12;
                let idx = extent::decode_idx(&buf[off..off + 12]);
                self.collect_extent_tree(idx.leaf as u32, runs, meta_blocks)?;
            }
        }
        Ok(())
    }

    /// Read the current set of leaf extents (flat list across all leaves).
    /// Equivalent to [`Self::read_extent_tree_state`] but discards the
    /// leaf-block tracking — used by read-only paths.
    fn read_extent_runs(&mut self) -> Result<Vec<ExtentRun>> {
        Ok(self.read_extent_tree_state()?.runs)
    }

    /// Stamp a new extent run list onto the staged inode, re-packing the
    /// tree into whatever depth the run count requires:
    ///   - ≤ 4 runs → depth-0 inline (no on-disk tree blocks).
    ///   - more → an [`extent::pack_extent_tree`] of depth ≥ 1: leaves,
    ///     then internal index levels, until the top fits the 4 inline
    ///     `i_block` idx slots.
    ///
    /// Every old on-disk tree block is freed first so `alloc_data_block`
    /// can re-use it; the fresh blocks are staged and tracked for
    /// `ext4_extent_tail` CRC stamping at flush.
    fn write_extent_runs_with_state(
        &mut self,
        runs: &[ExtentRun],
        old_state: &ExtentTreeState,
    ) -> Result<()> {
        let bs = self.ext.layout.block_size;
        let csum_tail = self.ext.has_metadata_csum();
        let ino = self.ino;

        // Free old tree blocks first so that alloc_data_block can re-use
        // them if convenient (and so the bitmap accounting stays accurate
        // if a later allocation fails).
        for &b in &old_state.meta_blocks {
            self.free_leaf_block(b);
        }

        // Build the tree bottom-up. For ≤ 4 runs this returns an inline
        // depth-0 tree and never touches the allocator.
        let (i_block_bytes, tree_blocks) = {
            let mut alloc = || self.ext.alloc_data_block();
            extent::pack_extent_tree(runs, bs, csum_tail, &mut alloc)?
        };
        for tb in tree_blocks {
            self.write_indirect_block(tb.phys, &tb.image)?;
            // Index nodes carry the same `ext4_extent_tail` CRC as leaves;
            // track every staged block for stamping at flush time.
            self.ext.track_extent_leaf_block(tb.phys, ino);
        }
        self.staged_inode_mut().block = extent::bytes_to_iblock(&i_block_bytes);
        Ok(())
    }

    /// Free a leaf block (depth-1 internal node). Also evicts the staged
    /// copy from `ext.data_blocks` so a future read of that block number
    /// hits the (now-unowned) on-disk content instead of the stale leaf,
    /// and drops the metadata_csum tracking record so no stale CRC is
    /// stamped at flush.
    fn free_leaf_block(&mut self, blk: u32) {
        self.ext.free_block(blk);
        self.ext.data_blocks.retain(|(b, _)| *b != blk);
        self.ext.untrack_extent_leaf_block(blk);
    }

    /// Sum the lengths of every leaf extent in the tree, then add the
    /// number of allocated leaf/idx metadata blocks. Used to update
    /// `blocks_512`.
    fn recompute_blocks_512_extent(&mut self) -> Result<()> {
        let state = self.read_extent_tree_state()?;
        let bs = self.ext.layout.block_size as u64;
        let mut data = 0u64;
        for r in &state.runs {
            let len = if r.len > MAX_LEN_PER_EXTENT {
                // Uninitialized extents (not emitted by us; tolerate on
                // read). The length encoded in ee_len - 32768.
                (r.len - MAX_LEN_PER_EXTENT) as u64
            } else {
                r.len as u64
            };
            data += len;
        }
        let meta = state.meta_blocks.len() as u64;
        let sectors = (data + meta) * (bs / 512);
        self.staged_inode_mut().blocks_512 = sectors as u32;
        Ok(())
    }

    /// Resolve logical block `n` against the extent tree without
    /// allocating. Returns 0 for a sparse hole (no extent covers `n`).
    /// Reads the leaf block on demand when the tree is depth-1.
    fn read_logical_block_extent(&mut self, n: u32) -> Result<u32> {
        let runs = self.read_extent_runs()?;
        for r in &runs {
            let len = if r.len > MAX_LEN_PER_EXTENT {
                r.len - MAX_LEN_PER_EXTENT
            } else {
                r.len
            };
            if n >= r.logical && n < r.logical + len as u32 {
                let phys = r.physical + (n - r.logical) as u64;
                return Ok(phys as u32);
            }
        }
        Ok(0)
    }

    /// Resolve logical block `n` against the extent tree, allocating a
    /// new physical block if `n` is uncovered. Merges with an adjacent
    /// extent when the freshly-allocated block lies right before / after
    /// one; otherwise appends a new leaf. Auto-promotes a depth-0 tree
    /// to depth-1 when the leaf count overflows 4.
    fn get_or_alloc_block_extent(&mut self, n: u32) -> Result<u32> {
        let state = self.read_extent_tree_state()?;
        let mut runs = state.runs.clone();

        // Already mapped?
        for r in &runs {
            let len = if r.len > MAX_LEN_PER_EXTENT {
                r.len - MAX_LEN_PER_EXTENT
            } else {
                r.len
            };
            if n >= r.logical && n < r.logical + len as u32 {
                let phys = r.physical + (n - r.logical) as u64;
                return Ok(phys as u32);
            }
        }

        // Need a fresh physical block.
        let new_phys = self.ext.alloc_data_block()? as u64;
        self.zero_block_on_disk(new_phys as u32)?;

        // Try to extend an existing extent whose tail meets the new block
        // both logically and physically. Only "initialized" extents are
        // mutable here — leave any (unexpected) uninitialised extents
        // alone.
        let mut merged = false;
        for r in runs.iter_mut() {
            if r.len >= MAX_LEN_PER_EXTENT {
                continue;
            }
            let tail_logical = r.logical + r.len as u32;
            let tail_phys = r.physical + r.len as u64;
            if tail_logical == n && tail_phys == new_phys {
                r.len += 1;
                merged = true;
                break;
            }
        }

        // If not, try to prepend to an extent that starts immediately
        // after the new block.
        if !merged {
            for r in runs.iter_mut() {
                if r.len >= MAX_LEN_PER_EXTENT {
                    continue;
                }
                if n + 1 == r.logical && new_phys + 1 == r.physical {
                    r.logical = n;
                    r.physical = new_phys;
                    r.len += 1;
                    merged = true;
                    break;
                }
            }
        }

        if !merged {
            // No capacity ceiling: write_extent_runs_with_state re-packs the
            // run list into an extent tree of whatever depth it needs.
            runs.push(ExtentRun {
                logical: n,
                len: 1,
                physical: new_phys,
            });
            runs.sort_by_key(|r| r.logical);
        }

        // After a merge two adjacent extents may now meet; coalesce
        // whenever the tail of [i] meets the head of [i+1].
        let mut i = 0;
        while i + 1 < runs.len() {
            let (a, b) = (runs[i], runs[i + 1]);
            let a_len = if a.len > MAX_LEN_PER_EXTENT {
                a.len - MAX_LEN_PER_EXTENT
            } else {
                a.len
            };
            if a.len < MAX_LEN_PER_EXTENT
                && a.logical + a_len as u32 == b.logical
                && a.physical + a_len as u64 == b.physical
                && a.len.saturating_add(b.len) <= MAX_LEN_PER_EXTENT
            {
                runs[i].len += b.len;
                runs.remove(i + 1);
            } else {
                i += 1;
            }
        }

        self.write_extent_runs_with_state(&runs, &state)?;
        Ok(new_phys as u32)
    }

    /// Free every block at logical index >= `from` from the extent tree.
    /// Updates the leaf list, freeing trailing extents entirely and
    /// shrinking the one (if any) that straddles `from`. Re-partitions
    /// the surviving runs across depth-0 or depth-1 as appropriate.
    fn free_blocks_from_extent(&mut self, from: u32) -> Result<()> {
        let state = self.read_extent_tree_state()?;
        let mut kept: Vec<ExtentRun> = Vec::with_capacity(state.runs.len());
        for r in &state.runs {
            let r = *r;
            let len = if r.len > MAX_LEN_PER_EXTENT {
                r.len - MAX_LEN_PER_EXTENT
            } else {
                r.len
            };
            if r.logical >= from {
                // Entirely beyond the new EOF — free the whole run.
                for off in 0..len as u32 {
                    self.ext.free_block((r.physical + off as u64) as u32);
                }
            } else if r.logical + len as u32 > from {
                // Straddles `from`. Keep [logical .. from), free the rest.
                let new_len = from - r.logical;
                for off in new_len..len as u32 {
                    self.ext.free_block((r.physical + off as u64) as u32);
                }
                let mut shortened = r;
                shortened.len = new_len as u16;
                kept.push(shortened);
            } else {
                // Entirely below `from` — keep as is.
                kept.push(r);
            }
        }
        self.write_extent_runs_with_state(&kept, &state)?;
        Ok(())
    }

    fn staged_inode(&self) -> &super::Inode {
        self.ext
            .inodes
            .iter()
            .find(|(i, _)| *i == self.ino)
            .map(|(_, i)| i)
            .expect("inode is staged at handle construction")
    }

    fn staged_inode_mut(&mut self) -> &mut super::Inode {
        self.ext
            .inodes
            .iter_mut()
            .find(|(i, _)| *i == self.ino)
            .map(|(_, i)| i)
            .expect("inode is staged at handle construction")
    }

    /// Read a single indirect-tree block. Returns a fresh `Vec`; checks
    /// the staged-block cache first so newly-allocated indirect blocks
    /// (which are not yet flushed to disk) are read from memory.
    fn read_indirect_block(&mut self, blk: u32) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; self.ext.layout.block_size as usize];
        self.ext.read_block(self.dev, blk, &mut buf)?;
        Ok(buf)
    }

    /// Stage a write to an indirect-tree block. Updates the in-memory
    /// copy so subsequent reads see the change; also writes through to
    /// the device so the new pointer is durable on a clean unmount even
    /// without an explicit sync (ext2's existing best-effort model).
    fn write_indirect_block(&mut self, blk: u32, bytes: &[u8]) -> Result<()> {
        let bs = self.ext.layout.block_size as u64;
        // Update the staged copy if present, otherwise add one.
        if let Some(slot) = self.ext.data_blocks.iter_mut().find(|(b, _)| *b == blk) {
            slot.1.clear();
            slot.1.extend_from_slice(bytes);
        } else {
            self.ext.data_blocks.push((blk, bytes.to_vec()));
        }
        self.dev.write_at(blk as u64 * bs, bytes)?;
        Ok(())
    }

    /// Resolve logical block `n` to its physical block number, allocating
    /// missing entries along the path. The newly-allocated indirect /
    /// data blocks are zeroed on disk before being returned so a
    /// partial-block write can RMW safely.
    fn get_or_alloc_block(&mut self, n: u32) -> Result<u32> {
        if self.is_extent() {
            return self.get_or_alloc_block_extent(n);
        }
        let bs = self.ext.layout.block_size;
        let ptrs = bs / 4;

        if (n as usize) < N_DIRECT {
            let mut blk = self.staged_inode().block[n as usize];
            if blk == 0 {
                blk = self.ext.alloc_data_block()?;
                self.zero_block_on_disk(blk)?;
                self.staged_inode_mut().block[n as usize] = blk;
            }
            return Ok(blk);
        }

        // Indirect ranges.
        let n_off = n - N_DIRECT as u32;
        if n_off < ptrs {
            // Single-indirect.
            let mut ind = self.staged_inode().block[IDX_INDIRECT];
            if ind == 0 {
                ind = self.ext.alloc_data_block()?;
                self.zero_block_on_disk(ind)?;
                self.staged_inode_mut().block[IDX_INDIRECT] = ind;
            }
            let mut buf = self.read_indirect_block(ind)?;
            let slot = n_off as usize * 4;
            let mut blk = read_u32_le(&buf, slot);
            if blk == 0 {
                blk = self.ext.alloc_data_block()?;
                self.zero_block_on_disk(blk)?;
                write_u32_le(&mut buf, slot, blk);
                self.write_indirect_block(ind, &buf)?;
            }
            return Ok(blk);
        }

        let n_off = n_off - ptrs;
        if n_off < ptrs * ptrs {
            // Double-indirect.
            let mut dind = self.staged_inode().block[IDX_DOUBLE_INDIRECT];
            if dind == 0 {
                dind = self.ext.alloc_data_block()?;
                self.zero_block_on_disk(dind)?;
                self.staged_inode_mut().block[IDX_DOUBLE_INDIRECT] = dind;
            }
            let mut outer = self.read_indirect_block(dind)?;
            let outer_slot = (n_off / ptrs) as usize * 4;
            let mut sub = read_u32_le(&outer, outer_slot);
            if sub == 0 {
                sub = self.ext.alloc_data_block()?;
                self.zero_block_on_disk(sub)?;
                write_u32_le(&mut outer, outer_slot, sub);
                self.write_indirect_block(dind, &outer)?;
            }
            let mut inner = self.read_indirect_block(sub)?;
            let inner_slot = ((n_off % ptrs) as usize) * 4;
            let mut blk = read_u32_le(&inner, inner_slot);
            if blk == 0 {
                blk = self.ext.alloc_data_block()?;
                self.zero_block_on_disk(blk)?;
                write_u32_le(&mut inner, inner_slot, blk);
                self.write_indirect_block(sub, &inner)?;
            }
            return Ok(blk);
        }

        let n_off = n_off - ptrs * ptrs;
        if n_off < ptrs * ptrs * ptrs {
            // Triple-indirect.
            let mut tind = self.staged_inode().block[IDX_TRIPLE_INDIRECT];
            if tind == 0 {
                tind = self.ext.alloc_data_block()?;
                self.zero_block_on_disk(tind)?;
                self.staged_inode_mut().block[IDX_TRIPLE_INDIRECT] = tind;
            }
            let mut l1 = self.read_indirect_block(tind)?;
            let l1_slot = (n_off / (ptrs * ptrs)) as usize * 4;
            let mut dind = read_u32_le(&l1, l1_slot);
            if dind == 0 {
                dind = self.ext.alloc_data_block()?;
                self.zero_block_on_disk(dind)?;
                write_u32_le(&mut l1, l1_slot, dind);
                self.write_indirect_block(tind, &l1)?;
            }
            let rem = n_off % (ptrs * ptrs);
            let mut l2 = self.read_indirect_block(dind)?;
            let l2_slot = (rem / ptrs) as usize * 4;
            let mut sub = read_u32_le(&l2, l2_slot);
            if sub == 0 {
                sub = self.ext.alloc_data_block()?;
                self.zero_block_on_disk(sub)?;
                write_u32_le(&mut l2, l2_slot, sub);
                self.write_indirect_block(dind, &l2)?;
            }
            let mut inner = self.read_indirect_block(sub)?;
            let inner_slot = ((rem % ptrs) as usize) * 4;
            let mut blk = read_u32_le(&inner, inner_slot);
            if blk == 0 {
                blk = self.ext.alloc_data_block()?;
                self.zero_block_on_disk(blk)?;
                write_u32_le(&mut inner, inner_slot, blk);
                self.write_indirect_block(sub, &inner)?;
            }
            return Ok(blk);
        }

        Err(crate::Error::Unsupported(
            "ext2: logical block exceeds triple-indirect range".into(),
        ))
    }

    /// Resolve logical block `n` without allocating. Returns 0 for an
    /// unallocated (hole) block. Mirrors [`Ext::file_block`] but supports
    /// double/triple-indirect on the read path so writes that allocated
    /// deep blocks can be read back through the same handle.
    fn read_logical_block(&mut self, n: u32) -> Result<u32> {
        if self.is_extent() {
            return self.read_logical_block_extent(n);
        }
        let bs = self.ext.layout.block_size;
        let ptrs = bs / 4;
        let inode = *self.staged_inode();
        if (n as usize) < N_DIRECT {
            return Ok(inode.block[n as usize]);
        }
        let n_off = n - N_DIRECT as u32;
        if n_off < ptrs {
            let ind = inode.block[IDX_INDIRECT];
            if ind == 0 {
                return Ok(0);
            }
            let buf = self.read_indirect_block(ind)?;
            return Ok(read_u32_le(&buf, n_off as usize * 4));
        }
        let n_off = n_off - ptrs;
        if n_off < ptrs * ptrs {
            let dind = inode.block[IDX_DOUBLE_INDIRECT];
            if dind == 0 {
                return Ok(0);
            }
            let outer = self.read_indirect_block(dind)?;
            let sub = read_u32_le(&outer, (n_off / ptrs) as usize * 4);
            if sub == 0 {
                return Ok(0);
            }
            let inner = self.read_indirect_block(sub)?;
            return Ok(read_u32_le(&inner, (n_off % ptrs) as usize * 4));
        }
        let n_off = n_off - ptrs * ptrs;
        if n_off < ptrs * ptrs * ptrs {
            let tind = inode.block[IDX_TRIPLE_INDIRECT];
            if tind == 0 {
                return Ok(0);
            }
            let l1 = self.read_indirect_block(tind)?;
            let dind = read_u32_le(&l1, (n_off / (ptrs * ptrs)) as usize * 4);
            if dind == 0 {
                return Ok(0);
            }
            let rem = n_off % (ptrs * ptrs);
            let l2 = self.read_indirect_block(dind)?;
            let sub = read_u32_le(&l2, (rem / ptrs) as usize * 4);
            if sub == 0 {
                return Ok(0);
            }
            let inner = self.read_indirect_block(sub)?;
            return Ok(read_u32_le(&inner, (rem % ptrs) as usize * 4));
        }
        Ok(0)
    }

    /// Free every block allocated at logical index >= `from`. Frees data
    /// blocks via the bitmap, and contracts the indirect tree by zeroing
    /// or freeing indirect blocks that no longer have any populated
    /// entries.
    fn free_blocks_from(&mut self, from: u32) -> Result<()> {
        if self.is_extent() {
            return self.free_blocks_from_extent(from);
        }
        let bs = self.ext.layout.block_size;
        let ptrs = bs / 4;
        let inode = *self.staged_inode();

        // 1. Direct pointers.
        for n in (from as usize)..N_DIRECT {
            let b = inode.block[n];
            if b != 0 {
                self.ext.free_block(b);
                self.staged_inode_mut().block[n] = 0;
            }
        }

        // 2. Single-indirect.
        let direct_end = N_DIRECT as u32;
        let single_end = direct_end + ptrs;
        if from < single_end {
            let ind = self.staged_inode().block[IDX_INDIRECT];
            if ind != 0 {
                let mut buf = self.read_indirect_block(ind)?;
                let first = from.saturating_sub(direct_end) as usize;
                let mut any_remaining = false;
                for i in 0..ptrs as usize {
                    let p = read_u32_le(&buf, i * 4);
                    if i >= first {
                        if p != 0 {
                            self.ext.free_block(p);
                            write_u32_le(&mut buf, i * 4, 0);
                        }
                    } else if p != 0 {
                        any_remaining = true;
                    }
                }
                if any_remaining {
                    self.write_indirect_block(ind, &buf)?;
                } else {
                    // Indirect block no longer needed.
                    self.ext.free_block(ind);
                    self.staged_inode_mut().block[IDX_INDIRECT] = 0;
                }
            }
        }

        // 3. Double-indirect.
        let double_end = single_end + ptrs * ptrs;
        if from < double_end {
            let dind = self.staged_inode().block[IDX_DOUBLE_INDIRECT];
            if dind != 0 {
                let mut outer = self.read_indirect_block(dind)?;
                let base = single_end;
                let mut any_outer = false;
                for i in 0..ptrs as usize {
                    let sub = read_u32_le(&outer, i * 4);
                    if sub == 0 {
                        continue;
                    }
                    let sub_start = base + (i as u32) * ptrs;
                    let sub_end = sub_start + ptrs;
                    if from >= sub_end {
                        any_outer = true;
                        continue;
                    }
                    let first = if from > sub_start {
                        (from - sub_start) as usize
                    } else {
                        0
                    };
                    let mut inner = self.read_indirect_block(sub)?;
                    let mut any_inner = false;
                    for j in 0..ptrs as usize {
                        let p = read_u32_le(&inner, j * 4);
                        if j >= first {
                            if p != 0 {
                                self.ext.free_block(p);
                                write_u32_le(&mut inner, j * 4, 0);
                            }
                        } else if p != 0 {
                            any_inner = true;
                        }
                    }
                    if any_inner {
                        self.write_indirect_block(sub, &inner)?;
                        any_outer = true;
                    } else {
                        self.ext.free_block(sub);
                        write_u32_le(&mut outer, i * 4, 0);
                    }
                }
                if any_outer {
                    self.write_indirect_block(dind, &outer)?;
                } else {
                    self.ext.free_block(dind);
                    self.staged_inode_mut().block[IDX_DOUBLE_INDIRECT] = 0;
                }
            }
        }

        // 4. Triple-indirect.
        let triple_end = double_end + ptrs * ptrs * ptrs;
        if from < triple_end {
            let tind = self.staged_inode().block[IDX_TRIPLE_INDIRECT];
            if tind != 0 {
                let mut l1 = self.read_indirect_block(tind)?;
                let base = double_end;
                let mut any_l1 = false;
                for i in 0..ptrs as usize {
                    let dind = read_u32_le(&l1, i * 4);
                    if dind == 0 {
                        continue;
                    }
                    let dind_start = base + (i as u32) * ptrs * ptrs;
                    let dind_end = dind_start + ptrs * ptrs;
                    if from >= dind_end {
                        any_l1 = true;
                        continue;
                    }
                    let mut l2 = self.read_indirect_block(dind)?;
                    let mut any_l2 = false;
                    for j in 0..ptrs as usize {
                        let sub = read_u32_le(&l2, j * 4);
                        if sub == 0 {
                            continue;
                        }
                        let sub_start = dind_start + (j as u32) * ptrs;
                        let sub_end = sub_start + ptrs;
                        if from >= sub_end {
                            any_l2 = true;
                            continue;
                        }
                        let first = if from > sub_start {
                            (from - sub_start) as usize
                        } else {
                            0
                        };
                        let mut inner = self.read_indirect_block(sub)?;
                        let mut any_inner = false;
                        for k in 0..ptrs as usize {
                            let p = read_u32_le(&inner, k * 4);
                            if k >= first {
                                if p != 0 {
                                    self.ext.free_block(p);
                                    write_u32_le(&mut inner, k * 4, 0);
                                }
                            } else if p != 0 {
                                any_inner = true;
                            }
                        }
                        if any_inner {
                            self.write_indirect_block(sub, &inner)?;
                            any_l2 = true;
                        } else {
                            self.ext.free_block(sub);
                            write_u32_le(&mut l2, j * 4, 0);
                        }
                    }
                    if any_l2 {
                        self.write_indirect_block(dind, &l2)?;
                        any_l1 = true;
                    } else {
                        self.ext.free_block(dind);
                        write_u32_le(&mut l1, i * 4, 0);
                    }
                }
                if any_l1 {
                    self.write_indirect_block(tind, &l1)?;
                } else {
                    self.ext.free_block(tind);
                    self.staged_inode_mut().block[IDX_TRIPLE_INDIRECT] = 0;
                }
            }
        }
        Ok(())
    }

    /// Overwrite an entire data block with zeros. Used right after
    /// alloc_data_block so the next RMW reads a deterministic value.
    fn zero_block_on_disk(&mut self, blk: u32) -> Result<()> {
        let bs = self.ext.layout.block_size as u64;
        let zeros = vec![0u8; bs as usize];
        self.dev.write_at(blk as u64 * bs, &zeros)?;
        Ok(())
    }

    /// Write `data` at byte offset `pos` of the file. Performs RMW on
    /// blocks where the write doesn't cover the whole block; whole-block
    /// writes go straight to the device. Allocates missing blocks via
    /// the indirect tree.
    fn write_at_pos(&mut self, pos: u64, data: &[u8]) -> Result<u64> {
        if data.is_empty() {
            return Ok(0);
        }
        let bs = self.ext.layout.block_size as u64;
        let mut written = 0u64;
        let mut cur_pos = pos;
        while written < data.len() as u64 {
            let n = (cur_pos / bs) as u32;
            let off_in_block = (cur_pos % bs) as usize;
            let space = bs as usize - off_in_block;
            let to_write = (data.len() - written as usize).min(space);

            let blk = self.get_or_alloc_block(n)?;
            let abs = blk as u64 * bs;

            if off_in_block == 0 && to_write == bs as usize {
                // Whole-block write.
                self.dev
                    .write_at(abs, &data[written as usize..written as usize + to_write])?;
            } else {
                // RMW.
                let mut buf = vec![0u8; bs as usize];
                self.dev.read_at(abs, &mut buf)?;
                buf[off_in_block..off_in_block + to_write]
                    .copy_from_slice(&data[written as usize..written as usize + to_write]);
                self.dev.write_at(abs, &buf)?;
            }
            written += to_write as u64;
            cur_pos += to_write as u64;
        }
        // Update inode size if we extended past the previous EOF.
        if cur_pos > self.len {
            self.set_inode_size(cur_pos)?;
        }
        Ok(written)
    }

    /// Read up to `out.len()` bytes from `pos`. Holes read back as zeros.
    fn read_at_pos(&mut self, pos: u64, out: &mut [u8]) -> Result<usize> {
        if pos >= self.len {
            return Ok(0);
        }
        let bs = self.ext.layout.block_size as u64;
        let remaining_in_file = self.len - pos;
        let mut read = 0usize;
        let max = (out.len() as u64).min(remaining_in_file) as usize;
        let mut cur_pos = pos;
        while read < max {
            let n = (cur_pos / bs) as u32;
            let off_in_block = (cur_pos % bs) as usize;
            let space = bs as usize - off_in_block;
            let to_read = (max - read).min(space);

            let blk = self.read_logical_block(n)?;
            if blk == 0 {
                // Hole → zeros.
                out[read..read + to_read].fill(0);
            } else {
                let mut buf = vec![0u8; bs as usize];
                self.dev.read_at(blk as u64 * bs, &mut buf)?;
                out[read..read + to_read]
                    .copy_from_slice(&buf[off_in_block..off_in_block + to_read]);
            }
            read += to_read;
            cur_pos += to_read as u64;
        }
        Ok(read)
    }

    /// Grow the file to `new_len`, allocating new blocks (zero-filled) as
    /// needed. Existing content is preserved. Updates `i_size`.
    fn grow_to(&mut self, new_len: u64) -> Result<()> {
        let bs = self.ext.layout.block_size as u64;
        let old_len = self.len;
        if new_len <= old_len {
            return Ok(());
        }
        // If the old length didn't fill its last block, the trailing
        // bytes of that block must read as zero. The freshly-allocated
        // tail blocks are already zeroed by `zero_block_on_disk`, but
        // the *existing* tail block needs a partial zero-fill.
        if !old_len.is_multiple_of(bs) {
            let last_n = (old_len / bs) as u32;
            let blk = self.read_logical_block(last_n)?;
            if blk != 0 {
                let off = (old_len % bs) as usize;
                let mut buf = vec![0u8; bs as usize];
                self.dev.read_at(blk as u64 * bs, &mut buf)?;
                for b in &mut buf[off..] {
                    *b = 0;
                }
                self.dev.write_at(blk as u64 * bs, &buf)?;
            }
        }
        // Make sure all logical blocks up through new_len-1 exist.
        let last_needed = if new_len == 0 {
            0
        } else {
            ((new_len - 1) / bs) as u32
        };
        let first_to_alloc = old_len.div_ceil(bs) as u32;
        for n in first_to_alloc..=last_needed {
            // Allocate (and zero) the block. We don't care about its
            // physical number — the indirect tree records it.
            let _ = self.get_or_alloc_block(n)?;
        }
        self.set_inode_size(new_len)?;
        Ok(())
    }

    /// Shrink the file to `new_len`. Frees blocks past the new tail.
    fn shrink_to(&mut self, new_len: u64) -> Result<()> {
        let bs = self.ext.layout.block_size as u64;
        if new_len >= self.len {
            return Ok(());
        }
        // First block whose logical index is wholly outside the new
        // length, i.e. `ceil(new_len / bs)`.
        let from = new_len.div_ceil(bs) as u32;
        self.free_blocks_from(from)?;
        self.set_inode_size(new_len)?;
        Ok(())
    }
}

impl<'a> Read for Ext2FileHandle<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.read_at_pos(self.pos, buf).map_err(io::Error::other)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl<'a> Write for Ext2FileHandle<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.write_at_pos(self.pos, buf).map_err(io::Error::other)?;
        self.pos += n;
        Ok(n as usize)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> Seek for Ext2FileHandle<'a> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(p) => p as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
            SeekFrom::End(d) => self.len as i128 + d as i128,
        };
        if new_pos < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek to negative offset",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

impl<'a> FileHandle for Ext2FileHandle<'a> {
    fn len(&self) -> u64 {
        self.len
    }

    fn set_len(&mut self, new_len: u64) -> Result<()> {
        if new_len > self.len {
            self.grow_to(new_len)?;
        } else if new_len < self.len {
            self.shrink_to(new_len)?;
        }
        self.recompute_blocks_512()?;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.recompute_blocks_512()?;
        self.ext.flush(self.dev)
    }
}

impl<'a> Drop for Ext2FileHandle<'a> {
    fn drop(&mut self) {
        // Best-effort: refresh blocks_512 so a subsequent flush sees the
        // right sector count. Don't propagate errors from Drop.
        let _ = self.recompute_blocks_512();
    }
}

/// Open a regular file for read/write at `path`. Accepts ext2/3/4 images.
/// If the image has a JBD2 journal with committed-but-not-checkpointed
/// transactions (`s_start != 0`), they are replayed onto the filesystem
/// before the handle attaches. Returns [`crate::Error::Unsupported`] if:
/// - the image is a journal device (`INCOMPAT_JOURNAL_DEV`)
/// - the inode's extent tree has `depth > 1` (depth-0 and depth-1 are
///   supported; multi-level trees with two or more idx levels are not
///   yet implemented).
pub(crate) fn open_file_rw_ext<'a>(
    ext: &'a mut Ext,
    dev: &'a mut dyn BlockDevice,
    path: &std::path::Path,
    flags: crate::fs::OpenFlags,
    meta: Option<crate::fs::FileMeta>,
) -> Result<Box<dyn FileHandle + 'a>> {
    let fi = ext.sb.feature_incompat;
    if fi & constants::feature::INCOMPAT_JOURNAL_DEV != 0 {
        return Err(crate::Error::Unsupported(
            "ext: image is an external journal device — partial writes not applicable".into(),
        ));
    }
    // Replay any committed-but-not-checkpointed transactions before we
    // start mutating in-place. A clean journal (s_start == 0) is a
    // no-op replay; an interrupted commit leaves data blocks in the log
    // ring that we apply here.
    if ext.sb.feature_compat & constants::feature::COMPAT_HAS_JOURNAL != 0 {
        let replayed = jbd2::replay_journal(ext, dev)?;
        if replayed {
            // Reload bitmaps and the in-memory group descriptors from
            // disk: replay rewrote them on the device but our in-memory
            // copy may have been stamped by an earlier flush.
            ext.reload_groups_from_disk(dev)?;
        }
    }

    let path_str = path
        .to_str()
        .ok_or_else(|| crate::Error::InvalidArgument(format!("ext: non-UTF-8 path {path:?}")))?;

    // Resolve or create.
    let ino = match ext.path_to_inode(dev, path_str) {
        Ok(ino) => {
            // Existing file — verify it's a regular file.
            let inode = ext.read_inode(dev, ino)?;
            if inode.mode & S_IFMT != S_IFREG {
                return Err(crate::Error::InvalidArgument(format!(
                    "ext: {path_str} is not a regular file"
                )));
            }
            ino
        }
        Err(_) if flags.create => {
            let meta = meta.ok_or_else(|| {
                crate::Error::InvalidArgument(
                    "ext: open_file_rw with create=true requires meta".into(),
                )
            })?;
            // Create an empty file via add_file_to_streaming, which
            // wires up the dir entry and stages the inode.
            let (parent, name) = super::split_path(path)?;
            let parent_str = parent.to_str().ok_or_else(|| {
                crate::Error::InvalidArgument("ext: non-UTF-8 parent path".into())
            })?;
            let parent_ino = ext.path_to_inode(dev, parent_str)?;
            let mut empty = std::io::Cursor::new(Vec::<u8>::new());
            ext.add_file_to_streaming(dev, parent_ino, name.as_bytes(), &mut empty, 0, meta)?
        }
        Err(e) => return Err(e),
    };

    let inode = ext.read_inode(dev, ino)?;
    let mut len = inode.file_size();

    let mut handle = Ext2FileHandle::new(ext, dev, ino, len)?;
    if flags.truncate && len > 0 {
        handle.shrink_to(0)?;
        handle.recompute_blocks_512()?;
        len = 0;
    }
    if flags.append {
        handle.pos = len;
    }
    Ok(Box::new(handle))
}

/// Open an already-resolved inode for read/write without traversing
/// the path namespace. Used by the FUSE adapter (which receives inode
/// numbers from the kernel) and by other consumers that already know
/// the inode. Same journal-replay + extent-depth gating as
/// `open_file_rw_ext`.
pub fn open_file_rw_ext_by_inode<'a>(
    ext: &'a mut Ext,
    dev: &'a mut dyn BlockDevice,
    ino: u32,
) -> Result<Ext2FileHandle<'a>> {
    let fi = ext.sb.feature_incompat;
    if fi & constants::feature::INCOMPAT_JOURNAL_DEV != 0 {
        return Err(crate::Error::Unsupported(
            "ext: image is an external journal device — partial writes not applicable".into(),
        ));
    }
    if ext.sb.feature_compat & constants::feature::COMPAT_HAS_JOURNAL != 0 {
        let replayed = jbd2::replay_journal(ext, dev)?;
        if replayed {
            ext.reload_groups_from_disk(dev)?;
        }
    }
    let inode = ext.read_inode(dev, ino)?;
    if inode.mode & S_IFMT != S_IFREG {
        return Err(crate::Error::InvalidArgument(format!(
            "ext: inode {ino} is not a regular file (mode={:#o})",
            inode.mode
        )));
    }
    let len = inode.file_size();
    Ext2FileHandle::new(ext, dev, ino, len)
}

#[inline]
fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

#[inline]
fn write_u32_le(buf: &mut [u8], off: usize, val: u32) {
    buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
}
