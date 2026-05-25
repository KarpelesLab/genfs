//! XFS write path — per-AG bump-pointer allocator + inode helpers.
//!
//! After [`super::format::format`] returns, the on-disk image is a
//! valid v5 filesystem with the root inode chunk allocated in AG 0
//! (root + rbmino + rsumino) and the kernel journal stub already
//! written. This module adds methods on [`Xfs`] for populating the
//! volume with files, directories, symlinks, special files, and
//! extended attributes — plus the `remove` companion for tearing
//! entries down.
//!
//! - **Allocator.** Per-AG bump pointer + per-AG freed-extent list.
//!   `alloc_blocks_in_any_ag` round-robins across AGs, preferring a
//!   freed extent of exact size before bumping the pointer. Freed
//!   space comes from [`Xfs::remove`]. The BNO / CNT trees are
//!   re-emitted with all the AG's current extents at
//!   [`Xfs::flush_writes`] time, coalescing adjacent ranges.
//!
//! - **Inodes.** Multi-chunk INOBT, multi-AG hopping. Allocation tries
//!   every existing chunk across AGs in round-robin order; if every
//!   chunk is full, it allocates a fresh 64-inode chunk in the next AG
//!   with space. [`Xfs::remove`] frees its inode back to the chunk
//!   bitmap and zeroes the on-disk slot.
//!
//! - **Directories.** Every directory is initialised as block-format
//!   (`encode_v5_block_dir`); growing past one directory block
//!   returns `Error::Unsupported`. The root directory is always
//!   block-format. (Promotion to leaf or node format is future work.)
//!
//! - **Symlinks.** ≤ 336 bytes → local (inline). 337 .. 4096 → remote,
//!   one v5 XSLM-headered block. Longer is `Error::Unsupported`.
//!
//! - **Files.** Streamed through a 64 KiB scratch buffer; allocated as
//!   one contiguous extent in whichever AG the round-robin picks.
//!
//! - **Xattrs.** [`Xfs::add_xattr`] / [`Xfs::remove_xattr`] /
//!   [`Xfs::read_xattrs`] support both **shortform** (inline LOCAL
//!   attribute fork) and **leaf form** (one full FS-block leaf, via
//!   [`super::xattr_leaf`]). The writer picks the cheapest form that
//!   fits; promotion shortform → leaf happens transparently when the
//!   inline area overflows. Node-form (multi-leaf dabtree) is not
//!   implemented and surfaces `Error::Unsupported`.
//!
//! ## What's not done
//!
//! - Reverse-mapping B+tree, reference-count B+tree, FINOBT — disabled.
//! - Sparse inodes — disabled.
//! - Journal — only the empty-unmount-record stub is written; the
//!   kernel may emit a `dirty log` warning on first mount because we
//!   don't stamp `h_crc` in the record header. Mounts still succeed.
//! - Mixed-format directories (leaf, node, B+tree on the write side) —
//!   only block-format is emitted.
//! - Hard links — every directory entry points at a freshly allocated
//!   inode; calling `add_file` twice with the same target creates two
//!   independent inodes.

use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::dir_batch::{DEFAULT_CAPACITY, DirBatch};

use super::Xfs;
use super::bmbt::Extent;
use super::dir::{
    XFS_DIR3_FT_BLKDEV, XFS_DIR3_FT_CHRDEV, XFS_DIR3_FT_DIR, XFS_DIR3_FT_FIFO,
    XFS_DIR3_FT_REG_FILE, XFS_DIR3_FT_SOCK, XFS_DIR3_FT_SYMLINK, encode_v5_block_dir,
    stamp_v5_dir_block_crc,
};
use super::format::{
    AG0_METADATA_BLOCKS, INODE_CHUNK_ALIGN, LOG_AGBLOCK, ROOT_CHUNK_AGBLOCK, XFS_ABTB_CRC_MAGIC,
    XFS_ABTC_CRC_MAGIC, XFS_BLOCKSIZE, XFS_BTREE_SBLOCK_V5_SIZE, XFS_IBT_CRC_MAGIC,
    XFS_INODES_PER_CHUNK, XFS_INODESIZE, XFS_INOPBLOCK, stamp_v5_btree_block_crc,
};
use super::inode::{
    S_IFBLK, S_IFCHR, S_IFDIR, S_IFIFO, S_IFLNK, S_IFREG, S_IFSOCK, V3DinodeBuilder, XfsTimestamp,
    stamp_v3_inode_crc,
};
use super::journal::DEFAULT_LOG_BLOCKS;
use super::symlink::XFS_SYMLINK_HDR_SIZE;

/// Streaming-write scratch buffer size — never grow this above 64 KiB.
pub const SCRATCH_SIZE: usize = 64 * 1024;

/// Special-file kind for [`Xfs::add_device`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Char,
    Block,
    Fifo,
    Socket,
}

impl DeviceKind {
    fn s_ifmt(self) -> u16 {
        match self {
            Self::Char => S_IFCHR,
            Self::Block => S_IFBLK,
            Self::Fifo => S_IFIFO,
            Self::Socket => S_IFSOCK,
        }
    }
    fn ftype(self) -> u8 {
        match self {
            Self::Char => XFS_DIR3_FT_CHRDEV,
            Self::Block => XFS_DIR3_FT_BLKDEV,
            Self::Fifo => XFS_DIR3_FT_FIFO,
            Self::Socket => XFS_DIR3_FT_SOCK,
        }
    }
}

/// Per-entry metadata supplied with each write call. Mirrors the shape
/// of the crate-wide `FileMeta`, but kept module-local so this writer
/// is self-contained.
#[derive(Debug, Clone, Copy)]
pub struct EntryMeta {
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub mtime: u32,
    pub atime: u32,
    pub ctime: u32,
}

impl Default for EntryMeta {
    fn default() -> Self {
        Self {
            mode: 0o644,
            uid: 0,
            gid: 0,
            mtime: 0,
            atime: 0,
            ctime: 0,
        }
    }
}

impl EntryMeta {
    fn ts(&self) -> (XfsTimestamp, XfsTimestamp, XfsTimestamp) {
        (
            XfsTimestamp {
                sec: self.atime,
                nsec: 0,
            },
            XfsTimestamp {
                sec: self.mtime,
                nsec: 0,
            },
            XfsTimestamp {
                sec: self.ctime,
                nsec: 0,
            },
        )
    }
}

/// In-memory state of an active inode chunk within a single AG.
#[derive(Debug, Clone)]
struct InodeChunk {
    /// AG-relative inode number of slot 0.
    startino_ag: u32,
    /// Bitmap: bit i = inode i is free.
    ir_free: u64,
    /// AG-relative block where the chunk lives (`startino_ag >>
    /// inopblog`). Currently informational — kept so future code that
    /// physically zeroes the chunk on allocation can find it.
    #[allow(dead_code)]
    agblock: u32,
}

impl InodeChunk {
    fn alloc(&mut self) -> Option<u32> {
        for i in 0..64 {
            let mask = 1u64 << i;
            if (self.ir_free & mask) != 0 {
                self.ir_free &= !mask;
                return Some(self.startino_ag + i);
            }
        }
        None
    }

    fn freecount(&self) -> u32 {
        self.ir_free.count_ones()
    }

    fn free_slot(&mut self, slot: u32) -> bool {
        if slot >= 64 {
            return false;
        }
        let mask = 1u64 << slot;
        if self.ir_free & mask != 0 {
            return false; // already free
        }
        self.ir_free |= mask;
        true
    }
}

/// Per-AG slice of the write state. Each AG has its own bump pointer,
/// inode-chunk list, and freed-extent list (returned to BNO/CNT on
/// flush).
#[derive(Debug, Clone)]
struct AgState {
    /// Bump pointer (AG-relative block); points at the next FS block
    /// the linear allocator hasn't handed out yet.
    next_agblock: u32,
    /// All inode chunks ever allocated in this AG.
    chunks: Vec<InodeChunk>,
    /// `(startblock, blockcount)` pairs of extents that were released
    /// by [`Xfs::remove`] and are available for reuse by future
    /// allocations in the same AG. Reclaim order is "newest first" —
    /// a fresh allocation tries the most recently freed extent before
    /// bumping the pointer.
    freed_extents: Vec<(u32, u32)>,
}

impl AgState {
    fn freecount_inodes(&self) -> u32 {
        self.chunks.iter().map(|c| c.freecount()).sum()
    }
    fn usedcount_inodes(&self) -> u32 {
        self.chunks.iter().map(|c| 64 - c.freecount()).sum()
    }
}

/// Side-table of write-only state. Lives in [`Xfs`] behind an `Option`
/// because read-only opens don't need it.
#[derive(Debug, Clone)]
pub struct WriteState {
    /// One [`AgState`] per allocation group.
    ags: Vec<AgState>,
    /// AG index to try first on the next inode allocation
    /// (round-robin).
    next_inode_ag: u32,
    /// AG index to try first on the next block allocation
    /// (round-robin).
    next_block_ag: u32,
    /// Default 16-byte UUID stamped into new metadata blocks. We pull
    /// the canonical UUID from the superblock at write-time instead,
    /// so this field is informational. Kept for future per-AG UUIDs.
    #[allow(dead_code)]
    uuid: [u8; 16],
    /// Total inodes allocated so far across all AGs.
    inodes_used: u64,
    /// Outstanding inodes free across all chunks across all AGs.
    inodes_free: u64,
    /// Pending directory entries `(name, child_ino, ftype)` keyed by
    /// parent directory inode. Staged instead of rewriting the parent's
    /// single dir block + inode on every child (O(N²) for a directory of
    /// N children). Serialized once on eviction or at flush; path lookups
    /// read it as an in-memory overlay so staged children stay visible.
    dir_batch: DirBatch<u64, (String, u64, u8)>,
}

impl WriteState {
    /// Build the write state corresponding to a freshly-formatted
    /// `agcount`-AG image. AG 0 starts with the root inode chunk and
    /// the bump pointer right after the log + first dir block; every
    /// other AG starts with no inode chunks and the bump pointer at
    /// the end of static metadata.
    pub fn initial(uuid: [u8; 16], agcount: u32) -> Self {
        let log_blocks = DEFAULT_LOG_BLOCKS;
        let mut ags = Vec::with_capacity(agcount as usize);
        for ag in 0..agcount {
            let (next_agblock, chunks) = if ag == 0 {
                (
                    // Free pool starts right after the log.
                    LOG_AGBLOCK + log_blocks,
                    vec![InodeChunk {
                        startino_ag: ROOT_CHUNK_AGBLOCK * (XFS_INOPBLOCK as u32),
                        // Slots 0 (root), 1 (rbmino), 2 (rsumino) are
                        // pre-allocated by the formatter; the rest are
                        // free.
                        ir_free: !0b111u64,
                        agblock: ROOT_CHUNK_AGBLOCK,
                    }],
                )
            } else {
                (AG0_METADATA_BLOCKS, Vec::new())
            };
            ags.push(AgState {
                next_agblock,
                chunks,
                freed_extents: Vec::new(),
            });
        }
        Self {
            ags,
            next_inode_ag: 0,
            next_block_ag: 0,
            uuid,
            inodes_used: 3,
            inodes_free: 61,
            dir_batch: DirBatch::new(DEFAULT_CAPACITY),
        }
    }

    /// Single-AG convenience wrapper (back-compat with previous API).
    pub fn initial_single_ag(uuid: [u8; 16]) -> Self {
        Self::initial(uuid, 1)
    }
}

impl Xfs {
    /// Initialise the in-memory write state assuming a freshly-formatted
    /// image (AG 0 has the root inode chunk pre-allocated; no other
    /// chunks anywhere; bump pointer right after the log in AG 0 and
    /// right after static metadata in every other AG). For images that
    /// already contain files, use [`Self::resume_writes`] instead — it
    /// reconstructs the write state by reading the on-disk AGF / AGI /
    /// INOBT / BNO headers.
    pub fn begin_writes(&mut self, uuid: [u8; 16]) {
        let agcount = self.sb.agcount.max(1);
        self.write_state = Some(WriteState::initial(uuid, agcount));
    }

    /// Reconstruct the in-memory write state from the on-disk AG
    /// headers — the inverse of [`Self::flush_writes`]. Reads each AG's
    /// AGI (for the INOBT root pointer + free-inode count), the INOBT
    /// leaf (for the list of inode chunks + each chunk's `ir_free`
    /// bitmap), the AGF (for the BNO root pointer + freeblks), and the
    /// BNO leaf (for the list of free-space extents). The free extent
    /// whose end abuts `agf_length` becomes the bump pointer's tail;
    /// all other free extents are re-attached to the AG's
    /// `freed_extents` list so future allocations may reuse them.
    ///
    /// **Limitations.** Single-leaf B+trees only (level == 0): if any
    /// AGI's INOBT has a `level != 0` root, or any AGF's BNO does, the
    /// image was produced by something other than this writer and we
    /// refuse with [`crate::Error::Unsupported`]. The classic 16-byte
    /// inobt record (no sparse-inode `holemask`/`count` half-words) is
    /// what the writer emits, so that's what we decode.
    pub fn resume_writes(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        let agblocks = self.sb.agblocks;
        let total_blocks = self.sb.dblocks as u32;
        let bs = self.sb.blocksize as u64;
        let sect = super::format::XFS_SECTSIZE as u64;
        let agcount = self.sb.agcount.max(1);
        let uuid = self.sb.uuid;

        let mut ags = Vec::with_capacity(agcount as usize);
        let mut inodes_used: u64 = 0;
        let mut inodes_free: u64 = 0;

        for ag in 0..agcount {
            let ag_byte = (ag as u64) * (agblocks as u64) * bs;
            let this_ag_blocks = if ag == agcount - 1 {
                total_blocks.saturating_sub(ag * agblocks).max(1)
            } else {
                agblocks
            };

            // -- AGI (sector 2): pull the INOBT root pointer + sanity-check magic.
            let mut agi = vec![0u8; sect as usize];
            dev.read_at(ag_byte + 2 * sect, &mut agi)?;
            let agi_magic = u32::from_be_bytes(agi[0..4].try_into().unwrap());
            if agi_magic != super::format::XFS_AGI_MAGIC {
                return Err(crate::Error::InvalidImage(format!(
                    "xfs: resume_writes: ag {ag} AGI magic {agi_magic:#010x} (expected XAGI)"
                )));
            }
            let inobt_root = u32::from_be_bytes(agi[20..24].try_into().unwrap());
            let inobt_level = u32::from_be_bytes(agi[24..28].try_into().unwrap());
            // Some flush_writes paths stamp `level = 1` even when the
            // leaf is empty (matches the formatter's behaviour). The
            // root we read is still a leaf; the "level" field in AGI is
            // really "number of B+tree levels above the root". Accept
            // values <= 1.
            if inobt_level > 1 {
                return Err(crate::Error::Unsupported(format!(
                    "xfs: resume_writes: ag {ag} INOBT has {inobt_level} levels (writer only emits level 0)"
                )));
            }

            // -- INOBT leaf: read every chunk record.
            let mut inobt = vec![0u8; bs as usize];
            dev.read_at(ag_byte + (inobt_root as u64) * bs, &mut inobt)?;
            let inobt_magic = u32::from_be_bytes(inobt[0..4].try_into().unwrap());
            if inobt_magic != super::format::XFS_IBT_CRC_MAGIC {
                return Err(crate::Error::InvalidImage(format!(
                    "xfs: resume_writes: ag {ag} INOBT magic {inobt_magic:#010x} (expected IAB3)"
                )));
            }
            let inobt_block_level = u16::from_be_bytes(inobt[4..6].try_into().unwrap());
            if inobt_block_level != 0 {
                return Err(crate::Error::Unsupported(format!(
                    "xfs: resume_writes: ag {ag} INOBT root block is level {inobt_block_level} (need leaf, level 0)"
                )));
            }
            let inobt_numrecs = u16::from_be_bytes(inobt[6..8].try_into().unwrap());
            let mut chunks = Vec::with_capacity(inobt_numrecs as usize);
            for i in 0..inobt_numrecs as usize {
                let off = XFS_BTREE_SBLOCK_V5_SIZE + i * 16;
                if off + 16 > inobt.len() {
                    return Err(crate::Error::InvalidImage(format!(
                        "xfs: resume_writes: ag {ag} INOBT record {i} overflows leaf block"
                    )));
                }
                let startino_ag = u32::from_be_bytes(inobt[off..off + 4].try_into().unwrap());
                let freecount = u32::from_be_bytes(inobt[off + 4..off + 8].try_into().unwrap());
                let ir_free = u64::from_be_bytes(inobt[off + 8..off + 16].try_into().unwrap());
                if ir_free.count_ones() != freecount {
                    return Err(crate::Error::InvalidImage(format!(
                        "xfs: resume_writes: ag {ag} chunk @ {startino_ag}: freecount {freecount} disagrees with ir_free popcount {}",
                        ir_free.count_ones()
                    )));
                }
                let inopblog = self.sb.inopblog as u32;
                let agblock = startino_ag >> inopblog;
                inodes_used += (64 - freecount) as u64;
                inodes_free += freecount as u64;
                chunks.push(InodeChunk {
                    startino_ag,
                    ir_free,
                    agblock,
                });
            }

            // -- AGF (sector 1): pull the BNO root pointer.
            let mut agf = vec![0u8; sect as usize];
            dev.read_at(ag_byte + sect, &mut agf)?;
            let agf_magic = u32::from_be_bytes(agf[0..4].try_into().unwrap());
            if agf_magic != super::format::XFS_AGF_MAGIC {
                return Err(crate::Error::InvalidImage(format!(
                    "xfs: resume_writes: ag {ag} AGF magic {agf_magic:#010x} (expected XAGF)"
                )));
            }
            let bno_root = u32::from_be_bytes(agf[16..20].try_into().unwrap());
            let bno_level = u32::from_be_bytes(agf[28..32].try_into().unwrap());
            if bno_level > 1 {
                return Err(crate::Error::Unsupported(format!(
                    "xfs: resume_writes: ag {ag} BNO has {bno_level} levels (writer only emits level 0)"
                )));
            }

            // -- BNO leaf: read every free-extent record.
            let mut bno = vec![0u8; bs as usize];
            dev.read_at(ag_byte + (bno_root as u64) * bs, &mut bno)?;
            let bno_magic = u32::from_be_bytes(bno[0..4].try_into().unwrap());
            if bno_magic != super::format::XFS_ABTB_CRC_MAGIC {
                return Err(crate::Error::InvalidImage(format!(
                    "xfs: resume_writes: ag {ag} BNO magic {bno_magic:#010x} (expected AB3B)"
                )));
            }
            let bno_block_level = u16::from_be_bytes(bno[4..6].try_into().unwrap());
            if bno_block_level != 0 {
                return Err(crate::Error::Unsupported(format!(
                    "xfs: resume_writes: ag {ag} BNO root block is level {bno_block_level} (need leaf, level 0)"
                )));
            }
            let bno_numrecs = u16::from_be_bytes(bno[6..8].try_into().unwrap());
            let mut free_extents: Vec<(u32, u32)> = Vec::with_capacity(bno_numrecs as usize);
            for i in 0..bno_numrecs as usize {
                let off = XFS_BTREE_SBLOCK_V5_SIZE + i * 8;
                if off + 8 > bno.len() {
                    return Err(crate::Error::InvalidImage(format!(
                        "xfs: resume_writes: ag {ag} BNO record {i} overflows leaf block"
                    )));
                }
                let startblock = u32::from_be_bytes(bno[off..off + 4].try_into().unwrap());
                let blockcount = u32::from_be_bytes(bno[off + 4..off + 8].try_into().unwrap());
                free_extents.push((startblock, blockcount));
            }

            // The tail-end free extent (one whose end abuts this AG's
            // length) is the bump-pointer tail; everything else is
            // recoverable through freed_extents. If no tail extent
            // exists the AG is full to the brim — pin next_agblock at
            // `this_ag_blocks`.
            let mut next_agblock = this_ag_blocks;
            let mut freed_extents: Vec<(u32, u32)> = Vec::new();
            for (s, c) in &free_extents {
                let end = s.saturating_add(*c);
                if end == this_ag_blocks {
                    next_agblock = *s;
                } else {
                    freed_extents.push((*s, *c));
                }
            }
            // If we never found a tail-aligned extent but there were
            // free extents, every one of them is a recoverable hole.
            // `next_agblock = this_ag_blocks` is still correct in that
            // case (no room past the last allocation).
            ags.push(AgState {
                next_agblock,
                chunks,
                freed_extents,
            });
        }

        self.write_state = Some(WriteState {
            ags,
            next_inode_ag: 0,
            next_block_ag: 0,
            uuid,
            inodes_used,
            inodes_free,
            dir_batch: DirBatch::new(DEFAULT_CAPACITY),
        });
        Ok(())
    }

    /// Ensure `write_state` is populated. If a previous [`format()`] or
    /// [`begin_writes`](Self::begin_writes)/[`resume_writes`](Self::resume_writes)
    /// call set it up already, this is a no-op; otherwise it reads the
    /// on-disk AG headers via [`Self::resume_writes`]. Called from the
    /// path-based mutators so that `Xfs::open` followed by `create_file`
    /// works without an explicit kickoff call.
    fn ensure_write_state(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        if self.write_state.is_none() {
            self.resume_writes(dev)?;
        }
        Ok(())
    }

    fn ws_mut(&mut self) -> Result<&mut WriteState> {
        self.write_state.as_mut().ok_or_else(|| {
            crate::Error::InvalidArgument(
                "xfs: write methods called before begin_writes() / format()".into(),
            )
        })
    }

    /// Allocate `n` contiguous AG blocks. Picks the next AG in
    /// round-robin order; falls back to other AGs if the current one
    /// is full. Reuses a recently-freed extent of the exact size if
    /// available. Returns `(ag, agblock_start)` — the caller is
    /// responsible for translating to FSB / device byte.
    fn alloc_blocks_in_any_ag(&mut self, n: u32) -> Result<(u32, u32)> {
        self.alloc_blocks_in_any_ag_aligned(n, 1)
    }

    /// Like [`alloc_blocks_in_any_ag`] but forces the start block to a
    /// multiple of `align` FS blocks. The skipped alignment gap is returned
    /// to the AG's freed-extent list so it stays accounted as free (and
    /// reusable). Used for inode chunks, which must satisfy `sb_inoalignmt`.
    fn alloc_blocks_in_any_ag_aligned(&mut self, n: u32, align: u32) -> Result<(u32, u32)> {
        let agblocks = self.sb.agblocks;
        let ws = self.ws_mut()?;
        let agcount = ws.ags.len() as u32;
        // Try each AG starting at `next_block_ag`, round-robin.
        for offset in 0..agcount {
            let ag = (ws.next_block_ag + offset) % agcount;
            let ag_state = &mut ws.ags[ag as usize];
            // Prefer a freed extent of exact size that satisfies alignment.
            if let Some(idx) = ag_state
                .freed_extents
                .iter()
                .position(|(s, c)| *c == n && s % align == 0)
            {
                let (start, _) = ag_state.freed_extents.swap_remove(idx);
                ws.next_block_ag = (ag + 1) % agcount;
                return Ok((ag, start));
            }
            // Otherwise bump-allocate, rounding the start up to `align`.
            let next = ag_state.next_agblock;
            let aligned = next.next_multiple_of(align);
            if aligned.checked_add(n).is_some_and(|end| end <= agblocks) {
                if aligned > next {
                    // Reclaim the alignment gap as free space.
                    ag_state.freed_extents.push((next, aligned - next));
                }
                ag_state.next_agblock = aligned + n;
                ws.next_block_ag = (ag + 1) % agcount;
                return Ok((ag, aligned));
            }
        }
        Err(crate::Error::InvalidArgument(format!(
            "xfs: out of space across all {agcount} AGs (requested {n} blocks)"
        )))
    }

    /// Allocate `n` contiguous AG blocks and return the **FSB** of the
    /// first allocated block (encodes the AG in the high bits via
    /// `agblklog`). All write-path callers funnel through this so
    /// `fsb_to_byte` yields the right device offset regardless of
    /// which AG was picked.
    pub(super) fn alloc_blocks_fsb(&mut self, n: u32) -> Result<u64> {
        let (ag, agblk) = self.alloc_blocks_in_any_ag(n)?;
        Ok(((ag as u64) << self.sb.agblklog as u32) | (agblk as u64))
    }

    /// Allocate one inode. Tries existing chunks across AGs in
    /// round-robin order; falls back to allocating a fresh 64-inode
    /// chunk when every existing chunk is full. Returns the absolute
    /// inode number.
    fn alloc_inode(&mut self, dev: &mut dyn BlockDevice) -> Result<u64> {
        let inopblog = self.sb.inopblog as u32;
        let agblklog = self.sb.agblklog as u32;
        // Round-robin across AGs, trying existing chunks first.
        let agcount = {
            let ws = self.ws_mut()?;
            ws.ags.len() as u32
        };
        for offset in 0..agcount {
            let try_result = {
                let ws = self.ws_mut()?;
                let ag = (ws.next_inode_ag + offset) % agcount;
                let ag_state = &mut ws.ags[ag as usize];
                let mut found = None;
                for chunk in &mut ag_state.chunks {
                    if let Some(rel) = chunk.alloc() {
                        found = Some(rel);
                        break;
                    }
                }
                found.map(|rel| (ag, rel))
            };
            if let Some((ag, rel)) = try_result {
                let ws = self.ws_mut()?;
                ws.inodes_used += 1;
                ws.inodes_free -= 1;
                ws.next_inode_ag = (ag + 1) % agcount;
                return Ok(((ag as u64) << (inopblog + agblklog)) | (rel as u64));
            }
        }
        // No existing chunk had room. Allocate a new chunk in the next
        // AG with space.
        let chunk_blocks = XFS_INODES_PER_CHUNK / (XFS_INOPBLOCK as u32);
        let (ag, agblk) = self.alloc_blocks_in_any_ag_aligned(chunk_blocks, INODE_CHUNK_ALIGN)?;
        let startino_ag = agblk * (XFS_INOPBLOCK as u32);
        // All 64 slots free; `alloc` takes slot 0. (A previous `!1` here
        // reserved slot 0 as *used* but it was never written, leaving a
        // zeroed inode the inode B-tree claimed was allocated — xfs_repair
        // flagged "bad magic, would clear".) Free slots stay zeroed, which
        // xfs_repair accepts as the free-inode pattern (matching the
        // formatter's zeroed root chunk).
        let mut chunk = InodeChunk {
            startino_ag,
            ir_free: !0u64,
            agblock: agblk,
        };
        let rel = chunk.alloc().expect("fresh chunk has 64 free inodes");
        {
            let ws = self.ws_mut()?;
            ws.ags[ag as usize].chunks.push(chunk);
            ws.inodes_used += 1;
            ws.inodes_free += 63;
            ws.next_inode_ag = (ag + 1) % (ws.ags.len() as u32);
        }
        // Initialise all 64 inode slots with valid (free, mode-0) v3 inodes:
        // XFS v5 verifies the magic + CRC of every inode in a chunk's
        // cluster, so leaving free slots zeroed makes xfs_repair report
        // "bad magic / CRC error". The caller overwrites slot 0 with the
        // real inode; later allocations overwrite the others.
        self.init_inode_chunk(dev, ag, startino_ag)?;
        Ok(((ag as u64) << (inopblog + agblklog)) | (rel as u64))
    }

    /// Stamp all 64 slots of a freshly allocated inode chunk with valid
    /// free (mode-0) v3 inodes in one 32 KiB write, so every inode in the
    /// chunk carries a correct magic, version, `di_next_unlinked` and CRC.
    fn init_inode_chunk(
        &mut self,
        dev: &mut dyn BlockDevice,
        ag: u32,
        startino_ag: u32,
    ) -> Result<()> {
        let inopblog = self.sb.inopblog as u32;
        let agblklog = self.sb.agblklog as u32;
        let isize = XFS_INODESIZE as usize;
        let uuid = self.uuid_for_writes();
        let zero = XfsTimestamp { sec: 0, nsec: 0 };
        let mut buf = vec![0u8; XFS_INODES_PER_CHUNK as usize * isize];
        for slot in 0..XFS_INODES_PER_CHUNK {
            let rel = startino_ag + slot;
            let ino = ((ag as u64) << (inopblog + agblklog)) | (rel as u64);
            let builder = V3DinodeBuilder {
                inodesize: isize,
                mode: 0, // free
                format: 0,
                uid: 0,
                gid: 0,
                nlink: 0,
                atime: zero,
                mtime: zero,
                ctime: zero,
                crtime: zero,
                size: 0,
                nblocks: 0,
                extsize: 0,
                nextents: 0,
                forkoff: 0,
                aformat: 0,
                flags: 0,
                generation: 0,
                di_ino: ino,
                uuid,
                flags2: 0,
            };
            let mut inode = builder.build();
            stamp_v3_inode_crc(&mut inode);
            let base = slot as usize * isize;
            buf[base..base + isize].copy_from_slice(&inode);
        }
        let first_ino = ((ag as u64) << (inopblog + agblklog)) | (startino_ag as u64);
        let off = self.ino_byte_offset(first_ino)?;
        dev.write_at(off, &buf)?;
        Ok(())
    }

    /// Free `n` blocks starting at FSB `start_fsb` back to its AG's
    /// freed-extent list. The next allocator pass may reuse them.
    pub(super) fn free_blocks_fsb(&mut self, start_fsb: u64, n: u32) -> Result<()> {
        let agblklog = self.sb.agblklog as u32;
        let ag = (start_fsb >> agblklog) as u32;
        let agblk = (start_fsb & ((1u64 << agblklog) - 1)) as u32;
        let ws = self.ws_mut()?;
        if (ag as usize) >= ws.ags.len() {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: free_blocks_fsb: ag {ag} out of range"
            )));
        }
        ws.ags[ag as usize].freed_extents.push((agblk, n));
        Ok(())
    }

    /// Free a previously allocated inode. Returns its (ag, slot)
    /// position for the caller's convenience.
    fn free_inode(&mut self, ino: u64) -> Result<(u32, u32)> {
        let inopblog = self.sb.inopblog as u32;
        let agblklog = self.sb.agblklog as u32;
        let slot_mask = (1u64 << inopblog) - 1;
        let agblk_mask = (1u64 << agblklog) - 1;
        let slot = (ino & slot_mask) as u32;
        let agrel_ino = ino & ((1u64 << (inopblog + agblklog)) - 1);
        let agrel_ino = agrel_ino as u32;
        let ag = (ino >> (inopblog + agblklog)) as u32;
        let _ = agblk_mask;
        let ws = self.ws_mut()?;
        if (ag as usize) >= ws.ags.len() {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: free_inode: ag {ag} out of range"
            )));
        }
        // Find the chunk that owns this ino (relative to AG).
        let ag_state = &mut ws.ags[ag as usize];
        for chunk in &mut ag_state.chunks {
            if agrel_ino >= chunk.startino_ag && agrel_ino < chunk.startino_ag + 64 {
                let slot_in_chunk = agrel_ino - chunk.startino_ag;
                if chunk.free_slot(slot_in_chunk) {
                    ws.inodes_used -= 1;
                    ws.inodes_free += 1;
                    return Ok((ag, slot));
                }
                return Err(crate::Error::InvalidArgument(format!(
                    "xfs: free_inode {ino}: slot already free"
                )));
            }
        }
        Err(crate::Error::InvalidImage(format!(
            "xfs: free_inode {ino}: no chunk owns this inode"
        )))
    }

    /// Write a v3 inode at `ino`'s slot, stamping the CRC. `literal` is
    /// the literal-area bytes (≤ inodesize - 176).
    fn write_inode(
        &mut self,
        dev: &mut dyn BlockDevice,
        ino: u64,
        builder: V3DinodeBuilder,
        literal: &[u8],
    ) -> Result<()> {
        let off = self.ino_byte_offset(ino)?;
        let mut buf = builder.build();
        let lit_max = (XFS_INODESIZE as usize) - 176;
        if literal.len() > lit_max {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: literal area {} > {lit_max}",
                literal.len()
            )));
        }
        buf[176..176 + literal.len()].copy_from_slice(literal);
        stamp_v3_inode_crc(&mut buf);
        dev.write_at(off, &buf)?;
        Ok(())
    }

    /// Read the parent-directory inode, append a new directory entry,
    /// re-encode the v5 block-format directory, and write it back. The
    /// caller has already allocated `child_ino`. Returns the updated
    /// parent's directory size in bytes.
    /// Stage a directory entry for `parent_ino` instead of rewriting the
    /// parent's dir block + inode immediately. The batch is serialized
    /// once — on eviction when a new directory enters a full cache, or at
    /// flush — and [`resolve_path`](super::Xfs::resolve_path) reads it as
    /// an overlay so staged children remain visible to path lookups.
    fn append_dir_entry(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u64,
        name: &str,
        child_ino: u64,
        ftype: u8,
    ) -> Result<()> {
        self.ensure_write_state(dev)?;
        let victim = self
            .ws_mut()?
            .dir_batch
            .stage(parent_ino, (name.to_string(), child_ino, ftype));
        if let Some((victim_ino, entries)) = victim {
            self.serialize_dir(dev, victim_ino, &entries)?;
        }
        Ok(())
    }

    /// Apply a directory's whole pending batch to its single on-disk dir
    /// block in one pass: decode the existing entries, append every
    /// staged entry, re-encode the block once, and rewrite the parent
    /// inode once (size + `nlink` bumped by the number of staged
    /// subdirectories). Equivalent to the old per-entry append repeated,
    /// but O(1) block rewrites instead of O(N).
    fn serialize_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u64,
        staged: &[(String, u64, u8)],
    ) -> Result<()> {
        if staged.is_empty() {
            return Ok(());
        }
        let (parent_buf, parent_core) = self.read_inode(dev, parent_ino)?;
        if !parent_core.is_dir() {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: parent inode {parent_ino} is not a directory"
            )));
        }
        let dir_block_size = self.sb.dir_block_size() as usize;

        // Existing on-disk extents (data + any leaf/free runs) and the
        // current entries (format-agnostic: block / leaf / node).
        let existing_extents = self.read_extent_list(dev, &parent_buf, &parent_core)?;
        let existing = self.read_dir_entries(dev, &parent_buf, &parent_core)?;
        let mut parent_parent = parent_ino;
        let mut merged: Vec<(String, u64, u8)> = Vec::with_capacity(existing.len() + staged.len());
        for e in existing {
            match e.name.as_str() {
                "." => {}
                ".." => parent_parent = e.inumber,
                _ => merged.push((e.name, e.inumber, e.ftype)),
            }
        }
        merged.extend(staged.iter().cloned());

        // Full record list including the synthetic "." / ".." at the front
        // (data block 0 carries them).
        let mut all: Vec<(String, u64, u8)> = Vec::with_capacity(merged.len() + 2);
        all.push((".".to_string(), parent_ino, XFS_DIR3_FT_DIR));
        all.push(("..".to_string(), parent_parent, XFS_DIR3_FT_DIR));
        all.extend(merged.iter().cloned());

        let plan = super::dir_build::plan_layout(&all, dir_block_size)?;

        let new_subdirs = staged
            .iter()
            .filter(|(_, _, ft)| *ft == XFS_DIR3_FT_DIR)
            .count() as u32;
        let nlink = parent_core.nlink + new_subdirs;
        let (atime, mtime, ctime) = (parent_core.atime, parent_core.mtime, parent_core.ctime);
        let uuid = self.uuid_for_writes();

        // ---- Block format: reuse the existing single dir block. ----
        if plan.format == super::dir_build::DirFormat::Block {
            if existing_extents.len() != 1 || existing_extents[0].blockcount != 1 {
                return Err(crate::Error::Unsupported(
                    "xfs: shrinking a promoted directory back to block format is unsupported"
                        .into(),
                ));
            }
            let self_extent = existing_extents[0];
            let phys_byte = self.fsb_to_byte(self_extent.startblock);
            let new_block = encode_v5_block_dir(
                dir_block_size,
                parent_ino,
                parent_parent,
                &merged,
                &uuid,
                phys_byte / 512,
            )?;
            dev.write_at(phys_byte, &new_block)?;
            self.rebuild_dir_inode(
                dev,
                parent_ino,
                &parent_core,
                nlink,
                atime,
                mtime,
                ctime,
                dir_block_size as u64,
                1,
                &[self_extent],
                &uuid,
            )?;
            return Ok(());
        }

        // ---- Leaf / node format: lay the directory out fresh in
        // contiguous runs (data | leaf-space | free-space). Free the old
        // blocks first so repeated promotions don't leak. ----
        for ext in &existing_extents {
            self.free_blocks_fsb(ext.startblock, ext.blockcount)?;
        }

        let n_data = plan.n_data();
        let n_leaf = plan.n_leafspace();
        let n_free = plan.n_free();
        let data_fsb = self.alloc_blocks_fsb(n_data as u32)?;
        let leaf_fsb = self.alloc_blocks_fsb(n_leaf as u32)?;
        let free_fsb = if n_free > 0 {
            self.alloc_blocks_fsb(n_free as u32)?
        } else {
            0
        };
        let leaf_db0 = super::dir_build::leaf_firstdb(dir_block_size);
        let free_db0 = super::dir_build::free_firstdb(dir_block_size);

        // Data blocks at logical 0..n_data.
        for (i, ents) in plan.data_blocks.iter().enumerate() {
            let phys = data_fsb + i as u64;
            let byte = self.fsb_to_byte(phys);
            let block = super::dir_build::build_data_block(
                ents,
                dir_block_size,
                parent_ino,
                &uuid,
                byte / 512,
            )?;
            dev.write_at(byte, &block)?;
        }

        match plan.format {
            super::dir_build::DirFormat::Leaf => {
                let byte = self.fsb_to_byte(leaf_fsb);
                let block = super::dir_build::build_leaf1_block(
                    &plan.leaf_ents,
                    &plan.bests,
                    dir_block_size,
                    parent_ino,
                    &uuid,
                    byte / 512,
                )?;
                dev.write_at(byte, &block)?;
            }
            super::dir_build::DirFormat::Node => {
                // Node root lives at the first leaf-space block (leaf_db0);
                // the M leafn blocks follow at leaf_db0 + 1 + j.
                let m = plan.leafn_counts.len();
                let mut node_children: Vec<(u32, u32)> = Vec::with_capacity(m);
                let mut start = 0usize;
                for (j, &cnt) in plan.leafn_counts.iter().enumerate() {
                    let slice = &plan.leaf_ents[start..start + cnt];
                    let leaf_db = leaf_db0 + 1 + j as u64;
                    let phys = leaf_fsb + 1 + j as u64;
                    let byte = self.fsb_to_byte(phys);
                    let forw = if j + 1 < m { (leaf_db + 1) as u32 } else { 0 };
                    let back = if j > 0 { (leaf_db - 1) as u32 } else { 0 };
                    let block = super::dir_build::build_leafn_block(
                        slice,
                        dir_block_size,
                        parent_ino,
                        &uuid,
                        byte / 512,
                        forw,
                        back,
                    )?;
                    dev.write_at(byte, &block)?;
                    let max_hash = slice.last().map(|(h, _)| *h).unwrap_or(0);
                    node_children.push((max_hash, leaf_db as u32));
                    start += cnt;
                }
                // Node root.
                let nbyte = self.fsb_to_byte(leaf_fsb);
                let node = super::dir_build::build_da_node_block(
                    &node_children,
                    dir_block_size,
                    parent_ino,
                    &uuid,
                    nbyte / 512,
                )?;
                dev.write_at(nbyte, &node)?;
                // Free block carrying the bests array.
                let fbyte = self.fsb_to_byte(free_fsb);
                let freeb = super::dir_build::build_free_block(
                    &plan.bests,
                    0,
                    dir_block_size,
                    parent_ino,
                    &uuid,
                    fbyte / 512,
                )?;
                dev.write_at(fbyte, &freeb)?;
            }
            super::dir_build::DirFormat::Block => unreachable!("handled above"),
        }

        let mut extents: Vec<Extent> = Vec::with_capacity(3);
        extents.push(Extent {
            offset: 0,
            startblock: data_fsb,
            blockcount: n_data as u32,
            unwritten: false,
        });
        extents.push(Extent {
            offset: leaf_db0,
            startblock: leaf_fsb,
            blockcount: n_leaf as u32,
            unwritten: false,
        });
        if n_free > 0 {
            extents.push(Extent {
                offset: free_db0,
                startblock: free_fsb,
                blockcount: n_free as u32,
                unwritten: false,
            });
        }
        let total_blocks = (n_data + n_leaf + n_free) as u64;
        let di_size = (n_data as u64) * dir_block_size as u64;
        self.rebuild_dir_inode(
            dev,
            parent_ino,
            &parent_core,
            nlink,
            atime,
            mtime,
            ctime,
            di_size,
            total_blocks,
            &extents,
            &uuid,
        )?;
        Ok(())
    }

    /// Rewrite a directory inode in EXTENTS format with the given extent
    /// list and metadata. `extents` must be sorted by logical offset.
    #[allow(clippy::too_many_arguments)]
    fn rebuild_dir_inode(
        &mut self,
        dev: &mut dyn BlockDevice,
        ino: u64,
        core: &super::inode::DinodeCore,
        nlink: u32,
        atime: XfsTimestamp,
        mtime: XfsTimestamp,
        ctime: XfsTimestamp,
        di_size: u64,
        nblocks: u64,
        extents: &[Extent],
        uuid: &[u8; 16],
    ) -> Result<()> {
        let builder = V3DinodeBuilder {
            inodesize: XFS_INODESIZE as usize,
            mode: core.mode,
            format: 2, // EXTENTS
            uid: core.uid,
            gid: core.gid,
            nlink,
            atime,
            mtime,
            ctime,
            crtime: mtime,
            size: di_size,
            nblocks,
            extsize: 0,
            nextents: extents.len() as u32,
            forkoff: 0,
            aformat: 2,
            flags: core.flags,
            generation: core.generation,
            di_ino: ino,
            uuid: *uuid,
            flags2: 0,
        };
        let mut lit = Vec::with_capacity(extents.len() * 16);
        for ext in extents {
            lit.extend_from_slice(&ext.encode());
        }
        self.write_inode(dev, ino, builder, &lit)
    }

    /// Serialize every pending directory batch (at flush, or before a
    /// read that needs the on-disk blocks current). No-op without write
    /// state.
    pub(crate) fn flush_dir_batches(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        let pending = match self.write_state.as_mut() {
            Some(ws) => ws.dir_batch.drain_all(),
            None => return Ok(()),
        };
        for (dir_ino, entries) in pending {
            self.serialize_dir(dev, dir_ino, &entries)?;
        }
        Ok(())
    }

    /// Serialize one directory's pending batch, if any. Used before a
    /// read path (`list` / `remove`) consumes that directory's on-disk
    /// block.
    pub(crate) fn flush_one_dir_batch(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_ino: u64,
    ) -> Result<()> {
        let entries = match self.write_state.as_mut() {
            Some(ws) => ws.dir_batch.take(&dir_ino),
            None => None,
        };
        if let Some(entries) = entries {
            self.serialize_dir(dev, dir_ino, &entries)?;
        }
        Ok(())
    }

    /// Look up a child inode staged (not yet serialized) under `dir_ino`
    /// by name. Lets [`resolve_path`](super::Xfs::resolve_path) see
    /// batched children as an in-memory overlay.
    pub(crate) fn pending_child_ino(&self, dir_ino: u64, name: &str) -> Option<u64> {
        self.write_state
            .as_ref()?
            .dir_batch
            .peek(&dir_ino)?
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, ino, _)| *ino)
    }

    /// Convenience: return the UUID we use for new metadata blocks.
    fn uuid_for_writes(&self) -> [u8; 16] {
        self.sb.uuid
    }

    /// Create a regular file at `name` under the directory inode
    /// `parent_ino`, streaming up to `size` bytes from `src` (which
    /// reads in 64 KiB chunks). Returns the new inode number.
    pub fn add_file<R: Read>(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u64,
        name: &str,
        meta: EntryMeta,
        size: u64,
        src: &mut R,
    ) -> Result<u64> {
        let bs = self.sb.blocksize as u64;
        let nblocks = if size == 0 { 0 } else { size.div_ceil(bs) } as u32;

        // Allocate the file data extent first.
        let startblock = if nblocks > 0 {
            self.alloc_blocks_fsb(nblocks)?
        } else {
            0
        };
        // Stream bytes through a fixed 64 KiB buffer.
        if nblocks > 0 {
            let mut scratch = [0u8; SCRATCH_SIZE];
            let mut remaining = size;
            let mut dev_offset = self.fsb_to_byte(startblock);
            while remaining > 0 {
                let want = (remaining.min(SCRATCH_SIZE as u64)) as usize;
                let n = read_exact_or_eof(src, &mut scratch[..want])?;
                if n == 0 {
                    return Err(crate::Error::InvalidArgument(format!(
                        "xfs: source for {name:?} returned EOF before {size} bytes (short by {remaining})"
                    )));
                }
                dev.write_at(dev_offset, &scratch[..n])?;
                // Zero-pad the tail of the last write if it didn't
                // cover a full FS block — keeps unallocated tail bytes
                // deterministically zero.
                dev_offset += n as u64;
                remaining -= n as u64;
            }
            // Pad up to the next FS-block boundary so the trailing
            // partial block reads back as the user's bytes followed by
            // zeros.
            let tail = (size % bs) as usize;
            if tail != 0 {
                let pad = (bs as usize) - tail;
                let zero = [0u8; SCRATCH_SIZE];
                let n = pad.min(SCRATCH_SIZE);
                dev.write_at(dev_offset, &zero[..n])?;
            }
        }
        // Allocate + write inode.
        let ino = self.alloc_inode(dev)?;
        let (atime, mtime, ctime) = meta.ts();
        let builder = V3DinodeBuilder {
            inodesize: XFS_INODESIZE as usize,
            mode: S_IFREG | (meta.mode & 0o7777),
            format: 2, // EXTENTS
            uid: meta.uid,
            gid: meta.gid,
            nlink: 1,
            atime,
            mtime,
            ctime,
            crtime: mtime,
            size,
            nblocks: nblocks as u64,
            extsize: 0,
            nextents: if nblocks > 0 { 1 } else { 0 },
            forkoff: 0,
            aformat: 2,
            flags: 0,
            generation: 1,
            di_ino: ino,
            uuid: self.uuid_for_writes(),
            flags2: 0,
        };
        let lit = if nblocks > 0 {
            let ext = Extent {
                offset: 0,
                startblock,
                blockcount: nblocks,
                unwritten: false,
            };
            ext.encode().to_vec()
        } else {
            Vec::new()
        };
        self.write_inode(dev, ino, builder, &lit)?;
        self.append_dir_entry(dev, parent_ino, name, ino, XFS_DIR3_FT_REG_FILE)?;
        Ok(ino)
    }

    /// Create a new directory `name` under `parent_ino`. The new
    /// directory is empty (block-format with just "." and "..") and
    /// occupies one freshly-allocated FS block.
    pub fn add_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u64,
        name: &str,
        meta: EntryMeta,
    ) -> Result<u64> {
        let dir_block_size = self.sb.dir_block_size() as usize;
        let dir_block_fsb = self.alloc_blocks_fsb(1)?;
        let ino = self.alloc_inode(dev)?;
        let uuid = self.uuid_for_writes();
        let phys_byte = self.fsb_to_byte(dir_block_fsb);
        let basic_blkno = phys_byte / 512;
        let block = encode_v5_block_dir(dir_block_size, ino, parent_ino, &[], &uuid, basic_blkno)?;
        dev.write_at(phys_byte, &block)?;

        let (atime, mtime, ctime) = meta.ts();
        let builder = V3DinodeBuilder {
            inodesize: XFS_INODESIZE as usize,
            mode: S_IFDIR | (meta.mode & 0o7777),
            format: 2, // EXTENTS
            uid: meta.uid,
            gid: meta.gid,
            nlink: 2,
            atime,
            mtime,
            ctime,
            crtime: mtime,
            size: dir_block_size as u64,
            nblocks: 1,
            extsize: 0,
            nextents: 1,
            forkoff: 0,
            aformat: 2,
            flags: 0,
            generation: 1,
            di_ino: ino,
            uuid,
            flags2: 0,
        };
        let ext = Extent {
            offset: 0,
            startblock: dir_block_fsb,
            blockcount: 1,
            unwritten: false,
        };
        self.write_inode(dev, ino, builder, &ext.encode())?;
        self.append_dir_entry(dev, parent_ino, name, ino, XFS_DIR3_FT_DIR)?;
        Ok(ino)
    }

    /// Create a symlink. Inline for ≤ literal-area bytes, otherwise one
    /// v5 XSLM-headered remote block.
    pub fn add_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u64,
        name: &str,
        target: &str,
        meta: EntryMeta,
    ) -> Result<u64> {
        let lit_max = (XFS_INODESIZE as usize) - 176;
        let target_bytes = target.as_bytes();
        let bs = self.sb.blocksize as usize;
        let max_remote = bs - XFS_SYMLINK_HDR_SIZE;
        let ino = self.alloc_inode(dev)?;
        let uuid = self.uuid_for_writes();
        let (atime, mtime, ctime) = meta.ts();
        if target_bytes.len() <= lit_max {
            // Inline (local) symlink.
            let builder = V3DinodeBuilder {
                inodesize: XFS_INODESIZE as usize,
                mode: S_IFLNK | 0o777,
                format: 1, // LOCAL
                uid: meta.uid,
                gid: meta.gid,
                nlink: 1,
                atime,
                mtime,
                ctime,
                crtime: mtime,
                size: target_bytes.len() as u64,
                nblocks: 0,
                extsize: 0,
                nextents: 0,
                forkoff: 0,
                aformat: 2,
                flags: 0,
                generation: 1,
                di_ino: ino,
                uuid,
                flags2: 0,
            };
            self.write_inode(dev, ino, builder, target_bytes)?;
        } else if target_bytes.len() <= max_remote {
            // One remote block.
            let blk_fsb = self.alloc_blocks_fsb(1)?;
            let blk_byte = self.fsb_to_byte(blk_fsb);
            let mut blkbuf = vec![0u8; bs];
            blkbuf[0..4].copy_from_slice(&super::symlink::XFS_SYMLINK_MAGIC.to_be_bytes());
            // crc at 4..8 — zero for now (we'll stamp below)
            // offset at 8..12 (1st block in file = 0)
            // ino at 16..24 = owner inode
            blkbuf[16..24].copy_from_slice(&ino.to_be_bytes());
            // blkno at 24..32 — basic block (FSB << 3)
            let basic_blkno = blk_byte / 512;
            blkbuf[24..32].copy_from_slice(&basic_blkno.to_be_bytes());
            // lsn at 32..40 zero
            blkbuf[40..56].copy_from_slice(&uuid);
            // Target bytes after the 56-byte header.
            blkbuf[XFS_SYMLINK_HDR_SIZE..XFS_SYMLINK_HDR_SIZE + target_bytes.len()]
                .copy_from_slice(target_bytes);
            // CRC at byte 4 (le32) of the v5 symlink header (same as
            // dir blocks).
            stamp_v5_dir_block_crc(&mut blkbuf);
            dev.write_at(blk_byte, &blkbuf)?;

            let builder = V3DinodeBuilder {
                inodesize: XFS_INODESIZE as usize,
                mode: S_IFLNK | 0o777,
                format: 2, // EXTENTS
                uid: meta.uid,
                gid: meta.gid,
                nlink: 1,
                atime,
                mtime,
                ctime,
                crtime: mtime,
                size: target_bytes.len() as u64,
                nblocks: 1,
                extsize: 0,
                nextents: 1,
                forkoff: 0,
                aformat: 2,
                flags: 0,
                generation: 1,
                di_ino: ino,
                uuid,
                flags2: 0,
            };
            let ext = Extent {
                offset: 0,
                startblock: blk_fsb,
                blockcount: 1,
                unwritten: false,
            };
            self.write_inode(dev, ino, builder, &ext.encode())?;
        } else {
            return Err(crate::Error::Unsupported(format!(
                "xfs: symlink target {} bytes > one-block remote limit ({max_remote})",
                target_bytes.len()
            )));
        }
        self.append_dir_entry(dev, parent_ino, name, ino, XFS_DIR3_FT_SYMLINK)?;
        Ok(ino)
    }

    /// Create a device node / FIFO / socket. The device numbers are
    /// packed Linux-style (major:minor in 32 bits).
    #[allow(clippy::too_many_arguments)]
    pub fn add_device(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u64,
        name: &str,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: EntryMeta,
    ) -> Result<u64> {
        let ino = self.alloc_inode(dev)?;
        let uuid = self.uuid_for_writes();
        let (atime, mtime, ctime) = meta.ts();
        // For dev nodes the literal area holds an 8-byte big-endian
        // packed dev number: major << 20 | minor (Linux MKDEV scheme).
        let packed = ((major as u64) << 20) | (minor as u64 & 0xFFFFF);
        let mut lit = [0u8; 8];
        lit.copy_from_slice(&packed.to_be_bytes());
        let builder = V3DinodeBuilder {
            inodesize: XFS_INODESIZE as usize,
            mode: kind.s_ifmt() | (meta.mode & 0o7777),
            format: 0, // DEV
            uid: meta.uid,
            gid: meta.gid,
            nlink: 1,
            atime,
            mtime,
            ctime,
            crtime: mtime,
            size: 0,
            nblocks: 0,
            extsize: 0,
            nextents: 0,
            forkoff: 0,
            aformat: 2,
            flags: 0,
            generation: 1,
            di_ino: ino,
            uuid,
            flags2: 0,
        };
        self.write_inode(dev, ino, builder, &lit)?;
        self.append_dir_entry(dev, parent_ino, name, ino, kind.ftype())?;
        Ok(ino)
    }

    /// Remove `name` from directory `parent_ino`. Frees the target
    /// inode's data extents back to the BNO/CNT free pool, marks the
    /// inode as unallocated in INOBT, zeroes its on-disk slot, and
    /// rewrites the parent dir block sans the removed entry. Returns
    /// the removed inode number.
    ///
    /// Rejects non-empty directories with [`crate::Error::InvalidArgument`].
    /// Counters are batched into the next `flush_writes` call.
    pub fn remove(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u64,
        name: &str,
    ) -> Result<u64> {
        if name == "." || name == ".." || name.is_empty() {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: cannot remove {name:?}"
            )));
        }
        // The splice below operates on on-disk dir blocks; serialize any
        // staged entries (for this parent and the target) first so they
        // are present to be removed and not re-added at flush.
        self.flush_one_dir_batch(dev, parent_ino)?;
        let (parent_buf, parent_core) = self.read_inode(dev, parent_ino)?;
        if !parent_core.is_dir() {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: parent inode {parent_ino} is not a directory"
            )));
        }
        // Block-format only: one extent, one FS block.
        let extents = self.read_extent_list(dev, &parent_buf, &parent_core)?;
        if extents.len() != 1 || extents[0].blockcount != 1 {
            return Err(crate::Error::Unsupported(
                "xfs: remove only supports block-format (single-extent) parents".into(),
            ));
        }
        let parent_self_extent = extents[0];
        let dir_block_size = self.sb.dir_block_size() as usize;
        let phys_byte = self.fsb_to_byte(parent_self_extent.startblock);
        let mut block = vec![0u8; dir_block_size];
        dev.read_at(phys_byte, &mut block)?;
        let existing = super::dir::decode_block_dir(&block, self.sb.is_v5())?;
        // Locate the entry by name.
        let target = existing.iter().find(|e| e.name == name).ok_or_else(|| {
            crate::Error::InvalidArgument(format!(
                "xfs: {name:?} not present in directory inode {parent_ino}"
            ))
        })?;
        let target_ino = target.inumber;
        let target_ftype = target.ftype;

        // Serialize the target's own staged children (if any) so the
        // emptiness check below sees them rather than a stale on-disk
        // block.
        self.flush_one_dir_batch(dev, target_ino)?;

        // Read the target inode; reject non-empty directories.
        let (target_buf, target_core) = self.read_inode(dev, target_ino)?;
        if target_core.is_dir() {
            let child_entries = self.read_dir_entries(dev, &target_buf, &target_core)?;
            let user_entries: Vec<_> = child_entries
                .iter()
                .filter(|e| e.name != "." && e.name != "..")
                .collect();
            if !user_entries.is_empty() {
                return Err(crate::Error::InvalidArgument(format!(
                    "xfs: directory {name:?} not empty (contains {} entries)",
                    user_entries.len()
                )));
            }
        }

        // Free the target inode's data extents (regular files /
        // remote-format symlinks / sub-dir blocks).
        if matches!(
            target_core.format,
            super::inode::DiFormat::Extents | super::inode::DiFormat::Btree
        ) {
            let target_extents = self.read_extent_list(dev, &target_buf, &target_core)?;
            for ext in &target_extents {
                self.free_blocks_fsb(ext.startblock, ext.blockcount)?;
            }
        }

        // Mark the inode unallocated.
        self.free_inode(target_ino)?;
        // Zero its on-disk slot so xfs_repair sees "di_magic = 0".
        let ino_off = self.ino_byte_offset(target_ino)?;
        let zero = vec![0u8; XFS_INODESIZE as usize];
        dev.write_at(ino_off, &zero)?;

        // Re-encode the parent dir block without the entry.
        let new_entries: Vec<(String, u64, u8)> = existing
            .into_iter()
            .filter(|e| e.name != "." && e.name != ".." && e.name != name)
            .map(|e| (e.name, e.inumber, e.ftype))
            .collect();
        let parent_parent = existing_parent(&block, self.sb.is_v5())?;
        let uuid = self.uuid_for_writes();
        let basic_blkno = phys_byte / 512;
        let new_block = encode_v5_block_dir(
            dir_block_size,
            parent_ino,
            parent_parent,
            &new_entries,
            &uuid,
            basic_blkno,
        )?;
        dev.write_at(phys_byte, &new_block)?;

        // Adjust parent nlink if we removed a subdirectory.
        let new_nlink = if target_ftype == XFS_DIR3_FT_DIR {
            parent_core.nlink.saturating_sub(1)
        } else {
            parent_core.nlink
        };
        let (atime, mtime, ctime) = (parent_core.atime, parent_core.mtime, parent_core.ctime);
        let builder = V3DinodeBuilder {
            inodesize: XFS_INODESIZE as usize,
            mode: parent_core.mode,
            format: 2, // EXTENTS
            uid: parent_core.uid,
            gid: parent_core.gid,
            nlink: new_nlink,
            atime,
            mtime,
            ctime,
            crtime: mtime,
            size: dir_block_size as u64,
            nblocks: 1,
            extsize: 0,
            nextents: 1,
            forkoff: 0,
            aformat: 2,
            flags: parent_core.flags,
            generation: parent_core.generation,
            di_ino: parent_ino,
            uuid,
            flags2: 0,
        };
        let mut lit = Vec::with_capacity(16);
        lit.extend_from_slice(&parent_self_extent.encode());
        self.write_inode(dev, parent_ino, builder, &lit)?;

        Ok(target_ino)
    }

    /// Attach a single extended attribute `(name, value)` to the inode.
    /// Stored in the shortform (LOCAL) attribute fork inside the
    /// inode's literal area. The total size of all attributes on the
    /// inode (header + entries) must fit in the attribute-fork half of
    /// the literal area — if it doesn't, an `Error::Unsupported` is
    /// returned (leaf-/node-format spill is future work).
    ///
    /// The first xattr added to a freshly-allocated inode pulls bytes
    /// away from the data fork via `di_forkoff`; we pick the minimum
    /// `forkoff` that just covers the encoded shortform area, rounded
    /// up to the nearest 8-byte boundary, leaving the remaining
    /// literal-area bytes available for the data fork.
    pub fn add_xattr(
        &mut self,
        dev: &mut dyn BlockDevice,
        ino: u64,
        name: &str,
        value: &[u8],
    ) -> Result<()> {
        if name.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "xfs: empty xattr name".into(),
            ));
        }
        let mut current = self.read_xattrs_vec(dev, ino)?;
        if let Some(slot) = current.iter_mut().find(|(n, _)| n == name) {
            slot.1 = value.to_vec();
        } else {
            current.push((name.to_string(), value.to_vec()));
        }
        self.rebuild_xattr_fork(dev, ino, &current)
    }

    /// Remove a single extended attribute. Returns `Ok(false)` if the
    /// attribute didn't exist on the inode (the caller can treat that
    /// as "already gone"); `Ok(true)` after a successful removal. When
    /// the last attribute is removed, the attribute fork is collapsed
    /// — forkoff goes back to 0, di_aformat back to 2 (the default
    /// "extents" indicator the writer uses for inodes with no attrs).
    pub fn remove_xattr(
        &mut self,
        dev: &mut dyn BlockDevice,
        ino: u64,
        name: &str,
    ) -> Result<bool> {
        let mut current = self.read_xattrs_vec(dev, ino)?;
        let before = current.len();
        current.retain(|(n, _)| n != name);
        if current.len() == before {
            return Ok(false);
        }
        self.rebuild_xattr_fork(dev, ino, &current)?;
        Ok(true)
    }

    /// Read all xattrs on this inode as a deterministic `Vec` (sorted
    /// by name) — used by [`add_xattr`] / [`remove_xattr`] so they
    /// produce byte-stable output regardless of `HashMap` iteration
    /// order.
    fn read_xattrs_vec(
        &self,
        dev: &mut dyn BlockDevice,
        ino: u64,
    ) -> Result<Vec<(String, Vec<u8>)>> {
        let (ino_buf, core) = self.read_inode(dev, ino)?;
        let mut v: Vec<(String, Vec<u8>)> = self
            .read_xattrs_from_core(dev, &ino_buf, &core)?
            .into_iter()
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(v)
    }

    /// Rebuild this inode's attribute fork in place with exactly the
    /// `attrs` set passed in. Picks the cheapest format that fits
    /// (shortform first, then a single leaf block), and frees any
    /// previously-allocated leaf block the inode pointed at. When
    /// `attrs` is empty, clears the fork entirely (forkoff = 0,
    /// aformat = 2 = "extents" sentinel).
    fn rebuild_xattr_fork(
        &mut self,
        dev: &mut dyn BlockDevice,
        ino: u64,
        attrs: &[(String, Vec<u8>)],
    ) -> Result<()> {
        let (ino_buf, core) = self.read_inode(dev, ino)?;
        let prev_aformat = if ino_buf.len() >= 84 { ino_buf[83] } else { 0 };
        let prev_leaf_fsb = if prev_aformat == 2 {
            let attr_start = core.literal_offset + (core.forkoff as usize) * 8;
            let inodesize = self.sb.inodesize as usize;
            // di_anextents lives at dinode offset 80..82 (BE u16) —
            // this is the actual xfsprogs layout. Don't trust the
            // older "88" comment in inode.rs.
            let anextents = if ino_buf.len() >= 82 {
                u16::from_be_bytes(ino_buf[80..82].try_into().unwrap())
            } else {
                0
            };
            if anextents == 1 && attr_start + 16 <= inodesize.min(ino_buf.len()) {
                let exts = super::bmbt::decode_extents(
                    &ino_buf[attr_start..inodesize.min(ino_buf.len())],
                    1,
                )?;
                Some(exts[0].startblock)
            } else {
                None
            }
        } else {
            None
        };

        // Determine the data-fork occupancy — we must NOT overwrite
        // the bytes the data fork already uses.
        //
        //   - LOCAL (1)    → core.size bytes
        //   - DEV (0)      → 8 bytes
        //   - EXTENTS (2)  → core.nextents * 16 bytes
        //   - BTREE (3)    → variable; we conservatively refuse.
        //
        // di_forkoff is in 8-byte words measured from the literal
        // area's start; the attribute fork lives at lit[forkoff*8 ..].
        let lit_size = (XFS_INODESIZE as usize) - 176;
        let data_fork_bytes = match core.format {
            super::inode::DiFormat::Local => core.size as usize,
            super::inode::DiFormat::Dev => 8,
            super::inode::DiFormat::Extents => (core.nextents as usize) * 16,
            super::inode::DiFormat::Btree => {
                return Err(crate::Error::Unsupported(
                    "xfs: rebuild_xattr_fork on BTREE-format inode not supported".into(),
                ));
            }
            super::inode::DiFormat::Unknown(b) => {
                return Err(crate::Error::Unsupported(format!(
                    "xfs: rebuild_xattr_fork on inode with unknown di_format {b}"
                )));
            }
        };
        let data_fork_words = data_fork_bytes.div_ceil(8);
        let data_fork_slice = ino_buf[176..176 + data_fork_bytes].to_vec();

        // Pick the cheapest format that holds these attrs.
        //   attrs empty            → forkoff=0, aformat=2 (no fork)
        //   shortform fits         → aformat=1
        //   else                   → aformat=2 + one leaf block
        let (aformat, attr_words, attr_payload, new_leaf_fsb): (u8, usize, Vec<u8>, Option<u64>) =
            if attrs.is_empty() {
                (2, 0, Vec::new(), None)
            } else {
                let shortform_encoded = super::xattr::encode_shortform(attrs).ok();
                let mut chosen = None;
                if let Some(enc) = &shortform_encoded {
                    let attr_words = enc.len().div_ceil(8);
                    let min_forkoff = data_fork_words.max(1);
                    let max_forkoff = (lit_size / 8).saturating_sub(attr_words);
                    if min_forkoff <= max_forkoff {
                        chosen = Some((1u8, attr_words, enc.clone(), None));
                    }
                }
                match chosen {
                    Some(x) => x,
                    None => self.encode_leaf_form_payload(dev, attrs, ino)?,
                }
            };

        // forkoff is the BOUNDARY between forks — i.e., data fork
        // length in 8-byte words. We need `forkoff*8 + attr_size <=
        // lit_size`. When the fork is empty we encode forkoff = 0.
        let forkoff: u8 = if attrs.is_empty() {
            0
        } else {
            let min_forkoff = data_fork_words.max(1);
            let max_forkoff = (lit_size / 8).saturating_sub(attr_words);
            if min_forkoff > max_forkoff {
                if let Some(fsb) = new_leaf_fsb {
                    // Roll back the leaf-block allocation we just made.
                    self.free_blocks_fsb(fsb, 1)?;
                }
                return Err(crate::Error::Unsupported(format!(
                    "xfs: xattr fork ({} bytes) does not fit alongside data fork",
                    attr_payload.len()
                )));
            }
            let f = min_forkoff as u8;
            let attr_off = (f as usize) * 8;
            if attr_off + attr_payload.len() > lit_size {
                if let Some(fsb) = new_leaf_fsb {
                    self.free_blocks_fsb(fsb, 1)?;
                }
                return Err(crate::Error::Unsupported(format!(
                    "xfs: xattr fork at forkoff={f} overruns literal area"
                )));
            }
            f
        };

        // For leaf form, di_anextents is 1 (one attr-fork extent).
        let anextents: u16 = if aformat == 2 && !attrs.is_empty() {
            1
        } else {
            0
        };

        let new_nblocks = core.nblocks + if new_leaf_fsb.is_some() { 1 } else { 0 }
            - if prev_leaf_fsb.is_some() { 1 } else { 0 };

        let builder = V3DinodeBuilder {
            inodesize: XFS_INODESIZE as usize,
            mode: core.mode,
            format: match core.format {
                super::inode::DiFormat::Local => 1,
                super::inode::DiFormat::Dev => 0,
                super::inode::DiFormat::Extents => 2,
                super::inode::DiFormat::Btree => 3,
                super::inode::DiFormat::Unknown(b) => b,
            },
            uid: core.uid,
            gid: core.gid,
            nlink: core.nlink,
            atime: core.atime,
            mtime: core.mtime,
            ctime: core.ctime,
            crtime: core.mtime,
            size: core.size,
            nblocks: new_nblocks,
            extsize: 0,
            nextents: core.nextents,
            forkoff,
            aformat,
            flags: core.flags,
            generation: core.generation,
            di_ino: ino,
            uuid: self.uuid_for_writes(),
            flags2: 0,
        };
        let mut buf = builder.build();
        // Restore the data fork.
        buf[176..176 + data_fork_bytes].copy_from_slice(&data_fork_slice);
        // Lay down the attribute fork at the forkoff boundary (if any).
        if forkoff != 0 && !attr_payload.is_empty() {
            let attr_off = (forkoff as usize) * 8;
            buf[176 + attr_off..176 + attr_off + attr_payload.len()].copy_from_slice(&attr_payload);
        }
        // di_anextents at offset 80..82.
        buf[80..82].copy_from_slice(&anextents.to_be_bytes());
        stamp_v3_inode_crc(&mut buf);
        let ino_off = self.ino_byte_offset(ino)?;
        dev.write_at(ino_off, &buf)?;

        // If we replaced/cleared an existing leaf block, free it.
        if let Some(old_fsb) = prev_leaf_fsb
            && Some(old_fsb) != new_leaf_fsb
        {
            self.free_blocks_fsb(old_fsb, 1)?;
        }
        Ok(())
    }

    /// Allocate + write a leaf-form attribute block for `attrs`, and
    /// return `(aformat=2, attr_fork_words, attr_fork_payload,
    /// Some(leaf_fsb))` — the payload is the packed bmbt extent record
    /// pointing at the new leaf block.
    fn encode_leaf_form_payload(
        &mut self,
        dev: &mut dyn BlockDevice,
        attrs: &[(String, Vec<u8>)],
        owner_ino: u64,
    ) -> Result<(u8, usize, Vec<u8>, Option<u64>)> {
        let bs = self.sb.blocksize as usize;
        let needed = super::xattr_leaf::min_leaf_block_size(attrs);
        if needed > bs {
            return Err(crate::Error::Unsupported(format!(
                "xfs: leaf xattr block needs {needed} bytes, > one FS block ({bs}); \
                 node-form attribute trees are not implemented"
            )));
        }
        let leaf_fsb = self.alloc_blocks_fsb(1)?;
        let uuid = self.uuid_for_writes();
        let byte = self.fsb_to_byte(leaf_fsb);
        let basic_blkno = byte / 512;
        let block = super::xattr_leaf::encode_v5_leaf(attrs, bs, owner_ino, &uuid, basic_blkno)?;
        dev.write_at(byte, &block)?;

        let ext = super::bmbt::Extent {
            offset: 0,
            startblock: leaf_fsb,
            blockcount: 1,
            unwritten: false,
        };
        let payload = ext.encode().to_vec();
        // 16 bytes = 2 8-byte words.
        Ok((2u8, 2usize, payload, Some(leaf_fsb)))
    }

    /// Read all extended attributes from an inode's attribute fork.
    /// Returns an empty map when `core.forkoff == 0` (no attr fork) or
    /// when the attr fork is empty. Supports shortform (LOCAL) attr
    /// forks only — extents and B-tree spills surface `Error::Unsupported`.
    pub fn read_xattrs(
        &self,
        dev: &mut dyn BlockDevice,
        ino: u64,
    ) -> Result<std::collections::HashMap<String, Vec<u8>>> {
        let (buf, core) = self.read_inode(dev, ino)?;
        self.read_xattrs_from_core(dev, &buf, &core)
    }

    /// Decode this inode's attr fork from its already-read on-disk
    /// bytes + core. Used by both [`read_xattrs`] and the read side of
    /// [`add_xattr`] (so updates round-trip without re-reading the
    /// inode). Needs `dev` so it can pull leaf-form attr blocks off
    /// disk when `di_aformat = EXTENTS`.
    fn read_xattrs_from_core(
        &self,
        dev: &mut dyn BlockDevice,
        ino_buf: &[u8],
        core: &super::inode::DinodeCore,
    ) -> Result<std::collections::HashMap<String, Vec<u8>>> {
        if core.forkoff == 0 {
            return Ok(std::collections::HashMap::new());
        }
        let attr_off = (core.forkoff as usize) * 8;
        let lit_start = core.literal_offset;
        let attr_start = lit_start + attr_off;
        let inodesize = self.sb.inodesize as usize;
        if attr_start >= inodesize {
            return Ok(std::collections::HashMap::new());
        }
        let attr_buf = &ino_buf[attr_start..inodesize.min(ino_buf.len())];
        // di_aformat=1 ⇒ shortform / LOCAL. The on-disk byte sits at
        // dinode offset 83 (di_aformat).
        let aformat = if ino_buf.len() >= 84 { ino_buf[83] } else { 0 };
        match aformat {
            1 => super::xattr::decode_shortform(attr_buf),
            0 => Ok(std::collections::HashMap::new()),
            2 => {
                // di_anextents lives at dinode offset 80..82 (BE u16)
                // per the on-disk xfsprogs layout.
                let anextents = if ino_buf.len() >= 82 {
                    u16::from_be_bytes(ino_buf[80..82].try_into().unwrap())
                } else {
                    0
                };
                if anextents == 0 {
                    return Ok(std::collections::HashMap::new());
                }
                // Decode the attr fork's extent list (packed records,
                // same shape as data-fork extents).
                let extents = super::bmbt::decode_extents(attr_buf, anextents as u32)?;
                // v1 supports a single-block leaf attr fork only.
                if extents.len() != 1 || extents[0].blockcount != 1 {
                    return Err(crate::Error::Unsupported(format!(
                        "xfs: leaf-form xattr fork with {} extents \
                         (multi-block / node-form deferred)",
                        extents.len()
                    )));
                }
                let ext = &extents[0];
                let bs = self.sb.blocksize as usize;
                let byte = self.fsb_to_byte_xattr(ext.startblock);
                let mut block = vec![0u8; bs];
                dev.read_at(byte, &mut block)?;
                super::xattr_leaf::decode_leaf(&block)
            }
            3 => Err(crate::Error::Unsupported(
                "xfs: btree-format attribute fork not supported on read".into(),
            )),
            other => Err(crate::Error::Unsupported(format!(
                "xfs: unknown attribute-fork format {other}"
            ))),
        }
    }

    /// FSB → byte offset (private mirror of the read-side helper so the
    /// xattr read path doesn't have to reach into `super` for it).
    fn fsb_to_byte_xattr(&self, fsb: u64) -> u64 {
        let ag = fsb >> self.sb.agblklog as u32;
        let agblk = fsb & ((1u64 << self.sb.agblklog as u32) - 1);
        ag * (self.sb.agblocks as u64) * (self.sb.blocksize as u64)
            + agblk * (self.sb.blocksize as u64)
    }

    /// Flush in-memory allocator state to disk: rewrite the AGF / AGI /
    /// BNO / CNT / INOBT roots + the superblock counters to reflect the
    /// current `WriteState`. Multi-AG safe: every AG's headers and
    /// per-AG B+trees are rewritten from `WriteState.ags[ag]`. Must be
    /// called once after all `add_*` / `remove` calls so the image is
    /// `xfs_repair -n` clean.
    pub fn flush_writes(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        // Serialize any pending directory batches first, so the AG
        // free-space / inode accounting below reflects the final state.
        self.flush_dir_batches(dev)?;
        let agblocks = self.sb.agblocks;
        let total_blocks = self.sb.dblocks as u32;
        let uuid = self.uuid_for_writes();
        let ws = self
            .write_state
            .as_ref()
            .ok_or_else(|| {
                crate::Error::InvalidArgument(
                    "xfs: flush_writes called before begin_writes()".into(),
                )
            })?
            .clone();
        let bs = XFS_BLOCKSIZE as u64;
        let agcount = ws.ags.len() as u32;

        let mut total_free_blocks_u64: u64 = 0;

        for (ag_idx, ag_state) in ws.ags.iter().enumerate() {
            let ag = ag_idx as u32;
            let ag_byte = (ag as u64) * (agblocks as u64) * bs;
            // The last AG can be short; quote the real block count.
            let this_ag_blocks = if ag == agcount - 1 {
                total_blocks.saturating_sub(ag * agblocks).max(1)
            } else {
                agblocks
            };
            // Collect this AG's free-space extents: the trailing
            // bump-pointer region plus any explicitly freed extents.
            let mut extents: Vec<(u32, u32)> = ag_state.freed_extents.clone();
            let tail_free = this_ag_blocks.saturating_sub(ag_state.next_agblock);
            if tail_free > 0 {
                extents.push((ag_state.next_agblock, tail_free));
            }
            // Sort by start-block, then coalesce adjacent extents.
            extents.sort_by_key(|(s, _)| *s);
            let mut coalesced: Vec<(u32, u32)> = Vec::with_capacity(extents.len());
            for (s, c) in extents {
                if let Some((ls, lc)) = coalesced.last_mut() {
                    if *ls + *lc == s {
                        *lc += c;
                        continue;
                    }
                }
                coalesced.push((s, c));
            }
            let total_free_in_ag: u32 = coalesced.iter().map(|(_, c)| *c).sum();
            let longest = coalesced.iter().map(|(_, c)| *c).max().unwrap_or(0);
            total_free_blocks_u64 += total_free_in_ag as u64;

            // -- BNO (block 4) — start-block order --------------------
            let mut bno = vec![0u8; XFS_BLOCKSIZE as usize];
            write_btree_header_for_ag(
                &mut bno,
                XFS_ABTB_CRC_MAGIC,
                0,
                coalesced.len() as u16,
                &uuid,
                ag,
                agblocks,
                4,
            );
            for (i, (s, c)) in coalesced.iter().enumerate() {
                let off = XFS_BTREE_SBLOCK_V5_SIZE + i * 8;
                if off + 8 > bno.len() {
                    return Err(crate::Error::Unsupported(
                        "xfs: too many free-space extents for a single-leaf BNO/CNT".into(),
                    ));
                }
                bno[off..off + 4].copy_from_slice(&s.to_be_bytes());
                bno[off + 4..off + 8].copy_from_slice(&c.to_be_bytes());
            }
            stamp_v5_btree_block_crc(&mut bno);
            dev.write_at(ag_byte + 4 * bs, &bno)?;

            // -- CNT (block 5) — blockcount order ---------------------
            let mut cnt_sorted = coalesced.clone();
            cnt_sorted.sort_by_key(|(_, c)| *c);
            let mut cnt = vec![0u8; XFS_BLOCKSIZE as usize];
            write_btree_header_for_ag(
                &mut cnt,
                XFS_ABTC_CRC_MAGIC,
                0,
                cnt_sorted.len() as u16,
                &uuid,
                ag,
                agblocks,
                5,
            );
            for (i, (s, c)) in cnt_sorted.iter().enumerate() {
                let off = XFS_BTREE_SBLOCK_V5_SIZE + i * 8;
                cnt[off..off + 4].copy_from_slice(&s.to_be_bytes());
                cnt[off + 4..off + 8].copy_from_slice(&c.to_be_bytes());
            }
            stamp_v5_btree_block_crc(&mut cnt);
            dev.write_at(ag_byte + 5 * bs, &cnt)?;

            // -- INOBT (block 6) — leaf with this AG's chunks ---------
            let mut inobt = vec![0u8; XFS_BLOCKSIZE as usize];
            write_btree_header_for_ag(
                &mut inobt,
                XFS_IBT_CRC_MAGIC,
                0,
                ag_state.chunks.len() as u16,
                &uuid,
                ag,
                agblocks,
                6,
            );
            for (i, chunk) in ag_state.chunks.iter().enumerate() {
                let off = XFS_BTREE_SBLOCK_V5_SIZE + i * 16;
                if off + 16 > inobt.len() {
                    return Err(crate::Error::Unsupported(
                        "xfs: too many inode chunks for a single-leaf INOBT".into(),
                    ));
                }
                inobt[off..off + 4].copy_from_slice(&chunk.startino_ag.to_be_bytes());
                inobt[off + 4..off + 8].copy_from_slice(&chunk.freecount().to_be_bytes());
                inobt[off + 8..off + 16].copy_from_slice(&chunk.ir_free.to_be_bytes());
            }
            stamp_v5_btree_block_crc(&mut inobt);
            dev.write_at(ag_byte + 6 * bs, &inobt)?;

            // -- AGF (sector 1, byte 512 of AG) ----------------------
            // AG headers are sector-aligned: the formatter laid them
            // down at offsets {0,512,1024,1536} of every AG, so we
            // overwrite at those same positions here.
            let sect = super::format::XFS_SECTSIZE as u64;
            let mut agf = vec![0u8; sect as usize];
            agf[0..4].copy_from_slice(&super::format::XFS_AGF_MAGIC.to_be_bytes());
            agf[4..8].copy_from_slice(&super::format::XFS_AGF_VERSION.to_be_bytes());
            agf[8..12].copy_from_slice(&ag.to_be_bytes());
            agf[12..16].copy_from_slice(&this_ag_blocks.to_be_bytes());
            agf[16..20].copy_from_slice(&4u32.to_be_bytes()); // bno root
            agf[20..24].copy_from_slice(&5u32.to_be_bytes()); // cnt root
            agf[24..28].copy_from_slice(&0u32.to_be_bytes()); // rmap root
            agf[28..32].copy_from_slice(&1u32.to_be_bytes());
            agf[32..36].copy_from_slice(&1u32.to_be_bytes());
            agf[36..40].copy_from_slice(&0u32.to_be_bytes()); // rmap level
            agf[40..44].copy_from_slice(&0u32.to_be_bytes());
            agf[44..48].copy_from_slice(&0u32.to_be_bytes());
            agf[48..52].copy_from_slice(&0u32.to_be_bytes());
            agf[52..56].copy_from_slice(&total_free_in_ag.to_be_bytes());
            agf[56..60].copy_from_slice(&longest.to_be_bytes());
            agf[60..64].copy_from_slice(&0u32.to_be_bytes());
            agf[64..80].copy_from_slice(&uuid);
            // rmap_blocks (80..84) stays 0 — no RMAPBT.
            // REFLINK fields, matching `format::build_agf`:
            //   84..88  refcount_blocks = 1
            //   88..92  refcount_root   = REFCNTBT_AGBLOCK (7)
            //   92..96  refcount_level  = 1
            // Phase 3b stage 1 leaves the REFCNTBT empty; the block at
            // AG-block 7 is laid down by `format::format()` and not
            // touched by the writer until `clone_range` (stage 2).
            agf[84..88].copy_from_slice(&1u32.to_be_bytes());
            agf[88..92].copy_from_slice(&super::format::REFCNTBT_AGBLOCK.to_be_bytes());
            agf[92..96].copy_from_slice(&1u32.to_be_bytes());
            super::format::stamp_v5_agf_crc(&mut agf);
            dev.write_at(ag_byte + sect, &agf)?;

            // -- AGI (sector 2, byte 1024 of AG) ---------------------
            let mut agi = vec![0u8; sect as usize];
            agi[0..4].copy_from_slice(&super::format::XFS_AGI_MAGIC.to_be_bytes());
            agi[4..8].copy_from_slice(&super::format::XFS_AGI_VERSION.to_be_bytes());
            agi[8..12].copy_from_slice(&ag.to_be_bytes());
            agi[12..16].copy_from_slice(&this_ag_blocks.to_be_bytes());
            let agi_count = ag_state.usedcount_inodes() + ag_state.freecount_inodes();
            agi[16..20].copy_from_slice(&agi_count.to_be_bytes());
            agi[20..24].copy_from_slice(&6u32.to_be_bytes()); // inobt root
            agi[24..28].copy_from_slice(&1u32.to_be_bytes()); // level
            agi[28..32].copy_from_slice(&ag_state.freecount_inodes().to_be_bytes());
            if let Some(last) = ag_state.chunks.last() {
                agi[32..36].copy_from_slice(&last.startino_ag.to_be_bytes());
            } else {
                agi[32..36].copy_from_slice(&u32::MAX.to_be_bytes());
            }
            agi[36..40].copy_from_slice(&u32::MAX.to_be_bytes()); // dirino (deprecated)
            for i in 0..64 {
                let off = 40 + i * 4;
                agi[off..off + 4].copy_from_slice(&u32::MAX.to_be_bytes());
            }
            agi[296..312].copy_from_slice(&uuid);
            agi[328..332].copy_from_slice(&0u32.to_be_bytes());
            agi[332..336].copy_from_slice(&0u32.to_be_bytes());
            super::format::stamp_v5_agi_crc(&mut agi);
            dev.write_at(ag_byte + 2 * sect, &agi)?;
        }

        // -- Re-stamp the primary superblock counters ---------------
        let mut sb = vec![0u8; XFS_BLOCKSIZE as usize];
        dev.read_at(0, &mut sb)?;
        // sb_icount = total inodes provisioned across all chunks (=
        // chunks × 64). xfs_repair compares this to the count it
        // derives from the INOBT and flags a mismatch otherwise.
        let chunk_total: u64 = ws.ags.iter().map(|a| (a.chunks.len() as u64) * 64).sum();
        sb[128..136].copy_from_slice(&chunk_total.to_be_bytes());
        sb[136..144].copy_from_slice(&ws.inodes_free.to_be_bytes());
        sb[144..152].copy_from_slice(&total_free_blocks_u64.to_be_bytes());
        super::format::stamp_v5_superblock_crc(&mut sb);
        dev.write_at(0, &sb)?;

        dev.sync()?;
        Ok(())
    }

    // ----------------------------------------------------------------
    // Path-based wrappers — convenience callers for the generic
    // `crate::fs::Filesystem` trait. Each splits `/a/b/c` into a parent
    // dir + a leaf name; the parent is resolved via `lookup_path_ino`
    // (returns root for `/`), then the matching inode-based method is
    // invoked.
    // ----------------------------------------------------------------

    /// Path-based equivalent of [`Self::add_file`].
    pub fn add_file_path<R: Read>(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        meta: EntryMeta,
        size: u64,
        src: &mut R,
    ) -> Result<u64> {
        self.ensure_write_state(dev)?;
        let (parent_ino, name) = self.split_path_for_create(dev, path)?;
        self.add_file(dev, parent_ino, &name, meta, size, src)
    }

    /// Clone `src_path` to a new file at `dst_path` by sharing extents:
    /// the destination inode points at the **same** physical FSBs as
    /// the source, and a refcount record is inserted into each
    /// affected AG's REFCNTBT with `refcount = 2`. Both the source and
    /// destination inodes get `XFS_DIFLAG2_REFLINK` set in `di_flags2`
    /// so `xfs_repair` sees a consistent picture.
    ///
    /// Constraints (this stage):
    /// * `src_path` must be a regular file in the EXTENTS data-fork
    ///   format (BMBT-format files are rejected — fstool's writer
    ///   never produces them anyway).
    /// * `dst_path` must not already exist; its parent must.
    /// * Sparse / unwritten / encrypted extents and `$ATTRIBUTE_LIST`-
    ///   like multi-record inodes are rejected with `Unsupported`.
    /// * A subsequent clone of the same range would bump the refcount
    ///   past 2; this initial implementation rejects overlap with
    ///   `Unsupported` rather than splitting / merging records (real
    ///   workloads will surface that need; we add it then).
    pub fn clone_file_path(
        &mut self,
        dev: &mut dyn BlockDevice,
        src_path: &str,
        dst_path: &str,
    ) -> Result<u64> {
        self.ensure_write_state(dev)?;

        // 1) Resolve src, validate kind + format.
        let (src_ino, mut src_buf, src_core) = self.resolve_path(dev, src_path)?;
        if !src_core.is_reg() {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: clone_file source {src_path:?} is not a regular file"
            )));
        }
        if src_core.format != super::inode::DiFormat::Extents {
            return Err(crate::Error::Unsupported(format!(
                "xfs: clone_file source is in {:?} format; only Extents is supported",
                src_core.format
            )));
        }

        // 2) Extent list (each record is `Extent { offset, startblock, blockcount, unwritten }`).
        let extents = self.read_extent_list(dev, &src_buf, &src_core)?;
        for e in &extents {
            if e.unwritten {
                return Err(crate::Error::Unsupported(
                    "xfs: clone_file: unwritten extents not supported".into(),
                ));
            }
        }

        // 3) Resolve destination parent + leaf; reject if leaf exists.
        let (parent_ino, leaf) = self.split_path_for_create(dev, dst_path)?;
        if self.dir_entry_exists(dev, parent_ino, &leaf)? {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: clone_file destination {dst_path:?} already exists"
            )));
        }

        // 4) Allocate the destination inode.
        let dst_ino = self.alloc_inode(dev)?;

        // 5) Re-read raw bytes the DinodeCore doesn't expose: extsize
        //    (72..76), aformat (83), existing flags2 (120..128).
        let src_extsize = u32::from_be_bytes(src_buf[72..76].try_into().unwrap());
        let src_aformat = src_buf[83];

        // 6) Build dst inode core (mirror src's, except di_ino + REFLINK flag).
        let dst_builder = V3DinodeBuilder {
            inodesize: XFS_INODESIZE as usize,
            mode: src_core.mode,
            format: 2, // EXTENTS
            uid: src_core.uid,
            gid: src_core.gid,
            nlink: 1,
            atime: src_core.atime,
            mtime: src_core.mtime,
            ctime: src_core.ctime,
            // No crtime on the existing core; reuse mtime — matches the
            // writer's other create paths.
            crtime: src_core.mtime,
            size: src_core.size,
            nblocks: src_core.nblocks,
            extsize: src_extsize,
            nextents: src_core.nextents,
            forkoff: src_core.forkoff,
            aformat: src_aformat,
            flags: src_core.flags,
            generation: 1,
            di_ino: dst_ino,
            uuid: self.uuid_for_writes(),
            flags2: super::inode::XFS_DIFLAG2_REFLINK,
        };

        // 7) Encode the extents verbatim into dst's literal area —
        //    same physical FSBs the source points at.
        let mut lit: Vec<u8> = Vec::with_capacity(extents.len() * 16);
        for e in &extents {
            lit.extend_from_slice(&e.encode());
        }

        // 8) Write dst inode + add the entry to dst's parent directory.
        self.write_inode(dev, dst_ino, dst_builder, &lit)?;
        self.append_dir_entry(dev, parent_ino, &leaf, dst_ino, XFS_DIR3_FT_REG_FILE)?;

        // 9) For each shared extent, insert a refcount record with
        //    refcount=2 into the AG that owns it.
        for e in &extents {
            self.insert_refcount_record_for_extent(dev, e.startblock, e.blockcount)?;
        }

        // 10) Set REFLINK on the source inode too — xfs_repair flags
        //     "inode has shared extents but REFLINK flag unset" as a
        //     corruption otherwise. OR the bit into di_flags2
        //     (120..128) and restamp the CRC.
        let mut src_flags2 = u64::from_be_bytes(src_buf[120..128].try_into().unwrap());
        src_flags2 |= super::inode::XFS_DIFLAG2_REFLINK;
        src_buf[120..128].copy_from_slice(&src_flags2.to_be_bytes());
        super::inode::stamp_v3_inode_crc(&mut src_buf);
        let src_off = self.ino_byte_offset(src_ino)?;
        dev.write_at(src_off, &src_buf)?;

        Ok(dst_ino)
    }

    /// Insert a refcount record `(startblock_in_ag, blockcount,
    /// refcount=2)` into the AG that owns the given absolute FSB.
    /// The REFCNTBT root lives at AG-block `REFCNTBT_AGBLOCK` of every
    /// AG (see `format::format()`).
    ///
    /// Records are sorted by `rc_startblock` ascending — the XFS
    /// refcount-btree's collation rule. `numrecs` is bumped and the
    /// block's CRC re-stamped.
    fn insert_refcount_record_for_extent(
        &mut self,
        dev: &mut dyn BlockDevice,
        startblock_fsb: u64,
        blockcount: u32,
    ) -> Result<()> {
        let agblklog = self.sb.agblklog as u32;
        let ag = (startblock_fsb >> agblklog) as u32;
        let ag_start_block = (startblock_fsb & ((1u64 << agblklog) - 1)) as u32;

        // Sanity: refuse to issue refcount records that straddle an AG
        // boundary. fstool's bump allocator (`alloc_blocks_fsb`) never
        // emits one — every extent is allocated from a single AG — but
        // assert anyway so a future change that breaks the invariant
        // doesn't silently corrupt the refcount-btree.
        let agblocks = self.sb.agblocks;
        if (ag_start_block as u64) + (blockcount as u64) > agblocks as u64 {
            return Err(crate::Error::Unsupported(format!(
                "xfs: clone_file extent at FSB {startblock_fsb} (ag {ag}, agblock \
                 {ag_start_block}, count {blockcount}) straddles AG boundary"
            )));
        }

        let bs = self.sb.blocksize as u64;
        let ag_byte = (ag as u64) * (agblocks as u64) * bs;
        let refc_block_off = ag_byte + (super::format::REFCNTBT_AGBLOCK as u64) * bs;

        let mut block = vec![0u8; bs as usize];
        dev.read_at(refc_block_off, &mut block)?;
        // Validate magic — guards against AG-layout mistakes.
        let magic = u32::from_be_bytes(block[0..4].try_into().unwrap());
        if magic != super::format::XFS_REFC_CRC_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: clone_file: REFCNTBT at ag {ag} has wrong magic {magic:#010x}"
            )));
        }
        let numrecs = u16::from_be_bytes(block[6..8].try_into().unwrap()) as usize;

        // Parse existing records (12 bytes each, starting at offset
        // `XFS_BTREE_SBLOCK_V5_SIZE` = 56).
        const REC_OFF: usize = super::format::XFS_BTREE_SBLOCK_V5_SIZE;
        const REC_SZ: usize = 12;
        let max_recs = (bs as usize - REC_OFF) / REC_SZ;
        if numrecs >= max_recs {
            return Err(crate::Error::Unsupported(format!(
                "xfs: REFCNTBT for ag {ag} is full ({numrecs}/{max_recs}); \
                 splitting to a multi-block tree is not implemented"
            )));
        }

        // Detect overlap with an existing record — reject for now
        // (stage 2 simplification). Real workloads triggering this need
        // refcount bump-or-split logic in stage 3.
        for i in 0..numrecs {
            let off = REC_OFF + i * REC_SZ;
            let rec_start = u32::from_be_bytes(block[off..off + 4].try_into().unwrap());
            let rec_count = u32::from_be_bytes(block[off + 4..off + 8].try_into().unwrap());
            let rec_end = rec_start.saturating_add(rec_count);
            let new_end = ag_start_block.saturating_add(blockcount);
            let overlaps = ag_start_block < rec_end && rec_start < new_end;
            if overlaps {
                return Err(crate::Error::Unsupported(format!(
                    "xfs: clone_file: extent (ag {ag}, agblock {ag_start_block}, \
                     count {blockcount}) overlaps existing refcount record at \
                     [{rec_start}..{rec_end}); refcount bump beyond 2 not implemented yet"
                )));
            }
        }

        // Sort-insert: find the position where rc_startblock keeps
        // ascending. The 'high bit of rc_startblock = COW staging
        // extent' convention is left clear (we share regular data).
        let mut insert_pos = numrecs;
        for i in 0..numrecs {
            let off = REC_OFF + i * REC_SZ;
            let rec_start = u32::from_be_bytes(block[off..off + 4].try_into().unwrap());
            if ag_start_block < rec_start {
                insert_pos = i;
                break;
            }
        }
        // Shift records >= insert_pos right by one slot.
        let tail_src = REC_OFF + insert_pos * REC_SZ;
        let tail_dst = tail_src + REC_SZ;
        let tail_len = (numrecs - insert_pos) * REC_SZ;
        block.copy_within(tail_src..tail_src + tail_len, tail_dst);
        // Write the new record.
        block[tail_src..tail_src + 4].copy_from_slice(&ag_start_block.to_be_bytes());
        block[tail_src + 4..tail_src + 8].copy_from_slice(&blockcount.to_be_bytes());
        block[tail_src + 8..tail_src + 12].copy_from_slice(&2u32.to_be_bytes());
        // Bump numrecs.
        let new_numrecs = (numrecs + 1) as u16;
        block[6..8].copy_from_slice(&new_numrecs.to_be_bytes());
        // Re-stamp CRC.
        stamp_v5_btree_block_crc(&mut block);
        dev.write_at(refc_block_off, &block)?;
        Ok(())
    }

    /// Best-effort existence check for `name` under directory `parent_ino`.
    /// Reads the directory's entries and looks for an exact name match.
    /// Used by `clone_file_path` to honour the trait contract ("dst
    /// must not already exist") before allocating any inode.
    fn dir_entry_exists(
        &self,
        dev: &mut dyn BlockDevice,
        parent_ino: u64,
        name: &str,
    ) -> Result<bool> {
        let (buf, core) = self.read_inode(dev, parent_ino)?;
        if !core.is_dir() {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: parent inode {parent_ino} is not a directory"
            )));
        }
        let entries = self.read_dir_entries(dev, &buf, &core)?;
        Ok(entries.iter().any(|e| e.name == name))
    }

    /// Path-based equivalent of [`Self::add_dir`].
    pub fn add_dir_path(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        meta: EntryMeta,
    ) -> Result<u64> {
        self.ensure_write_state(dev)?;
        let (parent_ino, name) = self.split_path_for_create(dev, path)?;
        self.add_dir(dev, parent_ino, &name, meta)
    }

    /// Path-based equivalent of [`Self::add_symlink`].
    pub fn add_symlink_path(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        target: &str,
        meta: EntryMeta,
    ) -> Result<u64> {
        self.ensure_write_state(dev)?;
        let (parent_ino, name) = self.split_path_for_create(dev, path)?;
        self.add_symlink(dev, parent_ino, &name, target, meta)
    }

    /// Path-based equivalent of [`Self::add_device`].
    pub fn add_device_path(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: EntryMeta,
    ) -> Result<u64> {
        self.ensure_write_state(dev)?;
        let (parent_ino, name) = self.split_path_for_create(dev, path)?;
        self.add_device(dev, parent_ino, &name, kind, major, minor, meta)
    }

    /// Path-based equivalent of [`Self::remove`].
    pub fn remove_path(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<u64> {
        self.ensure_write_state(dev)?;
        let (parent_ino, name) = self.split_path_for_create(dev, path)?;
        self.remove(dev, parent_ino, &name)
    }

    /// Split `path` into `(parent_inode, leaf_name)`. `path` must be
    /// absolute and non-empty. The parent directory must already exist.
    fn split_path_for_create(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<(u64, String)> {
        let trimmed = path.trim_start_matches('/');
        if trimmed.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "xfs: cannot create entry at root".into(),
            ));
        }
        let (parent, leaf) = match trimmed.rfind('/') {
            None => ("/", trimmed),
            Some(i) => (&path[..=i], &trimmed[i + 1..]),
        };
        let parent_ino = self.lookup_path_ino(dev, parent)?;
        Ok((parent_ino, leaf.to_string()))
    }

    /// Resolve an absolute path to its inode number. `/` returns the
    /// volume's root inode.
    pub fn lookup_path_ino(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<u64> {
        let (ino, _, _) = self.resolve_path(dev, path)?;
        Ok(ino)
    }
}

#[allow(dead_code)]
fn write_btree_header(
    buf: &mut [u8],
    magic: u32,
    level: u16,
    numrecs: u16,
    uuid: &[u8; 16],
    blkno_ag: u32,
) {
    write_btree_header_for_ag(buf, magic, level, numrecs, uuid, 0, 0, blkno_ag)
}

/// AG-aware variant of [`write_btree_header`]: stamps the AG number
/// into `bb_owner` and computes `bb_blkno` as the absolute basic-block
/// (= device byte / 512) so v5 BNO/CNT/INOBT roots checksum correctly.
#[allow(clippy::too_many_arguments)]
fn write_btree_header_for_ag(
    buf: &mut [u8],
    magic: u32,
    level: u16,
    numrecs: u16,
    uuid: &[u8; 16],
    ag: u32,
    agblocks: u32,
    blkno_ag: u32,
) {
    buf[0..4].copy_from_slice(&magic.to_be_bytes());
    buf[4..6].copy_from_slice(&level.to_be_bytes());
    buf[6..8].copy_from_slice(&numrecs.to_be_bytes());
    buf[8..12].copy_from_slice(&u32::MAX.to_be_bytes());
    buf[12..16].copy_from_slice(&u32::MAX.to_be_bytes());
    // bb_blkno = absolute basic-block number (device byte offset / 512)
    // = `(ag * agblocks + blkno_ag) * (blocksize / 512)`. xfs_repair
    // flags a btree block as "suspect" whenever this is wrong.
    let basic = ((ag as u64) * (agblocks as u64) + blkno_ag as u64) * (XFS_BLOCKSIZE as u64 / 512);
    buf[16..24].copy_from_slice(&basic.to_be_bytes());
    // lsn at 24..32 zero
    buf[32..48].copy_from_slice(uuid);
    buf[48..52].copy_from_slice(&ag.to_be_bytes()); // bb_owner = this AG
    // crc at 52..56 zero (caller stamps)
}

/// Read `..` from an already-on-disk block-format directory and return
/// its inumber.
fn existing_parent(block: &[u8], is_v5: bool) -> Result<u64> {
    let entries = super::dir::decode_block_dir(block, is_v5)?;
    for e in &entries {
        if e.name == ".." {
            return Ok(e.inumber);
        }
    }
    Err(crate::Error::InvalidImage(
        "xfs: directory missing \"..\" entry".into(),
    ))
}

/// Read up to `buf.len()` bytes from `r`, returning how many actually
/// came back (may be less than `buf.len()` only if `r` hit EOF). Wraps
/// `Read::read` in a loop because the trait may return short.
fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut total = 0;
    while total < buf.len() {
        let n = r.read(&mut buf[total..])?;
        if n == 0 {
            break;
        }
        total += n;
    }
    Ok(total).map_err(|e: std::convert::Infallible| match e {})
}

// Reuse this so AG0_METADATA_BLOCKS doesn't show as dead.
#[allow(dead_code)]
const _AG0_META: u32 = AG0_METADATA_BLOCKS;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn write_state_initial_has_root_only() {
        let ws = WriteState::initial_single_ag([0u8; 16]);
        assert_eq!(ws.ags.len(), 1);
        assert_eq!(ws.ags[0].chunks.len(), 1);
        // The formatter pre-allocates root + rbmino + rsumino, leaving
        // 61 inodes free in the root chunk.
        assert_eq!(ws.inodes_used, 3);
        assert_eq!(ws.inodes_free, 61);
    }

    #[test]
    fn add_file_then_list() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        let mut src = std::io::Cursor::new(b"hello world".to_vec());
        let _ino = xfs
            .add_file(
                &mut dev,
                rootino,
                "greet",
                EntryMeta::default(),
                11,
                &mut src,
            )
            .unwrap();
        xfs.flush_writes(&mut dev).unwrap();
        let entries = xfs.list_path(&mut dev, "/").unwrap();
        // Block-format dirs surface "." and ".." verbatim.
        let user: Vec<&crate::fs::DirEntry> = entries
            .iter()
            .filter(|e| e.name != "." && e.name != "..")
            .collect();
        assert_eq!(user.len(), 1);
        assert_eq!(user[0].name, "greet");
        assert_eq!(user[0].kind, crate::fs::EntryKind::Regular);
        // Read the file back.
        let mut r = xfs.open_file_reader(&mut dev, "/greet").unwrap();
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut out).unwrap();
        assert_eq!(&out, b"hello world");
    }

    #[test]
    fn add_dir_and_file_in_subdir() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        let sub = xfs
            .add_dir(&mut dev, rootino, "sub", EntryMeta::default())
            .unwrap();
        let mut src = std::io::Cursor::new(b"x".to_vec());
        xfs.add_file(&mut dev, sub, "f", EntryMeta::default(), 1, &mut src)
            .unwrap();
        xfs.flush_writes(&mut dev).unwrap();
        let entries = xfs.list_path(&mut dev, "/sub").unwrap();
        let user: Vec<&crate::fs::DirEntry> = entries
            .iter()
            .filter(|e| e.name != "." && e.name != "..")
            .collect();
        assert_eq!(user.len(), 1);
        assert_eq!(user[0].name, "f");
    }

    #[test]
    fn add_symlink_inline() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        xfs.add_symlink(
            &mut dev,
            rootino,
            "lnk",
            "/etc/hostname",
            EntryMeta::default(),
        )
        .unwrap();
        xfs.flush_writes(&mut dev).unwrap();
        let target = xfs.read_symlink(&mut dev, "/lnk").unwrap();
        assert_eq!(target, "/etc/hostname");
    }

    /// A symlink target longer than the inode literal area (336 bytes
    /// for a 512-byte v3 inode) must be stored in a separate v5
    /// XSLM-headered block and read back via the extent-list path.
    /// Reopening the image guarantees we're not just round-tripping
    /// through an in-memory cache.
    #[test]
    fn add_symlink_remote_round_trip() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        // 400-byte target — longer than the 336-byte literal area, so
        // add_symlink must promote to remote (extents) form.
        let target: String = "a/long/path/segment/repeated/".repeat(20);
        assert!(target.len() > 336 && target.len() < 4096);
        xfs.add_symlink(&mut dev, rootino, "lnk", &target, EntryMeta::default())
            .unwrap();
        xfs.flush_writes(&mut dev).unwrap();

        // Reopen the volume cold to make sure the on-disk extent + the
        // XSLM block are what carries the target — not a writer-side
        // cache.
        let xfs2 = super::super::Xfs::open(&mut dev).unwrap();
        let got = xfs2.read_symlink(&mut dev, "/lnk").unwrap();
        assert_eq!(got, target);
        // The symlink inode must be di_format=EXTENTS (2), not LOCAL.
        let (_lnk_ino, _buf, core) = xfs2.resolve_path(&mut dev, "/lnk").unwrap();
        assert_eq!(core.format, super::super::inode::DiFormat::Extents);
        assert!(core.is_symlink());
    }

    #[test]
    fn remove_regular_file() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        let mut src = std::io::Cursor::new(b"some data".to_vec());
        let ino = xfs
            .add_file(&mut dev, rootino, "tmp", EntryMeta::default(), 9, &mut src)
            .unwrap();
        // remove it
        let removed = xfs.remove(&mut dev, rootino, "tmp").unwrap();
        assert_eq!(removed, ino);
        xfs.flush_writes(&mut dev).unwrap();
        let entries = xfs.list_path(&mut dev, "/").unwrap();
        let user: Vec<&crate::fs::DirEntry> = entries
            .iter()
            .filter(|e| e.name != "." && e.name != "..")
            .collect();
        assert!(user.is_empty(), "expected dir to be empty, got {user:?}");
    }

    #[test]
    fn remove_empty_dir() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        xfs.add_dir(&mut dev, rootino, "sub", EntryMeta::default())
            .unwrap();
        xfs.remove(&mut dev, rootino, "sub").unwrap();
        xfs.flush_writes(&mut dev).unwrap();
        let entries = xfs.list_path(&mut dev, "/").unwrap();
        let user: Vec<&crate::fs::DirEntry> = entries
            .iter()
            .filter(|e| e.name != "." && e.name != "..")
            .collect();
        assert!(user.is_empty());
    }

    #[test]
    fn remove_nonempty_dir_fails() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        let sub = xfs
            .add_dir(&mut dev, rootino, "sub", EntryMeta::default())
            .unwrap();
        let mut src = std::io::Cursor::new(b"x".to_vec());
        xfs.add_file(&mut dev, sub, "f", EntryMeta::default(), 1, &mut src)
            .unwrap();
        let r = xfs.remove(&mut dev, rootino, "sub");
        assert!(matches!(r, Err(crate::Error::InvalidArgument(_))));
    }

    #[test]
    fn add_xattr_round_trip() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        let mut src = std::io::Cursor::new(b"hello".to_vec());
        let ino = xfs
            .add_file(&mut dev, rootino, "f", EntryMeta::default(), 5, &mut src)
            .unwrap();
        xfs.add_xattr(&mut dev, ino, "user.mime_type", b"text/plain")
            .unwrap();
        xfs.add_xattr(&mut dev, ino, "trusted.foo", b"bar").unwrap();
        let attrs = xfs.read_xattrs(&mut dev, ino).unwrap();
        assert_eq!(attrs.get("user.mime_type"), Some(&b"text/plain".to_vec()));
        assert_eq!(attrs.get("trusted.foo"), Some(&b"bar".to_vec()));
        // File contents still intact.
        let mut reader = xfs.open_file_reader(&mut dev, "/f").unwrap();
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut out).unwrap();
        assert_eq!(out, b"hello");
    }

    /// xattrs that overflow the inode literal area must promote to a
    /// single leaf block. After writing many small xattrs (well past
    /// what shortform's ~256-byte window holds), reopening the volume
    /// and reading them back must return the exact same set.
    #[test]
    fn xattr_leaf_form_round_trip() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        let mut src = std::io::Cursor::new(b"x".to_vec());
        let ino = xfs
            .add_file(&mut dev, rootino, "f", EntryMeta::default(), 1, &mut src)
            .unwrap();

        // 32 attrs × (~16-byte name + 32-byte value) ≈ 1.5 KiB —
        // definitely overflows shortform.
        let mut expected: std::collections::HashMap<String, Vec<u8>> =
            std::collections::HashMap::new();
        for i in 0..32 {
            let name = format!("user.attr_{i:02}");
            let value = format!("the value of attribute number {i:02}").into_bytes();
            xfs.add_xattr(&mut dev, ino, &name, &value).unwrap();
            expected.insert(name, value);
        }
        // Also mix in trusted.* / security.* to exercise the namespace
        // flag bits.
        xfs.add_xattr(&mut dev, ino, "trusted.acme.token", b"opaque")
            .unwrap();
        expected.insert("trusted.acme.token".into(), b"opaque".to_vec());
        xfs.add_xattr(&mut dev, ino, "security.selinux", b"unconfined_u")
            .unwrap();
        expected.insert("security.selinux".into(), b"unconfined_u".to_vec());

        xfs.flush_writes(&mut dev).unwrap();

        // Reopen the image to make sure the on-disk state alone is
        // sufficient — i.e. the leaf-form decode works end-to-end.
        let xfs2 = super::super::Xfs::open(&mut dev).unwrap();
        let got = xfs2.read_xattrs(&mut dev, ino).unwrap();
        assert_eq!(
            got.len(),
            expected.len(),
            "leaf-form attr count mismatch: got {got:#?}",
        );
        for (k, v) in &expected {
            assert_eq!(got.get(k), Some(v), "leaf-form mismatch for {k}");
        }

        // The inode's on-disk aformat must be EXTENTS (2), not LOCAL
        // (1) — confirming we actually exercised the leaf path.
        let (raw, _core) = xfs2.read_inode(&mut dev, ino).unwrap();
        assert_eq!(raw[83], 2, "expected leaf-form aformat=2 on disk");
        assert_eq!(
            u16::from_be_bytes(raw[80..82].try_into().unwrap()),
            1,
            "expected di_anextents=1 for one leaf-form block"
        );
    }

    /// Adding an xattr large enough to overflow shortform on its own
    /// must immediately promote to leaf form, not error.
    #[test]
    fn xattr_single_large_attr_promotes_to_leaf() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        let mut src = std::io::Cursor::new(b"x".to_vec());
        let ino = xfs
            .add_file(&mut dev, rootino, "f", EntryMeta::default(), 1, &mut src)
            .unwrap();
        // 1 KiB value — clearly past shortform's 255-byte cap.
        let value = vec![0x55u8; 1024];
        xfs.add_xattr(&mut dev, ino, "user.big", &value).unwrap();
        xfs.flush_writes(&mut dev).unwrap();

        let xfs2 = super::super::Xfs::open(&mut dev).unwrap();
        let got = xfs2.read_xattrs(&mut dev, ino).unwrap();
        assert_eq!(got.get("user.big"), Some(&value));
    }

    /// remove_xattr deletes from leaf and shortform alike, collapsing
    /// the fork when the last attribute is gone.
    #[test]
    fn xattr_remove_round_trip() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        let mut src = std::io::Cursor::new(b"x".to_vec());
        let ino = xfs
            .add_file(&mut dev, rootino, "f", EntryMeta::default(), 1, &mut src)
            .unwrap();

        // Shortform path.
        xfs.add_xattr(&mut dev, ino, "user.a", b"1").unwrap();
        xfs.add_xattr(&mut dev, ino, "user.b", b"2").unwrap();
        assert!(xfs.remove_xattr(&mut dev, ino, "user.a").unwrap());
        let attrs = xfs.read_xattrs(&mut dev, ino).unwrap();
        assert!(!attrs.contains_key("user.a"));
        assert_eq!(attrs.get("user.b"), Some(&b"2".to_vec()));
        // Remove the last → fork collapses.
        assert!(xfs.remove_xattr(&mut dev, ino, "user.b").unwrap());
        let attrs = xfs.read_xattrs(&mut dev, ino).unwrap();
        assert!(attrs.is_empty());
        let (raw, _core) = xfs.read_inode(&mut dev, ino).unwrap();
        assert_eq!(raw[82], 0, "forkoff should be cleared when last attr gone");

        // Leaf-form path: stuff enough attrs to overflow shortform,
        // then remove every other one and verify the remainder
        // round-trips.
        for i in 0..20 {
            let n = format!("user.k_{i:02}");
            let v = format!("value_{i:02}_padding_to_make_it_bigger");
            xfs.add_xattr(&mut dev, ino, &n, v.as_bytes()).unwrap();
        }
        // Removing a missing attr returns Ok(false).
        assert!(!xfs.remove_xattr(&mut dev, ino, "user.nope").unwrap());
        for i in (0..20).step_by(2) {
            let n = format!("user.k_{i:02}");
            assert!(xfs.remove_xattr(&mut dev, ino, &n).unwrap());
        }
        let attrs = xfs.read_xattrs(&mut dev, ino).unwrap();
        assert_eq!(attrs.len(), 10);
        for i in (1..20).step_by(2) {
            let n = format!("user.k_{i:02}");
            let v = format!("value_{i:02}_padding_to_make_it_bigger");
            assert_eq!(attrs.get(&n), Some(&v.into_bytes()));
        }
    }

    #[test]
    fn xattr_overwrite_replaces() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        let mut src = std::io::Cursor::new(b"x".to_vec());
        let ino = xfs
            .add_file(&mut dev, rootino, "g", EntryMeta::default(), 1, &mut src)
            .unwrap();
        xfs.add_xattr(&mut dev, ino, "user.k", b"v1").unwrap();
        xfs.add_xattr(&mut dev, ino, "user.k", b"v2").unwrap();
        let attrs = xfs.read_xattrs(&mut dev, ino).unwrap();
        assert_eq!(attrs.get("user.k"), Some(&b"v2".to_vec()));
        assert_eq!(attrs.len(), 1);
    }

    /// Manual smoke test: write a fresh XFS image to `/tmp/xfs_test.img`
    /// so the operator can run `xfs_db -r -c 'sb 0' -c 'p' /tmp/xfs_test.img`
    /// against it. Ignored by default; run with `cargo test --lib --
    /// --ignored xfs_writes_image_for_external_tools`.
    #[test]
    #[ignore]
    fn xfs_writes_image_for_external_tools() {
        use crate::block::FileBackend;
        let path = "/tmp/xfs_test.img";
        let _ = std::fs::remove_file(path);
        let f = std::fs::File::create(path).unwrap();
        // 768 MiB ⇒ 3 AGs of 256 MiB each so the multi-AG path is
        // exercised; mkfs.xfs's minimum is 300 MiB.
        f.set_len(768 * 1024 * 1024).unwrap();
        drop(f);
        let mut dev = FileBackend::open(std::path::Path::new(path)).unwrap();
        let opts = super::super::FormatOpts {
            uuid: [0x11u8; 16],
            ..Default::default()
        };
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0x11u8; 16]);
        let rootino = xfs.superblock().rootino;
        let mut src = std::io::Cursor::new(b"hello world".to_vec());
        xfs.add_file(
            &mut dev,
            rootino,
            "greet",
            EntryMeta::default(),
            11,
            &mut src,
        )
        .unwrap();
        let sub = xfs
            .add_dir(&mut dev, rootino, "sub", EntryMeta::default())
            .unwrap();
        let mut src2 = std::io::Cursor::new(b"hi".to_vec());
        let fino = xfs
            .add_file(&mut dev, sub, "leaf", EntryMeta::default(), 2, &mut src2)
            .unwrap();
        xfs.add_xattr(&mut dev, fino, "user.k", b"v").unwrap();
        // Also exercise leaf-form xattrs (overflows shortform).
        for i in 0..20 {
            xfs.add_xattr(
                &mut dev,
                fino,
                &format!("user.spill_{i:02}"),
                b"some longer value padding bytes",
            )
            .unwrap();
        }
        xfs.flush_writes(&mut dev).unwrap();
    }

    #[test]
    fn multi_ag_format_works() {
        // 512 MiB device → 2 AGs (256 MiB each at 65 536 blocks).
        let size = 512 * 1024 * 1024;
        let mut dev = MemoryBackend::new(size);
        let opts = super::super::FormatOpts::default();
        let xfs = super::super::format(&mut dev, &opts).unwrap();
        assert!(
            xfs.ag_count() >= 2,
            "expected ≥2 AGs, got {}",
            xfs.ag_count()
        );
        // Re-open via the read path and list root.
        let mut xfs2 = super::super::Xfs::open(&mut dev).unwrap();
        let entries = xfs2.list_path(&mut dev, "/").unwrap();
        // Empty dir = "." + "..".
        let user: Vec<_> = entries
            .iter()
            .filter(|e| e.name != "." && e.name != "..")
            .collect();
        assert!(user.is_empty());
        // Verify we can write a file and list it.
        xfs2.begin_writes([0u8; 16]);
        let rootino = xfs2.superblock().rootino;
        let mut src = std::io::Cursor::new(b"hi".to_vec());
        xfs2.add_file(&mut dev, rootino, "ma", EntryMeta::default(), 2, &mut src)
            .unwrap();
        xfs2.flush_writes(&mut dev).unwrap();
        let entries = xfs2.list_path(&mut dev, "/").unwrap();
        assert!(entries.iter().any(|e| e.name == "ma"));
    }

    /// Add a file, flush, then re-open via `Xfs::open` (no
    /// `begin_writes`) and add a second file. The second `add_file`
    /// must auto-resume the write state from the on-disk AG headers
    /// so the bump pointer skips the first file's blocks and the new
    /// inode lands in a free slot. Verify both files survive the
    /// final flush + reopen.
    #[test]
    fn reopen_as_writable_round_trip() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        // -- First lifecycle: format, add "a", flush.
        {
            let mut xfs = super::super::format(&mut dev, &opts).unwrap();
            xfs.begin_writes([0u8; 16]);
            let rootino = xfs.superblock().rootino;
            let mut src = std::io::Cursor::new(b"alpha".to_vec());
            xfs.add_file(&mut dev, rootino, "a", EntryMeta::default(), 5, &mut src)
                .unwrap();
            xfs.flush_writes(&mut dev).unwrap();
        }
        // -- Second lifecycle: re-open (no begin_writes), add "b" via
        // the path API which auto-resumes, then flush.
        {
            let mut xfs = super::super::Xfs::open(&mut dev).unwrap();
            // Sanity: "a" is visible through the reader.
            let entries = xfs.list_path(&mut dev, "/").unwrap();
            assert!(
                entries.iter().any(|e| e.name == "a"),
                "first file gone after reopen"
            );
            // No begin_writes — add_file_path will auto-resume.
            let mut src = std::io::Cursor::new(b"beta-data".to_vec());
            xfs.add_file_path(&mut dev, "/b", EntryMeta::default(), 9, &mut src)
                .unwrap();
            xfs.flush_writes(&mut dev).unwrap();
        }
        // -- Third reopen: verify both files are intact and readable.
        let xfs = super::super::Xfs::open(&mut dev).unwrap();
        let entries = xfs.list_path(&mut dev, "/").unwrap();
        let user: Vec<&crate::fs::DirEntry> = entries
            .iter()
            .filter(|e| e.name != "." && e.name != "..")
            .collect();
        let mut names: Vec<&str> = user.iter().map(|e| e.name.as_str()).collect();
        names.sort();
        assert_eq!(names, vec!["a", "b"], "both files visible after reopen");
        // Read each file's bytes.
        let mut r = xfs.open_file_reader(&mut dev, "/a").unwrap();
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut out).unwrap();
        assert_eq!(&out, b"alpha");
        let mut r = xfs.open_file_reader(&mut dev, "/b").unwrap();
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut out).unwrap();
        assert_eq!(&out, b"beta-data");
    }

    /// After flush_writes, `resume_writes` should reconstruct an
    /// allocator state where the inode count + bump pointer match the
    /// on-disk numbers exactly (modulo rounding from the writer's
    /// "extents-only" representation). Asserts the round-trip
    /// invariants.
    #[test]
    fn resume_writes_matches_on_disk_state() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.begin_writes([0u8; 16]);
        let rootino = xfs.superblock().rootino;
        let mut src = std::io::Cursor::new(b"hello".to_vec());
        xfs.add_file(&mut dev, rootino, "f", EntryMeta::default(), 5, &mut src)
            .unwrap();
        xfs.flush_writes(&mut dev).unwrap();

        // Capture the formatter's state so we can diff against the
        // reconstructed one.
        let pre = xfs.write_state.as_ref().unwrap().clone();

        // Reload through the read path and resume.
        let mut xfs2 = super::super::Xfs::open(&mut dev).unwrap();
        xfs2.resume_writes(&mut dev).unwrap();
        let post = xfs2.write_state.as_ref().unwrap();

        assert_eq!(pre.ags.len(), post.ags.len());
        assert_eq!(pre.inodes_used, post.inodes_used);
        assert_eq!(pre.inodes_free, post.inodes_free);
        for (i, (a, b)) in pre.ags.iter().zip(post.ags.iter()).enumerate() {
            assert_eq!(
                a.next_agblock, b.next_agblock,
                "ag {i}: bump pointer mismatch"
            );
            assert_eq!(
                a.chunks.len(),
                b.chunks.len(),
                "ag {i}: chunk count mismatch"
            );
            for (j, (ca, cb)) in a.chunks.iter().zip(b.chunks.iter()).enumerate() {
                assert_eq!(
                    ca.startino_ag, cb.startino_ag,
                    "ag {i} chunk {j}: startino mismatch"
                );
                assert_eq!(ca.ir_free, cb.ir_free, "ag {i} chunk {j}: ir_free mismatch");
            }
        }
    }

    /// Multiple re-open-and-add cycles. Each iteration creates a fresh
    /// inode + extent, flushes, then re-opens from scratch. Every file
    /// added in every iteration must still be visible at the end.
    #[test]
    fn reopen_add_repeatedly() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        // Initial format.
        {
            let xfs = super::super::format(&mut dev, &opts).unwrap();
            // No files written yet; just close out the formatter
            // through Drop (the format() return value carries its own
            // write_state).
            drop(xfs);
        }
        let names = ["one", "two", "three"];
        for n in &names {
            let mut xfs = super::super::Xfs::open(&mut dev).unwrap();
            let mut src = std::io::Cursor::new(n.as_bytes().to_vec());
            xfs.add_file_path(
                &mut dev,
                &format!("/{n}"),
                EntryMeta::default(),
                n.len() as u64,
                &mut src,
            )
            .unwrap();
            xfs.flush_writes(&mut dev).unwrap();
        }
        // Final reopen — every file is listed and readable.
        let xfs = super::super::Xfs::open(&mut dev).unwrap();
        let entries = xfs.list_path(&mut dev, "/").unwrap();
        let user_names: std::collections::HashSet<&str> = entries
            .iter()
            .filter(|e| e.name != "." && e.name != "..")
            .map(|e| e.name.as_str())
            .collect();
        for n in &names {
            assert!(user_names.contains(n), "file {n:?} missing after reopens");
        }
    }

    /// Batched directory writes: create many files in one directory in a
    /// single session (so they accumulate in the dir batch and serialize
    /// once at flush), then reopen and confirm every file is listed and
    /// its contents read back byte-exact.
    #[test]
    fn batched_many_files_one_dir_round_trip() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = super::super::FormatOpts::default();
        let mut xfs = super::super::format(&mut dev, &opts).unwrap();
        xfs.add_dir_path(&mut dev, "/d", EntryMeta::default())
            .unwrap();
        let n_files = 40usize;
        for i in 0..n_files {
            let body = format!("contents-of-file-{i:03}");
            let mut src = std::io::Cursor::new(body.clone().into_bytes());
            xfs.add_file_path(
                &mut dev,
                &format!("/d/f{i:03}"),
                EntryMeta::default(),
                body.len() as u64,
                &mut src,
            )
            .unwrap();
        }
        xfs.flush_writes(&mut dev).unwrap();

        let ro = super::super::Xfs::open(&mut dev).unwrap();
        let listed: std::collections::HashSet<String> = ro
            .list_path(&mut dev, "/d")
            .unwrap()
            .into_iter()
            .filter(|e| e.name != "." && e.name != "..")
            .map(|e| e.name)
            .collect();
        assert_eq!(
            listed.len(),
            n_files,
            "expected {n_files} files, got {listed:?}"
        );
        for i in 0..n_files {
            let name = format!("f{i:03}");
            assert!(listed.contains(&name), "missing {name}");
            let mut r = ro
                .open_file_reader(&mut dev, &format!("/d/{name}"))
                .unwrap();
            let mut got = String::new();
            std::io::Read::read_to_string(&mut r, &mut got).unwrap();
            assert_eq!(got, format!("contents-of-file-{i:03}"));
        }
    }
}
