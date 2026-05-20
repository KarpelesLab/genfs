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
//! - **Xattrs.** [`Xfs::add_xattr`] / [`Xfs::read_xattrs`] use the
//!   shortform (LOCAL) attribute fork; spill to leaf/node attr blocks
//!   surfaces `Error::Unsupported`.
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

use super::Xfs;
use super::bmbt::Extent;
use super::dir::{
    DataEntry, XFS_DIR3_FT_BLKDEV, XFS_DIR3_FT_CHRDEV, XFS_DIR3_FT_DIR, XFS_DIR3_FT_FIFO,
    XFS_DIR3_FT_REG_FILE, XFS_DIR3_FT_SOCK, XFS_DIR3_FT_SYMLINK, encode_v5_block_dir,
    stamp_v5_dir_block_crc,
};
use super::format::{
    AG0_METADATA_BLOCKS, LOG_AGBLOCK, ROOT_CHUNK_AGBLOCK, XFS_ABTB_CRC_MAGIC, XFS_ABTC_CRC_MAGIC,
    XFS_BLOCKSIZE, XFS_BTREE_SBLOCK_V5_SIZE, XFS_IBT_CRC_MAGIC, XFS_INODES_PER_CHUNK,
    XFS_INODESIZE, XFS_INOPBLOCK, stamp_v5_btree_block_crc,
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
        let agblocks = self.sb.agblocks;
        let ws = self.ws_mut()?;
        let agcount = ws.ags.len() as u32;
        // Try each AG starting at `next_block_ag`, round-robin.
        for offset in 0..agcount {
            let ag = (ws.next_block_ag + offset) % agcount;
            let ag_state = &mut ws.ags[ag as usize];
            // Prefer a freed extent of exact size.
            if let Some(idx) = ag_state.freed_extents.iter().position(|(_s, c)| *c == n) {
                let (start, _) = ag_state.freed_extents.swap_remove(idx);
                ws.next_block_ag = (ag + 1) % agcount;
                return Ok((ag, start));
            }
            // Otherwise bump-allocate.
            let next = ag_state.next_agblock;
            if next.checked_add(n).is_some_and(|end| end <= agblocks) {
                let start = next;
                ag_state.next_agblock = next + n;
                ws.next_block_ag = (ag + 1) % agcount;
                return Ok((ag, start));
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
    fn alloc_blocks_fsb(&mut self, n: u32) -> Result<u64> {
        let (ag, agblk) = self.alloc_blocks_in_any_ag(n)?;
        Ok(((ag as u64) << self.sb.agblklog as u32) | (agblk as u64))
    }

    /// Allocate one inode. Tries existing chunks across AGs in
    /// round-robin order; falls back to allocating a fresh 64-inode
    /// chunk when every existing chunk is full. Returns the absolute
    /// inode number.
    fn alloc_inode(&mut self) -> Result<u64> {
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
        let (ag, agblk) = self.alloc_blocks_in_any_ag(chunk_blocks)?;
        let startino_ag = agblk * (XFS_INOPBLOCK as u32);
        let mut chunk = InodeChunk {
            startino_ag,
            ir_free: !1u64,
            agblock: agblk,
        };
        let rel = chunk.alloc().expect("fresh chunk has 64 free inodes");
        let ws = self.ws_mut()?;
        ws.ags[ag as usize].chunks.push(chunk);
        ws.inodes_used += 1;
        ws.inodes_free += 63;
        ws.next_inode_ag = (ag + 1) % (ws.ags.len() as u32);
        Ok(((ag as u64) << (inopblog + agblklog)) | (rel as u64))
    }

    /// Free `n` blocks starting at FSB `start_fsb` back to its AG's
    /// freed-extent list. The next allocator pass may reuse them.
    fn free_blocks_fsb(&mut self, start_fsb: u64, n: u32) -> Result<()> {
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
    fn append_dir_entry(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_ino: u64,
        name: &str,
        child_ino: u64,
        ftype: u8,
    ) -> Result<u64> {
        // Read the parent inode. Block-format only.
        let (parent_buf, parent_core) = self.read_inode(dev, parent_ino)?;
        if !parent_core.is_dir() {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: parent inode {parent_ino} is not a directory"
            )));
        }
        // Only EXTENTS-format dirs with one extent supported by the
        // writer (i.e. block-format with one logical FS block).
        let extents = self.read_extent_list(dev, &parent_buf, &parent_core)?;
        if extents.len() != 1 || extents[0].blockcount != 1 {
            return Err(crate::Error::Unsupported(
                "xfs: write path only supports block-format (single-extent) directories".into(),
            ));
        }
        let parent_self_extent = extents[0];

        // Re-decode the existing block-format directory to recover the
        // user-visible entries (sans "." / "..").
        let dir_block_size = self.sb.dir_block_size() as usize;
        let mut block = vec![0u8; dir_block_size];
        let phys_byte = self.fsb_to_byte(parent_self_extent.startblock);
        dev.read_at(phys_byte, &mut block)?;
        let existing = super::dir::decode_block_dir(&block, self.sb.is_v5())?;
        let mut all: Vec<(String, u64, u8)> = existing
            .into_iter()
            .filter(|e| e.name != "." && e.name != "..")
            .map(|e: DataEntry| (e.name, e.inumber, e.ftype))
            .collect();
        all.push((name.to_string(), child_ino, ftype));

        // Find parent's parent. The shortform encoding kept it in the
        // first two records' inumber for ".."; we decoded them out. For
        // root, parent_of_parent == self.
        // To avoid re-reading the dir-block "..", we assume root for
        // now and store self-inum (which is correct for the root). For
        // child directories we'd need to track parent inum; we look it
        // up by reading the original "..":
        let parent_parent = existing_parent(&block, self.sb.is_v5())?;

        let basic_blkno = phys_byte / 512;
        let uuid = self.uuid_for_writes();
        let new_block = encode_v5_block_dir(
            dir_block_size,
            parent_ino,
            parent_parent,
            &all,
            &uuid,
            basic_blkno,
        )?;
        dev.write_at(phys_byte, &new_block)?;

        // Update parent inode: di_size = dir_block_size, di_nblocks =
        // 1, di_nextents = 1, mtime, ctime. We rebuild the inode in
        // place.
        let new_size = dir_block_size as u64;
        let (atime, mtime, ctime) = (parent_core.atime, parent_core.mtime, parent_core.ctime);
        let crtime = mtime;
        let nlink = if ftype == XFS_DIR3_FT_DIR {
            parent_core.nlink + 1 // new ".." backref
        } else {
            parent_core.nlink
        };
        let builder = V3DinodeBuilder {
            inodesize: XFS_INODESIZE as usize,
            mode: parent_core.mode,
            format: /*EXTENTS*/ 2,
            uid: parent_core.uid,
            gid: parent_core.gid,
            nlink,
            atime,
            mtime,
            ctime,
            crtime,
            size: new_size,
            nblocks: 1,
            extsize: 0,
            nextents: 1,
            forkoff: 0,
            aformat: 2,
            flags: parent_core.flags,
            generation: parent_core.generation,
            di_ino: parent_ino,
            uuid,
        };
        let mut lit = Vec::with_capacity(16);
        lit.extend_from_slice(&parent_self_extent.encode());
        self.write_inode(dev, parent_ino, builder, &lit)?;
        Ok(new_size)
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
        let ino = self.alloc_inode()?;
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
        let ino = self.alloc_inode()?;
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
        let ino = self.alloc_inode()?;
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
        let ino = self.alloc_inode()?;
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
        let (ino_buf, core) = self.read_inode(dev, ino)?;
        // Decode any existing xattrs from the inode's attr fork.
        let mut current: Vec<(String, Vec<u8>)> = self
            .read_xattrs_from_core(&ino_buf, &core)?
            .into_iter()
            .collect();
        // Replace or append.
        if let Some(slot) = current.iter_mut().find(|(n, _)| n == name) {
            slot.1 = value.to_vec();
        } else {
            current.push((name.to_string(), value.to_vec()));
        }
        // Encode the new shortform area.
        let encoded = super::xattr::encode_shortform(&current)?;
        // Determine the attr-fork allocation.
        //
        // Inode literal area starts at offset 176 (v3 / 512-B inode)
        // and is 336 bytes long. The data fork must keep its current
        // bytes intact — so we must NOT overwrite the bytes the data
        // fork already uses. The data fork's actual occupancy depends
        // on di_format:
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
                    "xfs: add_xattr to BTREE-format inode not supported".into(),
                ));
            }
            super::inode::DiFormat::Unknown(b) => {
                return Err(crate::Error::Unsupported(format!(
                    "xfs: add_xattr to inode with unknown di_format {b}"
                )));
            }
        };
        let data_fork_words = data_fork_bytes.div_ceil(8);
        // attr-fork bytes needed (round up to multiple of 8).
        let attr_words = encoded.len().div_ceil(8);
        // forkoff is the BOUNDARY between forks — i.e., data fork
        // length in 8-byte words. We need `forkoff*8 + attr_size <=
        // lit_size`. Pick the smallest forkoff that fits the data
        // fork.
        let min_forkoff = data_fork_words.max(1);
        let max_forkoff = (lit_size / 8).saturating_sub(attr_words);
        if min_forkoff > max_forkoff {
            return Err(crate::Error::Unsupported(format!(
                "xfs: shortform xattrs ({} bytes) don't fit in inode literal area",
                encoded.len()
            )));
        }
        let forkoff = min_forkoff as u8;
        let attr_off = (forkoff as usize) * 8;
        if attr_off + encoded.len() > lit_size {
            return Err(crate::Error::Unsupported(format!(
                "xfs: shortform xattrs at forkoff={forkoff} overrun literal area"
            )));
        }

        // Rebuild the inode in place. We keep the data fork bytes as-is.
        let data_fork_slice = &ino_buf[176..176 + data_fork_bytes];

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
            nblocks: core.nblocks,
            extsize: 0,
            nextents: core.nextents,
            forkoff,
            aformat: 1, // LOCAL
            flags: core.flags,
            generation: core.generation,
            di_ino: ino,
            uuid: self.uuid_for_writes(),
        };
        let mut buf = builder.build();
        // Restore the data fork.
        buf[176..176 + data_fork_bytes].copy_from_slice(data_fork_slice);
        // Lay down the attribute fork at the forkoff boundary.
        buf[176 + attr_off..176 + attr_off + encoded.len()].copy_from_slice(&encoded);
        stamp_v3_inode_crc(&mut buf);
        let ino_off = self.ino_byte_offset(ino)?;
        dev.write_at(ino_off, &buf)?;
        Ok(())
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
        self.read_xattrs_from_core(&buf, &core)
    }

    /// Decode this inode's attr fork from its already-read on-disk
    /// bytes + core. Used by both [`read_xattrs`] and the read side of
    /// [`add_xattr`] (so updates round-trip without re-reading the
    /// inode).
    fn read_xattrs_from_core(
        &self,
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
            2 => Err(crate::Error::Unsupported(
                "xfs: extents-format attribute fork not supported on read".into(),
            )),
            3 => Err(crate::Error::Unsupported(
                "xfs: btree-format attribute fork not supported on read".into(),
            )),
            other => Err(crate::Error::Unsupported(format!(
                "xfs: unknown attribute-fork format {other}"
            ))),
        }
    }

    /// Flush in-memory allocator state to disk: rewrite the AGF / AGI /
    /// BNO / CNT / INOBT roots + the superblock counters to reflect the
    /// current `WriteState`. Multi-AG safe: every AG's headers and
    /// per-AG B+trees are rewritten from `WriteState.ags[ag]`. Must be
    /// called once after all `add_*` / `remove` calls so the image is
    /// `xfs_repair -n` clean.
    pub fn flush_writes(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
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
                assert_eq!(
                    ca.ir_free, cb.ir_free,
                    "ag {i} chunk {j}: ir_free mismatch"
                );
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
}
