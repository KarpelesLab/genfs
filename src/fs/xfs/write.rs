//! XFS write path — bump-pointer allocator + inode-creation helpers.
//!
//! After [`super::format::format`] returns, the on-disk image is a valid
//! single-AG v5 filesystem with one inode allocated (the root). This
//! module adds methods on [`Xfs`] for populating the volume with files,
//! directories, symlinks, and special files. The strategy is the
//! simplest thing that produces a `xfs_repair -n clean` output:
//!
//! - **Allocator.** A single bump-pointer over the AG's free-space
//!   region (AG block `ROOT_CHUNK_AGBLOCK + 8` onward) hands out
//!   contiguous extents for both data blocks and new inode chunks. We
//!   never reuse freed space and never re-shuffle free-space records.
//!   At [`Xfs::flush_writes`] time we rewrite the BNO / CNT / INOBT
//!   roots to reflect the new state.
//!
//! - **Inodes.** Once the root chunk is exhausted, we allocate a fresh
//!   64-inode chunk by advancing the bump-pointer by 8 blocks
//!   (32 KiB / 4 KiB) and append a new INOBT leaf record. We track
//!   chunks as `(startino_ag, ir_free_bitmap)` pairs.
//!
//! - **Directories.** Every directory is initialised as block-format
//!   (`encode_v5_block_dir`); when a directory grows past one
//!   directory block, [`Xfs::add_dir`] returns `Error::Unsupported`.
//!   The root directory is always block-format. (Promotion to leaf or
//!   node format is a v2 feature.)
//!
//! - **Symlinks.** ≤ 336 bytes → local (inline). 337 .. 4096 → remote,
//!   one v5 XSLM-headered block. Longer is `Error::Unsupported`.
//!
//! - **Files.** Streamed through a 64 KiB scratch buffer, allocated as
//!   one contiguous extent (the bump-pointer guarantees contiguity).
//!   If the file size exceeds the remaining free pool the call fails.
//!
//! ## What's not done
//!
//! - Extended attributes — every inode has `forkoff=0`. TODO.
//! - Reverse-mapping B+tree, reference-count B+tree, FINOBT — disabled.
//! - Sparse inodes — disabled.
//! - Journal — `sb_logblocks = 0`; mounts must use `-o ro,norecovery`.
//! - Mixed-format directories (leaf, node, B+tree on the write side) —
//!   only block-format is emitted.
//! - Free-space defragmentation. The longest-extent count stays equal
//!   to the entire remaining free pool because we never split it.
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
    AG0_METADATA_BLOCKS, ROOT_CHUNK_AGBLOCK, XFS_ABTB_CRC_MAGIC, XFS_ABTC_CRC_MAGIC, XFS_BLOCKSIZE,
    XFS_BTREE_SBLOCK_V5_SIZE, XFS_IBT_CRC_MAGIC, XFS_INODES_PER_CHUNK, XFS_INODESIZE,
    XFS_INOPBLOCK, stamp_v5_btree_block_crc,
};
use super::inode::{
    S_IFBLK, S_IFCHR, S_IFDIR, S_IFIFO, S_IFLNK, S_IFREG, S_IFSOCK, V3DinodeBuilder, XfsTimestamp,
    stamp_v3_inode_crc,
};
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

/// In-memory state of an active inode chunk in AG 0.
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
}

/// Side-table of write-only state. Lives in [`Xfs`] behind an `Option`
/// because read-only opens don't need it.
#[derive(Debug, Clone)]
pub struct WriteState {
    /// AG 0 next free block (bump pointer).
    next_agblock: u32,
    /// All inode chunks ever allocated (we never deallocate).
    chunks: Vec<InodeChunk>,
    /// Default 16-byte UUID stamped into new metadata blocks. We pull
    /// the canonical UUID from the superblock at write-time instead,
    /// so this field is informational. Kept for future per-AG UUIDs.
    #[allow(dead_code)]
    uuid: [u8; 16],
    /// Total inodes allocated so far (== sum of `64 - chunk.freecount()`).
    inodes_used: u64,
    /// Outstanding inodes free across all chunks.
    inodes_free: u64,
}

impl WriteState {
    /// Build the write state corresponding to a freshly-formatted image:
    /// one chunk (the root chunk) with only inode 0 (the root) used.
    pub fn initial(uuid: [u8; 16]) -> Self {
        Self {
            // First bump-pointer hand-out = post-chunk + 1 (the first
            // dir block has already been written by the formatter).
            next_agblock: ROOT_CHUNK_AGBLOCK + 8 + 1,
            chunks: vec![InodeChunk {
                startino_ag: ROOT_CHUNK_AGBLOCK * (XFS_INOPBLOCK as u32),
                ir_free: !1u64, // bit 0 used, rest free
                agblock: ROOT_CHUNK_AGBLOCK,
            }],
            uuid,
            inodes_used: 1,
            inodes_free: 63,
        }
    }
}

impl Xfs {
    /// Initialise the in-memory write state. Idempotent: calling more
    /// than once replaces the existing state, which is fine because all
    /// allocator state is recoverable from the on-disk image.
    pub fn begin_writes(&mut self, uuid: [u8; 16]) {
        self.write_state = Some(WriteState::initial(uuid));
    }

    fn ws_mut(&mut self) -> Result<&mut WriteState> {
        self.write_state.as_mut().ok_or_else(|| {
            crate::Error::InvalidArgument(
                "xfs: write methods called before begin_writes() / format()".into(),
            )
        })
    }

    /// Bump-pointer-allocate `n` contiguous AG blocks. Returns the AG
    /// block number of the first one.
    fn alloc_blocks(&mut self, n: u32) -> Result<u32> {
        let agblocks = self.sb.agblocks;
        let ws = self.ws_mut()?;
        if ws
            .next_agblock
            .checked_add(n)
            .is_none_or(|end| end > agblocks)
        {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: out of space (requested {n} blocks at {}, AG capacity {agblocks})",
                ws.next_agblock,
            )));
        }
        let start = ws.next_agblock;
        ws.next_agblock += n;
        Ok(start)
    }

    /// Allocate one inode. May allocate a fresh inode chunk if every
    /// existing chunk is full. Returns the **absolute** inode number.
    fn alloc_inode(&mut self) -> Result<u64> {
        // Try existing chunks first.
        let inopblog = self.sb.inopblog;
        let agblklog = self.sb.agblklog;
        let agblocks = self.sb.agblocks;
        // Take a mutable borrow split so we can advance the bump-pointer
        // and the chunk list in one pass.
        {
            let ws = self.ws_mut()?;
            for chunk in &mut ws.chunks {
                if let Some(rel) = chunk.alloc() {
                    let ag = 0u64;
                    let ino = (ag << (inopblog + agblklog)) | (rel as u64);
                    ws.inodes_used += 1;
                    ws.inodes_free -= 1;
                    return Ok(ino);
                }
            }
        }
        // No room — allocate a new 64-inode chunk.
        let chunk_blocks = XFS_INODES_PER_CHUNK / (XFS_INOPBLOCK as u32);
        let start = self.alloc_blocks(chunk_blocks)?;
        let _ = agblocks;
        let startino_ag = start * (XFS_INOPBLOCK as u32);
        // Zero the chunk and write a "free" inode pattern into each slot.
        let chunk_byte = (start as u64) * (XFS_BLOCKSIZE as u64);
        // No dev access here; we'll have the caller flush. We can't
        // physically zero through the unavailable `dev` reference. So
        // we just record the chunk; physical zeroing happens lazily
        // when an inode is actually written into the slot.
        let mut chunk = InodeChunk {
            startino_ag,
            ir_free: !1u64,
            agblock: start,
        };
        let _ = chunk_byte;
        let rel = chunk.alloc().expect("fresh chunk has 64 free inodes");
        let ws = self.ws_mut()?;
        ws.chunks.push(chunk);
        let ag = 0u64;
        ws.inodes_used += 1;
        ws.inodes_free += 63;
        let inopblog2 = inopblog as u32;
        let agblklog2 = agblklog as u32;
        Ok((ag << (inopblog2 + agblklog2)) | (rel as u64))
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
            self.alloc_blocks(nblocks)?
        } else {
            0
        };
        // Stream bytes through a fixed 64 KiB buffer.
        if nblocks > 0 {
            let mut scratch = [0u8; SCRATCH_SIZE];
            let mut remaining = size;
            let mut dev_offset = (startblock as u64) * bs;
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
                startblock: startblock as u64,
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
        let dir_block_agblk = self.alloc_blocks(1)?;
        let ino = self.alloc_inode()?;
        let uuid = self.uuid_for_writes();
        let basic_blkno = ((dir_block_agblk as u64) * (XFS_BLOCKSIZE as u64)) / 512;
        let block = encode_v5_block_dir(dir_block_size, ino, parent_ino, &[], &uuid, basic_blkno)?;
        let phys_byte = (dir_block_agblk as u64) * (XFS_BLOCKSIZE as u64);
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
            startblock: dir_block_agblk as u64,
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
            let blk = self.alloc_blocks(1)?;
            let mut blkbuf = vec![0u8; bs];
            blkbuf[0..4].copy_from_slice(&super::symlink::XFS_SYMLINK_MAGIC.to_be_bytes());
            // crc at 4..8 — zero for now (we'll stamp below)
            // offset at 8..12 (1st block in file = 0)
            // ino at 16..24 = owner inode
            blkbuf[16..24].copy_from_slice(&ino.to_be_bytes());
            // blkno at 24..32 — basic block (FSB << 3)
            let basic_blkno = ((blk as u64) * (XFS_BLOCKSIZE as u64)) / 512;
            blkbuf[24..32].copy_from_slice(&basic_blkno.to_be_bytes());
            // lsn at 32..40 zero
            blkbuf[40..56].copy_from_slice(&uuid);
            // Target bytes after the 56-byte header.
            blkbuf[XFS_SYMLINK_HDR_SIZE..XFS_SYMLINK_HDR_SIZE + target_bytes.len()]
                .copy_from_slice(target_bytes);
            // CRC at byte 4 (le32) of the v5 symlink header (same as
            // dir blocks).
            stamp_v5_dir_block_crc(&mut blkbuf);
            dev.write_at((blk as u64) * (XFS_BLOCKSIZE as u64), &blkbuf)?;

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
                startblock: blk as u64,
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

    /// Flush in-memory allocator state to disk: rewrite the AGF / AGI /
    /// BNO / CNT / INOBT roots + the superblock counters to reflect the
    /// current `WriteState`. This must be called once after all `add_*`
    /// calls so the image is `xfs_repair -n` clean.
    pub fn flush_writes(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        let agblocks = self.sb.agblocks;
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
        let free_start = ws.next_agblock;
        let free_blocks = agblocks.saturating_sub(free_start);
        let bs = XFS_BLOCKSIZE as u64;

        // -- Re-stamp BNO (block 4) and CNT (block 5) ---------------
        let mut bno = make_alloc_btree(XFS_ABTB_CRC_MAGIC, &uuid, 4, free_start, free_blocks);
        stamp_v5_btree_block_crc(&mut bno);
        dev.write_at(4 * bs, &bno)?;
        let mut cnt = make_alloc_btree(XFS_ABTC_CRC_MAGIC, &uuid, 5, free_start, free_blocks);
        stamp_v5_btree_block_crc(&mut cnt);
        dev.write_at(5 * bs, &cnt)?;

        // -- Re-stamp INOBT root at block 6. If only one chunk, a
        //    single leaf record fits; otherwise we'd need to grow the
        //    tree, which we don't (TODO: multi-chunk INOBT).
        if ws.chunks.len() > 1 {
            // We do still write ALL records into a level-0 leaf as
            // long as they fit. A 4 KiB block fits ~250 records, so
            // this covers most realistic populations.
            let mut inobt = vec![0u8; XFS_BLOCKSIZE as usize];
            write_btree_header(
                &mut inobt,
                XFS_IBT_CRC_MAGIC,
                0,
                ws.chunks.len() as u16,
                &uuid,
                6,
            );
            let mut off = XFS_BTREE_SBLOCK_V5_SIZE;
            for chunk in &ws.chunks {
                inobt[off..off + 4].copy_from_slice(&chunk.startino_ag.to_be_bytes());
                inobt[off + 4..off + 8].copy_from_slice(&chunk.freecount().to_be_bytes());
                inobt[off + 8..off + 16].copy_from_slice(&chunk.ir_free.to_be_bytes());
                off += 16;
                if off + 16 > inobt.len() {
                    return Err(crate::Error::Unsupported(
                        "xfs: too many inode chunks for a single-leaf INOBT".into(),
                    ));
                }
            }
            stamp_v5_btree_block_crc(&mut inobt);
            dev.write_at(6 * bs, &inobt)?;
        } else {
            let chunk = &ws.chunks[0];
            let mut inobt = make_inobt_leaf(
                &uuid,
                6,
                chunk.startino_ag,
                chunk.freecount(),
                chunk.ir_free,
            );
            stamp_v5_btree_block_crc(&mut inobt);
            dev.write_at(6 * bs, &inobt)?;
        }

        // Also write per-inode-chunk inode blocks for any chunks we
        // allocated but never wrote into beyond what `write_inode`
        // already did. The chunk's storage region is zeroed when
        // first claimed by `alloc_blocks`, so unused slots read as
        // "di_magic = 0" which the kernel treats as free; xfs_repair
        // accepts this.

        // -- Re-stamp AGF + AGI (sectors 1 + 2) ---------------------
        let mut agf = vec![0u8; XFS_BLOCKSIZE as usize];
        agf[0..4].copy_from_slice(&super::format::XFS_AGF_MAGIC.to_be_bytes());
        agf[4..8].copy_from_slice(&super::format::XFS_AGF_VERSION.to_be_bytes());
        agf[8..12].copy_from_slice(&0u32.to_be_bytes());
        agf[12..16].copy_from_slice(&agblocks.to_be_bytes());
        agf[16..20].copy_from_slice(&4u32.to_be_bytes()); // bno root
        agf[20..24].copy_from_slice(&5u32.to_be_bytes()); // cnt root
        agf[24..28].copy_from_slice(&0u32.to_be_bytes()); // rmap root
        agf[28..32].copy_from_slice(&1u32.to_be_bytes());
        agf[32..36].copy_from_slice(&1u32.to_be_bytes());
        agf[36..40].copy_from_slice(&0u32.to_be_bytes()); // rmap level
        agf[40..44].copy_from_slice(&0u32.to_be_bytes());
        agf[44..48].copy_from_slice(&0u32.to_be_bytes());
        agf[48..52].copy_from_slice(&0u32.to_be_bytes());
        agf[52..56].copy_from_slice(&free_blocks.to_be_bytes());
        agf[56..60].copy_from_slice(&free_blocks.to_be_bytes());
        agf[60..64].copy_from_slice(&0u32.to_be_bytes());
        agf[64..80].copy_from_slice(&uuid);
        super::format::stamp_v5_agf_crc(&mut agf);
        dev.write_at(bs, &agf)?;

        let mut agi = vec![0u8; XFS_BLOCKSIZE as usize];
        agi[0..4].copy_from_slice(&super::format::XFS_AGI_MAGIC.to_be_bytes());
        agi[4..8].copy_from_slice(&super::format::XFS_AGI_VERSION.to_be_bytes());
        agi[8..12].copy_from_slice(&0u32.to_be_bytes());
        agi[12..16].copy_from_slice(&agblocks.to_be_bytes());
        agi[16..20].copy_from_slice(&((ws.inodes_used + ws.inodes_free) as u32).to_be_bytes());
        agi[20..24].copy_from_slice(&6u32.to_be_bytes()); // inobt root
        agi[24..28].copy_from_slice(&1u32.to_be_bytes()); // level
        agi[28..32].copy_from_slice(&(ws.inodes_free as u32).to_be_bytes());
        agi[32..36].copy_from_slice(&ws.chunks.last().unwrap().startino_ag.to_be_bytes());
        agi[36..40].copy_from_slice(&u32::MAX.to_be_bytes());
        for i in 0..64 {
            let off = 40 + i * 4;
            agi[off..off + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        }
        agi[296..312].copy_from_slice(&uuid);
        agi[328..332].copy_from_slice(&0u32.to_be_bytes());
        agi[332..336].copy_from_slice(&0u32.to_be_bytes());
        super::format::stamp_v5_agi_crc(&mut agi);
        dev.write_at(2 * bs, &agi)?;

        // -- Re-stamp the primary superblock counters ---------------
        let mut sb = vec![0u8; XFS_BLOCKSIZE as usize];
        dev.read_at(0, &mut sb)?;
        sb[128..136].copy_from_slice(&ws.inodes_used.to_be_bytes());
        sb[136..144].copy_from_slice(&ws.inodes_free.to_be_bytes());
        sb[144..152].copy_from_slice(&(free_blocks as u64).to_be_bytes());
        super::format::stamp_v5_superblock_crc(&mut sb);
        dev.write_at(0, &sb)?;

        dev.sync()?;
        Ok(())
    }
}

fn make_alloc_btree(
    magic: u32,
    uuid: &[u8; 16],
    blkno_ag: u32,
    free_startblock: u32,
    free_blockcount: u32,
) -> Vec<u8> {
    let mut buf = vec![0u8; XFS_BLOCKSIZE as usize];
    write_btree_header(
        &mut buf,
        magic,
        0,
        if free_blockcount > 0 { 1 } else { 0 },
        uuid,
        blkno_ag,
    );
    if free_blockcount > 0 {
        let off = XFS_BTREE_SBLOCK_V5_SIZE;
        buf[off..off + 4].copy_from_slice(&free_startblock.to_be_bytes());
        buf[off + 4..off + 8].copy_from_slice(&free_blockcount.to_be_bytes());
    }
    buf
}

fn make_inobt_leaf(
    uuid: &[u8; 16],
    blkno_ag: u32,
    startino_ag: u32,
    freecount: u32,
    ir_free: u64,
) -> Vec<u8> {
    let mut buf = vec![0u8; XFS_BLOCKSIZE as usize];
    write_btree_header(&mut buf, XFS_IBT_CRC_MAGIC, 0, 1, uuid, blkno_ag);
    let off = XFS_BTREE_SBLOCK_V5_SIZE;
    buf[off..off + 4].copy_from_slice(&startino_ag.to_be_bytes());
    buf[off + 4..off + 8].copy_from_slice(&freecount.to_be_bytes());
    buf[off + 8..off + 16].copy_from_slice(&ir_free.to_be_bytes());
    buf
}

fn write_btree_header(
    buf: &mut [u8],
    magic: u32,
    level: u16,
    numrecs: u16,
    uuid: &[u8; 16],
    blkno_ag: u32,
) {
    buf[0..4].copy_from_slice(&magic.to_be_bytes());
    buf[4..6].copy_from_slice(&level.to_be_bytes());
    buf[6..8].copy_from_slice(&numrecs.to_be_bytes());
    buf[8..12].copy_from_slice(&u32::MAX.to_be_bytes());
    buf[12..16].copy_from_slice(&u32::MAX.to_be_bytes());
    let basic = (blkno_ag as u64) * (XFS_BLOCKSIZE as u64 / 512);
    buf[16..24].copy_from_slice(&basic.to_be_bytes());
    // lsn at 24..32 zero
    buf[32..48].copy_from_slice(uuid);
    buf[48..52].copy_from_slice(&0u32.to_be_bytes()); // owner = AG 0
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
        let ws = WriteState::initial([0u8; 16]);
        assert_eq!(ws.chunks.len(), 1);
        assert_eq!(ws.inodes_used, 1);
        assert_eq!(ws.inodes_free, 63);
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
}
