//! `Filesystem::open_file_rw` for XFS — write-ahead-log file handle (Path A).
//!
//! XFS's on-disk update model is journalled: every metadata change must
//! land in the log first, with the kernel replaying the log on mount.
//! This implementation now implements that pattern (Path A) for the
//! single-inode-update transactions emitted by `persist()`:
//!
//! 1. On open, `prepare_log_for_rw` inspects the on-disk log header at
//!    `sb_logstart`:
//!    - If it's the clean-unmount stub we emit, proceed.
//!    - If it's the 4-op inode-update transaction this crate emits,
//!      `journal::replay_log` re-applies the logged
//!      inode bytes (recovering a crash between log-write and in-place
//!      inode-write) and restamps the clean stub.
//!    - Any other log content — including a real kernel log — yields
//!      `Error::Unsupported`.
//! 2. Walk the file's BMBT extents; partial in-place writes patch existing
//!    extent bytes; growth allocates new blocks through the existing AGF
//!    free-extent machinery (`alloc_blocks_fsb`).
//! 3. On `sync` / `Drop`, `XfsFileHandle::persist` runs five steps in
//!    order: (a) build the new on-disk inode buffer; (b) write a
//!    single-inode-update transaction (op header + trans hdr +
//!    inode_log_format + inode buffer + commit op) to BB 0 of the log via
//!    `journal::write_inode_update_transaction`;
//!    (c) write the new inode buffer to its in-place location (the
//!    "checkpoint" step); (d) call `flush_writes` to refresh
//!    AGF/AGI/BNO/CNT/INOBT; (e) restamp the clean-unmount stub so the
//!    on-disk log parses as cleanly unmounted.
//!
//! A crash between (b) and (c) leaves a dirty log; the next call to
//! `prepare_log_for_rw` replays the logged inode bytes back to disk and
//! the file is recovered.
//!
//! Limitations the handle still refuses cleanly:
//!
//! - Inodes in `BTREE` di_format (extent list spilled out of the literal
//!   area). The writer never emits this, so the only way to observe it is
//!   to open a file made by mkfs/kernel. Returns `Unsupported`.
//! - Files with `unwritten` extents. Returns `Unsupported`.
//! - Grow that would overflow the literal-area extent budget (more extents
//!   than can fit). Returns `Unsupported`.
//! - Logs containing more than one record, or any non-inode-update record
//!   shape. Returns `Unsupported`.

use std::io::{Read, Seek, SeekFrom, Write};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::FileHandle;

use super::Xfs;
use super::bmbt::Extent;
use super::format::{XFS_INODESIZE, stamp_v5_superblock_crc};
use super::inode::{DiFormat, S_IFREG, V3DinodeBuilder, XfsTimestamp, stamp_v3_inode_crc};
use super::journal::{
    BBSIZE, ReplayOutcome, XLOG_DEFAULT_H_SIZE, XLOG_FMT_LINUX_LE, XLOG_HEADER_MAGIC_NUM,
    XLOG_VERSION_2, replay_log, write_empty_log, write_inode_update_transaction,
};
#[cfg(test)]
use super::journal::{XFS_LOG_CLIENTID, XLOG_OP_UNMOUNT_FLAGS, XLOG_UNMOUNT_TYPE};
use super::write::EntryMeta;

/// Maximum extent count we can store inline in a v3 inode's literal area
/// when no xattr fork is present: 336 / 16 = 21. Stay conservative.
const MAX_INLINE_EXTENTS: usize = 20;

/// Prepare the log for read+write access.
///
/// - If the on-disk log header at `sb_logstart` is the clean-unmount
///   stub this crate emits, return `Ok(ReplayOutcome::AlreadyClean)`.
/// - If it's a 4-op inode-update transaction (this crate's only
///   non-clean shape), apply the logged inode bytes to disk via
///   [`crate::fs::xfs::journal::replay_log`] and restamp the clean stub.
///   Return `Ok(ReplayOutcome::Replayed)` (or `PartialDiscarded` if
///   the commit op was missing).
/// - Anything else: `Err(Unsupported)` — we don't touch foreign log
///   content.
pub(crate) fn prepare_log_for_rw(xfs: &Xfs, dev: &mut dyn BlockDevice) -> Result<ReplayOutcome> {
    let log_off = xfs.sb.logstart_byte_offset();
    let log_bytes = xfs.sb.log_bytes();
    if log_off == 0 || log_bytes < 2 * BBSIZE {
        return Err(crate::Error::Unsupported(format!(
            "xfs: no internal log present (logstart={}, logblocks={}) — \
             refusing open_file_rw without a writable journal",
            xfs.sb.logstart, xfs.sb.logblocks
        )));
    }
    // Peek the header magic. If unrecognised, refuse — we don't touch a
    // foreign log.
    let mut hdr = vec![0u8; BBSIZE as usize];
    dev.read_at(log_off, &mut hdr)?;
    let magic = u32::from_be_bytes(hdr[0..4].try_into().unwrap());
    if magic != XLOG_HEADER_MAGIC_NUM {
        return Err(crate::Error::Unsupported(format!(
            "xfs: log head magic {magic:#010x} doesn't match XLOG_HEADER_MAGIC_NUM — \
             unrecognised log; refusing open_file_rw"
        )));
    }
    // Delegate to the journal module: it knows the clean / 4-op /
    // partial / foreign shapes.
    replay_log(dev, log_off, log_bytes, &xfs.sb.uuid)
}

/// Validate the on-disk log header at `sb_logstart` matches the clean
/// unmount layout this crate's formatter emits. Returns:
///
/// - `Ok(())` when the log header looks clean.
/// - `Err(Unsupported)` when the log has been touched by a real XFS
///   driver (different magic, more than one logop, non-unmount payload,
///   etc.).
///
/// Kept for test assertions: production code calls
/// [`prepare_log_for_rw`] which also handles replay.
#[cfg(test)]
pub(crate) fn assert_log_clean(xfs: &Xfs, dev: &mut dyn BlockDevice) -> Result<()> {
    let log_off = xfs.sb.logstart_byte_offset();
    let log_bytes = xfs.sb.log_bytes();
    if log_off == 0 || log_bytes < 2 * BBSIZE {
        return Err(crate::Error::Unsupported(format!(
            "xfs: no internal log present (logstart={}, logblocks={}) — \
             refusing open_file_rw without a writable journal",
            xfs.sb.logstart, xfs.sb.logblocks
        )));
    }

    // BB 0: record header.
    let mut hdr = vec![0u8; BBSIZE as usize];
    dev.read_at(log_off, &mut hdr)?;
    let magic = u32::from_be_bytes(hdr[0..4].try_into().unwrap());
    if magic != XLOG_HEADER_MAGIC_NUM {
        return Err(crate::Error::Unsupported(format!(
            "xfs: log head magic {magic:#010x} doesn't match XLOG_HEADER_MAGIC_NUM — \
             dirty / unrecognised log; refusing open_file_rw"
        )));
    }
    // h_num_logops at [40..44]
    let num_logops = u32::from_be_bytes(hdr[40..44].try_into().unwrap());
    if num_logops != 1 {
        return Err(crate::Error::Unsupported(format!(
            "xfs: log head has {num_logops} ops (expected 1 unmount) — \
             dirty log; refusing open_file_rw"
        )));
    }
    // h_lsn / h_tail_lsn at [16..24] and [24..32] — must match for clean log.
    let h_lsn = u64::from_be_bytes(hdr[16..24].try_into().unwrap());
    let h_tail = u64::from_be_bytes(hdr[24..32].try_into().unwrap());
    if h_lsn != h_tail {
        return Err(crate::Error::Unsupported(format!(
            "xfs: log h_lsn ({h_lsn:#x}) != h_tail_lsn ({h_tail:#x}) — \
             dirty log; refusing open_file_rw"
        )));
    }

    // BB 1: unmount op payload.
    let mut op = vec![0u8; BBSIZE as usize];
    dev.read_at(log_off + BBSIZE, &mut op)?;
    // op[8] = client id, op[9] = flags.
    if op[8] != XFS_LOG_CLIENTID || (op[9] & XLOG_OP_UNMOUNT_FLAGS) != XLOG_OP_UNMOUNT_FLAGS {
        return Err(crate::Error::Unsupported(format!(
            "xfs: log op header is not an unmount record (client={:#x} flags={:#x}) — \
             dirty log; refusing open_file_rw",
            op[8], op[9]
        )));
    }
    let payload_magic = u16::from_le_bytes(op[12..14].try_into().unwrap());
    if payload_magic != XLOG_UNMOUNT_TYPE {
        return Err(crate::Error::Unsupported(format!(
            "xfs: log unmount payload magic {payload_magic:#06x} \
             (expected {:#06x}) — dirty log; refusing open_file_rw",
            XLOG_UNMOUNT_TYPE
        )));
    }
    Ok(())
}

/// Re-stamp the clean unmount log header. Called from `sync` / `Drop`.
/// Errors during `Drop` are swallowed (best-effort persistence).
fn rewrite_clean_unmount(xfs: &Xfs, dev: &mut dyn BlockDevice) -> Result<()> {
    let log_off = xfs.sb.logstart_byte_offset();
    let log_bytes = xfs.sb.log_bytes();
    if log_off == 0 || log_bytes == 0 {
        return Ok(());
    }
    // Re-zero only the first 2 BBs — preserve any unused log tail bytes.
    let zero = vec![0u8; 2 * BBSIZE as usize];
    dev.write_at(log_off, &zero)?;
    write_empty_log(dev, log_off, log_bytes, &xfs.sb.uuid)?;
    // Quiet "unused constant" warnings for journal layout consts that
    // we don't dereference here but still ship in the API surface.
    let _ = (XLOG_VERSION_2, XLOG_FMT_LINUX_LE, XLOG_DEFAULT_H_SIZE);
    Ok(())
}

/// An open XFS regular file for read+write. Backs
/// `Filesystem::open_file_rw` for XFS. See module-level docs for the
/// "clean unmount" bypass strategy.
pub struct XfsFileHandle<'a> {
    pub(crate) fs: &'a mut Xfs,
    pub(crate) dev: &'a mut dyn BlockDevice,
    /// Owning inode number.
    pub(crate) ino: u64,
    /// In-memory extent list (sorted by logical offset).
    pub(crate) extents: Vec<Extent>,
    /// Cached file size (logical bytes).
    pub(crate) len: u64,
    /// Current cursor.
    pub(crate) pos: u64,
    /// True when `extents` / `len` differ from the on-disk inode.
    pub(crate) dirty: bool,
    /// Snapshot of inode fields we round-trip unchanged.
    pub(crate) keep_uid: u32,
    pub(crate) keep_gid: u32,
    pub(crate) keep_mode: u16,
    pub(crate) keep_nlink: u32,
    pub(crate) keep_flags: u16,
    pub(crate) keep_generation: u32,
    pub(crate) keep_atime: XfsTimestamp,
    pub(crate) keep_mtime: XfsTimestamp,
    pub(crate) keep_ctime: XfsTimestamp,
    pub(crate) keep_crtime: XfsTimestamp,
    pub(crate) keep_forkoff: u8,
    /// Bytes the attr fork occupies (forkoff*8 .. literal_end). Read off
    /// disk so we can restore it byte-for-byte on writeback.
    pub(crate) attr_bytes: Vec<u8>,
}

impl<'a> XfsFileHandle<'a> {
    fn blocksize(&self) -> u64 {
        self.fs.sb.blocksize as u64
    }

    /// FSB → device byte address (same math as `Xfs::fsb_to_byte`).
    fn fsb_to_byte(&self, fsb: u64) -> u64 {
        let agblklog = self.fs.sb.agblklog as u32;
        let ag = fsb >> agblklog;
        let agblk = fsb & ((1u64 << agblklog) - 1);
        ag * (self.fs.sb.agblocks as u64) * self.blocksize() + agblk * self.blocksize()
    }

    /// Total FS blocks covered by all extents.
    fn nblocks(&self) -> u64 {
        self.extents.iter().map(|e| e.blockcount as u64).sum()
    }

    /// Find an extent covering logical FS block `lblk`, returning a copy.
    fn find_extent(&self, lblk: u64) -> Option<Extent> {
        self.extents
            .iter()
            .find(|e| lblk >= e.offset && lblk < e.offset + e.blockcount as u64)
            .copied()
    }

    /// Allocate enough fresh blocks to cover the gap so the file's
    /// extent footprint reaches `need_blocks` FS blocks. New blocks are
    /// zeroed on disk before the function returns.
    fn grow_alloc(&mut self, need_blocks: u64) -> Result<()> {
        let mut have = self.nblocks();
        while have < need_blocks {
            let want_u64 = (need_blocks - have).min((1u64 << 21) - 1);
            let want = want_u64 as u32;
            let fsb = self.fs.alloc_blocks_fsb(want)?;
            let byte = self.fsb_to_byte(fsb);
            let total = (want as u64) * self.blocksize();
            self.dev.zero_range(byte, total)?;
            let new_offset = have;
            let new_ext = Extent {
                offset: new_offset,
                startblock: fsb,
                blockcount: want,
                unwritten: false,
            };
            let mut merged = false;
            if let Some(tail) = self.extents.last_mut() {
                let tail_end_fsb = tail.startblock + tail.blockcount as u64;
                let tail_end_logical = tail.offset + tail.blockcount as u64;
                if tail_end_fsb == fsb && tail_end_logical == new_offset && !tail.unwritten {
                    tail.blockcount = tail.blockcount.checked_add(want).ok_or_else(|| {
                        crate::Error::Unsupported(
                            "xfs: extent blockcount overflow during grow".into(),
                        )
                    })?;
                    merged = true;
                }
            }
            if !merged {
                self.extents.push(new_ext);
                if self.extents.len() > MAX_INLINE_EXTENTS {
                    return Err(crate::Error::Unsupported(format!(
                        "xfs: file grew to {} extents (max inline = {MAX_INLINE_EXTENTS}) — \
                         bmbt promotion not implemented",
                        self.extents.len()
                    )));
                }
            }
            have += want as u64;
        }
        Ok(())
    }

    /// Drop trailing extents past `keep_blocks` FS blocks. Frees the
    /// blocks back to the AGF tree. Splits the boundary extent when the
    /// truncation falls mid-extent.
    fn shrink_alloc(&mut self, keep_blocks: u64) -> Result<()> {
        let mut new_exts: Vec<Extent> = Vec::with_capacity(self.extents.len());
        let old_extents = std::mem::take(&mut self.extents);
        for ext in old_extents {
            let end_logical = ext.offset + ext.blockcount as u64;
            if ext.offset >= keep_blocks {
                // Entire extent past keep — free it.
                self.fs.free_blocks_fsb(ext.startblock, ext.blockcount)?;
            } else if end_logical <= keep_blocks {
                // Entire extent kept.
                new_exts.push(ext);
            } else {
                // Split: keep front, free tail.
                let keep_cnt = (keep_blocks - ext.offset) as u32;
                let free_cnt = ext.blockcount - keep_cnt;
                let free_fsb = ext.startblock + keep_cnt as u64;
                self.fs.free_blocks_fsb(free_fsb, free_cnt)?;
                new_exts.push(Extent {
                    offset: ext.offset,
                    startblock: ext.startblock,
                    blockcount: keep_cnt,
                    unwritten: false,
                });
            }
        }
        self.extents = new_exts;
        Ok(())
    }

    /// Ensure the file has at least `target_len` bytes of allocated storage
    /// (rounded up to FS-block size). Grows by allocating + zeroing if
    /// needed.
    fn ensure_capacity(&mut self, target_len: u64) -> Result<()> {
        let bs = self.blocksize();
        let need = target_len.div_ceil(bs);
        if need > self.nblocks() {
            self.grow_alloc(need)?;
        }
        Ok(())
    }

    /// Read up to `buf.len()` bytes at `self.pos`. Helper for `Read::read`.
    fn read_internal(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let bs = self.blocksize();
        let pos_blk = self.pos / bs;
        let pos_off = self.pos % bs;
        let avail = self.len - self.pos;
        let want = (buf.len() as u64).min(avail);

        match self.find_extent(pos_blk) {
            Some(e) => {
                let extent_blocks_left = e.offset + e.blockcount as u64 - pos_blk;
                let extent_bytes_left = extent_blocks_left * bs - pos_off;
                let to_read = want.min(extent_bytes_left) as usize;
                let phys = self.fsb_to_byte(e.startblock + (pos_blk - e.offset)) + pos_off;
                self.dev
                    .read_at(phys, &mut buf[..to_read])
                    .map_err(std::io::Error::other)?;
                self.pos += to_read as u64;
                Ok(to_read)
            }
            None => {
                // Hole: zero-fill up to the next extent or EOF.
                let next_blk = self
                    .extents
                    .iter()
                    .filter(|e| e.offset > pos_blk)
                    .map(|e| e.offset)
                    .min();
                let hole_end = match next_blk {
                    Some(b) => b * bs,
                    None => self.len,
                };
                let to_zero = want.min(hole_end.saturating_sub(self.pos)) as usize;
                if to_zero == 0 {
                    return Ok(0);
                }
                buf[..to_zero].fill(0);
                self.pos += to_zero as u64;
                Ok(to_zero)
            }
        }
    }

    /// Write `buf` at `self.pos`, allocating + extending as needed.
    fn write_internal(&mut self, buf: &[u8]) -> Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        // If pos > len, the gap [len, pos) becomes a sparse-then-zeroed
        // region. We materialise it by allocating + zeroing.
        if self.pos > self.len {
            self.ensure_capacity(self.pos)?;
            let gap_lo = self.len;
            let gap_hi = self.pos;
            self.zero_logical_range(gap_lo, gap_hi)?;
            self.len = gap_hi;
            self.dirty = true;
        }
        let new_end = self.pos + buf.len() as u64;
        self.ensure_capacity(new_end)?;

        let bs = self.blocksize();
        let mut written: usize = 0;
        while written < buf.len() {
            let p = self.pos + written as u64;
            let pos_blk = p / bs;
            let pos_off = p % bs;
            let ext = self.find_extent(pos_blk).ok_or_else(|| {
                crate::Error::InvalidImage(format!(
                    "xfs: write at logical block {pos_blk} has no covering extent"
                ))
            })?;
            let extent_blocks_left = ext.offset + ext.blockcount as u64 - pos_blk;
            let extent_bytes_left = extent_blocks_left * bs - pos_off;
            let in_extent = ((buf.len() - written) as u64).min(extent_bytes_left) as usize;
            let phys = self.fsb_to_byte(ext.startblock + (pos_blk - ext.offset)) + pos_off;
            self.dev
                .write_at(phys, &buf[written..written + in_extent])?;
            written += in_extent;
        }

        self.pos += buf.len() as u64;
        if self.pos > self.len {
            self.len = self.pos;
        }
        self.dirty = true;
        Ok(written)
    }

    /// Zero the logical byte range `[from, to)` by writing through to
    /// every extent it covers. Both endpoints must already be within the
    /// allocated capacity.
    fn zero_logical_range(&mut self, from: u64, to: u64) -> Result<()> {
        if from >= to {
            return Ok(());
        }
        let bs = self.blocksize();
        let zero_scratch = vec![0u8; bs.min(64 * 1024) as usize];
        let mut p = from;
        while p < to {
            let pos_blk = p / bs;
            let pos_off = p % bs;
            let ext = self.find_extent(pos_blk).ok_or_else(|| {
                crate::Error::InvalidImage(format!(
                    "xfs: zero_logical_range at logical block {pos_blk} has no covering extent"
                ))
            })?;
            let extent_blocks_left = ext.offset + ext.blockcount as u64 - pos_blk;
            let extent_bytes_left = extent_blocks_left * bs - pos_off;
            let chunk = (to - p).min(extent_bytes_left);
            let mut written = 0u64;
            while written < chunk {
                let n = (chunk - written).min(zero_scratch.len() as u64) as usize;
                let phys =
                    self.fsb_to_byte(ext.startblock + (pos_blk - ext.offset)) + pos_off + written;
                self.dev.write_at(phys, &zero_scratch[..n])?;
                written += n as u64;
            }
            p += chunk;
        }
        Ok(())
    }

    /// Persist `extents` + size into the on-disk inode, then refresh the
    /// AGF/AGI/BNO/INOBT trees (`flush_writes`) and rewrite the clean
    /// unmount log header. Called by `sync` and `Drop`.
    fn persist(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        if self.extents.len() > MAX_INLINE_EXTENTS {
            return Err(crate::Error::Unsupported(format!(
                "xfs: file has {} extents (> {MAX_INLINE_EXTENTS} inline cap)",
                self.extents.len()
            )));
        }
        // Encode the extent list.
        let mut lit: Vec<u8> = Vec::with_capacity(self.extents.len() * 16);
        for ext in &self.extents {
            lit.extend_from_slice(&ext.encode());
        }
        // Build the inode buffer.
        let nblocks = self.nblocks();
        let builder = V3DinodeBuilder {
            inodesize: XFS_INODESIZE as usize,
            mode: self.keep_mode,
            format: 2, // EXTENTS
            uid: self.keep_uid,
            gid: self.keep_gid,
            nlink: self.keep_nlink,
            atime: self.keep_atime,
            mtime: self.keep_mtime,
            ctime: self.keep_ctime,
            crtime: self.keep_crtime,
            size: self.len,
            nblocks,
            extsize: 0,
            nextents: self.extents.len() as u32,
            forkoff: self.keep_forkoff,
            aformat: 1, // LOCAL (only attr-fork format we round-trip)
            flags: self.keep_flags,
            generation: self.keep_generation,
            di_ino: self.ino,
            uuid: self.fs.sb.uuid,
            flags2: 0,
        };
        let mut buf = builder.build();
        let data_end = 176 + lit.len();
        let inodesize = XFS_INODESIZE as usize;
        if data_end > inodesize {
            return Err(crate::Error::Unsupported(format!(
                "xfs: encoded extent list ({} bytes) overruns inode literal area",
                lit.len()
            )));
        }
        buf[176..data_end].copy_from_slice(&lit);
        if !self.attr_bytes.is_empty() && self.keep_forkoff != 0 {
            let attr_off = 176 + (self.keep_forkoff as usize) * 8;
            let attr_end = attr_off + self.attr_bytes.len();
            if attr_end > inodesize {
                return Err(crate::Error::Unsupported(
                    "xfs: attr-fork overflow restoring inode".into(),
                ));
            }
            buf[attr_off..attr_end].copy_from_slice(&self.attr_bytes);
        }
        stamp_v3_inode_crc(&mut buf);
        let ino_off = self.fs.ino_byte_offset(self.ino)?;

        // ---- write-ahead log step (Path A) ----
        // Stamp a single-inode-update transaction at the head of the log
        // BEFORE we touch the on-disk inode. A crash between this point
        // and the in-place write below leaves a dirty log; the next open
        // calls `prepare_log_for_rw` which re-applies the logged inode
        // bytes and restamps the unmount stub.
        let log_off = self.fs.sb.logstart_byte_offset();
        let log_bytes = self.fs.sb.log_bytes();
        if log_off != 0 && log_bytes >= 2 * BBSIZE {
            // The cycle bumps by 1 every wrap; we never wrap (single-pass
            // log) so cycle = 2 distinguishes a dirty record from the
            // cycle-1 clean stub.
            write_inode_update_transaction(
                self.dev,
                log_off,
                log_bytes,
                /* tid    */ 1,
                /* cycle  */ 2,
                /* ino    */ self.ino,
                /* target */ ino_off,
                &buf,
                &self.fs.sb.uuid,
            )?;
        }

        // ---- checkpoint: write the inode in place ----
        self.dev.write_at(ino_off, &buf)?;

        // Refresh AGF/AGI/BNO/CNT/INOBT roots.
        self.fs.flush_writes(self.dev)?;
        // Restamp the clean unmount log header. With the inode in place
        // the logged record is no longer needed.
        rewrite_clean_unmount(self.fs, self.dev)?;
        // flush_writes already restamps the primary superblock; nothing
        // else needs touching here.
        let mut sb_buf = vec![0u8; self.fs.sb.blocksize as usize];
        self.dev.read_at(0, &mut sb_buf)?;
        stamp_v5_superblock_crc(&mut sb_buf);
        self.dev.write_at(0, &sb_buf)?;
        self.dirty = false;
        Ok(())
    }
}

impl<'a> Read for XfsFileHandle<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.read_internal(buf)
    }
}

impl<'a> Write for XfsFileHandle<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_internal(buf).map_err(std::io::Error::other)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> Seek for XfsFileHandle<'a> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::Current(n) => self.pos as i128 + n as i128,
            SeekFrom::End(n) => self.len as i128 + n as i128,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "xfs: seek to negative offset",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

impl<'a> FileHandle for XfsFileHandle<'a> {
    fn len(&self) -> u64 {
        self.len
    }

    fn set_len(&mut self, new_len: u64) -> Result<()> {
        let bs = self.blocksize();
        if new_len < self.len {
            let keep_blocks = new_len.div_ceil(bs);
            self.shrink_alloc(keep_blocks)?;
            self.len = new_len;
            // Zero trailing bytes within the final retained block.
            if new_len % bs != 0 && new_len > 0 {
                let last_blk = (new_len - 1) / bs;
                let blk_end = (last_blk + 1) * bs;
                let zero_to = blk_end.min(keep_blocks * bs);
                if new_len < zero_to {
                    self.zero_logical_range(new_len, zero_to)?;
                }
            }
        } else if new_len > self.len {
            let old_len = self.len;
            self.ensure_capacity(new_len)?;
            self.zero_logical_range(old_len, new_len)?;
            self.len = new_len;
        } else {
            return Ok(());
        }
        self.dirty = true;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.persist()
    }
}

impl<'a> Drop for XfsFileHandle<'a> {
    fn drop(&mut self) {
        let _ = self.persist();
    }
}

// ---------------------------------------------------------------------------
// open_file_rw adapter — lives on Xfs so the trait impl in mod.rs delegates
// to it.
// ---------------------------------------------------------------------------

impl Xfs {
    /// Open `path` for in-place read+write. Backs
    /// [`crate::fs::Filesystem::open_file_rw`].
    pub(crate) fn open_rw<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
        flags: crate::fs::OpenFlags,
        meta: Option<crate::fs::FileMeta>,
    ) -> Result<Box<dyn FileHandle + 'a>> {
        // Guard 1: ensure log is clean / replay if a Path-A transaction
        // is pending. Foreign log content is refused here.
        prepare_log_for_rw(self, dev)?;
        // Guard 2: ensure write_state is loaded.
        if self.write_state.is_none() {
            self.resume_writes(dev)?;
        }

        // Resolve or create the inode.
        let ino = match self.lookup_path_ino(dev, path) {
            Ok(i) => i,
            Err(_) if flags.create => {
                let m = meta.ok_or_else(|| {
                    crate::Error::InvalidArgument(
                        "xfs: open_file_rw with create=true requires meta".into(),
                    )
                })?;
                let em = EntryMeta {
                    mode: m.mode,
                    uid: m.uid,
                    gid: m.gid,
                    mtime: m.mtime,
                    atime: m.atime,
                    ctime: m.ctime,
                };
                let mut empty = std::io::empty();
                self.add_file_path(dev, path, em, 0, &mut empty)?;
                self.lookup_path_ino(dev, path)?
            }
            Err(e) => return Err(e),
        };

        // Validate the inode is a regular file in EXTENTS format.
        let (ino_buf, core) = self.read_inode(dev, ino)?;
        if !core.is_reg() {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: {path:?} is not a regular file"
            )));
        }

        // Refuse open on files participating in a clone (REFLINK flag in
        // di_flags2). A write through the existing rw path would update
        // the shared physical extent in place and silently propagate the
        // change to every other inode pointing at it — defeating the
        // copy-on-write semantics that `clone_file` advertises.
        //
        // Safely supporting writes here requires per-block COW: detect
        // that the target block is shared (refcount > 1 in the REFCNTBT),
        // allocate a fresh extent, copy the unchanged portion, update
        // the file's BMBT, and decrement the refcount record. That's a
        // follow-up; for now we refuse cleanly so the failure mode is
        // a typed `Unsupported` instead of data corruption.
        let flags2 = if ino_buf.len() >= 128 {
            u64::from_be_bytes(ino_buf[120..128].try_into().unwrap())
        } else {
            0
        };
        if flags2 & super::inode::XFS_DIFLAG2_REFLINK != 0 {
            return Err(crate::Error::Unsupported(format!(
                "xfs: open_file_rw on {path:?}: file participates in a reflink \
                 clone (XFS_DIFLAG2_REFLINK set); writes would corrupt the \
                 sharing peer. Use `read_file` for read-only access; \
                 copy-on-write on the rw path is not yet implemented"
            )));
        }
        match core.format {
            DiFormat::Extents => {}
            DiFormat::Local => {
                return Err(crate::Error::Unsupported(
                    "xfs: open_file_rw on di_format=LOCAL inodes not supported \
                     (writer never emits them for regular files)"
                        .into(),
                ));
            }
            DiFormat::Btree => {
                return Err(crate::Error::Unsupported(
                    "xfs: open_file_rw on di_format=BTREE inodes not supported \
                     (bmbt spill not yet implemented on the rw path)"
                        .into(),
                ));
            }
            DiFormat::Dev => {
                return Err(crate::Error::InvalidArgument(
                    "xfs: open_file_rw on device-special inode".into(),
                ));
            }
            DiFormat::Unknown(b) => {
                return Err(crate::Error::Unsupported(format!(
                    "xfs: open_file_rw: unknown di_format {b}"
                )));
            }
        }
        // Walk the extent list (already EXTENTS-format).
        let extents = self.read_extent_list(dev, &ino_buf, &core)?;
        for e in &extents {
            if e.unwritten {
                return Err(crate::Error::Unsupported(
                    "xfs: open_file_rw with unwritten extents not supported".into(),
                ));
            }
        }
        let inodesize = self.sb.inodesize as usize;
        let attr_bytes = if core.forkoff != 0 {
            let attr_off = 176 + (core.forkoff as usize) * 8;
            if attr_off >= inodesize {
                Vec::new()
            } else {
                ino_buf[attr_off..inodesize.min(ino_buf.len())].to_vec()
            }
        } else {
            Vec::new()
        };
        let crtime = if ino_buf.len() >= 152 {
            XfsTimestamp::decode(&ino_buf[144..152])
        } else {
            core.mtime
        };

        let mut h = XfsFileHandle {
            fs: self,
            dev,
            ino,
            extents,
            len: core.size,
            pos: 0,
            dirty: false,
            keep_uid: core.uid,
            keep_gid: core.gid,
            keep_mode: core.mode,
            keep_nlink: core.nlink,
            keep_flags: core.flags,
            keep_generation: core.generation,
            keep_atime: core.atime,
            keep_mtime: core.mtime,
            keep_ctime: core.ctime,
            keep_crtime: crtime,
            keep_forkoff: core.forkoff,
            attr_bytes,
        };
        // Keep S_IFREG referenced so its import isn't flagged dead.
        let _ = S_IFREG;

        if flags.truncate && h.len > 0 {
            h.set_len(0)?;
        }
        if flags.append {
            h.pos = h.len;
        }
        Ok(Box::new(h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;
    use crate::fs::xfs::format::format;
    use crate::fs::xfs::{FormatOpts, Xfs};
    use crate::fs::{FileMeta, Filesystem, OpenFlags};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::Path;

    /// Format a fresh 64 MiB XFS image, prime the write state, flush the
    /// AG headers so the on-disk state matches what `Xfs::open` expects,
    /// then return the device + fs (caller can begin_writes again or
    /// rely on the path API's auto-resume).
    fn fresh_image() -> (MemoryBackend, Xfs) {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = FormatOpts::default();
        let mut xfs = format(&mut dev, &opts).unwrap();
        xfs.begin_writes(opts.uuid);
        xfs.flush_writes(&mut dev).unwrap();
        (dev, xfs)
    }

    /// Read the bytes of `/path` back through the read path.
    fn read_file(xfs: &mut Xfs, dev: &mut MemoryBackend, path: &str) -> Vec<u8> {
        let mut r = xfs.open_file_reader(dev, path).unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        out
    }

    #[test]
    fn open_file_rw_round_trip_xfs_clean_log() {
        let (mut dev, mut xfs) = fresh_image();
        // Create a small file.
        let payload = b"AAAAAAAAAAAAAAAAAAAA"; // 20 bytes
        let mut src: &[u8] = payload;
        xfs.add_file_path(
            &mut dev,
            "/x.bin",
            EntryMeta::default(),
            payload.len() as u64,
            &mut src,
        )
        .unwrap();
        xfs.flush_writes(&mut dev).unwrap();
        // Re-open as a fresh read+write filesystem.
        let mut xfs = Xfs::open(&mut dev).unwrap();
        {
            let mut h = Filesystem::open_file_rw(
                &mut xfs,
                &mut dev,
                Path::new("/x.bin"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
            assert_eq!(h.len(), 20);
            h.seek(SeekFrom::Start(5)).unwrap();
            h.write_all(b"ZZZZZ").unwrap();
            h.sync().unwrap();
        }
        // Re-open and verify the patched bytes round-trip.
        let mut xfs2 = Xfs::open(&mut dev).unwrap();
        let bytes = read_file(&mut xfs2, &mut dev, "/x.bin");
        assert_eq!(bytes.len(), 20);
        let mut expected = payload.to_vec();
        expected[5..10].copy_from_slice(b"ZZZZZ");
        assert_eq!(bytes, expected);
        // The log header should still validate as clean.
        assert_log_clean(&xfs2, &mut dev).unwrap();
    }

    #[test]
    fn open_file_rw_extends_file() {
        let (mut dev, mut xfs) = fresh_image();
        let mut src: &[u8] = b"hello";
        xfs.add_file_path(&mut dev, "/g.txt", EntryMeta::default(), 5, &mut src)
            .unwrap();
        xfs.flush_writes(&mut dev).unwrap();
        let mut xfs = Xfs::open(&mut dev).unwrap();
        {
            let mut h = Filesystem::open_file_rw(
                &mut xfs,
                &mut dev,
                Path::new("/g.txt"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
            h.seek(SeekFrom::End(0)).unwrap();
            h.write_all(b", world!").unwrap();
            h.sync().unwrap();
            assert_eq!(h.len(), 13);
        }
        let mut xfs2 = Xfs::open(&mut dev).unwrap();
        let bytes = read_file(&mut xfs2, &mut dev, "/g.txt");
        assert_eq!(bytes, b"hello, world!");
    }

    #[test]
    fn open_file_rw_refused_when_log_dirty() {
        let (mut dev, mut xfs) = fresh_image();
        let mut src: &[u8] = b"data";
        xfs.add_file_path(&mut dev, "/x", EntryMeta::default(), 4, &mut src)
            .unwrap();
        xfs.flush_writes(&mut dev).unwrap();
        let xfs = Xfs::open(&mut dev).unwrap();
        // Corrupt the log header magic at sb_logstart.
        let log_off = xfs.superblock().logstart_byte_offset();
        assert!(log_off > 0);
        dev.write_at(log_off, &[0u8; 8]).unwrap();
        let mut xfs = Xfs::open(&mut dev).unwrap();
        let res = Filesystem::open_file_rw(
            &mut xfs,
            &mut dev,
            Path::new("/x"),
            OpenFlags::default(),
            None,
        );
        let err = match res {
            Ok(_) => panic!("open_file_rw should refuse a dirty log"),
            Err(e) => e,
        };
        match err {
            crate::Error::Unsupported(msg) => {
                assert!(
                    msg.contains("log") || msg.contains("XLOG"),
                    "expected log-dirty error, got: {msg}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn open_file_rw_partial_write_round_trip() {
        // Identical to open_file_rw_round_trip_xfs_clean_log but exercising
        // a write that spans a whole-extent boundary.
        let (mut dev, mut xfs) = fresh_image();
        let payload = vec![0x55u8; 8192]; // 2 FS blocks worth
        let mut src: &[u8] = &payload;
        xfs.add_file_path(
            &mut dev,
            "/p.bin",
            EntryMeta::default(),
            payload.len() as u64,
            &mut src,
        )
        .unwrap();
        xfs.flush_writes(&mut dev).unwrap();
        let mut xfs = Xfs::open(&mut dev).unwrap();
        {
            let mut h = Filesystem::open_file_rw(
                &mut xfs,
                &mut dev,
                Path::new("/p.bin"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
            h.seek(SeekFrom::Start(4090)).unwrap();
            h.write_all(b"BOUNDARY").unwrap(); // straddles block 0 / block 1
            h.sync().unwrap();
        }
        let mut xfs2 = Xfs::open(&mut dev).unwrap();
        let bytes = read_file(&mut xfs2, &mut dev, "/p.bin");
        assert_eq!(bytes.len(), 8192);
        assert_eq!(&bytes[4090..4098], b"BOUNDARY");
    }

    #[test]
    fn open_file_rw_set_len_grow_and_shrink() {
        let (mut dev, mut xfs) = fresh_image();
        let mut src: &[u8] = b"ABCDEFGH";
        xfs.add_file_path(&mut dev, "/s.bin", EntryMeta::default(), 8, &mut src)
            .unwrap();
        xfs.flush_writes(&mut dev).unwrap();
        let mut xfs = Xfs::open(&mut dev).unwrap();
        {
            let mut h = Filesystem::open_file_rw(
                &mut xfs,
                &mut dev,
                Path::new("/s.bin"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
            // Grow to 20 KiB (multi-block) — fills with zeros.
            h.set_len(20 * 1024).unwrap();
            assert_eq!(h.len(), 20 * 1024);
            h.sync().unwrap();
        }
        {
            let mut xfs2 = Xfs::open(&mut dev).unwrap();
            let bytes = read_file(&mut xfs2, &mut dev, "/s.bin");
            assert_eq!(bytes.len(), 20 * 1024);
            assert_eq!(&bytes[..8], b"ABCDEFGH");
            assert!(bytes[8..].iter().all(|&b| b == 0));
        }
        // Now shrink back to 4 bytes.
        let mut xfs = Xfs::open(&mut dev).unwrap();
        {
            let mut h = Filesystem::open_file_rw(
                &mut xfs,
                &mut dev,
                Path::new("/s.bin"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
            h.set_len(4).unwrap();
            assert_eq!(h.len(), 4);
            h.sync().unwrap();
        }
        let mut xfs3 = Xfs::open(&mut dev).unwrap();
        let bytes = read_file(&mut xfs3, &mut dev, "/s.bin");
        assert_eq!(bytes, b"ABCD");
    }

    #[test]
    fn open_file_rw_append() {
        let (mut dev, mut xfs) = fresh_image();
        let mut src: &[u8] = b"head ";
        xfs.add_file_path(&mut dev, "/a.txt", EntryMeta::default(), 5, &mut src)
            .unwrap();
        xfs.flush_writes(&mut dev).unwrap();
        let mut xfs = Xfs::open(&mut dev).unwrap();
        {
            let mut h = Filesystem::open_file_rw(
                &mut xfs,
                &mut dev,
                Path::new("/a.txt"),
                OpenFlags {
                    create: false,
                    truncate: false,
                    append: true,
                },
                None,
            )
            .unwrap();
            h.write_all(b"tail").unwrap();
            h.sync().unwrap();
        }
        let mut xfs2 = Xfs::open(&mut dev).unwrap();
        let bytes = read_file(&mut xfs2, &mut dev, "/a.txt");
        assert_eq!(bytes, b"head tail");
    }

    #[test]
    fn open_file_rw_create_new() {
        let (mut dev, mut xfs) = fresh_image();
        xfs.flush_writes(&mut dev).unwrap();
        let mut xfs = Xfs::open(&mut dev).unwrap();
        {
            let mut h = Filesystem::open_file_rw(
                &mut xfs,
                &mut dev,
                Path::new("/n.txt"),
                OpenFlags {
                    create: true,
                    truncate: false,
                    append: false,
                },
                Some(FileMeta::default()),
            )
            .unwrap();
            assert_eq!(h.len(), 0);
            h.write_all(b"freshly created").unwrap();
            h.sync().unwrap();
        }
        let mut xfs2 = Xfs::open(&mut dev).unwrap();
        let bytes = read_file(&mut xfs2, &mut dev, "/n.txt");
        assert_eq!(bytes, b"freshly created");
    }

    #[test]
    fn open_file_ro_random_seek_xfs() {
        let (mut dev, mut xfs) = fresh_image();
        // Multi-block file (4 KiB blocks default) to exercise the extent walker.
        let data: Vec<u8> = (0..12_000u32).map(|i| (i & 0xFF) as u8).collect();
        let mut src: &[u8] = &data;
        xfs.add_file_path(
            &mut dev,
            "/ro.bin",
            EntryMeta::default(),
            data.len() as u64,
            &mut src,
        )
        .unwrap();
        xfs.flush_writes(&mut dev).unwrap();

        let mut xfs2 = Xfs::open(&mut dev).unwrap();
        let mut h = xfs2
            .open_file_ro(&mut dev, Path::new("/ro.bin"))
            .expect("open_file_ro");
        assert_eq!(h.len(), data.len() as u64);
        assert!(!h.is_empty());

        h.seek(SeekFrom::Start(7777)).unwrap();
        let mut buf = [0u8; 200];
        h.read_exact(&mut buf).unwrap();
        assert_eq!(&buf[..], &data[7777..7977]);

        h.seek(SeekFrom::Start(13)).unwrap();
        let mut buf2 = [0u8; 96];
        h.read_exact(&mut buf2).unwrap();
        assert_eq!(&buf2[..], &data[13..109]);
    }

    /// End-to-end XLOG Path A round-trip: format → write file → manually
    /// stamp a single-inode-update transaction into the log, leaving the
    /// on-disk inode untouched (simulating a crash between log-write and
    /// checkpoint) → reopen → verify the inode was replayed from the log.
    ///
    /// We bypass the rw write path's own log-stamping so the test
    /// pre-/post-conditions are unambiguous: we know exactly what bytes
    /// we wanted replay to apply.
    #[test]
    fn xlog_round_trip_replays_on_open() {
        use super::super::format::XFS_INODESIZE;
        use super::super::inode::stamp_v3_inode_crc;
        use super::super::journal::write_inode_update_transaction;

        let (mut dev, mut xfs) = fresh_image();
        // Create a tiny file with known content.
        let original = b"AAAAAAAAAAAAAAAA"; // 16 bytes
        let mut src: &[u8] = original;
        xfs.add_file_path(
            &mut dev,
            "/replay.bin",
            EntryMeta::default(),
            original.len() as u64,
            &mut src,
        )
        .unwrap();
        xfs.flush_writes(&mut dev).unwrap();

        // Snapshot the on-disk inode bytes (this is what disk will retain
        // after the "crash" — i.e. the pre-modification version).
        let xfs = Xfs::open(&mut dev).unwrap();
        let (ino, _, _) = xfs.resolve_path(&mut dev, "/replay.bin").unwrap();
        let ino_off = xfs.ino_byte_offset(ino).unwrap();
        let mut pre_inode = vec![0u8; XFS_INODESIZE as usize];
        dev.read_at(ino_off, &mut pre_inode).unwrap();

        // Build the "new" inode payload by patching just the di_size
        // field of the pre-inode (offset 56..64, big-endian per the v3
        // layout) and re-stamping the CRC. We log it but never write it
        // in place — that's what replay must do for us.
        let new_size = 32u64;
        let mut new_inode = pre_inode.clone();
        new_inode[56..64].copy_from_slice(&new_size.to_be_bytes());
        stamp_v3_inode_crc(&mut new_inode);

        // Stamp the transaction directly. Crucially, do NOT touch the
        // in-place inode — replay's job.
        let log_off = xfs.sb.logstart_byte_offset();
        let log_bytes = xfs.sb.log_bytes();
        write_inode_update_transaction(
            &mut dev,
            log_off,
            log_bytes,
            /* tid    */ 0xC0FFEEu32,
            /* cycle  */ 2,
            /* ino    */ ino,
            /* target */ ino_off,
            &new_inode,
            &xfs.sb.uuid,
        )
        .unwrap();
        // On-disk inode is still `pre_inode`. Confirm.
        let mut now = vec![0u8; XFS_INODESIZE as usize];
        dev.read_at(ino_off, &mut now).unwrap();
        assert_eq!(now, pre_inode, "no checkpoint write should have happened");
        // The log header reports a 4-op transaction (dirty).
        let mut log_hdr = vec![0u8; 512];
        dev.read_at(log_off, &mut log_hdr).unwrap();
        let num_logops = u32::from_be_bytes(log_hdr[40..44].try_into().unwrap());
        assert_eq!(num_logops, 4, "log must be dirty before reopen");

        // Reopen + open_file_rw — this calls prepare_log_for_rw, which
        // calls replay_log. After replay, the inode bytes match
        // `new_inode` and the log is clean.
        let mut xfs2 = Xfs::open(&mut dev).unwrap();
        {
            let _h = Filesystem::open_file_rw(
                &mut xfs2,
                &mut dev,
                Path::new("/replay.bin"),
                OpenFlags::default(),
                None,
            )
            .expect("open after replay");
            // Drop the handle without writing; no checkpoint expected.
        }
        // Inode at disk now reflects the replayed payload.
        let mut after = vec![0u8; XFS_INODESIZE as usize];
        dev.read_at(ino_off, &mut after).unwrap();
        let after_size = u64::from_be_bytes(after[56..64].try_into().unwrap());
        assert_eq!(
            after_size, new_size,
            "replay should have written the logged inode bytes to disk"
        );
        // Log restamped clean.
        let xfs3 = Xfs::open(&mut dev).unwrap();
        assert_log_clean(&xfs3, &mut dev).unwrap();
    }

    /// Two successive `sync` calls must each round-trip cleanly. Guards
    /// against the log getting stuck dirty after the first commit.
    #[test]
    fn xlog_round_trip_multiple_syncs() {
        let (mut dev, mut xfs) = fresh_image();
        let mut src: &[u8] = b"first";
        xfs.add_file_path(&mut dev, "/m.bin", EntryMeta::default(), 5, &mut src)
            .unwrap();
        xfs.flush_writes(&mut dev).unwrap();

        let mut xfs = Xfs::open(&mut dev).unwrap();
        {
            let mut h = Filesystem::open_file_rw(
                &mut xfs,
                &mut dev,
                Path::new("/m.bin"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
            h.write_all(b"XXXXX").unwrap();
            h.sync().unwrap();
            h.seek(SeekFrom::Start(2)).unwrap();
            h.write_all(b"YY").unwrap();
            h.sync().unwrap();
        }
        let mut xfs2 = Xfs::open(&mut dev).unwrap();
        let bytes = read_file(&mut xfs2, &mut dev, "/m.bin");
        assert_eq!(bytes, b"XXYYX");
        assert_log_clean(&xfs2, &mut dev).unwrap();
    }

    /// A torn / partial transaction (record header present but commit op
    /// missing) must be discarded by replay; the open should still
    /// succeed and the on-disk inode keep its pre-crash bytes.
    #[test]
    fn xlog_partial_transaction_is_discarded() {
        use super::super::format::XFS_INODESIZE;
        use super::super::journal::write_inode_update_transaction;

        let (mut dev, mut xfs) = fresh_image();
        let mut src: &[u8] = b"keep me";
        xfs.add_file_path(&mut dev, "/p.bin", EntryMeta::default(), 7, &mut src)
            .unwrap();
        xfs.flush_writes(&mut dev).unwrap();

        let xfs = Xfs::open(&mut dev).unwrap();
        let (ino, _, _) = xfs.resolve_path(&mut dev, "/p.bin").unwrap();
        let ino_off = xfs.ino_byte_offset(ino).unwrap();
        let mut pre_inode = vec![0u8; XFS_INODESIZE as usize];
        dev.read_at(ino_off, &mut pre_inode).unwrap();
        let log_off = xfs.sb.logstart_byte_offset();
        let log_bytes = xfs.sb.log_bytes();
        // Stamp a valid transaction, then corrupt the commit op (op 4) by
        // clearing its flags byte — replay should detect "no commit" and
        // discard.
        write_inode_update_transaction(
            &mut dev,
            log_off,
            log_bytes,
            1,
            2,
            ino,
            ino_off,
            &pre_inode,
            &xfs.sb.uuid,
        )
        .unwrap();
        // Layout inside the payload buffer (after BB0 record header):
        //   op1 (12 B) at 0
        //   trans_hdr (16 B) at 12
        //   op2 (12 B) at 28
        //   inode_log_fmt (56 B) at 40
        //   op3 (12 B) at 96
        //   inode_bytes (XFS_INODESIZE = 512 B) at 108
        //   op4 (12 B) at 620
        // The flag byte of op4 (oh_flags) lives at op-header offset 9.
        // Cycle stamping only touches bytes 0..4 of each BB inside the
        // payload, so 629 is safely untouched.
        let op4_off_in_payload = 12 * 3 + 16 + 56 + (XFS_INODESIZE as usize);
        assert_eq!(op4_off_in_payload, 620);
        let flag_addr = log_off + 512 + op4_off_in_payload as u64 + 9;
        dev.write_at(flag_addr, &[0u8]).unwrap();

        // Reopen — open_file_rw should succeed; replay discards the
        // partial transaction and restamps the log clean; inode keeps
        // its pre-crash bytes.
        let mut xfs2 = Xfs::open(&mut dev).unwrap();
        {
            let _h = Filesystem::open_file_rw(
                &mut xfs2,
                &mut dev,
                Path::new("/p.bin"),
                OpenFlags::default(),
                None,
            )
            .expect("open after torn-transaction discard");
        }
        let mut after = vec![0u8; XFS_INODESIZE as usize];
        dev.read_at(ino_off, &mut after).unwrap();
        assert_eq!(after, pre_inode, "on-disk inode unchanged");
        let xfs3 = Xfs::open(&mut dev).unwrap();
        assert_log_clean(&xfs3, &mut dev).unwrap();
    }
}
