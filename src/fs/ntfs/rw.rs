//! `Filesystem::open_file_rw` for NTFS — eager-write file handle.
//!
//! The handle owns mutable borrows of both [`super::Ntfs`] and the
//! [`BlockDevice`] for its lifetime. Bytes are written through to disk on
//! each `Write::write`. Mutating the file's logical length (grow, shrink,
//! create-with-truncate) is buffered until `sync()` (or drop), at which
//! point we re-emit the file's MFT record with an updated `$DATA`
//! attribute, an updated `$FILE_NAME` size pair, and free / allocate
//! clusters in the volume bitmap.
//!
//! ## Path A: real `$LogFile` records
//!
//! Each `sync()` runs a write-ahead-log commit through [`super::logfile`]:
//!
//! 1. Pack `(target_offset, redo_bytes, undo_bytes)` for every metadata
//!    write (the MFT record itself; optionally the parent dir's `$I30`
//!    size patch) into an `RCRD` page and stamp it into `$LogFile`.
//! 2. Mark the restart pages dirty (`CleanDismount` cleared) with the
//!    new `CurrentLsn`.
//! 3. Apply the in-place writes.
//! 4. Mark the restart pages clean.
//!
//! A crash between steps 2 and 4 leaves a recoverable image: on the
//! next open, `ensure_clean_log` walks the `RCRD` pages and re-applies
//! their `redo_bytes`, then re-stamps `CleanDismount`. Foreign LFS
//! consumers (kernel NTFS3, ntfs-3g, `ntfsfix`) see the restart pages
//! as a structurally valid v1.1 log with `CleanDismount` set, and
//! don't try to decode our private record payload (see
//! [`super::logfile`] for the payload format).
//!
//! Bulk `$DATA` byte writes are not journaled — they go straight to
//! disk inside `write_internal`. This matches the NTFS convention:
//! only metadata is logged, not user-data ranges.
//!
//! Volume-level state (`$Bitmap`, `$MFT`, `$MFTMirr`, boot sector) is
//! persisted by [`super::Ntfs::flush`] as today.

use std::io::{Read, Seek, SeekFrom, Write};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::FileHandle;

use super::attribute::{
    AttributeIter, AttributeKind, FileName, TYPE_DATA, TYPE_FILE_NAME, decode_utf16le,
};
use super::format::{
    self, build_file_name_value, build_non_resident_attr, build_resident_attr, encode_run_list,
};
use super::mft;
use super::run_list::Extent;

/// Open file handle. Lives as long as the borrowed `Ntfs` + device.
pub struct NtfsFileHandle<'a> {
    fs: &'a mut super::Ntfs,
    dev: &'a mut dyn BlockDevice,
    /// MFT record number of the file.
    rec_no: u64,
    /// MFT record byte size (cached).
    rec_size: usize,
    /// Logical sector size (used for USA fixup install).
    sector_size: usize,
    /// Cluster size in bytes.
    cluster_size: u64,
    /// Cursor position.
    pos: u64,
    /// Current logical file length (in bytes).
    len: u64,
    /// $DATA stream contents, when resident. `None` for non-resident.
    resident: Option<Vec<u8>>,
    /// $DATA extents, when non-resident. Always allocated runs — sparse
    /// LCNs are never produced by this writer.
    runs: Vec<Extent>,
    /// Decoded `$FILE_NAME` value bytes (so we can rebuild the record
    /// with refreshed real_size / allocated_size on sync).
    file_name_value: Vec<u8>,
    /// `parent_ref` from the decoded $FILE_NAME.
    parent_ref: u64,
    /// Cached `$STANDARD_INFORMATION` value bytes (preserved verbatim on
    /// rewrite — we don't touch timestamps).
    si_value: Vec<u8>,
    /// True if any state changed (length, runs, bytes). Drives the
    /// "rewrite MFT record" path on sync().
    dirty: bool,
}

impl<'a> NtfsFileHandle<'a> {
    /// Resolve VCN `vcn` to its physical byte offset, or `None` if the
    /// VCN falls in a sparse / unallocated region. (We never produce
    /// sparse runs ourselves, so this is conservative.)
    fn vcn_to_disk(&self, vcn: u64) -> Option<u64> {
        let cs = self.cluster_size;
        let mut walked: u64 = 0;
        for ext in &self.runs {
            if vcn < walked + ext.length {
                let local = vcn - walked;
                return ext.lcn.map(|lcn| (lcn + local) * cs);
            }
            walked += ext.length;
        }
        None
    }

    /// Allocated capacity in bytes (sum of all runs * cluster_size).
    fn allocated_bytes(&self) -> u64 {
        self.runs.iter().map(|r| r.length).sum::<u64>() * self.cluster_size
    }

    /// Read `out.len()` bytes at the current position. EOF returns 0.
    fn read_internal(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.len {
            return Ok(0);
        }
        let avail = self.len - self.pos;
        let want = (out.len() as u64).min(avail) as usize;
        if want == 0 {
            return Ok(0);
        }

        if let Some(ref bytes) = self.resident {
            let s = self.pos as usize;
            let e = s + want;
            out[..want].copy_from_slice(&bytes[s..e]);
            self.pos += want as u64;
            return Ok(want);
        }

        let cs = self.cluster_size;
        let vcn = self.pos / cs;
        let off_in_cluster = (self.pos % cs) as usize;
        let in_cluster = (cs as usize - off_in_cluster).min(want);
        let disk = match self.vcn_to_disk(vcn) {
            Some(p) => p,
            None => {
                // Conservative: treat as zero. (Should not happen for our
                // writer.)
                out[..in_cluster].fill(0);
                self.pos += in_cluster as u64;
                return Ok(in_cluster);
            }
        };
        let phys = disk + off_in_cluster as u64;
        self.dev
            .read_at(phys, &mut out[..in_cluster])
            .map_err(std::io::Error::other)?;
        self.pos += in_cluster as u64;
        Ok(in_cluster)
    }

    /// Allocate `extra_clusters` and append them to `self.runs`. Tries
    /// to extend the last run when possible, otherwise pushes a fresh
    /// run; chains via the format-time bitmap allocator (contiguous
    /// chunk per call).
    fn grow_runs(&mut self, extra_clusters: u64) -> Result<()> {
        if extra_clusters == 0 {
            return Ok(());
        }
        let w = self.fs.writer.as_mut().ok_or_else(|| {
            crate::Error::Unsupported(
                "ntfs: open_file_rw requires the writer (format the volume first)".into(),
            )
        })?;
        let start = w.alloc_clusters(extra_clusters)?;
        // Zero the newly-allocated region on disk so reading past
        // initialized_size sees clean data.
        let zero = vec![0u8; (extra_clusters * self.cluster_size) as usize];
        self.dev.write_at(start * self.cluster_size, &zero)?;

        // Try to merge with the last existing run.
        if let Some(last) = self.runs.last_mut() {
            if let Some(last_lcn) = last.lcn {
                if last_lcn + last.length == start {
                    last.length += extra_clusters;
                    self.dirty = true;
                    return Ok(());
                }
            }
        }
        self.runs.push(Extent {
            lcn: Some(start),
            length: extra_clusters,
        });
        self.dirty = true;
        Ok(())
    }

    /// Shrink `self.runs` so the total cluster count is `keep_clusters`,
    /// freeing the trailing clusters in the volume bitmap.
    fn shrink_runs(&mut self, keep_clusters: u64) -> Result<()> {
        let total: u64 = self.runs.iter().map(|r| r.length).sum();
        if keep_clusters >= total {
            return Ok(());
        }
        let w = self.fs.writer.as_mut().ok_or_else(|| {
            crate::Error::Unsupported(
                "ntfs: open_file_rw requires the writer (format the volume first)".into(),
            )
        })?;
        // Walk from the tail, freeing clusters until we hit keep_clusters.
        let mut remaining = total - keep_clusters;
        while remaining > 0 {
            let last = self.runs.last_mut().expect("non-empty runs");
            let take = remaining.min(last.length);
            if let Some(lcn) = last.lcn {
                // Clear bits [lcn + (last.length - take) ..  lcn + last.length).
                let free_start = lcn + (last.length - take);
                for c in free_start..lcn + last.length {
                    w.layout.bitmap.clear(c);
                    if w.layout.bitmap.next_hint > c {
                        w.layout.bitmap.next_hint = c;
                    }
                }
            }
            last.length -= take;
            remaining -= take;
            if last.length == 0 {
                self.runs.pop();
            }
        }
        w.dirty = true;
        self.dirty = true;
        Ok(())
    }

    /// Materialise a resident $DATA stream into a non-resident one.
    /// Allocates enough clusters to hold the current resident bytes, writes
    /// them to disk, and switches `self.resident` → `self.runs`.
    fn promote_to_non_resident(&mut self) -> Result<()> {
        let bytes = match self.resident.take() {
            Some(b) => b,
            None => return Ok(()),
        };
        // Allocate ceil(len / cluster_size) clusters and write the bytes.
        let cs = self.cluster_size;
        let need = bytes.len() as u64;
        let clusters = need.div_ceil(cs);
        if clusters == 0 {
            // Empty resident -> empty non-resident. Nothing to do.
            return Ok(());
        }
        let w = self
            .fs
            .writer
            .as_mut()
            .ok_or_else(|| crate::Error::Unsupported("ntfs: writer not initialised".into()))?;
        let lcn = w.alloc_clusters(clusters)?;
        // Pad bytes to a full cluster boundary.
        let mut padded = bytes;
        let pad_len = (clusters * cs) as usize - padded.len();
        padded.extend(std::iter::repeat_n(0u8, pad_len));
        self.dev.write_at(lcn * cs, &padded)?;
        self.runs.push(Extent {
            lcn: Some(lcn),
            length: clusters,
        });
        self.dirty = true;
        Ok(())
    }

    /// Ensure the file has at least `new_len` bytes of allocated capacity
    /// (and, for resident-only files, materialise non-resident storage
    /// when growth crosses the resident budget).
    fn ensure_capacity(&mut self, new_len: u64) -> Result<()> {
        // Decide the resident-vs-non-resident regime. We use a conservative
        // resident budget tied to the MFT record size (matches writer.rs).
        let resident_budget = self.rec_size.saturating_sub(232) as u64;

        if let Some(b) = self.resident.as_mut() {
            if new_len <= resident_budget {
                // Stay resident: just grow the buffer (zero-filled tail).
                if (b.len() as u64) < new_len {
                    b.resize(new_len as usize, 0);
                    self.dirty = true;
                }
                return Ok(());
            }
            // Promote to non-resident before growing.
            self.promote_to_non_resident()?;
        }
        // Non-resident path: extend run list to cover new_len bytes.
        let cs = self.cluster_size;
        let have = self.allocated_bytes();
        if new_len <= have {
            return Ok(());
        }
        let need_total_clusters = new_len.div_ceil(cs);
        let cur_clusters: u64 = self.runs.iter().map(|r| r.length).sum();
        let extra = need_total_clusters - cur_clusters;
        self.grow_runs(extra)
    }

    /// Write `buf` at the current cursor position, growing the file as
    /// needed. Eager — bytes hit disk inside this call (for non-resident);
    /// resident bytes hit disk on `sync()`.
    fn write_internal(&mut self, buf: &[u8]) -> Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        // Phase 1: if pos > len, the gap [len, pos) becomes a sparse
        // hole we fill with zero up to pos.
        if self.pos > self.len {
            self.ensure_capacity(self.pos)?;
            self.zero_range(self.len, self.pos)?;
            self.len = self.pos;
            self.dirty = true;
        }

        // Phase 2: ensure capacity covering [pos, pos + buf.len()).
        let new_end = self.pos + buf.len() as u64;
        self.ensure_capacity(new_end)?;

        // Phase 3: write the buffer. Resident path: poke the in-memory
        // buffer (and leave on-disk persistence to sync). Non-resident:
        // write through to disk now.
        if let Some(ref mut bytes) = self.resident {
            let s = self.pos as usize;
            let e = s + buf.len();
            if bytes.len() < e {
                bytes.resize(e, 0);
            }
            bytes[s..e].copy_from_slice(buf);
        } else {
            let cs = self.cluster_size;
            let mut written = 0usize;
            while written < buf.len() {
                let p = self.pos + written as u64;
                let vcn = p / cs;
                let off = (p % cs) as usize;
                let in_cluster = ((cs as usize) - off).min(buf.len() - written);
                let disk = self.vcn_to_disk(vcn).ok_or_else(|| {
                    crate::Error::InvalidImage(format!("ntfs: write past run list at VCN {vcn}"))
                })?;
                self.dev
                    .write_at(disk + off as u64, &buf[written..written + in_cluster])?;
                written += in_cluster;
            }
        }

        self.pos += buf.len() as u64;
        if self.pos > self.len {
            self.len = self.pos;
        }
        self.dirty = true;
        Ok(buf.len())
    }

    /// Zero the byte range `[from, to)` on disk (or in the resident
    /// buffer). Caller is responsible for having ensured capacity.
    fn zero_range(&mut self, from: u64, to: u64) -> Result<()> {
        if from >= to {
            return Ok(());
        }
        if let Some(ref mut bytes) = self.resident {
            let s = from as usize;
            let e = to as usize;
            if bytes.len() < e {
                bytes.resize(e, 0);
            }
            for b in &mut bytes[s..e] {
                *b = 0;
            }
            return Ok(());
        }
        let cs = self.cluster_size;
        let mut p = from;
        let zero = vec![0u8; cs as usize];
        while p < to {
            let vcn = p / cs;
            let off = (p % cs) as usize;
            let in_cluster = ((cs - off as u64).min(to - p)) as usize;
            let disk = self.vcn_to_disk(vcn).ok_or_else(|| {
                crate::Error::InvalidImage(format!("ntfs: zero_range past run list at VCN {vcn}"))
            })?;
            self.dev.write_at(disk + off as u64, &zero[..in_cluster])?;
            p += in_cluster as u64;
        }
        Ok(())
    }

    /// Rewrite the MFT record with up-to-date `$STANDARD_INFORMATION`,
    /// `$FILE_NAME`, and `$DATA` attributes. Called from `sync()` /
    /// `Drop`.
    ///
    /// Goes through the LFS journal: every in-place write of metadata
    /// (the MFT record itself plus the parent directory's `$I30` size
    /// pair) is captured as a redo/undo pair, packed into an RCRD page,
    /// and stamped into `$LogFile` before being applied. Restart pages
    /// are toggled `dirty → clean` around the commit so a crash in the
    /// middle leaves enough information to redo on the next open.
    fn flush_record(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }

        // Refresh the $FILE_NAME real_size + allocated_size. We rebuild
        // the value from scratch via build_file_name_value() so the
        // result matches what create_file() would have emitted.
        let (filetime, flags, namespace, name) = decode_file_name_meta(&self.file_name_value)?;
        let allocated = if self.resident.is_some() {
            // Resident attributes have no cluster allocation; report
            // real_size (rounded up to the cluster) as ntfs-3g expects.
            (self.len + self.cluster_size - 1) & !(self.cluster_size - 1)
        } else {
            self.runs.iter().map(|r| r.length).sum::<u64>() * self.cluster_size
        };
        let fn_value = build_file_name_value(
            self.parent_ref,
            &name,
            flags,
            self.len,
            allocated,
            filetime,
            namespace,
        );
        self.file_name_value = fn_value.clone();

        // Build the $DATA attribute.
        let data_attr = if let Some(ref bytes) = self.resident {
            // Resident: shrink to actual size (we may have padded earlier).
            let mut payload = bytes.clone();
            payload.truncate(self.len as usize);
            build_resident_attr(TYPE_DATA, &[], &payload, 0, 0)
        } else {
            // Non-resident: encode runs.
            let extents: Vec<(u64, u64)> = self
                .runs
                .iter()
                .filter_map(|r| r.lcn.map(|l| (l, r.length)))
                .collect();
            let total_clusters: u64 = self.runs.iter().map(|r| r.length).sum();
            let alloc_bytes = total_clusters * self.cluster_size;
            let (runs_bytes, last_vcn) = if extents.is_empty() {
                // Empty non-resident — shouldn't really happen; emit
                // empty run list + 0 last_vcn.
                (vec![0u8], 0u64)
            } else {
                (encode_run_list(&extents), total_clusters - 1)
            };
            build_non_resident_attr(
                TYPE_DATA,
                &[],
                &runs_bytes,
                0,
                last_vcn,
                alloc_bytes,
                self.len,
                self.len,
                0,
                0,
            )
        };

        // Re-emit the MFT record with [SI, FN, DATA].
        let si = build_resident_attr(
            super::attribute::TYPE_STANDARD_INFORMATION,
            &[],
            &self.si_value,
            0,
            0,
        );
        let fn_attr = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 1);

        let mut rec_buf = vec![0u8; self.rec_size];
        // Preserve flags from the existing on-disk record (just FLAG_IN_USE
        // for a regular file).
        let existing_flags = self.read_existing_flags()?;
        format::emit_record(
            &mut rec_buf,
            self.rec_size,
            self.rec_no,
            existing_flags,
            &[si, fn_attr, data_attr],
            self.sector_size,
            1,
        )?;

        // Pre-read the old MFT record bytes so we can journal the
        // (redo, undo) pair.
        let mft_off = {
            let w =
                self.fs.writer.as_ref().ok_or_else(|| {
                    crate::Error::Unsupported("ntfs: writer not initialised".into())
                })?;
            w.mft_offset(self.rec_no)?
        };
        let mut old_rec = vec![0u8; self.rec_size];
        self.dev.read_at(mft_off, &mut old_rec)?;

        // Gather pending journal entries. The MFT record is the
        // load-bearing one; the parent-dir index patch is best-effort
        // and we include it only when it actually changes bytes.
        let mut txn: Vec<super::logfile::RedoEntry> = Vec::new();
        txn.push(super::logfile::RedoEntry {
            target_offset: mft_off,
            redo_bytes: rec_buf.clone(),
            undo_bytes: old_rec,
        });
        if let Some(entry) = self.build_index_entry_redo()? {
            txn.push(entry);
        }

        // Commit through the journal: write LFS records, stamp restart
        // pages dirty, apply the in-place writes, stamp restart pages
        // clean.
        self.commit_txn(&txn)?;

        self.dirty = false;
        Ok(())
    }

    /// Build a `RedoEntry` for the parent directory's `$I30` size patch.
    /// Returns `Ok(None)` if no patch applies (no matching entry, or the
    /// bytes are already correct).
    fn build_index_entry_redo(&mut self) -> Result<Option<super::logfile::RedoEntry>> {
        let parent_rec_no = self.parent_ref & 0x0000_FFFF_FFFF_FFFF;
        let off = self
            .fs
            .writer
            .as_ref()
            .ok_or_else(|| crate::Error::Unsupported("ntfs: writer not initialised".into()))?
            .mft_offset(parent_rec_no)?;
        let mut rec = vec![0u8; self.rec_size];
        self.dev.read_at(off, &mut rec)?;
        let old_rec = rec.clone();
        if mft::apply_fixup(&mut rec, self.sector_size).is_err() {
            return Ok(None);
        }
        let hdr = mft::RecordHeader::parse(&rec)?;
        let bytes_in_use = hdr.bytes_in_use as usize;
        let first = hdr.first_attribute_offset as usize;
        let mut cursor = first;
        let mut root_off_in_rec: Option<(usize, usize, usize)> = None;
        let mut alloc_runs: Option<Vec<Extent>> = None;
        while cursor + 4 <= bytes_in_use {
            let tc = u32::from_le_bytes(rec[cursor..cursor + 4].try_into().unwrap());
            if tc == 0xFFFF_FFFF {
                break;
            }
            let len = u32::from_le_bytes(rec[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
            let non_resident = rec[cursor + 8] != 0;
            let name_len = rec[cursor + 9] as usize;
            let name_off =
                u16::from_le_bytes(rec[cursor + 10..cursor + 12].try_into().unwrap()) as usize;
            let attr_name = if name_len == 0 {
                String::new()
            } else {
                decode_utf16le(&rec[cursor + name_off..cursor + name_off + name_len * 2])
            };
            if attr_name == "$I30" {
                if tc == super::attribute::TYPE_INDEX_ROOT && !non_resident {
                    let value_off =
                        u16::from_le_bytes(rec[cursor + 0x14..cursor + 0x16].try_into().unwrap())
                            as usize;
                    let value_len =
                        u32::from_le_bytes(rec[cursor + 0x10..cursor + 0x14].try_into().unwrap())
                            as usize;
                    root_off_in_rec = Some((cursor, value_off, value_len));
                } else if tc == super::attribute::TYPE_INDEX_ALLOCATION && non_resident {
                    let runs_off =
                        u16::from_le_bytes(rec[cursor + 0x20..cursor + 0x22].try_into().unwrap())
                            as usize;
                    let runs_bytes = &rec[cursor + runs_off..cursor + len];
                    if let Ok(rs) = super::run_list::decode(runs_bytes) {
                        alloc_runs = Some(rs);
                    }
                }
            }
            cursor += len;
        }
        let Some((attr_start, value_off, value_len)) = root_off_in_rec else {
            return Ok(None);
        };
        let root_v_start = attr_start + value_off;
        let root_v_end = root_v_start + value_len;
        let root_val = &mut rec[root_v_start..root_v_end];
        if root_val.len() < 32 {
            return Ok(None);
        }
        let index_flags = root_val[28];
        let large_index = index_flags & 0x01 != 0;
        if !large_index {
            if patch_entries_for_record(root_val, 16, self.rec_no, self.len) {
                mft::install_fixup(&mut rec, self.sector_size, 1);
                if rec != old_rec {
                    return Ok(Some(super::logfile::RedoEntry {
                        target_offset: off,
                        redo_bytes: rec,
                        undo_bytes: old_rec,
                    }));
                }
            }
            return Ok(None);
        }
        let Some(runs) = alloc_runs else {
            return Ok(None);
        };
        let Some(first_run_lcn) = runs.first().and_then(|r| r.lcn) else {
            return Ok(None);
        };
        let block_size = self
            .fs
            .writer
            .as_ref()
            .map(|w| w.layout.index_record_size as usize)
            .unwrap_or(4096);
        let block_off = first_run_lcn * self.cluster_size;
        let mut block = vec![0u8; block_size];
        self.dev.read_at(block_off, &mut block)?;
        let old_block = block.clone();
        if mft::apply_fixup(&mut block, self.sector_size).is_err() {
            return Ok(None);
        }
        if block.len() < 0x20 {
            return Ok(None);
        }
        let first_entry_offset = u32::from_le_bytes(block[0x18..0x1C].try_into().unwrap()) as usize;
        let entries_start = 0x18 + first_entry_offset;
        if patch_entries_for_record(&mut block, entries_start, self.rec_no, self.len) {
            mft::install_fixup(&mut block, self.sector_size, 1);
            if block != old_block {
                return Ok(Some(super::logfile::RedoEntry {
                    target_offset: block_off,
                    redo_bytes: block,
                    undo_bytes: old_block,
                }));
            }
        }
        Ok(None)
    }

    /// Run a commit-time protocol:
    ///
    /// 1. Locate `$LogFile` and the current restart-area `CurrentLsn`.
    /// 2. Pack `entries` into one or more RCRD pages starting at the
    ///    record-page region (`logfile_offset + 2·LOG_PAGE_SIZE`).
    /// 3. Stamp both restart pages with the new `CurrentLsn` and
    ///    `CleanDismount` cleared (== "transaction in progress").
    /// 4. Apply the in-place writes.
    /// 5. Stamp both restart pages again with `CleanDismount` set.
    ///
    /// If the process dies between steps 3 and 5, the next open's
    /// `ensure_clean_log` sees the dirty marker, walks the RCRD pages,
    /// re-applies the redo bytes, and re-stamps clean.
    fn commit_txn(&mut self, entries: &[super::logfile::RedoEntry]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let (logfile_offset, log_size) = locate_logfile(self.fs, self.dev)?;
        if log_size < 4 * super::logfile::LOG_PAGE_SIZE as u64 {
            // Tiny log — fall back to direct writes (no journal).
            for e in entries {
                self.dev.write_at(e.target_offset, &e.redo_bytes)?;
            }
            return Ok(());
        }

        // Pick a starting LSN past the existing current_lsn.
        let prev_view = super::logfile::read_current_restart(self.dev, logfile_offset)?;
        let prev_lsn = prev_view.map(|(v, _)| v.current_lsn).unwrap_or(0);
        let mut next_lsn = prev_lsn + 1;

        // Pack into RCRD pages — one page per chunk, splitting on
        // overflow. The record-page region starts at offset
        // 2·LOG_PAGE_SIZE; pages cycle modulo (log_size - 2·LOG_PAGE_SIZE).
        let rcrd_region_start = logfile_offset + 2 * super::logfile::LOG_PAGE_SIZE as u64;
        let rcrd_region_size = log_size - 2 * super::logfile::LOG_PAGE_SIZE as u64;
        let mut page_idx: u64 = 0;
        let mut chunk_start = 0usize;
        let mut last_lsn = next_lsn;
        while chunk_start < entries.len() {
            // Find the largest prefix that fits in one page.
            let mut chunk_end = chunk_start + 1;
            while chunk_end <= entries.len() {
                match super::logfile::build_record_page(&entries[chunk_start..chunk_end], next_lsn)
                {
                    Ok(_) => chunk_end += 1,
                    Err(_) => break,
                }
            }
            if chunk_end == chunk_start + 1 {
                // Even one record doesn't fit — bail to direct writes.
                for e in entries {
                    self.dev.write_at(e.target_offset, &e.redo_bytes)?;
                }
                return Ok(());
            }
            chunk_end -= 1;
            let (page, ll) =
                super::logfile::build_record_page(&entries[chunk_start..chunk_end], next_lsn)?;
            let off = rcrd_region_start
                + (page_idx * super::logfile::LOG_PAGE_SIZE as u64) % rcrd_region_size;
            self.dev.write_at(off, &page)?;
            last_lsn = ll;
            next_lsn = ll + 1;
            page_idx += 1;
            chunk_start = chunk_end;
        }

        // Stamp restart pages: dirty, current_lsn = last_lsn.
        let dirty_page = super::logfile::build_restart_page(last_lsn, log_size, false);
        self.dev.write_at(logfile_offset, &dirty_page)?;
        self.dev.write_at(
            logfile_offset + super::logfile::LOG_PAGE_SIZE as u64,
            &dirty_page,
        )?;

        // Apply in-place writes.
        for e in entries {
            self.dev.write_at(e.target_offset, &e.redo_bytes)?;
        }

        // Stamp restart pages: clean.
        let clean_page = super::logfile::build_restart_page(last_lsn, log_size, true);
        self.dev.write_at(logfile_offset, &clean_page)?;
        self.dev.write_at(
            logfile_offset + super::logfile::LOG_PAGE_SIZE as u64,
            &clean_page,
        )?;
        Ok(())
    }

    /// Read the `flags` u16 from the current on-disk MFT record. Falls
    /// back to `FLAG_IN_USE` when the record can't be parsed.
    fn read_existing_flags(&mut self) -> Result<u16> {
        let off = self
            .fs
            .writer
            .as_ref()
            .ok_or_else(|| crate::Error::Unsupported("ntfs: writer not initialised".into()))?
            .mft_offset(self.rec_no)?;
        let mut rec = vec![0u8; self.rec_size];
        self.dev.read_at(off, &mut rec)?;
        if mft::apply_fixup(&mut rec, self.sector_size).is_err() {
            return Ok(mft::RecordHeader::FLAG_IN_USE);
        }
        let hdr = mft::RecordHeader::parse(&rec)?;
        Ok(hdr.flags)
    }
}

/// Resolve `$LogFile`'s on-disk byte offset and total size by reading
/// MFT record 2 and decoding its non-resident `$DATA` run list.
fn locate_logfile(fs: &mut super::Ntfs, dev: &mut dyn BlockDevice) -> Result<(u64, u64)> {
    let (rec_size, sector_size, cluster_size) = {
        let w = fs.writer.as_ref().expect("writer present");
        (
            w.layout.mft_record_size as usize,
            w.layout.bytes_per_sector as usize,
            w.cluster_size,
        )
    };
    let mft_off = fs
        .writer
        .as_ref()
        .expect("writer present")
        .mft_offset(super::format::REC_LOGFILE)?;
    let mut rec = vec![0u8; rec_size];
    dev.read_at(mft_off, &mut rec)?;
    mft::apply_fixup(&mut rec, sector_size)?;
    let hdr = mft::RecordHeader::parse(&rec)?;
    for attr_res in AttributeIter::new(&rec, hdr.first_attribute_offset as usize) {
        let attr = attr_res?;
        if attr.type_code == TYPE_DATA && attr.name.is_empty() {
            if let AttributeKind::NonResident {
                runs, real_size, ..
            } = attr.kind
            {
                if let Some(first) = runs.first().and_then(|r| r.lcn) {
                    return Ok((first * cluster_size, real_size));
                }
            }
        }
    }
    Ok((0, 0))
}

/// Apply the redo bytes of every entry in order, then re-stamp the two
/// restart pages with `CleanDismount` set and `current_lsn = 0`
/// (records will be overwritten on the next sync).
fn replay_and_clean(
    dev: &mut dyn BlockDevice,
    logfile_offset: u64,
    log_size: u64,
    entries: &[super::logfile::RedoEntry],
    _last_seq: u64,
) -> Result<()> {
    for e in entries {
        dev.write_at(e.target_offset, &e.redo_bytes)?;
    }
    // Zero the record-page region and re-stamp clean restart pages.
    let rstr_region = 2 * super::logfile::LOG_PAGE_SIZE as u64;
    if log_size > rstr_region {
        dev.zero_range(logfile_offset + rstr_region, log_size - rstr_region)?;
    }
    let page = super::logfile::build_restart_page(0, log_size, true);
    dev.write_at(logfile_offset, &page)?;
    dev.write_at(logfile_offset + super::logfile::LOG_PAGE_SIZE as u64, &page)?;
    Ok(())
}

/// Walk index entries starting at `start_off` in `buf`, looking for one
/// whose `file_ref` lower-48 bits equal `rec_no`. When found, patch the
/// embedded $FILE_NAME key's `real_size` (offset 48..56) and
/// `allocated_size` (offset 40..48) to `new_len` / `new_len ceil'd to
/// cluster`. Returns `true` if a patch was applied.
fn patch_entries_for_record(buf: &mut [u8], start_off: usize, rec_no: u64, new_len: u64) -> bool {
    let mut cursor = start_off;
    let mut changed = false;
    while cursor + 16 <= buf.len() {
        let entry_len =
            u16::from_le_bytes(buf[cursor + 8..cursor + 10].try_into().unwrap()) as usize;
        let flags = u32::from_le_bytes(buf[cursor + 12..cursor + 16].try_into().unwrap());
        if entry_len < 16 || cursor + entry_len > buf.len() {
            break;
        }
        let is_last = flags & 0x02 != 0;
        let key_len =
            u16::from_le_bytes(buf[cursor + 10..cursor + 12].try_into().unwrap()) as usize;
        if !is_last && key_len >= 66 {
            let file_ref = u64::from_le_bytes(buf[cursor..cursor + 8].try_into().unwrap());
            if (file_ref & 0x0000_FFFF_FFFF_FFFF) == rec_no {
                let key_off = cursor + 16;
                // Patch allocated_size / real_size (rounded to next 4 KiB
                // for the allocated value — matches what create_file emits).
                let allocated = (new_len + 4095) & !4095;
                buf[key_off + 40..key_off + 48].copy_from_slice(&allocated.to_le_bytes());
                buf[key_off + 48..key_off + 56].copy_from_slice(&new_len.to_le_bytes());
                changed = true;
                // Keep scanning — there may be more than one $FILE_NAME
                // namespace (Win32 + DOS) all referencing the same record.
            }
        }
        if is_last {
            break;
        }
        cursor += entry_len;
    }
    changed
}

/// Decode the (filetime, flags, namespace, name) of a `$FILE_NAME`
/// attribute value so we can rebuild it. We only carry forward what
/// `build_file_name_value` consumes.
fn decode_file_name_meta(v: &[u8]) -> Result<(u64, u32, u8, String)> {
    let fname = FileName::parse(v)?;
    Ok((
        fname.modified_time,
        fname.flags,
        fname.namespace,
        fname.name,
    ))
}

impl<'a> Read for NtfsFileHandle<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.read_internal(buf)
    }
}

impl<'a> Write for NtfsFileHandle<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write_internal(buf).map_err(std::io::Error::other)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> Seek for NtfsFileHandle<'a> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
            SeekFrom::End(d) => self.len as i128 + d as i128,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "ntfs: seek to negative offset",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

impl<'a> FileHandle for NtfsFileHandle<'a> {
    fn len(&self) -> u64 {
        self.len
    }

    fn set_len(&mut self, new_len: u64) -> Result<()> {
        if new_len == self.len {
            return Ok(());
        }
        if new_len > self.len {
            self.ensure_capacity(new_len)?;
            // Zero the freshly-exposed tail.
            let old_len = self.len;
            self.zero_range(old_len, new_len)?;
            self.len = new_len;
        } else {
            // Shrink: free trailing clusters (non-resident) or just
            // truncate the resident buffer.
            if let Some(ref mut bytes) = self.resident {
                bytes.truncate(new_len as usize);
            } else {
                let cs = self.cluster_size;
                let keep_clusters = new_len.div_ceil(cs);
                self.shrink_runs(keep_clusters)?;
            }
            self.len = new_len;
            if self.pos > self.len {
                self.pos = self.len;
            }
        }
        self.dirty = true;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.flush_record()?;
        // Push the volume-level state (bitmap, MFT-bitmap, boot, MFT rec 0)
        // so subsequent re-opens see the new allocations.
        self.fs.flush(self.dev)?;
        Ok(())
    }
}

impl<'a> Drop for NtfsFileHandle<'a> {
    fn drop(&mut self) {
        // Best-effort persistence on drop. Errors are swallowed —
        // callers who care about durability call sync() explicitly.
        let _ = self.flush_record();
    }
}

// ---------------------------------------------------------------------------
// open_file_rw adapter — lives on Ntfs so the trait impl in mod.rs can
// delegate to it.
// ---------------------------------------------------------------------------

impl super::Ntfs {
    /// Implementation of [`crate::fs::Filesystem::open_file_rw`].
    pub(super) fn open_rw<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
        flags: crate::fs::OpenFlags,
        meta: Option<crate::fs::FileMeta>,
    ) -> Result<Box<dyn FileHandle + 'a>> {
        // Make the handle writable. A freshly `open`ed image has
        // `writer: None`; reconstruct it from disk so reopened volumes
        // (e.g. NTFS inside a qcow2) accept byte-level edits.
        self.ensure_writer(dev)?;

        // Refuse if the existing $LogFile has non-zero content — that
        // would mean the volume carries a real journal we don't
        // implement. Format-emitted volumes have a zero log.
        Self::ensure_clean_log(self, dev)?;

        // Resolve path → MFT record (or create if requested).
        let rec_no = self.lookup_path(dev, path).ok();

        let rec_no = match rec_no {
            Some(r) => r,
            None => {
                if !flags.create {
                    return Err(crate::Error::InvalidArgument(format!(
                        "ntfs: no such file: {path:?}"
                    )));
                }
                let m = meta.ok_or_else(|| {
                    crate::Error::InvalidArgument(
                        "ntfs: open_file_rw create=true requires meta".into(),
                    )
                })?;
                // Create an empty file via the normal writer path.
                self.create_file(dev, path, crate::fs::FileSource::Zero(0), m)?;
                // Look up the freshly-minted record.
                self.lookup_path(dev, path)?
            }
        };

        // Decode the MFT record to extract $DATA / $FILE_NAME / $SI.
        let (rec_size, sector_size, cluster_size) = {
            let w = self.writer.as_ref().expect("writer present");
            (
                w.layout.mft_record_size as usize,
                w.layout.bytes_per_sector as usize,
                w.cluster_size,
            )
        };
        let off = self
            .writer
            .as_ref()
            .expect("writer present")
            .mft_offset(rec_no)?;
        let mut rec = vec![0u8; rec_size];
        dev.read_at(off, &mut rec)?;
        mft::apply_fixup(&mut rec, sector_size)?;
        let hdr = mft::RecordHeader::parse(&rec)?;
        if hdr.is_directory() {
            return Err(crate::Error::InvalidArgument(format!(
                "ntfs: {path:?} is a directory"
            )));
        }
        if !hdr.is_in_use() {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: record {rec_no} is not in use"
            )));
        }

        let mut resident: Option<Vec<u8>> = None;
        let mut runs: Vec<Extent> = Vec::new();
        let mut real_size: u64 = 0;
        let mut file_name_value: Option<Vec<u8>> = None;
        let mut si_value: Option<Vec<u8>> = None;
        let mut parent_ref: u64 = 0;
        for attr_res in AttributeIter::new(&rec, hdr.first_attribute_offset as usize) {
            let attr = attr_res?;
            match attr.type_code {
                super::attribute::TYPE_STANDARD_INFORMATION => {
                    if let AttributeKind::Resident { value, .. } = attr.kind {
                        si_value = Some(value.to_vec());
                    }
                }
                TYPE_FILE_NAME => {
                    if let AttributeKind::Resident { value, .. } = attr.kind {
                        // Prefer Win32 / POSIX / Win32+DOS over DOS-only.
                        let fname = FileName::parse(value)?;
                        let take = match fname.namespace {
                            FileName::NAMESPACE_DOS => file_name_value.is_none(),
                            _ => true,
                        };
                        if take {
                            file_name_value = Some(value.to_vec());
                            parent_ref = fname.parent_mft_ref;
                        }
                    }
                }
                TYPE_DATA if attr.name.is_empty() => match attr.kind {
                    AttributeKind::Resident { value, .. } => {
                        resident = Some(value.to_vec());
                        real_size = value.len() as u64;
                    }
                    AttributeKind::NonResident {
                        runs: rs,
                        real_size: r,
                        ..
                    } => {
                        runs = rs;
                        real_size = r;
                    }
                },
                _ => {}
            }
        }

        let file_name_value = file_name_value.ok_or_else(|| {
            crate::Error::InvalidImage(format!("ntfs: record {rec_no} missing $FILE_NAME"))
        })?;
        let si_value = si_value.unwrap_or_default();

        let mut handle = NtfsFileHandle {
            fs: self,
            dev,
            rec_no,
            rec_size,
            sector_size,
            cluster_size,
            pos: 0,
            len: real_size,
            resident,
            runs,
            file_name_value,
            parent_ref,
            si_value,
            dirty: false,
        };

        if flags.truncate && handle.len != 0 {
            handle.set_len(0)?;
        }
        if flags.append {
            handle.pos = handle.len;
        }

        Ok(Box::new(handle))
    }

    /// Inspect `$LogFile`. We accept three states:
    ///
    /// 1. **Structured + clean.** Restart pages parse as RSTR and carry
    ///    `CleanDismount`. No replay needed.
    /// 2. **Structured + dirty.** Restart pages parse as RSTR but lack
    ///    `CleanDismount`. We have records to replay; walk them and
    ///    apply the redo bytes in order, then re-stamp the restart pages
    ///    with `CleanDismount` set.
    /// 3. **All-zero log.** Legacy / unformatted log. Treated as clean.
    ///
    /// Any other shape (foreign journal, torn write) → refuse.
    fn ensure_clean_log(fs: &mut super::Ntfs, dev: &mut dyn BlockDevice) -> Result<()> {
        let (logfile_offset, log_size) = locate_logfile(fs, dev)?;
        if log_size == 0 {
            return Ok(());
        }

        // Peek the first 8 bytes — if they're zero, treat as legacy clean.
        let mut head = [0u8; 8];
        dev.read_at(logfile_offset, &mut head)?;
        if head.iter().all(|&b| b == 0) {
            return Ok(());
        }

        match super::logfile::read_current_restart(dev, logfile_offset)? {
            Some((view, _which)) if view.is_clean() => Ok(()),
            Some((view, _which)) => {
                // Dirty restart area — replay the records, then re-stamp
                // restart pages as clean.
                let entries = super::logfile::walk_records(dev, logfile_offset, log_size)?;
                let seq = super::logfile::lsn_seq(view.current_lsn).max(1);
                replay_and_clean(dev, logfile_offset, log_size, &entries, seq)
            }
            None => Err(crate::Error::Unsupported(
                "ntfs: open_file_rw refuses to mutate a volume with a non-LFS-shaped $LogFile"
                    .into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;
    use crate::fs::ntfs::Ntfs;
    use crate::fs::ntfs::format::FormatOpts;
    use crate::fs::{FileMeta, FileSource, Filesystem, OpenFlags};
    use std::path::Path;

    fn fresh(size: u64) -> (MemoryBackend, Ntfs) {
        let mut dev = MemoryBackend::new(size);
        let opts = FormatOpts {
            volume_label: "rw-test".into(),
            ..Default::default()
        };
        let ntfs = Ntfs::format(&mut dev, &opts).unwrap();
        (dev, ntfs)
    }

    fn read_all(fs: &mut Ntfs, dev: &mut MemoryBackend, path: &str) -> Vec<u8> {
        let mut r = fs.open_file_reader(dev, path).unwrap();
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).unwrap();
        buf
    }

    #[test]
    fn open_file_rw_partial_write_round_trip() {
        let (mut dev, mut fs) = fresh(8 * 1024 * 1024);
        let payload = b"AAAAAAAAAAAAAAAAAAAA";
        fs.create_file(
            &mut dev,
            "/x.bin",
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(payload.to_vec())),
                len: payload.len() as u64,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        {
            let mut h = Filesystem::open_file_rw(
                &mut fs,
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

        let bytes = read_all(&mut fs, &mut dev, "/x.bin");
        let mut expected = payload.to_vec();
        expected[5..10].copy_from_slice(b"ZZZZZ");
        assert_eq!(bytes, expected);
    }

    #[test]
    fn open_file_rw_extends_file() {
        let (mut dev, mut fs) = fresh(8 * 1024 * 1024);
        fs.create_file(
            &mut dev,
            "/g.txt",
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"hello".to_vec())),
                len: 5,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        {
            let mut h = Filesystem::open_file_rw(
                &mut fs,
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
        let bytes = read_all(&mut fs, &mut dev, "/g.txt");
        assert_eq!(bytes, b"hello, world!");
    }

    #[test]
    fn open_file_rw_set_len_grow_and_shrink() {
        let (mut dev, mut fs) = fresh(16 * 1024 * 1024);
        fs.create_file(
            &mut dev,
            "/s.bin",
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"ABCDEFGH".to_vec())),
                len: 8,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        {
            let mut h = Filesystem::open_file_rw(
                &mut fs,
                &mut dev,
                Path::new("/s.bin"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
            h.set_len(20 * 1024).unwrap();
            assert_eq!(h.len(), 20 * 1024);
            h.sync().unwrap();
        }
        {
            let bytes = read_all(&mut fs, &mut dev, "/s.bin");
            assert_eq!(bytes.len(), 20 * 1024);
            assert_eq!(&bytes[..8], b"ABCDEFGH");
            assert!(bytes[8..].iter().all(|&b| b == 0));
        }
        {
            let mut h = Filesystem::open_file_rw(
                &mut fs,
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
        let bytes = read_all(&mut fs, &mut dev, "/s.bin");
        assert_eq!(bytes, b"ABCD");
    }

    #[test]
    fn open_file_rw_append() {
        let (mut dev, mut fs) = fresh(8 * 1024 * 1024);
        fs.create_file(
            &mut dev,
            "/a.txt",
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"head ".to_vec())),
                len: 5,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        {
            let mut h = Filesystem::open_file_rw(
                &mut fs,
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
        let bytes = read_all(&mut fs, &mut dev, "/a.txt");
        assert_eq!(bytes, b"head tail");
    }

    #[test]
    fn open_file_rw_create_new() {
        let (mut dev, mut fs) = fresh(8 * 1024 * 1024);
        {
            let mut h = Filesystem::open_file_rw(
                &mut fs,
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
        let bytes = read_all(&mut fs, &mut dev, "/n.txt");
        assert_eq!(bytes, b"freshly created");
    }

    #[test]
    fn open_file_rw_refused_when_log_unparseable() {
        let (mut dev, mut fs) = fresh(8 * 1024 * 1024);
        // Corrupt both restart pages so neither parses as a valid LFS
        // restart area. We want the dirty-but-unrecognised path to
        // refuse.
        let (lcn, cs) = {
            let w = fs.writer.as_ref().unwrap();
            (w.layout.logfile_lcn, w.cluster_size)
        };
        let phys = lcn * cs;
        dev.write_at(phys, b"JUNKJUNK\x01\x02\x03\x04").unwrap();
        dev.write_at(phys + 4096, b"JUNKJUNK\x01\x02\x03\x04")
            .unwrap();

        let res = Filesystem::open_file_rw(
            &mut fs,
            &mut dev,
            Path::new("/somefile"),
            OpenFlags::default(),
            None,
        );
        match res {
            Err(crate::Error::Unsupported(msg)) => {
                assert!(msg.contains("LogFile") || msg.contains("log"));
            }
            Err(other) => panic!("expected Unsupported on dirty log, got {other:?}"),
            Ok(_) => panic!("expected Unsupported on dirty log, got Ok"),
        }
    }

    /// Simulate a crash mid-commit: write a file, then poison its MFT
    /// record in-place AND clear the restart-area's `CleanDismount` flag.
    /// Reopening should replay the LFS records — restoring the MFT
    /// record — and the file should read back correctly.
    #[test]
    fn open_file_rw_replay_restores_metadata() {
        let (mut dev, mut fs) = fresh(16 * 1024 * 1024);

        // Step 1: create the file and write through open_file_rw so an
        // LFS record carrying the redo MFT bytes lands in $LogFile. We
        // pick an initial payload short enough that the open_file_rw
        // overwrite truncates / refills it cleanly.
        let payload = b"abcdefghij".to_vec();
        fs.create_file(
            &mut dev,
            "/jnl.bin",
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(payload.clone())),
                len: payload.len() as u64,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        // Look up the MFT record number now (before we poison it) for
        // later verification.
        let rec_no = fs.lookup_path(&mut dev, "/jnl.bin").unwrap();
        let mft_off = fs.writer.as_ref().unwrap().mft_offset(rec_no).unwrap();
        let rec_size = fs.writer.as_ref().unwrap().layout.mft_record_size as usize;

        // Use open_file_rw to overwrite the contents — this populates the
        // log with a metadata-update record.
        {
            let mut h = Filesystem::open_file_rw(
                &mut fs,
                &mut dev,
                Path::new("/jnl.bin"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
            // Write 18 bytes (longer than the initial 10) so the MFT
            // record's $DATA length is bumped to 18 — recorded in the
            // journal entry.
            h.write_all(b"REPLAY MATERIAL XX").unwrap();
            h.sync().unwrap();
        }

        // Step 2: poison the in-place MFT record bytes — overwrite with
        // zeros so any read of it fails. Also flip the restart-area
        // CleanDismount flag off to simulate a crash between writing
        // the records and stamping clean.
        let zero_rec = vec![0u8; rec_size];
        dev.write_at(mft_off, &zero_rec).unwrap();
        let (lcn, cs) = {
            let w = fs.writer.as_ref().unwrap();
            (w.layout.logfile_lcn, w.cluster_size)
        };
        let logfile_offset = lcn * cs;
        // Read current restart page, parse, rebuild as dirty.
        let (view, _) = super::super::logfile::read_current_restart(&mut dev, logfile_offset)
            .unwrap()
            .expect("restart parses");
        let dirty =
            super::super::logfile::build_restart_page(view.current_lsn, view.file_size, false);
        dev.write_at(logfile_offset, &dirty).unwrap();
        dev.write_at(
            logfile_offset + super::super::logfile::LOG_PAGE_SIZE as u64,
            &dirty,
        )
        .unwrap();

        // Step 3: reopen the file via open_file_rw. The ensure_clean_log
        // path should replay the LFS records, restoring the MFT record;
        // a subsequent read should see the rewritten contents.
        {
            let _h = Filesystem::open_file_rw(
                &mut fs,
                &mut dev,
                Path::new("/jnl.bin"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
        }
        let bytes = read_all(&mut fs, &mut dev, "/jnl.bin");
        assert_eq!(bytes, b"REPLAY MATERIAL XX");

        // Step 4: subsequent open should see a clean log (the replay
        // path re-stamps CleanDismount) and not error out.
        {
            let mut h = Filesystem::open_file_rw(
                &mut fs,
                &mut dev,
                Path::new("/jnl.bin"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
            h.write_all(b"!").unwrap();
            h.sync().unwrap();
        }
        let bytes = read_all(&mut fs, &mut dev, "/jnl.bin");
        assert_eq!(&bytes[..1], b"!");
        // Length is preserved at 18; tail unchanged.
        assert_eq!(bytes.len(), 18);
        assert_eq!(&bytes[1..], b"EPLAY MATERIAL XX");
    }

    /// Two sync()s in a row on the same handle should each emit a valid
    /// LFS record and leave the restart area marked clean. The
    /// `ntfsfix --no-action` external sanity test for the resulting
    /// volume continues to pass (verified by the dedicated test below;
    /// this test asserts only the on-disk shape).
    #[test]
    fn open_file_rw_journal_idempotent_clean_close() {
        let (mut dev, mut fs) = fresh(16 * 1024 * 1024);
        fs.create_file(
            &mut dev,
            "/k.bin",
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"A".to_vec())),
                len: 1,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        for _ in 0..3 {
            let mut h = Filesystem::open_file_rw(
                &mut fs,
                &mut dev,
                Path::new("/k.bin"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
            h.write_all(b"Z").unwrap();
            h.sync().unwrap();
        }
        // Confirm clean shutdown shape after all writes.
        let (lcn, cs) = {
            let w = fs.writer.as_ref().unwrap();
            (w.layout.logfile_lcn, w.cluster_size)
        };
        let logfile_offset = lcn * cs;
        let (view, _) = super::super::logfile::read_current_restart(&mut dev, logfile_offset)
            .unwrap()
            .expect("restart parses");
        assert!(view.is_clean(), "restart area must be clean after sync");
    }

    /// External-tool round-trip: format, mutate via open_file_rw, then
    /// run `ntfsfix --no-action`. Skips when `ntfsfix` is not installed.
    #[test]
    fn open_file_rw_round_trip_ntfsfix_clean() {
        // Skip when ntfsfix isn't on PATH.
        if std::process::Command::new("ntfsfix")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_err()
        {
            eprintln!("ntfsfix not found — skipping");
            return;
        }

        let (mut dev, mut fs) = fresh(16 * 1024 * 1024);
        fs.create_file(
            &mut dev,
            "/round.bin",
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"original".to_vec())),
                len: 8,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        {
            let mut h = Filesystem::open_file_rw(
                &mut fs,
                &mut dev,
                Path::new("/round.bin"),
                OpenFlags::default(),
                None,
            )
            .unwrap();
            h.write_all(b"REWRITTEN").unwrap();
            h.sync().unwrap();
        }

        // Dump the image to a temp file and call ntfsfix on it.
        let mut tmp = std::env::temp_dir();
        tmp.push(format!("fstool-ntfs-rw-{}.img", std::process::id()));
        let buf = dev.as_slice().to_vec();
        std::fs::write(&tmp, &buf).unwrap();
        let out = std::process::Command::new("ntfsfix")
            .arg("--no-action")
            .arg(&tmp)
            .output()
            .expect("ntfsfix invocation");
        let _ = std::fs::remove_file(&tmp);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        // `ntfsfix` prints one of two report styles depending on whether
        // it had to repair anything:
        //
        //   * If the volume mounts cleanly (post-$Secure population), it
        //     reports "Mounting volume... OK" / "Processing of $MFT and
        //     $MFTMirr completed successfully." / "NTFS partition ...
        //     was processed successfully."
        //   * If something forced the journal-rebuild / MFT-mirror
        //     compare path, it prints "Comparing $MFTMirr to $MFT... OK"
        //     followed by the same "completed successfully" line.
        //
        // Either output indicates a structurally sound image.
        let combined = format!("{stdout}\n{stderr}");
        assert!(
            combined.contains("Processing of $MFT and $MFTMirr completed successfully."),
            "ntfsfix MFT processing failed: stdout={stdout}, stderr={stderr}"
        );
    }
}
