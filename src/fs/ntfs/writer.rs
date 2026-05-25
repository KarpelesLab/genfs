//! NTFS writer state + user-facing create_* methods.
//!
//! The writer owns the post-format state ([`format::LayoutResult`]) and
//! maintains a few mutable caches so user file/dir creations can be
//! performed without re-reading the whole volume on every call:
//!
//! * `bitmap`: in-memory mirror of `$Bitmap:$DATA`, flushed in `Ntfs::flush`.
//! * `mft_bitmap`: 1 bit per MFT record, marks used slots.
//! * `next_user_record`: monotonic hint for the next free MFT slot.
//!
//! Design choices:
//!
//! * Small directories use only `$INDEX_ROOT` (no `$INDEX_ALLOCATION`). When
//!   the root would overflow the resident attribute budget we promote the
//!   directory to `$INDEX_ALLOCATION` by emitting an `INDX` block and
//!   storing only a child pointer in the root.
//! * File data streams cluster-by-cluster through a 64 KiB scratch buffer
//!   — never reads the whole file into memory.
//! * Reparse points are written for symlinks (tag = IO_REPARSE_TAG_SYMLINK).
//! * Compression / encryption / sparse / ADS via writer / hard-links past
//!   the first $FILE_NAME / extended $Extend population are rejected with
//!   [`crate::Error::Unsupported`].

use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::{DeviceKind, FileMeta, FileSource};

use super::attribute::{
    FileName, TYPE_DATA, TYPE_FILE_NAME, TYPE_INDEX_ALLOCATION, TYPE_INDEX_ROOT,
    TYPE_REPARSE_POINT, TYPE_STANDARD_INFORMATION,
};
use super::format::{
    self, FIRST_USER_RECORD, FormatOpts, LayoutResult, REC_ROOT, build_file_name_value,
    build_non_resident_attr, build_resident_attr, build_si_value_with_security, emit_record,
    encode_single_run, insert_into_index_root, pack_mft_ref, rewrite_resident_attr,
    security_id_for, unix_to_filetime,
};
use super::mft;
use super::secure::SecurityClass;

/// Maximum bytes a directory's $INDEX_ROOT may grow to before we promote
/// to $INDEX_ALLOCATION. The MFT record is 1024 bytes; subtract space
/// for $SI + $FILE_NAME + headers and we have ~700 bytes for the root —
/// promote at 512 to stay well under any limit.
const MAX_INDEX_ROOT_BYTES: usize = 512;

/// 64 KiB scratch buffer for streaming file data.
const WRITE_SCRATCH: usize = 64 * 1024;

/// All mutable writer state. Lives on [`super::Ntfs`].
#[derive(Debug)]
pub struct WriterState {
    pub layout: LayoutResult,
    /// MFT bitmap: which record slots are in use.
    pub mft_bitmap: Vec<u8>,
    /// Hint for the next user record number to try.
    pub next_user_record: u64,
    /// Path-to-record cache for directories. Populated on demand from the
    /// in-memory $INDEX_ROOT after a create_dir.
    pub dir_cache: std::collections::HashMap<String, u64>,
    /// Cluster-size shortcut (matches layout.cluster_size).
    pub cluster_size: u64,
    /// Indicates we need to re-stamp boot/bitmap/MFT on next flush.
    pub dirty: bool,
}

impl WriterState {
    pub fn new(layout: LayoutResult) -> Self {
        let mft_records = layout.mft_records;
        let mut mft_bitmap = vec![0u8; mft_records.div_ceil(8) as usize];
        // Records 0..15 are in use after format.
        for r in 0..16u64 {
            mft_bitmap[(r / 8) as usize] |= 1u8 << ((r % 8) as u8);
        }
        let mut dir_cache = std::collections::HashMap::new();
        dir_cache.insert("/".to_string(), REC_ROOT);
        let cluster_size = layout.cluster_size as u64;
        Self {
            layout,
            mft_bitmap,
            next_user_record: FIRST_USER_RECORD,
            dir_cache,
            cluster_size,
            dirty: true,
        }
    }

    /// Allocate the next free MFT record number. Grows the MFT if all
    /// allocated records are in use (returns Unsupported if we'd need to
    /// grow into a non-contiguous extent — keeps the writer simple).
    fn allocate_mft_record(&mut self, dev: &mut dyn BlockDevice) -> Result<u64> {
        for r in self.next_user_record..self.layout.mft_records {
            let i = (r / 8) as usize;
            let m = 1u8 << ((r % 8) as u8);
            if self.mft_bitmap[i] & m == 0 {
                self.mft_bitmap[i] |= m;
                self.next_user_record = r + 1;
                return Ok(r);
            }
        }
        // Need to extend the MFT.
        self.extend_mft(dev)?;
        // Retry once.
        for r in self.next_user_record..self.layout.mft_records {
            let i = (r / 8) as usize;
            let m = 1u8 << ((r % 8) as u8);
            if self.mft_bitmap[i] & m == 0 {
                self.mft_bitmap[i] |= m;
                self.next_user_record = r + 1;
                return Ok(r);
            }
        }
        Err(crate::Error::Unsupported(
            "ntfs: writer cannot extend $MFT further".into(),
        ))
    }

    /// Allocate one additional MFT extent (8 clusters = 32 records).
    fn extend_mft(&mut self, _dev: &mut dyn BlockDevice) -> Result<()> {
        let new_clusters = 8u64;
        let new_lcn = self.layout.bitmap.allocate(new_clusters)?;
        // Extend in-memory state.
        let rec_size = self.layout.mft_record_size as u64;
        let new_records = new_clusters * self.cluster_size / rec_size;
        self.layout.mft_extents.push((new_lcn, new_clusters));
        self.layout.mft_records += new_records;
        // Grow MFT bitmap.
        let new_bm_size = self.layout.mft_records.div_ceil(8) as usize;
        while self.mft_bitmap.len() < new_bm_size {
            self.mft_bitmap.push(0);
        }
        self.dirty = true;
        Ok(())
    }

    /// Allocate `count` contiguous clusters.
    pub fn alloc_clusters(&mut self, count: u64) -> Result<u64> {
        self.dirty = true;
        self.layout.bitmap.allocate(count)
    }

    /// Compute the physical byte offset for MFT record `rec_no`.
    pub fn mft_offset(&self, rec_no: u64) -> Result<u64> {
        let rec_size = self.layout.mft_record_size as u64;
        let target = rec_no * rec_size;
        let mut walked: u64 = 0;
        for &(lcn, length) in &self.layout.mft_extents {
            let span = length * self.cluster_size;
            if target < walked + span {
                let local = target - walked;
                return Ok(lcn * self.cluster_size + local);
            }
            walked += span;
        }
        Err(crate::Error::InvalidImage(format!(
            "ntfs: record {rec_no} past end of $MFT"
        )))
    }
}

/// Public helpers exposed on [`super::Ntfs`].
impl super::Ntfs {
    /// Format `dev` as a fresh NTFS volume with `opts`. Returns a writable
    /// `Ntfs` handle whose internal state tracks the layout for subsequent
    /// `create_*` calls.
    pub fn format(dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<Self> {
        let layout = format::format_volume(dev, opts)?;
        let mut ntfs = Self::open(dev)?;
        ntfs.writer = Some(WriterState::new(layout));
        // Stamp the root directory's $I30 with entries for the canonical
        // system files (records 0..=15 except the root itself). Real-world
        // NTFS volumes index these in `$I30`; `ntfs-3g` refuses to mount a
        // volume whose root index doesn't carry them.
        ntfs.index_system_files_in_root(dev)?;
        Ok(ntfs)
    }

    /// Build $I30 index entries in the root directory for each system MFT
    /// record (records 0..=15) so `ntfs-3g` and other drivers find them
    /// under "/". The entries are pulled from each system record's own
    /// `$FILE_NAME` attribute so the indexed name + size + flags match
    /// exactly. Called once at the tail of `format()`.
    ///
    /// Skips:
    /// * Record 5 (the root itself — avoids self-reference in `$I30`).
    /// * Records whose MFT slot couldn't be located or doesn't carry a
    ///   resident `$FILE_NAME` (defensive — shouldn't happen post-format).
    fn index_system_files_in_root(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        // Records 0..=15 minus the root (5). `ntfs-3g` and the kernel use
        // a binary search keyed by the $UpCase-folded name when looking
        // entries up in $I30, so the entries MUST be stored in collation
        // order — inserting in MFT-record order produces an unsorted
        // index and lookups for late-alphabet names (`$Secure`,
        // `$UpCase`, `$Volume`) silently fail to find them.
        //
        // The well-known system file names are pure ASCII, so $UpCase
        // collation degenerates to ASCII-uppercase ordering. The list
        // below is pre-sorted on the uppercase form of each name.
        const SYSTEM_RECS: &[u64] = &[
            4,  // $AttrDef
            8,  // $BadClus
            6,  // $Bitmap
            7,  // $Boot
            11, // $Extend
            2,  // $LogFile
            0,  // $MFT
            1,  // $MFTMirr
            12, // $Reserved12
            13, // $Reserved13
            14, // $Reserved14
            15, // $Reserved15
            9,  // $Secure
            10, // $UpCase
            3,  // $Volume
        ];
        let (rec_size, sector_size) = {
            let w = self.writer.as_ref().expect("writer present");
            (
                w.layout.mft_record_size as usize,
                w.layout.bytes_per_sector as usize,
            )
        };
        for &rec_no in SYSTEM_RECS {
            let off = self
                .writer
                .as_ref()
                .expect("writer present")
                .mft_offset(rec_no)?;
            let mut rec = vec![0u8; rec_size];
            dev.read_at(off, &mut rec)?;
            mft::apply_fixup(&mut rec, sector_size)?;
            let Some(fn_value) = extract_resident_attr_value(&rec, TYPE_FILE_NAME, "") else {
                continue;
            };
            let is_dir = mft::RecordHeader::parse(&rec)
                .map(|h| h.is_directory())
                .unwrap_or(false);
            self.add_entry_to_dir(dev, REC_ROOT, &fn_value, rec_no, is_dir)?;
        }
        Ok(())
    }

    /// Create a regular file at `path` populated from `src` with metadata `meta`.
    pub fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        src: FileSource,
        meta: FileMeta,
    ) -> Result<()> {
        self.ensure_writer(dev)?;
        let (parent_path, base_name) = split_parent(path)?;
        let parent_rec = self.resolve_dir(dev, &parent_path)?;
        let file_size = src.len().map_err(crate::Error::from)?;
        let (mut reader, _len) = src.open().map_err(crate::Error::from)?;

        let writer = self.writer.as_mut().expect("writer present");
        let rec_no = writer.allocate_mft_record(dev)?;
        let filetime = unix_to_filetime(meta.mtime);

        // Decide resident vs non-resident.
        let rec_size = writer.layout.mft_record_size as usize;
        let cluster_size = writer.cluster_size;
        // Resident budget: rec_size minus headers (~232 bytes for $SI, $FN, terminator).
        let resident_budget = rec_size.saturating_sub(232);
        let (data_attr, alloc_clusters) = if (file_size as usize) <= resident_budget {
            // Read full file into a Vec (small).
            let mut buf = Vec::with_capacity(file_size as usize);
            // Bounded copy.
            let mut remaining = file_size;
            let mut tmp = [0u8; WRITE_SCRATCH];
            while remaining > 0 {
                let want = (remaining as usize).min(tmp.len());
                let n = reader.read(&mut tmp[..want]).map_err(crate::Error::from)?;
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                remaining -= n as u64;
            }
            (build_resident_attr(TYPE_DATA, &[], &buf, 0, 0), 0u64)
        } else {
            // Non-resident: allocate clusters, stream data through scratch.
            let need_clusters = file_size.div_ceil(cluster_size);
            let data_lcn = writer.alloc_clusters(need_clusters)?;
            // Stream-write through 64 KiB scratch.
            let mut scratch = vec![0u8; WRITE_SCRATCH];
            let mut written: u64 = 0;
            while written < file_size {
                let chunk = ((file_size - written) as usize).min(scratch.len());
                let mut filled = 0;
                while filled < chunk {
                    let n = reader
                        .read(&mut scratch[filled..chunk])
                        .map_err(crate::Error::from)?;
                    if n == 0 {
                        break;
                    }
                    filled += n;
                }
                if filled < chunk {
                    // Underflow — zero-pad and stop.
                    for b in &mut scratch[filled..chunk] {
                        *b = 0;
                    }
                }
                let phys = data_lcn * cluster_size + written;
                dev.write_at(phys, &scratch[..chunk])?;
                written += chunk as u64;
            }
            // Pad the final cluster to a cluster boundary so the trailing
            // cluster doesn't carry stale data.
            let last_cluster_end = need_clusters * cluster_size;
            if written < last_cluster_end {
                let pad_off = data_lcn * cluster_size + written;
                let pad_len = (last_cluster_end - written) as usize;
                let pad = vec![0u8; pad_len];
                dev.write_at(pad_off, &pad)?;
            }
            let runs = encode_single_run(data_lcn, need_clusters);
            let attr = build_non_resident_attr(
                TYPE_DATA,
                &[],
                &runs,
                0,
                need_clusters - 1,
                need_clusters * cluster_size,
                file_size,
                file_size,
                0,
                0,
            );
            (attr, need_clusters)
        };
        let _ = alloc_clusters;

        let parent_ref = pack_mft_ref(parent_rec, 1);
        let si = build_resident_attr(
            TYPE_STANDARD_INFORMATION,
            &[],
            &build_si_value_with_security(
                filetime,
                dos_attrs_from_mode(meta.mode, false),
                security_id_for(SecurityClass::User),
            ),
            0,
            0,
        );
        let fn_value = build_file_name_value(
            parent_ref,
            base_name,
            dos_flags_from_mode(meta.mode, false),
            file_size,
            (file_size + cluster_size - 1) & !(cluster_size - 1),
            filetime,
            FileName::NAMESPACE_WIN32,
        );
        let fn_attr = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 1);

        // Emit the new MFT record.
        let mut rec_buf = vec![0u8; rec_size];
        emit_record(
            &mut rec_buf,
            rec_size,
            rec_no,
            mft::RecordHeader::FLAG_IN_USE,
            &[si, fn_attr, data_attr],
            writer.layout.bytes_per_sector as usize,
            1,
        );
        let off = writer.mft_offset(rec_no)?;
        dev.write_at(off, &rec_buf)?;

        // Stage parent index entry.
        self.add_entry_to_dir(dev, parent_rec, &fn_value, rec_no, false)?;
        Ok(())
    }

    /// Create a directory at `path`.
    pub fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        meta: FileMeta,
    ) -> Result<()> {
        self.ensure_writer(dev)?;
        let (parent_path, base_name) = split_parent(path)?;
        let parent_rec = self.resolve_dir(dev, &parent_path)?;
        let writer = self.writer.as_mut().expect("writer present");
        let rec_no = writer.allocate_mft_record(dev)?;
        let rec_size = writer.layout.mft_record_size as usize;
        let filetime = unix_to_filetime(meta.mtime);
        let parent_ref = pack_mft_ref(parent_rec, 1);

        let si = build_resident_attr(
            TYPE_STANDARD_INFORMATION,
            &[],
            &build_si_value_with_security(
                filetime,
                dos_attrs_from_mode(meta.mode, true),
                security_id_for(SecurityClass::User),
            ),
            0,
            0,
        );
        let fn_value = build_file_name_value(
            parent_ref,
            base_name,
            dos_flags_from_mode(meta.mode, true),
            0,
            0,
            filetime,
            FileName::NAMESPACE_WIN32,
        );
        let fn_attr = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 1);
        let i30_name: Vec<u8> = "$I30"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        let idx_root = build_resident_attr(
            TYPE_INDEX_ROOT,
            &i30_name,
            &format::build_empty_index_root(),
            0,
            0,
        );
        let mut rec_buf = vec![0u8; rec_size];
        emit_record(
            &mut rec_buf,
            rec_size,
            rec_no,
            mft::RecordHeader::FLAG_IN_USE | mft::RecordHeader::FLAG_DIRECTORY,
            &[si, fn_attr, idx_root],
            writer.layout.bytes_per_sector as usize,
            1,
        );
        let off = writer.mft_offset(rec_no)?;
        dev.write_at(off, &rec_buf)?;

        // Stage parent index entry.
        self.add_entry_to_dir(dev, parent_rec, &fn_value, rec_no, true)?;

        // Cache for subsequent path resolves.
        let dir_path = normalize_path(path);
        if let Some(w) = self.writer.as_mut() {
            w.dir_cache.insert(dir_path, rec_no);
        }
        Ok(())
    }

    /// Create a symbolic link at `path` pointing at `target`.
    /// Emits a $REPARSE_POINT with `IO_REPARSE_TAG_SYMLINK` (0xA000000C).
    pub fn create_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        target: &str,
        meta: FileMeta,
    ) -> Result<()> {
        self.ensure_writer(dev)?;
        let (parent_path, base_name) = split_parent(path)?;
        let parent_rec = self.resolve_dir(dev, &parent_path)?;
        let writer = self.writer.as_mut().expect("writer present");
        let rec_no = writer.allocate_mft_record(dev)?;
        let rec_size = writer.layout.mft_record_size as usize;
        let filetime = unix_to_filetime(meta.mtime);
        let parent_ref = pack_mft_ref(parent_rec, 1);

        let si = build_resident_attr(
            TYPE_STANDARD_INFORMATION,
            &[],
            &build_si_value_with_security(
                filetime,
                0x400, // REPARSE_POINT attribute
                security_id_for(SecurityClass::User),
            ),
            0,
            0,
        );
        let fn_value = build_file_name_value(
            parent_ref,
            base_name,
            0x400, // reparse-point file
            0,
            0,
            filetime,
            FileName::NAMESPACE_WIN32,
        );
        let fn_attr = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 1);
        let empty_data = build_resident_attr(TYPE_DATA, &[], &[], 0, 0);

        // Reparse data: tag + symlink reparse buffer.
        // Symlink reparse buffer layout:
        //   substitute_name_offset (u16) substitute_name_length (u16)
        //   print_name_offset (u16) print_name_length (u16)
        //   flags (u32: 1 = relative path)
        //   PathBuffer (substitute name UTF-16, print name UTF-16)
        let target_utf16: Vec<u16> = target.encode_utf16().collect();
        let target_bytes: Vec<u8> = target_utf16.iter().flat_map(|u| u.to_le_bytes()).collect();
        let substitute_off = 0u16;
        let substitute_len = target_bytes.len() as u16;
        let print_off = substitute_len;
        let print_len = substitute_len;
        let flags: u32 = if target.starts_with('/') || target.starts_with('\\') {
            0
        } else {
            1
        };
        let mut reparse_data = Vec::new();
        reparse_data.extend_from_slice(&substitute_off.to_le_bytes());
        reparse_data.extend_from_slice(&substitute_len.to_le_bytes());
        reparse_data.extend_from_slice(&print_off.to_le_bytes());
        reparse_data.extend_from_slice(&print_len.to_le_bytes());
        reparse_data.extend_from_slice(&flags.to_le_bytes());
        reparse_data.extend_from_slice(&target_bytes);
        reparse_data.extend_from_slice(&target_bytes);

        let reparse_tag: u32 = 0xA000_000C;
        let reparse_len = reparse_data.len() as u16;
        let mut reparse_payload = Vec::new();
        reparse_payload.extend_from_slice(&reparse_tag.to_le_bytes());
        reparse_payload.extend_from_slice(&reparse_len.to_le_bytes());
        reparse_payload.extend_from_slice(&0u16.to_le_bytes()); // reserved
        reparse_payload.extend_from_slice(&reparse_data);

        let reparse_attr = build_resident_attr(TYPE_REPARSE_POINT, &[], &reparse_payload, 0, 0);

        let mut rec_buf = vec![0u8; rec_size];
        emit_record(
            &mut rec_buf,
            rec_size,
            rec_no,
            mft::RecordHeader::FLAG_IN_USE,
            &[si, fn_attr, empty_data, reparse_attr],
            writer.layout.bytes_per_sector as usize,
            1,
        );
        let off = writer.mft_offset(rec_no)?;
        dev.write_at(off, &rec_buf)?;

        self.add_entry_to_dir(dev, parent_rec, &fn_value, rec_no, false)?;
        Ok(())
    }

    /// Special-file creation (FIFO / device nodes / sockets). NTFS doesn't
    /// natively model these — we refuse with `Unsupported`.
    pub fn create_device(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &str,
        _kind: DeviceKind,
        _major: u32,
        _minor: u32,
        _meta: FileMeta,
    ) -> Result<()> {
        Err(crate::Error::Unsupported(
            "ntfs: special files (char/block/fifo/socket) are not representable".into(),
        ))
    }

    /// Reconstruct a [`WriterState`] from an already-formatted on-disk
    /// volume so a *reopened* image (`Ntfs::open`, `writer: None`) can be
    /// mutated — e.g. an NTFS filesystem living inside a qcow2 used as a
    /// read/write store. Everything [`Self::flush`] consumes is recovered
    /// from the boot sector plus the well-known system MFT records, using
    /// the same readers the read path is already validated on:
    ///
    /// * geometry + `volume_serial` from `self.boot`;
    /// * `$MFT` (rec 0) `$DATA` run list → `mft_extents` / `mft_records`;
    /// * `$MFT` (rec 0) `$BITMAP` run list + bytes → MFT record bitmap;
    /// * `$Bitmap` (rec 6) `$DATA` run list + bytes → cluster allocation
    ///   bitmap (so new allocations never collide with existing data);
    /// * recs 2/4/10 `$DATA` run lists → $LogFile/$AttrDef/$UpCase
    ///   locations (best-effort; `flush` never rewrites these).
    fn reconstruct_writer_state(&mut self, dev: &mut dyn BlockDevice) -> Result<WriterState> {
        let cluster_size = self.boot.cluster_size() as u64;
        let mft_record_size = self.boot.mft_record_size();
        let bytes_per_sector = self.boot.bytes_per_sector;
        let sectors_per_cluster = self.boot.sectors_per_cluster;
        let index_record_size = self.boot.index_record_size();
        let total_clusters = dev.total_size() / cluster_size;

        // --- $MFT (record 0): $DATA extents + $BITMAP ---
        let mft_set = self.load_record_set(dev, 0)?;
        let mft_rec = &mft_set[0].1;
        let mft_extents = extract_non_resident_runs(mft_rec, TYPE_DATA, "").ok_or_else(|| {
            crate::Error::InvalidImage("ntfs: $MFT (rec 0) has no non-resident $DATA".into())
        })?;
        let mft_clusters_total: u64 = mft_extents.iter().map(|&(_, len)| len).sum();
        let mft_records = mft_clusters_total * cluster_size / mft_record_size as u64;

        let mft_bitmap_runs = extract_non_resident_runs(mft_rec, super::attribute::TYPE_BITMAP, "")
            .ok_or_else(|| {
                crate::Error::Unsupported(
                    "ntfs: reopen-mutate requires a non-resident $MFT $BITMAP".into(),
                )
            })?;
        let (mft_bitmap_lcn, mft_bitmap_clusters) = mft_bitmap_runs[0];
        let mft_bitmap = read_runs_prefix(
            dev,
            &mft_bitmap_runs,
            cluster_size,
            mft_records.div_ceil(8) as usize,
        )?;

        // --- $Bitmap (record 6): cluster allocation bitmap ---
        let bm_set = self.load_record_set(dev, 6)?;
        let bm_rec = &bm_set[0].1;
        let bitmap_runs = extract_non_resident_runs(bm_rec, TYPE_DATA, "").ok_or_else(|| {
            crate::Error::InvalidImage("ntfs: $Bitmap (rec 6) has no non-resident $DATA".into())
        })?;
        let bitmap_lcn = bitmap_runs[0].0;
        let bitmap_clusters: u64 = bitmap_runs.iter().map(|&(_, len)| len).sum();
        let bitmap_bytes = read_runs_prefix(
            dev,
            &bitmap_runs,
            cluster_size,
            total_clusters.div_ceil(8) as usize,
        )?;
        let bitmap = format::BitmapAlloc {
            bytes: bitmap_bytes,
            total: total_clusters,
            next_hint: 0,
        };

        // --- best-effort system file locations (flush never rewrites them) ---
        let (logfile_lcn, logfile_clusters) = self.first_data_run(dev, 2);
        let (attrdef_lcn, attrdef_clusters) = self.first_data_run(dev, 4);
        let (upcase_lcn, upcase_clusters) = self.first_data_run(dev, 10);

        let layout = LayoutResult {
            cluster_size: cluster_size as u32,
            bytes_per_sector,
            sectors_per_cluster,
            total_clusters,
            mft_record_size,
            index_record_size,
            mft_extents,
            mft_records,
            mftmirr_lcn: self.boot.mft_mirr_lcn,
            bitmap_lcn,
            bitmap_clusters,
            bitmap,
            mft_bitmap_lcn,
            mft_bitmap_clusters,
            volume_serial: self.boot.volume_serial,
            upcase_lcn,
            upcase_clusters,
            logfile_lcn,
            logfile_clusters,
            attrdef_lcn,
            attrdef_clusters,
        };

        // First free MFT slot ≥ FIRST_USER_RECORD (0..16 are system records).
        let next_user_record = (FIRST_USER_RECORD..mft_records)
            .find(|&r| {
                let i = (r / 8) as usize;
                let m = 1u8 << ((r % 8) as u8);
                i >= mft_bitmap.len() || mft_bitmap[i] & m == 0
            })
            .unwrap_or(mft_records);

        let mut state = WriterState::new(layout);
        state.mft_bitmap = mft_bitmap;
        state.next_user_record = next_user_record;
        state.dirty = false;
        Ok(state)
    }

    /// Make this handle writable. `Ntfs::open` returns a read-only handle
    /// (`writer: None`); the first mutation lazily reconstructs the writer
    /// state from disk. No-op once a writer exists (after `format` or a
    /// prior mutation). Reads stay cheap — the cluster bitmap is only read
    /// here, on first write.
    pub(super) fn ensure_writer(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        if self.writer.is_none() {
            let state = self.reconstruct_writer_state(dev)?;
            self.writer = Some(state);
        }
        Ok(())
    }

    /// First non-resident `$DATA` run `(lcn, clusters)` of system record
    /// `rec_no`, or `(0, 0)` if absent. Best-effort: used only to populate
    /// `LayoutResult` fields `flush` never rewrites.
    fn first_data_run(&mut self, dev: &mut dyn BlockDevice, rec_no: u64) -> (u64, u64) {
        self.load_record_set(dev, rec_no)
            .ok()
            .and_then(|s| {
                extract_non_resident_runs(&s[0].1, TYPE_DATA, "").and_then(|r| r.first().copied())
            })
            .unwrap_or((0, 0))
    }

    /// Persist outstanding writer state to disk. Re-stamps $Bitmap, MFT
    /// record 0's $DATA run list (if MFT grew), and the boot sectors.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        let Some(w) = self.writer.as_mut() else {
            return Ok(());
        };
        if !w.dirty {
            return Ok(());
        }

        let cluster_size = w.cluster_size;
        let rec_size = w.layout.mft_record_size as usize;
        let bps = w.layout.bytes_per_sector as usize;

        // 1) Restamp $Bitmap data.
        {
            let off = w.layout.bitmap_lcn * cluster_size;
            dev.write_at(off, &w.layout.bitmap.bytes)?;
            let padded = w.layout.bitmap_clusters * cluster_size;
            if (w.layout.bitmap.bytes.len() as u64) < padded {
                let pad = vec![0u8; (padded - w.layout.bitmap.bytes.len() as u64) as usize];
                dev.write_at(off + w.layout.bitmap.bytes.len() as u64, &pad)?;
            }
        }
        // 2) Restamp MFT-internal bitmap.
        {
            let off = w.layout.mft_bitmap_lcn * cluster_size;
            dev.write_at(off, &w.mft_bitmap)?;
        }
        // 3) Re-stamp $MFT record 0 with the updated $DATA run list.
        {
            let mut rec_buf = vec![0u8; rec_size];
            let filetime = unix_to_filetime(0);
            let parent_root_ref = pack_mft_ref(REC_ROOT, 1);
            format::build_mft_record(
                &mut rec_buf,
                rec_size,
                parent_root_ref,
                &w.layout.mft_extents,
                w.layout.mft_records,
                w.layout.mft_bitmap_lcn,
                w.layout.mft_bitmap_clusters,
                filetime,
                cluster_size,
                bps,
            );
            // First extent's start is record 0's home.
            let off = w.layout.mft_extents[0].0 * cluster_size;
            dev.write_at(off, &rec_buf)?;
        }
        // 4) Boot sector: nothing structural changed, but stamp it again for
        //    durability — also re-write the backup at the last LBA.
        {
            let mut boot_buf = vec![0u8; bps];
            dev.read_at(0, &mut boot_buf)?;
            let last_lba_offset =
                (w.layout.total_clusters * (cluster_size / bps as u64) - 1) * bps as u64;
            dev.write_at(last_lba_offset, &boot_buf)?;
        }
        w.dirty = false;
        dev.sync()?;
        Ok(())
    }

    /// Resolve a directory path to its MFT record number. Caches results
    /// for subsequent lookups.
    fn resolve_dir(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<u64> {
        let norm = normalize_path(path);
        if let Some(w) = self.writer.as_ref() {
            if let Some(&rec) = w.dir_cache.get(&norm) {
                return Ok(rec);
            }
        }
        let rec = self.lookup_path(dev, &norm)?;
        if let Some(w) = self.writer.as_mut() {
            w.dir_cache.insert(norm, rec);
        }
        Ok(rec)
    }

    /// Append an entry pointing at `(file_ref, file_name_value)` into the
    /// $INDEX_ROOT of MFT record `dir_rec`. Re-reads, modifies, re-writes
    /// the record. If the root would overflow, promotes the index to
    /// $INDEX_ALLOCATION (a single block holding all entries).
    fn add_entry_to_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_rec: u64,
        file_name_value: &[u8],
        file_rec: u64,
        _is_directory: bool,
    ) -> Result<()> {
        let writer = self.writer.as_mut().expect("writer present");
        let rec_size = writer.layout.mft_record_size as usize;
        let sector_size = writer.layout.bytes_per_sector as usize;
        let cluster_size = writer.cluster_size;
        let _ = cluster_size;
        // Read the directory record.
        let off = writer.mft_offset(dir_rec)?;
        let mut rec = vec![0u8; rec_size];
        dev.read_at(off, &mut rec)?;
        mft::apply_fixup(&mut rec, sector_size)?;

        // Build the new index entry.
        let file_ref = pack_mft_ref(file_rec, 1);
        let entry = build_index_entry(file_ref, file_name_value, 0, None);

        // Find current $INDEX_ROOT $I30 value.
        let current =
            extract_resident_attr_value(&rec, TYPE_INDEX_ROOT, "$I30").ok_or_else(|| {
                crate::Error::InvalidImage("ntfs: directory missing $INDEX_ROOT".into())
            })?;

        // If already promoted (LARGE_INDEX flag set in the index header), the
        // root holds only a child-pointer terminator; new entries belong in
        // the allocation block.
        let is_large_index = current.len() >= 29 && current[28] & 0x01 != 0;
        if is_large_index {
            return self.insert_into_allocation_block(dev, dir_rec, &entry);
        }

        match insert_into_index_root(&current, &entry, MAX_INDEX_ROOT_BYTES) {
            Ok(new_value) => {
                rewrite_resident_attr(&mut rec, rec_size, TYPE_INDEX_ROOT, "$I30", &new_value)?;
                // Re-install fixup and write.
                mft::install_fixup(&mut rec, sector_size, 1);
                dev.write_at(off, &rec)?;
                Ok(())
            }
            Err(crate::Error::Unsupported(_)) => {
                // Promote the directory to $INDEX_ALLOCATION.
                self.promote_index_to_allocation(dev, dir_rec, &entry)
            }
            Err(e) => Err(e),
        }
    }

    /// Read the INDX allocation block for a promoted directory, insert
    /// `new_entry` into it, and write it back.
    fn insert_into_allocation_block(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_rec: u64,
        new_entry: &[u8],
    ) -> Result<()> {
        let writer = self.writer.as_mut().expect("writer present");
        let rec_size = writer.layout.mft_record_size as usize;
        let sector_size = writer.layout.bytes_per_sector as usize;
        let cluster_size = writer.cluster_size;
        let block_size = writer.layout.index_record_size as usize;

        let off = writer.mft_offset(dir_rec)?;
        let mut rec = vec![0u8; rec_size];
        dev.read_at(off, &mut rec)?;
        mft::apply_fixup(&mut rec, sector_size)?;

        // Find $INDEX_ALLOCATION $I30 runs.
        let alloc_runs = extract_non_resident_runs(&rec, TYPE_INDEX_ALLOCATION, "$I30")
            .ok_or_else(|| {
                crate::Error::InvalidImage(
                    "ntfs: promoted directory missing $INDEX_ALLOCATION".into(),
                )
            })?;
        let (alloc_lcn, _alloc_clusters) = alloc_runs[0];

        // Read the (sole) INDX block.
        let block_off = alloc_lcn * cluster_size;
        let mut block = vec![0u8; block_size];
        dev.read_at(block_off, &mut block)?;
        mft::apply_fixup(&mut block, sector_size)?;

        // Extract existing entries from the block.
        let first_entry_off = u32::from_le_bytes(block[0x18..0x1C].try_into().unwrap()) as usize;
        let bytes_in_use = u32::from_le_bytes(block[0x1C..0x20].try_into().unwrap()) as usize;
        let entries_start = 0x18 + first_entry_off;
        let entries_end = 0x18 + bytes_in_use;
        let mut existing_entries: Vec<Vec<u8>> = Vec::new();
        let mut cursor = entries_start;
        while cursor + 16 <= entries_end {
            let entry_len =
                u16::from_le_bytes(block[cursor + 8..cursor + 10].try_into().unwrap()) as usize;
            if entry_len < 16 || cursor + entry_len > entries_end {
                break;
            }
            let flags = u32::from_le_bytes(block[cursor + 12..cursor + 16].try_into().unwrap());
            let is_last = flags & 0x02 != 0;
            if is_last {
                break;
            }
            existing_entries.push(block[cursor..cursor + entry_len].to_vec());
            cursor += entry_len;
        }
        existing_entries.push(new_entry.to_vec());

        // Rebuild and write.
        let mut new_block = build_indx_block(block_size, sector_size, 0, &existing_entries)?;
        mft::install_fixup(&mut new_block, sector_size, 1);
        dev.write_at(block_off, &new_block)?;
        let _ = rec_size;
        Ok(())
    }

    /// Promote a directory's index from "small" (root-only) to "large"
    /// ($INDEX_ROOT pointing at a single $INDEX_ALLOCATION block holding
    /// all entries).
    fn promote_index_to_allocation(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_rec: u64,
        new_entry: &[u8],
    ) -> Result<()> {
        let writer = self.writer.as_mut().expect("writer present");
        let rec_size = writer.layout.mft_record_size as usize;
        let sector_size = writer.layout.bytes_per_sector as usize;
        let cluster_size = writer.cluster_size;
        let index_block_size = writer.layout.index_record_size as u64;
        let blocks_per_cluster = cluster_size / index_block_size;
        let clusters_per_block = if blocks_per_cluster == 0 {
            index_block_size.div_ceil(cluster_size)
        } else {
            1
        };

        // Read current directory record.
        let off = writer.mft_offset(dir_rec)?;
        let mut rec = vec![0u8; rec_size];
        dev.read_at(off, &mut rec)?;
        mft::apply_fixup(&mut rec, sector_size)?;

        // Pull the existing entries out of the root.
        let current =
            extract_resident_attr_value(&rec, TYPE_INDEX_ROOT, "$I30").ok_or_else(|| {
                crate::Error::InvalidImage(
                    "ntfs: directory missing $INDEX_ROOT for promotion".into(),
                )
            })?;
        let mut existing_entries: Vec<Vec<u8>> = Vec::new();
        let mut cursor = 16 + 16;
        let bytes_in_use = u32::from_le_bytes(current[20..24].try_into().unwrap()) as usize;
        let entries_end = 16 + bytes_in_use;
        while cursor + 16 <= entries_end {
            let entry_len =
                u16::from_le_bytes(current[cursor + 8..cursor + 10].try_into().unwrap()) as usize;
            let flags = u32::from_le_bytes(current[cursor + 12..cursor + 16].try_into().unwrap());
            if entry_len < 16 || cursor + entry_len > entries_end {
                break;
            }
            let is_last = flags & 0x02 != 0;
            if !is_last {
                existing_entries.push(current[cursor..cursor + entry_len].to_vec());
            } else {
                break;
            }
            cursor += entry_len;
        }
        existing_entries.push(new_entry.to_vec());

        // Build INDX block payload.
        let mut block_buf = build_indx_block(
            index_block_size as usize,
            sector_size,
            0, // this block's VCN
            &existing_entries,
        )?;
        // Install fixup on the INDX block (it uses USA the same way MFT does).
        mft::install_fixup(&mut block_buf, sector_size, 1);

        // Allocate clusters for $INDEX_ALLOCATION.
        let alloc_lcn = writer.alloc_clusters(clusters_per_block)?;
        let alloc_off = alloc_lcn * cluster_size;
        dev.write_at(alloc_off, &block_buf)?;
        if block_buf.len() < (clusters_per_block * cluster_size) as usize {
            let pad = vec![0u8; (clusters_per_block * cluster_size) as usize - block_buf.len()];
            dev.write_at(alloc_off + block_buf.len() as u64, &pad)?;
        }

        // Build the new $INDEX_ROOT with LARGE_INDEX flag + a single
        // "child" terminator entry pointing at VCN 0.
        let new_root = build_large_index_root(0);
        rewrite_resident_attr(&mut rec, rec_size, TYPE_INDEX_ROOT, "$I30", &new_root)?;

        // Add $INDEX_ALLOCATION attribute (named "$I30").
        let i30_name: Vec<u8> = "$I30"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        let runs = encode_single_run(alloc_lcn, clusters_per_block);
        let alloc_attr = build_non_resident_attr(
            TYPE_INDEX_ALLOCATION,
            &i30_name,
            &runs,
            0,
            clusters_per_block - 1,
            clusters_per_block * cluster_size,
            clusters_per_block * cluster_size,
            clusters_per_block * cluster_size,
            0,
            0,
        );
        // Add $BITMAP attribute for $INDEX_ALLOCATION (1 bit per block).
        // Resident: just one byte, with bit 0 set (block 0 in use).
        let bm_value = vec![0x01u8, 0, 0, 0, 0, 0, 0, 0]; // 8 bytes for alignment
        let bm_attr =
            build_resident_attr(super::attribute::TYPE_BITMAP, &i30_name, &bm_value, 0, 0);

        // Insert these new attributes before the terminator. Use
        // `append_attrs` for that.
        append_attrs(&mut rec, rec_size, &[alloc_attr, bm_attr])?;
        mft::install_fixup(&mut rec, sector_size, 1);
        dev.write_at(off, &rec)?;
        Ok(())
    }
}

/// Map a POSIX mode + isdir to an NTFS DOS-attrs / file_attributes word.
fn dos_attrs_from_mode(mode: u16, isdir: bool) -> u32 {
    let mut a = 0u32;
    if mode & 0o222 == 0 {
        a |= 0x01; // READONLY
    }
    if isdir {
        // The $STANDARD_INFORMATION.file_attributes does NOT carry the
        // directory bit (FN does that). But ARCHIVE is canonically set
        // on freshly-created files.
    } else {
        a |= 0x20; // ARCHIVE
    }
    a
}

/// Map a POSIX mode + isdir to a $FILE_NAME.flags value.
fn dos_flags_from_mode(mode: u16, isdir: bool) -> u32 {
    let mut a = dos_attrs_from_mode(mode, isdir);
    if isdir {
        a |= 0x1000_0000; // FILE_NAME flag DIRECTORY
    }
    a
}

/// Walk an MFT record looking for a non-resident attribute of `(type_code,
/// name)`. Returns the list of (LCN, length) extents covered by its run list.
/// Read `want_bytes` bytes from the start of a non-resident run list,
/// walking runs in order. Used to recover the cluster / MFT bitmaps when
/// reconstructing writer state for a reopened volume. The result is
/// padded with zeros (or truncated) to exactly `want_bytes`.
fn read_runs_prefix(
    dev: &mut dyn BlockDevice,
    runs: &[(u64, u64)],
    cluster_size: u64,
    want_bytes: usize,
) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::with_capacity(want_bytes);
    for &(lcn, clusters) in runs {
        if out.len() >= want_bytes {
            break;
        }
        let span = (clusters * cluster_size) as usize;
        let mut buf = vec![0u8; span];
        dev.read_at(lcn * cluster_size, &mut buf)?;
        out.extend_from_slice(&buf);
    }
    out.resize(want_bytes, 0);
    Ok(out)
}

fn extract_non_resident_runs(rec: &[u8], type_code: u32, name: &str) -> Option<Vec<(u64, u64)>> {
    let hdr = mft::RecordHeader::parse(rec).ok()?;
    let bytes_in_use = hdr.bytes_in_use as usize;
    let first = hdr.first_attribute_offset as usize;
    let mut cursor = first;
    while cursor + 4 <= bytes_in_use {
        let tc = u32::from_le_bytes(rec[cursor..cursor + 4].try_into().ok()?);
        if tc == 0xFFFF_FFFF {
            return None;
        }
        let len = u32::from_le_bytes(rec[cursor + 4..cursor + 8].try_into().ok()?) as usize;
        let non_resident = rec[cursor + 8] != 0;
        let name_len = rec[cursor + 9] as usize;
        let name_off = u16::from_le_bytes(rec[cursor + 10..cursor + 12].try_into().ok()?) as usize;
        let attr_name = if name_len == 0 {
            String::new()
        } else {
            super::attribute::decode_utf16le(
                &rec[cursor + name_off..cursor + name_off + name_len * 2],
            )
        };
        if tc == type_code && attr_name == name && non_resident {
            let runs_off =
                u16::from_le_bytes(rec[cursor + 0x20..cursor + 0x22].try_into().ok()?) as usize;
            let runs_bytes = &rec[cursor + runs_off..cursor + len];
            let extents = super::run_list::decode(runs_bytes).ok()?;
            let mut out = Vec::new();
            for e in extents {
                if let Some(lcn) = e.lcn {
                    out.push((lcn, e.length));
                }
            }
            return Some(out);
        }
        cursor += len;
    }
    None
}

/// Walk an MFT record looking for a resident attribute of `(type_code, name)`.
/// Returns its value bytes.
fn extract_resident_attr_value(rec: &[u8], type_code: u32, name: &str) -> Option<Vec<u8>> {
    let hdr = mft::RecordHeader::parse(rec).ok()?;
    let bytes_in_use = hdr.bytes_in_use as usize;
    let first = hdr.first_attribute_offset as usize;
    let mut cursor = first;
    while cursor + 4 <= bytes_in_use {
        let tc = u32::from_le_bytes(rec[cursor..cursor + 4].try_into().ok()?);
        if tc == 0xFFFF_FFFF {
            return None;
        }
        let len = u32::from_le_bytes(rec[cursor + 4..cursor + 8].try_into().ok()?) as usize;
        let non_resident = rec[cursor + 8] != 0;
        let name_len = rec[cursor + 9] as usize;
        let name_off = u16::from_le_bytes(rec[cursor + 10..cursor + 12].try_into().ok()?) as usize;
        let attr_name = if name_len == 0 {
            String::new()
        } else {
            super::attribute::decode_utf16le(
                &rec[cursor + name_off..cursor + name_off + name_len * 2],
            )
        };
        if tc == type_code && attr_name == name && !non_resident {
            let value_len =
                u32::from_le_bytes(rec[cursor + 0x10..cursor + 0x14].try_into().ok()?) as usize;
            let value_off =
                u16::from_le_bytes(rec[cursor + 0x14..cursor + 0x16].try_into().ok()?) as usize;
            return Some(rec[cursor + value_off..cursor + value_off + value_len].to_vec());
        }
        cursor += len;
    }
    None
}

/// Append the given attributes into `rec` just before the 0xFFFFFFFF
/// terminator. Updates bytes_in_use and stamps fresh attribute ids.
fn append_attrs(rec: &mut [u8], rec_size: usize, attrs: &[Vec<u8>]) -> Result<()> {
    let hdr = mft::RecordHeader::parse(rec)?;
    let bytes_in_use = hdr.bytes_in_use as usize;
    // Terminator is at bytes_in_use - 4.
    let term_pos = bytes_in_use - 4;
    let total_new: usize = attrs.iter().map(|a| a.len()).sum();
    if term_pos + total_new + 4 > rec_size {
        return Err(crate::Error::Unsupported(
            "ntfs: not enough room in MFT record for new attributes".into(),
        ));
    }
    // Next attribute id
    let mut next_attr_id = u16::from_le_bytes(rec[0x28..0x2A].try_into().unwrap());
    let mut cursor = term_pos;
    for a in attrs {
        rec[cursor..cursor + a.len()].copy_from_slice(a);
        rec[cursor + 14..cursor + 16].copy_from_slice(&next_attr_id.to_le_bytes());
        next_attr_id = next_attr_id.wrapping_add(1);
        cursor += a.len();
    }
    rec[cursor..cursor + 4].copy_from_slice(&[0xFFu8, 0xFF, 0xFF, 0xFF]);
    cursor += 4;
    let new_bytes_in_use = cursor as u32;
    rec[0x18..0x1C].copy_from_slice(&new_bytes_in_use.to_le_bytes());
    rec[0x28..0x2A].copy_from_slice(&next_attr_id.to_le_bytes());
    // Zero tail.
    for b in &mut rec[cursor..rec_size] {
        *b = 0;
    }
    Ok(())
}

/// Build a fresh INDX block carrying the given entries plus a terminator.
///
/// Block layout:
///   0x00..0x04   INDX magic
///   0x04..0x06   usa_offset (= 0x28)
///   0x06..0x08   usa_size   (1 + sectors)
///   0x08..0x10   LSN
///   0x10..0x18   this block's VCN
///   0x18         index header (16 bytes: first_entry_offset, bytes_in_use,
///                bytes_allocated, flags)
///   USA at 0x28..(0x28 + 2*usa_size)
///   first entry — chosen so it sits past the USA on an 8-byte boundary
///   so `first_entry_offset = entry_start - 0x18`.
fn build_indx_block(
    block_size: usize,
    sector_size: usize,
    vcn: u64,
    entries: &[Vec<u8>],
) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; block_size];
    buf[0..4].copy_from_slice(b"INDX");
    // usa_offset = 0x28
    buf[4..6].copy_from_slice(&0x28u16.to_le_bytes());
    let sectors = block_size / sector_size;
    let usa_size = sectors + 1;
    buf[6..8].copy_from_slice(&(usa_size as u16).to_le_bytes());
    // LSN at 8..16 = 0
    buf[16..24].copy_from_slice(&vcn.to_le_bytes());
    // Align entries past the USA (at 0x28 + 2*usa_size).
    let usa_end = 0x28 + 2 * usa_size;
    let entries_start = (usa_end + 7) & !7;
    // First entry offset is relative to the index header start at 0x18.
    let first_entry_offset = (entries_start - 0x18) as u32;
    let term_entry = {
        let mut e = vec![0u8; 16];
        e[8..10].copy_from_slice(&16u16.to_le_bytes());
        e[12..16].copy_from_slice(&0x02u32.to_le_bytes());
        e
    };
    let entries_total: usize = entries.iter().map(|e| e.len()).sum::<usize>() + term_entry.len();
    if entries_start + entries_total > block_size {
        return Err(crate::Error::Unsupported(
            "ntfs: directory entry overflow in single INDX block".into(),
        ));
    }
    // The index header's bytes_in_use counts from the index header start
    // (0x18). It includes the 16-byte header + (any padding from header to
    // entries) + entries.
    let bytes_in_use = (entries_start - 0x18) as u32 + entries_total as u32;
    let bytes_allocated = (block_size - 0x18) as u32;
    let flags: u8 = 0;
    buf[0x18..0x1C].copy_from_slice(&first_entry_offset.to_le_bytes());
    buf[0x1C..0x20].copy_from_slice(&bytes_in_use.to_le_bytes());
    buf[0x20..0x24].copy_from_slice(&bytes_allocated.to_le_bytes());
    buf[0x24] = flags;
    let mut cursor = entries_start;
    for e in entries {
        buf[cursor..cursor + e.len()].copy_from_slice(e);
        cursor += e.len();
    }
    buf[cursor..cursor + term_entry.len()].copy_from_slice(&term_entry);
    Ok(buf)
}

/// Build a LARGE-INDEX $INDEX_ROOT carrying only a terminator with a child
/// pointer at `child_vcn`. Used when a directory has been promoted to
/// $INDEX_ALLOCATION.
fn build_large_index_root(child_vcn: u64) -> Vec<u8> {
    let index_block_size = format::DEFAULT_INDEX_RECORD_SIZE;
    let cpib: i8 = 1;
    let mut v = Vec::new();
    v.extend_from_slice(&TYPE_FILE_NAME.to_le_bytes());
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&index_block_size.to_le_bytes());
    v.push(cpib as u8);
    v.extend_from_slice(&[0u8; 3]);
    let first_entry_offset = 16u32;
    // Terminator with child pointer = 24 bytes.
    let term_entry_len = 24u32;
    let bytes_in_use = 16u32 + term_entry_len;
    v.extend_from_slice(&first_entry_offset.to_le_bytes());
    v.extend_from_slice(&bytes_in_use.to_le_bytes());
    v.extend_from_slice(&bytes_in_use.to_le_bytes());
    v.push(0x01); // LARGE_INDEX
    v.extend_from_slice(&[0u8; 3]);
    // Terminator entry with HAS_CHILD + LAST.
    let mut term = vec![0u8; 24];
    term[8..10].copy_from_slice(&24u16.to_le_bytes());
    term[10..12].copy_from_slice(&0u16.to_le_bytes());
    term[12..16].copy_from_slice(&0x03u32.to_le_bytes()); // HAS_CHILD | LAST
    term[16..24].copy_from_slice(&child_vcn.to_le_bytes());
    v.extend_from_slice(&term);
    v
}

/// Build an index entry holding a $FILE_NAME key.
fn build_index_entry(
    file_ref: u64,
    file_name_value: &[u8],
    flags: u32,
    child_vcn: Option<u64>,
) -> Vec<u8> {
    let key_len = file_name_value.len();
    let mut payload_len = 16 + key_len;
    payload_len = (payload_len + 7) & !7;
    let entry_len = if child_vcn.is_some() {
        payload_len + 8
    } else {
        payload_len
    };
    let mut e = vec![0u8; entry_len];
    e[0..8].copy_from_slice(&file_ref.to_le_bytes());
    e[8..10].copy_from_slice(&(entry_len as u16).to_le_bytes());
    e[10..12].copy_from_slice(&(key_len as u16).to_le_bytes());
    let final_flags = flags | if child_vcn.is_some() { 0x01 } else { 0 };
    e[12..16].copy_from_slice(&final_flags.to_le_bytes());
    e[16..16 + key_len].copy_from_slice(file_name_value);
    if let Some(vcn) = child_vcn {
        let off = entry_len - 8;
        e[off..off + 8].copy_from_slice(&vcn.to_le_bytes());
    }
    e
}

/// Split `/a/b/c` into ("/a/b", "c"). Errors out for non-absolute paths.
fn split_parent(path: &str) -> Result<(String, &str)> {
    if !path.starts_with('/') {
        return Err(crate::Error::InvalidArgument(format!(
            "ntfs: path must be absolute, got {path:?}"
        )));
    }
    let trimmed = path.trim_end_matches('/');
    let last_slash = trimmed
        .rfind('/')
        .ok_or_else(|| crate::Error::InvalidArgument("ntfs: path has no name component".into()))?;
    let (parent, rest) = trimmed.split_at(last_slash);
    let parent = if parent.is_empty() {
        "/".to_string()
    } else {
        parent.to_string()
    };
    let base = &rest[1..]; // skip the slash
    if base.is_empty() {
        return Err(crate::Error::InvalidArgument(
            "ntfs: missing file name".into(),
        ));
    }
    Ok((parent, base))
}

fn normalize_path(path: &str) -> String {
    if path == "/" {
        return "/".to_string();
    }
    path.trim_end_matches('/').to_string()
}
