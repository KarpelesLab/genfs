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
use crate::fs::dir_batch::{DEFAULT_CAPACITY, DirBatch};
use crate::fs::{DeviceKind, FileMeta, FileSource};

use super::attribute::{
    FileName, TYPE_DATA, TYPE_FILE_NAME, TYPE_INDEX_ALLOCATION, TYPE_INDEX_ROOT,
    TYPE_REPARSE_POINT, TYPE_STANDARD_INFORMATION,
};
use super::format::{
    self, FIRST_USER_RECORD, FormatOpts, LayoutResult, REC_ROOT, build_file_name_value,
    build_non_resident_attr, build_resident_attr, build_si_value_with_security, emit_record,
    encode_single_run, pack_mft_ref, rewrite_resident_attr, security_id_for, unix_to_filetime,
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
    /// Pending `$I30` index entries per directory MFT record. Adding a
    /// child stages its (already-built) index entry here instead of
    /// re-reading, re-sorting, and re-writing the parent's index on
    /// every create — that batch is serialized once, on eviction or at
    /// flush. Keyed by the parent's MFT record number; the value is the
    /// raw index-entry bytes from `build_index_entry`.
    pub dir_batch: DirBatch<u64, Vec<u8>>,
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
            dir_batch: DirBatch::new(DEFAULT_CAPACITY),
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

    /// Grow the $MFT. Doubles the current MFT data size (min 8 clusters)
    /// so a large file count accrues in O(log n) extents — otherwise the
    /// hundreds of 8-cluster extents make record 0's $DATA runlist overflow
    /// the 1 KiB record. Also grows the non-resident MFT `$BITMAP`'s
    /// backing clusters (relocating it) when one bit per record no longer
    /// fits the current allocation.
    fn extend_mft(&mut self, _dev: &mut dyn BlockDevice) -> Result<()> {
        let cluster_size = self.cluster_size;
        let rec_size = self.layout.mft_record_size as u64;

        // Double the MFT data region.
        let cur_data: u64 = self.layout.mft_extents.iter().map(|(_, l)| *l).sum();
        let new_clusters = cur_data.max(8);
        let new_lcn = self.layout.bitmap.allocate(new_clusters)?;
        let new_records = new_clusters * cluster_size / rec_size;
        self.layout.mft_extents.push((new_lcn, new_clusters));
        self.layout.mft_records += new_records;

        // Grow the in-memory bitmap to one bit per record.
        let new_bm_size = self.layout.mft_records.div_ceil(8) as usize;
        while self.mft_bitmap.len() < new_bm_size {
            self.mft_bitmap.push(0);
        }
        // Ensure the on-disk $BITMAP region is large enough; relocate to a
        // fresh (doubled) contiguous run when it isn't. The old run leaks
        // into used space — negligible against the MFT itself.
        let needed_clusters = (new_bm_size as u64).div_ceil(cluster_size);
        if needed_clusters > self.layout.mft_bitmap_clusters {
            let grow_to = needed_clusters
                .max(self.layout.mft_bitmap_clusters * 2)
                .max(1);
            let new_bm_lcn = self.layout.bitmap.allocate(grow_to)?;
            self.layout.mft_bitmap_lcn = new_bm_lcn;
            self.layout.mft_bitmap_clusters = grow_to;
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

        let rec_size = writer.layout.mft_record_size as usize;
        let cluster_size = writer.cluster_size;

        // Build $STANDARD_INFORMATION and $FILE_NAME up front so the
        // resident-data budget is sized against their *actual* lengths.
        // A long file name makes $FILE_NAME large; a fixed estimate (the
        // old `rec_size - 232`) lets a ~750-byte body stay resident and
        // overflow the 1 KiB record.
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

        // Space left for a resident $DATA value after the record header +
        // USA, $SI, $FN, the resident-attr header and the 0xFFFFFFFF
        // terminator (plus slack for 8-byte alignment).
        const MFT_HEADER_OVERHEAD: usize = 64;
        const RESIDENT_DATA_HDR: usize = 24;
        const TERM_SLACK: usize = 16;
        let resident_budget = rec_size.saturating_sub(
            MFT_HEADER_OVERHEAD + si.len() + fn_attr.len() + RESIDENT_DATA_HDR + TERM_SLACK,
        );
        // Decide resident vs non-resident.
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

        // Emit the new MFT record ($SI / $FN built above).
        let mut rec_buf = vec![0u8; rec_size];
        emit_record(
            &mut rec_buf,
            rec_size,
            rec_no,
            mft::RecordHeader::FLAG_IN_USE,
            &[si, fn_attr, data_attr],
            writer.layout.bytes_per_sector as usize,
            1,
        )?;
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
        )?;
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
        )?;
        let off = writer.mft_offset(rec_no)?;
        dev.write_at(off, &rec_buf)?;

        self.add_entry_to_dir(dev, parent_rec, &fn_value, rec_no, false)?;
        Ok(())
    }

    /// Create a char or block device node. NTFS-3G represents these as
    /// regular files whose `$DATA` carries an `INTX_FILE` header — an
    /// 8-byte magic (`"IntxCHR\0"` / `"IntxBLK\0"`) followed by 8-byte
    /// LE `major` and 8-byte LE `minor` (layout from
    /// `/usr/include/ntfs-3g/layout.h`). The file is otherwise an
    /// ordinary small-resident-$DATA regular file, so the entire
    /// existing `create_file` path is reused.
    ///
    /// FIFOs and sockets have no `INTX_FILE` magic in `ntfs-3g`'s
    /// vocabulary (only `IntxLNK` / `IntxCHR` / `IntxBLK` are defined),
    /// so they are rejected with `Unsupported`. A consumer that needs
    /// FIFO/socket semantics on NTFS should pick one of the other
    /// writable backends (ext / FAT / HFS+).
    pub fn create_device(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: FileMeta,
    ) -> Result<()> {
        // Build the 24-byte INTX_FILE payload (magic + major + minor LE).
        // The magic bytes are the literal ASCII of `IntxCHR\0` / `IntxBLK\0`
        // — matching `INTX_FILE_TYPES` in ntfs-3g's `layout.h`. Wider
        // device numbers are accepted (u32 fits in the u64 field).
        let magic: &[u8; 8] = match kind {
            DeviceKind::Char => b"IntxCHR\0",
            DeviceKind::Block => b"IntxBLK\0",
            DeviceKind::Fifo | DeviceKind::Socket => {
                return Err(crate::Error::Unsupported(
                    "ntfs: FIFO / socket nodes are not representable in ntfs-3g's Intx \
                     vocabulary (only char / block / symlink); use ext or HFS+ instead"
                        .into(),
                ));
            }
        };
        let mut payload: Vec<u8> = Vec::with_capacity(24);
        payload.extend_from_slice(magic);
        payload.extend_from_slice(&(major as u64).to_le_bytes());
        payload.extend_from_slice(&(minor as u64).to_le_bytes());
        debug_assert_eq!(payload.len(), 24);

        // Hand the payload to create_file. The 24-byte body is well
        // under the resident-$DATA budget so the file stays resident
        // (no clusters allocated), exactly how `ntfs-3g` writes its
        // own device nodes.
        let len = payload.len() as u64;
        let src = FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(payload)),
            len,
        };
        self.create_file(dev, path, src, meta)
    }

    /// Remove the file / empty directory / symlink at `path`. The inverse
    /// of `create_file` / `create_dir` / `create_symlink`:
    ///
    /// 1. splice every parent `$I30` entry pointing at the target out of
    ///    `$INDEX_ROOT` (or, for promoted dirs, out of the single
    ///    `$INDEX_ALLOCATION` INDX block — no de-promotion);
    /// 2. free every non-resident attribute's clusters in `$Bitmap`
    ///    (unnamed `$DATA`, named ADS, `$INDEX_ALLOCATION`,
    ///    non-resident `$BITMAP`);
    /// 3. clear the MFT record's `FLAG_IN_USE`, bump its sequence
    ///    number, and clear the corresponding bit in the MFT bitmap.
    ///
    /// `flush` (caller-driven) restamps `$Bitmap` and the MFT bitmap.
    ///
    /// Refuses with `Unsupported` for:
    /// * the root or any system record (rec < `FIRST_USER_RECORD`),
    /// * records that have spilled into `$ATTRIBUTE_LIST` (writer
    ///   never produces those, so this only fires on third-party images),
    /// * non-empty directories,
    /// * `$DATA` carrying the encrypted / compressed / sparse flags
    ///   (the writer never produces them and freeing their runs by
    ///   physical extent count would be unsafe).
    pub fn remove(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<()> {
        self.ensure_writer(dev)?;
        let norm = normalize_path(path);
        if norm == "/" {
            return Err(crate::Error::InvalidArgument(
                "ntfs: refusing to remove the root directory".into(),
            ));
        }
        let target_rec = self.lookup_path(dev, &norm)?;
        if target_rec < FIRST_USER_RECORD {
            return Err(crate::Error::InvalidArgument(format!(
                "ntfs: refusing to remove system record {target_rec}"
            )));
        }
        let (parent_path, _base_name) = split_parent(&norm)?;
        let parent_rec = self.resolve_dir(dev, &parent_path)?;

        // Load the target's MFT record(s). $ATTRIBUTE_LIST spill is
        // rejected (matches the writer's create-side simplification).
        let records = self.load_record_set(dev, target_rec)?;
        if records.len() > 1 {
            return Err(crate::Error::Unsupported(
                "ntfs: remove of records with $ATTRIBUTE_LIST spill is not supported".into(),
            ));
        }
        let target_rec_bytes = records[0].1.clone();
        let target_hdr = mft::RecordHeader::parse(&target_rec_bytes)?;
        if !target_hdr.is_in_use() {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: record {target_rec} is not in use"
            )));
        }

        // Empty-directory check. read_directory dedups DOS/Win32 entries
        // and only surfaces the system `$`-prefixed entries at the root —
        // which is rejected above — so for a user-created dir an empty
        // index means an empty directory.
        if target_hdr.is_directory() {
            let kids = self.read_directory(dev, target_rec)?;
            if !kids.is_empty() {
                return Err(crate::Error::InvalidArgument(format!(
                    "ntfs: cannot remove non-empty directory ({} entries)",
                    kids.len()
                )));
            }
        }

        // Read the link count from the record header (offset 0x12, u16).
        // If > 1 this is a hard-linked file: unlink only from the given
        // parent, decrement, do NOT free clusters or the MFT record.
        let link_count = u16::from_le_bytes(target_rec_bytes[0x12..0x14].try_into().unwrap());
        let is_hardlink = link_count > 1;

        // Reject encrypted / compressed / sparse $DATA so we don't
        // mis-free clusters of a layout the writer can't produce.
        for attr_res in super::attribute::AttributeIter::new(
            &target_rec_bytes,
            target_hdr.first_attribute_offset as usize,
        ) {
            let attr = attr_res?;
            if attr.type_code == super::attribute::TYPE_DATA
                && (attr.flags
                    & (super::attribute::ATTR_FLAG_COMPRESSED
                        | super::attribute::ATTR_FLAG_ENCRYPTED
                        | super::attribute::ATTR_FLAG_SPARSE))
                    != 0
            {
                return Err(crate::Error::Unsupported(
                    "ntfs: remove of compressed / encrypted / sparse $DATA is not supported".into(),
                ));
            }
        }

        // 1) Splice every parent index entry pointing at target_rec.
        self.remove_entries_from_parent(dev, parent_rec, target_rec)?;

        if is_hardlink {
            // Hard-link unlink: decrement link count + write back.
            let mut rec_buf = target_rec_bytes.clone();
            let new_lc = link_count - 1;
            rec_buf[0x12..0x14].copy_from_slice(&new_lc.to_le_bytes());
            let off = self
                .writer
                .as_ref()
                .expect("writer present")
                .mft_offset(target_rec)?;
            let sector_size = self
                .writer
                .as_ref()
                .expect("writer present")
                .layout
                .bytes_per_sector as usize;
            mft::install_fixup(&mut rec_buf, sector_size, 1);
            dev.write_at(off, &rec_buf)?;
        } else {
            // 2) Free clusters from every non-resident stream.
            self.free_runlist_clusters(&target_rec_bytes)?;
            // 3) Free the MFT record.
            self.mark_record_free(dev, target_rec, target_rec_bytes)?;
        }

        // Drop dir_cache entry for the removed path (and the parent's
        // entry stays, of course). The path normalisation we computed
        // above is already in canonical form.
        if let Some(w) = self.writer.as_mut() {
            w.dir_cache.remove(&norm);
        }

        Ok(())
    }

    /// Splice every index entry whose `file_ref`'s low 48 bits equal
    /// `target_rec` out of the parent directory's `$I30`. Dispatches
    /// between the resident `$INDEX_ROOT` (small) and the single-block
    /// `$INDEX_ALLOCATION` (promoted) cases — never de-promotes.
    fn remove_entries_from_parent(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_rec: u64,
        target_rec: u64,
    ) -> Result<()> {
        let (rec_size, sector_size) = {
            let w = self.writer.as_ref().expect("writer present");
            (
                w.layout.mft_record_size as usize,
                w.layout.bytes_per_sector as usize,
            )
        };
        let off = self
            .writer
            .as_ref()
            .expect("writer present")
            .mft_offset(parent_rec)?;
        let mut rec = vec![0u8; rec_size];
        dev.read_at(off, &mut rec)?;
        mft::apply_fixup(&mut rec, sector_size)?;

        let current =
            extract_resident_attr_value(&rec, TYPE_INDEX_ROOT, "$I30").ok_or_else(|| {
                crate::Error::InvalidImage("ntfs: parent directory missing $INDEX_ROOT".into())
            })?;
        // LARGE_INDEX flag lives at value offset 28 (index-header byte 12).
        let is_large_index = current.len() >= 29 && current[28] & 0x01 != 0;
        if is_large_index {
            // Promoted: real entries live in the single INDX block.
            self.remove_entry_from_allocation_block(dev, parent_rec, target_rec)
        } else {
            // Small: rebuild $INDEX_ROOT with matching entries dropped.
            let new_value = format::remove_entry_from_index_root(&current, target_rec)?;
            rewrite_resident_attr(&mut rec, rec_size, TYPE_INDEX_ROOT, "$I30", &new_value)?;
            mft::install_fixup(&mut rec, sector_size, 1);
            dev.write_at(off, &rec)?;
            Ok(())
        }
    }

    /// Inverse of `insert_into_allocation_block`: read the single INDX
    /// block, drop entries whose `file_ref` low-48 == `target_rec`,
    /// rebuild via `build_indx_block`, write back.
    fn remove_entry_from_allocation_block(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_rec: u64,
        target_rec: u64,
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

        let alloc_runs = extract_non_resident_runs(&rec, TYPE_INDEX_ALLOCATION, "$I30")
            .ok_or_else(|| {
                crate::Error::InvalidImage(
                    "ntfs: promoted directory missing $INDEX_ALLOCATION".into(),
                )
            })?;
        let (alloc_lcn, _alloc_clusters) = alloc_runs[0];

        let block_off = alloc_lcn * cluster_size;
        let mut block = vec![0u8; block_size];
        dev.read_at(block_off, &mut block)?;
        mft::apply_fixup(&mut block, sector_size)?;

        let first_entry_off = u32::from_le_bytes(block[0x18..0x1C].try_into().unwrap()) as usize;
        let bytes_in_use = u32::from_le_bytes(block[0x1C..0x20].try_into().unwrap()) as usize;
        let entries_start = 0x18 + first_entry_off;
        let entries_end = 0x18 + bytes_in_use;
        const FILE_REF_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;

        let mut kept_entries: Vec<Vec<u8>> = Vec::new();
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
            let file_ref =
                u64::from_le_bytes(block[cursor..cursor + 8].try_into().unwrap()) & FILE_REF_MASK;
            if file_ref != target_rec {
                kept_entries.push(block[cursor..cursor + entry_len].to_vec());
            }
            cursor += entry_len;
        }

        let mut new_block = build_indx_block(block_size, sector_size, 0, &kept_entries)?;
        mft::install_fixup(&mut new_block, sector_size, 1);
        dev.write_at(block_off, &new_block)?;
        let _ = rec_size; // silence unused (mirrors insert path)
        Ok(())
    }

    /// Free every cluster owned by the target record's non-resident
    /// attributes (unnamed `$DATA`, named ADS, `$INDEX_ALLOCATION`,
    /// non-resident `$BITMAP`). Sets the writer dirty so flush restamps
    /// `$Bitmap`.
    fn free_runlist_clusters(&mut self, rec_bytes: &[u8]) -> Result<()> {
        let writer = self.writer.as_mut().expect("writer present");
        for (_tc, _name, runs) in for_each_non_resident_attr(rec_bytes) {
            for (lcn, length) in runs {
                for c in lcn..lcn.saturating_add(length) {
                    writer.layout.bitmap.clear(c);
                }
            }
        }
        writer.dirty = true;
        Ok(())
    }

    /// Mark `target_rec` free: clear `FLAG_IN_USE` in the header, bump
    /// the sequence number (NTFS convention so stale file references
    /// can be detected), reinstall fixup, write the record back, clear
    /// the bit in `mft_bitmap`, and lower `next_user_record` so the
    /// slot is reused on the next allocation. Sets `dirty`.
    fn mark_record_free(
        &mut self,
        dev: &mut dyn BlockDevice,
        target_rec: u64,
        mut rec_bytes: Vec<u8>,
    ) -> Result<()> {
        let writer = self.writer.as_mut().expect("writer present");
        let sector_size = writer.layout.bytes_per_sector as usize;
        let off = writer.mft_offset(target_rec)?;

        // Header: flags at 0x16 (u16), seq at 0x10 (u16).
        let mut flags = u16::from_le_bytes(rec_bytes[0x16..0x18].try_into().unwrap());
        flags &= !mft::RecordHeader::FLAG_IN_USE;
        rec_bytes[0x16..0x18].copy_from_slice(&flags.to_le_bytes());
        let seq = u16::from_le_bytes(rec_bytes[0x10..0x12].try_into().unwrap());
        let mut new_seq = seq.wrapping_add(1);
        if new_seq == 0 {
            new_seq = 1;
        }
        rec_bytes[0x10..0x12].copy_from_slice(&new_seq.to_le_bytes());

        mft::install_fixup(&mut rec_bytes, sector_size, 1);
        dev.write_at(off, &rec_bytes)?;

        // Clear the MFT bitmap bit + lower the allocation hint.
        let i = (target_rec / 8) as usize;
        let m = 1u8 << ((target_rec % 8) as u8);
        if i < writer.mft_bitmap.len() {
            writer.mft_bitmap[i] &= !m;
        }
        if target_rec < writer.next_user_record {
            writer.next_user_record = target_rec;
        }
        writer.dirty = true;
        Ok(())
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
        if self.writer.is_none() {
            return Ok(());
        }
        // Serialize every pending directory batch before re-stamping the
        // volume metadata — these writes may allocate clusters (index
        // promotion), which the bitmap restamp below must then capture.
        let pending = self
            .writer
            .as_mut()
            .expect("writer present")
            .dir_batch
            .drain_all();
        for (dir_rec, entries) in pending {
            self.serialize_dir(dev, dir_rec, &entries)?;
        }

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
        // 2) Restamp MFT-internal bitmap (padded to its full region; bits
        //    past the record count read as free).
        {
            let off = w.layout.mft_bitmap_lcn * cluster_size;
            dev.write_at(off, &w.mft_bitmap)?;
            let region = w.layout.mft_bitmap_clusters * cluster_size;
            if (w.mft_bitmap.len() as u64) < region {
                let pad = vec![0u8; (region - w.mft_bitmap.len() as u64) as usize];
                dev.write_at(off + w.mft_bitmap.len() as u64, &pad)?;
            }
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
        // 3b) Re-sync $MFTMirr with the current first four MFT records.
        //     Record 0's $DATA runlist changes when the MFT grows, so the
        //     mirror must be refreshed or ntfs-3g refuses to mount
        //     ("$MFTMirr does not match $MFT"); ntfsfix silently corrected
        //     it, which masked the staleness in tests.
        {
            let mirr_off = w.layout.mftmirr_lcn * cluster_size;
            let mft_start = w.layout.mft_extents[0].0 * cluster_size;
            let mut buf = vec![0u8; 4 * rec_size];
            dev.read_at(mft_start, &mut buf)?;
            dev.write_at(mirr_off, &buf)?;
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
        if let Some(w) = self.writer.as_ref()
            && let Some(&rec) = w.dir_cache.get(&norm)
        {
            return Ok(rec);
        }
        let rec = self.lookup_path(dev, &norm)?;
        if let Some(w) = self.writer.as_mut() {
            w.dir_cache.insert(norm, rec);
        }
        Ok(rec)
    }

    /// Stage an entry pointing at `(file_ref, file_name_value)` for the
    /// `$INDEX_ROOT` of MFT record `dir_rec`. The entry is built now (no
    /// disk I/O) and parked in the per-directory batch cache; the parent
    /// directory's index is serialized once — when this directory is
    /// evicted to make room for another, when the volume is flushed, or
    /// when something reads the directory back (see [`serialize_dir`] and
    /// the flush-on-read in `read_directory`). This turns the old
    /// re-read + re-sort + re-write **per child** (O(N²) for a big
    /// directory) into one pass.
    fn add_entry_to_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_rec: u64,
        file_name_value: &[u8],
        file_rec: u64,
        _is_directory: bool,
    ) -> Result<()> {
        let file_ref = pack_mft_ref(file_rec, 1);
        let entry = build_index_entry(file_ref, file_name_value, 0, None);
        let writer = self.writer.as_mut().expect("writer present");
        // A staged create changes the volume; make sure flush re-stamps.
        writer.dirty = true;
        let victim = writer.dir_batch.stage(dir_rec, entry);
        // If the cache evicted an older directory, serialize it now.
        if let Some((victim_rec, entries)) = victim {
            self.serialize_dir(dev, victim_rec, &entries)?;
        }
        Ok(())
    }

    /// Apply a directory's whole pending batch to its on-disk `$I30` in a
    /// single pass: read the directory record once, merge every staged
    /// entry, sort once, and write once — promoting `$INDEX_ROOT` →
    /// `$INDEX_ALLOCATION` at most once if the combined index overflows
    /// the resident budget. Output is byte-identical to having inserted
    /// the same entries one at a time.
    pub(super) fn serialize_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_rec: u64,
        entries: &[Vec<u8>],
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let writer = self.writer.as_mut().expect("writer present");
        let rec_size = writer.layout.mft_record_size as usize;
        let sector_size = writer.layout.bytes_per_sector as usize;
        // Read the directory record.
        let off = writer.mft_offset(dir_rec)?;
        let mut rec = vec![0u8; rec_size];
        dev.read_at(off, &mut rec)?;
        mft::apply_fixup(&mut rec, sector_size)?;

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
            return self.insert_into_allocation_block(dev, dir_rec, entries);
        }

        match format::insert_entries_into_index_root(&current, entries, MAX_INDEX_ROOT_BYTES) {
            Ok(new_value) => {
                rewrite_resident_attr(&mut rec, rec_size, TYPE_INDEX_ROOT, "$I30", &new_value)?;
                // Re-install fixup and write.
                mft::install_fixup(&mut rec, sector_size, 1);
                dev.write_at(off, &rec)?;
                Ok(())
            }
            Err(crate::Error::Unsupported(_)) => {
                // Promote the directory to $INDEX_ALLOCATION.
                self.promote_index_to_allocation(dev, dir_rec, entries)
            }
            Err(e) => Err(e),
        }
    }

    /// Read the INDX allocation block for a promoted directory, insert
    /// every entry in `new_entries` into it, and write it back once.
    fn insert_into_allocation_block(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_rec: u64,
        new_entries: &[Vec<u8>],
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
        existing_entries.extend(new_entries.iter().cloned());
        // Same rationale as `insert_into_index_root`: `ntfs-3g`'s
        // path lookup binary-searches the INDX block, so the on-disk
        // order must be the NTFS collation key.
        existing_entries.sort_by_key(|e| format::entry_sort_key(e));

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
        new_entries: &[Vec<u8>],
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
        existing_entries.extend(new_entries.iter().cloned());
        // Sort into NTFS collation order. The old per-entry flow relied on
        // a *following* `insert_into_allocation_block` to sort, but a
        // batched promotion is the final write, so it must sort here —
        // `ntfs-3g` binary-searches the INDX block.
        existing_entries.sort_by_key(|e| format::entry_sort_key(e));

        // Bulk-load a balanced B-tree of INDX blocks (one VCN each). A
        // small directory yields a single leaf at VCN 0; a large one a
        // multi-level tree whose root the $INDEX_ROOT points at.
        let (root_vcn, blocks) =
            build_index_btree(&existing_entries, index_block_size as usize, sector_size)?;
        let nblocks = blocks.len() as u64;

        // One VCN per INDX block, `clusters_per_block` clusters each, in
        // one contiguous run so VCN v → LCN base + v*cpb.
        let total_clusters = nblocks * clusters_per_block;
        let alloc_lcn = writer.alloc_clusters(total_clusters)?;
        let block_span = (clusters_per_block * cluster_size) as usize;
        for (vcn, block) in &blocks {
            let off = (alloc_lcn + vcn * clusters_per_block) * cluster_size;
            dev.write_at(off, block)?;
            if block.len() < block_span {
                let pad = vec![0u8; block_span - block.len()];
                dev.write_at(off + block.len() as u64, &pad)?;
            }
        }

        // $INDEX_ROOT: LARGE_INDEX with a single child terminator pointing
        // at the tree root's VCN.
        let new_root = build_large_index_root(root_vcn);
        rewrite_resident_attr(&mut rec, rec_size, TYPE_INDEX_ROOT, "$I30", &new_root)?;

        // Add $INDEX_ALLOCATION attribute (named "$I30").
        let i30_name: Vec<u8> = "$I30"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        let runs = encode_single_run(alloc_lcn, total_clusters);
        let alloc_size = total_clusters * cluster_size;
        let alloc_attr = build_non_resident_attr(
            TYPE_INDEX_ALLOCATION,
            &i30_name,
            &runs,
            0,
            total_clusters - 1,
            alloc_size,
            alloc_size,
            alloc_size,
            0,
            0,
        );
        // $BITMAP: one bit per INDX block (VCN), all in use.
        let bm_value = build_index_bitmap(nblocks);
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

/// `(type_code, attribute_name, runs)` — one entry per non-resident
/// attribute returned by [`for_each_non_resident_attr`].
type NonResidentAttrInfo = (u32, String, Vec<(u64, u64)>);

/// Sibling of [`extract_non_resident_runs`] that enumerates *every*
/// non-resident attribute in `rec` and returns its `(type_code, name,
/// runs)` tuples. The runs include only allocated extents (`lcn !=
/// None`), matching `extract_non_resident_runs`'s convention.
///
/// Used by `free_runlist_clusters` on the remove path: a single MFT
/// record can carry the unnamed `$DATA`, named `$DATA` ADS streams, a
/// directory's `$INDEX_ALLOCATION`, and a non-resident `$BITMAP` — all
/// must be freed when the file is removed.
fn for_each_non_resident_attr(rec: &[u8]) -> Vec<NonResidentAttrInfo> {
    let mut out: Vec<NonResidentAttrInfo> = Vec::new();
    let Ok(hdr) = mft::RecordHeader::parse(rec) else {
        return out;
    };
    let bytes_in_use = hdr.bytes_in_use as usize;
    let first = hdr.first_attribute_offset as usize;
    let mut cursor = first;
    while cursor + 4 <= bytes_in_use {
        let Ok(tc_b) = rec[cursor..cursor + 4].try_into() else {
            break;
        };
        let tc = u32::from_le_bytes(tc_b);
        if tc == 0xFFFF_FFFF {
            break;
        }
        let Ok(len_b) = rec[cursor + 4..cursor + 8].try_into() else {
            break;
        };
        let len = u32::from_le_bytes(len_b) as usize;
        if len == 0 || cursor + len > bytes_in_use {
            break;
        }
        let non_resident = rec[cursor + 8] != 0;
        let name_len = rec[cursor + 9] as usize;
        let name_off =
            u16::from_le_bytes(rec[cursor + 10..cursor + 12].try_into().unwrap_or([0; 2])) as usize;
        let attr_name = if name_len == 0 {
            String::new()
        } else {
            super::attribute::decode_utf16le(
                &rec[cursor + name_off..cursor + name_off + name_len * 2],
            )
        };
        if non_resident {
            let runs_off = u16::from_le_bytes(
                rec[cursor + 0x20..cursor + 0x22]
                    .try_into()
                    .unwrap_or([0; 2]),
            ) as usize;
            let runs_bytes = &rec[cursor + runs_off..cursor + len];
            if let Ok(extents) = super::run_list::decode(runs_bytes) {
                let runs: Vec<(u64, u64)> = extents
                    .into_iter()
                    .filter_map(|e| e.lcn.map(|lcn| (lcn, e.length)))
                    .collect();
                if !runs.is_empty() {
                    out.push((tc, attr_name, runs));
                }
            }
        }
        cursor += len;
    }
    out
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

/// Build one INDX block at `vcn` from already-encoded index entries, with
/// a terminator that ends a leaf (`None`, flags LAST) or an internal node
/// (`Some(child)`, flags HAS_CHILD|LAST + the rightmost child VCN). The
/// header's INDEX_NODE flag (0x24) is set for internal nodes; the USA
/// fixup is installed here.
fn build_indx_block_full(
    block_size: usize,
    sector_size: usize,
    vcn: u64,
    entries: &[Vec<u8>],
    terminator_child: Option<u64>,
) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; block_size];
    buf[0..4].copy_from_slice(b"INDX");
    buf[4..6].copy_from_slice(&0x28u16.to_le_bytes());
    let sectors = block_size / sector_size;
    let usa_size = sectors + 1;
    buf[6..8].copy_from_slice(&(usa_size as u16).to_le_bytes());
    buf[16..24].copy_from_slice(&vcn.to_le_bytes());
    let usa_end = 0x28 + 2 * usa_size;
    let entries_start = (usa_end + 7) & !7;
    let first_entry_offset = (entries_start - 0x18) as u32;
    let term_entry: Vec<u8> = match terminator_child {
        Some(child) => {
            let mut e = vec![0u8; 24];
            e[8..10].copy_from_slice(&24u16.to_le_bytes());
            e[12..16].copy_from_slice(&0x03u32.to_le_bytes()); // HAS_CHILD | LAST
            e[16..24].copy_from_slice(&child.to_le_bytes());
            e
        }
        None => {
            let mut e = vec![0u8; 16];
            e[8..10].copy_from_slice(&16u16.to_le_bytes());
            e[12..16].copy_from_slice(&0x02u32.to_le_bytes()); // LAST
            e
        }
    };
    let entries_total: usize = entries.iter().map(|e| e.len()).sum::<usize>() + term_entry.len();
    if entries_start + entries_total > block_size {
        return Err(crate::Error::Unsupported(
            "ntfs: INDX node entries overflow the block".into(),
        ));
    }
    let bytes_in_use = (entries_start - 0x18) as u32 + entries_total as u32;
    let bytes_allocated = (block_size - 0x18) as u32;
    buf[0x24] = if terminator_child.is_some() { 0x01 } else { 0 };
    buf[0x18..0x1C].copy_from_slice(&first_entry_offset.to_le_bytes());
    buf[0x1C..0x20].copy_from_slice(&bytes_in_use.to_le_bytes());
    buf[0x20..0x24].copy_from_slice(&bytes_allocated.to_le_bytes());
    let mut cursor = entries_start;
    for e in entries {
        buf[cursor..cursor + e.len()].copy_from_slice(e);
        cursor += e.len();
    }
    buf[cursor..cursor + term_entry.len()].copy_from_slice(&term_entry);
    mft::install_fixup(&mut buf, sector_size, 1);
    Ok(buf)
}

/// One pending B-tree item during bulk-load: the raw $I30 index entry plus
/// the VCN of the subtree sorting to its left (`None` at the leaf level).
#[derive(Clone)]
struct BtreeItem {
    raw: Vec<u8>,
    left_child: Option<u64>,
}

/// Index-entry bytes this item contributes: a leaf entry verbatim, or an
/// internal entry re-encoded with its left-child VCN appended (HAS_CHILD).
fn btree_item_entry(it: &BtreeItem) -> Vec<u8> {
    match it.left_child {
        None => it.raw.clone(),
        Some(child) => {
            let file_ref = u64::from_le_bytes(it.raw[0..8].try_into().unwrap());
            let key_len = u16::from_le_bytes(it.raw[10..12].try_into().unwrap()) as usize;
            build_index_entry(file_ref, &it.raw[16..16 + key_len], 0, Some(child))
        }
    }
}

/// One serialized INDX block: its VCN within `$INDEX_ALLOCATION` and the
/// block bytes (with USA fixup already installed).
type IndxBlock = (u64, Vec<u8>);

/// Bulk-load a balanced NTFS `$I30` B-tree from a sorted entry set. Splits
/// entries across leaf INDX blocks; between two siblings a separator entry
/// is pulled up into the parent (NTFS keeps real entries in internal
/// nodes), carrying the left sibling's VCN; the parent's terminator points
/// at the rightmost child. Repeats level by level until one node remains —
/// the root. Returns `(root_vcn, [(vcn, block_bytes)])` (fixups installed).
fn build_index_btree(
    entries: &[Vec<u8>],
    block_size: usize,
    sector_size: usize,
) -> Result<(u64, Vec<IndxBlock>)> {
    let sectors = block_size / sector_size;
    let usa_end = 0x28 + 2 * (sectors + 1);
    let entries_start = (usa_end + 7) & !7;

    let mut out: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut next_vcn = 0u64;
    let mut items: Vec<BtreeItem> = entries
        .iter()
        .map(|e| BtreeItem {
            raw: e.clone(),
            left_child: None,
        })
        .collect();
    let mut rightmost: Option<u64> = None;

    loop {
        let internal = items.first().is_some_and(|i| i.left_child.is_some());
        let term_len = if internal { 24 } else { 16 };
        let usable = block_size - entries_start - term_len;

        let build_node = |vcn: u64, group: &[BtreeItem], term_child: Option<u64>| {
            let encoded: Vec<Vec<u8>> = group.iter().map(btree_item_entry).collect();
            build_indx_block_full(block_size, sector_size, vcn, &encoded, term_child)
        };

        let mut parent: Vec<BtreeItem> = Vec::new();
        let mut cur: Vec<BtreeItem> = Vec::new();
        let mut cur_bytes = 0usize;
        let mut i = 0;
        while i < items.len() {
            let elen = btree_item_entry(&items[i]).len();
            if elen > usable {
                return Err(crate::Error::Unsupported(
                    "ntfs: a single directory entry is too large for an INDX block".into(),
                ));
            }
            if !cur.is_empty() && cur_bytes + elen > usable {
                let vcn = next_vcn;
                next_vcn += 1;
                out.push((vcn, build_node(vcn, &cur, items[i].left_child)?));
                parent.push(BtreeItem {
                    raw: items[i].raw.clone(),
                    left_child: Some(vcn),
                });
                cur.clear();
                cur_bytes = 0;
                i += 1; // separator pulled up
            } else {
                cur_bytes += elen;
                cur.push(items[i].clone());
                i += 1;
            }
        }
        let vcn = next_vcn;
        next_vcn += 1;
        out.push((vcn, build_node(vcn, &cur, rightmost)?));
        if parent.is_empty() {
            return Ok((vcn, out));
        }
        items = parent;
        rightmost = Some(vcn);
    }
}

/// Resident `$BITMAP` value marking the first `n` INDX blocks (VCNs) in
/// use, rounded up to an 8-byte multiple.
fn build_index_bitmap(n: u64) -> Vec<u8> {
    let nbytes = ((n as usize).div_ceil(8)).max(1);
    let nbytes = (nbytes + 7) & !7;
    let mut v = vec![0u8; nbytes];
    for i in 0..n as usize {
        v[i / 8] |= 1u8 << (i % 8);
    }
    v
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
