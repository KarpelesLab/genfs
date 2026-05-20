//! NTFS format-time layout: emit boot sector, $MFT, $MFTMirr, and all the
//! standard system files (records 0..15) for a fresh volume.
//!
//! ## Layout choices for fstool's writer
//!
//! We pick a conservative on-disk layout aimed at maximising compatibility
//! with ntfs-3g / kernel NTFS3 / chkdsk on Windows:
//!
//! * 512-byte sectors, **4 KiB clusters** (sectors_per_cluster = 8).
//! * **1024-byte MFT records** (`clusters_per_mft_record = -10`).
//! * **4096-byte index records** (`clusters_per_index_record = -12`).
//! * `$MFT` at LCN 4 (matches Microsoft's `format /fs:ntfs` default for
//!   small volumes).
//! * `$MFTMirr` at the volume midpoint cluster (mirrors records 0..3).
//! * `$Bitmap`, `$LogFile`, `$UpCase` etc. allocated dynamically by the
//!   cluster allocator and pointed at via non-resident attribute runs.
//! * Backup boot sector at the last LBA, per NT spec.
//!
//! The format emits **records 0..15** by name (with their well-known
//! contents), then leaves records 16+ free for user files / dirs that
//! `super::writer::WriterState` allocates on demand. Record 11 (`$Extend`) is
//! an empty directory; we do not populate `$ObjId / $Quota / $Reparse /
//! $UsnJrnl` — Windows will treat that as a v3.0 volume.
//!
//! After the system records are written, `Ntfs::format` stages index
//! entries for records 0..=15 (minus the root itself) into the root
//! directory's `$I30`. That matches how real-world NTFS volumes are laid
//! out — `ntfs-3g` and chkdsk walk `$I30` to discover the system files
//! even though most users never list them directly.
//!
//! The internal helper objects defined here are also used by
//! `super::writer::WriterState` for runtime mutations.

use crate::Result;
use crate::block::BlockDevice;

use super::attribute::{
    TYPE_BITMAP, TYPE_DATA, TYPE_FILE_NAME, TYPE_INDEX_ROOT, TYPE_STANDARD_INFORMATION,
    TYPE_VOLUME_NAME,
};
use super::boot::NTFS_OEM;
use super::mft;
use super::secure;
use super::upcase_gen::build_upcase_blob;

/// Hard-coded MFT record numbers (matches what the read path expects).
pub const REC_MFT: u64 = 0;
pub const REC_MFTMIRR: u64 = 1;
pub const REC_LOGFILE: u64 = 2;
pub const REC_VOLUME: u64 = 3;
pub const REC_ATTRDEF: u64 = 4;
pub const REC_ROOT: u64 = 5;
pub const REC_BITMAP: u64 = 6;
pub const REC_BOOT: u64 = 7;
pub const REC_BADCLUS: u64 = 8;
pub const REC_SECURE: u64 = 9;
pub const REC_UPCASE: u64 = 10;
pub const REC_EXTEND: u64 = 11;

/// First MFT record number available for user files / directories.
pub const FIRST_USER_RECORD: u64 = 16;

/// Default MFT record size in bytes.
pub const DEFAULT_MFT_RECORD_SIZE: u32 = 1024;
/// Default index record size in bytes.
pub const DEFAULT_INDEX_RECORD_SIZE: u32 = 4096;
/// Default cluster size in bytes (sectors_per_cluster = 8 × 512 BPS).
pub const DEFAULT_CLUSTER_SIZE: u32 = 4096;
/// Initial $MFT size in clusters (16 records × 1024 = 16 KiB → 4 clusters).
pub const INITIAL_MFT_CLUSTERS: u64 = 16;
/// $LogFile minimum size (NTFS spec demands ≥ 1 MiB; we emit a clean-shutdown
/// log of 64 KiB which kernel NTFS3 accepts as "empty / not replayable").
pub const LOGFILE_BYTES: u64 = 64 * 1024;

/// Options controlling a fresh NTFS format.
#[derive(Debug, Clone)]
pub struct FormatOpts {
    /// Logical sector size. Default 512.
    pub bytes_per_sector: u16,
    /// Sectors per cluster. Default 8 (→ 4 KiB clusters).
    pub sectors_per_cluster: u8,
    /// Volume name (becomes `$Volume:$VOLUME_NAME`). Empty = unnamed.
    pub volume_label: String,
    /// 64-bit volume serial number. Random by default.
    pub volume_serial: u64,
}

impl Default for FormatOpts {
    fn default() -> Self {
        Self {
            bytes_per_sector: 512,
            sectors_per_cluster: 8,
            volume_label: String::new(),
            volume_serial: random_serial(),
        }
    }
}

/// Produce a non-zero 64-bit value from the running clock — used as a
/// volume serial number. We avoid pulling `uuid` here because the bytes
/// don't need to be a UUID, just unique-per-format.
fn random_serial() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xDEAD_BEEF_CAFE_BABE);
    // XOR a thread address in for some entropy when called rapidly.
    let mix = &now as *const _ as usize as u64;
    now ^ mix.rotate_left(11)
}

/// In-memory cluster allocator used by both format-time layout and the
/// runtime writer. Backed by a byte slice that maps 1 bit per cluster.
#[derive(Debug, Clone)]
pub struct BitmapAlloc {
    /// LSB-first packed bitmap: bit `n` is cluster `n`.
    pub bytes: Vec<u8>,
    /// Total clusters in the volume.
    pub total: u64,
    /// Next cluster to consider when looking for a free run.
    pub next_hint: u64,
}

impl BitmapAlloc {
    pub fn new(total_clusters: u64) -> Self {
        let bytes = vec![0u8; total_clusters.div_ceil(8) as usize];
        Self {
            bytes,
            total: total_clusters,
            next_hint: 0,
        }
    }

    pub fn is_set(&self, cluster: u64) -> bool {
        if cluster >= self.total {
            return true;
        }
        let i = (cluster / 8) as usize;
        let m = 1u8 << ((cluster % 8) as u8);
        self.bytes[i] & m != 0
    }

    pub fn set(&mut self, cluster: u64) {
        if cluster < self.total {
            let i = (cluster / 8) as usize;
            let m = 1u8 << ((cluster % 8) as u8);
            self.bytes[i] |= m;
        }
    }

    pub fn clear(&mut self, cluster: u64) {
        if cluster < self.total {
            let i = (cluster / 8) as usize;
            let m = 1u8 << ((cluster % 8) as u8);
            self.bytes[i] &= !m;
        }
    }

    /// Mark a contiguous range allocated.
    pub fn set_range(&mut self, start: u64, length: u64) {
        for c in start..start.saturating_add(length).min(self.total) {
            self.set(c);
        }
    }

    /// Allocate `count` contiguous clusters. Falls back to scanning from the
    /// start if the hint-based search comes up empty. Returns the starting
    /// cluster on success.
    pub fn allocate(&mut self, count: u64) -> Result<u64> {
        if count == 0 {
            return Ok(self.next_hint);
        }
        let starts = [self.next_hint, 0];
        for &start in &starts {
            let mut run_start: Option<u64> = None;
            let mut run_len = 0u64;
            for c in start..self.total {
                if !self.is_set(c) {
                    if run_start.is_none() {
                        run_start = Some(c);
                        run_len = 0;
                    }
                    run_len += 1;
                    if run_len >= count {
                        let s = run_start.unwrap();
                        self.set_range(s, count);
                        self.next_hint = s + count;
                        return Ok(s);
                    }
                } else {
                    run_start = None;
                    run_len = 0;
                }
            }
        }
        Err(crate::Error::InvalidImage(
            "ntfs: out of free clusters during allocation".into(),
        ))
    }
}

/// Build the boot sector for a freshly formatted NTFS volume. `mft_lcn`
/// and `mftmirr_lcn` are the starting clusters of $MFT and $MFTMirr.
pub fn build_boot_sector(
    opts: &FormatOpts,
    total_sectors: u64,
    mft_lcn: u64,
    mftmirr_lcn: u64,
    mft_record_field: i8,
    index_record_field: i8,
) -> Vec<u8> {
    let mut b = vec![0u8; opts.bytes_per_sector as usize];
    // Jump
    b[0..3].copy_from_slice(&[0xEB, 0x52, 0x90]);
    // OEM ID
    b[3..11].copy_from_slice(NTFS_OEM);
    // BPB
    b[11..13].copy_from_slice(&opts.bytes_per_sector.to_le_bytes());
    b[13] = opts.sectors_per_cluster;
    // Reserved sectors = 0
    b[14..16].copy_from_slice(&0u16.to_le_bytes());
    // FATs = 0, root entries = 0, total sectors16 = 0, media = F8
    b[21] = 0xF8;
    // Sectors per track / heads / hidden sectors: harmless filler.
    b[24..26].copy_from_slice(&63u16.to_le_bytes());
    b[26..28].copy_from_slice(&255u16.to_le_bytes());
    // Total sectors (-1 of volume sectors per NTFS convention).
    let bpb_total = total_sectors.saturating_sub(1);
    b[0x28..0x30].copy_from_slice(&bpb_total.to_le_bytes());
    b[0x30..0x38].copy_from_slice(&mft_lcn.to_le_bytes());
    b[0x38..0x40].copy_from_slice(&mftmirr_lcn.to_le_bytes());
    b[0x40] = mft_record_field as u8;
    b[0x44] = index_record_field as u8;
    b[0x48..0x50].copy_from_slice(&opts.volume_serial.to_le_bytes());
    // checksum (0) at 0x50..0x54
    // Boot signature
    b[510] = 0x55;
    b[511] = 0xAA;
    b
}

/// Build the canonical `$AttrDef` payload. `$AttrDef` is a series of
/// 160-byte records, each describing one attribute type. We emit the
/// well-known table verbatim — Windows is happy as long as the layout
/// and the per-row sizes match the spec.
pub fn build_attrdef_payload() -> Vec<u8> {
    // Each entry is 160 bytes: name (128 = 64 UTF-16 chars) + type (4) +
    // display rule (4) + collation (4) + flags (4) + min_size (8) + max_size (8).
    #[allow(clippy::type_complexity)]
    let entries: &[(&str, u32, u32, u32, u32, u64, u64)] = &[
        ("$STANDARD_INFORMATION", 0x10, 0, 0, 0x40, 48, 72),
        ("$ATTRIBUTE_LIST", 0x20, 0, 0, 0x40, 0, u64::MAX),
        ("$FILE_NAME", 0x30, 0, 2, 0x42, 68, 578),
        ("$OBJECT_ID", 0x40, 0, 0, 0x40, 0, 256),
        ("$SECURITY_DESCRIPTOR", 0x50, 0, 0, 0x40, 0, u64::MAX),
        ("$VOLUME_NAME", 0x60, 0, 0, 0x40, 2, 256),
        ("$VOLUME_INFORMATION", 0x70, 0, 0, 0x40, 12, 12),
        ("$DATA", 0x80, 0, 0, 0, 0, u64::MAX),
        ("$INDEX_ROOT", 0x90, 0, 0, 0x40, 0, u64::MAX),
        ("$INDEX_ALLOCATION", 0xA0, 0, 0, 0, 0, u64::MAX),
        ("$BITMAP", 0xB0, 0, 0, 0, 0, u64::MAX),
        ("$REPARSE_POINT", 0xC0, 0, 0, 0x40, 0, 16384),
        ("$EA_INFORMATION", 0xD0, 0, 0, 0x40, 8, 8),
        ("$EA", 0xE0, 0, 0, 0, 0, 65536),
        ("$LOGGED_UTILITY_STREAM", 0x100, 0, 0, 0x40, 0, 65536),
    ];
    let mut out = Vec::with_capacity(entries.len() * 160);
    for (name, type_code, dr, collation, flags, min_size, max_size) in entries {
        let mut rec = vec![0u8; 160];
        // Name as UTF-16LE, up to 64 code units, NUL-terminated within 128 bytes.
        let mut idx = 0usize;
        for u in name.encode_utf16() {
            if idx + 2 > 128 {
                break;
            }
            rec[idx..idx + 2].copy_from_slice(&u.to_le_bytes());
            idx += 2;
        }
        rec[128..132].copy_from_slice(&type_code.to_le_bytes());
        rec[132..136].copy_from_slice(&dr.to_le_bytes());
        rec[136..140].copy_from_slice(&collation.to_le_bytes());
        rec[140..144].copy_from_slice(&flags.to_le_bytes());
        rec[144..152].copy_from_slice(&min_size.to_le_bytes());
        rec[152..160].copy_from_slice(&max_size.to_le_bytes());
        out.extend_from_slice(&rec);
    }
    out
}

/// Emit a fresh `FILE` record into `rec_buf`, sized at `rec_size`, with
/// the given flags and concatenated `attrs`. Also installs the USA fixup.
pub fn emit_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    mft_record_number: u64,
    flags: u16,
    attrs: &[Vec<u8>],
    sector_size: usize,
    usn: u16,
) {
    // Zero out
    for b in rec_buf.iter_mut() {
        *b = 0;
    }
    let rec = &mut rec_buf[..rec_size];

    rec[0..4].copy_from_slice(b"FILE");
    // usa_offset = 0x30 (canonical post-Win8 layout; works everywhere)
    let usa_offset: u16 = 0x30;
    rec[4..6].copy_from_slice(&usa_offset.to_le_bytes());
    let sectors = rec_size / sector_size;
    let usa_size = (sectors + 1) as u16;
    rec[6..8].copy_from_slice(&usa_size.to_le_bytes());
    // log file sequence number = 0
    // sequence number = 1 (must be non-zero so file_ref's sequence != 0)
    rec[0x10..0x12].copy_from_slice(&1u16.to_le_bytes());
    // hard link count = 1
    rec[0x12..0x14].copy_from_slice(&1u16.to_le_bytes());

    // first attribute offset (aligned past USA)
    let usa_end = usa_offset as usize + usa_size as usize * 2;
    let first_attr_off = ((usa_end + 7) & !7) as u16;
    rec[0x14..0x16].copy_from_slice(&first_attr_off.to_le_bytes());
    rec[0x16..0x18].copy_from_slice(&flags.to_le_bytes());

    // mft record number (Windows 10+ extension)
    rec[0x2C..0x30].copy_from_slice(&(mft_record_number as u32).to_le_bytes());

    let mut cursor = first_attr_off as usize;
    let mut next_attr_id: u16 = 1;
    for a in attrs {
        rec[cursor..cursor + a.len()].copy_from_slice(a);
        // Stamp attribute id (offset 14..16) — every attr gets a unique id.
        rec[cursor + 14..cursor + 16].copy_from_slice(&next_attr_id.to_le_bytes());
        next_attr_id = next_attr_id.wrapping_add(1);
        cursor += a.len();
    }
    let term = [0xFFu8, 0xFF, 0xFF, 0xFF];
    rec[cursor..cursor + 4].copy_from_slice(&term);
    cursor += 4;
    // next attribute id field at offset 0x28
    rec[0x28..0x2A].copy_from_slice(&next_attr_id.to_le_bytes());

    let bytes_in_use = cursor as u32;
    rec[0x18..0x1C].copy_from_slice(&bytes_in_use.to_le_bytes());
    rec[0x1C..0x20].copy_from_slice(&(rec_size as u32).to_le_bytes());

    // Install fixup last.
    mft::install_fixup(rec, sector_size, usn);
}

/// Build a resident attribute (header + resident-specific fields + value).
/// The attribute id at offset 14 is left as 0 — [`emit_record`] stamps it
/// with a unique id per record.
pub fn build_resident_attr(
    type_code: u32,
    name_utf16: &[u8],
    value: &[u8],
    flags: u16,
    indexed_flag: u8,
) -> Vec<u8> {
    let name_len_u16 = (name_utf16.len() / 2) as u8;
    let name_off: u16 = if name_utf16.is_empty() { 0 } else { 0x18 };
    let header_block_len = 0x18 + name_utf16.len();
    let header_block_aligned = (header_block_len + 7) & !7;
    let value_offset = header_block_aligned as u16;
    let total = (header_block_aligned + value.len() + 7) & !7;

    let mut hdr = vec![0u8; 16];
    hdr[0..4].copy_from_slice(&type_code.to_le_bytes());
    hdr[4..8].copy_from_slice(&(total as u32).to_le_bytes());
    hdr[8] = 0; // resident
    hdr[9] = name_len_u16;
    hdr[10..12].copy_from_slice(&name_off.to_le_bytes());
    hdr[12..14].copy_from_slice(&flags.to_le_bytes());
    // attr id stamped by emit_record
    let mut resident = vec![0u8; 8];
    resident[0..4].copy_from_slice(&(value.len() as u32).to_le_bytes());
    resident[4..6].copy_from_slice(&value_offset.to_le_bytes());
    resident[6] = indexed_flag;
    hdr.extend_from_slice(&resident);

    let mut buf = hdr;
    if !name_utf16.is_empty() {
        buf.extend_from_slice(name_utf16);
    }
    while buf.len() < header_block_aligned {
        buf.push(0);
    }
    buf.extend_from_slice(value);
    while buf.len() < total {
        buf.push(0);
    }
    buf
}

/// Build a non-resident attribute with the given mapping-pairs blob.
#[allow(clippy::too_many_arguments)]
pub fn build_non_resident_attr(
    type_code: u32,
    name_utf16: &[u8],
    runs: &[u8],
    starting_vcn: u64,
    last_vcn: u64,
    allocated: u64,
    real: u64,
    initialized: u64,
    flags: u16,
    compression_unit: u8,
) -> Vec<u8> {
    let name_len_u16 = (name_utf16.len() / 2) as u8;
    let header_block_len = 0x40 + name_utf16.len();
    let header_block_aligned = (header_block_len + 7) & !7;
    let runs_off = header_block_aligned as u16;
    let total = ((header_block_aligned + runs.len()) + 7) & !7;

    let name_off: u16 = if name_utf16.is_empty() { 0 } else { 0x40 };
    let mut hdr = vec![0u8; 16];
    hdr[0..4].copy_from_slice(&type_code.to_le_bytes());
    hdr[4..8].copy_from_slice(&(total as u32).to_le_bytes());
    hdr[8] = 1; // non-resident
    hdr[9] = name_len_u16;
    hdr[10..12].copy_from_slice(&name_off.to_le_bytes());
    hdr[12..14].copy_from_slice(&flags.to_le_bytes());
    let mut nonresident = vec![0u8; 0x30];
    nonresident[0x00..0x08].copy_from_slice(&starting_vcn.to_le_bytes());
    nonresident[0x08..0x10].copy_from_slice(&last_vcn.to_le_bytes());
    nonresident[0x10..0x12].copy_from_slice(&runs_off.to_le_bytes());
    nonresident[0x12..0x13].copy_from_slice(&[compression_unit]);
    nonresident[0x18..0x20].copy_from_slice(&allocated.to_le_bytes());
    nonresident[0x20..0x28].copy_from_slice(&real.to_le_bytes());
    nonresident[0x28..0x30].copy_from_slice(&initialized.to_le_bytes());
    hdr.extend_from_slice(&nonresident);
    let mut buf = hdr;
    if !name_utf16.is_empty() {
        buf.extend_from_slice(name_utf16);
    }
    while buf.len() < header_block_aligned {
        buf.push(0);
    }
    buf.extend_from_slice(runs);
    while buf.len() < total {
        buf.push(0);
    }
    buf
}

/// Encode a run list (mapping pairs) for a single contiguous extent at
/// `start_lcn` of length `length` clusters. Returns the bytes (followed
/// by the 0x00 terminator).
pub fn encode_single_run(start_lcn: u64, length: u64) -> Vec<u8> {
    let len_size = min_unsigned_bytes(length);
    let off_size = min_signed_bytes(start_lcn as i64);
    let mut out = Vec::with_capacity(1 + len_size + off_size + 1);
    out.push(((off_size as u8) << 4) | (len_size as u8));
    out.extend_from_slice(&length.to_le_bytes()[..len_size]);
    let lcn_bytes = start_lcn.to_le_bytes();
    out.extend_from_slice(&lcn_bytes[..off_size]);
    out.push(0); // terminator
    out
}

/// Encode a chain of contiguous extents starting at successive LCNs. The
/// caller passes (lcn, length) pairs in order; deltas are computed between
/// successive entries.
pub fn encode_run_list(extents: &[(u64, u64)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut prev_lcn: i64 = 0;
    for &(lcn, length) in extents {
        let len_size = min_unsigned_bytes(length);
        let delta = (lcn as i64) - prev_lcn;
        let off_size = min_signed_bytes(delta);
        out.push(((off_size as u8) << 4) | (len_size as u8));
        out.extend_from_slice(&length.to_le_bytes()[..len_size]);
        let delta_bytes = delta.to_le_bytes();
        out.extend_from_slice(&delta_bytes[..off_size]);
        prev_lcn = lcn as i64;
    }
    out.push(0);
    out
}

fn min_unsigned_bytes(v: u64) -> usize {
    if v == 0 {
        return 1;
    }
    let mut n = 0usize;
    let mut x = v;
    while x != 0 {
        n += 1;
        x >>= 8;
    }
    n
}

fn min_signed_bytes(v: i64) -> usize {
    if v == 0 {
        return 1;
    }
    let mut n = 1usize;
    while n < 8 {
        let bits = (n as u32) * 8;
        let lo = -(1i64 << (bits - 1));
        let hi = (1i64 << (bits - 1)) - 1;
        if v >= lo && v <= hi {
            return n;
        }
        n += 1;
    }
    8
}

/// Build an empty `$INDEX_ROOT` value with name `$I30` indexed by
/// `$FILE_NAME`. The root carries only a "terminator" entry (no real
/// children) so a directory created this way is initially empty.
pub fn build_empty_index_root() -> Vec<u8> {
    let index_block_size = DEFAULT_INDEX_RECORD_SIZE;
    // Cpib: bytes-per-index-block encoded the same way as MFT record size.
    // Positive: clusters; negative: 1<<(-v). With 4 KiB clusters / 4 KiB
    // index blocks, value = 1.
    let cpib: i8 = 1;
    let mut v = Vec::with_capacity(0x20);
    v.extend_from_slice(&TYPE_FILE_NAME.to_le_bytes());
    v.extend_from_slice(&1u32.to_le_bytes()); // collation = filename
    v.extend_from_slice(&index_block_size.to_le_bytes());
    v.push(cpib as u8);
    v.extend_from_slice(&[0u8; 3]);
    // Index header at offset 16
    let first_entry_offset = 16u32;
    let term_entry_len = 16u32;
    let bytes_in_use = 16u32 + term_entry_len;
    let bytes_allocated = bytes_in_use;
    let flags: u8 = 0; // SMALL_INDEX
    v.extend_from_slice(&first_entry_offset.to_le_bytes());
    v.extend_from_slice(&bytes_in_use.to_le_bytes());
    v.extend_from_slice(&bytes_allocated.to_le_bytes());
    v.push(flags);
    v.extend_from_slice(&[0u8; 3]);
    // Terminator entry: file_ref=0, entry_len=16, key_len=0, flags=LAST(0x02).
    let mut term = vec![0u8; 16];
    term[0..8].copy_from_slice(&0u64.to_le_bytes());
    term[8..10].copy_from_slice(&16u16.to_le_bytes());
    term[10..12].copy_from_slice(&0u16.to_le_bytes());
    term[12..16].copy_from_slice(&0x02u32.to_le_bytes());
    v.extend_from_slice(&term);
    v
}

/// Build a default `$STANDARD_INFORMATION` value (48 bytes). Times default
/// to the supplied FILETIME; file_attributes default to `attrs`. Equivalent
/// to [`build_si_value_with_security`] with a zero `security_id`.
pub fn build_si_value(filetime: u64, attrs: u32) -> Vec<u8> {
    build_si_value_with_security(filetime, attrs, 0)
}

/// Convenience wrapper around [`build_si_value_with_security`] for an
/// NTFS system record (MFT records 0..=15). Stamps the canonical
/// "System" security_id resolved through the format-time SD catalogue.
pub fn build_system_si_value(filetime: u64, attrs: u32) -> Vec<u8> {
    build_si_value_with_security(
        filetime,
        attrs,
        security_id_for(secure::SecurityClass::System),
    )
}

/// Build a `$STANDARD_INFORMATION` value, optionally with the NTFS 3.0+
/// extended footer carrying `security_id`. A zero `security_id` produces
/// the legacy 48-byte form (matches a freshly formatted v1.2 volume).
/// Non-zero ids produce a 72-byte value with the standard layout:
///
/// ```text
///   0x00..0x08  created  (FILETIME)
///   0x08..0x10  modified (FILETIME)
///   0x10..0x18  mft_changed (FILETIME)
///   0x18..0x20  accessed (FILETIME)
///   0x20..0x24  file_attributes
///   0x24..0x30  reserved (max_versions / version / class_id)
///   0x30..0x34  owner_id    (NTFS 3.0+)
///   0x34..0x38  security_id (NTFS 3.0+)
///   0x38..0x40  quota_charged (NTFS 3.0+)
///   0x40..0x48  usn  (NTFS 3.0+)
/// ```
pub fn build_si_value_with_security(filetime: u64, attrs: u32, security_id: u32) -> Vec<u8> {
    let len = if security_id != 0 { 72 } else { 48 };
    let mut v = vec![0u8; len];
    v[0..8].copy_from_slice(&filetime.to_le_bytes());
    v[8..16].copy_from_slice(&filetime.to_le_bytes());
    v[16..24].copy_from_slice(&filetime.to_le_bytes());
    v[24..32].copy_from_slice(&filetime.to_le_bytes());
    v[32..36].copy_from_slice(&attrs.to_le_bytes());
    if security_id != 0 {
        // security_id at 0x34..0x38
        v[0x34..0x38].copy_from_slice(&security_id.to_le_bytes());
    }
    v
}

/// Build a `$FILE_NAME` value pointing at parent record `parent_ref`
/// (low-48 bits + sequence in high 16 bits as packed by NTFS), with the
/// given `name` and flags / size.
pub fn build_file_name_value(
    parent_ref: u64,
    name: &str,
    flags: u32,
    real_size: u64,
    allocated_size: u64,
    filetime: u64,
    namespace: u8,
) -> Vec<u8> {
    let name_utf16: Vec<u8> = name.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    let mut v = vec![0u8; 66 + name_utf16.len()];
    v[0..8].copy_from_slice(&parent_ref.to_le_bytes());
    v[8..16].copy_from_slice(&filetime.to_le_bytes());
    v[16..24].copy_from_slice(&filetime.to_le_bytes());
    v[24..32].copy_from_slice(&filetime.to_le_bytes());
    v[32..40].copy_from_slice(&filetime.to_le_bytes());
    v[40..48].copy_from_slice(&allocated_size.to_le_bytes());
    v[48..56].copy_from_slice(&real_size.to_le_bytes());
    v[56..60].copy_from_slice(&flags.to_le_bytes());
    v[64] = (name_utf16.len() / 2) as u8;
    v[65] = namespace;
    v[66..].copy_from_slice(&name_utf16);
    v
}

/// Pack an MFT reference (low-48 record number + high-16 sequence).
pub fn pack_mft_ref(rec_no: u64, sequence: u16) -> u64 {
    (rec_no & 0x0000_FFFF_FFFF_FFFF) | ((sequence as u64) << 48)
}

/// Build an `$AttrDef` record (record 4). Carries a non-resident $DATA
/// holding `attrdef_bytes`, which must already be cluster-padded.
#[allow(clippy::too_many_arguments)]
pub fn build_attrdef_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    rec_no: u64,
    parent_ref: u64,
    attrdef_size: u64,
    attrdef_lcn: u64,
    attrdef_clusters: u64,
    filetime: u64,
    cluster_size: u64,
    sector_size: usize,
    name: &str,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, 0x06),
        0,
        0,
    );
    let fn_value = build_file_name_value(
        parent_ref,
        name,
        0x06, // hidden + system
        attrdef_size,
        attrdef_clusters * cluster_size,
        filetime,
        1, // Win32
    );
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    let data_runs = encode_single_run(attrdef_lcn, attrdef_clusters);
    let data = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &data_runs,
        0,
        attrdef_clusters - 1,
        attrdef_clusters * cluster_size,
        attrdef_size,
        attrdef_size,
        0,
        0,
    );
    emit_record(
        rec_buf,
        rec_size,
        rec_no,
        mft::RecordHeader::FLAG_IN_USE,
        &[si, fname, data],
        sector_size,
        1,
    );
}

/// Volume information attribute: 12-byte value = reserved (8) + major + minor + flags.
const TYPE_VOLUME_INFORMATION: u32 = 0x70;
pub fn build_volume_information() -> Vec<u8> {
    let mut v = vec![0u8; 12];
    // NTFS v3.1
    v[8] = 3;
    v[9] = 1;
    v
}

/// Build the $Volume record (record 3) — carries $VOLUME_NAME (UTF-16LE
/// volume label) and $VOLUME_INFORMATION (NTFS version).
pub fn build_volume_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    parent_ref: u64,
    label: &str,
    filetime: u64,
    sector_size: usize,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, 0x06),
        0,
        0,
    );
    let fn_value = build_file_name_value(parent_ref, "$Volume", 0x06, 0, 0, filetime, 1);
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    let label_utf16: Vec<u8> = label.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    let vol_name = build_resident_attr(TYPE_VOLUME_NAME, &[], &label_utf16, 0, 0);
    let vol_info = build_resident_attr(
        TYPE_VOLUME_INFORMATION,
        &[],
        &build_volume_information(),
        0,
        0,
    );
    emit_record(
        rec_buf,
        rec_size,
        REC_VOLUME,
        mft::RecordHeader::FLAG_IN_USE,
        &[si, fname, vol_name, vol_info],
        sector_size,
        1,
    );
}

/// Build the root directory's MFT record (record 5). The index is empty
/// initially — `Writer::add_entry_to_dir` mutates it as files are added.
pub fn build_root_record(rec_buf: &mut [u8], rec_size: usize, filetime: u64, sector_size: usize) {
    // Root carries the User-class SD (everyone full access) — it is the
    // user-visible top-level directory, not a system file.
    let root_si =
        build_si_value_with_security(filetime, 0x06, security_id_for(secure::SecurityClass::User));
    let si = build_resident_attr(TYPE_STANDARD_INFORMATION, &[], &root_si, 0, 0);
    // The root has a $FILE_NAME whose parent is itself.
    let root_ref = pack_mft_ref(REC_ROOT, 1);
    let fn_value = build_file_name_value(root_ref, ".", 0x10000006, 0, 0, filetime, 1); // dir | hidden | system
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    let i30_name: Vec<u8> = "$I30"
        .encode_utf16()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    let idx_root = build_resident_attr(TYPE_INDEX_ROOT, &i30_name, &build_empty_index_root(), 0, 0);
    emit_record(
        rec_buf,
        rec_size,
        REC_ROOT,
        mft::RecordHeader::FLAG_IN_USE | mft::RecordHeader::FLAG_DIRECTORY,
        &[si, fname, idx_root],
        sector_size,
        1,
    );
}

/// Build the $Bitmap record (record 6) with non-resident $DATA pointing at
/// the on-disk cluster bitmap.
#[allow(clippy::too_many_arguments)]
pub fn build_bitmap_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    parent_ref: u64,
    bitmap_bytes: u64,
    bitmap_lcn: u64,
    bitmap_clusters: u64,
    filetime: u64,
    cluster_size: u64,
    sector_size: usize,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, 0x06),
        0,
        0,
    );
    let fn_value = build_file_name_value(
        parent_ref,
        "$Bitmap",
        0x06,
        bitmap_bytes,
        bitmap_clusters * cluster_size,
        filetime,
        1,
    );
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    let runs = encode_single_run(bitmap_lcn, bitmap_clusters);
    let data = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &runs,
        0,
        bitmap_clusters - 1,
        bitmap_clusters * cluster_size,
        bitmap_bytes,
        bitmap_bytes,
        0,
        0,
    );
    emit_record(
        rec_buf,
        rec_size,
        REC_BITMAP,
        mft::RecordHeader::FLAG_IN_USE,
        &[si, fname, data],
        sector_size,
        1,
    );
}

/// Build the $Boot record (record 7): non-resident $DATA covering LBA 0.
pub fn build_boot_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    parent_ref: u64,
    total_bytes: u64,
    filetime: u64,
    sector_size: usize,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, 0x06),
        0,
        0,
    );
    let fn_value = build_file_name_value(
        parent_ref,
        "$Boot",
        0x06,
        sector_size as u64,
        sector_size as u64,
        filetime,
        1,
    );
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    // $Boot's $DATA covers the boot sector (one cluster). LCN 0.
    let runs = encode_single_run(0, 1);
    let data = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &runs,
        0,
        0,
        sector_size as u64,
        sector_size as u64,
        sector_size as u64,
        0,
        0,
    );
    let _ = total_bytes; // currently informational
    emit_record(
        rec_buf,
        rec_size,
        REC_BOOT,
        mft::RecordHeader::FLAG_IN_USE,
        &[si, fname, data],
        sector_size,
        1,
    );
}

/// Build $BadClus (record 8): a sparse non-resident $DATA named "$Bad".
pub fn build_badclus_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    parent_ref: u64,
    total_clusters: u64,
    filetime: u64,
    cluster_size: u64,
    sector_size: usize,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, 0x06),
        0,
        0,
    );
    let fn_value = build_file_name_value(parent_ref, "$BadClus", 0x06, 0, 0, filetime, 1);
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    // Unnamed empty $DATA.
    let data_unnamed = build_resident_attr(TYPE_DATA, &[], &[], 0, 0);
    // Named $DATA "$Bad": sparse extent spanning the volume.
    let bad_name: Vec<u8> = "$Bad"
        .encode_utf16()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    // Sparse run: length-only header.
    let mut sparse_runs = Vec::new();
    let len_size = min_unsigned_bytes(total_clusters);
    sparse_runs.push(len_size as u8); // off_size = 0
    sparse_runs.extend_from_slice(&total_clusters.to_le_bytes()[..len_size]);
    sparse_runs.push(0); // terminator
    let bad_data = build_non_resident_attr(
        TYPE_DATA,
        &bad_name,
        &sparse_runs,
        0,
        total_clusters - 1,
        total_clusters * cluster_size,
        total_clusters * cluster_size,
        0, // initialized_size = 0 (sparse)
        super::attribute::ATTR_FLAG_SPARSE,
        0,
    );
    emit_record(
        rec_buf,
        rec_size,
        REC_BADCLUS,
        mft::RecordHeader::FLAG_IN_USE,
        &[si, fname, data_unnamed, bad_data],
        sector_size,
        1,
    );
}

/// File-attribute bits for `$Secure`: HIDDEN | SYSTEM | VIEW_INDEX.
/// `ntfs-3g` cross-checks `$STANDARD_INFORMATION.file_attributes` against
/// the MFT record's `VIEW_INDEX` flag on mount, so we keep the two in
/// sync at format time.
const SECURE_FILE_ATTRS: u32 = 0x2000_0006;
/// MFT record flag bit for "VIEW_INDEX" (the same flag that `$Secure`,
/// `$Extend/$ObjId`, etc. carry on real volumes). Combined with
/// `FLAG_IN_USE` it yields the canonical 0x0009 record header flags.
const MFT_RECORD_FLAG_VIEW_INDEX: u16 = 0x0008;
/// First security_id mkntfs hands out. We follow the same convention so
/// downstream tools that special-case the low ids behave identically.
pub const FIRST_SECURITY_ID: u32 = 0x100;

/// The security id assigned to a given class on a fresh volume. The
/// catalogue is fixed at format time (see `build_security_catalogue`)
/// so this is a small switch rather than a runtime lookup.
pub fn security_id_for(class: secure::SecurityClass) -> u32 {
    FIRST_SECURITY_ID + class.catalogue_index()
}

/// Build the catalogue of security descriptors a fresh volume carries.
/// Each entry is `(security_id, sd_blob)`; the formatter feeds this
/// list straight into [`build_sds_stream_multi`] / [`build_secure_record`].
///
/// We currently emit two distinct SDs:
/// * `0x100` — the **User** SD (Everyone Full Access). Default for
///   user-visible files / directories and for the root.
/// * `0x101` — the **System** SD (SYSTEM + Administrators Full Access,
///   no Everyone entry). Applied to records 0..=15 (`$MFT`, `$Secure`,
///   `$LogFile`, ...).
fn build_security_catalogue() -> Vec<(u32, Vec<u8>)> {
    let classes = [secure::SecurityClass::User, secure::SecurityClass::System];
    classes
        .iter()
        .map(|&c| (security_id_for(c), build_security_descriptor(c)))
        .collect()
}

/// Build a default "everyone full access" SECURITY_DESCRIPTOR_RELATIVE.
/// Owner = SYSTEM, Group = SYSTEM, DACL = single ACE granting Everyone
/// FILE_ALL_ACCESS. Equivalent to `build_security_descriptor(SecurityClass::User)`.
pub fn build_default_security_descriptor() -> Vec<u8> {
    build_security_descriptor(secure::SecurityClass::User)
}

/// Build a self-relative `SECURITY_DESCRIPTOR` blob for the given class.
/// Each class produces a distinct DACL so the resulting SDs hash to
/// different values and occupy independent slots in `$Secure:$SDS`.
///
/// All variants share Owner = Group = Local SYSTEM (S-1-5-18), revision 1,
/// `SE_DACL_PRESENT | SE_SELF_RELATIVE = 0x8004` control flags, and the
/// canonical header layout (DACL → owner → group). Only the DACL differs.
pub fn build_security_descriptor(class: secure::SecurityClass) -> Vec<u8> {
    // SID: S-1-1-0 (Everyone) — one sub-authority.
    let mut everyone = vec![0u8; 12];
    everyone[0] = 1;
    everyone[1] = 1;
    everyone[2..8].copy_from_slice(&[0, 0, 0, 0, 0, 1]);
    everyone[8..12].copy_from_slice(&0u32.to_le_bytes());

    // SID: S-1-5-18 (Local SYSTEM).
    let mut system = vec![0u8; 12];
    system[0] = 1;
    system[1] = 1;
    system[2..8].copy_from_slice(&[0, 0, 0, 0, 0, 5]);
    system[8..12].copy_from_slice(&18u32.to_le_bytes());

    // SID: S-1-5-32-544 (BUILTIN\Administrators) — two sub-authorities.
    let mut administrators = vec![0u8; 16];
    administrators[0] = 1;
    administrators[1] = 2;
    administrators[2..8].copy_from_slice(&[0, 0, 0, 0, 0, 5]);
    administrators[8..12].copy_from_slice(&32u32.to_le_bytes());
    administrators[12..16].copy_from_slice(&544u32.to_le_bytes());

    fn ace_allowed(mask: u32, sid: &[u8]) -> Vec<u8> {
        let mut ace = Vec::with_capacity(8 + sid.len());
        ace.push(0u8); // ACCESS_ALLOWED
        ace.push(0u8); // flags
        let ace_size = 8u16 + sid.len() as u16;
        ace.extend_from_slice(&ace_size.to_le_bytes());
        ace.extend_from_slice(&mask.to_le_bytes());
        ace.extend_from_slice(sid);
        ace
    }

    // FILE_ALL_ACCESS = 0x001F01FF.
    let all_access = 0x001F_01FFu32;

    // DACL ACEs depend on the class.
    let aces: Vec<Vec<u8>> = match class {
        // Default / User: Everyone gets full access. Matches mkntfs's
        // default DACL for unowned files and the legacy
        // `build_default_security_descriptor` blob.
        secure::SecurityClass::Default | secure::SecurityClass::User => {
            vec![ace_allowed(all_access, &everyone)]
        }
        // System: SYSTEM + Administrators get full access; Everyone has
        // no entries. This is the canonical DACL `mkntfs` applies to
        // records 0..=15.
        secure::SecurityClass::System => {
            vec![
                ace_allowed(all_access, &system),
                ace_allowed(all_access, &administrators),
            ]
        }
    };

    // ACL header + ACEs.
    let mut acl = Vec::new();
    acl.push(2u8); // revision
    acl.push(0u8); // sbz1
    let aces_bytes: usize = aces.iter().map(|a| a.len()).sum();
    let acl_size = 8u16 + aces_bytes as u16;
    acl.extend_from_slice(&acl_size.to_le_bytes());
    acl.extend_from_slice(&(aces.len() as u16).to_le_bytes());
    acl.extend_from_slice(&0u16.to_le_bytes()); // sbz2
    for ace in &aces {
        acl.extend_from_slice(ace);
    }

    // SECURITY_DESCRIPTOR_RELATIVE layout (DACL first, then owner / group).
    let header_len = 20usize;
    let dacl_off = header_len as u32;
    let owner_off = dacl_off + acl.len() as u32;
    let group_off = owner_off + system.len() as u32;
    let mut sd = vec![0u8; header_len];
    sd[0] = 1; // revision
    sd[1] = 0; // sbz
    sd[2..4].copy_from_slice(&0x8004u16.to_le_bytes()); // control
    sd[4..8].copy_from_slice(&owner_off.to_le_bytes());
    sd[8..12].copy_from_slice(&group_off.to_le_bytes());
    sd[12..16].copy_from_slice(&0u32.to_le_bytes()); // sacl
    sd[16..20].copy_from_slice(&dacl_off.to_le_bytes());
    sd.extend_from_slice(&acl);
    sd.extend_from_slice(&system); // owner
    sd.extend_from_slice(&system); // group
    sd
}

/// Trivial 32-bit hash for `$SDH`. Real NTFS uses a specific polynomial
/// (each byte folded with `h = h*0x67 + b`) but our readers index by
/// security_id via `$SII`, not by hash; mkntfs and the kernel only need
/// `$SDH` to be self-consistent (the hash stored in the key must match
/// what `$SDS` says). We compute a stable hash so the entry-and-key pair
/// agrees with itself.
pub fn sd_hash(buf: &[u8]) -> u32 {
    let mut h: u32 = 0;
    for &b in buf {
        h = h.wrapping_mul(0x67).wrapping_add(b as u32);
    }
    h
}

/// Decoded view of one entry the formatter staged into `$Secure:$SDS`.
/// `sds_offset` is the byte offset within `$SDS` where the 20-byte SDS
/// entry header for this row begins; `sds_size` is the value stored in
/// that header's `size` field (20 bytes header + SD blob length, **NOT**
/// including the 16-byte alignment pad).
#[derive(Debug, Clone, Copy)]
pub struct SdsEntryLayout {
    pub security_id: u32,
    pub hash: u32,
    pub sds_offset: u64,
    pub sds_size: u32,
}

/// Build the bytes of `$Secure:$SDS` carrying a single SD with
/// `security_id = FIRST_SECURITY_ID`. Returns `(stream, entry_size)`
/// where `entry_size` is the size field stored inside the SDS entry
/// header (header + SD blob, NOT including the 16-byte alignment pad).
///
/// Single-entry shortcut used by tests and callers that want a minimal
/// `$Secure`. Multi-entry callers should use
/// [`build_sds_stream_multi`].
pub fn build_sds_stream(sd_blob: &[u8]) -> (Vec<u8>, u32) {
    let (bytes, entries) = build_sds_stream_multi(&[(FIRST_SECURITY_ID, sd_blob)]);
    let size = entries.first().map(|e| e.sds_size).unwrap_or(0);
    (bytes, size)
}

/// Build the bytes of `$Secure:$SDS` for `entries`, in order. Each SDS
/// entry is laid out as:
///
/// * 20-byte SDS entry header `(hash, security_id, offset_in_sds, size)`
/// * `sd_blob` immediately after.
/// * Zero-padding so the next entry starts on a 16-byte boundary.
///
/// Returns `(stream_bytes, layout)` where `layout[i]` describes where
/// each input entry landed (the caller will use this to build `$SDH` and
/// `$SII` entries pointing at the right `sds_offset` / `sds_size`).
pub fn build_sds_stream_multi(entries: &[(u32, &[u8])]) -> (Vec<u8>, Vec<SdsEntryLayout>) {
    let mut out = Vec::new();
    let mut layouts = Vec::with_capacity(entries.len());
    for &(security_id, sd_blob) in entries {
        let sds_offset = out.len() as u64;
        let entry_size_no_pad = 20u32 + sd_blob.len() as u32;
        let hash = sd_hash(sd_blob);
        out.extend_from_slice(&hash.to_le_bytes());
        out.extend_from_slice(&security_id.to_le_bytes());
        out.extend_from_slice(&sds_offset.to_le_bytes());
        out.extend_from_slice(&entry_size_no_pad.to_le_bytes());
        out.extend_from_slice(sd_blob);
        while out.len() % 16 != 0 {
            out.push(0);
        }
        layouts.push(SdsEntryLayout {
            security_id,
            hash,
            sds_offset,
            sds_size: entry_size_no_pad,
        });
    }
    (out, layouts)
}

/// Build $Secure (record 9) carrying one or more security descriptors.
///
/// `sds_lcn` / `sds_clusters` describe the non-resident `$DATA:$SDS`
/// extent allocated by [`format_volume`]; `sds_size` is the number of
/// bytes actually used at the start of that extent (sum of entry headers
/// + SD blobs, 16-byte padded between entries).
///
/// Each entry in `sds_entries` becomes one row in `$SDH` (keyed by
/// `(hash, id)`) and `$SII` (keyed by `id`). The order in `sds_entries`
/// matters because `build_secure_index_root` does not re-sort.
#[allow(clippy::too_many_arguments)]
pub fn build_secure_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    parent_ref: u64,
    filetime: u64,
    sds_lcn: u64,
    sds_clusters: u64,
    sds_size: u64,
    sds_entries: &[SdsEntryLayout],
    cluster_size: u64,
    sector_size: usize,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, SECURE_FILE_ATTRS),
        0,
        0,
    );
    let fn_value =
        build_file_name_value(parent_ref, "$Secure", SECURE_FILE_ATTRS, 0, 0, filetime, 1);
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);

    // $SDS: non-resident $DATA stream named "$SDS".
    let sds_name: Vec<u8> = "$SDS"
        .encode_utf16()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    let allocated = sds_clusters * cluster_size;
    let runs = encode_single_run(sds_lcn, sds_clusters);
    let sds = build_non_resident_attr(
        TYPE_DATA,
        &sds_name,
        &runs,
        0,
        sds_clusters.saturating_sub(1),
        allocated,
        sds_size,
        sds_size,
        0,
        0,
    );

    // $SDH index root: one entry per SDS row (in input order), then a
    // single terminator.
    let sdh_entries: Vec<Vec<u8>> = sds_entries
        .iter()
        .map(|e| build_sdh_entry(e.hash, e.security_id, e.sds_offset, e.sds_size))
        .collect();
    let sdh_root = build_secure_index_root(0x12, &sdh_entries);
    let sdh_name: Vec<u8> = "$SDH"
        .encode_utf16()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    let sdh = build_resident_attr(TYPE_INDEX_ROOT, &sdh_name, &sdh_root, 0, 0);

    // $SII index root: one entry per SDS row, keyed by ascending
    // security_id (`mkntfs` does the same so the leaf is collation-sorted).
    let mut sorted: Vec<&SdsEntryLayout> = sds_entries.iter().collect();
    sorted.sort_by_key(|e| e.security_id);
    let sii_entries: Vec<Vec<u8>> = sorted
        .iter()
        .map(|e| build_sii_entry(e.hash, e.security_id, e.sds_offset, e.sds_size))
        .collect();
    let sii_root = build_secure_index_root(0x10, &sii_entries);
    let sii_name: Vec<u8> = "$SII"
        .encode_utf16()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    let sii = build_resident_attr(TYPE_INDEX_ROOT, &sii_name, &sii_root, 0, 0);

    emit_record(
        rec_buf,
        rec_size,
        REC_SECURE,
        mft::RecordHeader::FLAG_IN_USE | MFT_RECORD_FLAG_VIEW_INDEX,
        &[si, fname, sds, sdh, sii],
        sector_size,
        1,
    );
}

/// Build an `$INDEX_ROOT` value for `$SDH` / `$SII`. `entries` are the
/// raw bytes of the real index entries in collation order; a 16-byte
/// terminator entry is appended here. The indexed-attribute type is
/// always 0 ("view index") for both streams; the caller passes the
/// appropriate collation (0x12 for SDH by hash+id, 0x10 for SII by
/// security_id alone).
fn build_secure_index_root(collation: u32, entries: &[Vec<u8>]) -> Vec<u8> {
    let index_block_size = DEFAULT_INDEX_RECORD_SIZE;
    let cpib: i8 = 1;
    let entries_total: usize = entries.iter().map(|e| e.len()).sum();
    let mut v = Vec::with_capacity(0x20 + entries_total + 16);
    // INDEX_ROOT header (16 bytes).
    v.extend_from_slice(&0u32.to_le_bytes()); // indexed attribute type = 0 ("view index")
    v.extend_from_slice(&collation.to_le_bytes());
    v.extend_from_slice(&index_block_size.to_le_bytes());
    v.push(cpib as u8);
    v.extend_from_slice(&[0u8; 3]);
    // Index header (16 bytes).
    let first_entry_offset = 16u32;
    let term_entry_len = 16u32;
    let bytes_in_use = 16u32 + entries_total as u32 + term_entry_len;
    let bytes_allocated = bytes_in_use;
    v.extend_from_slice(&first_entry_offset.to_le_bytes());
    v.extend_from_slice(&bytes_in_use.to_le_bytes());
    v.extend_from_slice(&bytes_allocated.to_le_bytes());
    v.push(0); // flags (SMALL_INDEX)
    v.extend_from_slice(&[0u8; 3]);
    for e in entries {
        v.extend_from_slice(e);
    }
    // Terminator entry.
    let mut term = vec![0u8; 16];
    term[8..10].copy_from_slice(&16u16.to_le_bytes());
    term[12..16].copy_from_slice(&0x02u32.to_le_bytes());
    v.extend_from_slice(&term);
    v
}

/// Build a single `$SDH` index entry: 48 bytes carrying an 8-byte key
/// `(hash, security_id)`, a 20-byte data payload mirroring the SDS entry
/// header, and a trailing 4-byte "II" tag (Unicode 'I','I').
fn build_sdh_entry(hash: u32, security_id: u32, sds_offset: u64, sds_size: u32) -> Vec<u8> {
    let entry_len: u16 = 48;
    let key_len: u16 = 8;
    let data_off: u16 = 0x18;
    let data_size: u16 = 20;
    let mut e = vec![0u8; entry_len as usize];
    e[0..2].copy_from_slice(&data_off.to_le_bytes());
    e[2..4].copy_from_slice(&data_size.to_le_bytes());
    e[8..10].copy_from_slice(&entry_len.to_le_bytes());
    e[10..12].copy_from_slice(&key_len.to_le_bytes());
    // Key at 16..24: hash + security_id.
    e[16..20].copy_from_slice(&hash.to_le_bytes());
    e[20..24].copy_from_slice(&security_id.to_le_bytes());
    // Data at 24..44: SDS-entry-header copy.
    e[24..28].copy_from_slice(&hash.to_le_bytes());
    e[28..32].copy_from_slice(&security_id.to_le_bytes());
    e[32..40].copy_from_slice(&sds_offset.to_le_bytes());
    e[40..44].copy_from_slice(&sds_size.to_le_bytes());
    // Trailing "II" tag at 44..48.
    e[44..46].copy_from_slice(&0x0049u16.to_le_bytes());
    e[46..48].copy_from_slice(&0x0049u16.to_le_bytes());
    e
}

/// Build a single `$SII` index entry: 40 bytes carrying a 4-byte key
/// (security_id) and a 20-byte data payload mirroring the SDS entry
/// header. Unlike `$SDH`, `$SII` entries have no trailing tag.
fn build_sii_entry(hash: u32, security_id: u32, sds_offset: u64, sds_size: u32) -> Vec<u8> {
    let entry_len: u16 = 40;
    let key_len: u16 = 4;
    let data_off: u16 = 0x14;
    let data_size: u16 = 20;
    let mut e = vec![0u8; entry_len as usize];
    e[0..2].copy_from_slice(&data_off.to_le_bytes());
    e[2..4].copy_from_slice(&data_size.to_le_bytes());
    e[8..10].copy_from_slice(&entry_len.to_le_bytes());
    e[10..12].copy_from_slice(&key_len.to_le_bytes());
    e[16..20].copy_from_slice(&security_id.to_le_bytes());
    e[20..24].copy_from_slice(&hash.to_le_bytes());
    e[24..28].copy_from_slice(&security_id.to_le_bytes());
    e[28..36].copy_from_slice(&sds_offset.to_le_bytes());
    e[36..40].copy_from_slice(&sds_size.to_le_bytes());
    e
}

/// Build $UpCase (record 10) with non-resident $DATA.
#[allow(clippy::too_many_arguments)]
pub fn build_upcase_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    parent_ref: u64,
    upcase_bytes: u64,
    upcase_lcn: u64,
    upcase_clusters: u64,
    filetime: u64,
    cluster_size: u64,
    sector_size: usize,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, 0x06),
        0,
        0,
    );
    let fn_value = build_file_name_value(
        parent_ref,
        "$UpCase",
        0x06,
        upcase_bytes,
        upcase_clusters * cluster_size,
        filetime,
        1,
    );
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    let runs = encode_single_run(upcase_lcn, upcase_clusters);
    let data = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &runs,
        0,
        upcase_clusters - 1,
        upcase_clusters * cluster_size,
        upcase_bytes,
        upcase_bytes,
        0,
        0,
    );
    emit_record(
        rec_buf,
        rec_size,
        REC_UPCASE,
        mft::RecordHeader::FLAG_IN_USE,
        &[si, fname, data],
        sector_size,
        1,
    );
}

/// Build $Extend (record 11): empty directory.
pub fn build_extend_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    parent_ref: u64,
    filetime: u64,
    sector_size: usize,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, 0x06),
        0,
        0,
    );
    let fn_value = build_file_name_value(parent_ref, "$Extend", 0x10000006, 0, 0, filetime, 1);
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    let i30_name: Vec<u8> = "$I30"
        .encode_utf16()
        .flat_map(|u| u.to_le_bytes())
        .collect();
    let idx_root = build_resident_attr(TYPE_INDEX_ROOT, &i30_name, &build_empty_index_root(), 0, 0);
    emit_record(
        rec_buf,
        rec_size,
        REC_EXTEND,
        mft::RecordHeader::FLAG_IN_USE | mft::RecordHeader::FLAG_DIRECTORY,
        &[si, fname, idx_root],
        sector_size,
        1,
    );
}

/// Build a reserved/unused but allocated record (records 12..15).
/// Marked IN_USE so chkdsk doesn't complain, but contains only a
/// $STANDARD_INFORMATION + $FILE_NAME placeholder.
pub fn build_reserved_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    rec_no: u64,
    parent_ref: u64,
    name: &str,
    filetime: u64,
    sector_size: usize,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, 0x06),
        0,
        0,
    );
    let fn_value = build_file_name_value(parent_ref, name, 0x06, 0, 0, filetime, 1);
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    emit_record(
        rec_buf,
        rec_size,
        rec_no,
        mft::RecordHeader::FLAG_IN_USE,
        &[si, fname],
        sector_size,
        1,
    );
}

/// Build the $LogFile record (record 2). The $DATA is non-resident,
/// pointing at LOGFILE_BYTES of zeroed clusters.
#[allow(clippy::too_many_arguments)]
pub fn build_logfile_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    parent_ref: u64,
    log_lcn: u64,
    log_clusters: u64,
    filetime: u64,
    cluster_size: u64,
    sector_size: usize,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, 0x06),
        0,
        0,
    );
    let log_bytes = log_clusters * cluster_size;
    let fn_value = build_file_name_value(
        parent_ref, "$LogFile", 0x06, log_bytes, log_bytes, filetime, 1,
    );
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    let runs = encode_single_run(log_lcn, log_clusters);
    let data = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &runs,
        0,
        log_clusters - 1,
        log_bytes,
        log_bytes,
        log_bytes,
        0,
        0,
    );
    emit_record(
        rec_buf,
        rec_size,
        REC_LOGFILE,
        mft::RecordHeader::FLAG_IN_USE,
        &[si, fname, data],
        sector_size,
        1,
    );
}

/// Build $MFT itself (record 0). $DATA is a non-resident attribute pointing
/// at the MFT's own clusters; $BITMAP is a non-resident attribute pointing
/// at the MFT-record bitmap.
#[allow(clippy::too_many_arguments)]
pub fn build_mft_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    parent_ref: u64,
    mft_data_extents: &[(u64, u64)],
    mft_records: u64,
    mft_bitmap_lcn: u64,
    mft_bitmap_clusters: u64,
    filetime: u64,
    cluster_size: u64,
    sector_size: usize,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, 0x06),
        0,
        0,
    );
    let fn_value = build_file_name_value(
        parent_ref,
        "$MFT",
        0x06,
        mft_records * rec_size as u64,
        mft_records * rec_size as u64,
        filetime,
        1,
    );
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    let runs = encode_run_list(mft_data_extents);
    let total_mft_clusters: u64 = mft_data_extents.iter().map(|(_, l)| l).sum();
    let data = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &runs,
        0,
        total_mft_clusters - 1,
        total_mft_clusters * cluster_size,
        mft_records * rec_size as u64,
        mft_records * rec_size as u64,
        0,
        0,
    );
    let bm_runs = encode_single_run(mft_bitmap_lcn, mft_bitmap_clusters);
    // $BITMAP holds one bit per record. Size = ceil(mft_records / 8).
    let bm_bytes = mft_records.div_ceil(8);
    let bitmap = build_non_resident_attr(
        TYPE_BITMAP,
        &[],
        &bm_runs,
        0,
        mft_bitmap_clusters - 1,
        mft_bitmap_clusters * cluster_size,
        bm_bytes,
        bm_bytes,
        0,
        0,
    );
    emit_record(
        rec_buf,
        rec_size,
        REC_MFT,
        mft::RecordHeader::FLAG_IN_USE,
        &[si, fname, data, bitmap],
        sector_size,
        1,
    );
}

/// Build $MFTMirr (record 1). Same shape as $MFT but the $DATA points at
/// the mirror's own clusters (4 records worth = enough to mirror records 0..3).
#[allow(clippy::too_many_arguments)]
pub fn build_mftmirr_record(
    rec_buf: &mut [u8],
    rec_size: usize,
    parent_ref: u64,
    mirror_lcn: u64,
    mirror_clusters: u64,
    filetime: u64,
    cluster_size: u64,
    sector_size: usize,
) {
    let si = build_resident_attr(
        TYPE_STANDARD_INFORMATION,
        &[],
        &build_system_si_value(filetime, 0x06),
        0,
        0,
    );
    let bytes = mirror_clusters * cluster_size;
    let fn_value = build_file_name_value(parent_ref, "$MFTMirr", 0x06, bytes, bytes, filetime, 1);
    let fname = build_resident_attr(TYPE_FILE_NAME, &[], &fn_value, 0, 0);
    let runs = encode_single_run(mirror_lcn, mirror_clusters);
    let data = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &runs,
        0,
        mirror_clusters - 1,
        bytes,
        bytes,
        bytes,
        0,
        0,
    );
    emit_record(
        rec_buf,
        rec_size,
        REC_MFTMIRR,
        mft::RecordHeader::FLAG_IN_USE,
        &[si, fname, data],
        sector_size,
        1,
    );
}

// ----- INDEX_ROOT mutation helpers used by the writer -------------------

/// Insert (or update) a single index entry in a small $INDEX_ROOT that
/// currently uses only the SMALL_INDEX layout. Returns the new value bytes
/// (with terminator entry preserved at the end).
///
/// Returns `Err(Unsupported)` if the resulting root would exceed the
/// `max_resident_bytes` budget — the caller should promote the directory
/// to $INDEX_ALLOCATION at that point.
pub fn insert_into_index_root(
    root_value: &[u8],
    new_entry: &[u8],
    max_resident_bytes: usize,
) -> Result<Vec<u8>> {
    // Layout: 16 bytes index-meta + 16 bytes index-header + entries (incl. terminator).
    if root_value.len() < 32 {
        return Err(crate::Error::InvalidImage(
            "ntfs: $INDEX_ROOT too small to mutate".into(),
        ));
    }
    let header_meta = &root_value[..16];
    let _first_entry_offset = u32::from_le_bytes(root_value[16..20].try_into().unwrap()) as usize;
    let bytes_in_use = u32::from_le_bytes(root_value[20..24].try_into().unwrap()) as usize;
    let flags = root_value[28];
    let entries_start = 16 + 16; // index header starts at 16; entries at 16+16
    let entries_end = 16 + bytes_in_use; // bytes_in_use is measured from index header start

    // Walk to find the terminator and the insertion point. We keep entries
    // in input order (caller is responsible for ordering by collation if
    // they care — for our SMALL_INDEX directories, NTFS will sort on next
    // mount).
    let mut cursor = entries_start;
    let mut entries: Vec<Vec<u8>> = Vec::new();
    let mut terminator: Vec<u8> = Vec::new();
    while cursor + 16 <= entries_end {
        let entry_len =
            u16::from_le_bytes(root_value[cursor + 8..cursor + 10].try_into().unwrap()) as usize;
        if entry_len < 16 || cursor + entry_len > entries_end {
            return Err(crate::Error::InvalidImage(
                "ntfs: malformed $INDEX_ROOT entry length".into(),
            ));
        }
        let e_flags = u32::from_le_bytes(root_value[cursor + 12..cursor + 16].try_into().unwrap());
        let is_last = e_flags & 0x02 != 0;
        let slice = root_value[cursor..cursor + entry_len].to_vec();
        if is_last {
            terminator = slice;
            break;
        } else {
            entries.push(slice);
        }
        cursor += entry_len;
    }
    if terminator.is_empty() {
        return Err(crate::Error::InvalidImage(
            "ntfs: $INDEX_ROOT missing terminator".into(),
        ));
    }
    entries.push(new_entry.to_vec());

    // Rebuild
    let entries_total: usize = entries.iter().map(|e| e.len()).sum::<usize>() + terminator.len();
    let new_bytes_in_use = 16 + entries_total;
    let new_total = 16 + new_bytes_in_use;
    if new_total > max_resident_bytes {
        return Err(crate::Error::Unsupported(
            "ntfs: directory index would overflow $INDEX_ROOT — promotion to $INDEX_ALLOCATION needed".into(),
        ));
    }
    let mut out = Vec::with_capacity(new_total);
    out.extend_from_slice(header_meta);
    // Index header
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&(new_bytes_in_use as u32).to_le_bytes());
    out.extend_from_slice(&(new_bytes_in_use as u32).to_le_bytes());
    out.push(flags);
    out.extend_from_slice(&[0u8; 3]);
    for e in &entries {
        out.extend_from_slice(e);
    }
    out.extend_from_slice(&terminator);
    Ok(out)
}

/// Compute the number of mirror clusters: covers records 0..3 (4 × rec_size).
pub fn mirror_clusters(rec_size: u32, cluster_size: u32) -> u64 {
    let mirror_bytes = 4 * rec_size as u64;
    mirror_bytes.div_ceil(cluster_size as u64)
}

/// Apply `fn(value: &[u8]) -> Vec<u8>` to a named resident attribute of
/// `type_code` in the given (already fixup-removed) MFT record. The
/// new value is written back in place, the attribute is grown by
/// the size delta, and the record's bytes_in_use is bumped.
///
/// Returns `Err(Unsupported)` if the rewrite would overflow the record
/// (callers should then split via $ATTRIBUTE_LIST, which the writer
/// currently does not do — promotion to $INDEX_ALLOCATION handles the
/// $I30 case before that limit is hit).
pub fn rewrite_resident_attr(
    rec: &mut [u8],
    rec_size: usize,
    type_code: u32,
    name: &str,
    new_value: &[u8],
) -> Result<()> {
    let hdr = mft::RecordHeader::parse(rec)?;
    let bytes_in_use = hdr.bytes_in_use as usize;
    let first = hdr.first_attribute_offset as usize;
    let mut cursor = first;
    loop {
        if cursor + 4 > bytes_in_use {
            return Err(crate::Error::InvalidImage(
                "ntfs: attribute walk past bytes_in_use".into(),
            ));
        }
        let tc = u32::from_le_bytes(rec[cursor..cursor + 4].try_into().unwrap());
        if tc == 0xFFFF_FFFF {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: attribute type 0x{:x} not found in record",
                type_code
            )));
        }
        let len = u32::from_le_bytes(rec[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let non_resident = rec[cursor + 8] != 0;
        let name_len = rec[cursor + 9] as usize;
        let name_off =
            u16::from_le_bytes(rec[cursor + 10..cursor + 12].try_into().unwrap()) as usize;
        let attr_name = if name_len == 0 {
            String::new()
        } else {
            super::attribute::decode_utf16le(
                &rec[cursor + name_off..cursor + name_off + name_len * 2],
            )
        };
        if tc == type_code && attr_name == name && !non_resident {
            // Resident value layout: 0x10 value_length(u32), 0x14 value_offset(u16), 0x16 indexed_flag
            let value_off =
                u16::from_le_bytes(rec[cursor + 0x14..cursor + 0x16].try_into().unwrap()) as usize;
            let header_block_len = value_off; // bytes up to value
            let new_total = (header_block_len + new_value.len() + 7) & !7;
            let old_total = len;
            // Shift tail (attrs after this one + terminator) accordingly.
            let after = cursor + old_total;
            let new_after = cursor + new_total;
            // Resulting record must fit.
            let tail_len = bytes_in_use - after;
            let new_bytes_in_use = new_after + tail_len;
            if new_bytes_in_use + 8 > rec_size {
                return Err(crate::Error::Unsupported(
                    "ntfs: resident attribute rewrite would overflow MFT record".into(),
                ));
            }
            // Copy tail (carefully: may overlap). Use a copy via Vec.
            let tail = rec[after..bytes_in_use].to_vec();
            // Pad out with zeros up to new_after to keep alignment.
            rec[cursor + 0x10..cursor + 0x14]
                .copy_from_slice(&(new_value.len() as u32).to_le_bytes());
            rec[cursor + 4..cursor + 8].copy_from_slice(&(new_total as u32).to_le_bytes());
            // Zero the old value region first
            for b in &mut rec[cursor + value_off..after] {
                *b = 0;
            }
            // Write new value
            rec[cursor + value_off..cursor + value_off + new_value.len()]
                .copy_from_slice(new_value);
            // Pad to new_total
            for b in &mut rec[cursor + value_off + new_value.len()..new_after] {
                *b = 0;
            }
            // Write tail
            for (i, b) in tail.iter().enumerate() {
                rec[new_after + i] = *b;
            }
            // Update bytes_in_use
            rec[0x18..0x1C].copy_from_slice(&(new_bytes_in_use as u32).to_le_bytes());
            // Clear anything beyond new_bytes_in_use (so the USA fixup doesn't fight stale state).
            for b in &mut rec[new_bytes_in_use..rec_size] {
                *b = 0;
            }
            return Ok(());
        }
        cursor += len;
    }
}

/// Lay out a fresh NTFS volume on `dev`. Writes boot sector, mirror, $MFT,
/// $MFTMirr, $LogFile, $Volume, $AttrDef, root, $Bitmap, $Boot, $BadClus,
/// $Secure, $UpCase, $Extend, and reserved slots 12..15. Returns the
/// post-format state needed by the runtime writer.
#[derive(Debug, Clone)]
pub struct LayoutResult {
    pub cluster_size: u32,
    pub bytes_per_sector: u16,
    pub sectors_per_cluster: u8,
    pub total_clusters: u64,
    pub mft_record_size: u32,
    pub index_record_size: u32,
    /// `$MFT`'s on-disk extents (LCN, length-in-clusters).
    pub mft_extents: Vec<(u64, u64)>,
    /// Number of MFT records currently allocated (header capacity).
    pub mft_records: u64,
    pub mftmirr_lcn: u64,
    pub bitmap_lcn: u64,
    pub bitmap_clusters: u64,
    pub bitmap: BitmapAlloc,
    pub mft_bitmap_lcn: u64,
    pub mft_bitmap_clusters: u64,
    pub volume_serial: u64,
    pub upcase_lcn: u64,
    pub upcase_clusters: u64,
    pub logfile_lcn: u64,
    pub logfile_clusters: u64,
    pub attrdef_lcn: u64,
    pub attrdef_clusters: u64,
}

/// Round `bytes` up to a multiple of `cluster_size` and return clusters.
fn ceil_div(num: u64, den: u64) -> u64 {
    num.div_ceil(den)
}

/// Lay out a fresh volume. The device's `total_size()` must be ≥ a few MiB.
#[allow(clippy::too_many_lines)]
pub fn format_volume(dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<LayoutResult> {
    let total_size = dev.total_size();
    if total_size < 4 * 1024 * 1024 {
        return Err(crate::Error::InvalidArgument(
            "ntfs: minimum supported volume size is 4 MiB".into(),
        ));
    }
    let bps = opts.bytes_per_sector as u32;
    let spc = opts.sectors_per_cluster as u32;
    if !bps.is_power_of_two() || !spc.is_power_of_two() {
        return Err(crate::Error::InvalidArgument(
            "ntfs: bytes_per_sector and sectors_per_cluster must be powers of two".into(),
        ));
    }
    let cluster_size = bps * spc;
    if cluster_size != DEFAULT_CLUSTER_SIZE {
        // Allow non-default but warn-trace; the writer logic below assumes 4 KiB.
        // It still works for other multiples, but we standardise here.
    }
    let total_sectors = total_size / bps as u64;
    let total_clusters = total_size / cluster_size as u64;

    let rec_size = DEFAULT_MFT_RECORD_SIZE;
    let mft_record_field: i8 = -10; // 1 << 10 = 1024
    let index_record_field: i8 = -12; // 1 << 12 = 4096
    let mft_clusters = INITIAL_MFT_CLUSTERS;
    let mft_records_capacity = mft_clusters * cluster_size as u64 / rec_size as u64;
    let mft_lcn = 4u64; // matches the read-side tests' expectation

    let mut bitmap = BitmapAlloc::new(total_clusters);
    // Reserve cluster 0 (boot sector + first sector) → 1 cluster.
    bitmap.set_range(0, 1);
    // Reserve $MFT clusters.
    bitmap.set_range(mft_lcn, mft_clusters);
    bitmap.next_hint = mft_lcn + mft_clusters;

    // Allocate $MFTMirr in the volume midpoint (or at fixed near-end if small).
    let mirror_clusters_n = mirror_clusters(rec_size, cluster_size);
    let mid_cluster = total_clusters / 2;
    // Try midpoint, fallback to allocator.
    let mftmirr_lcn = {
        let candidate = mid_cluster;
        if candidate + mirror_clusters_n <= total_clusters
            && (candidate..candidate + mirror_clusters_n).all(|c| !bitmap.is_set(c))
        {
            bitmap.set_range(candidate, mirror_clusters_n);
            candidate
        } else {
            bitmap.allocate(mirror_clusters_n)?
        }
    };

    // Allocate $LogFile.
    let logfile_clusters = ceil_div(LOGFILE_BYTES, cluster_size as u64);
    let logfile_lcn = bitmap.allocate(logfile_clusters)?;
    // Allocate $AttrDef payload.
    let attrdef = build_attrdef_payload();
    let attrdef_clusters = ceil_div(attrdef.len() as u64, cluster_size as u64);
    let attrdef_lcn = bitmap.allocate(attrdef_clusters)?;
    // Allocate $UpCase payload (128 KiB).
    let upcase = build_upcase_blob();
    let upcase_clusters = ceil_div(upcase.len() as u64, cluster_size as u64);
    let upcase_lcn = bitmap.allocate(upcase_clusters)?;
    // Allocate $Secure:$SDS — one cluster is more than enough for the
    // small SD catalogue we plant (≈ 100 bytes per entry, padded to 16).
    // We emit one SD per concrete `SecurityClass` actually in use on a
    // fresh volume: User (default) and System (records 0..=15).
    let security_catalogue = build_security_catalogue();
    let sds_entry_refs: Vec<(u32, &[u8])> = security_catalogue
        .iter()
        .map(|(id, sd)| (*id, sd.as_slice()))
        .collect();
    let (sds_stream, sds_layouts) = build_sds_stream_multi(&sds_entry_refs);
    let sds_used = sds_stream.len() as u64;
    let sds_clusters = ceil_div(sds_used.max(1), cluster_size as u64);
    let sds_lcn = bitmap.allocate(sds_clusters)?;
    // Allocate $Bitmap (will be rewritten on flush; size known now).
    let bitmap_bytes = bitmap.bytes.len() as u64;
    let bitmap_clusters = ceil_div(bitmap_bytes, cluster_size as u64);
    let bitmap_lcn = bitmap.allocate(bitmap_clusters)?;
    // Allocate $MFT bitmap (1 bit per MFT record).
    let mft_bitmap_bytes = ceil_div(mft_records_capacity, 8);
    let mft_bitmap_clusters = ceil_div(mft_bitmap_bytes, cluster_size as u64).max(1);
    let mft_bitmap_lcn = bitmap.allocate(mft_bitmap_clusters)?;

    let filetime = unix_to_filetime(0);

    let serial = opts.volume_serial;
    let boot = build_boot_sector(
        opts,
        total_sectors,
        mft_lcn,
        mftmirr_lcn,
        mft_record_field,
        index_record_field,
    );
    dev.write_at(0, &boot)?;
    // Backup boot sector at last LBA.
    let last_lba_offset = (total_sectors - 1) * bps as u64;
    dev.write_at(last_lba_offset, &boot)?;

    // Build records 0..15 into a contiguous MFT buffer.
    let mft_buf_size = (mft_clusters * cluster_size as u64) as usize;
    let mut mft_buf = vec![0u8; mft_buf_size];

    let parent_root_ref = pack_mft_ref(REC_ROOT, 1);

    // Record 0: $MFT
    {
        let r = &mut mft_buf
            [(REC_MFT as usize) * rec_size as usize..(REC_MFT as usize + 1) * rec_size as usize];
        build_mft_record(
            r,
            rec_size as usize,
            parent_root_ref,
            &[(mft_lcn, mft_clusters)],
            mft_records_capacity,
            mft_bitmap_lcn,
            mft_bitmap_clusters,
            filetime,
            cluster_size as u64,
            bps as usize,
        );
    }
    // Record 1: $MFTMirr
    {
        let r = &mut mft_buf[(REC_MFTMIRR as usize) * rec_size as usize
            ..(REC_MFTMIRR as usize + 1) * rec_size as usize];
        build_mftmirr_record(
            r,
            rec_size as usize,
            parent_root_ref,
            mftmirr_lcn,
            mirror_clusters_n,
            filetime,
            cluster_size as u64,
            bps as usize,
        );
    }
    // Record 2: $LogFile
    {
        let r = &mut mft_buf[(REC_LOGFILE as usize) * rec_size as usize
            ..(REC_LOGFILE as usize + 1) * rec_size as usize];
        build_logfile_record(
            r,
            rec_size as usize,
            parent_root_ref,
            logfile_lcn,
            logfile_clusters,
            filetime,
            cluster_size as u64,
            bps as usize,
        );
    }
    // Record 3: $Volume
    {
        let r = &mut mft_buf[(REC_VOLUME as usize) * rec_size as usize
            ..(REC_VOLUME as usize + 1) * rec_size as usize];
        build_volume_record(
            r,
            rec_size as usize,
            parent_root_ref,
            &opts.volume_label,
            filetime,
            bps as usize,
        );
    }
    // Record 4: $AttrDef
    {
        let r = &mut mft_buf[(REC_ATTRDEF as usize) * rec_size as usize
            ..(REC_ATTRDEF as usize + 1) * rec_size as usize];
        build_attrdef_record(
            r,
            rec_size as usize,
            REC_ATTRDEF,
            parent_root_ref,
            attrdef.len() as u64,
            attrdef_lcn,
            attrdef_clusters,
            filetime,
            cluster_size as u64,
            bps as usize,
            "$AttrDef",
        );
    }
    // Record 5: root directory
    {
        let r = &mut mft_buf
            [(REC_ROOT as usize) * rec_size as usize..(REC_ROOT as usize + 1) * rec_size as usize];
        build_root_record(r, rec_size as usize, filetime, bps as usize);
    }
    // Record 6: $Bitmap
    {
        let r = &mut mft_buf[(REC_BITMAP as usize) * rec_size as usize
            ..(REC_BITMAP as usize + 1) * rec_size as usize];
        build_bitmap_record(
            r,
            rec_size as usize,
            parent_root_ref,
            bitmap_bytes,
            bitmap_lcn,
            bitmap_clusters,
            filetime,
            cluster_size as u64,
            bps as usize,
        );
    }
    // Record 7: $Boot
    {
        let r = &mut mft_buf
            [(REC_BOOT as usize) * rec_size as usize..(REC_BOOT as usize + 1) * rec_size as usize];
        build_boot_record(
            r,
            rec_size as usize,
            parent_root_ref,
            total_size,
            filetime,
            bps as usize,
        );
    }
    // Record 8: $BadClus
    {
        let r = &mut mft_buf[(REC_BADCLUS as usize) * rec_size as usize
            ..(REC_BADCLUS as usize + 1) * rec_size as usize];
        build_badclus_record(
            r,
            rec_size as usize,
            parent_root_ref,
            total_clusters,
            filetime,
            cluster_size as u64,
            bps as usize,
        );
    }
    // Record 9: $Secure
    {
        let r = &mut mft_buf[(REC_SECURE as usize) * rec_size as usize
            ..(REC_SECURE as usize + 1) * rec_size as usize];
        build_secure_record(
            r,
            rec_size as usize,
            parent_root_ref,
            filetime,
            sds_lcn,
            sds_clusters,
            sds_used,
            &sds_layouts,
            cluster_size as u64,
            bps as usize,
        );
    }
    // Record 10: $UpCase
    {
        let r = &mut mft_buf[(REC_UPCASE as usize) * rec_size as usize
            ..(REC_UPCASE as usize + 1) * rec_size as usize];
        build_upcase_record(
            r,
            rec_size as usize,
            parent_root_ref,
            upcase.len() as u64,
            upcase_lcn,
            upcase_clusters,
            filetime,
            cluster_size as u64,
            bps as usize,
        );
    }
    // Record 11: $Extend
    {
        let r = &mut mft_buf[(REC_EXTEND as usize) * rec_size as usize
            ..(REC_EXTEND as usize + 1) * rec_size as usize];
        build_extend_record(
            r,
            rec_size as usize,
            parent_root_ref,
            filetime,
            bps as usize,
        );
    }
    // Records 12..15: reserved placeholders.
    for (rec_no, name) in
        (12u64..=15u64).zip(["$Reserved12", "$Reserved13", "$Reserved14", "$Reserved15"])
    {
        let r = &mut mft_buf
            [(rec_no as usize) * rec_size as usize..(rec_no as usize + 1) * rec_size as usize];
        build_reserved_record(
            r,
            rec_size as usize,
            rec_no,
            parent_root_ref,
            name,
            filetime,
            bps as usize,
        );
    }

    // Write MFT
    dev.write_at(mft_lcn * cluster_size as u64, &mft_buf)?;

    // Write $MFTMirr (mirror first 4 MFT records)
    let mirror_size = (4 * rec_size as u64).min(mirror_clusters_n * cluster_size as u64) as usize;
    dev.write_at(mftmirr_lcn * cluster_size as u64, &mft_buf[..mirror_size])?;

    // Pad rest of mirror with zeros to fill the allocated clusters.
    let mirror_byte_len = (mirror_clusters_n * cluster_size as u64) as usize;
    if mirror_byte_len > mirror_size {
        let zeros = vec![0u8; mirror_byte_len - mirror_size];
        dev.write_at(
            mftmirr_lcn * cluster_size as u64 + mirror_size as u64,
            &zeros,
        )?;
    }

    // Write $LogFile. We stamp two structurally-valid RSTR restart
    // pages with `CleanDismount` set; everything past those two pages
    // stays zero (== "no log records"). See [`super::logfile`].
    super::logfile::write_initial_logfile(
        dev,
        logfile_lcn * cluster_size as u64,
        logfile_clusters * cluster_size as u64,
    )?;

    // Write $AttrDef payload.
    {
        let off = attrdef_lcn * cluster_size as u64;
        dev.write_at(off, &attrdef)?;
        let padded_len = attrdef_clusters * cluster_size as u64;
        if (attrdef.len() as u64) < padded_len {
            let pad = vec![0u8; (padded_len - attrdef.len() as u64) as usize];
            dev.write_at(off + attrdef.len() as u64, &pad)?;
        }
    }
    // Write $UpCase payload.
    {
        let off = upcase_lcn * cluster_size as u64;
        dev.write_at(off, &upcase)?;
        let padded_len = upcase_clusters * cluster_size as u64;
        if (upcase.len() as u64) < padded_len {
            let pad = vec![0u8; (padded_len - upcase.len() as u64) as usize];
            dev.write_at(off + upcase.len() as u64, &pad)?;
        }
    }
    // Write $Secure:$SDS payload (one cluster, one entry).
    {
        let off = sds_lcn * cluster_size as u64;
        dev.write_at(off, &sds_stream)?;
        let padded_len = sds_clusters * cluster_size as u64;
        if (sds_stream.len() as u64) < padded_len {
            let pad = vec![0u8; (padded_len - sds_stream.len() as u64) as usize];
            dev.write_at(off + sds_stream.len() as u64, &pad)?;
        }
    }
    // Write $Bitmap (we'll re-stamp on flush; for now stamp current state).
    {
        let off = bitmap_lcn * cluster_size as u64;
        dev.write_at(off, &bitmap.bytes)?;
        let padded_len = bitmap_clusters * cluster_size as u64;
        if (bitmap.bytes.len() as u64) < padded_len {
            let pad = vec![0u8; (padded_len - bitmap.bytes.len() as u64) as usize];
            dev.write_at(off + bitmap.bytes.len() as u64, &pad)?;
        }
    }
    // Write the MFT bitmap. Initial state: records 0..15 in use.
    {
        let mut mb = vec![0u8; mft_bitmap_bytes as usize];
        for r in 0..16u64 {
            mb[(r / 8) as usize] |= 1u8 << ((r % 8) as u8);
        }
        let off = mft_bitmap_lcn * cluster_size as u64;
        dev.write_at(off, &mb)?;
        let padded = mft_bitmap_clusters * cluster_size as u64;
        if (mb.len() as u64) < padded {
            let pad = vec![0u8; (padded - mb.len() as u64) as usize];
            dev.write_at(off + mb.len() as u64, &pad)?;
        }
    }

    Ok(LayoutResult {
        cluster_size,
        bytes_per_sector: opts.bytes_per_sector,
        sectors_per_cluster: opts.sectors_per_cluster,
        total_clusters,
        mft_record_size: rec_size,
        index_record_size: DEFAULT_INDEX_RECORD_SIZE,
        mft_extents: vec![(mft_lcn, mft_clusters)],
        mft_records: mft_records_capacity,
        mftmirr_lcn,
        bitmap_lcn,
        bitmap_clusters,
        bitmap,
        mft_bitmap_lcn,
        mft_bitmap_clusters,
        volume_serial: serial,
        upcase_lcn,
        upcase_clusters,
        logfile_lcn,
        logfile_clusters,
        attrdef_lcn,
        attrdef_clusters,
    })
}

/// Convert a Unix timestamp (seconds since 1970-01-01) to NT FILETIME
/// (100 ns intervals since 1601-01-01).
pub fn unix_to_filetime(unix_secs: u32) -> u64 {
    const FILETIME_EPOCH_DIFF: u64 = 11_644_473_600; // seconds between 1601 and 1970
    let secs = unix_secs as u64 + FILETIME_EPOCH_DIFF;
    secs * 10_000_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_unsigned_bytes_basic() {
        assert_eq!(min_unsigned_bytes(0), 1);
        assert_eq!(min_unsigned_bytes(0xFF), 1);
        assert_eq!(min_unsigned_bytes(0x100), 2);
        assert_eq!(min_unsigned_bytes(0xFFFF), 2);
        assert_eq!(min_unsigned_bytes(0x10000), 3);
    }

    #[test]
    fn min_signed_bytes_basic() {
        assert_eq!(min_signed_bytes(0), 1);
        assert_eq!(min_signed_bytes(127), 1);
        assert_eq!(min_signed_bytes(-128), 1);
        assert_eq!(min_signed_bytes(128), 2);
        assert_eq!(min_signed_bytes(-129), 2);
    }

    #[test]
    fn encode_single_run_basic() {
        let r = encode_single_run(20, 1);
        // header: off_size=1, len_size=1 → 0x11; length=0x01; offset=0x14; terminator
        assert_eq!(r, vec![0x11, 0x01, 0x14, 0x00]);
    }

    #[test]
    fn upcase_blob_size() {
        let u = build_upcase_blob();
        assert_eq!(u.len(), 128 * 1024);
    }

    #[test]
    fn filetime_conversion_known() {
        // Unix epoch 0 → 1970-01-01 → FILETIME 116444736000000000.
        assert_eq!(unix_to_filetime(0), 116_444_736_000_000_000);
    }
}
