//! NTFS `$LogFile` — Log File Service (LFS) journal.
//!
//! Spec reference: <https://flatcap.github.io/linux-ntfs/ntfs/files/logfile.html>.
//! All structural knowledge in this module is derived from that page;
//! no foreign source is consulted.
//!
//! ## Path A — minimum-viable LFS for `open_file_rw`
//!
//! The journal lives in `$LogFile` (MFT record 2). On a default 64 KiB
//! log it carries:
//!
//! * Two 4 KiB **restart pages** at the start (`RSTR` magic). Each holds
//!   a restart-page header followed by a restart area and one or more
//!   client records. Only one client is present — `"NTFS"`.
//! * The rest of the file is divided into 4 KiB **record pages**
//!   (`RCRD` magic), each carrying a record-page header followed by one
//!   or more LFS records and their client payloads.
//!
//! Each restart / record page is multi-sector-fixup'd (USA) just like
//! MFT records.
//!
//! ### LSN packing
//!
//! Every record has a 64-bit LSN that doubles as a file-relative
//! locator.  The split is governed by `SeqNumberBits` stored in the
//! restart area:
//!
//! ```text
//!     FileOffset = ((lsn << SeqNumberBits) & 0xFFFFFFFFFFFFFFFF)
//!                  >> (SeqNumberBits - 3)
//! ```
//!
//! For a 64 KiB log with 4 KiB log pages we choose `SeqNumberBits = 50`,
//! leaving 14 bits of file offset (covering 16 KiB-worth of 8-byte
//! aligned positions — the four record pages that fit after the two
//! restart pages).
//!
//! ### Client payload — private format
//!
//! The spec page does not document the NTFS-client record body
//! (`NTFS_LOG_RECORD`).  To replay our own records without consulting
//! foreign source, we use a private payload format identified by a
//! magic prefix.  Foreign LFS consumers (kernel NTFS3, ntfs-3g,
//! `ntfsfix`) only inspect the restart-page chain to decide whether
//! the volume needs replay; when `CleanDismount` is set in the
//! restart area, they trust that the in-place metadata is consistent
//! and never decode the record stream.
//!
//! Each private payload is:
//!
//! ```text
//!     u32 magic   = b"FSTJ"
//!     u64 target_offset_on_disk
//!     u32 length
//!     u8[length] redo_bytes    // contents to apply on redo
//!     u8[length] undo_bytes    // contents to apply on undo
//! ```

use crate::Result;
use crate::block::BlockDevice;

use super::mft;

/// Restart-page / record-page size (matches our 4 KiB cluster default).
pub const LOG_PAGE_SIZE: usize = 4096;

/// Magic at the head of a restart page.
pub const RSTR_MAGIC: &[u8; 4] = b"RSTR";
/// Magic at the head of a record page.
pub const RCRD_MAGIC: &[u8; 4] = b"RCRD";

/// LFS major / minor version we stamp.
pub const LFS_MAJOR_VERSION: i16 = 1;
pub const LFS_MINOR_VERSION: i16 = 1;

/// `RESTART_AREA.Flags`.
pub const FLAG_SINGLE_PAGE_IO: u16 = 0x0001;
pub const FLAG_CLEAN_DISMOUNT: u16 = 0x0002;

/// Client list sentinel for "no client at this index."
pub const NO_CLIENT: u16 = 0xFFFF;

/// LFS_RECORD.RecordType values.
pub const RT_CLIENT_RECORD: u32 = 1;
pub const RT_CLIENT_RESTART: u32 = 2;

/// Private payload magic so we can distinguish our own records.
pub const PRIVATE_PAYLOAD_MAGIC: &[u8; 4] = b"FSTJ";

/// Sequence-number bit split: with a default 64 KiB log we have at most
/// 8 record-page positions (16 KiB span ÷ 2 KiB granularity per spec).
/// Choosing 50 bits for the sequence number leaves 14 file-offset bits,
/// which after the `>> (SeqNumberBits - 3) == >> 47` shift gives an
/// effective 17-bit offset (128 KiB span). Plenty for a 64 KiB log.
pub const SEQ_NUMBER_BITS: u32 = 50;

/// Bytes per sector used for the USA fixup. The format module always
/// emits 512-byte sectors; matching it keeps `apply_fixup` / `install_fixup`
/// happy.
pub const SECTOR_SIZE: usize = 512;

/// Convert an LSN into a byte offset within the log file.
pub fn lsn_to_file_offset(lsn: u64) -> u64 {
    let shifted = lsn.wrapping_shl(SEQ_NUMBER_BITS);
    shifted.wrapping_shr(SEQ_NUMBER_BITS - 3)
}

/// Convert a byte offset within the log file (8-byte aligned) plus a
/// sequence number into a packed LSN.
pub fn pack_lsn(seq: u64, file_offset: u64) -> u64 {
    let off_bits = 64 - SEQ_NUMBER_BITS;
    let off = (file_offset >> 3) & ((1u64 << off_bits) - 1);
    (seq << off_bits) | off
}

/// Extract the sequence-number portion of an LSN.
pub fn lsn_seq(lsn: u64) -> u64 {
    let off_bits = 64 - SEQ_NUMBER_BITS;
    lsn >> off_bits
}

/// Multi-sector header common to RSTR / RCRD.
///
/// Layout (8 bytes):
///   0..4  magic (b"RSTR" or b"RCRD")
///   4..6  usa_offset
///   6..8  usa_count (= 1 USN + N sector tails)
const _: () = ();

/// Build a 4 KiB restart page (`RSTR` magic) carrying a `RESTART_AREA`
/// plus one `LOG_CLIENT_RECORD` for the NTFS client.
///
/// * `current_lsn`: latest written record LSN (0 if none).
/// * `file_size`: total $LogFile size in bytes.
/// * `clean_dismount`: stamps `FLAG_CLEAN_DISMOUNT` when true.
///
/// The page is USA-fixed-up before return; write it to disk verbatim.
pub fn build_restart_page(
    current_lsn: u64,
    file_size: u64,
    clean_dismount: bool,
) -> [u8; LOG_PAGE_SIZE] {
    let mut page = [0u8; LOG_PAGE_SIZE];

    // --- MULTI_SECTOR_HEADER (offset 0..8) ---
    page[0..4].copy_from_slice(RSTR_MAGIC);
    // USA sits at offset 0x1E; covers LOG_PAGE_SIZE / SECTOR_SIZE sectors.
    let sectors = LOG_PAGE_SIZE / SECTOR_SIZE;
    let usa_offset: u16 = 0x1E;
    let usa_count: u16 = (sectors as u16) + 1;
    page[4..6].copy_from_slice(&usa_offset.to_le_bytes());
    page[6..8].copy_from_slice(&usa_count.to_le_bytes());

    // --- RESTART_PAGE_HEADER ---
    // 0x08 chkdsk_lsn (8) — zero (no chkdsk ever ran).
    // 0x10 system_page_size (u32) — restart page size.
    page[0x10..0x14].copy_from_slice(&(LOG_PAGE_SIZE as u32).to_le_bytes());
    // 0x14 log_page_size (u32).
    page[0x14..0x18].copy_from_slice(&(LOG_PAGE_SIZE as u32).to_le_bytes());
    // 0x18 restart_offset (u16) — offset of RESTART_AREA within this page.
    let restart_offset: u16 = 0x40; // first 64 bytes are header + USA region.
    page[0x18..0x1A].copy_from_slice(&restart_offset.to_le_bytes());
    // 0x1A minor_version (i16).
    page[0x1A..0x1C].copy_from_slice(&LFS_MINOR_VERSION.to_le_bytes());
    // 0x1C major_version (i16).
    page[0x1C..0x1E].copy_from_slice(&LFS_MAJOR_VERSION.to_le_bytes());
    // 0x1E.. USA bytes (filled by install_fixup below).

    // --- RESTART_AREA at +0x40 ---
    let ra = restart_offset as usize;
    let mut flags = 0u16;
    if clean_dismount {
        flags |= FLAG_CLEAN_DISMOUNT;
    }
    // 0x00 current_lsn (8).
    page[ra..ra + 8].copy_from_slice(&current_lsn.to_le_bytes());
    // 0x08 log_clients (2) — one client (NTFS).
    page[ra + 8..ra + 10].copy_from_slice(&1u16.to_le_bytes());
    // 0x0A client_free_list (2) — none.
    page[ra + 10..ra + 12].copy_from_slice(&NO_CLIENT.to_le_bytes());
    // 0x0C client_in_use_list (2) — client 0.
    page[ra + 12..ra + 14].copy_from_slice(&0u16.to_le_bytes());
    // 0x0E flags (2).
    page[ra + 14..ra + 16].copy_from_slice(&flags.to_le_bytes());
    // 0x10 seq_number_bits (u32).
    page[ra + 16..ra + 20].copy_from_slice(&SEQ_NUMBER_BITS.to_le_bytes());
    // 0x14 restart_area_length (u16). We allocate fixed 0xA0 (160) bytes
    // total — covers the 0x30 RESTART_AREA fields + 0x70 client record
    // beyond it.
    let restart_area_length: u16 = 0xA0;
    page[ra + 20..ra + 22].copy_from_slice(&restart_area_length.to_le_bytes());
    // 0x16 client_array_offset (u16) — relative to RESTART_AREA start.
    let client_array_offset: u16 = 0x30;
    page[ra + 22..ra + 24].copy_from_slice(&client_array_offset.to_le_bytes());
    // 0x18 file_size (8).
    page[ra + 24..ra + 32].copy_from_slice(&file_size.to_le_bytes());
    // 0x20 last_lsn_data_length (u32) — zero (no record points here).
    // 0x24 record_header_length (u16) — size of LFS_RECORD header (0x30).
    page[ra + 36..ra + 38].copy_from_slice(&0x30u16.to_le_bytes());
    // 0x26 log_page_data_offset (u16) — offset within an RCRD page where
    // record data begins. 0x40 (header + USA region).
    page[ra + 38..ra + 40].copy_from_slice(&0x40u16.to_le_bytes());
    // 0x28 revision_number (u32) — monotonic per restart-area write.
    // We always write 1; the only correctness need is that the
    // higher-revision page wins on tie-break, but we tie-break on
    // current_lsn first which is already monotonic.
    page[ra + 40..ra + 44].copy_from_slice(&1u32.to_le_bytes());

    // --- LOG_CLIENT_RECORD at +ra+0x30 ---
    let cr = ra + client_array_offset as usize;
    // 0x00 oldest_lsn (8) — 0: replay from the very first record.
    // 0x08 client_restart_lsn (8) — current_lsn.
    page[cr + 8..cr + 16].copy_from_slice(&current_lsn.to_le_bytes());
    // 0x10 prev_client (2), 0x12 next_client (2).
    page[cr + 16..cr + 18].copy_from_slice(&NO_CLIENT.to_le_bytes());
    page[cr + 18..cr + 20].copy_from_slice(&NO_CLIENT.to_le_bytes());
    // 0x14 seq_number (2) — incremented each restart; keep at 1.
    page[cr + 20..cr + 22].copy_from_slice(&1u16.to_le_bytes());
    // 0x16..0x1C padding.
    // 0x1C client_name_length (u32) — in bytes, UTF-16LE "NTFS" = 8 bytes.
    page[cr + 28..cr + 32].copy_from_slice(&8u32.to_le_bytes());
    // 0x20.. client_name (UTF-16LE).
    let name_utf16 = ['N', 'T', 'F', 'S'];
    let mut p = cr + 32;
    for c in name_utf16 {
        let u = c as u16;
        page[p..p + 2].copy_from_slice(&u.to_le_bytes());
        p += 2;
    }

    // USA fixup — install with USN = 1.
    mft::install_fixup(&mut page, SECTOR_SIZE, 1);
    page
}

/// Decoded view of a restart page's restart-area + client-record fields.
#[derive(Debug, Clone, Copy)]
pub struct RestartView {
    pub current_lsn: u64,
    pub client_restart_lsn: u64,
    pub file_size: u64,
    pub seq_number_bits: u32,
    pub flags: u16,
}

impl RestartView {
    pub fn is_clean(&self) -> bool {
        self.flags & FLAG_CLEAN_DISMOUNT != 0
    }
}

/// Try to parse `bytes` (expected to be one restart page) into a
/// `RestartView`. Performs the USA fixup; returns `None` if the page
/// fails validation (wrong magic, torn write, mis-shaped).
pub fn parse_restart_page(bytes: &[u8]) -> Option<RestartView> {
    if bytes.len() < LOG_PAGE_SIZE {
        return None;
    }
    let mut buf = vec![0u8; LOG_PAGE_SIZE];
    buf.copy_from_slice(&bytes[..LOG_PAGE_SIZE]);
    if &buf[0..4] != RSTR_MAGIC {
        return None;
    }
    mft::apply_fixup(&mut buf, SECTOR_SIZE).ok()?;
    let restart_offset = u16::from_le_bytes(buf[0x18..0x1A].try_into().unwrap()) as usize;
    if restart_offset + 0x2C > buf.len() {
        return None;
    }
    let ra = restart_offset;
    let current_lsn = u64::from_le_bytes(buf[ra..ra + 8].try_into().unwrap());
    let flags = u16::from_le_bytes(buf[ra + 14..ra + 16].try_into().unwrap());
    let seq_number_bits = u32::from_le_bytes(buf[ra + 16..ra + 20].try_into().unwrap());
    let file_size = u64::from_le_bytes(buf[ra + 24..ra + 32].try_into().unwrap());
    let client_array_offset =
        u16::from_le_bytes(buf[ra + 22..ra + 24].try_into().unwrap()) as usize;
    let client_restart_lsn = if ra + client_array_offset + 16 <= buf.len() {
        u64::from_le_bytes(
            buf[ra + client_array_offset + 8..ra + client_array_offset + 16]
                .try_into()
                .unwrap(),
        )
    } else {
        0
    };
    Some(RestartView {
        current_lsn,
        client_restart_lsn,
        file_size,
        seq_number_bits,
        flags,
    })
}

/// A single LFS record we want to append.  `redo_bytes` are written at
/// `target_offset` on redo; `undo_bytes` are written on undo.
#[derive(Debug, Clone)]
pub struct RedoEntry {
    pub target_offset: u64,
    pub redo_bytes: Vec<u8>,
    pub undo_bytes: Vec<u8>,
}

/// Build the bytes of one record page (`RCRD`) containing `records`.
///
/// `first_lsn` is the LSN of the first record on the page; subsequent
/// records receive incrementing LSNs by file position.
///
/// Returns `(page_bytes, last_lsn)` or an error if the records don't
/// fit in one page.  Callers split across pages as needed.
pub fn build_record_page(
    records: &[RedoEntry],
    first_lsn: u64,
) -> Result<([u8; LOG_PAGE_SIZE], u64)> {
    let mut page = [0u8; LOG_PAGE_SIZE];
    // --- MULTI_SECTOR_HEADER ---
    page[0..4].copy_from_slice(RCRD_MAGIC);
    let sectors = LOG_PAGE_SIZE / SECTOR_SIZE;
    let usa_offset: u16 = 0x28;
    let usa_count: u16 = (sectors as u16) + 1;
    page[4..6].copy_from_slice(&usa_offset.to_le_bytes());
    page[6..8].copy_from_slice(&usa_count.to_le_bytes());

    // --- RECORD_PAGE_HEADER ---
    // 0x08 last_lsn_or_file_offset (8) — last LSN starting on this page.
    // We fill this after packing.
    // 0x10 flags (u32) — RecordEnd = 1.
    page[0x10..0x14].copy_from_slice(&1u32.to_le_bytes());
    // 0x14 page_count (u16) — 1.
    page[0x14..0x16].copy_from_slice(&1u16.to_le_bytes());
    // 0x16 page_position (u16) — 1.
    page[0x16..0x18].copy_from_slice(&1u16.to_le_bytes());
    // 0x18 next_record_offset (u16) — filled after packing.
    // 0x20 last_end_lsn (u64) — filled after packing.

    // Records start at 0x40.
    let mut cursor: usize = 0x40;
    let mut last_lsn = first_lsn;
    let mut last_end_lsn = first_lsn;
    for (cur_lsn, r) in (first_lsn..).zip(records.iter()) {
        // Pack one record: 0x30 header + ClientData.
        // ClientData = magic(4) + target_offset(8) + length(4) + redo + undo.
        let cd_len = 4 + 8 + 4 + r.redo_bytes.len() + r.undo_bytes.len();
        // Align ClientData length up to 8 bytes (NTFS records align).
        let cd_padded = (cd_len + 7) & !7;
        let needed = 0x30 + cd_padded;
        if cursor + needed > LOG_PAGE_SIZE {
            return Err(crate::Error::InvalidImage(
                "ntfs: LFS record page would overflow — too many txn entries".into(),
            ));
        }

        // LFS_RECORD header.
        page[cursor..cursor + 8].copy_from_slice(&cur_lsn.to_le_bytes());
        // 0x08 client_previous_lsn — 0 (first record in our flat chain).
        // 0x10 client_undo_next_lsn — 0.
        page[cursor + 0x18..cursor + 0x1C].copy_from_slice(&(cd_len as u32).to_le_bytes());
        // 0x1C client_seq_number (u16), 0x1E client_index (u16).
        page[cursor + 0x1C..cursor + 0x1E].copy_from_slice(&1u16.to_le_bytes());
        page[cursor + 0x1E..cursor + 0x20].copy_from_slice(&0u16.to_le_bytes());
        // 0x20 record_type (u32).
        page[cursor + 0x20..cursor + 0x24].copy_from_slice(&RT_CLIENT_RECORD.to_le_bytes());
        // 0x24 transaction_id (u32) — 1.
        page[cursor + 0x24..cursor + 0x28].copy_from_slice(&1u32.to_le_bytes());
        // 0x28 flags (u16) — 0.
        // 0x2A..0x30 padding.

        // ClientData.
        let cd = cursor + 0x30;
        page[cd..cd + 4].copy_from_slice(PRIVATE_PAYLOAD_MAGIC);
        page[cd + 4..cd + 12].copy_from_slice(&r.target_offset.to_le_bytes());
        page[cd + 12..cd + 16].copy_from_slice(&(r.redo_bytes.len() as u32).to_le_bytes());
        let mut p = cd + 16;
        page[p..p + r.redo_bytes.len()].copy_from_slice(&r.redo_bytes);
        p += r.redo_bytes.len();
        page[p..p + r.undo_bytes.len()].copy_from_slice(&r.undo_bytes);

        last_lsn = cur_lsn;
        last_end_lsn = cur_lsn + 1;
        cursor += needed;
    }

    // last_lsn_or_file_offset (8) at +0x08.
    page[0x08..0x10].copy_from_slice(&last_lsn.to_le_bytes());
    // next_record_offset (u16) at +0x18.
    page[0x18..0x1A].copy_from_slice(&(cursor as u16).to_le_bytes());
    // last_end_lsn (u64) at +0x20.
    page[0x20..0x28].copy_from_slice(&last_end_lsn.to_le_bytes());

    mft::install_fixup(&mut page, SECTOR_SIZE, 1);
    Ok((page, last_lsn))
}

/// Parse the records out of one RCRD page. Returns the redo entries in
/// LSN order, or `None` if the page fails validation.
pub fn parse_record_page(bytes: &[u8]) -> Option<Vec<RedoEntry>> {
    if bytes.len() < LOG_PAGE_SIZE {
        return None;
    }
    let mut buf = vec![0u8; LOG_PAGE_SIZE];
    buf.copy_from_slice(&bytes[..LOG_PAGE_SIZE]);
    if &buf[0..4] != RCRD_MAGIC {
        return None;
    }
    mft::apply_fixup(&mut buf, SECTOR_SIZE).ok()?;
    let next_record_offset = u16::from_le_bytes(buf[0x18..0x1A].try_into().unwrap()) as usize;
    if next_record_offset < 0x40 || next_record_offset > buf.len() {
        return None;
    }

    let mut out = Vec::new();
    let mut cursor = 0x40usize;
    while cursor + 0x30 <= next_record_offset {
        // Read header.
        let cd_len =
            u32::from_le_bytes(buf[cursor + 0x18..cursor + 0x1C].try_into().unwrap()) as usize;
        if cd_len < 16 || cursor + 0x30 + cd_len > buf.len() {
            break;
        }
        let cd = cursor + 0x30;
        if &buf[cd..cd + 4] != PRIVATE_PAYLOAD_MAGIC {
            // Not our payload — skip the record but continue past it.
            let cd_padded = (cd_len + 7) & !7;
            cursor += 0x30 + cd_padded;
            continue;
        }
        let target_offset = u64::from_le_bytes(buf[cd + 4..cd + 12].try_into().unwrap());
        let length = u32::from_le_bytes(buf[cd + 12..cd + 16].try_into().unwrap()) as usize;
        if 16 + 2 * length > cd_len {
            break;
        }
        let redo_start = cd + 16;
        let undo_start = redo_start + length;
        let redo_bytes = buf[redo_start..redo_start + length].to_vec();
        let undo_bytes = buf[undo_start..undo_start + length].to_vec();
        out.push(RedoEntry {
            target_offset,
            redo_bytes,
            undo_bytes,
        });
        let cd_padded = (cd_len + 7) & !7;
        cursor += 0x30 + cd_padded;
    }
    Some(out)
}

/// Write a freshly-formatted, clean-shutdown $LogFile to disk.
///
/// Stamps two identical restart pages at the start of the log region;
/// the rest of the log stays zero. Both `current_lsn` fields are 0,
/// `CleanDismount` is set, so any LFS-aware consumer (kernel NTFS3,
/// ntfs-3g, `ntfsfix`) treats the volume as cleanly closed without
/// inspecting the record stream.
pub fn write_initial_logfile(
    dev: &mut dyn BlockDevice,
    logfile_offset: u64,
    log_size: u64,
) -> Result<()> {
    // Zero the entire region first.
    dev.zero_range(logfile_offset, log_size)?;
    // Stamp restart pages.
    let page = build_restart_page(0, log_size, true);
    dev.write_at(logfile_offset, &page)?;
    dev.write_at(logfile_offset + LOG_PAGE_SIZE as u64, &page)?;
    Ok(())
}

/// Read the two restart pages and return the one with the higher LSN.
/// Returns `None` if neither parses; returns `Some((view, page_index))`
/// when at least one is valid, where `page_index` ∈ {0, 1} identifies
/// which physical page won.
pub fn read_current_restart(
    dev: &mut dyn BlockDevice,
    logfile_offset: u64,
) -> Result<Option<(RestartView, u32)>> {
    let mut buf0 = vec![0u8; LOG_PAGE_SIZE];
    let mut buf1 = vec![0u8; LOG_PAGE_SIZE];
    dev.read_at(logfile_offset, &mut buf0)?;
    dev.read_at(logfile_offset + LOG_PAGE_SIZE as u64, &mut buf1)?;
    let v0 = parse_restart_page(&buf0);
    let v1 = parse_restart_page(&buf1);
    let out = match (v0, v1) {
        (Some(a), Some(b)) => {
            if a.current_lsn >= b.current_lsn {
                Some((a, 0u32))
            } else {
                Some((b, 1u32))
            }
        }
        (Some(a), None) => Some((a, 0u32)),
        (None, Some(b)) => Some((b, 1u32)),
        (None, None) => None,
    };
    Ok(out)
}

/// Walk every record page in `[logfile_offset + 2·LOG_PAGE_SIZE, end)`
/// and return the redo entries in on-disk order. Stops on the first
/// page that fails to parse.
pub fn walk_records(
    dev: &mut dyn BlockDevice,
    logfile_offset: u64,
    log_size: u64,
) -> Result<Vec<RedoEntry>> {
    let mut out = Vec::new();
    let start = logfile_offset + 2 * LOG_PAGE_SIZE as u64;
    let end = logfile_offset + log_size;
    let mut p = start;
    let mut buf = vec![0u8; LOG_PAGE_SIZE];
    while p + LOG_PAGE_SIZE as u64 <= end {
        dev.read_at(p, &mut buf)?;
        // Stop at the first non-RCRD (e.g. zero) page — that's "end of
        // log."
        if &buf[0..4] != RCRD_MAGIC {
            break;
        }
        match parse_record_page(&buf) {
            Some(entries) => out.extend(entries),
            None => break,
        }
        p += LOG_PAGE_SIZE as u64;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_page_round_trip_clean() {
        let page = build_restart_page(0xDEAD_BEEF, 64 * 1024, true);
        let view = parse_restart_page(&page).expect("must parse");
        assert!(view.is_clean());
        assert_eq!(view.current_lsn, 0xDEAD_BEEF);
        assert_eq!(view.file_size, 64 * 1024);
        assert_eq!(view.seq_number_bits, SEQ_NUMBER_BITS);
    }

    #[test]
    fn restart_page_round_trip_dirty() {
        let page = build_restart_page(0xCAFE, 64 * 1024, false);
        let view = parse_restart_page(&page).expect("must parse");
        assert!(!view.is_clean());
        assert_eq!(view.current_lsn, 0xCAFE);
    }

    #[test]
    fn record_page_round_trip_single_entry() {
        let entries = vec![RedoEntry {
            target_offset: 4096,
            redo_bytes: b"new bytes".to_vec(),
            undo_bytes: b"old bytes".to_vec(),
        }];
        let (page, last_lsn) = build_record_page(&entries, 10).unwrap();
        assert_eq!(last_lsn, 10);
        let parsed = parse_record_page(&page).expect("must parse");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].target_offset, 4096);
        assert_eq!(parsed[0].redo_bytes, b"new bytes");
        assert_eq!(parsed[0].undo_bytes, b"old bytes");
    }

    #[test]
    fn record_page_round_trip_multi_entry() {
        let entries = vec![
            RedoEntry {
                target_offset: 100,
                redo_bytes: vec![1, 2, 3],
                undo_bytes: vec![4, 5, 6],
            },
            RedoEntry {
                target_offset: 200,
                redo_bytes: vec![7; 32],
                undo_bytes: vec![8; 32],
            },
        ];
        let (page, last_lsn) = build_record_page(&entries, 100).unwrap();
        assert_eq!(last_lsn, 101);
        let parsed = parse_record_page(&page).expect("must parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].redo_bytes, vec![1, 2, 3]);
        assert_eq!(parsed[1].undo_bytes, vec![8; 32]);
    }

    #[test]
    fn lsn_packing_round_trip() {
        let lsn = pack_lsn(7, 0x1000);
        assert_eq!(lsn_seq(lsn), 7);
        let off = lsn_to_file_offset(lsn);
        assert_eq!(off, 0x1000);
    }

    #[test]
    fn restart_page_rejects_zero_bytes() {
        let buf = vec![0u8; LOG_PAGE_SIZE];
        assert!(parse_restart_page(&buf).is_none());
    }

    #[test]
    fn restart_page_rejects_short_buffer() {
        let buf = vec![0u8; 16];
        assert!(parse_restart_page(&buf).is_none());
    }
}
