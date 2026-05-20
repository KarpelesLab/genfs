//! Unit tests for the NTFS read implementation.
//!
//! We hand-craft minimal NTFS-like images (no `mkntfs` available) and
//! verify each layer in isolation: boot decode, USA fixup, attribute
//! walk, $INDEX_ROOT decode, then a full `list_path` / `open_file_reader`
//! / `read_xattrs` round trip on a single-directory image.

use super::*;
use crate::block::MemoryBackend;

const BPS: u16 = 512;
const SPC: u8 = 8; // 4 KiB clusters
const REC_SIZE: u32 = 1024; // 1 KiB MFT record

fn fake_boot(bps: u16, spc: u8, mft_lcn: u64, mft_rec: i8) -> Vec<u8> {
    let mut v = vec![0u8; 512];
    v[0..3].copy_from_slice(&[0xEB, 0x52, 0x90]);
    v[3..11].copy_from_slice(boot::NTFS_OEM);
    v[11..13].copy_from_slice(&bps.to_le_bytes());
    v[13] = spc;
    v[0x28..0x30].copy_from_slice(&1024u64.to_le_bytes());
    v[0x30..0x38].copy_from_slice(&mft_lcn.to_le_bytes());
    v[0x38..0x40].copy_from_slice(&(mft_lcn + 1).to_le_bytes());
    v[0x40] = mft_rec as u8;
    v[0x44] = 1;
    v[0x48..0x50].copy_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
    v
}

#[test]
fn decode_recognises_oem_id() {
    let buf = fake_boot(512, 8, 4, -10);
    let bs = boot::BootSector::decode(&buf).unwrap();
    assert_eq!(bs.bytes_per_sector, 512);
    assert_eq!(bs.sectors_per_cluster, 8);
    assert_eq!(bs.mft_record_size(), 1024);
    assert_eq!(bs.cluster_size(), 4096);
}

#[test]
fn decode_handles_positive_clusters_per_mft_record() {
    let buf = fake_boot(512, 8, 4, 2);
    let bs = boot::BootSector::decode(&buf).unwrap();
    assert_eq!(bs.mft_record_size(), 8192);
}

#[test]
fn decode_rejects_wrong_oem() {
    let mut buf = fake_boot(512, 8, 4, -10);
    buf[3..11].copy_from_slice(b"EXFAT   ");
    assert!(boot::BootSector::decode(&buf).is_none());
}

#[test]
fn probe_detects_ntfs() {
    let mut dev = MemoryBackend::new(4096);
    dev.write_at(0, &fake_boot(512, 8, 4, -10)).unwrap();
    assert!(probe(&mut dev).unwrap());
}

#[test]
fn fixup_roundtrip() {
    // Build a 1024-byte record where bytes 510..512 and 1022..1024 are
    // distinctive. install_fixup then apply_fixup must restore them.
    let mut buf = vec![0u8; 1024];
    buf[0..4].copy_from_slice(b"FILE");
    buf[4..6].copy_from_slice(&42u16.to_le_bytes()); // usa_offset
    buf[6..8].copy_from_slice(&3u16.to_le_bytes()); // usa_size (USN + 2 sectors)
    buf[510] = 0xAA;
    buf[511] = 0xBB;
    buf[1022] = 0xCC;
    buf[1023] = 0xDD;
    mft::install_fixup(&mut buf, 512, 0x7777);
    // The tails are now 0x77 0x77; originals are stashed in the USA.
    assert_eq!(buf[510], 0x77);
    assert_eq!(buf[511], 0x77);
    mft::apply_fixup(&mut buf, 512).unwrap();
    assert_eq!(buf[510], 0xAA);
    assert_eq!(buf[511], 0xBB);
    assert_eq!(buf[1022], 0xCC);
    assert_eq!(buf[1023], 0xDD);
}

#[test]
fn fixup_detects_torn_write() {
    let mut buf = vec![0u8; 1024];
    buf[0..4].copy_from_slice(b"FILE");
    buf[4..6].copy_from_slice(&42u16.to_le_bytes());
    buf[6..8].copy_from_slice(&3u16.to_le_bytes());
    mft::install_fixup(&mut buf, 512, 0x7777);
    // Corrupt the first sector's tail to simulate a torn write.
    buf[511] = 0x00;
    let err = mft::apply_fixup(&mut buf, 512).unwrap_err();
    assert!(matches!(err, crate::Error::InvalidImage(_)));
}

// ---------------------------------------------------------------------
// Minimal whole-volume fixture.
//
// Layout (cluster size = 4 KiB):
//   LBA 0:    boot sector
//   cluster 4 (offset 0x4000):  MFT starts here
//     record 0:  $MFT itself, with one non-resident $DATA run covering
//                clusters 4..6 (8 KiB of MFT).
//     record 5:  root dir, with $INDEX_ROOT $I30 containing the entry
//                for "hello.txt" + a tail entry, no $INDEX_ALLOCATION.
//     record 6:  hello.txt with resident $DATA = b"hi\n" and a named
//                $DATA "stream1" with b"AAAA".
//
// We write each record raw (no fixup) then call install_fixup to make
// the USAs valid.
// ---------------------------------------------------------------------

fn build_attr_header(
    type_code: u32,
    total_length: u32,
    non_resident: bool,
    name_len_u16: u8,
    name_off: u16,
    flags: u16,
    attr_id: u16,
) -> Vec<u8> {
    let mut h = vec![0u8; 16];
    h[0..4].copy_from_slice(&type_code.to_le_bytes());
    h[4..8].copy_from_slice(&total_length.to_le_bytes());
    h[8] = non_resident as u8;
    h[9] = name_len_u16;
    h[10..12].copy_from_slice(&name_off.to_le_bytes());
    h[12..14].copy_from_slice(&flags.to_le_bytes());
    h[14..16].copy_from_slice(&attr_id.to_le_bytes());
    h
}

/// Build a resident attribute (header + resident-specific fields + value).
fn build_resident_attr(type_code: u32, name_utf16: &[u8], value: &[u8], attr_id: u16) -> Vec<u8> {
    // Layout: 16 byte common header + 8 byte resident header + name (UTF16) + padding to 8 + value
    let name_len_u16 = (name_utf16.len() / 2) as u8;
    let name_off: u16 = if name_utf16.is_empty() { 0 } else { 0x18 };
    let header_block_len = 0x18 + name_utf16.len();
    let header_block_aligned = (header_block_len + 7) & !7;
    let total = header_block_aligned + value.len();
    let total = (total + 7) & !7;
    let value_offset = header_block_aligned as u16;

    let mut buf = Vec::with_capacity(total);
    let mut hdr = build_attr_header(
        type_code,
        total as u32,
        false,
        name_len_u16,
        name_off,
        0,
        attr_id,
    );
    // Resident-specific:
    let mut resident = vec![0u8; 8];
    resident[0..4].copy_from_slice(&(value.len() as u32).to_le_bytes());
    resident[4..6].copy_from_slice(&value_offset.to_le_bytes());
    resident[6] = 0;
    hdr.extend_from_slice(&resident);
    buf.extend_from_slice(&hdr);
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
fn build_non_resident_attr(
    type_code: u32,
    name_utf16: &[u8],
    runs: &[u8],
    starting_vcn: u64,
    last_vcn: u64,
    allocated: u64,
    real: u64,
    initialized: u64,
    attr_id: u16,
) -> Vec<u8> {
    // Layout: 16 byte common header + 0x30 non-resident header bytes + name + runs
    let name_len_u16 = (name_utf16.len() / 2) as u8;
    let header_block_len = 0x40 + name_utf16.len();
    let header_block_aligned = (header_block_len + 7) & !7;
    let runs_off = header_block_aligned as u16;
    let total = ((header_block_aligned + runs.len()) + 7) & !7;

    let name_off: u16 = if name_utf16.is_empty() { 0 } else { 0x40 };
    let mut hdr = build_attr_header(
        type_code,
        total as u32,
        true,
        name_len_u16,
        name_off,
        0,
        attr_id,
    );
    let mut nonresident = vec![0u8; 0x30];
    nonresident[0x00..0x08].copy_from_slice(&starting_vcn.to_le_bytes());
    nonresident[0x08..0x10].copy_from_slice(&last_vcn.to_le_bytes());
    nonresident[0x10..0x12].copy_from_slice(&runs_off.to_le_bytes());
    nonresident[0x12..0x14].copy_from_slice(&0u16.to_le_bytes()); // compression unit
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

fn build_record(record_size: usize, flags: u16, attrs: Vec<Vec<u8>>) -> Vec<u8> {
    let mut rec = vec![0u8; record_size];
    rec[0..4].copy_from_slice(b"FILE");
    // usa_offset = 0x2A (42), usa_size depends on sectors covered.
    rec[4..6].copy_from_slice(&0x2Au16.to_le_bytes());
    let sectors = record_size / BPS as usize;
    let usa_size = (sectors + 1) as u16;
    rec[6..8].copy_from_slice(&usa_size.to_le_bytes());
    // first_attr_offset = aligned past USA. USA occupies 0x2A..(0x2A + usa_size*2).
    let usa_end = 0x2A + usa_size as usize * 2;
    let first_attr_off = ((usa_end + 7) & !7) as u16; // align to 8
    rec[0x14..0x16].copy_from_slice(&first_attr_off.to_le_bytes());
    rec[0x16..0x18].copy_from_slice(&flags.to_le_bytes());

    let mut cursor = first_attr_off as usize;
    for a in &attrs {
        rec[cursor..cursor + a.len()].copy_from_slice(a);
        cursor += a.len();
    }
    // Terminator
    let term = [0xFFu8, 0xFF, 0xFF, 0xFF];
    rec[cursor..cursor + 4].copy_from_slice(&term);
    cursor += 4;

    let bytes_in_use = cursor as u32;
    rec[0x18..0x1C].copy_from_slice(&bytes_in_use.to_le_bytes());
    rec[0x1C..0x20].copy_from_slice(&(record_size as u32).to_le_bytes());

    mft::install_fixup(&mut rec, BPS as usize, 0x0001);
    rec
}

fn utf16_le(s: &str) -> Vec<u8> {
    s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect()
}

/// Build a $FILE_NAME attribute value (just the payload, no header).
fn build_file_name_value(
    parent_ref: u64,
    name: &str,
    flags: u32,
    real_size: u64,
    namespace: u8,
) -> Vec<u8> {
    let name_utf16 = utf16_le(name);
    let mut v = vec![0u8; 66 + name_utf16.len()];
    v[0..8].copy_from_slice(&parent_ref.to_le_bytes());
    // timestamps zero
    v[40..48].copy_from_slice(&real_size.to_le_bytes());
    v[48..56].copy_from_slice(&real_size.to_le_bytes());
    v[56..60].copy_from_slice(&flags.to_le_bytes());
    v[64] = (name_utf16.len() / 2) as u8;
    v[65] = namespace;
    v[66..].copy_from_slice(&name_utf16);
    v
}

/// Build an $INDEX_ROOT value with `entries` as raw entry blobs.
fn build_index_root_value(entries: &[Vec<u8>]) -> Vec<u8> {
    // 16 bytes header (indexed type + collation + index block size + cpib + padding)
    // 16 bytes index header
    // entries
    // The first 16 bytes:
    let mut v: Vec<u8> = Vec::new();
    v.extend_from_slice(&TYPE_FILE_NAME.to_le_bytes()); // indexed attr type
    v.extend_from_slice(&1u32.to_le_bytes()); // collation = filename
    v.extend_from_slice(&0u32.to_le_bytes()); // index block size (no allocation)
    v.push(0); // cpib
    v.extend_from_slice(&[0u8; 3]);

    // Index header at offset 16.
    let entries_total: usize = entries.iter().map(|e| e.len()).sum();
    let first_entry_offset = 16u32; // entries start right after the 16-byte index header
    let bytes_in_use = 16u32 + entries_total as u32;
    let bytes_allocated = bytes_in_use;
    let flags: u8 = 0; // SMALL_INDEX
    v.extend_from_slice(&first_entry_offset.to_le_bytes());
    v.extend_from_slice(&bytes_in_use.to_le_bytes());
    v.extend_from_slice(&bytes_allocated.to_le_bytes());
    v.push(flags);
    v.extend_from_slice(&[0u8; 3]);
    for e in entries {
        v.extend_from_slice(e);
    }
    v
}

/// Build an index entry holding a $FILE_NAME key. `child_vcn` adds a
/// child pointer (and the HAS_CHILD flag).
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
    let final_flags = flags
        | if child_vcn.is_some() {
            index::ENTRY_FLAG_HAS_CHILD
        } else {
            0
        };
    e[12..16].copy_from_slice(&final_flags.to_le_bytes());
    e[16..16 + key_len].copy_from_slice(file_name_value);
    if let Some(vcn) = child_vcn {
        let off = entry_len - 8;
        e[off..off + 8].copy_from_slice(&vcn.to_le_bytes());
    }
    e
}

/// Build a terminator (last) entry. Empty key, with optional child.
fn build_terminator_entry(child_vcn: Option<u64>) -> Vec<u8> {
    let mut payload_len = 16;
    let entry_len = if child_vcn.is_some() {
        payload_len += 8;
        payload_len
    } else {
        payload_len
    };
    let mut e = vec![0u8; entry_len];
    e[0..8].copy_from_slice(&0u64.to_le_bytes()); // file_ref = 0
    e[8..10].copy_from_slice(&(entry_len as u16).to_le_bytes());
    e[10..12].copy_from_slice(&0u16.to_le_bytes()); // key_len = 0
    let flags = index::ENTRY_FLAG_LAST
        | if child_vcn.is_some() {
            index::ENTRY_FLAG_HAS_CHILD
        } else {
            0
        };
    e[12..16].copy_from_slice(&flags.to_le_bytes());
    if let Some(vcn) = child_vcn {
        let off = entry_len - 8;
        e[off..off + 8].copy_from_slice(&vcn.to_le_bytes());
    }
    e
}

/// Builds a tiny but complete-enough NTFS image for the integration
/// tests. Returns (backend, mft byte offset).
fn build_tiny_image() -> MemoryBackend {
    let cluster_size = (BPS as u32) * (SPC as u32);
    let mft_lcn = 4u64;
    let mft_byte_off = mft_lcn * cluster_size as u64;

    // Total: 32 clusters = 128 KiB. Layout:
    //   cluster 4..6  : MFT (8 KiB == 8 records)
    //   cluster 8..9  : root dir's $INDEX_ALLOCATION (none, we use small index)
    //   cluster 10    : hello.txt's data... actually our $DATA is resident.
    //   cluster 12    : "stream1" named $DATA for hello.txt — resident too.
    let total_size = 32 * cluster_size as u64;
    let mut dev = MemoryBackend::new(total_size);

    // Boot sector
    let mut boot = fake_boot(BPS, SPC, mft_lcn, -10);
    dev.write_at(0, &boot[..]).unwrap();
    // Boot sector must specify the index-block-size field too; -12 = 4 KiB
    boot[0x44] = (-12i8) as u8;
    dev.write_at(0x44, &[(-12i8) as u8]).unwrap();

    // --- Record 0: $MFT ---
    // $STANDARD_INFORMATION (resident)
    let si_value = {
        let mut v = vec![0u8; 48];
        v[32..36].copy_from_slice(&0u32.to_le_bytes()); // attrs
        v
    };
    let si_attr = build_resident_attr(TYPE_STANDARD_INFORMATION, &[], &si_value, 0);
    // $FILE_NAME for $MFT (mostly cosmetic for record 0)
    let mft_fname_value = build_file_name_value(
        5,
        "$MFT",
        FileName::FLAG_DIRECTORY,
        8 * REC_SIZE as u64,
        FileName::NAMESPACE_WIN32,
    );
    let mft_fname_attr = build_resident_attr(TYPE_FILE_NAME, &[], &mft_fname_value, 1);
    // $DATA non-resident: one run, length=2 clusters, lcn=mft_lcn (4).
    // Run list: 0x11 (1-byte length, 1-byte offset) + 0x02 + 0x04 + 0x00.
    let mft_runs = vec![0x11u8, 0x02, 0x04, 0x00];
    let mft_data = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &mft_runs,
        0,
        1,
        2 * cluster_size as u64,
        8 * REC_SIZE as u64,
        8 * REC_SIZE as u64,
        2,
    );
    let rec0 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE,
        vec![si_attr.clone(), mft_fname_attr, mft_data],
    );
    dev.write_at(mft_byte_off, &rec0).unwrap();

    // --- Record 5: root directory ---
    let root_fname_value = build_file_name_value(
        5,
        ".",
        FileName::FLAG_DIRECTORY,
        0,
        FileName::NAMESPACE_WIN32,
    );
    let root_fname_attr = build_resident_attr(TYPE_FILE_NAME, &[], &root_fname_value, 1);

    // Build $INDEX_ROOT pointing at "hello.txt" (file ref = record 6
    // with sequence 1 in the high 16 bits).
    let hello_ref: u64 = 6 | (1u64 << 48);
    let hello_fn = build_file_name_value(5, "hello.txt", 0, 3, FileName::NAMESPACE_WIN32);
    let hello_entry = build_index_entry(hello_ref, &hello_fn, 0, None);
    let term_entry = build_terminator_entry(None);
    let idx_root_value = build_index_root_value(&[hello_entry, term_entry]);
    // Index root name = "$I30" (UTF-16LE).
    let i30_name = utf16_le("$I30");
    let idx_root_attr = build_resident_attr(TYPE_INDEX_ROOT, &i30_name, &idx_root_value, 2);
    let rec5 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE | mft::RecordHeader::FLAG_DIRECTORY,
        vec![si_attr.clone(), root_fname_attr, idx_root_attr],
    );
    dev.write_at(mft_byte_off + 5 * REC_SIZE as u64, &rec5)
        .unwrap();

    // --- Record 6: hello.txt ---
    let file_fname_value = build_file_name_value(5, "hello.txt", 0, 3, FileName::NAMESPACE_WIN32);
    let file_fname_attr = build_resident_attr(TYPE_FILE_NAME, &[], &file_fname_value, 1);
    let file_data_attr = build_resident_attr(TYPE_DATA, &[], b"hi\n", 2);
    let stream_name = utf16_le("stream1");
    let stream_data_attr = build_resident_attr(TYPE_DATA, &stream_name, b"AAAA", 3);
    let rec6 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE,
        vec![si_attr, file_fname_attr, file_data_attr, stream_data_attr],
    );
    dev.write_at(mft_byte_off + 6 * REC_SIZE as u64, &rec6)
        .unwrap();

    dev
}

#[test]
fn read_mft_record_zero() {
    let mut dev = build_tiny_image();
    let mut ntfs = Ntfs::open(&mut dev).unwrap();
    let mut buf = vec![0u8; REC_SIZE as usize];
    ntfs.read_mft_record(&mut dev, 0, &mut buf).unwrap();
    assert_eq!(&buf[0..4], b"FILE");
    let hdr = mft::RecordHeader::parse(&buf).unwrap();
    assert!(hdr.is_in_use());
}

#[test]
fn list_root_directory() {
    let mut dev = build_tiny_image();
    let mut ntfs = Ntfs::open(&mut dev).unwrap();
    let entries = ntfs.list_path(&mut dev, "/").unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "hello.txt");
    assert_eq!(entries[0].kind, crate::fs::EntryKind::Regular);
}

#[test]
fn read_hello_txt() {
    let mut dev = build_tiny_image();
    let mut ntfs = Ntfs::open(&mut dev).unwrap();
    let mut reader = ntfs.open_file_reader(&mut dev, "/hello.txt").unwrap();
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).unwrap();
    assert_eq!(buf, b"hi\n");
}

#[test]
fn read_xattrs_includes_dos_attrs_and_ads() {
    let mut dev = build_tiny_image();
    let mut ntfs = Ntfs::open(&mut dev).unwrap();
    let attrs = ntfs.read_xattrs(&mut dev, "/hello.txt").unwrap();
    assert!(attrs.contains_key(xattr_keys::DOS_ATTRS));
    assert!(attrs.contains_key(xattr_keys::TIMES_RAW));
    let ads_key = format!("{}stream1", xattr_keys::ADS_PREFIX);
    assert_eq!(
        attrs.get(&ads_key).map(|v| v.as_slice()),
        Some(b"AAAA" as &[u8])
    );
}

#[test]
fn lookup_path_missing_component() {
    let mut dev = build_tiny_image();
    let mut ntfs = Ntfs::open(&mut dev).unwrap();
    let err = ntfs.lookup_path(&mut dev, "/no_such_file").unwrap_err();
    assert!(matches!(err, crate::Error::InvalidImage(_)));
}

// --- Case-insensitive lookup via $UpCase ----------------------------------
//
// The tiny image has no $UpCase metadata file, so the driver should fall
// back to the identity table — lookups remain case-sensitive in that
// degraded mode. We exercise the case-folding code path by installing an
// ASCII-uppercasing UpCase directly into the cached field after open.

#[test]
fn case_insensitive_lookup_with_ascii_upcase() {
    let mut dev = build_tiny_image();
    let mut ntfs = Ntfs::open(&mut dev).unwrap();
    // Read record 0 first to bootstrap the MFT runs.
    let mut scratch = vec![0u8; REC_SIZE as usize];
    ntfs.read_mft_record(&mut dev, 0, &mut scratch).unwrap();

    // Install an ASCII-uppercasing UpCase table directly.
    let mut bytes = Vec::with_capacity(0x10000 * 2);
    for i in 0..0x10000u32 {
        let v = if (0x61..=0x7A).contains(&i) {
            i as u16 - 0x20
        } else {
            i as u16
        };
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    ntfs.upcase = Some(super::secure::UpcaseTable::from_bytes(&bytes));

    // Now "/HELLO.TXT" should resolve to the same record as "/hello.txt".
    let lower = ntfs.lookup_path(&mut dev, "/hello.txt").unwrap();
    let upper = ntfs.lookup_path(&mut dev, "/HELLO.TXT").unwrap();
    assert_eq!(lower, upper);
}

// --- $ATTRIBUTE_LIST spill ------------------------------------------------
//
// We fabricate a base record whose $DATA lives entirely in an extension
// record, referenced through an $ATTRIBUTE_LIST. The full attribute view
// is then assembled by `load_record_set` and `open_stream_by_record` must
// find $DATA across records.

#[test]
fn data_attribute_from_extension_record() {
    let cluster_size = (BPS as u32) * (SPC as u32);
    let mft_lcn = 4u64;
    let mft_byte_off = mft_lcn * cluster_size as u64;
    let total_size = 32 * cluster_size as u64;
    let mut dev = MemoryBackend::new(total_size);
    // Boot
    let boot = fake_boot(BPS, SPC, mft_lcn, -10);
    dev.write_at(0, &boot[..]).unwrap();
    dev.write_at(0x44, &[(-12i8) as u8]).unwrap();

    let si_value = vec![0u8; 48];
    let si_attr = build_resident_attr(TYPE_STANDARD_INFORMATION, &[], &si_value, 0);

    // Record 0: $MFT
    let mft_fname_value = build_file_name_value(
        5,
        "$MFT",
        FileName::FLAG_DIRECTORY,
        8 * REC_SIZE as u64,
        FileName::NAMESPACE_WIN32,
    );
    let mft_fname_attr = build_resident_attr(TYPE_FILE_NAME, &[], &mft_fname_value, 1);
    let mft_runs = vec![0x11u8, 0x02, 0x04, 0x00];
    let mft_data = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &mft_runs,
        0,
        1,
        2 * cluster_size as u64,
        8 * REC_SIZE as u64,
        8 * REC_SIZE as u64,
        2,
    );
    let rec0 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE,
        vec![si_attr.clone(), mft_fname_attr, mft_data],
    );
    dev.write_at(mft_byte_off, &rec0).unwrap();

    // Record 5: root directory pointing at "split.dat" (record 6).
    let root_fname_value = build_file_name_value(
        5,
        ".",
        FileName::FLAG_DIRECTORY,
        0,
        FileName::NAMESPACE_WIN32,
    );
    let root_fname_attr = build_resident_attr(TYPE_FILE_NAME, &[], &root_fname_value, 1);
    let split_ref: u64 = 6 | (1u64 << 48);
    let split_fn = build_file_name_value(5, "split.dat", 0, 6, FileName::NAMESPACE_WIN32);
    let split_entry = build_index_entry(split_ref, &split_fn, 0, None);
    let term_entry = build_terminator_entry(None);
    let idx_root_value = build_index_root_value(&[split_entry, term_entry]);
    let i30_name = utf16_le("$I30");
    let idx_root_attr = build_resident_attr(TYPE_INDEX_ROOT, &i30_name, &idx_root_value, 2);
    let rec5 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE | mft::RecordHeader::FLAG_DIRECTORY,
        vec![si_attr.clone(), root_fname_attr, idx_root_attr],
    );
    dev.write_at(mft_byte_off + 5 * REC_SIZE as u64, &rec5)
        .unwrap();

    // Record 6: base record. Holds $SI + $FILE_NAME + $ATTRIBUTE_LIST.
    // The $ATTRIBUTE_LIST points at record 7 for $DATA.
    let file_fname_value = build_file_name_value(5, "split.dat", 0, 6, FileName::NAMESPACE_WIN32);
    let file_fname_attr = build_resident_attr(TYPE_FILE_NAME, &[], &file_fname_value, 1);

    // Build $ATTRIBUTE_LIST entry: one row pointing at record 7 for $DATA.
    let mut alist_value: Vec<u8> = Vec::new();
    let entry_len: u16 = 0x20;
    alist_value.extend_from_slice(&TYPE_DATA.to_le_bytes()); // type
    alist_value.extend_from_slice(&entry_len.to_le_bytes()); // entry_len
    alist_value.push(0); // name_len
    alist_value.push(0x1A); // name_off
    alist_value.extend_from_slice(&0u64.to_le_bytes()); // starting_vcn
    let rec7_ref: u64 = 7 | (1u64 << 48);
    alist_value.extend_from_slice(&rec7_ref.to_le_bytes()); // mft ref to rec 7
    alist_value.extend_from_slice(&3u16.to_le_bytes()); // attr id
    while alist_value.len() < entry_len as usize {
        alist_value.push(0);
    }
    let alist_attr = build_resident_attr(TYPE_ATTRIBUTE_LIST, &[], &alist_value, 2);

    let rec6 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE,
        vec![si_attr.clone(), file_fname_attr, alist_attr],
    );
    dev.write_at(mft_byte_off + 6 * REC_SIZE as u64, &rec6)
        .unwrap();

    // Record 7: extension record carrying the actual $DATA = b"hello!".
    // base_record_ref points back at record 6 (low 48 bits) seq 1.
    let data_attr = build_resident_attr(TYPE_DATA, &[], b"hello!", 3);
    let mut rec7 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE,
        vec![data_attr],
    );
    // Patch base_record_ref at 0x20..0x28 to point at record 6.
    let base_ref: u64 = 6 | (1u64 << 48);
    rec7[0x20..0x28].copy_from_slice(&base_ref.to_le_bytes());
    // Re-install fixup since we touched bytes after build_record installed it.
    mft::install_fixup(&mut rec7, BPS as usize, 0x0001);
    dev.write_at(mft_byte_off + 7 * REC_SIZE as u64, &rec7)
        .unwrap();

    let mut ntfs = Ntfs::open(&mut dev).unwrap();
    let mut reader = ntfs.open_file_reader(&mut dev, "/split.dat").unwrap();
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).unwrap();
    assert_eq!(buf, b"hello!");
}

// --- LZNT1 compressed $DATA ----------------------------------------------
//
// Build a synthetic CU of 4 clusters (16 KiB at 4 KiB clusters). The CU's
// run list has 1 real cluster of LZNT1 data and 3 sparse clusters; the
// decoder should produce 6 bytes of output ("ABCABC"), then zero-fill to
// `real_size`.

#[test]
fn compressed_data_decodes_via_cu() {
    let cluster_size = (BPS as u32) * (SPC as u32);
    let mft_lcn = 4u64;
    let mft_byte_off = mft_lcn * cluster_size as u64;
    let total_size = 64 * cluster_size as u64;
    let mut dev = MemoryBackend::new(total_size);
    let boot = fake_boot(BPS, SPC, mft_lcn, -10);
    dev.write_at(0, &boot[..]).unwrap();
    dev.write_at(0x44, &[(-12i8) as u8]).unwrap();

    // Build a compressed CU payload: one chunk → "ABCABC".
    // (See compression::tests::decompresses_back_reference for the encoding.)
    let chunk_payload = vec![0x08u8, b'A', b'B', b'C', 0x00, 0x20];
    let chunk_len_minus_1 = chunk_payload.len() as u16 - 1;
    let header = 0xB000u16 | chunk_len_minus_1;
    let mut compressed = header.to_le_bytes().to_vec();
    compressed.extend_from_slice(&chunk_payload);
    // The driver expects one cluster of compressed data; pad to cluster.
    while compressed.len() < cluster_size as usize {
        compressed.push(0);
    }
    // Drop the data into cluster 20.
    let data_lcn = 20u64;
    dev.write_at(data_lcn * cluster_size as u64, &compressed)
        .unwrap();

    let si_value = vec![0u8; 48];
    let si_attr = build_resident_attr(TYPE_STANDARD_INFORMATION, &[], &si_value, 0);

    // Record 0: $MFT
    let mft_fname_attr = build_resident_attr(
        TYPE_FILE_NAME,
        &[],
        &build_file_name_value(
            5,
            "$MFT",
            FileName::FLAG_DIRECTORY,
            8 * REC_SIZE as u64,
            FileName::NAMESPACE_WIN32,
        ),
        1,
    );
    let mft_runs = vec![0x11u8, 0x02, 0x04, 0x00];
    let mft_data = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &mft_runs,
        0,
        1,
        2 * cluster_size as u64,
        8 * REC_SIZE as u64,
        8 * REC_SIZE as u64,
        2,
    );
    let rec0 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE,
        vec![si_attr.clone(), mft_fname_attr, mft_data],
    );
    dev.write_at(mft_byte_off, &rec0).unwrap();

    // Record 5: root dir indexing "lz.dat" → record 6.
    let lz_ref: u64 = 6 | (1u64 << 48);
    let lz_fn = build_file_name_value(5, "lz.dat", 0, 6, FileName::NAMESPACE_WIN32);
    let lz_entry = build_index_entry(lz_ref, &lz_fn, 0, None);
    let term = build_terminator_entry(None);
    let idx_root_value = build_index_root_value(&[lz_entry, term]);
    let i30_name = utf16_le("$I30");
    let idx_root_attr = build_resident_attr(TYPE_INDEX_ROOT, &i30_name, &idx_root_value, 2);
    let rec5 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE | mft::RecordHeader::FLAG_DIRECTORY,
        vec![
            si_attr.clone(),
            build_resident_attr(
                TYPE_FILE_NAME,
                &[],
                &build_file_name_value(
                    5,
                    ".",
                    FileName::FLAG_DIRECTORY,
                    0,
                    FileName::NAMESPACE_WIN32,
                ),
                1,
            ),
            idx_root_attr,
        ],
    );
    dev.write_at(mft_byte_off + 5 * REC_SIZE as u64, &rec5)
        .unwrap();

    // Record 6: lz.dat. Compressed $DATA, compression_unit=2 (1 << 2 = 4
    // clusters per CU). Run list: 1 real cluster at LCN 20, then 3 sparse
    // clusters. real_size = 6 (we only emit "ABCABC").
    // Run encoding: 0x11 0x01 0x14 (len=1, offset=+20), 0x01 0x03 (len=3 sparse), 0x00.
    let runs = vec![0x11u8, 0x01, 0x14, 0x01, 0x03, 0x00];
    let mut data_attr = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &runs,
        0,
        3,
        4 * cluster_size as u64,
        6,
        6,
        3,
    );
    // Set compression flag + compression_unit=2 inside the header.
    // Flags are at offset 12 of the attribute header.
    data_attr[12..14].copy_from_slice(&ATTR_FLAG_COMPRESSED.to_le_bytes());
    // compression_unit lives at attr_start + 0x22 (u16 low byte).
    data_attr[0x22] = 2;
    data_attr[0x23] = 0;
    let rec6 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE,
        vec![
            si_attr.clone(),
            build_resident_attr(
                TYPE_FILE_NAME,
                &[],
                &build_file_name_value(5, "lz.dat", 0, 6, FileName::NAMESPACE_WIN32),
                1,
            ),
            data_attr,
        ],
    );
    dev.write_at(mft_byte_off + 6 * REC_SIZE as u64, &rec6)
        .unwrap();

    let mut ntfs = Ntfs::open(&mut dev).unwrap();
    let mut reader = ntfs.open_file_reader(&mut dev, "/lz.dat").unwrap();
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).unwrap();
    assert_eq!(buf, b"ABCABC");
}

// --- Encrypted $DATA is rejected ----------------------------------------

#[test]
fn encrypted_data_is_unsupported() {
    let cluster_size = (BPS as u32) * (SPC as u32);
    let mft_lcn = 4u64;
    let mft_byte_off = mft_lcn * cluster_size as u64;
    let total_size = 32 * cluster_size as u64;
    let mut dev = MemoryBackend::new(total_size);
    let boot = fake_boot(BPS, SPC, mft_lcn, -10);
    dev.write_at(0, &boot[..]).unwrap();
    dev.write_at(0x44, &[(-12i8) as u8]).unwrap();

    let si_value = vec![0u8; 48];
    let si_attr = build_resident_attr(TYPE_STANDARD_INFORMATION, &[], &si_value, 0);

    let mft_fname_attr = build_resident_attr(
        TYPE_FILE_NAME,
        &[],
        &build_file_name_value(
            5,
            "$MFT",
            FileName::FLAG_DIRECTORY,
            8 * REC_SIZE as u64,
            FileName::NAMESPACE_WIN32,
        ),
        1,
    );
    let mft_runs = vec![0x11u8, 0x02, 0x04, 0x00];
    let mft_data = build_non_resident_attr(
        TYPE_DATA,
        &[],
        &mft_runs,
        0,
        1,
        2 * cluster_size as u64,
        8 * REC_SIZE as u64,
        8 * REC_SIZE as u64,
        2,
    );
    let rec0 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE,
        vec![si_attr.clone(), mft_fname_attr, mft_data],
    );
    dev.write_at(mft_byte_off, &rec0).unwrap();

    let efs_ref: u64 = 6 | (1u64 << 48);
    let efs_fn = build_file_name_value(5, "efs.dat", 0, 0, FileName::NAMESPACE_WIN32);
    let efs_entry = build_index_entry(efs_ref, &efs_fn, 0, None);
    let term = build_terminator_entry(None);
    let idx_root_value = build_index_root_value(&[efs_entry, term]);
    let i30_name = utf16_le("$I30");
    let idx_root_attr = build_resident_attr(TYPE_INDEX_ROOT, &i30_name, &idx_root_value, 2);
    let rec5 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE | mft::RecordHeader::FLAG_DIRECTORY,
        vec![
            si_attr.clone(),
            build_resident_attr(
                TYPE_FILE_NAME,
                &[],
                &build_file_name_value(
                    5,
                    ".",
                    FileName::FLAG_DIRECTORY,
                    0,
                    FileName::NAMESPACE_WIN32,
                ),
                1,
            ),
            idx_root_attr,
        ],
    );
    dev.write_at(mft_byte_off + 5 * REC_SIZE as u64, &rec5)
        .unwrap();

    // Record 6: efs.dat with ATTR_FLAG_ENCRYPTED set on $DATA.
    let mut data_attr = build_resident_attr(TYPE_DATA, &[], b"AAAA", 3);
    data_attr[12..14].copy_from_slice(&ATTR_FLAG_ENCRYPTED.to_le_bytes());
    let rec6 = build_record(
        REC_SIZE as usize,
        mft::RecordHeader::FLAG_IN_USE,
        vec![
            si_attr.clone(),
            build_resident_attr(
                TYPE_FILE_NAME,
                &[],
                &build_file_name_value(5, "efs.dat", 0, 0, FileName::NAMESPACE_WIN32),
                1,
            ),
            data_attr,
        ],
    );
    dev.write_at(mft_byte_off + 6 * REC_SIZE as u64, &rec6)
        .unwrap();

    let mut ntfs = Ntfs::open(&mut dev).unwrap();
    let result = ntfs.open_file_reader(&mut dev, "/efs.dat");
    match result {
        Ok(_) => panic!("expected Unsupported error for EFS-encrypted file"),
        Err(crate::Error::Unsupported(msg)) => assert!(msg.contains("EFS")),
        Err(other) => panic!("expected Unsupported, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// Writer tests.
//
// These format a fresh NTFS volume on an in-memory backend, exercise
// create_*, then re-open the volume and verify the read path can walk
// what was written.
// ---------------------------------------------------------------------

use super::format::FormatOpts;
use crate::fs::{FileMeta, FileSource};

fn fresh_volume(size: u64) -> (MemoryBackend, Ntfs) {
    let mut dev = MemoryBackend::new(size);
    let opts = FormatOpts {
        volume_label: "fstool-test".to_string(),
        ..Default::default()
    };
    let ntfs = Ntfs::format(&mut dev, &opts).unwrap();
    (dev, ntfs)
}

#[test]
fn writer_format_then_open_reads_boot_sector() {
    let (mut dev, _ntfs) = fresh_volume(8 * 1024 * 1024);
    // Re-open from device — verifies boot sector probe-able / decodable.
    assert!(probe(&mut dev).unwrap());
    let ntfs2 = Ntfs::open(&mut dev).unwrap();
    assert_eq!(ntfs2.cluster_size(), 4096);
    assert_eq!(ntfs2.mft_record_size(), 1024);
}

#[test]
fn writer_format_volume_has_root_directory() {
    let (mut dev, mut ntfs) = fresh_volume(8 * 1024 * 1024);
    let entries = ntfs.list_path(&mut dev, "/").unwrap();
    // Freshly formatted root should be empty.
    assert!(entries.is_empty());
}

#[test]
fn writer_creates_small_file_resident_data() {
    let (mut dev, mut ntfs) = fresh_volume(8 * 1024 * 1024);
    ntfs.create_file(
        &mut dev,
        "/hello.txt",
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(b"hi\n".to_vec())),
            len: 3,
        },
        FileMeta::default(),
    )
    .unwrap();
    ntfs.flush(&mut dev).unwrap();

    // Re-open and read.
    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();
    let entries = ntfs2.list_path(&mut dev, "/").unwrap();
    assert!(entries.iter().any(|e| e.name == "hello.txt"));
    let mut r = ntfs2.open_file_reader(&mut dev, "/hello.txt").unwrap();
    let mut buf = Vec::new();
    r.read_to_end(&mut buf).unwrap();
    assert_eq!(buf, b"hi\n");
}

#[test]
fn writer_creates_large_file_non_resident_data() {
    let (mut dev, mut ntfs) = fresh_volume(16 * 1024 * 1024);
    // 8000 bytes is larger than the resident budget and forces a
    // non-resident $DATA with a single cluster run.
    let payload: Vec<u8> = (0..8000).map(|i| (i & 0xFF) as u8).collect();
    ntfs.create_file(
        &mut dev,
        "/big.bin",
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(payload.clone())),
            len: payload.len() as u64,
        },
        FileMeta::default(),
    )
    .unwrap();
    ntfs.flush(&mut dev).unwrap();

    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();
    let mut r = ntfs2.open_file_reader(&mut dev, "/big.bin").unwrap();
    let mut buf = Vec::new();
    r.read_to_end(&mut buf).unwrap();
    assert_eq!(buf, payload);
}

#[test]
fn writer_creates_directory() {
    let (mut dev, mut ntfs) = fresh_volume(8 * 1024 * 1024);
    ntfs.create_dir(&mut dev, "/sub", FileMeta::default())
        .unwrap();
    ntfs.create_file(
        &mut dev,
        "/sub/note.txt",
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(b"x".to_vec())),
            len: 1,
        },
        FileMeta::default(),
    )
    .unwrap();
    ntfs.flush(&mut dev).unwrap();

    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();
    let root_entries = ntfs2.list_path(&mut dev, "/").unwrap();
    assert!(root_entries.iter().any(|e| e.name == "sub"));
    let sub_entries = ntfs2.list_path(&mut dev, "/sub").unwrap();
    assert!(sub_entries.iter().any(|e| e.name == "note.txt"));
}

#[test]
fn writer_creates_symlink() {
    let (mut dev, mut ntfs) = fresh_volume(8 * 1024 * 1024);
    ntfs.create_symlink(&mut dev, "/link", "target.txt", FileMeta::default())
        .unwrap();
    ntfs.flush(&mut dev).unwrap();

    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();
    let entries = ntfs2.list_path(&mut dev, "/").unwrap();
    assert!(entries.iter().any(|e| e.name == "link"));
    // Reparse data should be present on the entry.
    let xattrs = ntfs2.read_xattrs(&mut dev, "/link").unwrap();
    assert!(xattrs.contains_key(xattr_keys::REPARSE));
}

#[test]
fn writer_refuses_device_creation() {
    let (mut dev, mut ntfs) = fresh_volume(8 * 1024 * 1024);
    let err = ntfs
        .create_device(
            &mut dev,
            "/dev/null",
            crate::fs::DeviceKind::Char,
            1,
            3,
            FileMeta::default(),
        )
        .unwrap_err();
    assert!(matches!(err, crate::Error::Unsupported(_)));
}

#[test]
fn writer_create_without_format_errors() {
    // Open an existing read-only image (the tiny one) and try to create.
    let mut dev = build_tiny_image();
    let mut ntfs = Ntfs::open(&mut dev).unwrap();
    let err = ntfs
        .create_file(
            &mut dev,
            "/new.txt",
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"x".to_vec())),
                len: 1,
            },
            FileMeta::default(),
        )
        .unwrap_err();
    assert!(matches!(err, crate::Error::Unsupported(_)));
}

#[test]
fn writer_dir_promotes_to_index_allocation() {
    // Add enough entries to a directory that the $INDEX_ROOT overflows
    // its 512-byte budget and gets promoted to $INDEX_ALLOCATION.
    let (mut dev, mut ntfs) = fresh_volume(16 * 1024 * 1024);
    for i in 0..16 {
        let path = format!("/file_{i:02}.txt");
        ntfs.create_file(
            &mut dev,
            &path,
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"x".to_vec())),
                len: 1,
            },
            FileMeta::default(),
        )
        .unwrap();
    }
    ntfs.flush(&mut dev).unwrap();
    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();
    let entries = ntfs2.list_path(&mut dev, "/").unwrap();
    let names: std::collections::HashSet<String> = entries.iter().map(|e| e.name.clone()).collect();
    for i in 0..16 {
        assert!(
            names.contains(&format!("file_{i:02}.txt")),
            "missing file_{i:02}.txt"
        );
    }
}

#[test]
fn writer_streams_large_file_through_scratch_buffer() {
    // 200 KiB file forces multiple scratch buffers worth of streaming.
    let (mut dev, mut ntfs) = fresh_volume(64 * 1024 * 1024);
    let size = 200 * 1024;
    let payload: Vec<u8> = (0..size).map(|i| ((i * 7) & 0xFF) as u8).collect();
    ntfs.create_file(
        &mut dev,
        "/stream.bin",
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(payload.clone())),
            len: payload.len() as u64,
        },
        FileMeta::default(),
    )
    .unwrap();
    ntfs.flush(&mut dev).unwrap();
    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();
    let mut r = ntfs2.open_file_reader(&mut dev, "/stream.bin").unwrap();
    let mut buf = Vec::new();
    r.read_to_end(&mut buf).unwrap();
    assert_eq!(buf.len(), payload.len());
    assert_eq!(buf, payload);
}

#[test]
fn writer_zero_length_file() {
    let (mut dev, mut ntfs) = fresh_volume(8 * 1024 * 1024);
    ntfs.create_file(
        &mut dev,
        "/empty.txt",
        FileSource::Zero(0),
        FileMeta::default(),
    )
    .unwrap();
    ntfs.flush(&mut dev).unwrap();
    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();
    let entries = ntfs2.list_path(&mut dev, "/").unwrap();
    assert!(entries.iter().any(|e| e.name == "empty.txt"));
    let mut r = ntfs2.open_file_reader(&mut dev, "/empty.txt").unwrap();
    let mut buf = Vec::new();
    r.read_to_end(&mut buf).unwrap();
    assert_eq!(buf, Vec::<u8>::new());
}

#[test]
fn writer_nested_directories() {
    let (mut dev, mut ntfs) = fresh_volume(16 * 1024 * 1024);
    ntfs.create_dir(&mut dev, "/a", FileMeta::default())
        .unwrap();
    ntfs.create_dir(&mut dev, "/a/b", FileMeta::default())
        .unwrap();
    ntfs.create_dir(&mut dev, "/a/b/c", FileMeta::default())
        .unwrap();
    ntfs.create_file(
        &mut dev,
        "/a/b/c/deep.txt",
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(b"deep!".to_vec())),
            len: 5,
        },
        FileMeta::default(),
    )
    .unwrap();
    ntfs.flush(&mut dev).unwrap();
    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();
    let entries = ntfs2.list_path(&mut dev, "/a/b/c").unwrap();
    assert!(entries.iter().any(|e| e.name == "deep.txt"));
    let mut r = ntfs2.open_file_reader(&mut dev, "/a/b/c/deep.txt").unwrap();
    let mut buf = Vec::new();
    r.read_to_end(&mut buf).unwrap();
    assert_eq!(buf, b"deep!");
}

#[test]
fn writer_format_emits_upcase_table() {
    let (mut dev, mut ntfs) = fresh_volume(8 * 1024 * 1024);
    // After format, $UpCase record 10 should exist and have a non-trivial
    // $DATA stream that round-trips through the reader.
    let mut buf = vec![0u8; ntfs.mft_record_size() as usize];
    ntfs.read_mft_record(&mut dev, MFT_RECORD_UPCASE, &mut buf)
        .unwrap();
    let hdr = mft::RecordHeader::parse(&buf).unwrap();
    assert!(hdr.is_in_use());
}

#[test]
fn writer_root_index_contains_system_files() {
    // `format()` populates the root's `$I30` with index entries for every
    // canonical system MFT record (0..=15 minus the root itself). The
    // cross-FS view filters `$`-names out of `list_path("/")`, but the
    // on-disk index does carry them — verifies the layout that ntfs-3g
    // expects.
    let (mut dev, mut ntfs) = fresh_volume(16 * 1024 * 1024);
    ntfs.flush(&mut dev).unwrap();

    // Re-open and inspect the raw root directory entries.
    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();
    let raw = ntfs2.read_directory(&mut dev, MFT_RECORD_ROOT).unwrap();
    let names: std::collections::HashSet<String> = raw
        .iter()
        .filter_map(|e| e.file_name.as_ref().map(|f| f.name.clone()))
        .collect();

    // Every canonical NTFS system file should be indexed in the root.
    for expected in &[
        "$MFT", "$MFTMirr", "$LogFile", "$Volume", "$AttrDef", "$Bitmap", "$Boot", "$BadClus",
        "$Secure", "$UpCase", "$Extend",
    ] {
        assert!(
            names.contains(*expected),
            "expected root $I30 to index {expected:?}, got {:?}",
            names
        );
    }
    // The root must NOT index itself.
    assert!(
        !raw.iter()
            .any(|e| (e.file_ref & 0x0000_FFFF_FFFF_FFFF) == MFT_RECORD_ROOT),
        "root must not self-reference in its own $I30"
    );

    // The cross-FS `list_path("/")` must hide these system files so the
    // generic walker keeps seeing a clean user-facing view.
    let user_view = ntfs2.list_path(&mut dev, "/").unwrap();
    for entry in &user_view {
        assert!(
            !entry.name.starts_with('$'),
            "list_path(\"/\") leaked system file {:?}",
            entry.name
        );
    }
}

#[test]
fn writer_root_index_keeps_system_files_after_user_files_added() {
    // Add several user files to force an $INDEX_ROOT → $INDEX_ALLOCATION
    // promotion. The system-file entries planted by `format()` must
    // survive promotion and remain visible in the root's index.
    let (mut dev, mut ntfs) = fresh_volume(16 * 1024 * 1024);
    for i in 0..10 {
        let path = format!("/u_{i}.txt");
        ntfs.create_file(
            &mut dev,
            &path,
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"x".to_vec())),
                len: 1,
            },
            FileMeta::default(),
        )
        .unwrap();
    }
    ntfs.flush(&mut dev).unwrap();

    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();
    let raw = ntfs2.read_directory(&mut dev, MFT_RECORD_ROOT).unwrap();
    let names: std::collections::HashSet<String> = raw
        .iter()
        .filter_map(|e| e.file_name.as_ref().map(|f| f.name.clone()))
        .collect();
    for expected in &["$MFT", "$Volume", "$Bitmap", "$UpCase", "$Extend"] {
        assert!(
            names.contains(*expected),
            "post-promotion root $I30 missing {expected:?}"
        );
    }
    for i in 0..10 {
        let want = format!("u_{i}.txt");
        assert!(names.contains(&want), "user file {want:?} missing");
    }
}

#[test]
fn open_file_ro_random_seek_ntfs() {
    use crate::fs::Filesystem;
    use std::io::{Read, Seek, SeekFrom};

    let (mut dev, mut ntfs) = fresh_volume(8 * 1024 * 1024);
    // Large enough that the $DATA goes non-resident (one cluster is 4 KiB).
    let data: Vec<u8> = (0..20_000u32).map(|i| (i & 0xFF) as u8).collect();
    ntfs.create_file(
        &mut dev,
        "/ro.bin",
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(data.clone())),
            len: data.len() as u64,
        },
        FileMeta::default(),
    )
    .unwrap();
    ntfs.flush(&mut dev).unwrap();

    // Reopen — the read-only path doesn't need writer state.
    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();
    let mut h = ntfs2
        .open_file_ro(&mut dev, std::path::Path::new("/ro.bin"))
        .expect("open_file_ro");
    assert_eq!(h.len(), data.len() as u64);
    assert!(!h.is_empty());

    h.seek(SeekFrom::Start(9000)).unwrap();
    let mut buf = [0u8; 256];
    h.read_exact(&mut buf).unwrap();
    assert_eq!(&buf[..], &data[9000..9256]);

    h.seek(SeekFrom::Start(50)).unwrap();
    let mut buf2 = [0u8; 100];
    h.read_exact(&mut buf2).unwrap();
    assert_eq!(&buf2[..], &data[50..150]);
}

/// Helper: extract the `security_id` from `$STANDARD_INFORMATION` of an
/// MFT record. Returns `0` when the SI value isn't long enough to carry
/// the NTFS 3.0+ extension (i.e. the file was stamped with the legacy
/// 48-byte form).
fn read_security_id_for_record(ntfs: &mut Ntfs, dev: &mut MemoryBackend, rec_no: u64) -> u32 {
    let mut buf = vec![0u8; ntfs.mft_record_size() as usize];
    ntfs.read_mft_record(dev, rec_no, &mut buf).unwrap();
    let hdr = mft::RecordHeader::parse(&buf).unwrap();
    for attr_res in AttributeIter::new(&buf, hdr.first_attribute_offset as usize) {
        let attr = attr_res.unwrap();
        if attr.type_code != TYPE_STANDARD_INFORMATION {
            continue;
        }
        if let AttributeKind::Resident { value, .. } = attr.kind {
            if value.len() >= 0x38 {
                return u32::from_le_bytes(value[0x34..0x38].try_into().unwrap());
            }
            return 0;
        }
    }
    panic!("record {rec_no} has no resident $STANDARD_INFORMATION");
}

#[test]
fn writer_format_emits_multiple_security_descriptors() {
    // Multi-SD verification:
    //   * Format a fresh image.
    //   * Add at least one user file.
    //   * On reopen, $Secure:$SDS must carry >= 2 distinct (hash, security_id)
    //     pairs (User vs. System).
    //   * System records (e.g. $MFT at record 0) must have SI.security_id
    //     pointing at the System SD (FIRST_SECURITY_ID + 1).
    //   * User-visible files / directories (the root, and /u.txt) must
    //     have SI.security_id pointing at the User SD (FIRST_SECURITY_ID).
    use super::format::{FIRST_SECURITY_ID, security_id_for};
    use super::secure::SecurityClass;

    let (mut dev, mut ntfs) = fresh_volume(8 * 1024 * 1024);
    ntfs.create_file(
        &mut dev,
        "/u.txt",
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(b"hi\n".to_vec())),
            len: 3,
        },
        FileMeta::default(),
    )
    .unwrap();
    ntfs.flush(&mut dev).unwrap();

    let mut ntfs2 = Ntfs::open(&mut dev).unwrap();

    // Pull the entire $SDS stream and scan its SDS-entry headers (20 bytes
    // each, padded to a 16-byte boundary). We expect at least two distinct
    // entries with different (hash, security_id) pairs — one User, one
    // System.
    let mut sds = Vec::new();
    {
        let mut r = ntfs2
            .open_stream_by_record(&mut dev, MFT_RECORD_SECURE, "$SDS")
            .unwrap();
        r.read_to_end(&mut sds).unwrap();
    }

    let mut entries: Vec<(u32, u32, u32)> = Vec::new(); // (hash, security_id, size)
    let mut off = 0usize;
    while off + 0x14 <= sds.len() {
        let hash = u32::from_le_bytes(sds[off..off + 4].try_into().unwrap());
        let sid = u32::from_le_bytes(sds[off + 4..off + 8].try_into().unwrap());
        let _entry_off = u64::from_le_bytes(sds[off + 8..off + 16].try_into().unwrap());
        let size = u32::from_le_bytes(sds[off + 16..off + 20].try_into().unwrap());
        if size == 0 || (size as usize) < 0x14 {
            break;
        }
        if sid == 0 {
            // Padding region or trailing zeros — end of stream.
            break;
        }
        entries.push((hash, sid, size));
        // Advance by `size` and pad to 16-byte boundary.
        let advance = ((size as usize) + 0x0F) & !0x0F;
        off += advance;
    }

    assert!(
        entries.len() >= 2,
        "expected >= 2 SDS entries, got {}: {:?}",
        entries.len(),
        entries
    );

    // Collect distinct (hash, security_id) pairs and assert >= 2.
    use std::collections::HashSet;
    let distinct: HashSet<(u32, u32)> = entries.iter().map(|&(h, s, _)| (h, s)).collect();
    assert!(
        distinct.len() >= 2,
        "expected >= 2 distinct (hash, security_id) pairs in $SDS, got {distinct:?}"
    );

    // The two ids we currently emit are User (0x100) and System (0x101).
    let user_id = security_id_for(SecurityClass::User);
    let system_id = security_id_for(SecurityClass::System);
    assert_eq!(user_id, FIRST_SECURITY_ID);
    assert_eq!(system_id, FIRST_SECURITY_ID + 1);
    let ids: HashSet<u32> = entries.iter().map(|&(_, s, _)| s).collect();
    assert!(
        ids.contains(&user_id),
        "expected User security_id {user_id:#x} in $SDS, got {ids:?}"
    );
    assert!(
        ids.contains(&system_id),
        "expected System security_id {system_id:#x} in $SDS, got {ids:?}"
    );

    // The two distinct ids must hash to different values — otherwise the
    // catalogue collapsed back to a single descriptor.
    let user_hash = entries
        .iter()
        .find(|(_, s, _)| *s == user_id)
        .map(|(h, _, _)| *h)
        .unwrap();
    let system_hash = entries
        .iter()
        .find(|(_, s, _)| *s == system_id)
        .map(|(h, _, _)| *h)
        .unwrap();
    assert_ne!(
        user_hash, system_hash,
        "User and System SDs unexpectedly share a hash — catalogue collapsed?"
    );

    // System records must point at the System SD.
    let mft_sid = read_security_id_for_record(&mut ntfs2, &mut dev, MFT_RECORD_MFT);
    assert_eq!(
        mft_sid, system_id,
        "$MFT (record 0) should carry System security_id"
    );
    let secure_sid = read_security_id_for_record(&mut ntfs2, &mut dev, MFT_RECORD_SECURE);
    assert_eq!(
        secure_sid, system_id,
        "$Secure (record 9) should carry System security_id"
    );

    // Root and the user file must point at the User SD.
    let root_sid = read_security_id_for_record(&mut ntfs2, &mut dev, MFT_RECORD_ROOT);
    assert_eq!(
        root_sid, user_id,
        "root directory should carry User security_id"
    );
    let user_rec = ntfs2.lookup_path(&mut dev, "/u.txt").unwrap();
    let user_sid = read_security_id_for_record(&mut ntfs2, &mut dev, user_rec);
    assert_eq!(user_sid, user_id, "/u.txt should carry User security_id");

    // Cross-check: the resolve_security_descriptor path through $SII must
    // produce the same SD blob that build_security_descriptor(class) produces
    // for the User class — we exercise it through read_xattrs on the user
    // file.
    let xa_user = ntfs2.read_xattrs(&mut dev, "/u.txt").unwrap();
    let xa_user_sd = xa_user
        .get(xattr_keys::SECURITY)
        .expect("user file should carry a resolved security descriptor");
    let expected_user_sd = super::format::build_security_descriptor(SecurityClass::User);
    assert_eq!(
        xa_user_sd, &expected_user_sd,
        "resolved User SD differs from build_security_descriptor(User)"
    );
}
