//! exFAT directory entries — 32-byte typed records.
//!
//! Each "file" or "directory" in a parent directory is described by a
//! *set* of consecutive entries: one primary entry (FileDirectoryEntry,
//! 0x85) followed by `SecondaryCount` secondary entries, of which the
//! first is a StreamExtension (0xC0) and the remainder are FileName
//! entries (0xC1). Other entry types tracked here are AllocationBitmap
//! (0x81), UpcaseTable (0x82), and VolumeLabel (0x83).
//!
//! ## Common layout (offsets within a 32-byte entry)
//!
//! ```text
//!     0   1  EntryType  (bit 7 = InUse; bit 6 = Secondary; bit 5 = Critical)
//!     1  31  type-specific
//! ```
//!
//! ### FileDirectoryEntry (0x85)
//!
//! ```text
//!     0   1  EntryType = 0x85
//!     1   1  SecondaryCount        (>= 2: stream + name(s))
//!     2   2  SetChecksum
//!     4   2  FileAttributes        (bit 4 = directory)
//!     6   2  Reserved1
//!     8   4  CreateTimestamp
//!    12   4  LastModifiedTimestamp
//!    16   4  LastAccessedTimestamp
//!    20   1  Create10msIncrement
//!    21   1  LastModified10msIncrement
//!    22   1  CreateUtcOffset
//!    23   1  LastModifiedUtcOffset
//!    24   1  LastAccessedUtcOffset
//!    25   7  Reserved2
//! ```
//!
//! ### StreamExtension (0xC0)
//!
//! ```text
//!     0   1  EntryType = 0xC0
//!     1   1  GeneralSecondaryFlags  (bit 0 = AllocationPossible,
//!                                    bit 1 = NoFatChain)
//!     2   1  Reserved1
//!     3   1  NameLength             (in UTF-16 code units)
//!     4   2  NameHash
//!     6   2  Reserved2
//!     8   8  ValidDataLength
//!    16   4  Reserved3
//!    20   4  FirstCluster
//!    24   8  DataLength
//! ```
//!
//! ### FileName (0xC1)
//!
//! ```text
//!     0   1  EntryType = 0xC1
//!     1   1  GeneralSecondaryFlags  (0)
//!     2  30  FileName               (15 UTF-16 code units, LE)
//! ```
//!
//! ## SetChecksum
//!
//! The 16-bit checksum is computed over all entries in the set, in
//! sequence, byte by byte, skipping bytes 2 and 3 of the primary entry
//! (where the SetChecksum field itself lives). Each byte rotates the
//! accumulator right by one bit and adds the byte (mod 2^16).

/// Bytes per on-disk directory entry.
pub const ENTRY_SIZE: usize = 32;

/// Entry-type bytes we recognise.
pub const ENTRY_ALLOCATION_BITMAP: u8 = 0x81;
pub const ENTRY_UPCASE_TABLE: u8 = 0x82;
pub const ENTRY_VOLUME_LABEL: u8 = 0x83;
pub const ENTRY_FILE: u8 = 0x85;
pub const ENTRY_STREAM_EXTENSION: u8 = 0xC0;
pub const ENTRY_FILE_NAME: u8 = 0xC1;

/// Mask: an entry is "in use" when its high bit is set.
pub const ENTRY_INUSE: u8 = 0x80;

/// FileAttributes bits (from the FileDirectoryEntry primary).
pub const ATTR_READ_ONLY: u16 = 0x0001;
pub const ATTR_HIDDEN: u16 = 0x0002;
pub const ATTR_SYSTEM: u16 = 0x0004;
pub const ATTR_DIRECTORY: u16 = 0x0010;
pub const ATTR_ARCHIVE: u16 = 0x0020;

/// GeneralSecondaryFlags bits (StreamExtension).
pub const SECFLAG_ALLOC_POSSIBLE: u8 = 0x01;
pub const SECFLAG_NO_FAT_CHAIN: u8 = 0x02;

/// A fully-parsed file/directory entry set.
#[derive(Debug, Clone)]
pub struct FileEntrySet {
    pub file_attributes: u16,
    pub create_timestamp: u32,
    pub last_modified_timestamp: u32,
    pub last_accessed_timestamp: u32,
    /// True when bit 4 of FileAttributes is set.
    pub is_directory: bool,
    /// Flags from the StreamExtension entry.
    pub secondary_flags: u8,
    pub name_length: u8,
    pub name_hash: u16,
    pub valid_data_length: u64,
    pub first_cluster: u32,
    pub data_length: u64,
    /// File name as decoded UTF-16 → Rust `String`.
    pub name: String,
    /// Raw UTF-16 LE name code units (for case-insensitive matching with
    /// the up-case table — avoids re-encoding `name`).
    pub name_utf16: Vec<u16>,
}

impl FileEntrySet {
    /// True if the StreamExtension marks the file as contiguous (no FAT
    /// chain walk required).
    pub fn no_fat_chain(&self) -> bool {
        self.secondary_flags & SECFLAG_NO_FAT_CHAIN != 0
    }
}

/// A single 32-byte slot's identity.
#[derive(Debug, Clone)]
pub enum RawSlot<'a> {
    /// EntryType byte is 0x00 → end-of-directory marker; stop scanning.
    EndOfDirectory,
    /// EntryType high bit clear → unused/deleted slot; skip.
    Unused,
    /// AllocationBitmap (0x81). Carries `(bitmap_flags, first_cluster, data_length)`.
    AllocationBitmap {
        bitmap_flags: u8,
        first_cluster: u32,
        data_length: u64,
    },
    /// UpcaseTable (0x82). Carries `(checksum, first_cluster, data_length)`.
    UpcaseTable {
        checksum: u32,
        first_cluster: u32,
        data_length: u64,
    },
    /// VolumeLabel (0x83). Carries the raw UTF-16 units actually present.
    VolumeLabel(Vec<u16>),
    /// FileDirectoryEntry (0x85) — primary entry of a file set.
    File {
        secondary_count: u8,
        set_checksum: u16,
        bytes: &'a [u8; ENTRY_SIZE],
    },
    /// Other recognised secondary types within a file set are parsed in
    /// context by `parse_file_set`; this variant covers anything else.
    Other { entry_type: u8 },
}

/// Classify a single 32-byte slot.
pub fn classify_slot(slot: &[u8; ENTRY_SIZE]) -> RawSlot<'_> {
    let t = slot[0];
    if t == 0x00 {
        return RawSlot::EndOfDirectory;
    }
    if t & ENTRY_INUSE == 0 {
        return RawSlot::Unused;
    }
    match t {
        ENTRY_ALLOCATION_BITMAP => RawSlot::AllocationBitmap {
            bitmap_flags: slot[1],
            first_cluster: u32::from_le_bytes(slot[20..24].try_into().unwrap()),
            data_length: u64::from_le_bytes(slot[24..32].try_into().unwrap()),
        },
        ENTRY_UPCASE_TABLE => RawSlot::UpcaseTable {
            checksum: u32::from_le_bytes(slot[4..8].try_into().unwrap()),
            first_cluster: u32::from_le_bytes(slot[20..24].try_into().unwrap()),
            data_length: u64::from_le_bytes(slot[24..32].try_into().unwrap()),
        },
        ENTRY_VOLUME_LABEL => {
            let n = (slot[1] as usize).min(11);
            let mut units = Vec::with_capacity(n);
            for i in 0..n {
                let off = 2 + i * 2;
                units.push(u16::from_le_bytes(slot[off..off + 2].try_into().unwrap()));
            }
            RawSlot::VolumeLabel(units)
        }
        ENTRY_FILE => RawSlot::File {
            secondary_count: slot[1],
            set_checksum: u16::from_le_bytes(slot[2..4].try_into().unwrap()),
            bytes: slot,
        },
        other => RawSlot::Other { entry_type: other },
    }
}

/// Compute the SetChecksum over the bytes of all entries in a set. The
/// `set` slice must hold exactly `(1 + secondary_count) * ENTRY_SIZE`
/// bytes, primary first. Bytes 2 and 3 of the primary are skipped (where
/// the checksum field lives on-disk).
pub fn set_checksum(set: &[u8]) -> u16 {
    let mut sum: u16 = 0;
    for (i, &b) in set.iter().enumerate() {
        if i == 2 || i == 3 {
            continue;
        }
        sum = sum.rotate_right(1).wrapping_add(b as u16);
    }
    sum
}

/// Compute the NameHash over an up-cased UTF-16 name. Used by exFAT to
/// short-circuit name comparisons; we mirror the algorithm so we can
/// build a name's hash if needed but do not currently verify it.
pub fn name_hash(upcased_le_bytes: &[u8]) -> u16 {
    let mut hash: u16 = 0;
    for &b in upcased_le_bytes {
        hash = hash.rotate_right(1).wrapping_add(b as u16);
    }
    hash
}

/// Parse a single file entry set starting at `set[0]` (a primary 0x85).
/// `set` must hold `(1 + secondary_count) * ENTRY_SIZE` bytes. Returns
/// the parsed set or an error if the on-disk checksum doesn't validate
/// or the secondary entries don't match the expected layout.
pub fn parse_file_set(set: &[u8]) -> crate::Result<FileEntrySet> {
    if set.len() < 3 * ENTRY_SIZE {
        return Err(crate::Error::InvalidImage(
            "exfat: file entry set is shorter than 3 entries".into(),
        ));
    }
    let primary: &[u8; ENTRY_SIZE] = (&set[..ENTRY_SIZE]).try_into().unwrap();
    if primary[0] != ENTRY_FILE {
        return Err(crate::Error::InvalidImage(format!(
            "exfat: expected primary FileDirectoryEntry (0x85), got 0x{:02X}",
            primary[0]
        )));
    }
    let secondary_count = primary[1] as usize;
    let expected_len = (1 + secondary_count) * ENTRY_SIZE;
    if set.len() < expected_len {
        return Err(crate::Error::InvalidImage(format!(
            "exfat: file entry set wants {expected_len} bytes, got {}",
            set.len()
        )));
    }
    let on_disk_checksum = u16::from_le_bytes(primary[2..4].try_into().unwrap());
    let computed = set_checksum(&set[..expected_len]);
    if on_disk_checksum != computed {
        return Err(crate::Error::InvalidImage(format!(
            "exfat: file-set checksum mismatch (on-disk 0x{on_disk_checksum:04X}, \
             computed 0x{computed:04X})"
        )));
    }
    let file_attributes = u16::from_le_bytes(primary[4..6].try_into().unwrap());
    let create_timestamp = u32::from_le_bytes(primary[8..12].try_into().unwrap());
    let last_modified_timestamp = u32::from_le_bytes(primary[12..16].try_into().unwrap());
    let last_accessed_timestamp = u32::from_le_bytes(primary[16..20].try_into().unwrap());
    let is_directory = file_attributes & ATTR_DIRECTORY != 0;

    let stream: &[u8; ENTRY_SIZE] = (&set[ENTRY_SIZE..2 * ENTRY_SIZE]).try_into().unwrap();
    if stream[0] != ENTRY_STREAM_EXTENSION {
        return Err(crate::Error::InvalidImage(format!(
            "exfat: expected StreamExtension (0xC0) at slot 1, got 0x{:02X}",
            stream[0]
        )));
    }
    let secondary_flags = stream[1];
    let name_length = stream[3];
    let name_hash_disk = u16::from_le_bytes(stream[4..6].try_into().unwrap());
    let valid_data_length = u64::from_le_bytes(stream[8..16].try_into().unwrap());
    let first_cluster = u32::from_le_bytes(stream[20..24].try_into().unwrap());
    let data_length = u64::from_le_bytes(stream[24..32].try_into().unwrap());

    // Concatenate FileName entries.
    let mut name_utf16: Vec<u16> = Vec::with_capacity(name_length as usize);
    let name_entries = secondary_count.saturating_sub(1);
    for i in 0..name_entries {
        let off = (2 + i) * ENTRY_SIZE;
        let slot: &[u8; ENTRY_SIZE] = (&set[off..off + ENTRY_SIZE]).try_into().unwrap();
        if slot[0] != ENTRY_FILE_NAME {
            return Err(crate::Error::InvalidImage(format!(
                "exfat: expected FileName (0xC1) at slot {}, got 0x{:02X}",
                2 + i,
                slot[0]
            )));
        }
        for j in 0..15 {
            let p = 2 + j * 2;
            name_utf16.push(u16::from_le_bytes(slot[p..p + 2].try_into().unwrap()));
        }
    }
    name_utf16.truncate(name_length as usize);
    let name = String::from_utf16(&name_utf16).map_err(|_| {
        crate::Error::InvalidImage("exfat: file name is not valid UTF-16".into())
    })?;

    // We don't strictly validate name_hash — it's a hint, and verifying
    // requires the up-case table the caller may not yet have. Keep the
    // value visible for diagnostics.
    let _ = name_hash_disk;

    Ok(FileEntrySet {
        file_attributes,
        create_timestamp,
        last_modified_timestamp,
        last_accessed_timestamp,
        is_directory,
        secondary_flags,
        name_length,
        name_hash: name_hash_disk,
        valid_data_length,
        first_cluster,
        data_length,
        name,
        name_utf16,
    })
}

/// Decode a VolumeLabel entry's UTF-16 units into a Rust `String`. Lossy
/// decode — invalid surrogates fall back to U+FFFD.
pub fn decode_volume_label(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a primary FileDirectoryEntry byte slot. The checksum field
    /// (bytes 2..4) is left zero — callers either compute it or test the
    /// "before checksum" state.
    fn make_primary(secondary_count: u8, file_attributes: u16) -> [u8; ENTRY_SIZE] {
        let mut e = [0u8; ENTRY_SIZE];
        e[0] = ENTRY_FILE;
        e[1] = secondary_count;
        // 2..4 SetChecksum (filled later)
        e[4..6].copy_from_slice(&file_attributes.to_le_bytes());
        // timestamps left zero
        e
    }

    fn make_stream(
        secondary_flags: u8,
        name_length: u8,
        valid_data_length: u64,
        first_cluster: u32,
        data_length: u64,
    ) -> [u8; ENTRY_SIZE] {
        let mut e = [0u8; ENTRY_SIZE];
        e[0] = ENTRY_STREAM_EXTENSION;
        e[1] = secondary_flags;
        e[3] = name_length;
        // name_hash left zero
        e[8..16].copy_from_slice(&valid_data_length.to_le_bytes());
        e[20..24].copy_from_slice(&first_cluster.to_le_bytes());
        e[24..32].copy_from_slice(&data_length.to_le_bytes());
        e
    }

    fn make_name(units: &[u16]) -> [u8; ENTRY_SIZE] {
        let mut e = [0u8; ENTRY_SIZE];
        e[0] = ENTRY_FILE_NAME;
        for (i, &u) in units.iter().enumerate().take(15) {
            let off = 2 + i * 2;
            e[off..off + 2].copy_from_slice(&u.to_le_bytes());
        }
        e
    }

    #[test]
    fn checksum_skips_bytes_2_and_3() {
        // A trivial 3-entry set, all zero except types.
        let primary = make_primary(2, 0);
        let stream = make_stream(SECFLAG_NO_FAT_CHAIN, 5, 5, 4, 5);
        let name = make_name(&[b'H' as u16, b'E' as u16, b'L' as u16, b'L' as u16, b'O' as u16]);
        let mut set = Vec::new();
        set.extend_from_slice(&primary);
        set.extend_from_slice(&stream);
        set.extend_from_slice(&name);
        let csum1 = set_checksum(&set);
        // Mutating bytes 2..4 of the primary (the checksum field itself)
        // must not change the computed sum.
        set[2] = 0xAB;
        set[3] = 0xCD;
        let csum2 = set_checksum(&set);
        assert_eq!(csum1, csum2);
        // Mutating any other byte changes the checksum.
        set[5] = 0x55;
        let csum3 = set_checksum(&set);
        assert_ne!(csum1, csum3);
    }

    #[test]
    fn parse_known_good_set() {
        // Build "hello.txt" (9 code units → fits in one FileName entry).
        let name_units: Vec<u16> = "hello.txt".encode_utf16().collect();
        let mut primary = make_primary(2, 0); // not a directory
        let stream = make_stream(SECFLAG_NO_FAT_CHAIN, name_units.len() as u8, 11, 5, 11);
        let name = make_name(&name_units);

        let mut set = Vec::new();
        set.extend_from_slice(&primary);
        set.extend_from_slice(&stream);
        set.extend_from_slice(&name);
        // Compute SetChecksum over the set with primary bytes 2..4 = 0,
        // then write it back into the primary.
        let csum = set_checksum(&set);
        primary[2..4].copy_from_slice(&csum.to_le_bytes());
        set[..ENTRY_SIZE].copy_from_slice(&primary);

        let parsed = parse_file_set(&set).unwrap();
        assert_eq!(parsed.name, "hello.txt");
        assert_eq!(parsed.name_length, 9);
        assert_eq!(parsed.first_cluster, 5);
        assert_eq!(parsed.data_length, 11);
        assert_eq!(parsed.valid_data_length, 11);
        assert!(!parsed.is_directory);
        assert!(parsed.no_fat_chain());
    }

    #[test]
    fn parse_rejects_bad_checksum() {
        let name_units: Vec<u16> = "x".encode_utf16().collect();
        let primary = make_primary(2, 0);
        let stream = make_stream(0, 1, 0, 0, 0);
        let name = make_name(&name_units);
        let mut set = Vec::new();
        set.extend_from_slice(&primary);
        set.extend_from_slice(&stream);
        set.extend_from_slice(&name);
        // Don't fill in a valid checksum — primary[2..4] stays 0, but
        // the computed checksum is non-zero.
        let err = parse_file_set(&set).unwrap_err();
        match err {
            crate::Error::InvalidImage(msg) => assert!(msg.contains("checksum")),
            other => panic!("expected InvalidImage, got {other:?}"),
        }
    }

    #[test]
    fn parse_two_filename_entries() {
        // 37-char name spans three FileName entries (15 + 15 + 7).
        let name_str = "this_filename_is_just_long_enough.bin";
        let name_units: Vec<u16> = name_str.encode_utf16().collect();
        assert!(name_units.len() > 15);
        let n_name_entries = name_units.len().div_ceil(15);
        let secondary_count = (1 + n_name_entries) as u8;

        let mut primary = make_primary(secondary_count, 0);
        let stream = make_stream(
            SECFLAG_NO_FAT_CHAIN,
            name_units.len() as u8,
            100,
            10,
            100,
        );

        let mut set = Vec::new();
        set.extend_from_slice(&primary);
        set.extend_from_slice(&stream);
        for chunk in name_units.chunks(15) {
            let entry = make_name(chunk);
            set.extend_from_slice(&entry);
        }

        let csum = set_checksum(&set);
        primary[2..4].copy_from_slice(&csum.to_le_bytes());
        set[..ENTRY_SIZE].copy_from_slice(&primary);

        let parsed = parse_file_set(&set).unwrap();
        assert_eq!(parsed.name, name_str);
    }

    #[test]
    fn classify_basic() {
        let mut slot = [0u8; ENTRY_SIZE];
        // 0x00 → end of dir.
        matches!(classify_slot(&slot), RawSlot::EndOfDirectory);

        slot[0] = 0x05; // InUse bit clear
        matches!(classify_slot(&slot), RawSlot::Unused);

        slot[0] = ENTRY_VOLUME_LABEL;
        slot[1] = 3; // 3 chars
        slot[2..4].copy_from_slice(&(b'T' as u16).to_le_bytes());
        slot[4..6].copy_from_slice(&(b'M' as u16).to_le_bytes());
        slot[6..8].copy_from_slice(&(b'P' as u16).to_le_bytes());
        match classify_slot(&slot) {
            RawSlot::VolumeLabel(units) => {
                assert_eq!(decode_volume_label(&units), "TMP");
            }
            _ => panic!("expected VolumeLabel"),
        }
    }
}
