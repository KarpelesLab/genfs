//! `$Secure` and `$UpCase` metadata helpers.
//!
//! These two NTFS metadata files (records 9 and 10 respectively) carry
//! information that's referenced *by index* from ordinary file records:
//!
//! * **`$Secure`** holds shared security descriptors in its `$SDS` data
//!   stream and two B-tree indexes (`$SII` by security id, `$SDH` by hash).
//!   When a file's `$STANDARD_INFORMATION.security_id` is non-zero, the
//!   real self-relative `SECURITY_DESCRIPTOR` blob lives in `$SDS` and must
//!   be looked up via `$SII`.
//!
//! * **`$UpCase`** holds a Unicode upper-case table — exactly 65536 little-
//!   endian u16 entries. `$UpCase[c]` is the uppercase form of code point
//!   `c` (BMP only — surrogate-paired code points fall back to themselves).
//!   NTFS directory indexes are sorted by this table, and the kernel matches
//!   names case-insensitively by folding both sides through it.
//!
//! The structures here are stripped to what we need for the read path:
//! resolve a `security_id` to its raw SD bytes, and case-fold a `&str` for
//! directory lookup.

use crate::Result;

/// Logical class of security descriptor a file or directory should be
/// stamped with at format time. Real-world NTFS volumes carry many
/// distinct SDs (one per unique ACL); we materialise a small fixed
/// catalogue at format time and let each MFT record point at the right
/// entry through `$STANDARD_INFORMATION.security_id`.
///
/// * `System` — tighter ACL applied to NTFS system files (records 0..=15
///   on a fresh volume). DACL grants SYSTEM full control and Administrators
///   full control; Everyone has no entries.
/// * `User`   — looser ACL applied to user-visible files and directories
///   created through the writer. DACL grants Everyone full control (this
///   matches what `mkntfs` historically lays down for unowned files).
/// * `Default` — alias for `User`. Reserved so future expansions
///   (read-only, special-purpose ACLs) don't churn external call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SecurityClass {
    /// User-facing files and directories.
    Default,
    /// NTFS system files (records 0..=15).
    System,
    /// Explicit alias for `Default`; reserved for future use.
    User,
}

impl SecurityClass {
    /// Stable index inside the volume's security catalogue. Used by the
    /// formatter to lay out the SDS stream in a deterministic order so
    /// the `security_id` for each class is reproducible.
    pub fn catalogue_index(self) -> u32 {
        match self {
            // The User / Default classes share the same on-disk SD.
            SecurityClass::Default | SecurityClass::User => 0,
            SecurityClass::System => 1,
        }
    }
}

/// One row of the `$Secure:$SDS` data stream's "SDS entry" header. The
/// stream is laid out as packed entries, padded so each starts on a 16-byte
/// boundary within a 256 KiB block. Each entry's header occupies 0x14 bytes
/// and is immediately followed by `sd_size - 0x14` bytes of self-relative
/// `SECURITY_DESCRIPTOR`.
///
/// We parse only the fields we need to verify a hit + pull the SD out.
#[derive(Debug, Clone)]
pub struct SdsEntryHeader {
    pub hash: u32,
    pub security_id: u32,
    pub offset_in_sds: u64,
    pub size: u32,
}

impl SdsEntryHeader {
    /// Decode the 20-byte SDS entry header at the start of `buf`.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < 0x14 {
            return Err(crate::Error::InvalidImage(
                "ntfs: $Secure:$SDS entry header truncated".into(),
            ));
        }
        Ok(Self {
            hash: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            security_id: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            offset_in_sds: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            size: u32::from_le_bytes(buf[16..20].try_into().unwrap()),
        })
    }
}

/// One decoded $SII index entry. The key is the security id (LE u32); the
/// value carries `(hash, security_id, sds_offset, sds_size)`. We only need
/// the offset + size to seek into $SDS.
#[derive(Debug, Clone)]
pub struct SiiEntry {
    pub security_id: u32,
    pub sds_offset: u64,
    pub sds_size: u32,
}

/// Walk the `$Secure:$SII` index buffer, gathering `SiiEntry` rows from
/// the leaf-level entries. The on-disk index uses the same generic-index
/// framework as $I30 (`$INDEX_ROOT` / `$INDEX_ALLOCATION`), so we expose a
/// flat decoder that handles a single node's entry stream. Tree descent is
/// handled by the caller — for v1 we only consult the root node when it's
/// small enough; larger volumes can extend this to walk allocation blocks
/// using the existing `index::walk_index_node`.
///
/// Returns the list of non-terminator entries found in `buf` (the bytes
/// between `entries_start` and `entries_start + bytes_in_use` of the inner
/// index header).
pub fn walk_sii_node(buf: &[u8]) -> Result<Vec<SiiEntry>> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor + 16 <= buf.len() {
        // Generic index entry header: data_offset(u16), data_size(u16),
        // padding(u32), entry_len(u16), key_len(u16), flags(u32).
        // The $SII variant stores data right after the header (data_offset
        // points there) and the key is the 4-byte security id immediately
        // after the entry header.
        let data_offset = u16::from_le_bytes(buf[cursor..cursor + 2].try_into().unwrap()) as usize;
        let data_size =
            u16::from_le_bytes(buf[cursor + 2..cursor + 4].try_into().unwrap()) as usize;
        let entry_len =
            u16::from_le_bytes(buf[cursor + 8..cursor + 10].try_into().unwrap()) as usize;
        let key_len =
            u16::from_le_bytes(buf[cursor + 10..cursor + 12].try_into().unwrap()) as usize;
        let flags = u32::from_le_bytes(buf[cursor + 12..cursor + 16].try_into().unwrap());

        if entry_len < 16 || cursor + entry_len > buf.len() {
            break;
        }
        let is_last = flags & 0x02 != 0;
        if !is_last && key_len >= 4 && data_size >= 0x14 {
            let key_start = cursor + 16;
            let security_id = u32::from_le_bytes(buf[key_start..key_start + 4].try_into().unwrap());
            let data_start = cursor + data_offset;
            if data_start + data_size <= buf.len() {
                // Layout of the data portion mirrors the SDS entry header.
                let hdr = SdsEntryHeader::parse(&buf[data_start..data_start + data_size])?;
                out.push(SiiEntry {
                    security_id,
                    sds_offset: hdr.offset_in_sds,
                    sds_size: hdr.size,
                });
                let _ = security_id;
            }
        }
        cursor += entry_len;
        if is_last {
            break;
        }
    }
    Ok(out)
}

/// The `$UpCase` table. 65536 u16 entries; `case_fold(c)` returns the
/// uppercase variant for BMP code points.
#[derive(Clone)]
pub struct UpcaseTable {
    table: Vec<u16>,
}

impl UpcaseTable {
    /// Build the identity table — every code point folds to itself. Used
    /// when `$UpCase` isn't available (synthetic test images).
    pub fn identity() -> Self {
        let mut table = Vec::with_capacity(0x10000);
        for i in 0..=0xFFFFu16 {
            table.push(i);
        }
        Self { table }
    }

    /// Decode a `$UpCase` binary blob. Spec requires 128 KiB (65536 × u16);
    /// shorter buffers pad with identity, longer ones are truncated.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut table = Vec::with_capacity(0x10000);
        for i in 0..0x10000usize {
            let start = i * 2;
            if start + 2 <= bytes.len() {
                table.push(u16::from_le_bytes([bytes[start], bytes[start + 1]]));
            } else {
                table.push(i as u16);
            }
        }
        Self { table }
    }

    /// Case-fold one BMP code unit. Non-BMP code units (the surrogate pair
    /// range) are returned unchanged — NTFS itself does the same.
    pub fn fold_unit(&self, c: u16) -> u16 {
        self.table[c as usize]
    }

    /// Fold an entire `&str` to upper case for comparison purposes. We do
    /// this UTF-16 code-unit by code-unit so the lookup matches what NTFS
    /// did when it built its index ordering.
    pub fn fold_str(&self, s: &str) -> Vec<u16> {
        s.encode_utf16().map(|u| self.fold_unit(u)).collect()
    }

    /// Compare two `&str`s case-insensitively per the table.
    pub fn equals_ignore_case(&self, a: &str, b: &str) -> bool {
        // Quick reject on character count via the iterator (avoids
        // allocating when lengths differ).
        let mut au = a.encode_utf16();
        let mut bu = b.encode_utf16();
        loop {
            match (au.next(), bu.next()) {
                (None, None) => return true,
                (Some(x), Some(y)) => {
                    if self.fold_unit(x) != self.fold_unit(y) {
                        return false;
                    }
                }
                _ => return false,
            }
        }
    }
}

impl std::fmt::Debug for UpcaseTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UpcaseTable({} entries)", self.table.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_folds_self() {
        let t = UpcaseTable::identity();
        assert_eq!(t.fold_unit(0x0061), 0x0061);
        assert!(t.equals_ignore_case("abc", "abc"));
        assert!(!t.equals_ignore_case("abc", "abd"));
    }

    #[test]
    fn ascii_uppercase_table() {
        // Build a small table that uppercases ASCII a..z.
        let mut bytes = Vec::with_capacity(0x10000 * 2);
        for i in 0..0x10000u32 {
            let v = if (0x61..=0x7A).contains(&i) {
                i as u16 - 0x20
            } else {
                i as u16
            };
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let t = UpcaseTable::from_bytes(&bytes);
        assert_eq!(t.fold_unit(b'a' as u16), b'A' as u16);
        assert!(t.equals_ignore_case("HELLO.TXT", "hello.txt"));
        assert!(!t.equals_ignore_case("hello", "world"));
    }

    #[test]
    fn sds_header_parse() {
        let mut buf = vec![0u8; 0x14];
        buf[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        buf[4..8].copy_from_slice(&7u32.to_le_bytes());
        buf[8..16].copy_from_slice(&0x4000u64.to_le_bytes());
        buf[16..20].copy_from_slice(&0x100u32.to_le_bytes());
        let h = SdsEntryHeader::parse(&buf).unwrap();
        assert_eq!(h.hash, 0xDEAD_BEEF);
        assert_eq!(h.security_id, 7);
        assert_eq!(h.offset_in_sds, 0x4000);
        assert_eq!(h.size, 0x100);
    }
}
