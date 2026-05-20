//! GRF file-table decode + encode.
//!
//! Two on-disk dialects share a single in-memory representation.
//!
//! ## v0x102 / v0x103 (raw, encrypted filenames)
//!
//! The table starts at `header.offset + HEADER_SIZE` and is NOT
//! zlib-compressed; it's raw bytes that run to end-of-file. Each
//! entry is encoded as:
//!
//! | bytes | field                                                          |
//! |-------|----------------------------------------------------------------|
//! | 4     | `padded_fn_len + 2` (u32 LE)                                   |
//! | 2     | skip / padding (zero)                                          |
//! | N     | encrypted filename (N = `padded_fn_len`, multiple of 8)        |
//! | 17    | [`RawEntry`] — the 17-byte `grf_table_entry_data` C struct      |
//!
//! Filename encryption is the three-pass transform in
//! [`crate::fs::grf::crypt::decode_filename`]. The stored `len` and
//! `len_aligned` fields carry magic offsets (-`size`-715 and -37579
//! respectively) on top of the actual compressed lengths;
//! [`RawEntry::to_resolved`] undoes these.
//!
//! ## v0x200 (zlib-compressed, plain filenames)
//!
//! The table at `header.offset + HEADER_SIZE` starts with an 8-byte
//! header `[comp_size: u32][uncomp_size: u32]` followed by
//! `comp_size` bytes of zlib data. After inflation, each entry is:
//!
//! | bytes | field                                          |
//! |-------|------------------------------------------------|
//! | var   | null-terminated CP949 filename                 |
//! | 17    | [`RawEntry`] — same struct as above             |
//!
//! No magic offsets on `len` / `len_aligned` in v0x200.
//!
//! ## Common
//!
//! Entries with `flags & GRF_FLAG_FILE == 0` or `size == 0` are
//! directory markers / bogus rows — libgrf discards them on read
//! and we do the same.

use crate::Result;
use crate::fs::grf::crypt;
use crate::fs::grf::encoding;

/// `GRF_FLAG_FILE` — bit 0 set means "real file entry."
pub const GRF_FLAG_FILE: u8 = 0x01;
/// `GRF_FLAG_MIXCRYPT` — per-block mixed encryption (most files).
pub const GRF_FLAG_MIXCRYPT: u8 = 0x02;
/// `GRF_FLAG_DES` — every 8-byte block runs the cipher.
pub const GRF_FLAG_DES: u8 = 0x04;

/// Per-entry raw bytes as they appear on disk inside a parsed table.
/// `len` / `len_aligned` carry the version-specific magic offsets
/// in raw form; convert to a [`crate::fs::grf::Entry`] via
/// [`Self::to_resolved`] which strips them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RawEntry {
    pub len: u32,
    pub len_aligned: u32,
    pub size: u32,
    pub flags: u8,
    pub pos: u32,
}

impl RawEntry {
    pub const SIZE: usize = 17;

    fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(crate::Error::InvalidImage(format!(
                "grf: table entry needs {} bytes, got {}",
                Self::SIZE,
                buf.len()
            )));
        }
        Ok(Self {
            len: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            len_aligned: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
            size: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            flags: buf[12],
            pos: u32::from_le_bytes(buf[13..17].try_into().unwrap()),
        })
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.len.to_le_bytes());
        out.extend_from_slice(&self.len_aligned.to_le_bytes());
        out.extend_from_slice(&self.size.to_le_bytes());
        out.push(self.flags);
        out.extend_from_slice(&self.pos.to_le_bytes());
    }
}

/// A fully resolved table entry — version-specific magic offsets
/// already stripped, filename decoded to UTF-8.
#[derive(Debug, Clone)]
pub struct Entry {
    /// UTF-8 filename. Stored on disk as CP949.
    pub name: String,
    /// Uncompressed size of the file body.
    pub size: u32,
    /// Compressed length of the file body.
    pub len: u32,
    /// Compressed length rounded up to the GRF 4-byte alignment.
    pub len_aligned: u32,
    /// Byte offset of the compressed body, relative to the end of
    /// the GRF header (i.e. add [`crate::fs::grf::HEADER_SIZE`] for
    /// an absolute file offset).
    pub pos: u32,
    /// File-level flags: `GRF_FLAG_FILE | GRF_FLAG_MIXCRYPT |
    /// GRF_FLAG_DES`. The cipher cycle is recomputed by readers as
    /// needed.
    pub flags: u8,
}

impl Entry {
    /// Cycle counter for [`crate::fs::grf::crypt::decode_des_etc`].
    /// `MIXCRYPT` files cycle every `1 + floor(log10(len))` blocks;
    /// `DES` files cycle 0 (every block). Returns `None` if the
    /// entry isn't encrypted.
    pub fn crypto_cycle(&self) -> Option<i32> {
        if self.flags & GRF_FLAG_MIXCRYPT != 0 {
            let mut cycle = 1i32;
            let mut step = 10u32;
            while self.len >= step {
                cycle += 1;
                step = step.saturating_mul(10);
                if step == u32::MAX {
                    break;
                }
            }
            Some(cycle)
        } else if self.flags & GRF_FLAG_DES != 0 {
            Some(0)
        } else {
            None
        }
    }
}

/// Decode a v0x200 table — used after inflating the zlib payload.
/// Returns entries in the order they appear in the table (which is
/// also `pos`-sorted by libgrf convention).
pub(crate) fn decode_v200(buf: &[u8]) -> Result<Vec<Entry>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let nul = match buf[pos..].iter().position(|&b| b == 0) {
            Some(i) => i,
            None => {
                return Err(crate::Error::InvalidImage(
                    "grf: v0x200 table truncated mid-filename".into(),
                ));
            }
        };
        let name_bytes = &buf[pos..pos + nul];
        pos += nul + 1;
        if pos + RawEntry::SIZE > buf.len() {
            return Err(crate::Error::InvalidImage(
                "grf: v0x200 table truncated mid-entry".into(),
            ));
        }
        let raw = RawEntry::decode(&buf[pos..pos + RawEntry::SIZE])?;
        pos += RawEntry::SIZE;

        if raw.flags & GRF_FLAG_FILE == 0 || raw.size == 0 {
            // Directory marker or empty bogus row — libgrf skips
            // these and so do we.
            continue;
        }

        let name = encoding::cp949_to_utf8(name_bytes).into_owned();
        out.push(Entry {
            name,
            size: raw.size,
            len: raw.len,
            len_aligned: raw.len_aligned,
            pos: raw.pos,
            flags: raw.flags,
        });
    }
    Ok(out)
}

/// Decode a v0x102 / v0x103 table — raw bytes, encrypted filenames,
/// magic offsets on `len` / `len_aligned`. `nocrypt_extensions`
/// drives the heuristic that picks `MIXCRYPT` vs `DES` for entries
/// whose flags don't already carry one of those bits — see
/// [`apply_v102_crypto_heuristic`].
pub(crate) fn decode_v102(buf: &[u8]) -> Result<Vec<Entry>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        if pos + 6 > buf.len() {
            return Err(crate::Error::InvalidImage(
                "grf: v0x102 table truncated at length prefix".into(),
            ));
        }
        let padded_fn_len =
            u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()).wrapping_sub(2) as usize;
        pos += 6;
        if pos + padded_fn_len + RawEntry::SIZE > buf.len() {
            return Err(crate::Error::InvalidImage(
                "grf: v0x102 entry exceeds table bounds".into(),
            ));
        }
        // Decrypt the filename into a local buffer.
        let mut name_bytes = buf[pos..pos + padded_fn_len].to_vec();
        pos += padded_fn_len;
        crypt::decode_filename(&mut name_bytes);
        // The padded buffer ends with one or more zero bytes from the
        // original null-padding; trim to the first null.
        let real_len = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        name_bytes.truncate(real_len);

        let raw = RawEntry::decode(&buf[pos..pos + RawEntry::SIZE])?;
        pos += RawEntry::SIZE;

        if raw.flags & GRF_FLAG_FILE == 0 || raw.size == 0 {
            continue;
        }

        // v0x102/0x103 magic offsets on the size fields.
        let actual_len = raw.len.wrapping_sub(raw.size).wrapping_sub(715);
        let actual_aligned = raw.len_aligned.wrapping_sub(37579);

        let name = encoding::cp949_to_utf8(&name_bytes).into_owned();
        let mut entry = Entry {
            name,
            size: raw.size,
            len: actual_len,
            len_aligned: actual_aligned,
            pos: raw.pos,
            flags: raw.flags,
        };
        apply_v102_crypto_heuristic(&mut entry);
        out.push(entry);
    }
    Ok(out)
}

/// libgrf's per-extension heuristic: files with extensions
/// `.gnd / .gat / .act / .str` use `GRF_FLAG_DES`; everything else
/// uses `GRF_FLAG_MIXCRYPT`. Applied to v0x102/0x103 entries whose
/// raw flags don't already carry one of those bits.
fn apply_v102_crypto_heuristic(entry: &mut Entry) {
    if entry.flags & (GRF_FLAG_MIXCRYPT | GRF_FLAG_DES) != 0 {
        return;
    }
    const NO_MIX: &[&[u8; 3]] = &[b"gnd", b"gat", b"act", b"str"];
    let lower = entry.name.to_ascii_lowercase();
    let is_no_mix = lower.len() >= 4
        && lower.as_bytes()[lower.len() - 4] == b'.'
        && NO_MIX
            .iter()
            .any(|ext| &lower.as_bytes()[lower.len() - 3..] == ext.as_slice());
    if is_no_mix {
        entry.flags |= GRF_FLAG_DES;
    } else {
        entry.flags |= GRF_FLAG_MIXCRYPT;
    }
}

/// Encode a slice of entries as a v0x200-format uncompressed table
/// blob. The caller is responsible for zlib-compressing the result
/// before writing it to disk.
pub(crate) fn encode_v200(entries: &[Entry]) -> Vec<u8> {
    let mut out = Vec::new();
    for e in entries {
        let cp949 = encoding::utf8_to_cp949(&e.name);
        out.extend_from_slice(&cp949);
        out.push(0); // null terminator
        let raw = RawEntry {
            len: e.len,
            len_aligned: e.len_aligned,
            size: e.size,
            flags: e.flags,
            pos: e.pos,
        };
        raw.encode_into(&mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v200_round_trip() {
        let entries = vec![
            Entry {
                name: "data/info.txt".into(),
                size: 100,
                len: 50,
                len_aligned: 52,
                pos: 0,
                flags: GRF_FLAG_FILE,
            },
            Entry {
                name: "한글파일.txt".into(),
                size: 200,
                len: 80,
                len_aligned: 80,
                pos: 52,
                flags: GRF_FLAG_FILE,
            },
        ];
        let blob = encode_v200(&entries);
        let back = decode_v200(&blob).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].name, "data/info.txt");
        assert_eq!(back[1].name, "한글파일.txt");
        assert_eq!(back[1].len_aligned, 80);
    }

    #[test]
    fn v200_skips_directory_markers() {
        // A directory marker: flag=0, size=0. encode_v200 doesn't
        // emit these but a real GRF might; we should ignore them on
        // decode.
        let mut blob = Vec::new();
        blob.extend_from_slice(b"some/dir\0");
        RawEntry {
            len: 0,
            len_aligned: 0,
            size: 0,
            flags: 0,
            pos: 0,
        }
        .encode_into(&mut blob);
        blob.extend_from_slice(b"real.txt\0");
        RawEntry {
            len: 5,
            len_aligned: 8,
            size: 10,
            flags: GRF_FLAG_FILE,
            pos: 0,
        }
        .encode_into(&mut blob);

        let entries = decode_v200(&blob).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "real.txt");
    }

    #[test]
    fn crypto_cycle() {
        let e = Entry {
            name: "x".into(),
            size: 0,
            len: 9999,
            len_aligned: 0,
            pos: 0,
            flags: GRF_FLAG_FILE | GRF_FLAG_MIXCRYPT,
        };
        // len = 9999 → cycle = 1 + floor(log10(9999)) = 4
        assert_eq!(e.crypto_cycle(), Some(4));

        let e2 = Entry {
            name: "x".into(),
            size: 0,
            len: 100,
            len_aligned: 0,
            pos: 0,
            flags: GRF_FLAG_FILE | GRF_FLAG_DES,
        };
        assert_eq!(e2.crypto_cycle(), Some(0));

        let e3 = Entry {
            name: "x".into(),
            size: 0,
            len: 100,
            len_aligned: 0,
            pos: 0,
            flags: GRF_FLAG_FILE,
        };
        assert_eq!(e3.crypto_cycle(), None);
    }

    #[test]
    fn v102_heuristic_picks_des_for_gnd() {
        let mut e = Entry {
            name: "data/map.gnd".into(),
            size: 100,
            len: 50,
            len_aligned: 52,
            pos: 0,
            flags: GRF_FLAG_FILE,
        };
        apply_v102_crypto_heuristic(&mut e);
        assert_eq!(e.flags & GRF_FLAG_DES, GRF_FLAG_DES);
        assert_eq!(e.flags & GRF_FLAG_MIXCRYPT, 0);
    }

    #[test]
    fn v102_heuristic_picks_mixcrypt_for_others() {
        let mut e = Entry {
            name: "data/info.txt".into(),
            size: 100,
            len: 50,
            len_aligned: 52,
            pos: 0,
            flags: GRF_FLAG_FILE,
        };
        apply_v102_crypto_heuristic(&mut e);
        assert_eq!(e.flags & GRF_FLAG_MIXCRYPT, GRF_FLAG_MIXCRYPT);
        assert_eq!(e.flags & GRF_FLAG_DES, 0);
    }
}
