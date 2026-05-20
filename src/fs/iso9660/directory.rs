//! Directory record + directory walking per ECMA-119 §9.1 and §6.8.
//!
//! A directory is a stream of variable-length records, each starting
//! with a single-byte length-of-record. A record of length zero means
//! "no more records in this sector — skip to the next 2 KiB boundary".
//! Records cannot straddle a logical-sector boundary, so the walker
//! has to round up after a zero-length record.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::{DirEntry, EntryKind};

use super::SECTOR_SIZE;
use super::rock_ridge;
use super::vd::{SupplementaryVolumeDescriptor, decode_both_endian_u16, decode_both_endian_u32};

/// One directory record decoded from the on-disk byte stream. Carries
/// just the location/size/flags needed to address the file; long names,
/// posix attrs etc. live in the System Use Area which the walker layers
/// on top.
#[derive(Debug, Clone)]
pub struct DirRecord {
    /// Length of the encoded record on disk (`len_dr` in ECMA-119).
    /// Needed to step the walker forward.
    pub len_dr: u8,
    /// LBA of this entry's data extent.
    pub extent_lba: u32,
    /// Data length in bytes.
    pub length: u64,
    /// Directory record flags (ECMA-119 §9.1.6).
    pub flags: u8,
    /// 8.3 / Joliet identifier bytes; `1..len_fi+1` of the record.
    pub identifier: Vec<u8>,
    /// System Use Area bytes — RR entries live here when present.
    pub system_use: Vec<u8>,
}

impl DirRecord {
    pub fn is_dir(&self) -> bool {
        // Flags bit 1 (mask 0x02) = directory per ECMA-119 §9.1.6.
        self.flags & 0x02 != 0
    }

    /// Decode a record from `buf` starting at offset 0. Returns the
    /// parsed record. `buf` must be at least `len_dr` bytes long.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.is_empty() {
            return Err(crate::Error::InvalidImage(
                "iso9660: empty directory record buffer".into(),
            ));
        }
        let len_dr = buf[0];
        if len_dr < 33 || usize::from(len_dr) > buf.len() {
            return Err(crate::Error::InvalidImage(format!(
                "iso9660: dir record length {len_dr} out of bounds"
            )));
        }
        let extent_lba = decode_both_endian_u32(&buf[2..10], "extent_lba")?;
        let length = decode_both_endian_u32(&buf[10..18], "data_length")?;
        let flags = buf[25];
        let len_fi = buf[32] as usize;
        let id_start = 33;
        let id_end = id_start + len_fi;
        if id_end > buf.len() {
            return Err(crate::Error::InvalidImage(
                "iso9660: identifier overflows record length".into(),
            ));
        }
        let identifier = buf[id_start..id_end].to_vec();
        // System Use Area starts after the identifier, padded to even
        // length (ECMA-119 §9.1.12).
        let sua_start = id_end + if len_fi % 2 == 0 { 1 } else { 0 };
        let sua_end = usize::from(len_dr);
        let system_use = if sua_start < sua_end {
            buf[sua_start..sua_end].to_vec()
        } else {
            Vec::new()
        };
        Ok(Self {
            len_dr,
            extent_lba,
            length: u64::from(length),
            flags,
            identifier,
            system_use,
        })
    }
}

/// One entry surfaced to the caller. `name` is the cooked, human-
/// readable name (Joliet → UTF-16 decoded; Rock Ridge NM applied;
/// else 8.3 with `;version` suffix stripped).
#[derive(Debug, Clone)]
pub struct DirEntryRaw {
    pub name: String,
    pub record: DirRecord,
    pub symlink_target: Option<String>,
}

impl DirEntryRaw {
    pub(crate) fn into_dir_entry(self) -> DirEntry {
        let kind = if self.symlink_target.is_some() {
            EntryKind::Symlink
        } else if self.record.is_dir() {
            EntryKind::Dir
        } else {
            EntryKind::Regular
        };
        DirEntry {
            name: self.name,
            inode: self.record.extent_lba,
            kind,
        }
    }
}

/// Read every record in `dir`'s extent, returning the cooked entries.
/// Skips the first two ("." and "..") for callers but parses them so
/// Rock Ridge System Use Area at slot 0 (where `SP` would land) is
/// available.
pub fn read_directory(
    dev: &mut dyn BlockDevice,
    dir: &DirRecord,
    use_rock_ridge: bool,
    joliet_svd: Option<&SupplementaryVolumeDescriptor>,
) -> Result<Vec<DirEntryRaw>> {
    let mut out = Vec::new();
    let mut buf = vec![0u8; dir.length as usize];
    dev.read_at(u64::from(dir.extent_lba) * u64::from(SECTOR_SIZE), &mut buf)?;
    let mut cursor = 0usize;
    let len = buf.len();
    while cursor < len {
        let len_dr = buf[cursor] as usize;
        if len_dr == 0 {
            // Rest of this sector is padding — skip to the next sector.
            let sector_end = ((cursor / SECTOR_SIZE as usize) + 1) * SECTOR_SIZE as usize;
            cursor = sector_end;
            continue;
        }
        if cursor + len_dr > len {
            break;
        }
        let rec = DirRecord::decode(&buf[cursor..cursor + len_dr])?;
        cursor += len_dr;

        // Skip "." (ident=0x00) and ".." (ident=0x01).
        let ident = rec.identifier.clone();
        if ident.len() == 1 && (ident[0] == 0x00 || ident[0] == 0x01) {
            // Preserve them as "." / ".." for callers who want them.
            let name = if ident[0] == 0x00 { "." } else { ".." };
            out.push(DirEntryRaw {
                name: name.to_string(),
                record: rec,
                symlink_target: None,
            });
            continue;
        }

        // Cooked name: Joliet first (if requested), then Rock Ridge NM
        // override on top, else the 8.3 form.
        let mut cooked = if joliet_svd.is_some() {
            // Joliet identifiers are UCS-2 BE per Joliet §3.4.2.
            super::joliet::ucs2_be_to_string(&ident)
        } else {
            iso_basename(&ident)
        };

        let mut symlink_target = None;
        if use_rock_ridge {
            if let Some(rr) = rock_ridge::parse_system_use(dev, &rec.system_use) {
                if let Some(nm) = rr.alternate_name {
                    cooked = nm;
                }
                if let Some(sl) = rr.symlink_target {
                    symlink_target = Some(sl);
                }
            }
        }

        out.push(DirEntryRaw {
            name: cooked,
            record: rec,
            symlink_target,
        });
    }
    Ok(out)
}

/// Strip the `;version` suffix and lowercase the 8.3 form for
/// readability. Plain ISO 9660 identifiers are uppercase; modern tools
/// (and users) expect lowercase, so we case-fold for the cooked name.
fn iso_basename(ident: &[u8]) -> String {
    let s = String::from_utf8_lossy(ident);
    let base = match s.rsplit_once(';') {
        Some((stem, _ver)) => stem,
        None => &s,
    };
    // Trim trailing '.' that mkisofs leaves for "no extension" 8.3 names.
    let base = base.trim_end_matches('.');
    base.to_string()
}

// Silence the dead-code warning if the binary later peels this off.
#[allow(dead_code)]
fn _ref_size(buf: &[u8]) -> usize {
    decode_both_endian_u16(buf, "len")
        .map(|v| v as usize)
        .unwrap_or(0)
}
