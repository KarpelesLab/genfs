//! ZIP reader: locate the End-Of-Central-Directory, follow ZIP64 when
//! present, and walk the central directory into an [`ArchiveIndex`].

use super::encoding;
use super::{
    METHOD_DEFLATE, METHOD_STORE, SIG_CENTRAL, SIG_EOCD, SIG_ZIP64_EOCD, SIG_ZIP64_LOCATOR,
};
use crate::block::BlockDevice;
use crate::fs::archive::{ArchiveEntry, ArchiveIndex, DataLocator, EntryKind, Method};
use crate::{Error, Result};

const S_IFMT: u32 = 0o170000;
const S_IFLNK: u32 = 0o120000;

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn le64(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}

/// Central-directory geometry resolved from the EOCD (+ ZIP64).
struct Eocd {
    cd_offset: u64,
    cd_size: u64,
    total_entries: u64,
}

fn find_eocd(dev: &mut dyn BlockDevice) -> Result<Eocd> {
    let total = dev.total_size();
    // EOCD is 22 bytes + up to 65535 bytes of comment.
    let max_back = (22 + 0xffff).min(total);
    let start = total - max_back;
    let mut tail = vec![0u8; max_back as usize];
    dev.read_at(start, &mut tail)?;

    // Scan backward for the EOCD signature, validating the comment len.
    let mut eocd_pos = None;
    if tail.len() >= 22 {
        for i in (0..=tail.len() - 22).rev() {
            if le32(&tail, i) == SIG_EOCD {
                let comment_len = le16(&tail, i + 20) as usize;
                if i + 22 + comment_len == tail.len() {
                    eocd_pos = Some(i);
                    break;
                }
            }
        }
    }
    let i = eocd_pos
        .ok_or_else(|| Error::InvalidImage("zip: end-of-central-directory not found".into()))?;

    let mut cd_size = le32(&tail, i + 12) as u64;
    let mut cd_offset = le32(&tail, i + 16) as u64;
    let mut total_entries = le16(&tail, i + 10) as u64;

    // ZIP64: a locator sits 20 bytes before the EOCD.
    if i >= 20 && le32(&tail, i - 20) == SIG_ZIP64_LOCATOR {
        let z64_eocd_off = le64(&tail, i - 20 + 8);
        let mut rec = [0u8; 56];
        dev.read_at(z64_eocd_off, &mut rec)?;
        if le32(&rec, 0) != SIG_ZIP64_EOCD {
            return Err(Error::InvalidImage(
                "zip: ZIP64 locator points at a bad record".into(),
            ));
        }
        total_entries = le64(&rec, 32);
        cd_size = le64(&rec, 40);
        cd_offset = le64(&rec, 48);
    }

    if cd_offset + cd_size > total {
        return Err(Error::InvalidImage(
            "zip: central directory extends past end of archive".into(),
        ));
    }
    Ok(Eocd {
        cd_offset,
        cd_size,
        total_entries,
    })
}

/// Walk a per-entry extra field, applying ZIP64 overrides and reading a
/// Unix mtime from the `UT` (0x5455) extra. `comp`/`uncomp`/`offset`
/// are overridden from the ZIP64 (0x0001) field only for the values
/// that were `0xFFFFFFFF`.
fn apply_extras(extra: &[u8], comp: &mut u64, uncomp: &mut u64, offset: &mut u64, mtime: &mut u64) {
    let mut p = 0;
    while p + 4 <= extra.len() {
        let id = le16(extra, p);
        let len = le16(extra, p + 2) as usize;
        let body_start = p + 4;
        if body_start + len > extra.len() {
            break;
        }
        let body = &extra[body_start..body_start + len];
        match id {
            0x0001 => {
                // ZIP64: present fields, in order, for each 0xFFFFFFFF value.
                let mut q = 0;
                if *uncomp == 0xffff_ffff && q + 8 <= body.len() {
                    *uncomp = le64(body, q);
                    q += 8;
                }
                if *comp == 0xffff_ffff && q + 8 <= body.len() {
                    *comp = le64(body, q);
                    q += 8;
                }
                if *offset == 0xffff_ffff && q + 8 <= body.len() {
                    *offset = le64(body, q);
                }
            }
            0x5455
                // UT: flags byte, then mtime if bit 0 set.
                if !body.is_empty() && (body[0] & 1) != 0 && body.len() >= 5 => {
                    *mtime = le32(body, 1) as u64;
                }
            _ => {}
        }
        p = body_start + len;
    }
}

/// Map a ZIP compression-method id to our [`Method`].
fn method_for(id: u16) -> Method {
    match id {
        METHOD_STORE => Method::Stored,
        METHOD_DEFLATE => Method::Deflate,
        93 => Method::Codec(crate::compression::Algo::Zstd),
        other => Method::Unsupported(other),
    }
}

pub fn scan(dev: &mut dyn BlockDevice) -> Result<ArchiveIndex> {
    let eocd = find_eocd(dev)?;
    let mut cd = vec![0u8; eocd.cd_size as usize];
    dev.read_at(eocd.cd_offset, &mut cd)?;

    let mut idx = ArchiveIndex::new("zip");
    let mut p = 0usize;
    let mut seen = 0u64;
    while p + 46 <= cd.len() {
        if le32(&cd, p) != SIG_CENTRAL {
            break;
        }
        let version_made_by = le16(&cd, p + 4);
        let gp_flags = le16(&cd, p + 8);
        let method_id = le16(&cd, p + 10);
        let dos_time = le16(&cd, p + 12);
        let dos_date = le16(&cd, p + 14);
        let name_len = le16(&cd, p + 28) as usize;
        let extra_len = le16(&cd, p + 30) as usize;
        let comment_len = le16(&cd, p + 32) as usize;
        let external_attr = le32(&cd, p + 38);
        let mut comp = le32(&cd, p + 20) as u64;
        let mut uncomp = le32(&cd, p + 24) as u64;
        let mut local_offset = le32(&cd, p + 42) as u64;

        let name_off = p + 46;
        let extra_off = name_off + name_len;
        let comment_off = extra_off + extra_len;
        if comment_off + comment_len > cd.len() {
            return Err(Error::InvalidImage(
                "zip: central-directory record overruns".into(),
            ));
        }
        let name_bytes = &cd[name_off..extra_off];
        let extra = &cd[extra_off..comment_off];

        let mut mtime = super::dos_to_unix(dos_date, dos_time);
        apply_extras(extra, &mut comp, &mut uncomp, &mut local_offset, &mut mtime);

        let name = encoding::decode_name(name_bytes, gp_flags & 0x0800 != 0);

        // Resolve the data offset from the *local* header (its extra
        // field length may differ from the central one).
        let mut lh = [0u8; 30];
        dev.read_at(local_offset, &mut lh)?;
        let l_name = le16(&lh, 26) as u64;
        let l_extra = le16(&lh, 28) as u64;
        let data_off = local_offset + 30 + l_name + l_extra;

        // Host-OS byte (high byte of version-made-by) 3 == Unix → the
        // external-attributes high word is a Unix st_mode.
        let host_os = (version_made_by >> 8) & 0xff;
        let unix_mode = if host_os == 3 { external_attr >> 16 } else { 0 };

        let is_dir = name.ends_with('/');
        let kind = if is_dir {
            EntryKind::Dir
        } else if unix_mode & S_IFMT == S_IFLNK {
            EntryKind::Symlink
        } else {
            EntryKind::Regular
        };
        let mode = if unix_mode & 0o7777 != 0 {
            (unix_mode & 0o7777) as u16
        } else if is_dir {
            0o755
        } else {
            0o644
        };

        let method = method_for(method_id);
        let loc = DataLocator {
            offset: data_off,
            compressed_len: comp,
            uncompressed_len: uncomp,
            method,
        };

        let mut entry = ArchiveEntry {
            path: name,
            kind,
            mode,
            uid: 0,
            gid: 0,
            mtime,
            link_target: None,
            device_major: 0,
            device_minor: 0,
            data: None,
        };
        match kind {
            EntryKind::Regular => entry.data = Some(loc),
            EntryKind::Symlink => {
                // The body holds the link target; decode it now (targets
                // are tiny). Unsupported codecs leave it empty.
                if let Ok(mut r) = crate::fs::archive::reader::open(dev, loc) {
                    let mut t = String::new();
                    use std::io::Read;
                    if r.read_to_string(&mut t).is_ok() {
                        entry.link_target = Some(t);
                    }
                }
            }
            _ => {}
        }
        idx.push(entry);

        p = comment_off + comment_len;
        seen += 1;
    }

    if eocd.total_entries != 0 && seen != eocd.total_entries {
        // Not fatal — some tools miscount — but worth surfacing in logs.
        log::debug!(
            "zip: EOCD declared {} entries, walked {seen}",
            eocd.total_entries
        );
    }
    Ok(idx)
}
