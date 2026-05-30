//! `cpio` reader: walk the record chain (newc / newc-crc / odc) into an
//! [`ArchiveIndex`].

use super::{MAGIC_NEWC, MAGIC_NEWC_CRC, MAGIC_ODC, S_IFMT, TRAILER};
use crate::block::BlockDevice;
use crate::fs::archive::{ArchiveEntry, ArchiveIndex, DataLocator, EntryKind, Method};
use crate::{Error, Result};

const NEWC_HEADER_LEN: u64 = 110;
const ODC_HEADER_LEN: u64 = 76;
/// Upper bound on a single record's name length. Real cpio names are
/// path-length bounded; anything larger is a malicious header.
const MAX_NAMESIZE: u64 = 64 * 1024;

fn parse_radix(buf: &[u8], radix: u32, what: &str) -> Result<u64> {
    let s = std::str::from_utf8(buf)
        .map_err(|_| Error::InvalidImage(format!("cpio: non-ASCII {what} field")))?;
    u64::from_str_radix(s.trim(), radix)
        .map_err(|_| Error::InvalidImage(format!("cpio: bad {what} field {s:?}")))
}

/// One parsed header, normalised across newc/odc.
struct Hdr {
    mode: u32,
    uid: u32,
    gid: u32,
    mtime: u64,
    filesize: u64,
    rdevmajor: u32,
    rdevminor: u32,
    namesize: u64,
    header_len: u64,
    /// 4-byte alignment (newc) or 1 (odc) for both name and body.
    align: u64,
}

fn read_newc(dev: &mut dyn BlockDevice, pos: u64) -> Result<Hdr> {
    let mut h = [0u8; NEWC_HEADER_LEN as usize];
    dev.read_at(pos, &mut h)?;
    let f = |a: usize, b: usize, what: &str| parse_radix(&h[a..b], 16, what);
    Ok(Hdr {
        mode: f(14, 22, "mode")? as u32,
        uid: f(22, 30, "uid")? as u32,
        gid: f(30, 38, "gid")? as u32,
        mtime: f(46, 54, "mtime")?,
        filesize: f(54, 62, "filesize")?,
        rdevmajor: f(78, 86, "rdevmajor")? as u32,
        rdevminor: f(86, 94, "rdevminor")? as u32,
        namesize: f(94, 102, "namesize")?,
        header_len: NEWC_HEADER_LEN,
        align: 4,
    })
}

fn read_odc(dev: &mut dyn BlockDevice, pos: u64) -> Result<Hdr> {
    let mut h = [0u8; ODC_HEADER_LEN as usize];
    dev.read_at(pos, &mut h)?;
    let f = |a: usize, b: usize, what: &str| parse_radix(&h[a..b], 8, what);
    let rdev = f(36, 42, "rdev")? as u32;
    Ok(Hdr {
        mode: f(18, 24, "mode")? as u32,
        uid: f(24, 30, "uid")? as u32,
        gid: f(30, 36, "gid")? as u32,
        mtime: f(48, 59, "mtime")?,
        filesize: f(65, 76, "filesize")?,
        // odc packs rdev into one 16-bit field: major = high 8, minor = low 8.
        rdevmajor: (rdev >> 8) & 0xff,
        rdevminor: rdev & 0xff,
        namesize: f(59, 65, "namesize")?,
        header_len: ODC_HEADER_LEN,
        align: 1,
    })
}

fn align_up(n: u64, align: u64) -> u64 {
    if align <= 1 {
        n
    } else {
        n.div_ceil(align) * align
    }
}

fn kind_from_mode(mode: u32) -> EntryKind {
    match mode & S_IFMT {
        0o100000 => EntryKind::Regular,
        0o040000 => EntryKind::Dir,
        0o120000 => EntryKind::Symlink,
        0o020000 => EntryKind::Char,
        0o060000 => EntryKind::Block,
        0o010000 => EntryKind::Fifo,
        0o140000 => EntryKind::Socket,
        // Some producers leave the type bits clear for plain files.
        _ => EntryKind::Regular,
    }
}

pub fn scan(dev: &mut dyn BlockDevice) -> Result<ArchiveIndex> {
    let total = dev.total_size();
    let mut idx = ArchiveIndex::new("cpio");
    let mut pos: u64 = 0;

    while pos + 6 <= total {
        let mut magic = [0u8; 6];
        dev.read_at(pos, &mut magic)?;
        let hdr = if &magic == MAGIC_NEWC || &magic == MAGIC_NEWC_CRC {
            read_newc(dev, pos)?
        } else if &magic == MAGIC_ODC {
            read_odc(dev, pos)?
        } else {
            return Err(Error::InvalidImage(format!(
                "cpio: unrecognised record magic {magic:?} at offset {pos}"
            )));
        };

        // Name follows the header (namesize includes the trailing NUL).
        let name_off = pos + hdr.header_len;
        if hdr.namesize > MAX_NAMESIZE {
            return Err(Error::InvalidImage(format!(
                "cpio: namesize {} exceeds {MAX_NAMESIZE} limit",
                hdr.namesize
            )));
        }
        // Reject name/body that can't fit in what's left of the device
        // before allocating buffers sized from untrusted header fields.
        if name_off > total || hdr.namesize > total - name_off {
            return Err(Error::InvalidImage(
                "cpio: name extends past end of archive".into(),
            ));
        }
        let mut name_buf = vec![0u8; hdr.namesize as usize];
        dev.read_at(name_off, &mut name_buf)?;
        while name_buf.last() == Some(&0) {
            name_buf.pop();
        }
        let name = String::from_utf8_lossy(&name_buf).into_owned();

        // Records are 4-byte aligned from the archive start (newc) so a
        // file-offset alignment equals record-relative alignment. Use
        // checked arithmetic so a malicious filesize can't wrap the
        // record cursor past the end of the device.
        let body_off = align_up(name_off + hdr.namesize, hdr.align);
        if body_off > total || hdr.filesize > total - body_off {
            return Err(Error::InvalidImage(
                "cpio: file body extends past end of archive".into(),
            ));
        }
        let next = body_off
            .checked_add(hdr.filesize)
            .map(|end| align_up(end, hdr.align))
            .filter(|&next| next <= total)
            .ok_or_else(|| Error::InvalidImage("cpio: record advance overflows archive".into()))?;
        // Forward-progress guard: a zero-length record would loop forever.
        if next <= pos {
            return Err(Error::InvalidImage(
                "cpio: record makes no forward progress".into(),
            ));
        }

        if name == TRAILER {
            break;
        }

        let kind = kind_from_mode(hdr.mode);
        let mut entry = ArchiveEntry {
            path: name,
            kind,
            mode: (hdr.mode & 0o7777) as u16,
            uid: hdr.uid,
            gid: hdr.gid,
            mtime: hdr.mtime,
            link_target: None,
            device_major: hdr.rdevmajor,
            device_minor: hdr.rdevminor,
            data: None,
        };

        match kind {
            EntryKind::Regular | EntryKind::HardLink => {
                entry.data = Some(DataLocator {
                    offset: body_off,
                    compressed_len: hdr.filesize,
                    uncompressed_len: hdr.filesize,
                    method: Method::Stored,
                });
            }
            EntryKind::Symlink => {
                let mut tbuf = vec![0u8; hdr.filesize as usize];
                dev.read_at(body_off, &mut tbuf)?;
                entry.link_target = Some(String::from_utf8_lossy(&tbuf).into_owned());
            }
            _ => {}
        }

        idx.push(entry);
        pos = next;
    }

    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    /// Build a 110-byte newc header with the given namesize/filesize hex
    /// fields; everything else zero.
    fn newc_header(namesize: u32, filesize: u32) -> Vec<u8> {
        let mut h = Vec::with_capacity(NEWC_HEADER_LEN as usize);
        h.extend_from_slice(MAGIC_NEWC); // magic (6)
        // 13 eight-char hex fields follow. filesize is field index 6,
        // namesize is field index 10 (0-based, after magic).
        for i in 0..13 {
            let v = match i {
                6 => filesize,
                10 => namesize,
                _ => 0,
            };
            h.extend_from_slice(format!("{v:08X}").as_bytes());
        }
        assert_eq!(h.len(), NEWC_HEADER_LEN as usize);
        h
    }

    #[test]
    fn rejects_oversized_namesize() {
        let hdr = newc_header(0xFFFF_FFFF, 0);
        let mut dev = MemoryBackend::new(4096);
        dev.write_at(0, &hdr).unwrap();
        assert!(matches!(scan(&mut dev), Err(Error::InvalidImage(_))));
    }

    #[test]
    fn rejects_filesize_past_end() {
        // Small valid namesize ("a\0" padded), huge filesize.
        let hdr = newc_header(2, 0xFFFF_FFF0);
        let mut dev = MemoryBackend::new(4096);
        dev.write_at(0, &hdr).unwrap();
        dev.write_at(NEWC_HEADER_LEN, b"a\0").unwrap();
        assert!(matches!(scan(&mut dev), Err(Error::InvalidImage(_))));
    }
}
