//! `cpio` reader: walk the record chain (newc / newc-crc / odc) into an
//! [`ArchiveIndex`].

use super::{MAGIC_NEWC, MAGIC_NEWC_CRC, MAGIC_ODC, S_IFMT, TRAILER};
use crate::block::BlockDevice;
use crate::fs::archive::{ArchiveEntry, ArchiveIndex, DataLocator, EntryKind, Method};
use crate::{Error, Result};

const NEWC_HEADER_LEN: u64 = 110;
const ODC_HEADER_LEN: u64 = 76;

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
        let mut name_buf = vec![0u8; hdr.namesize as usize];
        dev.read_at(name_off, &mut name_buf)?;
        while name_buf.last() == Some(&0) {
            name_buf.pop();
        }
        let name = String::from_utf8_lossy(&name_buf).into_owned();

        // Records are 4-byte aligned from the archive start (newc) so a
        // file-offset alignment equals record-relative alignment.
        let body_off = align_up(name_off + hdr.namesize, hdr.align);
        let next = align_up(body_off + hdr.filesize, hdr.align);

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
