//! `ar` reader: walk the member chain into an [`ArchiveIndex`].

use crate::block::BlockDevice;
use crate::fs::archive::{ArchiveEntry, ArchiveIndex, DataLocator, EntryKind, Method};
use crate::{Error, Result};

const HEADER_LEN: u64 = 60;

/// Parse a fixed-width ASCII numeric header field; blanks → `default`.
fn parse_field<T: std::str::FromStr>(field: &[u8], default: T) -> T {
    std::str::from_utf8(field)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

pub fn scan(dev: &mut dyn BlockDevice) -> Result<ArchiveIndex> {
    let total = dev.total_size();
    let mut magic = [0u8; 8];
    dev.read_at(0, &mut magic)?;
    if &magic != super::MAGIC {
        return Err(Error::InvalidImage(
            "ar: missing `!<arch>` global header".into(),
        ));
    }

    let mut idx = ArchiveIndex::new("ar");
    let mut gnu_table: Option<Vec<u8>> = None;
    let mut pos: u64 = 8;

    while pos + HEADER_LEN <= total {
        let mut hdr = [0u8; 60];
        dev.read_at(pos, &mut hdr)?;
        if &hdr[58..60] != b"`\n" {
            // Not a valid header terminator — stop (trailing slack or
            // corruption). Real archives always align here.
            break;
        }

        let raw_name = std::str::from_utf8(&hdr[0..16])
            .map_err(|_| Error::InvalidImage("ar: non-UTF-8 name field".into()))?
            .trim_end();
        let size: u64 = parse_field(&hdr[48..58], u64::MAX);
        if size == u64::MAX {
            return Err(Error::InvalidImage("ar: unparseable size field".into()));
        }
        let mtime: u64 = parse_field(&hdr[16..28], 0);
        let uid: u32 = parse_field(&hdr[28..34], 0);
        let gid: u32 = parse_field(&hdr[34..40], 0);
        let mode: u16 = std::str::from_utf8(&hdr[40..48])
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim(), 8).ok())
            .map(|m| (m & 0o7777) as u16)
            .unwrap_or(0o644);

        let data_off = pos + HEADER_LEN;
        // Reject a body that can't fit before allocating buffers sized
        // from the untrusted size field.
        if size > total - data_off {
            return Err(Error::InvalidImage(
                "ar: member body extends past end of archive".into(),
            ));
        }
        // Advance past this member (bodies are padded to an even offset).
        // Use checked arithmetic so a crafted size can't wrap the cursor.
        let next = data_off
            .checked_add(size)
            .and_then(|end| end.checked_add(size & 1))
            .filter(|&next| next <= total)
            .ok_or_else(|| Error::InvalidImage("ar: record advance overflows archive".into()))?;
        // Forward-progress guard: each record is at least HEADER_LEN bytes,
        // so a valid `next` must move strictly past `pos`.
        if next <= pos {
            return Err(Error::InvalidImage(
                "ar: record makes no forward progress".into(),
            ));
        }

        // GNU long-name string table.
        if raw_name == "//" {
            let mut buf = vec![0u8; size as usize];
            dev.read_at(data_off, &mut buf)?;
            gnu_table = Some(buf);
            pos = next;
            continue;
        }
        // Symbol tables (GNU `/`, BSD `__.SYMDEF`) — not files.
        if raw_name == "/" || raw_name == "/SYM64/" {
            pos = next;
            continue;
        }

        let (name, body_off, body_len) = if let Some(rest) = raw_name.strip_prefix("#1/") {
            // BSD extended name: <len> bytes of name precede the data.
            let nlen: u64 = rest
                .trim()
                .parse()
                .map_err(|_| Error::InvalidImage("ar: bad #1/ name length".into()))?;
            if nlen > size {
                return Err(Error::InvalidImage(
                    "ar: #1/ name longer than member".into(),
                ));
            }
            let mut nbuf = vec![0u8; nlen as usize];
            dev.read_at(data_off, &mut nbuf)?;
            let nm = String::from_utf8_lossy(&nbuf)
                .trim_end_matches('\0')
                .to_string();
            (nm, data_off + nlen, size - nlen)
        } else if let Some(off_str) = raw_name.strip_prefix('/') {
            // GNU long name: "/<offset>" into the `//` table.
            let off: usize = off_str
                .trim()
                .parse()
                .map_err(|_| Error::InvalidImage("ar: bad GNU long-name offset".into()))?;
            let table = gnu_table
                .as_ref()
                .ok_or_else(|| Error::InvalidImage("ar: GNU name ref without // table".into()))?;
            if off > table.len() {
                return Err(Error::InvalidImage(
                    "ar: GNU name offset out of range".into(),
                ));
            }
            let end = table[off..]
                .iter()
                .position(|&b| b == b'\n' || b == b'/')
                .map(|p| off + p)
                .unwrap_or(table.len());
            (
                String::from_utf8_lossy(&table[off..end]).to_string(),
                data_off,
                size,
            )
        } else {
            // Short name; GNU appends a trailing '/'.
            (
                raw_name.strip_suffix('/').unwrap_or(raw_name).to_string(),
                data_off,
                size,
            )
        };

        // Skip BSD symbol tables resolved through #1/.
        if name.is_empty() || name.starts_with("__.SYMDEF") {
            pos = next;
            continue;
        }

        idx.push(ArchiveEntry {
            path: name,
            kind: EntryKind::Regular,
            mode,
            uid,
            gid,
            mtime,
            link_target: None,
            device_major: 0,
            device_minor: 0,
            data: Some(DataLocator {
                offset: body_off,
                compressed_len: body_len,
                uncompressed_len: body_len,
                method: Method::Stored,
            }),
        });

        pos = next;
    }

    Ok(idx)
}
