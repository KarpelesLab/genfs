//! Rock Ridge (IEEE P1282) System Use Sharing Protocol entries.
//!
//! We parse the entries we surface through the read API:
//!
//! - `SP` (System Use Protocol indicator) — must appear on the "." entry
//!   of the root directory for RR to be active.
//! - `RR` — bit-flags advertising which other entries to expect (older
//!   convention; newer media drops it).
//! - `NM` — alternate (long) name. Concatenates across `CONTINUE` flag.
//! - `PX` — POSIX file mode + nlink + uid + gid. Bytes 4..36, both-endian.
//! - `SL` — symlink target. Composed of `Component Records`.
//! - `CE` — continuation area pointer (LBA + offset + length). When set,
//!   the remaining SUA bytes live at that location.
//! - `TF` — timestamps. We pick mtime out of it for the user-facing API.
//!
//! Everything else (`PN`, `CL`, `PL`, `SF`, vendor-specific) is parsed far
//! enough to be skipped over.

use crate::Result;
use crate::block::BlockDevice;

use super::SECTOR_SIZE;
use super::directory::DirRecord;
use super::vd::PrimaryVolumeDescriptor;

/// Cooked Rock Ridge attributes for one directory entry. Only the
/// fields the read path needs are kept; the writer constructs SUA bytes
/// directly without going through this struct.
#[derive(Debug, Default, Clone)]
pub struct RockRidgeAttrs {
    pub alternate_name: Option<String>,
    pub symlink_target: Option<String>,
    pub mode: Option<u32>,
    pub uid: Option<u32>,
    pub gid: Option<u32>,
    pub mtime: Option<i64>,
}

/// Walk the System Use Area of a directory record and produce the
/// cooked attributes. Returns `None` if the SUA contained nothing of
/// interest (the caller can then fall back to defaults).
pub fn parse_system_use(dev: &mut dyn BlockDevice, sua: &[u8]) -> Option<RockRidgeAttrs> {
    let mut attrs = RockRidgeAttrs::default();
    let mut name = String::new();
    let mut symlink = String::new();
    let mut had_any = false;

    parse_block(sua, dev, &mut attrs, &mut name, &mut symlink, &mut had_any);

    if !had_any {
        return None;
    }
    if !name.is_empty() {
        attrs.alternate_name = Some(name);
    }
    if !symlink.is_empty() {
        attrs.symlink_target = Some(symlink);
    }
    Some(attrs)
}

/// Detect SP / ER on the root's "." entry. If present, Rock Ridge is
/// active for the whole volume.
pub fn root_has_rr(dev: &mut dyn BlockDevice, pvd: &PrimaryVolumeDescriptor) -> Result<bool> {
    // The root directory's first record (".") sits at the very start
    // of the root extent. Read one sector and decode.
    let mut buf = vec![0u8; SECTOR_SIZE as usize];
    dev.read_at(
        u64::from(pvd.root.extent_lba) * u64::from(SECTOR_SIZE),
        &mut buf,
    )?;
    if buf.is_empty() || buf[0] == 0 {
        return Ok(false);
    }
    let len_dr = buf[0] as usize;
    if len_dr < 33 || len_dr > buf.len() {
        return Ok(false);
    }
    let dot = DirRecord::decode(&buf[..len_dr])?;
    // SP signature: "SP" + 2 bytes len + version + 0xBE 0xEF magic.
    for window in dot.system_use.windows(4) {
        if &window[..2] == b"SP" && window[2] >= 7 {
            // Look for the SP magic at offset 4..6 of the SP entry.
            let off = dot.system_use.as_ptr() as usize - dot.system_use.as_ptr() as usize; // 0
            let _ = off;
            // Just check that SP is present and the magic 0xBE 0xEF
            // follows at the right offset.
            // Find the SP at its actual position in the SUA.
            for i in 0..(dot.system_use.len().saturating_sub(7)) {
                if &dot.system_use[i..i + 2] == b"SP"
                    && dot.system_use.get(i + 4) == Some(&0xBE)
                    && dot.system_use.get(i + 5) == Some(&0xEF)
                {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Walk a contiguous SUA block, mutating the accumulator in place.
/// `CE` entries recursively follow into a continuation area on disk.
fn parse_block(
    sua: &[u8],
    dev: &mut dyn BlockDevice,
    attrs: &mut RockRidgeAttrs,
    name_acc: &mut String,
    symlink_acc: &mut String,
    any: &mut bool,
) {
    let mut i = 0;
    while i + 4 <= sua.len() {
        let sig = &sua[i..i + 2];
        let len = sua[i + 2] as usize;
        if len < 4 || i + len > sua.len() {
            break;
        }
        let body = &sua[i + 4..i + len];
        match sig {
            b"NM" => {
                if let Some(payload) = body.get(1..) {
                    let cont_flag = body.first().copied().unwrap_or(0);
                    name_acc.push_str(&String::from_utf8_lossy(payload));
                    if cont_flag & 0x01 == 0 {
                        // No continuation — name complete.
                    }
                    *any = true;
                }
            }
            b"PX" => {
                if body.len() >= 32 {
                    if let Ok(mode) = super::vd::decode_both_endian_u32(&body[0..8], "PX.mode") {
                        attrs.mode = Some(mode);
                    }
                    if let Ok(_nlink) = super::vd::decode_both_endian_u32(&body[8..16], "PX.nlink")
                    {
                        // nlink not surfaced today.
                    }
                    if let Ok(uid) = super::vd::decode_both_endian_u32(&body[16..24], "PX.uid") {
                        attrs.uid = Some(uid);
                    }
                    if let Ok(gid) = super::vd::decode_both_endian_u32(&body[24..32], "PX.gid") {
                        attrs.gid = Some(gid);
                    }
                    *any = true;
                }
            }
            b"SL" => {
                // body[0] = flags, body[1..] = component records.
                let cont = body.first().copied().unwrap_or(0) & 0x01;
                let mut j = 1;
                while j + 2 <= body.len() {
                    let comp_flags = body[j];
                    let comp_len = body[j + 1] as usize;
                    if j + 2 + comp_len > body.len() {
                        break;
                    }
                    let comp_bytes = &body[j + 2..j + 2 + comp_len];
                    if !symlink_acc.is_empty() && (comp_flags & 0x08) == 0 {
                        symlink_acc.push('/');
                    }
                    // Bit flags per IEEE P1282 §4.1.3.1:
                    //   0x01: CONTINUE — concat without a slash
                    //   0x02: CURRENT  ("./")
                    //   0x04: PARENT   ("../")
                    //   0x08: ROOT     ("/")
                    if comp_flags & 0x02 != 0 {
                        symlink_acc.push('.');
                    } else if comp_flags & 0x04 != 0 {
                        symlink_acc.push_str("..");
                    } else if comp_flags & 0x08 != 0 {
                        symlink_acc.push('/');
                    } else {
                        symlink_acc.push_str(&String::from_utf8_lossy(comp_bytes));
                    }
                    j += 2 + comp_len;
                }
                let _ = cont;
                *any = true;
            }
            b"CE" => {
                if body.len() >= 24 {
                    let ce_lba = super::vd::decode_both_endian_u32(&body[0..8], "CE.lba").ok();
                    let ce_off = super::vd::decode_both_endian_u32(&body[8..16], "CE.offset").ok();
                    let ce_len = super::vd::decode_both_endian_u32(&body[16..24], "CE.len").ok();
                    if let (Some(lba), Some(off), Some(clen)) = (ce_lba, ce_off, ce_len) {
                        let mut buf = vec![0u8; clen as usize];
                        let abs = u64::from(lba) * u64::from(SECTOR_SIZE) + u64::from(off);
                        if dev.read_at(abs, &mut buf).is_ok() {
                            parse_block(&buf, dev, attrs, name_acc, symlink_acc, any);
                        }
                    }
                }
            }
            b"TF" => {
                // Timestamps. body[0] = flags bitmap (0x01 creation,
                // 0x02 modify, 0x04 access, ...). Each timestamp is 7
                // bytes (or 17 if flag bit 0x80 is set for long form).
                let flags = body.first().copied().unwrap_or(0);
                let long = flags & 0x80 != 0;
                let entry_size = if long { 17 } else { 7 };
                let bits = flags & 0x7F;
                let mut k = 1;
                // We want bit 0x02 (modify time).
                let mut bit = 0u8;
                while bit < 7 {
                    if bits & (1 << bit) != 0 && k + entry_size <= body.len() {
                        if bit == 1 {
                            // mtime
                            attrs.mtime = Some(decode_iso_short_time(&body[k..k + entry_size]));
                        }
                        k += entry_size;
                    }
                    bit += 1;
                }
                *any = true;
            }
            b"SP" | b"RR" | b"ER" | b"ES" | b"PD" | b"ST" | b"PN" | b"CL" | b"PL" | b"SF" => {
                *any = true;
            }
            _ => { /* unknown / vendor entry — skip */ }
        }
        i += len;
    }
}

/// ECMA-119 §9.1.5 short-form time: 7 bytes — years since 1900, month,
/// day, hour, minute, second, GMT offset (15-minute units).
fn decode_iso_short_time(buf: &[u8]) -> i64 {
    if buf.len() < 7 {
        return 0;
    }
    let years = i64::from(buf[0]);
    let month = i64::from(buf[1]);
    let day = i64::from(buf[2]);
    let hour = i64::from(buf[3]);
    let minute = i64::from(buf[4]);
    let second = i64::from(buf[5]);
    let gmt_off_qh = i8::from_le_bytes([buf[6]]) as i64;
    // Days-from-civil — Howard Hinnant.
    let y = 1900 + years - if month <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64;
    let m_shift = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * m_shift + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    days * 86400 + hour * 3600 + minute * 60 + second - gmt_off_qh * 15 * 60
}
