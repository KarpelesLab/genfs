//! ZIP archives as an fstool filesystem.
//!
//! - **Read:** a full central-directory scan — robust EOCD search past
//!   a trailing comment, ZIP64 (locator + EOCD record + per-entry
//!   extra field `0x0001`), Unix permissions/symlinks from the
//!   external-attributes field, and the encoding handling in
//!   [`encoding`]. `Stored` and `Deflate` bodies decode; other methods
//!   index but report `Unsupported` on read.
//! - **Write:** streams local headers + bodies at a bumping cursor
//!   (CRC-32 + sizes back-patched after each body), then the central
//!   directory + EOCD on flush. `Stored` and `Deflate`; ZIP64 is
//!   emitted only when an entry, offset, or count crosses the 32-bit
//!   limits. Names are written per the universal-zip technique (host =
//!   MS-DOS, bit 11 only for true UTF-8).

mod encoding;
mod scan;
mod write;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;

pub const SIG_LOCAL: u32 = 0x0403_4b50;
pub const SIG_CENTRAL: u32 = 0x0201_4b50;
pub const SIG_EOCD: u32 = 0x0605_4b50;
pub const SIG_ZIP64_EOCD: u32 = 0x0606_4b50;
pub const SIG_ZIP64_LOCATOR: u32 = 0x0706_4b50;

pub const METHOD_STORE: u16 = 0;
pub const METHOD_DEFLATE: u16 = 8;

/// Body compression chosen by the writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    Stored,
    Deflate,
}

/// ZIP creation options.
#[derive(Debug, Clone)]
pub struct ZipFormatOpts {
    pub method: Compression,
    /// DEFLATE level 0..=9 (6 = default).
    pub level: u32,
}

impl Default for ZipFormatOpts {
    fn default() -> Self {
        Self {
            method: Compression::Deflate,
            level: 6,
        }
    }
}

impl ZipFormatOpts {
    pub fn apply_options(&mut self, bag: &mut crate::format_opts::OptionMap) -> Result<()> {
        if let Some(c) = bag.take_str("compression") {
            self.method = match c.to_ascii_lowercase().as_str() {
                "store" | "stored" | "none" => Compression::Stored,
                "deflate" | "deflated" => Compression::Deflate,
                other => {
                    return Err(crate::Error::InvalidArgument(format!(
                        "zip: unknown compression {other:?} (use `stored` or `deflate`)"
                    )));
                }
            };
        }
        if let Some(l) = bag.take_u32("level")? {
            if l > 9 {
                return Err(crate::Error::InvalidArgument(format!(
                    "zip: compression level {l} out of range (0..=9)"
                )));
            }
            self.level = l;
        }
        Ok(())
    }
}

/// ZIP filesystem handle.
pub struct ZipFs(pub ArchiveFs);

impl ZipFs {
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self(ArchiveFs::from_index(scan::scan(dev)?)))
    }

    pub fn format(dev: &mut dyn BlockDevice, opts: &ZipFormatOpts) -> Result<Self> {
        Ok(Self(ArchiveFs::writer(
            "zip",
            Box::new(write::ZipWriter::new(dev, opts.clone())),
        )))
    }
}

impl crate::fs::FilesystemFactory for ZipFs {
    type FormatOpts = ZipFormatOpts;
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

crate::impl_archive_fs_filesystem!(ZipFs);

/// Convert a DOS date+time pair to seconds since the Unix epoch (UTC).
/// DOS time has 2-second resolution and a 1980 epoch.
pub(crate) fn dos_to_unix(dos_date: u16, dos_time: u16) -> u64 {
    let sec = ((dos_time & 0x1f) as u64) * 2;
    let min = ((dos_time >> 5) & 0x3f) as u64;
    let hour = ((dos_time >> 11) & 0x1f) as u64;
    let day = (dos_date & 0x1f) as u64;
    let month = ((dos_date >> 5) & 0x0f) as u64;
    let year = 1980 + ((dos_date >> 9) & 0x7f) as u64;
    days_from_civil(year, month.max(1), day.max(1)) * 86400 + hour * 3600 + min * 60 + sec
}

/// Convert seconds since the Unix epoch (UTC) to a DOS date+time pair.
/// Clamps below the 1980 epoch.
pub(crate) fn unix_to_dos(secs: u64) -> (u16, u16) {
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let (year, month, day) = civil_from_days(days);
    if year < 1980 {
        // DOS epoch floor: 1980-01-01 00:00:00.
        return (0x0021, 0);
    }
    let hour = (rem / 3600) as u16;
    let min = ((rem % 3600) / 60) as u16;
    let sec2 = ((rem % 60) / 2) as u16;
    let date = (((year - 1980) as u16) << 9) | ((month as u16) << 5) | (day as u16);
    let time = (hour << 11) | (min << 5) | sec2;
    (date, time)
}

/// Days from 1970-01-01 to `y-m-d` (Howard Hinnant's algorithm).
fn days_from_civil(y: u64, m: u64, d: u64) -> u64 {
    let y = if m <= 2 { y - 1 } else { y } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = ((m + 9) % 12) as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146097 + doe - 719468) as u64
}

/// Inverse of [`days_from_civil`]: civil `(year, month, day)` from a
/// day count since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, u64, u64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u64;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u64;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dos_time_round_trips() {
        // 2021-06-15 12:30:44 UTC.
        let secs = days_from_civil(2021, 6, 15) * 86400 + 12 * 3600 + 30 * 60 + 44;
        let (date, time) = unix_to_dos(secs);
        // DOS seconds have 2s resolution → 44 stays 44.
        assert_eq!(dos_to_unix(date, time), secs);
    }

    #[test]
    fn pre_1980_clamps() {
        let (date, time) = unix_to_dos(0); // 1970
        assert_eq!((date, time), (0x0021, 0));
    }
}
