//! Tar (USTAR) header — the 512-byte block at the start of every entry.
//!
//! ```text
//!     0  100  name
//!   100    8  mode (octal, NUL-terminated)
//!   108    8  uid  (octal)
//!   116    8  gid  (octal)
//!   124   12  size (octal)
//!   136   12  mtime (octal)
//!   148    8  chksum (octal, 6 digits + NUL + space; computed over header
//!                     with this field treated as 8 ASCII spaces)
//!   156    1  typeflag
//!   157  100  linkname
//!   257    6  magic   = "ustar\0"
//!   263    2  version = "00"
//!   265   32  uname
//!   297   32  gname
//!   329    8  devmajor
//!   337    8  devminor
//!   345  155  prefix (for paths > 100 bytes split at a slash boundary)
//! ```
//!
//! Long fields and xattrs use PAX extended headers (see
//! [`super::pax`]) — those headers immediately precede the real entry
//! and override the matching plain-header fields.

use crate::Result;

pub const BLOCK_SIZE: usize = 512;
pub const NAME_LEN: usize = 100;
pub const PREFIX_LEN: usize = 155;
pub const LINKNAME_LEN: usize = 100;

/// Upper bound on the in-memory body of a *metadata* entry (PAX header,
/// PAX global header, GNU long name / long link). These describe a single
/// path, linkpath, or a handful of records — kilobytes in practice. The
/// header `size` field is attacker-controlled, so cap it before allocating
/// to avoid OOM from a hostile image. 8 MiB is wildly generous for any
/// legitimate path/record list.
pub const MAX_META_BODY: usize = 8 * 1024 * 1024;

pub const TYPEFLAG_REG: u8 = b'0';
pub const TYPEFLAG_REG_OLD: u8 = b'\0'; // pre-ustar regular
pub const TYPEFLAG_HARDLINK: u8 = b'1';
pub const TYPEFLAG_SYMLINK: u8 = b'2';
pub const TYPEFLAG_CHAR: u8 = b'3';
pub const TYPEFLAG_BLOCK: u8 = b'4';
pub const TYPEFLAG_DIR: u8 = b'5';
pub const TYPEFLAG_FIFO: u8 = b'6';
pub const TYPEFLAG_CONT: u8 = b'7';
pub const TYPEFLAG_PAX: u8 = b'x';
pub const TYPEFLAG_PAX_GLOBAL: u8 = b'g';
pub const TYPEFLAG_GNU_LONGNAME: u8 = b'L';
pub const TYPEFLAG_GNU_LONGLINK: u8 = b'K';

/// Decoded ustar header. Numeric fields are stored as their parsed
/// values; PAX overrides are applied by the higher-level reader.
#[derive(Debug, Clone)]
pub struct Header {
    pub name: String,
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub mtime: u64,
    pub typeflag: u8,
    pub linkname: String,
    pub uname: String,
    pub gname: String,
    pub devmajor: u32,
    pub devminor: u32,
    pub prefix: String,
}

impl Header {
    /// Concatenate `prefix` + `name` per the ustar rule. PAX `path`
    /// overrides supersede this.
    pub fn full_name(&self) -> String {
        if self.prefix.is_empty() {
            self.name.clone()
        } else {
            format!("{}/{}", self.prefix, self.name)
        }
    }

    pub fn decode(block: &[u8; BLOCK_SIZE]) -> Result<Self> {
        let name = read_string(&block[0..100]);
        let mode = parse_octal_u64(&block[100..108])? as u16;
        let uid = parse_octal_u64(&block[108..116])? as u32;
        let gid = parse_octal_u64(&block[116..124])? as u32;
        let size = parse_octal_u64(&block[124..136])?;
        let mtime = parse_octal_u64(&block[136..148])?;
        let typeflag = block[156];
        let linkname = read_string(&block[157..257]);
        // Magic check is lenient: accept "ustar\0" (POSIX) and "ustar "
        // (GNU). Many tools omit the version too — don't be strict.
        let devmajor_raw = &block[329..337];
        let devminor_raw = &block[337..345];
        let devmajor = if devmajor_raw.iter().all(|&b| b == 0) {
            0
        } else {
            parse_octal_u64(devmajor_raw)? as u32
        };
        let devminor = if devminor_raw.iter().all(|&b| b == 0) {
            0
        } else {
            parse_octal_u64(devminor_raw)? as u32
        };
        Ok(Self {
            name,
            mode,
            uid,
            gid,
            size,
            mtime,
            typeflag,
            linkname,
            uname: read_string(&block[265..297]),
            gname: read_string(&block[297..329]),
            devmajor,
            devminor,
            prefix: read_string(&block[345..500]),
        })
    }

    pub fn encode(&self) -> Result<[u8; BLOCK_SIZE]> {
        let mut block = [0u8; BLOCK_SIZE];
        write_string(&mut block[0..NAME_LEN], &self.name);
        write_octal(&mut block[100..108], self.mode as u64, 7)?;
        write_octal(&mut block[108..116], self.uid as u64, 7)?;
        write_octal(&mut block[116..124], self.gid as u64, 7)?;
        write_octal(&mut block[124..136], self.size, 11)?;
        write_octal(&mut block[136..148], self.mtime, 11)?;
        // chksum starts as 8 ASCII spaces for the checksum computation.
        block[148..156].copy_from_slice(b"        ");
        block[156] = self.typeflag;
        write_string(&mut block[157..157 + LINKNAME_LEN], &self.linkname);
        block[257..263].copy_from_slice(b"ustar\0");
        block[263..265].copy_from_slice(b"00");
        write_string(&mut block[265..297], &self.uname);
        write_string(&mut block[297..329], &self.gname);
        write_octal(&mut block[329..337], self.devmajor as u64, 7)?;
        write_octal(&mut block[337..345], self.devminor as u64, 7)?;
        write_string(&mut block[345..345 + PREFIX_LEN], &self.prefix);

        // Compute checksum: simple unsigned sum of every byte in the
        // header, with the chksum field treated as 8 spaces (which we
        // initialised above).
        let sum: u32 = block.iter().map(|&b| b as u32).sum();
        // Write as 6-digit octal, NUL, space.
        let s = format!("{sum:06o}");
        block[148..154].copy_from_slice(s.as_bytes());
        block[154] = 0;
        block[155] = b' ';
        Ok(block)
    }

    /// Verify the header's stored checksum. Returns true if valid.
    pub fn checksum_ok(block: &[u8; BLOCK_SIZE]) -> bool {
        // Sum with checksum field replaced by spaces.
        let stored = match parse_octal_u64(&block[148..156]) {
            Ok(n) => n as u32,
            Err(_) => return false,
        };
        let mut sum: u32 = 0;
        for (i, &b) in block.iter().enumerate() {
            sum += if (148..156).contains(&i) {
                0x20
            } else {
                b as u32
            };
        }
        sum == stored
    }
}

/// True when `block` is all zero bytes — the EOF marker (two consecutive
/// zero blocks end the archive).
pub fn is_zero_block(block: &[u8; BLOCK_SIZE]) -> bool {
    block.iter().all(|&b| b == 0)
}

/// Read a NUL-terminated (or full-field) string out of an ustar field.
fn read_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Write `s` into a fixed-width field, NUL-padded.
fn write_string(dst: &mut [u8], s: &str) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(dst.len());
    dst[..n].copy_from_slice(&bytes[..n]);
    // (remaining bytes were already zero)
}

/// Parse an octal numeric field. Spaces and NULs are treated as
/// terminators. Empty fields parse as 0.
pub fn parse_octal_u64(bytes: &[u8]) -> Result<u64> {
    // GNU tar's "base-256 binary" extension uses the top bit; we don't
    // emit it but we accept it on read for big sizes.
    if let Some(&first) = bytes.first()
        && first & 0x80 != 0
    {
        let mut acc: u64 = (first & 0x7f) as u64;
        for &b in &bytes[1..] {
            acc = (acc << 8) | (b as u64);
        }
        return Ok(acc);
    }
    let mut acc: u64 = 0;
    for &b in bytes {
        if b == 0 || b == b' ' {
            break;
        }
        if !(b'0'..=b'7').contains(&b) {
            return Err(crate::Error::InvalidImage(format!(
                "tar: bad octal digit {b:#x}"
            )));
        }
        acc = acc * 8 + (b - b'0') as u64;
    }
    Ok(acc)
}

/// Write `n` as a zero-padded octal number into `dst[..digits]`, then a
/// NUL terminator at `dst[digits]`. Width includes the trailing NUL:
/// `dst.len() == digits + 1`.
pub fn write_octal(dst: &mut [u8], n: u64, digits: usize) -> Result<()> {
    let s = format!("{n:0>width$o}", width = digits);
    if s.len() > digits {
        return Err(crate::Error::Unsupported(format!(
            "tar: value {n} ({s} octal) does not fit in {digits} digits — use PAX"
        )));
    }
    dst[..digits].copy_from_slice(s.as_bytes());
    if digits < dst.len() {
        dst[digits] = 0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_basic_regular() {
        let h = Header {
            name: "hello.txt".into(),
            mode: 0o640,
            uid: 1000,
            gid: 1000,
            size: 13,
            mtime: 0x6000_0000,
            typeflag: TYPEFLAG_REG,
            linkname: String::new(),
            uname: "user".into(),
            gname: "group".into(),
            devmajor: 0,
            devminor: 0,
            prefix: String::new(),
        };
        let block = h.encode().unwrap();
        assert!(Header::checksum_ok(&block));
        let d = Header::decode(&block).unwrap();
        assert_eq!(d.name, "hello.txt");
        assert_eq!(d.mode, 0o640);
        assert_eq!(d.uid, 1000);
        assert_eq!(d.size, 13);
        assert_eq!(d.typeflag, TYPEFLAG_REG);
    }

    #[test]
    fn octal_parse_handles_space_and_nul() {
        assert_eq!(parse_octal_u64(b"0001000\0").unwrap(), 0o1000);
        assert_eq!(parse_octal_u64(b"123 \0\0\0\0\0").unwrap(), 0o123);
        assert!(parse_octal_u64(b"99\0").is_err());
    }

    #[test]
    fn detects_zero_block() {
        let zeros = [0u8; BLOCK_SIZE];
        assert!(is_zero_block(&zeros));
        let mut nz = [0u8; BLOCK_SIZE];
        nz[3] = 1;
        assert!(!is_zero_block(&nz));
    }
}
