//! `cpio` writer: emits the **newc** (`070701`) format.
//!
//! Each `create_*` streams one record straight to the device at a
//! bumping cursor (header + NUL-terminated name + 4-byte pad, then body
//! + 4-byte pad). `flush` writes the `TRAILER!!!` sentinel.

use crate::block::BlockDevice;
use crate::fs::archive::ArchiveBuilder;
use crate::fs::archive::tree;
use crate::fs::archive::writer::Cursor;
use crate::fs::{DeviceKind, FileMeta, FileSource};
use crate::{Error, Result};

const NEWC_HEADER_LEN: usize = 110;

pub struct CpioWriter {
    cursor: Cursor,
    ino: u32,
}

impl CpioWriter {
    pub fn new(dev: &dyn BlockDevice) -> Self {
        Self {
            cursor: Cursor::new(dev),
            ino: 1,
        }
    }

    fn next_ino(&mut self) -> u32 {
        let i = self.ino;
        self.ino = self.ino.wrapping_add(1).max(1);
        i
    }
}

fn pad4(n: u64) -> usize {
    ((4 - (n % 4)) % 4) as usize
}

fn put_hex(h: &mut [u8], off: usize, v: u32) {
    let s = format!("{v:08X}");
    h[off..off + 8].copy_from_slice(s.as_bytes());
}

/// Build a 110-byte newc header.
#[allow(clippy::too_many_arguments)]
fn newc_header(
    ino: u32,
    mode: u32,
    uid: u32,
    gid: u32,
    nlink: u32,
    mtime: u32,
    filesize: u32,
    rdevmajor: u32,
    rdevminor: u32,
    namesize: u32,
) -> [u8; NEWC_HEADER_LEN] {
    let mut h = [0u8; NEWC_HEADER_LEN];
    h[0..6].copy_from_slice(super::MAGIC_NEWC);
    put_hex(&mut h, 6, ino);
    put_hex(&mut h, 14, mode);
    put_hex(&mut h, 22, uid);
    put_hex(&mut h, 30, gid);
    put_hex(&mut h, 38, nlink);
    put_hex(&mut h, 46, mtime);
    put_hex(&mut h, 54, filesize);
    put_hex(&mut h, 62, 0); // devmajor
    put_hex(&mut h, 70, 0); // devminor
    put_hex(&mut h, 78, rdevmajor);
    put_hex(&mut h, 86, rdevminor);
    put_hex(&mut h, 94, namesize);
    put_hex(&mut h, 102, 0); // c_check (0 for newc)
    h
}

impl CpioWriter {
    /// Write a record whose body is the in-memory `inline` bytes (or
    /// empty when `inline` is empty). Used for dirs/symlinks/devices.
    #[allow(clippy::too_many_arguments)]
    fn write_inline(
        &mut self,
        dev: &mut dyn BlockDevice,
        name: &str,
        mode: u32,
        meta: &FileMeta,
        rdevmajor: u32,
        rdevminor: u32,
        inline: &[u8],
    ) -> Result<()> {
        let namesize = name.len() as u64 + 1;
        let ino = self.next_ino();
        let h = newc_header(
            ino,
            mode,
            meta.uid,
            meta.gid,
            1,
            meta.mtime,
            inline.len() as u32,
            rdevmajor,
            rdevminor,
            namesize as u32,
        );
        self.cursor.write(dev, &h)?;
        self.cursor.write(dev, name.as_bytes())?;
        self.cursor.write(dev, &[0u8])?;
        let np = pad4(NEWC_HEADER_LEN as u64 + namesize);
        if np > 0 {
            self.cursor.write(dev, &[0u8; 4][..np])?;
        }
        if !inline.is_empty() {
            self.cursor.write(dev, inline)?;
            let bp = pad4(inline.len() as u64);
            if bp > 0 {
                self.cursor.write(dev, &[0u8; 4][..bp])?;
            }
        }
        Ok(())
    }
}

impl ArchiveBuilder for CpioWriter {
    fn add_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        src: FileSource,
        meta: FileMeta,
    ) -> Result<()> {
        let name = tree::normalise_path(path);
        let name = name.trim_start_matches('/');
        if name.is_empty() {
            return Err(Error::InvalidArgument("cpio: empty file path".into()));
        }
        let (mut reader, len) = src.open()?;
        if len > u64::from(u32::MAX) {
            return Err(Error::Unsupported(
                "cpio: newc format caps individual files at 4 GiB".into(),
            ));
        }
        let namesize = name.len() as u64 + 1;
        let ino = self.next_ino();
        let h = newc_header(
            ino,
            0o100000 | u32::from(meta.mode),
            meta.uid,
            meta.gid,
            1,
            meta.mtime,
            len as u32,
            0,
            0,
            namesize as u32,
        );
        self.cursor.write(dev, &h)?;
        self.cursor.write(dev, name.as_bytes())?;
        self.cursor.write(dev, &[0u8])?;
        let np = pad4(NEWC_HEADER_LEN as u64 + namesize);
        if np > 0 {
            self.cursor.write(dev, &[0u8; 4][..np])?;
        }
        // Stream the body straight through (reader derefs to dyn Read).
        let mut remaining = len;
        let mut buf = [0u8; 64 * 1024];
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            reader.read_exact(&mut buf[..want]).map_err(Error::from)?;
            self.cursor.write(dev, &buf[..want])?;
            remaining -= want as u64;
        }
        let bp = pad4(len);
        if bp > 0 {
            self.cursor.write(dev, &[0u8; 4][..bp])?;
        }
        Ok(())
    }

    fn add_dir(&mut self, dev: &mut dyn BlockDevice, path: &str, meta: FileMeta) -> Result<()> {
        let name = tree::normalise_path(path);
        let name = name.trim_start_matches('/');
        if name.is_empty() {
            return Ok(()); // root is implicit
        }
        self.write_inline(dev, name, 0o040000 | u32::from(meta.mode), &meta, 0, 0, &[])
    }

    fn add_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        target: &str,
        meta: FileMeta,
    ) -> Result<()> {
        let name = tree::normalise_path(path);
        let name = name.trim_start_matches('/');
        // Symlink target is the record body (no trailing NUL).
        self.write_inline(
            dev,
            name,
            0o120000 | u32::from(meta.mode),
            &meta,
            0,
            0,
            target.as_bytes(),
        )
    }

    fn add_device(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: FileMeta,
    ) -> Result<()> {
        let name = tree::normalise_path(path);
        let name = name.trim_start_matches('/');
        let type_bits = match kind {
            DeviceKind::Char => 0o020000,
            DeviceKind::Block => 0o060000,
            DeviceKind::Fifo => 0o010000,
            DeviceKind::Socket => 0o140000,
        };
        self.write_inline(
            dev,
            name,
            type_bits | u32::from(meta.mode),
            &meta,
            major,
            minor,
            &[],
        )
    }

    fn finish(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        // TRAILER!!! record: one link, zero size.
        let namesize = super::TRAILER.len() as u64 + 1;
        let h = newc_header(0, 0, 0, 0, 1, 0, 0, 0, 0, namesize as u32);
        self.cursor.write(dev, &h)?;
        self.cursor.write(dev, super::TRAILER.as_bytes())?;
        self.cursor.write(dev, &[0u8])?;
        let np = pad4(NEWC_HEADER_LEN as u64 + namesize);
        if np > 0 {
            self.cursor.write(dev, &[0u8; 4][..np])?;
        }
        dev.sync()?;
        Ok(())
    }

    fn position(&self) -> u64 {
        self.cursor.position()
    }
}
