//! ZIP writer.
//!
//! Each `create_*` writes a local file header then streams the body at
//! a bumping cursor, computing the CRC-32 and compressed size on the
//! fly and back-patching them into the header afterwards (the device is
//! seekable). `flush` appends the central directory and the EOCD,
//! emitting ZIP64 structures only when an entry, offset, or count
//! crosses the 32-bit limit.
//!
//! Per the universal-zip technique we set the UTF-8 flag (bit 11) only
//! for non-ASCII names; because non-ASCII names are therefore never
//! unflagged, declaring host-OS = Unix (so we can carry POSIX mode bits
//! and symlinks for round-tripping) does not risk the Windows
//! mangling the note warns about.

#[cfg(feature = "gzip")]
use std::io::Write;

use super::encoding;
use super::{
    Compression, METHOD_DEFLATE, METHOD_STORE, SIG_CENTRAL, SIG_EOCD, SIG_LOCAL, SIG_ZIP64_EOCD,
    SIG_ZIP64_LOCATOR, ZipFormatOpts,
};
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveBuilder;
use crate::fs::archive::tree;
use crate::fs::archive::writer::Cursor;
use crate::fs::{DeviceKind, FileMeta, FileSource, ReadSeek};
use crate::{Error, Result};

const U32_MAX: u64 = 0xffff_ffff;
const HOST_UNIX: u16 = 3;
const S_IFLNK: u32 = 0o120000;

/// One central-directory record, accumulated until `finish`.
struct CdRec {
    name: String,
    utf8: bool,
    method: u16,
    crc: u32,
    comp_size: u64,
    uncomp_size: u64,
    local_offset: u64,
    dos_date: u16,
    dos_time: u16,
    /// Full Unix `st_mode` (type bits included) for the external attrs.
    unix_mode: u32,
    is_dir: bool,
}

pub struct ZipWriter {
    cursor: Cursor,
    opts: ZipFormatOpts,
    central: Vec<CdRec>,
}

/// `Write` sink that appends to the device through the shared cursor and
/// counts the bytes (for the DEFLATE compressed size). Only the DEFLATE
/// path uses it, so it's gated with the `gzip` feature.
#[cfg(feature = "gzip")]
struct CursorSink<'a> {
    cursor: &'a mut Cursor,
    dev: &'a mut dyn BlockDevice,
    written: u64,
}

#[cfg(feature = "gzip")]
impl Write for CursorSink<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.cursor
            .write(self.dev, buf)
            .map_err(std::io::Error::other)?;
        self.written += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn push_u16(v: &mut Vec<u8>, n: u16) {
    v.extend_from_slice(&n.to_le_bytes());
}
fn push_u32(v: &mut Vec<u8>, n: u32) {
    v.extend_from_slice(&n.to_le_bytes());
}
fn push_u64(v: &mut Vec<u8>, n: u64) {
    v.extend_from_slice(&n.to_le_bytes());
}

impl ZipWriter {
    pub fn new(dev: &dyn BlockDevice, opts: ZipFormatOpts) -> Self {
        Self {
            cursor: Cursor::new(dev),
            opts,
            central: Vec::new(),
        }
    }

    fn norm(path: &str) -> String {
        tree::normalise_path(path)
            .trim_start_matches('/')
            .to_string()
    }

    /// Pick the body method: honour the option, but force `Stored` for
    /// empty bodies and when DEFLATE isn't compiled in.
    fn body_method(&self, len: u64) -> Compression {
        if len == 0 {
            return Compression::Stored;
        }
        match self.opts.method {
            Compression::Stored => Compression::Stored,
            Compression::Deflate => {
                if cfg!(feature = "gzip") {
                    Compression::Deflate
                } else {
                    Compression::Stored
                }
            }
        }
    }

    /// Write a local file header + body and record the central entry.
    /// `body` streams `len` bytes; `inline` provides an in-memory body
    /// (symlink targets, empty dirs).
    #[allow(clippy::too_many_arguments)]
    fn write_entry(
        &mut self,
        dev: &mut dyn BlockDevice,
        name: &str,
        unix_mode: u32,
        meta: &FileMeta,
        is_dir: bool,
        mut body: Option<(Box<dyn ReadSeek + Send>, u64)>,
        inline: &[u8],
    ) -> Result<()> {
        let uncomp = body
            .as_ref()
            .map(|(_, n)| *n)
            .unwrap_or(inline.len() as u64);
        let method = if body.is_some() {
            self.body_method(uncomp)
        } else if inline.is_empty() {
            Compression::Stored
        } else {
            // small in-memory bodies stored verbatim
            Compression::Stored
        };
        let method_id = match method {
            Compression::Stored => METHOD_STORE,
            Compression::Deflate => METHOD_DEFLATE,
        };
        let utf8 = encoding::needs_utf8_flag(name);
        let (dos_date, dos_time) = super::unix_to_dos(u64::from(meta.mtime));
        let entry_zip64 = uncomp >= U32_MAX;
        let local_offset = self.cursor.position();
        let name_bytes = name.as_bytes();

        // --- local file header (30 bytes + name + optional zip64 extra) ---
        let mut hdr = Vec::with_capacity(30 + name_bytes.len() + 20);
        push_u32(&mut hdr, SIG_LOCAL);
        push_u16(&mut hdr, if entry_zip64 { 45 } else { 20 }); // version needed
        push_u16(&mut hdr, if utf8 { 0x0800 } else { 0 }); // gp flags
        push_u16(&mut hdr, method_id);
        push_u16(&mut hdr, dos_time);
        push_u16(&mut hdr, dos_date);
        push_u32(&mut hdr, 0); // crc — patched
        push_u32(&mut hdr, if entry_zip64 { U32_MAX as u32 } else { 0 }); // comp — patched
        push_u32(
            &mut hdr,
            if entry_zip64 {
                U32_MAX as u32
            } else {
                uncomp as u32
            },
        );
        push_u16(&mut hdr, name_bytes.len() as u16);
        push_u16(&mut hdr, if entry_zip64 { 20 } else { 0 }); // extra len
        hdr.extend_from_slice(name_bytes);
        if entry_zip64 {
            push_u16(&mut hdr, 0x0001);
            push_u16(&mut hdr, 16);
            push_u64(&mut hdr, uncomp);
            push_u64(&mut hdr, 0); // comp — patched
        }
        self.cursor.write(dev, &hdr)?;

        // --- body ---
        let (crc, comp_size) = match body.take() {
            Some((mut reader, len)) => self.stream_body(dev, &mut reader, len, method)?,
            None => {
                let crc = crc32fast::hash(inline);
                self.cursor.write(dev, inline)?;
                (crc, inline.len() as u64)
            }
        };

        // --- back-patch crc + sizes ---
        dev.write_at(local_offset + 14, &crc.to_le_bytes())?;
        if entry_zip64 {
            // comp lives in the zip64 extra (after id+size+uncomp = +8).
            let extra_data = local_offset + 30 + name_bytes.len() as u64 + 4;
            dev.write_at(extra_data + 8, &comp_size.to_le_bytes())?;
        } else {
            dev.write_at(local_offset + 18, &(comp_size as u32).to_le_bytes())?;
            dev.write_at(local_offset + 22, &(uncomp as u32).to_le_bytes())?;
        }

        self.central.push(CdRec {
            name: name.to_string(),
            utf8,
            method: method_id,
            crc,
            comp_size,
            uncomp_size: uncomp,
            local_offset,
            dos_date,
            dos_time,
            unix_mode,
            is_dir,
        });
        Ok(())
    }

    fn stream_body(
        &mut self,
        dev: &mut dyn BlockDevice,
        reader: &mut Box<dyn ReadSeek + Send>,
        len: u64,
        method: Compression,
    ) -> Result<(u32, u64)> {
        let mut crc = crc32fast::Hasher::new();
        let mut buf = vec![0u8; 64 * 1024];
        match method {
            Compression::Stored => {
                let mut remaining = len;
                while remaining > 0 {
                    let want = remaining.min(buf.len() as u64) as usize;
                    reader.read_exact(&mut buf[..want]).map_err(Error::from)?;
                    crc.update(&buf[..want]);
                    self.cursor.write(dev, &buf[..want])?;
                    remaining -= want as u64;
                }
                Ok((crc.finalize(), len))
            }
            Compression::Deflate => self.stream_deflate(dev, reader, len, &mut crc, &mut buf),
        }
    }

    #[cfg(feature = "gzip")]
    fn stream_deflate(
        &mut self,
        dev: &mut dyn BlockDevice,
        reader: &mut Box<dyn ReadSeek + Send>,
        len: u64,
        crc: &mut crc32fast::Hasher,
        buf: &mut [u8],
    ) -> Result<(u32, u64)> {
        let sink = CursorSink {
            cursor: &mut self.cursor,
            dev,
            written: 0,
        };
        let mut enc =
            flate2::write::DeflateEncoder::new(sink, flate2::Compression::new(self.opts.level));
        let mut remaining = len;
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            reader.read_exact(&mut buf[..want]).map_err(Error::from)?;
            crc.update(&buf[..want]);
            enc.write_all(&buf[..want]).map_err(Error::from)?;
            remaining -= want as u64;
        }
        let sink = enc.finish().map_err(Error::from)?;
        Ok((crc.clone().finalize(), sink.written))
    }

    #[cfg(not(feature = "gzip"))]
    fn stream_deflate(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _reader: &mut Box<dyn ReadSeek + Send>,
        _len: u64,
        _crc: &mut crc32fast::Hasher,
        _buf: &mut [u8],
    ) -> Result<(u32, u64)> {
        Err(Error::Unsupported(
            "zip: DEFLATE disabled — rebuild with `--features gzip`".into(),
        ))
    }
}

impl ArchiveBuilder for ZipWriter {
    fn add_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        src: FileSource,
        meta: FileMeta,
    ) -> Result<()> {
        let name = Self::norm(path);
        if name.is_empty() {
            return Err(Error::InvalidArgument("zip: empty file path".into()));
        }
        let (reader, len) = src.open()?;
        let unix_mode = 0o100000 | u32::from(meta.mode);
        self.write_entry(
            dev,
            &name,
            unix_mode,
            &meta,
            false,
            Some((reader, len)),
            &[],
        )
    }

    fn add_dir(&mut self, dev: &mut dyn BlockDevice, path: &str, meta: FileMeta) -> Result<()> {
        let mut name = Self::norm(path);
        if name.is_empty() {
            return Ok(()); // root is implicit
        }
        name.push('/'); // ZIP marks directories with a trailing slash
        let unix_mode = 0o040000 | u32::from(meta.mode);
        self.write_entry(dev, &name, unix_mode, &meta, true, None, &[])
    }

    fn add_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        target: &str,
        meta: FileMeta,
    ) -> Result<()> {
        let name = Self::norm(path);
        let unix_mode = S_IFLNK | u32::from(meta.mode);
        self.write_entry(dev, &name, unix_mode, &meta, false, None, target.as_bytes())
    }

    fn add_device(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &str,
        _kind: DeviceKind,
        _major: u32,
        _minor: u32,
        _meta: FileMeta,
    ) -> Result<()> {
        Err(Error::Unsupported(format!(
            "zip: cannot store device/special node {path:?} — ZIP has no such record"
        )))
    }

    fn finish(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        let cd_start = self.cursor.position();
        let records = std::mem::take(&mut self.central);

        for r in &records {
            let mut zip64 = Vec::new();
            if r.uncomp_size >= U32_MAX {
                push_u64(&mut zip64, r.uncomp_size);
            }
            if r.comp_size >= U32_MAX {
                push_u64(&mut zip64, r.comp_size);
            }
            if r.local_offset >= U32_MAX {
                push_u64(&mut zip64, r.local_offset);
            }
            let name_bytes = r.name.as_bytes();
            let external_attr = (r.unix_mode << 16) | if r.is_dir { 0x10 } else { 0 };

            let mut rec = Vec::with_capacity(46 + name_bytes.len() + zip64.len());
            push_u32(&mut rec, SIG_CENTRAL);
            push_u16(&mut rec, (HOST_UNIX << 8) | 45); // version made by
            push_u16(&mut rec, if zip64.is_empty() { 20 } else { 45 }); // needed
            push_u16(&mut rec, if r.utf8 { 0x0800 } else { 0 });
            push_u16(&mut rec, r.method);
            push_u16(&mut rec, r.dos_time);
            push_u16(&mut rec, r.dos_date);
            push_u32(&mut rec, r.crc);
            push_u32(&mut rec, r.comp_size.min(U32_MAX) as u32);
            push_u32(&mut rec, r.uncomp_size.min(U32_MAX) as u32);
            push_u16(&mut rec, name_bytes.len() as u16);
            push_u16(
                &mut rec,
                if zip64.is_empty() {
                    0
                } else {
                    zip64.len() as u16 + 4
                },
            );
            push_u16(&mut rec, 0); // comment len
            push_u16(&mut rec, 0); // disk start
            push_u16(&mut rec, 0); // internal attrs
            push_u32(&mut rec, external_attr);
            push_u32(&mut rec, r.local_offset.min(U32_MAX) as u32);
            rec.extend_from_slice(name_bytes);
            if !zip64.is_empty() {
                push_u16(&mut rec, 0x0001);
                push_u16(&mut rec, zip64.len() as u16);
                rec.extend_from_slice(&zip64);
            }
            self.cursor.write(dev, &rec)?;
        }

        let cd_end = self.cursor.position();
        let cd_size = cd_end - cd_start;
        let count = records.len() as u64;
        let need_zip64 = count > 0xffff || cd_size > U32_MAX || cd_start > U32_MAX;

        if need_zip64 {
            let z64_pos = self.cursor.position();
            let mut z = Vec::with_capacity(56);
            push_u32(&mut z, SIG_ZIP64_EOCD);
            push_u64(&mut z, 44); // size of remaining record
            push_u16(&mut z, (HOST_UNIX << 8) | 45);
            push_u16(&mut z, 45);
            push_u32(&mut z, 0); // disk
            push_u32(&mut z, 0); // disk w/ cd
            push_u64(&mut z, count);
            push_u64(&mut z, count);
            push_u64(&mut z, cd_size);
            push_u64(&mut z, cd_start);
            self.cursor.write(dev, &z)?;

            let mut loc = Vec::with_capacity(20);
            push_u32(&mut loc, SIG_ZIP64_LOCATOR);
            push_u32(&mut loc, 0); // disk of zip64 eocd
            push_u64(&mut loc, z64_pos);
            push_u32(&mut loc, 1); // total disks
            self.cursor.write(dev, &loc)?;
        }

        let mut eocd = Vec::with_capacity(22);
        push_u32(&mut eocd, SIG_EOCD);
        push_u16(&mut eocd, 0); // disk
        push_u16(&mut eocd, 0); // disk w/ cd
        push_u16(&mut eocd, count.min(0xffff) as u16);
        push_u16(&mut eocd, count.min(0xffff) as u16);
        push_u32(&mut eocd, cd_size.min(U32_MAX) as u32);
        push_u32(&mut eocd, cd_start.min(U32_MAX) as u32);
        push_u16(&mut eocd, 0); // comment len
        self.cursor.write(dev, &eocd)?;

        dev.sync()?;
        Ok(())
    }

    fn position(&self) -> u64 {
        self.cursor.position()
    }
}
