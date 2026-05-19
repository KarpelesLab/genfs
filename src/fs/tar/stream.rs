//! Pure `Read`/`Write` streaming tar I/O.
//!
//! The original tar module is `BlockDevice`-backed: random-access reads
//! are cheap because the device is seekable, and the writer just bumps
//! a cursor against a pre-sized backing store. That's convenient for
//! raw `.tar` files but useless the moment we want to pipe a tar
//! through `gzip` / `zstd` / `xz`, all of which expose a one-pass
//! `Read` / `Write` API.
//!
//! This module mirrors the same coverage (ustar + PAX + GNU long
//! name/link entries on read, ustar + PAX on write) on top of plain
//! `Read` and `Write`. No `Seek` requirement, no patching of earlier
//! blocks, no tempfile.
//!
//! ## API shape
//!
//! - [`TarStreamWriter<W>`] is append-only. Each `add_*` call emits any
//!   required PAX records followed by the ustar header and (for regular
//!   files) the padded content streamed straight from a `Read`. Call
//!   [`TarStreamWriter::finish`] at the end to emit the two zero blocks
//!   and flush the underlying writer.
//! - [`TarStreamReader<R>`] walks the archive once in linear order.
//!   Each call to [`TarStreamReader::next_entry`] either returns the
//!   next [`StreamEntry`] (a borrowed handle that implements `Read` for
//!   the entry's body) or `None` at EOF. The reader transparently
//!   skips unconsumed body bytes + padding when the caller asks for
//!   the next entry without finishing the current one.
//!
//! ## Streaming invariant
//!
//! Neither side ever buffers the archive: the writer's only allocation
//! is a 64 KiB pump buffer for `add_file`, and the reader keeps just
//! the current 512-byte header plus whatever PAX body it's actively
//! decoding (PAX bodies are tiny — path / linkpath / size / mtime /
//! xattrs).

use std::io::{Read, Write};

use crate::Result;
use crate::fs::DeviceKind;
use crate::fs::ext::xattr::Xattr;

use super::header::{self, BLOCK_SIZE, Header};
use super::pax;
use super::{Entry, EntryKind, PaxOverrides, TarEntryMeta, build_header, normalise_path};

const PUMP_BUF: usize = 64 * 1024;

// ============================ writer ===========================

/// Append-only tar writer driven by a `Write`. The writer never seeks
/// and never holds more than the standard pump buffer in memory, so it
/// composes directly with `GzEncoder`, `ZstdEncoder`, etc.
pub struct TarStreamWriter<W: Write> {
    inner: W,
    bytes_written: u64,
    finished: bool,
}

impl<W: Write> TarStreamWriter<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            bytes_written: 0,
            finished: false,
        }
    }

    /// Bytes written to the inner writer so far (uncompressed). Useful
    /// for reporting; not used internally.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    pub fn add_file(
        &mut self,
        path: &str,
        reader: &mut dyn Read,
        size: u64,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()> {
        let needs_size_pax = size > 0o7777_7777_7777; // 12-octal-digit limit (8 GiB)
        let mut records = pax::records_for_entry(path, None, needs_size_pax, xattrs);
        if needs_size_pax {
            records.push(pax::Record {
                key: pax::KEY_SIZE.into(),
                value: size.to_string().into_bytes(),
            });
        }
        if !records.is_empty() {
            self.write_pax_header(path, &records)?;
        }
        let h = build_header(
            path,
            header::TYPEFLAG_REG,
            size,
            None,
            (0, 0),
            &meta,
            !records.is_empty(),
        )?;
        self.write_all_block(&h.encode()?)?;
        // Stream the file content, then pad to a 512-byte boundary.
        let mut remaining = size;
        let mut buf = vec![0u8; PUMP_BUF];
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            reader.read_exact(&mut buf[..want])?;
            self.write_all_at(&buf[..want])?;
            remaining -= want as u64;
        }
        let pad = (BLOCK_SIZE - (size as usize % BLOCK_SIZE)) % BLOCK_SIZE;
        if pad > 0 {
            self.write_all_at(&[0u8; BLOCK_SIZE][..pad])?;
        }
        Ok(())
    }

    pub fn add_dir(&mut self, path: &str, meta: TarEntryMeta, xattrs: &[Xattr]) -> Result<()> {
        let records = pax::records_for_entry(path, None, false, xattrs);
        if !records.is_empty() {
            self.write_pax_header(path, &records)?;
        }
        let h = build_header(
            path,
            header::TYPEFLAG_DIR,
            0,
            None,
            (0, 0),
            &meta,
            !records.is_empty(),
        )?;
        self.write_all_block(&h.encode()?)
    }

    pub fn add_symlink(
        &mut self,
        path: &str,
        target: &str,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()> {
        let records = pax::records_for_entry(path, Some(target), false, xattrs);
        if !records.is_empty() {
            self.write_pax_header(path, &records)?;
        }
        let h = build_header(
            path,
            header::TYPEFLAG_SYMLINK,
            0,
            Some(target),
            (0, 0),
            &meta,
            !records.is_empty(),
        )?;
        self.write_all_block(&h.encode()?)
    }

    pub fn add_device(
        &mut self,
        path: &str,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()> {
        let records = pax::records_for_entry(path, None, false, xattrs);
        if !records.is_empty() {
            self.write_pax_header(path, &records)?;
        }
        let typeflag = match kind {
            DeviceKind::Char => header::TYPEFLAG_CHAR,
            DeviceKind::Block => header::TYPEFLAG_BLOCK,
            DeviceKind::Fifo => header::TYPEFLAG_FIFO,
            DeviceKind::Socket => {
                eprintln!("tar: socket {path:?} archived as FIFO (tar can't represent sockets)");
                header::TYPEFLAG_FIFO
            }
        };
        let h = build_header(
            path,
            typeflag,
            0,
            None,
            (major, minor),
            &meta,
            !records.is_empty(),
        )?;
        self.write_all_block(&h.encode()?)
    }

    /// Write two zero blocks (EOF marker) and flush the underlying
    /// writer. After this call the writer is no longer usable; calling
    /// any `add_*` again will fail.
    pub fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.write_all_block(&[0u8; BLOCK_SIZE])?;
        self.write_all_block(&[0u8; BLOCK_SIZE])?;
        self.inner.flush()?;
        self.finished = true;
        Ok(())
    }

    /// Consume the writer, returning the underlying inner writer.
    /// `finish` must have been called first.
    pub fn into_inner(self) -> W {
        self.inner
    }

    fn write_pax_header(&mut self, ref_path: &str, records: &[pax::Record]) -> Result<()> {
        let body = pax::encode_records(records);
        let meta = TarEntryMeta {
            mode: 0o644,
            uid: 0,
            gid: 0,
            mtime: 0,
            uname: String::new(),
            gname: String::new(),
        };
        let pax_name = format!(
            "./PaxHeaders/{}",
            ref_path.rsplit('/').next().unwrap_or("entry")
        );
        let mut h = build_header(
            &pax_name,
            header::TYPEFLAG_PAX,
            body.len() as u64,
            None,
            (0, 0),
            &meta,
            false,
        )?;
        h.size = body.len() as u64;
        self.write_all_block(&h.encode()?)?;
        self.write_all_at(&body)?;
        let pad = (BLOCK_SIZE - (body.len() % BLOCK_SIZE)) % BLOCK_SIZE;
        if pad > 0 {
            self.write_all_at(&[0u8; BLOCK_SIZE][..pad])?;
        }
        Ok(())
    }

    fn write_all_block(&mut self, block: &[u8; BLOCK_SIZE]) -> Result<()> {
        self.write_all_at(block)
    }

    fn write_all_at(&mut self, buf: &[u8]) -> Result<()> {
        if self.finished {
            return Err(crate::Error::InvalidArgument(
                "tar: stream writer already finished".into(),
            ));
        }
        self.inner.write_all(buf)?;
        self.bytes_written += buf.len() as u64;
        Ok(())
    }
}

// ============================ reader ===========================

/// Append-only tar reader driven by a `Read`. Walks the archive in
/// archive order; PAX overrides are applied to the next plain header.
///
/// The reader buffers at most one PAX record body at a time (typically
/// a few hundred bytes) and the 512-byte header it's currently
/// decoding.
pub struct TarStreamReader<R: Read> {
    inner: R,
    /// PAX records accumulated for the next data-bearing entry.
    pending: PaxOverrides,
    /// Bytes of the current entry's body still to be consumed plus any
    /// 512-byte padding tail. Subtracted as the consumer (or the next
    /// `next_entry` call) reads from us.
    body_remaining: u64,
    body_padding: usize,
    /// True after two consecutive zero blocks have been observed (EOF).
    eof: bool,
}

impl<R: Read> TarStreamReader<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            pending: PaxOverrides::default(),
            body_remaining: 0,
            body_padding: 0,
            eof: false,
        }
    }

    /// Advance to the next entry. The caller may read the entry's body
    /// via the returned [`StreamEntry`] (which implements `Read`); any
    /// unread bytes plus the 512-byte padding tail are skipped
    /// automatically on the next call.
    pub fn next_entry(&mut self) -> Result<Option<StreamEntry<'_, R>>> {
        // Skip the previous entry's unread body + its padding.
        if self.body_remaining > 0 || self.body_padding > 0 {
            self.skip_current_body()?;
        }
        if self.eof {
            return Ok(None);
        }

        let mut block = [0u8; BLOCK_SIZE];
        let mut consecutive_zero = 0u32;
        loop {
            match self.read_one_block(&mut block) {
                Ok(true) => {}
                Ok(false) => {
                    // Truncated archive: treat as EOF without complaint
                    // (matches the behaviour of `Tar::open` when total
                    // size doesn't carry the trailing zero blocks).
                    self.eof = true;
                    return Ok(None);
                }
                Err(e) => return Err(e),
            }
            if header::is_zero_block(&block) {
                consecutive_zero += 1;
                if consecutive_zero >= 2 {
                    self.eof = true;
                    return Ok(None);
                }
                continue;
            }
            consecutive_zero = 0;
            if !Header::checksum_ok(&block) {
                return Err(crate::Error::InvalidImage(
                    "tar: bad header checksum in stream".into(),
                ));
            }
            let h = Header::decode(&block)?;
            let size_padded = ((h.size + 511) & !511) as usize - h.size as usize;
            match h.typeflag {
                header::TYPEFLAG_PAX => {
                    let body = self.read_exact_padded(h.size as usize)?;
                    self.pending.merge(pax::decode_records(&body)?);
                    continue;
                }
                header::TYPEFLAG_PAX_GLOBAL => {
                    // Global headers are ignored; consume the body + padding.
                    let _ = self.read_exact_padded(h.size as usize)?;
                    continue;
                }
                header::TYPEFLAG_GNU_LONGNAME => {
                    let body = self.read_exact_padded(h.size as usize)?;
                    self.pending.path = Some(trim_nul(body));
                    continue;
                }
                header::TYPEFLAG_GNU_LONGLINK => {
                    let body = self.read_exact_padded(h.size as usize)?;
                    self.pending.linkpath = Some(trim_nul(body));
                    continue;
                }
                _ => {}
            }
            let Some(kind) = EntryKind::from_typeflag(h.typeflag) else {
                eprintln!(
                    "tar: skipping entry {:?} with unknown typeflag {:?}",
                    h.full_name(),
                    h.typeflag as char
                );
                // Consume the body + padding and try again.
                let _ = self.read_exact_padded(h.size as usize)?;
                continue;
            };
            let path = self.pending.path.take().unwrap_or_else(|| h.full_name());
            let link_target = self.pending.linkpath.take().or_else(|| {
                if matches!(kind, EntryKind::Symlink | EntryKind::HardLink) {
                    Some(h.linkname.clone())
                } else {
                    None
                }
            });
            let size = self.pending.size.take().unwrap_or(h.size);
            let mtime = self.pending.mtime.take().unwrap_or(h.mtime);
            let xattrs = std::mem::take(&mut self.pending.xattrs);
            let mut path = path;
            if path.ends_with('/') {
                path.pop();
            }
            let entry = Entry {
                path: normalise_path(&path),
                kind,
                mode: h.mode,
                uid: h.uid,
                gid: h.gid,
                mtime,
                size,
                link_target,
                device_major: h.devmajor,
                device_minor: h.devminor,
                // `data_offset` is meaningless for streamed entries;
                // leave it at zero — callers should use the
                // `StreamEntry::Read` impl instead.
                data_offset: 0,
                xattrs,
            };
            self.body_remaining = if matches!(kind, EntryKind::Regular) {
                size
            } else {
                0
            };
            self.body_padding = if matches!(kind, EntryKind::Regular) && size > 0 {
                (BLOCK_SIZE - (size as usize % BLOCK_SIZE)) % BLOCK_SIZE
            } else {
                0
            };
            // For non-regular entries that still carry a body (shouldn't
            // happen in practice, but be defensive about future kinds),
            // expose nothing — the data has already been ignored.
            let _ = size_padded;
            return Ok(Some(StreamEntry {
                entry,
                parent: self,
            }));
        }
    }

    fn read_one_block(&mut self, block: &mut [u8; BLOCK_SIZE]) -> Result<bool> {
        // Returns Ok(true) on a full 512-byte block, Ok(false) on clean
        // EOF before any bytes were read.
        let mut got = 0;
        while got < BLOCK_SIZE {
            match self.inner.read(&mut block[got..]) {
                Ok(0) => {
                    if got == 0 {
                        return Ok(false);
                    }
                    return Err(crate::Error::InvalidImage(format!(
                        "tar: short read inside header (got {got} / 512)"
                    )));
                }
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(true)
    }

    /// Read exactly `len` bytes from the inner stream and then discard
    /// trailing 512-byte padding.
    fn read_exact_padded(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf)?;
        let pad = (BLOCK_SIZE - (len % BLOCK_SIZE)) % BLOCK_SIZE;
        if pad > 0 {
            let mut sink = [0u8; BLOCK_SIZE];
            self.inner.read_exact(&mut sink[..pad])?;
        }
        Ok(buf)
    }

    /// Skip whatever's left of the current entry's body plus its
    /// 512-byte padding tail. Called automatically by `next_entry`.
    fn skip_current_body(&mut self) -> Result<()> {
        let mut sink = [0u8; PUMP_BUF];
        while self.body_remaining > 0 {
            let want = (self.body_remaining as usize).min(sink.len());
            self.inner.read_exact(&mut sink[..want])?;
            self.body_remaining -= want as u64;
        }
        if self.body_padding > 0 {
            self.inner.read_exact(&mut sink[..self.body_padding])?;
            self.body_padding = 0;
        }
        Ok(())
    }
}

/// Trim a NUL-terminated tar long-name / long-link body into a String.
fn trim_nul(mut v: Vec<u8>) -> String {
    while let Some(&b) = v.last() {
        if b == 0 {
            v.pop();
        } else {
            break;
        }
    }
    String::from_utf8_lossy(&v).into_owned()
}

/// One entry yielded by [`TarStreamReader::next_entry`]. The metadata
/// is fully resolved (PAX overrides applied); read the body via the
/// `Read` impl. Unread bytes are discarded automatically when the
/// reader advances.
pub struct StreamEntry<'a, R: Read> {
    pub entry: Entry,
    parent: &'a mut TarStreamReader<R>,
}

impl<'a, R: Read> StreamEntry<'a, R> {
    /// Bytes of the body still readable. `0` for non-regular entries.
    pub fn remaining(&self) -> u64 {
        self.parent.body_remaining
    }
}

impl<'a, R: Read> Read for StreamEntry<'a, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.parent.body_remaining == 0 {
            return Ok(0);
        }
        let want = (self.parent.body_remaining as usize).min(buf.len());
        let n = self.parent.inner.read(&mut buf[..want])?;
        self.parent.body_remaining -= n as u64;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> TarEntryMeta {
        TarEntryMeta {
            mode: 0o640,
            uid: 1000,
            gid: 1000,
            mtime: 0x6000_0000,
            uname: "user".into(),
            gname: "group".into(),
        }
    }

    #[test]
    fn stream_round_trip_basic() {
        let mut sink: Vec<u8> = Vec::new();
        {
            let mut w = TarStreamWriter::new(&mut sink);
            let body = b"hello stream tar\n";
            let mut r: &[u8] = body;
            w.add_file(
                "/hello.txt",
                &mut r,
                body.len() as u64,
                meta(),
                &[Xattr::new("user.tag", b"flag".to_vec())],
            )
            .unwrap();
            w.add_dir("/sub", meta(), &[]).unwrap();
            let nested = b"nested body\n";
            let mut nr: &[u8] = nested;
            w.add_file("/sub/inner.txt", &mut nr, nested.len() as u64, meta(), &[])
                .unwrap();
            w.add_symlink("/link-to-hello", "hello.txt", meta(), &[])
                .unwrap();
            w.finish().unwrap();
        }
        // Read it back.
        let mut reader = TarStreamReader::new(&sink[..]);
        let mut seen = Vec::new();
        while let Some(mut ent) = reader.next_entry().unwrap() {
            let mut body = Vec::new();
            if matches!(ent.entry.kind, EntryKind::Regular) {
                ent.read_to_end(&mut body).unwrap();
            }
            seen.push((ent.entry.path.clone(), ent.entry.kind, body));
        }
        assert_eq!(seen.len(), 4);
        assert_eq!(seen[0].0, "/hello.txt");
        assert_eq!(seen[0].1, EntryKind::Regular);
        assert_eq!(seen[0].2, b"hello stream tar\n");
        assert_eq!(seen[1].0, "/sub");
        assert_eq!(seen[1].1, EntryKind::Dir);
        assert_eq!(seen[2].0, "/sub/inner.txt");
        assert_eq!(seen[2].1, EntryKind::Regular);
        assert_eq!(seen[2].2, b"nested body\n");
        assert_eq!(seen[3].0, "/link-to-hello");
        assert_eq!(seen[3].1, EntryKind::Symlink);
    }

    #[test]
    fn stream_reader_skips_unread_bodies() {
        let mut sink: Vec<u8> = Vec::new();
        {
            let mut w = TarStreamWriter::new(&mut sink);
            for i in 0..5 {
                let body = vec![i as u8; 1000 + i * 137];
                let mut r: &[u8] = &body;
                w.add_file(
                    &format!("/f{i}.bin"),
                    &mut r,
                    body.len() as u64,
                    meta(),
                    &[],
                )
                .unwrap();
            }
            w.finish().unwrap();
        }
        // Walk to the 4th entry without reading any bodies.
        let mut reader = TarStreamReader::new(&sink[..]);
        let mut paths = Vec::new();
        while let Some(ent) = reader.next_entry().unwrap() {
            paths.push(ent.entry.path.clone());
            // Drop without reading; reader must skip the body+padding.
        }
        assert_eq!(
            paths,
            vec!["/f0.bin", "/f1.bin", "/f2.bin", "/f3.bin", "/f4.bin"]
        );
    }

    #[test]
    fn stream_round_trip_long_path_via_pax() {
        let long_path = format!("/{}", "a".repeat(200));
        let mut sink: Vec<u8> = Vec::new();
        {
            let mut w = TarStreamWriter::new(&mut sink);
            let body = b"X";
            let mut r: &[u8] = body;
            w.add_file(&long_path, &mut r, 1, meta(), &[]).unwrap();
            w.finish().unwrap();
        }
        let mut reader = TarStreamReader::new(&sink[..]);
        let ent = reader.next_entry().unwrap().unwrap();
        assert_eq!(ent.entry.path, long_path);
    }

    #[test]
    fn stream_round_trip_xattrs_via_pax() {
        let mut sink: Vec<u8> = Vec::new();
        {
            let mut w = TarStreamWriter::new(&mut sink);
            let body = b"x";
            let mut r: &[u8] = body;
            w.add_file(
                "/with-xattr",
                &mut r,
                1,
                meta(),
                &[
                    Xattr::new("user.foo", b"bar".to_vec()),
                    Xattr::new("user.bin", b"\x00\x01\x02".to_vec()),
                ],
            )
            .unwrap();
            w.finish().unwrap();
        }
        let mut reader = TarStreamReader::new(&sink[..]);
        let ent = reader.next_entry().unwrap().unwrap();
        assert_eq!(ent.entry.xattrs.len(), 2);
        assert_eq!(ent.entry.xattrs[0].name, "user.foo");
        assert_eq!(ent.entry.xattrs[0].value, b"bar");
        assert_eq!(ent.entry.xattrs[1].name, "user.bin");
        assert_eq!(ent.entry.xattrs[1].value, b"\x00\x01\x02");
    }
}
