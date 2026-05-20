//! Tar archives as a "filesystem" in fstool's `AnyFs` sense.
//!
//! Tar isn't a real filesystem — it's a sequential archive — but it
//! carries the same per-entry metadata as ext (path, mode, uid/gid,
//! mtime, symlink target, device numbers, xattrs). That makes it a
//! natural intermediate for `fstool repack`: ext → tar → ext mostly
//! preserves the tree (timestamps get rounded to whole seconds; xattrs
//! ride along as `SCHILY.xattr.*` PAX records).
//!
//! ## Scope
//!
//! - Read **ustar** + **PAX** entries with the records fstool cares
//!   about: `path`, `linkpath`, `size`, `mtime`, and `SCHILY.xattr.*`.
//! - Read GNU `L` / `K` long-name + long-link entries (the legacy form
//!   most existing archives use).
//! - Write **ustar** + **PAX** with the same coverage; PAX is emitted
//!   whenever a field doesn't fit the plain ustar (long path, long
//!   linkname, size > 8 GiB, non-ASCII path, xattrs present).
//! - No compression. No sparse files. No global PAX headers. Hard links
//!   are read but mapped to "regular file with same content" on
//!   destination — preserving link semantics across FS types is a
//!   separate problem.
//!
//! ## Random-access read path
//!
//! On `open`, the reader walks the whole archive once, recording each
//! entry's metadata and (for regular files) the byte offset where its
//! data begins. After that, `list_path` answers from the in-memory
//! tree and `open_file_reader` seeks straight to the data.

pub mod header;
pub mod pax;
pub mod stream;

pub use stream::{
    BoundedReader, IndexedEntry, StreamEntry, TarStreamIndex, TarStreamReader, TarStreamWriter,
};

use std::collections::HashMap;
use std::io::Read;

use header::{BLOCK_SIZE, Header};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::ext::xattr::Xattr;

/// Common write-side surface shared by [`TarWriter`] (BlockDevice-backed)
/// and [`TarStreamWriter`] (Write-backed). Callers that walk a source
/// filesystem and emit tar entries can be parametrised on this trait so
/// the same walker drives both back-ends.
pub trait TarSink {
    fn add_file(
        &mut self,
        path: &str,
        reader: &mut dyn Read,
        size: u64,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()>;
    fn add_dir(&mut self, path: &str, meta: TarEntryMeta, xattrs: &[Xattr]) -> Result<()>;
    fn add_symlink(
        &mut self,
        path: &str,
        target: &str,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()>;
    fn add_device(
        &mut self,
        path: &str,
        kind: crate::fs::DeviceKind,
        major: u32,
        minor: u32,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()>;
}

/// What a single tar entry represents on the destination side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    Regular,
    Dir,
    Symlink,
    HardLink,
    CharDev,
    BlockDev,
    Fifo,
}

impl EntryKind {
    fn from_typeflag(t: u8) -> Option<Self> {
        Some(match t {
            header::TYPEFLAG_REG | header::TYPEFLAG_REG_OLD | header::TYPEFLAG_CONT => {
                Self::Regular
            }
            header::TYPEFLAG_DIR => Self::Dir,
            header::TYPEFLAG_SYMLINK => Self::Symlink,
            header::TYPEFLAG_HARDLINK => Self::HardLink,
            header::TYPEFLAG_CHAR => Self::CharDev,
            header::TYPEFLAG_BLOCK => Self::BlockDev,
            header::TYPEFLAG_FIFO => Self::Fifo,
            _ => return None,
        })
    }
}

/// One archived entry, fully resolved (PAX overrides applied).
#[derive(Debug, Clone)]
pub struct Entry {
    pub path: String,
    pub kind: EntryKind,
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub mtime: u64,
    pub size: u64,
    pub link_target: Option<String>,
    pub device_major: u32,
    pub device_minor: u32,
    /// File offset of the data block immediately after the header.
    /// Only meaningful when `kind == Regular`.
    pub data_offset: u64,
    pub xattrs: Vec<Xattr>,
}

/// An opened tar archive.
pub struct Tar {
    entries: Vec<Entry>,
    /// Map from a *normalised* absolute path (always starts with `/`) to
    /// the index into `entries`.
    by_path: HashMap<String, usize>,
    /// Map from a normalised absolute directory path to the list of
    /// names contained directly underneath it (in archive order).
    children: HashMap<String, Vec<String>>,
}

impl Tar {
    /// Scan `dev` from offset 0, parsing every tar entry until the
    /// two-zero-block EOF marker. Returns the resolved in-memory index;
    /// no file contents are read.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let total = dev.total_size();
        let mut pos = 0u64;
        let mut block = [0u8; BLOCK_SIZE];
        let mut entries: Vec<Entry> = Vec::new();
        // PAX overrides that should be applied to the NEXT plain header.
        let mut pending: PaxOverrides = PaxOverrides::default();
        let mut consecutive_zero = 0u32;

        while pos + BLOCK_SIZE as u64 <= total {
            dev.read_at(pos, &mut block)?;
            if header::is_zero_block(&block) {
                consecutive_zero += 1;
                pos += BLOCK_SIZE as u64;
                if consecutive_zero >= 2 {
                    break;
                }
                continue;
            }
            consecutive_zero = 0;
            // Cheap sanity check: a real ustar header has the magic.
            // If the checksum is wrong we still bail out — almost
            // always indicates we walked off into garbage.
            if !Header::checksum_ok(&block) {
                return Err(crate::Error::InvalidImage(format!(
                    "tar: bad header checksum at offset {pos}"
                )));
            }
            let h = Header::decode(&block)?;
            let data_off = pos + BLOCK_SIZE as u64;
            let size_padded = (h.size + 511) & !511;
            match h.typeflag {
                header::TYPEFLAG_PAX => {
                    // Read the PAX body.
                    let body = read_exact_at(dev, data_off, h.size as usize)?;
                    pending.merge(pax::decode_records(&body)?);
                    pos = data_off + size_padded;
                    continue;
                }
                header::TYPEFLAG_PAX_GLOBAL => {
                    // Ignore global headers — we don't propagate them.
                    pos = data_off + size_padded;
                    continue;
                }
                header::TYPEFLAG_GNU_LONGNAME => {
                    let body = read_exact_at(dev, data_off, h.size as usize)?;
                    pending.path = Some(trim_nul(body));
                    pos = data_off + size_padded;
                    continue;
                }
                header::TYPEFLAG_GNU_LONGLINK => {
                    let body = read_exact_at(dev, data_off, h.size as usize)?;
                    pending.linkpath = Some(trim_nul(body));
                    pos = data_off + size_padded;
                    continue;
                }
                _ => {}
            }
            let Some(kind) = EntryKind::from_typeflag(h.typeflag) else {
                // Unknown typeflag — skip the entry but warn.
                eprintln!(
                    "tar: skipping entry {:?} with unknown typeflag {:?}",
                    h.full_name(),
                    h.typeflag as char
                );
                pos = data_off + size_padded;
                continue;
            };

            let path = pending.path.take().unwrap_or_else(|| h.full_name());
            let link_target = pending.linkpath.take().or_else(|| {
                if matches!(kind, EntryKind::Symlink | EntryKind::HardLink) {
                    Some(h.linkname.clone())
                } else {
                    None
                }
            });
            let size = pending.size.take().unwrap_or(h.size);
            let mtime = pending.mtime.take().unwrap_or(h.mtime);
            let xattrs = std::mem::take(&mut pending.xattrs);
            // Strip trailing `/` from directory paths so the index is
            // consistent.
            let mut path = path;
            if path.ends_with('/') {
                path.pop();
            }
            let path = normalise_path(&path);
            entries.push(Entry {
                path,
                kind,
                mode: h.mode,
                uid: h.uid,
                gid: h.gid,
                mtime,
                size,
                link_target,
                device_major: h.devmajor,
                device_minor: h.devminor,
                data_offset: data_off,
                xattrs,
            });

            pos = data_off
                + if matches!(kind, EntryKind::Regular) {
                    (size + 511) & !511
                } else {
                    0
                };
        }

        // Build the path → entry and parent → children indexes.
        let mut by_path = HashMap::new();
        let mut children: HashMap<String, Vec<String>> = HashMap::new();
        // The root "/" exists implicitly even if the archive doesn't
        // list it.
        children.entry("/".into()).or_default();
        for (i, e) in entries.iter().enumerate() {
            by_path.insert(e.path.clone(), i);
            let (parent, leaf) = split_path(&e.path);
            children
                .entry(parent.to_string())
                .or_default()
                .push(leaf.to_string());
            if matches!(e.kind, EntryKind::Dir) {
                children.entry(e.path.clone()).or_default();
            }
        }
        Ok(Self {
            entries,
            by_path,
            children,
        })
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn lookup(&self, path: &str) -> Option<&Entry> {
        let path = normalise_path(path);
        self.by_path.get(&path).map(|&i| &self.entries[i])
    }

    pub fn list_path(
        &self,
        _dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let path = normalise_path(path);
        let entries = self.children.get(&path).ok_or_else(|| {
            crate::Error::InvalidArgument(format!("tar: no such directory {path:?}"))
        })?;
        let mut out = Vec::with_capacity(entries.len());
        for name in entries {
            let child_path = if path == "/" {
                format!("/{name}")
            } else {
                format!("{path}/{name}")
            };
            // Tar can have a file listed without its parent dir.
            // Synthesise the missing parent on the fly: list it as Dir.
            let kind = self
                .by_path
                .get(&child_path)
                .map(|&i| match self.entries[i].kind {
                    EntryKind::Dir => crate::fs::EntryKind::Dir,
                    EntryKind::Regular | EntryKind::HardLink => crate::fs::EntryKind::Regular,
                    EntryKind::Symlink => crate::fs::EntryKind::Symlink,
                    EntryKind::CharDev => crate::fs::EntryKind::Char,
                    EntryKind::BlockDev => crate::fs::EntryKind::Block,
                    EntryKind::Fifo => crate::fs::EntryKind::Fifo,
                })
                .unwrap_or(crate::fs::EntryKind::Dir);
            // "inode": fold the entry index for stability across runs.
            let inode = self.by_path.get(&child_path).copied().unwrap_or(0) as u32 + 1;
            let size = if matches!(kind, crate::fs::EntryKind::Regular) {
                self.by_path
                    .get(&child_path)
                    .map(|&i| self.entries[i].size)
                    .unwrap_or(0)
            } else {
                0
            };
            out.push(crate::fs::DirEntry {
                name: name.clone(),
                inode,
                kind,
                size,
            });
        }
        Ok(out)
    }

    /// Open a streaming reader over a regular file's archive content.
    pub fn open_file_reader<'a>(
        &self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<TarFileReader<'a>> {
        let e = self
            .lookup(path)
            .ok_or_else(|| crate::Error::InvalidArgument(format!("tar: no such entry {path:?}")))?;
        if !matches!(e.kind, EntryKind::Regular | EntryKind::HardLink) {
            return Err(crate::Error::InvalidArgument(format!(
                "tar: {path:?} is not a regular file"
            )));
        }
        Ok(TarFileReader {
            dev,
            offset: e.data_offset,
            remaining: e.size,
        })
    }
}

/// Read-only `Filesystem` adapter so `inspect::open(dev)` can return a
/// `Box<dyn Filesystem>` that walks a tar archive. Writes return
/// `Unsupported` — tar archives are sequential and `repack` is the
/// only way to produce a new one.
impl crate::fs::Filesystem for Tar {
    fn create_file(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &std::path::Path,
        _src: crate::fs::FileSource,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(crate::Error::RepackOnly {
            kind: "tar",
            op: "write",
        })
    }

    fn create_dir(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &std::path::Path,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(crate::Error::RepackOnly {
            kind: "tar",
            op: "write",
        })
    }

    fn create_symlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &std::path::Path,
        _target: &std::path::Path,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(crate::Error::RepackOnly {
            kind: "tar",
            op: "write",
        })
    }

    fn create_device(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &std::path::Path,
        _kind: crate::fs::DeviceKind,
        _major: u32,
        _minor: u32,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(crate::Error::RepackOnly {
            kind: "tar",
            op: "write",
        })
    }

    fn remove(&mut self, _dev: &mut dyn BlockDevice, _path: &std::path::Path) -> Result<()> {
        Err(crate::Error::RepackOnly {
            kind: "tar",
            op: "write",
        })
    }

    fn list(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("tar: non-UTF-8 path".into()))?;
        Tar::list_path(self, dev, s)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn std::io::Read + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("tar: non-UTF-8 path".into()))?;
        let r = self.open_file_reader(dev, s)?;
        Ok(Box::new(r))
    }

    fn flush(&mut self, _dev: &mut dyn BlockDevice) -> Result<()> {
        Ok(())
    }

    fn supports_mutation(&self) -> bool {
        false
    }

    fn read_symlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<std::path::PathBuf> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("tar: non-UTF-8 path".into()))?;
        let entry = self.lookup(s).ok_or_else(|| {
            crate::Error::InvalidArgument(format!("tar: no entry at {s:?}"))
        })?;
        if !matches!(entry.kind, EntryKind::Symlink) {
            return Err(crate::Error::InvalidArgument(format!(
                "tar: {s:?} is not a symlink"
            )));
        }
        entry
            .link_target
            .clone()
            .map(std::path::PathBuf::from)
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!("tar: symlink {s:?} has no link target"))
            })
    }
}

#[derive(Default, Debug)]
struct PaxOverrides {
    path: Option<String>,
    linkpath: Option<String>,
    size: Option<u64>,
    mtime: Option<u64>,
    xattrs: Vec<Xattr>,
}

impl PaxOverrides {
    fn merge(&mut self, records: Vec<pax::Record>) {
        for r in records {
            match r.key.as_str() {
                pax::KEY_PATH => self.path = Some(String::from_utf8_lossy(&r.value).into_owned()),
                pax::KEY_LINKPATH => {
                    self.linkpath = Some(String::from_utf8_lossy(&r.value).into_owned())
                }
                pax::KEY_SIZE => {
                    if let Ok(s) = std::str::from_utf8(&r.value)
                        && let Ok(n) = s.parse::<u64>()
                    {
                        self.size = Some(n);
                    }
                }
                pax::KEY_MTIME => {
                    if let Ok(s) = std::str::from_utf8(&r.value) {
                        // mtime may carry a fractional part (`123.456`);
                        // we only support seconds-precision for now.
                        let secs = s.split('.').next().unwrap_or(s);
                        if let Ok(n) = secs.parse::<u64>() {
                            self.mtime = Some(n);
                        }
                    }
                }
                k => {
                    if let Some(name) = k.strip_prefix(pax::XATTR_PREFIX) {
                        self.xattrs.push(Xattr {
                            name: name.to_string(),
                            value: r.value,
                        });
                    }
                    // Other PAX keys (`charset`, `comment`, …) are
                    // silently ignored.
                }
            }
        }
    }
}

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

fn read_exact_at(dev: &mut dyn BlockDevice, offset: u64, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    dev.read_at(offset, &mut buf)?;
    Ok(buf)
}

/// Normalise a path to start with `/` and not end with `/` (root is `/`).
fn normalise_path(p: &str) -> String {
    let trimmed = p.trim_end_matches('/');
    if trimmed.is_empty() {
        return "/".into();
    }
    if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{trimmed}")
    }
}

fn split_path(p: &str) -> (&str, &str) {
    // p is normalised to start with '/' and not end with '/'.
    match p.rfind('/') {
        Some(0) => ("/", &p[1..]),
        Some(i) => (&p[..i], &p[i + 1..]),
        None => ("/", p),
    }
}

/// Streaming reader over one regular-file entry. Reads through the
/// backing `BlockDevice`'s `read_at`; nothing is buffered beyond the
/// caller's destination slice.
pub struct TarFileReader<'a> {
    dev: &'a mut dyn BlockDevice,
    offset: u64,
    remaining: u64,
}

impl<'a> Read for TarFileReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let want = (buf.len() as u64).min(self.remaining) as usize;
        self.dev
            .read_at(self.offset, &mut buf[..want])
            .map_err(std::io::Error::other)?;
        self.offset += want as u64;
        self.remaining -= want as u64;
        Ok(want)
    }
}

// -- writer side ---------------------------------------------------------

/// Sequential tar writer. Each `add_*` call emits a PAX header (if any
/// field requires it) plus the standard ustar header plus padded
/// content. Call [`Self::finish`] when done to write the two-zero-block
/// end marker.
pub struct TarWriter<'a> {
    dev: &'a mut dyn BlockDevice,
    cursor: u64,
    /// We grow the underlying device on demand. For a `FileBackend` this
    /// works because the backing file's `set_len` was called by the
    /// factory; we just track our own cursor against `total_size`.
    capacity: u64,
}

impl<'a> TarWriter<'a> {
    pub fn new(dev: &'a mut dyn BlockDevice) -> Self {
        let capacity = dev.total_size();
        Self {
            dev,
            cursor: 0,
            capacity,
        }
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
        self.write_block(&h.encode()?)?;
        // Stream the file content, padding to a 512-byte boundary.
        let mut remaining = size;
        let mut buf = [0u8; 64 * 1024];
        while remaining > 0 {
            let want = remaining.min(buf.len() as u64) as usize;
            reader
                .read_exact(&mut buf[..want])
                .map_err(crate::Error::from)?;
            self.write_at_cursor(&buf[..want])?;
            remaining -= want as u64;
        }
        // Pad to the next 512-byte boundary.
        let pad = (BLOCK_SIZE - (size as usize % BLOCK_SIZE)) % BLOCK_SIZE;
        if pad > 0 {
            self.write_at_cursor(&[0u8; BLOCK_SIZE][..pad])?;
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
        self.write_block(&h.encode()?)?;
        Ok(())
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
        self.write_block(&h.encode()?)?;
        Ok(())
    }

    pub fn add_device(
        &mut self,
        path: &str,
        kind: crate::fs::DeviceKind,
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
            crate::fs::DeviceKind::Char => header::TYPEFLAG_CHAR,
            crate::fs::DeviceKind::Block => header::TYPEFLAG_BLOCK,
            crate::fs::DeviceKind::Fifo => header::TYPEFLAG_FIFO,
            crate::fs::DeviceKind::Socket => {
                // tar doesn't represent sockets; emit a fifo to keep the
                // tree intact and warn.
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
        self.write_block(&h.encode()?)?;
        Ok(())
    }

    /// Finish the archive: write two zero blocks (EOF marker).
    pub fn finish(&mut self) -> Result<()> {
        self.write_block(&[0u8; BLOCK_SIZE])?;
        self.write_block(&[0u8; BLOCK_SIZE])?;
        self.dev.sync()?;
        Ok(())
    }

    /// Bytes written so far. Used by the CLI to truncate the backing
    /// file to the actual archive length (it may have been pre-sized).
    pub fn cursor(&self) -> u64 {
        self.cursor
    }

    fn write_block(&mut self, block: &[u8; BLOCK_SIZE]) -> Result<()> {
        self.write_at_cursor(block)
    }

    fn write_at_cursor(&mut self, buf: &[u8]) -> Result<()> {
        if self.cursor + buf.len() as u64 > self.capacity {
            return Err(crate::Error::OutOfBounds {
                offset: self.cursor,
                len: buf.len() as u64,
                size: self.capacity,
            });
        }
        self.dev.write_at(self.cursor, buf)?;
        self.cursor += buf.len() as u64;
        Ok(())
    }

    fn write_pax_header(&mut self, ref_path: &str, records: &[pax::Record]) -> Result<()> {
        let body = pax::encode_records(records);
        // PAX header is itself a tar entry with typeflag 'x'.
        let meta = TarEntryMeta {
            mode: 0o644,
            uid: 0,
            gid: 0,
            mtime: 0,
            uname: String::new(),
            gname: String::new(),
        };
        // Use a stable, ustar-safe name based on the leaf so we don't
        // accidentally trigger another long-name escalation.
        let pax_name = format!(
            "{}/PaxHeaders/{}",
            ".".strip_suffix('/').unwrap_or("."),
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
        // PAX header carries the size of `body` exactly.
        h.size = body.len() as u64;
        self.write_block(&h.encode()?)?;
        // Body + pad.
        self.write_at_cursor(&body)?;
        let pad = (BLOCK_SIZE - (body.len() % BLOCK_SIZE)) % BLOCK_SIZE;
        if pad > 0 {
            self.write_at_cursor(&[0u8; BLOCK_SIZE][..pad])?;
        }
        Ok(())
    }
}

/// Per-entry metadata for the writer. Mirrors a tar header's
/// non-content fields; xattrs are passed separately because they emit
/// PAX records.
#[derive(Debug, Clone)]
pub struct TarEntryMeta {
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub mtime: u64,
    pub uname: String,
    pub gname: String,
}

impl Default for TarEntryMeta {
    fn default() -> Self {
        Self {
            mode: 0o644,
            uid: 0,
            gid: 0,
            mtime: 0,
            uname: String::new(),
            gname: String::new(),
        }
    }
}

fn build_header(
    full_path: &str,
    typeflag: u8,
    size: u64,
    linkname: Option<&str>,
    dev: (u32, u32),
    meta: &TarEntryMeta,
    long_path_via_pax: bool,
) -> Result<Header> {
    // If the path fits ustar and we're not using PAX for it, fill name
    // (and prefix). Otherwise put a short placeholder in `name` — the
    // PAX `path` record overrides it on the read side.
    let (name, prefix) = if !long_path_via_pax && pax::path_fits_ustar(full_path) {
        split_path_for_ustar(full_path)
    } else {
        // Use the last 100 bytes of the path as the fallback name so
        // tools that ignore PAX still get *something* coherent.
        let leaf = full_path
            .rsplit('/')
            .next()
            .unwrap_or(full_path)
            .chars()
            .take(header::NAME_LEN)
            .collect::<String>();
        (leaf, String::new())
    };
    let (linkname_short, _link_via_pax) = match linkname {
        Some(t) if t.len() <= header::LINKNAME_LEN && t.is_ascii() => (t.to_string(), false),
        Some(t) => (
            t.chars().take(header::LINKNAME_LEN).collect::<String>(),
            true,
        ),
        None => (String::new(), false),
    };
    Ok(Header {
        name,
        mode: meta.mode & 0o7777,
        uid: meta.uid,
        gid: meta.gid,
        size,
        mtime: meta.mtime,
        typeflag,
        linkname: linkname_short,
        uname: meta.uname.clone(),
        gname: meta.gname.clone(),
        devmajor: dev.0,
        devminor: dev.1,
        prefix,
    })
}

/// Split a path that's known to fit ustar into `(name, prefix)`.
/// Caller has already checked [`pax::path_fits_ustar`].
fn split_path_for_ustar(path: &str) -> (String, String) {
    if path.len() <= header::NAME_LEN {
        return (path.to_string(), String::new());
    }
    // Find the LATEST `/` such that the suffix fits in 100 bytes and
    // the prefix fits in 155.
    let bytes = path.as_bytes();
    for i in (0..bytes.len()).rev() {
        if bytes[i] == b'/' && i <= header::PREFIX_LEN && bytes.len() - i - 1 <= header::NAME_LEN {
            return (path[i + 1..].to_string(), path[..i].to_string());
        }
    }
    // Shouldn't reach here if path_fits_ustar was true.
    (path.to_string(), String::new())
}

// ── TarSink trait impls so the same walker drives both back-ends ───

impl<'a> TarSink for TarWriter<'a> {
    fn add_file(
        &mut self,
        path: &str,
        reader: &mut dyn Read,
        size: u64,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()> {
        TarWriter::add_file(self, path, reader, size, meta, xattrs)
    }
    fn add_dir(&mut self, path: &str, meta: TarEntryMeta, xattrs: &[Xattr]) -> Result<()> {
        TarWriter::add_dir(self, path, meta, xattrs)
    }
    fn add_symlink(
        &mut self,
        path: &str,
        target: &str,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()> {
        TarWriter::add_symlink(self, path, target, meta, xattrs)
    }
    fn add_device(
        &mut self,
        path: &str,
        kind: crate::fs::DeviceKind,
        major: u32,
        minor: u32,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()> {
        TarWriter::add_device(self, path, kind, major, minor, meta, xattrs)
    }
}

impl<W: std::io::Write> TarSink for TarStreamWriter<W> {
    fn add_file(
        &mut self,
        path: &str,
        reader: &mut dyn Read,
        size: u64,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()> {
        TarStreamWriter::add_file(self, path, reader, size, meta, xattrs)
    }
    fn add_dir(&mut self, path: &str, meta: TarEntryMeta, xattrs: &[Xattr]) -> Result<()> {
        TarStreamWriter::add_dir(self, path, meta, xattrs)
    }
    fn add_symlink(
        &mut self,
        path: &str,
        target: &str,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()> {
        TarStreamWriter::add_symlink(self, path, target, meta, xattrs)
    }
    fn add_device(
        &mut self,
        path: &str,
        kind: crate::fs::DeviceKind,
        major: u32,
        minor: u32,
        meta: TarEntryMeta,
        xattrs: &[Xattr],
    ) -> Result<()> {
        TarStreamWriter::add_device(self, path, kind, major, minor, meta, xattrs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn round_trip_minimal_archive() {
        // Build a small archive in memory: one file with xattrs, one
        // symlink, one nested file. Read it back and check the entries.
        let mut dev = MemoryBackend::new(64 * 1024);
        {
            let mut w = TarWriter::new(&mut dev);
            let meta = TarEntryMeta {
                mode: 0o640,
                uid: 1000,
                gid: 1000,
                mtime: 0x6000_0000,
                uname: "user".into(),
                gname: "group".into(),
            };
            let content = b"hello tar\n";
            let mut r: &[u8] = content;
            w.add_file(
                "/hello.txt",
                &mut r,
                content.len() as u64,
                meta.clone(),
                &[Xattr::new("user.tag", b"flag".to_vec())],
            )
            .unwrap();
            w.add_dir("/sub", meta.clone(), &[]).unwrap();
            let nested = b"nested\n";
            let mut nr: &[u8] = nested;
            w.add_file(
                "/sub/inside.txt",
                &mut nr,
                nested.len() as u64,
                meta.clone(),
                &[],
            )
            .unwrap();
            w.add_symlink("/link-to-hello", "hello.txt", meta, &[])
                .unwrap();
            w.finish().unwrap();
        }
        let tar = Tar::open(&mut dev).unwrap();
        let hello = tar.lookup("/hello.txt").unwrap();
        assert_eq!(hello.kind, EntryKind::Regular);
        assert_eq!(hello.mode, 0o640);
        assert_eq!(hello.size, 10);
        assert_eq!(hello.xattrs.len(), 1);
        assert_eq!(hello.xattrs[0].name, "user.tag");
        assert_eq!(hello.xattrs[0].value, b"flag");

        let sym = tar.lookup("/link-to-hello").unwrap();
        assert_eq!(sym.kind, EntryKind::Symlink);
        assert_eq!(sym.link_target.as_deref(), Some("hello.txt"));

        let nested = tar.lookup("/sub/inside.txt").unwrap();
        let mut reader = tar.open_file_reader(&mut dev, "/sub/inside.txt").unwrap();
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"nested\n");
        assert_eq!(nested.size, 7);

        let root = tar.list_path(&mut dev, "/").unwrap();
        let names: Vec<&str> = root.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"hello.txt"));
        assert!(names.contains(&"sub"));
        assert!(names.contains(&"link-to-hello"));
    }
}
