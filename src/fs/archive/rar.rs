//! RAR (`.rar`) reader.
//!
//! Recognised by `detect_fs` via the `Rar!\x1A\x07` signature (byte 7 is
//! `0x01` for RAR5, `0x00` for RAR4). RAR compression is proprietary and
//! reverse-engineered; **creation is forbidden by the RAR licence**, so this
//! is read-only at best.
//!
//! With the `rar` Cargo feature this is a real **read-only RAR5** reader:
//! it walks the RAR5 block chain (vint-encoded headers), and extracts each
//! file's contiguous packed run — Store verbatim, or compressed via
//! `compcol::rar5` (LZ77/Huffman incl. the x86 E8/E8E9 filters). Non-solid
//! archives only.
//!
//! Out of scope (clean `Unsupported`): RAR4, solid archives, encryption,
//! the delta/ARM/other RAR5 filters compcol doesn't decode, multi-volume
//! sets. Without the `rar` feature this stays a detection-only scaffold.

use std::path::Path;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;
use crate::fs::{DirEntry, FileAttrs, FileReadHandle, Filesystem, MutationCapability};

/// RAR filesystem handle.
pub struct RarFs {
    fs: ArchiveFs,
    #[cfg(feature = "rar")]
    files: std::collections::HashMap<String, Option<imp::RarFile>>,
}

impl RarFs {
    #[cfg(feature = "rar")]
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        // Detection already matched `Rar!\x1A\x07`; branch on the version.
        // RAR5 marker is `Rar!\x1A\x07\x01\x00` — the version byte is [6]
        // (0x01 = RAR5, 0x00 = RAR4, whose marker is 7 bytes).
        let mut sig = [0u8; 8];
        dev.read_at(0, &mut sig)?;
        if sig[6] != 0x01 {
            // RAR4 (and the unlikely future v6+) — not implemented.
            return Ok(Self {
                fs: ArchiveFs::scaffold("rar"),
                files: std::collections::HashMap::new(),
            });
        }
        let p = imp::scan(dev)?;
        Ok(Self {
            fs: ArchiveFs::from_index(p.index),
            files: p.files,
        })
    }

    #[cfg(not(feature = "rar"))]
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self {
            fs: ArchiveFs::scaffold("rar"),
        })
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "rar: creating archives is not supported (RAR compression is proprietary)".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for RarFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl Filesystem for RarFs {
    fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &Path,
        src: crate::fs::FileSource,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        self.fs.create_file(dev, path, src, meta)
    }

    fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        self.fs.create_dir(dev, path, meta)
    }

    fn create_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &Path,
        target: &Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        self.fs.create_symlink(dev, path, target, meta)
    }

    fn create_device(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &Path,
        kind: crate::fs::DeviceKind,
        major: u32,
        minor: u32,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        self.fs.create_device(dev, path, kind, major, minor, meta)
    }

    fn remove(&mut self, dev: &mut dyn BlockDevice, path: &Path) -> Result<()> {
        self.fs.remove(dev, path)
    }

    fn list(&mut self, dev: &mut dyn BlockDevice, path: &Path) -> Result<Vec<DirEntry>> {
        self.fs.list(dev, path)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &Path,
    ) -> Result<Box<dyn std::io::Read + 'a>> {
        #[cfg(feature = "rar")]
        match self.lookup(path)? {
            imp::Lookup::File(f) => return imp::open_file(dev, &f),
            imp::Lookup::Unextractable(key) => {
                return Err(crate::Error::Unsupported(format!(
                    "rar: {key:?} cannot be extracted (solid, encrypted, or unsupported method)"
                )));
            }
            imp::Lookup::NotRegular => {}
        }
        self.fs.read_file(dev, path)
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &Path,
    ) -> Result<Box<dyn FileReadHandle + 'a>> {
        #[cfg(feature = "rar")]
        match self.lookup(path)? {
            imp::Lookup::File(f) => {
                use std::io::Read;
                let mut r = imp::open_file(dev, &f)?;
                let mut bytes = Vec::new();
                r.read_to_end(&mut bytes).map_err(crate::Error::from)?;
                return Ok(Box::new(imp::mem_handle(bytes)));
            }
            imp::Lookup::Unextractable(key) => {
                return Err(crate::Error::Unsupported(format!(
                    "rar: {key:?} cannot be extracted (solid, encrypted, or unsupported method)"
                )));
            }
            imp::Lookup::NotRegular => {}
        }
        self.fs.open_file_ro(dev, path)
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        self.fs.flush(dev)
    }

    fn streams_immediately(&self) -> bool {
        self.fs.streams_immediately()
    }

    fn image_len(&self) -> Option<u64> {
        self.fs.image_len()
    }

    fn mutation_capability(&self) -> MutationCapability {
        self.fs.mutation_capability()
    }

    fn read_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &Path,
    ) -> Result<std::path::PathBuf> {
        self.fs.read_symlink(dev, path)
    }

    fn getattr(&mut self, dev: &mut dyn BlockDevice, path: &Path) -> Result<FileAttrs> {
        self.fs.getattr(dev, path)
    }
}

#[cfg(feature = "rar")]
impl RarFs {
    fn lookup(&self, path: &Path) -> Result<imp::Lookup> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("rar: non-UTF-8 path".to_string()))?;
        let key = crate::fs::archive::tree::normalise_path(s);
        Ok(match self.files.get(&key) {
            Some(Some(f)) => imp::Lookup::File(*f),
            Some(None) => imp::Lookup::Unextractable(key),
            None => imp::Lookup::NotRegular,
        })
    }
}

#[cfg(feature = "rar")]
mod imp {
    //! RAR5 container parser + per-file decode.

    use std::collections::HashMap;
    use std::io::{self, Read};

    use crate::block::BlockDevice;
    use crate::fs::archive::reader::BoundedDevReader;
    use crate::fs::archive::{ArchiveEntry, ArchiveIndex, DataLocator, EntryKind, Method};
    use crate::{Error, Result};

    // RAR5 header types.
    const HEAD_FILE: u64 = 2;
    const HEAD_END: u64 = 5;
    // Common header flags.
    const HFLAG_EXTRA: u64 = 0x01;
    const HFLAG_DATA: u64 = 0x02;
    // File flags.
    const FFLAG_DIR: u64 = 0x01;
    const FFLAG_MTIME: u64 = 0x02;
    const FFLAG_CRC: u64 = 0x04;

    /// A RAR5 file's packed run + how to decode it.
    #[derive(Debug, Clone, Copy)]
    pub struct RarFile {
        pub data_offset: u64,
        pub pack_size: u64,
        pub unpack_size: u64,
        pub window: usize,
        pub store: bool,
    }

    pub enum Lookup {
        File(RarFile),
        Unextractable(String),
        NotRegular,
    }

    pub struct Parsed {
        pub index: ArchiveIndex,
        pub files: HashMap<String, Option<RarFile>>,
    }

    /// Read a RAR5 vint (base-128 LE, high bit = continue) from `buf` at
    /// `pos`. Returns `(value, bytes_consumed)`.
    fn read_vint(buf: &[u8], pos: usize) -> Option<(u64, usize)> {
        let mut val = 0u64;
        let mut shift = 0u32;
        let mut i = pos;
        loop {
            let b = *buf.get(i)?;
            i += 1;
            val |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Some((val, i - pos));
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
    }

    /// Cursor over a header byte slice.
    struct Cur<'a> {
        b: &'a [u8],
        p: usize,
    }
    impl<'a> Cur<'a> {
        fn vint(&mut self) -> Result<u64> {
            let (v, n) = read_vint(self.b, self.p)
                .ok_or_else(|| Error::InvalidImage("rar: truncated vint".into()))?;
            self.p += n;
            Ok(v)
        }
        fn u32le(&mut self) -> Result<u32> {
            let end = self.p + 4;
            let s = self
                .b
                .get(self.p..end)
                .ok_or_else(|| Error::InvalidImage("rar: truncated u32".into()))?;
            self.p = end;
            Ok(u32::from_le_bytes(s.try_into().unwrap()))
        }
        fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
            let b: &'a [u8] = self.b;
            let end = self.p + n;
            let s = b
                .get(self.p..end)
                .ok_or_else(|| Error::InvalidImage("rar: truncated field".into()))?;
            self.p = end;
            Ok(s)
        }
    }

    fn read_at(dev: &mut dyn BlockDevice, off: u64, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        dev.read_at(off, &mut buf)?;
        Ok(buf)
    }

    /// Parse a RAR5 archive into the directory tree + per-file table.
    pub fn scan(dev: &mut dyn BlockDevice) -> Result<Parsed> {
        let dev_len = dev.total_size();
        let mut index = ArchiveIndex::new("rar");
        let mut files: HashMap<String, Option<RarFile>> = HashMap::new();

        let mut pos: u64 = 8; // past the 8-byte signature
        while pos + 5 <= dev_len {
            // CRC32 (4) + HeaderSize vint. Read a small preamble.
            let pre_len = 16.min((dev_len - pos) as usize);
            let pre = read_at(dev, pos, pre_len)?;
            let (head_size, hs_len) = match read_vint(&pre, 4) {
                Some(v) => v,
                None => break,
            };
            let header_start = pos + 4 + hs_len as u64;
            let header_end = header_start + head_size;
            if header_end > dev_len {
                break;
            }
            let hdr = read_at(dev, header_start, head_size as usize)?;
            let mut c = Cur { b: &hdr, p: 0 };
            let htype = c.vint()?;
            let hflags = c.vint()?;
            let _extra_size = if hflags & HFLAG_EXTRA != 0 {
                c.vint()?
            } else {
                0
            };
            let data_size = if hflags & HFLAG_DATA != 0 {
                c.vint()?
            } else {
                0
            };

            if htype == HEAD_END {
                break;
            }
            if htype == HEAD_FILE {
                let fflags = c.vint()?;
                let unpack_size = c.vint()?;
                let _attributes = c.vint()?;
                if fflags & FFLAG_MTIME != 0 {
                    c.u32le()?;
                }
                if fflags & FFLAG_CRC != 0 {
                    c.u32le()?;
                }
                let comp = c.vint()?;
                let _host_os = c.vint()?;
                let name_len = c.vint()? as usize;
                let name = c.bytes(name_len)?;

                let path = crate::fs::archive::tree::normalise_path(&normalise_name(name));
                let is_dir = fflags & FFLAG_DIR != 0;
                if path != "/" && !is_dir {
                    let mut entry = ArchiveEntry::regular(
                        path.clone(),
                        DataLocator {
                            offset: 0,
                            compressed_len: 0,
                            uncompressed_len: unpack_size,
                            method: Method::Stored,
                        },
                    );
                    entry.kind = EntryKind::Regular;
                    index.push(entry);

                    // compression info: bit 6 solid, bits 7..=9 method,
                    // bits 10..=13 dict (window = 128 KiB << N).
                    let solid = comp & 0x40 != 0;
                    let method = (comp >> 7) & 0x7;
                    let dict_n = (comp >> 10) & 0xf;
                    let window = 0x20000usize << dict_n;
                    // method 0 = store; anything else is RAR5 LZ. Encryption
                    // (header type 4 anywhere, or per-file) and solid streams
                    // aren't handled → mark Unextractable.
                    let slice = if solid {
                        None
                    } else {
                        Some(RarFile {
                            data_offset: header_end,
                            pack_size: data_size,
                            unpack_size,
                            window,
                            store: method == 0,
                        })
                    };
                    files.insert(path, slice);
                } else if path != "/" && is_dir {
                    index.push(ArchiveEntry::dir(path));
                }
            }

            pos = header_end + data_size;
        }

        Ok(Parsed { index, files })
    }

    /// RAR stores forward-slash paths already; normalise to absolute.
    fn normalise_name(bytes: &[u8]) -> String {
        let raw = String::from_utf8_lossy(bytes);
        let mut out = String::new();
        for comp in raw.split(['/', '\\']) {
            if comp.is_empty() || comp == "." || comp == ".." {
                continue;
            }
            out.push('/');
            out.push_str(comp);
        }
        out
    }

    /// Open a file's bytes as a bounded reader: Store yields the packed run
    /// verbatim; compressed feeds `compcol::rar5` (capped at `unpack_size`).
    pub fn open_file<'a>(dev: &'a mut dyn BlockDevice, f: &RarFile) -> Result<Box<dyn Read + 'a>> {
        let run = BoundedDevReader::new(dev, f.data_offset, f.pack_size);
        if f.store {
            Ok(Box::new(run))
        } else {
            let dec = compcol::rar5::Decoder::with_unpack_size_and_window(f.unpack_size, f.window);
            Ok(Box::new(compcol::io::DecoderReader::new(run, dec)))
        }
    }

    /// Wrap decoded bytes as a seekable [`FileReadHandle`].
    pub fn mem_handle(bytes: Vec<u8>) -> MemHandle {
        let len = bytes.len() as u64;
        MemHandle {
            cur: io::Cursor::new(bytes),
            len,
        }
    }

    pub struct MemHandle {
        cur: io::Cursor<Vec<u8>>,
        len: u64,
    }
    impl Read for MemHandle {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.cur.read(buf)
        }
    }
    impl io::Seek for MemHandle {
        fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
            self.cur.seek(pos)
        }
    }
    impl crate::fs::FileReadHandle for MemHandle {
        fn len(&self) -> u64 {
            self.len
        }
    }
}

#[cfg(all(test, feature = "rar"))]
mod tests {
    use std::io::Read;
    use std::path::Path;

    use super::*;
    use crate::block::MemoryBackend;

    // The fixtures were created with `rar a -ma5` over these exact files and
    // verified with `unrar t`.
    fn hello() -> Vec<u8> {
        b"Hello, RAR5 reader!\n".repeat(50)
    }
    fn lorem() -> Vec<u8> {
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(40)
    }

    fn dev_from(arc: &[u8]) -> MemoryBackend {
        let mut dev = MemoryBackend::new(arc.len().max(1) as u64);
        dev.write_at(0, arc).unwrap();
        dev
    }

    fn read_file(arc: &[u8], path: &str) -> Result<Vec<u8>> {
        let mut dev = dev_from(arc);
        let mut fs = RarFs::open(&mut dev)?;
        let mut r = fs.read_file(&mut dev, Path::new(path))?;
        let mut out = Vec::new();
        r.read_to_end(&mut out).map_err(crate::Error::from)?;
        Ok(out)
    }

    #[test]
    fn store_archive_round_trip() {
        let arc = include_bytes!("testdata/store5.rar");
        assert_eq!(read_file(arc, "/hello.txt").unwrap(), hello());
        assert_eq!(read_file(arc, "/lorem.txt").unwrap(), lorem());

        let mut dev = dev_from(arc);
        let mut fs = RarFs::open(&mut dev).unwrap();
        let root = fs.list(&mut dev, Path::new("/")).unwrap();
        let names: Vec<_> = root.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"hello.txt") && names.contains(&"lorem.txt"),
            "{names:?}"
        );
    }

    /// Compressed (`-m3`) archive — exercises the `compcol::rar5` decoder.
    #[test]
    fn compressed_archive_round_trip() {
        let arc = include_bytes!("testdata/comp5.rar");
        assert_eq!(read_file(arc, "/hello.txt").unwrap(), hello());
        assert_eq!(read_file(arc, "/lorem.txt").unwrap(), lorem());
    }

    /// Cross-check our extraction of the committed compressed fixture against
    /// the reference `unrar` tool, when it's installed.
    #[test]
    fn matches_unrar_reference() {
        use std::process::Command;
        if Command::new("unrar").arg("--help").output().is_err() {
            eprintln!("skipping: unrar not installed");
            return;
        }
        let arc = include_bytes!("testdata/comp5.rar");
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(tmp.as_file_mut(), arc).unwrap();
        for name in ["hello.txt", "lorem.txt"] {
            // `unrar p` prints the file to stdout; `-inul`/`-idq` silence it.
            let out = Command::new("unrar")
                .args(["p", "-inul", tmp.path().to_str().unwrap(), name])
                .output()
                .unwrap();
            assert!(out.status.success(), "unrar p failed for {name}");
            assert_eq!(
                read_file(arc, &format!("/{name}")).unwrap(),
                out.stdout,
                "reader vs unrar mismatch for {name}"
            );
        }
    }

    /// Files that continue a solid stream aren't supported — they need the
    /// decoder state from the previous file. (The first file in a solid
    /// archive isn't flagged solid and still extracts.) `lorem.txt` is the
    /// second, solid-flagged member, so reading it returns a clean
    /// Unsupported.
    #[test]
    fn solid_member_is_unsupported() {
        let arc = include_bytes!("testdata/solid5.rar");
        assert_eq!(read_file(arc, "/hello.txt").unwrap(), hello()); // first: not solid
        let err = read_file(arc, "/lorem.txt").unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }
}
