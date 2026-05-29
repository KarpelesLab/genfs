//! Amiga LZX (`.lzx`, `LZX`) reader.
//!
//! Recognised by `detect_fs` via the `LZX` signature. This is the Amiga
//! archive format (Jonathan Forbes, 1995), distinct from the Microsoft LZX
//! *compression method* used inside CAB/CHM.
//!
//! With the `amiga-lzx` Cargo feature this is a real **read-only** reader.
//! The container is reverse-engineered (per `unlzx`): a 10-byte `LZX` info
//! header, then a sequence of 31-byte entry headers (each followed by its
//! filename and comment). Files are grouped — several consecutive entries
//! can share one compressed run (the "merged" group), owned by the entry
//! whose `pack_size > 0`; the others carry `pack_size == 0`. The group's
//! decompressed stream is the concatenation of its files in order, and each
//! file is a byte slice of it (offset = running sum of earlier sizes).
//!
//! Supported pack modes: 0 (store) and 2 (LZX, via `compcol::amiga_lzx`).
//! Files are streamed out of their group (skip-to-offset + length cap) so
//! memory stays bounded. Archive creation is not supported.
//!
//! Without the `amiga-lzx` feature this stays a detection-only scaffold.

use std::path::Path;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;
use crate::fs::{DirEntry, FileAttrs, FileReadHandle, Filesystem, MutationCapability};

/// Amiga LZX filesystem handle.
pub struct LzxFs {
    fs: ArchiveFs,
    #[cfg(feature = "amiga-lzx")]
    groups: Vec<imp::Group>,
    #[cfg(feature = "amiga-lzx")]
    files: std::collections::HashMap<String, Option<imp::FileSlice>>,
}

impl LzxFs {
    #[cfg(feature = "amiga-lzx")]
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let p = imp::scan(dev)?;
        Ok(Self {
            fs: ArchiveFs::from_index(p.index),
            groups: p.groups,
            files: p.files,
        })
    }

    #[cfg(not(feature = "amiga-lzx"))]
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self {
            fs: ArchiveFs::scaffold("lzx"),
        })
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "lzx: creating archives is not supported".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for LzxFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl Filesystem for LzxFs {
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
        #[cfg(feature = "amiga-lzx")]
        match self.lookup(path)? {
            imp::Lookup::Slice(slice, group) => {
                let mut r = imp::decode_group_reader(dev, &group)?;
                imp::skip_exact(&mut *r, slice.uoff)?;
                return Ok(Box::new(imp::LimitReader::new(r, slice.len)));
            }
            imp::Lookup::Unextractable(key) => {
                return Err(crate::Error::Unsupported(format!(
                    "lzx: {key:?} cannot be extracted (unsupported pack mode or truncated)"
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
        #[cfg(feature = "amiga-lzx")]
        match self.lookup(path)? {
            imp::Lookup::Slice(slice, group) => {
                use std::io::Read;
                let mut r = imp::decode_group_reader(dev, &group)?;
                imp::skip_exact(&mut *r, slice.uoff)?;
                let mut bytes = Vec::new();
                (&mut *r)
                    .take(slice.len)
                    .read_to_end(&mut bytes)
                    .map_err(crate::Error::from)?;
                return Ok(Box::new(imp::mem_handle(bytes)));
            }
            imp::Lookup::Unextractable(key) => {
                return Err(crate::Error::Unsupported(format!(
                    "lzx: {key:?} cannot be extracted (unsupported pack mode or truncated)"
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

#[cfg(feature = "amiga-lzx")]
impl LzxFs {
    fn lookup(&self, path: &Path) -> Result<imp::Lookup> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("lzx: non-UTF-8 path".to_string()))?;
        let key = crate::fs::archive::tree::normalise_path(s);
        Ok(match self.files.get(&key) {
            Some(Some(slice)) => imp::Lookup::Slice(*slice, self.groups[slice.group]),
            Some(None) => imp::Lookup::Unextractable(key),
            None => imp::Lookup::NotRegular,
        })
    }
}

#[cfg(feature = "amiga-lzx")]
mod imp {
    //! Amiga LZX container parser + streaming group decode.

    use std::collections::HashMap;
    use std::io::{self, Read};

    use compcol::Algorithm;

    use crate::block::BlockDevice;
    use crate::fs::archive::reader::BoundedDevReader;
    use crate::fs::archive::{ArchiveEntry, ArchiveIndex, DataLocator, EntryKind, Method};
    use crate::{Error, Result};

    /// A compressed group: one contiguous run owned by an entry with
    /// `pack_size > 0`, covering one or more (merged) files.
    #[derive(Debug, Clone, Copy)]
    pub struct Group {
        pub data_offset: u64,
        pub pack_size: u64,
        pub pack_mode: u8,
        pub total_uncomp: u64,
    }

    /// A file's slice of its group's decompressed stream.
    #[derive(Debug, Clone, Copy)]
    pub struct FileSlice {
        pub group: usize,
        pub uoff: u64,
        pub len: u64,
    }

    pub struct Parsed {
        pub index: ArchiveIndex,
        pub groups: Vec<Group>,
        pub files: HashMap<String, Option<FileSlice>>,
    }

    pub enum Lookup {
        Slice(FileSlice, Group),
        Unextractable(String),
        NotRegular,
    }

    #[inline]
    fn le32(b: &[u8], o: usize) -> u32 {
        u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }

    fn read_at(dev: &mut dyn BlockDevice, off: u64, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        dev.read_at(off, &mut buf)?;
        Ok(buf)
    }

    /// Parse the archive into the directory tree + group/file side tables.
    pub fn scan(dev: &mut dyn BlockDevice) -> Result<Parsed> {
        let dev_len = dev.total_size();
        let info = read_at(dev, 0, 10)?;
        if &info[0..3] != b"LZX" {
            return Err(Error::InvalidImage(
                "lzx: bad signature (expected LZX)".into(),
            ));
        }

        let mut index = ArchiveIndex::new("lzx");
        let mut groups: Vec<Group> = Vec::new();
        let mut files: HashMap<String, Option<FileSlice>> = HashMap::new();
        // Entries pending assignment to the group whose owner closes it.
        let mut pending: Vec<(String, u64, u64)> = Vec::new(); // (path, uoff, len)
        let mut merge_size: u64 = 0;

        let mut pos: u64 = 10;
        while pos + 31 <= dev_len {
            let h = read_at(dev, pos, 31)?;
            pos += 31;
            let unpack = le32(&h, 2) as u64;
            let pack = le32(&h, 6) as u64;
            let pack_mode = h[11];
            let comment_len = h[14] as u64;
            let name_len = h[30] as usize;

            let name_bytes = read_at(dev, pos, name_len)?;
            pos += name_len as u64;
            pos += comment_len; // skip the comment

            let path = crate::fs::archive::tree::normalise_path(&normalise_name(&name_bytes));
            if path != "/" {
                let mut entry = ArchiveEntry::regular(
                    path.clone(),
                    DataLocator {
                        offset: 0,
                        compressed_len: 0,
                        uncompressed_len: unpack,
                        method: Method::Stored,
                    },
                );
                entry.kind = EntryKind::Regular;
                index.push(entry);
                pending.push((path, merge_size, unpack));
            }
            merge_size += unpack;

            if pack > 0 {
                // This entry owns the compressed run for the pending group.
                let group_idx = groups.len();
                groups.push(Group {
                    data_offset: pos,
                    pack_size: pack,
                    pack_mode,
                    total_uncomp: merge_size,
                });
                let extractable = matches!(pack_mode, 0 | 2);
                for (p, uoff, len) in pending.drain(..) {
                    files.insert(
                        p,
                        extractable.then_some(FileSlice {
                            group: group_idx,
                            uoff,
                            len,
                        }),
                    );
                }
                pos += pack; // skip the packed data
                merge_size = 0;
            }
        }
        // Any entries with no owning run (truncated archive) can't be read.
        for (p, _, _) in pending.drain(..) {
            files.insert(p, None);
        }

        Ok(Parsed {
            index,
            groups,
            files,
        })
    }

    /// Amiga paths use `/` separators and may carry a volume/`:`; produce a
    /// normalised absolute slash path.
    fn normalise_name(bytes: &[u8]) -> String {
        // Latin-1 decode (Amiga filenames are 8-bit), then split on '/' and
        // drop any leading "volume:" component.
        let raw: String = bytes.iter().map(|&b| b as char).collect();
        let raw = raw.rsplit(':').next().unwrap_or(&raw);
        let mut out = String::new();
        for comp in raw.split('/') {
            if comp.is_empty() || comp == "." || comp == ".." {
                continue;
            }
            out.push('/');
            out.push_str(comp);
        }
        out
    }

    /// Build a lazy reader over `group`'s decompressed stream (bounded
    /// memory). Store yields the raw run; LZX feeds compcol's `amiga_lzx`
    /// decoder (its 4-byte length prefix prepended) over the run.
    pub fn decode_group_reader<'a>(
        dev: &'a mut dyn BlockDevice,
        group: &Group,
    ) -> Result<Box<dyn Read + 'a>> {
        match group.pack_mode {
            0 => Ok(Box::new(BoundedDevReader::new(
                dev,
                group.data_offset,
                group.pack_size,
            ))),
            2 => {
                let total = u32::try_from(group.total_uncomp)
                    .map_err(|_| Error::Unsupported("lzx: group larger than 4 GiB".into()))?;
                let framed = io::Cursor::new(total.to_le_bytes().to_vec()).chain(
                    BoundedDevReader::new(dev, group.data_offset, group.pack_size),
                );
                Ok(Box::new(compcol::io::DecoderReader::new(
                    framed,
                    compcol::amiga_lzx::AmigaLzx::decoder(),
                )))
            }
            other => Err(Error::Unsupported(format!(
                "lzx: pack mode {other} not supported"
            ))),
        }
    }

    /// Read and discard exactly `n` bytes from `r`.
    pub fn skip_exact(r: &mut dyn Read, mut n: u64) -> Result<()> {
        let mut scratch = [0u8; 64 * 1024];
        while n > 0 {
            let want = n.min(scratch.len() as u64) as usize;
            let got = r.read(&mut scratch[..want]).map_err(crate::Error::from)?;
            if got == 0 {
                return Err(Error::InvalidImage(
                    "lzx: group stream ended before the file offset".into(),
                ));
            }
            n -= got as u64;
        }
        Ok(())
    }

    /// Caps an owned boxed reader to `remaining` bytes.
    pub struct LimitReader<'a> {
        inner: Box<dyn Read + 'a>,
        remaining: u64,
    }

    impl<'a> LimitReader<'a> {
        pub fn new(inner: Box<dyn Read + 'a>, remaining: u64) -> Self {
            Self { inner, remaining }
        }
    }

    impl Read for LimitReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                return Ok(0);
            }
            let want = buf.len().min(self.remaining as usize);
            let n = self.inner.read(&mut buf[..want])?;
            self.remaining -= n as u64;
            Ok(n)
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

#[cfg(all(test, feature = "amiga-lzx"))]
mod tests {
    use std::io::Read;
    use std::path::Path;

    use super::*;
    use crate::block::MemoryBackend;

    fn dev_from(bytes: &[u8]) -> MemoryBackend {
        let mut dev = MemoryBackend::new(bytes.len().max(1) as u64);
        dev.write_at(0, bytes).unwrap();
        dev
    }

    fn read_file(arc: &[u8], path: &str) -> Result<Vec<u8>> {
        let mut dev = dev_from(arc);
        let mut fs = LzxFs::open(&mut dev)?;
        let mut r = fs.read_file(&mut dev, Path::new(path))?;
        let mut out = Vec::new();
        r.read_to_end(&mut out).map_err(crate::Error::from)?;
        Ok(out)
    }

    /// A genuine Amiga `.lzx` archive (two merged store files) created and
    /// verified with the reference `unlzx` extractor. Confirms our container
    /// parsing — info header, 31-byte entry headers, merged-group slicing —
    /// matches the real format, and exercises skip-to-offset (b.txt > 0).
    #[test]
    fn genuine_unlzx_store_fixture() {
        let arc = include_bytes!("testdata/amiga_store.lzx");
        let a = b"first merged store file AAAAAAAA\n".repeat(4);
        let b = b"second merged store file BBBBBBBB\n".repeat(4);
        assert_eq!(read_file(arc, "/a.txt").unwrap(), a);
        assert_eq!(read_file(arc, "/b.txt").unwrap(), b);

        // Tree + sizes reported correctly.
        let mut dev = dev_from(arc);
        let mut fs = LzxFs::open(&mut dev).unwrap();
        let root = fs.list(&mut dev, Path::new("/")).unwrap();
        let names: Vec<_> = root.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"a.txt") && names.contains(&"b.txt"),
            "{names:?}"
        );
    }

    /// LZX (pack mode 2) decode path: synthesize an archive whose group is
    /// compressed with compcol's `amiga_lzx` encoder, then decode it back
    /// through the reader (compcol decoder). Self-consistent — the real
    /// `unlzx` can't read compcol's uncompressed-block streams.
    #[test]
    fn lzx_mode_round_trip() {
        let data = b"Amiga LZX pack-mode-2 payload, repeated to have some bulk. ".repeat(20);
        let framed = compcol::vec::compress_to_vec::<compcol::amiga_lzx::AmigaLzx>(&data).unwrap();
        let bitstream = &framed[4..]; // drop compcol's 4-byte length prefix
        let name = b"data.bin";

        let mut h = vec![0u8; 31];
        h[2..6].copy_from_slice(&(data.len() as u32).to_le_bytes());
        h[6..10].copy_from_slice(&(bitstream.len() as u32).to_le_bytes());
        h[11] = 2; // pack mode = LZX
        h[30] = name.len() as u8;
        // header CRC isn't validated by the reader, so leave it zero.

        let mut arc = Vec::new();
        arc.extend_from_slice(b"LZX");
        arc.extend_from_slice(&[0u8; 7]);
        arc.extend_from_slice(&h);
        arc.extend_from_slice(name);
        arc.extend_from_slice(bitstream);

        assert_eq!(read_file(&arc, "/data.bin").unwrap(), data);
    }
}
