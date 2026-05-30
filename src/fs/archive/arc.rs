//! SEA ARC (`.arc`) reader.
//!
//! Recognised by `detect_fs` via the leading `0x1A` marker followed by a
//! method byte.
//!
//! With the `arc` Cargo feature this is a real **read-only** reader: it walks
//! the flat per-file header chain (`0x1A`, method, 8.3 name, sizes, CRC) and
//! indexes every member. The two *stored* methods (1 = old, 2 = with an
//! original-size field) decode today; the compressed methods — 3 (packed /
//! RLE90), 4 (squeezed), 5–9 (crunched / squashed LZW) — list correctly but
//! reading their bodies returns a clean `Unsupported`, pending ARC crunch /
//! squeeze codecs in `compcol`.
//!
//! Without the `arc` feature this stays a detection-only scaffold.

use std::path::Path;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;
use crate::fs::{DirEntry, FileAttrs, FileReadHandle, Filesystem, MutationCapability};

/// SEA ARC filesystem handle.
pub struct ArcFs {
    fs: ArchiveFs,
    #[cfg(feature = "arc")]
    files: std::collections::HashMap<String, imp::Entry>,
}

impl ArcFs {
    #[cfg(feature = "arc")]
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let p = imp::scan(dev)?;
        Ok(Self {
            fs: ArchiveFs::from_index(p.index),
            files: p.files,
        })
    }

    #[cfg(not(feature = "arc"))]
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self {
            fs: ArchiveFs::scaffold("arc"),
        })
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "arc: creating archives is not supported".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for ArcFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl Filesystem for ArcFs {
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
        #[cfg(feature = "arc")]
        match self.lookup(path)? {
            imp::Lookup::Stored(f) => return Ok(imp::open_stored(dev, &f)),
            imp::Lookup::Unsupported(method) => {
                return Err(crate::Error::Unsupported(format!(
                    "arc: compression method {method} is not decodable yet (no compcol arc codec)"
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
        #[cfg(feature = "arc")]
        match self.lookup(path)? {
            imp::Lookup::Stored(f) => {
                use std::io::Read;
                let mut r = imp::open_stored(dev, &f);
                let mut bytes = Vec::new();
                r.read_to_end(&mut bytes).map_err(crate::Error::from)?;
                return Ok(Box::new(imp::mem_handle(bytes)));
            }
            imp::Lookup::Unsupported(method) => {
                return Err(crate::Error::Unsupported(format!(
                    "arc: compression method {method} is not decodable yet (no compcol arc codec)"
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

#[cfg(feature = "arc")]
impl ArcFs {
    fn lookup(&self, path: &Path) -> Result<imp::Lookup> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("arc: non-UTF-8 path".to_string()))?;
        let key = crate::fs::archive::tree::normalise_path(s);
        Ok(match self.files.get(&key) {
            Some(imp::Entry::Stored(f)) => imp::Lookup::Stored(*f),
            Some(imp::Entry::Unsupported(m)) => imp::Lookup::Unsupported(*m),
            None => imp::Lookup::NotRegular,
        })
    }
}

#[cfg(feature = "arc")]
mod imp {
    //! SEA ARC container parser. Each member is `0x1A`, a 1-byte method, the
    //! 8.3 name, sizes/CRC, then the (possibly compressed) body.

    use std::collections::HashMap;
    use std::io::{self, Read};

    use crate::Result;
    use crate::block::BlockDevice;
    use crate::fs::archive::reader::BoundedDevReader;
    use crate::fs::archive::{ArchiveEntry, ArchiveIndex, DataLocator, EntryKind, Method};

    /// ARC archive marker that prefixes every header.
    const MARK: u8 = 0x1A;

    /// A stored member's verbatim byte range.
    #[derive(Debug, Clone, Copy)]
    pub struct ArcFile {
        pub data_offset: u64,
        pub size: u64,
    }

    /// What a path resolves to.
    pub enum Entry {
        /// Method 1/2 (stored) — readable as a raw byte range.
        Stored(ArcFile),
        /// A compressed method we can name but not yet decode.
        Unsupported(u8),
    }

    pub enum Lookup {
        Stored(ArcFile),
        Unsupported(u8),
        NotRegular,
    }

    pub struct Parsed {
        pub index: ArchiveIndex,
        pub files: HashMap<String, Entry>,
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

    /// Parse the archive into the directory tree + per-path decode table.
    pub fn scan(dev: &mut dyn BlockDevice) -> Result<Parsed> {
        let dev_len = dev.total_size();
        let mut index = ArchiveIndex::new("arc");
        let mut files: HashMap<String, Entry> = HashMap::new();

        let mut pos: u64 = 0;
        let mut guard = 0u32;
        // The shortest header is method 1: marker(1)+method(1)+name(13)+
        // csize(4)+date(2)+time(2)+crc(2) = 25 bytes.
        while pos + 25 <= dev_len {
            guard += 1;
            if guard > 1 << 20 {
                break;
            }
            let h = read_at(dev, pos, 29.min((dev_len - pos) as usize))?;
            if h[0] != MARK {
                break; // not a valid header; stop cleanly
            }
            let method = h[1];
            if method == 0 {
                break; // end-of-archive marker (0x1A 0x00)
            }

            // 13-byte NUL-terminated name field at offset 2.
            let name_field = &h[2..15];
            let name_end = name_field
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_field.len());
            let name: String = name_field[..name_end]
                .iter()
                .map(|&b| if b == b'\\' { '/' } else { b as char })
                .collect();

            let comp_size = le32(&h, 15) as u64;
            // Method 1 (old) has no original-size field → header is 25 bytes
            // and the body is stored (size == comp_size). Methods >= 2 add a
            // 4-byte original size, making the header 29 bytes.
            let (header_len, unpack_size) = if method == 1 {
                (25u64, comp_size)
            } else {
                if h.len() < 29 {
                    break;
                }
                (29u64, le32(&h, 25) as u64)
            };

            let data_offset = pos + header_len;
            let next = data_offset + comp_size;
            if next <= pos || next > dev_len {
                break;
            }

            let path = crate::fs::archive::tree::normalise_path(&name);
            if path != "/" {
                let mut entry = ArchiveEntry::regular(
                    path.clone(),
                    DataLocator {
                        offset: data_offset,
                        compressed_len: comp_size,
                        uncompressed_len: unpack_size,
                        method: Method::Stored,
                    },
                );
                entry.kind = EntryKind::Regular;
                index.push(entry);

                // Methods 1 and 2 are stored verbatim. Everything else
                // (RLE90 / squeeze / crunch / squash) needs a codec.
                if method == 1 || method == 2 {
                    files.insert(
                        path,
                        Entry::Stored(ArcFile {
                            data_offset,
                            size: comp_size,
                        }),
                    );
                } else {
                    files.insert(path, Entry::Unsupported(method));
                }
            }

            pos = next;
        }

        Ok(Parsed { index, files })
    }

    /// Open a stored member's verbatim byte range as a bounded reader.
    pub fn open_stored<'a>(dev: &'a mut dyn BlockDevice, f: &ArcFile) -> Box<dyn Read + 'a> {
        Box::new(BoundedDevReader::new(dev, f.data_offset, f.size))
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

#[cfg(all(test, feature = "arc"))]
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
        let mut fs = ArcFs::open(&mut dev)?;
        let mut r = fs.read_file(&mut dev, Path::new(path))?;
        let mut out = Vec::new();
        r.read_to_end(&mut out).map_err(crate::Error::from)?;
        Ok(out)
    }

    fn names(arc: &[u8]) -> Vec<String> {
        let mut dev = dev_from(arc);
        let mut fs = ArcFs::open(&mut dev).unwrap();
        fs.list(&mut dev, Path::new("/"))
            .unwrap()
            .iter()
            .map(|e| e.name.clone())
            .collect()
    }

    /// Method-2 (stored) members decode; the method-8 (crunched) member lists
    /// but reads as `Unsupported`. The fixture is hand-built to the SEA ARC
    /// spec (no reference extractor is installed).
    #[test]
    fn store_round_trip_and_crunch_unsupported() {
        let arc = include_bytes!("testdata/arc_store.arc");
        assert_eq!(
            read_file(arc, "/hello.txt").unwrap(),
            b"Hello, ARC reader!\n".repeat(3)
        );
        assert_eq!(
            read_file(arc, "/lorem.txt").unwrap(),
            b"Lorem ipsum dolor sit amet. ".repeat(40)
        );
        let n = names(arc);
        assert!(
            ["hello.txt", "lorem.txt", "crunch.bin"]
                .iter()
                .all(|x| n.iter().any(|y| y == x)),
            "{n:?}"
        );
        let err = read_file(arc, "/crunch.bin").unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }

    /// Method 1 (old stored, 25-byte header without an original-size field).
    #[test]
    fn method1_old_store() {
        let arc = include_bytes!("testdata/arc_m1.arc");
        assert_eq!(
            read_file(arc, "/old.txt").unwrap(),
            b"old-style ARC stored entry\n"
        );
    }
}
