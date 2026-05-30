//! StuffIt (`.sit`, `SIT!` / `StuffIt`) reader.
//!
//! Recognised by `detect_fs` via the `SIT!` (classic) or `StuffIt` (SIT5)
//! signature.
//!
//! With the `sit` Cargo feature this is a real **read-only** reader for the
//! **classic** StuffIt container (`SIT!`): it walks the 22-byte archive header
//! and the 112-byte per-file entry headers (resource + data forks, big-endian)
//! and indexes every member by its **data fork**, honouring the folder
//! start/end markers for nested paths.
//!
//! Decoding is wired for method 0 (store). The compressed methods (1 RLE90,
//! 2 LZW, 3 Huffman, 5 LZAH, 13 LZ+Huffman, 15 Arsenic, …) and the entire
//! **StuffIt 5** format (`StuffIt` magic) parse/detect but read as a clean
//! `Unsupported`, pending StuffIt codecs in `compcol`. (StuffIt 5 falls back
//! to a detection-only scaffold.)
//!
//! Without the `sit` feature this stays a detection-only scaffold.

use std::path::Path;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;
use crate::fs::{DirEntry, FileAttrs, FileReadHandle, Filesystem, MutationCapability};

/// StuffIt filesystem handle.
pub struct SitFs {
    fs: ArchiveFs,
    #[cfg(feature = "sit")]
    files: std::collections::HashMap<String, imp::Entry>,
}

impl SitFs {
    #[cfg(feature = "sit")]
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        // Classic `SIT!` is parsed; StuffIt 5 (`StuffIt` magic) is left as a
        // detection-only scaffold (its format is unrelated and unsupported).
        let mut magic = [0u8; 4];
        dev.read_at(0, &mut magic)?;
        if &magic != b"SIT!" && &magic != b"SITD" {
            return Ok(Self {
                fs: ArchiveFs::scaffold("sit"),
                files: std::collections::HashMap::new(),
            });
        }
        let p = imp::scan(dev)?;
        Ok(Self {
            fs: ArchiveFs::from_index(p.index),
            files: p.files,
        })
    }

    #[cfg(not(feature = "sit"))]
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self {
            fs: ArchiveFs::scaffold("sit"),
        })
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "sit: creating archives is not supported".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for SitFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl Filesystem for SitFs {
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
        #[cfg(feature = "sit")]
        match self.lookup(path)? {
            imp::Lookup::Stored(f) => return Ok(imp::open_stored(dev, &f)),
            imp::Lookup::Unsupported(method) => {
                return Err(crate::Error::Unsupported(format!(
                    "sit: data-fork method {method} is not decodable yet (no compcol sit codec)"
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
        #[cfg(feature = "sit")]
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
                    "sit: data-fork method {method} is not decodable yet (no compcol sit codec)"
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

#[cfg(feature = "sit")]
impl SitFs {
    fn lookup(&self, path: &Path) -> Result<imp::Lookup> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("sit: non-UTF-8 path".to_string()))?;
        let key = crate::fs::archive::tree::normalise_path(s);
        Ok(match self.files.get(&key) {
            Some(imp::Entry::Stored(f)) => imp::Lookup::Stored(*f),
            Some(imp::Entry::Unsupported(m)) => imp::Lookup::Unsupported(*m),
            None => imp::Lookup::NotRegular,
        })
    }
}

#[cfg(feature = "sit")]
mod imp {
    //! Classic StuffIt (`SIT!`) container parser. Each entry carries a resource
    //! fork then a data fork; we index the **data fork** as the file body.

    use std::collections::HashMap;
    use std::io::{self, Read};

    use crate::Result;
    use crate::block::BlockDevice;
    use crate::fs::archive::reader::BoundedDevReader;
    use crate::fs::archive::{ArchiveEntry, ArchiveIndex, DataLocator, EntryKind, Method};

    /// Per-entry header length (bytes).
    const ENTRY_HEADER: u64 = 112;
    /// `rsrc_method` sentinel: start of a folder.
    const FOLDER_START: u8 = 32;
    /// `rsrc_method` sentinel: end of a folder.
    const FOLDER_END: u8 = 33;

    /// A stored (method 0) data fork's verbatim byte range.
    #[derive(Debug, Clone, Copy)]
    pub struct SitFile {
        pub data_offset: u64,
        pub size: u64,
    }

    pub enum Entry {
        /// Data-fork method 0 (store) — readable as a raw byte range.
        Stored(SitFile),
        /// A compressed data-fork method we can name but not yet decode.
        Unsupported(u8),
    }

    pub enum Lookup {
        Stored(SitFile),
        Unsupported(u8),
        NotRegular,
    }

    pub struct Parsed {
        pub index: ArchiveIndex,
        pub files: HashMap<String, Entry>,
    }

    #[inline]
    fn be32(b: &[u8], o: usize) -> u32 {
        u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }

    fn read_at(dev: &mut dyn BlockDevice, off: u64, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        dev.read_at(off, &mut buf)?;
        Ok(buf)
    }

    /// Parse the classic `SIT!` archive into the directory tree + data-fork
    /// decode table.
    pub fn scan(dev: &mut dyn BlockDevice) -> Result<Parsed> {
        let dev_len = dev.total_size();
        let mut index = ArchiveIndex::new("sit");
        let mut files: HashMap<String, Entry> = HashMap::new();

        // Directory stack from FOLDER_START / FOLDER_END markers.
        let mut dir_stack: Vec<String> = Vec::new();

        let mut pos: u64 = 22; // past the 22-byte archive header
        let mut guard = 0u32;
        while pos + ENTRY_HEADER <= dev_len {
            guard += 1;
            if guard > 1 << 20 {
                break;
            }
            let h = read_at(dev, pos, ENTRY_HEADER as usize)?;
            let rmethod = h[0];
            let dmethod = h[1];
            let name_len = (h[2] as usize).min(63);
            let name: String = h[3..3 + name_len]
                .iter()
                .map(|&b| {
                    if b == b':' || b == b'/' {
                        '_'
                    } else {
                        b as char
                    }
                })
                .collect();

            if rmethod == FOLDER_END {
                dir_stack.pop();
                pos += ENTRY_HEADER;
                continue;
            }
            if rmethod == FOLDER_START {
                if !name.is_empty() {
                    dir_stack.push(name);
                    let path = current_path(&dir_stack, "");
                    if path != "/" {
                        index.push(ArchiveEntry::dir(path));
                    }
                }
                pos += ENTRY_HEADER;
                continue;
            }

            let rfork_clen = be32(&h, 92) as u64;
            let dfork_ulen = be32(&h, 88) as u64;
            let dfork_clen = be32(&h, 96) as u64;

            // Data fork follows the resource fork, which follows the header.
            let data_offset = pos + ENTRY_HEADER + rfork_clen;
            let next = pos + ENTRY_HEADER + rfork_clen + dfork_clen;
            if next <= pos || next > dev_len {
                break;
            }

            let path = current_path(&dir_stack, &name);
            if path != "/" {
                let mut entry = ArchiveEntry::regular(
                    path.clone(),
                    DataLocator {
                        offset: data_offset,
                        compressed_len: dfork_clen,
                        uncompressed_len: dfork_ulen,
                        method: Method::Stored,
                    },
                );
                entry.kind = EntryKind::Regular;
                index.push(entry);

                if dmethod == 0 {
                    files.insert(
                        path,
                        Entry::Stored(SitFile {
                            data_offset,
                            size: dfork_clen,
                        }),
                    );
                } else {
                    files.insert(path, Entry::Unsupported(dmethod));
                }
            }

            pos = next;
        }

        Ok(Parsed { index, files })
    }

    /// Join the folder stack and a (possibly empty) leaf into a normalised
    /// absolute path.
    fn current_path(dir_stack: &[String], leaf: &str) -> String {
        let mut raw = String::new();
        for d in dir_stack {
            raw.push('/');
            raw.push_str(d);
        }
        if !leaf.is_empty() {
            raw.push('/');
            raw.push_str(leaf);
        }
        crate::fs::archive::tree::normalise_path(&raw)
    }

    /// Open a stored data fork's verbatim byte range as a bounded reader.
    pub fn open_stored<'a>(dev: &'a mut dyn BlockDevice, f: &SitFile) -> Box<dyn Read + 'a> {
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

#[cfg(all(test, feature = "sit"))]
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
        let mut fs = SitFs::open(&mut dev)?;
        let mut r = fs.read_file(&mut dev, Path::new(path))?;
        let mut out = Vec::new();
        r.read_to_end(&mut out).map_err(crate::Error::from)?;
        Ok(out)
    }

    fn names(arc: &[u8]) -> Vec<String> {
        let mut dev = dev_from(arc);
        let mut fs = SitFs::open(&mut dev).unwrap();
        fs.list(&mut dev, Path::new("/"))
            .unwrap()
            .iter()
            .map(|e| e.name.clone())
            .collect()
    }

    /// Classic `SIT!` archive: the two stored (method 0) members decode from
    /// their data fork; the method-13 member lists but reads `Unsupported`.
    /// The fixture is hand-built to the classic StuffIt spec (no reference
    /// extractor is installed).
    #[test]
    fn classic_store_round_trip_and_method13_unsupported() {
        let arc = include_bytes!("testdata/test.sit");
        assert_eq!(
            read_file(arc, "/hello.txt").unwrap(),
            b"Hello, StuffIt reader!\n".repeat(3)
        );
        assert_eq!(
            read_file(arc, "/lorem.txt").unwrap(),
            b"Lorem ipsum dolor sit amet. ".repeat(40)
        );
        let n = names(arc);
        assert!(
            ["hello.txt", "lorem.txt", "deluxe.bin"]
                .iter()
                .all(|x| n.iter().any(|y| y == x)),
            "{n:?}"
        );
        let err = read_file(arc, "/deluxe.bin").unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }
}
