//! LHA / LZH (`.lzh`, `.lha`) reader.
//!
//! Recognised by `detect_fs` via the `-lh?-` / `-lz?-` method tag at offset 2.
//!
//! With the `lha` Cargo feature this is a real **read-only** reader: it walks
//! the header chain (levels 0, 1 and 2, including the level-1/2 extended-header
//! list for long names and directory components) and indexes every member.
//!
//! Decoding is wired for **`-lh0-`** (store) and **`-lhd-`** (directory) today;
//! the LZSS+Huffman methods (`-lh1-/-lh4-/-lh5-/-lh6-/-lh7-`, `-lzs-`, …) parse
//! and list correctly but reading their bodies returns a clean `Unsupported`
//! naming the method, pending an `lha` codec in `compcol`.
//!
//! Without the `lha` feature this stays a detection-only scaffold.

use std::path::Path;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;
use crate::fs::{DirEntry, FileAttrs, FileReadHandle, Filesystem, MutationCapability};

/// LHA filesystem handle.
pub struct LhaFs {
    fs: ArchiveFs,
    #[cfg(feature = "lha")]
    files: std::collections::HashMap<String, imp::Entry>,
}

impl LhaFs {
    #[cfg(feature = "lha")]
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let p = imp::scan(dev)?;
        Ok(Self {
            fs: ArchiveFs::from_index(p.index),
            files: p.files,
        })
    }

    #[cfg(not(feature = "lha"))]
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self {
            fs: ArchiveFs::scaffold("lha"),
        })
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "lha: creating archives is not supported".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for LhaFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl Filesystem for LhaFs {
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
        #[cfg(feature = "lha")]
        match self.lookup(path)? {
            imp::Lookup::Stored(f) => return Ok(imp::open_stored(dev, &f)),
            imp::Lookup::Unsupported(method) => {
                return Err(crate::Error::Unsupported(format!(
                    "lha: method -{method}- is not decodable yet (no compcol lha codec)"
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
        #[cfg(feature = "lha")]
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
                    "lha: method -{method}- is not decodable yet (no compcol lha codec)"
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

#[cfg(feature = "lha")]
impl LhaFs {
    fn lookup(&self, path: &Path) -> Result<imp::Lookup> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("lha: non-UTF-8 path".to_string()))?;
        let key = crate::fs::archive::tree::normalise_path(s);
        Ok(match self.files.get(&key) {
            Some(imp::Entry::Stored(f)) => imp::Lookup::Stored(*f),
            Some(imp::Entry::Unsupported(m)) => imp::Lookup::Unsupported(m.clone()),
            None => imp::Lookup::NotRegular,
        })
    }
}

#[cfg(feature = "lha")]
mod imp {
    //! LHA container parser. Each member is one contiguous packed run; for
    //! `-lh0-` (store) that run *is* the file content.

    use std::collections::HashMap;
    use std::io::{self, Read};

    use crate::block::BlockDevice;
    use crate::fs::archive::reader::BoundedDevReader;
    use crate::fs::archive::{ArchiveEntry, ArchiveIndex, DataLocator, EntryKind, Method};
    use crate::{Error, Result};

    /// A stored (`-lh0-`) member's verbatim byte range.
    #[derive(Debug, Clone, Copy)]
    pub struct LhaFile {
        pub data_offset: u64,
        pub size: u64,
    }

    /// What a path resolves to.
    pub enum Entry {
        /// `-lh0-` store — readable as a raw byte range.
        Stored(LhaFile),
        /// A compressed method we can name but not yet decode (the `lhX`/`lzX`
        /// tag, without the dashes).
        Unsupported(String),
    }

    pub enum Lookup {
        Stored(LhaFile),
        Unsupported(String),
        NotRegular,
    }

    pub struct Parsed {
        pub index: ArchiveIndex,
        pub files: HashMap<String, Entry>,
    }

    #[inline]
    fn le16(b: &[u8], o: usize) -> u16 {
        u16::from_le_bytes([b[o], b[o + 1]])
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

    /// Result of walking an extended-header chain.
    #[derive(Default)]
    struct ExtInfo {
        /// Total bytes the chain occupies (the level-1 skip-size adjustment).
        sum: u64,
        /// Filename override (ext type 0x01).
        name: Option<Vec<u8>>,
        /// Directory path, `0xFF`-separated (ext type 0x02).
        dir: Option<Vec<u8>>,
    }

    /// Walk a level-1/2 extended-header chain starting at `cur` whose first
    /// header is `size` bytes. Each header is `[type:1][data:size-3][next:2]`.
    fn walk_ext(
        dev: &mut dyn BlockDevice,
        mut cur: u64,
        mut size: u64,
        dev_len: u64,
    ) -> Result<ExtInfo> {
        let mut info = ExtInfo::default();
        let mut guard = 0u32;
        while size != 0 {
            guard += 1;
            if guard > 4096 {
                return Err(Error::InvalidImage(
                    "lha: runaway extended-header chain".into(),
                ));
            }
            if size < 3 || cur + size > dev_len {
                return Err(Error::InvalidImage("lha: truncated extended header".into()));
            }
            let h = read_at(dev, cur, size as usize)?;
            let htype = h[0];
            let data = &h[1..size as usize - 2];
            match htype {
                0x01 => info.name = Some(data.to_vec()), // filename
                0x02 => info.dir = Some(data.to_vec()),  // directory (0xFF-separated)
                _ => {}
            }
            let next = le16(&h, size as usize - 2) as u64;
            info.sum += size;
            cur += size;
            size = next;
        }
        Ok(info)
    }

    /// Parse the archive into the directory tree + per-path decode table.
    pub fn scan(dev: &mut dyn BlockDevice) -> Result<Parsed> {
        let dev_len = dev.total_size();
        let mut index = ArchiveIndex::new("lha");
        let mut files: HashMap<String, Entry> = HashMap::new();

        let mut pos: u64 = 0;
        let mut guard = 0u32;
        while pos + 21 <= dev_len {
            guard += 1;
            if guard > 1 << 20 {
                break;
            }
            // A single 0 byte (level 0/1) or a zero size word (level 2)
            // terminates the archive.
            let head = read_at(dev, pos, 26.min((dev_len - pos) as usize))?;
            if head[0] == 0 {
                break;
            }
            let level = head[20];

            // Method tag "-lhX-" / "-lzX-" at offset 2; inner 3 chars are the
            // method id (e.g. "lh0", "lh5", "lhd").
            let method_tag: String = head[3..6].iter().map(|&b| b as char).collect();

            let data_offset;
            let comp_size;
            let unpack_size;
            let next;
            let mut name_bytes: Vec<u8>;
            let mut dir_bytes: Option<Vec<u8>> = None;

            match level {
                0 | 1 => {
                    let header_size = head[0] as u64;
                    let base_end = pos + 2 + header_size;
                    if base_end > dev_len {
                        break;
                    }
                    unpack_size = le32(&head, 11) as u64;
                    let size_field = le32(&head, 7) as u64;
                    let name_len = head[21] as usize;
                    name_bytes = read_at(dev, pos + 22, name_len)?;
                    if level == 0 {
                        data_offset = base_end;
                        comp_size = size_field;
                        next = data_offset + comp_size;
                    } else {
                        // Level 1: the 2 bytes before `base_end` are the first
                        // extended-header size; `size_field` is the *skip size*
                        // (packed data + all extended headers).
                        let two = read_at(dev, base_end - 2, 2)?;
                        let first_ext = le16(&two, 0) as u64;
                        let ext = walk_ext(dev, base_end, first_ext, dev_len)?;
                        if let Some(n) = ext.name {
                            name_bytes = n;
                        }
                        dir_bytes = ext.dir;
                        data_offset = base_end + ext.sum;
                        comp_size = size_field.saturating_sub(ext.sum);
                        next = data_offset + comp_size;
                    }
                }
                2 => {
                    let total_header_size = le16(&head, 0) as u64;
                    if total_header_size < 26 || pos + total_header_size > dev_len {
                        break;
                    }
                    comp_size = le32(&head, 7) as u64;
                    unpack_size = le32(&head, 11) as u64;
                    let first_ext = le16(&head, 24) as u64;
                    let ext = walk_ext(dev, pos + 26, first_ext, dev_len)?;
                    name_bytes = ext.name.unwrap_or_default();
                    dir_bytes = ext.dir;
                    data_offset = pos + total_header_size;
                    next = data_offset + comp_size;
                }
                _ => break, // level 3+ not supported
            }

            if next <= pos || next > dev_len {
                break;
            }

            let path =
                crate::fs::archive::tree::normalise_path(&assemble_path(&dir_bytes, &name_bytes));

            if method_tag == "lhd" {
                // Directory entry.
                if path != "/" {
                    index.push(ArchiveEntry::dir(path));
                }
            } else if path != "/" {
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

                if method_tag == "lh0" {
                    files.insert(
                        path,
                        Entry::Stored(LhaFile {
                            data_offset,
                            size: comp_size,
                        }),
                    );
                } else {
                    files.insert(path, Entry::Unsupported(method_tag));
                }
            }

            pos = next;
        }

        Ok(Parsed { index, files })
    }

    /// Build the absolute slash path from an optional directory header
    /// (components separated by `0xFF` or `\`) and the filename.
    fn assemble_path(dir: &Option<Vec<u8>>, name: &[u8]) -> String {
        let mut raw = String::new();
        if let Some(d) = dir {
            for b in d {
                raw.push(if *b == 0xFF || *b == b'\\' {
                    '/'
                } else {
                    *b as char
                });
            }
            if !raw.ends_with('/') {
                raw.push('/');
            }
        }
        for &b in name {
            raw.push(if b == b'\\' { '/' } else { b as char });
        }
        // Drop a leading "drive:" if present, normalise components.
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

    /// Open a stored member's verbatim byte range as a bounded reader.
    pub fn open_stored<'a>(dev: &'a mut dyn BlockDevice, f: &LhaFile) -> Box<dyn Read + 'a> {
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

#[cfg(all(test, feature = "lha"))]
mod tests {
    use std::io::Read;
    use std::path::Path;

    use super::*;
    use crate::block::MemoryBackend;

    fn hello() -> Vec<u8> {
        b"Hello, LHA reader!\n".repeat(3)
    }
    fn lorem() -> Vec<u8> {
        b"Lorem ipsum dolor sit amet. ".repeat(40)
    }

    fn dev_from(bytes: &[u8]) -> MemoryBackend {
        let mut dev = MemoryBackend::new(bytes.len().max(1) as u64);
        dev.write_at(0, bytes).unwrap();
        dev
    }

    fn read_file(arc: &[u8], path: &str) -> Result<Vec<u8>> {
        let mut dev = dev_from(arc);
        let mut fs = LhaFs::open(&mut dev)?;
        let mut r = fs.read_file(&mut dev, Path::new(path))?;
        let mut out = Vec::new();
        r.read_to_end(&mut out).map_err(crate::Error::from)?;
        Ok(out)
    }

    fn names(arc: &[u8]) -> Vec<String> {
        let mut dev = dev_from(arc);
        let mut fs = LhaFs::open(&mut dev).unwrap();
        fs.list(&mut dev, Path::new("/"))
            .unwrap()
            .iter()
            .map(|e| e.name.clone())
            .collect()
    }

    /// Genuine `lha -z` (store) archives at all three header levels (0/1/2),
    /// each over the same two files. Confirms level-0/1/2 header parsing
    /// (incl. the level-1 skip-size/ext-header math and level-2 ext filename)
    /// and the `-lh0-` store decode path.
    #[test]
    fn store_levels_0_1_2_round_trip() {
        for arc in [
            &include_bytes!("testdata/store0.lzh")[..],
            &include_bytes!("testdata/store1.lzh")[..],
            &include_bytes!("testdata/store2.lzh")[..],
        ] {
            assert_eq!(read_file(arc, "/hello.txt").unwrap(), hello());
            assert_eq!(read_file(arc, "/lorem.txt").unwrap(), lorem());
            let n = names(arc);
            assert!(
                n.iter().any(|x| x == "hello.txt") && n.iter().any(|x| x == "lorem.txt"),
                "{n:?}"
            );
        }
    }

    /// Cross-check our store extraction against the reference `lha` tool when
    /// it's installed.
    #[test]
    fn matches_lha_reference() {
        use std::process::Command;
        if Command::new("lha").arg("-?").output().is_err() {
            eprintln!("skipping: lha not installed");
            return;
        }
        let arc = include_bytes!("testdata/store2.lzh");
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(tmp.as_file_mut(), arc).unwrap();
        for name in ["hello.txt", "lorem.txt"] {
            // `lha p` prints the member to stdout; `-q` quiets the banner.
            let out = Command::new("lha")
                .args(["pq", tmp.path().to_str().unwrap(), name])
                .output()
                .unwrap();
            // Only cross-check when the reference tool actually extracted.
            if !out.status.success() {
                eprintln!("skipping: `lha p` unavailable here");
                return;
            }
            assert_eq!(
                read_file(arc, &format!("/{name}")).unwrap(),
                out.stdout,
                "reader vs lha mismatch for {name}"
            );
        }
    }

    /// Genuine `-lh5-` archive: the container parses and lists correctly (sizes
    /// from the headers), but reading a member returns `Unsupported` until the
    /// compcol `lha` codec lands.
    #[test]
    fn lh5_parses_but_read_is_unsupported() {
        let arc = include_bytes!("testdata/comp5.lzh");
        let n = names(arc);
        assert!(
            n.iter().any(|x| x == "hello.txt") && n.iter().any(|x| x == "lorem.txt"),
            "{n:?}"
        );
        let err = read_file(arc, "/hello.txt").unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
    }
}
