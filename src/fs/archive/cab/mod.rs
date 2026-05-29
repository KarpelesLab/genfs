//! Microsoft Cabinet (`.cab`, `MSCF`) reader.
//!
//! Recognised by `detect_fs` via the `MSCF` signature. With the `cab` Cargo
//! feature this is a real **read-only** reader: it parses the cabinet (see
//! [`scan`]) and extracts a file by *streaming* its owning CFFOLDER through
//! `compcol`, skipping to the file's offset and capping at its length (see
//! [`folder`]) — so memory stays bounded even for folders that decompress to
//! many gigabytes. Supported folder methods: None (Store), MSZIP, LZX, and
//! Quantum. Spanned/multi-cabinet sets and archive creation are not
//! supported.
//!
//! Without the `cab` feature this stays a detection-only scaffold (reads
//! return a clean `Unsupported`).

#[cfg(feature = "cab")]
mod folder;
#[cfg(feature = "cab")]
mod scan;

use std::path::Path;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;
use crate::fs::{DirEntry, FileAttrs, FileReadHandle, Filesystem, MutationCapability};

/// CAB filesystem handle. `fs` carries the directory tree + per-file sizes
/// (and, without the `cab` feature, the scaffold); the side tables drive the
/// custom folder-decode read path.
pub struct CabFs {
    fs: ArchiveFs,
    #[cfg(feature = "cab")]
    folders: Vec<scan::Folder>,
    #[cfg(feature = "cab")]
    files: std::collections::HashMap<String, Option<scan::FileSlice>>,
}

impl CabFs {
    #[cfg(feature = "cab")]
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let p = scan::scan(dev)?;
        Ok(Self {
            fs: ArchiveFs::from_index(p.index),
            folders: p.folders,
            files: p.files,
        })
    }

    #[cfg(not(feature = "cab"))]
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self {
            fs: ArchiveFs::scaffold("cab"),
        })
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "cab: creating archives is not supported".into(),
        ))
    }

    /// Decode and slice out a file's bytes. `None` from the slice table
    /// means a known regular file we can't extract (spanned / missing
    /// folder); a missing key means it isn't a regular file (let the inner
    /// `ArchiveFs` produce the proper error). The slice carries an owned
    /// `Folder` clone so the caller can build a reader borrowing only `dev`.
    #[cfg(feature = "cab")]
    fn lookup(&self, path: &Path) -> Result<CabLookup> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("cab: non-UTF-8 path".to_string()))?;
        let key = crate::fs::archive::tree::normalise_path(s);
        Ok(match self.files.get(&key) {
            Some(Some(slice)) => CabLookup::Slice(*slice, self.folders[slice.folder].clone()),
            Some(None) => CabLookup::Unextractable(key),
            None => CabLookup::NotRegular,
        })
    }
}

/// Outcome of resolving a path to its folder slice (see [`CabFs::lookup`]).
#[cfg(feature = "cab")]
enum CabLookup {
    /// Extractable regular file: its slice + an owned copy of the folder.
    Slice(scan::FileSlice, scan::Folder),
    /// A regular file we can't extract (spans cabinets / unsupported method).
    Unextractable(String),
    /// Not a regular file — defer to the inner `ArchiveFs`.
    NotRegular,
}

impl crate::fs::FilesystemFactory for CabFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

/// Caps an owned boxed reader to `remaining` bytes (the `Read::take`
/// equivalent for a `Box<dyn Read>`, which isn't itself `Sized`-callable).
#[cfg(feature = "cab")]
struct LimitReader<'a> {
    inner: Box<dyn std::io::Read + 'a>,
    remaining: u64,
}

#[cfg(feature = "cab")]
impl std::io::Read for LimitReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let want = buf.len().min(self.remaining as usize);
        let n = self.inner.read(&mut buf[..want])?;
        self.remaining -= n as u64;
        Ok(n)
    }
}

/// Seekable in-memory handle over a decoded file body (for `open_file_ro`).
#[cfg(feature = "cab")]
struct MemHandle {
    cur: std::io::Cursor<Vec<u8>>,
    len: u64,
}

#[cfg(feature = "cab")]
impl std::io::Read for MemHandle {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.cur.read(buf)
    }
}
#[cfg(feature = "cab")]
impl std::io::Seek for MemHandle {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.cur.seek(pos)
    }
}
#[cfg(feature = "cab")]
impl FileReadHandle for MemHandle {
    fn len(&self) -> u64 {
        self.len
    }
}

impl Filesystem for CabFs {
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
        #[cfg(feature = "cab")]
        match self.lookup(path)? {
            CabLookup::Slice(slice, folder) => {
                // Stream the folder, skip to the file's offset, cap at its
                // length — bounded memory regardless of folder/file size.
                let mut r = folder::decode_folder_reader(dev, &folder)?;
                folder::skip_exact(&mut *r, slice.uncomp_offset)?;
                return Ok(Box::new(LimitReader {
                    inner: r,
                    remaining: slice.len,
                }));
            }
            CabLookup::Unextractable(key) => {
                return Err(crate::Error::Unsupported(format!(
                    "cab: {key:?} cannot be extracted (spans cabinets or uses an \
                     unsupported method)"
                )));
            }
            CabLookup::NotRegular => {}
        }
        self.fs.read_file(dev, path)
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &Path,
    ) -> Result<Box<dyn FileReadHandle + 'a>> {
        // Random access needs the bytes in memory; buffer the *file* (its
        // own length), not the whole folder. Huge files should use the
        // streaming `read_file` instead.
        #[cfg(feature = "cab")]
        match self.lookup(path)? {
            CabLookup::Slice(slice, folder) => {
                use std::io::Read;
                let mut r = folder::decode_folder_reader(dev, &folder)?;
                folder::skip_exact(&mut *r, slice.uncomp_offset)?;
                let mut bytes = Vec::new();
                (&mut *r)
                    .take(slice.len)
                    .read_to_end(&mut bytes)
                    .map_err(crate::Error::from)?;
                let len = bytes.len() as u64;
                return Ok(Box::new(MemHandle {
                    cur: std::io::Cursor::new(bytes),
                    len,
                }));
            }
            CabLookup::Unextractable(key) => {
                return Err(crate::Error::Unsupported(format!(
                    "cab: {key:?} cannot be extracted (spans cabinets or uses an \
                     unsupported method)"
                )));
            }
            CabLookup::NotRegular => {}
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

#[cfg(all(test, feature = "cab"))]
pub(crate) mod test_support {
    //! Minimal single-folder/single-file cabinet builder, shared by the
    //! in-crate unit tests and the `cabextract` cross-check integration test.

    /// Build a single-folder cabinet with one file `name` of logical size
    /// `cb_file` (offset 0), whose folder data is the given CFDATA `blocks`.
    pub fn build_cab(
        type_compress: u16,
        blocks: &[(Vec<u8>, u16)],
        cb_file: u32,
        name: &str,
    ) -> Vec<u8> {
        build_cab_files(type_compress, blocks, &[(name, cb_file, 0)])
    }

    /// Build a single-folder cabinet holding several files (each
    /// `(name, cb_file, uoff_folder_start)`) over the given CFDATA `blocks`.
    pub fn build_cab_files(
        type_compress: u16,
        blocks: &[(Vec<u8>, u16)],
        files: &[(&str, u32, u32)],
    ) -> Vec<u8> {
        let coff_files: u32 = 36 + 8;
        let cffile_size: usize = files.iter().map(|(n, _, _)| 16 + n.len() + 1).sum();
        let coff_cab_start = coff_files as usize + cffile_size;

        let mut out = Vec::new();
        // CFHEADER (36 bytes).
        out.extend_from_slice(b"MSCF");
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved1
        out.extend_from_slice(&0u32.to_le_bytes()); // cbCabinet — patched
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved2
        out.extend_from_slice(&coff_files.to_le_bytes()); // coffFiles
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved3
        out.push(3); // versionMinor
        out.push(1); // versionMajor
        out.extend_from_slice(&1u16.to_le_bytes()); // cFolders
        out.extend_from_slice(&(files.len() as u16).to_le_bytes()); // cFiles
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // setID
        out.extend_from_slice(&0u16.to_le_bytes()); // iCabinet
        // CFFOLDER (8 bytes).
        out.extend_from_slice(&(coff_cab_start as u32).to_le_bytes());
        out.extend_from_slice(&(blocks.len() as u16).to_le_bytes());
        out.extend_from_slice(&type_compress.to_le_bytes());
        // CFFILE[].
        for (name, cb_file, uoff) in files {
            out.extend_from_slice(&cb_file.to_le_bytes());
            out.extend_from_slice(&uoff.to_le_bytes()); // uoffFolderStart
            out.extend_from_slice(&0u16.to_le_bytes()); // iFolder
            out.extend_from_slice(&0u16.to_le_bytes()); // date
            out.extend_from_slice(&0u16.to_le_bytes()); // time
            out.extend_from_slice(&0u16.to_le_bytes()); // attribs
            out.extend_from_slice(name.as_bytes());
            out.push(0);
        }
        // CFDATA blocks (csum=0 → unchecked).
        for (payload, cb_uncomp) in blocks {
            out.extend_from_slice(&0u32.to_le_bytes()); // csum
            out.extend_from_slice(&(payload.len() as u16).to_le_bytes()); // cbData
            out.extend_from_slice(&cb_uncomp.to_le_bytes()); // cbUncomp
            out.extend_from_slice(payload);
        }
        let total = out.len() as u32;
        out[8..12].copy_from_slice(&total.to_le_bytes());
        out
    }

    /// MSZIP single block: `CK` + raw deflate of `content`.
    pub fn mszip_block(content: &[u8]) -> Vec<u8> {
        let mut p = b"CK".to_vec();
        p.extend_from_slice(
            &compcol::vec::compress_to_vec::<compcol::deflate::Deflate>(content).unwrap(),
        );
        p
    }

    /// LZX: compcol's stream minus its 5-byte standalone header; returns the
    /// CFDATA payload plus the `typeCompress` word (method 3 + window bits).
    pub fn lzx_block(content: &[u8]) -> (Vec<u8>, u16) {
        let framed = compcol::vec::compress_to_vec::<compcol::lzx::Lzx>(content).unwrap();
        let window_bits = framed[0] as u16;
        let payload = framed[5..].to_vec();
        (payload, (window_bits << 8) | 3)
    }
}

#[cfg(all(test, feature = "cab"))]
mod tests {
    use std::io::Read;
    use std::path::Path;

    use super::test_support::*;
    use super::*;
    use crate::block::MemoryBackend;

    fn dev_from(cab: &[u8]) -> MemoryBackend {
        let mut dev = MemoryBackend::new(cab.len() as u64);
        dev.write_at(0, cab).unwrap();
        dev
    }

    fn read_file(cab: &[u8], path: &str) -> Result<Vec<u8>> {
        let mut dev = dev_from(cab);
        let mut fs = CabFs::open(&mut dev)?;
        let mut r = fs.read_file(&mut dev, Path::new(path))?;
        let mut out = Vec::new();
        r.read_to_end(&mut out).map_err(crate::Error::from)?;
        Ok(out)
    }

    /// Extract a single-file cabinet's body with the reference `cabextract`
    /// tool (`-p` pipes file content to stdout). Returns `None` when the
    /// tool isn't installed, so the assertion is skipped on bare machines.
    fn cabextract(cab: &[u8]) -> Option<Vec<u8>> {
        use std::process::Command;
        let probe = Command::new("cabextract").arg("--version").output().ok()?;
        if !probe.status.success() {
            return None;
        }
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(tmp.as_file_mut(), cab).unwrap();
        let out = Command::new("cabextract")
            .arg("-q")
            .arg("-p")
            .arg(tmp.path())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "cabextract rejected the cabinet: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        Some(out.stdout)
    }

    /// Assert our reader and (when present) `cabextract` both recover
    /// `content` from a single-file cabinet `cab`.
    fn assert_extracts(cab: &[u8], path: &str, content: &[u8]) {
        assert_eq!(
            read_file(cab, path).unwrap(),
            content,
            "fstool reader mismatch"
        );
        if let Some(out) = cabextract(cab) {
            assert_eq!(
                out, content,
                "cabextract mismatch (container/codec interop)"
            );
        }
    }

    #[test]
    fn store_round_trip() {
        let content = b"hello cab store method, verbatim bytes!".repeat(4);
        let cab = build_cab(
            0,
            &[(content.clone(), content.len() as u16)],
            content.len() as u32,
            "a.txt",
        );
        assert_extracts(&cab, "/a.txt", &content);
    }

    #[test]
    fn mszip_single_block_round_trip() {
        let content = b"mszip single block payload; ".repeat(50);
        let block = mszip_block(&content);
        let cab = build_cab(
            1,
            &[(block, content.len() as u16)],
            content.len() as u32,
            "m.txt",
        );
        assert_extracts(&cab, "/m.txt", &content);
    }

    #[test]
    fn mszip_multi_block_round_trip() {
        // Two CFDATA blocks exercise the per-block decode + 32 KiB preset-
        // dictionary seeding path (compcol 0.4.3 / #22). cabextract
        // cross-checks the multi-block container framing.
        let chunk1 = b"first MSZIP block contents, repeated. ".repeat(30);
        let chunk2 = b"second MSZIP block contents, also repeated. ".repeat(30);
        let mut content = chunk1.clone();
        content.extend_from_slice(&chunk2);
        let blocks = vec![
            (mszip_block(&chunk1), chunk1.len() as u16),
            (mszip_block(&chunk2), chunk2.len() as u16),
        ];
        let cab = build_cab(1, &blocks, content.len() as u32, "m2.txt");
        assert_extracts(&cab, "/m2.txt", &content);
    }

    #[test]
    fn lzx_round_trip() {
        let content = b"LZX compressed cab folder data, repeated to compress well. ".repeat(40);
        let (payload, type_compress) = lzx_block(&content);
        let cab = build_cab(
            type_compress,
            &[(payload, content.len() as u16)],
            content.len() as u32,
            "l.txt",
        );
        assert_extracts(&cab, "/l.txt", &content);
    }

    #[test]
    fn second_file_in_folder_skips_to_offset() {
        // Two files share one Store folder; reading the second exercises
        // skip_exact (offset > 0) + the length cap.
        let a = b"first file contents AAAA".repeat(3);
        let b = b"second file contents BBBB".repeat(3);
        let mut folder = a.clone();
        folder.extend_from_slice(&b);
        let cab = build_cab_files(
            0,
            &[(folder.clone(), folder.len() as u16)],
            &[
                ("a.txt", a.len() as u32, 0),
                ("b.txt", b.len() as u32, a.len() as u32),
            ],
        );
        assert_eq!(read_file(&cab, "/a.txt").unwrap(), a);
        assert_eq!(read_file(&cab, "/b.txt").unwrap(), b);
    }

    #[test]
    fn lists_the_file() {
        let content = b"x".repeat(10);
        let cab = build_cab(0, &[(content.clone(), 10)], 10, "dir\\nested.txt");
        let mut dev = dev_from(&cab);
        let mut fs = CabFs::open(&mut dev).unwrap();
        let root = fs.list(&mut dev, Path::new("/")).unwrap();
        assert!(root.iter().any(|e| e.name == "dir"));
        let sub = fs.list(&mut dev, Path::new("/dir")).unwrap();
        assert_eq!(sub.len(), 1);
        assert_eq!(sub[0].name, "nested.txt");
        assert_eq!(sub[0].size, 10);
    }
}
