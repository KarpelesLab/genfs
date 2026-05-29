//! Microsoft Cabinet (`.cab`, `MSCF`) reader.
//!
//! Recognised by `detect_fs` via the `MSCF` signature. With the `cab` Cargo
//! feature this is a real **read-only** reader: it parses the cabinet (see
//! [`scan`]) and extracts files by decompressing the owning CFFOLDER via
//! `compcol` and slicing out the file's range (see [`folder`]). Supported
//! folder methods: None (Store), LZX, Quantum, and single-block MSZIP;
//! multi-block MSZIP awaits compcol preset-dictionary support
//! (KarpelesLab/compcol#22). Spanned/multi-cabinet sets and archive creation
//! are not supported.
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
    /// `ArchiveFs` produce the proper error).
    #[cfg(feature = "cab")]
    fn file_bytes(&mut self, dev: &mut dyn BlockDevice, path: &Path) -> Result<Option<Vec<u8>>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("cab: non-UTF-8 path".to_string()))?;
        let key = crate::fs::archive::tree::normalise_path(s);
        let slice = match self.files.get(&key) {
            Some(Some(slice)) => *slice,
            Some(None) => {
                return Err(crate::Error::Unsupported(format!(
                    "cab: {key:?} cannot be extracted (spans cabinets or uses an \
                     unsupported method)"
                )));
            }
            None => return Ok(None), // not a regular file
        };
        let folder = self.folders[slice.folder].clone();
        let bytes = folder::decode_folder(dev, &folder)?;
        let start = slice.uncomp_offset as usize;
        let end = start
            .checked_add(slice.len as usize)
            .ok_or_else(|| crate::Error::InvalidImage("cab: file slice overflow".into()))?;
        if end > bytes.len() {
            return Err(crate::Error::InvalidImage(format!(
                "cab: file slice {start}..{end} exceeds folder ({} bytes)",
                bytes.len()
            )));
        }
        Ok(Some(bytes[start..end].to_vec()))
    }
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
        {
            if let Some(bytes) = self.file_bytes(dev, path)? {
                return Ok(Box::new(std::io::Cursor::new(bytes)));
            }
        }
        self.fs.read_file(dev, path)
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &Path,
    ) -> Result<Box<dyn FileReadHandle + 'a>> {
        #[cfg(feature = "cab")]
        {
            if let Some(bytes) = self.file_bytes(dev, path)? {
                let len = bytes.len() as u64;
                return Ok(Box::new(MemHandle {
                    cur: std::io::Cursor::new(bytes),
                    len,
                }));
            }
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

    /// Build a cabinet with one folder (compression `type_compress`) holding
    /// one file `name` of logical size `cb_file`, whose folder data is the
    /// given CFDATA `blocks` (`(payload, cb_uncomp)` each).
    pub fn build_cab(
        type_compress: u16,
        blocks: &[(Vec<u8>, u16)],
        cb_file: u32,
        name: &str,
    ) -> Vec<u8> {
        let name_bytes = name.as_bytes();
        let coff_files: u32 = 36 + 8;
        let cffile_size = 16 + name_bytes.len() + 1;
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
        out.extend_from_slice(&1u16.to_le_bytes()); // cFiles
        out.extend_from_slice(&0u16.to_le_bytes()); // flags
        out.extend_from_slice(&0u16.to_le_bytes()); // setID
        out.extend_from_slice(&0u16.to_le_bytes()); // iCabinet
        // CFFOLDER (8 bytes).
        out.extend_from_slice(&(coff_cab_start as u32).to_le_bytes());
        out.extend_from_slice(&(blocks.len() as u16).to_le_bytes());
        out.extend_from_slice(&type_compress.to_le_bytes());
        // CFFILE.
        out.extend_from_slice(&cb_file.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // uoffFolderStart
        out.extend_from_slice(&0u16.to_le_bytes()); // iFolder
        out.extend_from_slice(&0u16.to_le_bytes()); // date
        out.extend_from_slice(&0u16.to_le_bytes()); // time
        out.extend_from_slice(&0u16.to_le_bytes()); // attribs
        out.extend_from_slice(name_bytes);
        out.push(0);
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
    fn mszip_multi_block_is_unsupported() {
        // Two MSZIP blocks → cross-block history needed (compcol#22).
        let b1 = mszip_block(b"first block");
        let b2 = mszip_block(b"second block");
        let cab = build_cab(1, &[(b1, 11), (b2, 12)], 23, "big.txt");
        let err = read_file(&cab, "/big.txt").unwrap_err();
        assert!(
            matches!(err, crate::Error::Unsupported(_)),
            "expected Unsupported, got {err:?}"
        );
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
