//! 7-Zip (`.7z`) reader.
//!
//! Recognised by `detect_fs` via the `37 7A BC AF 27 1C` signature.
//!
//! With the `sevenz` Cargo feature this is a real **read-only** reader for the
//! common single-coder case. It parses the 7z container — the 32-byte
//! signature header, the (optionally LZMA-packed `kEncodedHeader`) end header,
//! `StreamsInfo` (pack info, folders/coders, substreams) and `FilesInfo`
//! (UTF-16 names, empty-stream / empty-file vectors) — and maps every file to
//! its folder substream.
//!
//! Decoding is wired for folders whose single coder is **Copy**, **LZMA**,
//! **BZip2** or **Deflate** (solid folders are sliced per substream, decoded
//! once on demand). Folders using **LZMA2**, BCJ/Delta branch filters, PPMd,
//! AES (encryption) or any multi-coder pipeline list correctly but read as a
//! clean `Unsupported`, pending raw-LZMA2 + branch-filter codecs in `compcol`.
//!
//! Without the `sevenz` feature this stays a detection-only scaffold.

use std::path::Path;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;
use crate::fs::{DirEntry, FileAttrs, FileReadHandle, Filesystem, MutationCapability};

/// 7z filesystem handle.
pub struct SevenZFs {
    fs: ArchiveFs,
    #[cfg(feature = "sevenz")]
    inner: imp::Inner,
}

impl SevenZFs {
    #[cfg(feature = "sevenz")]
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let p = imp::scan(dev)?;
        Ok(Self {
            fs: ArchiveFs::from_index(p.index),
            inner: p.inner,
        })
    }

    #[cfg(not(feature = "sevenz"))]
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self {
            fs: ArchiveFs::scaffold("7z"),
        })
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "7z: creating archives is not supported".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for SevenZFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl Filesystem for SevenZFs {
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
        #[cfg(feature = "sevenz")]
        match self.inner.lookup(path)? {
            imp::Lookup::File(loc) => return self.inner.open_file(dev, &loc),
            imp::Lookup::Unsupported(reason) => return Err(crate::Error::Unsupported(reason)),
            imp::Lookup::NotRegular => {}
        }
        self.fs.read_file(dev, path)
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &Path,
    ) -> Result<Box<dyn FileReadHandle + 'a>> {
        #[cfg(feature = "sevenz")]
        match self.inner.lookup(path)? {
            imp::Lookup::File(loc) => {
                use std::io::Read;
                let mut r = self.inner.open_file(dev, &loc)?;
                let mut bytes = Vec::new();
                r.read_to_end(&mut bytes).map_err(crate::Error::from)?;
                return Ok(Box::new(imp::mem_handle(bytes)));
            }
            imp::Lookup::Unsupported(reason) => return Err(crate::Error::Unsupported(reason)),
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

#[cfg(feature = "sevenz")]
mod imp {
    //! 7z container parser + single-coder folder decode.

    use std::collections::HashMap;
    use std::io::{self, Read};
    use std::path::Path;

    use compcol::Algorithm;

    use crate::block::BlockDevice;
    use crate::fs::archive::reader::BoundedDevReader;
    use crate::fs::archive::{ArchiveEntry, ArchiveIndex, EntryKind};
    use crate::{Error, Result};

    // Property IDs.
    const K_HEADER: u8 = 0x01;
    const K_MAIN_STREAMS_INFO: u8 = 0x04;
    const K_FILES_INFO: u8 = 0x05;
    const K_PACK_INFO: u8 = 0x06;
    const K_UNPACK_INFO: u8 = 0x07;
    const K_SUBSTREAMS_INFO: u8 = 0x08;
    const K_SIZE: u8 = 0x09;
    const K_CRC: u8 = 0x0A;
    const K_FOLDER: u8 = 0x0B;
    const K_CODERS_UNPACK_SIZE: u8 = 0x0C;
    const K_NUM_UNPACK_STREAM: u8 = 0x0D;
    const K_EMPTY_STREAM: u8 = 0x0E;
    const K_EMPTY_FILE: u8 = 0x0F;
    const K_NAME: u8 = 0x11;
    const K_ENCODED_HEADER: u8 = 0x17;

    /// One coder inside a folder.
    struct Coder {
        id: Vec<u8>,
        n_in: u64,
        n_out: u64,
        attr: Vec<u8>,
    }

    /// A folder = a coder pipeline producing one logical output stream.
    struct Folder {
        coders: Vec<Coder>,
        num_bind_pairs: u64,
        unpack_sizes: Vec<u64>, // one per coder output stream, in order
        num_substreams: u64,
        substream_sizes: Vec<u64>, // length == num_substreams
    }

    impl Folder {
        fn total_in(&self) -> u64 {
            self.coders.iter().map(|c| c.n_in).sum()
        }
        fn total_out(&self) -> u64 {
            self.coders.iter().map(|c| c.n_out).sum()
        }
        fn num_packed_streams(&self) -> u64 {
            self.total_in() - self.num_bind_pairs
        }
        /// The folder's overall uncompressed size = its last coder's output.
        fn unpack_size(&self) -> u64 {
            *self.unpack_sizes.last().unwrap_or(&0)
        }
    }

    /// Precomputed decode info for a folder (single-coder only).
    struct FolderRun {
        pack_offset: u64,
        pack_size: u64,
        unpack_size: u64,
        coder_id: Vec<u8>,
        coder_attr: Vec<u8>,
        /// Cumulative substream offsets within the decoded folder output
        /// (length == num_substreams + 1).
        sub_offsets: Vec<u64>,
        decodable: Option<String>, // Some(reason) if the folder can't be decoded
    }

    /// A file's location: folder + byte slice within the folder output.
    #[derive(Clone)]
    pub struct FileLoc {
        pub folder: usize,
        pub off: u64,
        pub len: u64,
    }

    pub enum Entry {
        File(FileLoc),
        Unsupported(String),
    }

    pub enum Lookup {
        File(FileLoc),
        Unsupported(String),
        NotRegular,
    }

    pub struct Inner {
        folders: Vec<FolderRun>,
        files: HashMap<String, Entry>,
    }

    pub struct Parsed {
        pub index: ArchiveIndex,
        pub inner: Inner,
    }

    // ─── byte-cursor with the 7z number / bitvector primitives ──────────────

    struct Cur<'a> {
        b: &'a [u8],
        p: usize,
    }
    impl<'a> Cur<'a> {
        fn new(b: &'a [u8]) -> Self {
            Cur { b, p: 0 }
        }
        fn byte(&mut self) -> Result<u8> {
            let v = *self
                .b
                .get(self.p)
                .ok_or_else(|| Error::InvalidImage("7z: truncated header".into()))?;
            self.p += 1;
            Ok(v)
        }
        fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
            let s = self
                .b
                .get(self.p..self.p + n)
                .ok_or_else(|| Error::InvalidImage("7z: truncated header field".into()))?;
            self.p += n;
            Ok(s)
        }
        /// 7z variable-length number (REAL_UINT64).
        fn num(&mut self) -> Result<u64> {
            let first = self.byte()?;
            let mut mask = 0x80u8;
            let mut val = 0u64;
            for i in 0..8 {
                if first & mask == 0 {
                    val |= ((first & (mask.wrapping_sub(1))) as u64) << (8 * i);
                    return Ok(val);
                }
                val |= (self.byte()? as u64) << (8 * i);
                mask >>= 1;
            }
            Ok(val)
        }
        fn usize_num(&mut self) -> Result<usize> {
            let n = self.num()?;
            usize::try_from(n).map_err(|_| Error::InvalidImage("7z: number too large".into()))
        }
        /// Read a bit vector of `n` bits (MSB first within each byte).
        fn bits(&mut self, n: usize) -> Result<Vec<bool>> {
            let mut out = Vec::with_capacity(n);
            let mut cur = 0u8;
            let mut mask = 0u8;
            for _ in 0..n {
                if mask == 0 {
                    cur = self.byte()?;
                    mask = 0x80;
                }
                out.push(cur & mask != 0);
                mask >>= 1;
            }
            Ok(out)
        }
        /// A bit vector that may be prefixed by an "all defined" byte.
        fn bits_all_defined(&mut self, n: usize) -> Result<Vec<bool>> {
            if self.byte()? != 0 {
                Ok(vec![true; n])
            } else {
                self.bits(n)
            }
        }
    }

    fn read_at(dev: &mut dyn BlockDevice, off: u64, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        dev.read_at(off, &mut buf)?;
        Ok(buf)
    }

    // ─── StreamsInfo parsing ────────────────────────────────────────────────

    struct StreamsInfo {
        pack_pos: u64,
        pack_sizes: Vec<u64>,
        folders: Vec<Folder>,
    }

    fn parse_streams_info(c: &mut Cur) -> Result<StreamsInfo> {
        let mut pack_pos = 0u64;
        let mut pack_sizes = Vec::new();
        let mut folders: Vec<Folder> = Vec::new();

        let mut id = c.byte()?;
        if id == K_PACK_INFO {
            pack_pos = c.num()?;
            let n = c.usize_num()?;
            loop {
                let pid = c.byte()?;
                if pid == K_SIZE {
                    pack_sizes = (0..n).map(|_| c.num()).collect::<Result<_>>()?;
                } else if pid == 0 {
                    break;
                } else {
                    // skip kCRC etc. inside packinfo — not expected, bail.
                    return Err(Error::InvalidImage(
                        "7z: unexpected packinfo property".into(),
                    ));
                }
            }
            id = c.byte()?;
        }
        if id == K_UNPACK_INFO {
            let fid = c.byte()?;
            if fid != K_FOLDER {
                return Err(Error::InvalidImage("7z: expected kFolder".into()));
            }
            let num_folders = c.usize_num()?;
            let external = c.byte()?;
            if external != 0 {
                return Err(Error::Unsupported(
                    "7z: external folder definitions not supported".into(),
                ));
            }
            for _ in 0..num_folders {
                folders.push(parse_folder(c)?);
            }
            // kCodersUnpackSize: one size per output stream across all folders.
            let usid = c.byte()?;
            if usid != K_CODERS_UNPACK_SIZE {
                return Err(Error::InvalidImage("7z: expected kCodersUnpackSize".into()));
            }
            for f in folders.iter_mut() {
                let n_out = f.total_out() as usize;
                f.unpack_sizes = (0..n_out).map(|_| c.num()).collect::<Result<_>>()?;
            }
            // Optional kCRC then kEnd.
            loop {
                let pid = c.byte()?;
                if pid == 0 {
                    break;
                } else if pid == K_CRC {
                    let defined = c.bits_all_defined(folders.len())?;
                    let ndef = defined.iter().filter(|&&d| d).count();
                    c.bytes(ndef * 4)?; // skip CRCs
                } else {
                    return Err(Error::InvalidImage(
                        "7z: unexpected unpackinfo property".into(),
                    ));
                }
            }
            id = c.byte()?;
        }

        // Default: one substream per folder, sized to the folder's unpack size.
        for f in folders.iter_mut() {
            f.num_substreams = 1;
            f.substream_sizes = vec![f.unpack_size()];
        }

        if id == K_SUBSTREAMS_INFO {
            parse_substreams_info(c, &mut folders)?;
            id = c.byte()?;
        }
        if id != 0 {
            return Err(Error::InvalidImage(
                "7z: expected kEnd of StreamsInfo".into(),
            ));
        }
        Ok(StreamsInfo {
            pack_pos,
            pack_sizes,
            folders,
        })
    }

    fn parse_folder(c: &mut Cur) -> Result<Folder> {
        let num_coders = c.usize_num()?;
        let mut coders = Vec::with_capacity(num_coders);
        for _ in 0..num_coders {
            let flag = c.byte()?;
            let id_size = (flag & 0x0F) as usize;
            let id = c.bytes(id_size)?.to_vec();
            let (n_in, n_out) = if flag & 0x10 != 0 {
                (c.num()?, c.num()?)
            } else {
                (1, 1)
            };
            let attr = if flag & 0x20 != 0 {
                let ps = c.usize_num()?;
                c.bytes(ps)?.to_vec()
            } else {
                Vec::new()
            };
            coders.push(Coder {
                id,
                n_in,
                n_out,
                attr,
            });
        }
        let total_out: u64 = coders.iter().map(|c| c.n_out).sum();
        let total_in: u64 = coders.iter().map(|c| c.n_in).sum();
        let num_bind_pairs = total_out - 1;
        for _ in 0..num_bind_pairs {
            c.num()?; // in index
            c.num()?; // out index
        }
        let num_packed = total_in - num_bind_pairs;
        if num_packed > 1 {
            for _ in 0..num_packed {
                c.num()?; // packed stream index
            }
        }
        Ok(Folder {
            coders,
            num_bind_pairs,
            unpack_sizes: Vec::new(),
            num_substreams: 1,
            substream_sizes: Vec::new(),
        })
    }

    fn parse_substreams_info(c: &mut Cur, folders: &mut [Folder]) -> Result<()> {
        let mut id = c.byte()?;
        if id == K_NUM_UNPACK_STREAM {
            for f in folders.iter_mut() {
                f.num_substreams = c.num()?;
            }
            id = c.byte()?;
        }
        // kSize: for each folder with >1 substream, all-but-last sizes; the
        // last is the remainder of the folder's unpack size.
        // (When num_substreams == 1, the single size IS the folder size.)
        for f in folders.iter_mut() {
            let total = f.unpack_size();
            let n = f.num_substreams;
            if n == 0 {
                f.substream_sizes = Vec::new();
                continue;
            }
            let mut sizes = Vec::with_capacity(n as usize);
            let mut sum = 0u64;
            if id == K_SIZE {
                for _ in 0..n - 1 {
                    let s = c.num()?;
                    sum += s;
                    sizes.push(s);
                }
            }
            sizes.push(total.saturating_sub(sum));
            f.substream_sizes = sizes;
        }
        if id == K_SIZE {
            id = c.byte()?;
        }
        // Skip an optional kCRC then kEnd.
        loop {
            if id == 0 {
                break;
            } else if id == K_CRC {
                let total_streams: usize = folders.iter().map(|f| f.num_substreams as usize).sum();
                let defined = c.bits_all_defined(total_streams)?;
                let ndef = defined.iter().filter(|&&d| d).count();
                c.bytes(ndef * 4)?;
            } else {
                return Err(Error::InvalidImage(
                    "7z: unexpected substreams property".into(),
                ));
            }
            id = c.byte()?;
        }
        Ok(())
    }

    // ─── single-coder folder decode ─────────────────────────────────────────

    /// Build a reader over a folder's whole decompressed output. Only
    /// single-coder Copy / LZMA / BZip2 / Deflate folders are supported.
    fn folder_output_reader<'a>(
        dev: &'a mut dyn BlockDevice,
        run: &FolderRun,
    ) -> Result<Box<dyn Read + 'a>> {
        let packed = BoundedDevReader::new(dev, run.pack_offset, run.pack_size);
        match run.coder_id.as_slice() {
            // Copy.
            [0x00] => Ok(Box::new(packed)),
            // LZMA: synthesize a `.lzma`-alone header (5 props + 8-byte size)
            // from the coder attributes + folder unpack size, then decode.
            [0x03, 0x01, 0x01] => {
                if run.coder_attr.len() < 5 {
                    return Err(Error::InvalidImage("7z: LZMA props too short".into()));
                }
                let mut header = Vec::with_capacity(13);
                header.extend_from_slice(&run.coder_attr[..5]);
                header.extend_from_slice(&run.unpack_size.to_le_bytes());
                let framed = io::Cursor::new(header).chain(packed);
                Ok(Box::new(compcol::io::DecoderReader::new(
                    framed,
                    compcol::lzma::Lzma::decoder(),
                )))
            }
            // BZip2 (standard .bz2 stream).
            [0x04, 0x02, 0x02] => Ok(Box::new(compcol::io::DecoderReader::new(
                packed,
                compcol::bzip2::Bzip2::decoder(),
            ))),
            // Deflate (raw).
            [0x04, 0x01, 0x08] => Ok(Box::new(compcol::io::DecoderReader::new(
                packed,
                compcol::deflate::Deflate::decoder(),
            ))),
            other => Err(Error::Unsupported(format!(
                "7z: coder {} not supported",
                hex(other)
            ))),
        }
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    impl Inner {
        pub fn lookup(&self, path: &Path) -> Result<Lookup> {
            let s = path
                .to_str()
                .ok_or_else(|| Error::InvalidArgument("7z: non-UTF-8 path".into()))?;
            let key = crate::fs::archive::tree::normalise_path(s);
            Ok(match self.files.get(&key) {
                Some(Entry::File(loc)) => Lookup::File(loc.clone()),
                Some(Entry::Unsupported(r)) => Lookup::Unsupported(r.clone()),
                None => Lookup::NotRegular,
            })
        }

        pub fn open_file<'a>(
            &self,
            dev: &'a mut dyn BlockDevice,
            loc: &FileLoc,
        ) -> Result<Box<dyn Read + 'a>> {
            let run = &self.folders[loc.folder];
            if let Some(reason) = &run.decodable {
                return Err(Error::Unsupported(reason.clone()));
            }
            let mut r = folder_output_reader(dev, run)?;
            skip_exact(&mut *r, loc.off)?;
            Ok(Box::new(LimitReader {
                inner: r,
                remaining: loc.len,
            }))
        }
    }

    // ─── top-level scan ─────────────────────────────────────────────────────

    pub fn scan(dev: &mut dyn BlockDevice) -> Result<Parsed> {
        let dev_len = dev.total_size();
        let sig = read_at(dev, 0, 32)?;
        if &sig[0..6] != b"7z\xBC\xAF\x27\x1C" {
            return Err(Error::InvalidImage("7z: bad signature".into()));
        }
        let next_off = u64::from_le_bytes(sig[12..20].try_into().unwrap());
        let next_size = u64::from_le_bytes(sig[20..28].try_into().unwrap());
        let header_at = 32u64
            .checked_add(next_off)
            .ok_or_else(|| Error::InvalidImage("7z: header offset overflow".into()))?;
        if next_size == 0 {
            // Empty archive.
            return Ok(Parsed {
                index: ArchiveIndex::new("7z"),
                inner: Inner {
                    folders: Vec::new(),
                    files: HashMap::new(),
                },
            });
        }
        if header_at + next_size > dev_len {
            return Err(Error::InvalidImage("7z: header past end of file".into()));
        }
        let mut header = read_at(dev, header_at, next_size as usize)?;

        // The end header may be an LZMA-packed `kEncodedHeader`; decode it to
        // recover the real `kHeader` bytes.
        if header.first() == Some(&K_ENCODED_HEADER) {
            let mut c = Cur::new(&header[1..]);
            let si = parse_streams_info(&mut c)?;
            let runs = build_runs(&si, dev_len)?;
            if runs.len() != 1 {
                return Err(Error::Unsupported(
                    "7z: multi-folder encoded header not supported".into(),
                ));
            }
            if let Some(reason) = &runs[0].decodable {
                return Err(Error::Unsupported(reason.clone()));
            }
            let mut r = folder_output_reader(dev, &runs[0])?;
            let mut decoded = Vec::new();
            r.read_to_end(&mut decoded).map_err(Error::from)?;
            header = decoded;
        }

        let mut index = ArchiveIndex::new("7z");
        let mut files: HashMap<String, Entry> = HashMap::new();

        let mut c = Cur::new(&header);
        if c.byte()? != K_HEADER {
            return Err(Error::InvalidImage("7z: expected kHeader".into()));
        }

        let mut streams: Option<StreamsInfo> = None;
        let mut runs: Vec<FolderRun> = Vec::new();

        // Walk the kHeader properties.
        loop {
            let id = c.byte()?;
            match id {
                0 => break,
                K_MAIN_STREAMS_INFO => {
                    let si = parse_streams_info(&mut c)?;
                    runs = build_runs(&si, dev_len)?;
                    streams = Some(si);
                }
                K_FILES_INFO => {
                    build_files(&mut c, streams.as_ref(), &runs, &mut index, &mut files)?;
                }
                _ => {
                    // kArchiveProperties / kAdditionalStreamsInfo etc. — these
                    // appear before the streams we handle; bail rather than
                    // misparse.
                    return Err(Error::Unsupported(format!(
                        "7z: header property {id:#x} not supported"
                    )));
                }
            }
        }

        Ok(Parsed {
            index,
            inner: Inner {
                folders: runs,
                files,
            },
        })
    }

    /// Precompute each folder's pack offset/size + decodability.
    fn build_runs(si: &StreamsInfo, dev_len: u64) -> Result<Vec<FolderRun>> {
        let base = 32u64 + si.pack_pos;
        // Prefix sums of pack sizes give each pack stream's offset.
        let mut pack_off = Vec::with_capacity(si.pack_sizes.len() + 1);
        let mut acc = base;
        pack_off.push(acc);
        for &s in &si.pack_sizes {
            acc = acc
                .checked_add(s)
                .ok_or_else(|| Error::InvalidImage("7z: pack size overflow".into()))?;
            pack_off.push(acc);
        }

        let mut runs = Vec::with_capacity(si.folders.len());
        let mut pack_idx = 0usize;
        for f in &si.folders {
            let num_packed = f.num_packed_streams() as usize;
            let single = f.coders.len() == 1 && num_packed == 1 && f.num_bind_pairs == 0;
            let pack_offset = *pack_off.get(pack_idx).unwrap_or(&base);
            // Sum the sizes of this folder's packed streams.
            let mut pack_size = 0u64;
            for k in 0..num_packed {
                pack_size += *si.pack_sizes.get(pack_idx + k).unwrap_or(&0);
            }
            pack_idx += num_packed;

            // Cumulative substream offsets within the folder output.
            let mut sub_offsets = Vec::with_capacity(f.substream_sizes.len() + 1);
            let mut o = 0u64;
            sub_offsets.push(0);
            for &s in &f.substream_sizes {
                o += s;
                sub_offsets.push(o);
            }

            let decodable = if !single {
                Some("7z: multi-coder / filtered folder not supported".to_string())
            } else if pack_offset + pack_size > dev_len {
                Some("7z: folder pack data past end of file".to_string())
            } else {
                let id = f.coders[0].id.as_slice();
                match id {
                    [0x00] | [0x03, 0x01, 0x01] | [0x04, 0x02, 0x02] | [0x04, 0x01, 0x08] => None,
                    _ => Some(format!("7z: coder {} not supported", hex(id))),
                }
            };

            runs.push(FolderRun {
                pack_offset,
                pack_size,
                unpack_size: f.unpack_size(),
                coder_id: f.coders.first().map(|c| c.id.clone()).unwrap_or_default(),
                coder_attr: f.coders.first().map(|c| c.attr.clone()).unwrap_or_default(),
                sub_offsets,
                decodable,
            });
        }
        Ok(runs)
    }

    /// Parse FilesInfo and map stream-bearing files to folder substreams.
    fn build_files(
        c: &mut Cur,
        streams: Option<&StreamsInfo>,
        runs: &[FolderRun],
        index: &mut ArchiveIndex,
        files: &mut HashMap<String, Entry>,
    ) -> Result<()> {
        let num_files = c.usize_num()?;
        let mut empty_stream = vec![false; num_files];
        let mut empty_file: Vec<bool> = Vec::new();
        let mut names: Vec<String> = Vec::new();

        loop {
            let prop = c.byte()?;
            if prop == 0 {
                break;
            }
            let size = c.usize_num()?;
            let end = c.p + size;
            match prop {
                K_EMPTY_STREAM => {
                    empty_stream = c.bits(num_files)?;
                }
                K_EMPTY_FILE => {
                    let n_empty = empty_stream.iter().filter(|&&e| e).count();
                    empty_file = c.bits(n_empty)?;
                }
                K_NAME => {
                    let external = c.byte()?;
                    if external != 0 {
                        return Err(Error::Unsupported(
                            "7z: external names not supported".into(),
                        ));
                    }
                    let raw = c.bytes(end - c.p)?;
                    names = decode_names(raw, num_files)?;
                }
                _ => {
                    // kMTime / kWinAttributes / kCTime / kATime / kDummy / …
                    c.bytes(end - c.p)?;
                }
            }
            // Resync to the declared property end (defensive).
            c.p = end;
        }

        if names.len() != num_files {
            return Err(Error::InvalidImage("7z: name count mismatch".into()));
        }

        // Build a flat list of substreams (folder, offset, len) in order.
        struct Sub {
            folder: usize,
            off: u64,
            len: u64,
        }
        let mut subs: Vec<Sub> = Vec::new();
        if let Some(si) = streams {
            for (fi, f) in si.folders.iter().enumerate() {
                let run = &runs[fi];
                for (si2, &len) in f.substream_sizes.iter().enumerate() {
                    subs.push(Sub {
                        folder: fi,
                        off: run.sub_offsets[si2],
                        len,
                    });
                }
            }
        }

        let mut empty_idx = 0usize;
        let mut sub_idx = 0usize;
        for i in 0..num_files {
            let path = crate::fs::archive::tree::normalise_path(&names[i]);
            if empty_stream[i] {
                let is_empty_file = empty_file.get(empty_idx).copied().unwrap_or(false);
                empty_idx += 1;
                if is_empty_file {
                    if path != "/" {
                        let mut e = ArchiveEntry::regular(
                            path.clone(),
                            crate::fs::archive::DataLocator {
                                offset: 0,
                                compressed_len: 0,
                                uncompressed_len: 0,
                                method: crate::fs::archive::Method::Stored,
                            },
                        );
                        e.kind = EntryKind::Regular;
                        index.push(e);
                        // Empty file: a zero-length stored member.
                        files.insert(
                            path,
                            Entry::File(FileLoc {
                                folder: usize::MAX,
                                off: 0,
                                len: 0,
                            }),
                        );
                    }
                } else if path != "/" {
                    index.push(ArchiveEntry::dir(path));
                }
            } else {
                let sub = subs
                    .get(sub_idx)
                    .ok_or_else(|| Error::InvalidImage("7z: more files than substreams".into()))?;
                sub_idx += 1;
                if path != "/" {
                    let mut e = ArchiveEntry::regular(
                        path.clone(),
                        crate::fs::archive::DataLocator {
                            offset: 0,
                            compressed_len: 0,
                            uncompressed_len: sub.len,
                            method: crate::fs::archive::Method::Stored,
                        },
                    );
                    e.kind = EntryKind::Regular;
                    index.push(e);

                    let run = &runs[sub.folder];
                    if let Some(reason) = &run.decodable {
                        files.insert(path, Entry::Unsupported(reason.clone()));
                    } else {
                        files.insert(
                            path,
                            Entry::File(FileLoc {
                                folder: sub.folder,
                                off: sub.off,
                                len: sub.len,
                            }),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// UTF-16LE, NUL-terminated, back-slash separated names → forward-slash
    /// normalised component strings.
    fn decode_names(raw: &[u8], num_files: usize) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(num_files);
        let mut units: Vec<u16> = Vec::new();
        let mut i = 0;
        while i + 1 < raw.len() {
            let u = u16::from_le_bytes([raw[i], raw[i + 1]]);
            i += 2;
            if u == 0 {
                let s: String = String::from_utf16_lossy(&units)
                    .chars()
                    .map(|ch| if ch == '\\' { '/' } else { ch })
                    .collect();
                out.push(s);
                units.clear();
            } else {
                units.push(u);
            }
        }
        Ok(out)
    }

    /// Read and discard exactly `n` bytes.
    fn skip_exact(r: &mut dyn Read, mut n: u64) -> Result<()> {
        let mut scratch = [0u8; 64 * 1024];
        while n > 0 {
            let want = n.min(scratch.len() as u64) as usize;
            let got = r.read(&mut scratch[..want]).map_err(Error::from)?;
            if got == 0 {
                return Err(Error::InvalidImage("7z: folder stream ended early".into()));
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

#[cfg(all(test, feature = "sevenz"))]
mod tests {
    use std::io::Read;
    use std::path::Path;

    use super::*;
    use crate::block::MemoryBackend;

    fn hello() -> Vec<u8> {
        b"Hello, 7z reader!\n".repeat(3)
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
        let mut fs = SevenZFs::open(&mut dev)?;
        let mut r = fs.read_file(&mut dev, Path::new(path))?;
        let mut out = Vec::new();
        r.read_to_end(&mut out).map_err(crate::Error::from)?;
        Ok(out)
    }

    fn names(arc: &[u8]) -> Vec<String> {
        let mut dev = dev_from(arc);
        let mut fs = SevenZFs::open(&mut dev).unwrap();
        fs.list(&mut dev, Path::new("/"))
            .unwrap()
            .iter()
            .map(|e| e.name.clone())
            .collect()
    }

    /// Copy folders with an uncompressed (`-mhc=off`) header.
    #[test]
    fn copy_uncompressed_header() {
        let arc = include_bytes!("testdata/copy_nohc.7z");
        assert_eq!(read_file(arc, "/hello.txt").unwrap(), hello());
        assert_eq!(read_file(arc, "/lorem.txt").unwrap(), lorem());
    }

    /// Copy folders with a compressed (LZMA `kEncodedHeader`) end header —
    /// exercises the header-decode bootstrap.
    #[test]
    fn copy_compressed_header() {
        let arc = include_bytes!("testdata/copy_hc.7z");
        assert_eq!(read_file(arc, "/hello.txt").unwrap(), hello());
        assert_eq!(read_file(arc, "/lorem.txt").unwrap(), lorem());
    }

    /// LZMA, solid (both files in one folder) with a compressed header —
    /// exercises LZMA decode + substream slicing.
    #[test]
    fn lzma_solid() {
        let arc = include_bytes!("testdata/lzma.7z");
        assert_eq!(read_file(arc, "/hello.txt").unwrap(), hello());
        assert_eq!(read_file(arc, "/lorem.txt").unwrap(), lorem());
        let n = names(arc);
        assert!(
            n.iter().any(|x| x == "hello.txt") && n.iter().any(|x| x == "lorem.txt"),
            "{n:?}"
        );
    }

    /// BZip2 and Deflate single-coder folders decode via compcol.
    #[test]
    fn bzip2_and_deflate() {
        for arc in [
            &include_bytes!("testdata/bzip2.7z")[..],
            &include_bytes!("testdata/deflate.7z")[..],
        ] {
            assert_eq!(read_file(arc, "/hello.txt").unwrap(), hello());
            assert_eq!(read_file(arc, "/lorem.txt").unwrap(), lorem());
        }
    }

    /// LZMA2 (the 7-Zip default) lists correctly but reads `Unsupported`
    /// pending a raw-LZMA2 entry point in compcol.
    #[test]
    fn lzma2_lists_but_read_is_unsupported() {
        let arc = include_bytes!("testdata/lzma2.7z");
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

    /// Cross-check against the reference `7z` tool when installed.
    #[test]
    fn matches_7z_reference() {
        use std::process::Command;
        if Command::new("7z").arg("--help").output().is_err() {
            eprintln!("skipping: 7z not installed");
            return;
        }
        let arc = include_bytes!("testdata/lzma.7z");
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(tmp.as_file_mut(), arc).unwrap();
        for name in ["hello.txt", "lorem.txt"] {
            let out = Command::new("7z")
                .args(["x", "-so", tmp.path().to_str().unwrap(), name])
                .output()
                .unwrap();
            // `7z x -so` behaves inconsistently across platforms/builds; only
            // cross-check when the reference tool actually produced the file.
            if !out.status.success() {
                eprintln!("skipping: `7z x -so` unavailable here");
                return;
            }
            assert_eq!(
                read_file(arc, &format!("/{name}")).unwrap(),
                out.stdout,
                "reader vs 7z mismatch for {name}"
            );
        }
    }
}
