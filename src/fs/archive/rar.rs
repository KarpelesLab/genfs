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
//! `compcol::rar5` (LZ77/Huffman incl. the x86 E8/E8E9 filters).
//!
//! **Solid archives** are supported: a solid group is the concatenation of
//! its members' packed runs decoded as one continuous stream with a shared
//! LZ window (`compcol::rar5` itself is per-stream, so we drive a single
//! resumable decoder over the whole group ourselves). A persistent forward
//! cursor (`imp::LiveSolid`) lets a sequential walk — notably `repack` —
//! decompress the group exactly **once**; a backward/random read of an
//! earlier member rebuilds the cursor and re-decodes from the group start
//! (correct, bounded memory, no whole-group buffering).
//!
//! Out of scope (clean `Unsupported`): RAR4, encryption, stored members
//! inside a solid group, the delta/ARM/other RAR5 filters compcol doesn't
//! decode, multi-volume sets. Without the `rar` feature this stays a
//! detection-only scaffold.

use std::path::Path;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;
use crate::fs::{DirEntry, FileAttrs, FileReadHandle, Filesystem, MutationCapability};

/// RAR filesystem handle.
pub struct RarFs {
    fs: ArchiveFs,
    #[cfg(feature = "rar")]
    files: std::collections::HashMap<String, imp::Entry>,
    /// Solid groups (>1 member). `imp::Entry::Solid` indexes into this.
    #[cfg(feature = "rar")]
    groups: Vec<imp::SolidGroup>,
    /// Persistent forward decode cursor for the most-recently-read solid
    /// group, so a sequential walk decompresses the group only once.
    #[cfg(feature = "rar")]
    live: Option<imp::LiveSolid>,
    /// Count of solid-cursor (re)builds — instrumentation for the
    /// decode-once guarantee. An in-order walk of a group builds it once.
    #[cfg(feature = "rar")]
    rebuilds: usize,
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
                groups: Vec::new(),
                live: None,
                rebuilds: 0,
            });
        }
        let p = imp::scan(dev)?;
        Ok(Self {
            fs: ArchiveFs::from_index(p.index),
            files: p.files,
            groups: p.groups,
            live: None,
            rebuilds: 0,
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
            imp::Lookup::Plain(f) => return imp::open_file(dev, &f),
            imp::Lookup::Solid { group, member } => {
                return self.read_solid(dev, group, member);
            }
            imp::Lookup::Unextractable(key) => {
                return Err(crate::Error::Unsupported(format!(
                    "rar: {key:?} cannot be extracted (encrypted, stored-in-solid, or unsupported method)"
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
            imp::Lookup::Plain(f) => {
                use std::io::Read;
                let mut r = imp::open_file(dev, &f)?;
                let mut bytes = Vec::new();
                r.read_to_end(&mut bytes).map_err(crate::Error::from)?;
                return Ok(Box::new(imp::mem_handle(bytes)));
            }
            imp::Lookup::Solid { group, member } => {
                use std::io::Read;
                let mut r = self.read_solid(dev, group, member)?;
                let mut bytes = Vec::new();
                r.read_to_end(&mut bytes).map_err(crate::Error::from)?;
                return Ok(Box::new(imp::mem_handle(bytes)));
            }
            imp::Lookup::Unextractable(key) => {
                return Err(crate::Error::Unsupported(format!(
                    "rar: {key:?} cannot be extracted (encrypted, stored-in-solid, or unsupported method)"
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
            Some(imp::Entry::Plain(f)) => imp::Lookup::Plain(*f),
            Some(imp::Entry::Solid { group, member }) => imp::Lookup::Solid {
                group: *group,
                member: *member,
            },
            Some(imp::Entry::Unextractable) => imp::Lookup::Unextractable(key),
            None => imp::Lookup::NotRegular,
        })
    }

    /// Open member `member` of solid `group` as a bounded streaming reader,
    /// driving the persistent forward cursor so that an in-order walk decodes
    /// the group only once. A request for a member at or after the cursor
    /// advances it (skipping intervening output); a request before the cursor
    /// rebuilds it and re-decodes from the group start.
    fn read_solid<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        group: usize,
        member: usize,
    ) -> Result<Box<dyn std::io::Read + 'a>> {
        let RarFs {
            live,
            groups,
            rebuilds,
            ..
        } = self;
        let g = &groups[group];
        let start = g.starts[member];
        let remaining = g.starts[member + 1] - start;

        // Reuse the live cursor only if it's this group and hasn't already
        // advanced past where we need to start; otherwise rebuild from zero.
        let reusable = matches!(live, Some(l) if l.group == group && l.out_pos <= start);
        if !reusable {
            *live = Some(imp::LiveSolid::new(group, g.window, g.total));
            *rebuilds += 1;
        }
        let l = live.as_mut().unwrap();
        imp::skip_to(l, g, dev, start)?;
        Ok(Box::new(imp::SolidReader {
            live: l,
            areas: &g.areas,
            dev,
            remaining,
        }))
    }
}

#[cfg(feature = "rar")]
mod imp {
    //! RAR5 container parser + per-file decode.

    use std::collections::HashMap;
    use std::io::{self, Read};

    // Bring the `compcol::Decoder` trait methods (`decode` / `discard_output`
    // / `finish`) into scope without shadowing the concrete
    // `compcol::rar5::Decoder` type, plus the `Status` enum.
    use compcol::Decoder as _;
    use compcol::Status;

    use crate::block::BlockDevice;
    use crate::fs::archive::reader::BoundedDevReader;
    use crate::fs::archive::{ArchiveEntry, ArchiveIndex, DataLocator, EntryKind, Method};
    use crate::{Error, Result};

    /// Staging chunk pulled from the device per refill while driving the solid
    /// decoder. The decoder buffers input internally, so this only needs to be
    /// large enough to amortise `read_at` calls.
    const SOLID_CHUNK: usize = 64 * 1024;
    /// Ceiling on the LZ window we will allocate from a file's dict bits.
    /// RAR5's dict_n maxes at 15 (1 GiB nominal); a malicious archive can
    /// set it to force an OOM. Cap at 64 MiB — larger windows just mean a
    /// slightly less efficient (but correct) decode for honest archives,
    /// which in practice never exceed this for the files we extract.
    const MAX_WINDOW: usize = 64 * 1024 * 1024;

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

    /// What a path resolves to in the per-file table.
    pub enum Entry {
        /// Standalone member (its own 1-member group): decoded independently.
        Plain(RarFile),
        /// Member of a multi-member solid group; indices into `RarFs::groups`.
        Solid { group: usize, member: usize },
        /// Recognised but undecodable (encrypted, stored-in-solid, …).
        Unextractable,
    }

    pub enum Lookup {
        Plain(RarFile),
        Solid { group: usize, member: usize },
        Unextractable(String),
        NotRegular,
    }

    /// One member's packed run within a solid group.
    #[derive(Debug, Clone, Copy)]
    pub struct DataArea {
        pub offset: u64,
        pub pack: u64,
    }

    /// A solid group: members' packed runs form one continuous LZ stream
    /// sharing a single window. `starts[i]..starts[i+1]` is member `i`'s byte
    /// range in the decoded output; `starts` has `members + 1` entries.
    pub struct SolidGroup {
        pub areas: Vec<DataArea>,
        pub window: usize,
        pub starts: Vec<u64>,
        pub total: u64,
    }

    /// Persistent forward decode cursor over one solid group. The owned
    /// `compcol::rar5::Decoder` carries the LZ window across members; the
    /// compressed cursor (`in_area`/`in_off`) and the staging buffer survive
    /// across `read_file` calls so a member boundary mid-chunk is seamless.
    pub struct LiveSolid {
        pub group: usize,
        dec: compcol::rar5::Decoder,
        /// Decompressed bytes emitted **or** discarded so far (absolute in the
        /// group's output). The next sequential member begins exactly here.
        pub out_pos: u64,
        in_area: usize,
        in_off: u64,
        in_buf: Vec<u8>,
        in_consumed: usize,
        in_filled: usize,
    }

    impl LiveSolid {
        pub fn new(group: usize, window: usize, total: u64) -> Self {
            LiveSolid {
                group,
                dec: compcol::rar5::Decoder::with_unpack_size_and_window(total, window),
                out_pos: 0,
                in_area: 0,
                in_off: 0,
                in_buf: vec![0u8; SOLID_CHUNK],
                in_consumed: 0,
                in_filled: 0,
            }
        }
    }

    pub struct Parsed {
        pub index: ArchiveIndex,
        pub files: HashMap<String, Entry>,
        pub groups: Vec<SolidGroup>,
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

    /// One member collected during the scan, before groups are finalised.
    struct MemberB {
        path: String,
        data_offset: u64,
        pack: u64,
        unpack: u64,
        method: u64,
        window: usize,
    }

    /// Parse a RAR5 archive into the directory tree + per-file table.
    ///
    /// Files are grouped by solid runs: a file with the solid bit **clear**
    /// (or the very first file) starts a new group; a file with the bit
    /// **set** continues the current group. 1-member groups become
    /// [`Entry::Plain`] (independent decode, the non-solid path); multi-member
    /// groups become a [`SolidGroup`] with each member an [`Entry::Solid`].
    pub fn scan(dev: &mut dyn BlockDevice) -> Result<Parsed> {
        let dev_len = dev.total_size();
        let mut index = ArchiveIndex::new("rar");
        // Members collected per solid run, in archive order.
        let mut groups_b: Vec<Vec<MemberB>> = Vec::new();

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
                    // Compute the window with a 32-bit-safe shift and clamp
                    // it: dict_n up to 15 would otherwise demand up to 1 GiB.
                    // Also bound by the archive's real size — the window can
                    // never usefully exceed the bytes we can read.
                    let window = 0x20000u32
                        .checked_shl(dict_n as u32)
                        .map(|w| w as usize)
                        .unwrap_or(MAX_WINDOW)
                        .min(MAX_WINDOW)
                        .min((dev_len as usize).max(0x20000));
                    // A non-solid file (or the first file overall) starts a new
                    // group; a solid file continues the current one.
                    if !solid || groups_b.is_empty() {
                        groups_b.push(Vec::new());
                    }
                    groups_b.last_mut().unwrap().push(MemberB {
                        path,
                        data_offset: header_end,
                        pack: data_size,
                        unpack: unpack_size,
                        method,
                        window,
                    });
                } else if path != "/" && is_dir {
                    index.push(ArchiveEntry::dir(path));
                }
            }

            pos = header_end + data_size;
        }

        // Finalise groups into the per-file table + solid-group vector.
        let mut files: HashMap<String, Entry> = HashMap::new();
        let mut groups: Vec<SolidGroup> = Vec::new();
        for members in groups_b {
            if members.len() == 1 {
                let m = &members[0];
                files.insert(
                    m.path.clone(),
                    Entry::Plain(RarFile {
                        data_offset: m.data_offset,
                        pack_size: m.pack,
                        unpack_size: m.unpack,
                        window: m.window,
                        store: m.method == 0,
                    }),
                );
                continue;
            }
            // Multi-member solid group. A stored member isn't part of the LZ
            // bitstream, so it would break the continuous decode — refuse the
            // whole group cleanly rather than mis-decode.
            if members.iter().any(|m| m.method == 0) {
                for m in &members {
                    files.insert(m.path.clone(), Entry::Unextractable);
                }
                continue;
            }
            let gi = groups.len();
            let window = members[0].window;
            let mut areas = Vec::with_capacity(members.len());
            let mut starts = Vec::with_capacity(members.len() + 1);
            let mut total = 0u64;
            starts.push(0);
            for (mi, m) in members.iter().enumerate() {
                areas.push(DataArea {
                    offset: m.data_offset,
                    pack: m.pack,
                });
                total += m.unpack;
                starts.push(total);
                files.insert(
                    m.path.clone(),
                    Entry::Solid {
                        group: gi,
                        member: mi,
                    },
                );
            }
            groups.push(SolidGroup {
                areas,
                window,
                starts,
                total,
            });
        }

        Ok(Parsed {
            index,
            files,
            groups,
        })
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

    /// Refill the staging buffer from the next compressed bytes of the solid
    /// group, advancing past exhausted data areas. A fill never spans a data
    /// area (the decoder treats the concatenation as one stream regardless).
    /// Returns `false` once the group's whole compressed input is consumed.
    fn refill(
        in_buf: &mut [u8],
        in_consumed: &mut usize,
        in_filled: &mut usize,
        in_area: &mut usize,
        in_off: &mut u64,
        areas: &[DataArea],
        dev: &mut dyn BlockDevice,
    ) -> Result<bool> {
        while *in_area < areas.len() && *in_off >= areas[*in_area].pack {
            *in_area += 1;
            *in_off = 0;
        }
        if *in_area >= areas.len() {
            return Ok(false);
        }
        let a = areas[*in_area];
        let avail = a.pack - *in_off;
        let want = (in_buf.len() as u64).min(avail) as usize;
        dev.read_at(a.offset + *in_off, &mut in_buf[..want])?;
        *in_consumed = 0;
        *in_filled = want;
        *in_off += want as u64;
        Ok(true)
    }

    /// Advance the live cursor's decoder forward to absolute group offset
    /// `target`, discarding the intervening output without materialising it.
    /// For an in-order walk `target == out_pos` and this is a no-op.
    pub fn skip_to(
        l: &mut LiveSolid,
        g: &SolidGroup,
        dev: &mut dyn BlockDevice,
        target: u64,
    ) -> Result<()> {
        while l.out_pos < target {
            let LiveSolid {
                dec,
                in_buf,
                in_consumed,
                in_filled,
                in_area,
                in_off,
                out_pos,
                ..
            } = l;
            if *in_consumed >= *in_filled
                && !refill(
                    in_buf,
                    in_consumed,
                    in_filled,
                    in_area,
                    in_off,
                    &g.areas,
                    dev,
                )?
            {
                return Err(Error::InvalidImage(
                    "rar: solid stream ended before a member offset".into(),
                ));
            }
            let n = (target - *out_pos) as usize;
            let (p, _status) = dec
                .discard_output(&in_buf[*in_consumed..*in_filled], n)
                .map_err(|e| Error::InvalidImage(format!("rar: solid skip failed: {e:?}")))?;
            *in_consumed += p.consumed;
            *out_pos += p.written as u64;
            if p.consumed == 0 && p.written == 0 {
                return Err(Error::InvalidImage(
                    "rar: solid decode stalled during skip".into(),
                ));
            }
        }
        Ok(())
    }

    /// Streaming reader over one member of a solid group. Drives the shared
    /// resumable decoder (mirroring `compcol::io::DecoderReader`'s loop),
    /// pulling compressed bytes from the device on demand and capping output
    /// at the member's length so the cursor lands exactly on the next member.
    pub struct SolidReader<'a> {
        pub live: &'a mut LiveSolid,
        pub areas: &'a [DataArea],
        pub dev: &'a mut dyn BlockDevice,
        pub remaining: u64,
    }

    impl Read for SolidReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.remaining == 0 || buf.is_empty() {
                return Ok(0);
            }
            let want = (buf.len() as u64).min(self.remaining) as usize;
            loop {
                let LiveSolid {
                    dec,
                    in_buf,
                    in_consumed,
                    in_filled,
                    in_area,
                    in_off,
                    out_pos,
                    ..
                } = &mut *self.live;
                let had_input = if *in_consumed < *in_filled {
                    true
                } else {
                    refill(
                        in_buf,
                        in_consumed,
                        in_filled,
                        in_area,
                        in_off,
                        self.areas,
                        self.dev,
                    )
                    .map_err(io::Error::other)?
                };
                let (p, status) =
                    dec.decode(&in_buf[*in_consumed..*in_filled], &mut buf[..want])?;
                *in_consumed += p.consumed;
                *out_pos += p.written as u64;
                if p.written > 0 {
                    self.remaining -= p.written as u64;
                    return Ok(p.written);
                }
                if !had_input {
                    // No compressed input left and nothing emitted: flush the
                    // decoder's tail. A well-formed member ends here.
                    let (pf, _s) = dec.finish(&mut buf[..want])?;
                    if pf.written > 0 {
                        *out_pos += pf.written as u64;
                        self.remaining -= pf.written as u64;
                        return Ok(pf.written);
                    }
                    return Ok(0);
                }
                // Had input but produced nothing — normally the decoder
                // buffered an incomplete block (consumed all, InputEmpty) and
                // the next loop refills. Guard against a wedged decoder.
                if p.consumed == 0 && !matches!(status, Status::OutputFull) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "rar: solid decode made no progress",
                    ));
                }
            }
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

    /// Solid archive: the second member (`lorem.txt`) continues the first's
    /// LZ window, so it now decodes correctly via the shared cursor.
    #[test]
    fn solid_two_member_round_trip() {
        let arc = include_bytes!("testdata/solid5.rar");
        assert_eq!(read_file(arc, "/hello.txt").unwrap(), hello());
        assert_eq!(read_file(arc, "/lorem.txt").unwrap(), lorem());
    }

    // The `solidmulti5.rar` fixture was built with
    //   rar a -ma5 -s -m5 solidmulti5.rar alpha.txt beta.txt gamma.txt big.txt
    // over files that deliberately share a long recurring phrase, so later
    // members genuinely back-reference earlier members' window. `rar` stores
    // them alphabetically: alpha, beta, big, gamma.
    const SHARED: &str = "The quick brown fox jumps over the lazy dog while the lazy dog sleeps. ";
    fn alpha() -> Vec<u8> {
        format!(
            "{}{}",
            SHARED.repeat(30),
            "alpha-unique-padding ".repeat(20)
        )
        .into_bytes()
    }
    fn beta() -> Vec<u8> {
        format!("{}{}", "beta intro ".repeat(5), SHARED.repeat(50)).into_bytes()
    }
    fn gamma() -> Vec<u8> {
        format!(
            "{}{}",
            "Lorem ipsum dolor sit amet. ".repeat(15),
            SHARED.repeat(25)
        )
        .into_bytes()
    }
    fn big() -> Vec<u8> {
        format!("{}{}", SHARED.repeat(200), "0123456789 ".repeat(500)).into_bytes()
    }
    /// Members in the archive's stored (alphabetical) order.
    fn multi_members() -> [(&'static str, Vec<u8>); 4] {
        [
            ("alpha.txt", alpha()),
            ("beta.txt", beta()),
            ("big.txt", big()),
            ("gamma.txt", gamma()),
        ]
    }

    /// Every member of a 4-file solid group decodes to its exact contents.
    #[test]
    fn solid_multi_member_round_trip() {
        let arc = include_bytes!("testdata/solidmulti5.rar");
        for (name, want) in multi_members() {
            assert_eq!(
                read_file(arc, &format!("/{name}")).unwrap(),
                want,
                "mismatch for {name}"
            );
        }
    }

    /// Reading the whole solid group in archive order through one `RarFs`
    /// decompresses it **once**: the forward cursor is built a single time and
    /// never rewinds. (Reading an earlier member afterwards forces a rebuild.)
    #[test]
    fn solid_group_decoded_once_in_order() {
        let arc = include_bytes!("testdata/solidmulti5.rar");
        let mut dev = dev_from(arc);
        let mut fs = RarFs::open(&mut dev).unwrap();
        for (name, want) in multi_members() {
            let mut r = fs
                .read_file(&mut dev, Path::new(&format!("/{name}")))
                .unwrap();
            let mut out = Vec::new();
            r.read_to_end(&mut out).unwrap();
            drop(r);
            assert_eq!(out, want, "mismatch for {name}");
        }
        // One build for the group; the cursor advanced straight through.
        assert_eq!(fs.rebuilds, 1, "group should decode exactly once in order");
        let live = fs.live.as_ref().unwrap();
        assert_eq!(live.out_pos, fs.groups[live.group].total);

        // A backward read (earlier member) now forces exactly one rebuild.
        let mut r = fs.read_file(&mut dev, Path::new("/alpha.txt")).unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        drop(r);
        assert_eq!(out, alpha());
        assert_eq!(fs.rebuilds, 2, "backward seek should rebuild once");
    }

    /// Cross-check the solid fixture against the reference `unrar`.
    #[test]
    fn solid_matches_unrar_reference() {
        use std::process::Command;
        if Command::new("unrar").arg("--help").output().is_err() {
            eprintln!("skipping: unrar not installed");
            return;
        }
        let arc = include_bytes!("testdata/solidmulti5.rar");
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(tmp.as_file_mut(), arc).unwrap();
        for (name, _) in multi_members() {
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
}
