//! Classic **HFS** (Hierarchical File System, Mac OS ≤ 8 / System ≤ 8) —
//! read-only reader.
//!
//! Distinct from HFS+ (`src/fs/hfs_plus/`): classic HFS uses 512-byte B-tree
//! nodes, MacRoman Pascal names (≤ 31 chars), 16-bit allocation-block addressing
//! and a Master Directory Block (MDB) at volume offset 1024 with signature
//! `BD` (0x4244). Recognised by `detect_fs`; commonly found inside DiskCopy 4.2
//! images (see `src/block/diskcopy.rs`), which this reader sees transparently.
//!
//! Strategy: classic HFS volumes are small (floppies / small disks), so at
//! `open` we read the whole catalog + extents-overflow B-trees into memory and
//! build a parent→children map. That sidesteps HFS's exact key-ordering table
//! (we match names by case-insensitive scan, never by B-tree key comparison)
//! and keeps `list`/`cat` simple. Each file's **data fork** is exposed (the
//! resource fork is Mac metadata and ignored, like the StuffIt reader).

mod macroman;

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use crate::block::BlockDevice;
use crate::fs::{DirEntry, EntryKind, FileAttrs, Filesystem, MutationCapability};
use crate::{Error, Result};

/// HFS MDB lives two 512-byte sectors into the volume.
const MDB_OFFSET: u64 = 1024;
/// Classic HFS B-tree node size is fixed at 512 bytes.
const NODE_SIZE: usize = 512;
/// Root directory CNID.
const ROOT_CNID: u32 = 2;
/// Seconds between the Mac (1904) and Unix (1970) epochs.
const MAC_EPOCH_DELTA: u32 = 2_082_844_800;
/// Cap on the catalog / extents file sizes we'll buffer in RAM.
const MAX_TREE_BYTES: u64 = 64 * 1024 * 1024;

#[inline]
fn be16(b: &[u8], o: usize) -> u16 {
    u16::from_be_bytes([b[o], b[o + 1]])
}
#[inline]
fn be32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn round_up_even(x: usize) -> usize {
    (x + 1) & !1
}

/// A fork's first three (inline) extent descriptors: `(startBlock, blockCount)`.
type ExtRec = [(u16, u16); 3];

fn ext_rec(b: &[u8], o: usize) -> ExtRec {
    [
        (be16(b, o), be16(b, o + 2)),
        (be16(b, o + 4), be16(b, o + 6)),
        (be16(b, o + 8), be16(b, o + 10)),
    ]
}

/// One indexed catalog entry (a child of some directory).
struct Node {
    name: String,
    cnid: u32,
    is_dir: bool,
    /// Data-fork logical size (files only).
    size: u64,
    mtime: u32,
    /// Data-fork inline extents (files only).
    inline: ExtRec,
}

/// Classic HFS volume (read-only, in-memory catalog).
pub struct Hfs {
    /// Allocation block size in bytes (`drAlBlkSiz`).
    block_size: u32,
    /// Byte offset of allocation block 0 (`drAlBlSt` × 512).
    alloc_base: u64,
    /// Volume name (`drVN`, MacRoman-decoded).
    pub volume_name: String,
    /// parentCNID → children.
    children: HashMap<u32, Vec<Node>>,
    /// Extents-overflow map: (forkType, CNID, startBlock) → 3 extents.
    overflow: HashMap<(u8, u32, u16), ExtRec>,
}

impl Hfs {
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let dev_len = dev.total_size();
        if dev_len < MDB_OFFSET + 512 {
            return Err(Error::InvalidImage(
                "hfs: device too small for an MDB".into(),
            ));
        }
        let mut m = [0u8; 512];
        dev.read_at(MDB_OFFSET, &mut m)?;
        if &m[0..2] != b"BD" {
            return Err(Error::InvalidImage(
                "hfs: bad MDB signature (expected BD)".into(),
            ));
        }
        // An HFS volume that merely *wraps* an embedded HFS+ volume has
        // `drEmbedSigWord` == 'H+'. That belongs to the HFS+ reader, not here.
        if be16(&m, 0x7C) == 0x482B {
            return Err(Error::Unsupported(
                "hfs: HFS-wrapped HFS+ volume is not supported (embedded H+)".into(),
            ));
        }

        let block_size = be32(&m, 20);
        let al_bl_st = be16(&m, 28) as u64;
        if block_size == 0 || !block_size.is_multiple_of(512) {
            return Err(Error::InvalidImage(format!(
                "hfs: bad allocation block size {block_size}"
            )));
        }
        let alloc_base = al_bl_st * 512;

        // Volume name: Str27 at offset 36 (length byte + chars).
        let vn_len = (m[36] as usize).min(27);
        let volume_name = macroman::decode(&m[37..37 + vn_len]);

        // Special files: extents-overflow (CNID 3) and catalog (CNID 4).
        let xt_size = be32(&m, 130) as u64;
        let xt_ext = ext_rec(&m, 134);
        let ct_size = be32(&m, 146) as u64;
        let ct_ext = ext_rec(&m, 150);
        for (sz, what) in [(xt_size, "extents"), (ct_size, "catalog")] {
            if sz > MAX_TREE_BYTES || sz > dev_len {
                return Err(Error::InvalidImage(format!(
                    "hfs: {what} file size {sz} is implausible"
                )));
            }
        }

        let mut hfs = Hfs {
            block_size,
            alloc_base,
            volume_name,
            children: HashMap::new(),
            overflow: HashMap::new(),
        };

        // 1) Read the extents-overflow file (inline extents only — it can't
        //    reference itself) and build the overflow map.
        let xt_bytes = hfs.read_fork(dev, &xt_ext, xt_size, dev_len)?;
        hfs.build_overflow(&xt_bytes)?;

        // 2) Read the catalog file (inline + overflow for CNID 4) and build the
        //    parent→children map.
        let cat_ext = hfs.full_extents(4, 0, &ct_ext, ct_size);
        let cat_bytes = hfs.read_fork(dev, &cat_ext, ct_size, dev_len)?;
        hfs.build_catalog(&cat_bytes)?;

        Ok(hfs)
    }

    /// Byte ranges (`(device offset, length)`) covering `extents`, capped at
    /// `logical` bytes, bounds-checked against the device.
    fn fork_ranges(
        &self,
        extents: &[(u16, u16)],
        logical: u64,
        dev_len: u64,
    ) -> Result<Vec<(u64, u64)>> {
        let mut out = Vec::new();
        let mut acc = 0u64;
        for &(sb, bc) in extents {
            if bc == 0 || acc >= logical {
                continue;
            }
            let off = self
                .alloc_base
                .checked_add(sb as u64 * self.block_size as u64)
                .ok_or_else(|| Error::InvalidImage("hfs: extent offset overflow".into()))?;
            let mut len = bc as u64 * self.block_size as u64;
            if acc + len > logical {
                len = logical - acc;
            }
            if off.checked_add(len).is_none_or(|e| e > dev_len) {
                return Err(Error::InvalidImage("hfs: extent past end of device".into()));
            }
            out.push((off, len));
            acc += len;
        }
        Ok(out)
    }

    /// Read a fork fully into memory (used for the small catalog/extents files).
    fn read_fork(
        &self,
        dev: &mut dyn BlockDevice,
        extents: &[(u16, u16)],
        logical: u64,
        dev_len: u64,
    ) -> Result<Vec<u8>> {
        let ranges = self.fork_ranges(extents, logical, dev_len)?;
        let mut out = vec![0u8; logical as usize];
        let mut pos = 0usize;
        for (off, len) in ranges {
            let len = len as usize;
            dev.read_at(off, &mut out[pos..pos + len])?;
            pos += len;
        }
        Ok(out)
    }

    /// Chase the extents-overflow chain to gather all extents of a fork.
    fn full_extents(
        &self,
        cnid: u32,
        fork_type: u8,
        inline: &ExtRec,
        logical: u64,
    ) -> Vec<(u16, u16)> {
        let mut exts: Vec<(u16, u16)> = inline.iter().copied().filter(|&(_, c)| c != 0).collect();
        let mut covered: u32 = exts.iter().map(|&(_, c)| c as u32).sum();
        let need = logical.div_ceil(self.block_size.max(1) as u64) as u32;
        let mut guard = 0u32;
        while covered < need {
            guard += 1;
            if guard > 8192 {
                break;
            }
            let Some(rec) = self.overflow.get(&(fork_type, cnid, covered as u16)) else {
                break;
            };
            let mut progressed = false;
            for &(s, c) in rec {
                if c != 0 {
                    exts.push((s, c));
                    covered += c as u32;
                    progressed = true;
                }
            }
            if !progressed {
                break;
            }
        }
        exts
    }

    /// Walk every leaf record of a B-tree buffer, calling `f(key, data)`.
    fn walk_leaves<F>(buf: &[u8], mut f: F) -> Result<()>
    where
        F: FnMut(&[u8], &[u8]) -> Result<()>,
    {
        if buf.len() < NODE_SIZE {
            return Ok(());
        }
        // Header node (node 0) record 0 = BTHdrRec; bthFNode @ +24 within node.
        let first_leaf = be32(buf, 24);
        let n_nodes = be32(buf, 36);
        let total_nodes = (buf.len() / NODE_SIZE) as u32;

        let mut node_idx = first_leaf;
        let mut steps = n_nodes.min(total_nodes).max(1) as u64 + 1;
        while node_idx != 0 {
            if steps == 0 {
                return Err(Error::InvalidImage("hfs: B-tree leaf chain cycle".into()));
            }
            steps -= 1;
            if node_idx >= total_nodes {
                return Err(Error::InvalidImage(
                    "hfs: leaf node index out of range".into(),
                ));
            }
            let start = node_idx as usize * NODE_SIZE;
            let node = &buf[start..start + NODE_SIZE];
            let flink = be32(node, 0);
            let ntype = node[8];
            let nrecs = be16(node, 10) as usize;
            if ntype != 0xFF {
                break; // not a leaf node — chain ended
            }
            for i in 0..nrecs {
                let lo = NODE_SIZE - 2 * (i + 1);
                let o = be16(node, lo) as usize;
                let o2 = be16(node, lo - 2) as usize;
                if o < 14 || o2 > NODE_SIZE || o >= o2 {
                    continue;
                }
                let rec = &node[o..o2];
                let key_len = rec[0] as usize;
                let key_total = round_up_even(1 + key_len);
                if 1 + key_len > rec.len() || key_total > rec.len() {
                    continue;
                }
                f(&rec[1..1 + key_len], &rec[key_total..])?;
            }
            node_idx = flink;
        }
        Ok(())
    }

    fn build_overflow(&mut self, buf: &[u8]) -> Result<()> {
        let mut map = HashMap::new();
        Self::walk_leaves(buf, |key, data| {
            // Extents key: forkType(1), fileNum(4), startBlock(2).
            if key.len() < 7 || data.len() < 12 {
                return Ok(());
            }
            let fork_type = key[0];
            let cnid = be32(key, 1);
            let start_abn = be16(key, 5);
            map.insert((fork_type, cnid, start_abn), ext_rec(data, 0));
            Ok(())
        })?;
        self.overflow = map;
        Ok(())
    }

    fn build_catalog(&mut self, buf: &[u8]) -> Result<()> {
        let mut children: HashMap<u32, Vec<Node>> = HashMap::new();
        Self::walk_leaves(buf, |key, data| {
            // Catalog key: reserved(1), parentID(4), name (Pascal MacRoman).
            if key.len() < 6 || data.is_empty() {
                return Ok(());
            }
            let parent = be32(key, 1);
            let name_len = (key[5] as usize).min(31);
            if 6 + name_len > key.len() {
                return Ok(());
            }
            // Classic HFS uses `:` as the path separator, so `/` (0x2F) is a
            // *legal* character inside a filename (e.g. "A/ROSE Includes").
            // fstool — like every POSIX tool — separates path components with
            // `/`, so a raw `/` in a name would be mis-split. Swap it to `:`,
            // exactly as macOS's own BSD layer does when surfacing HFS names
            // (Finder shows `/`, the shell shows `:`). `:` itself can never
            // appear in a raw HFS name, so the mapping is unambiguous.
            let name = macroman::decode(&key[6..6 + name_len]).replace('/', ":");

            match data[0] {
                1 if data.len() >= 18 => {
                    // Directory record: dirDirID @ +6, dirMdDat @ +14.
                    children.entry(parent).or_default().push(Node {
                        name,
                        cnid: be32(data, 6),
                        is_dir: true,
                        size: 0,
                        mtime: be32(data, 14),
                        inline: [(0, 0); 3],
                    });
                }
                2 if data.len() >= 86 => {
                    // File record (after cdrType + cdrResrv2 at @0/@1):
                    // filFlNum @ +20, filLgLen (data fork) @ +26, filMdDat @
                    // +48, data-fork filExtRec @ +74.
                    children.entry(parent).or_default().push(Node {
                        name,
                        cnid: be32(data, 20),
                        is_dir: false,
                        size: be32(data, 26) as u64,
                        mtime: be32(data, 48),
                        inline: ext_rec(data, 74),
                    });
                }
                _ => {} // thread records (3/4) and anything else: ignore
            }
            Ok(())
        })?;
        self.children = children;
        Ok(())
    }

    /// Resolve a slash path to a catalog node (and its parent CNID for `/`).
    fn resolve(&self, path: &str) -> Option<Resolved<'_>> {
        let mut cnid = ROOT_CNID;
        let comps: Vec<&str> = path.split('/').filter(|c| !c.is_empty()).collect();
        if comps.is_empty() {
            return Some(Resolved::Dir(ROOT_CNID));
        }
        for (i, comp) in comps.iter().enumerate() {
            let kids = self.children.get(&cnid)?;
            let node = kids
                .iter()
                .find(|n| macroman::eq_ignore_case(&n.name, comp))?;
            let last = i + 1 == comps.len();
            if node.is_dir {
                cnid = node.cnid;
                if last {
                    return Some(Resolved::Dir(node.cnid));
                }
            } else if last {
                return Some(Resolved::File(node));
            } else {
                return None; // a file in the middle of the path
            }
        }
        Some(Resolved::Dir(cnid))
    }

    pub fn list_path(&self, path: &str) -> Result<Vec<DirEntry>> {
        let cnid = match self.resolve(path) {
            Some(Resolved::Dir(c)) => c,
            Some(Resolved::File(_)) => {
                return Err(Error::InvalidArgument(format!(
                    "hfs: {path:?} is not a directory"
                )));
            }
            None => {
                return Err(Error::InvalidArgument(format!(
                    "hfs: no such path {path:?}"
                )));
            }
        };
        let mut out = Vec::new();
        if let Some(kids) = self.children.get(&cnid) {
            for n in kids {
                out.push(DirEntry {
                    name: n.name.clone(),
                    inode: n.cnid,
                    kind: if n.is_dir {
                        EntryKind::Dir
                    } else {
                        EntryKind::Regular
                    },
                    size: n.size,
                });
            }
        }
        Ok(out)
    }

    pub fn open_file_reader<'a>(
        &self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<HfsFileReader<'a>> {
        let (cnid, size, inline) = match self.resolve(path) {
            Some(Resolved::File(n)) => (n.cnid, n.size, n.inline),
            Some(Resolved::Dir(_)) => {
                return Err(Error::InvalidArgument(format!(
                    "hfs: {path:?} is a directory"
                )));
            }
            None => {
                return Err(Error::InvalidArgument(format!(
                    "hfs: no such file {path:?}"
                )));
            }
        };
        let dev_len = dev.total_size();
        let extents = self.full_extents(cnid, 0, &inline, size);
        let ranges = self.fork_ranges(&extents, size, dev_len)?;
        Ok(HfsFileReader {
            dev,
            ranges,
            total: size,
            pos: 0,
        })
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(Error::Unsupported(
            "hfs: creating classic HFS volumes is not supported".into(),
        ))
    }
}

enum Resolved<'a> {
    Dir(u32),
    File(&'a Node),
}

/// Streaming reader over a file's data fork (extent byte ranges).
pub struct HfsFileReader<'a> {
    dev: &'a mut dyn BlockDevice,
    ranges: Vec<(u64, u64)>,
    total: u64,
    pos: u64,
}

impl HfsFileReader<'_> {
    /// Map a logical position to `(device offset, bytes available in extent)`.
    fn locate(&self, pos: u64) -> Option<(u64, u64)> {
        let mut walked = 0u64;
        for &(off, len) in &self.ranges {
            if pos < walked + len {
                let within = pos - walked;
                return Some((off + within, len - within));
            }
            walked += len;
        }
        None
    }
}

impl Read for HfsFileReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.total || buf.is_empty() {
            return Ok(0);
        }
        let (off, avail) = match self.locate(self.pos) {
            Some(v) => v,
            None => return Ok(0),
        };
        let want = (buf.len() as u64).min(avail).min(self.total - self.pos) as usize;
        self.dev
            .read_at(off, &mut buf[..want])
            .map_err(io::Error::other)?;
        self.pos += want as u64;
        Ok(want)
    }
}

impl Seek for HfsFileReader<'_> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(d) => self.total as i128 + d as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
        };
        if new < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "hfs: seek before start",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

impl crate::fs::FileReadHandle for HfsFileReader<'_> {
    fn len(&self) -> u64 {
        self.total
    }
}

impl crate::fs::FilesystemFactory for Hfs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl Filesystem for Hfs {
    fn create_file(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &Path,
        _src: crate::fs::FileSource,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(Error::Immutable {
            kind: "hfs",
            op: "add",
        })
    }

    fn create_dir(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &Path,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(Error::Immutable {
            kind: "hfs",
            op: "mkdir",
        })
    }

    fn create_symlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &Path,
        _target: &Path,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(Error::Immutable {
            kind: "hfs",
            op: "symlink",
        })
    }

    fn create_device(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &Path,
        _kind: crate::fs::DeviceKind,
        _major: u32,
        _minor: u32,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(Error::Immutable {
            kind: "hfs",
            op: "mknod",
        })
    }

    fn remove(&mut self, _dev: &mut dyn BlockDevice, _path: &Path) -> Result<()> {
        Err(Error::Immutable {
            kind: "hfs",
            op: "rm",
        })
    }

    fn list(&mut self, _dev: &mut dyn BlockDevice, path: &Path) -> Result<Vec<DirEntry>> {
        let s = path
            .to_str()
            .ok_or_else(|| Error::InvalidArgument("hfs: non-UTF-8 path".into()))?;
        self.list_path(s)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &Path,
    ) -> Result<Box<dyn Read + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| Error::InvalidArgument("hfs: non-UTF-8 path".into()))?;
        Ok(Box::new(self.open_file_reader(dev, s)?))
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &Path,
    ) -> Result<Box<dyn crate::fs::FileReadHandle + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| Error::InvalidArgument("hfs: non-UTF-8 path".into()))?;
        Ok(Box::new(self.open_file_reader(dev, s)?))
    }

    fn getattr(&mut self, _dev: &mut dyn BlockDevice, path: &Path) -> Result<FileAttrs> {
        let s = path
            .to_str()
            .ok_or_else(|| Error::InvalidArgument("hfs: non-UTF-8 path".into()))?;
        let (kind, size, mtime, inode) = match self.resolve(s) {
            Some(Resolved::Dir(cnid)) => (EntryKind::Dir, 0u64, 0u32, cnid),
            Some(Resolved::File(n)) => (EntryKind::Regular, n.size, n.mtime, n.cnid),
            None => return Err(Error::InvalidArgument(format!("hfs: no such path {s:?}"))),
        };
        let mtime = mtime.saturating_sub(MAC_EPOCH_DELTA);
        Ok(FileAttrs {
            kind,
            mode: if kind == EntryKind::Dir { 0o755 } else { 0o644 },
            uid: 0,
            gid: 0,
            size,
            blocks: size.div_ceil(512),
            nlink: if kind == EntryKind::Dir { 2 } else { 1 },
            atime: mtime,
            mtime,
            ctime: mtime,
            rdev: 0,
            inode,
        })
    }

    fn flush(&mut self, _dev: &mut dyn BlockDevice) -> Result<()> {
        Ok(())
    }

    fn mutation_capability(&self) -> MutationCapability {
        MutationCapability::Immutable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    const HELLO: &[u8] = b"Hello from classic HFS!\n";
    const DEEP: &[u8] = b"Deep file in sub.\n";

    fn put_u16(v: &mut [u8], o: usize, x: u16) {
        v[o..o + 2].copy_from_slice(&x.to_be_bytes());
    }
    fn put_u32(v: &mut [u8], o: usize, x: u32) {
        v[o..o + 4].copy_from_slice(&x.to_be_bytes());
    }
    fn put_ext(v: &mut [u8], o: usize, e: &[(u16, u16)]) {
        for (i, &(s, c)) in e.iter().enumerate() {
            put_u16(v, o + i * 4, s);
            put_u16(v, o + i * 4 + 2, c);
        }
    }

    fn file_record(cnid: u32, size: u32, start_block: u16) -> Vec<u8> {
        let mut d = vec![0u8; 102];
        d[0] = 2; // cdrType = file (cdrResrv2 @1)
        d[4..8].copy_from_slice(b"TEXT"); // FInfo fdType
        d[8..12].copy_from_slice(b"ttxt"); // FInfo fdCreator
        d[20..24].copy_from_slice(&cnid.to_be_bytes()); // filFlNum
        d[26..30].copy_from_slice(&size.to_be_bytes()); // filLgLen (data fork)
        d[30..34].copy_from_slice(&512u32.to_be_bytes()); // filPyLen (1 block)
        put_ext(&mut d, 74, &[(start_block, 1), (0, 0), (0, 0)]); // filExtRec
        d
    }
    fn dir_record(cnid: u32) -> Vec<u8> {
        let mut d = vec![0u8; 70];
        d[0] = 1; // cdrType = directory
        d[6..10].copy_from_slice(&cnid.to_be_bytes()); // dirDirID
        d
    }
    fn leaf_record(parent: u32, name: &[u8], data: &[u8]) -> Vec<u8> {
        let key_len = 6 + name.len();
        let mut r = vec![0u8; 1 + key_len];
        r[0] = key_len as u8;
        r[2..6].copy_from_slice(&parent.to_be_bytes());
        r[6] = name.len() as u8;
        r[7..7 + name.len()].copy_from_slice(name);
        r.resize(round_up_even(1 + key_len), 0); // pad key to even
        r.extend_from_slice(data);
        r
    }

    /// Build a minimal but genuine classic HFS volume: a 2-node catalog B-tree
    /// (header + one leaf) over `/hello.txt`, `/sub/`, `/sub/deep.txt`, with
    /// 512-byte allocation blocks. Layout (512-byte sectors): boot(0,1),
    /// MDB(2), bitmap(3), then alloc blocks from sector 4 — catalog(0,1),
    /// extents(2), hello data(3), deep data(4).
    fn build_volume() -> Vec<u8> {
        let block = 512usize;
        let al_bl_st = 4u16;
        let num_alloc_blocks = 5usize;
        let mut v = vec![0u8; (al_bl_st as usize + num_alloc_blocks + 1) * block];

        // MDB at 1024.
        let mdb = 1024;
        v[mdb..mdb + 2].copy_from_slice(b"BD");
        put_u32(&mut v, mdb + 20, block as u32); // drAlBlkSiz
        put_u16(&mut v, mdb + 28, al_bl_st); // drAlBlSt
        v[mdb + 36] = 7;
        v[mdb + 37..mdb + 44].copy_from_slice(b"TestVol"); // drVN
        put_u32(&mut v, mdb + 130, block as u32); // drXTFlSize
        put_ext(&mut v, mdb + 134, &[(2, 1), (0, 0), (0, 0)]); // drXTExtRec
        put_u32(&mut v, mdb + 146, (2 * block) as u32); // drCTFlSize
        put_ext(&mut v, mdb + 150, &[(0, 2), (0, 0), (0, 0)]); // drCTExtRec

        // Extents-overflow file (alloc block 2): a header node, no leaves.
        let xt = (al_bl_st as usize + 2) * block;
        v[xt + 8] = 1; // ndType = header
        put_u16(&mut v, xt + 32, 512); // bthNodeSize
        put_u32(&mut v, xt + 36, 1); // bthNNodes

        // Catalog file (alloc block 0): header node + one leaf node.
        let cat = al_bl_st as usize * block;
        v[cat + 8] = 1; // header node
        put_u32(&mut v, cat + 24, 1); // bthFNode = node 1
        put_u16(&mut v, cat + 32, 512); // bthNodeSize
        put_u32(&mut v, cat + 36, 2); // bthNNodes
        let leaf = cat + 512;
        v[leaf + 8] = 0xFF; // leaf node
        v[leaf + 9] = 1; // height
        let recs = [
            leaf_record(2, b"hello.txt", &file_record(17, HELLO.len() as u32, 3)),
            leaf_record(2, b"sub", &dir_record(16)),
            leaf_record(16, b"deep.txt", &file_record(18, DEEP.len() as u32, 4)),
            // A classic-Mac name with a `/` in it (path separator is `:` on
            // HFS). Reuses hello.txt's data block. Must surface as `A:B`.
            leaf_record(2, b"A/B", &file_record(19, HELLO.len() as u32, 3)),
        ];
        let mut pos = 14usize;
        let mut offs = vec![14u16];
        for r in &recs {
            v[leaf + pos..leaf + pos + r.len()].copy_from_slice(r);
            pos += r.len();
            offs.push(pos as u16);
        }
        put_u16(&mut v, leaf + 10, recs.len() as u16); // ndNRecs
        for (i, &o) in offs.iter().enumerate() {
            put_u16(&mut v, leaf + 512 - 2 * (i + 1), o);
        }

        // File data forks.
        let hd = (al_bl_st as usize + 3) * block;
        v[hd..hd + HELLO.len()].copy_from_slice(HELLO);
        let dd = (al_bl_st as usize + 4) * block;
        v[dd..dd + DEEP.len()].copy_from_slice(DEEP);

        v
    }

    fn dev_from(bytes: &[u8]) -> MemoryBackend {
        let mut dev = MemoryBackend::new(bytes.len() as u64);
        dev.write_at(0, bytes).unwrap();
        dev
    }

    fn read_all(fs: &mut Hfs, dev: &mut dyn BlockDevice, path: &str) -> Vec<u8> {
        let mut r = fs.read_file(dev, Path::new(path)).unwrap();
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut out).unwrap();
        out
    }

    /// Parse the synthetic volume directly: tree + nested dir + file extraction.
    #[test]
    fn synthetic_volume_round_trip() {
        let vol = build_volume();
        let mut dev = dev_from(&vol);
        let mut fs = Hfs::open(&mut dev).unwrap();
        assert_eq!(fs.volume_name, "TestVol");

        let root: Vec<_> = fs
            .list(&mut dev, Path::new("/"))
            .unwrap()
            .into_iter()
            .map(|e| (e.name, e.kind))
            .collect();
        assert!(root.contains(&("hello.txt".into(), EntryKind::Regular)));
        assert!(root.contains(&("sub".into(), EntryKind::Dir)));
        // A classic-Mac name with a literal `/` ("A/B") surfaces canonically as
        // "A:B" — the `/` (legal on HFS, whose separator is `:`) is swapped so
        // it can't be mistaken for a path separator.
        assert!(
            root.contains(&("A:B".into(), EntryKind::Regular)),
            "expected canonical 'A:B' in root: {root:?}"
        );

        let sub: Vec<_> = fs
            .list(&mut dev, Path::new("/sub"))
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(sub, vec!["deep.txt"]);

        assert_eq!(read_all(&mut fs, &mut dev, "/hello.txt"), HELLO);
        assert_eq!(read_all(&mut fs, &mut dev, "/sub/deep.txt"), DEEP);
        // Resolving the canonical "A:B" reaches the entry (its data fork
        // aliases hello.txt's block, so the bytes match HELLO).
        assert_eq!(read_all(&mut fs, &mut dev, "/A:B"), HELLO);
    }

    /// The same volume wrapped in a DiskCopy 4.2 header is detected + read
    /// transparently through the full `AnyFs` pipeline (Part A + Part B).
    #[test]
    fn diskcopy_wrapped_hfs_via_anyfs() {
        let vol = build_volume();
        let mut img = vec![0u8; 0x54];
        img[0x40..0x44].copy_from_slice(&(vol.len() as u32).to_be_bytes()); // data size
        img[0x50] = 3; // 1440k encoding
        img[0x52] = 0x01; // magic 0x0100
        img.extend_from_slice(&vol);

        let mem = dev_from(&img);
        let mut dc = crate::block::DiskCopy42Backend::new(Box::new(mem)).unwrap();
        let mut fs = crate::inspect::AnyFs::open(&mut dc).unwrap();
        assert_eq!(fs.kind_string(), "hfs");
        let names: Vec<_> = fs
            .list(&mut dc, "/")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(names.iter().any(|n| n == "hello.txt"));

        let mut out = Vec::new();
        fs.copy_file_to(&mut dc, "/sub/deep.txt", &mut out).unwrap();
        assert_eq!(out, DEEP);
    }
}
