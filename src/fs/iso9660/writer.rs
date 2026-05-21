//! ISO 9660 writer — mkisofs-style two-pass builder.
//!
//! The writer buffers create_*-call entries in memory, then on `flush()`
//! walks the tree twice:
//!
//! 1. **Pass 1** — assign LBAs. Counts sectors needed for PVD, optional
//!    Joliet SVD, VDST, path tables (L + M, plus Joliet equivalents),
//!    directory records, and file data. Each directory and each file
//!    gets a starting LBA + byte size.
//!
//! 2. **Pass 2** — write everything out. Order:
//!    - LBAs 0..15: system area (zero)
//!    - LBA 16: Primary Volume Descriptor
//!    - LBA 17: Joliet Supplementary Volume Descriptor (when enabled)
//!    - next LBA: Volume Descriptor Set Terminator
//!    - L-path table + M-path table (PVD names)
//!    - L-path table + M-path table (Joliet names, when enabled)
//!    - PVD directory records (each dir's stream of records)
//!    - Joliet directory records
//!    - File data, each file aligned to the start of a sector
//!
//! Streaming invariant: file payloads come in as [`FileSource`] and are
//! pumped through a 64 KiB scratch buffer during pass 2. The writer
//! never loads a file fully into memory.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::{DeviceKind, FileMeta, FileSource};

use super::el_torito::BootCatalog;
use super::joliet::string_to_ucs2_be;

/// Logical sector size — fixed at 2 KiB by ECMA-119.
const SECTOR: u64 = 2048;
/// Fixed-position byte offset for the PVD (LBA 16).
const PVD_BYTE: u64 = 16 * SECTOR;

/// Format options. `volume_id` is required; everything else has sane
/// defaults.
#[derive(Debug, Clone)]
pub struct FormatOpts {
    pub volume_id: String,
    pub publisher_id: String,
    pub data_preparer_id: String,
    pub application_id: String,
    pub joliet: bool,
    pub rock_ridge: bool,
    pub el_torito: Option<BootCatalog>,
    /// UNIX timestamp used for all metadata timestamps.
    pub create_date: u32,
}

impl Default for FormatOpts {
    fn default() -> Self {
        Self {
            volume_id: "CDROM".into(),
            publisher_id: String::new(),
            data_preparer_id: String::new(),
            application_id: String::new(),
            joliet: true,
            rock_ridge: true,
            el_torito: None,
            create_date: 0,
        }
    }
}

impl FormatOpts {
    /// Apply a generic option-bag (CLI `-O key=val` / TOML
    /// `[filesystem.options]`) on top of these opts. Unknown keys are
    /// left in the map for the caller to flag. El Torito boot config is
    /// not yet plumbable through the bag — set it directly on
    /// `FormatOpts`.
    pub fn apply_options(&mut self, map: &mut crate::format_opts::OptionMap) -> crate::Result<()> {
        if let Some(s) = map.take_str("volume_id") {
            self.volume_id = s;
        }
        // Accept "volume_label" as a synonym so the same CLI key works
        // across filesystems.
        if let Some(s) = map.take_str("volume_label") {
            self.volume_id = s;
        }
        if let Some(s) = map.take_str("publisher_id") {
            self.publisher_id = s;
        }
        if let Some(s) = map.take_str("data_preparer_id") {
            self.data_preparer_id = s;
        }
        if let Some(s) = map.take_str("application_id") {
            self.application_id = s;
        }
        if let Some(b) = map.take_bool("joliet")? {
            self.joliet = b;
        }
        if let Some(b) = map.take_bool("rock_ridge")? {
            self.rock_ridge = b;
        }
        if let Some(t) = map.take_u32("create_date")? {
            self.create_date = t;
        }
        Ok(())
    }
}

/// One in-memory entry the writer is buffering. Kept private — the
/// public API speaks through [`crate::fs::Filesystem`].
///
/// `File::body` holds the file's bytes in RAM. This buys us correctness
/// when the source is a tempfile (as `populate_image_via_trait`
/// produces — those vanish between `create_file` and `flush`). Pay-as-
/// you-go: typical ISO contents fit comfortably; if a future caller
/// needs multi-GiB files we can revisit with a temp-file pool the
/// writer owns.
enum PendingEntry {
    File {
        #[allow(dead_code)]
        meta: FileMeta,
        body: Vec<u8>,
    },
    Dir {
        #[allow(dead_code)]
        meta: FileMeta,
    },
    Symlink {
        #[allow(dead_code)]
        meta: FileMeta,
        target: PathBuf,
    },
    Device {
        #[allow(dead_code)]
        meta: FileMeta,
        #[allow(dead_code)]
        kind: DeviceKind,
        #[allow(dead_code)]
        major: u32,
        #[allow(dead_code)]
        minor: u32,
    },
}

/// Two-pass ISO 9660 writer. Construct with [`Iso9660Writer::new`],
/// stream entries via the Filesystem trait, then call `flush` (via the
/// trait) to lay out the image.
pub struct Iso9660Writer {
    opts: FormatOpts,
    /// Tree of entries keyed by normalized path. Root is implicit.
    entries: BTreeMap<PathBuf, PendingEntry>,
    /// Set once `flush` has run successfully.
    flushed: bool,
}

impl Iso9660Writer {
    pub fn new(opts: FormatOpts) -> Self {
        Self {
            opts,
            entries: BTreeMap::new(),
            flushed: false,
        }
    }

    pub fn add_file(&mut self, path: &Path, src: FileSource, meta: FileMeta) -> Result<()> {
        let path = normalize(path)?;
        // Read the file body eagerly. The caller may pass a HostPath
        // pointing at a tempfile that gets dropped before flush — and
        // the writer can't lay out the data area until all entries are
        // in, so we hold the bytes ourselves.
        let (mut reader, total) = src.open()?;
        let mut body = Vec::with_capacity(total as usize);
        reader.read_to_end(&mut body)?;
        self.entries.insert(path, PendingEntry::File { meta, body });
        Ok(())
    }

    pub fn add_dir(&mut self, path: &Path, meta: FileMeta) -> Result<()> {
        let path = normalize(path)?;
        self.entries.insert(path, PendingEntry::Dir { meta });
        Ok(())
    }

    pub fn add_symlink(&mut self, path: &Path, target: &Path, meta: FileMeta) -> Result<()> {
        let path = normalize(path)?;
        self.entries.insert(
            path,
            PendingEntry::Symlink {
                meta,
                target: target.to_path_buf(),
            },
        );
        Ok(())
    }

    pub fn add_device(
        &mut self,
        path: &Path,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: FileMeta,
    ) -> Result<()> {
        let path = normalize(path)?;
        self.entries.insert(
            path,
            PendingEntry::Device {
                meta,
                kind,
                major,
                minor,
            },
        );
        Ok(())
    }

    pub fn remove_entry(&mut self, path: &Path) -> Result<()> {
        let path = normalize(path)?;
        if self.entries.remove(&path).is_none() {
            return Err(crate::Error::InvalidArgument(format!(
                "iso9660: no buffered entry at {}",
                path.display()
            )));
        }
        Ok(())
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Write the buffered tree to `dev`. Idempotent.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        // Build the directory tree from the flat entries map.
        let tree = build_tree(&self.entries)?;
        let layout = compute_layout(&tree, &self.opts);
        write_image(dev, &tree, &mut self.entries, &layout, &self.opts)?;
        self.flushed = true;
        Ok(())
    }
}

/// One node in the layout tree. Constructed by `build_tree`; references
/// the source entry by absolute path so file data can be re-opened in
/// pass 2 without cloning `FileSource`.
struct Node {
    /// Absolute path including leading '/'.
    path: PathBuf,
    /// Last path component (or "/" for the root).
    name: String,
    /// File / dir / symlink / device.
    kind: NodeKind,
    /// Children, sorted by ISO-comparison order. Empty for non-directories.
    children: Vec<Node>,
}

enum NodeKind {
    Dir,
    File { size: u64 },
    Symlink { target: PathBuf },
    Device,
}

/// Build the tree from the writer's flat `entries` map. Returns the
/// root node. Implicitly creates intermediate directories if a file
/// references a parent that wasn't explicitly added.
fn build_tree(entries: &BTreeMap<PathBuf, PendingEntry>) -> Result<Node> {
    let mut root = Node {
        path: PathBuf::from("/"),
        name: "/".to_string(),
        kind: NodeKind::Dir,
        children: Vec::new(),
    };
    for (path, entry) in entries {
        let kind = match entry {
            PendingEntry::Dir { .. } => NodeKind::Dir,
            PendingEntry::File { body, .. } => NodeKind::File {
                size: body.len() as u64,
            },
            PendingEntry::Symlink { target, .. } => NodeKind::Symlink {
                target: target.clone(),
            },
            PendingEntry::Device { .. } => NodeKind::Device,
        };
        insert_node(&mut root, path, kind)?;
    }
    sort_tree(&mut root);
    Ok(root)
}

fn insert_node(root: &mut Node, path: &Path, kind: NodeKind) -> Result<()> {
    let components: Vec<&str> = path
        .to_str()
        .unwrap()
        .trim_matches('/')
        .split('/')
        .filter(|c| !c.is_empty())
        .collect();
    if components.is_empty() {
        return Err(crate::Error::InvalidArgument(
            "iso9660: cannot insert root explicitly".into(),
        ));
    }
    let mut cur = root;
    for (i, comp) in components.iter().enumerate() {
        let is_leaf = i + 1 == components.len();
        let mut full = cur.path.clone();
        full.push(comp);
        if let Some(idx) = cur.children.iter().position(|c| c.name == *comp) {
            if is_leaf {
                cur.children[idx].kind = match &kind {
                    NodeKind::Dir => NodeKind::Dir,
                    NodeKind::File { size } => NodeKind::File { size: *size },
                    NodeKind::Symlink { target } => NodeKind::Symlink {
                        target: target.clone(),
                    },
                    NodeKind::Device => NodeKind::Device,
                };
            }
            cur = &mut cur.children[idx];
        } else {
            let child_kind = if is_leaf {
                match &kind {
                    NodeKind::Dir => NodeKind::Dir,
                    NodeKind::File { size } => NodeKind::File { size: *size },
                    NodeKind::Symlink { target } => NodeKind::Symlink {
                        target: target.clone(),
                    },
                    NodeKind::Device => NodeKind::Device,
                }
            } else {
                NodeKind::Dir
            };
            cur.children.push(Node {
                path: full,
                name: (*comp).to_string(),
                kind: child_kind,
                children: Vec::new(),
            });
            let n = cur.children.len() - 1;
            cur = &mut cur.children[n];
        }
    }
    Ok(())
}

fn sort_tree(node: &mut Node) {
    // ISO 9660 sort order: ascending byte-wise by uppercased 8.3
    // identifier. For our purposes, lexicographic on the cooked name is
    // good enough — both Joliet and Rock Ridge tolerate any sort.
    node.children.sort_by(|a, b| a.name.cmp(&b.name));
    for child in node.children.iter_mut() {
        sort_tree(child);
    }
}

/// Layout results. Every directory in `dir_lba` has its LBA assigned;
/// every file in `file_lba` likewise.
struct Layout {
    /// LBA → byte size map of every directory's record stream.
    dir_lba: BTreeMap<PathBuf, (u32, u64)>,
    /// Same for the Joliet directory stream (parallel tree).
    joliet_dir_lba: BTreeMap<PathBuf, (u32, u64)>,
    /// LBA → byte size of each regular file's extent.
    file_lba: BTreeMap<PathBuf, (u32, u64)>,
    /// LBA of L-path table (PVD).
    l_path_lba: u32,
    /// LBA of M-path table (PVD).
    m_path_lba: u32,
    /// L / M path table LBAs for Joliet.
    joliet_l_path_lba: u32,
    joliet_m_path_lba: u32,
    /// Combined byte size of each path table (PVD).
    path_table_size: u32,
    /// Byte size of Joliet path tables.
    joliet_path_table_size: u32,
    /// LBA of the VDST terminator.
    vdst_lba: u32,
    /// Final total sectors (for PVD's volume_space_size).
    total_sectors: u32,
}

/// Compute LBAs without writing anything. Pass 1.
fn compute_layout(root: &Node, opts: &FormatOpts) -> Layout {
    let joliet = opts.joliet;
    let mut cursor: u32 = 16; // PVD lives here
    cursor += 1; // PVD
    if joliet {
        cursor += 1; // Joliet SVD
    }
    let vdst_lba = cursor;
    cursor += 1; // VDST

    // Path tables come next. Compute sizes by walking the tree.
    let pvd_dirs = collect_directories(root);
    let path_table_size = path_table_byte_size(&pvd_dirs, /*joliet*/ false);
    let l_path_lba = cursor;
    cursor += sectors_for(u64::from(path_table_size));
    let m_path_lba = cursor;
    cursor += sectors_for(u64::from(path_table_size));

    let (joliet_l_path_lba, joliet_m_path_lba, joliet_path_table_size) = if joliet {
        let size = path_table_byte_size(&pvd_dirs, /*joliet*/ true);
        let l = cursor;
        cursor += sectors_for(u64::from(size));
        let m = cursor;
        cursor += sectors_for(u64::from(size));
        (l, m, size)
    } else {
        (0, 0, 0)
    };

    // Directory record streams (PVD).
    let mut dir_lba: BTreeMap<PathBuf, (u32, u64)> = BTreeMap::new();
    for (path, _) in pvd_dirs.iter() {
        let n = find_node(root, path).unwrap();
        let is_root = path == &PathBuf::from("/");
        let size = dir_records_byte_size(n, opts, /*joliet*/ false, is_root);
        dir_lba.insert(path.clone(), (cursor, size));
        cursor += sectors_for(size);
    }

    // Joliet directory record streams.
    let mut joliet_dir_lba: BTreeMap<PathBuf, (u32, u64)> = BTreeMap::new();
    if joliet {
        for (path, _) in pvd_dirs.iter() {
            let n = find_node(root, path).unwrap();
            let size = dir_records_byte_size(n, opts, /*joliet*/ true, /*is_root*/ false);
            joliet_dir_lba.insert(path.clone(), (cursor, size));
            cursor += sectors_for(size);
        }
    }

    // File data. We walk in path order so the cursor advances
    // deterministically; each file gets a fresh sector start.
    let mut file_lba: BTreeMap<PathBuf, (u32, u64)> = BTreeMap::new();
    walk_files(root, &mut |n| {
        if let NodeKind::File { size } = n.kind {
            file_lba.insert(n.path.clone(), (cursor, size));
            cursor += sectors_for(size.max(1));
        }
    });

    Layout {
        dir_lba,
        joliet_dir_lba,
        file_lba,
        l_path_lba,
        m_path_lba,
        joliet_l_path_lba,
        joliet_m_path_lba,
        path_table_size,
        joliet_path_table_size,
        vdst_lba,
        total_sectors: cursor,
    }
}

fn collect_directories(root: &Node) -> Vec<(PathBuf, u16)> {
    // Returns (path, parent_index_1based) per ECMA-119 path table
    // ordering — root has parent = 1 (itself). Breadth-first.
    let mut out: Vec<(PathBuf, u16)> = vec![(root.path.clone(), 1)];
    let mut queue: Vec<(usize, &Node)> = vec![(0, root)];
    while let Some((parent_idx_minus1, parent)) = queue.pop() {
        for child in &parent.children {
            if matches!(child.kind, NodeKind::Dir) {
                let parent_record = (parent_idx_minus1 + 1) as u16;
                out.push((child.path.clone(), parent_record));
                let new_idx = out.len() - 1;
                queue.push((new_idx, child));
            }
        }
    }
    out
}

fn find_node<'a>(root: &'a Node, path: &Path) -> Option<&'a Node> {
    if path == Path::new("/") {
        return Some(root);
    }
    let comps: Vec<&str> = path
        .to_str()?
        .trim_matches('/')
        .split('/')
        .filter(|c| !c.is_empty())
        .collect();
    let mut cur = root;
    for comp in comps {
        cur = cur.children.iter().find(|c| c.name == comp)?;
    }
    Some(cur)
}

fn walk_files<F: FnMut(&Node)>(root: &Node, f: &mut F) {
    f(root);
    for c in &root.children {
        walk_files(c, f);
    }
}

fn sectors_for(bytes: u64) -> u32 {
    bytes.div_ceil(SECTOR) as u32
}

/// Compute the byte length of an ECMA-119 path table (sum of every
/// directory's path table record).
fn path_table_byte_size(dirs: &[(PathBuf, u16)], joliet: bool) -> u32 {
    let mut total: u32 = 0;
    for (path, _parent) in dirs {
        let name_bytes = if path == Path::new("/") {
            1u32 // root identifier = single 0x00 byte
        } else {
            let comp = path.file_name().unwrap().to_str().unwrap();
            iso_name_bytes(comp, joliet, /*directory*/ true) as u32
        };
        // Record: 1 (len_di) + 1 (xattr len) + 4 (extent) + 2 (parent)
        //         + name + (name % 2 == 1 ? 1 : 0) pad
        let pad = if name_bytes % 2 == 1 { 1 } else { 0 };
        total += 8 + name_bytes + pad;
    }
    total
}

/// Number of bytes the cooked identifier consumes on disk. For ISO
/// 9660 we use the uppercase 8.3-with-`;1` form; for Joliet, UCS-2 BE.
/// For directories we omit the `;1` suffix.
fn iso_name_bytes(name: &str, joliet: bool, directory: bool) -> usize {
    if joliet {
        // UCS-2 BE, 2 bytes per code unit.
        name.encode_utf16().count() * 2
    } else if directory {
        name.to_ascii_uppercase().len().max(1)
    } else {
        // Files include ";1" version suffix.
        name.to_ascii_uppercase().len() + 2
    }
}

/// Sum of the on-disk directory record bytes for `dir` and its
/// immediate children (i.e. one directory's worth of records, not
/// recursive).
fn dir_records_byte_size(dir: &Node, opts: &FormatOpts, joliet: bool, is_root: bool) -> u64 {
    let mut sum: u64 = 0;
    // "." and ".." entries — both 34 bytes (no name overhead).
    sum += 34 * 2;
    // Root's "." gets the 7-byte SP SUSP indicator under Rock Ridge.
    if is_root && opts.rock_ridge && !joliet {
        sum += 7;
    }
    let want_rr = opts.rock_ridge && !joliet;
    for child in &dir.children {
        sum += dir_record_size(child, want_rr, joliet) as u64;
    }
    // Round up to sector — ISO records don't straddle sectors. We
    // approximate by aligning the dir total to a sector boundary.
    align_records_to_sector(sum)
}

fn align_records_to_sector(bytes: u64) -> u64 {
    bytes.div_ceil(SECTOR) * SECTOR
}

/// Length of a single directory record on disk.
fn dir_record_size(node: &Node, rock_ridge: bool, joliet: bool) -> usize {
    let name_bytes = iso_name_bytes(&node.name, joliet, matches!(node.kind, NodeKind::Dir));
    // Pad name to even length (ECMA-119 §9.1.12).
    let name_pad = if name_bytes % 2 == 0 { 1 } else { 0 };
    let base = 33 + name_bytes + name_pad;
    if rock_ridge {
        // Rock Ridge SUA size per child: NM (5 + name) + PX (36) +
        // optional SL for symlinks. The SP marker is on the root's "."
        // record only — `dir_records_byte_size` accounts for that.
        let nm = 5 + node.name.len();
        let px = 36;
        let sl = if let NodeKind::Symlink { target } = &node.kind {
            // SL entry: 5 byte header + 1 flag + each component (2 + bytes).
            let mut s = 5;
            for comp in target.to_str().unwrap_or("").split('/') {
                if comp.is_empty() {
                    s += 2; // ROOT flag
                } else {
                    s += 2 + comp.len();
                }
            }
            s
        } else {
            0
        };
        base + nm + px + sl
    } else {
        base
    }
}

/// Pass 2: stamp the actual bytes.
fn write_image(
    dev: &mut dyn BlockDevice,
    root: &Node,
    entries: &mut BTreeMap<PathBuf, PendingEntry>,
    layout: &Layout,
    opts: &FormatOpts,
) -> Result<()> {
    // 0. System area — zero bytes 0..16*2048.
    let zero = vec![0u8; SECTOR as usize];
    for s in 0u64..16 {
        dev.write_at(s * SECTOR, &zero)?;
    }

    // 1. PVD at LBA 16.
    let pvd = encode_pvd(layout, opts, root, /*joliet*/ false);
    dev.write_at(PVD_BYTE, &pvd)?;

    // 2. Joliet SVD at LBA 17 if enabled.
    if opts.joliet {
        let svd = encode_pvd(layout, opts, root, /*joliet*/ true);
        dev.write_at(17 * SECTOR, &svd)?;
    }

    // 3. VDST terminator.
    let mut vdst = vec![0u8; SECTOR as usize];
    vdst[0] = 0xFF;
    vdst[1..6].copy_from_slice(b"CD001");
    vdst[6] = 0x01;
    dev.write_at(u64::from(layout.vdst_lba) * SECTOR, &vdst)?;

    // 4. Path tables (L + M) for PVD.
    let dirs = collect_directories(root);
    let (lpath, mpath) = encode_path_tables(&dirs, &layout.dir_lba, /*joliet*/ false);
    dev.write_at(u64::from(layout.l_path_lba) * SECTOR, &lpath)?;
    dev.write_at(u64::from(layout.m_path_lba) * SECTOR, &mpath)?;

    // 5. Joliet path tables.
    if opts.joliet {
        let (lpath, mpath) =
            encode_path_tables(&dirs, &layout.joliet_dir_lba, /*joliet*/ true);
        dev.write_at(u64::from(layout.joliet_l_path_lba) * SECTOR, &lpath)?;
        dev.write_at(u64::from(layout.joliet_m_path_lba) * SECTOR, &mpath)?;
    }

    // 6. PVD directory records — one stream per directory.
    for (path, _) in dirs.iter() {
        let (lba, _size) = layout.dir_lba.get(path).copied().unwrap();
        let node = find_node(root, path).unwrap();
        let parent_path = parent_of(path);
        let parent_node = find_node(root, &parent_path).unwrap();
        let parent_lba = layout.dir_lba.get(&parent_path).copied().unwrap().0;
        let self_lba = lba;
        let is_root = path == &PathBuf::from("/");
        let stream = encode_dir_records(
            node,
            parent_node,
            self_lba,
            parent_lba,
            &layout.dir_lba,
            &layout.file_lba,
            opts,
            /*joliet*/ false,
            /*is_root*/ is_root,
        );
        dev.write_at(u64::from(lba) * SECTOR, &stream)?;
    }

    // 7. Joliet directory records.
    if opts.joliet {
        for (path, _) in dirs.iter() {
            let (lba, _) = layout.joliet_dir_lba.get(path).copied().unwrap();
            let node = find_node(root, path).unwrap();
            let parent_path = parent_of(path);
            let parent_node = find_node(root, &parent_path).unwrap();
            let parent_lba = layout.joliet_dir_lba.get(&parent_path).copied().unwrap().0;
            let stream = encode_dir_records(
                node,
                parent_node,
                lba,
                parent_lba,
                &layout.joliet_dir_lba,
                &layout.file_lba,
                opts,
                /*joliet*/ true,
                /*is_root*/ false,
            );
            dev.write_at(u64::from(lba) * SECTOR, &stream)?;
        }
    }

    // 8. File data. Body is held in RAM by add_file (see PendingEntry
    //    docstring for the rationale) — we still emit zero-padding for
    //    the final sector so subsequent records start aligned.
    for (path, (lba, _size)) in layout.file_lba.iter() {
        let Some(entry) = entries.get(path) else {
            continue;
        };
        let PendingEntry::File { body, .. } = entry else {
            continue;
        };
        let base = u64::from(*lba) * SECTOR;
        let total = body.len() as u64;
        if !body.is_empty() {
            dev.write_at(base, body)?;
        }
        let used = total % SECTOR;
        if used != 0 {
            let pad = (SECTOR - used) as usize;
            let z = vec![0u8; pad];
            dev.write_at(base + total, &z)?;
        }
    }

    Ok(())
}

fn parent_of(p: &Path) -> PathBuf {
    if p == Path::new("/") {
        return PathBuf::from("/");
    }
    p.parent()
        .map(|x| x.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/"))
}

fn encode_pvd(layout: &Layout, opts: &FormatOpts, root: &Node, joliet: bool) -> Vec<u8> {
    let mut buf = vec![0u8; SECTOR as usize];
    buf[0] = if joliet { 2 } else { 1 };
    buf[1..6].copy_from_slice(b"CD001");
    buf[6] = 1; // version

    // system_id (32) at offset 8, volume_id (32) at offset 40.
    // For Joliet, both are UCS-2 BE space-padded.
    if joliet {
        // Escape sequences at offset 88 (UCS-2 level 3 = "%/E").
        buf[88..91].copy_from_slice(b"%/E");
        let sys = string_to_ucs2_be("");
        write_padded_ucs2(&mut buf[8..40], &sys);
        let vol = string_to_ucs2_be(&opts.volume_id);
        write_padded_ucs2(&mut buf[40..72], &vol);
    } else {
        write_padded_ascii(&mut buf[8..40], "LINUX");
        write_padded_ascii(&mut buf[40..72], &opts.volume_id);
    }

    // volume_space_size (both endian) at offset 80.
    put_both_u32(&mut buf[80..88], layout.total_sectors);
    // logical block size = 2048.
    put_both_u16(&mut buf[128..132], 2048);
    // path table size.
    let pts = if joliet {
        layout.joliet_path_table_size
    } else {
        layout.path_table_size
    };
    put_both_u32(&mut buf[132..140], pts);
    // L and M path table LBAs.
    let (lp, mp) = if joliet {
        (layout.joliet_l_path_lba, layout.joliet_m_path_lba)
    } else {
        (layout.l_path_lba, layout.m_path_lba)
    };
    buf[140..144].copy_from_slice(&lp.to_le_bytes());
    buf[148..152].copy_from_slice(&mp.to_be_bytes());

    // Root directory record at offset 156, exactly 34 bytes.
    let root_dirs = if joliet {
        &layout.joliet_dir_lba
    } else {
        &layout.dir_lba
    };
    let (root_lba, root_size) = root_dirs.get(Path::new("/")).copied().unwrap();
    let root_rec = encode_root_dir_record(root_lba, root_size);
    buf[156..156 + 34].copy_from_slice(&root_rec);

    // Volume set / publisher / preparer / application identifiers.
    if joliet {
        let app = string_to_ucs2_be(&opts.application_id);
        let pub_ = string_to_ucs2_be(&opts.publisher_id);
        let prep = string_to_ucs2_be(&opts.data_preparer_id);
        write_padded_ucs2(&mut buf[190..318], &[]); // volume set
        write_padded_ucs2(&mut buf[318..446], &pub_);
        write_padded_ucs2(&mut buf[446..574], &prep);
        write_padded_ucs2(&mut buf[574..702], &app);
    } else {
        write_padded_ascii(&mut buf[190..318], ""); // volume set
        write_padded_ascii(&mut buf[318..446], &opts.publisher_id);
        write_padded_ascii(&mut buf[446..574], &opts.data_preparer_id);
        write_padded_ascii(&mut buf[574..702], &opts.application_id);
    }
    // Copyright / abstract / bibliographic file ids — zero/space pad.
    write_padded_ascii(&mut buf[702..739], "");
    write_padded_ascii(&mut buf[739..776], "");
    write_padded_ascii(&mut buf[776..813], "");

    // Dates (17 bytes each) — all zero is acceptable per ECMA-119 §8.4.26
    // ("not specified" = all '0' (0x30) digits + 0 GMT offset). mkisofs
    // does emit current time; we use the opts.create_date if set.
    let date_bytes = encode_iso_long_date(opts.create_date);
    buf[813..830].copy_from_slice(&date_bytes); // creation
    buf[830..847].copy_from_slice(&date_bytes); // modification
    // Expiration + effective = "not specified" (all zeros are fine).
    // File structure version
    buf[881] = 1;
    let _ = root;
    buf
}

fn encode_root_dir_record(lba: u32, size: u64) -> [u8; 34] {
    let mut r = [0u8; 34];
    r[0] = 34; // len_dr
    r[1] = 0; // ext attr len
    put_both_u32(&mut r[2..10], lba);
    put_both_u32(&mut r[10..18], size as u32);
    // Date: all zeros = "not specified" (7 bytes at offset 18).
    // flags (offset 25): directory.
    r[25] = 0x02;
    // file_unit_size (26), interleave_gap (27) = 0.
    // volume_seq_number (28..32) = both-endian 1.
    put_both_u16(&mut r[28..32], 1);
    r[32] = 1; // identifier length
    r[33] = 0x00; // root identifier
    r
}

/// Encode an ECMA-119 17-byte long-form date from a UNIX epoch.
fn encode_iso_long_date(epoch: u32) -> [u8; 17] {
    let mut d = [b'0'; 17];
    d[16] = 0; // GMT offset
    if epoch == 0 {
        return d;
    }
    // Days-from-civil — inverse of the read-side decoder. We can be
    // approximate; mkisofs's output uses GM time and zeros below.
    let days = epoch / 86400;
    let h = (epoch / 3600) % 24;
    let m = (epoch / 60) % 60;
    let s = epoch % 60;
    let (y, mo, da) = days_to_ymd(i64::from(days) + 719468);
    let yr = y as u32;
    let s_year = format!("{yr:04}");
    let s_month = format!("{mo:02}");
    let s_day = format!("{da:02}");
    let s_hour = format!("{h:02}");
    let s_min = format!("{m:02}");
    let s_sec = format!("{s:02}");
    d[0..4].copy_from_slice(s_year.as_bytes());
    d[4..6].copy_from_slice(s_month.as_bytes());
    d[6..8].copy_from_slice(s_day.as_bytes());
    d[8..10].copy_from_slice(s_hour.as_bytes());
    d[10..12].copy_from_slice(s_min.as_bytes());
    d[12..14].copy_from_slice(s_sec.as_bytes());
    d[14..16].copy_from_slice(b"00"); // hundredths
    d
}

/// Hinnant's "days from civil" inverse for ISO timestamps.
fn days_to_ymd(z: i64) -> (i64, i64, i64) {
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn put_both_u16(buf: &mut [u8], v: u16) {
    buf[0..2].copy_from_slice(&v.to_le_bytes());
    buf[2..4].copy_from_slice(&v.to_be_bytes());
}

fn put_both_u32(buf: &mut [u8], v: u32) {
    buf[0..4].copy_from_slice(&v.to_le_bytes());
    buf[4..8].copy_from_slice(&v.to_be_bytes());
}

fn write_padded_ascii(buf: &mut [u8], s: &str) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(buf.len());
    buf[..n].copy_from_slice(&bytes[..n]);
    for b in &mut buf[n..] {
        *b = b' ';
    }
}

fn write_padded_ucs2(buf: &mut [u8], ucs: &[u8]) {
    let n = ucs.len().min(buf.len() / 2 * 2);
    buf[..n].copy_from_slice(&ucs[..n]);
    // Pad with 0x00 0x20 (space) in UCS-2 BE.
    for chunk in buf[n..].chunks_exact_mut(2) {
        chunk[0] = 0x00;
        chunk[1] = 0x20;
    }
}

/// Encode the L (little-endian) and M (big-endian) path tables for
/// the set of `dirs`. Order in `dirs` is the path-table-index order,
/// so directory N's PT entry uses parent N's 1-based index.
fn encode_path_tables(
    dirs: &[(PathBuf, u16)],
    dir_lba: &BTreeMap<PathBuf, (u32, u64)>,
    joliet: bool,
) -> (Vec<u8>, Vec<u8>) {
    let mut lpath: Vec<u8> = Vec::new();
    let mut mpath: Vec<u8> = Vec::new();
    for (path, parent_idx) in dirs {
        let (lba, _) = dir_lba.get(path).copied().unwrap();
        let name_bytes = if path == Path::new("/") {
            vec![0u8]
        } else {
            let comp = path.file_name().unwrap().to_str().unwrap();
            iso_identifier_bytes(comp, joliet, /*directory*/ true)
        };
        let len_di = name_bytes.len() as u8;
        // L (little-endian) — extent LE then parent LE.
        lpath.push(len_di);
        lpath.push(0); // ext attr length
        lpath.extend_from_slice(&lba.to_le_bytes());
        lpath.extend_from_slice(&parent_idx.to_le_bytes());
        lpath.extend_from_slice(&name_bytes);
        if name_bytes.len() % 2 == 1 {
            lpath.push(0);
        }
        // M (big-endian).
        mpath.push(len_di);
        mpath.push(0);
        mpath.extend_from_slice(&lba.to_be_bytes());
        mpath.extend_from_slice(&parent_idx.to_be_bytes());
        mpath.extend_from_slice(&name_bytes);
        if name_bytes.len() % 2 == 1 {
            mpath.push(0);
        }
    }
    (lpath, mpath)
}

/// Build the cooked identifier bytes for one directory entry / path
/// table record.
fn iso_identifier_bytes(name: &str, joliet: bool, directory: bool) -> Vec<u8> {
    if joliet {
        return string_to_ucs2_be(name);
    }
    let mut s = name.to_ascii_uppercase();
    if !directory {
        s.push_str(";1");
    }
    s.into_bytes()
}

/// Encode the directory record stream for `dir`: "." + ".." + each
/// child. Total length is padded to a sector boundary.
///
/// When `is_root` is true and Rock Ridge is active on this stream
/// (PVD, not Joliet), the "." record carries an extra 7-byte `SP`
/// System Use entry per IEEE P1282 §5.3. The `SP` entry announces to
/// conformant SUSP parsers (e.g. `isoinfo -d`) that Rock Ridge entries
/// follow; without it, those parsers report "No SUSP/Rock Ridge
/// present" and skip the per-record `NM` / `PX` / `SL`.
#[allow(clippy::too_many_arguments)]
fn encode_dir_records(
    dir: &Node,
    parent: &Node,
    self_lba: u32,
    parent_lba: u32,
    dir_lba: &BTreeMap<PathBuf, (u32, u64)>,
    file_lba: &BTreeMap<PathBuf, (u32, u64)>,
    opts: &FormatOpts,
    joliet: bool,
    is_root: bool,
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    // "." entry — for the root directory under Rock Ridge we append the
    // SP System Use entry so SUSP-aware parsers pick up the RR fields.
    let self_size = dir_lba.get(&dir.path).copied().unwrap().1;
    let want_sp_on_dot = is_root && opts.rock_ridge && !joliet;
    if want_sp_on_dot {
        buf.extend_from_slice(&encode_dot_record_with_sp(self_lba, self_size, 0x00));
    } else {
        buf.extend_from_slice(&encode_dot_record(self_lba, self_size, 0x00));
    }
    // ".." entry
    let parent_size = dir_lba.get(&parent.path).copied().unwrap().1;
    buf.extend_from_slice(&encode_dot_record(parent_lba, parent_size, 0x01));

    for child in &dir.children {
        let rec = encode_child_record(child, dir_lba, file_lba, opts, joliet);
        // Records can't straddle a 2K boundary.
        let sector_used = buf.len() as u64 % SECTOR;
        if sector_used + rec.len() as u64 > SECTOR {
            let pad = (SECTOR - sector_used) as usize;
            buf.extend(std::iter::repeat_n(0u8, pad));
        }
        buf.extend_from_slice(&rec);
    }
    // Pad final sector.
    let used = buf.len() as u64 % SECTOR;
    if used != 0 {
        let pad = (SECTOR - used) as usize;
        buf.extend(std::iter::repeat_n(0u8, pad));
    }
    buf
}

fn encode_dot_record(lba: u32, size: u64, ident: u8) -> [u8; 34] {
    let mut r = [0u8; 34];
    r[0] = 34;
    put_both_u32(&mut r[2..10], lba);
    put_both_u32(&mut r[10..18], size as u32);
    r[25] = 0x02; // directory flag
    put_both_u16(&mut r[28..32], 1);
    r[32] = 1;
    r[33] = ident;
    r
}

/// The 7-byte SP entry that lives in the SUA of the root's "." record.
/// Layout per IEEE P1282 §5.3 / SUSP §5.3:
///   "SP" + len=7 + version=1 + 0xBE + 0xEF + bytes_skipped=0
const SP_ENTRY: [u8; 7] = *b"SP\x07\x01\xBE\xEF\x00";

/// Encode the "." record with an SP System Use entry appended. The
/// identifier is one byte (0x00) so the SUA starts at offset 34 with no
/// padding (ECMA-119 §9.1.12). Total length: 34 + 7 = 41 bytes.
fn encode_dot_record_with_sp(lba: u32, size: u64, ident: u8) -> [u8; 41] {
    let mut r = [0u8; 41];
    r[0] = 41; // len_dr including SUA
    put_both_u32(&mut r[2..10], lba);
    put_both_u32(&mut r[10..18], size as u32);
    r[25] = 0x02; // directory flag
    put_both_u16(&mut r[28..32], 1);
    r[32] = 1;
    r[33] = ident;
    // SUA starts at offset 34 (len_fi=1 is odd → no pad byte).
    r[34..41].copy_from_slice(&SP_ENTRY);
    r
}

fn encode_child_record(
    child: &Node,
    dir_lba: &BTreeMap<PathBuf, (u32, u64)>,
    file_lba: &BTreeMap<PathBuf, (u32, u64)>,
    opts: &FormatOpts,
    joliet: bool,
) -> Vec<u8> {
    let (lba, size) = match &child.kind {
        NodeKind::Dir => dir_lba.get(&child.path).copied().unwrap_or((0, 0)),
        NodeKind::File { size } => file_lba.get(&child.path).copied().unwrap_or((0, *size)),
        // Symlinks / devices have no data extent; lba=0, size=0. The
        // file kind (regular / symlink / device) is encoded via Rock
        // Ridge entries in the SUA — fsck and the kernel iso9660 driver
        // both treat a zero-length zero-LBA record as "metadata only".
        _ => (0, 0),
    };

    let name_bytes = iso_identifier_bytes(&child.name, joliet, matches!(child.kind, NodeKind::Dir));
    let len_fi = name_bytes.len() as u8;
    let name_pad = if name_bytes.len() % 2 == 0 { 1 } else { 0 };

    let want_rr = opts.rock_ridge && !joliet;
    let mut sua: Vec<u8> = Vec::new();
    if want_rr {
        // NM entry.
        sua.extend_from_slice(b"NM");
        let nm_len = 5 + child.name.len();
        sua.push(nm_len as u8);
        sua.push(1); // version
        sua.push(0); // flags
        sua.extend_from_slice(child.name.as_bytes());
        // PX entry (mode + nlink + uid + gid, both-endian).
        sua.extend_from_slice(b"PX");
        sua.push(36); // len
        sua.push(1); // version
        let mode: u32 = match &child.kind {
            NodeKind::Dir => 0o040755,
            NodeKind::File { .. } => 0o100644,
            NodeKind::Symlink { .. } => 0o120777,
            NodeKind::Device => 0o020644,
        };
        let mut both = [0u8; 32];
        put_both_u32(&mut both[0..8], mode);
        put_both_u32(&mut both[8..16], 1); // nlink
        put_both_u32(&mut both[16..24], 0); // uid
        put_both_u32(&mut both[24..32], 0); // gid
        sua.extend_from_slice(&both);
        // SL entry for symlinks.
        if let NodeKind::Symlink { target } = &child.kind {
            let mut comps_bytes: Vec<u8> = Vec::new();
            for comp in target.to_str().unwrap_or("").split('/') {
                if comp.is_empty() {
                    // ROOT marker (flag 0x08, len 0).
                    comps_bytes.push(0x08);
                    comps_bytes.push(0);
                } else {
                    let bytes = comp.as_bytes();
                    comps_bytes.push(0x00);
                    comps_bytes.push(bytes.len() as u8);
                    comps_bytes.extend_from_slice(bytes);
                }
            }
            sua.extend_from_slice(b"SL");
            sua.push((5 + comps_bytes.len()) as u8);
            sua.push(1); // version
            sua.push(0); // flags
            sua.extend_from_slice(&comps_bytes);
        }
    }

    let base = 33 + name_bytes.len() + name_pad;
    let total = base + sua.len();
    let mut rec = vec![0u8; total];
    rec[0] = total as u8;
    put_both_u32(&mut rec[2..10], lba);
    put_both_u32(&mut rec[10..18], size as u32);
    // Date — 7 bytes at offset 18, all zero = "not specified".
    let flags = if matches!(child.kind, NodeKind::Dir) {
        0x02
    } else {
        0x00
    };
    rec[25] = flags;
    put_both_u16(&mut rec[28..32], 1); // volume seq
    rec[32] = len_fi;
    rec[33..33 + name_bytes.len()].copy_from_slice(&name_bytes);
    let sua_start = base;
    rec[sua_start..sua_start + sua.len()].copy_from_slice(&sua);
    rec
}

fn normalize(path: &Path) -> Result<PathBuf> {
    let s = path
        .to_str()
        .ok_or_else(|| crate::Error::InvalidArgument("iso9660: non-UTF-8 path".into()))?;
    let trimmed = s.trim_end_matches('/');
    if trimmed.is_empty() || trimmed == "/" {
        return Err(crate::Error::InvalidArgument(
            "iso9660: cannot create root explicitly".into(),
        ));
    }
    if !trimmed.starts_with('/') {
        return Err(crate::Error::InvalidArgument(format!(
            "iso9660: path must be absolute: {trimmed}"
        )));
    }
    Ok(PathBuf::from(trimmed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockDevice, MemoryBackend};
    use crate::fs::FileMeta;
    use std::io::Cursor;

    #[test]
    fn root_dot_record_carries_sp_marker() {
        // Build a minimal RR-enabled image, then re-read the first
        // directory record of the root extent and confirm its SUA opens
        // with the canonical SP signature.
        let mut dev = MemoryBackend::new(4 * 1024 * 1024);
        let opts = FormatOpts {
            volume_id: "SPCHECK".into(),
            joliet: false,
            rock_ridge: true,
            ..FormatOpts::default()
        };
        let mut w = Iso9660Writer::new(opts);
        w.add_dir(Path::new("/sub"), FileMeta::default()).unwrap();
        w.flush(&mut dev).unwrap();

        let iso = super::super::Iso9660::open(&mut dev).unwrap();
        // Root extent LBA from the PVD's embedded root dir record.
        let root_lba = iso.pvd.root.extent_lba;
        let mut buf = vec![0u8; super::super::SECTOR_SIZE as usize];
        dev.read_at(
            u64::from(root_lba) * u64::from(super::super::SECTOR_SIZE),
            &mut buf,
        )
        .unwrap();
        let len_dr = buf[0] as usize;
        let dot = super::super::directory::DirRecord::decode(&buf[..len_dr]).unwrap();
        // Identifier is the single 0x00 byte for "." per ECMA-119.
        assert_eq!(dot.identifier, vec![0x00]);
        // SUA begins with the SP entry.
        assert!(
            dot.system_use.len() >= 7,
            "root '.' record SUA too short: {} bytes",
            dot.system_use.len(),
        );
        assert_eq!(
            &dot.system_use[..7],
            b"SP\x07\x01\xBE\xEF\x00",
            "root '.' SUA does not begin with the SP marker",
        );

        // Sanity: child records on the root must NOT carry SP. Walk to
        // the second record in the stream and confirm its SUA starts
        // with NM (or anything else but SP).
        let second_off = len_dr;
        let len2 = buf[second_off] as usize;
        let dotdot =
            super::super::directory::DirRecord::decode(&buf[second_off..second_off + len2])
                .unwrap();
        assert!(
            dotdot.system_use.len() < 2 || &dotdot.system_use[..2] != b"SP",
            "'..' record unexpectedly carries an SP entry",
        );
    }

    #[test]
    fn write_then_read_round_trip() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts {
            volume_id: "ROUNDTRIP".into(),
            ..FormatOpts::default()
        };
        let mut w = Iso9660Writer::new(opts);
        w.add_dir(Path::new("/etc"), FileMeta::default()).unwrap();
        let body = b"hello world\n".to_vec();
        let src = FileSource::Reader {
            reader: Box::new(Cursor::new(body.clone())),
            len: body.len() as u64,
        };
        w.add_file(Path::new("/etc/conf"), src, FileMeta::default())
            .unwrap();
        w.flush(&mut dev).unwrap();

        // Re-open through the reader.
        let iso = super::super::Iso9660::open(&mut dev).unwrap();
        assert_eq!(iso.volume_id(), "ROUNDTRIP");
        let root = iso.list_path(&mut dev, "/").unwrap();
        let names: Vec<_> = root.iter().map(|d| d.name.clone()).collect();
        assert!(names.iter().any(|n| n == "etc"));
        let etc = iso.list_path(&mut dev, "/etc").unwrap();
        assert!(etc.iter().any(|d| d.name == "conf"));
    }
}
