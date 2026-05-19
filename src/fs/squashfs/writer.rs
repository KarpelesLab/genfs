//! SquashFS v4 writer.
//!
//! The writer is a two-phase builder:
//!
//! 1. **Buffering phase** — `create_dir`, `create_file`, `create_symlink`,
//!    etc. record entries in an in-memory tree. Regular file payloads
//!    live as [`crate::fs::FileSource`]s, never as `Vec<u8>` — large
//!    files are streamed on flush.
//! 2. **Flush phase** — `flush` performs the actual on-disk layout in a
//!    single pass:
//!    a. Data blocks (file payloads, then fragment block).
//!    b. Inode table metablocks.
//!    c. Directory table metablocks.
//!    d. Fragment table (one entry + L1 location array).
//!    e. Export table (1 ref per inode + L1 location array).
//!    f. Id lookup table (deduped uid/gid array + L1 location array).
//!    g. Xattr table (K/V metablocks, lookup metablocks, header).
//!    h. Superblock (last — needs every other section's location).
//!
//! On-disk addresses inside metadata are *relative* to their table's
//! base (matching the read path), except for the L1 arrays which carry
//! absolute byte offsets.
//!
//! Streaming invariant: regular files are read through a 64 KiB scratch
//! buffer; full block payloads (≤ block_size, default 128 KiB) are
//! held in memory only briefly during compression.

use std::collections::BTreeMap;
use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::DeviceKind;
use crate::fs::FileSource;
use crate::fs::squashfs::Compression;
use crate::fs::squashfs::inode::{
    INODE_BASIC_BLOCK, INODE_BASIC_CHAR, INODE_BASIC_DIR, INODE_BASIC_FIFO, INODE_BASIC_FILE,
    INODE_BASIC_SOCKET, INODE_BASIC_SYMLINK, INODE_EXT_BLOCK, INODE_EXT_CHAR, INODE_EXT_DIR,
    INODE_EXT_FIFO, INODE_EXT_FILE, INODE_EXT_SOCKET, INODE_EXT_SYMLINK,
};
use crate::fs::squashfs::metablock::{compression_to_algo, encode_metablock};
use crate::fs::squashfs::xattr::Xattr;

/// Maximum bytes per data block. We mirror the SquashFS default of 128 KiB.
pub const DEFAULT_BLOCK_SIZE: u32 = 131_072;

/// Per-entry metadata captured up front. Mirrors the public `FileMeta`
/// closely but kept private so the writer's user-facing API stays
/// minimal.
#[derive(Debug, Clone, Copy)]
pub struct EntryMeta {
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub mtime: u32,
}

impl Default for EntryMeta {
    fn default() -> Self {
        Self {
            mode: 0o644,
            uid: 0,
            gid: 0,
            mtime: 0,
        }
    }
}

/// Buffered entry kinds. Regular file payloads are kept as `FileSource`
/// to preserve the streaming invariant.
#[allow(dead_code)]
enum BuiltKind {
    /// Reserved for future use (e.g. device nodes); not currently emitted.
    Dir,
    File(FileSource),
    Symlink(String),
    /// A second directory entry pointing at an existing inode. The `String`
    /// is the *normalised* path of the source entry. We resolve it at
    /// flush() time and share its inode number + position.
    Hardlink(String),
    /// Device node (block / char / fifo / socket). Major/minor are packed
    /// using the Linux MKDEV formula at emission time.
    Device {
        kind: DeviceKind,
        major: u32,
        minor: u32,
    },
}

struct BuiltEntry {
    kind: BuiltKind,
    meta: EntryMeta,
    xattrs: Vec<Xattr>,
}

/// In-memory state held by [`super::Squashfs`] during writing.
pub struct WriteState {
    /// FS block size for data blocks.
    pub block_size: u32,
    /// Algorithm used to compress data, fragment, metablock, and lookup
    /// tables.
    pub compression: Compression,
    /// Directories created by the caller (parents are inserted implicitly).
    /// Key: absolute path with leading "/"; value: directory metadata.
    dirs: BTreeMap<String, EntryDir>,
    /// Files / symlinks indexed by their absolute path.
    files: BTreeMap<String, BuiltEntry>,
}

struct EntryDir {
    meta: EntryMeta,
    xattrs: Vec<Xattr>,
}

impl WriteState {
    pub fn new(block_size: u32, compression: Compression) -> Self {
        let mut dirs = BTreeMap::new();
        dirs.insert(
            "/".to_string(),
            EntryDir {
                meta: EntryMeta {
                    mode: 0o755,
                    ..Default::default()
                },
                xattrs: Vec::new(),
            },
        );
        Self {
            block_size,
            compression,
            dirs,
            files: BTreeMap::new(),
        }
    }

    pub fn create_dir(&mut self, path: &str, meta: EntryMeta, xattrs: Vec<Xattr>) -> Result<()> {
        let p = normalise_path(path)?;
        if p == "/" {
            // Update root metadata if caller chose to.
            if let Some(d) = self.dirs.get_mut("/") {
                d.meta = meta;
                d.xattrs = xattrs;
            }
            return Ok(());
        }
        let parent = parent_path(&p);
        self.ensure_parent(parent)?;
        self.dirs.insert(p, EntryDir { meta, xattrs });
        Ok(())
    }

    pub fn create_file(
        &mut self,
        path: &str,
        src: FileSource,
        meta: EntryMeta,
        xattrs: Vec<Xattr>,
    ) -> Result<()> {
        let p = normalise_path(path)?;
        if p == "/" {
            return Err(crate::Error::InvalidArgument(
                "squashfs: cannot create file at /".into(),
            ));
        }
        let parent = parent_path(&p);
        self.ensure_parent(parent)?;
        self.files.insert(
            p,
            BuiltEntry {
                kind: BuiltKind::File(src),
                meta,
                xattrs,
            },
        );
        Ok(())
    }

    pub fn create_symlink(
        &mut self,
        path: &str,
        target: &str,
        meta: EntryMeta,
        xattrs: Vec<Xattr>,
    ) -> Result<()> {
        let p = normalise_path(path)?;
        if p == "/" {
            return Err(crate::Error::InvalidArgument(
                "squashfs: cannot create symlink at /".into(),
            ));
        }
        let parent = parent_path(&p);
        self.ensure_parent(parent)?;
        self.files.insert(
            p,
            BuiltEntry {
                kind: BuiltKind::Symlink(target.to_string()),
                meta,
                xattrs,
            },
        );
        Ok(())
    }

    /// Register a hard link at `dst_path` pointing at the existing inode of
    /// `src_path`. SquashFS rejects hard links to directories, so the source
    /// must resolve to a regular file or symlink — anything else returns
    /// [`crate::Error::InvalidArgument`].
    pub fn create_hardlink(&mut self, src_path: &str, dst_path: &str) -> Result<()> {
        let src = normalise_path(src_path)?;
        let dst = normalise_path(dst_path)?;
        if dst == "/" {
            return Err(crate::Error::InvalidArgument(
                "squashfs: cannot create hardlink at /".into(),
            ));
        }
        if dst == src {
            return Err(crate::Error::InvalidArgument(
                "squashfs: hardlink source and destination are identical".into(),
            ));
        }
        // Resolve the source: must be a non-dir file/symlink/device.
        let (real_src, src_meta) = {
            let Some(entry) = self.files.get(&src) else {
                if self.dirs.contains_key(&src) {
                    return Err(crate::Error::InvalidArgument(
                        "squashfs: hardlinks to directories are not allowed".into(),
                    ));
                }
                return Err(crate::Error::InvalidArgument(format!(
                    "squashfs: hardlink source {src:?} does not exist"
                )));
            };
            // Disallow nesting hardlinks (collapse to the ultimate source).
            let real_src = match &entry.kind {
                BuiltKind::Hardlink(s) => s.clone(),
                _ => src.clone(),
            };
            (real_src, entry.meta)
        };
        let parent = parent_path(&dst);
        self.ensure_parent(parent)?;
        // Copy metadata + xattrs from source for completeness; consumers
        // who want different metadata should attach it to the source. The
        // hardlinked entry won't materialise its own inode anyway — the
        // shared inode wins.
        self.files.insert(
            dst,
            BuiltEntry {
                kind: BuiltKind::Hardlink(real_src),
                meta: src_meta,
                xattrs: Vec::new(),
            },
        );
        Ok(())
    }

    /// Buffer a new device node, FIFO or socket. `major`/`minor` are packed
    /// per the Linux `MKDEV(major, minor)` formula; only the low 20 bits of
    /// `minor` and low 12 bits of `major` are preserved (matching the
    /// SquashFS legacy 32-bit device format).
    pub fn create_device(
        &mut self,
        path: &str,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: EntryMeta,
        xattrs: Vec<Xattr>,
    ) -> Result<()> {
        let p = normalise_path(path)?;
        if p == "/" {
            return Err(crate::Error::InvalidArgument(
                "squashfs: cannot create device at /".into(),
            ));
        }
        let parent = parent_path(&p);
        self.ensure_parent(parent)?;
        self.files.insert(
            p,
            BuiltEntry {
                kind: BuiltKind::Device { kind, major, minor },
                meta,
                xattrs,
            },
        );
        Ok(())
    }

    fn ensure_parent(&mut self, parent: String) -> Result<()> {
        if parent == "/" {
            return Ok(());
        }
        if !self.dirs.contains_key(&parent) {
            // Recursively ensure the parent's parent, then insert.
            let pp = parent_path(&parent);
            self.ensure_parent(pp)?;
            self.dirs.insert(
                parent,
                EntryDir {
                    meta: EntryMeta {
                        mode: 0o755,
                        ..Default::default()
                    },
                    xattrs: Vec::new(),
                },
            );
        }
        Ok(())
    }

    /// Lay everything out on `dev` and return the populated superblock.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<super::Superblock> {
        // ---- 1) Assign inode numbers in deterministic order. ----
        //
        // Inode 1 = root; the rest follow the BTreeMap's natural order so
        // listings are alphabetical inside each directory. Hardlinks share
        // the source's inode number — they don't consume a slot.
        let mut inode_numbers: BTreeMap<String, u32> = BTreeMap::new();
        // Count hard links per source path so we can populate link_count
        // correctly when we emit the source inode.
        let mut hardlink_count: BTreeMap<String, u32> = BTreeMap::new();
        for entry in self.files.values() {
            if let BuiltKind::Hardlink(src) = &entry.kind {
                *hardlink_count.entry(src.clone()).or_insert(0) += 1;
            }
        }
        let mut next_inode: u32 = 1;
        inode_numbers.insert("/".into(), next_inode);
        next_inode += 1;
        // Combine dirs + files into a sorted path list, skipping "/" and
        // hardlinks (the latter are filled in below).
        let mut all_paths: Vec<&String> = Vec::new();
        for p in self.dirs.keys() {
            if p != "/" {
                all_paths.push(p);
            }
        }
        for (p, e) in &self.files {
            if !matches!(e.kind, BuiltKind::Hardlink(_)) {
                all_paths.push(p);
            }
        }
        all_paths.sort();
        for p in &all_paths {
            inode_numbers.insert((*p).clone(), next_inode);
            next_inode += 1;
        }
        // Now fill in inode numbers for hardlinks (they alias the source).
        for (p, e) in &self.files {
            if let BuiltKind::Hardlink(src) = &e.kind {
                let src_inode = *inode_numbers.get(src).ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "squashfs: hardlink source {src:?} not found at flush time"
                    ))
                })?;
                inode_numbers.insert(p.clone(), src_inode);
            }
        }
        let total_inodes = next_inode - 1;

        // ---- 2) Build the id table (dedup uid/gid). ----
        let mut id_table: Vec<u32> = Vec::new();
        let mut id_index: BTreeMap<u32, u16> = BTreeMap::new();
        let mut intern_id = |v: u32| -> Result<u16> {
            if let Some(&i) = id_index.get(&v) {
                return Ok(i);
            }
            if id_table.len() >= u16::MAX as usize {
                return Err(crate::Error::InvalidArgument(
                    "squashfs: id table overflow (>65535 distinct uids/gids)".into(),
                ));
            }
            let i = id_table.len() as u16;
            id_table.push(v);
            id_index.insert(v, i);
            Ok(i)
        };

        // ---- 3) Build the xattr lookup. Each distinct set => one entry. ----
        let mut xattr_sets: Vec<Vec<Xattr>> = Vec::new();
        let mut xattr_set_index: BTreeMap<Vec<(String, Vec<u8>)>, u32> = BTreeMap::new();
        let intern_xattr = |xs: &Vec<Xattr>,
                            xattr_sets: &mut Vec<Vec<Xattr>>,
                            xattr_set_index: &mut BTreeMap<Vec<(String, Vec<u8>)>, u32>|
         -> u32 {
            if xs.is_empty() {
                return u32::MAX;
            }
            let key: Vec<(String, Vec<u8>)> = xs
                .iter()
                .map(|x| (x.key.clone(), x.value.clone()))
                .collect();
            if let Some(&i) = xattr_set_index.get(&key) {
                return i;
            }
            let i = xattr_sets.len() as u32;
            xattr_sets.push(xs.clone());
            xattr_set_index.insert(key, i);
            i
        };

        // ---- 4) Phase A — write file data blocks + pack tails into a single fragment block.
        //
        // We accumulate per-file metadata: blocks_start, block_size words,
        // fragment_index, fragment_offset.
        let mut next_disk_offset: u64 = 96; // immediately after superblock
        // Write a placeholder superblock space (zeros).
        ensure_size(dev, 96)?;
        dev.write_at(0, &[0u8; 96])?;

        // LZ4 needs a compressor-options metablock immediately after the
        // superblock: 8 bytes of payload (`u32 version=1, u32 flags=0`)
        // wrapped in a 2-byte uncompressed-metablock header. We must
        // also set the SQUASHFS_COMP_OPT flag in the superblock (done
        // later, when we encode `sb`).
        if matches!(self.compression, Compression::Lz4) {
            // 2-byte header: length=8, high bit set (uncompressed).
            let header = ((8u16) | 0x8000).to_le_bytes();
            let body = [
                1u32.to_le_bytes(),
                0u32.to_le_bytes(), // flags: 0 = standard LZ4 (no HC)
            ]
            .concat();
            ensure_size(dev, 96 + 2 + body.len() as u64)?;
            dev.write_at(96, &header)?;
            dev.write_at(98, &body)?;
            next_disk_offset = 96 + 2 + body.len() as u64;
        }

        struct FileLayout {
            blocks_start: u64,
            block_size_words: Vec<u32>,
            fragment_index: u32,
            fragment_offset: u32,
            file_size: u64,
        }
        let mut file_layouts: BTreeMap<String, FileLayout> = BTreeMap::new();
        // Fragment block accumulator. Tails are packed into the *current*
        // fragment buffer; once it would exceed `block_size`, we emit it as
        // one fragment-table entry and start a fresh buffer. This produces
        // multiple fragment-table entries for trees with many small files.
        let mut frag_buf: Vec<u8> = Vec::new();
        // Fragment table entries built as we flush each frag buffer.
        let mut fragment_entries: Vec<(u64, u32)> = Vec::new(); // (disk_offset, size_word)

        let block_size = self.block_size;
        let mut scratch = vec![0u8; 65_536];

        // Local helper to flush the current frag buffer if it's non-empty.
        // Returns the index assigned to the just-flushed buffer.
        let flush_frag_buf = |frag_buf: &mut Vec<u8>,
                              fragment_entries: &mut Vec<(u64, u32)>,
                              compression: Compression,
                              dev: &mut dyn BlockDevice,
                              next_disk_offset: &mut u64|
         -> Result<()> {
            if frag_buf.is_empty() {
                return Ok(());
            }
            let start = *next_disk_offset;
            let size_word = emit_data_block(dev, frag_buf, compression, next_disk_offset)?;
            fragment_entries.push((start, size_word));
            frag_buf.clear();
            Ok(())
        };

        // Iterate files in deterministic order.
        let file_keys: Vec<String> = self.files.keys().cloned().collect();
        for path in &file_keys {
            let entry = self.files.get_mut(path).unwrap();
            // Only the regular-file variant emits data here; leave the
            // other variants untouched.
            let is_file = matches!(entry.kind, BuiltKind::File(_));
            if !is_file {
                continue;
            }
            // Swap the source out so we can consume it; replace with a
            // zero placeholder that we'll never read again.
            let placeholder = BuiltKind::File(FileSource::Zero(0));
            let BuiltKind::File(src) = std::mem::replace(&mut entry.kind, placeholder) else {
                unreachable!()
            };
            // Open the source.
            let len = src
                .len()
                .map_err(|e| crate::Error::Io(std::io::Error::other(e)))?;
            let (mut reader, total) = src
                .open()
                .map_err(|e| crate::Error::Io(std::io::Error::other(e)))?;
            let _ = len;
            let mut layout = FileLayout {
                blocks_start: next_disk_offset,
                block_size_words: Vec::new(),
                fragment_index: 0xFFFF_FFFF,
                fragment_offset: 0,
                file_size: total,
            };
            let mut consumed: u64 = 0;
            if total == 0 {
                file_layouts.insert(path.clone(), layout);
                continue;
            }
            if total < block_size as u64 {
                // Whole file goes to a fragment. Flush the current frag
                // buffer first if appending this tail would push it past
                // `block_size`.
                if frag_buf.len() as u64 + total > block_size as u64 {
                    flush_frag_buf(
                        &mut frag_buf,
                        &mut fragment_entries,
                        self.compression,
                        dev,
                        &mut next_disk_offset,
                    )?;
                }
                let off = frag_buf.len();
                copy_to_buf(&mut *reader, &mut scratch, total, &mut frag_buf)?;
                layout.fragment_index = fragment_entries.len() as u32;
                layout.fragment_offset = off as u32;
                file_layouts.insert(path.clone(), layout);
                continue;
            }
            let mut block_buf = vec![0u8; block_size as usize];
            while total - consumed >= block_size as u64 {
                read_exact(&mut *reader, &mut block_buf)?;
                let size_word =
                    emit_data_block(dev, &block_buf, self.compression, &mut next_disk_offset)?;
                layout.block_size_words.push(size_word);
                consumed += block_size as u64;
            }
            let tail = (total - consumed) as usize;
            if tail > 0 {
                if frag_buf.len() + tail > block_size as usize {
                    flush_frag_buf(
                        &mut frag_buf,
                        &mut fragment_entries,
                        self.compression,
                        dev,
                        &mut next_disk_offset,
                    )?;
                }
                let off = frag_buf.len();
                copy_to_buf(&mut *reader, &mut scratch, tail as u64, &mut frag_buf)?;
                layout.fragment_index = fragment_entries.len() as u32;
                layout.fragment_offset = off as u32;
            }
            file_layouts.insert(path.clone(), layout);
        }

        // Emit the final (possibly only) fragment block, if any.
        flush_frag_buf(
            &mut frag_buf,
            &mut fragment_entries,
            self.compression,
            dev,
            &mut next_disk_offset,
        )?;

        // ---- 5) Phase B — assign uid/gid + xattr indices, build inode metablocks. ----
        //
        // Inodes are laid out in the same order as inode numbers: root,
        // then each path in `all_paths` order. We record (block_rel,
        // offset_in_block) for each inode so directory entries can point
        // at them.

        // Helper: encode a directory inode header + payload.
        let mut inode_table_raw: Vec<u8> = Vec::new();
        // We need to know each directory's listing-byte position in the
        // directory table before we can write its inode. So: defer the
        // *directory* inode emission until after the directory table is
        // built. To do that in one pass we instead build the directory
        // table first.
        //
        // Strategy:
        //  1. Compute the inode-table layout for every *non-directory*
        //     inode, so that directory listings can know where the file
        //     inodes live.
        //  2. Build the directory listings — they reference
        //     non-directory inodes' (block, offset) directly, and
        //     reference subdirectory inodes via placeholder (back-patched
        //     once dir inodes are written).
        //
        // Simpler approach: write inodes in two passes. Pass 1 emits all
        // non-directory inodes (file/symlink). Pass 2 emits directory
        // inodes after we know listing offsets.
        let mut inode_positions: BTreeMap<String, (u32, u16)> = BTreeMap::new(); // path -> (block_rel, in_block_offset)

        // Helper: append a non-dir inode and return its raw_offset.
        let emit_inode = |raw: &mut Vec<u8>, bytes: &[u8]| -> usize {
            let off = raw.len();
            raw.extend_from_slice(bytes);
            off
        };

        // Track per-path raw byte offsets for non-dir inodes.
        let mut nondir_raw_offsets: BTreeMap<String, usize> = BTreeMap::new();
        // Track per-path raw byte offsets for dir inodes (filled in pass 2).
        // Root inode is laid out first to satisfy the "root inode at
        // inode_table offset (0,0)" convention.
        // Pass A: reserve the root inode slot (we don't know listing
        // offset yet); but we can still emit a placeholder if needed.
        //
        // Easier approach: emit file/symlink inodes first, then all dir
        // inodes (root last, so it ends up *not* at offset 0). The spec
        // doesn't require the root to be first — only that root_inode in
        // the superblock points at it. So we emit dirs *after* non-dirs.

        // ---- Build all non-directory inodes (skip hardlinks — they alias). ----
        for path in &file_keys {
            let entry = self.files.get(path).unwrap();
            // Hardlinks alias an existing inode; no inode emission.
            if matches!(entry.kind, BuiltKind::Hardlink(_)) {
                continue;
            }
            let uid_idx = intern_id(entry.meta.uid)?;
            let gid_idx = intern_id(entry.meta.gid)?;
            let inode_no = inode_numbers[path];
            // 1 (the entry itself) + N hardlinks pointing at it.
            let link_count: u32 = 1 + hardlink_count.get(path).copied().unwrap_or(0);
            // Intern the xattr set up front so we know whether we need
            // the extended inode form (which carries an xattr_index).
            let xattr_idx = intern_xattr(&entry.xattrs, &mut xattr_sets, &mut xattr_set_index);
            // Hardlinks force the extended form on the target so the
            // link_count field is materialised on disk.
            let force_ext = link_count > 1;
            match &entry.kind {
                BuiltKind::File(_) => {
                    let layout = &file_layouts[path];
                    if layout.blocks_start > u32::MAX as u64 {
                        return Err(crate::Error::InvalidImage(
                            "squashfs: blocks_start > 4GiB not supported by writer".into(),
                        ));
                    }
                    if layout.file_size > u32::MAX as u64 {
                        return Err(crate::Error::InvalidImage(
                            "squashfs: file > 4GiB not supported by writer".into(),
                        ));
                    }
                    let mut bytes = Vec::with_capacity(64 + layout.block_size_words.len() * 4);
                    if xattr_idx == u32::MAX && !force_ext {
                        // Basic file inode.
                        bytes.extend_from_slice(&INODE_BASIC_FILE.to_le_bytes());
                        bytes.extend_from_slice(&entry.meta.mode.to_le_bytes());
                        bytes.extend_from_slice(&uid_idx.to_le_bytes());
                        bytes.extend_from_slice(&gid_idx.to_le_bytes());
                        bytes.extend_from_slice(&entry.meta.mtime.to_le_bytes());
                        bytes.extend_from_slice(&inode_no.to_le_bytes());
                        bytes.extend_from_slice(&(layout.blocks_start as u32).to_le_bytes());
                        bytes.extend_from_slice(&layout.fragment_index.to_le_bytes());
                        bytes.extend_from_slice(&layout.fragment_offset.to_le_bytes());
                        bytes.extend_from_slice(&(layout.file_size as u32).to_le_bytes());
                    } else {
                        // Extended file inode: u64 blocks_start, u64 file_size,
                        // u64 sparse=0, u32 link_count, u32 fragment_index,
                        // u32 fragment_offset, u32 xattr_index.
                        bytes.extend_from_slice(&INODE_EXT_FILE.to_le_bytes());
                        bytes.extend_from_slice(&entry.meta.mode.to_le_bytes());
                        bytes.extend_from_slice(&uid_idx.to_le_bytes());
                        bytes.extend_from_slice(&gid_idx.to_le_bytes());
                        bytes.extend_from_slice(&entry.meta.mtime.to_le_bytes());
                        bytes.extend_from_slice(&inode_no.to_le_bytes());
                        bytes.extend_from_slice(&layout.blocks_start.to_le_bytes());
                        bytes.extend_from_slice(&layout.file_size.to_le_bytes());
                        bytes.extend_from_slice(&0u64.to_le_bytes()); // sparse
                        bytes.extend_from_slice(&link_count.to_le_bytes());
                        bytes.extend_from_slice(&layout.fragment_index.to_le_bytes());
                        bytes.extend_from_slice(&layout.fragment_offset.to_le_bytes());
                        bytes.extend_from_slice(&xattr_idx.to_le_bytes());
                    }
                    for sw in &layout.block_size_words {
                        bytes.extend_from_slice(&sw.to_le_bytes());
                    }
                    let off = emit_inode(&mut inode_table_raw, &bytes);
                    nondir_raw_offsets.insert(path.clone(), off);
                }
                BuiltKind::Symlink(target) => {
                    let mut bytes = Vec::new();
                    if xattr_idx == u32::MAX {
                        bytes.extend_from_slice(&INODE_BASIC_SYMLINK.to_le_bytes());
                    } else {
                        bytes.extend_from_slice(&INODE_EXT_SYMLINK.to_le_bytes());
                    }
                    bytes.extend_from_slice(&entry.meta.mode.to_le_bytes());
                    bytes.extend_from_slice(&uid_idx.to_le_bytes());
                    bytes.extend_from_slice(&gid_idx.to_le_bytes());
                    bytes.extend_from_slice(&entry.meta.mtime.to_le_bytes());
                    bytes.extend_from_slice(&inode_no.to_le_bytes());
                    bytes.extend_from_slice(&link_count.to_le_bytes());
                    bytes.extend_from_slice(&(target.len() as u32).to_le_bytes());
                    bytes.extend_from_slice(target.as_bytes());
                    if xattr_idx != u32::MAX {
                        bytes.extend_from_slice(&xattr_idx.to_le_bytes());
                    }
                    let off = emit_inode(&mut inode_table_raw, &bytes);
                    nondir_raw_offsets.insert(path.clone(), off);
                }
                BuiltKind::Device { kind, major, minor } => {
                    // Pack into u32 per Linux MKDEV: top 12 bits = major,
                    // bottom 20 bits = minor (legacy SquashFS format).
                    let dev_word: u32 = ((*major & 0xFFF) << 20) | (*minor & 0xF_FFFF);
                    let (basic_id, ext_id) = match kind {
                        DeviceKind::Block => (INODE_BASIC_BLOCK, INODE_EXT_BLOCK),
                        DeviceKind::Char => (INODE_BASIC_CHAR, INODE_EXT_CHAR),
                        DeviceKind::Fifo => (INODE_BASIC_FIFO, INODE_EXT_FIFO),
                        DeviceKind::Socket => (INODE_BASIC_SOCKET, INODE_EXT_SOCKET),
                    };
                    let use_ext = xattr_idx != u32::MAX || force_ext;
                    let mut bytes = Vec::new();
                    if use_ext {
                        bytes.extend_from_slice(&ext_id.to_le_bytes());
                    } else {
                        bytes.extend_from_slice(&basic_id.to_le_bytes());
                    }
                    bytes.extend_from_slice(&entry.meta.mode.to_le_bytes());
                    bytes.extend_from_slice(&uid_idx.to_le_bytes());
                    bytes.extend_from_slice(&gid_idx.to_le_bytes());
                    bytes.extend_from_slice(&entry.meta.mtime.to_le_bytes());
                    bytes.extend_from_slice(&inode_no.to_le_bytes());
                    bytes.extend_from_slice(&link_count.to_le_bytes());
                    // Block/char carry a device word; fifo/socket spec is
                    // device-less but we still write the word as 0 for
                    // block/char-shaped on-disk layout consistency. The
                    // SquashFS spec defines fifo/socket inodes as
                    // header + link_count only (no device field). To
                    // remain spec-correct, only emit the device word for
                    // block/char.
                    match kind {
                        DeviceKind::Block | DeviceKind::Char => {
                            bytes.extend_from_slice(&dev_word.to_le_bytes());
                        }
                        DeviceKind::Fifo | DeviceKind::Socket => {}
                    }
                    if use_ext {
                        bytes.extend_from_slice(&xattr_idx.to_le_bytes());
                    }
                    let off = emit_inode(&mut inode_table_raw, &bytes);
                    nondir_raw_offsets.insert(path.clone(), off);
                }
                BuiltKind::Hardlink(_) => unreachable!(),
                BuiltKind::Dir => unreachable!(),
            }
        }

        // ---- 6) Build directory listings. ----
        //
        // A directory listing per directory: each entry refers to a child
        // inode by (block_rel, in_block_offset). For child directories,
        // we don't know the offset yet — we'll patch later. To keep the
        // first pass simple, we collect listings as lists of
        // `(name, child_path, child_kind)`, then encode them once all
        // child inode positions are known.
        let mut listings: BTreeMap<String, Vec<(String, String, u16)>> = BTreeMap::new();
        for d in self.dirs.keys() {
            listings.insert(d.clone(), Vec::new());
        }
        for p in &file_keys {
            let parent = parent_path(p);
            let name = leaf_name(p).to_string();
            // For a hardlink, the directory entry points at the *source*
            // path's inode position. We resolve that here so the listing
            // emitter further down uses the right (block_rel, offset).
            let (kind, target_path) = match &self.files[p].kind {
                BuiltKind::File(_) => (INODE_BASIC_FILE, p.clone()),
                BuiltKind::Symlink(_) => (INODE_BASIC_SYMLINK, p.clone()),
                BuiltKind::Dir => (INODE_BASIC_DIR, p.clone()),
                BuiltKind::Hardlink(src) => {
                    // Source kind: peek at the source entry.
                    let k = match &self.files[src].kind {
                        BuiltKind::File(_) => INODE_BASIC_FILE,
                        BuiltKind::Symlink(_) => INODE_BASIC_SYMLINK,
                        BuiltKind::Device { kind, .. } => match kind {
                            DeviceKind::Block => INODE_BASIC_BLOCK,
                            DeviceKind::Char => INODE_BASIC_CHAR,
                            DeviceKind::Fifo => INODE_BASIC_FIFO,
                            DeviceKind::Socket => INODE_BASIC_SOCKET,
                        },
                        BuiltKind::Dir | BuiltKind::Hardlink(_) => INODE_BASIC_FILE,
                    };
                    (k, src.clone())
                }
                BuiltKind::Device { kind, .. } => {
                    let id = match kind {
                        DeviceKind::Block => INODE_BASIC_BLOCK,
                        DeviceKind::Char => INODE_BASIC_CHAR,
                        DeviceKind::Fifo => INODE_BASIC_FIFO,
                        DeviceKind::Socket => INODE_BASIC_SOCKET,
                    };
                    (id, p.clone())
                }
            };
            listings
                .get_mut(&parent)
                .unwrap()
                .push((name, target_path, kind));
        }
        for d in self.dirs.keys() {
            if d == "/" {
                continue;
            }
            let parent = parent_path(d);
            let name = leaf_name(d).to_string();
            listings
                .get_mut(&parent)
                .unwrap()
                .push((name, d.clone(), INODE_BASIC_DIR));
        }
        for v in listings.values_mut() {
            v.sort_by(|a, b| a.0.cmp(&b.0));
        }

        // ---- 7) Build directory inodes (with placeholder listing offsets). ----
        //
        // We need each directory inode's (block_rel, in_block_offset). To
        // get those we must emit inodes one by one into `inode_table_raw`
        // and convert raw offsets to (block_rel, in_block_offset) using
        // the per-block size map. But block sizes are only known after
        // encoding. So we encode raw bytes first, then re-chunk to compute
        // (block_rel, in_block_offset) for every recorded raw offset.

        // Reserve directory inode slots. We patch listing-block index and
        // listing-byte offset (which we don't know yet) later — but that
        // means we have to rewrite the inode metablocks afterwards. Two
        // options:
        //  A) Emit dir inodes after the directory table is fully built.
        //  B) Emit dir inodes with placeholders, then back-patch raw bytes
        //     and re-encode metablocks.
        // (A) is cleaner. So we leave dir-inode emission for last and
        // simply record the post-non-dir-inode raw_offset for each dir.
        let mut dir_raw_offsets: BTreeMap<String, usize> = BTreeMap::new();
        // Encode directory listings into a single raw byte buffer; record
        // each directory's (raw_offset, raw_size). One or more runs per
        // listing — see step 8.
        let mut dir_table_raw: Vec<u8> = Vec::new();
        let mut dir_listing_offsets: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // path -> (raw_offset, raw_size)

        // Decide which directories need the extended-dir inode form
        // (40 bytes instead of 32). We promote a dir to ext form when its
        // listing-size upper bound, OR its xattr set, requires it.
        //
        // Upper bound per directory: in the worst case every entry lives
        // in its own run, so each entry costs `12 (header) + 8 + name_len`
        // bytes. A basic-dir's `file_size` field is a u16 storing size+3,
        // so it caps at 65532 bytes of listing data.
        let mut dir_is_ext: BTreeMap<String, bool> = BTreeMap::new();
        for d in self.dirs.keys() {
            let mut upper: usize = 0;
            if let Some(entries) = listings.get(d) {
                for (name, _, _) in entries {
                    upper += 12 + 8 + name.len();
                }
            }
            let needs_ext_for_size = upper > 65_532;
            let needs_ext_for_xattr = !self.dirs[d].xattrs.is_empty();
            dir_is_ext.insert(d.clone(), needs_ext_for_size || needs_ext_for_xattr);
        }

        // Reserve appropriate space per directory inode (32 basic, 40 ext).
        for d in self.dirs.keys() {
            let off = inode_table_raw.len();
            let sz = if dir_is_ext[d] { 40 } else { 32 };
            inode_table_raw.extend_from_slice(&vec![0u8; sz]);
            dir_raw_offsets.insert(d.clone(), off);
        }

        // Compute inode metablock chunking (raw_offset -> (block_rel, in_block_offset)).
        // The disk payload itself is recomputed after the dir-inode back-patch.
        let (inode_block_rel_map, _) = chunk_raw_to_metablocks(&inode_table_raw, self.compression)?;
        let raw_to_pos = |raw_off: usize| -> (u32, u16) {
            let entry = inode_block_rel_map[raw_off / 8192];
            (entry, (raw_off % 8192) as u16)
        };
        for path in nondir_raw_offsets.keys() {
            let (b, o) = raw_to_pos(nondir_raw_offsets[path]);
            inode_positions.insert(path.clone(), (b, o));
        }
        for path in dir_raw_offsets.keys() {
            let (b, o) = raw_to_pos(dir_raw_offsets[path]);
            inode_positions.insert(path.clone(), (b, o));
        }

        // ---- 8) Encode directory listings. ----
        //
        // Each listing is one or more "runs", each starting with a 12-byte
        // header. We emit a single run per directory whose entries share
        // the same inode-table metablock; if a directory's children
        // straddle multiple inode blocks we split into runs accordingly.
        for d in self.dirs.keys() {
            let entries = &listings[d];
            let raw_start = dir_table_raw.len();
            if entries.is_empty() {
                dir_listing_offsets.insert(d.clone(), (raw_start, 0));
                continue;
            }
            // Group entries by their child's inode metablock (block_rel)
            // — within a run the entries' signed inode_number_offset is
            // also bounded to fit i16.
            let mut idx = 0;
            while idx < entries.len() {
                let (_name0, child0_path, _kind0) = &entries[idx];
                let (start_block, _) = inode_positions[child0_path];
                let base_inode = inode_numbers[child0_path];
                let mut run_entries: Vec<&(String, String, u16)> = Vec::new();
                while idx < entries.len() {
                    let (_n, cp, _k) = &entries[idx];
                    let (cb, _co) = inode_positions[cp];
                    if cb != start_block {
                        break;
                    }
                    let ci = inode_numbers[cp];
                    let diff = ci as i64 - base_inode as i64;
                    if !(-32768..=32767).contains(&diff) {
                        break;
                    }
                    // Limit run size to 256 entries (spec: count is u32
                    // stored as count-1, so up to 256 entries per run is
                    // conservative).
                    if run_entries.len() >= 256 {
                        break;
                    }
                    run_entries.push(&entries[idx]);
                    idx += 1;
                }
                // Header
                let count_minus_one = (run_entries.len() - 1) as u32;
                dir_table_raw.extend_from_slice(&count_minus_one.to_le_bytes());
                dir_table_raw.extend_from_slice(&start_block.to_le_bytes());
                dir_table_raw.extend_from_slice(&base_inode.to_le_bytes());
                // Entries
                for (name, child_path, kind) in run_entries {
                    let (_b, in_off) = inode_positions[child_path];
                    let ci = inode_numbers[child_path];
                    let diff = ci as i64 - base_inode as i64;
                    let signed = diff as i16;
                    dir_table_raw.extend_from_slice(&in_off.to_le_bytes());
                    dir_table_raw.extend_from_slice(&signed.to_le_bytes());
                    dir_table_raw.extend_from_slice(&kind.to_le_bytes());
                    let name_bytes = name.as_bytes();
                    // Stored as len-1.
                    let name_size = (name_bytes.len() - 1) as u16;
                    dir_table_raw.extend_from_slice(&name_size.to_le_bytes());
                    dir_table_raw.extend_from_slice(name_bytes);
                }
            }
            let raw_size = dir_table_raw.len() - raw_start;
            dir_listing_offsets.insert(d.clone(), (raw_start, raw_size));
        }

        // ---- 9) Chunk directory raw into metablocks. ----
        let (dir_block_rel_map, dir_disk_payload) =
            chunk_raw_to_metablocks(&dir_table_raw, self.compression)?;
        let dir_raw_to_pos = |raw_off: usize| -> (u32, u16) {
            let entry = dir_block_rel_map[raw_off / 8192];
            (entry, (raw_off % 8192) as u16)
        };

        // ---- 10) Patch directory inodes with real listing offsets. ----
        for d in self.dirs.keys() {
            let (raw_off, listing_size) = dir_listing_offsets[d];
            let (block_index, block_offset) = dir_raw_to_pos(raw_off);
            let parent_inode = if d == "/" {
                inode_numbers["/"]
            } else {
                let p = parent_path(d);
                inode_numbers[&p]
            };
            let inode_no = inode_numbers[d];
            let dir_meta = &self.dirs[d];
            let uid_idx = intern_id(dir_meta.meta.uid)?;
            let gid_idx = intern_id(dir_meta.meta.gid)?;
            let off = dir_raw_offsets[d];
            let link_count = count_subdirs(&listings, d) as u32 + 2;
            let xattr_idx = intern_xattr(&dir_meta.xattrs, &mut xattr_sets, &mut xattr_set_index);
            if dir_is_ext[d] {
                // ExtDir: 16-byte header + 24-byte payload + 0 index entries.
                let mut buf = [0u8; 40];
                buf[0..2].copy_from_slice(&INODE_EXT_DIR.to_le_bytes());
                buf[2..4].copy_from_slice(&dir_meta.meta.mode.to_le_bytes());
                buf[4..6].copy_from_slice(&uid_idx.to_le_bytes());
                buf[6..8].copy_from_slice(&gid_idx.to_le_bytes());
                buf[8..12].copy_from_slice(&dir_meta.meta.mtime.to_le_bytes());
                buf[12..16].copy_from_slice(&inode_no.to_le_bytes());
                // Payload: u32 link_count, u32 file_size (size+3),
                //          u32 block_index, u32 parent_inode,
                //          u16 index_count=0, u16 block_offset, u32 xattr.
                buf[16..20].copy_from_slice(&link_count.to_le_bytes());
                let stored: u32 = (listing_size as u32).saturating_add(3);
                buf[20..24].copy_from_slice(&stored.to_le_bytes());
                buf[24..28].copy_from_slice(&block_index.to_le_bytes());
                buf[28..32].copy_from_slice(&parent_inode.to_le_bytes());
                buf[32..34].copy_from_slice(&0u16.to_le_bytes()); // index_count
                buf[34..36].copy_from_slice(&block_offset.to_le_bytes());
                buf[36..40].copy_from_slice(&xattr_idx.to_le_bytes());
                inode_table_raw[off..off + 40].copy_from_slice(&buf);
            } else {
                // BasicDir: 16-byte header + 16-byte payload.
                let mut buf = [0u8; 32];
                buf[0..2].copy_from_slice(&INODE_BASIC_DIR.to_le_bytes());
                buf[2..4].copy_from_slice(&dir_meta.meta.mode.to_le_bytes());
                buf[4..6].copy_from_slice(&uid_idx.to_le_bytes());
                buf[6..8].copy_from_slice(&gid_idx.to_le_bytes());
                buf[8..12].copy_from_slice(&dir_meta.meta.mtime.to_le_bytes());
                buf[12..16].copy_from_slice(&inode_no.to_le_bytes());
                buf[16..20].copy_from_slice(&block_index.to_le_bytes());
                buf[20..24].copy_from_slice(&link_count.to_le_bytes());
                let stored = if listing_size == 0 {
                    3u16
                } else {
                    (listing_size as u16).saturating_add(3)
                };
                buf[24..26].copy_from_slice(&stored.to_le_bytes());
                buf[26..28].copy_from_slice(&block_offset.to_le_bytes());
                buf[28..32].copy_from_slice(&parent_inode.to_le_bytes());
                inode_table_raw[off..off + 32].copy_from_slice(&buf);
            }
        }

        // ---- 11) Re-encode inode metablocks after the back-patch. ----
        // (The chunk boundaries stay the same because we only touched
        // 32-byte windows that lie wholly within their metablocks: every
        // dir inode is 32 contiguous bytes, no straddling.)
        let (_, inode_disk_payload) = chunk_raw_to_metablocks(&inode_table_raw, self.compression)?;
        let inode_table_start = next_disk_offset;
        ensure_size(dev, next_disk_offset + inode_disk_payload.len() as u64)?;
        dev.write_at(inode_table_start, &inode_disk_payload)?;
        next_disk_offset += inode_disk_payload.len() as u64;

        // Directory table
        let directory_table_start = next_disk_offset;
        ensure_size(dev, next_disk_offset + dir_disk_payload.len() as u64)?;
        dev.write_at(directory_table_start, &dir_disk_payload)?;
        next_disk_offset += dir_disk_payload.len() as u64;

        // ---- 12) Fragment table. ----
        let fragment_count = fragment_entries.len() as u32;
        let fragment_table_start = if fragment_count == 0 {
            u64::MAX
        } else {
            // Build a metablock of fragment entries (16 bytes each).
            let mut frag_raw = Vec::with_capacity(fragment_entries.len() * 16);
            for (start, size_word) in &fragment_entries {
                frag_raw.extend_from_slice(&start.to_le_bytes());
                frag_raw.extend_from_slice(&size_word.to_le_bytes());
                frag_raw.extend_from_slice(&0u32.to_le_bytes()); // unused
            }
            let mb = encode_metablock(&frag_raw, self.compression)?;
            let mb_disk_offset = next_disk_offset;
            ensure_size(dev, next_disk_offset + mb.len() as u64)?;
            dev.write_at(mb_disk_offset, &mb)?;
            next_disk_offset += mb.len() as u64;
            // L1 array: a single u64 pointing at our metablock.
            let l1_offset = next_disk_offset;
            ensure_size(dev, next_disk_offset + 8)?;
            dev.write_at(l1_offset, &mb_disk_offset.to_le_bytes())?;
            next_disk_offset += 8;
            l1_offset
        };

        // ---- 13) Export table. ----
        // One u64 inode ref per inode number (in order 1..=total_inodes).
        let export_table_start = if total_inodes == 0 {
            u64::MAX
        } else {
            // Inverse map: inode_number -> path.
            // Build inv: inode number → path. Skip hardlink aliases so a
            // hard link's path doesn't shadow the real entry (which is the
            // one that has an `inode_positions` mapping).
            let mut inv: BTreeMap<u32, String> = BTreeMap::new();
            for (p, &i) in &inode_numbers {
                if let Some(e) = self.files.get(p)
                    && matches!(e.kind, BuiltKind::Hardlink(_))
                {
                    continue;
                }
                inv.insert(i, p.clone());
            }
            let mut raw: Vec<u8> = Vec::with_capacity(total_inodes as usize * 8);
            for i in 1..=total_inodes {
                let p = inv.get(&i).ok_or_else(|| {
                    crate::Error::InvalidImage("squashfs: gap in inode numbers".into())
                })?;
                let (block_rel, in_off) = inode_positions[p];
                let iref = ((block_rel as u64) << 16) | (in_off as u64);
                raw.extend_from_slice(&iref.to_le_bytes());
            }
            // Chunk into metablocks (8 KiB → 1024 entries).
            let mut mb_offsets_abs: Vec<u64> = Vec::new();
            let mut pos = 0usize;
            while pos < raw.len() {
                let end = (pos + 8192).min(raw.len());
                let mb = encode_metablock(&raw[pos..end], self.compression)?;
                let mb_off = next_disk_offset;
                ensure_size(dev, next_disk_offset + mb.len() as u64)?;
                dev.write_at(mb_off, &mb)?;
                next_disk_offset += mb.len() as u64;
                mb_offsets_abs.push(mb_off);
                pos = end;
            }
            let l1_offset = next_disk_offset;
            let mut l1 = Vec::with_capacity(mb_offsets_abs.len() * 8);
            for o in &mb_offsets_abs {
                l1.extend_from_slice(&o.to_le_bytes());
            }
            ensure_size(dev, next_disk_offset + l1.len() as u64)?;
            dev.write_at(l1_offset, &l1)?;
            next_disk_offset += l1.len() as u64;
            l1_offset
        };

        // ---- 14) Id lookup table. ----
        let id_count = id_table.len() as u16;
        let id_table_start = if id_count == 0 {
            u64::MAX
        } else {
            let mut raw: Vec<u8> = Vec::with_capacity(id_table.len() * 4);
            for v in &id_table {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            let mut mb_offsets_abs: Vec<u64> = Vec::new();
            let mut pos = 0usize;
            while pos < raw.len() {
                let end = (pos + 8192).min(raw.len());
                let mb = encode_metablock(&raw[pos..end], self.compression)?;
                let mb_off = next_disk_offset;
                ensure_size(dev, next_disk_offset + mb.len() as u64)?;
                dev.write_at(mb_off, &mb)?;
                next_disk_offset += mb.len() as u64;
                mb_offsets_abs.push(mb_off);
                pos = end;
            }
            let l1_offset = next_disk_offset;
            let mut l1 = Vec::with_capacity(mb_offsets_abs.len() * 8);
            for o in &mb_offsets_abs {
                l1.extend_from_slice(&o.to_le_bytes());
            }
            ensure_size(dev, next_disk_offset + l1.len() as u64)?;
            dev.write_at(l1_offset, &l1)?;
            next_disk_offset += l1.len() as u64;
            l1_offset
        };

        // ---- 15) Xattr table. ----
        let xattr_id_table_start = if xattr_sets.is_empty() {
            u64::MAX
        } else {
            let base = next_disk_offset;
            let (payload, hdr_off) =
                super::xattr::encode_xattr_table(&xattr_sets, base, self.compression)?;
            ensure_size(dev, base + payload.len() as u64)?;
            dev.write_at(base, &payload)?;
            next_disk_offset += payload.len() as u64;
            base + hdr_off
        };

        // ---- 16) Superblock. ----
        let bytes_used = next_disk_offset;
        let block_log = block_size.trailing_zeros() as u16;
        let root_ref = {
            let (b, o) = inode_positions["/"];
            ((b as u64) << 16) | (o as u64)
        };
        let comp_id = match self.compression {
            Compression::Gzip => 1,
            Compression::Lzma => 2,
            Compression::Lzo => 3,
            Compression::Xz => 4,
            Compression::Lz4 => 5,
            Compression::Zstd => 6,
            Compression::Unknown(_) => 0,
        };
        let mut sb = vec![0u8; 96];
        sb[0..4].copy_from_slice(&super::SQUASHFS_MAGIC.to_le_bytes());
        sb[4..8].copy_from_slice(&total_inodes.to_le_bytes());
        sb[8..12].copy_from_slice(&0u32.to_le_bytes()); // mkfs_time
        sb[12..16].copy_from_slice(&block_size.to_le_bytes());
        sb[16..20].copy_from_slice(&fragment_count.to_le_bytes());
        sb[20..22].copy_from_slice(&(comp_id as u16).to_le_bytes());
        sb[22..24].copy_from_slice(&block_log.to_le_bytes());
        // Flags: signal whether the export and id-table-uncompressed bits apply.
        // We always emit an export table + xattr table when applicable.
        let mut flags: u16 = 0;
        if export_table_start != u64::MAX {
            flags |= 0x0080;
        }
        if xattr_id_table_start == u64::MAX {
            flags |= 0x0200;
        }
        // LZ4 requires a compressor-options metablock immediately after
        // the superblock. It carries `u32 version=1` + `u32 flags` (we
        // emit flags=0 — non-HC). Set bit 0x0400 (SQUASHFS_COMP_OPT) so
        // unsquashfs reads the block.
        if matches!(self.compression, Compression::Lz4) {
            flags |= 0x0400;
        }
        sb[24..26].copy_from_slice(&flags.to_le_bytes());
        sb[26..28].copy_from_slice(&id_count.to_le_bytes());
        sb[28..30].copy_from_slice(&4u16.to_le_bytes()); // major
        sb[30..32].copy_from_slice(&0u16.to_le_bytes()); // minor
        sb[32..40].copy_from_slice(&root_ref.to_le_bytes());
        sb[40..48].copy_from_slice(&bytes_used.to_le_bytes());
        sb[48..56].copy_from_slice(&id_table_start.to_le_bytes());
        sb[56..64].copy_from_slice(&xattr_id_table_start.to_le_bytes());
        sb[64..72].copy_from_slice(&inode_table_start.to_le_bytes());
        sb[72..80].copy_from_slice(&directory_table_start.to_le_bytes());
        sb[80..88].copy_from_slice(&fragment_table_start.to_le_bytes());
        sb[88..96].copy_from_slice(&export_table_start.to_le_bytes());
        dev.write_at(0, &sb)?;
        dev.sync()?;

        super::Superblock::decode(&sb).ok_or_else(|| {
            crate::Error::InvalidImage("squashfs: writer produced invalid superblock".into())
        })
    }
}

/// Encode a raw byte stream as a sequence of 8 KiB metablocks. Returns the
/// concatenated on-disk payload (header+payload for each block) plus a
/// vector of *relative* block offsets — one per metablock, indexed by
/// `raw_offset / 8192`. The relative offsets are the bytes of the on-disk
/// payload (header + payload) preceding each metablock.
fn chunk_raw_to_metablocks(raw: &[u8], compression: Compression) -> Result<(Vec<u32>, Vec<u8>)> {
    let mut rel_offsets: Vec<u32> = Vec::new();
    let mut out: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    if raw.is_empty() {
        return Ok((rel_offsets, out));
    }
    while pos < raw.len() {
        rel_offsets.push(out.len() as u32);
        let end = (pos + 8192).min(raw.len());
        let mb = encode_metablock(&raw[pos..end], compression)?;
        out.extend_from_slice(&mb);
        pos = end;
    }
    Ok((rel_offsets, out))
}

/// Write `block` as one data block: compress when smaller, otherwise
/// emit raw with the "uncompressed" high bit set. Returns the encoded
/// size word for the file's block list.
fn emit_data_block(
    dev: &mut dyn BlockDevice,
    block: &[u8],
    compression: Compression,
    next_disk_offset: &mut u64,
) -> Result<u32> {
    let algo = compression_to_algo(compression);
    let (payload, size_word): (Vec<u8>, u32) = if let Some(a) = algo
        && a.enabled()
        && let Ok(c) = crate::compression::compress(a, block)
        && !c.is_empty()
        && c.len() < block.len()
    {
        let sw = c.len() as u32;
        (c, sw)
    } else {
        // Uncompressed: high bit set in the size word.
        let sw = block.len() as u32 | 0x0100_0000;
        (block.to_vec(), sw)
    };
    ensure_size(dev, *next_disk_offset + payload.len() as u64)?;
    dev.write_at(*next_disk_offset, &payload)?;
    *next_disk_offset += payload.len() as u64;
    Ok(size_word)
}

/// Make sure `dev` is at least `len` bytes long. Block backends with a
/// fixed capacity will already be sized correctly; the [`MemoryBackend`]
/// used for tests grows lazily, so this is a no-op when capacity is
/// sufficient.
fn ensure_size(dev: &mut dyn BlockDevice, len: u64) -> Result<()> {
    if dev.total_size() < len {
        return Err(crate::Error::OutOfBounds {
            offset: 0,
            len,
            size: dev.total_size(),
        });
    }
    Ok(())
}

/// Read exactly `out.len()` bytes from `r`. Errors with `InvalidImage`
/// (we treat truncated host files as a corrupt-input case).
fn read_exact(r: &mut dyn Read, out: &mut [u8]) -> Result<()> {
    let mut filled = 0;
    while filled < out.len() {
        let n = r.read(&mut out[filled..]).map_err(crate::Error::Io)?;
        if n == 0 {
            return Err(crate::Error::InvalidImage(
                "squashfs: source reader returned EOF before file size reached".into(),
            ));
        }
        filled += n;
    }
    Ok(())
}

/// Copy exactly `n` bytes from `r` into `dst`, using `scratch` as a 64 KiB
/// staging buffer (the streaming invariant).
fn copy_to_buf(r: &mut dyn Read, scratch: &mut [u8], n: u64, dst: &mut Vec<u8>) -> Result<()> {
    let mut remaining = n;
    while remaining > 0 {
        let want = remaining.min(scratch.len() as u64) as usize;
        let buf = &mut scratch[..want];
        let mut filled = 0;
        while filled < buf.len() {
            let m = r.read(&mut buf[filled..]).map_err(crate::Error::Io)?;
            if m == 0 {
                return Err(crate::Error::InvalidImage(
                    "squashfs: source reader returned EOF before declared length".into(),
                ));
            }
            filled += m;
        }
        dst.extend_from_slice(buf);
        remaining -= want as u64;
    }
    Ok(())
}

/// Normalise a user-supplied path to its canonical form: leading "/",
/// no trailing "/", no "//", no "." components. Rejects ".." (the writer
/// doesn't support relative-up references) and the empty string.
fn normalise_path(p: &str) -> Result<String> {
    if p.is_empty() {
        return Err(crate::Error::InvalidArgument("squashfs: empty path".into()));
    }
    let mut parts: Vec<&str> = Vec::new();
    for c in p.split('/') {
        match c {
            "" | "." => continue,
            ".." => {
                return Err(crate::Error::InvalidArgument(
                    "squashfs: '..' not allowed in writer paths".into(),
                ));
            }
            other => parts.push(other),
        }
    }
    if parts.is_empty() {
        Ok("/".to_string())
    } else {
        Ok(format!("/{}", parts.join("/")))
    }
}

/// Return the parent directory of `p`. For "/foo" returns "/"; for "/"
/// returns "/".
fn parent_path(p: &str) -> String {
    if p == "/" {
        return "/".into();
    }
    let trimmed = p.trim_end_matches('/');
    match trimmed.rsplit_once('/') {
        Some(("", _)) => "/".into(),
        Some((parent, _)) => parent.into(),
        None => "/".into(),
    }
}

/// Return the leaf name (final path component) of `p`. For "/" returns "".
fn leaf_name(p: &str) -> &str {
    let trimmed = p.trim_end_matches('/');
    match trimmed.rsplit_once('/') {
        Some((_, leaf)) => leaf,
        None => trimmed,
    }
}

/// Count the number of direct subdirectories listed under `path`, used
/// to derive a directory inode's `link_count` (POSIX convention:
/// `2 + subdir_count`, where the "2" accounts for "." and "..").
fn count_subdirs(listings: &BTreeMap<String, Vec<(String, String, u16)>>, path: &str) -> usize {
    listings
        .get(path)
        .map(|v| v.iter().filter(|(_, _, k)| *k == INODE_BASIC_DIR).count())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn normalise_path_handles_common_inputs() {
        assert_eq!(normalise_path("/").unwrap(), "/");
        assert_eq!(normalise_path("/a").unwrap(), "/a");
        assert_eq!(normalise_path("/a/").unwrap(), "/a");
        assert_eq!(normalise_path("/a//b/./c/").unwrap(), "/a/b/c");
        assert!(normalise_path("").is_err());
        assert!(normalise_path("/a/../b").is_err());
    }

    #[test]
    fn parent_and_leaf_round_trip() {
        assert_eq!(parent_path("/"), "/");
        assert_eq!(parent_path("/a"), "/");
        assert_eq!(parent_path("/a/b"), "/a");
        assert_eq!(parent_path("/a/b/c"), "/a/b");
        assert_eq!(leaf_name("/"), "");
        assert_eq!(leaf_name("/a"), "a");
        assert_eq!(leaf_name("/a/b"), "b");
    }

    /// Cover the full-block path: a file larger than `block_size` causes
    /// the writer to emit at least one full block of file data followed
    /// by a fragment tail. The reader reassembles both.
    #[test]
    fn write_then_read_multi_block_file() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        // Use 4 KiB blocks so we exercise the multi-block path without
        // allocating megabytes of test fixture.
        let mut state = WriteState::new(4096, Compression::Unknown(0));
        // 4096 + 1234 bytes = one full block + a tail.
        let mut payload = Vec::with_capacity(4096 + 1234);
        for i in 0..(4096 + 1234) {
            payload.push((i % 251) as u8);
        }
        state
            .create_file(
                "/big.bin",
                FileSource::Reader {
                    reader: Box::new(std::io::Cursor::new(payload.clone())),
                    len: payload.len() as u64,
                },
                EntryMeta::default(),
                Vec::new(),
            )
            .unwrap();
        state.flush(&mut dev).unwrap();
        let s = super::super::Squashfs::open(&mut dev).unwrap();
        let mut r = s.open_file_reader(&mut dev, "/big.bin").unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut buf).unwrap();
        drop(r);
        assert_eq!(buf, payload);
    }

    #[test]
    fn write_then_read_minimal_image() {
        // One regular file, one symlink, one nested dir.
        let mut dev = MemoryBackend::new(1024 * 1024);
        let mut state = WriteState::new(DEFAULT_BLOCK_SIZE, Compression::Unknown(0));
        state
            .create_dir(
                "/etc",
                EntryMeta {
                    mode: 0o755,
                    uid: 0,
                    gid: 0,
                    mtime: 12345,
                },
                Vec::new(),
            )
            .unwrap();
        state
            .create_file(
                "/etc/hello.txt",
                FileSource::Reader {
                    reader: Box::new(std::io::Cursor::new(b"hello".to_vec())),
                    len: 5,
                },
                EntryMeta {
                    mode: 0o644,
                    uid: 1000,
                    gid: 1000,
                    mtime: 12345,
                },
                vec![Xattr {
                    key: "user.color".into(),
                    value: b"orange".to_vec(),
                }],
            )
            .unwrap();
        state
            .create_symlink(
                "/lnk",
                "etc/hello.txt",
                EntryMeta {
                    mode: 0o777,
                    uid: 0,
                    gid: 0,
                    mtime: 0,
                },
                Vec::new(),
            )
            .unwrap();
        let sb = state.flush(&mut dev).unwrap();
        assert_eq!(sb.magic, super::super::SQUASHFS_MAGIC);
        assert_eq!(sb.major, 4);
        // 4 inodes: root, /etc, /etc/hello.txt, /lnk.
        assert_eq!(sb.inode_count, 4);
        // Re-open and walk.
        let s = super::super::Squashfs::open(&mut dev).unwrap();
        let entries = s.list_path(&mut dev, "/").unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"etc"));
        assert!(names.contains(&"lnk"));
        let etc = s.list_path(&mut dev, "/etc").unwrap();
        assert_eq!(etc.len(), 1);
        assert_eq!(etc[0].name, "hello.txt");
        let mut r = s.open_file_reader(&mut dev, "/etc/hello.txt").unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut buf).unwrap();
        drop(r);
        assert_eq!(buf, b"hello");
        let tgt = s.read_symlink(&mut dev, "/lnk").unwrap();
        assert_eq!(tgt, "etc/hello.txt");
    }
}
