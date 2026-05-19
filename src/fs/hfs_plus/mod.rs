//! HFS+ — Apple's legacy macOS filesystem (pre-2017). Read-only support.
//!
//! ## Status
//!
//! Read-only v1.1: open, list directories, stream regular file
//! contents, read symbolic-link targets, and resolve hard links to
//! their underlying indirect-node iNodes. Files whose data fork spills
//! beyond eight inline extents are read through the extents-overflow
//! B-tree.
//!
//! Implementation is based on Apple Technical Note TN1150, the
//! canonical public reference for HFS+ on-disk structures.
//!
//! ## Scope and deferred features
//!
//! * Write support is out of scope entirely.
//! * No journal replay — the on-disk catalog is read as-is.
//! * HFSX case-sensitive comparison is honoured at the catalog-key
//!   level; non-ASCII case folding follows a simplified table.
//! * Extended-attribute B-tree (`attributesFile`) is not parsed.
//!
//! ## Module layout
//!
//! * [`volume_header`] — the 512-byte `HFSPlusVolumeHeader` at offset 1024.
//! * [`btree`] — node descriptors, header records, record-offset tables.
//! * [`catalog`] — catalog keys, leaf records, lookup.
//! * [`extents`] — extents-overflow B-tree walker for spilled extents.

pub mod btree;
pub mod catalog;
pub mod extents;
pub mod volume_header;
pub mod writer;

pub use writer::FormatOpts;

use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;

use btree::ForkReader;
use catalog::{Catalog, CatalogFile, CatalogKey, CatalogRecord, ROOT_FOLDER_ID, UniStr, mode};
use extents::ExtentsOverflow;
use volume_header::{ForkData, VolumeHeader, read_volume_header};

/// Maximum bytes we'll ever read from a symlink's data fork; symlinks
/// are tiny path strings and a sane filesystem stays well under this.
const SYMLINK_MAX_BYTES: u64 = 4096;

/// An opened HFS+ volume.
pub struct HfsPlus {
    volume_header: VolumeHeader,
    catalog: Catalog,
    /// Extents-overflow B-tree, opened lazily-but-once at mount time.
    /// Absent only if the volume header has no extents-overflow file,
    /// which is unusual but technically permitted.
    overflow: Option<ExtentsOverflow>,
    /// CNID of the HFS+ private data directory (where `iNode<N>`
    /// indirect-node files live), if it exists on this volume.
    /// Resolved lazily on first hard-link encounter.
    private_dir_cnid: std::cell::Cell<Option<u32>>,
    /// `true` once we've attempted to resolve `private_dir_cnid`. We
    /// cache the absent result too so we don't re-scan every time.
    private_dir_resolved: std::cell::Cell<bool>,
    /// Cached volume name, taken from the root folder's thread record.
    volume_name: String,
    /// Writer state, present only while a freshly formatted volume
    /// (or one re-opened for mutation) is being built. `None` for
    /// read-only opens.
    writer: Option<writer::Writer>,
}

impl HfsPlus {
    /// Open an existing HFS+ volume on `dev`.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let vh = read_volume_header(dev)?;
        let case_sensitive = vh.is_hfsx();

        let cat_fork = ForkReader::from_inline(&vh.catalog_file, vh.block_size, "catalog")?;
        let catalog = Catalog::open(dev, cat_fork, case_sensitive)?;

        // Open the extents-overflow file if its fork has any blocks.
        // It can be empty on a fresh volume.
        let overflow = if vh.extents_file.total_blocks > 0 {
            let ext_fork =
                ForkReader::from_inline(&vh.extents_file, vh.block_size, "extents-overflow")?;
            Some(ExtentsOverflow::open(dev, ext_fork)?)
        } else {
            None
        };

        // The root folder's thread record is keyed by (ROOT_FOLDER_ID, "");
        // its name field is the volume name.
        let volume_name = lookup_thread_name(dev, &catalog, ROOT_FOLDER_ID)?
            .unwrap_or_else(|| "Untitled".to_string());

        Ok(Self {
            volume_header: vh,
            catalog,
            overflow,
            private_dir_cnid: std::cell::Cell::new(None),
            private_dir_resolved: std::cell::Cell::new(false),
            volume_name,
            writer: None,
        })
    }

    /// Format `dev` as a fresh HFS+ volume and return a handle that
    /// accepts subsequent `create_*` calls. Call [`flush`](Self::flush)
    /// when done to persist the catalog, bitmap, and volume header.
    pub fn format(dev: &mut dyn BlockDevice, opts: &writer::FormatOpts) -> Result<Self> {
        let (vh, w) = writer::format(dev, opts)?;
        let case_sensitive = vh.is_hfsx();
        let mut w = w;
        let mut vh_mut = vh.clone();
        writer::flush(&mut w, &mut vh_mut, dev)?;
        w.flushed = false;

        let cat_fork = ForkReader::from_inline(&vh_mut.catalog_file, vh_mut.block_size, "catalog")?;
        let catalog = Catalog::open(dev, cat_fork, case_sensitive)?;
        let overflow = if vh_mut.extents_file.total_blocks > 0 {
            let ext_fork = ForkReader::from_inline(
                &vh_mut.extents_file,
                vh_mut.block_size,
                "extents-overflow",
            )?;
            Some(ExtentsOverflow::open(dev, ext_fork)?)
        } else {
            None
        };
        let volume_name = w.volume_name.clone();
        Ok(Self {
            volume_header: vh_mut,
            catalog,
            overflow,
            private_dir_cnid: std::cell::Cell::new(None),
            private_dir_resolved: std::cell::Cell::new(false),
            volume_name,
            writer: Some(w),
        })
    }

    /// Create a directory at the given absolute path.
    pub fn create_dir(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &str,
        mode: u16,
        uid: u32,
        gid: u32,
    ) -> Result<u32> {
        let (parent_id, name) = self.resolve_create_target(path)?;
        let w = self
            .writer
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: volume is read-only".into()))?;
        let cnid = w.next_cnid;
        w.next_cnid = w
            .next_cnid
            .checked_add(1)
            .ok_or_else(|| crate::Error::Unsupported("hfs+: CNID space exhausted".into()))?;
        writer::insert_folder(w, parent_id, &name, cnid, mode, uid, gid)?;
        Ok(cnid)
    }

    /// Create a regular file at the given absolute path, streaming
    /// `len` bytes from `src` into freshly allocated allocation blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn create_file<R: std::io::Read>(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        src: &mut R,
        len: u64,
        mode: u16,
        uid: u32,
        gid: u32,
    ) -> Result<u32> {
        let (parent_id, name) = self.resolve_create_target(path)?;
        let w = self
            .writer
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: volume is read-only".into()))?;
        let cnid = w.next_cnid;
        w.next_cnid = w
            .next_cnid
            .checked_add(1)
            .ok_or_else(|| crate::Error::Unsupported("hfs+: CNID space exhausted".into()))?;
        let fork = writer::stream_data_to_blocks(w, dev, src, len)?;
        writer::insert_file(
            w,
            parent_id,
            &name,
            cnid,
            mode,
            uid,
            gid,
            *b"\0\0\0\0",
            *b"\0\0\0\0",
            &fork,
            false,
        )?;
        Ok(cnid)
    }

    /// Create a symlink at `path` whose target is the UTF-8 byte string
    /// `target`. Finder type `slnk` / creator `rhap`, mode S_IFLNK.
    pub fn create_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        target: &str,
        mode: u16,
        uid: u32,
        gid: u32,
    ) -> Result<u32> {
        let (parent_id, name) = self.resolve_create_target(path)?;
        let w = self
            .writer
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: volume is read-only".into()))?;
        let cnid = w.next_cnid;
        w.next_cnid = w
            .next_cnid
            .checked_add(1)
            .ok_or_else(|| crate::Error::Unsupported("hfs+: CNID space exhausted".into()))?;
        let fork = writer::write_inline_data(w, dev, target.as_bytes())?;
        writer::insert_file(
            w, parent_id, &name, cnid, mode, uid, gid, *b"slnk", *b"rhap", &fork, true,
        )?;
        Ok(cnid)
    }

    /// Remove a file, symlink, or empty directory at `path`.
    pub fn remove(&mut self, _dev: &mut dyn BlockDevice, path: &str) -> Result<()> {
        let (parent_id, name) = self.resolve_create_target(path)?;
        let w = self
            .writer
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: volume is read-only".into()))?;
        writer::remove_entry(w, parent_id, &name)
    }

    #[doc(hidden)]
    #[cfg(test)]
    pub fn test_writer(&self) -> Option<&writer::Writer> {
        self.writer.as_ref()
    }

    /// Persist in-memory writer state (bitmap, catalog tree, volume
    /// header) to disk. Idempotent; no-op on read-only handles.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        let Some(w) = self.writer.as_mut() else {
            return Ok(());
        };
        writer::flush(w, &mut self.volume_header, dev)?;
        let case_sensitive = self.volume_header.is_hfsx();
        let cat_fork = ForkReader::from_inline(
            &self.volume_header.catalog_file,
            self.volume_header.block_size,
            "catalog",
        )?;
        self.catalog = Catalog::open(dev, cat_fork, case_sensitive)?;
        w.flushed = false;
        Ok(())
    }

    /// Split `path` into its parent CNID and final name component,
    /// resolving all intermediates against the in-memory catalog.
    fn resolve_create_target(&self, path: &str) -> Result<(u32, UniStr)> {
        let parts = split_path(path);
        if parts.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "hfs+: cannot create at the root path".into(),
            ));
        }
        let (last, prefix) = parts.split_last().unwrap();
        let mut cnid = ROOT_FOLDER_ID;
        let w = self
            .writer
            .as_ref()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: volume is read-only".into()))?;
        for part in prefix {
            let name = UniStr::from_str_lossy(part);
            let (_, child_cnid, rec_type) = w.lookup(cnid, &name).ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "hfs+: parent component {part:?} does not exist"
                ))
            })?;
            if rec_type != catalog::REC_FOLDER {
                return Err(crate::Error::InvalidArgument(format!(
                    "hfs+: component {part:?} is not a directory"
                )));
            }
            cnid = child_cnid;
        }
        Ok((cnid, UniStr::from_str_lossy(last)))
    }

    /// Total byte capacity advertised by the volume header.
    pub fn total_bytes(&self) -> u64 {
        u64::from(self.volume_header.total_blocks) * u64::from(self.volume_header.block_size)
    }

    /// Allocation block size in bytes.
    pub fn block_size(&self) -> u32 {
        self.volume_header.block_size
    }

    /// Cached volume name (from the root thread record).
    pub fn volume_name(&self) -> &str {
        &self.volume_name
    }

    /// List the entries of a directory by absolute path. Returns one
    /// [`crate::fs::DirEntry`] per child, with `inode` set to the
    /// HFS+ CNID (catalog node id) — which is the closest analogue
    /// to a Unix inode number on this filesystem. Hard-link source
    /// entries are reported with the underlying iNode's `kind`
    /// (typically `Regular`) so callers see a uniform view.
    pub fn list_path(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let cnid = self.resolve_dir(dev, path)?;
        self.list_cnid(dev, cnid)
    }

    /// Open a regular file by absolute path for streaming reads.
    /// Transparently follows hard links (`hlnk`/`hfs+` indirect node
    /// entries) to the actual iNode in the HFS+ private-data directory.
    /// Returns `Unsupported` for symlinks — use
    /// [`read_symlink_target_path`](Self::read_symlink_target_path)
    /// for those.
    pub fn open_file_reader<'a>(
        &self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<HfsPlusFileReader<'a>> {
        let rec = self.lookup_path(dev, path)?;
        let file = match rec {
            CatalogRecord::File(f) => f,
            CatalogRecord::Folder(_) => {
                return Err(crate::Error::InvalidArgument(format!(
                    "hfs+: {path:?} is a directory, not a file"
                )));
            }
            CatalogRecord::Thread(_) => {
                return Err(crate::Error::InvalidImage(format!(
                    "hfs+: path {path:?} resolved to a thread record"
                )));
            }
        };
        let file = self.resolve_hard_link(dev, file)?;
        let mode_bits = file.bsd.file_mode & mode::S_IFMT;
        if file.is_symlink() || mode_bits == mode::S_IFLNK {
            return Err(crate::Error::InvalidArgument(format!(
                "hfs+: {path:?} is a symlink; use read_symlink_target_path"
            )));
        }
        if mode_bits != 0 && mode_bits != mode::S_IFREG {
            return Err(crate::Error::Unsupported(format!(
                "hfs+: {path:?} is not a regular file (mode {:#06o})",
                file.bsd.file_mode
            )));
        }
        let fork = self.open_data_fork(dev, &file)?;
        Ok(HfsPlusFileReader {
            dev,
            fork,
            remaining: file.data_fork.logical_size,
            position: 0,
        })
    }

    /// Read a symlink's target by absolute path. Returns the raw
    /// UTF-8 bytes stored in the file's data fork as a `String`
    /// (HFS+ symlinks are conventionally stored as UTF-8 path text
    /// without a NUL terminator).
    pub fn read_symlink_target_path(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<String> {
        let rec = self.lookup_path(dev, path)?;
        let file = match rec {
            CatalogRecord::File(f) => f,
            _ => {
                return Err(crate::Error::InvalidArgument(format!(
                    "hfs+: {path:?} is not a file (cannot be a symlink)"
                )));
            }
        };
        let file = self.resolve_hard_link(dev, file)?;
        self.read_symlink_target_inner(dev, &file, path)
    }

    /// Read a symlink's target by CNID. Useful when the caller has
    /// already discovered the target file ID via [`Self::list_path`] but
    /// doesn't have a stable path (e.g. multiple parents). Returns
    /// the link target as a UTF-8 string.
    pub fn read_symlink_target(&self, dev: &mut dyn BlockDevice, cnid: u32) -> Result<String> {
        let file = self.lookup_file_by_cnid(dev, cnid)?;
        let file = self.resolve_hard_link(dev, file)?;
        self.read_symlink_target_inner(dev, &file, &format!("CNID {cnid}"))
    }

    /// Total byte length of a regular file (after hard-link resolution),
    /// for callers that want to size a buffer before streaming.
    pub fn file_size(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<u64> {
        let rec = self.lookup_path(dev, path)?;
        let file = match rec {
            CatalogRecord::File(f) => f,
            _ => {
                return Err(crate::Error::InvalidArgument(format!(
                    "hfs+: {path:?} is not a file"
                )));
            }
        };
        let file = self.resolve_hard_link(dev, file)?;
        Ok(file.data_fork.logical_size)
    }

    // -- internal helpers ----------------------------------------------

    /// Walk the path one component at a time, returning the CNID of
    /// the *directory* it names. `"/"` and `""` both name the root.
    fn resolve_dir(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<u32> {
        let parts = split_path(path);
        let mut cnid = ROOT_FOLDER_ID;
        for part in parts {
            let rec = self.lookup_component(dev, cnid, part)?;
            match rec {
                CatalogRecord::Folder(f) => {
                    cnid = f.folder_id;
                }
                CatalogRecord::File(_) => {
                    return Err(crate::Error::InvalidArgument(format!(
                        "hfs+: {part:?} is not a directory (in {path:?})"
                    )));
                }
                CatalogRecord::Thread(_) => {
                    return Err(crate::Error::InvalidImage(format!(
                        "hfs+: lookup of {part:?} returned a thread record"
                    )));
                }
            }
        }
        Ok(cnid)
    }

    /// Walk the path and return the leaf record (file or folder).
    fn lookup_path(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<CatalogRecord> {
        let parts = split_path(path);
        if parts.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "hfs+: cannot resolve root \"/\" as a leaf entry".into(),
            ));
        }
        let (last, prefix) = parts.split_last().unwrap();
        let mut cnid = ROOT_FOLDER_ID;
        for part in prefix {
            let rec = self.lookup_component(dev, cnid, part)?;
            match rec {
                CatalogRecord::Folder(f) => cnid = f.folder_id,
                _ => {
                    return Err(crate::Error::InvalidArgument(format!(
                        "hfs+: {part:?} is not a directory (in {path:?})"
                    )));
                }
            }
        }
        self.lookup_component(dev, cnid, last)
    }

    /// Look up the immediate child named `name` under directory `parent_id`.
    fn lookup_component(
        &self,
        dev: &mut dyn BlockDevice,
        parent_id: u32,
        name: &str,
    ) -> Result<CatalogRecord> {
        let key = CatalogKey {
            parent_id,
            name: UniStr::from_str_lossy(name),
            encoded_len: 0,
        };
        self.catalog.lookup(dev, &key)?.ok_or_else(|| {
            crate::Error::InvalidArgument(format!(
                "hfs+: no such entry {name:?} under CNID {parent_id}"
            ))
        })
    }

    /// Look up a catalog *file* by its CNID. Two-step: first read the
    /// file's thread record (key = (cnid, "")) to recover its
    /// `(parent_id, name)`, then look up the actual record.
    fn lookup_file_by_cnid(&self, dev: &mut dyn BlockDevice, cnid: u32) -> Result<CatalogFile> {
        let thread_key = CatalogKey {
            parent_id: cnid,
            name: UniStr::default(),
            encoded_len: 0,
        };
        let (parent_id, name) = match self.catalog.lookup(dev, &thread_key)? {
            Some(CatalogRecord::Thread(t)) => (t.parent_id, t.name),
            Some(_) => {
                return Err(crate::Error::InvalidImage(format!(
                    "hfs+: CNID {cnid} thread key did not return a thread record"
                )));
            }
            None => {
                return Err(crate::Error::InvalidArgument(format!(
                    "hfs+: no thread record for CNID {cnid}"
                )));
            }
        };
        let child_key = CatalogKey {
            parent_id,
            name,
            encoded_len: 0,
        };
        match self.catalog.lookup(dev, &child_key)? {
            Some(CatalogRecord::File(f)) => Ok(f),
            Some(CatalogRecord::Folder(_)) => Err(crate::Error::InvalidArgument(format!(
                "hfs+: CNID {cnid} names a folder, not a file"
            ))),
            Some(CatalogRecord::Thread(_)) => Err(crate::Error::InvalidImage(format!(
                "hfs+: CNID {cnid} reverse-lookup returned a thread record"
            ))),
            None => Err(crate::Error::InvalidArgument(format!(
                "hfs+: no catalog file for CNID {cnid}"
            ))),
        }
    }

    /// If `file` is a hard-link indirect-node entry (`hlnk`/`hfs+`),
    /// resolve it to the underlying iNode file in the HFS+ private
    /// data directory. Otherwise return the file unchanged.
    fn resolve_hard_link(
        &self,
        dev: &mut dyn BlockDevice,
        file: CatalogFile,
    ) -> Result<CatalogFile> {
        if !file.is_hard_link() {
            return Ok(file);
        }
        let private_dir = match self.private_data_dir(dev)? {
            Some(cnid) => cnid,
            None => {
                return Err(crate::Error::InvalidImage(
                    "hfs+: hard-link record but no '\\0\\0\\0\\0HFS+ Private Data' \
                     directory found"
                        .into(),
                ));
            }
        };
        // The iNode number lives in BsdInfo.special on a hlnk record.
        let inode_id = file.bsd.special;
        let name = format!("iNode{inode_id}");
        let rec = self.lookup_component(dev, private_dir, &name)?;
        match rec {
            CatalogRecord::File(f) => {
                // Defence in depth: forbid recursive hard links.
                if f.is_hard_link() {
                    return Err(crate::Error::InvalidImage(format!(
                        "hfs+: iNode{inode_id} is itself a hard-link indirection"
                    )));
                }
                Ok(f)
            }
            _ => Err(crate::Error::InvalidImage(format!(
                "hfs+: iNode{inode_id} is not a regular file record"
            ))),
        }
    }

    /// Resolve, cache, and return the CNID of the HFS+ private data
    /// directory (Apple's well-known name with four leading NUL
    /// code units followed by "HFS+ Private Data"). Returns `None`
    /// on a volume that has no hard links — the directory only exists
    /// when at least one hard link has ever been created.
    fn private_data_dir(&self, dev: &mut dyn BlockDevice) -> Result<Option<u32>> {
        if self.private_dir_resolved.get() {
            return Ok(self.private_dir_cnid.get());
        }
        // UniStr cannot easily be built from a Rust string containing
        // null code units, so build it directly.
        let mut code_units: Vec<u16> = vec![0, 0, 0, 0];
        code_units.extend("HFS+ Private Data".encode_utf16());
        let key = CatalogKey {
            parent_id: ROOT_FOLDER_ID,
            name: UniStr { code_units },
            encoded_len: 0,
        };
        let cnid = match self.catalog.lookup(dev, &key)? {
            Some(CatalogRecord::Folder(f)) => Some(f.folder_id),
            _ => None,
        };
        self.private_dir_cnid.set(cnid);
        self.private_dir_resolved.set(true);
        Ok(cnid)
    }

    /// Build a `ForkReader` for `file`'s data fork, pulling extra
    /// extents from the extents-overflow B-tree if the inline eight
    /// are insufficient.
    fn open_data_fork(&self, dev: &mut dyn BlockDevice, file: &CatalogFile) -> Result<ForkReader> {
        self.open_fork(
            dev,
            &file.data_fork,
            file.file_id,
            extents::FORK_DATA,
            "data",
        )
    }

    fn open_fork(
        &self,
        dev: &mut dyn BlockDevice,
        fork: &ForkData,
        file_id: u32,
        fork_type: u8,
        what: &str,
    ) -> Result<ForkReader> {
        if u64::from(fork.total_blocks) <= fork.inline_blocks() {
            return ForkReader::from_inline(fork, self.volume_header.block_size, what);
        }
        let overflow = self.overflow.as_ref().ok_or_else(|| {
            crate::Error::InvalidImage(
                "hfs+: fork needs overflow extents but volume has no \
                 extents-overflow file"
                    .into(),
            )
        })?;
        let first_overflow_block = u32::try_from(fork.inline_blocks()).map_err(|_| {
            crate::Error::InvalidImage("hfs+: inline fork block count overflows u32".into())
        })?;
        let extra = overflow.find_extents(dev, file_id, fork_type, first_overflow_block)?;
        ForkReader::from_inline_plus_overflow(fork, &extra, self.volume_header.block_size, what)
    }

    /// Read up to `SYMLINK_MAX_BYTES` from a symlink's data fork and
    /// return the result as a Rust string. `descriptor` is a human-
    /// readable identifier (path or CNID) used only in error text.
    fn read_symlink_target_inner(
        &self,
        dev: &mut dyn BlockDevice,
        file: &CatalogFile,
        descriptor: &str,
    ) -> Result<String> {
        let mode_bits = file.bsd.file_mode & mode::S_IFMT;
        let by_mode = mode_bits == mode::S_IFLNK;
        let by_finder = file.is_symlink();
        if !by_mode && !by_finder {
            return Err(crate::Error::InvalidArgument(format!(
                "hfs+: {descriptor} is not a symlink (mode {:#06o}, \
                 FileInfo type {:?}, creator {:?})",
                file.bsd.file_mode,
                bytes_to_osstr(&file.file_type),
                bytes_to_osstr(&file.creator),
            )));
        }
        let len = file.data_fork.logical_size;
        if len == 0 {
            return Ok(String::new());
        }
        if len > SYMLINK_MAX_BYTES {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: {descriptor} symlink target too large ({len} bytes, \
                 max {SYMLINK_MAX_BYTES})"
            )));
        }
        let fork = self.open_data_fork(dev, file)?;
        let mut buf = vec![0u8; len as usize];
        fork.read(dev, 0, &mut buf)?;
        // HFS+ symlinks are UTF-8 path strings. Accept a trailing NUL
        // (some writers add one).
        if buf.last() == Some(&0) {
            buf.pop();
        }
        String::from_utf8(buf).map_err(|e| {
            crate::Error::InvalidImage(format!(
                "hfs+: {descriptor} symlink target is not valid UTF-8: {e}"
            ))
        })
    }

    /// Enumerate the direct children of folder `cnid` by scanning
    /// every leaf node of the catalog from the first leaf onwards
    /// and collecting entries whose key.parentID matches.
    fn list_cnid(&self, dev: &mut dyn BlockDevice, cnid: u32) -> Result<Vec<crate::fs::DirEntry>> {
        use crate::fs::{DirEntry as FsDirEntry, EntryKind};

        let mut out = Vec::new();
        let node_size = u32::from(self.catalog.header.node_size);
        let mut node_idx = self.catalog.header.first_leaf_node;
        while node_idx != 0 {
            let node = btree::read_node(dev, &self.catalog.fork, node_idx, node_size)?;
            let desc = btree::NodeDescriptor::decode(&node)?;
            if desc.kind != btree::KIND_LEAF {
                return Err(crate::Error::InvalidImage(format!(
                    "hfs+: leaf chain node {node_idx} has non-leaf kind {}",
                    desc.kind
                )));
            }
            let offs = btree::record_offsets(&node, desc.num_records)?;

            let mut passed_parent = false;
            for i in 0..desc.num_records as usize {
                let rec = btree::record_bytes(&node, &offs, i);
                let key = CatalogKey::decode(rec)?;
                use std::cmp::Ordering as O;
                match key.parent_id.cmp(&cnid) {
                    O::Less => continue,
                    O::Greater => {
                        passed_parent = true;
                        break;
                    }
                    O::Equal => {}
                }
                let body_start = align2(key.encoded_len);
                if body_start > rec.len() {
                    return Err(crate::Error::InvalidImage(
                        "hfs+: catalog key overruns record".into(),
                    ));
                }
                let body = &rec[body_start..];
                let record = CatalogRecord::decode(body)?;
                // Skip threads — they're metadata, not real children.
                let (kind, child_id) = match &record {
                    CatalogRecord::Folder(f) => (EntryKind::Dir, f.folder_id),
                    CatalogRecord::File(f) => {
                        // Hard-link entries: resolve the kind by
                        // peeking at the iNode so callers see a
                        // unified view. The link itself is the
                        // `child_id` we expose, matching what the
                        // catalog records.
                        if f.is_hard_link() {
                            let resolved_kind = self
                                .resolve_hard_link(dev, f.clone())
                                .map(|r| file_kind(&r))
                                .unwrap_or(EntryKind::Unknown);
                            (resolved_kind, f.file_id)
                        } else if f.is_symlink() {
                            (EntryKind::Symlink, f.file_id)
                        } else {
                            (file_kind(f), f.file_id)
                        }
                    }
                    CatalogRecord::Thread(_) => continue,
                };
                out.push(FsDirEntry {
                    name: key.name.to_string_lossy(),
                    inode: child_id,
                    kind,
                });
            }
            if passed_parent {
                break;
            }
            node_idx = desc.f_link;
        }
        Ok(out)
    }
}

/// Streaming reader for a regular HFS+ file. Holds the data-fork
/// extent list in `ForkReader` and walks it on demand from the
/// borrowed [`BlockDevice`].
pub struct HfsPlusFileReader<'a> {
    dev: &'a mut dyn BlockDevice,
    fork: ForkReader,
    /// Bytes of the file still to return.
    remaining: u64,
    /// Current fork-relative byte position.
    position: u64,
}

impl<'a> Read for HfsPlusFileReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 || buf.is_empty() {
            return Ok(0);
        }
        let want = (buf.len() as u64).min(self.remaining) as usize;
        self.fork
            .read(self.dev, self.position, &mut buf[..want])
            .map_err(std::io::Error::other)?;
        self.position += want as u64;
        self.remaining -= want as u64;
        Ok(want)
    }
}

/// Probe for the HFS+ volume-header signature `"H+"` (or `"HX"` for HFSX)
/// at offset 1024 of the volume.
pub fn probe(dev: &mut dyn BlockDevice) -> Result<bool> {
    if dev.total_size() < 1024 + 2 {
        return Ok(false);
    }
    let mut sig = [0u8; 2];
    dev.read_at(1024, &mut sig)?;
    Ok(&sig == b"H+" || &sig == b"HX")
}

// -- helpers --------------------------------------------------------------

/// Classify a `CatalogFile` into a [`crate::fs::EntryKind`] using
/// its BSD `file_mode` bits when set, falling back to "regular".
fn file_kind(f: &CatalogFile) -> crate::fs::EntryKind {
    use crate::fs::EntryKind;
    match f.bsd.file_mode & mode::S_IFMT {
        mode::S_IFREG => EntryKind::Regular,
        mode::S_IFDIR => EntryKind::Dir, // unusual on a file record, but be defensive
        mode::S_IFLNK => EntryKind::Symlink,
        mode::S_IFCHR => EntryKind::Char,
        mode::S_IFBLK => EntryKind::Block,
        mode::S_IFIFO => EntryKind::Fifo,
        mode::S_IFSOCK => EntryKind::Socket,
        0 => EntryKind::Regular,
        _ => EntryKind::Unknown,
    }
}

/// Read the thread record for `cnid`, returning the node name. Used
/// to discover the volume name from the root folder's thread.
fn lookup_thread_name(
    dev: &mut dyn BlockDevice,
    catalog: &Catalog,
    cnid: u32,
) -> Result<Option<String>> {
    let key = CatalogKey {
        parent_id: cnid,
        name: UniStr::default(),
        encoded_len: 0,
    };
    match catalog.lookup(dev, &key)? {
        Some(CatalogRecord::Thread(t)) => Ok(Some(t.name.to_string_lossy())),
        _ => Ok(None),
    }
}

/// Render an OSType 4-byte tag for diagnostics, preferring ASCII
/// when the tag is human-readable and falling back to hex otherwise.
fn bytes_to_osstr(b: &[u8; 4]) -> String {
    if b.iter().all(|&c| (0x20..=0x7E).contains(&c)) {
        format!("'{}'", String::from_utf8_lossy(b))
    } else {
        format!("0x{:02x}{:02x}{:02x}{:02x}", b[0], b[1], b[2], b[3])
    }
}

fn align2(n: usize) -> usize {
    n + (n & 1)
}

fn split_path(path: &str) -> Vec<&str> {
    path.split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect()
}

// -- tests ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn probe_recognises_h_plus_signature() {
        let mut dev = MemoryBackend::new(8192);
        dev.write_at(1024, b"H+").unwrap();
        assert!(probe(&mut dev).unwrap());
    }

    #[test]
    fn probe_recognises_hx_signature() {
        let mut dev = MemoryBackend::new(8192);
        dev.write_at(1024, b"HX").unwrap();
        assert!(probe(&mut dev).unwrap());
    }

    #[test]
    fn probe_rejects_unknown_signature() {
        let mut dev = MemoryBackend::new(8192);
        dev.write_at(1024, b"NO").unwrap();
        assert!(!probe(&mut dev).unwrap());
    }

    #[test]
    fn open_fails_on_garbage() {
        let mut dev = MemoryBackend::new(8192);
        // Default zeros => signature 0x0000, not H+ or HX.
        assert!(HfsPlus::open(&mut dev).is_err());
    }
}
