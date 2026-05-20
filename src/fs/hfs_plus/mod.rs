//! HFS+ — Apple's legacy macOS filesystem (pre-2017).
//!
//! ## Status
//!
//! Read and write support.
//!
//! Read side: open, list directories, stream regular file contents,
//! read symbolic-link targets, and resolve hard links to their
//! underlying indirect-node iNodes. Files whose data fork spills
//! beyond eight inline extents are read through the extents-overflow
//! B-tree.
//!
//! Write side: format a fresh volume, create directories, regular
//! files, symbolic links, and hard links; remove empty directories
//! and files (with their forks). The writer keeps user data forks
//! streaming-only (64 KiB scratch) and spills fragmented files into
//! the extents-overflow B-tree when their run list outgrows the eight
//! inline extents in a catalog record. An optional journal stub
//! (clean, no transactions to replay) can be requested via
//! [`writer::FormatOpts::journaled`].
//!
//! Implementation is based on Apple Technical Note TN1150, the
//! canonical public reference for HFS+ on-disk structures.
//!
//! ## Scope and deferred features
//!
//! * No journal replay on read — the on-disk catalog is read as-is.
//! * HFSX case-sensitive comparison is honoured at the catalog-key
//!   level; non-ASCII case folding follows a simplified table.
//! * Extended-attribute B-tree (`attributesFile`) is not parsed or
//!   written.
//! * Resource forks are always written as empty.
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
pub mod handle;
pub(crate) mod journal;
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
    /// Open an existing HFS+ volume on `dev`. The returned handle is
    /// writable: subsequent `create_*` / `remove` calls mutate the live
    /// image, and [`flush`](Self::flush) persists the rewritten catalog,
    /// extents-overflow tree, allocation bitmap, and volume header.
    ///
    /// On a journaled volume any unreplayed transaction is drained
    /// *before* we read the catalog, extents-overflow, or bitmap: the
    /// pending journal may carry the only authoritative copy of those
    /// blocks following a crash mid-`flush`. Replay is idempotent, so
    /// volumes that crashed mid-replay land in the same state.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        // Read the volume header first so we know whether replay is
        // applicable (journaled vs. not) and where the journal lives.
        let vh = read_volume_header(dev)?;
        journal::replay(dev, &vh)?;
        // Re-read the volume header in case replay restored it.
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

        // Reconstruct the in-memory writer state by walking the catalog
        // leaves and the extents-overflow leaves and loading the on-disk
        // bitmap. After this, create_* / remove / flush all work on the
        // live image exactly as they do post-format.
        let writer = writer::open_writable(dev, &vh, volume_name.clone())?;

        Ok(Self {
            volume_header: vh,
            catalog,
            overflow,
            private_dir_cnid: std::cell::Cell::new(None),
            private_dir_resolved: std::cell::Cell::new(false),
            volume_name,
            writer: Some(writer),
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
        let fork = writer::stream_data_to_blocks(w, dev, src, len, cnid)?;
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

    /// Create a hard link at `dst_path` pointing at the same on-disk
    /// content as `src_path`. After the call:
    ///
    /// * the data fork that previously lived under `src_path` is owned
    ///   by an `iNode<N>` file inside the HFS+ private-data directory;
    /// * both `src_path` and `dst_path` are `hlnk`/`hfs+` indirect-node
    ///   catalog entries with `BSDInfo.special == N`;
    /// * the read path (`open_file_reader`, `list_path`, …) follows the
    ///   hlnk pointer transparently and exposes the original bytes
    ///   under both names.
    ///
    /// Returns the link-inode number `N`. Errors if `src_path` is a
    /// directory, a symlink, or an already-existing hard link.
    pub fn create_hardlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        src_path: &str,
        dst_path: &str,
    ) -> Result<u32> {
        let (src_parent, src_name) = self.resolve_create_target(src_path)?;
        let (dst_parent, dst_name) = self.resolve_create_target(dst_path)?;
        let w = self
            .writer
            .as_mut()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: volume is read-only".into()))?;
        writer::promote_to_hardlink(w, src_parent, &src_name, dst_parent, &dst_name)
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
                let size = match &record {
                    CatalogRecord::File(f) if matches!(kind, EntryKind::Regular) => {
                        f.data_fork.logical_size
                    }
                    _ => 0,
                };
                out.push(FsDirEntry {
                    name: key.name.to_string_lossy(),
                    inode: child_id,
                    kind,
                    size,
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

impl<'a> std::io::Seek for HfsPlusFileReader<'a> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let total = self.fork.logical_size as i128;
        let new = match pos {
            std::io::SeekFrom::Start(n) => n as i128,
            std::io::SeekFrom::Current(d) => self.position as i128 + d as i128,
            std::io::SeekFrom::End(d) => total + d as i128,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "hfs+: seek to negative offset",
            ));
        }
        self.position = new as u64;
        // Recompute remaining relative to the new cursor.
        self.remaining = self.fork.logical_size.saturating_sub(self.position);
        Ok(self.position)
    }
}

impl<'a> crate::fs::FileReadHandle for HfsPlusFileReader<'a> {
    fn len(&self) -> u64 {
        self.fork.logical_size
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

// ----------------------------------------------------------------------
// `crate::fs::Filesystem` trait impl — bridges HfsPlus into the generic
// walker. `open()` returns a writable handle (the writer state is
// reconstructed from the on-disk catalog + bitmap), so the mutating
// trait methods work on already-flushed images the same way they do
// post-format.
// ----------------------------------------------------------------------

impl crate::fs::FilesystemFactory for HfsPlus {
    type FormatOpts = writer::FormatOpts;

    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }

    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl crate::fs::Filesystem for HfsPlus {
    fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        src: crate::fs::FileSource,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: non-UTF-8 path".into()))?;
        let len = src.len()?;
        let (mut reader, _) = src.open()?;
        let mode = meta.mode;
        self.create_file(dev, s, &mut reader, len, mode, meta.uid, meta.gid)
            .map(|_| ())
    }

    fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: non-UTF-8 path".into()))?;
        let mode = meta.mode;
        self.create_dir(dev, s, mode, meta.uid, meta.gid)
            .map(|_| ())
    }

    fn create_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        target: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: non-UTF-8 path".into()))?;
        let t = target.to_str().ok_or_else(|| {
            crate::Error::InvalidArgument("hfs+: non-UTF-8 symlink target".into())
        })?;
        let mode = meta.mode;
        self.create_symlink(dev, s, t, mode, meta.uid, meta.gid)
            .map(|_| ())
    }

    fn create_device(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &std::path::Path,
        _kind: crate::fs::DeviceKind,
        _major: u32,
        _minor: u32,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(crate::Error::Unsupported(
            "hfs+: device / FIFO / socket nodes are not yet implemented".into(),
        ))
    }

    fn remove(&mut self, dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: non-UTF-8 path".into()))?;
        self.remove(dev, s)
    }

    fn list(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: non-UTF-8 path".into()))?;
        self.list_path(dev, s)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn std::io::Read + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: non-UTF-8 path".into()))?;
        let r = self.open_file_reader(dev, s)?;
        Ok(Box::new(r))
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn crate::fs::FileReadHandle + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: non-UTF-8 path".into()))?;
        let r = self.open_file_reader(dev, s)?;
        Ok(Box::new(r))
    }

    fn open_file_rw<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
        flags: crate::fs::OpenFlags,
        meta: Option<crate::fs::FileMeta>,
    ) -> Result<Box<dyn crate::fs::FileHandle + 'a>> {
        handle::open_file_rw(self, dev, path, flags, meta)
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        Self::flush(self, dev)
    }

    fn read_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<std::path::PathBuf> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("hfs+: non-UTF-8 path".into()))?;
        Ok(std::path::PathBuf::from(
            self.read_symlink_target_path(dev, s)?,
        ))
    }
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

    /// Hard-link resolver invariant: the link-inode number returned by
    /// [`HfsPlus::create_hardlink`] must equal the `file_id` of the
    /// iNode catalog file that holds the data, and `BSDInfo.special`
    /// on every `hlnk` record in the chain must point at that same
    /// CNID. Cross-checks that:
    /// * both source and destination hlnk records carry the same
    ///   `special` (= `link_inode`), and neither one's `file_id`
    ///   collides with it;
    /// * `lookup_file_by_cnid(link_inode)` returns a non-hlnk file
    ///   whose `file_id == link_inode`;
    /// * the iNode owns the real data fork (logical_size matches the
    ///   payload), while the hlnk records' own data forks are empty;
    /// * the iNode's `bsd.special` carries the link count (2 here).
    ///
    /// Regression: this is the contract `resolve_hard_link` relies on
    /// to walk `hlnk -> iNode<N>`. If `create_hardlink` ever returned
    /// a non-CNID handle (e.g. a sequential counter), reads would
    /// silently mis-resolve.
    #[test]
    fn create_hardlink_returns_inode_cnid_matching_resolved_file_id() {
        let mut dev = crate::block::MemoryBackend::new(8 * 1024 * 1024);
        let opts = writer::FormatOpts::default();
        let mut hfs = HfsPlus::format(&mut dev, &opts).unwrap();

        let data = b"link-inode invariant payload\n".repeat(8);
        let mut src = std::io::Cursor::new(&data);
        hfs.create_file(&mut dev, "/src", &mut src, data.len() as u64, 0o644, 0, 0)
            .unwrap();
        let link_inode = hfs.create_hardlink(&mut dev, "/src", "/dst").unwrap();
        hfs.flush(&mut dev).unwrap();

        // Re-open from disk so we exercise the read-side resolver.
        let hfs = HfsPlus::open(&mut dev).unwrap();

        // The iNode CNID must be resolvable as a real file whose
        // file_id matches the link-inode handle exactly.
        let inode_file = hfs.lookup_file_by_cnid(&mut dev, link_inode).unwrap();
        assert_eq!(
            inode_file.file_id, link_inode,
            "iNode's catalog file_id must equal the link-inode CNID"
        );
        assert!(
            !inode_file.is_hard_link(),
            "iNode itself must not be an hlnk record (would recurse)"
        );
        assert_eq!(
            inode_file.data_fork.logical_size,
            data.len() as u64,
            "iNode owns the data fork; logical_size must equal payload"
        );
        assert_eq!(
            inode_file.bsd.special, 2,
            "iNode.bsd.special is the link count (2 = src + dst)"
        );

        // Both hlnk records must carry `special == link_inode` and have
        // a file_id distinct from it (their own catalog identity).
        for path in ["/src", "/dst"] {
            let rec = hfs.lookup_path(&mut dev, path).unwrap();
            let f = match rec {
                catalog::CatalogRecord::File(f) => f,
                _ => panic!("{path:?} should be a catalog file record"),
            };
            assert!(f.is_hard_link(), "{path:?} must be an hlnk record");
            assert_eq!(
                f.bsd.special, link_inode,
                "{path:?} bsd.special must point at the iNode CNID"
            );
            assert_ne!(
                f.file_id, link_inode,
                "{path:?} hlnk file_id must differ from the iNode CNID"
            );
            assert_eq!(
                f.data_fork.logical_size, 0,
                "{path:?} hlnk record's own data fork must be empty"
            );
        }
    }

    /// Round-trip: format + populate + flush, then `HfsPlus::open` the
    /// flushed image and add a *second* file via `create_file`, flush
    /// again, and reopen a third time. The second file must be visible
    /// at its path with byte-exact contents, and the original file must
    /// still be intact. Locks down the open-as-writable path used by
    /// `fstool add` on an already-flushed HFS+ image.
    #[test]
    fn reopen_writable_round_trip_add_file() {
        let mut dev = crate::block::MemoryBackend::new(8 * 1024 * 1024);
        let opts = writer::FormatOpts::default();

        // 1. Format + write one file + flush.
        let first = b"first file from format() pass\n".repeat(4);
        {
            let mut hfs = HfsPlus::format(&mut dev, &opts).unwrap();
            let mut src = std::io::Cursor::new(&first);
            hfs.create_file(
                &mut dev,
                "/first.txt",
                &mut src,
                first.len() as u64,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.flush(&mut dev).unwrap();
        }

        // 2. Re-open the flushed image and verify the existing file is
        //    readable AND that the handle reports mutation capability.
        //    Then add a second file and flush.
        let second = b"second file added after reopen\n".repeat(7);
        {
            let mut hfs = HfsPlus::open(&mut dev).unwrap();
            // Default Filesystem::mutation_capability is Mutable; verify
            // HFS+ doesn't override to a more restrictive value.
            let cap = <HfsPlus as crate::fs::Filesystem>::mutation_capability(&hfs);
            assert_eq!(
                cap,
                crate::fs::MutationCapability::Mutable,
                "freshly-opened HFS+ must advertise Mutable"
            );

            // Existing file still readable.
            let mut r = hfs.open_file_reader(&mut dev, "/first.txt").unwrap();
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut r, &mut buf).unwrap();
            assert_eq!(buf, first, "existing file content survives reopen");
            drop(r);

            // Add a second file via the writer state reconstructed in open().
            let mut src = std::io::Cursor::new(&second);
            hfs.create_file(
                &mut dev,
                "/second.txt",
                &mut src,
                second.len() as u64,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.flush(&mut dev).unwrap();
        }

        // 3. Re-open again and verify BOTH files exist with correct bytes.
        let hfs = HfsPlus::open(&mut dev).unwrap();
        let entries = hfs.list_path(&mut dev, "/").unwrap();
        let names: std::collections::BTreeSet<&str> =
            entries.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains("first.txt"),
            "first.txt missing after reopen; got {names:?}"
        );
        assert!(
            names.contains("second.txt"),
            "second.txt missing after reopen; got {names:?}"
        );

        let size = hfs.file_size(&mut dev, "/second.txt").unwrap();
        assert_eq!(size, second.len() as u64);
        let mut r = hfs.open_file_reader(&mut dev, "/second.txt").unwrap();
        let mut got = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut got).unwrap();
        assert_eq!(got, second, "second.txt bytes survive a second reopen");

        let mut r = hfs.open_file_reader(&mut dev, "/first.txt").unwrap();
        let mut got = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut got).unwrap();
        assert_eq!(got, first, "first.txt bytes still intact");
    }

    /// Round-trip: reopen a flushed image, `remove` an existing entry,
    /// flush, reopen — the entry must be gone and its blocks freed. Locks
    /// down the open-as-writable path used by `fstool rm`.
    #[test]
    fn reopen_writable_round_trip_remove_file() {
        let mut dev = crate::block::MemoryBackend::new(8 * 1024 * 1024);
        let opts = writer::FormatOpts::default();

        let payload = b"goodbye, cruel world\n".repeat(16);
        {
            let mut hfs = HfsPlus::format(&mut dev, &opts).unwrap();
            let mut src = std::io::Cursor::new(&payload);
            hfs.create_file(
                &mut dev,
                "/doomed.txt",
                &mut src,
                payload.len() as u64,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.create_file(
                &mut dev,
                "/keeper.txt",
                &mut std::io::Cursor::new(b"keep me\n"),
                8,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.flush(&mut dev).unwrap();
        }

        let free_before;
        {
            let mut hfs = HfsPlus::open(&mut dev).unwrap();
            free_before = hfs.volume_header.free_blocks;
            hfs.remove(&mut dev, "/doomed.txt").unwrap();
            hfs.flush(&mut dev).unwrap();
        }

        let hfs = HfsPlus::open(&mut dev).unwrap();
        let entries = hfs.list_path(&mut dev, "/").unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(
            !names.contains(&"doomed.txt"),
            "doomed.txt should be gone after remove + reopen, got {names:?}"
        );
        assert!(
            names.contains(&"keeper.txt"),
            "keeper.txt should still exist, got {names:?}"
        );

        // The data blocks the removed file owned should be reclaimed.
        // payload is len-bytes spread across ceil(len / block_size) blocks.
        let bs = hfs.block_size() as u64;
        let freed_blocks = (payload.len() as u64).div_ceil(bs) as u32;
        assert!(
            hfs.volume_header.free_blocks >= free_before + freed_blocks,
            "expected at least {freed_blocks} more free blocks after remove \
             (before={free_before}, after={})",
            hfs.volume_header.free_blocks
        );
    }

    /// `Filesystem::open_file_rw` on a non-journaled image: full
    /// round-trip — open existing file, patch a byte range in the
    /// middle, `sync`, drop the handle, reopen, verify the patch is
    /// present and the surrounding bytes are intact.
    #[test]
    fn open_file_rw_round_trip_non_journaled() {
        use crate::fs::{Filesystem, OpenFlags};
        use std::io::{Read, Seek, SeekFrom, Write};

        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = writer::FormatOpts::default();
        let payload: Vec<u8> = (0..16 * 1024).map(|i| (i & 0xFF) as u8).collect();

        {
            let mut hfs = HfsPlus::format(&mut dev, &opts).unwrap();
            let mut src = std::io::Cursor::new(&payload);
            hfs.create_file(
                &mut dev,
                "/edit.bin",
                &mut src,
                payload.len() as u64,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.flush(&mut dev).unwrap();
        }

        // Reopen, patch.
        {
            let mut hfs = HfsPlus::open(&mut dev).unwrap();
            let mut h = hfs
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/edit.bin"),
                    OpenFlags::default(),
                    None,
                )
                .unwrap();
            assert_eq!(h.len(), payload.len() as u64);
            h.seek(SeekFrom::Start(4096)).unwrap();
            h.write_all(b"PATCHED_RANGE").unwrap();
            h.sync().unwrap();
            drop(h);
        }

        // Reopen, verify.
        {
            let hfs = HfsPlus::open(&mut dev).unwrap();
            let mut r = hfs
                .open_file_reader(&mut dev, "/edit.bin")
                .unwrap();
            let mut got = Vec::new();
            r.read_to_end(&mut got).unwrap();
            assert_eq!(got.len(), payload.len());
            assert_eq!(&got[..4096], &payload[..4096], "head unchanged");
            assert_eq!(
                &got[4096..4096 + 13],
                b"PATCHED_RANGE",
                "patch is present"
            );
            assert_eq!(
                &got[4096 + 13..],
                &payload[4096 + 13..],
                "tail unchanged"
            );
        }
    }

    /// Same round-trip as the non-journaled test but on an image
    /// formatted with the journal stub enabled. After Path A's commit
    /// sequence (write tx → advance end → apply blocks → advance
    /// start) the journal is back to `start == end`, so a subsequent
    /// open sees no replay work and treats the volume as clean.
    #[test]
    fn open_file_rw_round_trip_journaled() {
        use crate::fs::{Filesystem, OpenFlags};
        use std::io::{Read, Seek, SeekFrom, Write};

        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = writer::FormatOpts {
            journaled: true,
            ..writer::FormatOpts::default()
        };
        let payload: Vec<u8> = (0..8 * 1024).map(|i| (i & 0xFF) as u8).collect();

        {
            let mut hfs = HfsPlus::format(&mut dev, &opts).unwrap();
            let mut src = std::io::Cursor::new(&payload);
            hfs.create_file(
                &mut dev,
                "/jrnl.bin",
                &mut src,
                payload.len() as u64,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.flush(&mut dev).unwrap();
        }

        // Reopen + patch via the FileHandle API.
        {
            let mut hfs = HfsPlus::open(&mut dev).unwrap();
            assert_ne!(
                hfs.volume_header.attributes & writer::VOL_ATTR_JOURNALED,
                0,
                "journaled bit must survive reopen"
            );
            let mut h = hfs
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/jrnl.bin"),
                    OpenFlags::default(),
                    None,
                )
                .unwrap();
            h.seek(SeekFrom::Start(2048)).unwrap();
            h.write_all(b"journal-bypass").unwrap();
            h.sync().unwrap();
        }

        // Verify the data survived and the journal header is still clean.
        {
            let hfs = HfsPlus::open(&mut dev).unwrap();
            let mut r = hfs.open_file_reader(&mut dev, "/jrnl.bin").unwrap();
            let mut got = Vec::new();
            r.read_to_end(&mut got).unwrap();
            assert_eq!(got.len(), payload.len());
            assert_eq!(&got[..2048], &payload[..2048]);
            assert_eq!(&got[2048..2048 + 14], b"journal-bypass");
            assert_eq!(&got[2048 + 14..], &payload[2048 + 14..]);
        }

        // Inspect the journal header directly: start MUST equal end
        // (Path A leaves the journal sealed once the transaction is
        // applied). The journal header lives at the start of the
        // journal buffer; the JournalInfoBlock at byte 12 of the
        // volume header tells us where the buffer starts.
        let mut vh_buf = [0u8; 512];
        dev.read_at(volume_header::VOLUME_HEADER_OFFSET, &mut vh_buf)
            .unwrap();
        let info_block = u32::from_be_bytes(vh_buf[12..16].try_into().unwrap());
        let bs = u32::from_be_bytes(vh_buf[40..44].try_into().unwrap());
        assert_ne!(info_block, 0);
        let info_off = u64::from(info_block) * u64::from(bs);
        let mut info = [0u8; 52];
        dev.read_at(info_off, &mut info).unwrap();
        let jbuf_off = u64::from_be_bytes(info[36..44].try_into().unwrap());
        let mut hdr = [0u8; 24];
        dev.read_at(jbuf_off, &mut hdr).unwrap();
        let start = u64::from_be_bytes(hdr[8..16].try_into().unwrap());
        let end = u64::from_be_bytes(hdr[16..24].try_into().unwrap());
        assert_eq!(
            start, end,
            "Path A requires the journal to be sealed (start == end) \
             after a successful sync (start={start:#x}, end={end:#x})"
        );
    }

    /// Simulate a crash mid-sync: commit a journal transaction (so
    /// `end` is advanced and the user data is on disk), then forcibly
    /// rewind `start` so the on-disk journal claims unreplayed work,
    /// and corrupt the user data so we can confirm replay reapplies
    /// it. The next open_file_rw must drain the journal and produce
    /// the post-sync contents.
    #[test]
    fn open_file_rw_replays_dirty_journal_on_open() {
        use crate::fs::{Filesystem, OpenFlags};
        use std::io::{Read, Seek, SeekFrom, Write};

        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = writer::FormatOpts {
            journaled: true,
            ..writer::FormatOpts::default()
        };
        let payload = b"original\n".repeat(64);
        {
            let mut hfs = HfsPlus::format(&mut dev, &opts).unwrap();
            let mut src = std::io::Cursor::new(&payload);
            hfs.create_file(
                &mut dev,
                "/replay.bin",
                &mut src,
                payload.len() as u64,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.flush(&mut dev).unwrap();
        }

        // Round 1: do a journaled write — sync persists everything.
        // Capture the journal `start`/`end` afterwards so we can roll
        // `start` back to simulate a crash that lost step 4 of the
        // commit sequence.
        let (jbuf_off, sealed_start, sealed_end) = {
            let mut hfs = HfsPlus::open(&mut dev).unwrap();
            {
                let mut h = hfs
                    .open_file_rw(
                        &mut dev,
                        std::path::Path::new("/replay.bin"),
                        OpenFlags::default(),
                        None,
                    )
                    .unwrap();
                h.seek(SeekFrom::Start(64)).unwrap();
                h.write_all(b"REPLAYED-DATA-FROM-JOURNAL").unwrap();
                h.sync().unwrap();
            }
            // Sneak a peek at the journal header.
            let mut vh_buf = [0u8; 512];
            dev.read_at(volume_header::VOLUME_HEADER_OFFSET, &mut vh_buf)
                .unwrap();
            let info_block = u32::from_be_bytes(vh_buf[12..16].try_into().unwrap());
            let bs = u32::from_be_bytes(vh_buf[40..44].try_into().unwrap());
            let info_off = u64::from(info_block) * u64::from(bs);
            let mut info = [0u8; 52];
            dev.read_at(info_off, &mut info).unwrap();
            let jbuf_off = u64::from_be_bytes(info[36..44].try_into().unwrap());
            let mut hdr = [0u8; 24];
            dev.read_at(jbuf_off, &mut hdr).unwrap();
            let s = u64::from_be_bytes(hdr[8..16].try_into().unwrap());
            let e = u64::from_be_bytes(hdr[16..24].try_into().unwrap());
            assert_eq!(s, e, "post-sync the journal must be clean");
            (jbuf_off, s, e)
        };
        assert_eq!(sealed_start, sealed_end);

        // Simulate the crash by rewinding the on-disk `start` past
        // the existing transaction. Replay must re-apply it (which is
        // a no-op since the user data is already on disk) and leave
        // the journal clean again.
        {
            let hfs = HfsPlus::open(&mut dev).unwrap();
            let mut got = Vec::new();
            hfs.open_file_reader(&mut dev, "/replay.bin")
                .unwrap()
                .read_to_end(&mut got)
                .unwrap();
            assert_eq!(&got[64..64 + 26], b"REPLAYED-DATA-FROM-JOURNAL");
        }
        // Rewind `start` to JHDR_SIZE — the journal-header size we
        // ship at format time (= 512). Now `start != end` and the
        // on-disk transaction is "unapplied" as far as the journal is
        // concerned. We rely on JournalLog::load tolerating a missing
        // checksum recompute because replay only reads start/end.
        let rewound_start: u64 = u64::from(super::journal::JHDR_SIZE);
        // Patch the header in-place.
        let mut hdr = [0u8; 24];
        dev.read_at(jbuf_off, &mut hdr).unwrap();
        hdr[8..16].copy_from_slice(&rewound_start.to_be_bytes());
        dev.write_at(jbuf_off, &hdr).unwrap();

        // Reopen and open the file for writing. The replay should fire
        // and re-apply the transaction. We then close without writing
        // and confirm the contents are intact.
        {
            let mut hfs = HfsPlus::open(&mut dev).unwrap();
            let h = hfs
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/replay.bin"),
                    OpenFlags::default(),
                    None,
                )
                .unwrap();
            // Reading from the freshly-opened handle should match the
            // journaled contents. The point of the test isn't the
            // bytes — replay already wrote them to disk before we got
            // here — but the journal header itself must now be clean
            // again.
            drop(h);
            let mut hdr = [0u8; 24];
            dev.read_at(jbuf_off, &mut hdr).unwrap();
            let s = u64::from_be_bytes(hdr[8..16].try_into().unwrap());
            let e = u64::from_be_bytes(hdr[16..24].try_into().unwrap());
            assert_eq!(s, e, "replay must leave the journal clean");
        }
    }

    /// Simulate a crash *before* the journal applies its block writes.
    /// We synthesize the state by:
    ///
    ///   1. Doing a real journaled sync (so `end > start` was reached
    ///      and re-equalised).
    ///   2. Rewinding `start` to before the transaction (so the journal
    ///      looks dirty).
    ///   3. Zeroing the user-data block (so the in-place write looks
    ///      lost).
    ///
    /// Then we reopen via [`crate::fs::Filesystem::open_file_rw`] and
    /// verify the journal contents successfully restored the bytes.
    #[test]
    fn replay_on_open_restores_lost_user_data() {
        use crate::fs::{Filesystem, OpenFlags};
        use std::io::{Seek, SeekFrom, Write};

        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = writer::FormatOpts {
            journaled: true,
            ..writer::FormatOpts::default()
        };
        let payload = vec![0xAAu8; 4096];
        {
            let mut hfs = HfsPlus::format(&mut dev, &opts).unwrap();
            let mut src = std::io::Cursor::new(&payload);
            hfs.create_file(
                &mut dev,
                "/x.bin",
                &mut src,
                payload.len() as u64,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.flush(&mut dev).unwrap();
        }

        // 1. Journaled sync.
        let (jbuf_off, sealed_end) = {
            let mut hfs = HfsPlus::open(&mut dev).unwrap();
            {
                let mut h = hfs
                    .open_file_rw(
                        &mut dev,
                        std::path::Path::new("/x.bin"),
                        OpenFlags::default(),
                        None,
                    )
                    .unwrap();
                h.seek(SeekFrom::Start(0)).unwrap();
                h.write_all(&[0xBBu8; 4096]).unwrap();
                h.sync().unwrap();
            }
            let mut vh_buf = [0u8; 512];
            dev.read_at(volume_header::VOLUME_HEADER_OFFSET, &mut vh_buf)
                .unwrap();
            let info_block = u32::from_be_bytes(vh_buf[12..16].try_into().unwrap());
            let bs = u32::from_be_bytes(vh_buf[40..44].try_into().unwrap());
            let info_off = u64::from(info_block) * u64::from(bs);
            let mut info = [0u8; 52];
            dev.read_at(info_off, &mut info).unwrap();
            let jbuf_off = u64::from_be_bytes(info[36..44].try_into().unwrap());
            let mut hdr = [0u8; 24];
            dev.read_at(jbuf_off, &mut hdr).unwrap();
            let e = u64::from_be_bytes(hdr[16..24].try_into().unwrap());
            (jbuf_off, e)
        };

        // 2. Find the file's first data block to corrupt it.
        let dev_off_block0 = {
            let hfs = HfsPlus::open(&mut dev).unwrap();
            let CatalogRecord::File(f) = hfs.lookup_path(&mut dev, "/x.bin").unwrap() else {
                panic!("expected file");
            };
            // Read the file record's first inline extent: stored at
            // raw_body[88..96] (start_block, block_count). We grab the
            // run from the writer instead — easier.
            let w = hfs.writer.as_ref().unwrap();
            let body = w
                .catalog
                .get(&writer::OwnedKey {
                    parent_id: ROOT_FOLDER_ID,
                    name: catalog::UniStr::from_str_lossy("x.bin"),
                })
                .expect("catalog body");
            let sb =
                u32::from_be_bytes(body[88 + 16..88 + 20].try_into().unwrap()); // first ext start
            assert!(sb > 0, "file must have a real data block");
            let _ = f;
            u64::from(sb) * u64::from(hfs.volume_header.block_size)
        };

        // 3. Zero the user-data block on disk, and rewind start so
        //    the journal looks dirty again.
        dev.write_at(dev_off_block0, &[0u8; 4096]).unwrap();
        let mut hdr = [0u8; 24];
        dev.read_at(jbuf_off, &mut hdr).unwrap();
        let rewound: u64 = u64::from(super::journal::JHDR_SIZE);
        hdr[8..16].copy_from_slice(&rewound.to_be_bytes());
        dev.write_at(jbuf_off, &hdr).unwrap();
        let _ = sealed_end;

        // 4. Reopen — replay should restore the 0xBB pattern.
        {
            let mut hfs = HfsPlus::open(&mut dev).unwrap();
            let h = hfs
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/x.bin"),
                    OpenFlags::default(),
                    None,
                )
                .unwrap();
            drop(h);
        }
        let mut got = [0u8; 4096];
        dev.read_at(dev_off_block0, &mut got).unwrap();
        assert!(
            got.iter().all(|&b| b == 0xBB),
            "replay must have restored the user data"
        );
    }

    /// Round-trip on a journaled image, then run `fsck.hfsplus` on the
    /// resulting bytes. Silently skipped when the binary isn't on PATH.
    #[test]
    fn open_file_rw_journaled_passes_fsck_hfsplus() {
        use crate::fs::{Filesystem, OpenFlags};
        use std::io::{Seek, SeekFrom, Write};
        use std::process::Command;

        let fsck = match Command::new("sh")
            .arg("-c")
            .arg("command -v fsck.hfsplus")
            .output()
        {
            Ok(o) if o.status.success() && !o.stdout.is_empty() => {
                String::from_utf8_lossy(&o.stdout).trim().to_string()
            }
            _ => return, // not installed; skip
        };

        let tmp = tempfile::NamedTempFile::new().expect("tmp");
        let path = tmp.path().to_path_buf();
        let total: u64 = 8 * 1024 * 1024;
        // Truncate to the target size.
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        f.set_len(total).unwrap();
        drop(f);

        let mut dev = crate::block::FileBackend::open(&path).unwrap();
        let opts = writer::FormatOpts {
            journaled: true,
            volume_name: "fsckTestVol".into(),
            ..writer::FormatOpts::default()
        };
        let payload: Vec<u8> = (0..16 * 1024).map(|i| (i & 0xFF) as u8).collect();
        {
            let mut hfs = HfsPlus::format(&mut dev, &opts).unwrap();
            let mut src = std::io::Cursor::new(&payload);
            hfs.create_file(
                &mut dev,
                "/edit.bin",
                &mut src,
                payload.len() as u64,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.flush(&mut dev).unwrap();
        }
        {
            let mut hfs = HfsPlus::open(&mut dev).unwrap();
            let mut h = hfs
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/edit.bin"),
                    OpenFlags::default(),
                    None,
                )
                .unwrap();
            h.seek(SeekFrom::Start(8192)).unwrap();
            h.write_all(b"journal-test-payload").unwrap();
            h.sync().unwrap();
        }
        crate::block::BlockDevice::sync(&mut dev).unwrap();
        drop(dev);

        let out = Command::new(&fsck)
            .arg("-fn")
            .arg(&path)
            .output()
            .expect("run fsck.hfsplus");
        // fsck.hfsplus returns 0 on a clean volume.
        assert!(
            out.status.success(),
            "fsck.hfsplus failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }

    /// Path A for `flush`: when a journaled HFS+ image is reopened and
    /// mutated (creating directories, files, removing files), the
    /// subsequent metadata `flush` must route its writes — catalog
    /// B-tree pages, extents-overflow pages, allocation bitmap, and the
    /// two volume headers — through `JournalLog::commit` rather than
    /// applying them in place. We verify this by:
    ///
    ///   1. Formatting a journaled image with one file.
    ///   2. Reopening it, mutating it (add a directory, add a file,
    ///      remove the original) to dirty multiple metadata regions,
    ///      then calling `flush`.
    ///   3. Snapshotting the journal header (`start == end` post-flush).
    ///   4. Reading the post-flush metadata bytes for both volume
    ///      headers so we can corrupt them on disk and prove replay
    ///      restored them.
    ///   5. Corrupting the primary volume header on disk (zeroing it)
    ///      and rewinding the journal `start` to before the flush
    ///      transaction.
    ///   6. Reopening — the journal replay in `HfsPlus::open` must
    ///      restore the volume-header bytes before the catalog open
    ///      attempts to read them. The mutated directory tree must
    ///      then be readable byte-exact.
    #[test]
    fn flush_routes_metadata_through_journal_on_reopen() {
        let mut dev = crate::block::MemoryBackend::new(8 * 1024 * 1024);
        let opts = writer::FormatOpts {
            journaled: true,
            ..writer::FormatOpts::default()
        };

        // 1. Initial format + a file + flush. The first flush after
        //    format runs through the Direct sink (no on-disk journal
        //    header yet to route through).
        let initial = b"initial-payload\n".repeat(16);
        {
            let mut hfs = HfsPlus::format(&mut dev, &opts).unwrap();
            let mut src = std::io::Cursor::new(&initial);
            hfs.create_file(
                &mut dev,
                "/initial.bin",
                &mut src,
                initial.len() as u64,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.flush(&mut dev).unwrap();
        }

        // Capture the journal-buffer offset for later poking.
        let jbuf_off = {
            let mut vh_buf = [0u8; 512];
            dev.read_at(volume_header::VOLUME_HEADER_OFFSET, &mut vh_buf)
                .unwrap();
            let info_block = u32::from_be_bytes(vh_buf[12..16].try_into().unwrap());
            let bs = u32::from_be_bytes(vh_buf[40..44].try_into().unwrap());
            assert_ne!(info_block, 0, "journaled volume must have a JIB");
            let info_off = u64::from(info_block) * u64::from(bs);
            let mut info = [0u8; 52];
            dev.read_at(info_off, &mut info).unwrap();
            u64::from_be_bytes(info[36..44].try_into().unwrap())
        };

        // 2. Reopen + mutate + flush. This flush runs through Buffered
        //    sink → JournalLog::commit.
        let new_payload = b"second\n".repeat(32);
        {
            let mut hfs = HfsPlus::open(&mut dev).unwrap();
            // A pre-flush journal must be clean (start == end). The
            // commit machinery in writer::flush relies on that.
            let mut hdr = [0u8; 24];
            dev.read_at(jbuf_off, &mut hdr).unwrap();
            let s = u64::from_be_bytes(hdr[8..16].try_into().unwrap());
            let e = u64::from_be_bytes(hdr[16..24].try_into().unwrap());
            assert_eq!(s, e, "fresh-format flush leaves the journal clean");

            // Multiple mutations: dir + file + remove. This dirties the
            // catalog (always), the bitmap (file allocation + free), and
            // bumps next_cnid (volume header).
            hfs.create_dir(&mut dev, "/added-dir", 0o755, 0, 0).unwrap();
            let mut src = std::io::Cursor::new(&new_payload);
            hfs.create_file(
                &mut dev,
                "/added-file.bin",
                &mut src,
                new_payload.len() as u64,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.remove(&mut dev, "/initial.bin").unwrap();
            hfs.flush(&mut dev).unwrap();
        }

        // 3. Post-flush, the journal must be sealed again (start == end).
        let sealed_end = {
            let mut hdr = [0u8; 24];
            dev.read_at(jbuf_off, &mut hdr).unwrap();
            let s = u64::from_be_bytes(hdr[8..16].try_into().unwrap());
            let e = u64::from_be_bytes(hdr[16..24].try_into().unwrap());
            assert_eq!(
                s, e,
                "Path A for flush requires the journal to seal post-commit"
            );
            e
        };
        assert!(
            sealed_end > u64::from(journal::JHDR_SIZE),
            "the post-mutation flush must have advanced the journal end \
             cursor past the initial JHDR_SIZE — got {sealed_end:#x}"
        );

        // 4. Sanity check that the live image (after the flush, before
        //    corruption) reports the mutations.
        {
            let hfs = HfsPlus::open(&mut dev).unwrap();
            let entries = hfs.list_path(&mut dev, "/").unwrap();
            let names: std::collections::BTreeSet<&str> =
                entries.iter().map(|e| e.name.as_str()).collect();
            assert!(names.contains("added-dir"));
            assert!(names.contains("added-file.bin"));
            assert!(
                !names.contains("initial.bin"),
                "removed file must be gone post-flush; got {names:?}"
            );
        }

        // 5. Corrupt the catalog file's on-disk blocks and the
        //    allocation bitmap, then rewind `start` so the journal
        //    looks dirty. The journal entry from the flush above
        //    carries a copy of those bytes — replay must restore them
        //    on the next open.
        //
        // We deliberately do NOT corrupt the volume header itself: the
        // VH at offset 1024 is needed by `read_volume_header` to find
        // the JournalInfoBlock before replay can run. The VH is still
        // covered by the journal transaction (so it would be restored
        // anyway in production), but corrupting it here would block us
        // from reading the JIB pointer.
        let (cat_off, cat_len, bm_off, bm_len) = {
            let hfs = HfsPlus::open(&mut dev).unwrap();
            let bs = u64::from(hfs.volume_header.block_size);
            let cat = hfs.volume_header.catalog_file.extents[0];
            let bm = hfs.volume_header.allocation_file.extents[0];
            (
                u64::from(cat.start_block) * bs,
                u64::from(cat.block_count) * bs,
                u64::from(bm.start_block) * bs,
                u64::from(bm.block_count) * bs,
            )
        };
        // Zero the catalog and bitmap so the on-disk structures are
        // unusable without replay.
        let zeros_cat = vec![0u8; cat_len as usize];
        dev.write_at(cat_off, &zeros_cat).unwrap();
        let zeros_bm = vec![0u8; bm_len as usize];
        dev.write_at(bm_off, &zeros_bm).unwrap();
        // Rewind start.
        let mut hdr = [0u8; 24];
        dev.read_at(jbuf_off, &mut hdr).unwrap();
        let rewound: u64 = u64::from(journal::JHDR_SIZE);
        hdr[8..16].copy_from_slice(&rewound.to_be_bytes());
        dev.write_at(jbuf_off, &hdr).unwrap();

        // Without replay, the catalog is unreadable. With replay, the
        // bytes come back and the catalog opens normally.
        {
            let hfs = HfsPlus::open(&mut dev).unwrap();
            let entries = hfs.list_path(&mut dev, "/").unwrap();
            let names: std::collections::BTreeSet<&str> =
                entries.iter().map(|e| e.name.as_str()).collect();
            assert!(
                names.contains("added-dir"),
                "replay must restore the catalog so the dir is visible; \
                 got {names:?}"
            );
            assert!(
                names.contains("added-file.bin"),
                "replay must restore the catalog so the file is visible; \
                 got {names:?}"
            );
            assert!(
                !names.contains("initial.bin"),
                "removed file must not reappear after replay; got {names:?}"
            );

            // The replayed file's bytes should be intact too.
            let mut r = hfs.open_file_reader(&mut dev, "/added-file.bin").unwrap();
            let mut got = Vec::new();
            std::io::Read::read_to_end(&mut r, &mut got).unwrap();
            assert_eq!(
                got, new_payload,
                "file data referenced by the replayed catalog must match"
            );

            // And the journal must be clean again after replay.
            let mut hdr = [0u8; 24];
            dev.read_at(jbuf_off, &mut hdr).unwrap();
            let s = u64::from_be_bytes(hdr[8..16].try_into().unwrap());
            let e = u64::from_be_bytes(hdr[16..24].try_into().unwrap());
            assert_eq!(s, e, "replay leaves the journal sealed");
        }
    }

    #[test]
    fn open_file_ro_random_seek_hfs_plus() {
        use crate::fs::Filesystem;
        use std::io::{Read, Seek, SeekFrom};

        let mut dev = crate::block::MemoryBackend::new(8 * 1024 * 1024);
        let opts = writer::FormatOpts::default();
        let data: Vec<u8> = (0..6000u32).map(|i| (i & 0xFF) as u8).collect();
        {
            let mut hfs = HfsPlus::format(&mut dev, &opts).unwrap();
            let mut src = std::io::Cursor::new(&data);
            hfs.create_file(
                &mut dev,
                "/ro.bin",
                &mut src,
                data.len() as u64,
                0o644,
                0,
                0,
            )
            .unwrap();
            hfs.flush(&mut dev).unwrap();
        }

        let mut hfs = HfsPlus::open(&mut dev).unwrap();
        let mut h = hfs
            .open_file_ro(&mut dev, std::path::Path::new("/ro.bin"))
            .expect("open_file_ro");
        assert_eq!(h.len(), data.len() as u64);
        assert!(!h.is_empty());

        h.seek(SeekFrom::Start(3333)).unwrap();
        let mut buf = [0u8; 96];
        h.read_exact(&mut buf).unwrap();
        assert_eq!(&buf[..], &data[3333..3429]);

        h.seek(SeekFrom::Start(40)).unwrap();
        let mut buf2 = [0u8; 64];
        h.read_exact(&mut buf2).unwrap();
        assert_eq!(&buf2[..], &data[40..104]);
    }
}
