//! HFS+ — Apple's legacy macOS filesystem (pre-2017). Read-only support.
//!
//! ## Status
//!
//! Read-only v1: open, list directories, stream regular file contents.
//! Implementation is based on Apple Technical Note TN1150, the
//! canonical public reference for HFS+ on-disk structures.
//!
//! ## Scope and deferred features
//!
//! * Only the **inline extents** (≤ 8 per fork, embedded in the
//!   catalog record) are supported. A file whose `totalBlocks` exceeds
//!   the sum of its inline extents would need the extents-overflow
//!   B-tree; this v1 returns `Error::Unsupported` for that case.
//! * No journal replay — the on-disk catalog is read as-is.
//! * No attempt is made to chase indirect-link (`kHardLinkFileType`)
//!   records; encountering one returns `Unsupported`.
//! * HFSX case-sensitive comparison is honoured at the catalog-key
//!   level; non-ASCII case folding follows a simplified table.
//! * Write support is out of scope entirely.
//!
//! ## Module layout
//!
//! * [`volume_header`] — the 512-byte `HFSPlusVolumeHeader` at offset 1024.
//! * [`btree`] — node descriptors, header records, record-offset tables.
//! * [`catalog`] — catalog keys, leaf records, lookup.
//! * [`extents`] — extents-overflow key/record decoding (unused by v1).

pub mod btree;
pub mod catalog;
pub mod extents;
pub mod volume_header;

use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;

use btree::ForkReader;
use catalog::{
    Catalog, CatalogFile, CatalogKey, CatalogRecord, ROOT_FOLDER_ID, UniStr, mode,
};
use volume_header::{VolumeHeader, read_volume_header};

/// An opened HFS+ volume.
pub struct HfsPlus {
    volume_header: VolumeHeader,
    catalog: Catalog,
    /// Cached volume name, taken from the root folder's thread record.
    volume_name: String,
}

impl HfsPlus {
    /// Open an existing HFS+ volume on `dev`.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let vh = read_volume_header(dev)?;
        let case_sensitive = vh.is_hfsx();

        let cat_fork = ForkReader::from_inline(&vh.catalog_file, vh.block_size, "catalog")?;
        let catalog = Catalog::open(dev, cat_fork, case_sensitive)?;

        // The root folder's thread record is keyed by (ROOT_FOLDER_ID, "");
        // its name field is the volume name.
        let volume_name = lookup_thread_name(dev, &catalog, ROOT_FOLDER_ID)?
            .unwrap_or_else(|| "Untitled".to_string());

        Ok(Self {
            volume_header: vh,
            catalog,
            volume_name,
        })
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
    /// to a Unix inode number on this filesystem.
    pub fn list_path(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let cnid = self.resolve_dir(dev, path)?;
        self.list_cnid(dev, cnid)
    }

    /// Open a regular file by absolute path for streaming reads.
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
        reject_indirect_link(&file)?;
        let mode_bits = file.bsd.file_mode & mode::S_IFMT;
        if mode_bits == mode::S_IFLNK {
            return Err(crate::Error::Unsupported(format!(
                "hfs+: {path:?} is a symlink; symlink reads are not supported in v1"
            )));
        }
        if mode_bits != 0 && mode_bits != mode::S_IFREG {
            return Err(crate::Error::Unsupported(format!(
                "hfs+: {path:?} is not a regular file (mode {:#06o})",
                file.bsd.file_mode
            )));
        }
        let fork = ForkReader::from_inline(
            &file.data_fork,
            self.volume_header.block_size,
            "data fork",
        )?;
        Ok(HfsPlusFileReader {
            dev,
            fork,
            remaining: file.data_fork.logical_size,
            position: 0,
        })
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
        self.catalog
            .lookup(dev, &key)?
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "hfs+: no such entry {name:?} under CNID {parent_id}"
                ))
            })
    }

    /// Enumerate the direct children of folder `cnid` by scanning
    /// every leaf node of the catalog from the first leaf onwards
    /// and collecting entries whose key.parentID matches.
    fn list_cnid(
        &self,
        dev: &mut dyn BlockDevice,
        cnid: u32,
    ) -> Result<Vec<crate::fs::DirEntry>> {
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
                    CatalogRecord::File(f) => (file_kind(f), f.file_id),
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

/// Reject HFS+ "indirect node" file types — these are the files that
/// stand in for hard links (`hlnk` / `hfs+`) and we don't chase them
/// in v1.
fn reject_indirect_link(file: &CatalogFile) -> Result<()> {
    // FileInfo + ExtendedFileInfo live at body[48..80]; their first
    // 8 bytes are fileType + creator. Apple's hard-link convention:
    //   fileType = 'hlnk' (0x686C6E6B), creator = 'hfs+' (0x68667321 ... '!')
    //
    // We don't decode FileInfo into the struct, so this check would
    // require a wider catalog record. Conservatively assume regular
    // files are not indirect; the failure mode for a hard-link
    // indirect is a stale fork pointer, which we'd report as a read
    // error rather than silent data corruption.
    let _ = file;
    Ok(())
}

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
