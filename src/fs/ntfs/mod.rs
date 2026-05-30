//! NTFS — Microsoft's NT File System. Read implementation.
//!
//! ## Status
//!
//! Detection, MFT decode, attribute decode, directory walking via
//! $INDEX_ROOT + $INDEX_ALLOCATION, and streaming reads of $DATA streams
//! (resident, non-resident, sparse, LZNT1-compressed, and alternate data
//! streams) are implemented. The driver follows `$ATTRIBUTE_LIST` spill
//! across multiple MFT records, resolves shared security descriptors via
//! `$Secure:$SDS` (looked up through `$Secure:$SII`), and case-folds
//! directory lookups through the `$UpCase` table. Write support is out
//! of scope.
//!
//! ## Reference
//!
//! - Microsoft "[MS-FSCC] File System Control Codes":
//!   <https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-fscc/>
//! - Microsoft "[MS-XCA]" §2.5 ("LZNT1 Algorithm Details"):
//!   <https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-xca/>
//! - Linux kernel "NTFS3" docs:
//!   <https://docs.kernel.org/filesystems/ntfs3.html>
//! - "NTFS Documentation" by Richard Russon and Yuval Fledel.
//!
//! ## Attribute model — non-Unix metadata
//!
//! NTFS metadata doesn't map cleanly onto POSIX. The shape we adopt for
//! cross-FS conversion is:
//!
//! | NTFS concept                              | xattr key                                | Notes                                                    |
//! |-------------------------------------------|------------------------------------------|----------------------------------------------------------|
//! | `$STANDARD_INFORMATION.file_attributes`   | `user.ntfs.dos_attrs`                    | 32-bit LE: READONLY/HIDDEN/SYSTEM/ARCHIVE/COMPRESSED/etc |
//! | Object ID GUID (`$OBJECT_ID`)             | `user.ntfs.object_id`                    | 16 bytes raw GUID                                        |
//! | Reparse point tag + data (`$REPARSE_POINT`)| `user.ntfs.reparse`                     | Tag (LE u32) prepended to raw reparse data               |
//! | Alternate Data Streams (named `$DATA`)    | `user.ntfs.ads.<name>`                   | Per-stream xattr; binary stream contents                 |
//! | `$SECURITY_DESCRIPTOR` (raw NT SD blob)   | `system.ntfs_security`                   | Resident attribute, OR resolved from `$Secure:$SDS`      |
//! |                                           |                                          | via `$STANDARD_INFORMATION.security_id`                  |
//! | Short (8.3) filename                      | `user.ntfs.short_name`                   | UTF-16LE per `$FILE_NAME` with namespace=DOS             |
//! | Last-write / creation / change / access   | inode timestamps + `user.ntfs.times.raw` | The latter holds all four NT-FILETIME (100 ns) values    |
//!
//! ### NTFS → NTFS round-trip guarantee
//!
//! The cross-FS xattr mapping above is lossy at the sub-100ns level. For
//! NTFS-to-NTFS transfers, the writer copies raw attribute byte streams
//! verbatim rather than going through this mapping.
//!
//! ### Reparse points
//!
//! `$REPARSE_POINT` data is surfaced via `user.ntfs.reparse` (tag + raw
//! data). This driver does NOT follow junctions, symlinks or any other
//! reparse-point type — the target's bytes are intentionally exposed as-is
//! so the caller can decide whether to interpret them. A symlink read as
//! a "file" via `open_file_reader` returns the reparse point's
//! `$DATA` (typically empty) rather than dereferencing the link.

use std::collections::HashMap;
use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;

pub mod attribute;
pub mod attribute_list;
pub mod boot;
pub mod compression;
pub mod format;
pub mod index;
pub mod logfile;
pub mod mft;
pub mod run_list;
pub mod rw;
pub mod secure;
pub mod upcase_gen;
pub mod writer;

use attribute::{
    ATTR_FLAG_COMPRESSED, ATTR_FLAG_ENCRYPTED, AttributeIter, AttributeKind, FileName,
    StandardInformation, TYPE_ATTRIBUTE_LIST, TYPE_DATA, TYPE_FILE_NAME, TYPE_INDEX_ALLOCATION,
    TYPE_INDEX_ROOT, TYPE_OBJECT_ID, TYPE_REPARSE_POINT, TYPE_SECURITY_DESCRIPTOR,
    TYPE_STANDARD_INFORMATION,
};
use boot::BootSector;
use index::IndexEntry;
use run_list::Extent;
use secure::UpcaseTable;

/// Hard-coded MFT record numbers reserved by NTFS.
pub const MFT_RECORD_MFT: u64 = 0;
pub const MFT_RECORD_ROOT: u64 = 5;
pub const MFT_RECORD_SECURE: u64 = 9;
pub const MFT_RECORD_UPCASE: u64 = 10;

/// Cap on the size of a single security descriptor we'll pull out of
/// `$Secure:$SDS` and surface as `system.ntfs_security`. SDs are normally
/// well under 1 KiB; the 64 KiB cap exists purely as a sanity check
/// against malformed images.
const MAX_SECURITY_DESCRIPTOR_BYTES: u64 = 64 * 1024;

pub fn probe(dev: &mut dyn BlockDevice) -> Result<bool> {
    if dev.total_size() < 11 {
        return Ok(false);
    }
    let mut head = [0u8; 11];
    dev.read_at(0, &mut head)?;
    Ok(&head[3..11] == boot::NTFS_OEM)
}

/// Convenience re-export of the boot-sector decoder for tests/inspect.
pub use boot::BootSector as ExportedBootSector;

pub struct Ntfs {
    boot: BootSector,
    /// Cached MFT run list: where to read MFT record N from. Empty before
    /// `load_mft_runs` has been called.
    mft_runs: Vec<Extent>,
    /// Cached `$UpCase` table for case-insensitive directory lookups.
    /// `None` means "haven't tried yet"; `Some(identity)` means we tried
    /// and the image didn't expose one — names are compared exactly.
    upcase: Option<UpcaseTable>,
    /// Cache of decoded `$Secure:$SII` entries (security_id -> SDS slice).
    /// `None` means "haven't tried yet". An empty `Some(_)` means we tried
    /// and the image had no usable `$Secure`.
    sii_cache: Option<HashMap<u32, (u64, u32)>>,
    /// Writer state — populated only after `Ntfs::format` (or
    /// `Ntfs::open_for_write`). Read-only opens leave this `None`.
    writer: Option<writer::WriterState>,
}

impl Ntfs {
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        if dev.total_size() < 512 {
            return Err(crate::Error::InvalidImage(
                "ntfs: device too small to hold a boot sector".into(),
            ));
        }
        let mut buf = [0u8; 512];
        dev.read_at(0, &mut buf)?;
        let boot = BootSector::decode(&buf).ok_or_else(|| {
            crate::Error::InvalidImage("ntfs: boot sector OEM ID is not 'NTFS    '".into())
        })?;
        Ok(Self {
            boot,
            mft_runs: Vec::new(),
            upcase: None,
            sii_cache: None,
            writer: None,
        })
    }

    pub fn total_bytes(&self) -> u64 {
        self.boot.total_sectors * u64::from(self.boot.bytes_per_sector)
    }

    pub fn cluster_size(&self) -> u32 {
        self.boot.cluster_size()
    }

    pub fn bytes_per_sector(&self) -> u16 {
        self.boot.bytes_per_sector
    }

    pub fn sectors_per_cluster(&self) -> u8 {
        self.boot.sectors_per_cluster
    }

    pub fn mft_record_size(&self) -> u32 {
        self.boot.mft_record_size()
    }

    pub fn volume_serial(&self) -> u64 {
        self.boot.volume_serial
    }

    pub fn boot_sector(&self) -> &BootSector {
        &self.boot
    }

    /// Read MFT record N from disk into `out`. `out` must be at least
    /// `mft_record_size` bytes; the function applies USA fixup before
    /// returning. The first call lazily loads $MFT's own run list by
    /// reading record 0 directly from `mft_lcn` (bootstrap).
    pub fn read_mft_record(
        &mut self,
        dev: &mut dyn BlockDevice,
        rec: u64,
        out: &mut [u8],
    ) -> Result<()> {
        let rec_size = self.boot.mft_record_size() as usize;
        if out.len() < rec_size {
            return Err(crate::Error::InvalidArgument(
                "ntfs: MFT scratch buffer too small".into(),
            ));
        }
        let out = &mut out[..rec_size];

        // Bootstrap: read record 0 from the BPB-anchored MFT LCN. From
        // record 0 we extract $MFT's $DATA run list and cache it.
        if self.mft_runs.is_empty() {
            let base = self.boot.mft_lcn * u64::from(self.boot.cluster_size());
            // Record 0 is at the very start of the MFT — its index times
            // record_size is zero, so the read offset is just `base`.
            dev.read_at(base, out)?;
            mft::apply_fixup(out, self.boot.bytes_per_sector as usize)?;
            // Now decode record 0's attributes to find $DATA's run list.
            let header = mft::RecordHeader::parse(out)?;
            for attr_res in AttributeIter::new(out, header.first_attribute_offset as usize) {
                let attr = attr_res?;
                if attr.type_code == TYPE_DATA && attr.name.is_empty() {
                    match attr.kind {
                        AttributeKind::NonResident { runs, .. } => {
                            self.mft_runs = runs;
                        }
                        AttributeKind::Resident { .. } => {
                            return Err(crate::Error::InvalidImage(
                                "ntfs: $MFT $DATA is resident — impossible".into(),
                            ));
                        }
                    }
                    break;
                }
            }
            if self.mft_runs.is_empty() {
                return Err(crate::Error::InvalidImage(
                    "ntfs: could not locate $MFT $DATA run list in record 0".into(),
                ));
            }
            if rec == 0 {
                return Ok(()); // already loaded
            }
        }

        // For all other records, map record `rec` through the MFT $DATA
        // run list. `mft_runs` is in clusters; the record offset within
        // the MFT (in bytes) is `rec * rec_size`.
        let mft_byte_offset = rec
            .checked_mul(rec_size as u64)
            .ok_or_else(|| crate::Error::InvalidImage("ntfs: MFT offset overflow".into()))?;
        let cluster_size = u64::from(self.boot.cluster_size());
        let mut vcn_bytes: u64 = 0;
        let mut found = false;
        for ext in &self.mft_runs {
            let ext_bytes = ext.length * cluster_size;
            if mft_byte_offset < vcn_bytes + ext_bytes {
                let local = mft_byte_offset - vcn_bytes;
                match ext.lcn {
                    Some(lcn) => {
                        let phys = lcn * cluster_size + local;
                        dev.read_at(phys, out)?;
                    }
                    None => {
                        return Err(crate::Error::InvalidImage(
                            "ntfs: requested MFT record sits in a sparse run".into(),
                        ));
                    }
                }
                found = true;
                break;
            }
            vcn_bytes += ext_bytes;
        }
        if !found {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: MFT record {rec} is past the end of $MFT"
            )));
        }
        mft::apply_fixup(out, self.boot.bytes_per_sector as usize)?;
        Ok(())
    }

    /// Read the base record `rec_no` plus, if it has an `$ATTRIBUTE_LIST`,
    /// every extension record named in that list. Returns a vector of
    /// `(record_number, record_bytes)` pairs ordered base-first.
    fn load_record_set(
        &mut self,
        dev: &mut dyn BlockDevice,
        rec_no: u64,
    ) -> Result<Vec<(u64, Vec<u8>)>> {
        let rec_size = self.boot.mft_record_size() as usize;
        let mut base = vec![0u8; rec_size];
        self.read_mft_record(dev, rec_no, &mut base)?;
        let mut records: Vec<(u64, Vec<u8>)> = vec![(rec_no, base)];

        // Look for $ATTRIBUTE_LIST in the base record.
        let base_bytes = records[0].1.clone();
        let hdr = mft::RecordHeader::parse(&base_bytes)?;
        let mut alist_bytes: Option<Vec<u8>> = None;
        for attr_res in AttributeIter::new(&base_bytes, hdr.first_attribute_offset as usize) {
            let attr = attr_res?;
            if attr.type_code != TYPE_ATTRIBUTE_LIST {
                continue;
            }
            match attr.kind {
                AttributeKind::Resident { value, .. } => {
                    alist_bytes = Some(value.to_vec());
                }
                AttributeKind::NonResident {
                    real_size, runs, ..
                } => {
                    // Non-resident $ATTRIBUTE_LIST: stream it cluster by
                    // cluster through a dedicated reader. This is uncommon
                    // (the list rarely overflows a record) but legal.
                    let mut reader = NonResidentReader {
                        dev: &mut *dev,
                        cluster_size: self.boot.cluster_size() as u64,
                        runs,
                        real_size,
                        initialized_size: real_size,
                        pos: 0,
                        cluster_buf: vec![0u8; self.boot.cluster_size() as usize],
                        cached_vcn: u64::MAX,
                        cached_cluster_filled: false,
                    };
                    let mut buf = Vec::with_capacity(real_size as usize);
                    reader.read_to_end(&mut buf).map_err(crate::Error::from)?;
                    alist_bytes = Some(buf);
                }
            }
            break;
        }

        let Some(alist_bytes) = alist_bytes else {
            return Ok(records);
        };
        let entries = attribute_list::decode(&alist_bytes)?;
        let mut seen = std::collections::HashSet::new();
        seen.insert(rec_no);
        for entry in entries {
            let extension_rec = entry.record_number();
            if extension_rec == rec_no {
                // The list also names attributes that live in the base
                // record — skip those.
                continue;
            }
            if !seen.insert(extension_rec) {
                continue;
            }
            let mut buf = vec![0u8; rec_size];
            self.read_mft_record(dev, extension_rec, &mut buf)?;
            records.push((extension_rec, buf));
        }
        Ok(records)
    }

    /// Walk a directory's index, returning the (file_ref, FileName) of each
    /// entry. Skips DOS-namespace duplicates (those are covered by the
    /// Win32+DOS combined entry that has both names).
    pub fn read_directory(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_rec: u64,
    ) -> Result<Vec<IndexEntry>> {
        // Flush any entries staged for this directory by the writer's
        // batch cache so the on-disk `$I30` we are about to read reflects
        // every child created so far (transparency for list / path
        // lookups / remove).
        if let Some(w) = self.writer.as_mut()
            && let Some(entries) = w.dir_batch.take(&dir_rec)
        {
            self.serialize_dir(dev, dir_rec, &entries)?;
        }
        let records = self.load_record_set(dev, dir_rec)?;
        let hdr = mft::RecordHeader::parse(&records[0].1)?;
        if !hdr.is_in_use() {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: directory record {dir_rec} is not in use"
            )));
        }

        // Locate $INDEX_ROOT (must be named "$I30") and optional
        // $INDEX_ALLOCATION (same name) across the merged record set.
        let mut root_value: Option<Vec<u8>> = None;
        let mut alloc_runs: Option<Vec<Extent>> = None;
        for (_rec, rec_buf) in &records {
            let h = mft::RecordHeader::parse(rec_buf)?;
            for attr_res in AttributeIter::new(rec_buf, h.first_attribute_offset as usize) {
                let attr = attr_res?;
                if attr.name != "$I30" {
                    continue;
                }
                match (attr.type_code, attr.kind) {
                    (TYPE_INDEX_ROOT, AttributeKind::Resident { value, .. }) => {
                        root_value = Some(value.to_vec());
                    }
                    (TYPE_INDEX_ALLOCATION, AttributeKind::NonResident { runs, .. }) => {
                        // Multiple $INDEX_ALLOCATION segments are appended
                        // in starting_vcn order via load_record_set, so
                        // just chain runs as we encounter them. NTFS only
                        // splits these for very large directories.
                        match alloc_runs.as_mut() {
                            Some(existing) => existing.extend(runs),
                            None => alloc_runs = Some(runs),
                        }
                    }
                    _ => {}
                }
            }
        }
        let root_value = root_value.ok_or_else(|| {
            crate::Error::InvalidImage(format!("ntfs: record {dir_rec} has no $INDEX_ROOT $I30"))
        })?;
        let root_hdr = index::IndexRootHeader::parse(&root_value)?;
        let entries_start = root_hdr.header_offset + root_hdr.first_entry_offset as usize;
        let entries_len = (root_hdr.bytes_in_use as usize).saturating_sub(16);

        let mut out = Vec::new();
        let mut visited_blocks = std::collections::HashSet::<u64>::new();
        let root_children = index::walk_index_node(&root_value, entries_start, entries_len, |e| {
            out.push(e.clone());
        })?;

        if let Some(runs) = alloc_runs {
            let block_size = root_hdr.index_block_size as usize;
            for vcn in root_children {
                self.descend_index(dev, &runs, block_size, vcn, &mut out, &mut visited_blocks)?;
            }
        }

        // Dedup DOS-namespace duplicates: NTFS stores a separate index
        // entry for the DOS short name when the file has both Win32 and
        // DOS names. The same `file_ref` shows up twice in that case;
        // we drop the DOS-namespace one and keep the Win32 long name.
        let mut seen_refs = std::collections::HashMap::<u64, usize>::new();
        let mut filtered: Vec<IndexEntry> = Vec::new();
        for entry in out.into_iter() {
            let key = entry.file_ref;
            let is_dos = entry
                .file_name
                .as_ref()
                .map(|fn_| fn_.namespace == FileName::NAMESPACE_DOS)
                .unwrap_or(false);
            if is_dos && seen_refs.contains_key(&key) {
                continue;
            }
            if let Some(&idx) = seen_refs.get(&key) {
                // If we already kept a DOS entry but now get a Win32 one,
                // replace it.
                let prior_is_dos = filtered[idx]
                    .file_name
                    .as_ref()
                    .map(|fn_| fn_.namespace == FileName::NAMESPACE_DOS)
                    .unwrap_or(false);
                if prior_is_dos && !is_dos {
                    filtered[idx] = entry;
                }
                continue;
            }
            seen_refs.insert(key, filtered.len());
            filtered.push(entry);
        }
        Ok(filtered)
    }

    fn descend_index(
        &mut self,
        dev: &mut dyn BlockDevice,
        alloc_runs: &[Extent],
        block_size: usize,
        vcn: u64,
        out: &mut Vec<IndexEntry>,
        visited: &mut std::collections::HashSet<u64>,
    ) -> Result<()> {
        if !visited.insert(vcn) {
            return Err(crate::Error::InvalidImage(
                "ntfs: cycle in $INDEX_ALLOCATION tree".into(),
            ));
        }
        let cluster_size = u64::from(self.boot.cluster_size());
        let target_bytes = vcn * cluster_size;
        let mut walked: u64 = 0;
        let mut block_buf = vec![0u8; block_size];
        let mut found_offset: Option<u64> = None;
        for ext in alloc_runs {
            let span = ext.length * cluster_size;
            if target_bytes < walked + span {
                let local = target_bytes - walked;
                match ext.lcn {
                    Some(lcn) => {
                        found_offset = Some(lcn * cluster_size + local);
                    }
                    None => {
                        return Err(crate::Error::InvalidImage(
                            "ntfs: $INDEX_ALLOCATION points to a sparse VCN".into(),
                        ));
                    }
                }
                break;
            }
            walked += span;
        }
        let phys = found_offset.ok_or_else(|| {
            crate::Error::InvalidImage(format!("ntfs: index VCN {vcn} not in run list"))
        })?;
        dev.read_at(phys, &mut block_buf)?;
        mft::apply_fixup(&mut block_buf, self.boot.bytes_per_sector as usize)?;
        let blk_hdr = index::IndexBlockHeader::parse(&block_buf)?;
        let entries_start = blk_hdr.entries_start();
        let entries_len = blk_hdr.entries_byte_len();
        let children = index::walk_index_node(&block_buf, entries_start, entries_len, |e| {
            out.push(e.clone());
        })?;
        for child in children {
            self.descend_index(dev, alloc_runs, block_size, child, out, visited)?;
        }
        Ok(())
    }

    /// Resolve a path to its MFT record number. Path components are matched
    /// case-insensitively through the `$UpCase` table (when available). The
    /// root path "/" maps to record 5.
    pub fn lookup_path(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<u64> {
        if !path.starts_with('/') {
            return Err(crate::Error::InvalidArgument(format!(
                "ntfs: path must be absolute, got {path:?}"
            )));
        }
        self.ensure_upcase(dev)?;
        let mut current = MFT_RECORD_ROOT;
        for component in path.split('/').filter(|s| !s.is_empty()) {
            let entries = self.read_directory(dev, current)?;
            let mut next: Option<u64> = None;
            for entry in entries {
                if let Some(fname) = entry.file_name {
                    if fname.namespace == FileName::NAMESPACE_DOS {
                        // DOS-namespace entries are covered by the matching
                        // Win32 entry; skip to avoid double matching.
                        continue;
                    }
                    let matches = match self.upcase.as_ref() {
                        Some(t) => t.equals_ignore_case(&fname.name, component),
                        None => fname.name == component,
                    };
                    if matches {
                        next = Some(entry.file_ref & 0x0000_FFFF_FFFF_FFFF);
                        break;
                    }
                }
            }
            current = next.ok_or_else(|| {
                crate::Error::InvalidImage(format!("ntfs: path component {component:?} not found"))
            })?;
        }
        Ok(current)
    }

    /// Public list-path API: walks `path`, returns directory entries.
    pub fn list_path(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let rec = self.lookup_path(dev, path)?;
        let entries = self.read_directory(dev, rec)?;
        // At the root, `$I30` indexes the canonical system files
        // (`$MFT`, `$Volume`, `$Bitmap`, …, `$Extend`, the reserved
        // slots 12..15). Hide them from the cross-FS view so the
        // generic walker only sees user-visible entries — they're
        // still present on disk for `ntfs-3g` / chkdsk to find.
        let is_root = rec == MFT_RECORD_ROOT;
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            if let Some(fname) = entry.file_name {
                if is_root && fname.name.starts_with('$') {
                    continue;
                }
                let kind = if fname.is_directory() {
                    crate::fs::EntryKind::Dir
                } else {
                    crate::fs::EntryKind::Regular
                };
                // The MFT reference's upper 16 bits are the sequence
                // number; the public inode field is a u32 so we truncate
                // to the low 32 bits of the record number. Callers
                // doing a real cross-FS map should use `lookup_path`.
                let rec_no = (entry.file_ref & 0x0000_FFFF_FFFF_FFFF) as u32;
                let size = if fname.is_directory() {
                    0
                } else {
                    fname.real_size
                };
                out.push(crate::fs::DirEntry {
                    name: fname.name,
                    inode: rec_no,
                    kind,
                    size,
                });
            }
        }
        Ok(out)
    }

    /// Open the default unnamed $DATA stream of `path` as a streaming
    /// reader. The reader pulls one cluster at a time through an
    /// internal scratch buffer.
    pub fn open_file_reader<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<Box<dyn Read + 'a>> {
        let rec_no = self.lookup_path(dev, path)?;
        self.open_stream_by_record(dev, rec_no, "")
    }

    /// Open a named stream by MFT record + name. `""` means the default
    /// unnamed $DATA. Used by both `open_file_reader` and ADS extraction.
    ///
    /// Honours `$ATTRIBUTE_LIST` spill: if the stream's runs are split
    /// across multiple MFT records, all segments are gathered and chained
    /// by `starting_vcn` before the reader is constructed.
    ///
    /// Compressed `$DATA` (LZNT1) is decoded on the fly, one 16-cluster
    /// "compression unit" at a time. Encrypted `$DATA` (EFS) is refused
    /// with [`crate::Error::Unsupported`].
    pub fn open_stream_by_record<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        rec_no: u64,
        stream_name: &str,
    ) -> Result<Box<dyn Read + 'a>> {
        let records = self.load_record_set(dev, rec_no)?;
        let hdr = mft::RecordHeader::parse(&records[0].1)?;
        if !hdr.is_in_use() {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: record {rec_no} is not in use"
            )));
        }

        // Gather every $DATA segment matching the requested name across
        // all records in the set. For non-resident attributes we'll merge
        // their run lists; for resident ones we expect exactly one match.
        let mut resident_bytes: Option<Vec<u8>> = None;
        // (starting_vcn, last_vcn, allocated, real, initialized, comp_unit, runs)
        type Segment = (u64, u64, u64, u64, u64, u8, Vec<Extent>);
        let mut segments: Vec<Segment> = Vec::new();
        let mut is_encrypted = false;
        let mut is_compressed = false;
        for (_rec, rec_buf) in &records {
            let h = mft::RecordHeader::parse(rec_buf)?;
            for attr_res in AttributeIter::new(rec_buf, h.first_attribute_offset as usize) {
                let attr = attr_res?;
                if attr.type_code != TYPE_DATA {
                    continue;
                }
                if attr.name != stream_name {
                    continue;
                }
                if attr.flags & ATTR_FLAG_ENCRYPTED != 0 {
                    is_encrypted = true;
                }
                if attr.flags & ATTR_FLAG_COMPRESSED != 0 {
                    is_compressed = true;
                }
                match attr.kind {
                    AttributeKind::Resident { value, .. } => {
                        resident_bytes = Some(value.to_vec());
                    }
                    AttributeKind::NonResident {
                        starting_vcn,
                        last_vcn,
                        allocated_size,
                        real_size,
                        initialized_size,
                        compression_unit,
                        runs,
                    } => {
                        segments.push((
                            starting_vcn,
                            last_vcn,
                            allocated_size,
                            real_size,
                            initialized_size,
                            compression_unit,
                            runs,
                        ));
                    }
                }
            }
        }

        if is_encrypted {
            return Err(crate::Error::Unsupported(
                "ntfs: encrypted $DATA (EFS) is not supported".into(),
            ));
        }

        if let Some(bytes) = resident_bytes {
            return Ok(Box::new(ResidentReader { bytes, pos: 0 }));
        }

        if segments.is_empty() {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: stream {stream_name:?} not found on record {rec_no}"
            )));
        }

        // Sort and merge segments by starting_vcn. The first segment's
        // header carries the canonical real_size / initialized_size /
        // compression_unit; later segments only contribute runs.
        segments.sort_by_key(|s| s.0);
        let real_size = segments[0].3;
        let initialized_size = segments[0].4;
        let compression_unit = segments[0].5;
        let mut runs: Vec<Extent> = Vec::new();
        for seg in &segments {
            runs.extend(seg.6.iter().copied());
        }

        let cluster_size = self.boot.cluster_size() as u64;
        if is_compressed && compression_unit > 0 {
            let cu_clusters = 1u64 << compression_unit;
            return Ok(Box::new(CompressedReader::new(
                dev,
                cluster_size,
                cu_clusters,
                runs,
                real_size,
                initialized_size,
            )));
        }

        Ok(Box::new(NonResidentReader {
            dev,
            cluster_size,
            runs,
            real_size,
            initialized_size,
            pos: 0,
            cluster_buf: vec![0u8; cluster_size as usize],
            cached_vcn: u64::MAX,
            cached_cluster_filled: false,
        }))
    }

    /// Open the default unnamed $DATA stream of `path` as a seekable
    /// reader. Backs [`crate::fs::Filesystem::open_file_ro`]. The
    /// returned reader is one of resident / non-resident / compressed
    /// depending on how the stream is stored; all three implement
    /// `Read + Seek + FileReadHandle`.
    pub fn open_file_seekable<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<NtfsSeekableReader<'a>> {
        let rec_no = self.lookup_path(dev, path)?;
        let records = self.load_record_set(dev, rec_no)?;
        let hdr = mft::RecordHeader::parse(&records[0].1)?;
        if !hdr.is_in_use() {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: record {rec_no} is not in use"
            )));
        }
        if hdr.is_directory() {
            return Err(crate::Error::InvalidArgument(format!(
                "ntfs: {path:?} is a directory"
            )));
        }

        let stream_name = "";
        let mut resident_bytes: Option<Vec<u8>> = None;
        type Segment = (u64, u64, u64, u64, u64, u8, Vec<Extent>);
        let mut segments: Vec<Segment> = Vec::new();
        let mut is_encrypted = false;
        let mut is_compressed = false;
        for (_rec, rec_buf) in &records {
            let h = mft::RecordHeader::parse(rec_buf)?;
            for attr_res in AttributeIter::new(rec_buf, h.first_attribute_offset as usize) {
                let attr = attr_res?;
                if attr.type_code != TYPE_DATA {
                    continue;
                }
                if attr.name != stream_name {
                    continue;
                }
                if attr.flags & ATTR_FLAG_ENCRYPTED != 0 {
                    is_encrypted = true;
                }
                if attr.flags & ATTR_FLAG_COMPRESSED != 0 {
                    is_compressed = true;
                }
                match attr.kind {
                    AttributeKind::Resident { value, .. } => {
                        resident_bytes = Some(value.to_vec());
                    }
                    AttributeKind::NonResident {
                        starting_vcn,
                        last_vcn,
                        allocated_size,
                        real_size,
                        initialized_size,
                        compression_unit,
                        runs,
                    } => {
                        segments.push((
                            starting_vcn,
                            last_vcn,
                            allocated_size,
                            real_size,
                            initialized_size,
                            compression_unit,
                            runs,
                        ));
                    }
                }
            }
        }
        if is_encrypted {
            return Err(crate::Error::Unsupported(
                "ntfs: encrypted $DATA (EFS) is not supported".into(),
            ));
        }
        if let Some(bytes) = resident_bytes {
            return Ok(NtfsSeekableReader::Resident(ResidentReader {
                bytes,
                pos: 0,
            }));
        }
        if segments.is_empty() {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: stream {stream_name:?} not found on record {rec_no}"
            )));
        }
        segments.sort_by_key(|s| s.0);
        let real_size = segments[0].3;
        let initialized_size = segments[0].4;
        let compression_unit = segments[0].5;
        let mut runs: Vec<Extent> = Vec::new();
        for seg in &segments {
            runs.extend(seg.6.iter().copied());
        }
        let cluster_size = self.boot.cluster_size() as u64;
        if is_compressed && compression_unit > 0 {
            let cu_clusters = 1u64 << compression_unit;
            return Ok(NtfsSeekableReader::Compressed(CompressedReader::new(
                dev,
                cluster_size,
                cu_clusters,
                runs,
                real_size,
                initialized_size,
            )));
        }
        Ok(NtfsSeekableReader::NonResident(NonResidentReader {
            dev,
            cluster_size,
            runs,
            real_size,
            initialized_size,
            pos: 0,
            cluster_buf: vec![0u8; cluster_size as usize],
            cached_vcn: u64::MAX,
            cached_cluster_filled: false,
        }))
    }

    /// Collect cross-FS xattr metadata for `path` using the
    /// `xattr_keys` mapping. Note: streams (ADS) bigger than memory
    /// would be a problem; we cap them at 1 MiB and surface
    /// `Unsupported` if exceeded.
    pub fn read_xattrs(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<HashMap<String, Vec<u8>>> {
        let rec_no = self.lookup_path(dev, path)?;
        let records = self.load_record_set(dev, rec_no)?;

        let mut out: HashMap<String, Vec<u8>> = HashMap::new();
        let mut ads_names: Vec<String> = Vec::new();
        let mut win32_short_name: Option<Vec<u8>> = None;
        let mut security_id: Option<u32> = None;
        let mut have_inline_security = false;
        for (_rec, rec_buf) in &records {
            let h = mft::RecordHeader::parse(rec_buf)?;
            for attr_res in AttributeIter::new(rec_buf, h.first_attribute_offset as usize) {
                let attr = attr_res?;
                match attr.type_code {
                    TYPE_STANDARD_INFORMATION => {
                        if let AttributeKind::Resident { value, .. } = attr.kind {
                            let si = StandardInformation::parse(value)?;
                            out.insert(
                                xattr_keys::DOS_ATTRS.into(),
                                si.file_attributes.to_le_bytes().to_vec(),
                            );
                            out.insert(xattr_keys::TIMES_RAW.into(), si.times_raw().to_vec());
                            // NTFS >= 3.0 extended $STANDARD_INFORMATION
                            // adds owner_id / security_id / quota / USN
                            // starting at offset 0x30. security_id is at
                            // 0x34 — a non-zero value points at $Secure:$SII
                            // for the shared SD.
                            if value.len() >= 0x38 {
                                let id = u32::from_le_bytes(value[0x34..0x38].try_into().unwrap());
                                if id != 0 {
                                    security_id = Some(id);
                                }
                            }
                        }
                    }
                    TYPE_FILE_NAME => {
                        if let AttributeKind::Resident { value, .. } = attr.kind {
                            let fname = FileName::parse(value)?;
                            if fname.namespace == FileName::NAMESPACE_DOS
                                || fname.namespace == FileName::NAMESPACE_WIN32_DOS
                            {
                                let raw_utf16: Vec<u8> = fname
                                    .name
                                    .encode_utf16()
                                    .flat_map(|u| u.to_le_bytes())
                                    .collect();
                                win32_short_name = Some(raw_utf16);
                            }
                        }
                    }
                    TYPE_OBJECT_ID => {
                        if let AttributeKind::Resident { value, .. } = attr.kind {
                            // First 16 bytes are the GUID.
                            let take = value.len().min(16);
                            out.insert(xattr_keys::OBJECT_ID.into(), value[..take].to_vec());
                        }
                    }
                    TYPE_SECURITY_DESCRIPTOR => {
                        if let AttributeKind::Resident { value, .. } = attr.kind {
                            out.insert(xattr_keys::SECURITY.into(), value.to_vec());
                            have_inline_security = true;
                        }
                        // Non-resident inline SDs are unusual; $Secure
                        // handles the common shared-SD case below.
                    }
                    TYPE_REPARSE_POINT => {
                        if let AttributeKind::Resident { value, .. } = attr.kind {
                            out.insert(xattr_keys::REPARSE.into(), value.to_vec());
                        }
                    }
                    TYPE_DATA if !attr.name.is_empty() && !ads_names.contains(&attr.name) => {
                        ads_names.push(attr.name.clone());
                    }
                    _ => {}
                }
            }
        }
        if let Some(name) = win32_short_name {
            out.insert(xattr_keys::SHORT_NAME.into(), name);
        }

        // Resolve shared security descriptor via $Secure if applicable.
        if !have_inline_security
            && let Some(id) = security_id
            && let Some(sd) = self.resolve_security_descriptor(dev, id)?
        {
            out.insert(xattr_keys::SECURITY.into(), sd);
        }

        // Pull each ADS payload through the streaming reader, with a
        // 1 MiB safety cap.
        for name in ads_names {
            let key = format!("{}{}", xattr_keys::ADS_PREFIX, name);
            let mut reader = self.open_stream_by_record(dev, rec_no, &name)?;
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            loop {
                let n = reader.read(&mut chunk).map_err(crate::Error::from)?;
                if n == 0 {
                    break;
                }
                if buf.len() + n > 1024 * 1024 {
                    return Err(crate::Error::Unsupported(format!(
                        "ntfs: ADS {name:?} exceeds 1 MiB cap for xattr passthrough"
                    )));
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            out.insert(key, buf);
        }

        Ok(out)
    }

    /// Lazily load `$UpCase` (record 10) into the `Ntfs::upcase` cache.
    /// Failure to read it falls back to an identity table — synthetic test
    /// images may not carry `$UpCase`, and case-sensitive comparison is
    /// the safe degraded behaviour. Any error encountered is silently
    /// swallowed in favour of the identity table.
    fn ensure_upcase(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        if self.upcase.is_some() {
            return Ok(());
        }
        // Attempt to open record 10's default $DATA stream.
        match self.read_metadata_stream(dev, MFT_RECORD_UPCASE, "", 256 * 1024) {
            Ok(bytes) => {
                self.upcase = Some(UpcaseTable::from_bytes(&bytes));
            }
            Err(_) => {
                self.upcase = Some(UpcaseTable::identity());
            }
        }
        Ok(())
    }

    /// Read up to `cap` bytes of `(rec_no, stream_name)` into a `Vec<u8>`.
    /// Used for `$UpCase` and `$Secure:$SDS`. Bypasses path lookup (so it
    /// doesn't recurse into upcase / the security file itself).
    fn read_metadata_stream(
        &mut self,
        dev: &mut dyn BlockDevice,
        rec_no: u64,
        stream_name: &str,
        cap: usize,
    ) -> Result<Vec<u8>> {
        let mut reader = self.open_stream_by_record(dev, rec_no, stream_name)?;
        let mut out = Vec::new();
        let mut tmp = [0u8; 8192];
        loop {
            let n = reader.read(&mut tmp).map_err(crate::Error::from)?;
            if n == 0 {
                break;
            }
            if out.len() + n > cap {
                let take = cap - out.len();
                out.extend_from_slice(&tmp[..take]);
                break;
            }
            out.extend_from_slice(&tmp[..n]);
        }
        Ok(out)
    }

    /// Resolve a `security_id` (from `$STANDARD_INFORMATION`) to its raw
    /// self-relative `SECURITY_DESCRIPTOR` bytes. The id is keyed through
    /// `$Secure:$SII`, which points at an offset + size within
    /// `$Secure:$SDS`. The SDS entry is prefixed with a 20-byte header
    /// (hash, id, offset, size) followed by the SD payload.
    ///
    /// Returns `Ok(None)` if `$Secure` is missing / unreadable, or if the
    /// id doesn't match any SII entry. Returns `Ok(Some(_))` with the SD
    /// bytes only, capped at `MAX_SECURITY_DESCRIPTOR_BYTES`.
    fn resolve_security_descriptor(
        &mut self,
        dev: &mut dyn BlockDevice,
        security_id: u32,
    ) -> Result<Option<Vec<u8>>> {
        // Build / reuse the $SII cache.
        if self.sii_cache.is_none() {
            let cache = self.build_sii_cache(dev).unwrap_or_default();
            self.sii_cache = Some(cache);
        }
        let cache = self.sii_cache.as_ref().expect("sii_cache populated");
        let Some(&(offset, size)) = cache.get(&security_id) else {
            return Ok(None);
        };
        if size as u64 > MAX_SECURITY_DESCRIPTOR_BYTES {
            return Ok(None);
        }

        // Read `size` bytes from $Secure:$SDS starting at `offset`.
        let mut reader = self.open_stream_by_record(dev, MFT_RECORD_SECURE, "$SDS")?;
        // Skip to `offset`.
        let mut skipped: u64 = 0;
        let mut sink = [0u8; 8192];
        while skipped < offset {
            let want = (offset - skipped).min(sink.len() as u64) as usize;
            let n = reader.read(&mut sink[..want]).map_err(crate::Error::from)?;
            if n == 0 {
                return Ok(None);
            }
            skipped += n as u64;
        }
        let mut blob = vec![0u8; size as usize];
        let mut filled = 0;
        while filled < blob.len() {
            let n = reader
                .read(&mut blob[filled..])
                .map_err(crate::Error::from)?;
            if n == 0 {
                blob.truncate(filled);
                break;
            }
            filled += n;
        }
        if blob.len() < 0x14 {
            return Ok(None);
        }
        // First 20 bytes are the SDS-entry header; the SD payload follows.
        let entry_size = u32::from_le_bytes(blob[16..20].try_into().unwrap()) as usize;
        if entry_size <= 0x14 || entry_size > blob.len() {
            return Ok(None);
        }
        let sd = blob[0x14..entry_size].to_vec();
        Ok(Some(sd))
    }

    /// Walk `$Secure:$SII` (an `$INDEX_ROOT` + optional `$INDEX_ALLOCATION`
    /// keyed by security_id) and return a `security_id -> (offset, size)`
    /// map into `$SDS`.
    fn build_sii_cache(&mut self, dev: &mut dyn BlockDevice) -> Result<HashMap<u32, (u64, u32)>> {
        let records = self.load_record_set(dev, MFT_RECORD_SECURE)?;
        let mut root_value: Option<Vec<u8>> = None;
        let mut alloc_runs: Option<Vec<Extent>> = None;
        let mut index_block_size: u32 = 0;
        for (_rec, rec_buf) in &records {
            let h = mft::RecordHeader::parse(rec_buf)?;
            for attr_res in AttributeIter::new(rec_buf, h.first_attribute_offset as usize) {
                let attr = attr_res?;
                if attr.name != "$SII" {
                    continue;
                }
                match (attr.type_code, attr.kind) {
                    (TYPE_INDEX_ROOT, AttributeKind::Resident { value, .. }) => {
                        let hdr = index::IndexRootHeader::parse(value)?;
                        index_block_size = hdr.index_block_size;
                        root_value = Some(value.to_vec());
                    }
                    (TYPE_INDEX_ALLOCATION, AttributeKind::NonResident { runs, .. }) => {
                        match alloc_runs.as_mut() {
                            Some(existing) => existing.extend(runs),
                            None => alloc_runs = Some(runs),
                        }
                    }
                    _ => {}
                }
            }
        }
        let Some(root_value) = root_value else {
            return Ok(HashMap::new());
        };
        let root_hdr = index::IndexRootHeader::parse(&root_value)?;
        let entries_start = root_hdr.header_offset + root_hdr.first_entry_offset as usize;
        let entries_len = (root_hdr.bytes_in_use as usize).saturating_sub(16);
        let mut cache = HashMap::new();
        let entry_buf = &root_value
            [entries_start..entries_start + entries_len.min(root_value.len() - entries_start)];
        for e in secure::walk_sii_node(entry_buf)? {
            cache.insert(e.security_id, (e.sds_offset, e.sds_size));
        }

        // Walk allocation blocks if any. Each INDX block carries the same
        // entry stream layout we just decoded for the root.
        if let Some(runs) = alloc_runs
            && index_block_size > 0
        {
            let cluster_size = u64::from(self.boot.cluster_size());
            let block_size = index_block_size as usize;
            let mut visited = std::collections::HashSet::<u64>::new();
            // Iterate every block in the run list rather than tree-
            // descending — $SII isn't deep in practice and a flat
            // scan keeps the cache builder simple.
            let mut walked: u64 = 0;
            for ext in &runs {
                let span = ext.length * cluster_size;
                if let Some(lcn) = ext.lcn {
                    let mut local: u64 = 0;
                    while local < span {
                        let phys = lcn * cluster_size + local;
                        if visited.insert(phys) {
                            let mut blk = vec![0u8; block_size];
                            if dev.read_at(phys, &mut blk).is_ok()
                                && mft::apply_fixup(&mut blk, self.boot.bytes_per_sector as usize)
                                    .is_ok()
                                && let Ok(blk_hdr) = index::IndexBlockHeader::parse(&blk)
                            {
                                let s = blk_hdr.entries_start();
                                let l = blk_hdr.entries_byte_len();
                                if s + l <= blk.len() {
                                    let entries = &blk[s..s + l];
                                    if let Ok(rows) = secure::walk_sii_node(entries) {
                                        for r in rows {
                                            cache.insert(r.security_id, (r.sds_offset, r.sds_size));
                                        }
                                    }
                                }
                            }
                        }
                        local += block_size as u64;
                    }
                }
                walked += span;
            }
            let _ = walked;
        }
        Ok(cache)
    }
}

/// Streaming reader over a resident $DATA value (whole payload already in
/// the MFT record).
pub struct ResidentReader {
    bytes: Vec<u8>,
    pos: usize,
}

impl Read for ResidentReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = (self.bytes.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.bytes[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

impl std::io::Seek for ResidentReader {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let total = self.bytes.len() as i128;
        let new = match pos {
            std::io::SeekFrom::Start(n) => n as i128,
            std::io::SeekFrom::Current(d) => self.pos as i128 + d as i128,
            std::io::SeekFrom::End(d) => total + d as i128,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "ntfs: seek to negative offset",
            ));
        }
        self.pos = new as usize;
        Ok(self.pos as u64)
    }
}

/// Streaming reader over a non-resident $DATA stream. Reads at most one
/// cluster from disk at a time into `cluster_buf`. Bytes past
/// `initialized_size` (but before `real_size`) read as zero — that's the
/// NTFS "valid data length" semantics.
pub struct NonResidentReader<'a> {
    dev: &'a mut dyn BlockDevice,
    cluster_size: u64,
    runs: Vec<Extent>,
    real_size: u64,
    initialized_size: u64,
    pos: u64,
    cluster_buf: Vec<u8>,
    cached_vcn: u64,
    cached_cluster_filled: bool,
}

impl<'a> NonResidentReader<'a> {
    /// Find the physical byte offset of VCN `vcn`. Returns `None` for
    /// sparse extents.
    fn map_vcn(&self, vcn: u64) -> std::io::Result<Option<u64>> {
        let mut walked: u64 = 0;
        for ext in &self.runs {
            if vcn < walked + ext.length {
                let local = vcn - walked;
                return Ok(ext.lcn.map(|lcn| (lcn + local) * self.cluster_size));
            }
            walked += ext.length;
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("ntfs: VCN {vcn} past end of run list"),
        ))
    }
}

impl<'a> std::io::Seek for NonResidentReader<'a> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let total = self.real_size as i128;
        let new = match pos {
            std::io::SeekFrom::Start(n) => n as i128,
            std::io::SeekFrom::Current(d) => self.pos as i128 + d as i128,
            std::io::SeekFrom::End(d) => total + d as i128,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "ntfs: seek to negative offset",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

impl<'a> Read for NonResidentReader<'a> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.real_size {
            return Ok(0);
        }
        let cs = self.cluster_size;
        let vcn = self.pos / cs;
        let off = (self.pos % cs) as usize;
        if !self.cached_cluster_filled || self.cached_vcn != vcn {
            let phys = self.map_vcn(vcn)?;
            match phys {
                Some(p) => {
                    self.dev
                        .read_at(p, &mut self.cluster_buf)
                        .map_err(std::io::Error::other)?;
                }
                None => {
                    // Sparse: read as zero.
                    self.cluster_buf.fill(0);
                }
            }
            self.cached_vcn = vcn;
            self.cached_cluster_filled = true;
        }
        let remaining_file = self.real_size - self.pos;
        let n = ((cs - off as u64) as usize)
            .min(out.len())
            .min(remaining_file as usize);
        // Zero-fill the tail past initialized_size.
        if self.pos + n as u64 <= self.initialized_size {
            out[..n].copy_from_slice(&self.cluster_buf[off..off + n]);
        } else if self.pos >= self.initialized_size {
            out[..n].fill(0);
        } else {
            let copy_n = (self.initialized_size - self.pos) as usize;
            out[..copy_n].copy_from_slice(&self.cluster_buf[off..off + copy_n]);
            out[copy_n..n].fill(0);
        }
        self.pos += n as u64;
        Ok(n)
    }
}

/// Streaming reader over an LZNT1-compressed non-resident `$DATA` stream.
///
/// NTFS groups clusters into "compression units" of `1 << compression_unit`
/// clusters each (16 at the canonical 4 KiB-cluster / `compression_unit=4`
/// case). One unit on disk is one of:
///
/// * **All-zero** — the unit's run list slice is wholly sparse. We yield
///   `cu_size` bytes of zero.
/// * **Stored** — exactly `cu_clusters` clusters of real data with no
///   sparse tail; the unit is held verbatim. We pass those bytes through.
/// * **Compressed** — fewer than `cu_clusters` clusters of data followed
///   by sparse tail (NTFS deallocates the saved tail). We LZNT1-decode
///   the real prefix into a `cu_size`-byte buffer.
///
/// The reader keeps one decoded compression unit cached so reads inside
/// the same unit are zero-cost after the initial fetch.
pub struct CompressedReader<'a> {
    dev: &'a mut dyn BlockDevice,
    cluster_size: u64,
    cu_clusters: u64,
    cu_size: u64,
    runs: Vec<Extent>,
    real_size: u64,
    initialized_size: u64,
    pos: u64,
    /// Scratch buffer for one CU's compressed-on-disk bytes (up to
    /// `cu_size`).
    src_buf: Vec<u8>,
    /// Decoded CU contents. Always exactly `cu_size` bytes long.
    out_buf: Vec<u8>,
    /// Which compression unit (counted in CUs from the start of the
    /// attribute) is currently materialized in `out_buf`, or `u64::MAX`
    /// if none.
    cached_cu_index: u64,
}

impl<'a> CompressedReader<'a> {
    fn new(
        dev: &'a mut dyn BlockDevice,
        cluster_size: u64,
        cu_clusters: u64,
        runs: Vec<Extent>,
        real_size: u64,
        initialized_size: u64,
    ) -> Self {
        let cu_size = cluster_size * cu_clusters;
        Self {
            dev,
            cluster_size,
            cu_clusters,
            cu_size,
            runs,
            real_size,
            initialized_size,
            pos: 0,
            src_buf: vec![0u8; cu_size as usize],
            out_buf: vec![0u8; cu_size as usize],
            cached_cu_index: u64::MAX,
        }
    }

    /// Resolve the `i`th run-list cluster (counted as VCN) to its on-disk
    /// (lcn, length-remaining-in-run) tuple, or `None` for sparse.
    fn map_vcn(&self, vcn: u64) -> std::io::Result<Option<u64>> {
        let mut walked: u64 = 0;
        for ext in &self.runs {
            if vcn < walked + ext.length {
                let local = vcn - walked;
                return Ok(ext.lcn.map(|lcn| (lcn + local) * self.cluster_size));
            }
            walked += ext.length;
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("ntfs: VCN {vcn} past end of run list"),
        ))
    }

    /// Walk `cu_clusters` consecutive VCNs and decide how many of them have
    /// a real LCN. The first `real_clusters` VCNs of the CU carry data;
    /// the remaining `cu_clusters - real_clusters` are sparse (i.e. the
    /// compression saved those clusters). All-zero CUs report 0.
    fn count_real_clusters_in_cu(&self, base_vcn: u64) -> std::io::Result<u64> {
        let mut count = 0u64;
        for k in 0..self.cu_clusters {
            let phys = self.map_vcn(base_vcn + k)?;
            if phys.is_some() {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Materialize CU number `cu_index` into `self.out_buf`.
    fn load_cu(&mut self, cu_index: u64) -> std::io::Result<()> {
        if self.cached_cu_index == cu_index {
            return Ok(());
        }
        let base_vcn = cu_index * self.cu_clusters;
        let real_clusters = self.count_real_clusters_in_cu(base_vcn)?;
        if real_clusters == 0 {
            // All-sparse CU → all zero.
            for b in &mut self.out_buf {
                *b = 0;
            }
        } else if real_clusters == self.cu_clusters {
            // Stored verbatim — concatenate every cluster's bytes.
            for k in 0..self.cu_clusters {
                let phys = self
                    .map_vcn(base_vcn + k)?
                    .ok_or_else(|| std::io::Error::other("ntfs: stored-CU sparse cluster"))?;
                let lo = (k * self.cluster_size) as usize;
                let hi = lo + self.cluster_size as usize;
                self.dev
                    .read_at(phys, &mut self.out_buf[lo..hi])
                    .map_err(std::io::Error::other)?;
            }
        } else {
            // Compressed: first `real_clusters` clusters carry an LZNT1
            // stream; the rest of the CU is sparse padding.
            let src_len = (real_clusters * self.cluster_size) as usize;
            self.src_buf.resize(src_len, 0);
            for k in 0..real_clusters {
                let phys = self.map_vcn(base_vcn + k)?.ok_or_else(|| {
                    std::io::Error::other("ntfs: compressed-CU sparse mid-cluster")
                })?;
                let lo = (k * self.cluster_size) as usize;
                let hi = lo + self.cluster_size as usize;
                self.dev
                    .read_at(phys, &mut self.src_buf[lo..hi])
                    .map_err(std::io::Error::other)?;
            }
            compression::decompress_unit(&self.src_buf, &mut self.out_buf)
                .map_err(std::io::Error::other)?;
        }
        self.cached_cu_index = cu_index;
        Ok(())
    }
}

impl<'a> std::io::Seek for CompressedReader<'a> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let total = self.real_size as i128;
        let new = match pos {
            std::io::SeekFrom::Start(n) => n as i128,
            std::io::SeekFrom::Current(d) => self.pos as i128 + d as i128,
            std::io::SeekFrom::End(d) => total + d as i128,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "ntfs: seek to negative offset",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

impl<'a> Read for CompressedReader<'a> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.real_size {
            return Ok(0);
        }
        let cu_index = self.pos / self.cu_size;
        let off = (self.pos % self.cu_size) as usize;
        self.load_cu(cu_index)?;
        let remaining_file = self.real_size - self.pos;
        let n = ((self.cu_size - off as u64) as usize)
            .min(out.len())
            .min(remaining_file as usize);
        if self.pos + n as u64 <= self.initialized_size {
            out[..n].copy_from_slice(&self.out_buf[off..off + n]);
        } else if self.pos >= self.initialized_size {
            out[..n].fill(0);
        } else {
            let copy_n = (self.initialized_size - self.pos) as usize;
            out[..copy_n].copy_from_slice(&self.out_buf[off..off + copy_n]);
            out[copy_n..n].fill(0);
        }
        self.pos += n as u64;
        Ok(n)
    }
}

/// Seekable wrapper over NTFS's three flavours of $DATA reader.
/// Returned by [`Ntfs::open_file_seekable`] and used to back
/// [`crate::fs::Filesystem::open_file_ro`]. Implements
/// `Read + Seek + FileReadHandle`, dispatching to the variant.
pub enum NtfsSeekableReader<'a> {
    Resident(ResidentReader),
    NonResident(NonResidentReader<'a>),
    Compressed(CompressedReader<'a>),
}

impl<'a> Read for NtfsSeekableReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Resident(r) => r.read(buf),
            Self::NonResident(r) => r.read(buf),
            Self::Compressed(r) => r.read(buf),
        }
    }
}

impl<'a> std::io::Seek for NtfsSeekableReader<'a> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        match self {
            Self::Resident(r) => r.seek(pos),
            Self::NonResident(r) => r.seek(pos),
            Self::Compressed(r) => r.seek(pos),
        }
    }
}

impl<'a> crate::fs::FileReadHandle for NtfsSeekableReader<'a> {
    fn len(&self) -> u64 {
        match self {
            Self::Resident(r) => r.bytes.len() as u64,
            Self::NonResident(r) => r.real_size,
            Self::Compressed(r) => r.real_size,
        }
    }
}

/// Names for the xattr namespace this driver will use when round-tripping
/// NTFS metadata through other filesystems.
pub mod xattr_keys {
    /// $STANDARD_INFORMATION.file_attributes (32-bit LE).
    pub const DOS_ATTRS: &str = "user.ntfs.dos_attrs";
    /// $OBJECT_ID GUID (16 raw bytes).
    pub const OBJECT_ID: &str = "user.ntfs.object_id";
    /// Reparse-point tag (LE u32) followed by raw reparse data. The driver
    /// surfaces this as-is and does NOT follow junctions, symlinks, or any
    /// other reparse-point type during path resolution or reads.
    pub const REPARSE: &str = "user.ntfs.reparse";
    /// Alternate Data Streams; full key is `user.ntfs.ads.<name>`.
    pub const ADS_PREFIX: &str = "user.ntfs.ads.";
    /// Self-relative NT SECURITY_DESCRIPTOR blob. Sourced from either a
    /// resident `$SECURITY_DESCRIPTOR` attribute or, when the file uses a
    /// shared SD, resolved via `$STANDARD_INFORMATION.security_id` against
    /// `$Secure:$SII` → `$Secure:$SDS`.
    pub const SECURITY: &str = "system.ntfs_security";
    /// Short 8.3 filename (UTF-16LE from a $FILE_NAME with namespace=DOS).
    pub const SHORT_NAME: &str = "user.ntfs.short_name";
    /// Raw NT-FILETIME quadruple (create, modify, change, access) at 100 ns
    /// granularity, 4 × 8 = 32 bytes LE.
    pub const TIMES_RAW: &str = "user.ntfs.times.raw";
}

// ----------------------------------------------------------------------
// `crate::fs::Filesystem` trait impl — bridges Ntfs into the generic
// walker. Like HfsPlus, `open()` returns a read-only handle; trait
// mutators only succeed after `format()`.
// ----------------------------------------------------------------------

impl crate::fs::FilesystemFactory for Ntfs {
    type FormatOpts = format::FormatOpts;

    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }

    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl crate::fs::Filesystem for Ntfs {
    // Consumes the FileSource during create_file, so let the
    // streaming repack buffer small files in memory instead of
    // spilling each to a temp file (see create_file_streaming).
    fn streams_immediately(&self) -> bool {
        true
    }

    /// NTFS supports full in-place edits. A handle from [`Ntfs::open`]
    /// starts read-only (`writer: None`), but the first mutation lazily
    /// reconstructs the writer state from disk (see
    /// `writer::ensure_writer`), so a reopened image — e.g. one inside a
    /// qcow2 used as a read/write store — accepts add / `open_file_rw`
    /// just like a freshly-formatted one.
    fn mutation_capability(&self) -> crate::fs::MutationCapability {
        crate::fs::MutationCapability::Mutable
    }

    fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        src: crate::fs::FileSource,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ntfs: non-UTF-8 path".into()))?;
        self.create_file(dev, s, src, meta)
    }

    fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ntfs: non-UTF-8 path".into()))?;
        self.create_dir(dev, s, meta)
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
            .ok_or_else(|| crate::Error::InvalidArgument("ntfs: non-UTF-8 path".into()))?;
        let t = target.to_str().ok_or_else(|| {
            crate::Error::InvalidArgument("ntfs: non-UTF-8 symlink target".into())
        })?;
        self.create_symlink(dev, s, t, meta)
    }

    fn create_device(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        kind: crate::fs::DeviceKind,
        major: u32,
        minor: u32,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ntfs: non-UTF-8 path".into()))?;
        self.create_device(dev, s, kind, major, minor, meta)
    }

    fn remove(&mut self, dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ntfs: non-UTF-8 path".into()))?;
        self.remove(dev, s)
    }

    fn list(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ntfs: non-UTF-8 path".into()))?;
        self.list_path(dev, s)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn std::io::Read + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ntfs: non-UTF-8 path".into()))?;
        let r = self.open_file_reader(dev, s)?;
        Ok(Box::new(r))
    }

    /// NTFS has no POSIX ownership/mode; we surface what maps cleanly:
    /// the four timestamps (NT-FILETIME → Unix) and a mode synthesised
    /// from the DOS attribute bits (directory + read-only). uid/gid stay
    /// 0. The kind/size come from the directory index (authoritative, and
    /// the size is what the repack walker streams). Native NTFS metadata
    /// (DOS attrs, ADS, security, …) round-trips via the trait
    /// [`crate::fs::Filesystem::list_xattrs`] impl below.
    fn getattr(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<crate::fs::FileAttrs> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ntfs: non-UTF-8 path".into()))?;
        let norm = s.trim_end_matches('/');
        if norm.is_empty() {
            return Ok(crate::fs::FileAttrs {
                kind: crate::fs::EntryKind::Dir,
                mode: 0o755,
                uid: 0,
                gid: 0,
                size: 0,
                blocks: 0,
                nlink: 2,
                atime: 0,
                mtime: 0,
                ctime: 0,
                rdev: 0,
                inode: MFT_RECORD_ROOT as u32,
            });
        }
        // kind / size / inode from the parent's index — the same source
        // the trait default uses, and the only authoritative file size.
        let (parent, name) = norm.rsplit_once('/').unwrap_or(("", norm));
        let parent = if parent.is_empty() { "/" } else { parent };
        let de = self
            .list_path(dev, parent)?
            .into_iter()
            .find(|e| e.name == name)
            .ok_or_else(|| crate::Error::InvalidArgument(format!("ntfs: no entry at {s:?}")))?;

        // mode + times from $STANDARD_INFORMATION (surfaced as xattrs).
        let xa = self.read_xattrs(dev, s).unwrap_or_default();
        let dos = xa
            .get(xattr_keys::DOS_ATTRS)
            .filter(|v| v.len() >= 4)
            .map(|v| u32::from_le_bytes(v[0..4].try_into().unwrap()))
            .unwrap_or(0);
        let read_only = dos & 0x1 != 0; // FILE_ATTRIBUTE_READONLY
        let mode = match de.kind {
            crate::fs::EntryKind::Dir => 0o755,
            _ if read_only => 0o444,
            _ => 0o644,
        };
        // TIMES_RAW = [creation, modified, mft_changed, accessed] FILETIMEs.
        let filetime_to_unix = |ft: u64| (ft / 10_000_000).saturating_sub(11_644_473_600) as u32;
        let pick = |off: usize| -> u32 {
            xa.get(xattr_keys::TIMES_RAW)
                .filter(|v| v.len() >= off + 8)
                .map(|v| filetime_to_unix(u64::from_le_bytes(v[off..off + 8].try_into().unwrap())))
                .unwrap_or(0)
        };
        Ok(crate::fs::FileAttrs {
            kind: de.kind,
            mode,
            uid: 0,
            gid: 0,
            size: de.size,
            blocks: de.size.div_ceil(512),
            nlink: 1,
            atime: pick(24),
            mtime: pick(8),
            ctime: pick(16),
            rdev: 0,
            inode: de.inode,
        })
    }

    /// Surface NTFS's native metadata (DOS attributes, object id, reparse
    /// data, alternate data streams, security descriptor, short name, raw
    /// timestamps) as `user.ntfs.*` / `system.ntfs_security` xattrs so it
    /// round-trips through repack.
    fn list_xattrs(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::XattrPair>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ntfs: non-UTF-8 path".into()))?;
        let mut pairs: Vec<crate::fs::XattrPair> = self
            .read_xattrs(dev, s)?
            .into_iter()
            .map(|(name, value)| crate::fs::XattrPair { name, value })
            .collect();
        pairs.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(pairs)
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn crate::fs::FileReadHandle + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ntfs: non-UTF-8 path".into()))?;
        let r = self.open_file_seekable(dev, s)?;
        Ok(Box::new(r))
    }

    fn open_file_rw<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
        flags: crate::fs::OpenFlags,
        meta: Option<crate::fs::FileMeta>,
    ) -> Result<Box<dyn crate::fs::FileHandle + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("ntfs: non-UTF-8 path".into()))?;
        self.open_rw(dev, s, flags, meta)
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        Self::flush(self, dev)
    }
}

#[cfg(test)]
mod tests;
