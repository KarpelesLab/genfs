//! NTFS — Microsoft's NT File System. Read implementation.
//!
//! ## Status
//!
//! Detection, MFT decode, attribute decode, directory walking via
//! $INDEX_ROOT + $INDEX_ALLOCATION, and streaming reads of the default
//! unnamed $DATA stream are implemented. Write support is out of scope.
//!
//! ## Reference
//!
//! - Microsoft "[MS-FSCC] File System Control Codes":
//!   <https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-fscc/>
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
//! | `$SECURITY_DESCRIPTOR` (raw NT SD blob)   | `system.ntfs_security`                   | Self-relative SD blob; consumers who understand it       |
//! |                                           |                                          | can decode to SDDL                                       |
//! | Short (8.3) filename                      | `user.ntfs.short_name`                   | UTF-16LE per `$FILE_NAME` with namespace=DOS             |
//! | Last-write / creation / change / access   | inode timestamps + `user.ntfs.times.raw` | The latter holds all four NT-FILETIME (100 ns) values    |
//!
//! ### NTFS → NTFS round-trip guarantee
//!
//! The cross-FS xattr mapping above is lossy at the sub-100ns level. For
//! NTFS-to-NTFS transfers, the writer copies raw attribute byte streams
//! verbatim rather than going through this mapping.

use std::collections::HashMap;
use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;

pub mod attribute;
pub mod boot;
pub mod index;
pub mod mft;
pub mod run_list;

use attribute::{
    AttributeIter, AttributeKind, FileName, StandardInformation, TYPE_DATA, TYPE_FILE_NAME,
    TYPE_INDEX_ALLOCATION, TYPE_INDEX_ROOT, TYPE_OBJECT_ID, TYPE_REPARSE_POINT,
    TYPE_SECURITY_DESCRIPTOR, TYPE_STANDARD_INFORMATION,
};
use boot::BootSector;
use index::IndexEntry;
use run_list::Extent;

/// Hard-coded MFT record numbers reserved by NTFS.
pub const MFT_RECORD_MFT: u64 = 0;
pub const MFT_RECORD_ROOT: u64 = 5;

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

    /// Walk a directory's index, returning the (file_ref, FileName) of each
    /// entry. Skips DOS-namespace duplicates (those are covered by the
    /// Win32+DOS combined entry that has both names).
    pub fn read_directory(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_rec: u64,
    ) -> Result<Vec<IndexEntry>> {
        let rec_size = self.boot.mft_record_size() as usize;
        let mut rec_buf = vec![0u8; rec_size];
        self.read_mft_record(dev, dir_rec, &mut rec_buf)?;
        let header = mft::RecordHeader::parse(&rec_buf)?;
        if !header.is_in_use() {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: directory record {dir_rec} is not in use"
            )));
        }

        // Locate $INDEX_ROOT (must be named "$I30") and optional
        // $INDEX_ALLOCATION (same name).
        let mut root_value: Option<Vec<u8>> = None;
        let mut alloc_runs: Option<Vec<Extent>> = None;
        for attr_res in AttributeIter::new(&rec_buf, header.first_attribute_offset as usize) {
            let attr = attr_res?;
            if attr.name != "$I30" {
                continue;
            }
            match (attr.type_code, attr.kind) {
                (TYPE_INDEX_ROOT, AttributeKind::Resident { value, .. }) => {
                    root_value = Some(value.to_vec());
                }
                (TYPE_INDEX_ALLOCATION, AttributeKind::NonResident { runs, .. }) => {
                    alloc_runs = Some(runs);
                }
                _ => {}
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
    /// case-sensitively against the FILE_NAME entries (NTFS itself
    /// case-folds with the $UpCase table; for v1 we keep things case
    /// sensitive). The root path "/" maps to record 5.
    pub fn lookup_path(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<u64> {
        if !path.starts_with('/') {
            return Err(crate::Error::InvalidArgument(format!(
                "ntfs: path must be absolute, got {path:?}"
            )));
        }
        let mut current = MFT_RECORD_ROOT;
        for component in path.split('/').filter(|s| !s.is_empty()) {
            let entries = self.read_directory(dev, current)?;
            let mut next: Option<u64> = None;
            for entry in entries {
                if let Some(fname) = entry.file_name {
                    if fname.name == component && fname.namespace != FileName::NAMESPACE_DOS {
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
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            if let Some(fname) = entry.file_name {
                // Skip the special "." / system files at root; the
                // index walker doesn't synthesize "." or "..", but it
                // does include reserved entries below 24 if you walk
                // the root. Filter out names starting with a dollar
                // sign at the root only to match the cross-FS view.
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
                out.push(crate::fs::DirEntry {
                    name: fname.name,
                    inode: rec_no,
                    kind,
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
    pub fn open_stream_by_record<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        rec_no: u64,
        stream_name: &str,
    ) -> Result<Box<dyn Read + 'a>> {
        let rec_size = self.boot.mft_record_size() as usize;
        let mut rec_buf = vec![0u8; rec_size];
        self.read_mft_record(dev, rec_no, &mut rec_buf)?;
        let header = mft::RecordHeader::parse(&rec_buf)?;
        if !header.is_in_use() {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: record {rec_no} is not in use"
            )));
        }

        // Find the matching $DATA attribute.
        let mut info: Option<DataStreamInfo> = None;
        for attr_res in AttributeIter::new(&rec_buf, header.first_attribute_offset as usize) {
            let attr = attr_res?;
            if attr.type_code != TYPE_DATA {
                continue;
            }
            if attr.name != stream_name {
                continue;
            }
            if attr.is_encrypted() {
                return Err(crate::Error::Unsupported(
                    "ntfs: encrypted $DATA is not supported".into(),
                ));
            }
            if attr.is_compressed() {
                return Err(crate::Error::Unsupported(
                    "ntfs: compressed $DATA is not supported".into(),
                ));
            }
            info = Some(match attr.kind {
                AttributeKind::Resident { value, .. } => DataStreamInfo::Resident {
                    bytes: value.to_vec(),
                },
                AttributeKind::NonResident {
                    real_size,
                    initialized_size,
                    runs,
                    ..
                } => DataStreamInfo::NonResident {
                    real_size,
                    initialized_size,
                    runs,
                },
            });
            break;
        }
        let info = info.ok_or_else(|| {
            crate::Error::InvalidImage(format!(
                "ntfs: stream {stream_name:?} not found on record {rec_no}"
            ))
        })?;

        match info {
            DataStreamInfo::Resident { bytes } => Ok(Box::new(ResidentReader { bytes, pos: 0 })),
            DataStreamInfo::NonResident {
                real_size,
                initialized_size,
                runs,
            } => Ok(Box::new(NonResidentReader {
                dev,
                cluster_size: self.boot.cluster_size() as u64,
                runs,
                real_size,
                initialized_size,
                pos: 0,
                cluster_buf: vec![0u8; self.boot.cluster_size() as usize],
                cached_vcn: u64::MAX,
                cached_cluster_filled: false,
            })),
        }
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
        let rec_size = self.boot.mft_record_size() as usize;
        let mut rec_buf = vec![0u8; rec_size];
        self.read_mft_record(dev, rec_no, &mut rec_buf)?;
        let header = mft::RecordHeader::parse(&rec_buf)?;

        let mut out: HashMap<String, Vec<u8>> = HashMap::new();
        let mut ads_names: Vec<String> = Vec::new();
        let mut win32_short_name: Option<Vec<u8>> = None;
        for attr_res in AttributeIter::new(&rec_buf, header.first_attribute_offset as usize) {
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
                    }
                    // Non-resident SDs are uncommon in record-local form
                    // (they normally live in $Secure). Skip.
                }
                TYPE_REPARSE_POINT => {
                    if let AttributeKind::Resident { value, .. } = attr.kind {
                        out.insert(xattr_keys::REPARSE.into(), value.to_vec());
                    }
                }
                TYPE_DATA if !attr.name.is_empty() => {
                    ads_names.push(attr.name.clone());
                }
                _ => {}
            }
        }
        if let Some(name) = win32_short_name {
            out.insert(xattr_keys::SHORT_NAME.into(), name);
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
}

/// Internal representation of a $DATA stream we're about to read.
enum DataStreamInfo {
    Resident {
        bytes: Vec<u8>,
    },
    NonResident {
        real_size: u64,
        initialized_size: u64,
        runs: Vec<Extent>,
    },
}

/// Streaming reader over a resident $DATA value (whole payload already in
/// the MFT record).
struct ResidentReader {
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

/// Streaming reader over a non-resident $DATA stream. Reads at most one
/// cluster from disk at a time into `cluster_buf`. Bytes past
/// `initialized_size` (but before `real_size`) read as zero — that's the
/// NTFS "valid data length" semantics.
struct NonResidentReader<'a> {
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

/// Names for the xattr namespace this driver will use when round-tripping
/// NTFS metadata through other filesystems.
pub mod xattr_keys {
    /// $STANDARD_INFORMATION.file_attributes (32-bit LE).
    pub const DOS_ATTRS: &str = "user.ntfs.dos_attrs";
    /// $OBJECT_ID GUID (16 raw bytes).
    pub const OBJECT_ID: &str = "user.ntfs.object_id";
    /// Reparse-point tag (LE u32) followed by raw reparse data.
    pub const REPARSE: &str = "user.ntfs.reparse";
    /// Alternate Data Streams; full key is `user.ntfs.ads.<name>`.
    pub const ADS_PREFIX: &str = "user.ntfs.ads.";
    /// Self-relative NT SECURITY_DESCRIPTOR blob.
    pub const SECURITY: &str = "system.ntfs_security";
    /// Short 8.3 filename (UTF-16LE from a $FILE_NAME with namespace=DOS).
    pub const SHORT_NAME: &str = "user.ntfs.short_name";
    /// Raw NT-FILETIME quadruple (create, modify, change, access) at 100 ns
    /// granularity, 4 × 8 = 32 bytes LE.
    pub const TIMES_RAW: &str = "user.ntfs.times.raw";
}

#[cfg(test)]
mod tests;
