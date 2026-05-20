//! HFS+ writer — TN1150 "Volume Initialization" + "Catalog File".
//!
//! This module turns the read-only HFS+ implementation into a one-shot
//! filesystem builder. Layout follows TN1150 with these v1 choices:
//!
//! * `nodeSize = 8192` for all B-trees (the HFS+ default).
//! * Single contiguous extent for every special file: allocation bitmap,
//!   extents-overflow, catalog, attributes (empty), startup (empty).
//! * Plain HFS+ (signature `"H+"`, version 4). HFSX, journaling, and
//!   resource forks are intentionally out of scope.
//! * No iCloud/Spotlight/HFS+ private data directories — these only
//!   appear on volumes with hard links.
//!
//! ## On-disk layout we emit (in allocation-block order)
//!
//! ```text
//! block 0                : volume header (1 KiB pad + 512 B header)
//! block 1..              : allocation bitmap (size depends on volume)
//!     ..N                : extents-overflow B-tree (one node = 8 KiB)
//!     ..M                : catalog B-tree
//!     ..                 : user data (allocated bump-style during build)
//!     last 1 block       : alternate volume header (last 1 KiB of last block)
//! ```
//!
//! All special-file fork data is kept as one contiguous extent so the
//! reader's `ForkReader::from_inline` works without consulting the
//! extents-overflow tree.

use std::collections::BTreeMap;
use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;

use super::btree::{HEADER_REC_SIZE, KIND_HEADER, KIND_INDEX, KIND_LEAF, NODE_DESCRIPTOR_SIZE};
use super::catalog::{
    REC_FILE, REC_FILE_THREAD, REC_FOLDER, REC_FOLDER_THREAD, ROOT_FOLDER_ID, ROOT_PARENT_ID,
    UniStr, compare_unistr,
};
use super::extents::{EXTENT_KEY_PAYLOAD_LEN, EXTENT_RECORD_SIZE, FORK_DATA};
use super::volume_header::{
    ExtentDescriptor, FORK_DATA_SIZE, FORK_EXTENT_COUNT, ForkData, SIG_HFS_PLUS,
    VOLUME_HEADER_OFFSET, VolumeHeader,
};

/// Default B-tree node size for fresh volumes (8 KiB, matching mkfs.hfsplus).
pub const DEFAULT_NODE_SIZE: u32 = 8192;

/// Default allocation block size for fresh volumes (4 KiB).
pub const DEFAULT_BLOCK_SIZE: u32 = 4096;

/// Volume attribute set on a cleanly unmounted volume.
pub const VOL_ATTR_UNMOUNTED: u32 = 1 << 8;

/// `kHFSVolumeJournaledMask` per TN1150: set in
/// `HFSPlusVolumeHeader.attributes` when the volume carries a journal.
pub const VOL_ATTR_JOURNALED: u32 = 1 << 13;

/// `JournalInfoBlock.flags`: journal lives inside the filesystem (the
/// volume's own journal-buffer reserved blocks rather than an external
/// device). TN1150 names this `kJIJournalInFSMask`.
pub const JI_JOURNAL_IN_FS: u32 = 0x0000_0002;

/// Journal-header magic number, ASCII `"JNLx"` in big-endian word order.
pub const JOURNAL_HEADER_MAGIC: u32 = 0x4a4e_4c78;

/// Journal-header endian marker. Apple writes `0x12345678` natively
/// (host byte order on the writing machine); we write it big-endian
/// because every other multi-byte field in an HFS+ volume header is
/// big-endian, so a kernel comparing the on-disk value to its native
/// representation will detect endian mismatch the same way it does for
/// the rest of the volume.
pub const JOURNAL_HEADER_ENDIAN: u32 = 0x1234_5678;

/// Default size of the in-volume journal buffer when `FormatOpts::journaled`
/// is set, expressed in allocation blocks. macOS picks 8 MiB by default;
/// we ship a stub journal sized at 16 blocks (64 KiB at 4 KiB block_size),
/// which is enough room to hold a credible empty journal header and to
/// satisfy a kernel that mounts the volume read-only.
pub const DEFAULT_JOURNAL_BUFFER_BLOCKS: u32 = 16;

/// File-type modes used on disk (`HFSPlusBSDInfo.fileMode`).
mod m {
    pub const S_IFDIR: u16 = 0o040000;
    pub const S_IFREG: u16 = 0o100000;
    pub const S_IFLNK: u16 = 0o120000;
}

/// Options for [`super::HfsPlus::format`].
#[derive(Debug, Clone)]
pub struct FormatOpts {
    /// Allocation block size in bytes (must be a power of two ≥ 512).
    pub block_size: u32,
    /// B-tree node size in bytes (must be a power of two ≥ 512).
    pub node_size: u32,
    /// Number of allocation blocks reserved for the extents-overflow
    /// B-tree. v1 ships an empty single-node tree; one node is enough.
    pub extents_nodes: u32,
    /// Initial number of allocation blocks reserved for the catalog
    /// B-tree. The catalog grows in place — if your build outgrows this
    /// allotment, [`super::HfsPlus::flush`] returns an error. Sized
    /// generously (32 nodes by default).
    pub catalog_nodes: u32,
    /// UTF-8 volume name. Stored on the root folder's thread record.
    pub volume_name: String,
    /// Seconds since 1904-01-01 used for createDate / modifyDate.
    /// HFS+ uses a 1904 epoch; supply `0` to leave dates zeroed.
    pub create_date: u32,
    /// Emit a journal stub: reserve a journal-info block + journal
    /// buffer, set the `kHFSVolumeJournaledMask` bit, and lay down a
    /// clean (transaction-count-zero) journal header. The kernel will
    /// see a journaled volume with no replay work to do. Default false.
    pub journaled: bool,
}

impl Default for FormatOpts {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            node_size: DEFAULT_NODE_SIZE,
            // 4 = 1 header + 3 leaves' worth of headroom; enough for
            // ~300 overflow records before the tree fills, while still
            // tiny on disk (32 KiB at default node_size). Volumes
            // without any fragmented files leave most of this empty.
            extents_nodes: 4,
            catalog_nodes: 32,
            volume_name: "Untitled".into(),
            create_date: 0,
            journaled: false,
        }
    }
}

/// A catalog key in the form we keep in memory while building. We
/// store the raw `UniStr` plus the case-folded form so the BTreeMap
/// ordering matches the HFS+ ordering rules exactly. Always built in
/// case-insensitive mode (plain HFS+).
#[derive(Debug, Clone, Eq)]
pub(crate) struct OwnedKey {
    pub parent_id: u32,
    pub name: UniStr,
}

impl OwnedKey {
    fn thread(cnid: u32) -> Self {
        Self {
            parent_id: cnid,
            name: UniStr::default(),
        }
    }
}

impl PartialEq for OwnedKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl PartialOrd for OwnedKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OwnedKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.parent_id.cmp(&other.parent_id) {
            std::cmp::Ordering::Equal => compare_unistr(&self.name, &other.name, false),
            o => o,
        }
    }
}

/// In-memory writable state attached to an [`super::HfsPlus`] for builds.
///
/// During the build phase the on-disk image is incomplete: bitmaps,
/// catalog nodes and the volume header are only written by [`flush`].
/// Regular-file data, however, *is* streamed straight to disk as
/// `create_file` runs — that's how we keep memory bounded.
pub struct Writer {
    pub(crate) block_size: u32,
    pub(crate) node_size: u32,
    pub(crate) total_blocks: u32,
    pub(crate) volume_name: String,
    pub(crate) create_date: u32,
    /// Next CNID to hand out for new files / folders.
    pub(crate) next_cnid: u32,
    /// In-memory allocation bitmap. `bits[i]` set => block `i` is in use.
    /// The bitmap itself lives in the allocation file's extents on disk.
    pub(crate) bitmap: Vec<u8>,
    /// Bump-pointer cursor for the allocator. We use first-fit only after
    /// `remove` has freed blocks; otherwise the cursor advances monotonically.
    pub(crate) next_alloc: u32,
    pub(crate) free_blocks: u32,

    /// Catalog records, keyed in HFS+ catalog-order. Values are the
    /// encoded record bytes (without the leading key — we re-encode the
    /// key from the BTreeMap key on flush).
    pub(crate) catalog: BTreeMap<OwnedKey, Vec<u8>>,

    /// Extents-overflow records keyed by `(fork_type, file_id, start_block)`.
    /// Each value is a fixed-size group of up to eight `(start, count)`
    /// runs (zero-count terminated). Populated whenever a fork's run
    /// list outgrows the eight inline extents kept in the catalog.
    pub(crate) overflow_extents: BTreeMap<(u8, u32, u32), [ExtentDescriptor; FORK_EXTENT_COUNT]>,

    /// Fork data for the five special files. The extents we record here
    /// are immutable for the lifetime of the build (we size up front and
    /// don't grow them) so a fork that runs out of space is a hard error.
    pub(crate) allocation_file: ForkData,
    pub(crate) extents_file: ForkData,
    pub(crate) catalog_file: ForkData,
    pub(crate) attributes_file: ForkData,
    pub(crate) startup_file: ForkData,

    /// CNID of the `\0\0\0\0HFS+ Private Data` directory, created lazily
    /// on the first `create_hardlink` call. `None` on volumes that
    /// have never seen a hard link.
    pub(crate) private_dir_cnid: Option<u32>,

    /// Journal-info block (allocation-block number) and journal-buffer
    /// start + length (also in allocation blocks), set on a journaled
    /// format. Both are zero on an unjournaled volume.
    pub(crate) journal_info_block: u32,
    pub(crate) journal_buffer_start: u32,
    pub(crate) journal_buffer_blocks: u32,

    /// True once [`flush`] has been called successfully.
    pub(crate) flushed: bool,
}

impl Writer {
    /// Whether `cnid` is currently a directory in the in-memory catalog.
    pub(crate) fn is_dir(&self, cnid: u32) -> bool {
        let key = OwnedKey::thread(cnid);
        let Some(body) = self.catalog.get(&key) else {
            return false;
        };
        body.len() >= 2 && i16::from_be_bytes([body[0], body[1]]) == REC_FOLDER_THREAD
    }

    /// Return the CNID of the child named `name` under `parent_id`, if any.
    pub(crate) fn lookup(&self, parent_id: u32, name: &UniStr) -> Option<(OwnedKey, u32, i16)> {
        let key = OwnedKey {
            parent_id,
            name: name.clone(),
        };
        let body = self.catalog.get(&key)?;
        if body.len() < 12 {
            return None;
        }
        let rec_type = i16::from_be_bytes([body[0], body[1]]);
        let cnid = match rec_type {
            REC_FOLDER => u32::from_be_bytes(body[8..12].try_into().unwrap()),
            REC_FILE => u32::from_be_bytes(body[8..12].try_into().unwrap()),
            _ => return None,
        };
        Some((key, cnid, rec_type))
    }

    /// Increment the valence (child count) of `parent_id` by `delta`,
    /// in-place inside the encoded folder record. Silently does nothing
    /// if the parent isn't a folder record (used for root, where the
    /// folder record is keyed under `(ROOT_PARENT_ID, volume_name)`).
    pub(crate) fn bump_valence(&mut self, parent_id: u32, delta: i32) -> Result<()> {
        // We need the (parent_parent, parent_name) of `parent_id`, which
        // is what its thread record stores.
        let thread_key = OwnedKey::thread(parent_id);
        let Some(thread_body) = self.catalog.get(&thread_key) else {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+ writer: no thread record for parent CNID {parent_id}"
            )));
        };
        if thread_body.len() < 8 {
            return Err(crate::Error::InvalidImage(
                "hfs+ writer: short thread record".into(),
            ));
        }
        let pp = u32::from_be_bytes(thread_body[4..8].try_into().unwrap());
        let (pname, _) = UniStr::decode(&thread_body[8..])?;
        let folder_key = OwnedKey {
            parent_id: pp,
            name: pname,
        };
        let body = self.catalog.get_mut(&folder_key).ok_or_else(|| {
            crate::Error::InvalidImage(format!(
                "hfs+ writer: no folder record for CNID {parent_id}"
            ))
        })?;
        if body.len() < 8 || i16::from_be_bytes([body[0], body[1]]) != REC_FOLDER {
            return Err(crate::Error::InvalidImage(
                "hfs+ writer: parent CNID does not name a folder record".into(),
            ));
        }
        let cur = i64::from(u32::from_be_bytes(body[4..8].try_into().unwrap()));
        let new = (cur + i64::from(delta)).max(0) as u32;
        body[4..8].copy_from_slice(&new.to_be_bytes());
        Ok(())
    }

    /// Allocate `n` contiguous blocks. Bump-pointer first; on failure
    /// scans the bitmap for the first free run of `n` blocks. Returns
    /// the start block index.
    pub(crate) fn allocate(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Err(crate::Error::InvalidArgument(
                "hfs+ writer: zero-block allocation".into(),
            ));
        }
        if self.free_blocks < n {
            return Err(crate::Error::Unsupported(format!(
                "hfs+ writer: out of space ({} free, need {n})",
                self.free_blocks
            )));
        }
        // Try the bump cursor first — common path during build.
        if self.next_alloc + n <= self.total_blocks && self.range_is_free(self.next_alloc, n) {
            let start = self.next_alloc;
            self.set_used(start, n);
            self.next_alloc = start + n;
            self.free_blocks -= n;
            return Ok(start);
        }
        // Fallback: first-fit scan.
        let start = self.first_fit(n).ok_or_else(|| {
            crate::Error::Unsupported(format!(
                "hfs+ writer: fragmented; no run of {n} blocks available"
            ))
        })?;
        self.set_used(start, n);
        if start + n > self.next_alloc {
            self.next_alloc = start + n;
        }
        self.free_blocks -= n;
        Ok(start)
    }

    /// Allocate up to `max` contiguous free blocks (at least 1, exactly
    /// the size of the largest free run discoverable from the bump
    /// cursor / first-fit scan, capped at `max`). Used by the streaming
    /// file writer so it can lay user data down across multiple runs
    /// when no single run is big enough.
    pub(crate) fn allocate_largest_run(&mut self, max: u32) -> Result<ExtentDescriptor> {
        if max == 0 {
            return Err(crate::Error::InvalidArgument(
                "hfs+ writer: zero-block allocation".into(),
            ));
        }
        if self.free_blocks == 0 {
            return Err(crate::Error::Unsupported(
                "hfs+ writer: out of space (0 free blocks)".into(),
            ));
        }
        // Find the largest free run discoverable from the start of the
        // bitmap. Prefer one that begins at or after `next_alloc` so we
        // keep the bump-pointer behaviour on the happy path.
        let mut best: Option<(u32, u32)> = None;
        let mut run_start: u32 = 0;
        let mut run_len: u32 = 0;
        for bit in 0..self.total_blocks {
            let by = (bit / 8) as usize;
            let mask = 1u8 << (7 - (bit & 7));
            if self.bitmap[by] & mask == 0 {
                if run_len == 0 {
                    run_start = bit;
                }
                run_len += 1;
                if run_len >= max {
                    best = Some((run_start, max));
                    break;
                }
            } else {
                if run_len > 0 {
                    let candidate = (run_start, run_len);
                    match best {
                        Some((_, blen)) if blen >= run_len => {}
                        _ => best = Some(candidate),
                    }
                }
                run_len = 0;
            }
        }
        if let Some((s, l)) = best {
            let take = l.min(max);
            self.set_used(s, take);
            if s + take > self.next_alloc {
                self.next_alloc = s + take;
            }
            self.free_blocks -= take;
            return Ok(ExtentDescriptor {
                start_block: s,
                block_count: take,
            });
        }
        // Tail of the scan: a free run still in progress.
        if run_len > 0 {
            let take = run_len.min(max);
            self.set_used(run_start, take);
            if run_start + take > self.next_alloc {
                self.next_alloc = run_start + take;
            }
            self.free_blocks -= take;
            return Ok(ExtentDescriptor {
                start_block: run_start,
                block_count: take,
            });
        }
        Err(crate::Error::Unsupported(
            "hfs+ writer: fragmented bitmap reports free blocks but no run found".into(),
        ))
    }

    /// Free `n` blocks starting at `start`. Clears the corresponding
    /// bitmap bits and grows `free_blocks` accordingly.
    pub(crate) fn free(&mut self, start: u32, n: u32) {
        for i in 0..n {
            let bit = start + i;
            let by = (bit / 8) as usize;
            let mask = 1u8 << (7 - (bit & 7));
            if by < self.bitmap.len() && self.bitmap[by] & mask != 0 {
                self.bitmap[by] &= !mask;
                self.free_blocks += 1;
            }
        }
        // Allow the next bump alloc to consider freshly freed blocks
        // by rewinding when appropriate.
        if start < self.next_alloc {
            self.next_alloc = start;
        }
    }

    fn range_is_free(&self, start: u32, n: u32) -> bool {
        for i in 0..n {
            let bit = start + i;
            let by = (bit / 8) as usize;
            let mask = 1u8 << (7 - (bit & 7));
            if by >= self.bitmap.len() {
                return false;
            }
            if self.bitmap[by] & mask != 0 {
                return false;
            }
        }
        true
    }

    fn set_used(&mut self, start: u32, n: u32) {
        for i in 0..n {
            let bit = start + i;
            let by = (bit / 8) as usize;
            let mask = 1u8 << (7 - (bit & 7));
            self.bitmap[by] |= mask;
        }
    }

    fn first_fit(&self, n: u32) -> Option<u32> {
        let mut run: u32 = 0;
        let mut run_start: u32 = 0;
        for bit in 0..self.total_blocks {
            let by = (bit / 8) as usize;
            let mask = 1u8 << (7 - (bit & 7));
            if self.bitmap[by] & mask == 0 {
                if run == 0 {
                    run_start = bit;
                }
                run += 1;
                if run == n {
                    return Some(run_start);
                }
            } else {
                run = 0;
            }
        }
        None
    }
}

// ----------------------------------------------------------------------
// Encoding helpers
// ----------------------------------------------------------------------

/// Encode an HFSUniStr255 (u16 length + UTF-16 BE code units).
fn encode_unistr(s: &UniStr, out: &mut Vec<u8>) {
    let n = s.code_units.len().min(255) as u16;
    out.extend_from_slice(&n.to_be_bytes());
    for &cu in s.code_units.iter().take(255) {
        out.extend_from_slice(&cu.to_be_bytes());
    }
}

/// Encode a catalog key. Returns the encoded bytes (padded to even length).
pub(crate) fn encode_catalog_key(parent_id: u32, name: &UniStr) -> Vec<u8> {
    // key_length covers parentID (4) + HFSUniStr255 (2 + 2*N).
    let n = name.code_units.len().min(255);
    let key_length = (4 + 2 + 2 * n) as u16;
    let mut out = Vec::with_capacity(2 + key_length as usize + 1);
    out.extend_from_slice(&key_length.to_be_bytes());
    out.extend_from_slice(&parent_id.to_be_bytes());
    encode_unistr(name, &mut out);
    if out.len() % 2 != 0 {
        out.push(0);
    }
    out
}

/// Encode an extents-overflow key. Always 12 bytes (already even).
fn encode_extent_key(fork_type: u8, file_id: u32, start_block: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + EXTENT_KEY_PAYLOAD_LEN);
    out.extend_from_slice(&(EXTENT_KEY_PAYLOAD_LEN as u16).to_be_bytes());
    out.push(fork_type);
    out.push(0);
    out.extend_from_slice(&file_id.to_be_bytes());
    out.extend_from_slice(&start_block.to_be_bytes());
    out
}

/// Encode an HFSPlusBSDInfo (16 bytes).
fn encode_bsd(out: &mut Vec<u8>, file_mode: u16, uid: u32, gid: u32, special: u32) {
    out.extend_from_slice(&uid.to_be_bytes());
    out.extend_from_slice(&gid.to_be_bytes());
    out.push(0); // admin_flags
    out.push(0); // owner_flags
    out.extend_from_slice(&file_mode.to_be_bytes());
    out.extend_from_slice(&special.to_be_bytes());
}

/// Encode an HFSPlusForkData (80 bytes) by hand-rolling — `ForkData` in
/// the read code is `Copy` but lacks `encode`.
fn encode_fork(fork: &ForkData, out: &mut Vec<u8>) {
    out.extend_from_slice(&fork.logical_size.to_be_bytes());
    out.extend_from_slice(&fork.clump_size.to_be_bytes());
    out.extend_from_slice(&fork.total_blocks.to_be_bytes());
    for ext in &fork.extents {
        out.extend_from_slice(&ext.start_block.to_be_bytes());
        out.extend_from_slice(&ext.block_count.to_be_bytes());
    }
}

/// Encode a `HFSPlusCatalogFolder` record body (88 bytes).
pub(crate) fn encode_folder_body(
    folder_id: u32,
    valence: u32,
    file_mode: u16,
    uid: u32,
    gid: u32,
    create_date: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(88);
    out.extend_from_slice(&REC_FOLDER.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes()); // flags
    out.extend_from_slice(&valence.to_be_bytes()); // valence
    out.extend_from_slice(&folder_id.to_be_bytes()); // folderID
    // five 4-byte dates
    for _ in 0..5 {
        out.extend_from_slice(&create_date.to_be_bytes());
    }
    // BSDInfo (16 bytes)
    encode_bsd(&mut out, file_mode, uid, gid, 0);
    // FolderInfo (16 bytes) — leave as zero
    out.extend_from_slice(&[0u8; 16]);
    // ExtendedFolderInfo (16 bytes) — leave as zero
    out.extend_from_slice(&[0u8; 16]);
    out.extend_from_slice(&0u32.to_be_bytes()); // textEncoding
    out.extend_from_slice(&0u32.to_be_bytes()); // reserved
    debug_assert_eq!(out.len(), 88);
    out
}

/// Encode an `HFSPlusCatalogFile` record body (248 bytes).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_file_body(
    file_id: u32,
    file_mode: u16,
    uid: u32,
    gid: u32,
    create_date: u32,
    file_type: [u8; 4],
    creator: [u8; 4],
    data_fork: &ForkData,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(248);
    out.extend_from_slice(&REC_FILE.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes()); // flags
    out.extend_from_slice(&0u32.to_be_bytes()); // reserved1
    out.extend_from_slice(&file_id.to_be_bytes());
    // five 4-byte dates
    for _ in 0..5 {
        out.extend_from_slice(&create_date.to_be_bytes());
    }
    encode_bsd(&mut out, file_mode, uid, gid, 0);
    // FileInfo (16 bytes): fileType, creator, then 8 reserved bytes.
    out.extend_from_slice(&file_type);
    out.extend_from_slice(&creator);
    out.extend_from_slice(&[0u8; 8]);
    // ExtendedFileInfo (16 bytes) — leave as zero
    out.extend_from_slice(&[0u8; 16]);
    out.extend_from_slice(&0u32.to_be_bytes()); // textEncoding
    out.extend_from_slice(&0u32.to_be_bytes()); // reserved2
    encode_fork(data_fork, &mut out);
    // Resource fork: empty.
    encode_fork(&ForkData::default(), &mut out);
    debug_assert_eq!(out.len(), 248);
    out
}

/// Encode a thread record body. `record_type` should be `REC_FOLDER_THREAD`
/// or `REC_FILE_THREAD`.
pub(crate) fn encode_thread_body(record_type: i16, parent_id: u32, name: &UniStr) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + 2 + 2 * name.code_units.len());
    out.extend_from_slice(&record_type.to_be_bytes());
    out.extend_from_slice(&0u16.to_be_bytes()); // reserved
    out.extend_from_slice(&parent_id.to_be_bytes());
    encode_unistr(name, &mut out);
    out
}

/// HFSPlusForkData encoded directly into a fixed-size byte array.
fn fork_to_array(fork: &ForkData) -> [u8; FORK_DATA_SIZE] {
    let mut out = [0u8; FORK_DATA_SIZE];
    out[0..8].copy_from_slice(&fork.logical_size.to_be_bytes());
    out[8..12].copy_from_slice(&fork.clump_size.to_be_bytes());
    out[12..16].copy_from_slice(&fork.total_blocks.to_be_bytes());
    for (i, ext) in fork.extents.iter().enumerate() {
        let off = 16 + i * 8;
        out[off..off + 4].copy_from_slice(&ext.start_block.to_be_bytes());
        out[off + 4..off + 8].copy_from_slice(&ext.block_count.to_be_bytes());
    }
    out
}

/// Encode a 512-byte HFSPlusVolumeHeader from a [`VolumeHeader`] plus
/// the extra metadata fields not retained in the read-side struct.
pub(crate) fn encode_volume_header(
    vh: &VolumeHeader,
    next_allocation: u32,
    file_count: u32,
    folder_count: u32,
    create_date: u32,
    journal_info_block: u32,
) -> [u8; VolumeHeader::ENCODED_SIZE] {
    let mut b = [0u8; VolumeHeader::ENCODED_SIZE];
    b[0..2].copy_from_slice(&vh.signature);
    b[2..4].copy_from_slice(&vh.version.to_be_bytes());
    b[4..8].copy_from_slice(&vh.attributes.to_be_bytes());
    b[8..12].copy_from_slice(b"10.0"); // lastMountedVersion (cosmetic)
    b[12..16].copy_from_slice(&journal_info_block.to_be_bytes());
    b[16..20].copy_from_slice(&create_date.to_be_bytes());
    b[20..24].copy_from_slice(&create_date.to_be_bytes()); // modifyDate
    b[24..28].copy_from_slice(&0u32.to_be_bytes()); // backupDate
    b[28..32].copy_from_slice(&0u32.to_be_bytes()); // checkedDate
    b[32..36].copy_from_slice(&file_count.to_be_bytes());
    b[36..40].copy_from_slice(&folder_count.to_be_bytes());
    b[40..44].copy_from_slice(&vh.block_size.to_be_bytes());
    b[44..48].copy_from_slice(&vh.total_blocks.to_be_bytes());
    b[48..52].copy_from_slice(&vh.free_blocks.to_be_bytes());
    b[52..56].copy_from_slice(&next_allocation.to_be_bytes());
    // rsrcClumpSize / dataClumpSize: per-volume defaults inherited by
    // file forks that don't carry a custom clump size. fsck rejects
    // a zero value here as "Volume header needs minor repair" (its
    // own default is 16 KiB but it will silently use 64 KiB if we
    // pre-populate). Use 64 KiB — matches `newfs_hfs` for the same
    // block-size range and keeps clump arithmetic on power-of-two.
    b[56..60].copy_from_slice(&65536u32.to_be_bytes()); // rsrcClumpSize
    b[60..64].copy_from_slice(&65536u32.to_be_bytes()); // dataClumpSize
    b[64..68].copy_from_slice(&vh.next_catalog_id.to_be_bytes());
    b[68..72].copy_from_slice(&0u32.to_be_bytes()); // writeCount
    b[72..80].copy_from_slice(&1u64.to_be_bytes()); // encodingsBitmap (MacRoman bit)
    // finderInfo[8]: zero.

    let forks: [(usize, &ForkData); 5] = [
        (0x070, &vh.allocation_file),
        (0x0C0, &vh.extents_file),
        (0x110, &vh.catalog_file),
        (0x160, &vh.attributes_file),
        (0x1B0, &vh.startup_file),
    ];
    for (off, fork) in forks {
        let enc = fork_to_array(fork);
        b[off..off + FORK_DATA_SIZE].copy_from_slice(&enc);
    }
    b
}

// ----------------------------------------------------------------------
// B-tree serialisation
// ----------------------------------------------------------------------

/// One record's worth of bytes (key + body) waiting to be packed into a
/// node. We pack records into leaves first-fit-by-order, then build the
/// matching index nodes.
struct PackedRecord {
    key: Vec<u8>,
    body: Vec<u8>,
}

impl PackedRecord {
    fn encoded_len(&self) -> usize {
        // Records in a node consume `key_padded + body`. The 2-byte
        // offset table entry is accounted for separately.
        self.key.len() + self.body.len()
    }
}

/// Per-node packing result.
struct PackedNode {
    /// First record's key — used as the parent index entry.
    first_key: Vec<u8>,
    /// Encoded bytes of the node, length == `node_size`.
    bytes: Vec<u8>,
}

/// Pack a sequence of records (already in key-order) into a chain of
/// leaf nodes of size `node_size`. Returns the leaf bytes and the
/// per-leaf "first key" used to build the index above.
fn pack_leaves(records: &[PackedRecord], node_size: u32) -> Result<Vec<PackedNode>> {
    let ns = node_size as usize;
    let mut leaves: Vec<PackedNode> = Vec::new();
    let mut cur: Vec<&PackedRecord> = Vec::new();
    let mut cur_bytes: usize = NODE_DESCRIPTOR_SIZE;
    let mut cur_offsets: usize = 2; // trailing offset to free space

    for rec in records {
        let rec_size = rec.encoded_len();
        let with_rec = cur_bytes + rec_size + cur_offsets + 2;
        if with_rec > ns && !cur.is_empty() {
            leaves.push(write_node(KIND_LEAF, 1, &cur, ns)?);
            cur.clear();
            cur_bytes = NODE_DESCRIPTOR_SIZE;
            cur_offsets = 2;
        }
        // Even after flushing, a single record might be too large.
        if NODE_DESCRIPTOR_SIZE + rec_size + 2 + 2 > ns {
            return Err(crate::Error::Unsupported(format!(
                "hfs+ writer: record too large for node_size {ns} \
                 (record {} bytes)",
                rec_size
            )));
        }
        cur.push(rec);
        cur_bytes += rec_size;
        cur_offsets += 2;
    }
    if !cur.is_empty() {
        leaves.push(write_node(KIND_LEAF, 1, &cur, ns)?);
    }
    Ok(leaves)
}

/// Encode a node containing `records`. The node descriptor's `fLink`
/// and `bLink` fields are left zero — the caller patches them up.
fn write_node(
    kind: i8,
    height: u8,
    records: &[&PackedRecord],
    node_size: usize,
) -> Result<PackedNode> {
    let mut node = vec![0u8; node_size];
    // Descriptor
    node[8] = kind as u8;
    node[9] = height;
    node[10..12].copy_from_slice(&(records.len() as u16).to_be_bytes());

    // Records
    let mut cursor = NODE_DESCRIPTOR_SIZE;
    let mut offsets: Vec<u16> = Vec::with_capacity(records.len() + 1);
    for rec in records {
        offsets.push(cursor as u16);
        let end = cursor + rec.key.len();
        if end > node_size {
            return Err(crate::Error::Unsupported(
                "hfs+ writer: record overflowed node during packing".into(),
            ));
        }
        node[cursor..end].copy_from_slice(&rec.key);
        cursor = end;
        let end2 = cursor + rec.body.len();
        node[cursor..end2].copy_from_slice(&rec.body);
        cursor = end2;
    }
    offsets.push(cursor as u16);
    // Write offset table at the END of the node (growing backward).
    for (i, &o) in offsets.iter().enumerate() {
        let pos = node_size - 2 * (i + 1);
        node[pos..pos + 2].copy_from_slice(&o.to_be_bytes());
    }
    let first_key = if records.is_empty() {
        Vec::new()
    } else {
        records[0].key.clone()
    };
    Ok(PackedNode {
        first_key,
        bytes: node,
    })
}

/// Serialise a B-tree into nodes. Returns `(nodes, root_index,
/// first_leaf_index, last_leaf_index, tree_depth, leaf_record_count)`.
/// Node 0 is the header node; nodes 1.. are leaves followed by index
/// nodes. The caller is responsible for ensuring `nodes_capacity`
/// holds at least the number of nodes we emit.
fn build_btree(
    records: Vec<PackedRecord>,
    node_size: u32,
    nodes_capacity: u32,
) -> Result<BuiltTree> {
    let ns = node_size as usize;
    let leaf_count_initial = records.len();

    if records.is_empty() {
        // mkfs.hfsplus writes an empty B-tree as a single header node
        // with rootNode = 0 / treeDepth = 0 / firstLeafNode = 0. The
        // structural fix that satisfies fsck is the map-record size
        // (see `header_node`), not synthesising an empty leaf.
        return Ok(BuiltTree {
            nodes: vec![header_node(
                node_size,
                0,
                0,
                0,
                0,
                0,
                nodes_capacity,
                nodes_capacity.saturating_sub(1),
            )?],
            tree_depth: 0,
            root_node: 0,
            first_leaf: 0,
            last_leaf: 0,
            leaf_records: 0,
        });
    }

    // Pack leaves.
    let leaves_raw = pack_leaves(&records, node_size)?;
    let leaf_first_node = 1u32;
    let mut nodes: Vec<Vec<u8>> = Vec::with_capacity(nodes_capacity as usize);
    nodes.push(Vec::new()); // placeholder for header node at index 0

    // Emit leaves with proper bLink/fLink.
    let leaf_count = leaves_raw.len() as u32;
    for (i, pn) in leaves_raw.iter().enumerate() {
        let mut node = pn.bytes.clone();
        let f_link = if i + 1 < leaves_raw.len() {
            leaf_first_node + i as u32 + 1
        } else {
            0
        };
        let b_link = if i == 0 {
            0
        } else {
            leaf_first_node + i as u32 - 1
        };
        node[0..4].copy_from_slice(&f_link.to_be_bytes());
        node[4..8].copy_from_slice(&b_link.to_be_bytes());
        nodes.push(node);
    }

    // Build index levels.
    let mut tree_depth: u16 = 1; // leaves count as depth 1
    let mut level_first_keys: Vec<Vec<u8>> =
        leaves_raw.iter().map(|p| p.first_key.clone()).collect();
    let mut level_first_node = leaf_first_node;

    while level_first_keys.len() > 1 {
        // Build records for one index level above. Each record has the
        // child node's first key followed by a 4-byte child pointer.
        let mut idx_records: Vec<PackedRecord> = Vec::with_capacity(level_first_keys.len());
        for (i, key) in level_first_keys.iter().enumerate() {
            let child_idx = level_first_node + i as u32;
            let mut body = Vec::with_capacity(4);
            body.extend_from_slice(&child_idx.to_be_bytes());
            idx_records.push(PackedRecord {
                key: key.clone(),
                body,
            });
        }

        // Pack into nodes of `node_size`.
        let packed = pack_leaves(&idx_records, node_size)?;
        if packed.is_empty() {
            break;
        }
        let next_first_node = nodes.len() as u32;
        let next_count = packed.len() as u32;
        for (i, pn) in packed.iter().enumerate() {
            // Index nodes' fLink/bLink chain siblings at the same level.
            let mut node = pn.bytes.clone();
            // Set node kind to INDEX and height = tree_depth + 1.
            node[8] = KIND_INDEX as u8;
            node[9] = (tree_depth + 1) as u8;
            let f_link = if i + 1 < packed.len() {
                next_first_node + i as u32 + 1
            } else {
                0
            };
            let b_link = if i == 0 {
                0
            } else {
                next_first_node + i as u32 - 1
            };
            node[0..4].copy_from_slice(&f_link.to_be_bytes());
            node[4..8].copy_from_slice(&b_link.to_be_bytes());
            nodes.push(node);
        }
        tree_depth += 1;
        // The next level above this one uses the first key of every
        // node we just produced.
        level_first_keys = packed.into_iter().map(|p| p.first_key).collect();
        level_first_node = next_first_node;
        // Sanity: pack should reduce key count strictly when len>1.
        if level_first_keys.len() == next_count as usize && next_count > 1 {
            // Loop continues with smaller count next iteration.
        }
        if next_count == 1 {
            break;
        }
        let _ = ns; // silence unused if no further use
    }

    let root_node = if leaf_count == 1 {
        leaf_first_node
    } else {
        // The root is the single index node at the top — that's the
        // last node we pushed in the loop above.
        (nodes.len() as u32).saturating_sub(1)
    };
    let first_leaf = leaf_first_node;
    let last_leaf = leaf_first_node + leaf_count - 1;

    // Fill node 0 (header) with the right values.
    nodes[0] = header_node(
        node_size,
        tree_depth,
        root_node,
        leaf_count_initial as u32,
        first_leaf,
        last_leaf,
        nodes_capacity,
        nodes_capacity.saturating_sub(nodes.len() as u32),
    )?;

    if nodes.len() as u32 > nodes_capacity {
        return Err(crate::Error::Unsupported(format!(
            "hfs+ writer: B-tree needs {} nodes but only {nodes_capacity} \
             were reserved (raise FormatOpts::catalog_nodes)",
            nodes.len()
        )));
    }

    Ok(BuiltTree {
        nodes,
        tree_depth,
        root_node,
        first_leaf,
        last_leaf,
        leaf_records: leaf_count_initial as u32,
    })
}

#[allow(dead_code)]
struct BuiltTree {
    nodes: Vec<Vec<u8>>,
    tree_depth: u16,
    root_node: u32,
    first_leaf: u32,
    last_leaf: u32,
    leaf_records: u32,
}

#[allow(clippy::too_many_arguments)]
fn header_node(
    node_size: u32,
    tree_depth: u16,
    root_node: u32,
    leaf_records: u32,
    first_leaf: u32,
    last_leaf: u32,
    total_nodes: u32,
    free_nodes: u32,
) -> Result<Vec<u8>> {
    let ns = node_size as usize;
    let mut node = vec![0u8; ns];
    // Descriptor: kind = header, height = 0, numRecords = 3
    // (header rec, user rec, map rec).
    node[8] = KIND_HEADER as u8;
    node[9] = 0;
    node[10..12].copy_from_slice(&3u16.to_be_bytes());

    // BTHeaderRec at offset 14.
    let h = NODE_DESCRIPTOR_SIZE;
    node[h..h + 2].copy_from_slice(&tree_depth.to_be_bytes());
    node[h + 2..h + 6].copy_from_slice(&root_node.to_be_bytes());
    node[h + 6..h + 10].copy_from_slice(&leaf_records.to_be_bytes());
    node[h + 10..h + 14].copy_from_slice(&first_leaf.to_be_bytes());
    node[h + 14..h + 18].copy_from_slice(&last_leaf.to_be_bytes());
    node[h + 18..h + 20].copy_from_slice(&(node_size as u16).to_be_bytes());
    // maxKeyLength: 516 for catalog (2 + 4 + 2 + 510 incl. 255 UTF-16 chars),
    // 10 for extents-overflow. Picking 516 is conservative and harmless;
    // mkfs.hfsplus uses 516 for catalogs and 10 for the others.
    node[h + 20..h + 22].copy_from_slice(&516u16.to_be_bytes());
    node[h + 22..h + 26].copy_from_slice(&total_nodes.to_be_bytes());
    node[h + 26..h + 30].copy_from_slice(&free_nodes.to_be_bytes());
    // reserved1 at +30 (2 bytes), zero.
    // clumpSize at +32 (4 bytes) — match Apple's hint of node_size * 8.
    node[h + 32..h + 36].copy_from_slice(&(node_size).to_be_bytes());
    // btreeType (+36): 0 = HFS+, keyCompareType (+37): 0xCF (case-fold) for plain HFS+.
    node[h + 36] = 0;
    node[h + 37] = 0xCF;
    // attributes (+38): kBTBigKeysMask (2) | kBTVariableIndexKeysMask (4) = 6
    node[h + 38..h + 42].copy_from_slice(&6u32.to_be_bytes());
    // reserved3 (16 u32 words) -- zero.

    // Record offsets table: 4 entries at the tail (8 bytes total).
    // mkfs.hfsplus stretches the map record across ALL remaining space
    // — TN1150 §"B-Tree Header Record" says "The size of the first map
    // record is the size of the rest of the header node" — so the map
    // ends at `node_size - 8` (where the offsets table starts). fsck
    // rejects a header whose map record is short of that boundary.
    let used_blocks = total_nodes - free_nodes;
    let user_off = NODE_DESCRIPTOR_SIZE + HEADER_REC_SIZE; // 120
    let map_off = user_off + 128; // 248
    let offsets_table = 2 * 4; // 4 entries × u16
    let free_off = ns - offsets_table;
    if free_off <= map_off {
        return Err(crate::Error::Unsupported(format!(
            "hfs+ writer: node_size {ns} too small for header layout"
        )));
    }
    let map_rec_size = free_off - map_off;
    // The map needs at least one byte to hold the header's own bit.
    if (total_nodes as usize).div_ceil(8) > map_rec_size {
        return Err(crate::Error::Unsupported(format!(
            "hfs+ writer: {total_nodes}-node tree exceeds map record capacity ({map_rec_size} bytes)"
        )));
    }
    let offs = [
        NODE_DESCRIPTOR_SIZE as u16,
        user_off as u16,
        map_off as u16,
        free_off as u16,
    ];
    for (i, &o) in offs.iter().enumerate() {
        let pos = ns - 2 * (i + 1);
        node[pos..pos + 2].copy_from_slice(&o.to_be_bytes());
    }
    // Populate the map bits for nodes 0..used_blocks (these nodes are
    // in use). MSB-first within each byte, matching f2fs_test_bit and
    // every other HFS+ bitmap convention.
    for b in 0..used_blocks {
        let bi = b as usize;
        node[map_off + bi / 8] |= 1u8 << (7 - (bi & 7));
    }
    Ok(node)
}

// ----------------------------------------------------------------------
// Format
// ----------------------------------------------------------------------

/// Build a fresh HFS+ volume on `dev` and return a `Writer` ready to
/// receive `create_*` calls.
pub fn format(dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<(VolumeHeader, Writer)> {
    if !(opts.block_size.is_power_of_two() && opts.block_size >= 512) {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ format: block_size {} is not a power of two ≥ 512",
            opts.block_size
        )));
    }
    if !(opts.node_size.is_power_of_two() && opts.node_size >= 512) {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ format: node_size {} is not a power of two ≥ 512",
            opts.node_size
        )));
    }
    let bs = opts.block_size;
    let total_size = dev.total_size();
    if total_size < u64::from(bs) * 8 {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ format: device {total_size} bytes too small for block_size {bs}"
        )));
    }
    let total_blocks_u64 = total_size / u64::from(bs);
    if total_blocks_u64 > u64::from(u32::MAX) {
        return Err(crate::Error::InvalidArgument(
            "hfs+ format: more than 2^32 - 1 allocation blocks".into(),
        ));
    }
    let total_blocks = total_blocks_u64 as u32;

    // Zero the entire formatted region so unused bytes read as zero.
    dev.zero_range(0, u64::from(total_blocks) * u64::from(bs))?;

    // ---- layout: place special files starting at block 1.
    let mut cursor: u32 = 1;

    // Allocation bitmap: one bit per allocation block, rounded up to a
    // whole number of blocks.
    let bitmap_bytes = (total_blocks as u64).div_ceil(8);
    let alloc_blocks_needed = bitmap_bytes.div_ceil(u64::from(bs));
    let alloc_blocks_needed = u32::try_from(alloc_blocks_needed).map_err(|_| {
        crate::Error::InvalidArgument("hfs+ format: bitmap too large for u32".into())
    })?;
    // Allocation bitmap: clumpSize matches the allocation unit (block).
    let allocation_file = layout_special(&mut cursor, alloc_blocks_needed, bs, bs)?;

    // Extents-overflow and catalog B-trees: per TN1150 + mkfs.hfsplus
    // convention, `clumpSize` on a B-tree fork equals the B-tree's
    // node size (the BTHeaderRec.clumpSize and ForkData.clumpSize agree).
    // fsck.hfsplus rejects an extents-overflow fork whose clumpSize is
    // smaller than its declared nodeSize ("Invalid B-tree node size").
    let ext_blocks = blocks_for_nodes(opts.extents_nodes, opts.node_size, bs)?;
    let extents_file = layout_special(&mut cursor, ext_blocks, bs, opts.node_size)?;

    let cat_blocks = blocks_for_nodes(opts.catalog_nodes, opts.node_size, bs)?;
    let catalog_file = layout_special(&mut cursor, cat_blocks, bs, opts.node_size)?;

    // Attributes B-tree and startup file: empty (zero fork data).
    let attributes_file = ForkData::default();
    let startup_file = ForkData::default();

    // Journal stub: one block for the JournalInfoBlock, plus N blocks
    // of journal buffer. Reserved up-front so the bitmap reflects the
    // usage from format time onward.
    let (journal_info_block, journal_buffer_start, journal_buffer_blocks) = if opts.journaled {
        let info_start = cursor;
        cursor = cursor.checked_add(1).ok_or_else(|| {
            crate::Error::InvalidArgument("hfs+ format: journal layout overflow".into())
        })?;
        let buf_start = cursor;
        cursor = cursor
            .checked_add(DEFAULT_JOURNAL_BUFFER_BLOCKS)
            .ok_or_else(|| {
                crate::Error::InvalidArgument("hfs+ format: journal buffer overflow".into())
            })?;
        (info_start, buf_start, DEFAULT_JOURNAL_BUFFER_BLOCKS)
    } else {
        (0, 0, 0)
    };

    // Sanity: don't run past end of device.
    if cursor > total_blocks {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ format: special files need {cursor} blocks but volume \
             only has {total_blocks}"
        )));
    }

    // Initialise the in-memory bitmap with all-used for blocks 0..cursor,
    // plus tail bits beyond total_blocks set so the allocator never
    // picks them. Block 0 is the volume-header pre-area.
    let mut bitmap = vec![0u8; bitmap_bytes as usize];
    for b in 0..cursor {
        let by = (b / 8) as usize;
        bitmap[by] |= 1u8 << (7 - (b & 7));
    }
    // Pad bits beyond total_blocks within the last byte.
    let tail_bits = (bitmap_bytes * 8) as u32;
    for b in total_blocks..tail_bits {
        let by = (b / 8) as usize;
        bitmap[by] |= 1u8 << (7 - (b & 7));
    }
    // Mark the very last block as used too: we reserve it for the
    // alternate volume header.
    let last_block = total_blocks - 1;
    let by = (last_block / 8) as usize;
    let mask = 1u8 << (7 - (last_block & 7));
    if bitmap[by] & mask == 0 {
        bitmap[by] |= mask;
    }

    let mut free_blocks = total_blocks - cursor;
    if total_blocks > 0 {
        // Account for alternate volume header (block last_block).
        if last_block >= cursor {
            free_blocks = free_blocks.saturating_sub(1);
        }
    }

    let volume_name_unistr = UniStr::from_str_lossy(&opts.volume_name);

    let mut writer = Writer {
        block_size: bs,
        node_size: opts.node_size,
        total_blocks,
        volume_name: opts.volume_name.clone(),
        create_date: opts.create_date,
        next_cnid: 16, // CNIDs 0..15 are reserved per TN1150
        bitmap,
        next_alloc: cursor,
        free_blocks,
        catalog: BTreeMap::new(),
        overflow_extents: BTreeMap::new(),
        allocation_file,
        extents_file,
        catalog_file,
        attributes_file,
        startup_file,
        private_dir_cnid: None,
        journal_info_block,
        journal_buffer_start,
        journal_buffer_blocks,
        flushed: false,
    };

    // Seed the root folder + thread.
    let root_thread_body =
        encode_thread_body(REC_FOLDER_THREAD, ROOT_PARENT_ID, &volume_name_unistr);
    writer
        .catalog
        .insert(OwnedKey::thread(ROOT_FOLDER_ID), root_thread_body);

    let root_folder_body = encode_folder_body(
        ROOT_FOLDER_ID,
        0,
        m::S_IFDIR | 0o755,
        0,
        0,
        opts.create_date,
    );
    writer.catalog.insert(
        OwnedKey {
            parent_id: ROOT_PARENT_ID,
            name: volume_name_unistr,
        },
        root_folder_body,
    );

    // Journaled volumes carry the journal-info block + journal buffer
    // as in-volume allocation blocks reserved at format time. fsck
    // walks the catalog to validate the bitmap and treats any
    // bitmap-allocated block not owned by a catalog file as an
    // orphan ("Volume bitmap needs minor repair for orphaned
    // blocks"). The standard fix is to create two hidden files under
    // the root — `.journal_info_block` and `.journal` — whose data
    // forks exactly cover those reserved blocks, giving fsck a
    // catalog owner for every bit we set.
    if opts.journaled {
        insert_journal_files(
            &mut writer,
            journal_info_block,
            journal_buffer_start,
            journal_buffer_blocks,
        )?;
    }

    // Build the in-memory VolumeHeader we'll keep alongside the writer.
    let vh = VolumeHeader {
        signature: SIG_HFS_PLUS,
        version: 4,
        attributes: VOL_ATTR_UNMOUNTED,
        block_size: bs,
        total_blocks,
        free_blocks: writer.free_blocks,
        next_catalog_id: writer.next_cnid,
        allocation_file: writer.allocation_file,
        extents_file: writer.extents_file,
        catalog_file: writer.catalog_file,
        attributes_file: writer.attributes_file,
        startup_file: writer.startup_file,
    };

    Ok((vh, writer))
}

fn blocks_for_nodes(nodes: u32, node_size: u32, block_size: u32) -> Result<u32> {
    if nodes == 0 {
        return Err(crate::Error::InvalidArgument(
            "hfs+ format: special-file node count must be > 0".into(),
        ));
    }
    let bytes = u64::from(nodes) * u64::from(node_size);
    let blocks = bytes.div_ceil(u64::from(block_size));
    u32::try_from(blocks).map_err(|_| {
        crate::Error::InvalidArgument("hfs+ format: special-file too large for u32".into())
    })
}

fn layout_special(
    cursor: &mut u32,
    blocks: u32,
    block_size: u32,
    clump_size: u32,
) -> Result<ForkData> {
    let start = *cursor;
    *cursor = cursor
        .checked_add(blocks)
        .ok_or_else(|| crate::Error::InvalidArgument("hfs+ format: layout overflow".into()))?;
    let mut extents = [ExtentDescriptor::default(); FORK_EXTENT_COUNT];
    extents[0] = ExtentDescriptor {
        start_block: start,
        block_count: blocks,
    };
    Ok(ForkData {
        logical_size: u64::from(blocks) * u64::from(block_size),
        clump_size,
        total_blocks: blocks,
        extents,
    })
}

// ----------------------------------------------------------------------
// File creation: streaming bytes into bump-allocated blocks
// ----------------------------------------------------------------------

/// Stream `len` bytes from `src` into newly-allocated allocation blocks
/// on `dev`, returning the resulting `ForkData` for `file_id` to embed
/// in a catalog file record. Uses a 64 KiB scratch buffer; never loads
/// the file in memory.
///
/// If the file fragments past the eight inline extents that fit in
/// `HFSPlusForkData`, the writer queues the spill into
/// `writer.overflow_extents` keyed by `(FORK_DATA, file_id, start_fork_block)`.
/// `flush()` later turns the queued entries into extents-overflow
/// B-tree records.
pub(crate) fn stream_data_to_blocks<R: Read>(
    writer: &mut Writer,
    dev: &mut dyn BlockDevice,
    src: &mut R,
    len: u64,
    file_id: u32,
) -> Result<ForkData> {
    if len == 0 {
        return Ok(ForkData {
            logical_size: 0,
            clump_size: writer.block_size,
            total_blocks: 0,
            extents: [ExtentDescriptor::default(); FORK_EXTENT_COUNT],
        });
    }
    let bs = u64::from(writer.block_size);
    let total_blocks_u64 = len.div_ceil(bs);
    let total_blocks = u32::try_from(total_blocks_u64).map_err(|_| {
        crate::Error::InvalidArgument("hfs+ writer: file size overflows u32 blocks".into())
    })?;

    // Collect the run list as we allocate, so we can record any spill
    // beyond the 8 inline extents in the extents-overflow tree later.
    let mut runs: Vec<ExtentDescriptor> = Vec::new();
    let mut remaining = total_blocks;
    while remaining > 0 {
        let run = writer.allocate_largest_run(remaining)?;
        remaining -= run.block_count;
        runs.push(run);
    }

    // Now stream the source bytes into the runs in order.
    let mut buf = vec![0u8; 64 * 1024];
    let mut written: u64 = 0;
    for run in &runs {
        let run_bytes = u64::from(run.block_count) * bs;
        let mut run_off = u64::from(run.start_block) * bs;
        let mut run_remaining = run_bytes;
        while run_remaining > 0 && written < len {
            let want = ((len - written).min(run_remaining)).min(buf.len() as u64) as usize;
            let mut filled = 0;
            while filled < want {
                let n = src.read(&mut buf[filled..want]).map_err(crate::Error::Io)?;
                if n == 0 {
                    return Err(crate::Error::InvalidArgument(format!(
                        "hfs+ writer: source ended early at {} of {len} bytes",
                        written + filled as u64
                    )));
                }
                filled += n;
            }
            dev.write_at(run_off, &buf[..filled])?;
            run_off += filled as u64;
            run_remaining -= filled as u64;
            written += filled as u64;
        }
    }
    // Zero-pad the slack space in the last allocated run (the device
    // was zeroed at format time, but a block freed by `remove` and
    // re-handed-out via first-fit may carry stale bytes).
    if let Some(last) = runs.last() {
        let last_bytes = u64::from(last.block_count) * bs;
        let pre_last: u64 = runs
            .iter()
            .take(runs.len() - 1)
            .map(|e| u64::from(e.block_count) * bs)
            .sum();
        let used_in_last = len - pre_last;
        if used_in_last < last_bytes {
            let zero = vec![0u8; (last_bytes - used_in_last) as usize];
            let zero_off = u64::from(last.start_block) * bs + used_in_last;
            dev.write_at(zero_off, &zero)?;
        }
    }

    // Pack the run list back into a ForkData (first 8) + extents-overflow
    // records (remainder, eight per record).
    let mut extents = [ExtentDescriptor::default(); FORK_EXTENT_COUNT];
    let inline_count = runs.len().min(FORK_EXTENT_COUNT);
    for (slot, ext) in extents.iter_mut().zip(runs.iter().take(inline_count)) {
        *slot = *ext;
    }

    if runs.len() > FORK_EXTENT_COUNT {
        record_fork_overflow(
            writer,
            FORK_DATA,
            file_id,
            &extents,
            &runs[FORK_EXTENT_COUNT..],
        );
    }

    Ok(ForkData {
        logical_size: len,
        clump_size: writer.block_size,
        total_blocks,
        extents,
    })
}

/// Register the run list past the eight inline extents into the writer's
/// pending extents-overflow records. `inline_extents` is the 8-slot
/// ForkData array, used to compute the fork-block where the overflow
/// records start. `overflow_runs` are the 9th onward extents in fork order.
fn record_fork_overflow(
    writer: &mut Writer,
    fork_type: u8,
    file_id: u32,
    inline_extents: &[ExtentDescriptor; FORK_EXTENT_COUNT],
    overflow_runs: &[ExtentDescriptor],
) {
    // Fork-block index at which the first overflow record begins is the
    // total block count of the inline extents.
    let mut start_block: u32 = inline_extents
        .iter()
        .map(|e| e.block_count)
        .fold(0u32, |a, b| a.saturating_add(b));

    for chunk in overflow_runs.chunks(FORK_EXTENT_COUNT) {
        let mut group = [ExtentDescriptor::default(); FORK_EXTENT_COUNT];
        for (slot, ext) in group.iter_mut().zip(chunk.iter()) {
            *slot = *ext;
        }
        writer
            .overflow_extents
            .insert((fork_type, file_id, start_block), group);
        for ext in chunk {
            start_block = start_block.saturating_add(ext.block_count);
        }
    }
}

/// Stream a slice's bytes into newly-allocated blocks. Used for symlink
/// targets which are tiny strings, so we don't bother with the chunked
/// read loop above.
pub(crate) fn write_inline_data(
    writer: &mut Writer,
    dev: &mut dyn BlockDevice,
    bytes: &[u8],
) -> Result<ForkData> {
    if bytes.is_empty() {
        return Ok(ForkData::default());
    }
    let bs = writer.block_size;
    let total_blocks = u32::try_from((bytes.len() as u64).div_ceil(u64::from(bs)))
        .map_err(|_| crate::Error::InvalidArgument("hfs+ writer: data too large".into()))?;
    let start = writer.allocate(total_blocks)?;
    let device_off = u64::from(start) * u64::from(bs);
    dev.write_at(device_off, bytes)?;
    let mut extents = [ExtentDescriptor::default(); FORK_EXTENT_COUNT];
    extents[0] = ExtentDescriptor {
        start_block: start,
        block_count: total_blocks,
    };
    Ok(ForkData {
        logical_size: bytes.len() as u64,
        clump_size: bs,
        total_blocks,
        extents,
    })
}

// ----------------------------------------------------------------------
// Catalog mutation helpers
// ----------------------------------------------------------------------

/// Insert a new directory child with the given encoded folder body.
/// The caller must have allocated `folder_id` from `writer.next_cnid`.
pub(crate) fn insert_folder(
    writer: &mut Writer,
    parent_id: u32,
    name: &UniStr,
    folder_id: u32,
    mode: u16,
    uid: u32,
    gid: u32,
) -> Result<()> {
    if name.code_units.is_empty() {
        return Err(crate::Error::InvalidArgument(
            "hfs+ writer: directory name must not be empty".into(),
        ));
    }
    if !writer.is_dir(parent_id) {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ writer: parent CNID {parent_id} is not a directory"
        )));
    }
    let folder_key = OwnedKey {
        parent_id,
        name: name.clone(),
    };
    if writer.catalog.contains_key(&folder_key) {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ writer: entry {:?} already exists under CNID {parent_id}",
            name.to_string_lossy()
        )));
    }
    let body = encode_folder_body(
        folder_id,
        0,
        mode | m::S_IFDIR,
        uid,
        gid,
        writer.create_date,
    );
    writer.catalog.insert(folder_key, body);

    let thread = encode_thread_body(REC_FOLDER_THREAD, parent_id, name);
    writer.catalog.insert(OwnedKey::thread(folder_id), thread);

    writer.bump_valence(parent_id, 1)?;
    Ok(())
}

/// Insert a file record with the given encoded body and a thread record.
#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_file(
    writer: &mut Writer,
    parent_id: u32,
    name: &UniStr,
    file_id: u32,
    mode: u16,
    uid: u32,
    gid: u32,
    file_type: [u8; 4],
    creator: [u8; 4],
    data_fork: &ForkData,
    is_symlink: bool,
) -> Result<()> {
    if name.code_units.is_empty() {
        return Err(crate::Error::InvalidArgument(
            "hfs+ writer: file name must not be empty".into(),
        ));
    }
    if !writer.is_dir(parent_id) {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ writer: parent CNID {parent_id} is not a directory"
        )));
    }
    let key = OwnedKey {
        parent_id,
        name: name.clone(),
    };
    if writer.catalog.contains_key(&key) {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ writer: entry {:?} already exists under CNID {parent_id}",
            name.to_string_lossy()
        )));
    }
    let mode_full = mode | if is_symlink { m::S_IFLNK } else { m::S_IFREG };
    let body = encode_file_body(
        file_id,
        mode_full,
        uid,
        gid,
        writer.create_date,
        file_type,
        creator,
        data_fork,
    );
    writer.catalog.insert(key, body);

    let thread = encode_thread_body(REC_FILE_THREAD, parent_id, name);
    writer.catalog.insert(OwnedKey::thread(file_id), thread);

    writer.bump_valence(parent_id, 1)?;
    Ok(())
}

/// Build the well-known HFS+ private-data directory name. The name
/// consists of four NUL code units followed by the literal text
/// `"HFS+ Private Data"`. Apple chose those leading NULs so the
/// directory sorts ahead of any user-supplied name in the root and
/// so it never collides with a real file name.
pub(crate) fn private_data_dir_name() -> UniStr {
    let mut code_units: Vec<u16> = vec![0, 0, 0, 0];
    code_units.extend("HFS+ Private Data".encode_utf16());
    UniStr { code_units }
}

/// Ensure the HFS+ private-data directory exists under the root and
/// return its CNID. Created lazily on the first hard-link insert.
/// Create the two hidden catalog files that a journaled HFS+ volume
/// uses to give the journal-info block + journal buffer a catalog
/// owner. Without them, fsck reports the journal blocks as orphans
/// (their bits are set in the bitmap but nothing in the catalog
/// claims them).
///
/// Naming + tagging follow what Apple writes on a journaled volume:
/// * `.journal` — covers the `journal_buffer_blocks` blocks starting
///   at `buffer_start`.
/// * `.journal_info_block` — covers the single block at `info_block`.
/// * Both have `fileType = "jrnl"`, `creator = "hfs+"`, and Finder
///   `finderFlags = 0x5000` (kIsInvisible + kNameLocked) so the
///   names never surface in user-facing listings.
/// * `fileMode = S_IFREG` only (mode 0 — system-only access).
/// * `BSDInfo.special = 1` (Apple convention; not the link-inode
///   field — these aren't hlnk records).
pub(crate) fn insert_journal_files(
    writer: &mut Writer,
    info_block: u32,
    buffer_start: u32,
    buffer_blocks: u32,
) -> Result<()> {
    if buffer_blocks == 0 {
        return Ok(());
    }
    let bs = writer.block_size;
    // .journal first — Apple lays it down at CNID 16 (first user CNID).
    let journal_cnid = writer.next_cnid;
    writer.next_cnid = writer
        .next_cnid
        .checked_add(1)
        .ok_or_else(|| crate::Error::Unsupported("hfs+: CNID space exhausted".into()))?;
    let journal_fork = ForkData {
        logical_size: u64::from(buffer_blocks) * u64::from(bs),
        clump_size: bs,
        total_blocks: buffer_blocks,
        extents: {
            let mut ex = [ExtentDescriptor::default(); FORK_EXTENT_COUNT];
            ex[0] = ExtentDescriptor {
                start_block: buffer_start,
                block_count: buffer_blocks,
            };
            ex
        },
    };
    insert_journal_entry(writer, ".journal", journal_cnid, &journal_fork)?;

    // .journal_info_block — 1 block.
    let info_cnid = writer.next_cnid;
    writer.next_cnid = writer
        .next_cnid
        .checked_add(1)
        .ok_or_else(|| crate::Error::Unsupported("hfs+: CNID space exhausted".into()))?;
    let info_fork = ForkData {
        logical_size: u64::from(bs),
        clump_size: bs,
        total_blocks: 1,
        extents: {
            let mut ex = [ExtentDescriptor::default(); FORK_EXTENT_COUNT];
            ex[0] = ExtentDescriptor {
                start_block: info_block,
                block_count: 1,
            };
            ex
        },
    };
    insert_journal_entry(writer, ".journal_info_block", info_cnid, &info_fork)?;
    Ok(())
}

fn insert_journal_entry(
    writer: &mut Writer,
    name: &str,
    cnid: u32,
    fork: &ForkData,
) -> Result<()> {
    let name_uni = UniStr::from_str_lossy(name);
    let key = OwnedKey {
        parent_id: ROOT_FOLDER_ID,
        name: name_uni.clone(),
    };
    let mut body = Vec::with_capacity(248);
    body.extend_from_slice(&REC_FILE.to_be_bytes());
    // flags = kHFSThreadExistsMask only.
    body.extend_from_slice(&0x0002u16.to_be_bytes());
    body.extend_from_slice(&0u32.to_be_bytes()); // reserved1
    body.extend_from_slice(&cnid.to_be_bytes());
    for _ in 0..5 {
        body.extend_from_slice(&writer.create_date.to_be_bytes());
    }
    // BSDInfo: owner=0, group=0, ownerFlags=0, fileMode = S_IFREG (no
    // permission bits — the journal files are system-private),
    // special=1 (Apple convention).
    body.extend_from_slice(&0u32.to_be_bytes()); // ownerID
    body.extend_from_slice(&0u32.to_be_bytes()); // groupID
    body.push(0); // adminFlags
    body.push(0); // ownerFlags
    body.extend_from_slice(&m::S_IFREG.to_be_bytes()); // fileMode
    body.extend_from_slice(&1u32.to_be_bytes()); // special
    // FileInfo: fileType = "jrnl", creator = "hfs+", finderFlags = 0x5000.
    body.extend_from_slice(b"jrnl");
    body.extend_from_slice(b"hfs+");
    body.extend_from_slice(&0x5000u16.to_be_bytes()); // finderFlags
    body.extend_from_slice(&[0u8; 6]); // location + reservedField
    body.extend_from_slice(&[0u8; 16]); // ExtendedFileInfo
    body.extend_from_slice(&0u32.to_be_bytes()); // textEncoding
    body.extend_from_slice(&0u32.to_be_bytes()); // reserved2
    encode_fork(fork, &mut body);
    encode_fork(&ForkData::default(), &mut body); // empty resource fork
    debug_assert_eq!(body.len(), 248);
    writer.catalog.insert(key, body);
    writer.catalog.insert(
        OwnedKey::thread(cnid),
        encode_thread_body(REC_FILE_THREAD, ROOT_FOLDER_ID, &name_uni),
    );
    writer.bump_valence(ROOT_FOLDER_ID, 1)?;
    Ok(())
}

pub(crate) fn ensure_private_dir(writer: &mut Writer) -> Result<u32> {
    if let Some(cnid) = writer.private_dir_cnid {
        return Ok(cnid);
    }
    let cnid = writer.next_cnid;
    writer.next_cnid = writer
        .next_cnid
        .checked_add(1)
        .ok_or_else(|| crate::Error::Unsupported("hfs+: CNID space exhausted".into()))?;
    let name = private_data_dir_name();
    // mode 0700 / uid root / gid root, S_IFDIR is or'ed in by insert_folder.
    insert_folder(writer, ROOT_FOLDER_ID, &name, cnid, 0o700, 0, 0)?;
    writer.private_dir_cnid = Some(cnid);
    // Apple marks the Private Data directory as invisible in Finder:
    // FolderInfo.frFlags |= kIsInvisible (0x4000). The FolderInfo block
    // sits at byte 48..64 of the 88-byte folder body; frFlags is the
    // u16 at byte 48 + 8 = 56.
    let folder_key = OwnedKey {
        parent_id: ROOT_FOLDER_ID,
        name,
    };
    if let Some(body) = writer.catalog.get_mut(&folder_key) {
        if body.len() >= 58 {
            let cur = u16::from_be_bytes([body[56], body[57]]);
            body[56..58].copy_from_slice(&(cur | 0x4000).to_be_bytes());
        }
    }
    Ok(cnid)
}

/// Decode the data fork out of a stored file catalog record body.
/// Used by `create_hardlink` to move an existing file's payload into
/// a fresh iNode entry.
pub(crate) fn extract_file_fork(body: &[u8]) -> Result<ForkData> {
    if body.len() < 88 + FORK_DATA_SIZE {
        return Err(crate::Error::InvalidImage(
            "hfs+ writer: short catalog file body".into(),
        ));
    }
    let mut buf = [0u8; FORK_DATA_SIZE];
    buf.copy_from_slice(&body[88..88 + FORK_DATA_SIZE]);
    Ok(ForkData::decode(&buf))
}

/// Read the BSD `(uid, gid, mode)` triple out of a file catalog body.
pub(crate) fn extract_file_perms(body: &[u8]) -> Result<(u32, u32, u16)> {
    if body.len() < 48 {
        return Err(crate::Error::InvalidImage(
            "hfs+ writer: short catalog file body".into(),
        ));
    }
    let uid = u32::from_be_bytes(body[32..36].try_into().unwrap());
    let gid = u32::from_be_bytes(body[36..40].try_into().unwrap());
    let mode = u16::from_be_bytes(body[42..44].try_into().unwrap());
    Ok((uid, gid, mode))
}

/// Insert a hard-link "indirect node" record. See [`encode_hardlink_body`]
/// for the on-disk shape and chain semantics. `prev_link_cnid` /
/// `next_link_cnid` are CNIDs of the surrounding hlnk records in the
/// chain; `0` marks the head/tail.
#[allow(clippy::too_many_arguments)]
pub(crate) fn insert_hardlink_entry(
    writer: &mut Writer,
    parent_id: u32,
    name: &UniStr,
    file_id: u32,
    inode_cnid: u32,
    prev_link_cnid: u32,
    next_link_cnid: u32,
) -> Result<()> {
    if name.code_units.is_empty() {
        return Err(crate::Error::InvalidArgument(
            "hfs+ writer: hard-link name must not be empty".into(),
        ));
    }
    if !writer.is_dir(parent_id) {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ writer: parent CNID {parent_id} is not a directory"
        )));
    }
    let key = OwnedKey {
        parent_id,
        name: name.clone(),
    };
    if writer.catalog.contains_key(&key) {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ writer: entry {:?} already exists under CNID {parent_id}",
            name.to_string_lossy()
        )));
    }
    let body = encode_hardlink_body(
        file_id,
        inode_cnid,
        prev_link_cnid,
        next_link_cnid,
        writer.create_date,
    );
    writer.catalog.insert(key, body);
    let thread = encode_thread_body(REC_FILE_THREAD, parent_id, name);
    writer.catalog.insert(OwnedKey::thread(file_id), thread);
    writer.bump_valence(parent_id, 1)?;
    Ok(())
}

/// Encode a hard-link "indirect node" catalog file body. Same 248-byte
/// shape as `encode_file_body`, with these hardlink-specific values:
/// * FileInfo tags: (`'hlnk'`, `'hfs+'`)
/// * `HFSPlusBSDInfo.special` = the iNode file's CNID
/// * `HFSPlusBSDInfo.ownerID` = previous link CNID (0 if first in chain)
/// * `HFSPlusBSDInfo.groupID` = next link CNID (0 if last in chain)
/// * `HFSPlusBSDInfo.ownerFlags` = `UF_IMMUTABLE` (`0x02`) — Apple
///   marks hlnks immutable since their bytes are owned by the iNode.
/// * `HFSPlusBSDInfo.fileMode` = `S_IFREG | 0o444` — the hlnk record
///   itself reports read-only; the real mode lives on the iNode.
/// * `FileInfo.finderFlags` = `kHasBeenInited` (`0x0100`)
///
/// fsck.hfsplus walks the chain through `ownerID` / `groupID`, not by
/// scanning siblings, so missing or zeroed chain pointers cause it
/// to under-count links and report "Incorrect number of file hard
/// links". The owner/group fields are not the file's real uid/gid —
/// userspace pulls those from the iNode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_hardlink_body(
    file_id: u32,
    inode_cnid: u32,
    prev_link_cnid: u32,
    next_link_cnid: u32,
    create_date: u32,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(248);
    out.extend_from_slice(&REC_FILE.to_be_bytes());
    // flags = kHFSHasLinkChainMask (0x0020) | kHFSThreadExistsMask
    // (0x0002). Apple sets both on every hlnk record — fsck uses
    // ThreadExists to skip the lookup when walking the chain.
    out.extend_from_slice(&0x0022u16.to_be_bytes()); // flags
    out.extend_from_slice(&0u32.to_be_bytes()); // reserved1
    out.extend_from_slice(&file_id.to_be_bytes());
    for _ in 0..5 {
        out.extend_from_slice(&create_date.to_be_bytes());
    }
    // BSDInfo: hijacked layout for hlnk records (see fn doc).
    out.extend_from_slice(&prev_link_cnid.to_be_bytes()); // ownerID
    out.extend_from_slice(&next_link_cnid.to_be_bytes()); // groupID
    out.push(0); // adminFlags
    out.push(0x02); // ownerFlags = UF_IMMUTABLE
    out.extend_from_slice(&(m::S_IFREG | 0o444u16).to_be_bytes()); // fileMode
    out.extend_from_slice(&inode_cnid.to_be_bytes()); // special = iNode CNID
    // FileInfo: fileType "hlnk", creator "hfs+", finderFlags = kHasBeenInited.
    out.extend_from_slice(b"hlnk");
    out.extend_from_slice(b"hfs+");
    out.extend_from_slice(&0x0100u16.to_be_bytes()); // finderFlags
    out.extend_from_slice(&[0u8; 6]); // location(4) + reservedField(2)
    // ExtendedFileInfo
    out.extend_from_slice(&[0u8; 16]);
    out.extend_from_slice(&0u32.to_be_bytes()); // textEncoding
    out.extend_from_slice(&0u32.to_be_bytes()); // reserved2
    encode_fork(&ForkData::default(), &mut out);
    encode_fork(&ForkData::default(), &mut out);
    debug_assert_eq!(out.len(), 248);
    out
}

/// Promote an existing file to a hard-link pair.
///
/// On entry: `(src_parent, src_name)` names a regular file (not a
/// symlink, not already an hlnk) in the catalog, and `(dst_parent,
/// dst_name)` is an unused name in an existing directory.
///
/// On success:
/// * a fresh `iNode<N>` file lives in the HFS+ private-data directory
///   carrying the source's data fork (no bytes are moved on disk —
///   the catalog is the only thing that changes);
/// * the source entry is overwritten with an `hlnk`/`hfs+` indirect
///   record pointing at `N`;
/// * the destination entry is created as a second `hlnk`/`hfs+`
///   indirect record also pointing at `N`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn promote_to_hardlink(
    writer: &mut Writer,
    src_parent: u32,
    src_name: &UniStr,
    dst_parent: u32,
    dst_name: &UniStr,
) -> Result<u32> {
    // Forbid self-link: a hard link to itself would corrupt the catalog.
    if src_parent == dst_parent
        && compare_unistr(src_name, dst_name, false) == std::cmp::Ordering::Equal
    {
        return Err(crate::Error::InvalidArgument(
            "hfs+ writer: source and destination hard-link paths are the same".into(),
        ));
    }
    if !writer.is_dir(dst_parent) {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ writer: destination parent CNID {dst_parent} is not a directory"
        )));
    }
    let dst_key = OwnedKey {
        parent_id: dst_parent,
        name: dst_name.clone(),
    };
    if writer.catalog.contains_key(&dst_key) {
        return Err(crate::Error::InvalidArgument(format!(
            "hfs+ writer: destination {:?} already exists",
            dst_name.to_string_lossy()
        )));
    }
    let src_key = OwnedKey {
        parent_id: src_parent,
        name: src_name.clone(),
    };
    let src_body = writer.catalog.get(&src_key).ok_or_else(|| {
        crate::Error::InvalidArgument(format!(
            "hfs+ writer: source {:?} not found under CNID {src_parent}",
            src_name.to_string_lossy()
        ))
    })?;
    if src_body.len() < 248 || i16::from_be_bytes([src_body[0], src_body[1]]) != REC_FILE {
        return Err(crate::Error::InvalidArgument(
            "hfs+ writer: hard-link source must be an existing regular file".into(),
        ));
    }
    // Reject symlinks and already-hardlink sources.
    let mut file_type = [0u8; 4];
    file_type.copy_from_slice(&src_body[48..52]);
    let mut creator = [0u8; 4];
    creator.copy_from_slice(&src_body[52..56]);
    if &file_type == b"slnk" {
        return Err(crate::Error::Unsupported(
            "hfs+ writer: hard-linking a symbolic link is not supported".into(),
        ));
    }
    if &file_type == b"hlnk" && &creator == b"hfs+" {
        return Err(crate::Error::Unsupported(
            "hfs+ writer: hard-linking an existing hard-link entry is not supported".into(),
        ));
    }
    let src_file_id = u32::from_be_bytes(src_body[8..12].try_into().unwrap());
    let (uid, gid, mode_full) = extract_file_perms(src_body)?;
    let src_fork = extract_file_fork(src_body)?;
    let src_body_clone = src_body.clone();

    // Lazily create the private-data directory. After this call the
    // directory's CNID is in `writer.private_dir_cnid`.
    let private_dir = ensure_private_dir(writer)?;

    // Allocate the iNode catalog file's CNID. Per Apple's hardlink
    // convention (visible from a native HFS+ volume with hardlinks),
    // the iNode "number" carried by each `hlnk` record in
    // `HFSPlusBSDInfo.special` is the iNode file's CNID, and its
    // catalog name is `iNode<CNID>` — *not* a parallel counter. fsck
    // walks PD's children, parses the name suffix back into a CNID,
    // and cross-checks the special field. A separate counter desyncs
    // the two and fsck under-counts the chain.
    let inode_cnid = writer.next_cnid;
    let link_inode = inode_cnid;
    writer.next_cnid = writer
        .next_cnid
        .checked_add(1)
        .ok_or_else(|| crate::Error::Unsupported("hfs+: CNID space exhausted".into()))?;

    // Build the iNode<N> file. Re-use the source's data fork verbatim —
    // no bytes change on disk — but rebuild the body with the new
    // file_id so the catalog stays consistent.
    //
    // Apple's hardlink convention (verified against a native HFS+
    // image with hardlinks):
    // * iNode file name = `iNode<CNID>` (decimal CNID of this very
    //   file). fsck parses the suffix back to find the iNode and
    //   cross-check against the `special` field of each hlnk.
    // * `FileInfo.fileType` / `FileInfo.creator` are ZEROED — the
    //   iNode is not a user-facing file and carries no Finder tags.
    //   ('iNod'/'hfs+' tags would mark it as a Finder-recognised
    //   document type, which it isn't.)
    // * Flags: `kHFSHasLinkChainMask (0x0020)` + `kHFSThreadExistsMask
    //   (0x0002)` + Apple's iNode-marker bit `0x0080`. fsck rejects
    //   the iNode as "not a real hardlink target" if the marker bit
    //   is missing and reports "Incorrect number of file hard links".
    // * `BSDInfo.special` = link count (= number of hlnk records
    //   pointing at this iNode).
    let inode_name_str = format!("iNode{link_inode}");
    let inode_name = UniStr::from_str_lossy(&inode_name_str);
    let mut inode_body = encode_file_body(
        inode_cnid,
        mode_full,
        uid,
        gid,
        writer.create_date,
        [0u8; 4], // fileType (zero — Apple convention)
        [0u8; 4], // creator
        &src_fork,
    );
    inode_body[2..4].copy_from_slice(&0x00a2u16.to_be_bytes());
    // The "reserved1" slot at body[4..8] in HFSPlusCatalogFile is
    // overloaded by Apple's hardlink convention as `firstLinkChainID`
    // — the CNID of the first hlnk in the chain. fsck reads this
    // when verifying iNodes ("Error getting first link ID for inode
    // = N" / "first link ID = 0 is < 16" debug output if missing).
    // src_file_id is our head hlnk (chain prev = 0).
    inode_body[4..8].copy_from_slice(&src_file_id.to_be_bytes());
    // BSDInfo starts at byte 32; `special` is the last u32 of the
    // 16-byte BSDInfo struct (byte 32+12 = 44). Set link count = 2
    // (one for the src hlnk, one for the dst).
    inode_body[44..48].copy_from_slice(&2u32.to_be_bytes());
    writer.catalog.insert(
        OwnedKey {
            parent_id: private_dir,
            name: inode_name.clone(),
        },
        inode_body,
    );
    writer.catalog.insert(
        OwnedKey::thread(inode_cnid),
        encode_thread_body(REC_FILE_THREAD, private_dir, &inode_name),
    );
    writer.bump_valence(private_dir, 1)?;

    // If the source file had spilled extents, re-key them under the
    // iNode CNID instead of the old source CNID.
    if src_file_id != inode_cnid {
        let stolen_keys: Vec<(u8, u32, u32)> = writer
            .overflow_extents
            .range((FORK_DATA, src_file_id, 0)..=(FORK_DATA, src_file_id, u32::MAX))
            .map(|(k, _)| *k)
            .collect();
        for k in stolen_keys {
            if let Some(group) = writer.overflow_extents.remove(&k) {
                writer
                    .overflow_extents
                    .insert((FORK_DATA, inode_cnid, k.2), group);
            }
        }
    }

    // Allocate the destination's CNID up-front so we can wire the
    // chain pointers symmetrically. Chain order: src → dst (src is the
    // head, dst is the tail). The iNode itself doesn't sit in the
    // doubly-linked chain; it just carries the link count.
    let dst_cnid = writer.next_cnid;
    writer.next_cnid = writer
        .next_cnid
        .checked_add(1)
        .ok_or_else(|| crate::Error::Unsupported("hfs+: CNID space exhausted".into()))?;

    // Replace the source entry with an hlnk record (head of chain:
    // prev=0, next=dst_cnid). We do this in place (same catalog key)
    // by overwriting the stored body — preserves the existing source
    // file_id (its CNID), which keeps any open thread record under
    // that CNID valid.
    let _ = src_body_clone; // anchored for safety; no longer needed.
    let _ = (mode_full, uid, gid); // permissions live on the iNode, not the hlnks.
    let src_hlnk = encode_hardlink_body(
        src_file_id,
        inode_cnid,
        0,        // prev: head of chain
        dst_cnid, // next: dst hlnk
        writer.create_date,
    );
    writer.catalog.insert(src_key, src_hlnk);

    // Create the destination hlnk record (tail of chain: prev=src,
    // next=0).
    insert_hardlink_entry(
        writer,
        dst_parent,
        dst_name,
        dst_cnid,
        inode_cnid,
        src_file_id, // prev: src hlnk
        0,           // next: tail
    )?;

    // Mark every directory that now contains a hardlink-chain entry
    // with kHFSHasChildLinkMask (0x0040). Apple's HFS+ driver and
    // fsck propagate this up: Private Data holds the iNode (which has
    // HasLinkChain), and src_parent / dst_parent each hold an hlnk
    // (which has HasLinkChain). Without these bits the structural
    // walk reports "out of order" on the Private Data directory.
    set_folder_flag(writer, private_dir, 0x0040)?;
    set_folder_flag(writer, src_parent, 0x0040)?;
    if dst_parent != src_parent {
        set_folder_flag(writer, dst_parent, 0x0040)?;
    }

    Ok(link_inode)
}

/// Set bits in the flags field of the folder record for `cnid`. Used to
/// propagate `kHFSHasChildLinkMask` after a hardlink is created.
pub(crate) fn set_folder_flag(writer: &mut Writer, cnid: u32, mask: u16) -> Result<()> {
    // Find the folder record by walking via the thread (cnid → parent + name).
    let thread_key = OwnedKey::thread(cnid);
    let Some(thread_body) = writer.catalog.get(&thread_key).cloned() else {
        return Ok(()); // root sometimes has its own thread layout; skip.
    };
    if thread_body.len() < 8 {
        return Ok(());
    }
    let parent_parent = u32::from_be_bytes(thread_body[4..8].try_into().unwrap());
    let (name, _) = UniStr::decode(&thread_body[8..])?;
    let folder_key = OwnedKey {
        parent_id: parent_parent,
        name,
    };
    let Some(body) = writer.catalog.get_mut(&folder_key) else {
        return Ok(());
    };
    if body.len() < 4 || i16::from_be_bytes([body[0], body[1]]) != REC_FOLDER {
        return Ok(());
    }
    let cur = u16::from_be_bytes([body[2], body[3]]);
    body[2..4].copy_from_slice(&(cur | mask).to_be_bytes());
    Ok(())
}

/// Remove an entry and (for files) free its data fork.
pub(crate) fn remove_entry(writer: &mut Writer, parent_id: u32, name: &UniStr) -> Result<()> {
    let key = OwnedKey {
        parent_id,
        name: name.clone(),
    };
    let body = writer.catalog.get(&key).ok_or_else(|| {
        crate::Error::InvalidArgument(format!(
            "hfs+ writer: no entry {:?} under CNID {parent_id}",
            name.to_string_lossy()
        ))
    })?;
    if body.len() < 2 {
        return Err(crate::Error::InvalidImage(
            "hfs+ writer: short catalog body".into(),
        ));
    }
    let rec_type = i16::from_be_bytes([body[0], body[1]]);
    let cnid = match rec_type {
        REC_FOLDER => {
            let valence = u32::from_be_bytes(body[4..8].try_into().unwrap());
            if valence != 0 {
                return Err(crate::Error::InvalidArgument(format!(
                    "hfs+ writer: directory {:?} not empty ({valence} children)",
                    name.to_string_lossy()
                )));
            }
            u32::from_be_bytes(body[8..12].try_into().unwrap())
        }
        REC_FILE => {
            let cnid = u32::from_be_bytes(body[8..12].try_into().unwrap());
            // Decode data fork to find blocks to free.
            // dataFork starts at offset 88, 80 bytes.
            let mut buf = [0u8; FORK_DATA_SIZE];
            buf.copy_from_slice(&body[88..88 + FORK_DATA_SIZE]);
            let fork = ForkData::decode(&buf);
            for ext in &fork.extents {
                if ext.block_count == 0 {
                    continue;
                }
                writer.free(ext.start_block, ext.block_count);
            }
            // Drain any spilled extents-overflow records for this file
            // and free the blocks they describe. Records keyed by
            // (FORK_DATA, cnid, _) belong to this file.
            let overflow_keys: Vec<(u8, u32, u32)> = writer
                .overflow_extents
                .range((FORK_DATA, cnid, 0)..=(FORK_DATA, cnid, u32::MAX))
                .map(|(k, _)| *k)
                .collect();
            for key in overflow_keys {
                if let Some(group) = writer.overflow_extents.remove(&key) {
                    for ext in &group {
                        if ext.block_count == 0 {
                            continue;
                        }
                        writer.free(ext.start_block, ext.block_count);
                    }
                }
            }
            cnid
        }
        _ => {
            return Err(crate::Error::InvalidArgument(format!(
                "hfs+ writer: entry {:?} is a thread record (cannot remove)",
                name.to_string_lossy()
            )));
        }
    };
    writer.catalog.remove(&key);
    writer.catalog.remove(&OwnedKey::thread(cnid));
    writer.bump_valence(parent_id, -1)?;
    Ok(())
}

// ----------------------------------------------------------------------
// Flush
// ----------------------------------------------------------------------

/// Serialise the in-memory state (catalog tree, extents-overflow tree,
/// allocation bitmap, volume header) to disk. Idempotent: calling it
/// twice is a no-op after the first success.
pub fn flush(writer: &mut Writer, vh: &mut VolumeHeader, dev: &mut dyn BlockDevice) -> Result<()> {
    if writer.flushed {
        return Ok(());
    }
    // 1. Build catalog records list in key order.
    let mut records: Vec<PackedRecord> = Vec::with_capacity(writer.catalog.len());
    for (key, body) in &writer.catalog {
        let key_bytes = encode_catalog_key(key.parent_id, &key.name);
        records.push(PackedRecord {
            key: key_bytes,
            body: body.clone(),
        });
    }
    let cat_total_nodes = {
        let bytes = u64::from(writer.catalog_file.total_blocks) * u64::from(writer.block_size);
        u32::try_from(bytes / u64::from(writer.node_size)).map_err(|_| {
            crate::Error::Unsupported("hfs+ writer: catalog node count overflow".into())
        })?
    };

    let built = build_btree(records, writer.node_size, cat_total_nodes)?;
    write_btree_to_fork(
        dev,
        &built.nodes,
        &writer.catalog_file,
        writer.node_size,
        writer.block_size,
    )?;
    // 2. Extents-overflow tree. Empty when no fork has spilled past 8
    //    inline extents; otherwise the records we queued in
    //    `writer.overflow_extents` get serialised here.
    let ext_total_nodes = {
        let bytes = u64::from(writer.extents_file.total_blocks) * u64::from(writer.block_size);
        u32::try_from(bytes / u64::from(writer.node_size)).map_err(|_| {
            crate::Error::Unsupported("hfs+ writer: extents-overflow node count overflow".into())
        })?
    };
    let mut ext_records: Vec<PackedRecord> = Vec::with_capacity(writer.overflow_extents.len());
    for ((fork_type, file_id, start_block), group) in &writer.overflow_extents {
        let key = encode_extent_key(*fork_type, *file_id, *start_block);
        let mut body = vec![0u8; EXTENT_RECORD_SIZE];
        for (i, ext) in group.iter().enumerate() {
            let off = i * 8;
            body[off..off + 4].copy_from_slice(&ext.start_block.to_be_bytes());
            body[off + 4..off + 8].copy_from_slice(&ext.block_count.to_be_bytes());
        }
        ext_records.push(PackedRecord { key, body });
    }
    let ext_built = build_btree(ext_records, writer.node_size, ext_total_nodes)?;
    let mut ext_nodes_owned: Vec<Vec<u8>> = ext_built.nodes;
    // Patch the extents-overflow header for the geometry the shared
    // `header_node()` helper got wrong:
    //   - maxKeyLength: 10 (extents key) vs. 516 (catalog key)
    //   - keyCompareType: 0 binary  vs. 0xCF catalog case-fold
    //   - attributes: kBTBigKeysMask only (extents keys are fixed
    //     size, so kBTVariableIndexKeysMask = 4 must NOT be set)
    if let Some(header) = ext_nodes_owned.first_mut() {
        let h = NODE_DESCRIPTOR_SIZE;
        if header.len() >= h + HEADER_REC_SIZE {
            header[h + 20..h + 22].copy_from_slice(&(EXTENT_KEY_PAYLOAD_LEN as u16).to_be_bytes());
            header[h + 37] = 0; // binary compare
            header[h + 38..h + 42].copy_from_slice(&2u32.to_be_bytes()); // kBTBigKeysMask
        }
    }
    write_btree_to_fork(
        dev,
        &ext_nodes_owned,
        &writer.extents_file,
        writer.node_size,
        writer.block_size,
    )?;

    // 3. Allocation bitmap.
    let bm_off =
        u64::from(writer.allocation_file.extents[0].start_block) * u64::from(writer.block_size);
    dev.write_at(bm_off, &writer.bitmap)?;
    // Pad the rest of the allocation-file blocks with zero already done
    // by zero_range at format time.

    // 4. Journal stub. Emitted only when the volume was formatted with
    //    `FormatOpts::journaled = true`. We write a self-consistent
    //    `JournalInfoBlock` followed by a zeroed journal buffer whose
    //    first 512 bytes carry a clean journal header (no transactions
    //    to replay). The kernel will see a journaled volume that does
    //    not need replay on mount.
    if writer.journal_buffer_blocks > 0 {
        let bs = u64::from(writer.block_size);
        let jbuf_offset = u64::from(writer.journal_buffer_start) * bs;
        let jbuf_size = u64::from(writer.journal_buffer_blocks) * bs;
        let info_off = u64::from(writer.journal_info_block) * bs;
        let info = encode_journal_info_block(jbuf_offset, jbuf_size);
        dev.write_at(info_off, &info)?;
        let hdr = encode_journal_header(jbuf_size);
        dev.write_at(jbuf_offset, &hdr)?;
        // Mark the journaled attribute in the volume header.
        vh.attributes |= VOL_ATTR_JOURNALED;
    }

    // 5. Volume header (primary + alternate).
    vh.free_blocks = writer.free_blocks;
    vh.next_catalog_id = writer.next_cnid;
    // Count files vs. folders by scanning catalog (record types 1/2).
    let mut file_count: u32 = 0;
    let mut folder_count: u32 = 0;
    for body in writer.catalog.values() {
        if body.len() < 2 {
            continue;
        }
        match i16::from_be_bytes([body[0], body[1]]) {
            REC_FOLDER => folder_count += 1,
            REC_FILE => file_count += 1,
            _ => {}
        }
    }
    // Root folder is not counted in folder_count per TN1150 ("does not
    // include the root folder").
    folder_count = folder_count.saturating_sub(1);

    let buf = encode_volume_header(
        vh,
        writer.next_alloc,
        file_count,
        folder_count,
        writer.create_date,
        writer.journal_info_block,
    );
    dev.write_at(VOLUME_HEADER_OFFSET, &buf)?;

    // Alternate volume header lives in the volume's last 1024-byte
    // sector. Compute the offset as (total_size - 1024); pad sector 1024
    // bytes to a full 512-byte sector by writing only the 512-byte
    // header into the right place.
    let total = u64::from(writer.total_blocks) * u64::from(writer.block_size);
    if total >= 1024 {
        let alt_off = total - 1024;
        dev.write_at(alt_off, &buf)?;
    }

    dev.sync()?;
    writer.flushed = true;
    Ok(())
}

/// Encode a 180-byte `JournalInfoBlock` for an in-volume journal. The
/// rest of the block is left zero (matches what Apple does — the
/// reserved area is large and unused).
///
/// TN1150 layout, condensed:
///
/// ```text
/// 0       4   flags                  (kJIJournalInFSMask = 2)
/// 4       32  device_signature[8]    (zero for in-volume journals)
/// 36      8   offset                 (byte offset of journal buffer)
/// 44      8   size                   (byte size of journal buffer)
/// 52      37  ext_jnl_uuid           (zeroed)
/// 89      48  machine_serial_num     (zeroed)
/// ...     ... reserved
/// ```
fn encode_journal_info_block(buf_offset: u64, buf_size: u64) -> [u8; 512] {
    let mut b = [0u8; 512];
    b[0..4].copy_from_slice(&JI_JOURNAL_IN_FS.to_be_bytes());
    // device_signature[8]: 32 bytes of zero.
    b[36..44].copy_from_slice(&buf_offset.to_be_bytes());
    b[44..52].copy_from_slice(&buf_size.to_be_bytes());
    // ext_jnl_uuid + machine_serial_num + reserved: zeroed.
    b
}

/// Encode the journal header that lives at the start of the journal
/// buffer. With `start == end == jhdr_size` and no transactions queued,
/// the kernel concludes there is nothing to replay.
///
/// TN1150 / Apple `journal.h`:
///
/// ```text
/// 0    4   magic       0x4a4e4c78 "JNLx"
/// 4    4   endian      0x12345678
/// 8    8   start       (= jhdr_size, no transactions)
/// 16   8   end         (= start)
/// 24   8   size        journal buffer size in bytes
/// 32   4   blhdr_size  block-list-header size (== sector size, 512)
/// 36   4   checksum    CRC over the header w/ this field 0
/// 40   4   jhdr_size   size of this header (== 512)
/// ```
fn encode_journal_header(buf_size: u64) -> [u8; 512] {
    let mut b = [0u8; 512];
    let jhdr_size: u32 = 512;
    b[0..4].copy_from_slice(&JOURNAL_HEADER_MAGIC.to_be_bytes());
    b[4..8].copy_from_slice(&JOURNAL_HEADER_ENDIAN.to_be_bytes());
    b[8..16].copy_from_slice(&u64::from(jhdr_size).to_be_bytes());
    b[16..24].copy_from_slice(&u64::from(jhdr_size).to_be_bytes());
    b[24..32].copy_from_slice(&buf_size.to_be_bytes());
    b[32..36].copy_from_slice(&jhdr_size.to_be_bytes()); // blhdr_size (use jhdr_size)
    // Checksum over the header with the checksum field zeroed.
    b[36..40].copy_from_slice(&0u32.to_be_bytes());
    b[40..44].copy_from_slice(&jhdr_size.to_be_bytes());
    let csum = journal_header_checksum(&b);
    b[36..40].copy_from_slice(&csum.to_be_bytes());
    b
}

/// CRC-32 over the journal header bytes with the checksum field
/// zeroed. We use Apple's variant (CRC-32 with reflected polynomial
/// 0xEDB88320, initial 0xFFFFFFFF, finalise without XOR) — that's
/// the same algorithm zlib calls "CRC32" minus the final XOR.
fn journal_header_checksum(buf: &[u8]) -> u32 {
    // Compute over the entire 512-byte header. Apple's journal code
    // only covers the journal-header struct (jhdr_size bytes), which is
    // exactly the 512-byte sector we built.
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in buf {
        let mut c = (crc ^ u32::from(byte)) & 0xff;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
        }
        crc = (crc >> 8) ^ c;
    }
    crc
}

/// Write a sequence of pre-encoded B-tree nodes into the fork's first
/// extent on disk. Panics if the nodes don't fit (the caller has
/// already validated capacity).
fn write_btree_to_fork(
    dev: &mut dyn BlockDevice,
    nodes: &[Vec<u8>],
    fork: &ForkData,
    node_size: u32,
    block_size: u32,
) -> Result<()> {
    if nodes.is_empty() {
        return Ok(());
    }
    let mut node_idx = 0usize;
    for ext in &fork.extents {
        if ext.block_count == 0 {
            continue;
        }
        let ext_bytes = u64::from(ext.block_count) * u64::from(block_size);
        let nodes_here = (ext_bytes / u64::from(node_size)) as usize;
        let mut off = u64::from(ext.start_block) * u64::from(block_size);
        for _ in 0..nodes_here {
            if node_idx >= nodes.len() {
                return Ok(());
            }
            dev.write_at(off, &nodes[node_idx])?;
            off += u64::from(node_size);
            node_idx += 1;
        }
    }
    if node_idx < nodes.len() {
        return Err(crate::Error::Unsupported(format!(
            "hfs+ writer: only wrote {} of {} B-tree nodes (fork capacity exhausted)",
            node_idx,
            nodes.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn format_emits_valid_volume_header() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts {
            volume_name: "TestVol".into(),
            ..FormatOpts::default()
        };
        let (mut vh, mut writer) = format(&mut dev, &opts).unwrap();
        flush(&mut writer, &mut vh, &mut dev).unwrap();

        // Verify probe and re-open work.
        assert!(crate::fs::hfs_plus::probe(&mut dev).unwrap());
        let hfs = crate::fs::hfs_plus::HfsPlus::open(&mut dev).unwrap();
        assert_eq!(hfs.volume_name(), "TestVol");
        assert_eq!(hfs.block_size(), DEFAULT_BLOCK_SIZE);
    }

    #[test]
    fn root_directory_is_empty() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts::default();
        let (mut vh, mut writer) = format(&mut dev, &opts).unwrap();
        flush(&mut writer, &mut vh, &mut dev).unwrap();
        let hfs = crate::fs::hfs_plus::HfsPlus::open(&mut dev).unwrap();
        let entries = hfs.list_path(&mut dev, "/").unwrap();
        assert!(entries.is_empty(), "fresh root should be empty");
    }

    #[test]
    fn allocate_then_free_reclaims_space() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts::default();
        let (_vh, mut writer) = format(&mut dev, &opts).unwrap();
        let before = writer.free_blocks;
        let blk = writer.allocate(3).unwrap();
        assert!(writer.free_blocks <= before - 3);
        writer.free(blk, 3);
        assert_eq!(writer.free_blocks, before);
    }

    #[test]
    fn encode_volume_header_roundtrip() {
        let vh = VolumeHeader {
            signature: SIG_HFS_PLUS,
            version: 4,
            attributes: VOL_ATTR_UNMOUNTED,
            block_size: 4096,
            total_blocks: 1024,
            free_blocks: 1000,
            next_catalog_id: 17,
            allocation_file: ForkData::default(),
            extents_file: ForkData::default(),
            catalog_file: ForkData::default(),
            attributes_file: ForkData::default(),
            startup_file: ForkData::default(),
        };
        let buf = encode_volume_header(&vh, 24, 0, 0, 0, 0);
        let decoded = VolumeHeader::decode(&buf).unwrap();
        assert_eq!(decoded.signature, SIG_HFS_PLUS);
        assert_eq!(decoded.version, 4);
        assert_eq!(decoded.block_size, 4096);
        assert_eq!(decoded.total_blocks, 1024);
        assert_eq!(decoded.free_blocks, 1000);
        assert_eq!(decoded.next_catalog_id, 17);
    }

    #[test]
    fn create_dir_appears_in_listing() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts::default();
        let mut hfs = crate::fs::hfs_plus::HfsPlus::format(&mut dev, &opts).unwrap();
        hfs.create_dir(&mut dev, "/foo", 0o755, 0, 0).unwrap();
        hfs.flush(&mut dev).unwrap();
        let entries = hfs.list_path(&mut dev, "/").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "foo");
        assert_eq!(entries[0].kind, crate::fs::EntryKind::Dir);

        // Re-open from scratch and verify persistence.
        let hfs2 = crate::fs::hfs_plus::HfsPlus::open(&mut dev).unwrap();
        let entries = hfs2.list_path(&mut dev, "/").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "foo");
    }

    #[test]
    fn create_file_stores_data() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts::default();
        let mut hfs = crate::fs::hfs_plus::HfsPlus::format(&mut dev, &opts).unwrap();

        let data = b"hello, hfs+ world";
        let mut src = std::io::Cursor::new(data);
        hfs.create_file(
            &mut dev,
            "/hi.txt",
            &mut src,
            data.len() as u64,
            0o644,
            0,
            0,
        )
        .unwrap();
        hfs.flush(&mut dev).unwrap();

        // Read back via the read path on a freshly opened volume.
        let hfs2 = crate::fs::hfs_plus::HfsPlus::open(&mut dev).unwrap();
        let size = hfs2.file_size(&mut dev, "/hi.txt").unwrap();
        assert_eq!(size, data.len() as u64);
        let mut reader = hfs2.open_file_reader(&mut dev, "/hi.txt").unwrap();
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn create_symlink_roundtrip() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts::default();
        let mut hfs = crate::fs::hfs_plus::HfsPlus::format(&mut dev, &opts).unwrap();
        hfs.create_symlink(&mut dev, "/link", "../target/path", 0o777, 0, 0)
            .unwrap();
        hfs.flush(&mut dev).unwrap();

        let hfs2 = crate::fs::hfs_plus::HfsPlus::open(&mut dev).unwrap();
        let target = hfs2.read_symlink_target_path(&mut dev, "/link").unwrap();
        assert_eq!(target, "../target/path");
        let entries = hfs2.list_path(&mut dev, "/").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kind, crate::fs::EntryKind::Symlink);
    }

    #[test]
    fn nested_directories_resolve() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts::default();
        let mut hfs = crate::fs::hfs_plus::HfsPlus::format(&mut dev, &opts).unwrap();
        hfs.create_dir(&mut dev, "/a", 0o755, 0, 0).unwrap();
        hfs.create_dir(&mut dev, "/a/b", 0o755, 0, 0).unwrap();
        let data = b"deep";
        let mut src = std::io::Cursor::new(data);
        hfs.create_file(
            &mut dev,
            "/a/b/c.txt",
            &mut src,
            data.len() as u64,
            0o644,
            0,
            0,
        )
        .unwrap();
        hfs.flush(&mut dev).unwrap();

        let hfs2 = crate::fs::hfs_plus::HfsPlus::open(&mut dev).unwrap();
        let entries = hfs2.list_path(&mut dev, "/a/b").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "c.txt");
        let size = hfs2.file_size(&mut dev, "/a/b/c.txt").unwrap();
        assert_eq!(size, data.len() as u64);
    }

    #[test]
    fn remove_file_frees_blocks_and_drops_entry() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts::default();
        let mut hfs = crate::fs::hfs_plus::HfsPlus::format(&mut dev, &opts).unwrap();

        let before = {
            let w = hfs_test_writer(&hfs);
            w.free_blocks
        };
        let data = vec![0xAAu8; 12_000]; // ~3 blocks at 4 KiB.
        let mut src = std::io::Cursor::new(&data);
        hfs.create_file(&mut dev, "/big", &mut src, data.len() as u64, 0o644, 0, 0)
            .unwrap();
        hfs.remove(&mut dev, "/big").unwrap();
        let after = {
            let w = hfs_test_writer(&hfs);
            w.free_blocks
        };
        assert_eq!(before, after, "remove must reclaim allocated blocks");

        hfs.flush(&mut dev).unwrap();
        let hfs2 = crate::fs::hfs_plus::HfsPlus::open(&mut dev).unwrap();
        let entries = hfs2.list_path(&mut dev, "/").unwrap();
        assert!(entries.is_empty(), "removed file must not be listed");
    }

    #[test]
    fn many_directories_force_btree_split() {
        // 200 directories under root should exceed a single 8 KiB leaf
        // (each leaf record is ~120 bytes), forcing leaves to split and
        // an index level to be built.
        let mut dev = MemoryBackend::new(16 * 1024 * 1024);
        let opts = FormatOpts {
            catalog_nodes: 64,
            ..FormatOpts::default()
        };
        let mut hfs = crate::fs::hfs_plus::HfsPlus::format(&mut dev, &opts).unwrap();
        for i in 0..200 {
            hfs.create_dir(&mut dev, &format!("/d{i:03}"), 0o755, 0, 0)
                .unwrap();
        }
        hfs.flush(&mut dev).unwrap();
        let hfs2 = crate::fs::hfs_plus::HfsPlus::open(&mut dev).unwrap();
        let entries = hfs2.list_path(&mut dev, "/").unwrap();
        assert_eq!(entries.len(), 200);
    }

    /// Test-only: pry the in-memory writer out of an `HfsPlus` so we can
    /// inspect free_blocks across mutating calls without flushing.
    fn hfs_test_writer(hfs: &crate::fs::hfs_plus::HfsPlus) -> &Writer {
        hfs.test_writer().expect("writable handle")
    }

    /// Fragment the writer's allocation bitmap so that the only free
    /// space is a series of `hole_count` evenly-sized holes (each
    /// `hole_blocks` blocks long), each separated from the next by at
    /// least one used "fence" block. Returns the hole starting-block
    /// indices. Used by the spill tests below to coerce
    /// `stream_data_to_blocks` into producing many small extents.
    fn make_holes(writer: &mut Writer, hole_blocks: u32, hole_count: u32) -> Vec<u32> {
        let mut holes = Vec::with_capacity(hole_count as usize);
        // Reserve a fence block before the first hole.
        let _ = writer.allocate(1).unwrap();
        for _ in 0..hole_count {
            holes.push(writer.allocate(hole_blocks).unwrap());
            let _ = writer.allocate(1).unwrap();
        }
        // Drain everything that's still free past the fenced region so
        // the only free space left, post-free, is the holes themselves.
        while writer.free_blocks > 0 {
            if writer.allocate_largest_run(writer.free_blocks).is_err() {
                break;
            }
        }
        for &h in &holes {
            writer.free(h, hole_blocks);
        }
        holes
    }

    #[test]
    fn extents_overflow_spill_round_trips() {
        // 16 MiB volume so the bitmap, catalog, and overflow trees
        // leave us enough room to fragment.
        let mut dev = MemoryBackend::new(16 * 1024 * 1024);
        let opts = FormatOpts {
            volume_name: "Spill".into(),
            extents_nodes: 4, // header + room for a couple of leaves
            ..FormatOpts::default()
        };
        let (mut vh, mut writer) = format(&mut dev, &opts).unwrap();

        // Carve 16 single-block holes so a write of 16 blocks must
        // produce 16 separate extents — 8 inline + 8 spilled into the
        // extents-overflow B-tree.
        let holes = make_holes(&mut writer, 1, 16);
        assert_eq!(holes.len(), 16);

        // The data we'll write: distinct bytes per block so the
        // ordering of the runs is verifiable on read-back.
        let bs = writer.block_size as usize;
        let mut payload = vec![0u8; bs * 16];
        for (block_idx, chunk) in payload.chunks_mut(bs).enumerate() {
            chunk.fill((block_idx as u8).wrapping_add(0x10));
        }

        // Pick a fresh CNID for the file we're about to materialise.
        let file_cnid = writer.next_cnid;
        writer.next_cnid += 1;
        let mut src = std::io::Cursor::new(&payload);
        let fork = stream_data_to_blocks(
            &mut writer,
            &mut dev,
            &mut src,
            payload.len() as u64,
            file_cnid,
        )
        .unwrap();

        // We expect exactly 8 inline extents and 8 overflow extents.
        assert_eq!(fork.total_blocks, 16);
        assert_eq!(
            fork.extents.iter().filter(|e| e.block_count > 0).count(),
            FORK_EXTENT_COUNT,
            "all 8 inline extents should be populated"
        );
        let overflow_record_count = writer
            .overflow_extents
            .range((FORK_DATA, file_cnid, 0)..=(FORK_DATA, file_cnid, u32::MAX))
            .count();
        assert_eq!(
            overflow_record_count, 1,
            "8 spilled extents fit in a single overflow record"
        );

        // Wire the catalog entry up under root.
        let name = UniStr::from_str_lossy("spill.bin");
        insert_file(
            &mut writer,
            ROOT_FOLDER_ID,
            &name,
            file_cnid,
            0o644,
            0,
            0,
            *b"\0\0\0\0",
            *b"\0\0\0\0",
            &fork,
            false,
        )
        .unwrap();

        flush(&mut writer, &mut vh, &mut dev).unwrap();

        // Re-open and read back, byte-for-byte.
        let hfs = crate::fs::hfs_plus::HfsPlus::open(&mut dev).unwrap();
        let size = hfs.file_size(&mut dev, "/spill.bin").unwrap();
        assert_eq!(size, payload.len() as u64);
        let mut reader = hfs.open_file_reader(&mut dev, "/spill.bin").unwrap();
        let mut got = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut got).unwrap();
        assert_eq!(got, payload, "spilled file must read back byte-exact");
    }

    #[test]
    fn extents_overflow_spill_many_records() {
        // Force at least 20 separate extents — more than fits in one
        // overflow leaf record (each record holds 8 extents) — to
        // exercise multi-record packing in the overflow tree.
        let mut dev = MemoryBackend::new(16 * 1024 * 1024);
        let opts = FormatOpts {
            volume_name: "Spill2".into(),
            extents_nodes: 4,
            ..FormatOpts::default()
        };
        let (mut vh, mut writer) = format(&mut dev, &opts).unwrap();
        let holes = make_holes(&mut writer, 1, 20);
        assert_eq!(holes.len(), 20);

        let bs = writer.block_size as usize;
        let mut payload = vec![0u8; bs * 20];
        for (i, chunk) in payload.chunks_mut(bs).enumerate() {
            chunk.fill(((i + 1) as u8).wrapping_mul(7));
        }
        let cnid = writer.next_cnid;
        writer.next_cnid += 1;
        let mut src = std::io::Cursor::new(&payload);
        let fork =
            stream_data_to_blocks(&mut writer, &mut dev, &mut src, payload.len() as u64, cnid)
                .unwrap();
        assert_eq!(fork.total_blocks, 20);
        // 8 inline + 12 spilled, packed 8-per-record => 2 overflow records.
        let overflow_record_count = writer
            .overflow_extents
            .range((FORK_DATA, cnid, 0)..=(FORK_DATA, cnid, u32::MAX))
            .count();
        assert_eq!(overflow_record_count, 2);

        insert_file(
            &mut writer,
            ROOT_FOLDER_ID,
            &UniStr::from_str_lossy("multi.bin"),
            cnid,
            0o644,
            0,
            0,
            *b"\0\0\0\0",
            *b"\0\0\0\0",
            &fork,
            false,
        )
        .unwrap();
        flush(&mut writer, &mut vh, &mut dev).unwrap();

        let hfs = crate::fs::hfs_plus::HfsPlus::open(&mut dev).unwrap();
        let mut reader = hfs.open_file_reader(&mut dev, "/multi.bin").unwrap();
        let mut got = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut got).unwrap();
        assert_eq!(got, payload);
    }

    #[test]
    fn remove_frees_overflow_extents() {
        let mut dev = MemoryBackend::new(16 * 1024 * 1024);
        let opts = FormatOpts {
            extents_nodes: 4,
            ..FormatOpts::default()
        };
        let (mut vh, mut writer) = format(&mut dev, &opts).unwrap();
        let _holes = make_holes(&mut writer, 1, 12);

        let before_free = writer.free_blocks;

        let bs = writer.block_size as usize;
        let payload = vec![0xAB; bs * 12];
        let cnid = writer.next_cnid;
        writer.next_cnid += 1;
        let mut src = std::io::Cursor::new(&payload);
        let fork =
            stream_data_to_blocks(&mut writer, &mut dev, &mut src, payload.len() as u64, cnid)
                .unwrap();
        let name = UniStr::from_str_lossy("to-remove.bin");
        insert_file(
            &mut writer,
            ROOT_FOLDER_ID,
            &name,
            cnid,
            0o644,
            0,
            0,
            *b"\0\0\0\0",
            *b"\0\0\0\0",
            &fork,
            false,
        )
        .unwrap();
        // Sanity: overflow records were created.
        assert!(!writer.overflow_extents.is_empty());

        // Now remove and verify both inline + overflow blocks come back.
        remove_entry(&mut writer, ROOT_FOLDER_ID, &name).unwrap();
        assert!(writer.overflow_extents.is_empty());
        assert_eq!(writer.free_blocks, before_free);

        flush(&mut writer, &mut vh, &mut dev).unwrap();
    }

    #[test]
    fn create_hardlink_round_trips() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts::default();
        let mut hfs = crate::fs::hfs_plus::HfsPlus::format(&mut dev, &opts).unwrap();

        // Source file with distinctive bytes.
        let data = b"hard-link payload\n".repeat(64); // ~ 1 KiB
        let mut src = std::io::Cursor::new(&data);
        hfs.create_file(&mut dev, "/src", &mut src, data.len() as u64, 0o644, 0, 0)
            .unwrap();
        let link_inode = hfs.create_hardlink(&mut dev, "/src", "/dst").unwrap();
        // The iNode "number" returned by `create_hardlink` is the CNID
        // of the iNode file in PD — Apple uses the CNID as the link
        // number, so the value depends on how many CNIDs have been
        // handed out before. With `/src` taking the first user CNID
        // (16) and PD taking 17, the iNode lands at 18.
        assert_eq!(link_inode, 18, "first user file occupies CNIDs 16+");
        hfs.flush(&mut dev).unwrap();

        // Re-open from disk and verify both names yield the same bytes.
        let hfs2 = crate::fs::hfs_plus::HfsPlus::open(&mut dev).unwrap();
        for path in ["/src", "/dst"] {
            let size = hfs2.file_size(&mut dev, path).unwrap();
            assert_eq!(size, data.len() as u64, "size for {path}");
            let mut reader = hfs2.open_file_reader(&mut dev, path).unwrap();
            let mut got = Vec::new();
            std::io::Read::read_to_end(&mut reader, &mut got).unwrap();
            assert_eq!(got, data, "bytes for {path}");
        }

        // The HFS+ private-data directory must exist with one iNode child.
        let root = hfs2.list_path(&mut dev, "/").unwrap();
        assert!(
            root.iter().any(|e| e.name.contains("HFS+ Private Data")),
            "private-data directory should appear in root listing, got {:?}",
            root.iter().map(|e| &e.name).collect::<Vec<_>>()
        );
    }

    #[test]
    fn create_hardlink_rejects_self_link_and_symlink() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts::default();
        let mut hfs = crate::fs::hfs_plus::HfsPlus::format(&mut dev, &opts).unwrap();

        let data = b"x";
        let mut src = std::io::Cursor::new(&data[..]);
        hfs.create_file(&mut dev, "/file", &mut src, 1, 0o644, 0, 0)
            .unwrap();
        // Same source / destination path: error.
        assert!(hfs.create_hardlink(&mut dev, "/file", "/file").is_err());

        // Symlinks may not be hard-linked.
        hfs.create_symlink(&mut dev, "/sym", "/file", 0o777, 0, 0)
            .unwrap();
        assert!(hfs.create_hardlink(&mut dev, "/sym", "/sym-link").is_err());
    }

    #[test]
    fn create_hardlink_then_third_link_shares_inode() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts::default();
        let mut hfs = crate::fs::hfs_plus::HfsPlus::format(&mut dev, &opts).unwrap();

        let data = b"three-way link\n";
        let mut src = std::io::Cursor::new(&data[..]);
        hfs.create_file(&mut dev, "/a", &mut src, data.len() as u64, 0o644, 0, 0)
            .unwrap();
        let _ = hfs.create_hardlink(&mut dev, "/a", "/b").unwrap();
        // Creating a hard link *to an existing hard link* is not
        // supported in v1 (the source would itself already be an hlnk
        // record, which `promote_to_hardlink` rejects).
        assert!(hfs.create_hardlink(&mut dev, "/a", "/c").is_err());
    }

    #[test]
    fn format_journaled_sets_attribute_bit() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts {
            journaled: true,
            ..FormatOpts::default()
        };
        let (mut vh, mut writer) = format(&mut dev, &opts).unwrap();
        assert!(writer.journal_buffer_blocks > 0);
        assert!(writer.journal_info_block != 0);
        flush(&mut writer, &mut vh, &mut dev).unwrap();
        assert!(
            vh.attributes & VOL_ATTR_JOURNALED != 0,
            "journaled volume must set kHFSVolumeJournaledMask",
        );

        // Read the volume header off disk and confirm the bit + the
        // journal-info-block pointer survive the round trip.
        let parsed = crate::fs::hfs_plus::volume_header::read_volume_header(&mut dev).unwrap();
        assert!(parsed.attributes & VOL_ATTR_JOURNALED != 0);

        // The journal info block field is at offset 0x00C in the
        // 512-byte header. Read it directly.
        let mut raw = [0u8; 512];
        dev.read_at(VOLUME_HEADER_OFFSET, &mut raw).unwrap();
        let jib = u32::from_be_bytes(raw[12..16].try_into().unwrap());
        assert_eq!(jib, writer.journal_info_block);

        // The journal header at the start of the journal buffer must
        // begin with the JNLx magic.
        let mut hdr = [0u8; 16];
        dev.read_at(
            u64::from(writer.journal_buffer_start) * u64::from(writer.block_size),
            &mut hdr,
        )
        .unwrap();
        let magic = u32::from_be_bytes(hdr[0..4].try_into().unwrap());
        assert_eq!(magic, JOURNAL_HEADER_MAGIC);
        let endian = u32::from_be_bytes(hdr[4..8].try_into().unwrap());
        assert_eq!(endian, JOURNAL_HEADER_ENDIAN);
    }

    #[test]
    fn format_unjournaled_does_not_touch_journal_fields() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = FormatOpts::default();
        let (mut vh, mut writer) = format(&mut dev, &opts).unwrap();
        flush(&mut writer, &mut vh, &mut dev).unwrap();
        assert_eq!(vh.attributes & VOL_ATTR_JOURNALED, 0);
        // No blocks reserved for journal.
        assert_eq!(writer.journal_info_block, 0);
        assert_eq!(writer.journal_buffer_blocks, 0);
    }
}
