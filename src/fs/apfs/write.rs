//! APFS image writer — best-effort, single-volume, no checkpoint replay.
//!
//! ## Scope
//!
//! This produces a minimal but *readable* APFS image: an NXSB at block 0
//! plus an additional NXSB inside the checkpoint descriptor area, a
//! container omap, a stub spaceman, one volume with its own omap, an
//! fs-tree (single- or multi-leaf), and any number of regular files,
//! directories, symlinks and xattrs rooted at inode 2.
//!
//! Layout (block numbers given for `block_size = 4096`):
//!
//! ```text
//!   0      NXSB           container label
//!   1      checkpoint_map (resolves spaceman ephemeral oid → block 17)
//!   2      NXSB           live checkpoint (xid=2)
//!   3..16  reserved xp_desc slots for future NXSB checkpoints
//!          (used by reopen-and-write — see [`crate::fs::apfs::Apfs::open_writable`])
//!  17      spaceman_phys  real header + inline single-CIB address array
//!  18      omap_phys_t    container omap header
//!  19      APSB           volume superblock
//!  20      omap_phys_t    volume omap header
//!  21..    bump area — fs-tree leaves, omap leaves/internal nodes,
//!          file-extent data blocks. The CIB and per-chunk allocation
//!          bitmap(s) referenced by the spaceman are also bump-allocated
//!          at the *end* of this area, after every other block address
//!          has been pinned.
//! ```
//!
//! ## xp_desc area sizing
//!
//! The writer reserves `XP_DESC_BLOCKS` (16) blocks for the checkpoint
//! descriptor area instead of the bare minimum (2). The extra slots are
//! never used by `finish()` — they stay zeroed and are valid "no-op"
//! entries that the reader ignores when scanning for the latest-xid
//! NXSB. The reopen-and-write path
//! ([`crate::fs::apfs::Apfs::open_writable`]) consumes these slots one
//! per `sync()` so each new checkpoint goes at a fresh xp_desc block,
//! leaving the previous checkpoint's NXSB untouched (COW).
//!
//! Metadata and data blocks past the fixed prefix are bump-allocated:
//! metadata first, then data. The on-disk locations of the various
//! omap/fsroot roots are recorded by [`ApfsWriter::finish`] and stamped
//! into the NXSB/APSB before they're written.
//!
//! ## Space manager
//!
//! The `spaceman` module emits a structurally-correct
//! `spaceman_phys_t` describing the entire container as a single
//! device. A single chunk-info-block (CIB) lists one `chunk_info_t` per
//! container chunk (`blocks_per_chunk = 8 * block_size`); each non-empty
//! chunk points at its own allocation-bitmap block where set bits mean
//! "used". The bitmaps reflect every block the writer touched (NXSB
//! copies, checkpoint map, spaceman/CIB/bitmaps themselves, omap and
//! fs-tree nodes, and file-extent data blocks).
//!
//! The writer also reserves a small internal-pool (IP) ring near the
//! start of the container, populates the three space-manager
//! free-queue B-trees (`SFQ_IP`, `SFQ_MAIN`, `SFQ_TIER2`) as empty
//! fixed-KV trees, and writes one main-device allocation zone covering
//! the whole device into `sm_datazone`. The IP ring is never allocated
//! from during formatting — every block is just bump-allocated so the
//! bitmap correctly marks the ring blocks as "used" — but the
//! `spaceman_phys_t` fields describing the ring are fully populated
//! per the Apple File System Reference.
//!
//! ## Multi-leaf fs-tree
//!
//! When the staged fs-tree records overflow a single leaf, the writer
//! emits multiple leaf blocks (each ≤ `block_size - btree_info_t`) and
//! builds a single internal-node *root* above them. The internal root's
//! keys are the first record key of each leaf and its values are 8-byte
//! virtual oids; each leaf's vid is added to the volume omap so the
//! reader can resolve `(vid → paddr)` during descent. With ~4 KiB blocks
//! the internal root can address hundreds of leaves before it itself
//! overflows; if that ever happens, `finish` returns `Unsupported`.
//!
//! ## Multi-leaf omaps
//!
//! The container and volume omaps follow the same split rule, except the
//! internal-node child pointers are physical block addresses (the omap
//! is a physical tree, not virtual).
//!
//! ## Streaming invariant
//!
//! File bytes are copied through a 64 KiB scratch buffer; the writer
//! never loads a whole file into memory. Metadata blocks are each
//! bounded by `block_size`.
//!
//! ## Public API
//!
//! ```ignore
//! let mut w = ApfsWriter::new(&mut dev, total_blocks, 4096, "MyVolume")?;
//! let dir = w.add_dir(2, "subdir", 0o755)?;
//! w.add_file_from_reader(dir, "hello.txt", 0o644, &mut some_reader, size)?;
//! w.add_symlink(2, "link", 0o777, "target")?;
//! w.add_xattr(dir, "user.note", b"hello")?;
//! w.finish()?;
//! ```
//!
//! ## Limits
//!
//! - No checkpoint replay is performed (we don't pretend to be journal-
//!   recoverable).
//! - Multi-leaf trees can only have one internal level (root + leaves);
//!   if the internal root overflows, `finish` returns `Unsupported`.
//! - Xattr values larger than [`APFS_XATTR_MAX_EMBEDDED_SIZE`] (3804
//!   bytes) require a `j_xattr_dstream_t` referencing a separate
//!   dstream object. That layout is deferred — [`ApfsWriter::add_xattr`]
//!   returns `Unsupported` for oversized values.
//! - No snapshots, no encryption, no clones, no compression.
//! - The image must be at least ~32 blocks long.

use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;

use super::btree::{BTNODE_FIXED_KV_SIZE, BTNODE_LEAF, BTNODE_ROOT, BTREE_INFO_SIZE};
use super::checksum::fletcher64;
use super::jrec::{
    APFS_TYPE_DIR_REC, APFS_TYPE_DSTREAM_ID, APFS_TYPE_FILE_EXTENT, APFS_TYPE_INODE,
    APFS_TYPE_XATTR, DT_DIR, DT_LNK, DT_REG, INO_EXT_TYPE_DSTREAM, J_INODE_VAL_FIXED_SIZE,
    OBJ_ID_MASK, OBJ_TYPE_SHIFT,
};
use super::obj::{
    OBJECT_TYPE_BTREE, OBJECT_TYPE_BTREE_NODE, OBJECT_TYPE_CHECKPOINT_MAP, OBJECT_TYPE_FS,
    OBJECT_TYPE_FSTREE, OBJECT_TYPE_NX_SUPERBLOCK, OBJECT_TYPE_OMAP,
};
use super::spaceman::{self, DEFAULT_IP_BLOCK_COUNT, SpacemanLayout, blocks_per_chunk};
use super::superblock::{APFS_MAGIC, NX_MAGIC, NX_MAX_FILE_SYSTEMS};

/// `OBJ_VIRTUAL = 0` (the default flag); `OBJ_PHYSICAL = 0x4000_0000`;
/// `OBJ_EPHEMERAL = 0x8000_0000`. We use OBJ_PHYSICAL for everything
/// that lives at a fixed block address, OBJ_VIRTUAL (= 0) for omap-
/// resolved entities.
const OBJ_PHYSICAL: u32 = 0x4000_0000;
const OBJ_EPHEMERAL: u32 = 0x8000_0000;

/// 64 KiB scratch buffer for streamed file copies.
const COPY_BUF: usize = 64 * 1024;

/// Magic xid we use for every object we write. Larger than any plausible
/// label-NXSB xid (1) so our blocks "win" the most-recent-checkpoint
/// pick in the reader.
pub(crate) const WRITE_XID: u64 = 2;

/// Number of blocks reserved for the checkpoint descriptor area.
/// Sized to hold the chkmap stub (1 block), the live NXSB
/// (1 block), and 14 future-checkpoint NXSB slots for the
/// reopen-and-write path.
pub(crate) const XP_DESC_BLOCKS: u32 = 16;

/// Physical block of the chkmap stub (first xp_desc slot).
pub(crate) const CHKMAP_PADDR: u64 = 1;
/// Physical block of the initial live NXSB (second xp_desc slot).
pub(crate) const NXSB_LIVE_PADDR: u64 = 2;
/// Physical block of the spaceman_phys (just past the xp_desc area).
pub(crate) const SPACEMAN_PADDR: u64 = (XP_DESC_BLOCKS as u64) + 1;
/// Physical block of the container omap_phys_t header.
pub(crate) const CONT_OMAP_PADDR: u64 = SPACEMAN_PADDR + 1;
/// Physical block of the volume superblock (APSB).
pub(crate) const APSB_PADDR: u64 = CONT_OMAP_PADDR + 1;
/// Physical block of the volume omap_phys_t header.
pub(crate) const VOL_OMAP_PADDR: u64 = APSB_PADDR + 1;
/// First bump-allocated block. Everything past this is COW-friendly
/// scratch space for fs-tree leaves, omap leaves, and file extents.
pub(crate) const BUMP_BLOCK_START: u64 = VOL_OMAP_PADDR + 1;

/// Inode-2 default-data-stream object id constant. The fs-tree pairs
/// every regular-file inode with a `private_id` pointing at the dstream
/// object id; we encode them with `private_id == inode_oid` (i.e. the
/// dstream and inode share an id), which is permitted by the spec and
/// is also what genuine APFS volumes do for normal files.
const DSTREAM_ID_SHARES_INODE: bool = true;

/// `XATTR_DATA_EMBEDDED` flag in `j_xattr_val_t.flags`. When set the
/// xattr value is stored inline immediately after the val header.
const XATTR_DATA_EMBEDDED: u16 = 0x0002;

/// Hard cap on the embedded-xattr value size, taken from the Apple File
/// System Reference (constant `APFS_XATTR_MAX_EMBEDDED_SIZE`). Values up
/// to and including this size can be stored inline; larger ones require
/// a `j_xattr_dstream_t` (not implemented).
pub const APFS_XATTR_MAX_EMBEDDED_SIZE: usize = 3804;

/// First virtual oid we assign to fs-tree leaves. We start well past the
/// reserved-range used by other writer constants.
const FS_LEAF_VID_BASE: u64 = 0x1_0000;

/// One pending fs-tree record (key + value bytes), used by the in-memory
/// builder before serialization.
#[derive(Clone)]
struct FsRecord {
    key: Vec<u8>,
    val: Vec<u8>,
}

/// Type/category sort order tag for fs-tree records — encodes the APFS
/// record-ordering rules: ascending oid first, then ascending kind,
/// then a type-specific tail. We pre-compute a comparator key per
/// record so the final sort is straightforward.
fn fs_record_sort_key(rec: &FsRecord) -> (u64, u8, Vec<u8>) {
    let hdr = u64::from_le_bytes(rec.key[0..8].try_into().unwrap());
    let oid = hdr & OBJ_ID_MASK;
    let kind = (hdr >> OBJ_TYPE_SHIFT) as u8;
    let tail = if rec.key.len() > 8 {
        rec.key[8..].to_vec()
    } else {
        Vec::new()
    };
    (oid, kind, tail)
}

/// A streaming APFS writer. Construct one with [`ApfsWriter::new`],
/// build the filesystem tree with [`ApfsWriter::add_file_from_reader`],
/// [`ApfsWriter::add_dir`], [`ApfsWriter::add_symlink`], and
/// [`ApfsWriter::add_xattr`], then call [`ApfsWriter::finish`] to
/// materialise the on-disk image.
///
/// The writer owns a `&mut BlockDevice`; the device must already be
/// sized large enough (zero-extended is fine — we'll overwrite the
/// header and any blocks we use).
pub struct ApfsWriter<'a> {
    dev: &'a mut dyn BlockDevice,
    block_size: u32,
    total_blocks: u64,
    /// Index of the next free metadata/data block (bump-pointer allocated
    /// from `bump_block_start`). Metadata extents are reserved before
    /// data extents inside `finish()` by pre-incrementing this counter
    /// for each fs-tree / omap leaf we need.
    next_block: u64,
    /// First block reserved for bump allocation (metadata leaves + file
    /// extents). The fixed-position blocks (NXSB copies, chkmap,
    /// spaceman, container/volume omap headers, APSB) all live below
    /// this.
    bump_block_start: u64,
    /// Volume name, written into the APSB.
    volume_name: String,
    /// Container UUID (random-ish — derived from the volume name).
    container_uuid: [u8; 16],
    /// Volume UUID (derived from volume name + a salt).
    volume_uuid: [u8; 16],
    /// Pending fs-tree records, sorted on `finish()`.
    records: Vec<FsRecord>,
    /// Next free inode object id. Inode 2 is the root directory.
    next_oid: u64,
    /// Number of files added (for APSB stats).
    num_files: u64,
    /// Number of directories added beyond the root (for APSB stats).
    num_directories: u64,
    /// Number of symlinks added (for APSB stats).
    num_symlinks: u64,
    /// True once `finish()` has been called.
    finished: bool,
    /// XID stamped on every emitted block. Defaults to [`WRITE_XID`] for
    /// fresh `format`-driven writers; the reopen-and-write path bumps
    /// this for each subsequent checkpoint via [`ApfsWriter::new_checkpoint`].
    xid: u64,
    /// Block address where the new NXSB will be written. The format
    /// path uses [`NXSB_LIVE_PADDR`]; reopen-and-write picks the next
    /// free slot in the xp_desc area.
    nxsb_paddr: u64,
    /// True when the label NXSB at block 0 should be refreshed (only on
    /// format).
    write_label_nxsb: bool,
}

// ───────────────────────── shared record builders ─────────────────────────
//
// Free `pub(crate)` functions that build the on-disk (key, value) byte
// pairs for every fs-tree record kind we emit. Both the single-pass
// [`ApfsWriter`] (format / pending-write path) and the in-place mutators
// in [`super::Apfs`] (Write-state checkpoint COWs via
// [`super::rw::commit_with_mutator`]) push records through the same
// builders, so the byte layouts only need to live in one place.

/// Build the `j_drec_key_t` (plain) or `j_drec_hashed_key_t` (hashed)
/// key + `j_drec_val_t` value bytes for a directory entry. The
/// layout follows the volume's `apfs_incompatible_features` flags:
///
/// - `Plain` (the default for volumes our own writer formats):
///   `(hdr:u64, name_len:u16, name + NUL)`. Sort order in the
///   fs-tree is by (oid, kind, name).
/// - `Hashed` (set by `APFS_INCOMPAT_NORMALIZATION_INSENSITIVE`,
///   the macOS default for user data volumes):
///   `(hdr:u64, name_len_and_hash:u32 LE, name + NUL)`. Sort order
///   is by (oid, kind, hash, name). The hash is computed by
///   [`apfs_drec_name_len_and_hash`] from the NFD-normalised
///   (optionally case-folded per `case_fold`) name.
///
/// Value bytes are layout-independent.
///
/// Errors when the name is too long to encode in the 16-bit or
/// 10-bit `name_len` field, depending on layout — POSIX caps at
/// 255 bytes anyway, but the type allows more.
pub(crate) fn build_drec_record(
    parent_oid: u64,
    name: &str,
    target_oid: u64,
    dtype: u16,
    layout: super::fstree::DrecKeyLayout,
    case_fold: bool,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let nlen = name.len() + 1; // includes trailing NUL
    let mut key;
    let hdr = ((APFS_TYPE_DIR_REC as u64) << OBJ_TYPE_SHIFT) | (parent_oid & OBJ_ID_MASK);
    match layout {
        super::fstree::DrecKeyLayout::Plain => {
            if nlen > u16::MAX as usize {
                return Err(crate::Error::InvalidArgument(
                    "apfs writer: directory entry name too long".into(),
                ));
            }
            key = Vec::with_capacity(10 + nlen);
            key.extend_from_slice(&hdr.to_le_bytes());
            key.extend_from_slice(&(nlen as u16).to_le_bytes());
        }
        super::fstree::DrecKeyLayout::Hashed => {
            // Hashed layout packs (hash:22, name_len:10) into a u32.
            if nlen > 0x3FF {
                return Err(crate::Error::InvalidArgument(
                    "apfs writer: hashed-key directory entry name too long".into(),
                ));
            }
            key = Vec::with_capacity(12 + nlen);
            key.extend_from_slice(&hdr.to_le_bytes());
            let packed = apfs_drec_name_len_and_hash(name, case_fold);
            key.extend_from_slice(&packed.to_le_bytes());
        }
    }
    key.extend_from_slice(name.as_bytes());
    key.push(0);

    // j_drec_val_t: file_id u64 + date_added u64 (zero) + flags u16
    let mut val = vec![0u8; 18];
    val[0..8].copy_from_slice(&target_oid.to_le_bytes());
    val[16..18].copy_from_slice(&dtype.to_le_bytes());
    Ok((key, val))
}

/// Pack a drec's name length and 22-bit hash into the `u32 LE` that
/// lives in `j_drec_hashed_key_t.name_len_and_hash`. The hash is
/// CRC32C over the NFD-decomposed (and optionally case-folded) name
/// encoded as UTF-32 LE; low 10 bits of the result are the raw name
/// length **including** the trailing NUL.
///
/// Best-effort match for Apple's kernel hash: ASCII and common BMP
/// names match byte-for-byte. Uncommon combining sequences and
/// supplementary-plane characters may differ because Apple's
/// `UnicodeTables_v10.h` carries documented deviations (4-char
/// decomposition cap, iota-subscript CCC override, restricted
/// codepoint domain) we don't reproduce. If a name's hash diverges
/// from the kernel's, the new drec sorts into the wrong B-tree
/// bucket and the kernel won't find it on the next mount — same
/// failure mode as an off-by-one in our impl.
pub(crate) fn apfs_drec_name_len_and_hash(name: &str, case_fold: bool) -> u32 {
    use caseless::Caseless;
    use unicode_normalization::UnicodeNormalization;
    let chars: Vec<u32> = if case_fold {
        name.nfd().default_case_fold().map(|c| c as u32).collect()
    } else {
        name.nfd().map(|c| c as u32).collect()
    };
    let bytes: Vec<u8> = chars.iter().flat_map(|c| c.to_le_bytes()).collect();
    let crc = crc32c::crc32c(&bytes);
    let name_len = (name.len() + 1) as u32; // include trailing NUL
    ((crc & 0x003F_FFFF) << 10) | (name_len & 0x0000_03FF)
}

/// Build the `j_inode_key_t` + `j_inode_val_t` record bytes. For
/// regular files / symlinks (`dstream_size > 0` OR mode is `S_IFREG` /
/// `S_IFLNK`) an `INO_EXT_TYPE_DSTREAM` xfield carrying `size` and
/// `alloced_size` (block-rounded) is appended. Directory inodes get
/// no xfield. `nlink` is seeded as 2 for directories and 1 for
/// everything else, matching the single-pass writer's convention.
///
/// `block_size` is needed to round `alloced_size` to a whole-block
/// boundary inside the DSTREAM xfield value.
pub(crate) fn build_inode_record(
    oid: u64,
    parent_oid: u64,
    mode: u16,
    dstream_size: u64,
    block_size: u32,
    mtime_ns: u64,
) -> (Vec<u8>, Vec<u8>) {
    let has_dstream =
        dstream_size > 0 || (mode & 0o170_000 == 0o100_000) || (mode & 0o170_000 == 0o120_000);
    let mut val = vec![0u8; J_INODE_VAL_FIXED_SIZE];
    val[0..8].copy_from_slice(&parent_oid.to_le_bytes());
    let private_id = if has_dstream && DSTREAM_ID_SHARES_INODE {
        oid
    } else {
        0
    };
    val[8..16].copy_from_slice(&private_id.to_le_bytes());
    // Stamp all four time fields with `mtime_ns`. APFS records
    // create_time/mod_time/change_time/access_time as four separate
    // u64-ns slots; we collapse them to a single value because most
    // callers only have an mtime to give. Per-field timestamps can
    // be refined later via Apfs::set_times / Filesystem::set_attrs.
    val[16..24].copy_from_slice(&mtime_ns.to_le_bytes()); // create_time
    val[24..32].copy_from_slice(&mtime_ns.to_le_bytes()); // mod_time
    val[32..40].copy_from_slice(&mtime_ns.to_le_bytes()); // change_time
    val[40..48].copy_from_slice(&mtime_ns.to_le_bytes()); // access_time
    // internal_flags: 0.
    let nlink: i32 = if mode & 0o170_000 == 0o040_000 { 2 } else { 1 };
    val[56..60].copy_from_slice(&nlink.to_le_bytes());
    // owner/group: 0 (root).
    val[80..82].copy_from_slice(&mode.to_le_bytes());
    val[84..92].copy_from_slice(&dstream_size.to_le_bytes());

    if has_dstream {
        // xfield blob: 1 entry (DSTREAM), 40 bytes value, padded to 8.
        let mut xfields = Vec::new();
        xfields.extend_from_slice(&1u16.to_le_bytes()); // num_exts
        xfields.extend_from_slice(&40u16.to_le_bytes()); // used_data
        // x_field_t = (type:u8, flags:u8, size:u16)
        xfields.push(INO_EXT_TYPE_DSTREAM);
        xfields.push(0);
        xfields.extend_from_slice(&40u16.to_le_bytes());
        // value: j_dstream_t (40 bytes)
        let mut ds = [0u8; 40];
        ds[0..8].copy_from_slice(&dstream_size.to_le_bytes());
        let bs = block_size as u64;
        let alloc = dstream_size.div_ceil(bs) * bs;
        ds[8..16].copy_from_slice(&alloc.to_le_bytes());
        xfields.extend_from_slice(&ds);
        val.extend_from_slice(&xfields);
    }

    let mut key = vec![0u8; 8];
    let hdr = ((APFS_TYPE_INODE as u64) << OBJ_TYPE_SHIFT) | (oid & OBJ_ID_MASK);
    key.copy_from_slice(&hdr.to_le_bytes());
    (key, val)
}

/// Build the FILE_EXTENT + DSTREAM_ID record pair for a single extent
/// of `dstream_oid`. Always returns two records — the FILE_EXTENT
/// carrying `(logical_addr, length, phys_block)` and the DSTREAM_ID
/// counter the reader uses to locate the dstream's extents under this
/// oid.
pub(crate) fn build_file_extent_records(
    dstream_oid: u64,
    logical_addr: u64,
    length: u64,
    phys_block: u64,
    block_size: u32,
) -> [(Vec<u8>, Vec<u8>); 2] {
    // FILE_EXTENT
    let mut fe_key = vec![0u8; 16];
    let fe_hdr = ((APFS_TYPE_FILE_EXTENT as u64) << OBJ_TYPE_SHIFT) | (dstream_oid & OBJ_ID_MASK);
    fe_key[0..8].copy_from_slice(&fe_hdr.to_le_bytes());
    fe_key[8..16].copy_from_slice(&logical_addr.to_le_bytes());
    let mut fe_val = vec![0u8; 24];
    let bs = block_size as u64;
    let alloc = length.div_ceil(bs) * bs;
    fe_val[0..8].copy_from_slice(&alloc.to_le_bytes()); // length (low 56 bits)
    fe_val[8..16].copy_from_slice(&phys_block.to_le_bytes());
    // crypto_id stays zero.

    // DSTREAM_ID (just a refcnt=0 marker; reader uses presence not value)
    let mut ds_key = vec![0u8; 8];
    let ds_hdr = ((APFS_TYPE_DSTREAM_ID as u64) << OBJ_TYPE_SHIFT) | (dstream_oid & OBJ_ID_MASK);
    ds_key.copy_from_slice(&ds_hdr.to_le_bytes());
    let ds_val = vec![0u8; 4];
    [(fe_key, fe_val), (ds_key, ds_val)]
}

/// Build an XATTR record for `(parent_oid, name, value)`. Embedded
/// values only — values larger than [`APFS_XATTR_MAX_EMBEDDED_SIZE`]
/// (3804 bytes) are rejected (dstream-backed xattrs are not yet
/// supported). Errors on a name that doesn't fit in a 16-bit
/// `name_len` field.
pub(crate) fn build_xattr_record(
    parent_oid: u64,
    name: &str,
    value: &[u8],
) -> Result<(Vec<u8>, Vec<u8>)> {
    if value.len() > APFS_XATTR_MAX_EMBEDDED_SIZE {
        return Err(crate::Error::Unsupported(format!(
            "apfs writer: xattr value of {} bytes exceeds embedded limit ({}); \
             dstream xattrs are not supported",
            value.len(),
            APFS_XATTR_MAX_EMBEDDED_SIZE
        )));
    }
    let name_bytes = name.as_bytes();
    let nlen = name_bytes.len() + 1; // includes trailing NUL
    if nlen > u16::MAX as usize {
        return Err(crate::Error::InvalidArgument(
            "apfs writer: xattr name too long".into(),
        ));
    }
    // Key: j_xattr_key_t
    let mut key = Vec::with_capacity(10 + nlen);
    let hdr = ((APFS_TYPE_XATTR as u64) << OBJ_TYPE_SHIFT) | (parent_oid & OBJ_ID_MASK);
    key.extend_from_slice(&hdr.to_le_bytes());
    key.extend_from_slice(&(nlen as u16).to_le_bytes());
    key.extend_from_slice(name_bytes);
    key.push(0);
    // Value: j_xattr_val_t
    let mut val = Vec::with_capacity(4 + value.len());
    val.extend_from_slice(&XATTR_DATA_EMBEDDED.to_le_bytes());
    val.extend_from_slice(&(value.len() as u16).to_le_bytes());
    val.extend_from_slice(value);
    Ok((key, val))
}

impl<'a> ApfsWriter<'a> {
    /// Create a writer over `dev`. `total_blocks * block_size` must fit
    /// inside `dev.total_size()`. `block_size` must be a power of two
    /// between 512 and 65 536; APFS in the wild is always 4096.
    pub fn new(
        dev: &'a mut dyn BlockDevice,
        total_blocks: u64,
        block_size: u32,
        volume_name: &str,
    ) -> Result<Self> {
        if !(512..=65_536).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(crate::Error::InvalidArgument(format!(
                "apfs writer: block_size {block_size} is not a sensible power of two"
            )));
        }
        let needed = total_blocks.checked_mul(block_size as u64).ok_or_else(|| {
            crate::Error::InvalidArgument("apfs writer: total_blocks * block_size overflows".into())
        })?;
        if needed > dev.total_size() {
            return Err(crate::Error::InvalidArgument(format!(
                "apfs writer: image needs {} bytes but dev is {} bytes",
                needed,
                dev.total_size()
            )));
        }
        // The minimum (~32 blocks) accommodates the expanded
        // xp_desc area plus the fixed-prefix metadata blocks plus
        // enough bump room for a one-record fs-tree.
        if total_blocks < 64 {
            return Err(crate::Error::InvalidArgument(
                "apfs writer: need at least 64 blocks".into(),
            ));
        }
        let container_uuid = derive_uuid(volume_name.as_bytes(), b"container");
        let volume_uuid = derive_uuid(volume_name.as_bytes(), b"volume");

        // See module-level layout. Bump area begins at BUMP_BLOCK_START
        // (block 21 at the default xp_desc sizing).
        let bump_block_start: u64 = BUMP_BLOCK_START;
        let mut w = Self {
            dev,
            block_size,
            total_blocks,
            next_block: bump_block_start,
            bump_block_start,
            volume_name: volume_name.to_string(),
            container_uuid,
            volume_uuid,
            records: Vec::new(),
            next_oid: 16, // inode 2 is root; we hand out 16+ for new inodes
            num_files: 0,
            num_directories: 0,
            num_symlinks: 0,
            finished: false,
            xid: WRITE_XID,
            nxsb_paddr: NXSB_LIVE_PADDR,
            write_label_nxsb: true,
        };
        // Seed the root inode record (oid = 2). Times stay at 0
        // (epoch 1970) at format — callers refine via set_times /
        // set_attrs.
        w.add_inode_record(2, 0, mode_dir(0o755), 0, 0)?;
        Ok(w)
    }

    /// Construct a "checkpoint refresh" writer that emits a brand-new
    /// NXSB + fs-tree + omap snapshot of the volume's state into fresh
    /// blocks past `bump_block_start` and a new xp_desc slot at
    /// `nxsb_paddr`. Used by the reopen-and-write path
    /// ([`crate::fs::apfs::Apfs::open_writable`]).
    ///
    /// Unlike [`ApfsWriter::new`], this constructor does not seed any
    /// records — the caller is expected to push the full record set
    /// gathered from the existing fs-tree before calling
    /// [`ApfsWriter::finish`]. The label NXSB at block 0 is left
    /// untouched (only the new xp_desc slot is written), so a crash
    /// part-way through a checkpoint leaves the previous valid
    /// checkpoint discoverable.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_checkpoint(
        dev: &'a mut dyn BlockDevice,
        total_blocks: u64,
        block_size: u32,
        volume_name: &str,
        container_uuid: [u8; 16],
        volume_uuid: [u8; 16],
        xid: u64,
        nxsb_paddr: u64,
        bump_block_start: u64,
        next_oid: u64,
        num_files: u64,
        num_directories: u64,
        num_symlinks: u64,
    ) -> Result<Self> {
        if !(512..=65_536).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(crate::Error::InvalidArgument(format!(
                "apfs writer: block_size {block_size} is not a sensible power of two"
            )));
        }
        if bump_block_start < BUMP_BLOCK_START || bump_block_start >= total_blocks {
            return Err(crate::Error::InvalidArgument(format!(
                "apfs writer: bump_block_start {bump_block_start} out of range \
                 [{BUMP_BLOCK_START}, {total_blocks})"
            )));
        }
        // Valid NXSB slot range: [NXSB_LIVE_PADDR, XP_DESC_BLOCKS+1).
        // NXSB_LIVE_PADDR (= 2) is the format-time slot, reusable
        // once the xp_desc ring wraps around. Slot 0 (block 1 =
        // CHKMAP_PADDR) is reserved for the chkmap and out of the
        // range. XP_DESC_BLOCKS = 16, base = 1, so the last valid
        // paddr is 16.
        if nxsb_paddr < NXSB_LIVE_PADDR || nxsb_paddr >= (XP_DESC_BLOCKS as u64 + 1) {
            return Err(crate::Error::InvalidArgument(format!(
                "apfs writer: nxsb_paddr {nxsb_paddr} not a valid xp_desc slot \
                 (expected {} ≤ slot < {})",
                NXSB_LIVE_PADDR,
                XP_DESC_BLOCKS as u64 + 1
            )));
        }
        Ok(Self {
            dev,
            block_size,
            total_blocks,
            next_block: bump_block_start,
            bump_block_start,
            volume_name: volume_name.to_string(),
            container_uuid,
            volume_uuid,
            records: Vec::new(),
            next_oid,
            num_files,
            num_directories,
            num_symlinks,
            finished: false,
            xid,
            nxsb_paddr,
            write_label_nxsb: false,
        })
    }

    /// Push a pre-built fs-tree record into the writer's record buffer.
    /// Used by the reopen-and-write path to inject the records dumped
    /// from the existing fs-tree.
    pub(crate) fn push_raw_record(&mut self, key: Vec<u8>, val: Vec<u8>) {
        self.records.push(FsRecord { key, val });
    }

    /// Add an empty subdirectory under `parent_oid`. Returns the new
    /// directory's object id (use it as `parent_oid` for nested
    /// children).
    pub fn add_dir(&mut self, parent_oid: u64, name: &str, mode: u16) -> Result<u64> {
        self.add_dir_at_time(parent_oid, name, mode, 0)
    }

    /// Like [`Self::add_dir`] but stamps `mtime_ns` (UNIX ns since
    /// epoch) into the inode's four time fields. Used by callers
    /// that carry user-supplied timestamps through to the writer
    /// (the [`crate::fs::Filesystem`] adapter, mainly).
    pub fn add_dir_at_time(
        &mut self,
        parent_oid: u64,
        name: &str,
        mode: u16,
        mtime_ns: u64,
    ) -> Result<u64> {
        let oid = self.alloc_oid();
        self.add_drec(parent_oid, name, oid, DT_DIR)?;
        self.add_inode_record(oid, parent_oid, mode_dir(mode), 0, mtime_ns)?;
        self.num_directories += 1;
        Ok(oid)
    }

    /// Add a symlink under `parent_oid`. The link target is stored
    /// inline in the inode's name xfield-style data — we use a simple
    /// regular-file extent containing the target string.
    ///
    /// On real APFS, symlink targets live in an xattr named
    /// `com.apple.fs.symlink`; for v1 we use a regular file extent
    /// because it's easier to round-trip and our reader treats it the
    /// same way (size + extents). The DT_LNK type bit is set so
    /// directory listings still report it as a symlink.
    pub fn add_symlink(
        &mut self,
        parent_oid: u64,
        name: &str,
        mode: u16,
        target: &str,
    ) -> Result<u64> {
        self.add_symlink_at_time(parent_oid, name, mode, target, 0)
    }

    /// Like [`Self::add_symlink`] but stamps `mtime_ns` into the
    /// inode's time fields.
    pub fn add_symlink_at_time(
        &mut self,
        parent_oid: u64,
        name: &str,
        mode: u16,
        target: &str,
        mtime_ns: u64,
    ) -> Result<u64> {
        let oid = self.alloc_oid();
        self.add_drec(parent_oid, name, oid, DT_LNK)?;
        let target_bytes = target.as_bytes();
        let extent_paddr = self.allocate_extent_for_size(target_bytes.len() as u64)?;
        // Copy the target string into the allocated extent (streaming
        // not needed at this size, but use the same path for parity).
        let extent_len_blocks = self.bytes_to_blocks(target_bytes.len() as u64);
        let mut block = vec![0u8; self.block_size as usize];
        for i in 0..extent_len_blocks {
            let off = (i as usize) * self.block_size as usize;
            let end = (off + self.block_size as usize).min(target_bytes.len());
            if off < target_bytes.len() {
                block.fill(0);
                let chunk = &target_bytes[off..end];
                block[..chunk.len()].copy_from_slice(chunk);
                self.write_block(extent_paddr + i, &block)?;
            }
        }
        self.add_file_extent(oid, 0, target_bytes.len() as u64, extent_paddr)?;
        self.add_inode_record(
            oid,
            parent_oid,
            mode_lnk(mode),
            target_bytes.len() as u64,
            mtime_ns,
        )?;
        self.num_symlinks += 1;
        Ok(oid)
    }

    /// Add a regular file under `parent_oid`. Bytes are streamed from
    /// `reader` through a 64 KiB scratch buffer into a freshly
    /// allocated single extent.
    ///
    /// `size` MUST match the byte count produced by `reader`. We trust
    /// the caller — if they overshoot or undershoot, the on-disk
    /// inode's `j_dstream.size` may not agree with the actual extent.
    pub fn add_file_from_reader<R: Read>(
        &mut self,
        parent_oid: u64,
        name: &str,
        mode: u16,
        reader: &mut R,
        size: u64,
    ) -> Result<u64> {
        self.add_file_from_reader_at_time(parent_oid, name, mode, reader, size, 0)
    }

    /// Like [`Self::add_file_from_reader`] but stamps `mtime_ns`
    /// into the inode's time fields.
    pub fn add_file_from_reader_at_time<R: Read>(
        &mut self,
        parent_oid: u64,
        name: &str,
        mode: u16,
        reader: &mut R,
        size: u64,
        mtime_ns: u64,
    ) -> Result<u64> {
        let oid = self.alloc_oid();
        self.add_drec(parent_oid, name, oid, DT_REG)?;

        if size == 0 {
            // No extent records — empty file.
            self.add_inode_record(oid, parent_oid, mode_reg(mode), 0, mtime_ns)?;
            self.num_files += 1;
            return Ok(oid);
        }

        let extent_paddr = self.allocate_extent_for_size(size)?;
        let blocks = self.bytes_to_blocks(size);

        // Stream the file into the allocated extent, block at a time.
        let mut block_buf = vec![0u8; self.block_size as usize];
        let mut scratch = vec![0u8; COPY_BUF];
        let mut bytes_left = size;
        let mut block_idx: u64 = 0;
        let mut block_off: usize = 0;
        block_buf.fill(0);
        while bytes_left > 0 {
            let want = scratch.len().min(bytes_left as usize);
            // Fully fill `scratch[..want]` even if reader returns short.
            let mut got = 0;
            while got < want {
                let n = reader
                    .read(&mut scratch[got..want])
                    .map_err(crate::Error::Io)?;
                if n == 0 {
                    break;
                }
                got += n;
            }
            if got == 0 {
                // Reader truncated — pad remaining file with zeros so
                // the on-disk size matches `size`.
                break;
            }
            // Copy `got` bytes into the per-block buffer, flushing when
            // it fills.
            let mut consumed = 0;
            while consumed < got {
                let space = self.block_size as usize - block_off;
                let n = space.min(got - consumed);
                block_buf[block_off..block_off + n]
                    .copy_from_slice(&scratch[consumed..consumed + n]);
                block_off += n;
                consumed += n;
                bytes_left -= n as u64;
                if block_off == self.block_size as usize {
                    self.write_block(extent_paddr + block_idx, &block_buf)?;
                    block_idx += 1;
                    block_off = 0;
                    block_buf.fill(0);
                }
            }
        }
        // Flush the partial trailing block.
        if block_off > 0 {
            // Zero-pad the rest of the buffer (already zeroed since we
            // call fill(0) each cycle, but be explicit).
            for b in &mut block_buf[block_off..] {
                *b = 0;
            }
            self.write_block(extent_paddr + block_idx, &block_buf)?;
            block_idx += 1;
        }
        // If we still owe more blocks (reader truncated), zero them.
        while block_idx < blocks {
            block_buf.fill(0);
            self.write_block(extent_paddr + block_idx, &block_buf)?;
            block_idx += 1;
        }

        self.add_file_extent(oid, 0, size, extent_paddr)?;
        self.add_inode_record(oid, parent_oid, mode_reg(mode), size, mtime_ns)?;
        self.num_files += 1;
        Ok(oid)
    }

    /// Add an extended attribute under `parent_oid`. The name follows
    /// APFS conventions (caller picks the namespace prefix, e.g.
    /// `"user.note"` or `"com.apple.fs.symlink"`).
    ///
    /// Values up to [`APFS_XATTR_MAX_EMBEDDED_SIZE`] (3804 bytes) are
    /// stored inline in the `j_xattr_val_t`. Larger values would need a
    /// `j_xattr_dstream_t` referencing a separate dstream and are
    /// currently rejected with [`crate::Error::Unsupported`].
    ///
    /// The xattr appears in the fs-tree under
    /// `(parent_oid, APFS_TYPE_XATTR, name)` and is visible via
    /// [`super::Apfs::read_xattrs`].
    pub fn add_xattr(&mut self, parent_oid: u64, name: &str, value: &[u8]) -> Result<()> {
        let (key, val) = build_xattr_record(parent_oid, name, value)?;
        self.records.push(FsRecord { key, val });
        Ok(())
    }

    /// Materialise the on-disk image. After this call the writer is
    /// drained; further `add_*` calls return `Unsupported`.
    pub fn finish(mut self) -> Result<()> {
        if self.finished {
            return Err(crate::Error::Unsupported(
                "apfs writer: finish() called twice".into(),
            ));
        }
        self.finished = true;

        // Sort records into APFS canonical order.
        self.records.sort_by(|a, b| {
            let ka = fs_record_sort_key(a);
            let kb = fs_record_sort_key(b);
            ka.cmp(&kb)
        });

        let bs = self.block_size as usize;
        let cur_xid = self.xid;

        // ---- Block addresses (see module-level layout) ----
        // On format (write_label_nxsb = true) we use the fixed layout.
        // On a checkpoint refresh we allocate cont_omap / APSB / vol_omap
        // out of the bump area so the previous checkpoint's blocks remain
        // intact (COW). The spaceman + chkmap aren't consulted by the
        // reader on open, so we keep pointing the new NXSB at the same
        // spaceman_paddr (and chkmap_paddr) — refreshing them would
        // require allocating new xp_desc slots beyond what this v1
        // implementation supports.
        let nxsb_label_paddr: u64 = 0;
        let chkmap_paddr: u64 = CHKMAP_PADDR;
        let nxsb_live_paddr: u64 = self.nxsb_paddr;
        let spaceman_paddr: u64 = SPACEMAN_PADDR;
        let cont_omap_paddr: u64 = if self.write_label_nxsb {
            CONT_OMAP_PADDR
        } else {
            self.alloc_block()?
        };
        let apsb_paddr: u64 = if self.write_label_nxsb {
            APSB_PADDR
        } else {
            self.alloc_block()?
        };
        let vol_omap_paddr: u64 = if self.write_label_nxsb {
            VOL_OMAP_PADDR
        } else {
            self.alloc_block()?
        };

        // Virtual oids assigned to: container volume (=1024), spaceman
        // (=512), volume omap target -> fsroot (=fsroot_vid=2). The
        // container omap maps volume_vid → APSB paddr; the volume omap
        // maps fsroot_vid → fsroot paddr.
        let volume_vid: u64 = 1024;
        let spaceman_vid: u64 = 512;
        let reaper_vid: u64 = 513;
        let fsroot_vid: u64 = 2;

        // ---- Plan fs-tree (single leaf or multi-leaf with internal root) ----
        let leaf_payload_cap = leaf_payload_capacity(bs);
        // Each leaf has a per-entry ToC overhead of 8 bytes (kvloc_t) plus
        // the key+val payload. We pack greedily by sort order, ensuring
        // each leaf fits inside `leaf_payload_cap`.
        let leaves = pack_records_into_leaves(&self.records, leaf_payload_cap)?;

        // Reserve metadata block addresses for fs-tree leaves (and the
        // internal root if needed) BEFORE we allocate any file-extent
        // data blocks. Otherwise extent blocks would land before the
        // metadata, which is fine functionally but harder to reason
        // about. NOTE: per-spec layout, all extent allocations done by
        // add_file_from_reader / add_symlink already happened during
        // record building, so by the time we get here `next_block` has
        // grown past the data. We allocate fresh metadata blocks
        // AFTER data — that's still legal because we record their
        // actual paddrs into the omap.
        let fs_leaf_paddrs: Vec<u64> = (0..leaves.len())
            .map(|_| self.alloc_block())
            .collect::<Result<Vec<_>>>()?;
        // Compute the fs-tree root paddr & vid:
        //   single leaf  → root *is* the leaf (vid = fsroot_vid).
        //   multi leaf   → root is an internal node we add on top;
        //                  fsroot_vid maps to its paddr.
        let (fsroot_paddr, vol_omap_entries) = if leaves.len() <= 1 {
            // Single root-leaf at fsroot_vid.
            let leaf_paddr = fs_leaf_paddrs.first().copied().unwrap_or(0);
            // If `leaves` is empty we'd never get here (we always have
            // at least one record), but be defensive.
            (leaf_paddr, vec![(fsroot_vid, cur_xid, leaf_paddr)])
        } else {
            // Multi-leaf: assign vids to each leaf and add an internal
            // root above them. The internal root itself gets fsroot_vid.
            let mut entries = Vec::with_capacity(leaves.len() + 1);
            let mut child_vids = Vec::with_capacity(leaves.len());
            for (i, &paddr) in fs_leaf_paddrs.iter().enumerate() {
                let vid = FS_LEAF_VID_BASE + i as u64;
                entries.push((vid, cur_xid, paddr));
                child_vids.push(vid);
            }
            let root_paddr = self.alloc_block()?;
            entries.push((fsroot_vid, cur_xid, root_paddr));
            entries.sort_by_key(|e| e.0);
            (root_paddr, entries)
        };

        // ---- Write fs-tree leaves ----
        for (i, leaf_records) in leaves.iter().enumerate() {
            let is_root = leaves.len() == 1;
            let vid = if is_root {
                fsroot_vid
            } else {
                FS_LEAF_VID_BASE + i as u64
            };
            let leaf_block = build_fs_leaf(leaf_records, bs, vid, is_root, cur_xid)?;
            self.write_block(fs_leaf_paddrs[i], &leaf_block)?;
        }
        if leaves.len() > 1 {
            // Build internal root keyed by first record of each leaf,
            // values = leaf vids.
            let mut sep_entries: Vec<(Vec<u8>, u64)> = Vec::with_capacity(leaves.len());
            for (i, leaf) in leaves.iter().enumerate() {
                let sep_key = leaf[0].key.clone();
                let vid = FS_LEAF_VID_BASE + i as u64;
                sep_entries.push((sep_key, vid));
            }
            let internal_block = build_fs_internal_root(&sep_entries, bs, fsroot_vid, cur_xid)?;
            self.write_block(fsroot_paddr, &internal_block)?;
        }

        // ---- Volume omap (single- or multi-leaf) ----
        // vol_omap_entries is sorted by oid above.
        let vol_omap_root_paddr = self.write_omap_tree(&vol_omap_entries)?;
        let vol_omap_phys = build_omap_phys(bs, vol_omap_paddr, vol_omap_root_paddr, cur_xid)?;
        self.write_block(vol_omap_paddr, &vol_omap_phys)?;

        // ---- APSB ----
        let apsb_block = self.build_apsb(bs, apsb_paddr, vol_omap_paddr, fsroot_vid)?;
        self.write_block(apsb_paddr, &apsb_block)?;

        // ---- Container omap (single entry, but goes through the same
        //      multi-leaf path so we exercise the writer uniformly) ----
        let cont_omap_root_paddr = self.write_omap_tree(&[(volume_vid, cur_xid, apsb_paddr)])?;
        let cont_omap_phys = build_omap_phys(bs, cont_omap_paddr, cont_omap_root_paddr, cur_xid)?;
        self.write_block(cont_omap_paddr, &cont_omap_phys)?;

        // ---- Spaceman: CIB + per-chunk bitmaps + spaceman_phys ----
        //
        // The CIB and bitmap blocks are bump-allocated AFTER every other
        // metadata/data block so they themselves can be marked "used"
        // in their own bitmap without circular reasoning. Pre-reserve
        // their addresses, then build the spaceman over the closed
        // set of allocations.
        //
        // We additionally reserve:
        //   * a contiguous IP ring (DEFAULT_IP_BLOCK_COUNT blocks) for
        //     ephemeral metadata; we never allocate out of it during
        //     formatting,
        //   * a single IP bitmap block at the start of the ring,
        //   * three blocks for the empty SFQ_IP / SFQ_MAIN / SFQ_TIER2
        //     free-queue B-tree roots.
        //
        // Every one of these blocks is bump-allocated here, so all of
        // them end up inside the `(0, self.next_block)` used-range and
        // the spaceman bitmap correctly marks them used.
        // The spaceman + chkmap are only rewritten on format. The reader
        // doesn't consult them during open, so leaving the previous
        // checkpoint's blocks intact keeps the previous checkpoint
        // bootable until the new one is signed.
        if self.write_label_nxsb {
            let bpc = blocks_per_chunk(bs);
            let chunks: u64 = self.total_blocks.div_ceil(bpc);

            // IP ring: ip_bm_size_in_blocks bitmap block(s) followed by
            // ip_block_count ring blocks. We use one bitmap block — at a
            // 4 KiB block size that bitmap can track 32 768 blocks, vastly
            // more than the ring needs.
            let ip_bm_size_in_blocks: u32 = 1;
            let ip_block_count = DEFAULT_IP_BLOCK_COUNT;
            let ip_bm_base = self.alloc_block()?;
            for _ in 1..ip_bm_size_in_blocks {
                // Future-proof: if we ever want a larger ring's bitmap.
                let _ = self.alloc_block()?;
            }
            let ip_base = self.alloc_block()?;
            for _ in 1..ip_block_count {
                let _ = self.alloc_block()?;
            }

            // Three empty SFQ B-tree roots.
            let free_queue_paddrs: [u64; 3] = [
                self.alloc_block()?,
                self.alloc_block()?,
                self.alloc_block()?,
            ];

            let cib_paddr = self.alloc_block()?;
            let mut bitmap_paddrs: Vec<u64> = Vec::with_capacity(chunks as usize);
            for _ in 0..chunks {
                bitmap_paddrs.push(self.alloc_block()?);
            }

            let used_ranges: Vec<(u64, u64)> = vec![(0, self.next_block)];
            let layout = SpacemanLayout {
                block_size: bs,
                total_blocks: self.total_blocks,
                xid: cur_xid,
                spaceman_oid: spaceman_vid,
                cib_paddr,
                bitmap_paddrs: bitmap_paddrs.clone(),
                used_ranges,
                ip_base,
                ip_block_count,
                ip_bm_base,
                ip_bm_size_in_blocks,
                free_queue_paddrs,
            };
            let emitted = spaceman::build_spaceman(&layout)?;
            self.write_block(spaceman_paddr, &emitted.spaceman_block)?;
            self.write_block(cib_paddr, &emitted.cib_block)?;
            for (paddr, bmap) in bitmap_paddrs.iter().zip(emitted.bitmap_blocks.iter()) {
                self.write_block(*paddr, bmap)?;
            }
            // IP bitmap block (no obj_phys_t header per spec).
            self.write_block(ip_bm_base, &emitted.ip_bm_block)?;
            // Three empty free-queue B-tree roots.
            for (paddr, fq_block) in free_queue_paddrs
                .iter()
                .zip(emitted.free_queue_blocks.iter())
            {
                self.write_block(*paddr, fq_block)?;
            }

            // ---- Checkpoint map: one entry resolving spaceman ephemeral oid
            //      to its physical block. xp_desc readers (incl. fsck_apfs)
            //      walk this to find the spaceman.
            let chkmap = build_chkmap(
                bs,
                chkmap_paddr,
                spaceman::OBJECT_TYPE_SPACEMAN | OBJ_EPHEMERAL,
                spaceman_vid,
                spaceman_paddr,
                cur_xid,
            )?;
            self.write_block(chkmap_paddr, &chkmap)?;
        }

        // ---- NXSB ----
        // Live (or new-checkpoint) NXSB goes into the current xp_desc
        // slot. The label copy at block 0 is only refreshed on format
        // — it acts as a stable "where to find the xp_desc area"
        // hint and must NOT be rewritten mid-checkpoint or readers may
        // see a torn label.
        let nxsb = self.build_nxsb(
            bs,
            nxsb_live_paddr,
            cont_omap_paddr,
            spaceman_vid,
            reaper_vid,
            volume_vid,
        )?;
        self.write_block(nxsb_live_paddr, &nxsb)?;
        if self.write_label_nxsb {
            let nxsb_label = self.build_nxsb(
                bs,
                nxsb_label_paddr,
                cont_omap_paddr,
                spaceman_vid,
                reaper_vid,
                volume_vid,
            )?;
            self.write_block(nxsb_label_paddr, &nxsb_label)?;
        }

        let _ = fsroot_paddr;
        Ok(())
    }

    // ---- internal helpers ----

    fn alloc_oid(&mut self) -> u64 {
        let o = self.next_oid;
        self.next_oid = self.next_oid.checked_add(1).unwrap_or(o);
        o
    }

    fn bytes_to_blocks(&self, n: u64) -> u64 {
        let bs = self.block_size as u64;
        n.div_ceil(bs).max(1)
    }

    /// Bump-allocate one block from the metadata/data area. Errors if
    /// the image would run out of room.
    fn alloc_block(&mut self) -> Result<u64> {
        if self.next_block >= self.total_blocks {
            return Err(crate::Error::InvalidArgument(format!(
                "apfs writer: image full at {} blocks",
                self.total_blocks
            )));
        }
        let p = self.next_block;
        self.next_block += 1;
        Ok(p)
    }

    fn allocate_extent_for_size(&mut self, size: u64) -> Result<u64> {
        if size == 0 {
            return Err(crate::Error::InvalidArgument(
                "apfs writer: cannot allocate a zero-length extent".into(),
            ));
        }
        let blocks = self.bytes_to_blocks(size);
        let start = self.next_block;
        let end = start
            .checked_add(blocks)
            .ok_or_else(|| crate::Error::InvalidArgument("apfs writer: extent oob".into()))?;
        if end > self.total_blocks {
            return Err(crate::Error::InvalidArgument(format!(
                "apfs writer: extent of {blocks} blocks past end of image"
            )));
        }
        self.next_block = end;
        Ok(start)
    }

    fn write_block(&mut self, paddr: u64, buf: &[u8]) -> Result<()> {
        let off = paddr.saturating_mul(self.block_size as u64);
        self.dev.write_at(off, buf)
    }

    fn add_drec(&mut self, parent_oid: u64, name: &str, target_oid: u64, dtype: u16) -> Result<()> {
        // Plain drec key — `ApfsWriter::new` always formats without
        // APFS_INCOMPAT_NORMALIZATION_INSENSITIVE, so the single-pass
        // writer's drecs stay plain-key. The Write-state mutators
        // (Apfs::create_*_at, rename, link) pass the volume's actual
        // layout instead.
        let (key, val) = build_drec_record(
            parent_oid,
            name,
            target_oid,
            dtype,
            super::fstree::DrecKeyLayout::Plain,
            false,
        )?;
        self.records.push(FsRecord { key, val });
        Ok(())
    }

    fn add_inode_record(
        &mut self,
        oid: u64,
        parent_oid: u64,
        mode: u16,
        dstream_size: u64,
        mtime_ns: u64,
    ) -> Result<()> {
        let (key, val) = build_inode_record(
            oid,
            parent_oid,
            mode,
            dstream_size,
            self.block_size,
            mtime_ns,
        );
        self.records.push(FsRecord { key, val });
        Ok(())
    }

    fn add_file_extent(
        &mut self,
        dstream_oid: u64,
        logical_addr: u64,
        length: u64,
        phys_block: u64,
    ) -> Result<()> {
        let pair = build_file_extent_records(
            dstream_oid,
            logical_addr,
            length,
            phys_block,
            self.block_size,
        );
        for (key, val) in pair {
            self.records.push(FsRecord { key, val });
        }
        Ok(())
    }

    /// Emit an omap as one or more leaf blocks plus, if necessary, a
    /// single internal-root block. Returns the paddr of the tree's
    /// root (leaf for a single-leaf tree, internal node otherwise).
    fn write_omap_tree(&mut self, entries: &[(u64, u64, u64)]) -> Result<u64> {
        let bs = self.block_size as usize;
        let xid = self.xid;
        let leaves = pack_omap_into_leaves(entries, omap_leaf_capacity(bs))?;
        if leaves.len() == 1 {
            // Single leaf: it's the root and carries the trailing
            // btree_info_t.
            let paddr = self.alloc_block()?;
            let block = build_omap_leaf_node(bs, &leaves[0], true, xid)?;
            self.write_block(paddr, &block)?;
            return Ok(paddr);
        }
        // Multi-leaf: emit each leaf at its own paddr, then a single
        // internal root with 16-byte keys and 8-byte child paddrs.
        let mut leaf_paddrs: Vec<u64> = Vec::with_capacity(leaves.len());
        let mut sep_entries: Vec<((u64, u64), u64)> = Vec::with_capacity(leaves.len());
        for chunk in &leaves {
            let paddr = self.alloc_block()?;
            let block = build_omap_leaf_node(bs, chunk, false, xid)?;
            self.write_block(paddr, &block)?;
            leaf_paddrs.push(paddr);
            sep_entries.push(((chunk[0].0, chunk[0].1), paddr));
        }
        let root_paddr = self.alloc_block()?;
        let root_block = build_omap_internal_root(bs, &sep_entries, xid)?;
        self.write_block(root_paddr, &root_block)?;
        Ok(root_paddr)
    }

    /// Build the APSB block at offset `apsb_paddr`.
    fn build_apsb(
        &self,
        bs: usize,
        apsb_paddr: u64,
        vol_omap_paddr: u64,
        fsroot_vid: u64,
    ) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; bs];
        // obj_phys
        buf[8..16].copy_from_slice(&apsb_paddr.to_le_bytes()); // oid (we'll override below)
        buf[16..24].copy_from_slice(&self.xid.to_le_bytes());
        // o_type = OBJECT_TYPE_FS | OBJ_VIRTUAL (default 0).
        buf[24..28].copy_from_slice(&OBJECT_TYPE_FS.to_le_bytes());

        buf[32..36].copy_from_slice(&APFS_MAGIC.to_le_bytes());
        buf[36..40].copy_from_slice(&0u32.to_le_bytes()); // fs_index
        // features / ro_compat / incompat: 0 (plain drec layout, no
        // sealed/encryption).
        // apfs_root_tree_type at offset 116
        buf[116..120].copy_from_slice(&(OBJECT_TYPE_BTREE).to_le_bytes());
        buf[120..124].copy_from_slice(&(OBJECT_TYPE_BTREE).to_le_bytes());
        buf[124..128].copy_from_slice(&(OBJECT_TYPE_BTREE).to_le_bytes());
        buf[128..136].copy_from_slice(&vol_omap_paddr.to_le_bytes()); // omap_oid
        buf[136..144].copy_from_slice(&fsroot_vid.to_le_bytes()); // root_tree_oid
        buf[144..152].copy_from_slice(&0u64.to_le_bytes()); // extentref_tree_oid
        buf[152..160].copy_from_slice(&0u64.to_le_bytes()); // snap_meta_tree_oid (none)
        buf[160..168].copy_from_slice(&0u64.to_le_bytes()); // revert_to_xid
        buf[168..176].copy_from_slice(&0u64.to_le_bytes()); // revert_to_sblock_oid
        buf[176..184].copy_from_slice(&self.next_oid.to_le_bytes()); // next_obj_id
        buf[184..192].copy_from_slice(&self.num_files.to_le_bytes());
        buf[192..200].copy_from_slice(&self.num_directories.to_le_bytes());
        buf[200..208].copy_from_slice(&self.num_symlinks.to_le_bytes());
        // num_other_fsobjects: 0
        // num_snapshots: 0
        // total_blocks_alloced / freed: 0
        buf[240..256].copy_from_slice(&self.volume_uuid);
        // last_mod_time, fs_flags
        const APFS_FS_UNENCRYPTED: u64 = 0x0000_0001;
        buf[264..272].copy_from_slice(&APFS_FS_UNENCRYPTED.to_le_bytes());
        // formatted_by / modified_by: zero
        // volname at offset 704 (256 bytes)
        let name_bytes = self.volume_name.as_bytes();
        let n = name_bytes.len().min(255);
        buf[704..704 + n].copy_from_slice(&name_bytes[..n]);
        buf[704 + n] = 0;
        // next_doc_id at 960, role at 964 — zero.

        // Final checksum.
        sign_block(&mut buf);
        Ok(buf)
    }

    /// Build the live NXSB at `paddr`.
    fn build_nxsb(
        &self,
        bs: usize,
        paddr: u64,
        cont_omap_paddr: u64,
        spaceman_vid: u64,
        reaper_vid: u64,
        volume_vid: u64,
    ) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; bs];
        // obj_phys
        buf[8..16].copy_from_slice(&paddr.to_le_bytes()); // oid (physical = block)
        buf[16..24].copy_from_slice(&self.xid.to_le_bytes());
        buf[24..28].copy_from_slice(&(OBJECT_TYPE_NX_SUPERBLOCK | OBJ_EPHEMERAL).to_le_bytes());

        buf[32..36].copy_from_slice(&NX_MAGIC.to_le_bytes());
        buf[36..40].copy_from_slice(&self.block_size.to_le_bytes());
        buf[40..48].copy_from_slice(&self.total_blocks.to_le_bytes());
        // features / ro_compat / incompat: 0 (we ship a vanilla container)
        buf[72..88].copy_from_slice(&self.container_uuid);
        buf[88..96].copy_from_slice(&(self.next_oid + 1024).to_le_bytes()); // next_oid
        buf[96..104].copy_from_slice(&(self.xid + 1).to_le_bytes()); // next_xid
        // xp_desc area: XP_DESC_BLOCKS blocks (chkmap stub at block 1 +
        // the live NXSB at block 2 + XP_DESC_BLOCKS-2 spare slots for
        // future checkpoint rotation via Apfs::open_writable).
        // xp_desc_base = 1; reader scans this range looking for the
        // largest-xid NXSB.
        buf[104..108].copy_from_slice(&XP_DESC_BLOCKS.to_le_bytes()); // xp_desc_blocks
        buf[108..112].copy_from_slice(&1u32.to_le_bytes()); // xp_data_blocks
        buf[112..120].copy_from_slice(&CHKMAP_PADDR.to_le_bytes()); // xp_desc_base
        buf[120..128].copy_from_slice(&SPACEMAN_PADDR.to_le_bytes()); // xp_data_base = spaceman_paddr
        // xp_desc_next = paddr+1 (next free xp_desc slot after this NXSB).
        // open_writable advances this on each checkpoint.
        buf[128..132].copy_from_slice(&((paddr + 1) as u32).to_le_bytes());
        buf[132..136].copy_from_slice(&1u32.to_le_bytes()); // xp_data_next
        buf[136..140].copy_from_slice(&0u32.to_le_bytes()); // xp_desc_index
        // xp_desc_len = number of slots used so far up to & including this
        // NXSB (chkmap @ 1 + every NXSB written so far).
        buf[140..144].copy_from_slice(&((paddr) as u32).to_le_bytes()); // xp_desc_len
        buf[144..148].copy_from_slice(&0u32.to_le_bytes()); // xp_data_index
        buf[148..152].copy_from_slice(&1u32.to_le_bytes()); // xp_data_len
        buf[152..160].copy_from_slice(&spaceman_vid.to_le_bytes());
        buf[160..168].copy_from_slice(&cont_omap_paddr.to_le_bytes()); // omap_oid
        buf[168..176].copy_from_slice(&reaper_vid.to_le_bytes()); // reaper_oid
        buf[176..180].copy_from_slice(&0u32.to_le_bytes()); // test_type
        buf[180..184].copy_from_slice(&(NX_MAX_FILE_SYSTEMS as u32).to_le_bytes());
        // fs_oid[0] = volume_vid
        buf[184..192].copy_from_slice(&volume_vid.to_le_bytes());
        // rest of fs_oid[] = 0

        // Silence unused-variable warning when the writer ends up not
        // needing to walk the bump pointer afterwards.
        let _ = self.bump_block_start;
        sign_block(&mut buf);
        Ok(buf)
    }
}

// ---- block builders (free fns) ----

/// Compute the Fletcher-64 checksum over the rest of `buf` and write it
/// into the first 8 bytes.
fn sign_block(buf: &mut [u8]) {
    let cksum = fletcher64(buf);
    buf[0..8].copy_from_slice(&cksum.to_le_bytes());
}

/// Build an omap_phys_t (header) block. `paddr` is the block this
/// header lives at; `tree_paddr` is the physical block of the tree's
/// root.
fn build_omap_phys(bs: usize, paddr: u64, tree_paddr: u64, xid: u64) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; bs];
    buf[8..16].copy_from_slice(&paddr.to_le_bytes()); // oid
    buf[16..24].copy_from_slice(&xid.to_le_bytes()); // xid
    buf[24..28].copy_from_slice(&(OBJECT_TYPE_OMAP | OBJ_PHYSICAL).to_le_bytes());
    // flags / snap_count / tree_type / snapshot_tree_type left zero
    // (snapshot_tree_type is u32 at offset 44).
    buf[40..44].copy_from_slice(&(OBJECT_TYPE_BTREE | OBJ_PHYSICAL).to_le_bytes()); // tree_type
    buf[48..56].copy_from_slice(&tree_paddr.to_le_bytes());
    // snapshot_tree_oid stays 0.
    sign_block(&mut buf);
    Ok(buf)
}

/// Effective payload capacity for an omap leaf node (root or otherwise).
/// `key_size + val_size = 32`; ToC entries cost 4 bytes each. We
/// conservatively use the root layout (subtract `btree_info_t`) for both
/// root and non-root leaves so the per-leaf record count stays
/// consistent.
fn omap_leaf_capacity(bs: usize) -> usize {
    // The packing math: each entry adds 4 (ToC) + 16 (key) + 16 (val) = 36
    // bytes. Available = bs - obj_header(56) - btree_info_t(40) for the
    // root case. For non-root leaves we'd have 40 more bytes, but we use
    // the conservative figure so the splitter doesn't need to know which
    // leaf is the root.
    bs - 56 - BTREE_INFO_SIZE
}

/// Pack an omap entry list (already sorted by `(oid, xid)`) into one or
/// more leaf groups. Each group fits inside `cap` bytes when accounting
/// for the 4-byte ToC entry + 16+16 byte fixed-KV payload.
fn pack_omap_into_leaves(
    entries: &[(u64, u64, u64)],
    cap: usize,
) -> Result<Vec<Vec<(u64, u64, u64)>>> {
    let per = 4 + 16 + 16;
    let max_per_leaf = cap / per;
    if max_per_leaf == 0 {
        return Err(crate::Error::Unsupported(
            "apfs writer: block too small to fit any omap entry".into(),
        ));
    }
    if entries.is_empty() {
        return Ok(vec![Vec::new()]);
    }
    // Verify ascending order so the split is meaningful.
    let mut out: Vec<Vec<(u64, u64, u64)>> = Vec::new();
    for chunk in entries.chunks(max_per_leaf) {
        out.push(chunk.to_vec());
    }
    Ok(out)
}

/// Build an omap leaf node (fixed-KV 16/16). When `is_root` is true the
/// node carries `BTNODE_ROOT` and a trailing `btree_info_t`.
fn build_omap_leaf_node(
    bs: usize,
    entries: &[(u64, u64, u64)],
    is_root: bool,
    xid: u64,
) -> Result<Vec<u8>> {
    let mut block = vec![0u8; bs];
    let obj_type = if is_root {
        OBJECT_TYPE_BTREE | OBJ_PHYSICAL
    } else {
        OBJECT_TYPE_BTREE_NODE | OBJ_PHYSICAL
    };
    block[16..24].copy_from_slice(&xid.to_le_bytes());
    block[24..28].copy_from_slice(&obj_type.to_le_bytes());

    let mut flags = BTNODE_LEAF | BTNODE_FIXED_KV_SIZE;
    if is_root {
        flags |= BTNODE_ROOT;
    }
    block[32..34].copy_from_slice(&flags.to_le_bytes());
    block[34..36].copy_from_slice(&0u16.to_le_bytes()); // level
    block[36..40].copy_from_slice(&(entries.len() as u32).to_le_bytes());

    let toc_len = entries.len() * 4;
    block[40..42].copy_from_slice(&0u16.to_le_bytes());
    block[42..44].copy_from_slice(&(toc_len as u16).to_le_bytes());

    let toc_base = 56;
    let keys_start = toc_base + toc_len;
    let vals_end = if is_root { bs - BTREE_INFO_SIZE } else { bs };
    if entries.len() * 32 + toc_len + 56 > vals_end {
        return Err(crate::Error::Unsupported(
            "apfs writer: omap leaf overflowed single block".into(),
        ));
    }
    for (i, &(oid, xid, paddr)) in entries.iter().enumerate() {
        let k_off = (i * 16) as u16;
        let v_off = ((i + 1) * 16) as u16;
        block[toc_base + i * 4..toc_base + i * 4 + 2].copy_from_slice(&k_off.to_le_bytes());
        block[toc_base + i * 4 + 2..toc_base + i * 4 + 4].copy_from_slice(&v_off.to_le_bytes());

        let ks = keys_start + k_off as usize;
        block[ks..ks + 8].copy_from_slice(&oid.to_le_bytes());
        block[ks + 8..ks + 16].copy_from_slice(&xid.to_le_bytes());

        let vs = vals_end - v_off as usize;
        // flags(0) + size(0) + paddr
        block[vs + 8..vs + 16].copy_from_slice(&paddr.to_le_bytes());
    }
    if is_root {
        // Trailing btree_info_t: bt_key_size=16, bt_val_size=16
        let info_off = bs - BTREE_INFO_SIZE;
        block[info_off + 8..info_off + 12].copy_from_slice(&16u32.to_le_bytes());
        block[info_off + 12..info_off + 16].copy_from_slice(&16u32.to_le_bytes());
    }
    sign_block(&mut block);
    Ok(block)
}

/// Build a fixed-KV internal omap root whose keys are 16-byte
/// `(oid, xid)` pairs and whose value slots hold 8-byte physical block
/// addresses of child leaves. The root carries the trailing
/// `btree_info_t`.
fn build_omap_internal_root(bs: usize, entries: &[((u64, u64), u64)], xid: u64) -> Result<Vec<u8>> {
    let mut block = vec![0u8; bs];
    block[16..24].copy_from_slice(&xid.to_le_bytes());
    block[24..28].copy_from_slice(&(OBJECT_TYPE_BTREE | OBJ_PHYSICAL).to_le_bytes());

    let flags = BTNODE_ROOT | BTNODE_FIXED_KV_SIZE;
    block[32..34].copy_from_slice(&flags.to_le_bytes());
    block[34..36].copy_from_slice(&1u16.to_le_bytes()); // level = 1
    block[36..40].copy_from_slice(&(entries.len() as u32).to_le_bytes());

    let toc_len = entries.len() * 4;
    block[40..42].copy_from_slice(&0u16.to_le_bytes());
    block[42..44].copy_from_slice(&(toc_len as u16).to_le_bytes());

    let toc_base = 56;
    let keys_start = toc_base + toc_len;
    let vals_end = bs - BTREE_INFO_SIZE;
    // Each entry uses 4 (ToC) + 16 (key) + 8 (child paddr) = 28 bytes.
    if entries.len() * 28 + 56 + BTREE_INFO_SIZE > bs {
        return Err(crate::Error::Unsupported(format!(
            "apfs writer: {} omap internal entries overflow one block",
            entries.len()
        )));
    }

    for (i, &((oid, xid), child_paddr)) in entries.iter().enumerate() {
        let k_off = (i * 16) as u16;
        // Internal values are 8 bytes per slot.
        let v_off = ((i + 1) * 8) as u16;
        block[toc_base + i * 4..toc_base + i * 4 + 2].copy_from_slice(&k_off.to_le_bytes());
        block[toc_base + i * 4 + 2..toc_base + i * 4 + 4].copy_from_slice(&v_off.to_le_bytes());

        let ks = keys_start + k_off as usize;
        block[ks..ks + 8].copy_from_slice(&oid.to_le_bytes());
        block[ks + 8..ks + 16].copy_from_slice(&xid.to_le_bytes());

        let vs = vals_end - v_off as usize;
        block[vs..vs + 8].copy_from_slice(&child_paddr.to_le_bytes());
    }
    // Root info: bt_key_size=16, bt_val_size=16 (leaf payload size; the
    // reader needs this to know how big leaf entries are).
    let info_off = bs - BTREE_INFO_SIZE;
    block[info_off + 8..info_off + 12].copy_from_slice(&16u32.to_le_bytes());
    block[info_off + 12..info_off + 16].copy_from_slice(&16u32.to_le_bytes());
    sign_block(&mut block);
    Ok(block)
}

/// Effective payload capacity for an fs-tree leaf (root and non-root use
/// the same conservative figure that includes the trailing
/// `btree_info_t`). Each record consumes 8 ToC bytes + key + val.
fn leaf_payload_capacity(bs: usize) -> usize {
    bs - 56 - BTREE_INFO_SIZE
}

/// Pack a sorted record list into one or more leaf groups, each of
/// which fits inside `cap` bytes when accounting for ToC overhead.
fn pack_records_into_leaves(records: &[FsRecord], cap: usize) -> Result<Vec<Vec<FsRecord>>> {
    if records.is_empty() {
        return Ok(vec![Vec::new()]);
    }
    let mut leaves: Vec<Vec<FsRecord>> = Vec::new();
    let mut cur: Vec<FsRecord> = Vec::new();
    let mut cur_bytes: usize = 0;
    for r in records {
        let needed = 8 + r.key.len() + r.val.len();
        if needed > cap {
            return Err(crate::Error::Unsupported(format!(
                "apfs writer: single fs-tree record ({} bytes) does not fit in a leaf (cap {})",
                needed, cap
            )));
        }
        if cur_bytes + needed > cap && !cur.is_empty() {
            leaves.push(std::mem::take(&mut cur));
            cur_bytes = 0;
        }
        cur.push(r.clone());
        cur_bytes += needed;
    }
    if !cur.is_empty() {
        leaves.push(cur);
    }
    Ok(leaves)
}

/// Build a variable-KV fs-tree leaf node holding the given (already-
/// sorted) `records`. When `is_root` is true the node carries
/// `BTNODE_ROOT` and a trailing `btree_info_t`. Both root-leaf (depth 1)
/// and non-root leaf (depth 2) callers use this builder; the root flag
/// is set accordingly.
fn build_fs_leaf(
    records: &[FsRecord],
    bs: usize,
    vid: u64,
    is_root: bool,
    xid: u64,
) -> Result<Vec<u8>> {
    let mut block = vec![0u8; bs];
    // obj_phys — real APFS fsroots are BTREE objects whose subtype is
    // FSTREE (the BTREE constant identifies the on-disk B-tree object;
    // FSTREE goes in the subtype slot to identify the tree's contents).
    block[8..16].copy_from_slice(&vid.to_le_bytes());
    block[16..24].copy_from_slice(&xid.to_le_bytes());
    let obj_type = if is_root {
        OBJECT_TYPE_BTREE
    } else {
        OBJECT_TYPE_BTREE_NODE
    };
    block[24..28].copy_from_slice(&obj_type.to_le_bytes());
    block[28..32].copy_from_slice(&OBJECT_TYPE_FSTREE.to_le_bytes());

    let mut flags = BTNODE_LEAF;
    if is_root {
        flags |= BTNODE_ROOT;
    }
    block[32..34].copy_from_slice(&flags.to_le_bytes());
    block[34..36].copy_from_slice(&0u16.to_le_bytes()); // level
    block[36..40].copy_from_slice(&(records.len() as u32).to_le_bytes());

    let toc_len = records.len() * 8;
    block[40..42].copy_from_slice(&0u16.to_le_bytes());
    block[42..44].copy_from_slice(&(toc_len as u16).to_le_bytes());

    let toc_base = 56;
    let keys_start = toc_base + toc_len;
    let vals_end = if is_root { bs - BTREE_INFO_SIZE } else { bs };

    // Compute total bytes needed.
    let mut total_keys = 0usize;
    let mut total_vals = 0usize;
    for r in records {
        total_keys += r.key.len();
        total_vals += r.val.len();
    }
    if keys_start + total_keys + total_vals > vals_end {
        return Err(crate::Error::Unsupported(format!(
            "apfs writer: {} fs-tree records don't fit in one leaf (need {} key bytes + {} val bytes)",
            records.len(),
            total_keys,
            total_vals,
        )));
    }

    let mut k_cursor: usize = 0;
    let mut v_cursor_back: usize = 0;
    for (i, r) in records.iter().enumerate() {
        let k_off = k_cursor as u16;
        let k_len = r.key.len() as u16;
        v_cursor_back += r.val.len();
        let v_off = v_cursor_back as u16;
        let v_len = r.val.len() as u16;
        block[toc_base + i * 8..toc_base + i * 8 + 2].copy_from_slice(&k_off.to_le_bytes());
        block[toc_base + i * 8 + 2..toc_base + i * 8 + 4].copy_from_slice(&k_len.to_le_bytes());
        block[toc_base + i * 8 + 4..toc_base + i * 8 + 6].copy_from_slice(&v_off.to_le_bytes());
        block[toc_base + i * 8 + 6..toc_base + i * 8 + 8].copy_from_slice(&v_len.to_le_bytes());
        let ks = keys_start + k_off as usize;
        block[ks..ks + r.key.len()].copy_from_slice(&r.key);
        let vs = vals_end - v_off as usize;
        block[vs..vs + r.val.len()].copy_from_slice(&r.val);
        k_cursor += r.key.len();
    }

    if is_root {
        // Trailing btree_info_t. We leave bt_key_size/bt_val_size at 0
        // (variable-KV tree). bt_key_count carries the record count so
        // tooling that inspects the root can sanity-check.
        let info_off = bs - BTREE_INFO_SIZE;
        block[info_off + 24..info_off + 32].copy_from_slice(&(records.len() as u64).to_le_bytes());
    }

    sign_block(&mut block);
    Ok(block)
}

/// Build a variable-KV fs-tree *internal* root pointing at child leaves
/// via virtual oids. Each `(key_bytes, child_vid)` entry is laid out
/// using the same kvloc_t-style ToC as a leaf, but the value bytes are
/// always 8 (the child vid in little-endian).
fn build_fs_internal_root(
    entries: &[(Vec<u8>, u64)],
    bs: usize,
    root_vid: u64,
    xid: u64,
) -> Result<Vec<u8>> {
    let mut block = vec![0u8; bs];
    block[8..16].copy_from_slice(&root_vid.to_le_bytes());
    block[16..24].copy_from_slice(&xid.to_le_bytes());
    block[24..28].copy_from_slice(&OBJECT_TYPE_BTREE.to_le_bytes());
    block[28..32].copy_from_slice(&OBJECT_TYPE_FSTREE.to_le_bytes());

    let flags = BTNODE_ROOT; // not leaf, var-KV
    block[32..34].copy_from_slice(&flags.to_le_bytes());
    block[34..36].copy_from_slice(&1u16.to_le_bytes()); // level = 1
    block[36..40].copy_from_slice(&(entries.len() as u32).to_le_bytes());

    let toc_len = entries.len() * 8;
    block[40..42].copy_from_slice(&0u16.to_le_bytes());
    block[42..44].copy_from_slice(&(toc_len as u16).to_le_bytes());

    let toc_base = 56;
    let keys_start = toc_base + toc_len;
    let vals_end = bs - BTREE_INFO_SIZE;

    let mut total_keys = 0usize;
    for (kb, _) in entries {
        total_keys += kb.len();
    }
    let total_vals = entries.len() * 8;
    if keys_start + total_keys + total_vals > vals_end {
        return Err(crate::Error::Unsupported(format!(
            "apfs writer: {} fs-tree internal entries don't fit in one root \
             (need {} key bytes + {} val bytes)",
            entries.len(),
            total_keys,
            total_vals,
        )));
    }

    let mut k_cursor: usize = 0;
    let mut v_cursor_back: usize = 0;
    for (i, (kb, child_vid)) in entries.iter().enumerate() {
        let k_off = k_cursor as u16;
        let k_len = kb.len() as u16;
        v_cursor_back += 8;
        let v_off = v_cursor_back as u16;
        let v_len = 8u16;
        block[toc_base + i * 8..toc_base + i * 8 + 2].copy_from_slice(&k_off.to_le_bytes());
        block[toc_base + i * 8 + 2..toc_base + i * 8 + 4].copy_from_slice(&k_len.to_le_bytes());
        block[toc_base + i * 8 + 4..toc_base + i * 8 + 6].copy_from_slice(&v_off.to_le_bytes());
        block[toc_base + i * 8 + 6..toc_base + i * 8 + 8].copy_from_slice(&v_len.to_le_bytes());

        let ks = keys_start + k_off as usize;
        block[ks..ks + kb.len()].copy_from_slice(kb);
        let vs = vals_end - v_off as usize;
        block[vs..vs + 8].copy_from_slice(&child_vid.to_le_bytes());

        k_cursor += kb.len();
    }
    // Trailing btree_info_t — record the (recursive) entry count.
    let info_off = bs - BTREE_INFO_SIZE;
    block[info_off + 24..info_off + 32].copy_from_slice(&(entries.len() as u64).to_le_bytes());
    sign_block(&mut block);
    Ok(block)
}

/// Build a `checkpoint_map_phys_t` with a single `checkpoint_mapping_t`
/// entry. Fsck walks this map to resolve the NXSB's ephemeral
/// `nx_spaceman_oid` to the physical block where `spaceman_phys_t` was
/// written.
///
/// On-disk layout (per Apple File System Reference):
///
/// ```text
///   0..32   obj_phys_t
///  32..36   cpm_flags (CHECKPOINT_MAP_LAST = 0x1)
///  36..40   cpm_count (number of entries, here 1)
///  40..80   checkpoint_mapping_t (40 bytes per entry)
/// ```
fn build_chkmap(
    bs: usize,
    paddr: u64,
    entry_type: u32,
    entry_oid: u64,
    entry_paddr: u64,
    xid: u64,
) -> Result<Vec<u8>> {
    if bs < 80 {
        return Err(crate::Error::Unsupported(format!(
            "apfs: block size {bs} too small for a checkpoint map"
        )));
    }
    let mut buf = vec![0u8; bs];
    buf[8..16].copy_from_slice(&paddr.to_le_bytes());
    buf[16..24].copy_from_slice(&xid.to_le_bytes());
    buf[24..28].copy_from_slice(&(OBJECT_TYPE_CHECKPOINT_MAP | OBJ_PHYSICAL).to_le_bytes());
    // cpm_flags = CHECKPOINT_MAP_LAST.
    buf[32..36].copy_from_slice(&1u32.to_le_bytes());
    // cpm_count = 1.
    buf[36..40].copy_from_slice(&1u32.to_le_bytes());

    // checkpoint_mapping_t (40 bytes) at offset 40.
    // cpm_type / cpm_subtype mirror the target object's o_type/o_subtype.
    buf[40..44].copy_from_slice(&entry_type.to_le_bytes());
    buf[44..48].copy_from_slice(&0u32.to_le_bytes()); // cpm_subtype
    // cpm_size = block size (we copy one block at this paddr).
    buf[48..52].copy_from_slice(&(bs as u32).to_le_bytes());
    // cpm_pad = 0 (already)
    // cpm_fs_oid = 0 (not a volume-scoped object)
    // cpm_oid = ephemeral oid of the target
    buf[56..64].copy_from_slice(&entry_oid.to_le_bytes());
    // cpm_paddr = physical address of the target
    buf[64..72].copy_from_slice(&entry_paddr.to_le_bytes());
    // cpm_offset (at offset 72..80) stays zero
    sign_block(&mut buf);
    Ok(buf)
}

/// Derive a deterministic 16-byte UUID-like value from a name + a salt.
/// We avoid a UUID dependency by mixing the inputs through a simple
/// xor-shift PRNG seeded with the bytes; not cryptographically random,
/// but stable across runs which is all we want.
fn derive_uuid(name: &[u8], salt: &[u8]) -> [u8; 16] {
    let mut seed: u64 = 0x6a09_e667_f3bc_c908;
    for &b in name.iter().chain(salt.iter()) {
        seed = seed.wrapping_mul(0x0100_0193).wrapping_add(b as u64);
        seed ^= seed >> 27;
    }
    let mut out = [0u8; 16];
    for chunk in out.chunks_mut(8) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        chunk.copy_from_slice(&seed.to_le_bytes());
    }
    // Set RFC4122 variant + version-4-ish bits so it looks UUID-shaped.
    out[6] = (out[6] & 0x0f) | 0x40;
    out[8] = (out[8] & 0x3f) | 0x80;
    out
}

pub(crate) fn mode_reg(perm: u16) -> u16 {
    0o100_000 | (perm & 0o7777)
}
pub(crate) fn mode_dir(perm: u16) -> u16 {
    0o040_000 | (perm & 0o7777)
}
pub(crate) fn mode_lnk(perm: u16) -> u16 {
    0o120_000 | (perm & 0o7777)
}

#[cfg(test)]
mod tests {
    use super::super::Apfs;
    use super::*;
    use crate::block::MemoryBackend;
    use std::io::{Cursor, Read};

    /// Build a small image with one regular file and round-trip it
    /// through the read path.
    #[test]
    fn write_then_read_single_small_file() {
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        {
            let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "TestVol").unwrap();
            let data = b"hello apfs world";
            let mut r = Cursor::new(data);
            w.add_file_from_reader(2, "hi.txt", 0o644, &mut r, data.len() as u64)
                .unwrap();
            w.finish().unwrap();
        }
        // Now read it back.
        let apfs = Apfs::open(&mut dev).expect("opens the new image");
        assert_eq!(apfs.volume_name(), "TestVol");
        let entries = apfs.list_path(&mut dev, "/").expect("list root");
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"hi.txt"), "got names: {names:?}");

        let mut reader = apfs.open_file_reader(&mut dev, "/hi.txt").unwrap();
        let mut out = Vec::new();
        reader.read_to_end(&mut out).unwrap();
        assert_eq!(out, b"hello apfs world");
    }

    /// A nested directory round-trips through list_path.
    #[test]
    fn write_then_read_nested_dir() {
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        {
            let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "Vol").unwrap();
            let dir = w.add_dir(2, "sub", 0o755).unwrap();
            let data = b"nested";
            let mut r = Cursor::new(data);
            w.add_file_from_reader(dir, "f", 0o644, &mut r, data.len() as u64)
                .unwrap();
            w.finish().unwrap();
        }
        let apfs = Apfs::open(&mut dev).unwrap();
        let root = apfs.list_path(&mut dev, "/").unwrap();
        assert!(root.iter().any(|e| e.name == "sub"));
        let sub = apfs.list_path(&mut dev, "/sub").unwrap();
        assert!(sub.iter().any(|e| e.name == "f"));
        let mut r = apfs.open_file_reader(&mut dev, "/sub/f").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, b"nested");
    }

    /// Symlinks survive a write+read round-trip and report as DT_LNK.
    #[test]
    fn write_then_read_symlink() {
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        {
            let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "Vol").unwrap();
            w.add_symlink(2, "link", 0o777, "/dev/null").unwrap();
            w.finish().unwrap();
        }
        let apfs = Apfs::open(&mut dev).unwrap();
        let entries = apfs.list_path(&mut dev, "/").unwrap();
        let link = entries.iter().find(|e| e.name == "link").unwrap();
        assert!(matches!(link.kind, crate::fs::EntryKind::Symlink));
    }

    /// Multi-volume API: a freshly written image with one volume
    /// reports exactly one populated slot, and `open_volume(0)` opens
    /// the same volume as `Apfs::open`.
    #[test]
    fn list_and_open_single_volume_slot() {
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        {
            let w = ApfsWriter::new(&mut dev, total_blocks, bs, "OnlyVol").unwrap();
            w.finish().unwrap();
        }
        let vols = Apfs::list_volumes(&mut dev).unwrap();
        assert_eq!(vols.len(), 1);
        assert_eq!(vols[0].name, "OnlyVol");
        assert!(!vols[0].encrypted);
        assert_eq!(vols[0].index, 0);
        let apfs = Apfs::open_volume(&mut dev, 0).unwrap();
        assert_eq!(apfs.volume_name(), "OnlyVol");
        // Opening a missing slot fails cleanly.
        let e = Apfs::open_volume(&mut dev, 5).unwrap_err();
        assert!(matches!(e, crate::Error::InvalidArgument(_)));
    }

    /// `list_snapshots` on a volume without snapshots returns an empty
    /// vector (apfs_snap_meta_tree_oid == 0).
    #[test]
    fn list_snapshots_empty_when_no_tree() {
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        {
            let w = ApfsWriter::new(&mut dev, total_blocks, bs, "Vol").unwrap();
            w.finish().unwrap();
        }
        let apfs = Apfs::open(&mut dev).unwrap();
        let snaps = apfs.list_snapshots(&mut dev).unwrap();
        assert!(snaps.is_empty());
        // open_snapshot with a non-existent xid should fail cleanly.
        let e = apfs.open_snapshot(&mut dev, 42).unwrap_err();
        assert!(matches!(e, crate::Error::InvalidArgument(_)));
    }

    /// Streaming invariant — a larger file (multi-block) still works.
    #[test]
    fn write_then_read_multiblock_file() {
        let total_blocks = 128u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let payload: Vec<u8> = (0..(20 * 1024)).map(|i| (i % 256) as u8).collect();
        {
            let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "Vol").unwrap();
            let mut r = Cursor::new(payload.clone());
            w.add_file_from_reader(2, "blob", 0o644, &mut r, payload.len() as u64)
                .unwrap();
            w.finish().unwrap();
        }
        let apfs = Apfs::open(&mut dev).unwrap();
        let mut r = apfs.open_file_reader(&mut dev, "/blob").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, payload);
    }

    /// Embedded xattrs round-trip through `read_xattrs`.
    #[test]
    fn write_then_read_embedded_xattrs() {
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        {
            let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "Vol").unwrap();
            let mut r = Cursor::new(b"x");
            w.add_file_from_reader(2, "f", 0o644, &mut r, 1).unwrap();
            // Look up the file's oid by scanning the directory.
            // The first new file gets oid 17 (we start at 16, root used 16's
            // slot? actually next_oid begins at 16 — first alloc returns 16).
            // But we don't depend on the value; we'll re-look it up on read.
            w.add_xattr(2, "user.note", b"hello world").unwrap();
            w.add_xattr(2, "user.lang", b"en_US").unwrap();
            w.finish().unwrap();
        }
        let apfs = Apfs::open(&mut dev).unwrap();
        let xs = apfs.read_xattrs(&mut dev, "/").unwrap();
        assert_eq!(
            xs.get("user.note").map(Vec::as_slice),
            Some(&b"hello world"[..])
        );
        assert_eq!(xs.get("user.lang").map(Vec::as_slice), Some(&b"en_US"[..]));
    }

    /// Xattrs attached to a non-root inode are surfaced under that
    /// inode's path (not under "/").
    #[test]
    fn xattr_attached_to_file_path() {
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        {
            let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "Vol").unwrap();
            let mut r = Cursor::new(b"data");
            let oid = w
                .add_file_from_reader(2, "doc.txt", 0o644, &mut r, 4)
                .unwrap();
            w.add_xattr(oid, "com.apple.lastuseddate#PS", &[1, 2, 3, 4])
                .unwrap();
            w.finish().unwrap();
        }
        let apfs = Apfs::open(&mut dev).unwrap();
        // Root has no xattrs.
        let root_xs = apfs.read_xattrs(&mut dev, "/").unwrap();
        assert!(root_xs.is_empty());
        // The file does.
        let xs = apfs.read_xattrs(&mut dev, "/doc.txt").unwrap();
        assert_eq!(
            xs.get("com.apple.lastuseddate#PS").map(Vec::as_slice),
            Some(&[1u8, 2, 3, 4][..])
        );
    }

    /// Oversized xattr values return `Unsupported`.
    #[test]
    fn xattr_too_big_unsupported() {
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "Vol").unwrap();
        let big = vec![0u8; APFS_XATTR_MAX_EMBEDDED_SIZE + 1];
        let e = w.add_xattr(2, "user.too_big", &big).unwrap_err();
        assert!(matches!(e, crate::Error::Unsupported(_)));
    }

    /// Multi-leaf fs-tree: enough small files to overflow a single 4 KiB
    /// leaf, then read them back via the reader's multi-level walker.
    #[test]
    fn write_then_read_multi_leaf_fs_tree() {
        let total_blocks = 256u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let n_files: usize = 80; // each file ≈ 50 bytes of fs-tree records → ~4 KiB leaf @ 4 KiB blocks
        let payloads: Vec<Vec<u8>> = (0..n_files)
            .map(|i| format!("file-{i:03}-payload").into_bytes())
            .collect();
        {
            let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "Vol").unwrap();
            for (i, p) in payloads.iter().enumerate() {
                let name = format!("f{i:03}");
                let mut r = Cursor::new(p.clone());
                w.add_file_from_reader(2, &name, 0o644, &mut r, p.len() as u64)
                    .unwrap();
            }
            w.finish().unwrap();
        }
        let apfs = Apfs::open(&mut dev).unwrap();
        let entries = apfs.list_path(&mut dev, "/").unwrap();
        assert!(
            entries.len() >= n_files,
            "expected at least {} entries, got {}",
            n_files,
            entries.len()
        );
        // Spot-check a few file contents.
        for &i in &[0usize, 1, n_files / 2, n_files - 1] {
            let name = format!("/f{i:03}");
            let mut r = apfs.open_file_reader(&mut dev, &name).unwrap();
            let mut out = Vec::new();
            r.read_to_end(&mut out).unwrap();
            assert_eq!(out, payloads[i], "mismatch for {name}");
        }
    }

    /// `pack_records_into_leaves` splits when the cumulative byte count
    /// would exceed `cap`, and never emits an empty leaf.
    #[test]
    fn pack_records_splits_on_capacity() {
        let r = |k: u8, v_len: usize| FsRecord {
            key: vec![k; 8],
            val: vec![0u8; v_len],
        };
        let recs = vec![r(1, 100), r(2, 100), r(3, 100)];
        // cap = 8 + (8 + 100) = 116 → exactly one record per leaf.
        let leaves = pack_records_into_leaves(&recs, 116).unwrap();
        assert_eq!(leaves.len(), 3);
        // cap = 2 * (8 + 8 + 100) = 232 → two records per leaf.
        let leaves = pack_records_into_leaves(&recs, 232).unwrap();
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].len(), 2);
        assert_eq!(leaves[1].len(), 1);
    }

    // ---- Spaceman tests ----
    //
    // `fsck_apfs` is not packaged for Linux (no `apt`/`brew` binary is
    // available on this host or in CI), so we can't run it directly.
    // Instead we round-trip a tiny image through the writer and verify
    // that the emitted `spaceman_phys_t` + CIB + bitmap blocks are
    // internally consistent and agree with the writer's known
    // allocations. If a host ever does have `fsck_apfs`, see the
    // separate `apfs_fsck_external` ignored test for the wiring.

    use crate::block::BlockDevice;
    use crate::fs::apfs::spaceman::{count_used_bits, decode_cib_entries, decode_spaceman};

    /// Read the `paddr`-th 4 KiB block off `dev` as a Vec<u8>.
    fn read_block(dev: &mut dyn BlockDevice, paddr: u64, bs: u32) -> Vec<u8> {
        let mut buf = vec![0u8; bs as usize];
        dev.read_at(paddr * bs as u64, &mut buf).unwrap();
        buf
    }

    /// Verify spaceman+CIB+bitmap consistency: every chunk_info entry's
    /// free_count agrees with its bitmap's cleared-bit count, and the
    /// total free_count agrees with the spaceman header.
    fn assert_spaceman_consistent(dev: &mut dyn BlockDevice, bs: u32, total_blocks: u64) {
        // Spaceman lives at SPACEMAN_PADDR.
        let sm = read_block(dev, SPACEMAN_PADDR, bs);
        let dec = decode_spaceman(&sm).expect("decode spaceman_phys_t");
        assert_eq!(dec.block_size, bs);
        assert_eq!(dec.blocks_per_chunk, bs * 8);
        assert_eq!(dec.main_block_count, total_blocks);
        assert_eq!(dec.main_cab_count, 0);
        assert_eq!(dec.main_cib_count, 1);
        assert_eq!(dec.main_chunk_count, total_blocks.div_ceil((bs * 8) as u64));

        // CIB block (decoded from sm.cib_paddr).
        let cib = read_block(dev, dec.cib_paddr, bs);
        let entries = decode_cib_entries(&cib).expect("decode CIB");
        assert_eq!(entries.len() as u64, dec.main_chunk_count);

        let mut free_total: u64 = 0;
        let bpc = bs * 8;
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(e.addr, (i as u64) * bpc as u64, "chunk addr off");
            let expected_blocks =
                ((i as u64 + 1) * bpc as u64).min(total_blocks) - (i as u64) * bpc as u64;
            assert_eq!(
                e.block_count as u64, expected_blocks,
                "chunk {} block_count off",
                i
            );
            // Cross-check free_count vs bitmap cleared bits.
            let bmap = read_block(dev, e.bitmap_addr, bs);
            let used = count_used_bits(&bmap, e.block_count);
            let free = e.block_count - used;
            assert_eq!(
                free, e.free_count,
                "chunk {} free_count {} disagrees with bitmap (used={})",
                i, e.free_count, used
            );
            free_total += free as u64;
        }
        assert_eq!(
            free_total, dec.main_free_count,
            "spaceman free_count disagrees with sum of chunk free counts",
        );
    }

    /// Smoke test: the spaceman emitted for a tiny image accurately
    /// reflects the writer's allocations.
    #[test]
    fn spaceman_reflects_bump_allocations() {
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        {
            let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "Vol").unwrap();
            let data = b"abc";
            let mut r = Cursor::new(data);
            w.add_file_from_reader(2, "f", 0o644, &mut r, data.len() as u64)
                .unwrap();
            w.finish().unwrap();
        }
        assert_spaceman_consistent(&mut dev, bs, total_blocks);
        // Every used block in our bitmap must actually be a block we
        // wrote into. For the smoke check, just confirm block 0 (NXSB
        // label) is marked used and the last block of the container
        // (which the writer never touches) is marked free.
        let sm = read_block(&mut dev, SPACEMAN_PADDR, bs);
        let dec = decode_spaceman(&sm).unwrap();
        let cib = read_block(&mut dev, dec.cib_paddr, bs);
        let entries = decode_cib_entries(&cib).unwrap();
        let bmap = read_block(&mut dev, entries[0].bitmap_addr, bs);
        assert!(bmap[0] & 0x01 != 0, "block 0 (NXSB label) must be used");
        let last_bit = (total_blocks - 1) as usize;
        assert!(
            bmap[last_bit / 8] & (1 << (last_bit % 8)) == 0,
            "last block of container must be free"
        );
    }

    /// Larger image: multi-block file, still inside chunk 0. The
    /// spaceman bitmap must report the file's extent as "used".
    #[test]
    fn spaceman_marks_file_extent_used() {
        let total_blocks = 256u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let payload: Vec<u8> = (0..(20 * 1024)).map(|i| (i % 256) as u8).collect();
        // First pass: write *just* metadata (empty image) to see how
        // many blocks the writer naturally consumes.
        let used_blocks_before = {
            let mut dev1 = MemoryBackend::new(total_blocks * bs as u64);
            let w1 = ApfsWriter::new(&mut dev1, total_blocks, bs, "V").unwrap();
            w1.finish().unwrap();
            let sm1 = read_block(&mut dev1, SPACEMAN_PADDR, bs);
            let d1 = decode_spaceman(&sm1).unwrap();
            total_blocks - d1.main_free_count
        };
        {
            let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "V").unwrap();
            let mut r = Cursor::new(payload.clone());
            w.add_file_from_reader(2, "blob", 0o644, &mut r, payload.len() as u64)
                .unwrap();
            w.finish().unwrap();
        }
        assert_spaceman_consistent(&mut dev, bs, total_blocks);
        let sm = read_block(&mut dev, SPACEMAN_PADDR, bs);
        let d = decode_spaceman(&sm).unwrap();
        let used_blocks_after = total_blocks - d.main_free_count;
        // Writing a 20 KiB file consumes at least 5 extra data blocks
        // (plus its fs-tree-record bookkeeping). The bitmap must
        // reflect that growth.
        assert!(
            used_blocks_after >= used_blocks_before + 5,
            "expected at least 5 more used blocks after adding 20 KiB file \
             (was {used_blocks_before}, now {used_blocks_after})"
        );
    }

    /// A freshly-formatted spaceman publishes non-zero IP-ring
    /// metadata, three valid free-queue B-tree oids, and at least one
    /// allocation zone covering the main device.
    #[test]
    fn spaceman_populates_ip_sfq_and_datazone() {
        let total_blocks = 128u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        {
            let w = ApfsWriter::new(&mut dev, total_blocks, bs, "Vol").unwrap();
            w.finish().unwrap();
        }
        let sm = read_block(&mut dev, SPACEMAN_PADDR, bs);
        let dec = decode_spaceman(&sm).expect("decode spaceman_phys_t");

        // sm_flags must publish SM_FLAG_VERSIONED so the reader trusts
        // sm_version (=1).
        assert_eq!(dec.flags & 0x1, 0x1, "SM_FLAG_VERSIONED must be set");

        // Issue 1 — internal-pool fields are non-zero.
        assert_ne!(
            dec.ip_bm_tx_multiplier, 0,
            "sm_ip_bm_tx_multiplier must be populated"
        );
        assert_ne!(dec.ip_block_count, 0, "sm_ip_block_count must be non-zero");
        assert_ne!(
            dec.ip_bm_size_in_blocks, 0,
            "sm_ip_bm_size_in_blocks must be non-zero"
        );
        assert_ne!(
            dec.ip_bm_block_count, 0,
            "sm_ip_bm_block_count must be non-zero"
        );
        assert_ne!(dec.ip_bm_base, 0, "sm_ip_bm_base must be non-zero");
        assert_ne!(dec.ip_base, 0, "sm_ip_base must be non-zero");
        // Sanity: the ring must lie inside the container.
        assert!(
            dec.ip_base + dec.ip_block_count <= total_blocks,
            "IP ring (base={}, len={}) extends past container ({} blocks)",
            dec.ip_base,
            dec.ip_block_count,
            total_blocks
        );

        // Issue 2 — three non-zero SFQ tree oids.
        for (i, oid) in dec.sfq_tree_oids.iter().enumerate() {
            assert_ne!(*oid, 0, "sm_fq[{i}].sfq_tree_oid must be non-zero");
        }
        // Each SFQ root must be a readable B-tree node.
        for (i, &paddr) in dec.sfq_tree_oids.iter().enumerate() {
            let node_block = read_block(&mut dev, paddr, bs);
            // BTNODE_ROOT|BTNODE_LEAF|BTNODE_FIXED_KV_SIZE at offset 32.
            let flags = u16::from_le_bytes(node_block[32..34].try_into().unwrap());
            assert_eq!(
                flags & 0x0007,
                0x0007,
                "SFQ[{i}] root must be a fixed-KV leaf root (flags={flags:#x})"
            );
            // nkeys must be 0 on a fresh image.
            let nkeys = u32::from_le_bytes(node_block[36..40].try_into().unwrap());
            assert_eq!(nkeys, 0, "SFQ[{i}] root must be empty (nkeys={nkeys})");
            // Trailing btree_info_t: bt_key_size=16, bt_val_size=8.
            let info_off = bs as usize - 40;
            let bt_key_size =
                u32::from_le_bytes(node_block[info_off + 8..info_off + 12].try_into().unwrap());
            let bt_val_size =
                u32::from_le_bytes(node_block[info_off + 12..info_off + 16].try_into().unwrap());
            assert_eq!(bt_key_size, 16, "SFQ[{i}] bt_key_size must be 16");
            assert_eq!(bt_val_size, 8, "SFQ[{i}] bt_val_size must be 8");
        }

        // Issue 3 — at least one allocation zone covers the main
        // device.
        assert_eq!(
            dec.datazone_main_zone0_start, 0,
            "main-device zone 0 must start at block 0"
        );
        assert_ne!(
            dec.datazone_main_zone0_end, 0,
            "main-device zone 0 saz_zone_end must be populated"
        );
        assert_eq!(
            dec.datazone_main_zone0_end, total_blocks,
            "main-device zone 0 must cover the whole device"
        );

        // Adding the IP ring + SFQ + IP-bitmap blocks must not break
        // bitmap-vs-allocation consistency.
        assert_spaceman_consistent(&mut dev, bs, total_blocks);
    }

    /// The checkpoint map at block 1 must resolve the NXSB's
    /// `nx_spaceman_oid` to the spaceman's physical block.
    #[test]
    fn checkpoint_map_resolves_spaceman_oid() {
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        {
            let w = ApfsWriter::new(&mut dev, total_blocks, bs, "V").unwrap();
            w.finish().unwrap();
        }
        // NXSB at NXSB_LIVE_PADDR: nx_spaceman_oid is at offset 152..160.
        let nxsb = read_block(&mut dev, NXSB_LIVE_PADDR, bs);
        let sm_oid = u64::from_le_bytes(nxsb[152..160].try_into().unwrap());
        // Checkpoint map at CHKMAP_PADDR.
        let chk = read_block(&mut dev, CHKMAP_PADDR, bs);
        // cpm_count at offset 36, first entry at 40 with cpm_oid at +16
        // (within the entry), cpm_paddr at +24.
        let cpm_count = u32::from_le_bytes(chk[36..40].try_into().unwrap());
        assert_eq!(cpm_count, 1, "expected one checkpoint-map entry");
        let entry = 40usize;
        let cpm_oid = u64::from_le_bytes(chk[entry + 16..entry + 24].try_into().unwrap());
        let cpm_paddr = u64::from_le_bytes(chk[entry + 24..entry + 32].try_into().unwrap());
        assert_eq!(
            cpm_oid, sm_oid,
            "chkmap oid must match NXSB nx_spaceman_oid"
        );
        assert_eq!(
            cpm_paddr, SPACEMAN_PADDR,
            "spaceman lives at SPACEMAN_PADDR"
        );
    }

    /// External `fsck_apfs` smoke test — runs only when the binary is
    /// installed (typically on macOS). Builds a tiny image to a temp
    /// file, then shells out and asserts a clean exit. Linux hosts skip
    /// this test gracefully.
    #[test]
    #[ignore = "requires fsck_apfs (macOS host); see test body"]
    fn apfs_fsck_external() {
        use std::process::Command;
        if Command::new("fsck_apfs").arg("-h").output().is_err() {
            // Tool unavailable — skip without failing.
            return;
        }
        let dir = std::env::temp_dir();
        let path = dir.join("genfs-apfs-spaceman-fsck.img");
        let total_blocks = 256u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        {
            let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "FsckVol").unwrap();
            let mut r = Cursor::new(b"hello");
            w.add_file_from_reader(2, "f.txt", 0o644, &mut r, 5)
                .unwrap();
            w.finish().unwrap();
        }
        // Drain memory backend to file.
        let mut buf = vec![0u8; (total_blocks * bs as u64) as usize];
        dev.read_at(0, &mut buf).unwrap();
        std::fs::write(&path, &buf).unwrap();
        let status = Command::new("fsck_apfs")
            .arg("-n")
            .arg(&path)
            .status()
            .unwrap();
        assert!(status.success(), "fsck_apfs reported errors");
    }
}
