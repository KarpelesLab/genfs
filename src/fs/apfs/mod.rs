//! APFS — Apple's modern macOS / iOS filesystem. Read + best-effort
//! single-volume write support.
//!
//! ## Scope
//!
//! APFS is large. This module implements the on-disk parsers and a
//! best-effort walker plus a single-volume writer:
//!
//! 1. The **container superblock** (`nx_superblock_t`) at block 0 is
//!    decoded. The checkpoint descriptor area is then scanned for the
//!    most-recent valid NXSB copy (the live checkpoint).
//! 2. The container's **object map** (`omap_phys_t`) is loaded and its
//!    root B-tree node is read. The omap walker descends multi-level
//!    trees by binary-searching internal nodes; a small LRU node cache
//!    keeps memory bounded while honouring the streaming invariant.
//! 3. Every populated `nx_fs_oid[]` slot is resolved through the
//!    container omap to a physical block; the **APFS volume
//!    superblock** (`apfs_superblock_t`) is decoded per volume.
//! 4. The volume's own omap is loaded; the volume root tree (`fsroot`,
//!    `OBJECT_TYPE_FSTREE`) is located.
//! 5. The fsroot is walked top-down using a virtual-oid-aware
//!    multi-level B-tree walker. Directory listings and file extents
//!    are gathered via prefix range scans over the fs-tree.
//! 6. The volume's **snapshot-metadata tree** (`apfs_snap_meta_tree_oid`)
//!    is read as a physical leaf-only tree and yields a list of
//!    `(xid, name, sblock_paddr, create_time)` tuples. Snapshots can be
//!    opened via [`Apfs::open_snapshot`] / [`Apfs::open_snapshot_by_name`].
//! 7. The [`mod@write`] submodule produces minimal APFS images from
//!    scratch — see [`write::ApfsWriter`] for the library-only API.
//!    The [`crate::fs::Filesystem`] trait is also wired up via
//!    [`Apfs::format`]: `format → create_dir / create_file /
//!    create_symlink → flush` materialises an image and transitions
//!    the [`Apfs`] to read mode. After flush the volume is sealed
//!    (no further mutation), matching the writer's single-pass model.
//!
//! ## Honest limitations
//!
//! - **Drec layout detection.** The walker chooses between hashed and
//!   plain drec keys based on `APFS_INCOMPAT_NORMALIZATION_INSENSITIVE`,
//!   and falls back to the other layout on decode error. APFS's normalized
//!   hash function isn't implemented here; range scans iterate by
//!   `(parent_oid, DIR_REC)` prefix and filter by name in the caller, so
//!   correctness doesn't depend on the hash.
//! - **Snapshots** are read-only, single-leaf-tree only — multi-level
//!   snap-meta trees return `Unsupported`. Snapshot lookups use the
//!   snap-meta tree's `sblock_oid` directly and re-bind the volume's
//!   omap to the snapshot's xid.
//! - **No encryption.** Encrypted volumes are detected via `apfs_fs_flags`
//!   and refused with `Unsupported`.
//! - **No sealed-volume hashes / integrity tree.** Sealed volumes are
//!   detected via `APFS_INCOMPAT_SEALED_VOLUME` and refused.
//! - **No FusionDrive tiering** (the secondary `nx_efi_jumpstart`
//!   structures, tier-2 omap, etc.).
//! - **Xattrs:** embedded xattrs are surfaced by
//!   [`Apfs::read_xattrs`]. Dstream-backed xattrs (`XATTR_DATA_STREAM`)
//!   are silently skipped on read and refused on write.
//! - **No resource forks, no clones, no compressed files**
//!   (`UF_COMPRESSED` files are read as if they had no data).
//! - **Fletcher-64 checksum is computed but not enforced** by default —
//!   we accept blocks whose checksum fails and emit no warnings, because
//!   real images often disagree with the spec in subtle ways and we'd
//!   rather return data than refuse it. The checksum helper is still
//!   exposed for callers that want to enforce it.
//! - **Writer** supports multi-leaf fs-trees and multi-leaf omaps with
//!   one internal level above the leaves. Trees too large for that
//!   single internal level return `Unsupported`. See [`mod@write`] for
//!   the per-feature limits.
//!
//! ## References
//!
//! All field names and constants come from Apple's public *Apple File
//! System Reference* (PDF). No GPL code from libfsapfs or other
//! reverse-engineering projects was consulted.

pub mod btree;
pub mod checksum;
pub mod fstree;
pub mod jrec;
pub mod obj;
pub mod omap;
pub(crate) mod rw;
pub mod snap;
pub(crate) mod spaceman;
pub mod superblock;
pub mod write;

use crate::Result;
use crate::block::BlockDevice;

use fstree::{DrecKeyLayout, FsKeyTarget, FsTreeCtx, RangeScan};
use jrec::{
    APFS_TYPE_DIR_REC, APFS_TYPE_FILE_EXTENT, APFS_TYPE_INODE, APFS_TYPE_XATTR, DT_DIR, DT_LNK,
    DT_REG, DrecKey, DrecVal, FileExtentVal, InodeVal, OBJ_ID_MASK, OBJ_TYPE_SHIFT,
};
use obj::{OBJECT_TYPE_MASK, ObjPhys};
use omap::{OmapPhys, lookup as omap_lookup};
use snap::{SnapMetaVal, decode_snap_meta_key};
use superblock::{ApfsSuperblock, NX_MAGIC, NxSuperblock};

/// Root inode object id in every APFS volume.
const ROOT_DIR_INO: u64 = 2;

/// In-memory state for an opened APFS container/volume.
///
/// The fs-tree caches are kept behind a `RefCell` so the public API
/// can remain `&self`-callable even though multi-level walks
/// internally mutate cache state. Callers that need read-from-multiple
/// threads should wrap the whole `Apfs` in a `Mutex` — there is no
/// internal locking.
///
/// An [`Apfs`] can be in one of two states:
///
/// - **Read state** (the default after [`Apfs::open`] / [`Apfs::open_volume`]):
///   the volume has been parsed off the device and the reader caches
///   are live.
/// - **Pending-write state** (after [`Apfs::format`]): the device has
///   not yet been written. Buffered `create_*` operations are queued
///   in memory; [`crate::fs::Filesystem::flush`] drains the queue
///   into an [`write::ApfsWriter`] and then transitions the [`Apfs`]
///   to read state by re-parsing the just-written image.
pub struct Apfs {
    /// Effective block size (`nx_block_size`).
    block_size: u32,
    /// `nx_block_count * nx_block_size`.
    total_bytes: u64,
    /// Volume name (UTF-8, trimmed of trailing NUL).
    volume_name: String,
    /// Internal state: either pending-write (buffered ops) or read
    /// (parsed fs-tree, ready for queries).
    state: ApfsState,
}

/// Internal state machine for [`Apfs`]. See the [`Apfs`] doc-comment
/// for what each state means.
enum ApfsState {
    /// Read mode: the volume has been parsed and is ready for queries.
    Read(ReadState),
    /// Pending-write mode: buffered `create_*` ops waiting for
    /// [`crate::fs::Filesystem::flush`].
    PendingWrite(PendingWrite),
    /// Reopen-and-write mode: the volume is parsed (read state nested
    /// inside) and we additionally carry the checkpoint metadata
    /// needed to write a new NXSB at the next xp_desc slot via
    /// [`crate::fs::Filesystem::open_file_rw`] + `sync`.
    Write(WriteState),
}

/// Reopen-and-write state: a parsed read state plus enough writer
/// scaffolding to commit a fresh checkpoint.
pub(crate) struct WriteState {
    /// Underlying parsed volume — used for fs-tree reads inside the
    /// rw handle and for the read APIs.
    pub(crate) read: ReadState,
    /// Volume name (mirrors the outer [`Apfs::volume_name`]). Recomputed
    /// from the APSB at open time.
    pub(crate) volume_name: String,
    /// Container UUID from the live NXSB.
    pub(crate) container_uuid: [u8; 16],
    /// Volume UUID from the APSB.
    pub(crate) volume_uuid: [u8; 16],
    /// Total blocks in the container (`nx_block_count`).
    pub(crate) total_blocks: u64,
    /// `next_oid` to assign on the next mutation (mirrors APSB's value).
    pub(crate) next_oid: u64,
    /// `apfs_num_files` snapshotted at open time. Incremented locally
    /// on each new create — currently not used because rw is read+write
    /// on existing files only, but kept so future create paths can
    /// inherit cleanly.
    pub(crate) num_files: u64,
    /// `apfs_num_directories` (excluding root) snapshotted at open.
    pub(crate) num_directories: u64,
    /// `apfs_num_symlinks` snapshotted at open.
    pub(crate) num_symlinks: u64,
    /// Current checkpoint xid (the one driving the read view).
    pub(crate) cur_xid: u64,
    /// Next free xp_desc slot inside the xp_desc area. The new NXSB
    /// will be written here on sync; we then advance this pointer.
    pub(crate) next_xp_desc_slot: u64,
    /// High-water mark for bump allocation. Computed at open time from
    /// the spaceman's used-block count and grown by every new metadata
    /// or extent block.
    pub(crate) bump_high_water: u64,
}

/// Read-mode caches: everything needed to walk the fs-tree.
pub(crate) struct ReadState {
    /// `apfs_snap_meta_tree_oid` — physical block of the snapshot
    /// metadata B-tree root, or zero when the volume has no snapshots.
    snap_meta_tree_oid: u64,
    /// Which slot in `nx_fs_oid[]` produced this volume. A snapshot
    /// view inherits the parent volume's slot.
    volume_index: usize,
    /// Fs-tree root block — kept around because every fs-tree walk
    /// starts here. Internal-node children are virtual oids resolved
    /// through the volume omap in `fs_ctx`.
    fsroot_block: Vec<u8>,
    /// Volume omap context (omap root block + caches + target xid).
    /// Wrapped in `RefCell` so `&self` methods can still mutate the
    /// LRU caches.
    fs_ctx: std::cell::RefCell<FsTreeCtx>,
    /// Drec layout (hashed vs plain) chosen at open time based on
    /// `apfs_incompatible_features` flags.
    drec_layout: DrecKeyLayout,
}

/// Pending-write buffer: an ordered list of `create_*` operations
/// plus a path → oid map. Drained by [`crate::fs::Filesystem::flush`].
struct PendingWrite {
    /// Image geometry recorded at `format()` time.
    total_blocks: u64,
    /// Map of every directory created so far to its inode oid. The
    /// root "/" maps to 2 from the start. Walking nested paths during
    /// `create_*` requires the parent to be in this map.
    dir_oid: std::collections::HashMap<std::path::PathBuf, u64>,
    /// Buffered create operations, replayed in order on `flush()`.
    ops: Vec<PendingOp>,
    /// Synthetic oid counter used to populate `dir_oid` for nested
    /// directories at buffer time. Mirrors the writer's
    /// `alloc_oid` sequence so flushed oids stay consistent with
    /// what callers were told to expect.
    next_oid: u64,
}

/// A single buffered create operation.
enum PendingOp {
    /// `create_dir` — `parent_oid` is the oid of the directory the
    /// new dir lives under; `name` is the leaf name.
    Dir {
        parent_oid: u64,
        name: String,
        mode: u16,
        mtime_ns: u64,
    },
    /// `create_file` — same shape as `Dir`. The file's bytes are
    /// captured into `data` at buffer time because [`crate::fs::FileSource`]
    /// is consumed by the trait call.
    File {
        parent_oid: u64,
        name: String,
        mode: u16,
        mtime_ns: u64,
        data: Vec<u8>,
    },
    /// `create_symlink` — same shape as `Dir`, plus the link target.
    Symlink {
        parent_oid: u64,
        name: String,
        mode: u16,
        mtime_ns: u64,
        target: String,
    },
}

/// Convert a [`crate::fs::FileMeta`]'s u32 epoch-second `mtime` into
/// the u64 nanosecond timestamp APFS stores in its inode time fields.
/// Returns 0 (1970-01-01) when `meta.mtime` is 0, matching POSIX
/// "unset" semantics.
fn meta_mtime_ns(meta: &crate::fs::FileMeta) -> u64 {
    (meta.mtime as u64).saturating_mul(1_000_000_000)
}

impl std::fmt::Debug for Apfs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state_name = match &self.state {
            ApfsState::Read(_) => "Read",
            ApfsState::PendingWrite(_) => "PendingWrite",
            ApfsState::Write(_) => "Write",
        };
        f.debug_struct("Apfs")
            .field("block_size", &self.block_size)
            .field("total_bytes", &self.total_bytes)
            .field("volume_name", &self.volume_name)
            .field("state", &state_name)
            .finish_non_exhaustive()
    }
}

/// `APFS_INCOMPAT_NORMALIZATION_INSENSITIVE` — when set, drec keys use
/// the hashed layout (`j_drec_hashed_key_t`); otherwise the plain
/// layout (`j_drec_key_t`) is in use.
const APFS_INCOMPAT_NORMALIZATION_INSENSITIVE: u64 = 0x0000_0008;

/// Bundle of container-level state used by every volume opener so we
/// don't have to re-scan the checkpoint descriptor area on each call.
#[derive(Debug, Clone)]
struct ContainerCtx {
    live_sb: NxSuperblock,
    omap_root: Vec<u8>,
    block_size: u32,
    total_bytes: u64,
    /// Offset within the xp_desc area (0-based, relative to
    /// `live_sb.xp_desc_base`) where the live NXSB was found. Used by
    /// [`Apfs::open_writable`] to pick the next slot via ring math:
    /// the new checkpoint writes one slot past the live one, wrapping
    /// at `xp_desc_blocks` and skipping slot 0 (the chkmap). `None`
    /// when no valid NXSB was found in the xp_desc area (fresh image
    /// with only the label NXSB at block 0).
    live_xp_desc_offset: Option<u64>,
}

/// Public summary of one populated `nx_fs_oid[]` slot, returned by
/// [`Apfs::list_volumes`].
#[derive(Debug, Clone)]
pub struct VolumeInfo {
    /// Index into `nx_fs_oid[]` (0-based). Pass this to
    /// [`Apfs::open_volume`].
    pub index: usize,
    /// Virtual oid recorded in `nx_fs_oid[index]`.
    pub vol_oid: u64,
    /// Volume name from the volume's APSB (`apfs_volname`).
    pub name: String,
    /// `apfs_role` — the volume's role byte (0 = no role).
    pub role: u16,
    /// True when the volume's `APFS_FS_UNENCRYPTED` flag is clear — i.e.
    /// the volume is encrypted and won't open.
    pub encrypted: bool,
    /// Volume UUID.
    pub uuid: [u8; 16],
}

/// Public summary of one snapshot, returned by [`Apfs::list_snapshots`].
#[derive(Debug, Clone)]
pub struct SnapshotInfo {
    /// Transaction id at which the snapshot was taken.
    pub xid: u64,
    /// Snapshot name (UTF-8, trimmed of trailing NUL).
    pub name: String,
    /// Per-snapshot APSB physical block address (`sblock_oid`).
    pub sblock_paddr: u64,
    /// `j_snap_metadata_val_t.create_time` (nanoseconds since 1970-01-01).
    pub create_time: u64,
}

impl Apfs {
    /// Return `&ReadState` if in read or write mode, else an error.
    /// Used by every reader API to refuse cleanly when the [`Apfs`]
    /// is still buffering writes.
    fn read_state(&self) -> Result<&ReadState> {
        match &self.state {
            ApfsState::Read(r) => Ok(r),
            ApfsState::Write(w) => Ok(&w.read),
            ApfsState::PendingWrite(_) => Err(crate::Error::Unsupported(
                "apfs: filesystem is in pending-write mode; call flush() first".into(),
            )),
        }
    }

    /// Format an empty single-volume APFS image on `dev`. The returned
    /// [`Apfs`] is in pending-write mode: [`crate::fs::Filesystem::create_file`],
    /// [`crate::fs::Filesystem::create_dir`], and
    /// [`crate::fs::Filesystem::create_symlink`] buffer their effects
    /// in memory, and [`crate::fs::Filesystem::flush`] materialises the
    /// on-disk image and transitions the [`Apfs`] to read mode.
    ///
    /// `total_blocks * block_size` must fit inside `dev.total_size()`.
    /// `block_size` follows the same constraints as
    /// [`write::ApfsWriter::new`] (power of two between 512 and 65 536;
    /// 4096 is the conventional value).
    ///
    /// Note: the image is only written to `dev` when `flush()` is
    /// called. Before that, `dev` is not touched (other than the
    /// size sanity-check performed inside [`write::ApfsWriter::new`],
    /// which is non-destructive).
    pub fn format(
        dev: &mut dyn BlockDevice,
        total_blocks: u64,
        block_size: u32,
        volume_name: &str,
    ) -> Result<Self> {
        // Sanity-check geometry up front by constructing (and discarding)
        // a writer. This validates the same invariants `flush()` will
        // re-check, so callers find out about a bad geometry immediately
        // instead of after a series of `create_*` calls.
        let _ = write::ApfsWriter::new(dev, total_blocks, block_size, volume_name)?;
        let mut dir_oid = std::collections::HashMap::new();
        dir_oid.insert(std::path::PathBuf::from("/"), ROOT_DIR_INO);
        Ok(Self {
            block_size,
            total_bytes: total_blocks.saturating_mul(block_size as u64),
            volume_name: volume_name.to_string(),
            state: ApfsState::PendingWrite(PendingWrite {
                total_blocks,
                dir_oid,
                ops: Vec::new(),
                // Mirror `ApfsWriter::new`: writer starts oid counter at 16.
                next_oid: 16,
            }),
        })
    }

    /// Decode the container, find the active checkpoint, locate the
    /// first populated volume slot, and cache its fs-tree root block.
    /// Errors out with `Unsupported` when the image trips one of the
    /// explicit limitations listed at module level.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let ctx = load_container(dev)?;
        let vol_index = ctx
            .live_sb
            .fs_oid
            .iter()
            .position(|&o| o != 0)
            .ok_or_else(|| {
                crate::Error::InvalidImage("apfs: container has no volumes in nx_fs_oid".into())
            })?;
        Self::open_volume_with_ctx(dev, &ctx, vol_index, None)
    }

    /// Open a previously-flushed APFS container for in-place mutation.
    ///
    /// Unlike [`Apfs::open`], the returned [`Apfs`] is in **write
    /// state**: [`crate::fs::Filesystem::open_file_rw`] can be invoked
    /// on existing files, and `sync()` on the returned handle commits
    /// a new on-disk checkpoint (a fresh NXSB written into the next
    /// xp_desc slot, leaving the previous valid checkpoint intact for
    /// crash safety).
    ///
    /// Each successful `sync` on a returned handle consumes one
    /// xp_desc slot; slots are reused in ring-buffer order so the
    /// number of consecutive checkpoint commits is unbounded. The
    /// chkmap at slot 0 is preserved (we never overwrite it).
    ///
    /// Refuses volumes formatted with
    /// `APFS_INCOMPAT_NORMALIZATION_INSENSITIVE` (the default for
    /// macOS-created APFS volumes): the on-disk drec key carries a
    /// 22-bit hash of the Unicode-normalized + case-folded leaf
    /// name, and we don't implement that hash. Writing a drec with
    /// a wrong-sorted key would leave the B-tree in an order the
    /// kernel can't traverse on lookup. Read-only access to such a
    /// volume keeps working via [`Apfs::open`].
    pub fn open_writable(dev: &mut dyn BlockDevice) -> Result<Self> {
        let read_only = Apfs::open(dev)?;
        // Re-decode the container so we can pull the checkpoint
        // metadata we need (xid, container UUID, live slot offset).
        let ctx = load_container(dev)?;
        let block_size = ctx.block_size;
        let total_blocks = ctx.live_sb.block_count;
        let total_bytes = ctx.total_bytes;
        let container_uuid = ctx.live_sb.uuid;
        let cur_xid = ctx.live_sb.obj.xid;

        // Refuse hashed-key (case-insensitive) volumes up front. We
        // could attach in Write state, but every subsequent drec we
        // add would sort wrong in the B-tree: the key carries a
        // 22-bit hash of the normalized name, and our writer has
        // no normalization hash. Silently corrupting the catalog
        // is far worse than refusing — Apfs::open on the same image
        // keeps working for read-only access.
        match find_volume_paddr(dev, &ctx) {
            Ok((_, paddr)) => {
                let mut apsb_block = vec![0u8; block_size as usize];
                dev.read_at(paddr * block_size as u64, &mut apsb_block)?;
                let apsb = ApfsSuperblock::decode(&apsb_block)?;
                if apsb.incompatible_features & APFS_INCOMPAT_NORMALIZATION_INSENSITIVE != 0 {
                    return Err(crate::Error::Unsupported(
                        "apfs: volume is APFS_INCOMPAT_NORMALIZATION_INSENSITIVE \
                         (case-insensitive); the writer doesn't implement the \
                         APFS normalization hash so writes would corrupt the \
                         drec B-tree. Use Apfs::open for read-only access."
                            .into(),
                    ));
                }
            }
            Err(_) => {
                // No volumes / unreadable APSB — defer to the rest
                // of open_writable to surface a precise error.
            }
        }

        // Pick the next xp_desc slot via ring math: the new NXSB
        // writes one offset past the live one, wrapping at
        // `xp_desc_blocks`. Slot 0 holds the chkmap and is reserved —
        // valid NXSB slots are [1, xp_desc_blocks), so we skip slot 0
        // when wrapping. The reader's `find_live_nxsb` walks every
        // slot and picks the highest-xid valid NXSB, so ring rotation
        // round-trips through the read side without further work.
        //
        // For a fresh image with no live NXSB in the xp_desc area
        // (label NXSB at block 0 only), start at slot 1 (the
        // canonical first NXSB slot = `NXSB_LIVE_PADDR`).
        let blocks = ctx.live_sb.xp_desc_blocks as u64;
        let nxsb_slots = blocks.saturating_sub(1).max(1); // skip slot 0
        let next_offset = match ctx.live_xp_desc_offset {
            Some(off) if off >= 1 => ((off - 1 + 1) % nxsb_slots) + 1,
            // Live NXSB came from slot 0 somehow, or no live NXSB at
            // all — start at the first NXSB slot.
            _ => 1,
        };
        let next_xp_desc_slot = ctx.live_sb.xp_desc_base + next_offset;

        // Read the APSB to capture volume_uuid + counters + next_oid.
        let (vol_index, apsb_paddr) = find_volume_paddr(dev, &ctx)?;
        let mut apsb_block = vec![0u8; block_size as usize];
        dev.read_at(apsb_paddr * block_size as u64, &mut apsb_block)?;
        let apsb = ApfsSuperblock::decode(&apsb_block)?;

        // Extract next_obj_id from APSB (offset 176..184) — not on the
        // pub struct, so peek directly.
        let next_oid = u64::from_le_bytes(apsb_block[176..184].try_into().unwrap());

        // Use the spaceman's used-block accounting as the bump-allocator
        // high-water mark. Falling back to a conservative guess if the
        // spaceman can't be parsed: assume the whole front half of the
        // disk is in use. Honest: this means we lose half the capacity,
        // but it's safe.
        let bump_high_water = read_spaceman_high_water(dev, &ctx).unwrap_or(total_blocks / 2);

        // Build the write state, embedding the read-mode bits we just
        // parsed via Apfs::open above.
        let read = match read_only.state {
            ApfsState::Read(r) => r,
            _ => unreachable!("Apfs::open always returns Read"),
        };
        let _ = vol_index;
        Ok(Self {
            block_size,
            total_bytes,
            volume_name: apsb.volname.clone(),
            state: ApfsState::Write(WriteState {
                read,
                volume_name: apsb.volname.clone(),
                container_uuid,
                volume_uuid: apsb.vol_uuid,
                total_blocks,
                next_oid,
                num_files: apsb.num_files,
                num_directories: apsb.num_directories,
                num_symlinks: apsb.num_symlinks,
                cur_xid,
                next_xp_desc_slot,
                bump_high_water,
            }),
        })
    }

    /// List every populated `nx_fs_oid[]` slot. Encrypted volumes are
    /// returned with `encrypted = true` so callers can surface them in
    /// UIs even though [`Apfs::open_volume`] will refuse them.
    pub fn list_volumes(dev: &mut dyn BlockDevice) -> Result<Vec<VolumeInfo>> {
        let ctx = load_container(dev)?;
        let target_xid = ctx.live_sb.obj.xid;
        let mut out = Vec::new();
        for (i, &vol_oid) in ctx.live_sb.fs_oid.iter().enumerate() {
            if vol_oid == 0 {
                continue;
            }
            let mut dev_reader = DevReader {
                dev,
                block_size: ctx.block_size,
            };
            let vol_loc =
                match omap_lookup(&ctx.omap_root, vol_oid, target_xid, &mut |paddr, buf| {
                    dev_reader.read(paddr, buf)
                })? {
                    Some(v) => v,
                    None => continue,
                };
            let mut apsb_block = vec![0u8; ctx.block_size as usize];
            dev_reader.read(vol_loc.paddr, &mut apsb_block)?;
            let apsb = match ApfsSuperblock::decode(&apsb_block) {
                Ok(s) => s,
                Err(_) => continue,
            };
            const APFS_FS_UNENCRYPTED: u64 = 0x0000_0001;
            // `apfs_role` lives at offset 964; superblock.rs doesn't
            // capture it directly, so peek here.
            let role = if apsb_block.len() >= 966 {
                u16::from_le_bytes(apsb_block[964..966].try_into().unwrap())
            } else {
                0
            };
            out.push(VolumeInfo {
                index: i,
                vol_oid,
                name: apsb.volname.clone(),
                role,
                encrypted: apsb.fs_flags & APFS_FS_UNENCRYPTED == 0,
                uuid: apsb.vol_uuid,
            });
        }
        Ok(out)
    }

    /// Open the volume at `nx_fs_oid[index]`. Use [`Apfs::list_volumes`]
    /// to enumerate slots first.
    pub fn open_volume(dev: &mut dyn BlockDevice, index: usize) -> Result<Self> {
        let ctx = load_container(dev)?;
        Self::open_volume_with_ctx(dev, &ctx, index, None)
    }

    /// Internal: open a specific volume slot, optionally rebound to a
    /// snapshot view via `(sblock_paddr, snap_xid)`.
    fn open_volume_with_ctx(
        dev: &mut dyn BlockDevice,
        ctx: &ContainerCtx,
        index: usize,
        snapshot: Option<(u64, u64)>,
    ) -> Result<Self> {
        if index >= ctx.live_sb.fs_oid.len() {
            return Err(crate::Error::InvalidArgument(format!(
                "apfs: volume index {index} out of range"
            )));
        }
        let vol_oid = ctx.live_sb.fs_oid[index];
        if vol_oid == 0 {
            return Err(crate::Error::InvalidArgument(format!(
                "apfs: nx_fs_oid[{index}] is empty"
            )));
        }
        let block_size = ctx.block_size;
        let target_xid = ctx.live_sb.obj.xid;
        let mut dev_reader = DevReader { dev, block_size };

        // Resolve the APSB physical block: either look it up live in the
        // container omap, or use a snapshot-supplied physical block.
        let apsb_paddr = match snapshot {
            Some((p, _)) => p,
            None => {
                let vol_loc =
                    omap_lookup(&ctx.omap_root, vol_oid, target_xid, &mut |paddr, buf| {
                        dev_reader.read(paddr, buf)
                    })?
                    .ok_or_else(|| {
                        crate::Error::InvalidImage(format!(
                            "apfs: container omap has no entry for volume oid {vol_oid:#x}"
                        ))
                    })?;
                vol_loc.paddr
            }
        };

        let mut apsb_block = vec![0u8; block_size as usize];
        dev_reader.read(apsb_paddr, &mut apsb_block)?;
        let apsb = ApfsSuperblock::decode(&apsb_block)?;

        // Bail early on encrypted volumes — we can't decrypt anything.
        const APFS_FS_UNENCRYPTED: u64 = 0x0000_0001;
        if apsb.fs_flags & APFS_FS_UNENCRYPTED == 0 {
            return Err(crate::Error::Unsupported(
                "apfs: encrypted volumes are not supported (read)".into(),
            ));
        }
        // Sealed-volume integrity hashes are not honoured.
        const APFS_INCOMPAT_SEALED_VOLUME: u64 = 0x0000_0080;
        if apsb.incompatible_features & APFS_INCOMPAT_SEALED_VOLUME != 0 {
            return Err(crate::Error::Unsupported(
                "apfs: sealed volumes (integrity hashes) are not supported".into(),
            ));
        }

        // ---- Volume omap ----
        let vol_omap_phys =
            read_object::<OmapPhys>(dev_reader.dev, apsb.omap_oid, block_size, OmapPhys::decode)?;

        let mut vol_omap_root = vec![0u8; block_size as usize];
        dev_reader.read(vol_omap_phys.tree_oid, &mut vol_omap_root)?;

        // For snapshots the omap xid is the snapshot's xid; for the
        // live volume it's the APSB's own xid.
        let omap_xid = match snapshot {
            Some((_, xid)) => xid,
            None => apsb.obj.xid,
        };

        // ---- Resolve fsroot through the volume omap (multi-level safe) ----
        let fsroot_loc = omap_lookup(
            &vol_omap_root,
            apsb.root_tree_oid,
            omap_xid,
            &mut |paddr, buf| dev_reader.read(paddr, buf),
        )?
        .ok_or_else(|| {
            crate::Error::InvalidImage(format!(
                "apfs: volume omap has no entry for root_tree_oid {:#x} @ xid {omap_xid}",
                apsb.root_tree_oid
            ))
        })?;

        let mut fsroot_block = vec![0u8; block_size as usize];
        dev_reader.read(fsroot_loc.paddr, &mut fsroot_block)?;
        // Sanity-check the root is a btree (internal or leaf — both work).
        let fsroot_obj = ObjPhys::decode(&fsroot_block)?;
        let ot = fsroot_obj.type_and_flags & OBJECT_TYPE_MASK;
        if ot != obj::OBJECT_TYPE_BTREE && ot != obj::OBJECT_TYPE_BTREE_NODE {
            return Err(crate::Error::InvalidImage(format!(
                "apfs: fsroot o_type {ot:#x} is not a btree"
            )));
        }

        let drec_layout =
            if apsb.incompatible_features & APFS_INCOMPAT_NORMALIZATION_INSENSITIVE != 0 {
                DrecKeyLayout::Hashed
            } else {
                DrecKeyLayout::Plain
            };

        let fs_ctx = FsTreeCtx::new(vol_omap_root, omap_xid, block_size as usize);

        Ok(Self {
            block_size,
            total_bytes: ctx.total_bytes,
            volume_name: apsb.volname,
            state: ApfsState::Read(ReadState {
                snap_meta_tree_oid: apsb.snap_meta_tree_oid,
                volume_index: index,
                fsroot_block,
                fs_ctx: std::cell::RefCell::new(fs_ctx),
                drec_layout,
            }),
        })
    }

    /// List every snapshot recorded in the volume's snapshot-metadata
    /// tree. The snap-meta tree is a physical B-tree; we read its root
    /// block directly. Multi-level snap-meta trees return `Unsupported`
    /// cleanly.
    pub fn list_snapshots(&self, dev: &mut dyn BlockDevice) -> Result<Vec<SnapshotInfo>> {
        let rs = self.read_state()?;
        if rs.snap_meta_tree_oid == 0 {
            return Ok(Vec::new());
        }
        let mut root = vec![0u8; self.block_size as usize];
        let off = rs.snap_meta_tree_oid.saturating_mul(self.block_size as u64);
        dev.read_at(off, &mut root)?;
        let node = btree::BTreeNode::decode(&root)?;
        if !node.is_leaf() {
            return Err(crate::Error::Unsupported(
                "apfs: multi-level snapshot-metadata trees are not supported".into(),
            ));
        }
        let mut out = Vec::new();
        for i in 0..node.nkeys {
            let (kb, vb) = node.entry_at(i, 0, 0)?;
            let (kind, xid) = match decode_snap_meta_key(kb) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if kind != jrec::APFS_TYPE_SNAP_METADATA {
                continue;
            }
            let meta = match SnapMetaVal::decode(vb) {
                Ok(m) => m,
                Err(_) => continue,
            };
            out.push(SnapshotInfo {
                xid,
                name: meta.name,
                sblock_paddr: meta.sblock_oid,
                create_time: meta.create_time,
            });
        }
        Ok(out)
    }

    /// Open a snapshot of this volume by transaction id.
    ///
    /// The returned [`Apfs`] reads through the same volume omap as
    /// `self` but filters lookups by the snapshot's xid, so it sees
    /// the on-disk state that existed at that xid.
    pub fn open_snapshot(&self, dev: &mut dyn BlockDevice, xid: u64) -> Result<Self> {
        let vol_index = self.read_state()?.volume_index;
        let snaps = self.list_snapshots(dev)?;
        let snap = snaps
            .iter()
            .find(|s| s.xid == xid)
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!("apfs: no snapshot with xid {xid}"))
            })?
            .clone();
        let ctx = load_container(dev)?;
        Self::open_volume_with_ctx(dev, &ctx, vol_index, Some((snap.sblock_paddr, snap.xid)))
    }

    /// Open a snapshot of this volume by name.
    pub fn open_snapshot_by_name(&self, dev: &mut dyn BlockDevice, name: &str) -> Result<Self> {
        let vol_index = self.read_state()?.volume_index;
        let snaps = self.list_snapshots(dev)?;
        let snap = snaps
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!("apfs: no snapshot named {name:?}"))
            })?
            .clone();
        let ctx = load_container(dev)?;
        Self::open_volume_with_ctx(dev, &ctx, vol_index, Some((snap.sblock_paddr, snap.xid)))
    }

    /// List the children of `path`. Only absolute paths starting at "/"
    /// are accepted; "" and "/" both resolve to the root directory.
    ///
    /// Walks the fs-tree top-down using a multi-level B-tree walker; the
    /// tree may be of any depth.
    pub fn list_path(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let target_oid = self.resolve_path_to_oid(dev, path)?;
        self.list_dir(dev, target_oid)
    }

    /// Read every extended attribute attached to the inode at `path`.
    ///
    /// APFS xattrs are individual fs-tree records keyed by
    /// `(parent_oid, APFS_TYPE_XATTR, name)`; this walks that prefix
    /// range and returns each `(name → value)` pair as a `HashMap`.
    ///
    /// Only embedded (`XATTR_DATA_EMBEDDED`) xattrs are returned;
    /// dstream-backed xattrs (`XATTR_DATA_STREAM`) are silently skipped
    /// because the writer never emits them and our reader doesn't yet
    /// resolve the secondary dstream object. Callers that need
    /// large-xattr support should run their own scan via the lower-level
    /// `fstree::RangeScan` API.
    pub fn read_xattrs(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<std::collections::HashMap<String, Vec<u8>>> {
        let target_oid = self.resolve_path_to_oid(dev, path)?;
        let rs = self.read_state()?;
        let target = FsKeyTarget {
            oid: target_oid,
            kind: APFS_TYPE_XATTR,
            tail: &[],
            drec_layout: rs.drec_layout,
        };
        let block_size = self.block_size;
        let mut ctx = rs.fs_ctx.borrow_mut();
        let mut out = std::collections::HashMap::new();
        let mut scan = RangeScan::start(&rs.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })?;
        while let Some((kb, vb)) = scan.next(&mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })? {
            // Key: j_xattr_key_t = j_key_t(8) + u16 name_len + name[name_len]
            if kb.len() < 10 {
                continue;
            }
            let nlen = u16::from_le_bytes(kb[8..10].try_into().unwrap()) as usize;
            if 10 + nlen > kb.len() || nlen == 0 {
                continue;
            }
            let raw = &kb[10..10 + nlen];
            let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
            let name = String::from_utf8_lossy(&raw[..end]).into_owned();
            // Value: j_xattr_val_t = u16 flags + u16 xdata_len + xdata
            if vb.len() < 4 {
                continue;
            }
            let flags = u16::from_le_bytes(vb[0..2].try_into().unwrap());
            let xdata_len = u16::from_le_bytes(vb[2..4].try_into().unwrap()) as usize;
            const XATTR_DATA_EMBEDDED: u16 = 0x0002;
            if flags & XATTR_DATA_EMBEDDED == 0 {
                // Dstream-backed xattr — skip silently.
                continue;
            }
            if 4 + xdata_len > vb.len() {
                continue;
            }
            let value = vb[4..4 + xdata_len].to_vec();
            out.insert(name, value);
        }
        Ok(out)
    }

    /// Open a regular file for streaming reads. The returned reader
    /// borrows `dev` so it can fetch data blocks lazily.
    pub fn open_file_reader<'a>(
        &self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<ApfsFileReader<'a>> {
        let target_oid = self.resolve_path_to_oid(dev, path)?;
        let (size, dstream_oid) = self.lookup_inode_size(dev, target_oid)?;
        let extents = self.collect_extents(dev, dstream_oid.unwrap_or(target_oid))?;
        Ok(ApfsFileReader {
            dev,
            block_size: self.block_size,
            extents,
            size,
            cursor: 0,
        })
    }

    /// Change the permission bits (low 12 bits of `mode`) of the
    /// inode at `path`. Preserves the file-type bits.
    ///
    /// Writes a fresh APFS checkpoint — same COW pathway
    /// `rw::commit_with_mutator` uses for every other in-place
    /// mutation. The on-disk byte affected lives at offset 80..82
    /// of `j_inode_val_t`.
    pub fn chmod(&mut self, dev: &mut dyn BlockDevice, path: &str, mode_perms: u16) -> Result<()> {
        let target_oid = self.resolve_path_to_oid(dev, path)?;
        let new_perms = mode_perms & 0o7777;
        rw::commit_with_mutator(self, dev, |cx| {
            let records = &mut *cx.records;
            patch_inode_record(records, target_oid, |val| {
                if val.len() < jrec::J_INODE_VAL_FIXED_SIZE {
                    return;
                }
                let cur_mode = u16::from_le_bytes(val[80..82].try_into().unwrap());
                let new_mode = (cur_mode & 0xF000) | new_perms;
                val[80..82].copy_from_slice(&new_mode.to_le_bytes());
            });
            Ok(())
        })
    }

    /// Change ownership (owner/group) of the inode at `path`. POSIX
    /// `chown`. `j_inode_val_t` carries owner at offset 72..76 and
    /// group at 76..80.
    pub fn chown(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        owner: u32,
        group: u32,
    ) -> Result<()> {
        let target_oid = self.resolve_path_to_oid(dev, path)?;
        rw::commit_with_mutator(self, dev, |cx| {
            let records = &mut *cx.records;
            patch_inode_record(records, target_oid, |val| {
                if val.len() < jrec::J_INODE_VAL_FIXED_SIZE {
                    return;
                }
                val[72..76].copy_from_slice(&owner.to_le_bytes());
                val[76..80].copy_from_slice(&group.to_le_bytes());
            });
            Ok(())
        })
    }

    /// Stamp the modification / access / change timestamps on the
    /// inode at `path`. Each argument is a UNIX timestamp in
    /// nanoseconds (APFS native precision); `None` leaves the field
    /// unchanged. Layout: `mod_time` at 24..32, `change_time` at
    /// 32..40, `access_time` at 40..48 — all u64 little-endian.
    pub fn set_times(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        mtime_ns: Option<u64>,
        ctime_ns: Option<u64>,
        atime_ns: Option<u64>,
    ) -> Result<()> {
        let target_oid = self.resolve_path_to_oid(dev, path)?;
        rw::commit_with_mutator(self, dev, |cx| {
            let records = &mut *cx.records;
            patch_inode_record(records, target_oid, |val| {
                if val.len() < jrec::J_INODE_VAL_FIXED_SIZE {
                    return;
                }
                if let Some(m) = mtime_ns {
                    val[24..32].copy_from_slice(&m.to_le_bytes());
                }
                if let Some(c) = ctime_ns {
                    val[32..40].copy_from_slice(&c.to_le_bytes());
                }
                if let Some(a) = atime_ns {
                    val[40..48].copy_from_slice(&a.to_le_bytes());
                }
            });
            Ok(())
        })
    }

    /// Remove a dirent. Mirrors POSIX `unlink` for files and
    /// `rmdir` for empty directories. Decrements the target inode's
    /// link count; when the last link is gone, drops the inode +
    /// dstream + file_extent + xattr records (the bytes themselves
    /// stay on disk — the COW allocator never frees, only forgets).
    /// Refuses non-empty directories.
    pub fn remove_path(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<()> {
        let (parent_oid, name) = self.resolve_parent_and_name(dev, path)?;
        rw::commit_with_mutator(self, dev, |cx| {
            let records = &mut *cx.records;
            let (target_oid, dtype) = find_drec(records, parent_oid, &name).ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "apfs: no such entry {name:?} under inode {parent_oid}"
                ))
            })?;
            // Empty-dir check for directories.
            if dtype == DT_DIR && drec_count_for(records, target_oid) > 0 {
                return Err(crate::Error::InvalidArgument(format!(
                    "apfs: directory {name:?} is not empty"
                )));
            }
            // Always drop the dirent.
            remove_drec(records, parent_oid, &name);
            // For dirs: drop the inode (always — POSIX dirs never
            // hardlink); decrement parent's nchildren.
            if dtype == DT_DIR {
                remove_all_records_for_oid(records, target_oid);
                patch_inode_record(records, parent_oid, |v| {
                    if v.len() >= jrec::J_INODE_VAL_FIXED_SIZE {
                        let cur = i32::from_le_bytes(v[56..60].try_into().unwrap());
                        v[56..60].copy_from_slice(&(cur - 1).to_le_bytes());
                    }
                });
                return Ok(());
            }
            // Non-dir: hardlink-aware unlink. Decrement target's
            // nlink; only purge the inode + payload when the last
            // link goes.
            let mut last_link = false;
            patch_inode_record(records, target_oid, |v| {
                if v.len() >= jrec::J_INODE_VAL_FIXED_SIZE {
                    let cur = i32::from_le_bytes(v[56..60].try_into().unwrap());
                    let new = cur.saturating_sub(1);
                    v[56..60].copy_from_slice(&new.to_le_bytes());
                    if new <= 0 {
                        last_link = true;
                    }
                }
            });
            if last_link {
                remove_all_records_for_oid(records, target_oid);
            }
            // Decrement parent nchildren for non-dir removal too.
            patch_inode_record(records, parent_oid, |v| {
                if v.len() >= jrec::J_INODE_VAL_FIXED_SIZE {
                    let cur = i32::from_le_bytes(v[56..60].try_into().unwrap());
                    v[56..60].copy_from_slice(&(cur - 1).to_le_bytes());
                }
            });
            Ok(())
        })
    }

    /// Rename a dirent. Cross-directory moves of a directory rewrite
    /// the target inode's `parent_id` and adjust both parents'
    /// nchildren. Hardlinks survive a rename (we never touch the
    /// target inode's data).
    pub fn rename(
        &mut self,
        dev: &mut dyn BlockDevice,
        old_path: &str,
        new_path: &str,
    ) -> Result<()> {
        let (old_parent, old_name) = self.resolve_parent_and_name(dev, old_path)?;
        let (new_parent, new_name) = self.resolve_parent_and_name(dev, new_path)?;
        rw::commit_with_mutator(self, dev, |cx| {
            let records = &mut *cx.records;
            let (target_oid, dtype) =
                find_drec(records, old_parent, &old_name).ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "apfs: rename source {old_name:?} not found"
                    ))
                })?;
            if find_drec(records, new_parent, &new_name).is_some() {
                return Err(crate::Error::InvalidArgument(format!(
                    "apfs: rename target {new_name:?} already exists"
                )));
            }
            remove_drec(records, old_parent, &old_name);
            // Commit-1 stub: pass Plain/false. Commit-2 threads the
            // volume's real DrecKeyLayout + case_fold from WriteState.
            let (k, v) = write::build_drec_record(
                new_parent,
                &new_name,
                target_oid,
                dtype,
                DrecKeyLayout::Plain,
                false,
            )?;
            records.push((k, v));
            if old_parent != new_parent {
                // Update both parents' nchildren.
                patch_inode_record(records, old_parent, |v| {
                    if v.len() >= jrec::J_INODE_VAL_FIXED_SIZE {
                        let cur = i32::from_le_bytes(v[56..60].try_into().unwrap());
                        v[56..60].copy_from_slice(&(cur - 1).to_le_bytes());
                    }
                });
                patch_inode_record(records, new_parent, |v| {
                    if v.len() >= jrec::J_INODE_VAL_FIXED_SIZE {
                        let cur = i32::from_le_bytes(v[56..60].try_into().unwrap());
                        v[56..60].copy_from_slice(&(cur + 1).to_le_bytes());
                    }
                });
                // Directory move also rewrites the target's parent
                // pointer at offset 0..8 of its j_inode_val_t.
                if dtype == DT_DIR {
                    patch_inode_record(records, target_oid, |v| {
                        if v.len() >= 8 {
                            v[0..8].copy_from_slice(&new_parent.to_le_bytes());
                        }
                    });
                }
            }
            Ok(())
        })
    }

    /// Create a hardlink to an existing non-directory inode. Writes
    /// a fresh drec under `new_parent_path` pointing at the target
    /// and bumps the target inode's nlink. POSIX disallows hardlinks
    /// to directories — we refuse them up front.
    pub fn link(
        &mut self,
        dev: &mut dyn BlockDevice,
        existing_path: &str,
        new_path: &str,
    ) -> Result<()> {
        let target_oid = self.resolve_path_to_oid(dev, existing_path)?;
        let (new_parent, new_name) = self.resolve_parent_and_name(dev, new_path)?;
        // Peek at the source's dtype before we commit so we can refuse
        // dir hardlinks cleanly.
        let target_dtype = self.lookup_inode_dtype(dev, target_oid)?;
        if target_dtype == DT_DIR {
            return Err(crate::Error::InvalidArgument(
                "apfs: cannot hardlink to a directory".into(),
            ));
        }
        rw::commit_with_mutator(self, dev, |cx| {
            let records = &mut *cx.records;
            if find_drec(records, new_parent, &new_name).is_some() {
                return Err(crate::Error::InvalidArgument(format!(
                    "apfs: link target {new_name:?} already exists"
                )));
            }
            // Commit-1 stub: Plain/false. Commit-2 swaps in real values.
            let (k, v) = write::build_drec_record(
                new_parent,
                &new_name,
                target_oid,
                target_dtype,
                DrecKeyLayout::Plain,
                false,
            )?;
            records.push((k, v));
            patch_inode_record(records, target_oid, |v| {
                if v.len() >= jrec::J_INODE_VAL_FIXED_SIZE {
                    let cur = i32::from_le_bytes(v[56..60].try_into().unwrap());
                    v[56..60].copy_from_slice(&(cur + 1).to_le_bytes());
                }
            });
            patch_inode_record(records, new_parent, |v| {
                if v.len() >= jrec::J_INODE_VAL_FIXED_SIZE {
                    let cur = i32::from_le_bytes(v[56..60].try_into().unwrap());
                    v[56..60].copy_from_slice(&(cur + 1).to_le_bytes());
                }
            });
            Ok(())
        })
    }

    /// Create a new regular file at `path` containing `data` with the
    /// given permission bits. Writes a fresh APFS checkpoint via the
    /// same COW pathway every other Write-state mutation uses
    /// (`rw::commit_with_mutator`). Allocates a single fresh extent
    /// for the body — empty files emit only the inode + drec records.
    ///
    /// Errors when the parent directory doesn't exist, the leaf name
    /// already exists in that directory, or the device doesn't have
    /// enough free blocks for the new extent.
    pub fn create_file_at(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        data: &[u8],
        mode: u16,
        mtime_ns: u64,
    ) -> Result<()> {
        let (parent_oid, name) = self.resolve_parent_and_name(dev, path)?;
        rw::commit_with_mutator(self, dev, |cx| {
            if find_drec(cx.records, parent_oid, &name).is_some() {
                return Err(crate::Error::InvalidArgument(format!(
                    "apfs: create_file: {name:?} already exists under inode {parent_oid}"
                )));
            }
            let oid = cx.alloc_oid();
            let bs = cx.block_size();
            let size = data.len() as u64;
            // Allocate one extent + write the body. Empty files skip
            // the extent and the matching FILE_EXTENT / DSTREAM_ID
            // records — mirrors the single-pass writer's convention.
            if size > 0 {
                let paddr = cx.alloc_extent(size)?;
                cx.write_extent_bytes(paddr, data)?;
                for (k, v) in write::build_file_extent_records(oid, 0, size, paddr, bs) {
                    cx.records.push((k, v));
                }
            }
            // INODE for the new file, then the DREC under the parent.
            let (ik, iv) = write::build_inode_record(
                oid,
                parent_oid,
                write::mode_reg(mode),
                size,
                bs,
                mtime_ns,
            );
            cx.records.push((ik, iv));
            // Commit-1 stub: Plain/false. Commit-2 swaps in real values.
            let (dk, dv) = write::build_drec_record(
                parent_oid,
                &name,
                oid,
                DT_REG,
                DrecKeyLayout::Plain,
                false,
            )?;
            cx.records.push((dk, dv));
            // Bump the parent dir's nchildren counter.
            patch_inode_record(cx.records, parent_oid, |v| {
                if v.len() >= jrec::J_INODE_VAL_FIXED_SIZE {
                    let cur = i32::from_le_bytes(v[56..60].try_into().unwrap());
                    v[56..60].copy_from_slice(&(cur + 1).to_le_bytes());
                }
            });
            cx.note_new_file();
            Ok(())
        })
    }

    /// Create an empty directory at `path` with the given permission
    /// bits. Writes a fresh checkpoint via `rw::commit_with_mutator`.
    /// Errors when the parent doesn't exist or the leaf name is
    /// already taken.
    pub fn create_dir_at(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        mode: u16,
        mtime_ns: u64,
    ) -> Result<()> {
        let (parent_oid, name) = self.resolve_parent_and_name(dev, path)?;
        rw::commit_with_mutator(self, dev, |cx| {
            if find_drec(cx.records, parent_oid, &name).is_some() {
                return Err(crate::Error::InvalidArgument(format!(
                    "apfs: create_dir: {name:?} already exists under inode {parent_oid}"
                )));
            }
            let oid = cx.alloc_oid();
            let bs = cx.block_size();
            // Directories carry no DSTREAM xfield (dstream_size = 0
            // + mode is S_IFDIR → has_dstream is false in
            // build_inode_record).
            let (ik, iv) =
                write::build_inode_record(oid, parent_oid, write::mode_dir(mode), 0, bs, mtime_ns);
            cx.records.push((ik, iv));
            // Commit-1 stub: Plain/false. Commit-2 swaps in real values.
            let (dk, dv) = write::build_drec_record(
                parent_oid,
                &name,
                oid,
                DT_DIR,
                DrecKeyLayout::Plain,
                false,
            )?;
            cx.records.push((dk, dv));
            patch_inode_record(cx.records, parent_oid, |v| {
                if v.len() >= jrec::J_INODE_VAL_FIXED_SIZE {
                    let cur = i32::from_le_bytes(v[56..60].try_into().unwrap());
                    v[56..60].copy_from_slice(&(cur + 1).to_le_bytes());
                }
            });
            cx.note_new_dir();
            Ok(())
        })
    }

    /// Create a symbolic link at `path` pointing at `target` (the
    /// raw string is stored verbatim — APFS doesn't normalise it).
    /// Symlink targets are stored as a single-extent file body, the
    /// same way `Apfs::read_symlink` reads them back.
    pub fn create_symlink_at(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        target: &str,
        mode: u16,
        mtime_ns: u64,
    ) -> Result<()> {
        let (parent_oid, name) = self.resolve_parent_and_name(dev, path)?;
        let target_bytes = target.as_bytes();
        let size = target_bytes.len() as u64;
        if size == 0 {
            return Err(crate::Error::InvalidArgument(
                "apfs: symlink target must be non-empty".into(),
            ));
        }
        rw::commit_with_mutator(self, dev, |cx| {
            if find_drec(cx.records, parent_oid, &name).is_some() {
                return Err(crate::Error::InvalidArgument(format!(
                    "apfs: create_symlink: {name:?} already exists under inode {parent_oid}"
                )));
            }
            let oid = cx.alloc_oid();
            let bs = cx.block_size();
            let paddr = cx.alloc_extent(size)?;
            cx.write_extent_bytes(paddr, target_bytes)?;
            for (k, v) in write::build_file_extent_records(oid, 0, size, paddr, bs) {
                cx.records.push((k, v));
            }
            let (ik, iv) = write::build_inode_record(
                oid,
                parent_oid,
                write::mode_lnk(mode),
                size,
                bs,
                mtime_ns,
            );
            cx.records.push((ik, iv));
            // Commit-1 stub: Plain/false. Commit-2 swaps in real values.
            let (dk, dv) = write::build_drec_record(
                parent_oid,
                &name,
                oid,
                DT_LNK,
                DrecKeyLayout::Plain,
                false,
            )?;
            cx.records.push((dk, dv));
            patch_inode_record(cx.records, parent_oid, |v| {
                if v.len() >= jrec::J_INODE_VAL_FIXED_SIZE {
                    let cur = i32::from_le_bytes(v[56..60].try_into().unwrap());
                    v[56..60].copy_from_slice(&(cur + 1).to_le_bytes());
                }
            });
            cx.note_new_symlink();
            Ok(())
        })
    }

    /// Set an embedded extended attribute on the inode at `path`. If
    /// the xattr already exists it's replaced; otherwise it's
    /// inserted. Refuses values larger than 3804 bytes
    /// (`APFS_XATTR_MAX_EMBEDDED_SIZE`) — dstream-backed xattrs are
    /// not yet implemented.
    pub fn set_xattr(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        name: &str,
        value: &[u8],
    ) -> Result<()> {
        let target_oid = self.resolve_path_to_oid(dev, path)?;
        rw::commit_with_mutator(self, dev, |cx| {
            // Drop any pre-existing record for the same (target_oid,
            // XATTR, name) so the replace is idempotent.
            remove_xattr_record(cx.records, target_oid, name);
            let (k, v) = write::build_xattr_record(target_oid, name, value)?;
            cx.records.push((k, v));
            Ok(())
        })
    }

    /// Remove an embedded extended attribute from the inode at
    /// `path`. No-op when the named xattr doesn't exist (mirrors
    /// `removexattr(2)` semantics on most platforms — we don't
    /// distinguish "no such attribute" from success here).
    pub fn remove_xattr(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        name: &str,
    ) -> Result<()> {
        let target_oid = self.resolve_path_to_oid(dev, path)?;
        rw::commit_with_mutator(self, dev, |cx| {
            remove_xattr_record(cx.records, target_oid, name);
            Ok(())
        })
    }

    /// Total container capacity in bytes (`nx_block_count * nx_block_size`).
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Container block size (`nx_block_size`) in bytes.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Volume name from the volume superblock.
    pub fn volume_name(&self) -> &str {
        &self.volume_name
    }

    // ---- internal walker helpers ----

    /// Resolve the parent inode + leaf name of a path. Errors when
    /// the path is "/" (which has no parent in our model) or any
    /// intermediate component is missing.
    fn resolve_parent_and_name(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<(u64, String)> {
        let comps: Vec<&str> = split_path(path);
        if comps.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "apfs: cannot operate on the root directory itself".into(),
            ));
        }
        let leaf = comps.last().unwrap().to_string();
        let mut cur = ROOT_DIR_INO;
        for part in &comps[..comps.len() - 1] {
            cur = self.find_drec_child(dev, cur, part)?.ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "apfs: no such entry {part:?} under {path:?}"
                ))
            })?;
        }
        Ok((cur, leaf))
    }

    /// Resolve an inode's dtype (DT_REG / DT_DIR / DT_LNK). Cheaper
    /// than a full record dump and used as a pre-flight check for
    /// link() (which refuses dirs without committing a checkpoint).
    fn lookup_inode_dtype(&self, dev: &mut dyn BlockDevice, oid: u64) -> Result<u16> {
        let rs = self.read_state()?;
        let target = FsKeyTarget {
            oid,
            kind: APFS_TYPE_INODE,
            tail: &[],
            drec_layout: rs.drec_layout,
        };
        let block_size = self.block_size;
        let mut ctx = rs.fs_ctx.borrow_mut();
        let mut scan = RangeScan::start(&rs.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })?;
        if let Some((_kb, vb)) = scan.next(&mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })? {
            let ino = InodeVal::decode(&vb)?;
            return Ok(match ino.mode & 0o170_000 {
                0o040_000 => DT_DIR,
                0o120_000 => DT_LNK,
                _ => DT_REG,
            });
        }
        Err(crate::Error::InvalidArgument(format!(
            "apfs: no inode record for oid {oid:#x}"
        )))
    }

    /// Read and decode the full inode record for `oid`.
    fn read_inode_val(&self, dev: &mut dyn BlockDevice, oid: u64) -> Result<InodeVal> {
        let rs = self.read_state()?;
        let target = FsKeyTarget {
            oid,
            kind: APFS_TYPE_INODE,
            tail: &[],
            drec_layout: rs.drec_layout,
        };
        let block_size = self.block_size;
        let mut ctx = rs.fs_ctx.borrow_mut();
        let mut scan = RangeScan::start(&rs.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })?;
        if let Some((_kb, vb)) = scan.next(&mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })? {
            return InodeVal::decode(&vb);
        }
        Err(crate::Error::InvalidArgument(format!(
            "apfs: no inode record for oid {oid:#x}"
        )))
    }

    /// Walk path components, resolving each name through its parent's
    /// directory records. Returns the target's object id.
    pub fn resolve_path_to_oid(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<u64> {
        let mut cur = ROOT_DIR_INO;
        for part in split_path(path) {
            cur = self.find_drec_child(dev, cur, part)?.ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "apfs: no such entry {part:?} under {path:?}"
                ))
            })?;
        }
        Ok(cur)
    }

    /// Find the child named `name` under directory inode `parent_oid`
    /// using a range scan over the drec records under that parent.
    fn find_drec_child(
        &self,
        dev: &mut dyn BlockDevice,
        parent_oid: u64,
        name: &str,
    ) -> Result<Option<u64>> {
        let rs = self.read_state()?;
        let layout = rs.drec_layout;
        let target = FsKeyTarget {
            oid: parent_oid,
            kind: APFS_TYPE_DIR_REC,
            tail: &[],
            drec_layout: layout,
        };
        let block_size = self.block_size;
        let mut ctx = rs.fs_ctx.borrow_mut();
        let mut scan = RangeScan::start(&rs.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })?;
        while let Some((kb, vb)) = scan.next(&mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })? {
            let key = match layout {
                DrecKeyLayout::Hashed => {
                    DrecKey::decode_hashed(&kb).or_else(|_| DrecKey::decode_plain(&kb))?
                }
                DrecKeyLayout::Plain => {
                    DrecKey::decode_plain(&kb).or_else(|_| DrecKey::decode_hashed(&kb))?
                }
            };
            if key.name == name {
                let val = DrecVal::decode(&vb)?;
                return Ok(Some(val.file_id));
            }
        }
        Ok(None)
    }

    /// List the entries inside `dir_oid` by range-scanning all drec
    /// records whose key shares the `(dir_oid, DIR_REC)` prefix.
    fn list_dir(
        &self,
        dev: &mut dyn BlockDevice,
        dir_oid: u64,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        use crate::fs::{DirEntry as FsDirEntry, EntryKind};
        let rs = self.read_state()?;
        let layout = rs.drec_layout;
        let target = FsKeyTarget {
            oid: dir_oid,
            kind: APFS_TYPE_DIR_REC,
            tail: &[],
            drec_layout: layout,
        };
        let block_size = self.block_size;
        let mut ctx = rs.fs_ctx.borrow_mut();
        let mut out = Vec::new();
        let mut scan = RangeScan::start(&rs.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })?;
        while let Some((kb, vb)) = scan.next(&mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })? {
            let key = match layout {
                DrecKeyLayout::Hashed => {
                    match DrecKey::decode_hashed(&kb).or_else(|_| DrecKey::decode_plain(&kb)) {
                        Ok(k) => k,
                        Err(_) => continue,
                    }
                }
                DrecKeyLayout::Plain => {
                    match DrecKey::decode_plain(&kb).or_else(|_| DrecKey::decode_hashed(&kb)) {
                        Ok(k) => k,
                        Err(_) => continue,
                    }
                }
            };
            let val = match DrecVal::decode(&vb) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let kind = match val.dtype() {
                DT_DIR => EntryKind::Dir,
                DT_REG => EntryKind::Regular,
                DT_LNK => EntryKind::Symlink,
                jrec::DT_FIFO => EntryKind::Fifo,
                jrec::DT_CHR => EntryKind::Char,
                jrec::DT_BLK => EntryKind::Block,
                jrec::DT_SOCK => EntryKind::Socket,
                _ => EntryKind::Unknown,
            };
            out.push(FsDirEntry {
                name: key.name,
                inode: val.file_id as u32,
                kind,
                size: 0,
            });
        }
        Ok(out)
    }

    /// Find the inode record for `oid` and return `(size, dstream_oid)`.
    /// Inode records have an empty type-specific tail; a single
    /// `(oid, APFS_TYPE_INODE)` range scan yields exactly the one we
    /// want (at most one record per oid in a fs-tree).
    fn lookup_inode_size(&self, dev: &mut dyn BlockDevice, oid: u64) -> Result<(u64, Option<u64>)> {
        let rs = self.read_state()?;
        let target = FsKeyTarget {
            oid,
            kind: APFS_TYPE_INODE,
            tail: &[],
            drec_layout: rs.drec_layout,
        };
        let block_size = self.block_size;
        let mut ctx = rs.fs_ctx.borrow_mut();
        let mut scan = RangeScan::start(&rs.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })?;
        if let Some((_kb, vb)) = scan.next(&mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })? {
            let ino = InodeVal::decode(&vb)?;
            const S_IFMT: u16 = 0o170_000;
            const S_IFREG: u16 = 0o100_000;
            const S_IFLNK: u16 = 0o120_000;
            let mt = ino.mode & S_IFMT;
            if mt != S_IFREG && mt != S_IFLNK {
                return Err(crate::Error::InvalidArgument(format!(
                    "apfs: oid {oid:#x} is not a regular file (mode {:#o})",
                    ino.mode
                )));
            }
            let size = ino.dstream.map(|d| d.size).unwrap_or(0);
            let dstream_oid = if ino.private_id != 0 && ino.private_id != oid {
                Some(ino.private_id)
            } else {
                None
            };
            return Ok((size, dstream_oid));
        }
        Err(crate::Error::InvalidArgument(format!(
            "apfs: no inode record for oid {oid:#x}"
        )))
    }

    /// Range-scan all `j_file_extent` records keyed under `dstream_oid`,
    /// returning them in ascending `logical_addr` order. The B-tree
    /// itself yields entries in sorted order; we sort defensively to
    /// tolerate badly written images.
    fn collect_extents(
        &self,
        dev: &mut dyn BlockDevice,
        dstream_oid: u64,
    ) -> Result<Vec<(u64, FileExtentVal)>> {
        let rs = self.read_state()?;
        let target = FsKeyTarget {
            oid: dstream_oid,
            kind: APFS_TYPE_FILE_EXTENT,
            tail: &[],
            drec_layout: rs.drec_layout,
        };
        let block_size = self.block_size;
        let mut ctx = rs.fs_ctx.borrow_mut();
        let mut out = Vec::new();
        let mut scan = RangeScan::start(&rs.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })?;
        while let Some((kb, vb)) = scan.next(&mut ctx, &mut |paddr, buf| {
            read_at_paddr(dev, paddr, block_size, buf)
        })? {
            let logical_addr = if kb.len() >= 16 {
                u64::from_le_bytes(kb[8..16].try_into().unwrap())
            } else {
                continue;
            };
            let val = match FileExtentVal::decode(&vb) {
                Ok(v) => v,
                Err(_) => continue,
            };
            out.push((logical_addr, val));
        }
        out.sort_by_key(|(la, _)| *la);
        Ok(out)
    }
}

/// `Filesystem` adapter for APFS. An [`Apfs`] opened via
/// [`Apfs::open`] is read-only — mutation calls return `Unsupported`.
/// An [`Apfs`] returned by [`Apfs::format`] is in pending-write mode:
/// `create_file` / `create_dir` / `create_symlink` buffer operations
/// in memory and `flush` drains them into a fresh image through
/// [`write::ApfsWriter`]. After `flush` the [`Apfs`] is in read mode
/// and behaves like a freshly-opened image.
impl crate::fs::Filesystem for Apfs {
    fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        src: crate::fs::FileSource,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        // Buffer the source bytes — both code paths need to know the
        // file size up front (PendingWrite to stage the op, Write to
        // bump-allocate an extent). APFS' streaming write happens
        // inside the writer; at this layer we materialise the body
        // first.
        let (mut reader, size) = src
            .open()
            .map_err(|e| crate::Error::Io(std::io::Error::other(e)))?;
        let mut data = Vec::with_capacity(size.min(64 * 1024 * 1024) as usize);
        let n = std::io::Read::read_to_end(&mut reader, &mut data)
            .map_err(|e| crate::Error::Io(std::io::Error::other(e)))?;
        if n as u64 != size {
            // Pad with zeros so dstream.size still matches what the
            // user asked for; mirrors the writer's truncation handling.
            data.resize(size as usize, 0);
        }
        drop(reader);
        match &self.state {
            ApfsState::Read(_) => Err(crate::Error::Unsupported(
                "apfs: create_file on a read-only image (use Apfs::open_writable for re-opens)"
                    .into(),
            )),
            ApfsState::Write(_) => {
                let path_str = path
                    .to_str()
                    .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
                self.create_file_at(dev, path_str, &data, meta.mode, meta_mtime_ns(&meta))
            }
            ApfsState::PendingWrite(_) => {
                let pw = pending_write_mut(&mut self.state)?;
                let (parent_oid, name) = pw.resolve_parent(path)?;
                pw.ops.push(PendingOp::File {
                    parent_oid,
                    name,
                    mode: meta.mode,
                    mtime_ns: meta_mtime_ns(&meta),
                    data,
                });
                // Consume an oid slot so future creates see the same
                // sequence the writer will use. add_file_from_reader
                // allocates one oid.
                pw.next_oid = pw.next_oid.saturating_add(1);
                Ok(())
            }
        }
    }

    fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        match &self.state {
            ApfsState::Read(_) => Err(crate::Error::Unsupported(
                "apfs: create_dir on a read-only image".into(),
            )),
            ApfsState::Write(_) => {
                let path_str = path
                    .to_str()
                    .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
                self.create_dir_at(dev, path_str, meta.mode, meta_mtime_ns(&meta))
            }
            ApfsState::PendingWrite(_) => {
                let pw = pending_write_mut(&mut self.state)?;
                let (parent_oid, name) = pw.resolve_parent(path)?;
                // Assign a deterministic oid that matches what the
                // writer will hand out for this position in the call
                // sequence, and remember it so nested children of this
                // directory can resolve their parent path later.
                let new_oid = pw.next_oid;
                pw.next_oid = pw.next_oid.saturating_add(1);
                pw.dir_oid.insert(path.to_path_buf(), new_oid);
                pw.ops.push(PendingOp::Dir {
                    parent_oid,
                    name,
                    mode: meta.mode,
                    mtime_ns: meta_mtime_ns(&meta),
                });
                Ok(())
            }
        }
    }

    fn create_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        target: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let target_str = target
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 symlink target".into()))?
            .to_string();
        match &self.state {
            ApfsState::Read(_) => Err(crate::Error::Unsupported(
                "apfs: create_symlink on a read-only image".into(),
            )),
            ApfsState::Write(_) => {
                let path_str = path
                    .to_str()
                    .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
                self.create_symlink_at(dev, path_str, &target_str, meta.mode, meta_mtime_ns(&meta))
            }
            ApfsState::PendingWrite(_) => {
                let pw = pending_write_mut(&mut self.state)?;
                let (parent_oid, name) = pw.resolve_parent(path)?;
                pw.ops.push(PendingOp::Symlink {
                    parent_oid,
                    name,
                    mode: meta.mode,
                    mtime_ns: meta_mtime_ns(&meta),
                    target: target_str,
                });
                pw.next_oid = pw.next_oid.saturating_add(1);
                Ok(())
            }
        }
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
        // APFS supports DT_CHR / DT_BLK / DT_FIFO / DT_SOCK via the
        // rdev xfield, but neither the single-pass writer nor the
        // Write-state mutators emit one yet. Use cases are rare; this
        // is out of scope for the current mutation surface.
        Err(crate::Error::Unsupported(
            "apfs: device nodes are not supported by the writer".into(),
        ))
    }

    fn remove(&mut self, dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<()> {
        match &self.state {
            ApfsState::Write(_) => {
                let s = path
                    .to_str()
                    .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
                self.remove_path(dev, s)
            }
            _ => Err(crate::Error::Unsupported(
                "apfs: remove requires Apfs::open_writable (PendingWrite is single-pass; \
                 Read is immutable)"
                    .into(),
            )),
        }
    }

    fn rename(
        &mut self,
        dev: &mut dyn BlockDevice,
        old_path: &std::path::Path,
        new_path: &std::path::Path,
    ) -> Result<()> {
        match &self.state {
            ApfsState::Write(_) => {
                let old = old_path.to_str().ok_or_else(|| {
                    crate::Error::InvalidArgument("apfs: non-UTF-8 rename source".into())
                })?;
                let new = new_path.to_str().ok_or_else(|| {
                    crate::Error::InvalidArgument("apfs: non-UTF-8 rename target".into())
                })?;
                Apfs::rename(self, dev, old, new)
            }
            _ => Err(crate::Error::Unsupported(
                "apfs: rename requires Apfs::open_writable".into(),
            )),
        }
    }

    fn hardlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        target_path: &std::path::Path,
        new_path: &std::path::Path,
    ) -> Result<()> {
        match &self.state {
            ApfsState::Write(_) => {
                let tgt = target_path.to_str().ok_or_else(|| {
                    crate::Error::InvalidArgument("apfs: non-UTF-8 hardlink target".into())
                })?;
                let new = new_path.to_str().ok_or_else(|| {
                    crate::Error::InvalidArgument("apfs: non-UTF-8 hardlink path".into())
                })?;
                self.link(dev, tgt, new)
            }
            _ => Err(crate::Error::Unsupported(
                "apfs: hardlink requires Apfs::open_writable".into(),
            )),
        }
    }

    fn set_attrs(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        attrs: crate::fs::SetAttrs,
    ) -> Result<()> {
        if !matches!(&self.state, ApfsState::Write(_)) {
            return Err(crate::Error::Unsupported(
                "apfs: set_attrs requires Apfs::open_writable".into(),
            ));
        }
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
        let target_oid = self.resolve_path_to_oid(dev, s)?;
        // Bundle every requested field into one commit_with_mutator
        // closure so the multi-field case costs one checkpoint, not
        // three. Layout offsets per j_inode_val_t: mode 80..82
        // (preserve type bits), uid 72..76, gid 76..80, mtime 24..32,
        // ctime 32..40, atime 40..48 (all u64 ns; the trait's
        // SetAttrs gives u32 epoch seconds, so multiply by 1e9).
        const NS_PER_SEC: u64 = 1_000_000_000;
        rw::commit_with_mutator(self, dev, |cx| {
            patch_inode_record(cx.records, target_oid, |v| {
                if v.len() < jrec::J_INODE_VAL_FIXED_SIZE {
                    return;
                }
                if let Some(mode) = attrs.mode {
                    let cur = u16::from_le_bytes(v[80..82].try_into().unwrap());
                    let new = (cur & 0xF000) | (mode & 0o7777);
                    v[80..82].copy_from_slice(&new.to_le_bytes());
                }
                if let Some(uid) = attrs.uid {
                    v[72..76].copy_from_slice(&uid.to_le_bytes());
                }
                if let Some(gid) = attrs.gid {
                    v[76..80].copy_from_slice(&gid.to_le_bytes());
                }
                if let Some(mtime) = attrs.mtime {
                    let ns = mtime as u64 * NS_PER_SEC;
                    v[24..32].copy_from_slice(&ns.to_le_bytes());
                }
                if let Some(ctime) = attrs.ctime {
                    let ns = ctime as u64 * NS_PER_SEC;
                    v[32..40].copy_from_slice(&ns.to_le_bytes());
                }
                if let Some(atime) = attrs.atime {
                    let ns = atime as u64 * NS_PER_SEC;
                    v[40..48].copy_from_slice(&ns.to_le_bytes());
                }
            });
            Ok(())
        })
    }

    fn truncate(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        new_size: u64,
    ) -> Result<()> {
        if !matches!(&self.state, ApfsState::Write(_)) {
            return Err(crate::Error::Unsupported(
                "apfs: truncate requires Apfs::open_writable".into(),
            ));
        }
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
        // Route through the rw handle's set_len + sync — that
        // pathway already handles the COW checkpoint, dstream
        // resize, and extent rewrite. Shrinking truncates the file
        // body; growing extends with zeros.
        let mut h = rw::ApfsFileHandle::open(self, dev, s, crate::fs::OpenFlags::default())?;
        crate::fs::FileHandle::set_len(&mut h, new_size)?;
        crate::fs::FileHandle::sync(&mut h)?;
        Ok(())
    }

    fn list_xattrs(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::XattrPair>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
        // Apfs::read_xattrs returns a HashMap; flatten into the
        // trait's Vec<XattrPair> in lexicographic name order for
        // a stable listing.
        let map = self.read_xattrs(dev, s)?;
        let mut pairs: Vec<_> = map
            .into_iter()
            .map(|(name, value)| crate::fs::XattrPair { name, value })
            .collect();
        pairs.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(pairs)
    }

    fn set_xattr(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        name: &str,
        value: &[u8],
    ) -> Result<()> {
        if !matches!(&self.state, ApfsState::Write(_)) {
            return Err(crate::Error::Unsupported(
                "apfs: set_xattr requires Apfs::open_writable".into(),
            ));
        }
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
        Apfs::set_xattr(self, dev, s, name, value)
    }

    fn remove_xattr(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        name: &str,
    ) -> Result<()> {
        if !matches!(&self.state, ApfsState::Write(_)) {
            return Err(crate::Error::Unsupported(
                "apfs: remove_xattr requires Apfs::open_writable".into(),
            ));
        }
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
        Apfs::remove_xattr(self, dev, s, name)
    }

    fn list(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
        Apfs::list_path(self, dev, s)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn std::io::Read + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
        let r = self.open_file_reader(dev, s)?;
        Ok(Box::new(r))
    }

    fn getattr(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<crate::fs::FileAttrs> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
        let oid = self.resolve_path_to_oid(dev, s)?;
        let ino = self.read_inode_val(dev, oid)?;
        let kind = match ino.mode & 0o170_000 {
            0o100_000 => crate::fs::EntryKind::Regular,
            0o040_000 => crate::fs::EntryKind::Dir,
            0o120_000 => crate::fs::EntryKind::Symlink,
            0o020_000 => crate::fs::EntryKind::Char,
            0o060_000 => crate::fs::EntryKind::Block,
            0o010_000 => crate::fs::EntryKind::Fifo,
            0o140_000 => crate::fs::EntryKind::Socket,
            _ => crate::fs::EntryKind::Regular,
        };
        let size = ino.dstream.map(|d| d.size).unwrap_or(0);
        // APFS timestamps are nanoseconds since the Unix epoch.
        let to_secs = |ns: u64| (ns / 1_000_000_000) as u32;
        let nlink = if kind == crate::fs::EntryKind::Dir {
            2
        } else {
            ino.nchildren_or_nlink.max(1) as u32
        };
        Ok(crate::fs::FileAttrs {
            kind,
            mode: ino.mode & 0o7777,
            uid: ino.owner,
            gid: ino.group,
            size,
            blocks: size.div_ceil(512),
            nlink,
            atime: to_secs(ino.access_time),
            mtime: to_secs(ino.mod_time),
            ctime: to_secs(ino.change_time),
            rdev: 0,
            inode: oid as u32,
        })
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn crate::fs::FileReadHandle + 'a>> {
        // `open_file_reader` already gates on `read_state()` — it
        // returns `Unsupported` when we're in PendingWrite mode.
        // The ApfsFileReader is Read+Seek+FileReadHandle by virtue
        // of the impls just above.
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
        let h = self.open_file_reader(dev, s)?;
        Ok(Box::new(h))
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        // Take ownership of the pending buffer so we can drop our
        // borrow on `self.state` before re-opening below.
        let pw = match std::mem::replace(
            &mut self.state,
            ApfsState::PendingWrite(PendingWrite {
                total_blocks: 0,
                dir_oid: std::collections::HashMap::new(),
                ops: Vec::new(),
                next_oid: 16,
            }),
        ) {
            ApfsState::Read(r) => {
                // Nothing to flush — restore the read view we just took
                // out and return Ok.
                self.state = ApfsState::Read(r);
                return Ok(());
            }
            ApfsState::Write(w) => {
                // Open-file handles have already sync'd on Drop; nothing
                // to do here. Restore the write state we just swapped
                // out and return Ok.
                self.state = ApfsState::Write(w);
                return Ok(());
            }
            ApfsState::PendingWrite(p) => p,
        };
        // Materialise the image via ApfsWriter, replaying our buffered
        // operations in order.
        let block_size = self.block_size;
        let volume_name = self.volume_name.clone();
        {
            let mut w = write::ApfsWriter::new(dev, pw.total_blocks, block_size, &volume_name)?;
            for op in pw.ops {
                match op {
                    PendingOp::Dir {
                        parent_oid,
                        name,
                        mode,
                        mtime_ns,
                    } => {
                        w.add_dir_at_time(parent_oid, &name, mode, mtime_ns)?;
                    }
                    PendingOp::File {
                        parent_oid,
                        name,
                        mode,
                        mtime_ns,
                        data,
                    } => {
                        let len = data.len() as u64;
                        let mut r = std::io::Cursor::new(data);
                        w.add_file_from_reader_at_time(
                            parent_oid, &name, mode, &mut r, len, mtime_ns,
                        )?;
                    }
                    PendingOp::Symlink {
                        parent_oid,
                        name,
                        mode,
                        mtime_ns,
                        target,
                    } => {
                        w.add_symlink_at_time(parent_oid, &name, mode, &target, mtime_ns)?;
                    }
                }
            }
            w.finish()?;
        }
        // Re-parse the just-written image into read state. We open the
        // single volume we just wrote and adopt its state.
        let fresh = Apfs::open(dev)?;
        // Take fresh's state. fresh's other fields (block_size, etc.)
        // must already match ours since they describe the same image.
        debug_assert_eq!(fresh.block_size, self.block_size);
        debug_assert_eq!(fresh.total_bytes, self.total_bytes);
        self.state = fresh.state;
        self.volume_name = fresh.volume_name;
        Ok(())
    }

    fn mutation_capability(&self) -> crate::fs::MutationCapability {
        match &self.state {
            // In pending-write mode the writer is single-pass / append-
            // only with no remove or partial-write hooks. WholeFileOnly
            // is the closest fit ("can add whole files, can't patch").
            ApfsState::PendingWrite(_) => crate::fs::MutationCapability::WholeFileOnly,
            // Once flushed via `Apfs::open` the image is sealed — the
            // writer can't re-open and mutate. Mirrors ISO 9660 / SquashFS.
            ApfsState::Read(_) => crate::fs::MutationCapability::Immutable,
            // After `Apfs::open_writable` we can patch existing files'
            // bytes via `open_file_rw`. Create / remove of whole files
            // isn't wired yet, but partial writes work, so `Mutable`
            // is the closest fit.
            ApfsState::Write(_) => crate::fs::MutationCapability::Mutable,
        }
    }

    fn open_file_rw<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
        flags: crate::fs::OpenFlags,
        meta: Option<crate::fs::FileMeta>,
    ) -> Result<Box<dyn crate::fs::FileHandle + 'a>> {
        let _ = meta;
        if flags.create {
            return Err(crate::Error::Unsupported(
                "apfs: open_file_rw with create=true is not supported (open_writable \
                 only edits existing files)"
                    .into(),
            ));
        }
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 path".into()))?;
        let h = rw::ApfsFileHandle::open(self, dev, s, flags)?;
        Ok(Box::new(h))
    }
}

impl crate::fs::FilesystemFactory for Apfs {
    type FormatOpts = ApfsFormatOpts;

    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Apfs::format(dev, opts.total_blocks, opts.block_size, &opts.volume_name)
    }

    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Apfs::open(dev)
    }
}

/// Format options for [`Apfs::format`] via the [`crate::fs::FilesystemFactory`]
/// surface. Mirrors the positional arguments on the inherent
/// `Apfs::format` method.
#[derive(Debug, Clone)]
pub struct ApfsFormatOpts {
    /// Image size in blocks. Must satisfy
    /// `total_blocks * block_size <= dev.total_size()`.
    pub total_blocks: u64,
    /// Block size in bytes. Power of two between 512 and 65 536.
    /// 4096 is the conventional APFS value.
    pub block_size: u32,
    /// Volume label written into the APSB.
    pub volume_name: String,
}

impl Default for ApfsFormatOpts {
    /// Sensible defaults: 64 blocks × 4096 bytes = 256 KiB image,
    /// volume named "APFS".
    fn default() -> Self {
        Self {
            total_blocks: 64,
            block_size: 4096,
            volume_name: "APFS".to_string(),
        }
    }
}

impl ApfsFormatOpts {
    /// Apply a generic option-bag (CLI `-O key=val` / TOML
    /// `[filesystem.options]`) on top of these opts. Unknown keys are
    /// left in the map for the caller to flag.
    pub fn apply_options(&mut self, map: &mut crate::format_opts::OptionMap) -> crate::Result<()> {
        if let Some(sz) = map.take_size("block_size")? {
            self.block_size = sz as u32;
        }
        if let Some(n) = map.take_u64("total_blocks")? {
            self.total_blocks = n;
        }
        if let Some(s) = map.take_str("volume_name") {
            self.volume_name = s;
        }
        // Accept "volume_label" as a synonym so the same CLI key works
        // across filesystems.
        if let Some(s) = map.take_str("volume_label") {
            self.volume_name = s;
        }
        Ok(())
    }
}

/// Borrow the [`PendingWrite`] inside `state`, or return an error if
/// the [`Apfs`] is in read or write mode.
fn pending_write_mut(state: &mut ApfsState) -> Result<&mut PendingWrite> {
    match state {
        ApfsState::PendingWrite(p) => Ok(p),
        ApfsState::Read(_) => Err(crate::Error::Unsupported(
            "apfs: filesystem has already been flushed; mutation after flush is not supported"
                .into(),
        )),
        ApfsState::Write(_) => Err(crate::Error::Unsupported(
            "apfs: filesystem is open for in-place writes (open_file_rw); \
             create_* / remove are not supported in this mode"
                .into(),
        )),
    }
}

impl PendingWrite {
    /// Split `path` into `(parent_oid, leaf_name)`. The parent must
    /// already exist in `dir_oid` (created via `create_dir`, or "/")
    /// — implicit parent creation is not supported.
    fn resolve_parent(&self, path: &std::path::Path) -> Result<(u64, String)> {
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("/"));
        let parent_buf = if parent.as_os_str().is_empty() {
            std::path::PathBuf::from("/")
        } else {
            parent.to_path_buf()
        };
        let leaf = path
            .file_name()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: empty leaf name".into()))?
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 leaf name".into()))?
            .to_string();
        let parent_oid = *self.dir_oid.get(&parent_buf).ok_or_else(|| {
            crate::Error::InvalidArgument(format!(
                "apfs: parent directory {:?} not found; call create_dir() for it first",
                parent_buf
            ))
        })?;
        Ok((parent_oid, leaf))
    }
}

/// Read a physical block (by block number) from `dev` into `buf`.
fn read_at_paddr(
    dev: &mut dyn BlockDevice,
    paddr: u64,
    block_size: u32,
    buf: &mut [u8],
) -> Result<()> {
    let off = paddr.saturating_mul(block_size as u64);
    dev.read_at(off, buf)
}

/// Streaming reader over an APFS regular file. Walks the cached extent
/// list, reading one extent's bytes at a time. Sparse extents
/// (`phys_block_num == 0`) yield zero bytes.
pub struct ApfsFileReader<'a> {
    dev: &'a mut dyn BlockDevice,
    block_size: u32,
    /// `(logical_addr, extent)` pairs sorted by logical_addr.
    extents: Vec<(u64, FileExtentVal)>,
    /// Logical file size (from `j_dstream.size`).
    size: u64,
    /// Logical cursor in the file.
    cursor: u64,
}

impl<'a> std::io::Seek for ApfsFileReader<'a> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let target_i128 = match pos {
            std::io::SeekFrom::Start(n) => n as i128,
            std::io::SeekFrom::Current(d) => self.cursor as i128 + d as i128,
            std::io::SeekFrom::End(d) => self.size as i128 + d as i128,
        };
        if target_i128 < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "apfs: seek to negative offset",
            ));
        }
        // Clamp past-EOF so subsequent reads return 0 instead of
        // reading random bytes past the end of the file's extents.
        self.cursor = (target_i128 as u128).min(self.size as u128) as u64;
        Ok(self.cursor)
    }
}

impl<'a> crate::fs::FileReadHandle for ApfsFileReader<'a> {
    fn len(&self) -> u64 {
        self.size
    }
}

impl<'a> std::io::Read for ApfsFileReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.cursor >= self.size || buf.is_empty() {
            return Ok(0);
        }
        // Find the extent that covers `cursor`. We scan linearly; real
        // files have a small number of extents.
        let mut covering: Option<(u64, FileExtentVal)> = None;
        for &(la, ev) in &self.extents {
            if la <= self.cursor && self.cursor < la + ev.length {
                covering = Some((la, ev));
                break;
            }
        }
        // No extent covers this range — treat as a sparse hole and zero up
        // to the next extent boundary (or EOF).
        let (la, ev) = match covering {
            Some(pair) => pair,
            None => {
                let next = self
                    .extents
                    .iter()
                    .map(|(la, _)| *la)
                    .find(|&la| la > self.cursor)
                    .unwrap_or(self.size);
                let want = (next.min(self.size) - self.cursor).min(buf.len() as u64) as usize;
                buf[..want].fill(0);
                self.cursor += want as u64;
                return Ok(want);
            }
        };
        let off_in_extent = self.cursor - la;
        let avail_in_extent = ev.length - off_in_extent;
        let want = avail_in_extent
            .min(buf.len() as u64)
            .min(self.size - self.cursor) as usize;
        if ev.phys_block_num == 0 {
            // Sparse extent — zero bytes.
            buf[..want].fill(0);
        } else {
            let abs_off = ev.phys_block_num * self.block_size as u64 + off_in_extent;
            self.dev
                .read_at(abs_off, &mut buf[..want])
                .map_err(std::io::Error::other)?;
        }
        self.cursor += want as u64;
        Ok(want)
    }
}

/// Walk the dumped fs-tree record list, find the `INODE` record for
/// `target_oid`, and run `patcher` against its value bytes. The
/// patcher mutates in place — typically to flip mode bits, owner,
/// group, or timestamps. Used by [`Apfs::chmod`], [`Apfs::chown`],
/// [`Apfs::set_times`], and the dirent/hardlink mutators.
pub(crate) fn patch_inode_record<F>(records: &mut [(Vec<u8>, Vec<u8>)], target_oid: u64, patcher: F)
where
    F: FnOnce(&mut Vec<u8>),
{
    for (k, v) in records.iter_mut() {
        if k.len() < 8 {
            continue;
        }
        let hdr = u64::from_le_bytes(k[0..8].try_into().unwrap());
        let oid = hdr & OBJ_ID_MASK;
        let kind = (hdr >> OBJ_TYPE_SHIFT) as u8;
        if oid == target_oid && kind == APFS_TYPE_INODE {
            patcher(v);
            return;
        }
    }
}

/// Find the drec record under `parent_oid` with `name`, returning
/// `(target_oid, dtype)`. Uses the plain-layout drec key format
/// (`hdr(8) + name_len(2) + name + NUL`); hashed-layout images
/// already get normalized to plain when our writer emits them.
pub(crate) fn find_drec(
    records: &[(Vec<u8>, Vec<u8>)],
    parent_oid: u64,
    name: &str,
) -> Option<(u64, u16)> {
    for (k, v) in records {
        if k.len() < 10 {
            continue;
        }
        let hdr = u64::from_le_bytes(k[0..8].try_into().unwrap());
        let oid = hdr & OBJ_ID_MASK;
        let kind = (hdr >> OBJ_TYPE_SHIFT) as u8;
        if oid != parent_oid || kind != APFS_TYPE_DIR_REC {
            continue;
        }
        let nlen = u16::from_le_bytes(k[8..10].try_into().unwrap()) as usize;
        if k.len() < 10 + nlen || nlen == 0 {
            continue;
        }
        // The name is stored with a trailing NUL inside `nlen`.
        let stored = &k[10..10 + nlen - 1];
        if stored != name.as_bytes() {
            continue;
        }
        if v.len() < 18 {
            return None;
        }
        let target_oid = u64::from_le_bytes(v[0..8].try_into().unwrap());
        let dtype = u16::from_le_bytes(v[16..18].try_into().unwrap());
        return Some((target_oid, dtype));
    }
    None
}

/// Remove the drec under `parent_oid` matching `name`. No-op when
/// not found.
pub(crate) fn remove_drec(records: &mut Vec<(Vec<u8>, Vec<u8>)>, parent_oid: u64, name: &str) {
    records.retain(|(k, _)| {
        if k.len() < 10 {
            return true;
        }
        let hdr = u64::from_le_bytes(k[0..8].try_into().unwrap());
        let oid = hdr & OBJ_ID_MASK;
        let kind = (hdr >> OBJ_TYPE_SHIFT) as u8;
        if oid != parent_oid || kind != APFS_TYPE_DIR_REC {
            return true;
        }
        let nlen = u16::from_le_bytes(k[8..10].try_into().unwrap()) as usize;
        if k.len() < 10 + nlen || nlen == 0 {
            return true;
        }
        let stored = &k[10..10 + nlen - 1];
        stored != name.as_bytes()
    });
}

/// Count direct children of a directory inode by scanning drec
/// records keyed under `parent_oid`. Used by `remove_path` to
/// reject non-empty directories.
pub(crate) fn drec_count_for(records: &[(Vec<u8>, Vec<u8>)], parent_oid: u64) -> usize {
    let mut n = 0usize;
    for (k, _) in records {
        if k.len() < 8 {
            continue;
        }
        let hdr = u64::from_le_bytes(k[0..8].try_into().unwrap());
        let oid = hdr & OBJ_ID_MASK;
        let kind = (hdr >> OBJ_TYPE_SHIFT) as u8;
        if oid == parent_oid && kind == APFS_TYPE_DIR_REC {
            n += 1;
        }
    }
    n
}

/// Remove an XATTR record with key `(target_oid, XATTR, name)` if it
/// exists. The xattr's key shape (per `build_xattr_record`) is:
/// `j_key_t[0..8] | name_len:u16[8..10] | name_bytes | NUL` — so the
/// 10..(10+name.len()) range is the UTF-8 name. Used by both
/// `Apfs::set_xattr` (drop-then-replace) and `Apfs::remove_xattr`.
pub(crate) fn remove_xattr_record(
    records: &mut Vec<(Vec<u8>, Vec<u8>)>,
    target_oid: u64,
    name: &str,
) {
    let name_bytes = name.as_bytes();
    records.retain(|(k, _)| {
        if k.len() < 10 + name_bytes.len() + 1 {
            return true;
        }
        let hdr = u64::from_le_bytes(k[0..8].try_into().unwrap());
        let oid = hdr & OBJ_ID_MASK;
        let kind = (hdr >> OBJ_TYPE_SHIFT) as u8;
        if oid != target_oid || kind != APFS_TYPE_XATTR {
            return true;
        }
        // Key carries a trailing NUL; ensure the name + NUL matches.
        &k[10..10 + name_bytes.len()] != name_bytes || k[10 + name_bytes.len()] != 0
    });
}

/// Drop every record keyed on `target_oid` — inode + dstream_id +
/// file_extent + xattr — when the last hardlink to a file (or any
/// dir) has been removed. The underlying data blocks stay on disk
/// (APFS COW; nothing ever "frees"). Subsequent checkpoints will
/// re-emit the surviving record set without these.
pub(crate) fn remove_all_records_for_oid(records: &mut Vec<(Vec<u8>, Vec<u8>)>, target_oid: u64) {
    records.retain(|(k, _)| {
        if k.len() < 8 {
            return true;
        }
        let hdr = u64::from_le_bytes(k[0..8].try_into().unwrap());
        let oid = hdr & OBJ_ID_MASK;
        oid != target_oid
    });
}

/// Probe for the APFS container superblock magic `"NXSB"` at offset
/// 32 of LBA 0 (block 0 is the container superblock; its `nx_magic`
/// field lives at offset 32 in the `nx_superblock_t` layout).
pub fn probe(dev: &mut dyn BlockDevice) -> Result<bool> {
    if dev.total_size() < 64 {
        return Ok(false);
    }
    let mut head = [0u8; 64];
    dev.read_at(0, &mut head)?;
    Ok(&head[32..36] == b"NXSB")
}

// ---- small free helpers below ----

/// Load the container-level state required to (re-)open any volume on
/// `dev`: the live NXSB plus its already-loaded container omap root.
fn load_container(dev: &mut dyn BlockDevice) -> Result<ContainerCtx> {
    // ---- Container superblock at block 0 ----
    let mut block0 = vec![0u8; 4096];
    dev.read_at(0, &mut block0)?;
    let label_sb = NxSuperblock::decode(&block0)?;
    let block_size = label_sb.block_size;
    if block_size == 0 || block_size > 65_536 || !block_size.is_power_of_two() {
        return Err(crate::Error::InvalidImage(format!(
            "apfs: nx_block_size {block_size} is not a sensible power of two"
        )));
    }

    // Re-read block 0 at the real block size in case it differs.
    let mut block0 = vec![0u8; block_size as usize];
    dev.read_at(0, &mut block0)?;
    let label_sb = NxSuperblock::decode(&block0)?;

    // ---- Walk the checkpoint descriptor area for the live NXSB ----
    let (live_sb, live_xp_desc_offset) = match find_live_nxsb(dev, &label_sb, block_size)? {
        Some((sb, off)) => (sb, Some(off)),
        None => (label_sb.clone(), None),
    };

    let total_bytes = live_sb.block_count.saturating_mul(block_size as u64);

    // ---- Container omap ----
    let omap_phys = read_object::<OmapPhys>(dev, live_sb.omap_oid, block_size, OmapPhys::decode)?;
    let mut omap_root_block = vec![0u8; block_size as usize];
    dev.read_at(
        omap_phys.tree_oid.saturating_mul(block_size as u64),
        &mut omap_root_block,
    )?;

    Ok(ContainerCtx {
        live_sb,
        omap_root: omap_root_block,
        block_size,
        total_bytes,
        live_xp_desc_offset,
    })
}

/// Resolve the first populated `nx_fs_oid[]` slot in `ctx` to its APSB
/// physical block. Returns `(slot_index, paddr)`.
fn find_volume_paddr(dev: &mut dyn BlockDevice, ctx: &ContainerCtx) -> Result<(usize, u64)> {
    let vol_index = ctx
        .live_sb
        .fs_oid
        .iter()
        .position(|&o| o != 0)
        .ok_or_else(|| crate::Error::InvalidImage("apfs: container has no volumes".into()))?;
    let vol_oid = ctx.live_sb.fs_oid[vol_index];
    let target_xid = ctx.live_sb.obj.xid;
    let mut dev_reader = DevReader {
        dev,
        block_size: ctx.block_size,
    };
    let val = omap_lookup(&ctx.omap_root, vol_oid, target_xid, &mut |paddr, buf| {
        dev_reader.read(paddr, buf)
    })?
    .ok_or_else(|| {
        crate::Error::InvalidImage(format!(
            "apfs: container omap has no entry for volume oid {vol_oid:#x}"
        ))
    })?;
    Ok((vol_index, val.paddr))
}

/// Read the spaceman_phys_t and return the bump-allocator high-water mark
/// (= `total_blocks - main_free_count`). Returns `None` if the spaceman
/// can't be parsed, which is non-fatal because we have a conservative
/// fallback.
fn read_spaceman_high_water(dev: &mut dyn BlockDevice, ctx: &ContainerCtx) -> Option<u64> {
    let mut block = vec![0u8; ctx.block_size as usize];
    let off = write::SPACEMAN_PADDR * ctx.block_size as u64;
    if dev.read_at(off, &mut block).is_err() {
        return None;
    }
    // Inline decode of the relevant spaceman_phys_t fields. We avoid the
    // full `decode_spaceman()` helper because it's gated behind `cfg(test)`.
    // Verify the object type bits match spaceman before trusting the layout.
    let otype = u32::from_le_bytes(block[24..28].try_into().ok()?);
    if otype & 0x0000_ffff != spaceman::OBJECT_TYPE_SPACEMAN {
        return None;
    }
    let main_block_count = u64::from_le_bytes(block[48..56].try_into().ok()?);
    let main_free_count = u64::from_le_bytes(block[72..80].try_into().ok()?);
    Some(main_block_count - main_free_count)
}

/// Walk the checkpoint descriptor area looking for an NXSB whose xid is
/// strictly larger than the label NXSB's xid. Returns the chosen super-
/// block along with the slot offset (0-based, relative to
/// `xp_desc_base`) it lived at — or `None` if the label is already the
/// best. The slot offset is used by [`Apfs::open_writable`] to ring-
/// advance to the next free slot.
fn find_live_nxsb(
    dev: &mut dyn BlockDevice,
    label: &NxSuperblock,
    block_size: u32,
) -> Result<Option<(NxSuperblock, u64)>> {
    let n = label.xp_desc_blocks as u64;
    let base = label.xp_desc_base;
    let mut best: Option<(NxSuperblock, u64)> = None;
    let mut buf = vec![0u8; block_size as usize];
    for i in 0..n {
        let paddr = base.saturating_add(i);
        let off = paddr.saturating_mul(block_size as u64);
        if off + block_size as u64 > dev.total_size() {
            continue;
        }
        dev.read_at(off, &mut buf)?;
        // Quick check for NXSB magic before full decode.
        if buf.len() < 36 || &buf[32..36] != b"NXSB".as_slice() {
            // Could be a checkpoint_map_phys_t — skip.
            // Magic comparison: NX_MAGIC LE
            let mw = u32::from_le_bytes(buf[32..36].try_into().unwrap_or([0; 4]));
            if mw != NX_MAGIC {
                continue;
            }
        }
        let sb = match NxSuperblock::decode(&buf) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let better = match &best {
            None => sb.obj.xid >= label.obj.xid,
            Some((b, _)) => sb.obj.xid > b.obj.xid,
        };
        if better {
            best = Some((sb, i));
        }
    }
    Ok(best)
}

/// Read the block at physical address `paddr` and decode it via `decode`.
fn read_object<T>(
    dev: &mut dyn BlockDevice,
    paddr: u64,
    block_size: u32,
    decode: impl Fn(&[u8]) -> Result<T>,
) -> Result<T> {
    let mut buf = vec![0u8; block_size as usize];
    let off = paddr.checked_mul(block_size as u64).ok_or_else(|| {
        crate::Error::InvalidImage(format!("apfs: paddr {paddr} overflows when multiplied"))
    })?;
    let end = off.checked_add(block_size as u64).ok_or_else(|| {
        crate::Error::InvalidImage(format!("apfs: paddr {paddr} +block size overflows"))
    })?;
    if end > dev.total_size() {
        return Err(crate::Error::InvalidImage(format!(
            "apfs: object paddr {paddr} out of device bounds"
        )));
    }
    dev.read_at(off, &mut buf)?;
    decode(&buf)
}

/// Helper that wraps a `BlockDevice` + block size for the omap lookup
/// callback. Lookups are by physical block number.
struct DevReader<'a> {
    dev: &'a mut dyn BlockDevice,
    block_size: u32,
}

impl<'a> DevReader<'a> {
    fn read(&mut self, paddr: u64, buf: &mut [u8]) -> Result<()> {
        let off = paddr.saturating_mul(self.block_size as u64);
        self.dev.read_at(off, buf)
    }
}

/// Split an absolute or relative POSIX path into non-empty components,
/// ignoring `.` segments.
fn split_path(path: &str) -> Vec<&str> {
    path.split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;
    use crate::fs::apfs::obj::OBJECT_TYPE_NX_SUPERBLOCK;

    /// `probe` returns false for a blank device.
    #[test]
    fn probe_blank_device_false() {
        let mut dev = MemoryBackend::new(8192);
        assert!(!probe(&mut dev).unwrap());
    }

    /// `probe` returns true when the magic is in place.
    #[test]
    fn probe_with_magic() {
        let mut dev = MemoryBackend::new(8192);
        dev.write_at(32, b"NXSB").unwrap();
        assert!(probe(&mut dev).unwrap());
    }

    /// `open` on a blank device should fail at the magic check.
    #[test]
    fn open_blank_fails() {
        let mut dev = MemoryBackend::new(64 * 1024);
        let e = Apfs::open(&mut dev).unwrap_err();
        match e {
            crate::Error::InvalidImage(_) => {}
            other => panic!("expected InvalidImage, got {other:?}"),
        }
    }

    #[test]
    fn split_path_basics() {
        assert!(split_path("/").is_empty());
        assert!(split_path("").is_empty());
        assert!(split_path(".").is_empty());
        assert_eq!(split_path("/foo/bar"), vec!["foo", "bar"]);
        assert_eq!(split_path("foo/./bar/"), vec!["foo", "bar"]);
    }

    /// Wire the lookup against a hand-rolled container that gets only as
    /// far as the NXSB; deeper structures aren't here, so we only check
    /// that the proper error is returned past the superblock layer.
    #[test]
    fn open_with_nxsb_only_errors_cleanly() {
        let mut dev = MemoryBackend::new(64 * 4096);
        // Build a minimal NXSB at block 0.
        let mut buf = vec![0u8; 4096];
        buf[24..28].copy_from_slice(&OBJECT_TYPE_NX_SUPERBLOCK.to_le_bytes());
        buf[32..36].copy_from_slice(&NX_MAGIC.to_le_bytes());
        buf[36..40].copy_from_slice(&4096u32.to_le_bytes()); // block size
        buf[40..48].copy_from_slice(&64u64.to_le_bytes()); // block count
        // xp_desc area: 0 blocks → we'll use the label.
        buf[104..108].copy_from_slice(&0u32.to_le_bytes());
        buf[112..120].copy_from_slice(&0u64.to_le_bytes());
        // omap_oid: pointing past the device.
        buf[160..168].copy_from_slice(&u64::MAX.to_le_bytes());
        // max_file_systems = 1, fs_oid[0] = 7
        buf[180..184].copy_from_slice(&1u32.to_le_bytes());
        buf[184..192].copy_from_slice(&7u64.to_le_bytes());
        dev.write_at(0, &buf).unwrap();

        let e = Apfs::open(&mut dev).unwrap_err();
        // Either InvalidImage or OutOfBounds depending on where we
        // stumble — both are acceptable "I couldn't read that block".
        assert!(matches!(
            e,
            crate::Error::InvalidImage(_) | crate::Error::OutOfBounds { .. }
        ));
    }

    /// list_volumes on a blank device fails the same way `open` does
    /// (InvalidImage at the magic check).
    #[test]
    fn list_volumes_blank_fails() {
        let mut dev = MemoryBackend::new(64 * 1024);
        let e = Apfs::list_volumes(&mut dev).unwrap_err();
        assert!(matches!(e, crate::Error::InvalidImage(_)));
    }

    /// End-to-end through the [`crate::fs::Filesystem`] trait:
    /// `format` → `create_dir` → `create_file` → `flush` → `list` /
    /// `read_file`. This is the round-trip the task brief calls out as
    /// the success bar for trait wiring.
    #[test]
    fn trait_round_trip_format_create_flush_read() {
        use crate::fs::{FileMeta, FileSource, Filesystem};
        use std::io::Read;

        let total_blocks = 128u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);

        let mut apfs = Apfs::format(&mut dev, total_blocks, bs, "TraitVol").unwrap();
        // Capability advertises WholeFileOnly while buffering.
        assert!(matches!(
            apfs.mutation_capability(),
            crate::fs::MutationCapability::WholeFileOnly
        ));

        // create_dir + create_file under root and under that dir.
        apfs.create_dir(
            &mut dev,
            std::path::Path::new("/sub"),
            FileMeta::with_mode(0o755),
        )
        .unwrap();
        let payload: Vec<u8> = b"hello via trait".to_vec();
        apfs.create_file(
            &mut dev,
            std::path::Path::new("/sub/hello.txt"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(payload.clone())),
                len: payload.len() as u64,
            },
            FileMeta::with_mode(0o644),
        )
        .unwrap();
        // A file at root too, to exercise the "/"-as-parent path.
        apfs.create_file(
            &mut dev,
            std::path::Path::new("/note"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"top-level".to_vec())),
                len: 9,
            },
            FileMeta::with_mode(0o600),
        )
        .unwrap();

        // Flush materialises the image and transitions to read mode.
        apfs.flush(&mut dev).unwrap();
        assert!(matches!(
            apfs.mutation_capability(),
            crate::fs::MutationCapability::Immutable
        ));

        // Verify by listing and reading through the same trait surface.
        let root = apfs.list(&mut dev, std::path::Path::new("/")).unwrap();
        let names: Vec<&str> = root.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"sub"), "root listing: {names:?}");
        assert!(names.contains(&"note"), "root listing: {names:?}");

        let sub = apfs.list(&mut dev, std::path::Path::new("/sub")).unwrap();
        let sub_names: Vec<&str> = sub.iter().map(|e| e.name.as_str()).collect();
        assert!(
            sub_names.contains(&"hello.txt"),
            "/sub listing: {sub_names:?}"
        );

        let mut buf = Vec::new();
        apfs.read_file(&mut dev, std::path::Path::new("/sub/hello.txt"))
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        assert_eq!(buf, payload);

        buf.clear();
        apfs.read_file(&mut dev, std::path::Path::new("/note"))
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        assert_eq!(buf, b"top-level");
    }

    /// `open_file_ro` returns a Read+Seek+len handle in Read state;
    /// seeking + reading lands the right bytes.
    #[test]
    fn open_file_ro_random_seek_round_trip() {
        use crate::fs::{FileMeta, FileSource, Filesystem};
        use std::io::{Read, Seek, SeekFrom};

        let total_blocks = 128u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let body: Vec<u8> = (0..4096u32).map(|i| (i & 0xff) as u8).collect();
        let mut apfs = Apfs::format(&mut dev, total_blocks, bs, "Vol").unwrap();
        apfs.create_file(
            &mut dev,
            std::path::Path::new("/data.bin"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(body.clone())),
                len: body.len() as u64,
            },
            FileMeta::with_mode(0o644),
        )
        .unwrap();
        apfs.flush(&mut dev).unwrap();

        let mut h = apfs
            .open_file_ro(&mut dev, std::path::Path::new("/data.bin"))
            .unwrap();
        assert_eq!(h.len(), body.len() as u64);
        // Seek mid-file + read.
        h.seek(SeekFrom::Start(1000)).unwrap();
        let mut chunk = [0u8; 32];
        h.read_exact(&mut chunk).unwrap();
        assert_eq!(&chunk[..], &body[1000..1032]);
        // SeekFrom::End past EOF clamps; read returns 0.
        let where_ = h.seek(SeekFrom::End(100)).unwrap();
        assert_eq!(where_, body.len() as u64);
        let n = h.read(&mut chunk).unwrap();
        assert_eq!(n, 0);
    }

    /// `open_file_ro` in pending-write (pre-flush) mode is
    /// Unsupported.
    #[test]
    fn open_file_ro_refused_pre_flush() {
        use crate::fs::Filesystem;
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let mut apfs = Apfs::format(&mut dev, total_blocks, bs, "Vol").unwrap();
        let err = match apfs.open_file_ro(&mut dev, std::path::Path::new("/x")) {
            Ok(_) => panic!("open_file_ro must refuse in pending-write mode"),
            Err(e) => e,
        };
        assert!(matches!(err, crate::Error::Unsupported(_)));
    }

    /// Once flushed, further `create_*` calls return Unsupported (the
    /// writer doesn't support post-flush mutation).
    #[test]
    fn trait_create_after_flush_is_unsupported() {
        use crate::fs::{FileMeta, FileSource, Filesystem};
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let mut apfs = Apfs::format(&mut dev, total_blocks, bs, "Vol").unwrap();
        apfs.flush(&mut dev).unwrap();
        let e = apfs
            .create_file(
                &mut dev,
                std::path::Path::new("/late"),
                FileSource::Zero(0),
                FileMeta::default(),
            )
            .unwrap_err();
        assert!(matches!(e, crate::Error::Unsupported(_)));
    }

    /// Creating a file under a parent that hasn't been declared via
    /// `create_dir` is an InvalidArgument (no implicit-parent magic).
    #[test]
    fn trait_create_file_requires_existing_parent() {
        use crate::fs::{FileMeta, FileSource, Filesystem};
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let mut apfs = Apfs::format(&mut dev, total_blocks, bs, "Vol").unwrap();
        let e = apfs
            .create_file(
                &mut dev,
                std::path::Path::new("/nope/file"),
                FileSource::Zero(0),
                FileMeta::default(),
            )
            .unwrap_err();
        assert!(matches!(e, crate::Error::InvalidArgument(_)));
    }

    /// Reading from a freshly-formatted (still-buffering) [`Apfs`]
    /// refuses cleanly with Unsupported — the device hasn't been
    /// written yet.
    #[test]
    fn read_before_flush_is_unsupported() {
        use crate::fs::Filesystem;
        let total_blocks = 64u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let mut apfs = Apfs::format(&mut dev, total_blocks, bs, "Vol").unwrap();
        let e = apfs.list(&mut dev, std::path::Path::new("/")).unwrap_err();
        assert!(matches!(e, crate::Error::Unsupported(_)));
    }

    /// Format → flush → drop → re-`open_writable` succeeds and yields a
    /// `Write`-mode `Apfs` whose capability is `Mutable`.
    #[test]
    fn open_writable_round_trip_basic() {
        use crate::fs::{FileMeta, FileSource, Filesystem};
        let total_blocks = 256u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let mut apfs = Apfs::format(&mut dev, total_blocks, bs, "Vol").unwrap();
        apfs.create_file(
            &mut dev,
            std::path::Path::new("/note.txt"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"hello".to_vec())),
                len: 5,
            },
            FileMeta::with_mode(0o644),
        )
        .unwrap();
        apfs.flush(&mut dev).unwrap();
        drop(apfs);

        let apfs = Apfs::open_writable(&mut dev).unwrap();
        assert!(matches!(apfs.state, ApfsState::Write(_)));
        assert!(matches!(
            apfs.mutation_capability(),
            crate::fs::MutationCapability::Mutable
        ));
    }

    /// End-to-end: format → flush → `open_writable` → `open_file_rw` →
    /// write new bytes → `sync` → reopen → `read_file` returns identical
    /// bytes.
    #[test]
    fn rw_round_trip_overwrite_existing_file() {
        use crate::fs::{FileMeta, FileSource, Filesystem, OpenFlags};
        use std::io::{Read, Write};
        let total_blocks = 256u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let mut apfs = Apfs::format(&mut dev, total_blocks, bs, "Vol").unwrap();
        // Original bytes.
        let original: Vec<u8> = b"original-payload".to_vec();
        apfs.create_file(
            &mut dev,
            std::path::Path::new("/note.txt"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(original.clone())),
                len: original.len() as u64,
            },
            FileMeta::with_mode(0o644),
        )
        .unwrap();
        apfs.flush(&mut dev).unwrap();
        drop(apfs);

        // Reopen for writes and overwrite via open_file_rw.
        let mut apfs = Apfs::open_writable(&mut dev).unwrap();
        let new_payload: Vec<u8> = b"REWRITTEN-PAYLOAD-DATA".to_vec();
        {
            let mut h = apfs
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/note.txt"),
                    OpenFlags {
                        truncate: true,
                        ..OpenFlags::default()
                    },
                    None,
                )
                .unwrap();
            h.write_all(&new_payload).unwrap();
            h.sync().unwrap();
        }
        drop(apfs);

        // Reopen read-only and verify.
        let mut apfs = Apfs::open(&mut dev).unwrap();
        let mut buf = Vec::new();
        apfs.read_file(&mut dev, std::path::Path::new("/note.txt"))
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        assert_eq!(buf, new_payload);
    }

    /// Multi-session write: each `open_writable` consumes exactly one
    /// xp_desc slot, and the previous checkpoint stays intact for
    /// crash-safety.
    #[test]
    fn rw_round_trip_two_consecutive_sessions() {
        use crate::fs::{FileMeta, FileSource, Filesystem, OpenFlags};
        use std::io::{Read, Write};
        let total_blocks = 256u64;
        let bs = 4096u32;
        let mut dev = MemoryBackend::new(total_blocks * bs as u64);
        let mut apfs = Apfs::format(&mut dev, total_blocks, bs, "Vol").unwrap();
        apfs.create_file(
            &mut dev,
            std::path::Path::new("/log"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"v0".to_vec())),
                len: 2,
            },
            FileMeta::with_mode(0o644),
        )
        .unwrap();
        apfs.flush(&mut dev).unwrap();
        drop(apfs);

        // Session 1.
        let mut apfs = Apfs::open_writable(&mut dev).unwrap();
        {
            let mut h = apfs
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/log"),
                    OpenFlags {
                        truncate: true,
                        ..OpenFlags::default()
                    },
                    None,
                )
                .unwrap();
            h.write_all(b"v1-after-first-commit").unwrap();
            h.sync().unwrap();
        }
        drop(apfs);

        // Session 2.
        let mut apfs = Apfs::open_writable(&mut dev).unwrap();
        {
            let mut h = apfs
                .open_file_rw(
                    &mut dev,
                    std::path::Path::new("/log"),
                    OpenFlags {
                        truncate: true,
                        ..OpenFlags::default()
                    },
                    None,
                )
                .unwrap();
            h.write_all(b"v2-final-payload-here").unwrap();
            h.sync().unwrap();
        }
        drop(apfs);

        // Final check.
        let mut apfs = Apfs::open(&mut dev).unwrap();
        let mut buf = Vec::new();
        apfs.read_file(&mut dev, std::path::Path::new("/log"))
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        assert_eq!(buf, b"v2-final-payload-here");
    }
}
