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
    DT_REG, DrecKey, DrecVal, FileExtentVal, InodeVal,
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
    },
    /// `create_file` — same shape as `Dir`. The file's bytes are
    /// captured into `data` at buffer time because [`crate::fs::FileSource`]
    /// is consumed by the trait call.
    File {
        parent_oid: u64,
        name: String,
        mode: u16,
        data: Vec<u8>,
    },
    /// `create_symlink` — same shape as `Dir`, plus the link target.
    Symlink {
        parent_oid: u64,
        name: String,
        mode: u16,
        target: String,
    },
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
    /// Current scope: works on a freshly-formatted-then-flushed image
    /// and supports up to `XP_DESC_BLOCKS - 2` subsequent checkpoint
    /// commits before the xp_desc area fills up; create / remove are
    /// still refused. See the `rw` module for the per-feature limits.
    pub fn open_writable(dev: &mut dyn BlockDevice) -> Result<Self> {
        let read_only = Apfs::open(dev)?;
        // Re-decode the container so we can pull the checkpoint
        // metadata we need (xid, xp_desc_index, container UUID).
        let ctx = load_container(dev)?;
        let block_size = ctx.block_size;
        let total_blocks = ctx.live_sb.block_count;
        let total_bytes = ctx.total_bytes;
        let container_uuid = ctx.live_sb.uuid;
        let cur_xid = ctx.live_sb.obj.xid;

        // The reader picks the highest-xid NXSB inside xp_desc, so we
        // know which slot that lives at — that's `xp_desc_index +
        // xp_desc_len - 1` modulo the ring. Because our writer keeps
        // xp_desc_index = 0 and xp_desc_len strictly forward, the next
        // free slot is `xp_desc_base + xp_desc_len`.
        let next_xp_desc_slot = ctx.live_sb.xp_desc_base + ctx.live_sb.xp_desc_len as u64;
        if next_xp_desc_slot >= ctx.live_sb.xp_desc_base + ctx.live_sb.xp_desc_blocks as u64 {
            return Err(crate::Error::Unsupported(
                "apfs: xp_desc area is full — checkpoint rotation isn't \
                 implemented yet"
                    .into(),
            ));
        }

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

    /// Walk path components, resolving each name through its parent's
    /// directory records. Returns the target's object id.
    fn resolve_path_to_oid(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<u64> {
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
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        src: crate::fs::FileSource,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        // Pull bytes out of `src` while we still have it. APFS' writer
        // streams from a Read, but we have to buffer here because the
        // writer doesn't exist yet — flush() builds it. For real-world
        // use this means create_file is bounded by available RAM. Empty
        // files are fine.
        let (mut reader, size) = src
            .open()
            .map_err(|e| crate::Error::Io(std::io::Error::other(e)))?;
        let mut data = Vec::with_capacity(size.min(64 * 1024 * 1024) as usize);
        let n = std::io::Read::read_to_end(&mut reader, &mut data)
            .map_err(|e| crate::Error::Io(std::io::Error::other(e)))?;
        if n as u64 != size {
            // Pad with zeros so dstream.size still matches what the user
            // asked for; mirrors the writer's truncation handling.
            data.resize(size as usize, 0);
        }
        let pw = pending_write_mut(&mut self.state)?;
        let (parent_oid, name) = pw.resolve_parent(path)?;
        pw.ops.push(PendingOp::File {
            parent_oid,
            name,
            mode: meta.mode,
            data,
        });
        // Consume an oid slot so future creates see the same sequence
        // the writer will use. add_file_from_reader allocates one oid.
        pw.next_oid = pw.next_oid.saturating_add(1);
        Ok(())
    }

    fn create_dir(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let pw = pending_write_mut(&mut self.state)?;
        let (parent_oid, name) = pw.resolve_parent(path)?;
        // Assign a deterministic oid that matches what the writer will
        // hand out for this position in the call sequence, and remember
        // it so nested children of this directory can resolve their
        // parent path on subsequent calls.
        let new_oid = pw.next_oid;
        pw.next_oid = pw.next_oid.saturating_add(1);
        pw.dir_oid.insert(path.to_path_buf(), new_oid);
        pw.ops.push(PendingOp::Dir {
            parent_oid,
            name,
            mode: meta.mode,
        });
        Ok(())
    }

    fn create_symlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        target: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let target_str = target
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("apfs: non-UTF-8 symlink target".into()))?
            .to_string();
        let pw = pending_write_mut(&mut self.state)?;
        let (parent_oid, name) = pw.resolve_parent(path)?;
        pw.ops.push(PendingOp::Symlink {
            parent_oid,
            name,
            mode: meta.mode,
            target: target_str,
        });
        pw.next_oid = pw.next_oid.saturating_add(1);
        Ok(())
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
            "apfs: device nodes are not supported by the writer".into(),
        ))
    }

    fn remove(&mut self, _dev: &mut dyn BlockDevice, _path: &std::path::Path) -> Result<()> {
        Err(crate::Error::Unsupported(
            "apfs: remove is not supported (writer is single-pass)".into(),
        ))
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
                    } => {
                        w.add_dir(parent_oid, &name, mode)?;
                    }
                    PendingOp::File {
                        parent_oid,
                        name,
                        mode,
                        data,
                    } => {
                        let len = data.len() as u64;
                        let mut r = std::io::Cursor::new(data);
                        w.add_file_from_reader(parent_oid, &name, mode, &mut r, len)?;
                    }
                    PendingOp::Symlink {
                        parent_oid,
                        name,
                        mode,
                        target,
                    } => {
                        w.add_symlink(parent_oid, &name, mode, &target)?;
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
    let live_sb = find_live_nxsb(dev, &label_sb, block_size)?.unwrap_or(label_sb.clone());

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
/// block (or `None` if the label is already the best).
fn find_live_nxsb(
    dev: &mut dyn BlockDevice,
    label: &NxSuperblock,
    block_size: u32,
) -> Result<Option<NxSuperblock>> {
    let n = label.xp_desc_blocks as u64;
    let base = label.xp_desc_base;
    let mut best: Option<NxSuperblock> = None;
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
            Some(b) => sb.obj.xid > b.obj.xid,
        };
        if better {
            best = Some(sb);
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
