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
//!    scratch — see [`write::ApfsWriter`] for the API.
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
pub mod snap;
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
pub struct Apfs {
    /// Effective block size (`nx_block_size`).
    block_size: u32,
    /// `nx_block_count * nx_block_size`.
    total_bytes: u64,
    /// Volume name (UTF-8, trimmed of trailing NUL).
    volume_name: String,
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

impl std::fmt::Debug for Apfs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Apfs")
            .field("block_size", &self.block_size)
            .field("total_bytes", &self.total_bytes)
            .field("volume_name", &self.volume_name)
            .field("drec_layout", &self.drec_layout)
            .field("volume_index", &self.volume_index)
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
            snap_meta_tree_oid: apsb.snap_meta_tree_oid,
            volume_index: index,
            fsroot_block,
            fs_ctx: std::cell::RefCell::new(fs_ctx),
            drec_layout,
        })
    }

    /// List every snapshot recorded in the volume's snapshot-metadata
    /// tree. The snap-meta tree is a physical B-tree; we read its root
    /// block directly. Multi-level snap-meta trees return `Unsupported`
    /// cleanly.
    pub fn list_snapshots(&self, dev: &mut dyn BlockDevice) -> Result<Vec<SnapshotInfo>> {
        if self.snap_meta_tree_oid == 0 {
            return Ok(Vec::new());
        }
        let mut root = vec![0u8; self.block_size as usize];
        let off = self
            .snap_meta_tree_oid
            .saturating_mul(self.block_size as u64);
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
        let snaps = self.list_snapshots(dev)?;
        let snap = snaps
            .iter()
            .find(|s| s.xid == xid)
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!("apfs: no snapshot with xid {xid}"))
            })?
            .clone();
        let ctx = load_container(dev)?;
        Self::open_volume_with_ctx(
            dev,
            &ctx,
            self.volume_index,
            Some((snap.sblock_paddr, snap.xid)),
        )
    }

    /// Open a snapshot of this volume by name.
    pub fn open_snapshot_by_name(&self, dev: &mut dyn BlockDevice, name: &str) -> Result<Self> {
        let snaps = self.list_snapshots(dev)?;
        let snap = snaps
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!("apfs: no snapshot named {name:?}"))
            })?
            .clone();
        let ctx = load_container(dev)?;
        Self::open_volume_with_ctx(
            dev,
            &ctx,
            self.volume_index,
            Some((snap.sblock_paddr, snap.xid)),
        )
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
        let target = FsKeyTarget {
            oid: target_oid,
            kind: APFS_TYPE_XATTR,
            tail: &[],
            drec_layout: self.drec_layout,
        };
        let block_size = self.block_size;
        let mut ctx = self.fs_ctx.borrow_mut();
        let mut out = std::collections::HashMap::new();
        let mut scan =
            RangeScan::start(&self.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
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
        let layout = self.drec_layout;
        let target = FsKeyTarget {
            oid: parent_oid,
            kind: APFS_TYPE_DIR_REC,
            tail: &[],
            drec_layout: layout,
        };
        let block_size = self.block_size;
        let mut ctx = self.fs_ctx.borrow_mut();
        let mut scan =
            RangeScan::start(&self.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
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
        let layout = self.drec_layout;
        let target = FsKeyTarget {
            oid: dir_oid,
            kind: APFS_TYPE_DIR_REC,
            tail: &[],
            drec_layout: layout,
        };
        let block_size = self.block_size;
        let mut ctx = self.fs_ctx.borrow_mut();
        let mut out = Vec::new();
        let mut scan =
            RangeScan::start(&self.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
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
            });
        }
        Ok(out)
    }

    /// Find the inode record for `oid` and return `(size, dstream_oid)`.
    /// Inode records have an empty type-specific tail; a single
    /// `(oid, APFS_TYPE_INODE)` range scan yields exactly the one we
    /// want (at most one record per oid in a fs-tree).
    fn lookup_inode_size(&self, dev: &mut dyn BlockDevice, oid: u64) -> Result<(u64, Option<u64>)> {
        let target = FsKeyTarget {
            oid,
            kind: APFS_TYPE_INODE,
            tail: &[],
            drec_layout: self.drec_layout,
        };
        let block_size = self.block_size;
        let mut ctx = self.fs_ctx.borrow_mut();
        let mut scan =
            RangeScan::start(&self.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
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
        let target = FsKeyTarget {
            oid: dstream_oid,
            kind: APFS_TYPE_FILE_EXTENT,
            tail: &[],
            drec_layout: self.drec_layout,
        };
        let block_size = self.block_size;
        let mut ctx = self.fs_ctx.borrow_mut();
        let mut out = Vec::new();
        let mut scan =
            RangeScan::start(&self.fsroot_block, &target, &mut ctx, &mut |paddr, buf| {
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

/// Read-only `Filesystem` adapter so `inspect::open(dev)` can return a
/// `Box<dyn Filesystem>` that walks an APFS container. Writes return
/// `Unsupported` — the APFS writer is library-only and not yet trait
/// wired.
impl crate::fs::Filesystem for Apfs {
    fn create_file(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &std::path::Path,
        _src: crate::fs::FileSource,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(crate::Error::Unsupported(
            "apfs: read-only on this trait surface".into(),
        ))
    }

    fn create_dir(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &std::path::Path,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(crate::Error::Unsupported(
            "apfs: read-only on this trait surface".into(),
        ))
    }

    fn create_symlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &std::path::Path,
        _target: &std::path::Path,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(crate::Error::Unsupported(
            "apfs: read-only on this trait surface".into(),
        ))
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
            "apfs: read-only on this trait surface".into(),
        ))
    }

    fn remove(&mut self, _dev: &mut dyn BlockDevice, _path: &std::path::Path) -> Result<()> {
        Err(crate::Error::Unsupported(
            "apfs: read-only on this trait surface".into(),
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

    fn flush(&mut self, _dev: &mut dyn BlockDevice) -> Result<()> {
        Ok(())
    }

    fn supports_mutation(&self) -> bool {
        false
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
}
