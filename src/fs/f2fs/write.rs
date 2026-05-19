//! F2FS write driver — in-memory builder + flush to a [`BlockDevice`].
//!
//! ## Lifecycle
//!
//! 1. [`super::F2fs::format`] writes the static regions (superblock × 2,
//!    blank SIT / NAT / SSA, root inode) and returns an open [`super::F2fs`]
//!    backed by a [`Writer`] with the root pre-loaded.
//! 2. Callers issue any number of [`Writer::add_file`] /
//!    [`Writer::add_dir`] / [`Writer::add_symlink`] / [`Writer::remove`]
//!    operations. Inode and dentry mutations stay in RAM; **file data
//!    blocks stream straight to the main area** in 4 KiB chunks (max 64
//!    KiB pump buffer at any one time).
//! 3. [`super::F2fs::flush`] rewrites the changed inodes / direct nodes,
//!    rebuilds the NAT page(s) for the affected nids, and stamps a fresh
//!    CP0 head + summary. Calling `flush` repeatedly is idempotent.
//!
//! ## Inline-data fast path
//!
//! Files whose total size fits in the literal area (≤ `3712 - 8 - 27 = 3677`
//! bytes — we conservatively cap at `MAX_INLINE_DATA = 3672`) are encoded
//! with `F2FS_INLINE_DATA`. They consume zero data blocks.
//!
//! ## Inline dentry
//!
//! Directories that own ≤ [`super::dir::INLINE_DENTRY_NR`] entries with
//! ≤ 8-byte names *and* whose entries fit the literal area continue to
//! live in the inode's inline region. Once that budget is exceeded the
//! writer "spills" the directory to a regular 4 KiB dentry block (no
//! mixed-mode: one or the other).
//!
//! ## Limitations (returned as [`crate::Error::Unsupported`] when hit)
//!
//! - Compression, encryption, project quotas — no writer support.
//! - Inline xattr — not synthesised.
//!
//! ## Recently added
//!
//! - Hard links — [`Writer::add_hardlink`] points a new dentry at an existing
//!   nid and bumps `i_links`. Only nids tracked by the writer in the current
//!   session can be hard-linked (fresh-image lifetime).
//! - Triple-indirect node trees — [`Writer::add_file`] now grows through
//!   `i_nid[NID_TRIPLE_INDIRECT]` for files beyond the double-indirect
//!   threshold (~8 GiB).
//! - Multi-block spilled dentries — once a directory's child list overflows
//!   the first 4 KiB dentry block, additional blocks are allocated through
//!   the same direct/indirect-node chain a file uses.

use std::collections::BTreeMap;
use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::{DeviceKind, FileMeta, FileSource};

use super::checkpoint::Checkpoint;
use super::constants::{
    ADDRS_PER_BLOCK, ADDRS_PER_INODE, F2FS_BLK_CSUM_OFFSET, F2FS_BLKSIZE, F2FS_DATA_EXIST,
    F2FS_FT_BLKDEV, F2FS_FT_CHRDEV, F2FS_FT_DIR, F2FS_FT_FIFO, F2FS_FT_REG_FILE, F2FS_FT_SOCK,
    F2FS_FT_SYMLINK, F2FS_INLINE_DATA, F2FS_INLINE_DENTRY, F2FS_SLOT_LEN, NID_DIRECT_1,
    NID_DIRECT_2, NID_INDIRECT_1, NID_INDIRECT_2, NID_TRIPLE_INDIRECT, NIDS_PER_BLOCK,
    NIDS_PER_INODE, NR_DENTRY_IN_BLOCK, S_IFBLK, S_IFCHR, S_IFDIR, S_IFIFO, S_IFLNK, S_IFREG,
    S_IFSOCK, SIZE_OF_DENTRY_BITMAP, SIZE_OF_DIR_ENTRY, SIZE_OF_RESERVED,
};
use super::dir::INLINE_DENTRY_NR;
use super::format::Geometry;
use super::inode::I_ADDR_OFFSET;
use super::superblock::Superblock;

/// Pump buffer used to stream `FileSource` content from host into 4 KiB
/// data blocks. 64 KiB is the cap promised in the module docs.
const PUMP_BUF: usize = 64 * 1024;

/// Inline-data byte budget. The inode literal area runs `0xD0..F2FS_BLK_CSUM_OFFSET`
/// = 3884 bytes; we reserve the trailing 8 bytes for the node-footer
/// nid/ino and leave a 6-byte head guard.
pub const MAX_INLINE_DATA: usize = ADDRS_PER_INODE * 4 + NIDS_PER_INODE * 4 - 64;

/// One on-disk inode tracked by the writer.
#[derive(Debug, Clone)]
pub(crate) struct InodeRec {
    pub nid: u32,
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub links: u32,
    pub size: u64,
    pub blocks: u64,
    pub atime: u32,
    pub ctime: u32,
    pub mtime: u32,
    pub flags: u32,
    pub inline_flags: u8,
    /// 923 in-inode direct pointers.
    pub i_addr: [u32; ADDRS_PER_INODE],
    /// 5 node-id slots: 0,1 direct nodes; 2,3 indirect; 4 triple.
    pub i_nid: [u32; NIDS_PER_INODE],
    /// Inline payload (data or dentry) — when one of those flags is set
    /// this is the literal area, ≤ `MAX_INLINE_DATA` bytes.
    pub inline_payload: Vec<u8>,
    /// Each block in `i_addr` (and downstream node blocks) has a physical
    /// main-area address allocated when the data was streamed in. The
    /// writer never touches the device after this point.
    pub on_disk_block: u32,
}

/// Tracked direct-node block (1018 data-block pointers + a backing nid).
#[derive(Debug, Clone)]
pub(crate) struct DirectNode {
    pub nid: u32,
    pub on_disk_block: u32,
    /// 1018 data-block pointers.
    pub addrs: [u32; ADDRS_PER_BLOCK],
}

/// Tracked indirect-node block — 1018 node-id slots pointing at direct
/// nodes. The writer only allocates these when a single file overflows
/// both `i_nid[0]` and `i_nid[1]`.
#[derive(Debug, Clone)]
pub(crate) struct IndirectNode {
    pub nid: u32,
    pub on_disk_block: u32,
    pub nids: [u32; NIDS_PER_BLOCK],
}

/// One emitted dentry (block-format or inline). The hash is the F2FS
/// "dot hash" mirror of the name — we keep it 0 for v1; the reader
/// doesn't enforce it and `fsck.f2fs` only warns on mismatches.
#[derive(Debug, Clone)]
pub(crate) struct Dentry {
    pub hash: u32,
    pub ino: u32,
    pub file_type: u8,
    pub name: Vec<u8>,
}

/// Per-directory child list, indexed by directory nid.
type DirChildren = BTreeMap<u32, Vec<Dentry>>;

/// The mutable bookkeeping shared by every `add_*` call. Held inside
/// [`super::F2fs`] when the FS was opened for writing.
#[derive(Debug)]
pub struct Writer {
    pub(crate) geom: Geometry,
    #[allow(dead_code)]
    pub(crate) sb: Superblock,
    /// Next nid to hand out. Skips the three reserved nids 0,1,2 (meta,
    /// node, root) — we hand out from 4 upward, with nid 3 reserved for
    /// the root directory.
    pub(crate) next_nid: u32,
    /// Next NODE block to allocate. Lives in one of `node_segments`.
    /// When the current segment fills, the allocator pulls a fresh
    /// segment from the free-segment pool.
    pub(crate) next_node_blk: u32,
    /// Next DATA block to allocate. Same lifecycle as `next_node_blk`
    /// but for data-typed segments.
    pub(crate) next_data_blk: u32,
    /// Main-segment offsets currently or previously written by
    /// CURSEG_HOT_NODE. The LAST entry is the active one (the curseg's
    /// `segno`), all earlier entries are full. SIT entries for every
    /// segment in this list get `(CURSEG_HOT_NODE << 10) | valid_count`
    /// in their `vblocks` so fsck's `fsck_chk_curseg_info` is happy.
    pub(crate) node_segments: Vec<u32>,
    /// Main-segment offsets written by CURSEG_HOT_DATA.
    pub(crate) data_segments: Vec<u32>,
    /// Next free main-segment offset for spillover allocation. Starts
    /// at 6 — the 6 reserved cursegs occupy segments 0..5.
    pub(crate) next_free_seg: u32,
    /// All inodes we've created or modified — keyed by nid. The root is
    /// always present.
    pub(crate) inodes: BTreeMap<u32, InodeRec>,
    /// All direct-node blocks for the inodes above.
    pub(crate) direct_nodes: BTreeMap<u32, DirectNode>,
    /// All indirect-node blocks.
    pub(crate) indirect_nodes: BTreeMap<u32, IndirectNode>,
    /// Directory contents, indexed by directory nid.
    pub(crate) children: DirChildren,
    /// Inline-dentry directories that have already spilled to a block
    /// because they got too big. Once a dir spills it stays block-format.
    pub(crate) spilled_dirs: BTreeMap<u32, ()>,
}

impl Writer {
    /// Build the initial writer state — only the root inode is recorded.
    pub(crate) fn new(geom: Geometry, sb: Superblock) -> Self {
        let mut inodes = BTreeMap::new();
        let mut children = BTreeMap::new();
        let root_phys = geom.main_blkaddr; // first main-area block

        inodes.insert(
            3,
            InodeRec {
                nid: 3,
                mode: S_IFDIR | 0o755,
                uid: 0,
                gid: 0,
                links: 2,
                size: F2FS_BLKSIZE as u64,
                blocks: 0,
                atime: 0,
                ctime: 0,
                mtime: 0,
                flags: 0,
                inline_flags: F2FS_INLINE_DENTRY,
                i_addr: [0; ADDRS_PER_INODE],
                i_nid: [0; NIDS_PER_INODE],
                inline_payload: Vec::new(),
                on_disk_block: root_phys,
            },
        );
        children.insert(3, Vec::new());

        // Reserved layout for the 6 cursegs (matches mkfs.f2fs):
        //   main seg 0 → CURSEG_HOT_NODE (root inode lives here)
        //   main seg 1 → CURSEG_WARM_NODE (unused, kept empty)
        //   main seg 2 → CURSEG_COLD_NODE (unused, kept empty)
        //   main seg 3 → CURSEG_HOT_DATA
        //   main seg 4 → CURSEG_WARM_DATA (unused, kept empty)
        //   main seg 5 → CURSEG_COLD_DATA (unused, kept empty)
        //   main seg 6+ → free pool for hot_node / hot_data spillover.
        let data_seg_offset = 3u32 * geom.blocks_per_seg;
        Self {
            geom,
            sb,
            next_nid: 4,
            next_node_blk: root_phys + 1, // reserve root inode block (seg 0)
            next_data_blk: root_phys + data_seg_offset,
            node_segments: vec![0],
            data_segments: vec![3],
            next_free_seg: 6,
            inodes,
            direct_nodes: BTreeMap::new(),
            indirect_nodes: BTreeMap::new(),
            children,
            spilled_dirs: BTreeMap::new(),
        }
    }

    /// Allocate a fresh nid.
    fn alloc_nid(&mut self) -> u32 {
        let n = self.next_nid;
        self.next_nid += 1;
        n
    }

    /// Returns the absolute block address of segment `seg` (main offset).
    fn seg_to_blk(&self, seg: u32) -> u32 {
        self.geom.main_blkaddr + seg * self.geom.blocks_per_seg
    }

    /// Claim a fresh main-area segment from the free pool. Used when a
    /// curseg fills up and needs spillover capacity.
    fn alloc_fresh_segment(&mut self) -> Result<u32> {
        let s = self.next_free_seg;
        if s >= self.geom.segment_count_main {
            return Err(crate::Error::InvalidArgument(
                "f2fs: main area exhausted (no free segments)".into(),
            ));
        }
        self.next_free_seg += 1;
        Ok(s)
    }

    /// Allocate one NODE block (CURSEG_HOT_NODE). When the active node
    /// segment fills, claim a new one from the free pool and switch to
    /// it; the CP head's `cur_node_segno[0]` will reflect the latest.
    fn alloc_node_block(&mut self) -> Result<u32> {
        let active_seg = *self
            .node_segments
            .last()
            .expect("node_segments seeded in new()");
        let seg_end = self.seg_to_blk(active_seg) + self.geom.blocks_per_seg;
        if self.next_node_blk >= seg_end {
            let fresh = self.alloc_fresh_segment()?;
            self.node_segments.push(fresh);
            self.next_node_blk = self.seg_to_blk(fresh);
        }
        let blk = self.next_node_blk;
        self.next_node_blk += 1;
        Ok(blk)
    }

    /// Allocate one DATA block (CURSEG_HOT_DATA). Same spillover behavior.
    fn alloc_data_block(&mut self) -> Result<u32> {
        let active_seg = *self
            .data_segments
            .last()
            .expect("data_segments seeded in new()");
        let seg_end = self.seg_to_blk(active_seg) + self.geom.blocks_per_seg;
        if self.next_data_blk >= seg_end {
            let fresh = self.alloc_fresh_segment()?;
            self.data_segments.push(fresh);
            self.next_data_blk = self.seg_to_blk(fresh);
        }
        let blk = self.next_data_blk;
        self.next_data_blk += 1;
        Ok(blk)
    }

    /// Resolve a posix-style path to the inode that owns it. Returns
    /// `(parent_nid, leaf_name_bytes, existing_child_nid?)`.
    fn resolve_for_create(&self, path: &std::path::Path) -> Result<(u32, Vec<u8>, Option<u32>)> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("f2fs: non-UTF-8 path".into()))?;
        if s == "/" || s.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "f2fs: can't create root".into(),
            ));
        }
        let parts: Vec<&str> = s
            .trim_matches('/')
            .split('/')
            .filter(|p| !p.is_empty())
            .collect();
        if parts.is_empty() {
            return Err(crate::Error::InvalidArgument(format!(
                "f2fs: empty path {s:?}"
            )));
        }
        // Walk the tree by descending through known children.
        let mut cur = 3u32;
        for comp in &parts[..parts.len() - 1] {
            let kids = self.children.get(&cur).ok_or_else(|| {
                crate::Error::InvalidArgument(format!("f2fs: parent of {s:?} not a known dir"))
            })?;
            let found = kids
                .iter()
                .find(|d| d.name == comp.as_bytes())
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!("f2fs: no such dir {comp:?} in {s:?}"))
                })?;
            cur = found.ino;
        }
        let leaf = parts[parts.len() - 1].as_bytes().to_vec();
        let existing = self
            .children
            .get(&cur)
            .and_then(|kids| kids.iter().find(|d| d.name == leaf).map(|d| d.ino));
        Ok((cur, leaf, existing))
    }

    /// Append `data` to a fresh inode `nid`'s body, allocating blocks /
    /// direct nodes / indirect nodes as needed. Returns the new total
    /// size and block count.
    fn stream_into_inode(
        &mut self,
        dev: &mut dyn BlockDevice,
        nid: u32,
        mut src: Box<dyn crate::fs::ReadSeek + Send>,
        total: u64,
    ) -> Result<(u64, u64)> {
        // Inline fast path.
        if (total as usize) <= MAX_INLINE_DATA {
            let mut buf = vec![0u8; total as usize];
            if !buf.is_empty() {
                src.read_exact(&mut buf)?;
            }
            let ino = self
                .inodes
                .get_mut(&nid)
                .ok_or_else(|| crate::Error::InvalidImage("f2fs: ghost nid".into()))?;
            ino.inline_flags |= F2FS_INLINE_DATA | F2FS_DATA_EXIST;
            ino.inline_payload = buf;
            return Ok((total, 0));
        }

        let bs = F2FS_BLKSIZE as u64;
        let mut pump = vec![0u8; PUMP_BUF];
        let mut block_buf = vec![0u8; F2FS_BLKSIZE];
        let mut pump_pos = 0usize;
        let mut pump_len = 0usize;
        let mut logical_idx: u64 = 0;
        let mut written: u64 = 0;
        let mut blocks_alloc: u64 = 0;

        while written < total {
            // Refill pump.
            if pump_pos == pump_len {
                let want = ((total - written) as usize).min(pump.len());
                let n = src.read(&mut pump[..want])?;
                if n == 0 {
                    return Err(crate::Error::Io(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "f2fs: source ended early",
                    )));
                }
                pump_len = n;
                pump_pos = 0;
            }
            // Fill exactly one 4 KiB block from pump (possibly across refills).
            block_buf.fill(0);
            let mut filled = 0usize;
            let remaining = (total - written) as usize;
            let block_target = F2FS_BLKSIZE.min(remaining);
            while filled < block_target {
                if pump_pos == pump_len {
                    let want = ((total - written - filled as u64) as usize).min(pump.len());
                    let n = src.read(&mut pump[..want])?;
                    if n == 0 {
                        return Err(crate::Error::Io(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "f2fs: source ended early",
                        )));
                    }
                    pump_len = n;
                    pump_pos = 0;
                }
                let take = (pump_len - pump_pos).min(block_target - filled);
                block_buf[filled..filled + take].copy_from_slice(&pump[pump_pos..pump_pos + take]);
                pump_pos += take;
                filled += take;
            }
            // Allocate + write the data block.
            let phys = self.alloc_data_block()?;
            dev.write_at(phys as u64 * bs, &block_buf)?;
            self.place_data_block(nid, logical_idx, phys)?;
            logical_idx += 1;
            written += filled as u64;
            blocks_alloc += 1;
        }
        Ok((total, blocks_alloc))
    }

    /// Stamp `phys` into the right slot of `nid`'s pointer tree.
    fn place_data_block(&mut self, nid: u32, idx: u64, phys: u32) -> Result<()> {
        if idx < ADDRS_PER_INODE as u64 {
            let ino = self
                .inodes
                .get_mut(&nid)
                .ok_or_else(|| crate::Error::InvalidImage("f2fs: ghost nid".into()))?;
            ino.i_addr[idx as usize] = phys;
            return Ok(());
        }
        let mut rel = idx - ADDRS_PER_INODE as u64;

        // Two direct nodes (each 1018 ptrs).
        for nid_slot in [NID_DIRECT_1, NID_DIRECT_2] {
            let span = ADDRS_PER_BLOCK as u64;
            if rel < span {
                let dnode_nid = self.ensure_direct_node(nid, nid_slot)?;
                let d = self
                    .direct_nodes
                    .get_mut(&dnode_nid)
                    .expect("just inserted");
                d.addrs[rel as usize] = phys;
                return Ok(());
            }
            rel -= span;
        }

        // Two indirect nodes (each 1018 × 1018 ptrs).
        for nid_slot in [NID_INDIRECT_1, NID_INDIRECT_2] {
            let span = (NIDS_PER_BLOCK as u64) * (ADDRS_PER_BLOCK as u64);
            if rel < span {
                let outer = (rel / ADDRS_PER_BLOCK as u64) as usize;
                let inner = (rel % ADDRS_PER_BLOCK as u64) as usize;
                let ind_nid = self.ensure_indirect_node(nid, nid_slot)?;
                let dnode_nid = self.ensure_dnode_under_indirect(ind_nid, outer)?;
                let d = self
                    .direct_nodes
                    .get_mut(&dnode_nid)
                    .expect("just inserted");
                d.addrs[inner] = phys;
                return Ok(());
            }
            rel -= span;
        }

        // Triple-indirect region: i_nid[4] → top-indirect (1018 nids of
        // indirect nodes) → each indirect (1018 nids of direct nodes) →
        // each direct (1018 data ptrs). Span = 1018^3 blocks ≈ 4 TiB.
        let triple_span = (NIDS_PER_BLOCK as u64).pow(2) * (ADDRS_PER_BLOCK as u64);
        if rel < triple_span {
            let outer = (rel / ((NIDS_PER_BLOCK as u64) * (ADDRS_PER_BLOCK as u64))) as usize;
            let mid = ((rel / (ADDRS_PER_BLOCK as u64)) % (NIDS_PER_BLOCK as u64)) as usize;
            let inner = (rel % (ADDRS_PER_BLOCK as u64)) as usize;
            let top_nid = self.ensure_indirect_node(nid, NID_TRIPLE_INDIRECT)?;
            let mid_nid = self.ensure_indirect_under_indirect(top_nid, outer)?;
            let dnode_nid = self.ensure_dnode_under_indirect(mid_nid, mid)?;
            let d = self
                .direct_nodes
                .get_mut(&dnode_nid)
                .expect("just inserted");
            d.addrs[inner] = phys;
            return Ok(());
        }

        Err(crate::Error::Unsupported(format!(
            "f2fs: file exceeds the triple-indirect addressable limit (idx={idx})"
        )))
    }

    /// Ensure an indirect-node block exists at top-indirect `parent`'s
    /// `outer` slot. Symmetric to [`Self::ensure_dnode_under_indirect`] but
    /// for the top tier of the triple-indirect tree (where the child is an
    /// indirect node, not a direct node).
    fn ensure_indirect_under_indirect(&mut self, parent: u32, outer: usize) -> Result<u32> {
        let ind = self
            .indirect_nodes
            .get_mut(&parent)
            .ok_or_else(|| crate::Error::InvalidImage("f2fs: ghost triple-top nid".into()))?;
        if ind.nids[outer] != 0 {
            return Ok(ind.nids[outer]);
        }
        let inid = {
            let n = self.next_nid;
            self.next_nid += 1;
            n
        };
        let phys = self.alloc_node_block()?;
        self.indirect_nodes.get_mut(&parent).unwrap().nids[outer] = inid;
        self.indirect_nodes.insert(
            inid,
            IndirectNode {
                nid: inid,
                on_disk_block: phys,
                nids: [0; NIDS_PER_BLOCK],
            },
        );
        Ok(inid)
    }

    /// Ensure a direct-node block exists under `inode.i_nid[slot]`,
    /// allocating one if not.
    fn ensure_direct_node(&mut self, parent_nid: u32, slot: usize) -> Result<u32> {
        let ino = self
            .inodes
            .get_mut(&parent_nid)
            .ok_or_else(|| crate::Error::InvalidImage("f2fs: ghost nid".into()))?;
        if ino.i_nid[slot] != 0 {
            return Ok(ino.i_nid[slot]);
        }
        let dnid = {
            let n = self.next_nid;
            self.next_nid += 1;
            n
        };
        let phys = self.alloc_node_block()?;
        // Re-borrow to set the slot.
        self.inodes.get_mut(&parent_nid).unwrap().i_nid[slot] = dnid;
        self.direct_nodes.insert(
            dnid,
            DirectNode {
                nid: dnid,
                on_disk_block: phys,
                addrs: [0; ADDRS_PER_BLOCK],
            },
        );
        Ok(dnid)
    }

    /// Ensure an indirect node block under `inode.i_nid[slot]`.
    fn ensure_indirect_node(&mut self, parent_nid: u32, slot: usize) -> Result<u32> {
        let ino = self
            .inodes
            .get_mut(&parent_nid)
            .ok_or_else(|| crate::Error::InvalidImage("f2fs: ghost nid".into()))?;
        if ino.i_nid[slot] != 0 {
            return Ok(ino.i_nid[slot]);
        }
        let inid = {
            let n = self.next_nid;
            self.next_nid += 1;
            n
        };
        let phys = self.alloc_node_block()?;
        self.inodes.get_mut(&parent_nid).unwrap().i_nid[slot] = inid;
        self.indirect_nodes.insert(
            inid,
            IndirectNode {
                nid: inid,
                on_disk_block: phys,
                nids: [0; NIDS_PER_BLOCK],
            },
        );
        Ok(inid)
    }

    /// Ensure a direct-node block exists at indirect `parent`'s `outer`
    /// slot.
    fn ensure_dnode_under_indirect(&mut self, parent: u32, outer: usize) -> Result<u32> {
        let ind = self
            .indirect_nodes
            .get_mut(&parent)
            .ok_or_else(|| crate::Error::InvalidImage("f2fs: ghost indirect nid".into()))?;
        if ind.nids[outer] != 0 {
            return Ok(ind.nids[outer]);
        }
        let dnid = {
            let n = self.next_nid;
            self.next_nid += 1;
            n
        };
        let phys = self.alloc_node_block()?;
        self.indirect_nodes.get_mut(&parent).unwrap().nids[outer] = dnid;
        self.direct_nodes.insert(
            dnid,
            DirectNode {
                nid: dnid,
                on_disk_block: phys,
                addrs: [0; ADDRS_PER_BLOCK],
            },
        );
        Ok(dnid)
    }

    /// Create a regular file. Caller streams via `src`.
    pub fn add_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        src: FileSource,
        meta: FileMeta,
    ) -> Result<u32> {
        let (parent_nid, leaf, existing) = self.resolve_for_create(path)?;
        if existing.is_some() {
            return Err(crate::Error::InvalidArgument(format!(
                "f2fs: path exists: {path:?}"
            )));
        }
        let nid = self.alloc_nid();
        let inode_blk = self.alloc_node_block()?;
        let mode = S_IFREG | (meta.mode & 0x0FFF);
        self.inodes.insert(
            nid,
            InodeRec {
                nid,
                mode,
                uid: meta.uid,
                gid: meta.gid,
                links: 1,
                size: 0,
                blocks: 0,
                atime: meta.atime,
                ctime: meta.ctime,
                mtime: meta.mtime,
                flags: 0,
                inline_flags: 0,
                i_addr: [0; ADDRS_PER_INODE],
                i_nid: [0; NIDS_PER_INODE],
                inline_payload: Vec::new(),
                on_disk_block: inode_blk,
            },
        );

        let (reader, total) = src.open()?;
        let (size, blocks) = self.stream_into_inode(dev, nid, reader, total)?;
        let ino = self.inodes.get_mut(&nid).unwrap();
        ino.size = size;
        ino.blocks = blocks;

        self.attach_to_parent(parent_nid, &leaf, nid, F2FS_FT_REG_FILE)?;
        Ok(nid)
    }

    /// Create an empty directory.
    pub fn add_dir(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        meta: FileMeta,
    ) -> Result<u32> {
        let (parent_nid, leaf, existing) = self.resolve_for_create(path)?;
        if existing.is_some() {
            return Err(crate::Error::InvalidArgument(format!(
                "f2fs: path exists: {path:?}"
            )));
        }
        let nid = self.alloc_nid();
        let inode_blk = self.alloc_node_block()?;
        let mode = S_IFDIR | (meta.mode & 0x0FFF);
        self.inodes.insert(
            nid,
            InodeRec {
                nid,
                mode,
                uid: meta.uid,
                gid: meta.gid,
                links: 2,
                size: F2FS_BLKSIZE as u64,
                blocks: 0,
                atime: meta.atime,
                ctime: meta.ctime,
                mtime: meta.mtime,
                flags: 0,
                inline_flags: F2FS_INLINE_DENTRY,
                i_addr: [0; ADDRS_PER_INODE],
                i_nid: [0; NIDS_PER_INODE],
                inline_payload: Vec::new(),
                on_disk_block: inode_blk,
            },
        );
        self.children.insert(nid, Vec::new());

        // Parent gains a link (".." back-reference).
        if let Some(parent) = self.inodes.get_mut(&parent_nid) {
            parent.links += 1;
        }
        self.attach_to_parent(parent_nid, &leaf, nid, F2FS_FT_DIR)?;
        Ok(nid)
    }

    /// Create a symlink pointing at `target`.
    pub fn add_symlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        target: &std::path::Path,
        meta: FileMeta,
    ) -> Result<u32> {
        let (parent_nid, leaf, existing) = self.resolve_for_create(path)?;
        if existing.is_some() {
            return Err(crate::Error::InvalidArgument(format!(
                "f2fs: path exists: {path:?}"
            )));
        }
        let target_bytes = target
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("f2fs: non-UTF-8 symlink target".into()))?
            .as_bytes()
            .to_vec();
        if target_bytes.len() > MAX_INLINE_DATA {
            return Err(crate::Error::Unsupported(
                "f2fs: writer only handles symlink targets up to MAX_INLINE_DATA".into(),
            ));
        }
        let nid = self.alloc_nid();
        let inode_blk = self.alloc_node_block()?;
        let mode = S_IFLNK | (meta.mode & 0x0FFF);
        self.inodes.insert(
            nid,
            InodeRec {
                nid,
                mode,
                uid: meta.uid,
                gid: meta.gid,
                links: 1,
                size: target_bytes.len() as u64,
                blocks: 0,
                atime: meta.atime,
                ctime: meta.ctime,
                mtime: meta.mtime,
                flags: 0,
                inline_flags: F2FS_INLINE_DATA | F2FS_DATA_EXIST,
                i_addr: [0; ADDRS_PER_INODE],
                i_nid: [0; NIDS_PER_INODE],
                inline_payload: target_bytes,
                on_disk_block: inode_blk,
            },
        );

        self.attach_to_parent(parent_nid, &leaf, nid, F2FS_FT_SYMLINK)?;
        Ok(nid)
    }

    /// Create a special file (chr/blk/fifo/sock).
    pub fn add_device(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: FileMeta,
    ) -> Result<u32> {
        let (parent_nid, leaf, existing) = self.resolve_for_create(path)?;
        if existing.is_some() {
            return Err(crate::Error::InvalidArgument(format!(
                "f2fs: path exists: {path:?}"
            )));
        }
        let (mode_type, ft) = match kind {
            DeviceKind::Char => (S_IFCHR, F2FS_FT_CHRDEV),
            DeviceKind::Block => (S_IFBLK, F2FS_FT_BLKDEV),
            DeviceKind::Fifo => (S_IFIFO, F2FS_FT_FIFO),
            DeviceKind::Socket => (S_IFSOCK, F2FS_FT_SOCK),
        };
        let nid = self.alloc_nid();
        let inode_blk = self.alloc_node_block()?;
        // Pack devt into the first 8 bytes of inline_payload.
        let mut payload = vec![0u8; 8];
        let devt =
            ((major as u64) << 8) | (minor as u64 & 0xFF) | ((minor as u64 & 0xFFFFFF00) << 12);
        payload[..8].copy_from_slice(&devt.to_le_bytes());

        self.inodes.insert(
            nid,
            InodeRec {
                nid,
                mode: mode_type | (meta.mode & 0x0FFF),
                uid: meta.uid,
                gid: meta.gid,
                links: 1,
                size: 0,
                blocks: 0,
                atime: meta.atime,
                ctime: meta.ctime,
                mtime: meta.mtime,
                flags: 0,
                inline_flags: F2FS_INLINE_DATA | F2FS_DATA_EXIST,
                i_addr: [0; ADDRS_PER_INODE],
                i_nid: [0; NIDS_PER_INODE],
                inline_payload: payload,
                on_disk_block: inode_blk,
            },
        );

        self.attach_to_parent(parent_nid, &leaf, nid, ft)?;
        Ok(nid)
    }

    /// Create a hard link from `dst_path` to the inode at `src_path`.
    /// The source must be a path the writer already created in the current
    /// session — fresh-image writers don't re-decode on-disk inodes from
    /// the device, so previously-flushed nids that aren't tracked here
    /// can't be hard-linked. Symlinks and directories are rejected (Unix
    /// only permits hard links to non-directory files; we additionally
    /// disallow symlinks for cross-FS portability).
    pub fn add_hardlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        src_path: &std::path::Path,
        dst_path: &std::path::Path,
    ) -> Result<u32> {
        let src_nid = self.resolve_existing(src_path)?;
        let src = self
            .inodes
            .get(&src_nid)
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "f2fs: hardlink source {src_path:?} not tracked by this writer"
                ))
            })?
            .clone();
        if src.mode & super::constants::S_IFMT == S_IFDIR {
            return Err(crate::Error::InvalidArgument(
                "f2fs: cannot hard-link a directory".into(),
            ));
        }
        if src.mode & super::constants::S_IFMT == S_IFLNK {
            return Err(crate::Error::InvalidArgument(
                "f2fs: cannot hard-link a symbolic link".into(),
            ));
        }
        let (parent_nid, leaf, existing) = self.resolve_for_create(dst_path)?;
        if existing.is_some() {
            return Err(crate::Error::InvalidArgument(format!(
                "f2fs: path exists: {dst_path:?}"
            )));
        }
        // Re-borrow mutably to bump link count now that the dst slot is free.
        let src_mut = self
            .inodes
            .get_mut(&src_nid)
            .expect("source inode disappeared between lookup and update");
        src_mut.links = src_mut.links.saturating_add(1);
        // Recompute file_type from the existing mode so we never lie in
        // the dentry table.
        let ft = super::dir::file_type_from_mode(src.mode);
        self.attach_to_parent(parent_nid, &leaf, src_nid, ft)?;
        Ok(src_nid)
    }

    /// Resolve a posix-style path to an existing inode this writer tracks.
    /// Unlike [`Self::resolve_for_create`], the leaf must exist.
    fn resolve_existing(&self, path: &std::path::Path) -> Result<u32> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("f2fs: non-UTF-8 path".into()))?;
        if s == "/" || s.is_empty() {
            return Ok(3);
        }
        let parts: Vec<&str> = s
            .trim_matches('/')
            .split('/')
            .filter(|p| !p.is_empty())
            .collect();
        let mut cur = 3u32;
        for comp in &parts {
            let kids = self.children.get(&cur).ok_or_else(|| {
                crate::Error::InvalidArgument(format!("f2fs: parent of {s:?} not a known dir"))
            })?;
            let found = kids
                .iter()
                .find(|d| d.name == comp.as_bytes())
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!("f2fs: no such entry {comp:?} in {s:?}"))
                })?;
            cur = found.ino;
        }
        Ok(cur)
    }

    /// Add a dentry to the parent. Spills inline → block when needed.
    fn attach_to_parent(
        &mut self,
        parent_nid: u32,
        name: &[u8],
        child_nid: u32,
        file_type: u8,
    ) -> Result<()> {
        let dentry = Dentry {
            hash: 0,
            ino: child_nid,
            file_type,
            name: name.to_vec(),
        };
        self.children.entry(parent_nid).or_default().push(dentry);
        // Re-check whether the parent still fits in inline space; spill
        // to a regular dentry block if not.
        let kids = self.children.get(&parent_nid).unwrap();
        let inline_ok = fits_in_inline(kids);
        if !inline_ok {
            // Spill: clear INLINE_DENTRY and allocate a 4 KiB dentry block.
            let ino = self
                .inodes
                .get_mut(&parent_nid)
                .ok_or_else(|| crate::Error::InvalidImage("f2fs: ghost parent".into()))?;
            if ino.inline_flags & F2FS_INLINE_DENTRY != 0 {
                ino.inline_flags &= !F2FS_INLINE_DENTRY;
                ino.inline_payload.clear();
            }
            self.spilled_dirs.insert(parent_nid, ());
            // Allocate one data block for now. Multi-block dentry growth
            // is handled at flush time.
            if ino.i_addr[0] == 0 {
                let phys = self.alloc_data_block()?;
                self.inodes.get_mut(&parent_nid).unwrap().i_addr[0] = phys;
                self.inodes.get_mut(&parent_nid).unwrap().blocks = 1;
            }
        }
        Ok(())
    }

    /// Remove a file / empty dir / symlink at `path`. Returns
    /// `InvalidArgument` for a non-empty directory.
    pub fn remove(&mut self, _dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<()> {
        let (parent_nid, leaf, existing) = self.resolve_for_create(path)?;
        let child = existing
            .ok_or_else(|| crate::Error::InvalidArgument(format!("f2fs: not found: {path:?}")))?;
        // Non-empty directory?
        if let Some(grand) = self.children.get(&child) {
            if !grand.is_empty() {
                return Err(crate::Error::InvalidArgument(format!(
                    "f2fs: directory not empty: {path:?}"
                )));
            }
        }
        // Detach.
        if let Some(kids) = self.children.get_mut(&parent_nid) {
            kids.retain(|d| d.name != leaf);
        }
        // Drop bookkeeping for the child (its on-disk blocks become
        // "abandoned" but that's fine for a fresh-image writer; the
        // checkpoint will simply omit the nid from NAT).
        self.inodes.remove(&child);
        self.children.remove(&child);
        // If the removed entry was a directory, the parent loses a link.
        if let Some(parent) = self.inodes.get_mut(&parent_nid) {
            if parent.links > 1 {
                parent.links -= 1;
            }
        }
        Ok(())
    }

    /// Persist every dirty inode + dnode + indirect node, refresh NAT,
    /// and stamp a fresh CP head. After this returns the on-disk image
    /// is internally consistent.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        let bs = F2FS_BLKSIZE as u64;

        // 1) Write any dentry block(s) for directories that have spilled.
        //
        // For every spilled directory we greedily pack its child list into
        // 4 KiB dentry blocks (NR_DENTRY_IN_BLOCK slots per block). The
        // first block reuses the pre-allocated `i_addr[0]`; subsequent
        // blocks are routed through `place_data_block`, which transparently
        // handles direct-in-inode / direct-node / indirect-node / triple
        // indirect placement — exactly like a regular file.
        for (&dir_nid, _) in self.spilled_dirs.clone().iter() {
            let kids = self.children.get(&dir_nid).cloned().unwrap_or_default();
            let chunks = split_dentries_per_block(&kids);
            if chunks.is_empty() {
                continue;
            }
            // Block 0 — already pre-allocated when the directory first
            // spilled (see `attach_to_parent`). If somehow not (paranoid
            // path) allocate now.
            let mut block_addrs: Vec<u32> = Vec::with_capacity(chunks.len());
            let first_phys = {
                let ino = self
                    .inodes
                    .get(&dir_nid)
                    .ok_or_else(|| crate::Error::InvalidImage("f2fs: ghost dir".into()))?;
                ino.i_addr[0]
            };
            if first_phys == 0 {
                let phys = self.alloc_data_block()?;
                self.inodes.get_mut(&dir_nid).unwrap().i_addr[0] = phys;
                block_addrs.push(phys);
            } else {
                block_addrs.push(first_phys);
            }
            // Blocks 1..N — allocate + register.
            for idx in 1..chunks.len() {
                let phys = self.alloc_data_block()?;
                self.place_data_block(dir_nid, idx as u64, phys)?;
                block_addrs.push(phys);
            }
            // Stamp dentry blocks to disk.
            for (idx, chunk) in chunks.iter().enumerate() {
                let blk = encode_block_dentry(chunk);
                dev.write_at(block_addrs[idx] as u64 * bs, &blk)?;
            }
            // Update inode size / blocks to reflect the dentry block(s).
            let ino = self.inodes.get_mut(&dir_nid).unwrap();
            ino.size = (chunks.len() as u64) * F2FS_BLKSIZE as u64;
            ino.blocks = chunks.len() as u64;
        }

        // 2) Write every direct-node block.
        for (_, d) in self.direct_nodes.iter() {
            let ino = find_owner_of_dnode(self, d.nid);
            let blk = encode_direct_node_with_crc(&d.addrs, d.nid, ino);
            dev.write_at(d.on_disk_block as u64 * bs, &blk)?;
        }
        // 3) Write every indirect-node block.
        for (_, ind) in self.indirect_nodes.iter() {
            let ino = find_owner_of_indirect(self, ind.nid);
            let blk = encode_indirect_node_with_crc(&ind.nids, ind.nid, ino);
            dev.write_at(ind.on_disk_block as u64 * bs, &blk)?;
        }
        // Build parent_of: every inode's parent nid (root self-parents).
        let mut parent_of: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        parent_of.insert(3, 3); // root → root
        for (&parent, kids) in &self.children {
            for kid in kids {
                parent_of.insert(kid.ino, parent);
            }
        }

        // Recompute every inode's `i_blocks`. fsck.f2fs counts:
        //   inode block (1) + direct-node blocks owned + indirect-node
        //   blocks owned + non-zero i_addr / dnode addr entries.
        // Earlier writer paths set this to "data blocks streamed" which
        // missed the inode block itself plus any node-tree overhead.
        let mut blocks_for: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        for (&nid, ino) in self.inodes.iter() {
            let mut count: u64 = 1; // the inode block
            for a in ino.i_addr.iter() {
                if *a != 0 {
                    count += 1;
                }
            }
            blocks_for.insert(nid, count);
        }
        for (_, d) in self.direct_nodes.iter() {
            let owner = find_owner_of_dnode(self, d.nid);
            // The direct-node block itself.
            *blocks_for.entry(owner).or_insert(1) += 1;
            for a in d.addrs.iter() {
                if *a != 0 {
                    *blocks_for.entry(owner).or_insert(1) += 1;
                }
            }
        }
        for (_, ind) in self.indirect_nodes.iter() {
            let owner = find_owner_of_indirect(self, ind.nid);
            *blocks_for.entry(owner).or_insert(1) += 1;
        }

        // 4) Write every inode block.
        for (_, ino) in self.inodes.iter() {
            let kids = if ino.inline_flags & F2FS_INLINE_DENTRY != 0 {
                self.children.get(&ino.nid).cloned()
            } else {
                None
            };
            let parent_nid = *parent_of.get(&ino.nid).unwrap_or(&ino.nid);
            let blocks_count = *blocks_for.get(&ino.nid).unwrap_or(&1);
            let blk = encode_inode_block(ino, kids.as_deref(), parent_nid, blocks_count);
            dev.write_at(ino.on_disk_block as u64 * bs, &blk)?;
        }

        // 5) NAT — write both halves (we don't distinguish active/shadow
        //    on a fresh image). Each pack maps nid → (version, ino, block).
        let nat_pages_per_pack =
            (self.geom.segment_count_nat * self.geom.blocks_per_seg) as usize / 2;
        let mut pages: Vec<Vec<(u8, u32, u32)>> = vec![Vec::new(); nat_pages_per_pack];
        // Walk the union of all node records and assign each to its page.
        let mut all_nodes: Vec<(u32, u32, u32)> = Vec::new(); // (nid, ino_owner, block)
        for (_, ino) in self.inodes.iter() {
            all_nodes.push((ino.nid, ino.nid, ino.on_disk_block));
        }
        for (_, d) in self.direct_nodes.iter() {
            all_nodes.push((d.nid, find_owner_of_dnode(self, d.nid), d.on_disk_block));
        }
        for (_, ind) in self.indirect_nodes.iter() {
            all_nodes.push((
                ind.nid,
                find_owner_of_indirect(self, ind.nid),
                ind.on_disk_block,
            ));
        }
        // Special inodes nid=1 (node_ino) and nid=2 (meta_ino) must have
        // valid block_addr entries in the NAT — fsck.f2fs (Android fork)
        // rejects any slot whose `ne.ino != 0` but `block_addr == 0`,
        // and also rejects slots whose `ne.ino == 0` but `block_addr !=
        // 0`. mkfs.f2fs writes `block_addr = 1` for both (a placeholder
        // in the SB region — not in the main area). Match that exactly.
        // Tuple is (nid, ino, block_addr); version is hard-coded below.
        all_nodes.push((1, 1, 1)); // node_ino → ino=1, blk_addr=1
        all_nodes.push((2, 2, 1)); // meta_ino → ino=2, blk_addr=1

        for (nid, owner, blk) in all_nodes {
            let page_idx = (nid as usize) / super::constants::NAT_ENTRY_PER_BLOCK;
            let slot = (nid as usize) % super::constants::NAT_ENTRY_PER_BLOCK;
            if page_idx >= pages.len() {
                return Err(crate::Error::InvalidArgument(format!(
                    "f2fs: nid {nid} exceeds NAT capacity"
                )));
            }
            while pages[page_idx].len() <= slot {
                pages[page_idx].push((0, 0, 0));
            }
            // mkfs.f2fs uses version=0 across the board on a fresh image.
            pages[page_idx][slot] = (0, owner, blk);
        }
        for (pidx, slots) in pages.iter().enumerate() {
            let page = super::format::encode_nat_page(slots);
            for half in [0u32, 1u32] {
                let phys = self.geom.nat_blkaddr + half * (nat_pages_per_pack as u32) + pidx as u32;
                dev.write_at(phys as u64 * bs, &page)?;
            }
        }

        // 6) SIT — one entry per main-area segment. Each entry is
        //    `vblocks u16 | valid_map[64] | mtime u64` (74 bytes packed).
        //    `vblocks = (curseg_type << 10) | valid_count`. fsck.f2fs's
        //    `fsck_chk_curseg_info` rejects the image if the curseg's
        //    segment SIT entry doesn't carry the matching high bits.
        //
        //    For each segment in `node_segments` we set
        //    `(CURSEG_HOT_NODE << 10) | n` where n is the count of
        //    blocks actually placed there; same for `data_segments`
        //    with `CURSEG_HOT_DATA`. Segments 1, 2, 4, 5 stay at zero
        //    valid_count but carry the WARM/COLD type bits so the
        //    unused-but-named cursegs validate.
        let main_segs = self.geom.segment_count_main as usize;
        let bps = self.geom.blocks_per_seg as usize;
        let sit_segments_per_half = self.geom.segment_count_sit / 2;
        let sit_pages_per_half = (sit_segments_per_half * self.geom.blocks_per_seg) as usize;
        let sit_entries_per_block = F2FS_BLKSIZE / 74;
        let mut sit_pages: Vec<Vec<u8>> = vec![vec![0u8; F2FS_BLKSIZE]; sit_pages_per_half];

        // Per-segment derived state: (curseg_type bits, valid_block_count).
        // None entries stay zero (unused segments).
        let mut seg_state: Vec<(u16, usize)> = vec![(0, 0); main_segs];
        // Reserved/unused curseg "homes" carry their type bits but zero count.
        seg_state[1] = (4, 0); // CURSEG_WARM_NODE
        seg_state[2] = (5, 0); // CURSEG_COLD_NODE
        seg_state[4] = (1, 0); // CURSEG_WARM_DATA
        seg_state[5] = (2, 0); // CURSEG_COLD_DATA
        // Walk node_segments: full earlier ones, partial-current last.
        let node_active_blk = self.next_node_blk;
        for (i, &seg) in self.node_segments.iter().enumerate() {
            let count = if i + 1 < self.node_segments.len() {
                bps
            } else {
                (node_active_blk - self.seg_to_blk(seg)) as usize
            };
            seg_state[seg as usize] = (3, count); // CURSEG_HOT_NODE
        }
        let data_active_blk = self.next_data_blk;
        for (i, &seg) in self.data_segments.iter().enumerate() {
            let count = if i + 1 < self.data_segments.len() {
                bps
            } else {
                (data_active_blk - self.seg_to_blk(seg)) as usize
            };
            seg_state[seg as usize] = (0, count); // CURSEG_HOT_DATA
        }

        for (segno, (curseg_type, valid_bits)) in seg_state.iter().enumerate() {
            let page_idx = segno / sit_entries_per_block;
            if page_idx >= sit_pages.len() {
                break;
            }
            let entry_idx = segno % sit_entries_per_block;
            let off = entry_idx * 74;
            let page = &mut sit_pages[page_idx];
            let vblocks = (curseg_type << 10) | ((*valid_bits as u16) & 0x03FF);
            page[off..off + 2].copy_from_slice(&vblocks.to_le_bytes());
            for bit in 0..*valid_bits {
                let byte_idx = off + 2 + (bit / 8);
                let mask = 1u8 << (bit % 8);
                page[byte_idx] |= mask;
            }
        }
        // Write both halves of the SIT (shadow paging — content is
        // identical on a fresh image with an empty SIT version bitmap).
        for half in 0..2u32 {
            for (pidx, page) in sit_pages.iter().enumerate() {
                let phys = self.geom.sit_blkaddr + half * (sit_pages_per_half as u32) + pidx as u32;
                dev.write_at(phys as u64 * bs, page)?;
            }
        }

        // 7) SSA — blank, with CRC footers.
        let ssa_blocks = self.geom.segment_count_ssa * self.geom.blocks_per_seg;
        let ssa_blk = super::format::encode_ssa_page();
        for i in 0..ssa_blocks {
            dev.write_at((self.geom.ssa_blkaddr + i) as u64 * bs, &ssa_blk)?;
        }

        // 8) Checkpoint pack: write a fresh CP0 as an 8-block pack —
        //    matching what Android f2fs-tools' mkfs.f2fs produces for a
        //    fresh image. CP1 is deliberately zeroed so the reader picks
        //    CP0.
        //
        //    F2FS layout per the kernel docs §"Checkpoint" and
        //    `__write_checkpoint` (NR_CURSEG_DATA_TYPE + NR_CURSEG_NODE_TYPE = 6):
        //      block cp_blkaddr+0                     head CP
        //      blocks cp_blkaddr+1 .. cp_blkaddr+3    hot/warm/cold data summaries
        //      blocks cp_blkaddr+4 .. cp_blkaddr+6    hot/warm/cold node summaries
        //      block cp_blkaddr+7                     footer CP (= head copy)
        //    fsck.f2fs reads `cp_page_2 = cp_addr + total - 1` as a
        //    second CP head, AND walks the NAT journal at
        //    `cp_pack_start_sum` (which lives in the hot_data summary
        //    at cp_blkaddr+1). With that block all-zeros, n_nats=0
        //    and n_sits=0 in the journal slot, so the journal walk
        //    produces zero phantom entries.
        // Total allocated main-area blocks: sum across every segment
        // that node/data spillover claimed.
        let bps_u32 = self.geom.blocks_per_seg;
        let node_used_total: u32 = self
            .node_segments
            .iter()
            .enumerate()
            .map(|(i, &seg)| {
                if i + 1 < self.node_segments.len() {
                    bps_u32
                } else {
                    self.next_node_blk - self.seg_to_blk(seg)
                }
            })
            .sum();
        let data_used_total: u32 = self
            .data_segments
            .iter()
            .enumerate()
            .map(|(i, &seg)| {
                if i + 1 < self.data_segments.len() {
                    bps_u32
                } else {
                    self.next_data_blk - self.seg_to_blk(seg)
                }
            })
            .sum();
        let valid_blocks = (node_used_total as u64) + (data_used_total as u64);
        // fsck.f2fs (Android fork + Ubuntu 24.04 build) refuses a CP
        // that has any of `rsvd_segment_count == 0`,
        // `overprov_segment_count == 0`, or `fsmeta < F2FS_MIN_SEGMENT`
        // (typically 9). Our fsmeta is
        // `segment_count_ckpt + sit + nat + ssa + rsvd_segment_count` =
        // 2 + 1 + 2 + 1 + rsvd. So `rsvd >= 3` lifts fsmeta to 9.
        // Picking 5 for both leaves comfortable headroom on any image
        // big enough to format at all (we require ≥ 64 blocks).
        let main_segs = self.geom.segment_count_main;
        let rsvd_segment_count = 5u32.min(main_segs.saturating_sub(2));
        let overprov_segment_count = 5u32.min(main_segs.saturating_sub(2));
        // `user_block_count` is the user-visible block count after
        // reserving the GC + over-provisioned segments. fsck rejects
        // `user_block_count == 0` and `user_block_count >=
        // segment_count_main * blocks_per_seg`.
        let bps_u64 = self.geom.blocks_per_seg as u64;
        let usable_segs = main_segs
            .saturating_sub(rsvd_segment_count)
            .saturating_sub(overprov_segment_count);
        let user_block_count = (usable_segs as u64) * bps_u64;
        // SIT and NAT each occupy `segment_count_X` segments, split into
        // two halves for shadow paging. The versioning bitmap covers
        // one half — one bit per data block in that half:
        //   bytes = ((segs / 2) << log_blocks_per_seg) / 8
        // fsck.f2fs's `sanity_check_ckpt` rejects any CP whose
        // bitmap sizes don't match this formula exactly.
        let log_bps = self.geom.log_blocks_per_seg;
        let sit_ver_bitmap_bytesize = ((self.geom.segment_count_sit / 2) << log_bps) / 8;
        let nat_ver_bitmap_bytesize = ((self.geom.segment_count_nat / 2) << log_bps) / 8;
        // Curseg layout: 3 node cursegs (HOT/WARM/COLD), 3 data
        // cursegs. HOT_NODE / HOT_DATA segnos track the LAST entry in
        // `node_segments` / `data_segments` (the active segment after
        // any spillover); their blkoff is the offset within that
        // segment of the next free block. WARM/COLD point at the
        // reserved segments 1, 2, 4, 5 (always empty, blkoff=0).
        let hot_node_seg = *self.node_segments.last().unwrap();
        let hot_data_seg = *self.data_segments.last().unwrap();
        let hot_node_blkoff = (self.next_node_blk - self.seg_to_blk(hot_node_seg)) as u16;
        let hot_data_blkoff = (self.next_data_blk - self.seg_to_blk(hot_data_seg)) as u16;
        let cur_node_segno = [hot_node_seg, 1, 2];
        let cur_node_blkoff = [hot_node_blkoff, 0, 0];
        let cur_data_segno = [hot_data_seg, 4, 5];
        let cur_data_blkoff = [hot_data_blkoff, 0, 0];
        let cp = Checkpoint {
            version: 1,
            user_block_count,
            valid_block_count: valid_blocks,
            rsvd_segment_count,
            overprov_segment_count,
            flags: super::constants::CP_UMOUNT_FLAG,
            cp_pack_start_sum: 1,
            cp_pack_total_block_count: 8,
            cp_payload: 0,
            head_blkaddr: self.geom.cp_blkaddr,
            nat_ver_bitmap_bytesize,
            sit_ver_bitmap_bytesize,
            cur_nat_pack: 0,
            cur_sit_pack: 0,
            nat_journal: Vec::new(),
            cur_node_segno,
            cur_node_blkoff,
            cur_data_segno,
            cur_data_blkoff,
        };
        let cp_bytes = super::checkpoint::encode_cp_head_writer(&cp);
        let total = cp.cp_pack_total_block_count as u64;
        // Head at block cp_blkaddr+0.
        dev.write_at(self.geom.cp_blkaddr as u64 * bs, &cp_bytes)?;
        // Summary blocks 1 .. total-2: all zeros (the journal slots in
        // the hot_data summary at +1 thus have n_nats=0 / n_sits=0,
        // so fsck's NAT/SIT journal walks produce no entries).
        let zero = vec![0u8; F2FS_BLKSIZE];
        for off in 1..(total - 1) {
            dev.write_at((self.geom.cp_blkaddr as u64 + off) * bs, &zero)?;
        }
        // Footer at block cp_blkaddr + total - 1 (must duplicate head).
        dev.write_at((self.geom.cp_blkaddr as u64 + total - 1) * bs, &cp_bytes)?;
        // Mark CP1 invalid by zeroing every block it would occupy.
        // Zero magic and zero checksum_offset both fail
        // validate_checkpoint, so any of the 8 blocks landing on
        // cp_page_1 / cp_page_2 will reject CP1.
        let cp1_base = (self.geom.cp_blkaddr + self.geom.blocks_per_seg) as u64;
        for off in 0..total {
            dev.write_at((cp1_base + off) * bs, &zero)?;
        }
        dev.sync()?;
        Ok(())
    }
}

fn find_owner_of_dnode(w: &Writer, dnid: u32) -> u32 {
    for (_, ino) in w.inodes.iter() {
        for s in ino.i_nid.iter() {
            if *s == dnid {
                return ino.nid;
            }
        }
    }
    for (_, ind) in w.indirect_nodes.iter() {
        if ind.nids.contains(&dnid) {
            return find_owner_of_indirect(w, ind.nid);
        }
    }
    dnid
}

fn find_owner_of_indirect(w: &Writer, inid: u32) -> u32 {
    // First check inodes — direct/indirect/triple slots.
    for (_, ino) in w.inodes.iter() {
        for s in ino.i_nid.iter() {
            if *s == inid {
                return ino.nid;
            }
        }
    }
    // Triple-indirect: the indirect node may itself live under another
    // (top-level) indirect node. Walk up the chain.
    for (_, parent) in w.indirect_nodes.iter() {
        if parent.nid == inid {
            continue;
        }
        if parent.nids.contains(&inid) {
            return find_owner_of_indirect(w, parent.nid);
        }
    }
    inid
}

/// Greedy-pack a child list into 4 KiB dentry blocks. Each block holds at
/// most `NR_DENTRY_IN_BLOCK` slots; a single dentry consumes
/// `ceil(name_len / F2FS_SLOT_LEN)` slots (or 1 slot for an empty name).
/// Returns the per-block child slices in the order they should be written.
fn split_dentries_per_block(entries: &[Dentry]) -> Vec<Vec<Dentry>> {
    if entries.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<Vec<Dentry>> = Vec::new();
    let mut cur: Vec<Dentry> = Vec::new();
    let mut cur_slots = 0usize;
    for e in entries {
        let need = e.name.len().div_ceil(F2FS_SLOT_LEN).max(1);
        if cur_slots + need > NR_DENTRY_IN_BLOCK {
            out.push(std::mem::take(&mut cur));
            cur_slots = 0;
        }
        cur.push(e.clone());
        cur_slots += need;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// True if a list of dentries still fits in the inline-dentry layout.
fn fits_in_inline(entries: &[Dentry]) -> bool {
    let mut slot = 0usize;
    for e in entries {
        let need = e.name.len().div_ceil(F2FS_SLOT_LEN).max(1);
        if slot + need > INLINE_DENTRY_NR {
            return false;
        }
        slot += need;
    }
    true
}

/// Encode a 4 KiB dentry block from a child list.
fn encode_block_dentry(entries: &[Dentry]) -> Vec<u8> {
    let mut buf = vec![0u8; F2FS_BLKSIZE];
    let bitmap_off = 0usize;
    let dentries_off = SIZE_OF_DENTRY_BITMAP + SIZE_OF_RESERVED;
    let names_off = dentries_off + NR_DENTRY_IN_BLOCK * SIZE_OF_DIR_ENTRY;
    let mut slot = 0;
    for e in entries {
        let need = e.name.len().div_ceil(F2FS_SLOT_LEN).max(1);
        if slot + need > NR_DENTRY_IN_BLOCK {
            break;
        }
        set_bit(
            &mut buf[bitmap_off..bitmap_off + SIZE_OF_DENTRY_BITMAP],
            slot,
        );
        let de_off = dentries_off + slot * SIZE_OF_DIR_ENTRY;
        buf[de_off..de_off + 4].copy_from_slice(&e.hash.to_le_bytes());
        buf[de_off + 4..de_off + 8].copy_from_slice(&e.ino.to_le_bytes());
        buf[de_off + 8..de_off + 10].copy_from_slice(&(e.name.len() as u16).to_le_bytes());
        buf[de_off + 10] = e.file_type;
        let name_start = names_off + slot * F2FS_SLOT_LEN;
        let name_end = name_start + e.name.len();
        buf[name_start..name_end].copy_from_slice(&e.name);
        slot += need;
    }
    // Footer CRC — covers everything but the trailing 4 bytes.
    let crc = super::constants::f2fs_crc32(&buf[..F2FS_BLK_CSUM_OFFSET]);
    buf[F2FS_BLK_CSUM_OFFSET..F2FS_BLK_CSUM_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    buf
}

/// Encode the inline-dentry payload region used inside an inode block.
fn encode_inline_dentry_payload(dir_nid: u32, parent_nid: u32, entries: &[Dentry]) -> Vec<u8> {
    // F2FS requires the first two dentries of every directory to be
    // "." (pointing at self) and ".." (pointing at parent). Root's
    // parent is itself. fsck.f2fs (`fsck_chk_inline_dentries`) prints
    // `dots: 0` and refuses to proceed if either is missing.
    let dot_entries = [
        Dentry {
            hash: 0,
            ino: dir_nid,
            file_type: F2FS_FT_DIR,
            name: b".".to_vec(),
        },
        Dentry {
            hash: 0,
            ino: parent_nid,
            file_type: F2FS_FT_DIR,
            name: b"..".to_vec(),
        },
    ];
    let all: Vec<&Dentry> = dot_entries.iter().chain(entries.iter()).collect();
    let bitmap_bytes = INLINE_DENTRY_NR.div_ceil(8);
    // INLINE_RESERVED_SIZE (per fsck.f2fs's
    // `INLINE_RESERVED_SIZE` macro) = MAX_INLINE_DATA - (NR * (DE +
    // SLOT) + bitmap_size) = 3488 - (182 * 19 + 23) = 7. Off-by-six
    // here means fsck reads bitmap byte 0 and dentry byte 6 from the
    // same address, scrambling the layout.
    let reserved = 7usize;
    let dentries_off = bitmap_bytes + reserved;
    let names_off = dentries_off + INLINE_DENTRY_NR * SIZE_OF_DIR_ENTRY;
    let total = names_off + INLINE_DENTRY_NR * F2FS_SLOT_LEN;
    let mut buf = vec![0u8; total];
    let mut slot = 0usize;
    for e in all {
        let need = e.name.len().div_ceil(F2FS_SLOT_LEN).max(1);
        if slot + need > INLINE_DENTRY_NR {
            break;
        }
        set_bit(&mut buf[..bitmap_bytes], slot);
        let de_off = dentries_off + slot * SIZE_OF_DIR_ENTRY;
        buf[de_off..de_off + 4].copy_from_slice(&e.hash.to_le_bytes());
        buf[de_off + 4..de_off + 8].copy_from_slice(&e.ino.to_le_bytes());
        buf[de_off + 8..de_off + 10].copy_from_slice(&(e.name.len() as u16).to_le_bytes());
        buf[de_off + 10] = e.file_type;
        let name_start = names_off + slot * F2FS_SLOT_LEN;
        let name_end = name_start + e.name.len();
        buf[name_start..name_end].copy_from_slice(&e.name);
        slot += need;
    }
    buf
}

#[inline]
fn set_bit(bitmap: &mut [u8], i: usize) {
    let byte = i / 8;
    let bit = i % 8;
    if byte < bitmap.len() {
        bitmap[byte] |= 1 << bit;
    }
}

/// Encode an inode block from an [`InodeRec`] (optionally inlining a
/// dentry payload).
/// Stamp the 24-byte `struct node_footer` at the end of a 4 KiB node
/// block: nid u32, ino u32, flag u32, cp_ver u64, next_blkaddr u32.
/// `cp_ver` matches the CP head's `checkpoint_ver` (we always write 1
/// on a fresh image — see write.rs flush()).
fn write_node_footer(buf: &mut [u8], nid: u32, ino: u32) {
    const NODE_FOOTER_OFFSET: usize = F2FS_BLKSIZE - 24;
    let o = NODE_FOOTER_OFFSET;
    buf[o..o + 4].copy_from_slice(&nid.to_le_bytes());
    buf[o + 4..o + 8].copy_from_slice(&ino.to_le_bytes());
    buf[o + 8..o + 12].copy_from_slice(&0u32.to_le_bytes()); // flag
    buf[o + 12..o + 20].copy_from_slice(&1u64.to_le_bytes()); // cp_ver
    buf[o + 20..o + 24].copy_from_slice(&0u32.to_le_bytes()); // next_blkaddr
}

fn encode_inode_block(
    ino: &InodeRec,
    inline_children: Option<&[Dentry]>,
    parent_nid: u32,
    blocks: u64,
) -> Vec<u8> {
    let mut buf = vec![0u8; F2FS_BLKSIZE];
    buf[0x00..0x02].copy_from_slice(&ino.mode.to_le_bytes());
    buf[0x03] = ino.inline_flags;
    buf[0x04..0x08].copy_from_slice(&ino.uid.to_le_bytes());
    buf[0x08..0x0C].copy_from_slice(&ino.gid.to_le_bytes());
    buf[0x0C..0x10].copy_from_slice(&ino.links.to_le_bytes());
    buf[0x10..0x18].copy_from_slice(&ino.size.to_le_bytes());
    buf[0x18..0x20].copy_from_slice(&blocks.to_le_bytes());
    buf[0x20..0x28].copy_from_slice(&(ino.atime as u64).to_le_bytes());
    buf[0x28..0x30].copy_from_slice(&(ino.ctime as u64).to_le_bytes());
    buf[0x30..0x38].copy_from_slice(&(ino.mtime as u64).to_le_bytes());
    buf[0x50..0x54].copy_from_slice(&ino.flags.to_le_bytes());

    // Inline payload sits at `i_addr[DEF_INLINE_RESERVED_SIZE]` (=
    // i_addr[1]), NOT at `i_addr[0]` — fsck.f2fs's `inline_data_addr`
    // skips the first `__le32` slot. Mismatching this by 4 bytes was
    // why fsck reported `dots: 0` despite the writer emitting both
    // "." and "..".
    const INLINE_PAYLOAD_OFFSET: usize = I_ADDR_OFFSET + 4;
    if ino.inline_flags & F2FS_INLINE_DENTRY != 0 {
        let kids = inline_children.unwrap_or(&[]);
        let payload = encode_inline_dentry_payload(ino.nid, parent_nid, kids);
        let n = payload.len().min(buf.len() - INLINE_PAYLOAD_OFFSET - 8);
        buf[INLINE_PAYLOAD_OFFSET..INLINE_PAYLOAD_OFFSET + n].copy_from_slice(&payload[..n]);
    } else if ino.inline_flags & F2FS_INLINE_DATA != 0 {
        let payload = &ino.inline_payload;
        let n = payload.len().min(buf.len() - INLINE_PAYLOAD_OFFSET - 8);
        buf[INLINE_PAYLOAD_OFFSET..INLINE_PAYLOAD_OFFSET + n].copy_from_slice(&payload[..n]);
    } else {
        for (i, a) in ino.i_addr.iter().enumerate() {
            let o = I_ADDR_OFFSET + i * 4;
            buf[o..o + 4].copy_from_slice(&a.to_le_bytes());
        }
        let nid_off = I_ADDR_OFFSET + ADDRS_PER_INODE * 4;
        for (i, a) in ino.i_nid.iter().enumerate() {
            let o = nid_off + i * 4;
            buf[o..o + 4].copy_from_slice(&a.to_le_bytes());
        }
    }

    // Node blocks carry no trailing CRC — the last 24 bytes are the
    // `struct node_footer` instead. fsck.f2fs reads footer.nid and
    // footer.ino to validate the block, and writing a u32 at offset
    // 4092 would clobber next_blkaddr.
    write_node_footer(&mut buf, ino.nid, ino.nid);
    buf
}

/// 4 KiB direct-node block: 1018 le u32 data-block pointers + node_footer.
fn encode_direct_node_with_crc(ptrs: &[u32; ADDRS_PER_BLOCK], nid: u32, ino: u32) -> Vec<u8> {
    let mut buf = vec![0u8; F2FS_BLKSIZE];
    const FOOTER_OFFSET: usize = F2FS_BLKSIZE - 24;
    for (i, p) in ptrs.iter().enumerate() {
        let o = i * 4;
        if o + 4 > FOOTER_OFFSET {
            break;
        }
        buf[o..o + 4].copy_from_slice(&p.to_le_bytes());
    }
    write_node_footer(&mut buf, nid, ino);
    buf
}

/// 4 KiB indirect-node block: 1018 le u32 nids + node_footer.
fn encode_indirect_node_with_crc(nids: &[u32; NIDS_PER_BLOCK], nid: u32, ino: u32) -> Vec<u8> {
    let mut buf = vec![0u8; F2FS_BLKSIZE];
    const FOOTER_OFFSET: usize = F2FS_BLKSIZE - 24;
    for (i, n) in nids.iter().enumerate() {
        let o = i * 4;
        if o + 4 > FOOTER_OFFSET {
            break;
        }
        buf[o..o + 4].copy_from_slice(&n.to_le_bytes());
    }
    write_node_footer(&mut buf, nid, ino);
    buf
}

// Make BlockDevice see ReadSeek — already provided by crate::fs.
// (Kept here only as documentation; no extra impls.)
#[allow(dead_code)]
fn _dummy_use_of_reader_type(_: Box<dyn crate::fs::ReadSeek + Send>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn fits_in_inline_handles_long_names() {
        // 22 entries × 8-byte names each = 22 slots. Plus 11-byte name → 2
        // slots. Well within 182.
        let mut entries: Vec<Dentry> = Vec::new();
        for i in 0..30 {
            entries.push(Dentry {
                hash: 0,
                ino: 100 + i,
                file_type: F2FS_FT_REG_FILE,
                name: format!("longishfilename_{i}").into_bytes(),
            });
        }
        // 30 entries × 2 slots = 60 slots, fits.
        assert!(fits_in_inline(&entries));
        // 200 entries → overflow inline.
        let mut overflow: Vec<Dentry> = Vec::new();
        for i in 0..200 {
            overflow.push(Dentry {
                hash: 0,
                ino: i,
                file_type: F2FS_FT_REG_FILE,
                name: b"x".to_vec(),
            });
        }
        assert!(!fits_in_inline(&overflow));
    }

    /// Greedy chunker emits no chunks for an empty list, one chunk for
    /// trivially-small lists, and exactly the right number of chunks when
    /// the slot budget overflows.
    #[test]
    fn split_dentries_per_block_packs_greedily() {
        // Empty.
        assert!(split_dentries_per_block(&[]).is_empty());

        // One entry — fits in one block.
        let one = vec![Dentry {
            hash: 0,
            ino: 7,
            file_type: F2FS_FT_REG_FILE,
            name: b"a".to_vec(),
        }];
        assert_eq!(split_dentries_per_block(&one).len(), 1);

        // 300 entries with 1-slot names → ceil(300 / 214) = 2 blocks.
        let mut many: Vec<Dentry> = Vec::new();
        for i in 0..300 {
            many.push(Dentry {
                hash: 0,
                ino: i,
                file_type: F2FS_FT_REG_FILE,
                name: b"x".to_vec(),
            });
        }
        let chunks = split_dentries_per_block(&many);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), 214);
        assert_eq!(chunks[1].len(), 86);
    }

    /// Synthesize a tracked file inode at a block index that falls into
    /// the triple-indirect region, then verify the writer plumbed all
    /// three tiers correctly (top-indirect, mid-indirect, direct).
    #[test]
    fn place_data_block_builds_triple_indirect_tree() {
        // Build a minimal Writer state from a formatted device. We use
        // `super::super::F2fs::format` so the geometry + root inode are
        // wired up correctly; then we reach into the writer to invoke
        // `place_data_block` directly.
        let mut dev = MemoryBackend::new(2 * 1024 * 1024);
        let opts = super::super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::super::FormatOpts::default()
        };
        let mut fs = super::super::F2fs::format(&mut dev, &opts).unwrap();
        let writer = fs.writer.as_mut().expect("formatted handle has a writer");

        // Allocate an inode record by hand at nid 4 (next_nid).
        let inode_nid = writer.next_nid;
        writer.next_nid += 1;
        let inode_blk = writer.alloc_node_block().unwrap();
        writer.inodes.insert(
            inode_nid,
            InodeRec {
                nid: inode_nid,
                mode: S_IFREG | 0o644,
                uid: 0,
                gid: 0,
                links: 1,
                size: 0,
                blocks: 0,
                atime: 0,
                ctime: 0,
                mtime: 0,
                flags: 0,
                inline_flags: 0,
                i_addr: [0; ADDRS_PER_INODE],
                i_nid: [0; NIDS_PER_INODE],
                inline_payload: Vec::new(),
                on_disk_block: inode_blk,
            },
        );

        // Threshold of the triple region. Anything just past it lands in
        // the triple branch.
        let double_span = (NIDS_PER_BLOCK as u64) * (ADDRS_PER_BLOCK as u64) * 2;
        let triple_base = ADDRS_PER_INODE as u64 + (ADDRS_PER_BLOCK as u64) * 2 + double_span;
        let idx = triple_base + 7;

        let phys = writer.alloc_data_block().unwrap();
        writer.place_data_block(inode_nid, idx, phys).unwrap();

        // Now we expect: i_nid[NID_TRIPLE_INDIRECT] points at a top
        // indirect block, that top block's slot 0 points at a mid indirect
        // block, that mid block's slot 0 points at a direct block, and
        // that direct block's slot 7 holds `phys`.
        let ino = writer.inodes.get(&inode_nid).unwrap();
        let top_nid = ino.i_nid[NID_TRIPLE_INDIRECT];
        assert!(top_nid != 0, "triple top nid was never assigned");
        let top = writer.indirect_nodes.get(&top_nid).unwrap();
        let mid_nid = top.nids[0];
        assert!(mid_nid != 0, "triple mid nid was never assigned");
        let mid = writer.indirect_nodes.get(&mid_nid).unwrap();
        let dnode_nid = mid.nids[0];
        assert!(dnode_nid != 0, "direct node under triple-mid missing");
        let dnode = writer.direct_nodes.get(&dnode_nid).unwrap();
        assert_eq!(dnode.addrs[7], phys);
        // And the inode hasn't accidentally clobbered its in-inode array.
        assert_eq!(ino.i_addr[0], 0);
    }

    /// End-to-end triple-indirect: create a real file whose final block
    /// sits past the double-indirect span. The body is a sparse-zero
    /// `FileSource::Zero`, so we don't pay for 8 GiB of memory backing —
    /// the writer streams pump-buffer-sized zero blocks straight onto a
    /// fake-large device.
    ///
    /// Streaming the whole file is *still* expensive (each block is a
    /// 4 KiB write); to keep the test under a second we shrink the run by
    /// writing only the *last* block of the file via a manual placement —
    /// reusing the placement path already exercised by
    /// `place_data_block_builds_triple_indirect_tree` — then patching the
    /// inode size by hand so the reader knows where the file ends.
    ///
    /// Reader walks the same logical_to_physical chain and recovers the
    /// single non-zero block byte, validating triple-indirect end-to-end.
    #[test]
    fn triple_indirect_round_trip_synthetic() {
        let mut dev = MemoryBackend::new(2 * 1024 * 1024);
        let opts = super::super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::super::FormatOpts::default()
        };
        let mut fs = super::super::F2fs::format(&mut dev, &opts).unwrap();

        // Attach a fresh file to the root so the dentry path resolves.
        let inode_nid = {
            let writer = fs.writer.as_mut().unwrap();
            let nid = writer.next_nid;
            writer.next_nid += 1;
            let inode_blk = writer.alloc_node_block().unwrap();
            writer.inodes.insert(
                nid,
                InodeRec {
                    nid,
                    mode: S_IFREG | 0o644,
                    uid: 0,
                    gid: 0,
                    links: 1,
                    size: 0,
                    blocks: 0,
                    atime: 0,
                    ctime: 0,
                    mtime: 0,
                    flags: 0,
                    inline_flags: 0,
                    i_addr: [0; ADDRS_PER_INODE],
                    i_nid: [0; NIDS_PER_INODE],
                    inline_payload: Vec::new(),
                    on_disk_block: inode_blk,
                },
            );
            writer
                .attach_to_parent(3, b"sparse.bin", nid, F2FS_FT_REG_FILE)
                .unwrap();
            nid
        };

        // Plant a single data block deep in the triple-indirect span. We
        // sit one block past the boundary so the math is unambiguous.
        let double_span = (NIDS_PER_BLOCK as u64) * (ADDRS_PER_BLOCK as u64) * 2;
        let triple_base = ADDRS_PER_INODE as u64 + (ADDRS_PER_BLOCK as u64) * 2 + double_span;
        let idx = triple_base; // first block of the triple region

        let phys = {
            let writer = fs.writer.as_mut().unwrap();
            let p = writer.alloc_data_block().unwrap();
            writer.place_data_block(inode_nid, idx, p).unwrap();
            // Mark inode size so the reader serves exactly one byte from
            // the planted block.
            let ino = writer.inodes.get_mut(&inode_nid).unwrap();
            ino.size = idx * F2FS_BLKSIZE as u64 + 1;
            ino.blocks = 1;
            p
        };

        // Stamp a sentinel byte into the planted block.
        dev.write_at(phys as u64 * F2FS_BLKSIZE as u64, &{
            let mut b = vec![0u8; F2FS_BLKSIZE];
            b[0] = 0xAB;
            b
        })
        .unwrap();

        fs.flush(&mut dev).unwrap();

        // Re-open with the read driver and confirm the reader walked the
        // triple-indirect tree all the way to our planted block. We seek
        // to the last logical block by reading the full file (just the
        // last byte is meaningful — everything else is hole-zero).
        let mut fs2 = super::super::F2fs::open(&mut dev).unwrap();
        let mut r = fs2.open_file_reader(&mut dev, "/sparse.bin").unwrap();
        // The file is enormous; stream and only inspect the last byte.
        let total_bytes = idx * F2FS_BLKSIZE as u64 + 1;
        let mut last = 0u8;
        let mut left = total_bytes;
        let mut buf = vec![0u8; 64 * 1024];
        while left > 0 {
            let want = (left as usize).min(buf.len());
            let n = r.read(&mut buf[..want]).unwrap();
            assert!(n > 0, "reader ran dry before EOF");
            last = buf[n - 1];
            left -= n as u64;
        }
        assert_eq!(last, 0xAB);
    }

    /// Sanity: `find_owner_of_indirect` now chains up through a parent
    /// triple-top indirect node back to the inode.
    #[test]
    fn owner_chain_climbs_triple_tree() {
        let mut dev = MemoryBackend::new(2 * 1024 * 1024);
        let opts = super::super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::super::FormatOpts::default()
        };
        let mut fs = super::super::F2fs::format(&mut dev, &opts).unwrap();
        let writer = fs.writer.as_mut().unwrap();
        let inode_nid = writer.next_nid;
        writer.next_nid += 1;
        let inode_blk = writer.alloc_node_block().unwrap();
        writer.inodes.insert(
            inode_nid,
            InodeRec {
                nid: inode_nid,
                mode: S_IFREG | 0o644,
                uid: 0,
                gid: 0,
                links: 1,
                size: 0,
                blocks: 0,
                atime: 0,
                ctime: 0,
                mtime: 0,
                flags: 0,
                inline_flags: 0,
                i_addr: [0; ADDRS_PER_INODE],
                i_nid: [0; NIDS_PER_INODE],
                inline_payload: Vec::new(),
                on_disk_block: inode_blk,
            },
        );
        let double_span = (NIDS_PER_BLOCK as u64) * (ADDRS_PER_BLOCK as u64) * 2;
        let triple_base = ADDRS_PER_INODE as u64 + (ADDRS_PER_BLOCK as u64) * 2 + double_span;
        let phys = writer.alloc_data_block().unwrap();
        writer
            .place_data_block(inode_nid, triple_base, phys)
            .unwrap();

        // The mid-indirect (a child of the top-indirect) should report
        // the inode as its owner, since the chain crosses the top.
        let ino = writer.inodes.get(&inode_nid).unwrap();
        let top_nid = ino.i_nid[NID_TRIPLE_INDIRECT];
        let top = writer.indirect_nodes.get(&top_nid).unwrap();
        let mid_nid = top.nids[0];
        assert_eq!(super::find_owner_of_indirect(writer, mid_nid), inode_nid);
    }
}
