//! APFS in-place read/write file handle.
//!
//! Backs [`crate::fs::Filesystem::open_file_rw`] on an APFS volume that
//! was opened via [`super::Apfs::open_writable`]. Scope:
//!
//! - **Existing-file edits only.** `open_file_rw(create = true)` is
//!   rejected — the handle assumes the catalog already contains the
//!   target file. Whole-file create / remove / rename remain
//!   unsupported.
//! - **Single-extent semantics.** On sync, the file's entire byte
//!   range is rewritten as a single new fresh extent. We don't try to
//!   patch existing extents in place because APFS extents are
//!   immutable units of the omap-resolved fs-tree; doing partial COW
//!   would require splitting + re-stitching the extent record set,
//!   which is deferred.
//! - **Checkpoint-bounded.** Each `sync()` consumes one xp_desc slot.
//!   The format-time xp_desc area has [`super::write::XP_DESC_BLOCKS`]
//!   slots; we use one on format and the remainder are spare. Once
//!   they're exhausted further syncs return `Unsupported`.
//!
//! ## Crash safety
//!
//! On `sync()` we COW the whole metadata stack:
//!
//! 1. Allocate fresh blocks past the previous bump high-water for the
//!    rewritten file extent.
//! 2. Allocate fresh blocks for new fs-tree leaves, the new volume
//!    omap, a new APSB, and a new container omap.
//! 3. Write the new NXSB into the next free xp_desc slot — and only
//!    then is the new checkpoint discoverable.
//!
//! A crash between steps 1–2 leaves the old NXSB pointing at the old
//! metadata stack, so the next open sees the previous checkpoint
//! intact. Step 3's NXSB write is the only commit point: it's a
//! single-block write, and the reader picks the highest-xid valid
//! NXSB in the xp_desc area, so a torn NXSB block (failed Fletcher-64)
//! falls back to the previous slot.

use std::io::{Read, Seek, SeekFrom, Write};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::FileHandle;

use super::Apfs;
use super::ApfsState;
use super::fstree::FsTreeCtx;
use super::jrec::{
    APFS_TYPE_DSTREAM_ID, APFS_TYPE_FILE_EXTENT, APFS_TYPE_INODE, INO_EXT_TYPE_DSTREAM,
    J_INODE_VAL_FIXED_SIZE, OBJ_ID_MASK, OBJ_TYPE_SHIFT,
};
use super::read_at_paddr;
use super::write;

/// Read/write handle for one APFS regular file.
///
/// The handle holds `&mut Apfs` + `&mut dyn BlockDevice` for its full
/// lifetime. On `Drop` it does **not** auto-sync — call `sync()` to
/// commit the new checkpoint. Dropping without sync silently
/// discards every pending byte (mirroring `std::fs::File`).
pub struct ApfsFileHandle<'a> {
    fs: &'a mut Apfs,
    dev: &'a mut dyn BlockDevice,
    /// Target inode object id (file_id from the drec record we resolved).
    target_oid: u64,
    /// In-memory image of the file's current bytes. Populated from the
    /// existing extents at open time; mutated by `Write::write` and
    /// `set_len`.
    contents: Vec<u8>,
    /// Read / write cursor (byte offset into `contents`).
    pos: u64,
    /// True once any mutation has happened since the last sync. Cleared
    /// on `sync()`.
    dirty: bool,
}

impl<'a> ApfsFileHandle<'a> {
    /// Resolve `path`, load the file's current bytes into RAM, and
    /// return a handle ready for reads/writes.
    pub(super) fn open(
        fs: &'a mut Apfs,
        dev: &'a mut dyn BlockDevice,
        path: &str,
        flags: crate::fs::OpenFlags,
    ) -> Result<Self> {
        // Refuse if not in write mode.
        match &fs.state {
            ApfsState::Write(_) => {}
            ApfsState::Read(_) => {
                return Err(crate::Error::Unsupported(
                    "apfs: open_file_rw requires Apfs::open_writable (not Apfs::open)".into(),
                ));
            }
            ApfsState::PendingWrite(_) => {
                return Err(crate::Error::Unsupported(
                    "apfs: open_file_rw is not available in pending-write mode".into(),
                ));
            }
        }

        // Resolve path → oid via fs's reader API.
        let target_oid = fs.resolve_path_to_oid(dev, path)?;

        // Read all current bytes.
        let mut contents = {
            let mut r = fs.open_file_reader(dev, path)?;
            let mut buf = Vec::new();
            std::io::Read::read_to_end(&mut r, &mut buf)
                .map_err(|e| crate::Error::Io(std::io::Error::other(e)))?;
            buf
        };

        if flags.truncate {
            contents.clear();
        }
        let pos = if flags.append {
            contents.len() as u64
        } else {
            0
        };

        Ok(Self {
            fs,
            dev,
            target_oid,
            contents,
            pos,
            // Truncate counts as a mutation that needs committing on
            // sync; create=false guarantees we didn't synthesize a new
            // inode here.
            dirty: flags.truncate,
        })
    }
}

impl<'a> Read for ApfsFileHandle<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.contents.len() as u64 {
            return Ok(0);
        }
        let start = self.pos as usize;
        let end = (start + buf.len()).min(self.contents.len());
        let n = end - start;
        buf[..n].copy_from_slice(&self.contents[start..end]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl<'a> Write for ApfsFileHandle<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let end = (self.pos as usize)
            .checked_add(buf.len())
            .ok_or_else(|| std::io::Error::other("apfs: write offset overflow"))?;
        if end > self.contents.len() {
            self.contents.resize(end, 0);
        }
        let start = self.pos as usize;
        self.contents[start..end].copy_from_slice(buf);
        self.pos = end as u64;
        self.dirty = true;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // Per the FileHandle contract, `flush` is a hint; we leave
        // bytes in `contents` until `sync()` actually commits a
        // checkpoint. (Eagerly committing a checkpoint per Write::flush
        // would burn xp_desc slots fast.)
        Ok(())
    }
}

impl<'a> Seek for ApfsFileHandle<'a> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let target: i128 = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
            SeekFrom::End(d) => self.contents.len() as i128 + d as i128,
        };
        if target < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "apfs: seek to negative offset",
            ));
        }
        self.pos = target as u64;
        Ok(self.pos)
    }
}

impl<'a> FileHandle for ApfsFileHandle<'a> {
    fn len(&self) -> u64 {
        self.contents.len() as u64
    }

    fn set_len(&mut self, new_len: u64) -> Result<()> {
        let new_len = new_len as usize;
        if new_len != self.contents.len() {
            self.contents.resize(new_len, 0);
            self.dirty = true;
            if self.pos > new_len as u64 {
                self.pos = new_len as u64;
            }
        }
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let target_oid = self.target_oid;
        let new_bytes = std::mem::take(&mut self.contents);
        let new_size = new_bytes.len() as u64;
        commit_with_mutator(self.fs, self.dev, |cx| {
            let bs = cx.block_size() as u64;
            // Drop the target's existing FILE_EXTENT / DSTREAM_ID records;
            // a fresh single extent is allocated and written below.
            cx.records.retain(|(k, _)| {
                if k.len() < 8 {
                    return true;
                }
                let hdr = u64::from_le_bytes(k[0..8].try_into().unwrap());
                let oid = hdr & OBJ_ID_MASK;
                let kind = (hdr >> OBJ_TYPE_SHIFT) as u8;
                !(oid == target_oid
                    && (kind == APFS_TYPE_FILE_EXTENT || kind == APFS_TYPE_DSTREAM_ID))
            });
            // Patch the inode's DSTREAM xfield to advertise the new size.
            for (k, v) in cx.records.iter_mut() {
                if k.len() < 8 {
                    continue;
                }
                let hdr = u64::from_le_bytes(k[0..8].try_into().unwrap());
                let oid = hdr & OBJ_ID_MASK;
                let kind = (hdr >> OBJ_TYPE_SHIFT) as u8;
                if oid == target_oid && kind == APFS_TYPE_INODE {
                    patch_inode_dstream_size(v, new_size, bs);
                }
            }
            // Allocate a fresh extent for the new bytes (skipped for an
            // empty file — no FILE_EXTENT / DSTREAM_ID then either).
            if new_size > 0 {
                let extent_paddr = cx.alloc_extent(new_size)?;
                cx.write_extent_bytes(extent_paddr, &new_bytes)?;
                let block_size_u32 = cx.block_size();
                for (k, v) in super::write::build_file_extent_records(
                    target_oid,
                    0,
                    new_size,
                    extent_paddr,
                    block_size_u32,
                ) {
                    cx.records.push((k, v));
                }
            }
            Ok(())
        })?;
        self.dirty = false;
        Ok(())
    }
}

/// Mutation context handed to closures passed into
/// [`commit_with_mutator`]. Carries everything a Write-state
/// mutator might touch:
///
/// - the dumped fs-tree records (read/write),
/// - the underlying device (for extent body writes),
/// - the bump-allocator high-water (advanced when a new extent is
///   allocated),
/// - the per-volume counters the APSB tracks (`next_oid`,
///   `num_files`, `num_directories`, `num_symlinks`).
///
/// Counters and `bump_high_water` are read back by
/// [`commit_with_mutator`] after the closure returns and threaded
/// through to [`write::ApfsWriter::new_checkpoint`], so the new
/// checkpoint's APSB reflects whatever the closure did.
pub(crate) struct MutatorCx<'a> {
    /// In-memory dump of every existing fs-tree record. Mutators
    /// add / remove / patch entries here; the order doesn't matter
    /// (the writer sorts on flush).
    pub records: &'a mut Vec<(Vec<u8>, Vec<u8>)>,
    dev: &'a mut dyn BlockDevice,
    block_size: u32,
    total_blocks: u64,
    bump_high_water: u64,
    next_oid: u64,
    num_files: u64,
    num_directories: u64,
    num_symlinks: u64,
}

impl<'a> MutatorCx<'a> {
    /// Block size in bytes (4096 on every APFS image in the wild).
    pub(crate) fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Bump-allocate a fresh extent of `len_bytes` (rounded up to a
    /// whole block). Returns the physical block number the extent
    /// starts at. Subsequent calls return non-overlapping ranges.
    /// Errors when the device doesn't have enough remaining blocks.
    pub(crate) fn alloc_extent(&mut self, len_bytes: u64) -> Result<u64> {
        if len_bytes == 0 {
            return Ok(0);
        }
        let bs = self.block_size as u64;
        let blocks = len_bytes.div_ceil(bs);
        let start = self.bump_high_water;
        let end = start.checked_add(blocks).ok_or_else(|| {
            crate::Error::InvalidArgument("apfs: extent allocation overflow".into())
        })?;
        if end > self.total_blocks {
            return Err(crate::Error::InvalidArgument(format!(
                "apfs: not enough free blocks to allocate {len_bytes} bytes \
                 (need {blocks}, have {})",
                self.total_blocks - self.bump_high_water
            )));
        }
        self.bump_high_water = end;
        Ok(start)
    }

    /// Write `bytes` into the extent starting at `paddr`. Pads the
    /// final block with zeros if `bytes.len()` isn't a multiple of
    /// `block_size`. The caller is responsible for `paddr` having
    /// come from a recent [`Self::alloc_extent`] call sized for
    /// `bytes.len()`.
    pub(crate) fn write_extent_bytes(&mut self, paddr: u64, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let bs = self.block_size as usize;
        let bs_u64 = bs as u64;
        let blocks = bytes.len().div_ceil(bs);
        let mut blk = vec![0u8; bs];
        for i in 0..blocks {
            let off = i * bs;
            let end = (off + bs).min(bytes.len());
            blk.fill(0);
            blk[..end - off].copy_from_slice(&bytes[off..end]);
            self.dev.write_at((paddr + i as u64) * bs_u64, &blk)?;
        }
        Ok(())
    }

    /// Reserve and return the next free APFS object id. Bumps the
    /// internal counter so a second call returns a different value.
    #[allow(dead_code)] // used by commit-4+ callers
    pub(crate) fn alloc_oid(&mut self) -> u64 {
        let oid = self.next_oid;
        self.next_oid = self.next_oid.saturating_add(1);
        oid
    }

    /// Increment the APSB's `apfs_num_files` counter. Call this once
    /// per regular file added by the mutator.
    #[allow(dead_code)] // used by commit-4+ callers
    pub(crate) fn note_new_file(&mut self) {
        self.num_files = self.num_files.saturating_add(1);
    }

    /// Increment the APSB's `apfs_num_directories` counter.
    #[allow(dead_code)] // used by commit-4+ callers
    pub(crate) fn note_new_dir(&mut self) {
        self.num_directories = self.num_directories.saturating_add(1);
    }

    /// Increment the APSB's `apfs_num_symlinks` counter.
    #[allow(dead_code)] // used by commit-4+ callers
    pub(crate) fn note_new_symlink(&mut self) {
        self.num_symlinks = self.num_symlinks.saturating_add(1);
    }
}

/// Commit a checkpoint after running `mutator` against the dumped
/// fs-tree record list. Use this for any Write-state mutation:
/// inode attribute patches, directory entry add / remove / rename,
/// xattr edits, file creation (extent allocation via
/// [`MutatorCx::alloc_extent`]), file body rewrites — all go through
/// here, so the COW pathway lives in exactly one place.
///
/// On return, `fs.state` is refreshed to a fresh write state that
/// reflects the newly-written checkpoint.
pub(crate) fn commit_with_mutator<F>(
    fs: &mut Apfs,
    dev: &mut dyn BlockDevice,
    mutator: F,
) -> Result<()>
where
    F: FnOnce(&mut MutatorCx<'_>) -> Result<()>,
{
    let (
        block_size,
        total_blocks,
        volume_name,
        container_uuid,
        volume_uuid,
        cur_xid,
        next_xp_desc_slot,
        bump_high_water,
        next_oid,
        num_files,
        num_directories,
        num_symlinks,
    ) = match &fs.state {
        ApfsState::Write(w) => (
            fs.block_size,
            w.total_blocks,
            w.volume_name.clone(),
            w.container_uuid,
            w.volume_uuid,
            w.cur_xid,
            w.next_xp_desc_slot,
            w.bump_high_water,
            w.next_oid,
            w.num_files,
            w.num_directories,
            w.num_symlinks,
        ),
        _ => {
            return Err(crate::Error::Unsupported(
                "apfs: commit called outside write state".into(),
            ));
        }
    };

    let mut records = dump_all_records(fs, dev)?;
    // Build a context the mutator can both read records out of and
    // append fresh extents to. `bump_high_water` / counters are
    // copied out after the mutator returns and threaded through to
    // the writer so the new APSB reflects the closure's work.
    let mut cx = MutatorCx {
        records: &mut records,
        dev,
        block_size,
        total_blocks,
        bump_high_water,
        next_oid,
        num_files,
        num_directories,
        num_symlinks,
    };
    mutator(&mut cx)?;
    let new_bump_high_water = cx.bump_high_water;
    let new_next_oid = cx.next_oid;
    let new_num_files = cx.num_files;
    let new_num_directories = cx.num_directories;
    let new_num_symlinks = cx.num_symlinks;
    // `cx`'s last use is the line above; NLL releases its borrows of
    // `records` and `dev` here, so the writer block below can
    // reborrow `dev`. An explicit drop(cx) would be a no-op (no Drop
    // impl on MutatorCx) — clippy flags it under drop_non_drop.

    let new_xid = cur_xid + 1;
    {
        let mut w = write::ApfsWriter::new_checkpoint(
            dev,
            total_blocks,
            block_size,
            &volume_name,
            container_uuid,
            volume_uuid,
            new_xid,
            next_xp_desc_slot,
            new_bump_high_water,
            new_next_oid,
            new_num_files,
            new_num_directories,
            new_num_symlinks,
        )?;
        for (k, v) in records {
            w.push_raw_record(k, v);
        }
        w.finish()?;
    }

    let refreshed = Apfs::open_writable(dev)?;
    debug_assert!(matches!(refreshed.state, ApfsState::Write(_)));
    fs.state = refreshed.state;
    fs.volume_name = refreshed.volume_name;
    Ok(())
}

/// Walk every record in the fs-tree and return them as `(key, val)`
/// byte pairs. Sorted by `(oid, kind, tail)` per APFS canonical order
/// because `RangeScan` yields entries in that order anyway, but we
/// re-sort defensively.
fn dump_all_records(fs: &mut Apfs, dev: &mut dyn BlockDevice) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
    // Use the read state's caches.
    let rs = fs.read_state()?;
    let block_size = fs.block_size;
    let mut ctx = rs.fs_ctx.borrow_mut();
    let mut out = Vec::new();
    // We scan once per (oid, kind) pair we expect, but that's expensive.
    // The simpler tactic: walk every distinct oid we see by starting
    // each scan at (0, INODE) and following the natural tree order.
    // But RangeScan stops at the prefix boundary, so we'd need to
    // restart per (oid, kind).
    //
    // Instead, walk by repeated range-scans across each known record
    // kind. To know which oids exist, first walk every drec record
    // under root to enumerate dirs+files, then recurse into subdirs.
    // That's complex. The fast workaround: walk the tree via the leaf
    // iterator directly by re-using RangeScan with a sentinel start
    // (oid=0, kind=0) that matches the empty prefix.
    //
    // RangeScan's `stop_oid/stop_kind` are derived from the start
    // target, so scanning with oid=0,kind=0 stops immediately on the
    // first non-matching key. To dump the *whole* tree we instead walk
    // each leaf physically. The fs-tree has at most one internal level
    // — so the omap tells us where the leaves live.

    // Pull every fs-tree leaf paddr out of the omap.
    let leaf_paddrs = collect_fs_leaf_paddrs(&rs.fsroot_block, &mut ctx, &mut |paddr, buf| {
        read_at_paddr(dev, paddr, block_size, buf)
    })?;
    drop(ctx);

    if leaf_paddrs.is_empty() {
        // Single root-leaf tree: walk records directly out of the
        // cached fsroot_block.
        let node = super::btree::BTreeNode::decode(&rs.fsroot_block)?;
        if node.is_leaf() {
            for i in 0..node.nkeys {
                let (kb, vb) = node.entry_at(i, 0, 0)?;
                out.push((kb.to_vec(), vb.to_vec()));
            }
        }
    } else {
        for paddr in leaf_paddrs {
            let mut blk = vec![0u8; block_size as usize];
            read_at_paddr(dev, paddr, block_size, &mut blk)?;
            let node = super::btree::BTreeNode::decode(&blk)?;
            if !node.is_leaf() {
                continue;
            }
            for i in 0..node.nkeys {
                let (kb, vb) = node.entry_at(i, 0, 0)?;
                out.push((kb.to_vec(), vb.to_vec()));
            }
        }
    }

    // Defensive sort by (oid, kind, tail).
    out.sort_by(|a, b| {
        let ka = sort_key_for(&a.0);
        let kb = sort_key_for(&b.0);
        ka.cmp(&kb)
    });
    Ok(out)
}

fn sort_key_for(key: &[u8]) -> (u64, u8, Vec<u8>) {
    if key.len() < 8 {
        return (0, 0, key.to_vec());
    }
    let hdr = u64::from_le_bytes(key[0..8].try_into().unwrap());
    let oid = hdr & OBJ_ID_MASK;
    let kind = (hdr >> OBJ_TYPE_SHIFT) as u8;
    let tail = key[8..].to_vec();
    (oid, kind, tail)
}

/// Walk the fs-tree from `fsroot_block` and return the physical block
/// addresses of every leaf node. Internal-node children are resolved
/// through the volume omap inside `ctx`.
fn collect_fs_leaf_paddrs<F>(
    fsroot_block: &[u8],
    ctx: &mut FsTreeCtx,
    read_block: &mut F,
) -> Result<Vec<u64>>
where
    F: FnMut(u64, &mut [u8]) -> Result<()>,
{
    let root_node = super::btree::BTreeNode::decode(fsroot_block)?;
    if root_node.is_leaf() {
        // The root is itself a leaf — we don't know its own paddr,
        // but we can return an empty list and have the caller walk
        // the root's records directly. To keep the API uniform,
        // synthesize a single leaf by writing the root bytes to a
        // scratch buffer at "phantom paddr" 0 ... actually we just
        // expose the records here.
        //
        // The cleaner path: collect_fs_leaf_paddrs returns physical
        // paddrs of leaves; if the tree is a single root-leaf we
        // return an empty Vec and have the caller fall back to
        // reading the root directly. Marker: 0 paddr.
        return Ok(vec![]);
    }
    // Internal root: each child is a virtual oid (kvloc, 8-byte child
    // vid value). Resolve through the omap.
    let mut out = Vec::with_capacity(root_node.nkeys as usize);
    for i in 0..root_node.nkeys {
        let (_, vb) = root_node.entry_at(i, 0, 8)?;
        if vb.len() < 8 {
            continue;
        }
        let child_vid = u64::from_le_bytes(vb[0..8].try_into().unwrap());
        let paddr = ctx.resolve_vid(child_vid, read_block)?;
        let mut child_blk = vec![0u8; ctx.block_size];
        read_block(paddr, &mut child_blk)?;
        let child_node = super::btree::BTreeNode::decode(&child_blk)?;
        if child_node.is_leaf() {
            out.push(paddr);
        } else {
            // We don't support deeper trees here — fall back to
            // surfacing what we found.
            return Err(crate::Error::Unsupported(
                "apfs: fs-tree depth > 2 isn't supported by the rw path".into(),
            ));
        }
    }
    Ok(out)
}

/// Patch an inode value's DSTREAM xfield to advertise `new_size` and
/// the allocated size matching `block_size`-aligned-up.
fn patch_inode_dstream_size(val: &mut [u8], new_size: u64, block_size: u64) {
    if val.len() < J_INODE_VAL_FIXED_SIZE {
        return;
    }
    // The fixed inode value carries `total_size` at offset 84..92.
    val[84..92].copy_from_slice(&new_size.to_le_bytes());
    // Find the trailing xfield blob (offset 92..) and update the DSTREAM
    // value's size + alloced_size fields.
    if val.len() <= J_INODE_VAL_FIXED_SIZE {
        return;
    }
    let xfields = &mut val[J_INODE_VAL_FIXED_SIZE..];
    if xfields.len() < 4 {
        return;
    }
    let num_exts = u16::from_le_bytes(xfields[0..2].try_into().unwrap()) as usize;
    // Iterate x_field_t headers; the value blob follows after all
    // headers.
    let headers_end = 4 + num_exts * 4;
    if xfields.len() < headers_end {
        return;
    }
    let mut value_cursor = headers_end;
    for i in 0..num_exts {
        let hdr_off = 4 + i * 4;
        let kind = xfields[hdr_off];
        let size =
            u16::from_le_bytes(xfields[hdr_off + 2..hdr_off + 4].try_into().unwrap()) as usize;
        if kind == INO_EXT_TYPE_DSTREAM && size >= 16 {
            // j_dstream_t: u64 size, u64 alloced_size, ...
            let alloc = new_size.div_ceil(block_size) * block_size;
            xfields[value_cursor..value_cursor + 8].copy_from_slice(&new_size.to_le_bytes());
            xfields[value_cursor + 8..value_cursor + 16].copy_from_slice(&alloc.to_le_bytes());
            return;
        }
        value_cursor += size.next_multiple_of(8);
        if value_cursor > xfields.len() {
            return;
        }
    }
}

// Tests for the rw module live alongside the rest of the apfs
// integration tests in src/fs/apfs/mod.rs (round-trip format → mutate →
// reopen → read).
