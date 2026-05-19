//! APFS image writer — best-effort, single-volume, no checkpoint replay.
//!
//! ## Scope
//!
//! This produces a minimal but *readable* APFS image: an NXSB at block 0
//! plus an additional NXSB inside the checkpoint descriptor area, a
//! container omap with a single leaf node, a stub spaceman, one volume
//! with its own omap + fs-tree leaf, and any number of regular files,
//! directories, and symlinks rooted at inode 2.
//!
//! Layout (block numbers given for `block_size = 4096`):
//!
//! ```text
//!   0      NXSB           container label
//!   1      NXSB           live checkpoint
//!   2      checkpoint_map (zero entries; just a placeholder)
//!   3      omap_phys_t    container omap header
//!   4      btree leaf     container omap root (maps vid -> APSB paddr)
//!   5      spaceman_phys  stub (we don't track allocation in v1)
//!   6      APSB           volume superblock
//!   7      omap_phys_t    volume omap header
//!   8      btree leaf     volume omap root (maps vid -> fsroot paddr)
//!   9      btree leaf     fsroot — fs-tree with inode/drec/extent records
//!  10..    data blocks    file extents (one extent per regular file)
//! ```
//!
//! The spaceman entry is intentionally a stub: we don't maintain the
//! container's free-space bitmaps. This is fine for read-only consumers
//! (our reader doesn't touch the spaceman) but means mounting this
//! image on macOS would refuse to write to it.
//!
//! ## Streaming invariant
//!
//! File bytes are copied through a 64 KiB scratch buffer; the writer
//! never loads a whole file into memory. The fs-tree leaf and the few
//! metadata blocks are bounded by the size of one container block
//! (typically 4 KiB) — the v1 limit is therefore "everything fits in
//! one fs-tree leaf node", which in practice is around ~50 small
//! files/dirs at 4 KiB blocks.
//!
//! ## Public API
//!
//! ```ignore
//! let mut w = ApfsWriter::new(&mut dev, total_blocks, 4096, "MyVolume")?;
//! let dir = w.add_dir(2, "subdir", 0o755)?;
//! w.add_file_from_reader(dir, "hello.txt", 0o644, &mut some_reader, size)?;
//! w.add_symlink(2, "link", 0o777, "target")?;
//! w.finish()?;
//! ```
//!
//! ## Limits
//!
//! - No checkpoint replay is performed (we don't pretend to be journal-
//!   recoverable).
//! - The fs-tree, both omaps, and the snap-meta tree must each fit in a
//!   single leaf node. Exceeding this returns
//!   [`crate::Error::Unsupported`] from [`ApfsWriter::finish`].
//! - No snapshots, no encryption, no xattrs, no clones, no
//!   compression.
//! - The image must be at least ~32 blocks long.

use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;

use super::btree::{BTNODE_FIXED_KV_SIZE, BTNODE_LEAF, BTNODE_ROOT, BTREE_INFO_SIZE};
use super::checksum::fletcher64;
use super::jrec::{
    APFS_TYPE_DIR_REC, APFS_TYPE_DSTREAM_ID, APFS_TYPE_FILE_EXTENT, APFS_TYPE_INODE, DT_DIR,
    DT_LNK, DT_REG, INO_EXT_TYPE_DSTREAM, J_INODE_VAL_FIXED_SIZE, OBJ_ID_MASK, OBJ_TYPE_SHIFT,
};
use super::obj::{
    OBJECT_TYPE_BTREE, OBJECT_TYPE_CHECKPOINT_MAP, OBJECT_TYPE_FS, OBJECT_TYPE_FSTREE,
    OBJECT_TYPE_NX_SUPERBLOCK, OBJECT_TYPE_OMAP,
};
use super::superblock::{APFS_MAGIC, NX_MAGIC, NX_MAX_FILE_SYSTEMS};

/// `OBJECT_TYPE_SPACEMAN` — we only emit this constant for the stub
/// block; not otherwise used.
const OBJECT_TYPE_SPACEMAN: u32 = 0x0000_0005;

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
const WRITE_XID: u64 = 2;

/// Inode-2 default-data-stream object id constant. The fs-tree pairs
/// every regular-file inode with a `private_id` pointing at the dstream
/// object id; we encode them with `private_id == inode_oid` (i.e. the
/// dstream and inode share an id), which is permitted by the spec and
/// is also what genuine APFS volumes do for normal files.
const DSTREAM_ID_SHARES_INODE: bool = true;

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
/// [`ApfsWriter::add_dir`], and [`ApfsWriter::add_symlink`], then call
/// [`ApfsWriter::finish`] to materialise the on-disk image.
///
/// The writer owns a `&mut BlockDevice`; the device must already be
/// sized large enough (zero-extended is fine — we'll overwrite the
/// header and any blocks we use).
pub struct ApfsWriter<'a> {
    dev: &'a mut dyn BlockDevice,
    block_size: u32,
    total_blocks: u64,
    /// Index of the next free data block (allocated bump-pointer
    /// fashion from `data_block_start`).
    next_data_block: u64,
    /// First block reserved for file-extent data (everything before
    /// this is metadata).
    data_block_start: u64,
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
        if total_blocks < 32 {
            return Err(crate::Error::InvalidArgument(
                "apfs writer: need at least 32 blocks".into(),
            ));
        }
        let container_uuid = derive_uuid(volume_name.as_bytes(), b"container");
        let volume_uuid = derive_uuid(volume_name.as_bytes(), b"volume");

        let mut w = Self {
            dev,
            block_size,
            total_blocks,
            next_data_block: 0,
            data_block_start: 10, // see module-level layout doc
            volume_name: volume_name.to_string(),
            container_uuid,
            volume_uuid,
            records: Vec::new(),
            next_oid: 16, // inode 2 is root; we hand out 16+ for new inodes
            num_files: 0,
            num_directories: 0,
            num_symlinks: 0,
            finished: false,
        };
        w.next_data_block = w.data_block_start;
        // Seed the root inode record (oid = 2).
        w.add_inode_record(2, 0, mode_dir(0o755), 0)?;
        Ok(w)
    }

    /// Add an empty subdirectory under `parent_oid`. Returns the new
    /// directory's object id (use it as `parent_oid` for nested
    /// children).
    pub fn add_dir(&mut self, parent_oid: u64, name: &str, mode: u16) -> Result<u64> {
        let oid = self.alloc_oid();
        self.add_drec(parent_oid, name, oid, DT_DIR)?;
        self.add_inode_record(oid, parent_oid, mode_dir(mode), 0)?;
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
        self.add_inode_record(oid, parent_oid, mode_lnk(mode), target_bytes.len() as u64)?;
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
        let oid = self.alloc_oid();
        self.add_drec(parent_oid, name, oid, DT_REG)?;

        if size == 0 {
            // No extent records — empty file.
            self.add_inode_record(oid, parent_oid, mode_reg(mode), 0)?;
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
        self.add_inode_record(oid, parent_oid, mode_reg(mode), size)?;
        self.num_files += 1;
        Ok(oid)
    }

    /// Materialise the on-disk image: serialize the fs-tree leaf, the
    /// volume omap, the APSB, the container omap, the spaceman stub,
    /// and finally the NXSB copies. After this call the writer is
    /// drained and further `add_*` calls return `Unsupported`.
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
        let block_size = self.block_size;

        // ---- Fixed block addresses (see module-level layout) ----
        let nxsb_label_paddr: u64 = 0;
        let nxsb_live_paddr: u64 = 1;
        let chkmap_paddr: u64 = 2;
        let cont_omap_paddr: u64 = 3;
        let cont_omap_root_paddr: u64 = 4;
        let spaceman_paddr: u64 = 5;
        let apsb_paddr: u64 = 6;
        let vol_omap_paddr: u64 = 7;
        let vol_omap_root_paddr: u64 = 8;
        let fsroot_paddr: u64 = 9;

        // Virtual oids assigned to: container volume (=1024), spaceman
        // (=512), volume omap target -> fsroot (=fsroot_vid). The
        // container omap maps volume_vid → APSB paddr; the volume omap
        // maps fsroot_vid → fsroot paddr.
        let volume_vid: u64 = 1024;
        let spaceman_vid: u64 = 512;
        let reaper_vid: u64 = 513;
        let _vol_omap_vid: u64 = 2048;
        let fsroot_vid: u64 = 2;

        // ---- fs-tree leaf ----
        let fsroot_block = build_fs_leaf(&self.records, bs, fsroot_vid)?;
        self.write_block(fsroot_paddr, &fsroot_block)?;

        // ---- Volume omap root (single leaf) ----
        let vol_omap_root = build_omap_leaf(bs, &[(fsroot_vid, WRITE_XID, fsroot_paddr)])?;
        self.write_block(vol_omap_root_paddr, &vol_omap_root)?;

        // ---- Volume omap header ----
        let vol_omap_phys = build_omap_phys(bs, vol_omap_paddr, vol_omap_root_paddr)?;
        self.write_block(vol_omap_paddr, &vol_omap_phys)?;

        // ---- APSB ----
        let apsb_block = self.build_apsb(bs, apsb_paddr, vol_omap_paddr, fsroot_vid)?;
        self.write_block(apsb_paddr, &apsb_block)?;

        // ---- Spaceman stub ----
        let spaceman_block = build_spaceman_stub(bs, spaceman_vid)?;
        self.write_block(spaceman_paddr, &spaceman_block)?;

        // ---- Container omap root + header ----
        let cont_omap_root = build_omap_leaf(bs, &[(volume_vid, WRITE_XID, apsb_paddr)])?;
        self.write_block(cont_omap_root_paddr, &cont_omap_root)?;
        let cont_omap_phys = build_omap_phys(bs, cont_omap_paddr, cont_omap_root_paddr)?;
        self.write_block(cont_omap_paddr, &cont_omap_phys)?;

        // ---- Checkpoint map (placeholder, 0 entries) ----
        let chkmap = build_chkmap_stub(bs)?;
        self.write_block(chkmap_paddr, &chkmap)?;

        // ---- NXSB (live + label copy) ----
        let nxsb = self.build_nxsb(
            bs,
            nxsb_live_paddr,
            cont_omap_paddr,
            spaceman_vid,
            reaper_vid,
            volume_vid,
        )?;
        self.write_block(nxsb_live_paddr, &nxsb)?;
        // Label copy at block 0 — same content but oid=0 like normal
        // (the spec says label NXSB has oid 1, but in practice both
        // copies look identical). We just reuse the live copy here.
        let nxsb_label = self.build_nxsb(
            bs,
            nxsb_label_paddr,
            cont_omap_paddr,
            spaceman_vid,
            reaper_vid,
            volume_vid,
        )?;
        self.write_block(nxsb_label_paddr, &nxsb_label)?;

        let _ = block_size; // silence unused warning if ever
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

    fn allocate_extent_for_size(&mut self, size: u64) -> Result<u64> {
        if size == 0 {
            return Err(crate::Error::InvalidArgument(
                "apfs writer: cannot allocate a zero-length extent".into(),
            ));
        }
        let blocks = self.bytes_to_blocks(size);
        let start = self.next_data_block;
        let end = start
            .checked_add(blocks)
            .ok_or_else(|| crate::Error::InvalidArgument("apfs writer: extent oob".into()))?;
        if end > self.total_blocks {
            return Err(crate::Error::InvalidArgument(format!(
                "apfs writer: extent of {blocks} blocks past end of image"
            )));
        }
        self.next_data_block = end;
        Ok(start)
    }

    fn write_block(&mut self, paddr: u64, buf: &[u8]) -> Result<()> {
        let off = paddr.saturating_mul(self.block_size as u64);
        self.dev.write_at(off, buf)
    }

    fn add_drec(&mut self, parent_oid: u64, name: &str, target_oid: u64, dtype: u16) -> Result<()> {
        // Plain drec key (we ship images without
        // APFS_INCOMPAT_NORMALIZATION_INSENSITIVE so our reader picks
        // the plain layout).
        let nlen = name.len() + 1; // includes trailing NUL
        if nlen > u16::MAX as usize {
            return Err(crate::Error::InvalidArgument(
                "apfs writer: directory entry name too long".into(),
            ));
        }
        let mut key = Vec::with_capacity(10 + nlen);
        let hdr = ((APFS_TYPE_DIR_REC as u64) << OBJ_TYPE_SHIFT) | (parent_oid & OBJ_ID_MASK);
        key.extend_from_slice(&hdr.to_le_bytes());
        key.extend_from_slice(&(nlen as u16).to_le_bytes());
        key.extend_from_slice(name.as_bytes());
        key.push(0);

        // j_drec_val_t (18 bytes minimum)
        let mut val = vec![0u8; 18];
        val[0..8].copy_from_slice(&target_oid.to_le_bytes());
        // date_added at offset 8..16 — leave zero
        val[16..18].copy_from_slice(&dtype.to_le_bytes());

        self.records.push(FsRecord { key, val });
        Ok(())
    }

    fn add_inode_record(
        &mut self,
        oid: u64,
        parent_oid: u64,
        mode: u16,
        dstream_size: u64,
    ) -> Result<()> {
        // Build j_inode_val_t with no trailing xfields for directories;
        // for regular files / symlinks, add a DSTREAM xfield carrying
        // the file size.
        let has_dstream =
            dstream_size > 0 || (mode & 0o170_000 == 0o100_000) || (mode & 0o170_000 == 0o120_000);
        let mut val = vec![0u8; J_INODE_VAL_FIXED_SIZE];
        val[0..8].copy_from_slice(&parent_oid.to_le_bytes());
        // private_id = oid for files/symlinks (shared dstream id)
        let private_id = if has_dstream && DSTREAM_ID_SHARES_INODE {
            oid
        } else {
            0
        };
        val[8..16].copy_from_slice(&private_id.to_le_bytes());
        // Times: leave zero.
        // internal_flags: 0.
        // nchildren_or_nlink: 1 (for files) or 2 (for dirs at minimum)
        let nlink: i32 = if mode & 0o170_000 == 0o040_000 { 2 } else { 1 };
        val[56..60].copy_from_slice(&nlink.to_le_bytes());
        // owner/group: 0 (root).
        val[80..82].copy_from_slice(&mode.to_le_bytes());
        val[84..92].copy_from_slice(&dstream_size.to_le_bytes());

        if has_dstream {
            // xfield blob: 1 entry (DSTREAM), 40 bytes value, padded to 8.
            let mut xfields = Vec::new();
            xfields.extend_from_slice(&1u16.to_le_bytes()); // num_exts
            // used_data covers value bytes (40)
            xfields.extend_from_slice(&40u16.to_le_bytes());
            // x_field_t = (type:u8, flags:u8, size:u16)
            xfields.push(INO_EXT_TYPE_DSTREAM);
            xfields.push(0);
            xfields.extend_from_slice(&40u16.to_le_bytes());
            // value: j_dstream_t (40 bytes)
            let mut ds = [0u8; 40];
            ds[0..8].copy_from_slice(&dstream_size.to_le_bytes());
            // alloced_size — round up to block-size multiple
            let bs = self.block_size as u64;
            let alloc = dstream_size.div_ceil(bs) * bs;
            ds[8..16].copy_from_slice(&alloc.to_le_bytes());
            xfields.extend_from_slice(&ds);
            val.extend_from_slice(&xfields);
        }

        let mut key = vec![0u8; 8];
        let hdr = ((APFS_TYPE_INODE as u64) << OBJ_TYPE_SHIFT) | (oid & OBJ_ID_MASK);
        key.copy_from_slice(&hdr.to_le_bytes());

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
        let mut key = vec![0u8; 16];
        let hdr = ((APFS_TYPE_FILE_EXTENT as u64) << OBJ_TYPE_SHIFT) | (dstream_oid & OBJ_ID_MASK);
        key[0..8].copy_from_slice(&hdr.to_le_bytes());
        key[8..16].copy_from_slice(&logical_addr.to_le_bytes());

        // j_file_extent_val_t (24 bytes)
        let mut val = vec![0u8; 24];
        let bs = self.block_size as u64;
        let alloc = length.div_ceil(bs) * bs;
        val[0..8].copy_from_slice(&alloc.to_le_bytes()); // length (low 56 bits)
        val[8..16].copy_from_slice(&phys_block.to_le_bytes());
        // crypto_id stays zero.
        self.records.push(FsRecord { key, val });

        // Also a DSTREAM_ID record (just a counter to one) so the
        // reader can locate the dstream's extents under this oid.
        let mut dkey = vec![0u8; 8];
        let dhdr = ((APFS_TYPE_DSTREAM_ID as u64) << OBJ_TYPE_SHIFT) | (dstream_oid & OBJ_ID_MASK);
        dkey.copy_from_slice(&dhdr.to_le_bytes());
        let dval = vec![0u8; 4]; // j_dstream_id_val_t = refcnt: u32
        self.records.push(FsRecord {
            key: dkey,
            val: dval,
        });
        Ok(())
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
        buf[16..24].copy_from_slice(&WRITE_XID.to_le_bytes());
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
        buf[16..24].copy_from_slice(&WRITE_XID.to_le_bytes());
        buf[24..28].copy_from_slice(&(OBJECT_TYPE_NX_SUPERBLOCK | OBJ_EPHEMERAL).to_le_bytes());

        buf[32..36].copy_from_slice(&NX_MAGIC.to_le_bytes());
        buf[36..40].copy_from_slice(&self.block_size.to_le_bytes());
        buf[40..48].copy_from_slice(&self.total_blocks.to_le_bytes());
        // features / ro_compat / incompat: 0 (we ship a vanilla container)
        buf[72..88].copy_from_slice(&self.container_uuid);
        buf[88..96].copy_from_slice(&(self.next_oid + 1024).to_le_bytes()); // next_oid
        buf[96..104].copy_from_slice(&(WRITE_XID + 1).to_le_bytes()); // next_xid
        // xp_desc area: 2 blocks (chkmap stub + the live NXSB itself)
        buf[104..108].copy_from_slice(&2u32.to_le_bytes()); // xp_desc_blocks
        buf[108..112].copy_from_slice(&1u32.to_le_bytes()); // xp_data_blocks
        buf[112..120].copy_from_slice(&1u64.to_le_bytes()); // xp_desc_base
        buf[120..128].copy_from_slice(&3u64.to_le_bytes()); // xp_data_base (unused)
        buf[128..132].copy_from_slice(&2u32.to_le_bytes()); // xp_desc_next
        buf[132..136].copy_from_slice(&1u32.to_le_bytes()); // xp_data_next
        buf[136..140].copy_from_slice(&0u32.to_le_bytes()); // xp_desc_index
        buf[140..144].copy_from_slice(&2u32.to_le_bytes()); // xp_desc_len
        buf[144..148].copy_from_slice(&0u32.to_le_bytes()); // xp_data_index
        buf[148..152].copy_from_slice(&0u32.to_le_bytes()); // xp_data_len
        buf[152..160].copy_from_slice(&spaceman_vid.to_le_bytes());
        buf[160..168].copy_from_slice(&cont_omap_paddr.to_le_bytes()); // omap_oid
        buf[168..176].copy_from_slice(&reaper_vid.to_le_bytes()); // reaper_oid
        buf[176..180].copy_from_slice(&0u32.to_le_bytes()); // test_type
        buf[180..184].copy_from_slice(&(NX_MAX_FILE_SYSTEMS as u32).to_le_bytes());
        // fs_oid[0] = volume_vid
        buf[184..192].copy_from_slice(&volume_vid.to_le_bytes());
        // rest of fs_oid[] = 0

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
/// root leaf.
fn build_omap_phys(bs: usize, paddr: u64, tree_paddr: u64) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; bs];
    buf[8..16].copy_from_slice(&paddr.to_le_bytes()); // oid
    buf[16..24].copy_from_slice(&WRITE_XID.to_le_bytes()); // xid
    buf[24..28].copy_from_slice(&(OBJECT_TYPE_OMAP | OBJ_PHYSICAL).to_le_bytes());
    // flags / snap_count / tree_type / snapshot_tree_type left zero
    // (snapshot_tree_type is u32 at offset 44).
    buf[40..44].copy_from_slice(&(OBJECT_TYPE_BTREE | OBJ_PHYSICAL).to_le_bytes()); // tree_type
    buf[48..56].copy_from_slice(&tree_paddr.to_le_bytes());
    // snapshot_tree_oid stays 0.
    sign_block(&mut buf);
    Ok(buf)
}

/// Build an omap leaf node holding the given `(vid, xid, paddr)` triples
/// in ascending `(vid, xid)` order. Fixed-KV layout (16/16).
fn build_omap_leaf(bs: usize, entries: &[(u64, u64, u64)]) -> Result<Vec<u8>> {
    let mut block = vec![0u8; bs];
    // obj_phys
    block[24..28].copy_from_slice(&(OBJECT_TYPE_BTREE | OBJ_PHYSICAL).to_le_bytes());
    block[16..24].copy_from_slice(&WRITE_XID.to_le_bytes());

    // Header
    let flags = BTNODE_ROOT | BTNODE_LEAF | BTNODE_FIXED_KV_SIZE;
    block[32..34].copy_from_slice(&flags.to_le_bytes());
    block[34..36].copy_from_slice(&0u16.to_le_bytes()); // level
    block[36..40].copy_from_slice(&(entries.len() as u32).to_le_bytes());
    let toc_len = entries.len() * 4;
    block[40..42].copy_from_slice(&0u16.to_le_bytes());
    block[42..44].copy_from_slice(&(toc_len as u16).to_le_bytes());

    let toc_base = 56;
    let keys_start = toc_base + toc_len;
    let vals_end = bs - BTREE_INFO_SIZE;
    if entries.len() * 32 + toc_len > bs - BTREE_INFO_SIZE - toc_base {
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

    // Trailing btree_info_t: bt_key_size=16, bt_val_size=16
    let info_off = bs - BTREE_INFO_SIZE;
    block[info_off + 8..info_off + 12].copy_from_slice(&16u32.to_le_bytes());
    block[info_off + 12..info_off + 16].copy_from_slice(&16u32.to_le_bytes());

    sign_block(&mut block);
    Ok(block)
}

/// Build a variable-KV fs-tree leaf node holding `records` (already
/// sorted). Fits in a single block; we return `Unsupported` on
/// overflow so callers see a clean error.
fn build_fs_leaf(records: &[FsRecord], bs: usize, fsroot_vid: u64) -> Result<Vec<u8>> {
    let mut block = vec![0u8; bs];
    // obj_phys — real APFS fsroots are BTREE objects whose subtype is
    // FSTREE (the BTREE constant identifies the on-disk B-tree object;
    // FSTREE goes in the subtype slot to identify the tree's contents).
    block[8..16].copy_from_slice(&fsroot_vid.to_le_bytes());
    block[16..24].copy_from_slice(&WRITE_XID.to_le_bytes());
    block[24..28].copy_from_slice(&OBJECT_TYPE_BTREE.to_le_bytes());
    block[28..32].copy_from_slice(&OBJECT_TYPE_FSTREE.to_le_bytes());

    let flags = BTNODE_ROOT | BTNODE_LEAF;
    block[32..34].copy_from_slice(&flags.to_le_bytes());
    block[34..36].copy_from_slice(&0u16.to_le_bytes()); // level
    block[36..40].copy_from_slice(&(records.len() as u32).to_le_bytes());

    let toc_len = records.len() * 8;
    block[40..42].copy_from_slice(&0u16.to_le_bytes());
    block[42..44].copy_from_slice(&(toc_len as u16).to_le_bytes());

    let toc_base = 56;
    let keys_start = toc_base + toc_len;
    let vals_end = bs - BTREE_INFO_SIZE;

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

    // Trailing btree_info_t (we leave fixed sizes 0 since this is a
    // variable-KV tree; bt_key_count carries the record count).
    let info_off = bs - BTREE_INFO_SIZE;
    block[info_off + 24..info_off + 32].copy_from_slice(&(records.len() as u64).to_le_bytes());

    sign_block(&mut block);
    Ok(block)
}

/// Build a spaceman stub block — minimal `spaceman_phys_t` with the
/// fields our reader doesn't consume. APFS readers that don't enforce
/// allocation policy ignore the contents.
fn build_spaceman_stub(bs: usize, oid: u64) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; bs];
    buf[8..16].copy_from_slice(&oid.to_le_bytes());
    buf[16..24].copy_from_slice(&WRITE_XID.to_le_bytes());
    buf[24..28].copy_from_slice(&(OBJECT_TYPE_SPACEMAN | OBJ_EPHEMERAL).to_le_bytes());
    sign_block(&mut buf);
    Ok(buf)
}

/// Build a stub checkpoint_map_phys_t with zero entries. Only present so
/// the xp_desc area has a well-formed companion block for the live
/// NXSB.
fn build_chkmap_stub(bs: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; bs];
    buf[24..28].copy_from_slice(&(OBJECT_TYPE_CHECKPOINT_MAP | OBJ_PHYSICAL).to_le_bytes());
    buf[16..24].copy_from_slice(&WRITE_XID.to_le_bytes());
    // flags (u32) at offset 32, count (u32) at 36 — both zero is fine.
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

fn mode_reg(perm: u16) -> u16 {
    0o100_000 | (perm & 0o7777)
}
fn mode_dir(perm: u16) -> u16 {
    0o040_000 | (perm & 0o7777)
}
fn mode_lnk(perm: u16) -> u16 {
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
}
