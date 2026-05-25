//! XFS — read-only support.
//!
//! Implements enough of XFS v4/v5 to inspect the volume and stream files
//! out of it: the superblock, inodes with the v3 (CRC) and v2 cores,
//! shortform / block / leaf / node directories, single-level B-tree
//! (`BTREE` di_format) directories (deeper trees surface
//! `Error::Unsupported`), the linear extent list and B-tree extent maps
//! (`BTREE` di_format) for regular files, and both inline and remote
//! symlinks. Realtime files, encryption, reverse-mapping btrees, and the
//! journal are out of scope — encountering any of them returns
//! `Error::Unsupported` with a message naming the feature.
//!
//! ## Inode-number arithmetic
//!
//! XFS inode numbers are 64-bit and pack three fields: the AG index, the
//! AG-relative block number, and the slot inside that block.
//!
//! ```text
//!   ag    = ino >> (sb_agblklog + sb_inopblog)
//!   blk   = (ino >> sb_inopblog) & ((1 << sb_agblklog) - 1)
//!   slot  = ino & ((1 << sb_inopblog) - 1)
//! ```
//!
//! The on-disk byte address of an inode is then:
//!
//! ```text
//!   ag*sb_agblocks*sb_blocksize + blk*sb_blocksize + slot*sb_inodesize
//! ```
//!
//! ## Reference
//!
//! - Linux kernel documentation: <https://docs.kernel.org/admin-guide/filesystems/xfs.html>
//! - XFS Filesystem Structure (the "PDF"): <https://mirrors.edge.kernel.org/pub/linux/utils/fs/xfs/docs/xfs_filesystem_structure.pdf>
//!
//! This implementation is written from those documents only — no kernel
//! or xfsprogs source was consulted.

pub mod bmbt;
pub mod dir;
pub mod dir_build;
pub mod format;
pub mod inode;
pub mod journal;
pub mod rw;
pub mod superblock;
pub mod symlink;
pub mod write;
pub mod xattr;
pub mod xattr_leaf;

pub use format::{FormatOpts, format};
pub use write::{DeviceKind, EntryMeta, WriteState};

use crate::Result;
use crate::block::BlockDevice;

use self::bmbt::{BmbtLayout, Extent};
use self::dir::DataEntry;
use self::inode::{DiFormat, DinodeCore};
use self::superblock::Superblock;

/// In-memory state of an opened XFS volume. Owns the parsed superblock; all
/// other reads go through the passed-in `BlockDevice`. When the volume
/// was just formatted via [`format()`] (or [`begin_writes`](Xfs::begin_writes)
/// was called after open), the `write_state` field tracks the
/// bump-pointer allocator + INOBT chunks for incremental adds.
pub struct Xfs {
    pub(crate) sb: Superblock,
    pub(crate) write_state: Option<WriteState>,
}

impl Xfs {
    /// Open an XFS volume by reading + validating the superblock at the
    /// start of AG0. Does not walk any other metadata yet.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        if dev.total_size() < 512 {
            return Err(crate::Error::InvalidImage(
                "xfs: device too small to hold a superblock".into(),
            ));
        }
        // The superblock occupies the first sector; we read 512 bytes which
        // covers every field we touch in `Superblock::decode`.
        let mut buf = [0u8; 512];
        dev.read_at(0, &mut buf)?;
        let sb = Superblock::decode(&buf)?;
        // Refuse exotic configurations that would silently break reads.
        if sb.rblocks != 0 {
            return Err(crate::Error::Unsupported(
                "xfs: realtime subvolume not supported".into(),
            ));
        }
        Ok(Self {
            sb,
            write_state: None,
        })
    }

    /// Total bytes the volume claims: `sb_dblocks * sb_blocksize`.
    pub fn total_bytes(&self) -> u64 {
        self.sb.total_bytes()
    }

    /// FS block size in bytes.
    pub fn block_size(&self) -> u32 {
        self.sb.blocksize
    }

    /// Bytes per on-disk inode.
    pub fn inode_size(&self) -> u32 {
        self.sb.inodesize as u32
    }

    /// Number of allocation groups.
    pub fn ag_count(&self) -> u32 {
        self.sb.agcount
    }

    /// Borrow the superblock for diagnostics.
    pub fn superblock(&self) -> &Superblock {
        &self.sb
    }

    // -- inode addressing ------------------------------------------------

    /// Decompose an inode number into (ag, ag-relative block, slot).
    fn split_ino(&self, ino: u64) -> (u64, u64, u64) {
        let inopblog = self.sb.inopblog as u32;
        let agblklog = self.sb.agblklog as u32;
        let slot_mask = (1u64 << inopblog) - 1;
        let blk_mask = (1u64 << agblklog) - 1;
        let slot = ino & slot_mask;
        let blk = (ino >> inopblog) & blk_mask;
        let ag = ino >> (inopblog + agblklog);
        (ag, blk, slot)
    }

    /// Absolute byte offset on the device for inode `ino`.
    fn ino_byte_offset(&self, ino: u64) -> Result<u64> {
        let (ag, blk, slot) = self.split_ino(ino);
        if ag >= self.sb.agcount as u64 {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: inode {ino} references ag {ag} but agcount = {}",
                self.sb.agcount
            )));
        }
        let ag_bytes = ag * (self.sb.agblocks as u64) * (self.sb.blocksize as u64);
        let blk_bytes = blk * (self.sb.blocksize as u64);
        let slot_bytes = slot * (self.sb.inodesize as u64);
        Ok(ag_bytes + blk_bytes + slot_bytes)
    }

    /// Read an inode by number, returning the raw bytes plus the decoded core.
    fn read_inode(&self, dev: &mut dyn BlockDevice, ino: u64) -> Result<(Vec<u8>, DinodeCore)> {
        let off = self.ino_byte_offset(ino)?;
        let mut buf = vec![0u8; self.sb.inodesize as usize];
        dev.read_at(off, &mut buf)?;
        let core = DinodeCore::decode(&buf)?;
        // For v3 inodes the di_ino field should match the address we used.
        if let Some(self_ino) = core.di_ino
            && self_ino != ino
        {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: inode {ino}: di_ino self-reference is {self_ino}"
            )));
        }
        Ok((buf, core))
    }

    // -- directory walk --------------------------------------------------

    /// Resolve an absolute path to an inode number + the decoded inode.
    fn resolve_path(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<(u64, Vec<u8>, DinodeCore)> {
        let mut cur_ino = self.sb.rootino;
        let (buf, core) = self.read_inode(dev, cur_ino)?;
        // The root inode must be a directory; otherwise the volume is broken.
        if !core.is_dir() {
            return Err(crate::Error::InvalidImage(
                "xfs: root inode is not a directory".into(),
            ));
        }
        let mut cur_buf = buf;
        let mut cur_core = core;
        for part in split_path(path) {
            let dir_entries = self.read_dir_entries(dev, &cur_buf, &cur_core)?;
            // On-disk entry, or one still staged in the directory's batch
            // (created this session but not yet serialized).
            cur_ino = match dir_entries.iter().find(|e| e.name == part) {
                Some(found) => found.inumber,
                None => self.pending_child_ino(cur_ino, part).ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "xfs: no such entry {part:?} under {path:?}"
                    ))
                })?,
            };
            let (b, c) = self.read_inode(dev, cur_ino)?;
            cur_buf = b;
            cur_core = c;
        }
        Ok((cur_ino, cur_buf, cur_core))
    }

    /// Layout descriptor for B-tree / FSB walking.
    fn bmbt_layout(&self) -> BmbtLayout {
        BmbtLayout {
            blocksize: self.sb.blocksize,
            agblocks: self.sb.agblocks,
            agblklog: self.sb.agblklog,
            is_v5: self.sb.is_v5(),
        }
    }

    /// Convert an FSB (filesystem block number) to a device byte offset.
    fn fsb_to_byte(&self, fsb: u64) -> u64 {
        let ag = fsb >> self.sb.agblklog as u32;
        let agblk = fsb & ((1u64 << self.sb.agblklog as u32) - 1);
        ag * (self.sb.agblocks as u64) * (self.sb.blocksize as u64)
            + agblk * (self.sb.blocksize as u64)
    }

    /// Read the extent list for a file/dir inode. Returns the decoded
    /// extents regardless of whether the inode is `EXTENTS` (inline list)
    /// or `BTREE` (walked tree). Other formats are an error.
    fn read_extent_list(
        &self,
        dev: &mut dyn BlockDevice,
        ino_buf: &[u8],
        core: &DinodeCore,
    ) -> Result<Vec<Extent>> {
        let lit = core.literal_area(ino_buf, self.sb.inodesize as usize);
        match core.format {
            DiFormat::Extents => bmbt::decode_extents(lit, core.nextents),
            DiFormat::Btree => {
                let layout = self.bmbt_layout();
                bmbt::walk_btree(dev, &layout, lit)
            }
            other => Err(crate::Error::InvalidArgument(format!(
                "xfs: extent list requested for non-extent inode (format={other:?})"
            ))),
        }
    }

    /// Read every directory data/block referenced by a directory inode's
    /// extent list (block + leaf format directories), parse each one, and
    /// return the concatenated list of entries. Skips the leaf index block
    /// at logical offset >= `XFS_DIR2_LEAF_OFFSET / dirblksize`.
    fn read_extent_dir_entries(
        &self,
        dev: &mut dyn BlockDevice,
        ino_buf: &[u8],
        core: &DinodeCore,
    ) -> Result<Vec<DataEntry>> {
        let extents = self.read_extent_list(dev, ino_buf, core)?;
        self.decode_dir_entries_from_extents(dev, &extents)
    }

    /// Walk a `di_format == BTREE` directory inode: decode its bmbt
    /// root via [`bmbt::read_btree_dir_extents`] (any depth), then
    /// enumerate every data block the resulting extent list covers
    /// using the same decoder as the `block` / `leaf` paths.
    fn read_btree_dir_entries(
        &self,
        dev: &mut dyn BlockDevice,
        ino_buf: &[u8],
        core: &DinodeCore,
    ) -> Result<Vec<DataEntry>> {
        let lit = core.literal_area(ino_buf, self.sb.inodesize as usize);
        let layout = self.bmbt_layout();
        let extents = bmbt::read_btree_dir_extents(dev, &layout, lit)?;
        self.decode_dir_entries_from_extents(dev, &extents)
    }

    /// Shared back-end for [`read_extent_dir_entries`] and
    /// [`read_btree_dir_entries`]. Given an extent list (in `dir-FSB`
    /// units), decide between block-format (single dir block carrying
    /// data + a trailing leaf array) and data-block iteration (leaf /
    /// node / btree formats), then decode every data block.
    fn decode_dir_entries_from_extents(
        &self,
        dev: &mut dyn BlockDevice,
        extents: &[Extent],
    ) -> Result<Vec<DataEntry>> {
        // First, read logical block 0 to decide between block- and leaf-
        // format. Block format: a single dir block, contains both data
        // records and a trailing leaf-entry array.
        if extents.is_empty() {
            return Ok(Vec::new());
        }
        let dir_block_size = self.sb.dir_block_size() as u64;
        let fs_blocks_per_dir_block = (dir_block_size / self.sb.blocksize as u64).max(1);
        // Logical address of the leaf block in a leaf-format directory.
        // The kernel uses 32 GiB / dir_block_size; convert to FS-block units
        // since extents are stored in FS-block units.
        let leaf_dir_block_addr_fsblk = dir::XFS_DIR2_LEAF_FIRSTDB_BYTES / self.sb.blocksize as u64;

        // Read the very first block to peek at the magic.
        let first = self.read_dir_block_at_logical(dev, extents, 0, dir_block_size as usize)?;
        let is_v5 = self.sb.is_v5();
        if dir::is_block_format(&first)? {
            // Block format: a single directory block.
            return dir::decode_block_dir(&first, is_v5);
        }
        // Leaf or node format: walk every distinct directory-block-aligned
        // logical address that some extent covers (capped at the leaf-
        // offset boundary, which separates the data-block address range
        // from the leaf-index / free-index ranges).
        let mut out = Vec::new();
        // Cap is the LESSER of (last covered logical block + 1) and the
        // leaf-block boundary.
        let max_covered = extents
            .iter()
            .map(|e| e.offset + e.blockcount as u64)
            .max()
            .unwrap_or(0);
        let upper = max_covered.min(leaf_dir_block_addr_fsblk);
        let mut lblk = 0u64;
        while lblk < upper {
            // Is `lblk` covered by any extent?
            let covered = extents
                .iter()
                .any(|e| lblk >= e.offset && lblk < e.offset + e.blockcount as u64);
            if covered {
                let block =
                    self.read_dir_block_at_logical(dev, extents, lblk, dir_block_size as usize)?;
                if block.len() >= 4 {
                    let magic = u32::from_be_bytes(block[0..4].try_into().unwrap());
                    if magic == dir::XFS_DIR3_DATA_MAGIC || magic == dir::XFS_DIR2_DATA_MAGIC {
                        let mut entries = dir::decode_data_block(&block, is_v5)?;
                        out.append(&mut entries);
                    }
                    // Other magics (block / leaf / node / free) are either
                    // skipped (leaf/node/free are pure index) or impossible
                    // here (block-format was handled above).
                }
            }
            lblk += fs_blocks_per_dir_block;
        }
        Ok(out)
    }

    /// Read `dir_block_size` bytes from the directory at logical FS-block
    /// `lblk`, by mapping through the extent list. Returns the assembled
    /// buffer.
    fn read_dir_block_at_logical(
        &self,
        dev: &mut dyn BlockDevice,
        extents: &[Extent],
        lblk: u64,
        dir_block_size: usize,
    ) -> Result<Vec<u8>> {
        let bs = self.sb.blocksize as u64;
        let fs_blocks_per_dir_block = (dir_block_size as u64 / bs).max(1);
        let mut out = vec![0u8; dir_block_size];
        for i in 0..fs_blocks_per_dir_block {
            let target = lblk + i;
            let ext = extents
                .iter()
                .find(|e| target >= e.offset && target < e.offset + e.blockcount as u64)
                .ok_or_else(|| {
                    crate::Error::InvalidImage(format!(
                        "xfs: dir logical block {target} not covered by extent list"
                    ))
                })?;
            let phys_fsb = ext.startblock + (target - ext.offset);
            let phys_byte = self.fsb_to_byte(phys_fsb);
            let off = (i * bs) as usize;
            dev.read_at(phys_byte, &mut out[off..off + bs as usize])?;
        }
        Ok(out)
    }

    /// Read a directory's entries given the inode's bytes + core. Handles
    /// shortform (local), block, leaf, and node format directories.
    /// Returns `Error::Unsupported` for the `BTREE` di_format.
    fn read_dir_entries(
        &self,
        dev: &mut dyn BlockDevice,
        ino_buf: &[u8],
        core: &DinodeCore,
    ) -> Result<Vec<DataEntry>> {
        if !core.is_dir() {
            return Err(crate::Error::InvalidArgument(
                "xfs: target is not a directory".into(),
            ));
        }
        match core.format {
            DiFormat::Local => {
                let lit = core.literal_area(ino_buf, self.sb.inodesize as usize);
                let has_ftype = core.version >= 3;
                let (_parent, entries) = dir::decode_shortform(lit, has_ftype)?;
                // Promote ShortformEntry → DataEntry (identical fields).
                Ok(entries
                    .into_iter()
                    .map(|e| DataEntry {
                        name: e.name,
                        inumber: e.inumber,
                        ftype: e.ftype,
                    })
                    .collect())
            }
            DiFormat::Extents => self.read_extent_dir_entries(dev, ino_buf, core),
            // BTREE-format directories — the inode-fork holds a bmbt root
            // (same shape as for files); leaves still contain XDD3/XDB3
            // data blocks plus a LEAFN / NODE index above the
            // XFS_DIR2_LEAF_OFFSET boundary. The reader walks the bmbt
            // through arbitrary depth (root → internal → leaves) and
            // returns the union of data-fork extents.
            DiFormat::Btree => self.read_btree_dir_entries(dev, ino_buf, core),
            DiFormat::Dev => Err(crate::Error::InvalidImage(
                "xfs: directory inode has di_format=dev".into(),
            )),
            DiFormat::Unknown(b) => Err(crate::Error::Unsupported(format!(
                "xfs: directory inode has unknown di_format {b}"
            ))),
        }
    }

    /// List the entries of a directory by absolute path. `/` resolves to
    /// the root directory.
    pub fn list_path(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let (_ino, buf, core) = self.resolve_path(dev, path)?;
        let entries = self.read_dir_entries(dev, &buf, &core)?;
        Ok(dir::data_entries_to_generic(&entries))
    }

    /// Read the target of a symlink at an absolute path. Supports both
    /// inline (local) and remote (extent-list) symlink targets.
    pub fn read_symlink(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<String> {
        let (_ino, buf, core) = self.resolve_path(dev, path)?;
        if !core.is_symlink() {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: {path:?} is not a symlink"
            )));
        }
        match core.format {
            DiFormat::Local => {
                let lit = core.literal_area(&buf, self.sb.inodesize as usize);
                symlink::decode_local(lit, core.size)
            }
            DiFormat::Extents => {
                let extents = self.read_extent_list(dev, &buf, &core)?;
                let layout = self.bmbt_layout();
                symlink::decode_remote(dev, &layout, &extents, core.size)
            }
            DiFormat::Btree => Err(crate::Error::Unsupported(
                "xfs: B-tree (di_format=BTREE) symlinks not implemented".into(),
            )),
            DiFormat::Dev => Err(crate::Error::InvalidArgument(
                "xfs: symlink inode with di_format=dev".into(),
            )),
            DiFormat::Unknown(b) => Err(crate::Error::Unsupported(format!(
                "xfs: symlink inode with unknown di_format {b}"
            ))),
        }
    }

    /// Open a regular file by absolute path for streaming reads.
    pub fn open_file_reader<'a>(
        &self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<XfsFileReader<'a>> {
        let (_ino, buf, core) = self.resolve_path(dev, path)?;
        if !core.is_reg() {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: {path:?} is not a regular file"
            )));
        }
        match core.format {
            DiFormat::Extents | DiFormat::Btree => {
                let extents = self.read_extent_list(dev, &buf, &core)?;
                for e in &extents {
                    if e.unwritten {
                        return Err(crate::Error::Unsupported(
                            "xfs: unwritten extents not supported in read path".into(),
                        ));
                    }
                }
                let layout = self.bmbt_layout();
                Ok(XfsFileReader {
                    dev,
                    extents,
                    blocksize: self.sb.blocksize as u64,
                    agblocks: self.sb.agblocks as u64,
                    agblklog: self.sb.agblklog,
                    size: core.size,
                    pos: 0,
                    inline: None,
                    _layout: layout,
                })
            }
            DiFormat::Local => {
                // Tiny files stored entirely in the inode literal area.
                // The file's bytes are the first `size` bytes of the literal.
                let lit = core.literal_area(&buf, self.sb.inodesize as usize);
                if (core.size as usize) > lit.len() {
                    return Err(crate::Error::InvalidImage(format!(
                        "xfs: local-format file claims size {} > literal area {}",
                        core.size,
                        lit.len()
                    )));
                }
                let data = lit[..core.size as usize].to_vec();
                Ok(XfsFileReader {
                    dev,
                    extents: Vec::new(),
                    blocksize: self.sb.blocksize as u64,
                    agblocks: self.sb.agblocks as u64,
                    agblklog: self.sb.agblklog,
                    size: core.size,
                    pos: 0,
                    inline: Some(data),
                    _layout: self.bmbt_layout(),
                })
            }
            DiFormat::Dev => Err(crate::Error::InvalidArgument(
                "xfs: device-special inode has no file data".into(),
            )),
            DiFormat::Unknown(b) => Err(crate::Error::Unsupported(format!(
                "xfs: file inode has unknown di_format {b}"
            ))),
        }
    }
}

/// Streaming reader for an XFS regular file. Walks the inode's extent list
/// on demand, reading at most one `read()`-buffer worth at a time.
pub struct XfsFileReader<'a> {
    dev: &'a mut dyn BlockDevice,
    extents: Vec<Extent>,
    blocksize: u64,
    /// AG width in blocks (for FSB → byte translation).
    agblocks: u64,
    /// log2 of the *power-of-two AG stride* used to pack FSBs. Comes from
    /// `sb_agblklog`; the address is `(ag << agblklog) | agblk`.
    agblklog: u8,
    size: u64,
    /// Logical byte position of the next byte to return.
    pos: u64,
    /// Set when the file was tiny enough to live entirely in the inode's
    /// literal area (`DiFormat::Local`). When present, `extents` is empty
    /// and reads come from this buffer.
    inline: Option<Vec<u8>>,
    /// Layout (kept for symmetry with other readers; not used by the hot
    /// path).
    _layout: BmbtLayout,
}

impl<'a> XfsFileReader<'a> {
    /// FSB → device byte address. Identical math to [`Xfs::fsb_to_byte`].
    fn fsb_to_byte(&self, fsb: u64) -> u64 {
        let ag = fsb >> self.agblklog as u32;
        let agblk = fsb & ((1u64 << self.agblklog as u32) - 1);
        ag * self.agblocks * self.blocksize + agblk * self.blocksize
    }
}

impl<'a> std::io::Read for XfsFileReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.size || buf.is_empty() {
            return Ok(0);
        }
        let want = buf.len() as u64;
        let avail = self.size - self.pos;
        let n = want.min(avail) as usize;

        if let Some(inline) = &self.inline {
            let start = self.pos as usize;
            let end = start + n;
            buf[..n].copy_from_slice(&inline[start..end]);
            self.pos += n as u64;
            return Ok(n);
        }

        // Find the extent covering `pos`. Logical units are FS blocks.
        let pos_blk = self.pos / self.blocksize;
        let pos_off = self.pos % self.blocksize;
        let extent = self
            .extents
            .iter()
            .find(|e| pos_blk >= e.offset && pos_blk < e.offset + e.blockcount as u64);
        let n_done = match extent {
            Some(e) => {
                // Read across this extent until the next extent boundary
                // or the user's buffer is full.
                let extent_blocks_left = e.offset + e.blockcount as u64 - pos_blk;
                let extent_bytes_left = extent_blocks_left * self.blocksize - pos_off;
                let to_read = (n as u64).min(extent_bytes_left) as usize;
                let phys_fsb = e.startblock + (pos_blk - e.offset);
                let phys_byte = self.fsb_to_byte(phys_fsb) + pos_off;
                self.dev
                    .read_at(phys_byte, &mut buf[..to_read])
                    .map_err(std::io::Error::other)?;
                to_read
            }
            None => {
                // Hole — synthesise zeros up to the next extent or EOF.
                let next_extent_block = self
                    .extents
                    .iter()
                    .filter(|e| e.offset > pos_blk)
                    .map(|e| e.offset)
                    .min();
                let next_byte = next_extent_block
                    .map(|b| b * self.blocksize)
                    .unwrap_or(self.size);
                let hole_bytes = next_byte.saturating_sub(self.pos);
                let to_zero = (n as u64).min(hole_bytes) as usize;
                buf[..to_zero].fill(0);
                to_zero
            }
        };
        // Defensive: don't loop forever if both branches yielded zero.
        if n_done == 0 {
            return Ok(0);
        }
        self.pos += n_done as u64;
        Ok(n_done)
    }
}

impl<'a> std::io::Seek for XfsFileReader<'a> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let total = self.size as i128;
        let new = match pos {
            std::io::SeekFrom::Start(n) => n as i128,
            std::io::SeekFrom::Current(d) => self.pos as i128 + d as i128,
            std::io::SeekFrom::End(d) => total + d as i128,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "xfs: seek to negative offset",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

impl<'a> crate::fs::FileReadHandle for XfsFileReader<'a> {
    fn len(&self) -> u64 {
        self.size
    }
}

/// Probe for the XFS superblock magic `"XFSB"` at offset 0 of LBA 0.
pub fn probe(dev: &mut dyn BlockDevice) -> Result<bool> {
    if dev.total_size() < 512 {
        return Ok(false);
    }
    let mut head = [0u8; 4];
    dev.read_at(0, &mut head)?;
    Ok(&head == b"XFSB")
}

/// Split a `/`-rooted path into non-empty components. Treats `/`, `""`,
/// and `.` as "the root" (empty vec). Multiple slashes are collapsed.
fn split_path(path: &str) -> Vec<&str> {
    path.split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect()
}

// ----------------------------------------------------------------------
// `crate::fs::Filesystem` trait impl. Mutators forward to the path
// wrappers added in `write.rs`; reads use the existing path API.
// ----------------------------------------------------------------------

impl crate::fs::FilesystemFactory for Xfs {
    type FormatOpts = format::FormatOpts;

    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        format::format(dev, opts)
    }

    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl crate::fs::Filesystem for Xfs {
    /// XFS images opt in to the REFLINK feature (`format::format`
    /// stamps `XFS_SB_FEAT_RO_COMPAT_REFLINK` and an empty per-AG
    /// REFCNTBT root), and `clone_file_path` shares extents through
    /// the refcount-btree. `clone_range` for arbitrary sub-file
    /// ranges into a pre-existing destination is stage 3 (write-on-
    /// shared CoW) — for now we expose only whole-file cloning, where
    /// the destination must not already exist.
    fn clone_capability(&self) -> crate::fs::CloneCapability {
        crate::fs::CloneCapability::WholeFile
    }

    fn clone_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        src: &std::path::Path,
        dst: &std::path::Path,
    ) -> Result<()> {
        let s = src
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 src path".into()))?;
        let d = dst
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 dst path".into()))?;
        self.clone_file_path(dev, s, d).map(|_| ())
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
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 path".into()))?;
        let len = src.len()?;
        let (mut reader, _) = src.open()?;
        let em = entry_meta_from(meta);
        self.add_file_path(dev, s, em, len, &mut reader).map(|_| ())
    }

    fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 path".into()))?;
        self.add_dir_path(dev, s, entry_meta_from(meta)).map(|_| ())
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
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 path".into()))?;
        let t = target
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 symlink target".into()))?;
        self.add_symlink_path(dev, s, t, entry_meta_from(meta))
            .map(|_| ())
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
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 path".into()))?;
        let dk = match kind {
            crate::fs::DeviceKind::Char => crate::fs::xfs::write::DeviceKind::Char,
            crate::fs::DeviceKind::Block => crate::fs::xfs::write::DeviceKind::Block,
            crate::fs::DeviceKind::Fifo => crate::fs::xfs::write::DeviceKind::Fifo,
            crate::fs::DeviceKind::Socket => crate::fs::xfs::write::DeviceKind::Socket,
        };
        self.add_device_path(dev, s, dk, major, minor, entry_meta_from(meta))
            .map(|_| ())
    }

    fn remove(&mut self, dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 path".into()))?;
        // Serialize staged directory entries so the on-disk blocks the
        // remove path reads (and splices) are current.
        self.flush_dir_batches(dev)?;
        self.remove_path(dev, s).map(|_| ())
    }

    fn list(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 path".into()))?;
        // Materialize any staged children of the listed directory so the
        // listing reflects them.
        if let Ok(ino) = self.lookup_path_ino(dev, s) {
            self.flush_one_dir_batch(dev, ino)?;
        }
        self.list_path(dev, s)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn std::io::Read + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 path".into()))?;
        let r = self.open_file_reader(dev, s)?;
        Ok(Box::new(r))
    }

    fn getattr(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<crate::fs::FileAttrs> {
        use crate::fs::xfs::inode as xi;
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 path".into()))?;
        let (ino, _buf, core) = self.resolve_path(dev, s)?;
        let kind = match core.mode & xi::S_IFMT {
            xi::S_IFREG => crate::fs::EntryKind::Regular,
            xi::S_IFDIR => crate::fs::EntryKind::Dir,
            xi::S_IFLNK => crate::fs::EntryKind::Symlink,
            xi::S_IFCHR => crate::fs::EntryKind::Char,
            xi::S_IFBLK => crate::fs::EntryKind::Block,
            0o010_000 => crate::fs::EntryKind::Fifo,
            0o140_000 => crate::fs::EntryKind::Socket,
            _ => crate::fs::EntryKind::Regular,
        };
        Ok(crate::fs::FileAttrs {
            kind,
            mode: core.mode & 0o7777,
            uid: core.uid,
            gid: core.gid,
            size: core.size,
            blocks: core.size.div_ceil(512),
            nlink: core.nlink,
            atime: core.atime.sec,
            mtime: core.mtime.sec,
            ctime: core.ctime.sec,
            // Device-node rdev lives in the data fork; not surfaced yet.
            rdev: 0,
            inode: ino as u32,
        })
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        self.flush_writes(dev)
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn crate::fs::FileReadHandle + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 path".into()))?;
        let r = self.open_file_reader(dev, s)?;
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
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 path".into()))?;
        Xfs::open_rw(self, dev, s, flags, meta)
    }

    fn read_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<std::path::PathBuf> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: non-UTF-8 path".into()))?;
        Ok(std::path::PathBuf::from(Xfs::read_symlink(self, dev, s)?))
    }
}

/// Convert public [`crate::fs::FileMeta`] to XFS's private
/// [`write::EntryMeta`]. XFS keeps `atime`/`ctime` (XFS supports them)
/// so all fields round-trip.
fn entry_meta_from(meta: crate::fs::FileMeta) -> crate::fs::xfs::write::EntryMeta {
    crate::fs::xfs::write::EntryMeta {
        mode: meta.mode,
        uid: meta.uid,
        gid: meta.gid,
        atime: meta.atime,
        mtime: meta.mtime,
        ctime: meta.ctime,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    /// Build a minimal XFS image to drive the reader through realistic
    /// addressing math. Layout: 4 KiB blocks, 4 AGs of 8 blocks each, 256-byte
    /// inodes, 16 inodes per block. Total 128 KiB. The image only has a valid
    /// superblock at offset 0; deeper structures (AG headers, inode bitmaps)
    /// are zero, so `open` succeeds but `list_path("/")` will fail because the
    /// root-inode bytes are zero (no IN magic). That's fine — these tests
    /// only exercise the addressing math.
    fn minimal_image() -> MemoryBackend {
        let blocksize = 4096u32;
        let agblocks = 8u32;
        let agcount = 4u32;
        let inodesize = 256u16;
        let inopblock = (blocksize as u16) / inodesize;
        let dblocks = (agblocks as u64) * (agcount as u64);
        let rootino = 128u64;
        let buf = superblock::synth_sb_for_tests(
            blocksize,
            dblocks,
            agblocks,
            agcount,
            inodesize,
            inopblock,
            rootino,
            superblock::XFS_SB_VERSION_5,
        );
        let total = dblocks * (blocksize as u64);
        let mut dev = MemoryBackend::new(total);
        dev.write_at(0, &buf).unwrap();
        dev
    }

    #[test]
    fn open_reads_superblock() {
        let mut dev = minimal_image();
        let xfs = Xfs::open(&mut dev).unwrap();
        assert_eq!(xfs.block_size(), 4096);
        assert_eq!(xfs.inode_size(), 256);
        assert_eq!(xfs.ag_count(), 4);
        assert_eq!(xfs.total_bytes(), 32 * 4096);
    }

    #[test]
    fn probe_returns_true_for_xfs() {
        let mut dev = minimal_image();
        assert!(probe(&mut dev).unwrap());
    }

    #[test]
    fn probe_returns_false_for_garbage() {
        let mut dev = MemoryBackend::new(4096);
        assert!(!probe(&mut dev).unwrap());
    }

    #[test]
    fn ino_split_matches_manual_math() {
        let mut dev = minimal_image();
        let xfs = Xfs::open(&mut dev).unwrap();
        // 16 inodes per block, agblocks=8 ⇒ inopblog=4, agblklog=3.
        assert_eq!(xfs.sb.inopblog, 4);
        assert_eq!(xfs.sb.agblklog, 3);
        // Inode 128 = 0b1000_0000:
        //   slot  = 128 & 0xf  = 0
        //   blk   = (128>>4) & 0x7 = 0
        //   ag    = 128 >> 7   = 1
        let (ag, blk, slot) = xfs.split_ino(128);
        assert_eq!((ag, blk, slot), (1, 0, 0));
        // Inode 0: everything zero.
        let (ag, blk, slot) = xfs.split_ino(0);
        assert_eq!((ag, blk, slot), (0, 0, 0));
        // Inode (ag=2, blk=3, slot=5) = 2*128 + 3*16 + 5 = 256 + 48 + 5 = 309.
        let (ag, blk, slot) = xfs.split_ino(309);
        assert_eq!((ag, blk, slot), (2, 3, 5));
    }

    #[test]
    fn ino_byte_offset_rejects_out_of_range_ag() {
        let mut dev = minimal_image();
        let xfs = Xfs::open(&mut dev).unwrap();
        // ag=4 is one past the end (agcount=4 ⇒ valid 0..3).
        // 4 * 128 = 512 (since agblklog+inopblog = 7).
        let bad_ino = 4u64 * 128;
        assert!(matches!(
            xfs.ino_byte_offset(bad_ino),
            Err(crate::Error::InvalidImage(_))
        ));
    }

    #[test]
    fn ino_byte_offset_arithmetic() {
        let mut dev = minimal_image();
        let xfs = Xfs::open(&mut dev).unwrap();
        // Inode 128 = ag=1, blk=0, slot=0.
        // Byte = 1 * 8 * 4096 + 0 + 0 = 32768.
        let off = xfs.ino_byte_offset(128).unwrap();
        assert_eq!(off, 8 * 4096);
        // Inode 309 = ag=2, blk=3, slot=5.
        // Byte = 2*8*4096 + 3*4096 + 5*256
        //      = 65536 + 12288 + 1280 = 79104.
        let off = xfs.ino_byte_offset(309).unwrap();
        assert_eq!(off, 2 * 8 * 4096 + 3 * 4096 + 5 * 256);
    }

    #[test]
    fn split_path_handles_edge_cases() {
        assert!(split_path("/").is_empty());
        assert!(split_path("").is_empty());
        assert!(split_path("/.").is_empty());
        assert_eq!(split_path("/a/b"), vec!["a", "b"]);
        assert_eq!(split_path("a//b///c"), vec!["a", "b", "c"]);
    }

    /// A handcrafted v3 inode + a shortform directory pointing at one entry,
    /// then verify the full `list_path("/")` plumbing.
    #[test]
    fn list_root_shortform() {
        // Use the synthetic superblock from `minimal_image`, then carve a
        // root inode at the address `split_ino(rootino)` yields.
        let mut dev = minimal_image();
        let xfs = Xfs::open(&mut dev).unwrap();
        let rootino = xfs.sb.rootino;
        let off = xfs.ino_byte_offset(rootino).unwrap();
        let mut ino_buf = vec![0u8; 256];
        ino_buf[0..2].copy_from_slice(&inode::XFS_DINODE_MAGIC.to_be_bytes());
        ino_buf[2..4].copy_from_slice(&(inode::S_IFDIR | 0o755).to_be_bytes());
        ino_buf[4] = 3; // v3
        ino_buf[5] = 1; // local
        ino_buf[16..20].copy_from_slice(&2u32.to_be_bytes()); // nlink
        ino_buf[152..160].copy_from_slice(&rootino.to_be_bytes()); // di_ino
        // Literal area starts at 176 — write the shortform header + entries.
        let lit_off = 176;
        ino_buf[lit_off] = 1; // count
        ino_buf[lit_off + 1] = 0; // i8count
        ino_buf[lit_off + 2..lit_off + 6].copy_from_slice(&(rootino as u32).to_be_bytes());
        // One entry: name "hi" → ino 200, ftype=REG.
        let entry_off = lit_off + 6;
        ino_buf[entry_off] = 2; // namelen
        ino_buf[entry_off + 1] = 0;
        ino_buf[entry_off + 2] = 0; // dir-offset placeholder
        ino_buf[entry_off + 3..entry_off + 5].copy_from_slice(b"hi");
        ino_buf[entry_off + 5] = dir::XFS_DIR3_FT_REG_FILE;
        ino_buf[entry_off + 6..entry_off + 10].copy_from_slice(&200u32.to_be_bytes());
        dev.write_at(off, &ino_buf).unwrap();

        // Re-open and list.
        let xfs = Xfs::open(&mut dev).unwrap();
        let entries = xfs.list_path(&mut dev, "/").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "hi");
        assert_eq!(entries[0].inode, 200);
        assert_eq!(entries[0].kind, crate::fs::EntryKind::Regular);
    }

    /// Build the same minimal image as `list_root_shortform`, but route the
    /// shortform entry "lnk" → ino 200, then carve inode 200 as an inline
    /// symlink with target "/etc/hostname". Verify `read_symlink` returns
    /// that target.
    #[test]
    fn read_inline_symlink_end_to_end() {
        let mut dev = minimal_image();
        let xfs = Xfs::open(&mut dev).unwrap();
        let rootino = xfs.sb.rootino;

        // Carve the root dir inode as before but with one entry pointing at
        // ino 200 (ftype = SYMLINK).
        let root_off = xfs.ino_byte_offset(rootino).unwrap();
        let mut root_buf = vec![0u8; 256];
        root_buf[0..2].copy_from_slice(&inode::XFS_DINODE_MAGIC.to_be_bytes());
        root_buf[2..4].copy_from_slice(&(inode::S_IFDIR | 0o755).to_be_bytes());
        root_buf[4] = 3; // v3
        root_buf[5] = 1; // local
        root_buf[16..20].copy_from_slice(&2u32.to_be_bytes()); // nlink
        root_buf[152..160].copy_from_slice(&rootino.to_be_bytes()); // di_ino
        let lit_off = 176;
        root_buf[lit_off] = 1; // count
        root_buf[lit_off + 1] = 0; // i8count
        root_buf[lit_off + 2..lit_off + 6].copy_from_slice(&(rootino as u32).to_be_bytes());
        let e = lit_off + 6;
        root_buf[e] = 3; // namelen "lnk"
        root_buf[e + 1] = 0;
        root_buf[e + 2] = 0;
        root_buf[e + 3..e + 6].copy_from_slice(b"lnk");
        root_buf[e + 6] = dir::XFS_DIR3_FT_SYMLINK;
        root_buf[e + 7..e + 11].copy_from_slice(&200u32.to_be_bytes());
        dev.write_at(root_off, &root_buf).unwrap();

        // Carve inode 200 as an inline symlink. Inode 200 in our scheme:
        //   inopblog=4, agblklog=3 ⇒ ag=200>>7=1, blk=(200>>4)&7=4, slot=200&15=8.
        //   byte = 1*8*4096 + 4*4096 + 8*256 = 32768 + 16384 + 2048 = 51200.
        let off200 = xfs.ino_byte_offset(200).unwrap();
        let mut buf = vec![0u8; 256];
        buf[0..2].copy_from_slice(&inode::XFS_DINODE_MAGIC.to_be_bytes());
        buf[2..4].copy_from_slice(&(inode::S_IFLNK | 0o777).to_be_bytes());
        buf[4] = 3; // v3
        buf[5] = 1; // local
        buf[16..20].copy_from_slice(&1u32.to_be_bytes()); // nlink
        let target = "/etc/hostname";
        buf[56..64].copy_from_slice(&(target.len() as u64).to_be_bytes());
        buf[152..160].copy_from_slice(&200u64.to_be_bytes()); // di_ino
        let lit = 176;
        buf[lit..lit + target.len()].copy_from_slice(target.as_bytes());
        dev.write_at(off200, &buf).unwrap();

        let xfs = Xfs::open(&mut dev).unwrap();
        let got = xfs.read_symlink(&mut dev, "/lnk").unwrap();
        assert_eq!(got, target);
    }

    /// Hand-craft a v3 root inode in `BTREE` di_format whose bmbt root
    /// (in the inode's literal area) points at a single-level-0 leaf
    /// block, which in turn maps directory logical block 0 to a single
    /// physical block holding a v5 XDB3 block-format directory with two
    /// user entries (plus the synthetic `.` / `..`). This regression
    /// test asserts that `list_path("/")` no longer returns
    /// `Error::Unsupported` for BTREE-format directory inodes, and that
    /// every entry decoded by `decode_block_dir` is surfaced in the
    /// resulting `Vec<DirEntry>`.
    #[test]
    fn list_root_btree_dir_single_level() {
        let mut dev = minimal_image();
        let xfs = Xfs::open(&mut dev).unwrap();
        let rootino = xfs.sb.rootino;
        let inodesize = xfs.sb.inodesize as usize;
        assert_eq!(inodesize, 256, "minimal_image is fixed at 256-byte inodes");

        // Pick two free FSBs in AG 2: FSB 16 = (ag=2, blk=0) for the
        // BMBT leaf node, FSB 18 = (ag=2, blk=2) for the directory data
        // block. AG 1 is reserved for the root-inode chunk in this image
        // layout; AG 2 has nothing carved into it.
        let leaf_fsb: u64 = 16;
        let data_fsb: u64 = 18;

        // ----- Step 1: dir data block at FSB `data_fsb`. -----
        // Use the production encoder so we hit the same XDB3 layout
        // (header + per-entry leaf array + bestfree) that mkfs.xfs and
        // xfs_repair recognise. Two user entries: "alpha" and "beta".
        let block_basic_blkno = xfs.fsb_to_byte(data_fsb) / 512;
        let entries = vec![
            ("alpha".to_string(), 300u64, dir::XFS_DIR3_FT_REG_FILE),
            ("beta".to_string(), 301u64, dir::XFS_DIR3_FT_DIR),
        ];
        let dir_block = dir::encode_v5_block_dir(
            xfs.sb.dir_block_size() as usize,
            rootino,
            rootino,
            &entries,
            &[0u8; 16],
            block_basic_blkno,
        )
        .unwrap();
        let data_byte = xfs.fsb_to_byte(data_fsb);
        dev.write_at(data_byte, &dir_block).unwrap();

        // ----- Step 2: BMBT leaf block at FSB `leaf_fsb`. -----
        // v5 BMA3 header (72 B) + one packed extent record (16 B):
        //   offset = 0, startblock = data_fsb, blockcount = 1.
        let mut leaf = vec![0u8; xfs.sb.blocksize as usize];
        leaf[0..4].copy_from_slice(&bmbt::XFS_BMAP_CRC_MAGIC.to_be_bytes());
        leaf[4..6].copy_from_slice(&0u16.to_be_bytes()); // level
        leaf[6..8].copy_from_slice(&1u16.to_be_bytes()); // numrecs
        let extent = bmbt::Extent {
            offset: 0,
            startblock: data_fsb,
            blockcount: 1,
            unwritten: false,
        };
        leaf[72..72 + 16].copy_from_slice(&extent.encode());
        let leaf_byte = xfs.fsb_to_byte(leaf_fsb);
        dev.write_at(leaf_byte, &leaf).unwrap();

        // ----- Step 3: root inode at di_format = BTREE. -----
        let off = xfs.ino_byte_offset(rootino).unwrap();
        let mut ino_buf = vec![0u8; inodesize];
        ino_buf[0..2].copy_from_slice(&inode::XFS_DINODE_MAGIC.to_be_bytes());
        ino_buf[2..4].copy_from_slice(&(inode::S_IFDIR | 0o755).to_be_bytes());
        ino_buf[4] = 3; // v3
        ino_buf[5] = 3; // BTREE
        ino_buf[16..20].copy_from_slice(&2u32.to_be_bytes()); // nlink
        ino_buf[152..160].copy_from_slice(&rootino.to_be_bytes()); // di_ino

        // bmbt root in the literal area at offset 176, length 80 bytes:
        //   [0..2]  level    = 1
        //   [2..4]  numrecs  = 1
        //   [4..12] keys[0]  = 0      (br_startoff)
        //   [72..80] ptrs[0] = leaf_fsb   (tail layout — decode_root prefers it)
        let lit_off = 176;
        let lit_end = inodesize;
        let lit_len = lit_end - lit_off;
        assert_eq!(lit_len, 80, "literal area for 256-byte v3 inode is 80 B");
        ino_buf[lit_off..lit_off + 2].copy_from_slice(&1u16.to_be_bytes());
        ino_buf[lit_off + 2..lit_off + 4].copy_from_slice(&1u16.to_be_bytes());
        ino_buf[lit_off + 4..lit_off + 12].copy_from_slice(&0u64.to_be_bytes());
        ino_buf[lit_end - 8..lit_end].copy_from_slice(&leaf_fsb.to_be_bytes());

        dev.write_at(off, &ino_buf).unwrap();

        // ----- Step 4: list and validate. -----
        let xfs = Xfs::open(&mut dev).unwrap();
        let listed = xfs.list_path(&mut dev, "/").unwrap();
        // encode_v5_block_dir injects synthetic "." and "..", so the
        // dir block has 4 entries total.
        let names: Vec<&str> = listed.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"alpha"), "missing 'alpha' in {names:?}");
        assert!(names.contains(&"beta"), "missing 'beta' in {names:?}");
        assert!(names.contains(&"."), "missing '.' in {names:?}");
        assert!(names.contains(&".."), "missing '..' in {names:?}");
        let alpha = listed.iter().find(|e| e.name == "alpha").unwrap();
        assert_eq!(alpha.inode, 300);
        assert_eq!(alpha.kind, crate::fs::EntryKind::Regular);
        let beta = listed.iter().find(|e| e.name == "beta").unwrap();
        assert_eq!(beta.inode, 301);
        assert_eq!(beta.kind, crate::fs::EntryKind::Dir);
    }

    /// Multi-level B-tree directory: a synthetic root inode in BTREE
    /// di_format whose bmbt root sits at level 2 (root → 2 internal
    /// nodes → 4 leaf nodes). The leaves point at 20 directory data
    /// blocks, each holding ~252 entries; the test injects 4200 user
    /// entries total (above the 4096 threshold called out in the
    /// issue) and asserts every name surfaces through `list_path`.
    #[test]
    fn list_root_btree_dir_multi_level() {
        // Larger image: 4 AGs of 64 4-KiB blocks = 1 MiB.
        let blocksize = 4096u32;
        let agblocks = 64u32;
        let agcount = 4u32;
        let inodesize = 256u16;
        let inopblock = (blocksize as u16) / inodesize;
        let dblocks = (agblocks as u64) * (agcount as u64);
        let rootino = 128u64;
        let sb_buf = superblock::synth_sb_for_tests(
            blocksize,
            dblocks,
            agblocks,
            agcount,
            inodesize,
            inopblock,
            rootino,
            superblock::XFS_SB_VERSION_5,
        );
        let total = dblocks * (blocksize as u64);
        let mut dev = MemoryBackend::new(total);
        dev.write_at(0, &sb_buf).unwrap();

        let xfs = Xfs::open(&mut dev).unwrap();

        // Hand out FSBs in AG 2 for the dir data blocks and the bmbt
        // nodes (AG 1 holds the root-inode chunk in this layout).
        // FSBs are global indices: ag*64 + agblk.
        let ag2_base: u64 = 2 * (agblocks as u64);
        let data_fsbs: Vec<u64> = (0..20).map(|i| ag2_base + i).collect();
        // Leaves at FSBs ag2+30..34 (4 leaves), internals at ag2+40..42
        // (2 nodes). Root sits in the inode literal area itself.
        let leaf_fsbs: [u64; 4] = [ag2_base + 30, ag2_base + 31, ag2_base + 32, ag2_base + 33];
        let intern_fsbs: [u64; 2] = [ag2_base + 40, ag2_base + 41];

        // Layout the directory entries across 20 data blocks. The first
        // data block also carries the synthetic "." and "..".
        let dir_block_size = xfs.sb.dir_block_size() as usize;
        // Generate 4200 user entries with 4-char hex names ("0000" .. "1067")
        // so each padded record is exactly 16 bytes (252/block at 4 KiB).
        let mut all_user: Vec<(String, u64, u8)> = Vec::with_capacity(4200);
        for i in 0..4200u32 {
            let name = format!("{i:04x}");
            let inum = 1_000 + i as u64;
            all_user.push((name, inum, dir::XFS_DIR3_FT_REG_FILE));
        }
        // Distribute roughly evenly across the 20 blocks.
        let per_block = all_user.len().div_ceil(data_fsbs.len());
        let uuid = [0u8; 16];
        for (i, fsb) in data_fsbs.iter().enumerate() {
            let slice_lo = i * per_block;
            let slice_hi = ((i + 1) * per_block).min(all_user.len());
            let mut block_entries: Vec<(String, u64, u8)> = if i == 0 {
                // First data block also carries "." and "..".
                vec![
                    (".".to_string(), rootino, dir::XFS_DIR3_FT_DIR),
                    ("..".to_string(), rootino, dir::XFS_DIR3_FT_DIR),
                ]
            } else {
                Vec::new()
            };
            block_entries.extend(all_user[slice_lo..slice_hi].iter().cloned());
            let block_basic_blkno = xfs.fsb_to_byte(*fsb) / 512;
            let block = dir::encode_v5_data_block(
                dir_block_size,
                &block_entries,
                &uuid,
                block_basic_blkno,
                rootino,
            )
            .unwrap();
            dev.write_at(xfs.fsb_to_byte(*fsb), &block).unwrap();
        }

        // Build 4 BMBT leaf blocks, each mapping 5 consecutive dir
        // logical blocks to physical FSBs from `data_fsbs`. Leaf 0:
        // dir-block 0..4 → data_fsbs[0..5]; leaf 1: 5..9; etc.
        let hdr = 72; // v5 bmbt block header
        for (li, leaf_fsb) in leaf_fsbs.iter().enumerate() {
            let mut leaf = vec![0u8; blocksize as usize];
            leaf[0..4].copy_from_slice(&bmbt::XFS_BMAP_CRC_MAGIC.to_be_bytes());
            leaf[4..6].copy_from_slice(&0u16.to_be_bytes()); // level=0
            leaf[6..8].copy_from_slice(&5u16.to_be_bytes()); // numrecs=5
            for j in 0..5 {
                let logical_blk = (li * 5 + j) as u64;
                let phys_fsb = data_fsbs[li * 5 + j];
                let e = bmbt::Extent {
                    offset: logical_blk,
                    startblock: phys_fsb,
                    blockcount: 1,
                    unwritten: false,
                };
                leaf[hdr + j * 16..hdr + (j + 1) * 16].copy_from_slice(&e.encode());
            }
            dev.write_at(xfs.fsb_to_byte(*leaf_fsb), &leaf).unwrap();
        }

        // Build 2 BMBT internal (level-1) blocks, each pointing at 2 leaves.
        let max_recs = (blocksize as usize - hdr) / (bmbt::BMBT_KEY_SIZE + bmbt::BMBT_PTR_SIZE);
        for (ni, intern_fsb) in intern_fsbs.iter().enumerate() {
            let mut node = vec![0u8; blocksize as usize];
            node[0..4].copy_from_slice(&bmbt::XFS_BMAP_CRC_MAGIC.to_be_bytes());
            node[4..6].copy_from_slice(&1u16.to_be_bytes()); // level=1
            node[6..8].copy_from_slice(&2u16.to_be_bytes()); // numrecs=2
            // keys = the logical-block start of each child leaf.
            // Leaf li covers logical blocks [li*5, li*5+5).
            let l0 = ni * 2;
            let l1 = ni * 2 + 1;
            let key0 = (l0 * 5) as u64;
            let key1 = (l1 * 5) as u64;
            node[hdr..hdr + 8].copy_from_slice(&key0.to_be_bytes());
            node[hdr + 8..hdr + 16].copy_from_slice(&key1.to_be_bytes());
            let ptrs_start = hdr + max_recs * bmbt::BMBT_KEY_SIZE;
            node[ptrs_start..ptrs_start + 8].copy_from_slice(&leaf_fsbs[l0].to_be_bytes());
            node[ptrs_start + 8..ptrs_start + 16].copy_from_slice(&leaf_fsbs[l1].to_be_bytes());
            dev.write_at(xfs.fsb_to_byte(*intern_fsb), &node).unwrap();
        }

        // Carve the root inode in BTREE di_format with a level-2 bmbt root.
        let off = xfs.ino_byte_offset(rootino).unwrap();
        let mut ino_buf = vec![0u8; inodesize as usize];
        ino_buf[0..2].copy_from_slice(&inode::XFS_DINODE_MAGIC.to_be_bytes());
        ino_buf[2..4].copy_from_slice(&(inode::S_IFDIR | 0o755).to_be_bytes());
        ino_buf[4] = 3; // v3
        ino_buf[5] = 3; // BTREE
        ino_buf[16..20].copy_from_slice(&2u32.to_be_bytes()); // nlink
        ino_buf[152..160].copy_from_slice(&rootino.to_be_bytes()); // di_ino

        // bmbt root: level=2, numrecs=2.
        let lit_off = 176;
        let lit_end = inodesize as usize;
        ino_buf[lit_off..lit_off + 2].copy_from_slice(&2u16.to_be_bytes());
        ino_buf[lit_off + 2..lit_off + 4].copy_from_slice(&2u16.to_be_bytes());
        ino_buf[lit_off + 4..lit_off + 12].copy_from_slice(&0u64.to_be_bytes()); // key 0
        ino_buf[lit_off + 12..lit_off + 20].copy_from_slice(&((2 * 5) as u64).to_be_bytes()); // key 1
        // tail layout for ptrs.
        ino_buf[lit_end - 16..lit_end - 8].copy_from_slice(&intern_fsbs[0].to_be_bytes());
        ino_buf[lit_end - 8..lit_end].copy_from_slice(&intern_fsbs[1].to_be_bytes());
        dev.write_at(off, &ino_buf).unwrap();

        // List and validate.
        let xfs = Xfs::open(&mut dev).unwrap();
        let listed = xfs.list_path(&mut dev, "/").unwrap();
        // Must surface every user entry plus "." and "..".
        assert!(
            listed.len() >= all_user.len() + 2,
            "expected >= {} entries, got {}",
            all_user.len() + 2,
            listed.len()
        );
        let names: std::collections::HashSet<&str> =
            listed.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains("."), "missing '.'");
        assert!(names.contains(".."), "missing '..'");
        for (name, _, _) in &all_user {
            assert!(names.contains(name.as_str()), "missing user entry {name:?}");
        }
        // Spot-check that 4096+ entries came through, as the issue
        // explicitly calls out.
        assert!(
            listed.len() >= 4096,
            "multi-level dir should yield ≥ 4096 entries, got {}",
            listed.len()
        );
    }
}
