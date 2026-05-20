//! F2FS — Flash-Friendly File System. Read driver.
//!
//! ## Status
//!
//! Read-only. Implemented:
//!
//! - Both superblock copies parsed; we accept whichever validates.
//! - Live checkpoint pack picked by version + CRC32 over the head block.
//! - NAT lookups: checkpoint NAT journal first, then on-disk NAT pages
//!   from whichever pack the checkpoint nominated.
//! - Inode block decoded (mode, size, inline flags, i_addr, i_nid).
//! - Directory walk: regular 4 KiB dentry blocks and the inline-dentry
//!   path for tiny dirs.
//! - File streaming via [`F2fs::open_file_reader`], honouring
//!   `INLINE_DATA` and the full direct → indirect → triple-indirect
//!   chain. Block reads are 4 KiB at a time; nothing larger than a node
//!   block is ever buffered.
//!
//! ## Unsupported (returned as [`crate::Error::Unsupported`])
//!
//! - Encryption (`f2fs_encryption_v2`).
//! - File-level compression (lzo / lz4 / zstd).
//! - Quota inodes, project quotas.
//! - Hard-link namespace traversal beyond `parent → name → inode`.
//! - Crash-recovery / fsync roll-forward replay.
//! - Writing anything (log-structured writes are a follow-up).
//!
//! ## References
//!
//! - <https://docs.kernel.org/filesystems/f2fs.html>
//! - "F2FS: A New File System for Flash Storage" — Lee et al., USENIX
//!   FAST '15.
//!
//! ## Implementation files
//!
//! - [`superblock`]: SB layout + load.
//! - [`checkpoint`]: CP pack discovery, CRC32 validation, NAT journal.
//! - [`nat`]: nid → physical block lookup.
//! - [`inode`]: inode + direct/indirect node block decoders.
//! - [`dir`]: dentry block + inline-dentry walker.
//! - [`mod@file`]: data-block resolver and streaming `Read`er.

pub mod checkpoint;
pub mod constants;
pub mod dir;
pub mod file;
pub mod format;
pub mod inode;
pub mod nat;
pub mod rw;
pub mod superblock;
pub mod write;

use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;

pub use file::FileReader;
pub use format::{FormatOpts, Geometry, plan_geometry};
pub use superblock::{F2FS_MAGIC, SB_OFFSET_BACKUP, SB_OFFSET_PRIMARY, Superblock};
pub use write::Writer;

use checkpoint::Checkpoint;
use constants::{F2FS_BLKSIZE, S_IFDIR, S_IFMT, S_IFREG};
use dir::{RawDentry, decode_dentry_block, decode_inline_dentries};
use inode::{F2fsInode, decode_inode_block};

/// Probe for either F2FS superblock copy.
pub fn probe(dev: &mut dyn BlockDevice) -> Result<bool> {
    if dev.total_size() < SB_OFFSET_BACKUP + 4 {
        return Ok(false);
    }
    let mut head = [0u8; 4];
    dev.read_at(SB_OFFSET_PRIMARY, &mut head)?;
    if u32::from_le_bytes(head) == F2FS_MAGIC {
        return Ok(true);
    }
    dev.read_at(SB_OFFSET_BACKUP, &mut head)?;
    Ok(u32::from_le_bytes(head) == F2FS_MAGIC)
}

/// Mounted F2FS volume.
///
/// The struct caches the superblock and checkpoint pack at open time so
/// every `list_path` / `open_file_reader` call is a flat sequence of
/// 4 KiB reads against the block device with no further metadata-pack
/// re-scanning. Opening through [`F2fs::format`] additionally arms a
/// [`Writer`] for in-place create / remove / flush operations.
pub struct F2fs {
    sb: Superblock,
    cp: Checkpoint,
    pub(crate) writer: Option<Writer>,
}

impl F2fs {
    /// Open the volume read-only and lock in the live checkpoint pack.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let sb = superblock::load(dev)?;
        let cp = Checkpoint::load(dev, &sb)?;
        Ok(Self {
            sb,
            cp,
            writer: None,
        })
    }

    /// Format `dev` as a fresh F2FS volume and return a handle ready for
    /// `create_*` / `remove` / `flush` calls. Any prior content of `dev`
    /// is overwritten.
    pub fn format(dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<Self> {
        let bs = F2FS_BLKSIZE as u64;
        let total_blocks = dev.total_size() / bs;
        let geom = format::plan_geometry(total_blocks, opts)?;
        dev.zero_range(0, geom.main_blkaddr as u64 * bs)?;
        format::write_superblocks(dev, &geom, opts)?;
        format::wipe_metadata_region(dev, &geom)?;

        let sb = superblock::load(dev)?;
        let writer = Writer::new(geom, sb.clone());
        let mut me = Self {
            sb,
            cp: Checkpoint {
                version: 0,
                user_block_count: total_blocks,
                valid_block_count: 0,
                rsvd_segment_count: 0,
                overprov_segment_count: 0,
                flags: 0,
                cp_pack_start_sum: 1,
                cp_pack_total_block_count: 2,
                cp_payload: 0,
                head_blkaddr: geom.cp_blkaddr,
                nat_ver_bitmap_bytesize: 64,
                sit_ver_bitmap_bytesize: 64,
                cur_nat_pack: 0,
                cur_sit_pack: 0,
                nat_journal: Vec::new(),
                cur_node_segno: [0, 1, 2],
                cur_node_blkoff: [0, 0, 0],
                cur_data_segno: [3, 4, 5],
                cur_data_blkoff: [0, 0, 0],
                free_segment_count: 0,
                valid_node_count: 0,
                valid_inode_count: 0,
                next_free_nid: 0,
            },
            writer: Some(writer),
        };
        me.flush(dev)?;
        me.cp = Checkpoint::load(dev, &me.sb)?;
        Ok(me)
    }

    /// Create a regular file at `path` populated from `src`. The parent
    /// directory must exist; the leaf must not. Returns the new nid.
    pub fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        src: crate::fs::FileSource,
        meta: crate::fs::FileMeta,
    ) -> Result<u32> {
        self.writer_mut()?.add_file(dev, path, src, meta)
    }

    /// Create a directory at `path`.
    pub fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<u32> {
        self.writer_mut()?.add_dir(dev, path, meta)
    }

    /// Create a symbolic link.
    pub fn create_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        target: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<u32> {
        self.writer_mut()?.add_symlink(dev, path, target, meta)
    }

    /// Create a hard link at `dst` pointing at the inode currently
    /// addressed by `src`. The two paths share an inode and `i_links` on
    /// that inode is bumped. `src` must be a non-directory, non-symlink
    /// path created in the current writer session (fresh-image semantics —
    /// see [`Writer::add_hardlink`] docs).
    pub fn create_hardlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        src: &std::path::Path,
        dst: &std::path::Path,
    ) -> Result<u32> {
        self.writer_mut()?.add_hardlink(dev, src, dst)
    }

    /// Create a device / FIFO / socket node.
    pub fn create_device(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        kind: crate::fs::DeviceKind,
        major: u32,
        minor: u32,
        meta: crate::fs::FileMeta,
    ) -> Result<u32> {
        self.writer_mut()?
            .add_device(dev, path, kind, major, minor, meta)
    }

    /// Remove a file / empty directory / symlink. Non-empty directories
    /// are rejected with [`crate::Error::InvalidArgument`].
    pub fn remove(&mut self, dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<()> {
        self.writer_mut()?.remove(dev, path)
    }

    /// Persist in-memory writer state to `dev`. Idempotent. No-op on
    /// read-only handles.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        let Some(w) = self.writer.as_mut() else {
            return Ok(());
        };
        w.flush(dev)?;
        if let Ok(cp) = Checkpoint::load(dev, &self.sb) {
            self.cp = cp;
        }
        Ok(())
    }

    fn writer_mut(&mut self) -> Result<&mut Writer> {
        self.writer.as_mut().ok_or_else(|| {
            crate::Error::Unsupported(
                "f2fs: handle opened read-only; use F2fs::format for write access".into(),
            )
        })
    }

    pub fn total_bytes(&self) -> u64 {
        self.sb.block_count << self.sb.log_blocksize
    }

    pub fn block_size(&self) -> u32 {
        1u32 << self.sb.log_blocksize
    }

    pub fn volume_name(&self) -> &str {
        &self.sb.volume_name
    }

    pub fn superblock(&self) -> &Superblock {
        &self.sb
    }

    pub fn checkpoint(&self) -> &Checkpoint {
        &self.cp
    }

    /// Resolve a posix-style path (`/a/b/c`) to its node id, starting at
    /// the root.
    pub fn resolve_path(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<u32> {
        let mut ino = self.sb.root_ino;
        if path == "/" || path.is_empty() {
            return Ok(ino);
        }
        for comp in path.trim_matches('/').split('/') {
            if comp.is_empty() {
                continue;
            }
            let (inode_block, inode) = self.read_inode(dev, ino)?;
            if inode.mode & S_IFMT != S_IFDIR {
                return Err(crate::Error::InvalidArgument(format!(
                    "f2fs: '{path}': '{comp}' parent is not a directory"
                )));
            }
            let entries = self.list_inode(dev, &inode, &inode_block)?;
            let Some(found) = entries.iter().find(|e| e.name == comp.as_bytes()) else {
                return Err(crate::Error::InvalidArgument(format!(
                    "f2fs: '{path}': '{comp}' not found"
                )));
            };
            ino = found.ino;
        }
        Ok(ino)
    }

    /// Read the inode block for `nid`, return both the raw 4 KiB block
    /// (needed for the inline-dentry / inline-data payload area) and the
    /// decoded metadata view.
    pub fn read_inode(&self, dev: &mut dyn BlockDevice, nid: u32) -> Result<(Vec<u8>, F2fsInode)> {
        let addr = nat::lookup_node(dev, &self.sb, &self.cp, nid)?;
        let mut buf = vec![0u8; F2FS_BLKSIZE];
        dev.read_at(addr.block as u64 * self.sb.block_size() as u64, &mut buf)?;
        let inode = decode_inode_block(&buf)?;
        Ok((buf, inode))
    }

    /// List the entries of a directory inode (handles inline + block).
    pub fn list_inode(
        &self,
        dev: &mut dyn BlockDevice,
        inode: &F2fsInode,
        inode_block: &[u8],
    ) -> Result<Vec<RawDentry>> {
        if inode.is_inline_dentry() {
            return decode_inline_dentries(inode, inode_block);
        }
        // Walk every populated data block in the directory's logical
        // sequence. Each 4 KiB block decodes independently — we just
        // concatenate the dentries.
        let total_blocks = inode.size.div_ceil(F2FS_BLKSIZE as u64);
        let mut out = Vec::new();
        let mut buf = vec![0u8; F2FS_BLKSIZE];
        for i in 0..total_blocks {
            let phys = file::logical_to_physical(dev, &self.sb, &self.cp, inode, i)?;
            if phys == 0 || phys == constants::NEW_ADDR {
                continue;
            }
            dev.read_at(phys as u64 * self.sb.block_size() as u64, &mut buf)?;
            let entries = decode_dentry_block(&buf)?;
            out.extend(entries);
        }
        Ok(out)
    }

    /// List the entries at `path`.
    pub fn list_path(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let ino = self.resolve_path(dev, path)?;
        let (inode_block, inode) = self.read_inode(dev, ino)?;
        if inode.mode & S_IFMT != S_IFDIR {
            return Err(crate::Error::InvalidArgument(format!(
                "f2fs: '{path}' is not a directory"
            )));
        }
        let raws = self.list_inode(dev, &inode, &inode_block)?;
        // Hide "." and ".." from user-visible listings — the on-disk
        // entries are there for fsck.f2fs but applications normally
        // expect them filtered out (matching `ls -A` behavior).
        Ok(raws
            .into_iter()
            .filter(|d| d.name.as_slice() != b"." && d.name.as_slice() != b"..")
            .map(|d| d.into_dir_entry())
            .collect())
    }

    /// Open a regular file at `path` for streaming reads. Returns a
    /// boxed `Read` borrowing both `self` (the cached SB + CP) and the
    /// device. At most 4 KiB of file payload is buffered.
    pub fn open_file_reader<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<Box<dyn Read + 'a>> {
        let r = self.open_file_seekable(dev, path)?;
        Ok(Box::new(r))
    }

    /// Like [`Self::open_file_reader`] but returns the concrete
    /// [`FileReader`] type so callers can `Seek` on it. Backs
    /// [`crate::fs::Filesystem::open_file_ro`].
    pub fn open_file_seekable<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<FileReader<'a>> {
        let ino = self.resolve_path(dev, path)?;
        let (inode_block, inode) = self.read_inode(dev, ino)?;
        if inode.mode & S_IFMT != S_IFREG {
            return Err(crate::Error::InvalidArgument(format!(
                "f2fs: '{path}' is not a regular file"
            )));
        }
        let _ = inode;
        FileReader::new(dev, self.sb.clone(), self.cp.clone(), inode_block)
    }
}

impl crate::fs::FilesystemFactory for F2fs {
    type FormatOpts = FormatOpts;

    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }

    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl crate::fs::Filesystem for F2fs {
    fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        src: crate::fs::FileSource,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        self.create_file(dev, path, src, meta).map(|_| ())
    }

    fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        self.create_dir(dev, path, meta).map(|_| ())
    }

    fn create_symlink(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        target: &std::path::Path,
        meta: crate::fs::FileMeta,
    ) -> Result<()> {
        self.create_symlink(dev, path, target, meta).map(|_| ())
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
        self.create_device(dev, path, kind, major, minor, meta)
            .map(|_| ())
    }

    fn remove(&mut self, dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<()> {
        self.remove(dev, path)
    }

    fn list(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("f2fs: non-UTF-8 path".into()))?;
        self.list_path(dev, s)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn Read + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("f2fs: non-UTF-8 path".into()))?;
        self.open_file_reader(dev, s)
    }

    fn open_file_ro<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn crate::fs::FileReadHandle + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("f2fs: non-UTF-8 path".into()))?;
        let r = self.open_file_seekable(dev, s)?;
        Ok(Box::new(r))
    }

    fn open_file_rw<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
        flags: crate::fs::OpenFlags,
        meta: Option<crate::fs::FileMeta>,
    ) -> Result<Box<dyn crate::fs::FileHandle + 'a>> {
        let handle = rw::F2fsFileHandle::open(self, dev, path, flags, meta)?;
        Ok(Box::new(handle))
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        Self::flush(self, dev)
    }
}

#[cfg(test)]
mod tests {
    use super::checkpoint::{
        Checkpoint, NatJournalEntry, encode_cp_head, encode_nat_journal_block,
    };
    use super::constants::{
        ADDRS_PER_BLOCK, ADDRS_PER_INODE, F2FS_BLKSIZE, F2FS_FT_DIR, F2FS_FT_REG_FILE,
        F2FS_INLINE_DATA, F2FS_INLINE_DENTRY, NR_DENTRY_IN_BLOCK, S_IFDIR, S_IFREG,
    };
    use super::dir::{RawDentry, encode_dentry_block, encode_inline_dentries_payload};
    use super::inode::{F2fsInode, encode_direct_node, encode_indirect_node, encode_inode_block};
    use super::nat::encode_nat_entry;
    use super::superblock::F2FS_MAGIC;
    use super::*;
    use crate::block::MemoryBackend;
    use crate::fs::EntryKind;

    /// Layout knobs for [`build_image`].
    struct ImageLayout {
        blocks: u64,
        sb_blocks: u32,      // 2 (primary + backup)
        cp_segs: u32,        // 1
        sit_segs: u32,       // 1
        nat_segs: u32,       // 2 (1 even pack + 1 odd pack)
        ssa_segs: u32,       // 1
        blocks_per_seg: u32, // 1 → tiny test image (smallest legal value here)
    }

    impl ImageLayout {
        fn default_tiny() -> Self {
            Self {
                blocks: 256,
                sb_blocks: 2,
                cp_segs: 2, // two packs
                sit_segs: 1,
                nat_segs: 2,
                ssa_segs: 1,
                blocks_per_seg: 4,
            }
        }
    }

    /// Build a synthetic, internally consistent F2FS image whose
    /// root directory holds the given entries. Each entry references an
    /// inode whose body is provided by the caller.
    fn build_image(
        layout: &ImageLayout,
        root_entries: &[(String, F2fsInode, Option<Vec<u8>>)],
    ) -> MemoryBackend {
        let bs = F2FS_BLKSIZE as u64;
        let dev_size = layout.blocks * bs;
        let mut dev = MemoryBackend::new(dev_size);

        // Region geometry. Block-addressed.
        let cp_blkaddr = layout.sb_blocks; // right after the two SB blocks
        let sit_blkaddr = cp_blkaddr + layout.cp_segs * layout.blocks_per_seg;
        let nat_blkaddr = sit_blkaddr + layout.sit_segs * layout.blocks_per_seg;
        let ssa_blkaddr = nat_blkaddr + layout.nat_segs * layout.blocks_per_seg;
        let main_blkaddr = ssa_blkaddr + layout.ssa_segs * layout.blocks_per_seg;
        let main_segs = (layout.blocks as u32 - main_blkaddr) / layout.blocks_per_seg;

        // ---- superblock ----
        let mut sb_buf = vec![0u8; 0x400];
        sb_buf[0..4].copy_from_slice(&F2FS_MAGIC.to_le_bytes());
        sb_buf[4..6].copy_from_slice(&1u16.to_le_bytes());
        sb_buf[6..8].copy_from_slice(&15u16.to_le_bytes());
        sb_buf[8..12].copy_from_slice(&9u32.to_le_bytes()); // log_sectorsize
        sb_buf[0x10..0x14].copy_from_slice(&12u32.to_le_bytes()); // log_blocksize
        let log_bps = layout.blocks_per_seg.trailing_zeros();
        sb_buf[0x14..0x18].copy_from_slice(&log_bps.to_le_bytes());
        sb_buf[0x18..0x1C].copy_from_slice(&1u32.to_le_bytes()); // segs_per_sec
        sb_buf[0x1C..0x20].copy_from_slice(&1u32.to_le_bytes()); // secs_per_zone
        sb_buf[0x24..0x2C].copy_from_slice(&layout.blocks.to_le_bytes());
        sb_buf[0x2C..0x30].copy_from_slice(&main_segs.to_le_bytes()); // section_count
        sb_buf[0x30..0x34].copy_from_slice(
            &(main_segs + layout.cp_segs + layout.sit_segs + layout.nat_segs + layout.ssa_segs)
                .to_le_bytes(),
        );
        sb_buf[0x34..0x38].copy_from_slice(&layout.cp_segs.to_le_bytes());
        sb_buf[0x38..0x3C].copy_from_slice(&layout.sit_segs.to_le_bytes());
        sb_buf[0x3C..0x40].copy_from_slice(&layout.nat_segs.to_le_bytes());
        sb_buf[0x40..0x44].copy_from_slice(&layout.ssa_segs.to_le_bytes());
        sb_buf[0x44..0x48].copy_from_slice(&main_segs.to_le_bytes());
        sb_buf[0x48..0x4C].copy_from_slice(&0u32.to_le_bytes()); // segment0_blkaddr
        sb_buf[0x4C..0x50].copy_from_slice(&cp_blkaddr.to_le_bytes());
        sb_buf[0x50..0x54].copy_from_slice(&sit_blkaddr.to_le_bytes());
        sb_buf[0x54..0x58].copy_from_slice(&nat_blkaddr.to_le_bytes());
        sb_buf[0x58..0x5C].copy_from_slice(&ssa_blkaddr.to_le_bytes());
        sb_buf[0x5C..0x60].copy_from_slice(&main_blkaddr.to_le_bytes());
        sb_buf[0x60..0x64].copy_from_slice(&3u32.to_le_bytes()); // root_ino
        sb_buf[0x64..0x68].copy_from_slice(&1u32.to_le_bytes()); // node_ino
        sb_buf[0x68..0x6C].copy_from_slice(&2u32.to_le_bytes()); // meta_ino
        // volume_name = "test" at 0x7C
        let name = "test".encode_utf16().collect::<Vec<u16>>();
        for (i, c) in name.iter().enumerate() {
            sb_buf[0x7C + i * 2..0x7C + i * 2 + 2].copy_from_slice(&c.to_le_bytes());
        }
        dev.write_at(SB_OFFSET_PRIMARY, &sb_buf).unwrap();
        dev.write_at(SB_OFFSET_BACKUP, &sb_buf).unwrap();

        // ---- main-area allocator ----
        let mut next_main_blk = main_blkaddr;
        let mut alloc = || {
            let b = next_main_blk;
            next_main_blk += 1;
            assert!(next_main_blk < layout.blocks as u32);
            b
        };

        // Allocate physical blocks for every entry's inode and body.
        // Build the root dentry block first so we can compute its physical block.
        let mut nat_entries: Vec<NatJournalEntry> = Vec::new();
        let root_ino = 3u32;

        // Pre-allocate inode blocks (one per inode) + body blocks.
        let mut inode_blocks: Vec<(u32, F2fsInode, Option<Vec<u8>>, u32)> = Vec::new();
        // (nid, inode-meta, optional body bytes, inode physical block)
        // First the children:
        let mut child_entries_for_dir: Vec<RawDentry> = Vec::new();
        // Self-entries "." and ".." (only when there's space; not strictly required by our reader)
        for (child_nid, (name, ino, body)) in (100u32..).zip(root_entries.iter()) {
            let phys_inode = alloc();
            let ft = if ino.mode & S_IFMT == S_IFDIR {
                F2FS_FT_DIR
            } else {
                F2FS_FT_REG_FILE
            };
            child_entries_for_dir.push(RawDentry {
                hash: 0,
                ino: child_nid,
                file_type: ft,
                name: name.as_bytes().to_vec(),
            });
            inode_blocks.push((child_nid, ino.clone(), body.clone(), phys_inode));
        }
        // Now allocate the root inode + (optionally) the root dentry block.
        let root_phys_inode = alloc();
        let mut root_inode = F2fsInode {
            mode: S_IFDIR | 0o755,
            size: 0,
            uid: 0,
            gid: 0,
            links: 2,
            atime: 0,
            ctime: 0,
            mtime: 0,
            blocks: 1,
            generation: 0,
            flags: 0,
            inline_flags: 0,
            i_addr: [0; super::constants::ADDRS_PER_INODE],
            i_nid: [0; super::constants::NIDS_PER_INODE],
        };
        let want_inline_root = child_entries_for_dir.iter().all(|e| e.name.len() <= 8)
            && child_entries_for_dir.len() <= 4;
        if want_inline_root {
            // Pack into the inline area.
            root_inode.inline_flags |= F2FS_INLINE_DENTRY;
            root_inode.size = F2FS_BLKSIZE as u64; // doesn't matter, list path doesn't use size for inline
        } else {
            let root_dir_blk = alloc();
            let buf = encode_dentry_block(&child_entries_for_dir);
            dev.write_at(root_dir_blk as u64 * bs, &buf).unwrap();
            root_inode.size = F2FS_BLKSIZE as u64;
            root_inode.i_addr[0] = root_dir_blk;
        }

        // Now write each child's body (data blocks) + inode block.
        for (nid, mut ino, body, phys_inode) in inode_blocks {
            if let Some(bytes) = &body {
                if !ino.is_inline_data() && !bytes.is_empty() {
                    let n_blocks = bytes.len().div_ceil(F2FS_BLKSIZE);
                    for i in 0..n_blocks {
                        let phys = alloc();
                        let start = i * F2FS_BLKSIZE;
                        let end = (start + F2FS_BLKSIZE).min(bytes.len());
                        let mut blk = vec![0u8; F2FS_BLKSIZE];
                        blk[..end - start].copy_from_slice(&bytes[start..end]);
                        dev.write_at(phys as u64 * bs, &blk).unwrap();
                        if i < super::constants::ADDRS_PER_INODE {
                            ino.i_addr[i] = phys;
                        }
                    }
                }
            }
            let blk = encode_inode_block(&ino);
            dev.write_at(phys_inode as u64 * bs, &blk).unwrap();
            nat_entries.push(NatJournalEntry {
                nid,
                ino: nid,
                block_addr: phys_inode,
                version: 0,
            });
        }

        // Write inline-dentry payload INTO root inode block (after encode) if applicable.
        if root_inode.inline_flags & F2FS_INLINE_DENTRY != 0 {
            // Encode the inode block first, then overlay the inline dentry region
            // at I_ADDR_OFFSET + 4 (i_addr[DEF_INLINE_RESERVED_SIZE]).
            // Layout: bitmap | reserved | dentries | names.
            let mut blk = encode_inode_block(&root_inode);
            let payload = encode_inline_dentries_payload(&child_entries_for_dir);
            let off = super::inode::I_ADDR_OFFSET + 4;
            let n = payload.len().min(blk.len() - off - 8); // leave footer alone
            blk[off..off + n].copy_from_slice(&payload[..n]);
            // Re-stamp the CRC32 footer.
            let crc = crate::fs::f2fs::constants::f2fs_crc32(
                &blk[..super::constants::F2FS_BLK_CSUM_OFFSET],
            );
            blk[super::constants::F2FS_BLK_CSUM_OFFSET..super::constants::F2FS_BLK_CSUM_OFFSET + 4]
                .copy_from_slice(&crc.to_le_bytes());
            dev.write_at(root_phys_inode as u64 * bs, &blk).unwrap();
        } else {
            let blk = encode_inode_block(&root_inode);
            dev.write_at(root_phys_inode as u64 * bs, &blk).unwrap();
        }
        nat_entries.push(NatJournalEntry {
            nid: root_ino,
            ino: root_ino,
            block_addr: root_phys_inode,
            version: 0,
        });

        // ---- checkpoint pack #0 ----
        let cp = Checkpoint {
            version: 1,
            user_block_count: layout.blocks,
            valid_block_count: (next_main_blk - main_blkaddr) as u64,
            rsvd_segment_count: 0,
            overprov_segment_count: 0,
            flags: 0,
            cp_pack_start_sum: 1, // summary right after head
            cp_pack_total_block_count: 2,
            cp_payload: 0,
            head_blkaddr: cp_blkaddr,
            nat_ver_bitmap_bytesize: 64,
            sit_ver_bitmap_bytesize: 64,
            cur_nat_pack: 0,
            cur_sit_pack: 0,
            nat_journal: Vec::new(),
            cur_node_segno: [0, 1, 2],
            cur_node_blkoff: [0, 0, 0],
            cur_data_segno: [3, 4, 5],
            cur_data_blkoff: [0, 0, 0],
            free_segment_count: 0,
            valid_node_count: 0,
            valid_inode_count: 0,
            next_free_nid: 0,
        };
        let cp_head = encode_cp_head(&cp);
        dev.write_at(cp_blkaddr as u64 * bs, &cp_head).unwrap();
        let cp_sum = encode_nat_journal_block(&nat_entries);
        dev.write_at((cp_blkaddr as u64 + 1) * bs, &cp_sum).unwrap();

        dev
    }

    #[test]
    fn open_and_list_root_with_block_dentries() {
        // A few children with long names, forcing a real dentry block.
        let layout = ImageLayout::default_tiny();
        let mk_regular = |size: u64, payload: &[u8]| -> (F2fsInode, Option<Vec<u8>>) {
            let mut ino = F2fsInode {
                mode: S_IFREG | 0o644,
                size,
                uid: 0,
                gid: 0,
                links: 1,
                atime: 0,
                ctime: 0,
                mtime: 0,
                blocks: 1,
                generation: 0,
                flags: 0,
                inline_flags: 0,
                i_addr: [0; super::constants::ADDRS_PER_INODE],
                i_nid: [0; super::constants::NIDS_PER_INODE],
            };
            if size as usize <= 3000 && !payload.is_empty() {
                ino.inline_flags |= F2FS_INLINE_DATA;
            }
            (ino, Some(payload.to_vec()))
        };

        let (a_ino, a_body) = mk_regular(11, b"hello world");
        let (b_ino, b_body) = mk_regular(0, b"");
        let entries = vec![
            ("longish_filename_a.txt".to_string(), a_ino, a_body),
            ("b".to_string(), b_ino, b_body),
        ];
        let mut dev = build_image(&layout, &entries);
        let mut f = F2fs::open(&mut dev).unwrap();
        assert_eq!(f.volume_name(), "test");
        let list = f.list_path(&mut dev, "/").unwrap();
        let names: Vec<_> = list.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"longish_filename_a.txt"));
        assert!(names.contains(&"b"));
        let a = list
            .iter()
            .find(|e| e.name == "longish_filename_a.txt")
            .unwrap();
        assert_eq!(a.kind, EntryKind::Regular);
    }

    #[test]
    fn inline_data_round_trips() {
        let layout = ImageLayout::default_tiny();
        let payload = b"hello inline F2FS!";
        let mut ino = F2fsInode {
            mode: S_IFREG | 0o644,
            size: payload.len() as u64,
            uid: 0,
            gid: 0,
            links: 1,
            atime: 0,
            ctime: 0,
            mtime: 0,
            blocks: 0,
            generation: 0,
            flags: 0,
            inline_flags: F2FS_INLINE_DATA,
            i_addr: [0; super::constants::ADDRS_PER_INODE],
            i_nid: [0; super::constants::NIDS_PER_INODE],
        };
        // Write the literal bytes into the inline payload region of i_addr.
        // F2FS lays out inline data starting at `i_addr[DEF_INLINE_RESERVED_SIZE]`
        // (= i_addr[1]); slot 0 is reserved (matches `inline_data_addr` in
        // fsck.f2fs).
        let bytes_as_words = payload;
        for (i, b) in bytes_as_words.iter().enumerate() {
            let slot = (i / 4) + 1;
            let off = i % 4;
            let mut bs4 = ino.i_addr[slot].to_le_bytes();
            bs4[off] = *b;
            ino.i_addr[slot] = u32::from_le_bytes(bs4);
        }

        let entries = vec![("hi.txt".to_string(), ino, Some(payload.to_vec()))];
        let mut dev = build_image(&layout, &entries);
        let mut f = F2fs::open(&mut dev).unwrap();
        let mut r = f.open_file_reader(&mut dev, "/hi.txt").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, payload);
    }

    #[test]
    fn streaming_read_walks_direct_pointers() {
        // Build a file > 1 block so we exercise the i_addr path.
        let layout = ImageLayout::default_tiny();
        let block_size = F2FS_BLKSIZE;
        let payload: Vec<u8> = (0..(block_size * 3))
            .map(|i| (i as u8).wrapping_mul(31))
            .collect();
        let ino = F2fsInode {
            mode: S_IFREG | 0o644,
            size: payload.len() as u64,
            uid: 0,
            gid: 0,
            links: 1,
            atime: 0,
            ctime: 0,
            mtime: 0,
            blocks: 3,
            generation: 0,
            flags: 0,
            inline_flags: 0,
            i_addr: [0; super::constants::ADDRS_PER_INODE],
            i_nid: [0; super::constants::NIDS_PER_INODE],
        };
        let entries = vec![("big.bin".to_string(), ino, Some(payload.clone()))];
        let mut dev = build_image(&layout, &entries);
        let mut f = F2fs::open(&mut dev).unwrap();
        let mut r = f.open_file_reader(&mut dev, "/big.bin").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out.len(), payload.len());
        assert_eq!(out, payload);
    }

    #[test]
    fn rejects_invalid_checkpoint_crc() {
        let layout = ImageLayout::default_tiny();
        let entries: Vec<(String, F2fsInode, Option<Vec<u8>>)> = vec![];
        let mut dev = build_image(&layout, &entries);
        // Corrupt the CP head CRC by flipping the first byte.
        let cp_blk = 2u64; // matches build_image layout (sb_blocks=2)
        let mut byte = [0u8; 1];
        dev.read_at(cp_blk * F2FS_BLKSIZE as u64, &mut byte)
            .unwrap();
        byte[0] ^= 0xFF;
        dev.write_at(cp_blk * F2FS_BLKSIZE as u64, &byte).unwrap();
        // Re-stamp the second CP pack (segment-1) the same way.
        dev.read_at(
            (cp_blk + layout.blocks_per_seg as u64) * F2FS_BLKSIZE as u64,
            &mut byte,
        )
        .unwrap();
        byte[0] ^= 0xFF;
        dev.write_at(
            (cp_blk + layout.blocks_per_seg as u64) * F2FS_BLKSIZE as u64,
            &byte,
        )
        .unwrap();
        // Now both packs have the wrong CRC.
        let err = F2fs::open(&mut dev).err().expect("should fail");
        assert!(matches!(err, crate::Error::InvalidImage(_)));
    }

    #[test]
    fn picks_higher_version_cp_when_both_valid() {
        let layout = ImageLayout::default_tiny();
        let entries: Vec<(String, F2fsInode, Option<Vec<u8>>)> = vec![];
        let mut dev = build_image(&layout, &entries);
        // Build a second CP head with version=2 and write it into the
        // second pack location (cp_blkaddr + blocks_per_seg).
        let cp_blk = 2u32;
        let cp2 = Checkpoint {
            version: 99,
            user_block_count: 0,
            valid_block_count: 0,
            rsvd_segment_count: 0,
            overprov_segment_count: 0,
            flags: 0,
            cp_pack_start_sum: 1,
            cp_pack_total_block_count: 2,
            cp_payload: 0,
            head_blkaddr: cp_blk + layout.blocks_per_seg,
            nat_ver_bitmap_bytesize: 0,
            sit_ver_bitmap_bytesize: 0,
            cur_nat_pack: 1,
            cur_sit_pack: 1,
            nat_journal: Vec::new(),
            cur_node_segno: [0, 1, 2],
            cur_node_blkoff: [0, 0, 0],
            cur_data_segno: [3, 4, 5],
            cur_data_blkoff: [0, 0, 0],
            free_segment_count: 0,
            valid_node_count: 0,
            valid_inode_count: 0,
            next_free_nid: 0,
        };
        let buf = encode_cp_head(&cp2);
        let addr = (cp_blk + layout.blocks_per_seg) as u64 * F2FS_BLKSIZE as u64;
        dev.write_at(addr, &buf).unwrap();
        // Also write a (small / empty) NAT journal so the loader has something to read.
        let sum = encode_nat_journal_block(&[]);
        dev.write_at(addr + F2FS_BLKSIZE as u64, &sum).unwrap();
        let f = F2fs::open(&mut dev).unwrap();
        assert_eq!(f.checkpoint().version, 99);
        assert_eq!(f.checkpoint().cur_nat_pack, 1);
    }

    #[test]
    fn probe_detects_primary_copy() {
        let mut dev = MemoryBackend::new(64 * 1024);
        // Just enough to set magic at primary.
        dev.write_at(SB_OFFSET_PRIMARY, &F2FS_MAGIC.to_le_bytes())
            .unwrap();
        assert!(probe(&mut dev).unwrap());
    }

    #[test]
    fn probe_detects_backup_copy() {
        let mut dev = MemoryBackend::new(64 * 1024);
        dev.write_at(SB_OFFSET_BACKUP, &F2FS_MAGIC.to_le_bytes())
            .unwrap();
        assert!(probe(&mut dev).unwrap());
    }

    #[test]
    fn open_reports_geometry() {
        let layout = ImageLayout::default_tiny();
        let entries: Vec<(String, F2fsInode, Option<Vec<u8>>)> = vec![];
        let mut dev = build_image(&layout, &entries);
        let f = F2fs::open(&mut dev).unwrap();
        assert_eq!(f.block_size(), 4096);
        assert_eq!(f.volume_name(), "test");
        assert_eq!(f.total_bytes(), layout.blocks * 4096);
    }

    /// Silence the warning that NR_DENTRY_IN_BLOCK is unused in tests.
    #[test]
    fn nr_dentry_in_block_is_214() {
        assert_eq!(NR_DENTRY_IN_BLOCK, 214);
    }

    // Silence unused-helper warning when `encode_nat_entry` is not used
    // in the tiny path (we go through the journal).
    #[test]
    fn _exercise_encode_nat_entry() {
        let mut page = vec![0u8; F2FS_BLKSIZE];
        encode_nat_entry(&mut page, 0, 1, 3, 100);
        assert_eq!(page[0], 1);
    }

    #[test]
    fn inline_dentry_path_lists_short_names() {
        let layout = ImageLayout::default_tiny();
        let ino = F2fsInode {
            mode: S_IFREG | 0o644,
            size: 0,
            uid: 0,
            gid: 0,
            links: 1,
            atime: 0,
            ctime: 0,
            mtime: 0,
            blocks: 0,
            generation: 0,
            flags: 0,
            inline_flags: 0,
            i_addr: [0; ADDRS_PER_INODE],
            i_nid: [0; super::constants::NIDS_PER_INODE],
        };
        let entries = vec![
            ("a".to_string(), ino.clone(), None),
            ("bb".to_string(), ino.clone(), None),
        ];
        let mut dev = build_image(&layout, &entries);
        let mut f = F2fs::open(&mut dev).unwrap();
        let list = f.list_path(&mut dev, "/").unwrap();
        let names: Vec<_> = list.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"bb"));
        for e in &list {
            assert_eq!(e.kind, EntryKind::Regular);
        }
    }

    #[test]
    fn open_file_reader_rejects_directory() {
        let layout = ImageLayout::default_tiny();
        let entries: Vec<(String, F2fsInode, Option<Vec<u8>>)> = vec![];
        let mut dev = build_image(&layout, &entries);
        let mut f = F2fs::open(&mut dev).unwrap();
        let err = f.open_file_reader(&mut dev, "/").err().unwrap();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    #[test]
    fn list_path_missing_returns_error() {
        let layout = ImageLayout::default_tiny();
        let entries: Vec<(String, F2fsInode, Option<Vec<u8>>)> = vec![];
        let mut dev = build_image(&layout, &entries);
        let mut f = F2fs::open(&mut dev).unwrap();
        let err = f.list_path(&mut dev, "/nope").err().unwrap();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    /// Exercise a file extending past the in-inode 923-pointer region
    /// into a direct-node block referenced by `i_nid[0]`.
    #[test]
    fn streaming_read_walks_direct_node_indirection() {
        let bs = F2FS_BLKSIZE as u64;
        let total_blocks: u64 = 2048;
        let mut dev = MemoryBackend::new(total_blocks * bs);

        let cp_blkaddr: u32 = 2;
        let blocks_per_seg: u32 = 4;
        let sit_blkaddr = cp_blkaddr + 2 * blocks_per_seg;
        let nat_blkaddr = sit_blkaddr + blocks_per_seg;
        let ssa_blkaddr = nat_blkaddr + 2 * blocks_per_seg;
        let main_blkaddr = ssa_blkaddr + blocks_per_seg;

        // ---- superblock ----
        let mut sb_buf = vec![0u8; 0x400];
        sb_buf[0..4].copy_from_slice(&F2FS_MAGIC.to_le_bytes());
        sb_buf[4..6].copy_from_slice(&1u16.to_le_bytes());
        sb_buf[8..12].copy_from_slice(&9u32.to_le_bytes());
        sb_buf[0x10..0x14].copy_from_slice(&12u32.to_le_bytes());
        let log_bps = blocks_per_seg.trailing_zeros();
        sb_buf[0x14..0x18].copy_from_slice(&log_bps.to_le_bytes());
        sb_buf[0x18..0x1C].copy_from_slice(&1u32.to_le_bytes());
        sb_buf[0x1C..0x20].copy_from_slice(&1u32.to_le_bytes());
        sb_buf[0x24..0x2C].copy_from_slice(&total_blocks.to_le_bytes());
        sb_buf[0x30..0x34].copy_from_slice(&((total_blocks as u32) / blocks_per_seg).to_le_bytes());
        sb_buf[0x34..0x38].copy_from_slice(&2u32.to_le_bytes());
        sb_buf[0x38..0x3C].copy_from_slice(&1u32.to_le_bytes());
        sb_buf[0x3C..0x40].copy_from_slice(&2u32.to_le_bytes());
        sb_buf[0x40..0x44].copy_from_slice(&1u32.to_le_bytes());
        sb_buf[0x44..0x48].copy_from_slice(
            &((total_blocks as u32 - main_blkaddr) / blocks_per_seg).to_le_bytes(),
        );
        sb_buf[0x4C..0x50].copy_from_slice(&cp_blkaddr.to_le_bytes());
        sb_buf[0x50..0x54].copy_from_slice(&sit_blkaddr.to_le_bytes());
        sb_buf[0x54..0x58].copy_from_slice(&nat_blkaddr.to_le_bytes());
        sb_buf[0x58..0x5C].copy_from_slice(&ssa_blkaddr.to_le_bytes());
        sb_buf[0x5C..0x60].copy_from_slice(&main_blkaddr.to_le_bytes());
        sb_buf[0x60..0x64].copy_from_slice(&3u32.to_le_bytes());
        sb_buf[0x64..0x68].copy_from_slice(&1u32.to_le_bytes());
        sb_buf[0x68..0x6C].copy_from_slice(&2u32.to_le_bytes());
        dev.write_at(SB_OFFSET_PRIMARY, &sb_buf).unwrap();
        dev.write_at(SB_OFFSET_BACKUP, &sb_buf).unwrap();

        let mut next = main_blkaddr;
        let mut alloc = || {
            let b = next;
            next += 1;
            b
        };

        // File: ADDRS_PER_INODE + 3 blocks of pseudo-random data.
        let n_blocks = ADDRS_PER_INODE + 3;
        let payload: Vec<u8> = (0..(n_blocks * F2FS_BLKSIZE))
            .map(|i| (i as u8).wrapping_mul(7))
            .collect();
        let mut data_blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            data_blocks.push(alloc());
        }
        for (i, &b) in data_blocks.iter().enumerate() {
            let s = i * F2FS_BLKSIZE;
            let e = ((i + 1) * F2FS_BLKSIZE).min(payload.len());
            let mut blk = vec![0u8; F2FS_BLKSIZE];
            blk[..e - s].copy_from_slice(&payload[s..e]);
            dev.write_at(b as u64 * bs, &blk).unwrap();
        }
        // Direct node block.
        let mut dnode_ptrs = vec![0u32; ADDRS_PER_BLOCK];
        dnode_ptrs[0] = data_blocks[ADDRS_PER_INODE];
        dnode_ptrs[1] = data_blocks[ADDRS_PER_INODE + 1];
        dnode_ptrs[2] = data_blocks[ADDRS_PER_INODE + 2];
        let dnode_blk = alloc();
        dev.write_at(dnode_blk as u64 * bs, &encode_direct_node(&dnode_ptrs))
            .unwrap();

        // File inode.
        let mut file_inode = F2fsInode {
            mode: S_IFREG | 0o644,
            size: (n_blocks * F2FS_BLKSIZE) as u64,
            uid: 0,
            gid: 0,
            links: 1,
            atime: 0,
            ctime: 0,
            mtime: 0,
            blocks: n_blocks as u64,
            generation: 0,
            flags: 0,
            inline_flags: 0,
            i_addr: [0; ADDRS_PER_INODE],
            i_nid: [0; super::constants::NIDS_PER_INODE],
        };
        file_inode.i_addr[..ADDRS_PER_INODE].copy_from_slice(&data_blocks[..ADDRS_PER_INODE]);
        file_inode.i_nid[super::constants::NID_DIRECT_1] = 200;
        let file_inode_blk = alloc();
        dev.write_at(file_inode_blk as u64 * bs, &encode_inode_block(&file_inode))
            .unwrap();

        // Root inode with inline-dentry holding "big.bin".
        let root_inode = F2fsInode {
            mode: S_IFDIR | 0o755,
            size: F2FS_BLKSIZE as u64,
            uid: 0,
            gid: 0,
            links: 2,
            atime: 0,
            ctime: 0,
            mtime: 0,
            blocks: 0,
            generation: 0,
            flags: 0,
            inline_flags: F2FS_INLINE_DENTRY,
            i_addr: [0; ADDRS_PER_INODE],
            i_nid: [0; super::constants::NIDS_PER_INODE],
        };
        let root_inode_blk = alloc();
        let root_entries = vec![RawDentry {
            hash: 0,
            ino: 100,
            file_type: F2FS_FT_REG_FILE,
            name: b"big.bin".to_vec(),
        }];
        let mut blk = encode_inode_block(&root_inode);
        let payload_buf = encode_inline_dentries_payload(&root_entries);
        let off = super::inode::I_ADDR_OFFSET + 4;
        let n = payload_buf.len().min(blk.len() - off - 8);
        blk[off..off + n].copy_from_slice(&payload_buf[..n]);
        let crc =
            crate::fs::f2fs::constants::f2fs_crc32(&blk[..super::constants::F2FS_BLK_CSUM_OFFSET]);
        blk[super::constants::F2FS_BLK_CSUM_OFFSET..super::constants::F2FS_BLK_CSUM_OFFSET + 4]
            .copy_from_slice(&crc.to_le_bytes());
        dev.write_at(root_inode_blk as u64 * bs, &blk).unwrap();

        // Checkpoint + NAT journal (3 entries: root inode, file inode, direct node).
        let nat_entries = vec![
            NatJournalEntry {
                nid: 3,
                ino: 3,
                block_addr: root_inode_blk,
                version: 0,
            },
            NatJournalEntry {
                nid: 100,
                ino: 100,
                block_addr: file_inode_blk,
                version: 0,
            },
            NatJournalEntry {
                nid: 200,
                ino: 100,
                block_addr: dnode_blk,
                version: 0,
            },
        ];
        let cp = Checkpoint {
            version: 1,
            user_block_count: total_blocks,
            valid_block_count: (next - main_blkaddr) as u64,
            rsvd_segment_count: 0,
            overprov_segment_count: 0,
            flags: 0,
            cp_pack_start_sum: 1,
            cp_pack_total_block_count: 2,
            cp_payload: 0,
            head_blkaddr: cp_blkaddr,
            nat_ver_bitmap_bytesize: 64,
            sit_ver_bitmap_bytesize: 64,
            cur_nat_pack: 0,
            cur_sit_pack: 0,
            nat_journal: Vec::new(),
            cur_node_segno: [0, 1, 2],
            cur_node_blkoff: [0, 0, 0],
            cur_data_segno: [3, 4, 5],
            cur_data_blkoff: [0, 0, 0],
            free_segment_count: 0,
            valid_node_count: 0,
            valid_inode_count: 0,
            next_free_nid: 0,
        };
        dev.write_at(cp_blkaddr as u64 * bs, &encode_cp_head(&cp))
            .unwrap();
        dev.write_at(
            (cp_blkaddr as u64 + 1) * bs,
            &encode_nat_journal_block(&nat_entries),
        )
        .unwrap();

        // Walk the live API.
        let mut f = F2fs::open(&mut dev).unwrap();
        let list = f.list_path(&mut dev, "/").unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "big.bin");

        let mut r = f.open_file_reader(&mut dev, "/big.bin").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out.len(), payload.len());
        assert_eq!(out, payload);
    }

    /// `encode_indirect_node` is exercised by the indirect-node smoke
    /// test: build a block that decodes back to the same nids.
    #[test]
    fn indirect_node_roundtrip() {
        let nids = vec![10u32, 11, 12, 13];
        let blk = encode_indirect_node(&nids);
        let got = super::inode::decode_indirect_node(&blk).unwrap();
        assert_eq!(got[..4], nids[..]);
        assert_eq!(got.len(), super::constants::NIDS_PER_BLOCK);
    }

    // -------- Writer round-trip tests --------

    /// Build a small device, format it, and re-open. The empty root must
    /// list cleanly.
    #[test]
    fn format_creates_readable_empty_root() {
        // 1 MiB device (256 × 4 KiB blocks). Plenty of room for a tiny
        // log2(blocks_per_seg)=2 layout (4 blocks/segment).
        let mut dev = MemoryBackend::new(1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            volume_label: "fstool".into(),
            ..super::FormatOpts::default()
        };
        let _fs = F2fs::format(&mut dev, &opts).unwrap();
        let mut fs = F2fs::open(&mut dev).unwrap();
        let list = fs.list_path(&mut dev, "/").unwrap();
        assert!(list.is_empty());
        assert_eq!(fs.volume_name(), "fstool");
    }

    /// Create a tiny file (inline path) and read it back through the
    /// read driver.
    #[test]
    fn format_then_create_file_inline() {
        let mut dev = MemoryBackend::new(1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let mut fs = F2fs::format(&mut dev, &opts).unwrap();
        let payload = b"hello fresh f2fs writer";
        let src = crate::fs::FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(payload.to_vec())),
            len: payload.len() as u64,
        };
        fs.create_file(
            &mut dev,
            std::path::Path::new("/hello.txt"),
            src,
            crate::fs::FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        // Reopen and confirm.
        let mut fs2 = F2fs::open(&mut dev).unwrap();
        let list = fs2.list_path(&mut dev, "/").unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "hello.txt");
        let mut r = fs2.open_file_reader(&mut dev, "/hello.txt").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, payload);
    }

    /// Create a file big enough to spill out of the inline area.
    #[test]
    fn format_then_create_file_block() {
        let mut dev = MemoryBackend::new(2 * 1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let mut fs = F2fs::format(&mut dev, &opts).unwrap();
        // Payload size: 3 blocks worth of data (> MAX_INLINE_DATA).
        let payload: Vec<u8> = (0..(F2FS_BLKSIZE * 3))
            .map(|i| (i as u8).wrapping_mul(13))
            .collect();
        let src = crate::fs::FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(payload.clone())),
            len: payload.len() as u64,
        };
        fs.create_file(
            &mut dev,
            std::path::Path::new("/big.bin"),
            src,
            crate::fs::FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        let mut fs2 = F2fs::open(&mut dev).unwrap();
        let mut r = fs2.open_file_reader(&mut dev, "/big.bin").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out.len(), payload.len());
        assert_eq!(out, payload);
    }

    /// Directory creation: mkdir, then file inside, then list both.
    #[test]
    fn format_then_create_dir_and_child() {
        let mut dev = MemoryBackend::new(1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let mut fs = F2fs::format(&mut dev, &opts).unwrap();
        fs.create_dir(
            &mut dev,
            std::path::Path::new("/etc"),
            crate::fs::FileMeta::with_mode(0o755),
        )
        .unwrap();
        let src = crate::fs::FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(b"x=1".to_vec())),
            len: 3,
        };
        fs.create_file(
            &mut dev,
            std::path::Path::new("/etc/config"),
            src,
            crate::fs::FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        let mut fs2 = F2fs::open(&mut dev).unwrap();
        let root_list = fs2.list_path(&mut dev, "/").unwrap();
        assert!(root_list.iter().any(|e| e.name == "etc"));
        let etc = root_list.iter().find(|e| e.name == "etc").unwrap();
        assert_eq!(etc.kind, EntryKind::Dir);
        let etc_list = fs2.list_path(&mut dev, "/etc").unwrap();
        assert_eq!(etc_list.len(), 1);
        assert_eq!(etc_list[0].name, "config");
        let mut r = fs2.open_file_reader(&mut dev, "/etc/config").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, b"x=1");
    }

    /// Symlink creation + read-back via the dentry's file-type byte.
    #[test]
    fn format_then_create_symlink() {
        let mut dev = MemoryBackend::new(1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let mut fs = F2fs::format(&mut dev, &opts).unwrap();
        fs.create_symlink(
            &mut dev,
            std::path::Path::new("/link"),
            std::path::Path::new("./target"),
            crate::fs::FileMeta::with_mode(0o777),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        let mut fs2 = F2fs::open(&mut dev).unwrap();
        let list = fs2.list_path(&mut dev, "/").unwrap();
        let link = list.iter().find(|e| e.name == "link").unwrap();
        assert_eq!(link.kind, EntryKind::Symlink);
    }

    /// Remove a file and check it's gone after re-open.
    #[test]
    fn format_then_remove_file() {
        let mut dev = MemoryBackend::new(1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let mut fs = F2fs::format(&mut dev, &opts).unwrap();
        let src = crate::fs::FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(b"tmp".to_vec())),
            len: 3,
        };
        fs.create_file(
            &mut dev,
            std::path::Path::new("/scratch"),
            src,
            crate::fs::FileMeta::default(),
        )
        .unwrap();
        fs.remove(&mut dev, std::path::Path::new("/scratch"))
            .unwrap();
        fs.flush(&mut dev).unwrap();

        let mut fs2 = F2fs::open(&mut dev).unwrap();
        let list = fs2.list_path(&mut dev, "/").unwrap();
        assert!(list.is_empty());
    }

    /// Read-only handle rejects writes with a helpful message.
    #[test]
    fn read_only_open_rejects_create() {
        let mut dev = MemoryBackend::new(1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let _ = F2fs::format(&mut dev, &opts).unwrap();
        let mut fs_ro = F2fs::open(&mut dev).unwrap();
        let src = crate::fs::FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(b"!".to_vec())),
            len: 1,
        };
        let err = fs_ro
            .create_file(
                &mut dev,
                std::path::Path::new("/x"),
                src,
                crate::fs::FileMeta::default(),
            )
            .err()
            .unwrap();
        assert!(matches!(err, crate::Error::Unsupported(_)));
    }

    /// Many small files force the directory to spill from inline to a
    /// 4 KiB dentry block. Both formats must read back consistently.
    #[test]
    fn dir_spill_to_block_layout() {
        let mut dev = MemoryBackend::new(4 * 1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let mut fs = F2fs::format(&mut dev, &opts).unwrap();
        // 100 files with 9-byte names → 2 slots per dentry, 200 slots
        // total. That overflows the inline-dentry 182-slot budget and
        // forces a spill to a 4 KiB block-format dentry (214 slots, so
        // 100 × 2 = 200 slots still fits).
        let n_files = 100;
        for i in 0..n_files {
            let name = format!("ab{i:07}"); // 9 bytes (ab + 7 digits)
            let src = crate::fs::FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(vec![b'.'; 1])),
                len: 1,
            };
            fs.create_file(
                &mut dev,
                std::path::Path::new(&format!("/{name}")),
                src,
                crate::fs::FileMeta::default(),
            )
            .unwrap();
        }
        fs.flush(&mut dev).unwrap();
        let mut fs2 = F2fs::open(&mut dev).unwrap();
        let list = fs2.list_path(&mut dev, "/").unwrap();
        assert_eq!(list.len(), n_files);
        for entry in &list {
            assert_eq!(entry.kind, EntryKind::Regular);
        }
    }

    /// A file that overflows the in-inode 923 direct pointers must
    /// allocate at least one direct-node block.
    #[test]
    fn writer_overflows_into_direct_node() {
        use super::constants::ADDRS_PER_INODE;
        let mut dev = MemoryBackend::new(16 * 1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let mut fs = F2fs::format(&mut dev, &opts).unwrap();
        // ADDRS_PER_INODE + 5 blocks of pseudo-random data.
        let n_blocks = ADDRS_PER_INODE + 5;
        let payload: Vec<u8> = (0..(n_blocks * F2FS_BLKSIZE))
            .map(|i| (i as u8).wrapping_mul(17))
            .collect();
        let src = crate::fs::FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(payload.clone())),
            len: payload.len() as u64,
        };
        fs.create_file(
            &mut dev,
            std::path::Path::new("/huge.bin"),
            src,
            crate::fs::FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        let mut fs2 = F2fs::open(&mut dev).unwrap();
        let mut r = fs2.open_file_reader(&mut dev, "/huge.bin").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out.len(), payload.len());
        assert_eq!(out, payload);
    }

    /// Hard link: a second name in the same directory pointing at the same
    /// inode. Both paths read back identical bytes, and `i_links == 2`.
    #[test]
    fn hardlink_two_names_same_inode() {
        let mut dev = MemoryBackend::new(1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let mut fs = F2fs::format(&mut dev, &opts).unwrap();
        let payload = b"hard-linked payload";
        let src = crate::fs::FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(payload.to_vec())),
            len: payload.len() as u64,
        };
        let src_nid = fs
            .create_file(
                &mut dev,
                std::path::Path::new("/a.txt"),
                src,
                crate::fs::FileMeta::default(),
            )
            .unwrap();
        let dst_nid = fs
            .create_hardlink(
                &mut dev,
                std::path::Path::new("/a.txt"),
                std::path::Path::new("/b.txt"),
            )
            .unwrap();
        assert_eq!(src_nid, dst_nid, "hardlinks share their nid");
        fs.flush(&mut dev).unwrap();

        let mut fs2 = F2fs::open(&mut dev).unwrap();
        let list = fs2.list_path(&mut dev, "/").unwrap();
        let names: Vec<_> = list.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"a.txt"));
        assert!(names.contains(&"b.txt"));
        // Both dentries point at the same inode.
        let a = list.iter().find(|e| e.name == "a.txt").unwrap();
        let b = list.iter().find(|e| e.name == "b.txt").unwrap();
        assert_eq!(a.inode, b.inode);

        // Bytes match through either path.
        for name in &["/a.txt", "/b.txt"] {
            let mut r = fs2.open_file_reader(&mut dev, name).unwrap();
            let mut out = Vec::new();
            r.read_to_end(&mut out).unwrap();
            assert_eq!(out, payload);
        }
        // i_links == 2 on the shared inode.
        let (_, ino) = fs2.read_inode(&mut dev, a.inode).unwrap();
        assert_eq!(ino.links, 2);
    }

    /// Hard-linking a directory is forbidden by POSIX; the writer must say
    /// so explicitly.
    #[test]
    fn hardlink_rejects_directory() {
        let mut dev = MemoryBackend::new(1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let mut fs = F2fs::format(&mut dev, &opts).unwrap();
        fs.create_dir(
            &mut dev,
            std::path::Path::new("/d"),
            crate::fs::FileMeta::default(),
        )
        .unwrap();
        let err = fs
            .create_hardlink(
                &mut dev,
                std::path::Path::new("/d"),
                std::path::Path::new("/d2"),
            )
            .err()
            .unwrap();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    /// Multi-block spill: enough children with long names to overflow a
    /// single 4 KiB dentry block. The reader must walk both blocks.
    #[test]
    fn dir_spills_across_multiple_dentry_blocks() {
        let mut dev = MemoryBackend::new(8 * 1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let mut fs = F2fs::format(&mut dev, &opts).unwrap();
        // 300 files, each with a 17-byte name → 3 slots per dentry.
        // 300 × 3 = 900 slots > NR_DENTRY_IN_BLOCK (214) → 5 blocks.
        let n = 300usize;
        for i in 0..n {
            let name = format!("file_with_name_{i:03}"); // 17 bytes
            let src = crate::fs::FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(vec![b'.'; 1])),
                len: 1,
            };
            fs.create_file(
                &mut dev,
                std::path::Path::new(&format!("/{name}")),
                src,
                crate::fs::FileMeta::default(),
            )
            .unwrap();
        }
        fs.flush(&mut dev).unwrap();

        let mut fs2 = F2fs::open(&mut dev).unwrap();
        let list = fs2.list_path(&mut dev, "/").unwrap();
        assert_eq!(list.len(), n);
        // Confirm we actually went multi-block (size > one block).
        let (_, root) = fs2.read_inode(&mut dev, 3).unwrap();
        assert!(
            root.size > F2FS_BLKSIZE as u64,
            "expected multi-block dentry layout, got size={}",
            root.size
        );
        // Every named entry must round-trip.
        let names: std::collections::HashSet<_> = list.iter().map(|e| e.name.clone()).collect();
        for i in 0..n {
            assert!(names.contains(&format!("file_with_name_{i:03}")));
        }
    }

    #[test]
    fn open_file_ro_random_seek_f2fs() {
        use crate::fs::Filesystem;
        use std::io::{Read, Seek, SeekFrom};

        let mut dev = MemoryBackend::new(2 * 1024 * 1024);
        let opts = super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::FormatOpts::default()
        };
        let mut fs = F2fs::format(&mut dev, &opts).unwrap();
        // Multi-block file so the read driver walks the pointer tree.
        let data: Vec<u8> = (0..(F2FS_BLKSIZE * 2 + 137))
            .map(|i| (i as u8).wrapping_mul(7))
            .collect();
        fs.create_file(
            &mut dev,
            std::path::Path::new("/ro.bin"),
            crate::fs::FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(data.clone())),
                len: data.len() as u64,
            },
            crate::fs::FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        // Re-open the on-disk image (no live writer) and exercise the
        // read-only path.
        let mut fs2 = F2fs::open(&mut dev).unwrap();
        let mut h = fs2
            .open_file_ro(&mut dev, std::path::Path::new("/ro.bin"))
            .expect("open_file_ro");
        assert_eq!(h.len(), data.len() as u64);
        assert!(!h.is_empty());

        let off = F2FS_BLKSIZE as u64 + 257;
        h.seek(SeekFrom::Start(off)).unwrap();
        let mut buf = [0u8; 64];
        h.read_exact(&mut buf).unwrap();
        assert_eq!(&buf[..], &data[off as usize..off as usize + 64]);

        h.seek(SeekFrom::Start(11)).unwrap();
        let mut buf2 = [0u8; 32];
        h.read_exact(&mut buf2).unwrap();
        assert_eq!(&buf2[..], &data[11..43]);
    }
}
