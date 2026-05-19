//! XFS — read-only support.
//!
//! Implements enough of XFS v4/v5 to inspect the volume and stream small
//! files out of it: the superblock, inodes with the v3 (CRC) and v2 cores,
//! shortform directories, and the linear extent list for `EXTENTS`-format
//! files. Block-format directories, leaf/node/B-tree directories, B-tree
//! files, realtime files, encryption, reverse-mapping btrees, and the
//! journal are deliberately out of scope — encountering any of them
//! returns `Error::Unsupported` with a message naming the feature.
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
pub mod inode;
pub mod superblock;

use crate::Result;
use crate::block::BlockDevice;

use self::bmbt::Extent;
use self::dir::ShortformEntry;
use self::inode::{DiFormat, DinodeCore};
use self::superblock::Superblock;

/// In-memory state of an opened XFS volume. Owns the parsed superblock; all
/// other reads go through the passed-in `BlockDevice`.
pub struct Xfs {
    sb: Superblock,
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
        Ok(Self { sb })
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
            let dir_entries = self.read_dir_entries(&cur_buf, &cur_core)?;
            let found = dir_entries.iter().find(|e| e.name == part).ok_or_else(|| {
                crate::Error::InvalidArgument(format!("xfs: no such entry {part:?} under {path:?}"))
            })?;
            cur_ino = found.inumber;
            let (b, c) = self.read_inode(dev, cur_ino)?;
            cur_buf = b;
            cur_core = c;
        }
        Ok((cur_ino, cur_buf, cur_core))
    }

    /// Read a directory's entries given the inode's bytes + core.
    fn read_dir_entries(&self, ino_buf: &[u8], core: &DinodeCore) -> Result<Vec<ShortformEntry>> {
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
                Ok(entries)
            }
            DiFormat::Extents => Err(crate::Error::Unsupported(
                "xfs: block / leaf+ directories not implemented (only shortform)".into(),
            )),
            DiFormat::Btree => Err(crate::Error::Unsupported(
                "xfs: B-tree directories not implemented".into(),
            )),
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
        let entries = self.read_dir_entries(&buf, &core)?;
        Ok(dir::shortform_to_generic(&entries))
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
            DiFormat::Extents => {
                let lit = core.literal_area(&buf, self.sb.inodesize as usize);
                let extents = bmbt::decode_extents(lit, core.nextents)?;
                for e in &extents {
                    if e.unwritten {
                        return Err(crate::Error::Unsupported(
                            "xfs: unwritten extents not supported in read path".into(),
                        ));
                    }
                }
                Ok(XfsFileReader {
                    dev,
                    extents,
                    blocksize: self.sb.blocksize as u64,
                    size: core.size,
                    pos: 0,
                    inline: None,
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
                    size: core.size,
                    pos: 0,
                    inline: Some(data),
                })
            }
            DiFormat::Btree => Err(crate::Error::Unsupported(
                "xfs: B-tree extent files not implemented".into(),
            )),
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
    size: u64,
    /// Logical byte position of the next byte to return.
    pos: u64,
    /// Set when the file was tiny enough to live entirely in the inode's
    /// literal area (`DiFormat::Local`). When present, `extents` is empty
    /// and reads come from this buffer.
    inline: Option<Vec<u8>>,
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
                let phys_byte = (e.startblock + (pos_blk - e.offset)) * self.blocksize + pos_off;
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
}
