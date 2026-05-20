//! F2FS regular-file streaming reader.
//!
//! Walks `i_addr` → direct node → indirect node → double / triple indirect
//! pointer chains to map a logical block index to a physical block, then
//! pulls 4 KiB at a time from the device. Honours the inline-data flag.
//!
//! Reference: kernel docs §"Index Structure" + FAST '15 Figure 2.
//!
//! Streaming invariant: at most one 4 KiB block plus one node block plus
//! one indirect node block resident at any time (under 16 KiB heap).

use std::io::{Read, Seek, SeekFrom};

use super::constants::{
    ADDRS_PER_BLOCK, ADDRS_PER_INODE, F2FS_BLKSIZE, NEW_ADDR, NID_DIRECT_1, NID_DIRECT_2,
    NID_INDIRECT_1, NID_INDIRECT_2, NID_TRIPLE_INDIRECT, NIDS_PER_BLOCK, NULL_ADDR,
};
use super::inode::{F2fsInode, decode_direct_node, decode_indirect_node, decode_inode_block};
use super::nat::lookup_node;
use crate::Result;
use crate::block::BlockDevice;

use super::checkpoint::Checkpoint;
use super::superblock::Superblock;

/// Resolve the physical block address for the `idx`-th 4 KiB block of a
/// file. Returns `NULL_ADDR` for an unallocated hole (caller fills zero)
/// or the on-disk block number otherwise.
pub fn logical_to_physical(
    dev: &mut dyn BlockDevice,
    sb: &Superblock,
    cp: &Checkpoint,
    inode: &F2fsInode,
    idx: u64,
) -> Result<u32> {
    // Region 1: direct in-inode pointers.
    if idx < ADDRS_PER_INODE as u64 {
        return Ok(inode.i_addr[idx as usize]);
    }
    let mut rel = idx - ADDRS_PER_INODE as u64;

    // Region 2: two direct node blocks (each 1018 ptrs).
    for nid_idx in [NID_DIRECT_1, NID_DIRECT_2] {
        let span = ADDRS_PER_BLOCK as u64;
        if rel < span {
            return resolve_via_direct_node(dev, sb, cp, inode.i_nid[nid_idx], rel as usize);
        }
        rel -= span;
    }

    // Region 3: two indirect node blocks (each 1018 nids → 1018*1018 ptrs).
    for nid_idx in [NID_INDIRECT_1, NID_INDIRECT_2] {
        let span = (NIDS_PER_BLOCK as u64) * (ADDRS_PER_BLOCK as u64);
        if rel < span {
            let outer = (rel / ADDRS_PER_BLOCK as u64) as usize;
            let inner = (rel % ADDRS_PER_BLOCK as u64) as usize;
            return resolve_via_indirect_node(dev, sb, cp, inode.i_nid[nid_idx], outer, inner);
        }
        rel -= span;
    }

    // Region 4: one triple-indirect node block.
    let span = (NIDS_PER_BLOCK as u64).pow(2) * ADDRS_PER_BLOCK as u64;
    if rel < span {
        let outer = (rel / ((NIDS_PER_BLOCK as u64) * ADDRS_PER_BLOCK as u64)) as usize;
        let mid = ((rel / ADDRS_PER_BLOCK as u64) % NIDS_PER_BLOCK as u64) as usize;
        let inner = (rel % ADDRS_PER_BLOCK as u64) as usize;
        // Top indirect block → 1018 nids of indirect blocks.
        let top_nid = inode.i_nid[NID_TRIPLE_INDIRECT];
        if top_nid == 0 {
            return Ok(NULL_ADDR);
        }
        let top_blk = lookup_node(dev, sb, cp, top_nid)?.block;
        let mut blk = vec![0u8; F2FS_BLKSIZE];
        dev.read_at(top_blk as u64 * sb.block_size() as u64, &mut blk)?;
        let nids = decode_indirect_node(&blk)?;
        let mid_nid = nids[outer];
        if mid_nid == 0 {
            return Ok(NULL_ADDR);
        }
        return resolve_via_indirect_node(dev, sb, cp, mid_nid, mid, inner);
    }

    Err(crate::Error::InvalidImage(format!(
        "f2fs: logical block {idx} exceeds maximum file size"
    )))
}

fn resolve_via_direct_node(
    dev: &mut dyn BlockDevice,
    sb: &Superblock,
    cp: &Checkpoint,
    nid: u32,
    inner: usize,
) -> Result<u32> {
    if nid == 0 {
        return Ok(NULL_ADDR);
    }
    let blk = lookup_node(dev, sb, cp, nid)?.block;
    let mut buf = vec![0u8; F2FS_BLKSIZE];
    dev.read_at(blk as u64 * sb.block_size() as u64, &mut buf)?;
    let ptrs = decode_direct_node(&buf)?;
    Ok(ptrs[inner])
}

fn resolve_via_indirect_node(
    dev: &mut dyn BlockDevice,
    sb: &Superblock,
    cp: &Checkpoint,
    nid: u32,
    outer: usize,
    inner: usize,
) -> Result<u32> {
    if nid == 0 {
        return Ok(NULL_ADDR);
    }
    let blk = lookup_node(dev, sb, cp, nid)?.block;
    let mut buf = vec![0u8; F2FS_BLKSIZE];
    dev.read_at(blk as u64 * sb.block_size() as u64, &mut buf)?;
    let nids = decode_indirect_node(&buf)?;
    resolve_via_direct_node(dev, sb, cp, nids[outer], inner)
}

/// Streaming `Read`er over a regular file's body. Pulls one 4 KiB block
/// at a time; the underlying buffer never exceeds 4 KiB.
pub struct FileReader<'a> {
    pub(crate) dev: &'a mut dyn BlockDevice,
    pub(crate) sb: Superblock,
    pub(crate) cp: Checkpoint,
    pub(crate) inode: F2fsInode,
    pub(crate) inode_block: Vec<u8>,
    pub(crate) pos: u64,
    pub(crate) block_buf: Vec<u8>,
    pub(crate) cached_block: u64,
}

impl<'a> FileReader<'a> {
    pub(crate) fn new(
        dev: &'a mut dyn BlockDevice,
        sb: Superblock,
        cp: Checkpoint,
        inode_block: Vec<u8>,
    ) -> Result<Self> {
        let inode = decode_inode_block(&inode_block)?;
        Ok(Self {
            dev,
            sb,
            cp,
            inode,
            inode_block,
            pos: 0,
            block_buf: vec![0u8; F2FS_BLKSIZE],
            cached_block: u64::MAX,
        })
    }

    fn fill_logical_block(&mut self, idx: u64) -> std::io::Result<()> {
        if self.cached_block == idx {
            return Ok(());
        }
        if self.inode.is_inline_data() {
            // Whole "file" lives in the inode's literal area; we only
            // ever serve block index 0, and the block_buf gets a copy of
            // the inline payload padded with zeros.
            self.block_buf.fill(0);
            let payload = self.inode.inline_payload(&self.inode_block);
            let n = payload.len().min(F2FS_BLKSIZE);
            self.block_buf[..n].copy_from_slice(&payload[..n]);
            self.cached_block = idx;
            return Ok(());
        }
        let phys = logical_to_physical(self.dev, &self.sb, &self.cp, &self.inode, idx)
            .map_err(std::io::Error::other)?;
        if phys == NULL_ADDR || phys == NEW_ADDR {
            self.block_buf.fill(0);
        } else {
            self.dev
                .read_at(
                    phys as u64 * self.sb.block_size() as u64,
                    &mut self.block_buf,
                )
                .map_err(std::io::Error::other)?;
        }
        self.cached_block = idx;
        Ok(())
    }
}

impl<'a> Read for FileReader<'a> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let total = self.inode.size;
        if self.pos >= total || out.is_empty() {
            return Ok(0);
        }
        let bs = F2FS_BLKSIZE as u64;
        let idx = self.pos / bs;
        let off = (self.pos % bs) as usize;
        self.fill_logical_block(idx)?;
        let remaining_in_block = F2FS_BLKSIZE - off;
        let remaining_in_file = (total - self.pos) as usize;
        let n = out.len().min(remaining_in_block).min(remaining_in_file);
        out[..n].copy_from_slice(&self.block_buf[off..off + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl<'a> Seek for FileReader<'a> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let total = self.inode.size as i128;
        let new = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
            SeekFrom::End(d) => total + d as i128,
        };
        if new < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "f2fs: seek to negative offset",
            ));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

impl<'a> FileReader<'a> {
    /// Logical file length in bytes (for [`crate::fs::FileReadHandle`]).
    pub fn file_len(&self) -> u64 {
        self.inode.size
    }
}

impl<'a> crate::fs::FileReadHandle for FileReader<'a> {
    fn len(&self) -> u64 {
        self.inode.size
    }
}
