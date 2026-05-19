//! F2FS inode / dnode block parsing.
//!
//! Every node block (inode, direct node, indirect node) is 4 KiB and
//! ends with an 8-byte `node_footer` (we read only `nid` and `ino`).
//! For an *inode* block the first 0xD0 bytes hold metadata, followed by
//! the 923-slot `i_addr` array of u32 data-block pointers and a 5-slot
//! `i_nid` array of node IDs (direct → indirect → triple).
//!
//! When the `i_inline` flag set says `INLINE_DATA` or `INLINE_DENTRY`
//! the literal payload starts at the same offset as `i_addr` (the
//! pointer array is repurposed as inline storage).
//!
//! Reference: FAST '15 paper §2.2 + kernel docs §"Index Structure".

use super::constants::{
    ADDRS_PER_INODE, F2FS_BLKSIZE, F2FS_INLINE_DATA, F2FS_INLINE_DENTRY, NIDS_PER_INODE,
};

#[cfg(test)]
use super::constants::F2FS_BLK_CSUM_OFFSET;

/// The inode-block metadata we care about for read.
#[derive(Debug, Clone)]
pub struct F2fsInode {
    pub mode: u16,
    pub size: u64,
    pub uid: u32,
    pub gid: u32,
    pub links: u32,
    pub atime: u32,
    pub ctime: u32,
    pub mtime: u32,
    pub blocks: u64,
    pub generation: u32,
    pub flags: u32,
    pub inline_flags: u8,
    /// 923 direct data-block pointers (some of them may be reused for
    /// inline data/dentry; see [`Self::is_inline_data`]).
    pub i_addr: [u32; ADDRS_PER_INODE],
    /// 5 node IDs: indices 0 and 1 are direct, 2 and 3 are indirect, 4 is triple-indirect.
    pub i_nid: [u32; NIDS_PER_INODE],
}

impl F2fsInode {
    #[inline]
    pub fn is_inline_data(&self) -> bool {
        self.inline_flags & F2FS_INLINE_DATA != 0
    }

    #[inline]
    pub fn is_inline_dentry(&self) -> bool {
        self.inline_flags & F2FS_INLINE_DENTRY != 0
    }

    /// Raw bytes view of the inline-data / inline-dentry payload region
    /// inside the inode block. The literal data starts where `i_addr`
    /// would have been (offset 0xD0) and runs for `4 * 923 + 4 * 5` =
    /// 3712 bytes, the longest contiguous run inside the inode block
    /// before the node footer.
    pub fn inline_payload<'a>(&self, block: &'a [u8]) -> &'a [u8] {
        let off = I_ADDR_OFFSET;
        let end = (off + ADDRS_PER_INODE * 4 + NIDS_PER_INODE * 4).min(block.len());
        &block[off..end]
    }
}

/// Offset of `i_addr[0]` inside an inode block. The 0xD0 prefix holds:
/// mode, links, size, atime/ctime/mtime, blocks, uid, gid, generation,
/// flags, plus a few extent-cache fields and the inline bitfield.
pub(crate) const I_ADDR_OFFSET: usize = 0xD0;

/// Decode a 4 KiB inode block.
pub fn decode_inode_block(buf: &[u8]) -> crate::Result<F2fsInode> {
    if buf.len() < F2FS_BLKSIZE {
        return Err(crate::Error::InvalidImage(
            "f2fs: short read on inode block".into(),
        ));
    }
    let r16 = |o: usize| u16::from_le_bytes(buf[o..o + 2].try_into().unwrap());
    let r32 = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
    let r64 = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());

    // Inode-block prefix layout (publicly documented F2FS on-disk format):
    //   0x00 u16 i_mode
    //   0x02 u8  i_advise
    //   0x03 u8  i_inline
    //   0x04 u32 i_uid
    //   0x08 u32 i_gid
    //   0x0C u32 i_links
    //   0x10 u64 i_size
    //   0x18 u64 i_blocks
    //   0x20 u64 i_atime
    //   0x28 u64 i_ctime
    //   0x30 u64 i_mtime
    //   0x38 u32 i_atime_nsec
    //   0x3C u32 i_ctime_nsec
    //   0x40 u32 i_mtime_nsec
    //   0x44 u32 i_generation
    //   0x48 u32 i_current_depth
    //   0x4C u32 i_xattr_nid
    //   0x50 u32 i_flags
    //   0x54 u32 i_pino
    //   0x58 u32 i_namelen
    //   ... (i_name, i_dir_level, i_ext, reserved) → ends by I_ADDR_OFFSET
    //   0xD0 u32 i_addr[923]
    //   ...  u32 i_nid[5]   immediately follows i_addr
    //   F2FS_BLKSIZE - 8 .. F2FS_BLKSIZE-4 : node_footer.nid, .ino
    let mode = r16(0x00);
    let inline_flags = buf[0x03];
    let uid = r32(0x04);
    let gid = r32(0x08);
    let links = r32(0x0C);
    let size = r64(0x10);
    let blocks = r64(0x18);
    let atime = r64(0x20) as u32;
    let ctime = r64(0x28) as u32;
    let mtime = r64(0x30) as u32;
    let generation = r32(0x44);
    let flags = r32(0x50);

    let mut i_addr = [0u32; ADDRS_PER_INODE];
    for (i, slot) in i_addr.iter_mut().enumerate() {
        let o = I_ADDR_OFFSET + i * 4;
        *slot = r32(o);
    }
    let nid_off = I_ADDR_OFFSET + ADDRS_PER_INODE * 4;
    let mut i_nid = [0u32; NIDS_PER_INODE];
    for (i, slot) in i_nid.iter_mut().enumerate() {
        let o = nid_off + i * 4;
        *slot = r32(o);
    }

    Ok(F2fsInode {
        mode,
        size,
        uid,
        gid,
        links,
        atime,
        ctime,
        mtime,
        blocks,
        generation,
        flags,
        inline_flags,
        i_addr,
        i_nid,
    })
}

/// Decode a direct node block — 1018 u32 data-block pointers, then a
/// trailing footer we ignore.
pub fn decode_direct_node(buf: &[u8]) -> crate::Result<Vec<u32>> {
    if buf.len() < F2FS_BLKSIZE {
        return Err(crate::Error::InvalidImage(
            "f2fs: short read on direct node block".into(),
        ));
    }
    let mut out = Vec::with_capacity(super::constants::ADDRS_PER_BLOCK);
    for i in 0..super::constants::ADDRS_PER_BLOCK {
        let o = i * 4;
        out.push(u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()));
    }
    Ok(out)
}

/// Decode an indirect node block — 1018 u32 node IDs.
pub fn decode_indirect_node(buf: &[u8]) -> crate::Result<Vec<u32>> {
    if buf.len() < F2FS_BLKSIZE {
        return Err(crate::Error::InvalidImage(
            "f2fs: short read on indirect node block".into(),
        ));
    }
    let mut out = Vec::with_capacity(super::constants::NIDS_PER_BLOCK);
    for i in 0..super::constants::NIDS_PER_BLOCK {
        let o = i * 4;
        out.push(u32::from_le_bytes(buf[o..o + 4].try_into().unwrap()));
    }
    Ok(out)
}

/// Test helper: produce a 4 KiB inode block given a populated [`F2fsInode`].
/// Used by the integrated tests to round-trip a synthetic image through
/// [`decode_inode_block`].
#[cfg(test)]
pub(crate) fn encode_inode_block(ino: &F2fsInode) -> Vec<u8> {
    let mut buf = vec![0u8; F2FS_BLKSIZE];
    buf[0x00..0x02].copy_from_slice(&ino.mode.to_le_bytes());
    buf[0x03] = ino.inline_flags;
    buf[0x04..0x08].copy_from_slice(&ino.uid.to_le_bytes());
    buf[0x08..0x0C].copy_from_slice(&ino.gid.to_le_bytes());
    buf[0x0C..0x10].copy_from_slice(&ino.links.to_le_bytes());
    buf[0x10..0x18].copy_from_slice(&ino.size.to_le_bytes());
    buf[0x18..0x20].copy_from_slice(&ino.blocks.to_le_bytes());
    buf[0x20..0x28].copy_from_slice(&(ino.atime as u64).to_le_bytes());
    buf[0x28..0x30].copy_from_slice(&(ino.ctime as u64).to_le_bytes());
    buf[0x30..0x38].copy_from_slice(&(ino.mtime as u64).to_le_bytes());
    buf[0x44..0x48].copy_from_slice(&ino.generation.to_le_bytes());
    buf[0x50..0x54].copy_from_slice(&ino.flags.to_le_bytes());
    for (i, a) in ino.i_addr.iter().enumerate() {
        let o = I_ADDR_OFFSET + i * 4;
        buf[o..o + 4].copy_from_slice(&a.to_le_bytes());
    }
    let nid_off = I_ADDR_OFFSET + ADDRS_PER_INODE * 4;
    for (i, a) in ino.i_nid.iter().enumerate() {
        let o = nid_off + i * 4;
        buf[o..o + 4].copy_from_slice(&a.to_le_bytes());
    }
    // CRC32 footer.
    let crc = crc32fast::hash(&buf[..F2FS_BLK_CSUM_OFFSET]);
    buf[F2FS_BLK_CSUM_OFFSET..F2FS_BLK_CSUM_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    buf
}

/// Test helper: build a direct-node block from a slice of u32 pointers.
#[cfg(test)]
pub(crate) fn encode_direct_node(ptrs: &[u32]) -> Vec<u8> {
    let mut buf = vec![0u8; F2FS_BLKSIZE];
    for (i, p) in ptrs.iter().enumerate() {
        if i >= super::constants::ADDRS_PER_BLOCK {
            break;
        }
        let o = i * 4;
        buf[o..o + 4].copy_from_slice(&p.to_le_bytes());
    }
    buf
}

/// Test helper: build an indirect-node block from a slice of nids.
#[cfg(test)]
pub(crate) fn encode_indirect_node(nids: &[u32]) -> Vec<u8> {
    let mut buf = vec![0u8; F2FS_BLKSIZE];
    for (i, p) in nids.iter().enumerate() {
        if i >= super::constants::NIDS_PER_BLOCK {
            break;
        }
        let o = i * 4;
        buf[o..o + 4].copy_from_slice(&p.to_le_bytes());
    }
    buf
}
