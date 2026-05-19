//! F2FS NAT — Node Address Table.
//!
//! A NAT entry is `{ version: u8, ino: u32, block_addr: u32 }` (9 bytes).
//! 455 entries fit in one 4 KiB NAT page; the entry for nid `N` lives in
//! page `N / 455` at slot `N % 455`. The NAT region as a whole is
//! shadow-copied: each 4 KiB "logical" entry has two physical pages,
//! one in the "even" half and one in the "odd" half, selected by a
//! NAT-version bitmap held in the checkpoint pack. v1 picks the
//! `cur_nat_pack` half wholesale (every page from the same side) — full
//! per-page bitmap handling is straightforward to add later if needed.
//!
//! Lookups consult the checkpoint's in-memory NAT journal first: any
//! `nid` present there is the freshest version regardless of what the
//! on-disk page says.
//!
//! Reference: kernel docs §"NAT" + FAST '15 §2.2.

use crate::Result;
use crate::block::BlockDevice;

use super::checkpoint::Checkpoint;
use super::constants::{F2FS_BLKSIZE, NAT_ENTRY_PER_BLOCK, NAT_ENTRY_SIZE};
use super::superblock::Superblock;

/// Resolved physical block address for a node id.
#[derive(Debug, Clone, Copy)]
pub struct NodeAddr {
    pub block: u32,
    pub version: u8,
    pub ino: u32,
}

/// Look up the on-disk block holding the node `nid`.
///
/// 1. Consult the checkpoint's NAT journal (newest in-memory state).
/// 2. Otherwise read the appropriate NAT page from the active pack and
///    decode the entry at `nid % 455`.
pub fn lookup_node(
    dev: &mut dyn BlockDevice,
    sb: &Superblock,
    cp: &Checkpoint,
    nid: u32,
) -> Result<NodeAddr> {
    if let Some(j) = cp.nat_journal_lookup(nid) {
        if j.block_addr == 0 {
            return Err(crate::Error::InvalidImage(format!(
                "f2fs: nid {nid} unallocated (journal block_addr=0)"
            )));
        }
        return Ok(NodeAddr {
            block: j.block_addr,
            version: j.version,
            ino: j.ino,
        });
    }

    let bs = sb.block_size() as u64;
    let page_idx = (nid as usize) / NAT_ENTRY_PER_BLOCK;
    let slot = (nid as usize) % NAT_ENTRY_PER_BLOCK;

    // Each "logical" NAT page has two physical copies, laid out
    // contiguously: even-half pages occupy the first half of the NAT
    // region, odd-half pages the second half. (This is the bitmap-less
    // simplification described above.)
    let blocks_per_seg = sb.blocks_per_seg();
    let nat_total_blocks = sb.segment_count_nat * blocks_per_seg;
    let half = nat_total_blocks / 2;
    let pack = cp.cur_nat_pack as u32;
    let phys_page = sb.nat_blkaddr + pack * half + page_idx as u32;
    if (phys_page - sb.nat_blkaddr) >= nat_total_blocks {
        return Err(crate::Error::InvalidImage(format!(
            "f2fs: nid {nid} out of NAT range"
        )));
    }

    let mut page = vec![0u8; F2FS_BLKSIZE];
    dev.read_at(phys_page as u64 * bs, &mut page)?;

    let o = slot * NAT_ENTRY_SIZE;
    if o + NAT_ENTRY_SIZE > page.len() {
        return Err(crate::Error::InvalidImage(
            "f2fs: NAT slot past end of page".into(),
        ));
    }
    let version = page[o];
    let ino = u32::from_le_bytes(page[o + 1..o + 5].try_into().unwrap());
    let block_addr = u32::from_le_bytes(page[o + 5..o + 9].try_into().unwrap());
    if block_addr == 0 {
        return Err(crate::Error::InvalidImage(format!(
            "f2fs: nid {nid} has block_addr=0 (unallocated)"
        )));
    }
    Ok(NodeAddr {
        block: block_addr,
        version,
        ino,
    })
}

/// Encode a single NAT entry into a page slot. Test helper that mirrors
/// the decoder above so the on-disk layout stays in sync.
#[cfg(test)]
pub(crate) fn encode_nat_entry(page: &mut [u8], slot: usize, version: u8, ino: u32, block: u32) {
    let o = slot * NAT_ENTRY_SIZE;
    page[o] = version;
    page[o + 1..o + 5].copy_from_slice(&ino.to_le_bytes());
    page[o + 5..o + 9].copy_from_slice(&block.to_le_bytes());
}
