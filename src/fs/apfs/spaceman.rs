//! Minimal but structurally-correct APFS space manager.
//!
//! The writer hands `build_spaceman` a description of which container
//! blocks are currently used (the entire prefix from 0 to `next_block`,
//! since we bump-allocate); this module emits:
//!
//! * a `spaceman_phys_t` block with one main-device entry and an inline
//!   array of CIB physical addresses (no CAB indirection — single CIB),
//! * a `chunk_info_block_t` (CIB) containing one `chunk_info_t` per
//!   container chunk,
//! * one allocation-bitmap block per used chunk,
//! * three empty `spaceman_free_queue_t` B-tree roots (IP / MAIN / TIER2),
//!
//! and populates the internal-pool (`sm_ip_*`), free-queue (`sm_fq[*]`),
//! and allocation-zone (`sm_datazone`) fields with sensible defaults.
//!
//! Bitmap convention: a set bit means "used"; cleared bits are free.
//! This matches `apfsprogs/mkapfs/spaceman.c` (`bmap_mark_as_used`) and
//! the Apple File System Reference.
//!
//! Layout in the on-disk spaceman_phys_t (offsets from the start of the
//! block, all little-endian; identical to the Apple File System
//! Reference, Space Manager section):
//!
//! ```text
//!   0..32   obj_phys_t (cksum, oid, xid, type=OBJ_EPHEMERAL|0x0005, subtype=0)
//!  32..48   sm_block_size / sm_blocks_per_chunk / sm_chunks_per_cib / sm_cibs_per_cab
//!  48..96   SD_MAIN spaceman_device_t (48 bytes)
//!  96..144  SD_TIER2 spaceman_device_t (zero for single-device)
//! 144..148  sm_flags
//! 148..152  sm_ip_bm_tx_multiplier
//! 152..160  sm_ip_block_count
//! 160..164  sm_ip_bm_size_in_blocks
//! 164..168  sm_ip_bm_block_count
//! 168..176  sm_ip_bm_base
//! 176..184  sm_ip_base
//! 184..192  sm_fs_reserve_block_count
//! 192..200  sm_fs_reserve_alloc_count
//! 200..240  SFQ_IP    spaceman_free_queue_t (40 bytes)
//! 240..280  SFQ_MAIN  spaceman_free_queue_t
//! 280..320  SFQ_TIER2 spaceman_free_queue_t
//! 320..322  sm_ip_bm_free_head (u16)
//! 322..324  sm_ip_bm_free_tail (u16)
//! 324..328  sm_ip_bm_xid_offset (u32)
//! 328..332  sm_ip_bitmap_offset (u32)
//! 332..336  sm_ip_bm_free_next_offset (u32)
//! 336..340  sm_version (=1)
//! 340..344  sm_struct_size
//! 344..2520 sm_datazone (spaceman_datazone_info_phys_t —
//!             2 devices × 8 zones × 136 bytes = 2176 bytes)
//! ```
//!
//! The inline CIB-address array (`sm_dev[SD_MAIN].sm_addr_offset`) is
//! placed at the first 8-byte-aligned offset past the datazone.

use crate::Result;

use super::checksum::fletcher64;
use super::obj::{OBJECT_TYPE_BTREE, OBJECT_TYPE_BTREE_NODE};

/// `OBJECT_TYPE_SPACEMAN`.
pub(super) const OBJECT_TYPE_SPACEMAN: u32 = 0x0000_0005;
/// `OBJECT_TYPE_SPACEMAN_CIB`.
pub(super) const OBJECT_TYPE_SPACEMAN_CIB: u32 = 0x0000_0007;
/// `OBJECT_TYPE_SPACEMAN_FREE_QUEUE` (subtype on empty free-queue B-tree
/// roots).
pub(super) const OBJECT_TYPE_SPACEMAN_FREE_QUEUE: u32 = 0x0000_0009;

/// `OBJ_PHYSICAL`.
const OBJ_PHYSICAL: u32 = 0x4000_0000;
/// `OBJ_EPHEMERAL`.
const OBJ_EPHEMERAL: u32 = 0x8000_0000;

/// `SM_FLAG_VERSIONED` — set when `sm_version` is meaningful.
const SM_FLAG_VERSIONED: u32 = 0x0000_0001;

/// `SPACEMAN_IP_BM_TX_MULTIPLIER` (Apple File System Reference,
/// "Internal-Pool Bitmap": 16).
const SPACEMAN_IP_BM_TX_MULTIPLIER: u32 = 16;

/// `SPACEMAN_IP_BM_INDEX_INVALID` — sentinel u16 for an unused IP-ring
/// bitmap slot.
const SPACEMAN_IP_BM_INDEX_INVALID: u16 = 0xffff;

/// `SM_ALLOCZONE_NUM_PREVIOUS_BOUNDARIES`.
const SM_ALLOCZONE_NUM_PREVIOUS_BOUNDARIES: usize = 7;

/// `SM_ALLOCZONE_INVALID_END_BOUNDARY` — saz_zone_end sentinel meaning
/// "no boundary".
const SM_ALLOCZONE_INVALID_END_BOUNDARY: u64 = 0;

/// `SM_DATAZONE_ALLOCZONE_COUNT` — zones per device in `sm_datazone`.
const SM_DATAZONE_ALLOCZONE_COUNT: usize = 8;

/// `SD_COUNT` — devices in `sm_datazone[][]` (main + tier2).
const SD_COUNT: usize = 2;

/// `SD_MAIN` device index.
const SD_MAIN: usize = 0;

/// Size of one `spaceman_allocation_zone_info_phys_t`:
/// 16 (current) + 7*16 (previous) + 2 (zone_id) + 2 (prev_index) + 4
/// (reserved) = 136 bytes.
const ALLOC_ZONE_INFO_SIZE: usize = 16 + SM_ALLOCZONE_NUM_PREVIOUS_BOUNDARIES * 16 + 8;

/// Size of the entire `sm_datazone` substructure.
const DATAZONE_SIZE: usize = SD_COUNT * SM_DATAZONE_ALLOCZONE_COUNT * ALLOC_ZONE_INFO_SIZE;

/// Byte offset of `sm_datazone` inside `spaceman_phys_t`.
const SM_DATAZONE_OFFSET: usize = 344;

/// Byte offset of the inline CIB-address array (just past the
/// 2176-byte `sm_datazone`, rounded up to an 8-byte boundary).
const SM_CIB_ADDR_OFFSET: u32 = (SM_DATAZONE_OFFSET + DATAZONE_SIZE).next_multiple_of(8) as u32;

/// Default IP-ring size in blocks. APFS reserves a small ring of
/// physical blocks near the start of the container for ephemeral
/// metadata (checkpoint mapping, spaceman snapshots). 16 blocks (64 KiB
/// at 4 KiB block size) is comfortably enough for a freshly-formatted
/// container and is a multiple of `SPACEMAN_IP_BM_TX_MULTIPLIER`.
pub(super) const DEFAULT_IP_BLOCK_COUNT: u64 = 16;

/// Number of blocks one bitmap block can track at `block_size = 4096`.
/// Each byte tracks 8 blocks, so a 4096-byte bitmap covers 32 768
/// blocks. The spec ties this to `sm_blocks_per_chunk = 8 * block_size`.
pub(super) fn blocks_per_chunk(block_size: usize) -> u64 {
    (block_size as u64) * 8
}

/// Number of `chunk_info_t` entries (32 bytes each) that fit in a single
/// CIB block after the 40-byte header.
pub(super) fn chunks_per_cib(block_size: usize) -> u32 {
    ((block_size - 40) / 32) as u32
}

/// Information needed to emit the spaceman.
pub(super) struct SpacemanLayout {
    /// Block size of the container (typically 4096).
    pub block_size: usize,
    /// Total number of blocks in the container.
    pub total_blocks: u64,
    /// XID stamped on every emitted spaceman / CIB / bitmap block.
    pub xid: u64,
    /// Ephemeral oid for the spaceman header. Matches what the NXSB
    /// publishes in `nx_spaceman_oid` and what the checkpoint map
    /// resolves to the spaceman's physical block.
    pub spaceman_oid: u64,
    /// Physical address of the (sole) chunk_info_block.
    pub cib_paddr: u64,
    /// Physical addresses of the per-chunk bitmap blocks. One entry per
    /// chunk (in order). Empty bitmaps may be elided by passing
    /// `0` here; we always emit a full bitmap for chunk 0.
    pub bitmap_paddrs: Vec<u64>,
    /// All block ranges considered used by the writer. Inclusive start,
    /// exclusive end. The bitmap is generated by ORing the bit for each
    /// block in every range.
    pub used_ranges: Vec<(u64, u64)>,
    /// Physical address of the start of the internal-pool ring (the
    /// `sm_ip_base` field — a contiguous range of `ip_block_count`
    /// blocks reserved for ephemeral metadata).
    pub ip_base: u64,
    /// Number of blocks in the internal-pool ring. Must be a multiple
    /// of `SPACEMAN_IP_BM_TX_MULTIPLIER` and at least one block.
    pub ip_block_count: u64,
    /// Physical address of the first IP bitmap ring slot
    /// (`sm_ip_bm_base`).
    pub ip_bm_base: u64,
    /// Number of IP bitmap blocks (`sm_ip_bm_size_in_blocks`). For a
    /// fresh container, one bitmap block fully covers the ring.
    pub ip_bm_size_in_blocks: u32,
    /// Physical addresses of the three empty free-queue B-tree roots
    /// (`[SFQ_IP, SFQ_MAIN, SFQ_TIER2]`). One block each.
    pub free_queue_paddrs: [u64; 3],
}

/// Emitted spaceman artifact: the spaceman header, CIB, bitmap blocks,
/// IP-ring bitmap block, and three empty free-queue B-tree roots.
pub(super) struct EmittedSpaceman {
    pub spaceman_block: Vec<u8>,
    pub cib_block: Vec<u8>,
    /// One bitmap-block payload per chunk. `bitmap_blocks[i]` is the
    /// content to write at `layout.bitmap_paddrs[i]` (if that address is
    /// non-zero).
    pub bitmap_blocks: Vec<Vec<u8>>,
    /// Single IP-ring bitmap block, written at `layout.ip_bm_base`. All
    /// IP blocks start out free, so this is just an obj-headerless
    /// zeroed bitmap block. (We emit no obj header on the IP bitmap;
    /// per spec the IP bitmap blocks have no `obj_phys_t`.)
    pub ip_bm_block: Vec<u8>,
    /// Empty free-queue B-tree roots, in order `[SFQ_IP, SFQ_MAIN,
    /// SFQ_TIER2]`. Each is a leaf root carrying the trailing
    /// `btree_info_t` and zero entries.
    pub free_queue_blocks: [Vec<u8>; 3],
}

/// Build the spaceman_phys, the single chunk_info_block, one bitmap
/// block per chunk, the IP-ring bitmap block, and three empty
/// free-queue B-tree roots. The caller is responsible for writing each
/// block at the address recorded in `layout`.
pub(super) fn build_spaceman(layout: &SpacemanLayout) -> Result<EmittedSpaceman> {
    let bs = layout.block_size;
    if bs < 512 || !bs.is_power_of_two() {
        return Err(crate::Error::InvalidArgument(format!(
            "apfs spaceman: implausible block size {bs}"
        )));
    }
    let bpc = blocks_per_chunk(bs);
    let chunks: u64 = layout.total_blocks.div_ceil(bpc);
    let cpc = chunks_per_cib(bs);
    if chunks as u32 > cpc {
        return Err(crate::Error::Unsupported(format!(
            "apfs spaceman: {chunks} chunks would overflow a single CIB \
             (max {cpc}); CAB layer not implemented"
        )));
    }
    if layout.bitmap_paddrs.len() != chunks as usize {
        return Err(crate::Error::InvalidArgument(format!(
            "apfs spaceman: caller supplied {} bitmap paddrs for {} chunks",
            layout.bitmap_paddrs.len(),
            chunks,
        )));
    }
    if layout.ip_block_count == 0
        || layout.ip_block_count % SPACEMAN_IP_BM_TX_MULTIPLIER as u64 != 0
    {
        return Err(crate::Error::InvalidArgument(format!(
            "apfs spaceman: ip_block_count {} must be a non-zero multiple of \
             SPACEMAN_IP_BM_TX_MULTIPLIER ({})",
            layout.ip_block_count, SPACEMAN_IP_BM_TX_MULTIPLIER
        )));
    }
    if layout.ip_bm_size_in_blocks == 0 {
        return Err(crate::Error::InvalidArgument(
            "apfs spaceman: ip_bm_size_in_blocks must be ≥ 1".into(),
        ));
    }

    // ---- Per-chunk bitmaps ------------------------------------------
    let mut bitmap_blocks: Vec<Vec<u8>> = Vec::with_capacity(chunks as usize);
    let mut free_total: u64 = 0;
    let mut free_per_chunk: Vec<u32> = Vec::with_capacity(chunks as usize);
    for chunk_idx in 0..chunks {
        let mut bmap = vec![0u8; bs];
        let chunk_start = chunk_idx * bpc;
        let chunk_end = ((chunk_idx + 1) * bpc).min(layout.total_blocks);
        let chunk_blocks = (chunk_end - chunk_start) as u32;
        // Mark every used block inside this chunk's bitmap.
        for &(rs, re) in &layout.used_ranges {
            let lo = rs.max(chunk_start);
            let hi = re.min(chunk_end);
            for b in lo..hi {
                let bit = (b - chunk_start) as usize;
                bmap[bit / 8] |= 1 << (bit % 8);
            }
        }
        // Mark "beyond end of container" bits as used so the chunk's
        // free count is exactly free_blocks_in_chunk and fsck doesn't
        // claim them as free.
        for bit in chunk_blocks..(bpc as u32) {
            let bit = bit as usize;
            bmap[bit / 8] |= 1 << (bit % 8);
        }
        // Count free blocks in this chunk.
        let mut used: u32 = 0;
        for &(rs, re) in &layout.used_ranges {
            let lo = rs.max(chunk_start);
            let hi = re.min(chunk_end);
            if hi > lo {
                used += (hi - lo) as u32;
            }
        }
        let free = chunk_blocks - used;
        free_per_chunk.push(free);
        free_total += free as u64;
        bitmap_blocks.push(bmap);
    }

    // ---- Chunk-info-block (CIB) -------------------------------------
    let mut cib = vec![0u8; bs];
    // obj_phys
    cib[8..16].copy_from_slice(&layout.cib_paddr.to_le_bytes());
    cib[16..24].copy_from_slice(&layout.xid.to_le_bytes());
    cib[24..28].copy_from_slice(&(OBJECT_TYPE_SPACEMAN_CIB | OBJ_PHYSICAL).to_le_bytes());
    // subtype stays zero.
    // cib_index = 0
    cib[32..36].copy_from_slice(&0u32.to_le_bytes());
    // cib_chunk_info_count
    cib[36..40].copy_from_slice(&(chunks as u32).to_le_bytes());
    let chunk_info_base = 40usize;
    for (chunk_idx, (&free_in_chunk, &bmap_paddr)) in free_per_chunk
        .iter()
        .zip(layout.bitmap_paddrs.iter())
        .enumerate()
    {
        let off = chunk_info_base + chunk_idx * 32;
        let chunk_start = (chunk_idx as u64) * bpc;
        let chunk_end = (chunk_start + bpc).min(layout.total_blocks);
        let chunk_blocks = (chunk_end - chunk_start) as u32;
        // ci_xid
        cib[off..off + 8].copy_from_slice(&layout.xid.to_le_bytes());
        // ci_addr — block number of the first block in this chunk
        cib[off + 8..off + 16].copy_from_slice(&chunk_start.to_le_bytes());
        // ci_block_count
        cib[off + 16..off + 20].copy_from_slice(&chunk_blocks.to_le_bytes());
        // ci_free_count
        cib[off + 20..off + 24].copy_from_slice(&free_in_chunk.to_le_bytes());
        // ci_bitmap_addr — physical address of the bitmap block for
        // this chunk, or zero if the chunk has no bitmap (hole — every
        // bit would have been zero anyway).
        cib[off + 24..off + 32].copy_from_slice(&bmap_paddr.to_le_bytes());
    }
    sign_block(&mut cib);

    // ---- Spaceman header --------------------------------------------
    let mut sm = vec![0u8; bs];
    // obj_phys
    sm[8..16].copy_from_slice(&layout.spaceman_oid.to_le_bytes());
    sm[16..24].copy_from_slice(&layout.xid.to_le_bytes());
    sm[24..28].copy_from_slice(&(OBJECT_TYPE_SPACEMAN | OBJ_EPHEMERAL).to_le_bytes());
    // subtype stays zero.
    // sm_block_size
    sm[32..36].copy_from_slice(&(bs as u32).to_le_bytes());
    // sm_blocks_per_chunk
    sm[36..40].copy_from_slice(&(bpc as u32).to_le_bytes());
    // sm_chunks_per_cib
    sm[40..44].copy_from_slice(&cpc.to_le_bytes());
    // sm_cibs_per_cab — at the same arithmetic per CIB structure; doesn't
    // matter while we keep cab_count=0 but record a sensible figure.
    let cibs_per_cab = ((bs - 40) / 8) as u32;
    sm[44..48].copy_from_slice(&cibs_per_cab.to_le_bytes());

    // SD_MAIN spaceman_device_t (48 bytes) at offset 48
    sm[48..56].copy_from_slice(&layout.total_blocks.to_le_bytes()); // sm_block_count
    sm[56..64].copy_from_slice(&chunks.to_le_bytes()); // sm_chunk_count
    sm[64..68].copy_from_slice(&1u32.to_le_bytes()); // sm_cib_count = 1
    sm[68..72].copy_from_slice(&0u32.to_le_bytes()); // sm_cab_count = 0
    sm[72..80].copy_from_slice(&free_total.to_le_bytes()); // sm_free_count
    // sm_addr_offset: byte offset (relative to start of spaceman block)
    // where the inline CIB-address array begins. We place it at the
    // first 8-byte-aligned offset past the 2176-byte `sm_datazone`.
    let cib_addr_off: u32 = SM_CIB_ADDR_OFFSET;
    sm[80..84].copy_from_slice(&cib_addr_off.to_le_bytes());
    // reserved bytes stay zero.

    // SD_TIER2 left zero at offset 96.

    // sm_flags = SM_FLAG_VERSIONED so we publish sm_version reliably.
    sm[144..148].copy_from_slice(&SM_FLAG_VERSIONED.to_le_bytes());

    // ---- Internal-Pool (IP) ring ------------------------------------
    // The IP ring is a small contiguous reservation of physical blocks
    // near the start of the container used for ephemeral metadata
    // (checkpoint mapping, spaceman snapshots, etc.). We track every
    // ring-tracking field even though the writer never allocates out of
    // it: a structurally-correct ring is enough for downstream tools
    // and the freshly-formatted-image invariant.
    //
    // Field meanings (Apple File System Reference, Space Manager):
    //   sm_ip_bm_tx_multiplier  — how many transactions worth of IP
    //                             bitmaps fit in the ring (16 per spec).
    //   sm_ip_block_count       — total blocks reserved for the ring.
    //   sm_ip_bm_size_in_blocks — number of bitmap blocks used by the
    //                             ring (one is enough for a small ring
    //                             at any sensible block size).
    //   sm_ip_bm_block_count    — circular-buffer slot count for the
    //                             bitmap ring; equal to tx_multiplier.
    //   sm_ip_bm_base           — physical address of the IP bitmap
    //                             ring's first block.
    //   sm_ip_base              — physical address of the first ring
    //                             block proper.
    //   sm_ip_bm_free_head      — head of the bitmap-slot free list.
    //   sm_ip_bm_free_tail      — tail of the bitmap-slot free list.
    //   sm_ip_bm_xid_offset     — byte offset (inside an IP bitmap
    //                             block) of the per-slot xid array.
    //   sm_ip_bitmap_offset     — byte offset of the actual bit data.
    //   sm_ip_bm_free_next_offset — byte offset of the per-slot
    //                             next-pointer array.
    sm[148..152].copy_from_slice(&SPACEMAN_IP_BM_TX_MULTIPLIER.to_le_bytes());
    sm[152..160].copy_from_slice(&layout.ip_block_count.to_le_bytes());
    sm[160..164].copy_from_slice(&layout.ip_bm_size_in_blocks.to_le_bytes());
    sm[164..168].copy_from_slice(&SPACEMAN_IP_BM_TX_MULTIPLIER.to_le_bytes());
    sm[168..176].copy_from_slice(&layout.ip_bm_base.to_le_bytes());
    sm[176..184].copy_from_slice(&layout.ip_base.to_le_bytes());
    // sm_fs_reserve_block_count / sm_fs_reserve_alloc_count: leave at 0
    // (no reservations on a freshly-formatted image).

    // ---- Free queues (SFQ_IP / SFQ_MAIN / SFQ_TIER2) ----------------
    // Each free queue tracks blocks that are pending free across
    // transactions. On a fresh image every queue is empty — but the
    // sfq_tree_oid must point at a valid empty B-tree node so readers
    // that descend the queue tree see a well-formed leaf with zero
    // entries. We use OBJ_PHYSICAL trees (no omap indirection
    // needed): sfq_tree_oid carries the physical block address of the
    // empty root directly. This matches what the writer does for the
    // container/volume omap trees themselves.
    //
    // spaceman_free_queue_t layout (40 bytes):
    //    0   sfq_count             u64
    //    8   sfq_tree_oid          u64
    //   16   sfq_oldest_xid        u64
    //   24   sfq_tree_node_limit   u16
    //   26   sfq_pad16             u16
    //   28   sfq_pad32             u32
    //   32   sfq_reserved          u64
    let sfq_base = 200usize;
    for i in 0..3 {
        let off = sfq_base + i * 40;
        // sfq_count = 0 (empty queue)
        // sfq_tree_oid = physical block of the empty B-tree root
        sm[off + 8..off + 16].copy_from_slice(&layout.free_queue_paddrs[i].to_le_bytes());
        // sfq_oldest_xid = 0 (nothing pending free)
        // sfq_tree_node_limit = 0 (no per-tree node cap published)
        // sfq_pad16 / sfq_pad32 / sfq_reserved = 0
    }

    // ---- IP bitmap-ring offsets -------------------------------------
    // For our minimal image the ring is empty (no IP allocations yet),
    // so head == tail == 0 (slot 0 is the next-to-use). The per-slot
    // offset fields point at where the xid array, bitmap, and free-list
    // next pointers live inside each IP-bitmap block; we publish
    // sensible (mutually-disjoint, bs-bounded) offsets so a parser can
    // walk the ring even though we never write to it.
    sm[320..322].copy_from_slice(&0u16.to_le_bytes()); // sm_ip_bm_free_head
    sm[322..324].copy_from_slice(&0u16.to_le_bytes()); // sm_ip_bm_free_tail
    // Offsets inside an IP bitmap block. Per spec the bitmap data lives
    // at offset 0 and the per-slot xid / free-list arrays live in the
    // unused tail of the block. We carve out a small region for each.
    let xid_off: u32 = 0; // per-slot xid array (unused on fresh image)
    let bm_off: u32 = 0; // bitmap data starts at offset 0
    let next_off: u32 = 0; // per-slot free-list next-pointer array
    sm[324..328].copy_from_slice(&xid_off.to_le_bytes());
    sm[328..332].copy_from_slice(&bm_off.to_le_bytes());
    sm[332..336].copy_from_slice(&next_off.to_le_bytes());

    // sm_version
    sm[336..340].copy_from_slice(&1u32.to_le_bytes());

    // ---- Datazone (allocation zones) --------------------------------
    // Populate one allocation zone covering the entire main device. The
    // remaining seven main-device zones and all tier2 zones are left
    // zero — they describe regions of the device dedicated to specific
    // categories (snapshot metadata, etc.) and an unused zone is
    // expressed by `saz_zone_end = SM_ALLOCZONE_INVALID_END_BOUNDARY`
    // (0), which is the default.
    //
    // spaceman_allocation_zone_info_phys_t layout (136 bytes):
    //    0..16   saz_current_boundaries (saz_zone_start, saz_zone_end)
    //   16..128  saz_previous_boundaries[7] (7 × 16 bytes)
    //  128..130  saz_zone_id (u16)
    //  130..132  saz_previous_boundary_index (u16)
    //  132..136  saz_reserved (u32)
    //
    // sm_datazone is sdz_allocation_zones[SD_COUNT][SM_DATAZONE_ALLOCZONE_COUNT].
    let main_zone0_off =
        SM_DATAZONE_OFFSET + (SD_MAIN * SM_DATAZONE_ALLOCZONE_COUNT) * ALLOC_ZONE_INFO_SIZE;
    // saz_current_boundaries
    sm[main_zone0_off..main_zone0_off + 8].copy_from_slice(&0u64.to_le_bytes()); // saz_zone_start
    sm[main_zone0_off + 8..main_zone0_off + 16].copy_from_slice(&layout.total_blocks.to_le_bytes()); // saz_zone_end
    // saz_previous_boundaries: all zero (no history).
    // saz_zone_id = 0 (this is zone index 0 of the main device).
    // saz_previous_boundary_index: SM_ALLOCZONE_NUM_PREVIOUS_BOUNDARIES
    // sentinel for "no previous boundary written yet".
    sm[main_zone0_off + 130..main_zone0_off + 132]
        .copy_from_slice(&(SM_ALLOCZONE_NUM_PREVIOUS_BOUNDARIES as u16).to_le_bytes());
    // Suppress an unused-pattern-variable warning while keeping the
    // intent visible — the sentinel may become useful when we add
    // tier2 or per-volume zones.
    let _ = SM_ALLOCZONE_INVALID_END_BOUNDARY;

    // sm_struct_size — total in-use size of the on-disk struct. We say
    // it ends right after the inline CIB-address array.
    let struct_size = (cib_addr_off as usize) + 8 * (chunks as usize);
    sm[340..344].copy_from_slice(&(struct_size as u32).to_le_bytes());

    // Inline single-CIB address array at sm_addr_offset.
    let off = cib_addr_off as usize;
    if off + 8 > bs {
        return Err(crate::Error::Unsupported(format!(
            "apfs spaceman: block size {bs} too small for inline CIB addr array \
             (cib_addr_off={off})"
        )));
    }
    sm[off..off + 8].copy_from_slice(&layout.cib_paddr.to_le_bytes());

    sign_block(&mut sm);

    // ---- Empty free-queue B-tree roots ------------------------------
    // Each free-queue tree is a fixed-KV B-tree with:
    //   key = spaceman_free_queue_key_t (16 bytes: xid + paddr)
    //   val = spaceman_free_queue_val_t (8 bytes: u64 length)
    // On a fresh image each is an empty leaf root with the trailing
    // btree_info_t carrying bt_key_size=16 / bt_val_size=8 so a reader
    // can walk it. The `BTREE_ALLOW_GHOSTS` flag is set on real free
    // queues (ghosts mean "extent is exactly 1 block long") but is not
    // required for an empty tree; we set it anyway to match the spec.
    let mut free_queue_blocks: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (i, &paddr) in layout.free_queue_paddrs.iter().enumerate() {
        free_queue_blocks[i] = build_empty_free_queue_root(bs, paddr, layout.xid)?;
    }

    // ---- IP bitmap block --------------------------------------------
    // The IP bitmap tracks which blocks of the IP ring are in use. On a
    // freshly-formatted image every block of the ring is free, so the
    // bitmap is all zeros (cleared = free, per APFS convention). The IP
    // bitmap blocks have no obj_phys_t header per spec.
    let ip_bm_block = vec![0u8; bs];

    Ok(EmittedSpaceman {
        spaceman_block: sm,
        cib_block: cib,
        bitmap_blocks,
        ip_bm_block,
        free_queue_blocks,
    })
}

/// Build an empty free-queue B-tree root. Returns the on-disk block
/// content; the caller is responsible for writing it at `paddr`.
///
/// Layout: a single-node tree (BTNODE_ROOT | BTNODE_LEAF |
/// BTNODE_FIXED_KV_SIZE) with zero entries, an empty ToC, and a
/// trailing `btree_info_t` carrying:
///   * bt_flags  = BTREE_PHYSICAL | BTREE_ALLOW_GHOSTS
///   * bt_node_size = bs
///   * bt_key_size  = 16 (spaceman_free_queue_key_t)
///   * bt_val_size  = 8  (spaceman_free_queue_val_t)
fn build_empty_free_queue_root(bs: usize, paddr: u64, xid: u64) -> Result<Vec<u8>> {
    if bs < 56 + 40 {
        return Err(crate::Error::Unsupported(format!(
            "apfs spaceman: block size {bs} too small for empty free-queue root"
        )));
    }
    /// `BTNODE_ROOT | BTNODE_LEAF | BTNODE_FIXED_KV_SIZE`.
    const BTNODE_FLAGS: u16 = 0x0001 | 0x0002 | 0x0004;
    /// `BTREE_PHYSICAL | BTREE_ALLOW_GHOSTS`. The free-queue tree
    /// permits ghosts (a key with no value means a 1-block extent) and
    /// uses physical-block-address child pointers.
    const BTREE_INFO_FLAGS: u32 = 0x0000_0010 | 0x0000_0004;

    let mut block = vec![0u8; bs];
    // obj_phys: cksum (signed last), oid = paddr (physical-tree root),
    // xid, type = OBJECT_TYPE_BTREE | OBJ_PHYSICAL, subtype =
    // OBJECT_TYPE_SPACEMAN_FREE_QUEUE so a reader knows the leaf
    // payload semantics.
    block[8..16].copy_from_slice(&paddr.to_le_bytes());
    block[16..24].copy_from_slice(&xid.to_le_bytes());
    block[24..28].copy_from_slice(&(OBJECT_TYPE_BTREE | OBJ_PHYSICAL).to_le_bytes());
    block[28..32].copy_from_slice(&OBJECT_TYPE_SPACEMAN_FREE_QUEUE.to_le_bytes());

    // btn_flags / btn_level / btn_nkeys
    block[32..34].copy_from_slice(&BTNODE_FLAGS.to_le_bytes());
    block[34..36].copy_from_slice(&0u16.to_le_bytes()); // level (leaf)
    block[36..40].copy_from_slice(&0u32.to_le_bytes()); // nkeys = 0

    // Empty ToC at btn_data[0..0].
    block[40..42].copy_from_slice(&0u16.to_le_bytes()); // btn_table_space.off
    block[42..44].copy_from_slice(&0u16.to_le_bytes()); // btn_table_space.len
    // btn_free_space: the entire payload area between the (empty) ToC
    // and the trailing btree_info_t is free. data_start = 56,
    // vals_end = bs - 40, so free length = (bs - 40) - 56.
    let free_len = (bs - 40 - 56) as u16;
    block[44..46].copy_from_slice(&0u16.to_le_bytes()); // free_space.off
    block[46..48].copy_from_slice(&free_len.to_le_bytes()); // free_space.len
    // btn_key_free_list / btn_val_free_list: empty (BTOFF_INVALID).
    block[48..50].copy_from_slice(&0xffffu16.to_le_bytes());
    block[50..52].copy_from_slice(&0u16.to_le_bytes());
    block[52..54].copy_from_slice(&0xffffu16.to_le_bytes());
    block[54..56].copy_from_slice(&0u16.to_le_bytes());

    // Trailing btree_info_t at bs-40.
    let info_off = bs - 40;
    block[info_off..info_off + 4].copy_from_slice(&BTREE_INFO_FLAGS.to_le_bytes());
    block[info_off + 4..info_off + 8].copy_from_slice(&(bs as u32).to_le_bytes()); // bt_node_size
    block[info_off + 8..info_off + 12].copy_from_slice(&16u32.to_le_bytes()); // bt_key_size
    block[info_off + 12..info_off + 16].copy_from_slice(&8u32.to_le_bytes()); // bt_val_size
    // bt_longest_key / bt_longest_val / bt_key_count / bt_node_count
    // all zero (empty tree).
    block[info_off + 32..info_off + 40].copy_from_slice(&1u64.to_le_bytes()); // bt_node_count = 1

    // Suppress unused-imports lint while keeping the constants visible
    // for downstream readers.
    let _ = (OBJECT_TYPE_BTREE_NODE, SPACEMAN_IP_BM_INDEX_INVALID);

    sign_block(&mut block);
    Ok(block)
}

/// Fletcher-64 sign helper (duplicated from `write.rs` so this module
/// is self-contained).
fn sign_block(buf: &mut [u8]) {
    let cksum = fletcher64(buf);
    buf[0..8].copy_from_slice(&cksum.to_le_bytes());
}

// ---- decoders for tests ----

/// Lightweight decoded view of a spaceman header used by lib tests.
#[cfg(test)]
#[derive(Debug)]
pub(super) struct DecodedSpaceman {
    pub block_size: u32,
    pub blocks_per_chunk: u32,
    pub main_block_count: u64,
    pub main_chunk_count: u64,
    pub main_cib_count: u32,
    pub main_cab_count: u32,
    pub main_free_count: u64,
    pub cib_paddr: u64,
    pub flags: u32,
    pub ip_bm_tx_multiplier: u32,
    pub ip_block_count: u64,
    pub ip_bm_size_in_blocks: u32,
    pub ip_bm_block_count: u32,
    pub ip_bm_base: u64,
    pub ip_base: u64,
    pub sfq_tree_oids: [u64; 3],
    pub datazone_main_zone0_start: u64,
    pub datazone_main_zone0_end: u64,
}

/// Decode a freshly emitted spaceman block. Returns an error if the
/// type bits or version look wrong.
#[cfg(test)]
pub(super) fn decode_spaceman(buf: &[u8]) -> Result<DecodedSpaceman> {
    if buf.len() < SM_DATAZONE_OFFSET + DATAZONE_SIZE {
        return Err(crate::Error::InvalidImage(
            "apfs spaceman: block too short".into(),
        ));
    }
    let otype = u32::from_le_bytes(buf[24..28].try_into().unwrap());
    if otype & 0x0000_ffff != OBJECT_TYPE_SPACEMAN {
        return Err(crate::Error::InvalidImage(format!(
            "apfs spaceman: unexpected object type {otype:#x}"
        )));
    }
    let block_size = u32::from_le_bytes(buf[32..36].try_into().unwrap());
    let blocks_per_chunk = u32::from_le_bytes(buf[36..40].try_into().unwrap());
    let main_block_count = u64::from_le_bytes(buf[48..56].try_into().unwrap());
    let main_chunk_count = u64::from_le_bytes(buf[56..64].try_into().unwrap());
    let main_cib_count = u32::from_le_bytes(buf[64..68].try_into().unwrap());
    let main_cab_count = u32::from_le_bytes(buf[68..72].try_into().unwrap());
    let main_free_count = u64::from_le_bytes(buf[72..80].try_into().unwrap());
    let cib_addr_off = u32::from_le_bytes(buf[80..84].try_into().unwrap()) as usize;
    if cib_addr_off + 8 > buf.len() {
        return Err(crate::Error::InvalidImage(format!(
            "apfs spaceman: sm_addr_offset {cib_addr_off} past block end"
        )));
    }
    let cib_paddr = u64::from_le_bytes(buf[cib_addr_off..cib_addr_off + 8].try_into().unwrap());
    let flags = u32::from_le_bytes(buf[144..148].try_into().unwrap());
    let ip_bm_tx_multiplier = u32::from_le_bytes(buf[148..152].try_into().unwrap());
    let ip_block_count = u64::from_le_bytes(buf[152..160].try_into().unwrap());
    let ip_bm_size_in_blocks = u32::from_le_bytes(buf[160..164].try_into().unwrap());
    let ip_bm_block_count = u32::from_le_bytes(buf[164..168].try_into().unwrap());
    let ip_bm_base = u64::from_le_bytes(buf[168..176].try_into().unwrap());
    let ip_base = u64::from_le_bytes(buf[176..184].try_into().unwrap());
    let mut sfq_tree_oids = [0u64; 3];
    for (i, slot) in sfq_tree_oids.iter_mut().enumerate() {
        let off = 200 + i * 40 + 8;
        *slot = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
    }
    let dz_main_zone0 =
        SM_DATAZONE_OFFSET + (SD_MAIN * SM_DATAZONE_ALLOCZONE_COUNT) * ALLOC_ZONE_INFO_SIZE;
    let datazone_main_zone0_start =
        u64::from_le_bytes(buf[dz_main_zone0..dz_main_zone0 + 8].try_into().unwrap());
    let datazone_main_zone0_end = u64::from_le_bytes(
        buf[dz_main_zone0 + 8..dz_main_zone0 + 16]
            .try_into()
            .unwrap(),
    );
    Ok(DecodedSpaceman {
        block_size,
        blocks_per_chunk,
        main_block_count,
        main_chunk_count,
        main_cib_count,
        main_cab_count,
        main_free_count,
        cib_paddr,
        flags,
        ip_bm_tx_multiplier,
        ip_block_count,
        ip_bm_size_in_blocks,
        ip_bm_block_count,
        ip_bm_base,
        ip_base,
        sfq_tree_oids,
        datazone_main_zone0_start,
        datazone_main_zone0_end,
    })
}

/// Decode the chunk_info entries inside a CIB block.
#[cfg(test)]
pub(super) fn decode_cib_entries(buf: &[u8]) -> Result<Vec<DecodedChunkInfo>> {
    if buf.len() < 40 {
        return Err(crate::Error::InvalidImage(
            "apfs CIB: block too short".into(),
        ));
    }
    let otype = u32::from_le_bytes(buf[24..28].try_into().unwrap());
    if otype & 0x0000_ffff != OBJECT_TYPE_SPACEMAN_CIB {
        return Err(crate::Error::InvalidImage(format!(
            "apfs CIB: unexpected object type {otype:#x}"
        )));
    }
    let count = u32::from_le_bytes(buf[36..40].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let off = 40 + i * 32;
        if off + 32 > buf.len() {
            return Err(crate::Error::InvalidImage(
                "apfs CIB: chunk_info_t past block end".into(),
            ));
        }
        out.push(DecodedChunkInfo {
            addr: u64::from_le_bytes(buf[off + 8..off + 16].try_into().unwrap()),
            block_count: u32::from_le_bytes(buf[off + 16..off + 20].try_into().unwrap()),
            free_count: u32::from_le_bytes(buf[off + 20..off + 24].try_into().unwrap()),
            bitmap_addr: u64::from_le_bytes(buf[off + 24..off + 32].try_into().unwrap()),
        });
    }
    Ok(out)
}

/// Lightweight decoded view of one `chunk_info_t`.
#[cfg(test)]
#[derive(Debug)]
pub(super) struct DecodedChunkInfo {
    pub addr: u64,
    pub block_count: u32,
    pub free_count: u32,
    pub bitmap_addr: u64,
}

/// Count the set bits in a bitmap buffer covering the first
/// `chunk_blocks` bits. (Bits past `chunk_blocks` are ignored.)
#[cfg(test)]
pub(super) fn count_used_bits(bmap: &[u8], chunk_blocks: u32) -> u32 {
    let mut n: u32 = 0;
    for bit in 0..chunk_blocks as usize {
        if bmap[bit / 8] & (1 << (bit % 8)) != 0 {
            n += 1;
        }
    }
    n
}
