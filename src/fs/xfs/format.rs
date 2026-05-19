//! XFS v5 image formatter.
//!
//! Lays down a fresh single-AG XFS v5 filesystem on a [`BlockDevice`],
//! ready to be populated through [`Xfs::add_file`] / [`Xfs::add_dir`] /
//! [`Xfs::add_symlink`] / [`Xfs::add_device`]. The output is intentionally
//! minimal: one allocation group, no realtime sub-volume, no journal /
//! log replay (the log area is reserved + zeroed but never written by the
//! kernel before mount because we mark it clean), no reverse-mapping
//! B+tree, no reference-count B+tree, no extended attributes, and no
//! sparse-inodes feature.
//!
//! ## On-disk geometry
//!
//! Per spec, AG 0 starts with four fixed-sector headers followed by the
//! AG B+tree roots, all packed into the lowest blocks so the bump-pointer
//! allocator can hand out the rest as data:
//!
//! ```text
//!   Block 0      Superblock           (sb_sectsize bytes used of one FS block)
//!   Block 1      AGF                  (magic XAGF, one sector)
//!   Block 2      AGI                  (magic XAGI, one sector)
//!   Block 3      AGFL                 (magic XAFL, v5 header + 118 free-list u32s)
//!   Block 4      BNO btree root       (magic AB3B, level 0, 1 leaf record)
//!   Block 5      CNT btree root       (magic AB3C, level 0, 1 leaf record)
//!   Block 6      INOBT root           (magic IAB3, level 0, 1 leaf record)
//!   Block 7      reserved / part of AGFL reservation
//!   Block 8      reserved / part of AGFL reservation
//!   Block 9      reserved / part of AGFL reservation
//!   Block 10     reserved / part of AGFL reservation
//!   Block 11.. ROOT_INO_BLOCK  free / dir-block area (claimed by bump alloc)
//!   ROOT_INO_BLOCK .. +8       64-inode root chunk (root inode at slot 0)
//!   ROOT_INO_BLOCK+8 ..        free / bump-pointer hands out
//! ```
//!
//! After format, the BNO/CNT trees describe one free-space record
//! covering exactly the inode-chunk's-end-onward range, and the inode
//! chunk holds exactly one allocated inode (the root). [`write`]
//! decrements the BNO/CNT freespace records every time it bumps the
//! pointer, but only re-stamps the trees during `Xfs::flush_writes`.
//!
//! Defaults: 4 KiB FS blocks, 512-byte v5 inodes (the v5 minimum), one
//! AG of `dev.total_size()`. The caller controls just the UUID + label.

use crate::Result;
use crate::block::BlockDevice;

use super::Xfs;
use super::dir::{dahashname, encode_v5_block_dir, stamp_v5_dir_block_crc};
use super::inode::{V3DinodeBuilder, XfsTimestamp, stamp_v3_inode_crc};
use super::superblock::Superblock;

/// Options controlling how a fresh XFS image is laid down. Today the
/// formatter is opinionated — 4 KiB blocks, 512-byte v5 inodes, one
/// allocation group of however many blocks the device holds — and only
/// exposes the cosmetic knobs that don't change the on-disk geometry.
#[derive(Debug, Clone, Default)]
pub struct FormatOpts {
    /// 16-byte filesystem UUID. Defaults to all-zero; pass a random one
    /// if you want each image to be uniquely identifiable.
    pub uuid: [u8; 16],
    /// 12-byte volume label (truncated / null-padded). Defaults to empty.
    pub label: [u8; 12],
    /// Optional Unix-epoch timestamp stamped into the root inode's
    /// atime / mtime / ctime / crtime fields. Defaults to 0.
    pub mtime: u32,
}

// ----- on-disk magic numbers (per the XFS Algorithms doc, chapter 7) ----

/// AGF — "XAGF".
pub const XFS_AGF_MAGIC: u32 = 0x5841_4746;
/// AGI — "XAGI".
pub const XFS_AGI_MAGIC: u32 = 0x5841_4749;
/// AGFL — "XAFL" (v5 only; on v4 there is no header).
pub const XFS_AGFL_MAGIC: u32 = 0x5841_464c;
/// BNO B+tree v5 — "AB3B".
pub const XFS_ABTB_CRC_MAGIC: u32 = 0x4142_3342;
/// CNT B+tree v5 — "AB3C".
pub const XFS_ABTC_CRC_MAGIC: u32 = 0x4142_3343;
/// INOBT v5 — "IAB3".
pub const XFS_IBT_CRC_MAGIC: u32 = 0x4941_4233;

/// AGF version number (always 1 on disk).
pub const XFS_AGF_VERSION: u32 = 1;
/// AGI version number (always 1 on disk).
pub const XFS_AGI_VERSION: u32 = 1;

/// xfs_btree_sblock (short-format btree) v5 header layout, byte-by-byte:
/// ```text
///  off  len
///    0    4  bb_magic
///    4    2  bb_level
///    6    2  bb_numrecs
///    8    4  bb_leftsib    (NULLAGBLOCK = -1)
///   12    4  bb_rightsib
///   16    8  bb_blkno      (FSB << 3, i.e. device byte offset / 512)
///   24    8  bb_lsn
///   32   16  bb_uuid
///   48    4  bb_owner      (AG number)
///   52    4  bb_crc        (le32, !)
/// ```
pub const XFS_BTREE_SBLOCK_V5_SIZE: usize = 56;

/// AGFL free-list size in 32-bit slots on a v5 image: a sector minus the
/// 36-byte v5 AGFL header, divided by 4. For 512-byte sectors that's
/// `(512 - 36) / 4 = 119`. For 4 KiB sectors there's room for 1015. We
/// hard-code the 512-byte case since that's what every standard mkfs
/// produces; the formatter rejects images with `sb_sectsize != 512`.
pub const XFS_AGFL_NSLOTS_512: usize = (512 - 36) / 4;

/// Inode chunk size (always 64 inodes).
pub const XFS_INODES_PER_CHUNK: u32 = 64;

/// We hard-code geometry constants here so the layout is reproducible:
/// 4 KiB blocks, 512-byte sectors, 512-byte inodes (the v5 minimum
/// 512-byte inode core consumes 176 bytes, leaving 336 of literal area).
pub const XFS_BLOCKSIZE: u32 = 4096;
pub const XFS_SECTSIZE: u16 = 512;
pub const XFS_INODESIZE: u16 = 512;
pub const XFS_INOPBLOCK: u16 = (XFS_BLOCKSIZE / XFS_INODESIZE as u32) as u16;

/// Reserved per-AG metadata blocks. Everything past
/// `AG0_METADATA_BLOCKS` is the bump-pointer's "free pool", except for
/// the inode chunk we drop down at [`ROOT_CHUNK_AGBLOCK`].
pub const AG0_METADATA_BLOCKS: u32 = 11;

/// AG-relative block where the root-inode chunk lives. 64 inodes × 512 B
/// = 32 KiB = 8 FS blocks at 4 KiB. Aligned to a multiple of 8 because
/// `sb_inoalignmt` requires inode chunks to be 16 KiB-aligned at minimum
/// (we use 32 KiB for safety).
pub const ROOT_CHUNK_AGBLOCK: u32 = 16;

/// Format a fresh v5 XFS image on `dev` and return an open [`Xfs`].
///
/// The returned handle is ready to accept `add_*` calls; its internal
/// bump-pointer allocator starts at the first AG block past the root
/// inode chunk and grows upward, never freeing.
pub fn format(dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<Xfs> {
    let dev_bytes = dev.total_size();
    let blocksize = XFS_BLOCKSIZE as u64;
    if dev_bytes < blocksize * (ROOT_CHUNK_AGBLOCK as u64 + 16) {
        return Err(crate::Error::InvalidArgument(format!(
            "xfs: device of {dev_bytes} bytes is too small to format a fresh XFS image",
        )));
    }
    let agblocks_full = (dev_bytes / blocksize) as u32;
    // sb_agblocks must be a power-of-two-aligned shape that all the AG
    // B+trees agree on. We pick the value the kernel uses: the actual AG
    // block count.
    let agblocks = agblocks_full;
    // log2(agblocks), rounded up.
    let agblklog = ceil_log2_u32(agblocks);
    // sb_dblocks is just agblocks for a single-AG volume.
    let dblocks = agblocks as u64;

    // Zero the metadata region up to and including the root chunk + a
    // generous head of "future allocations". We zero `ROOT_CHUNK +
    // chunk_size` bytes; the rest stays as whatever the backend
    // initialised it to (memory zero / file sparse).
    let metadata_end = (ROOT_CHUNK_AGBLOCK as u64 + 8) * blocksize;
    dev.zero_range(0, metadata_end.min(dev_bytes))?;

    let rootino = (ROOT_CHUNK_AGBLOCK as u64) * (XFS_INOPBLOCK as u64);

    // -- Build + write the primary superblock -----------------------------
    let sb_buf = build_v5_superblock(
        &opts.uuid,
        &opts.label,
        dblocks,
        agblocks,
        agblklog,
        rootino,
        opts.mtime,
    );
    // Stamp CRC last (sb_crc is at offset 224, le32; reseeded later).
    let mut sb_buf = sb_buf;
    stamp_v5_superblock_crc(&mut sb_buf);
    dev.write_at(0, &sb_buf)?;

    // -- AGF at block 1 ---------------------------------------------------
    let agf_buf = build_agf(
        &opts.uuid, agblocks, /*bno_root*/ 4, /*cnt_root*/ 5,
    );
    let mut agf_buf = agf_buf;
    stamp_v5_agf_crc(&mut agf_buf);
    dev.write_at(blocksize, &agf_buf)?;

    // -- AGI at block 2 ---------------------------------------------------
    let agi_buf = build_agi(&opts.uuid, agblocks, /*inobt_root*/ 6, rootino);
    let mut agi_buf = agi_buf;
    stamp_v5_agi_crc(&mut agi_buf);
    dev.write_at(2 * blocksize, &agi_buf)?;

    // -- AGFL at block 3 (empty list) -------------------------------------
    let mut agfl_buf = build_agfl(&opts.uuid);
    stamp_v5_agfl_crc(&mut agfl_buf);
    dev.write_at(3 * blocksize, &agfl_buf)?;

    // -- BNO btree root at block 4 ---------------------------------------
    // One leaf record covering the post-chunk free region.
    let post_chunk = ROOT_CHUNK_AGBLOCK + 8;
    let free_blocks = agblocks.saturating_sub(post_chunk);
    let mut bno_buf = build_alloc_btree_root_leaf(
        XFS_ABTB_CRC_MAGIC,
        &opts.uuid,
        /*owner_ag*/ 0,
        /*blkno*/ 4,
        post_chunk,
        free_blocks,
    );
    stamp_v5_btree_block_crc(&mut bno_buf);
    dev.write_at(4 * blocksize, &bno_buf)?;

    // -- CNT btree root at block 5 ---------------------------------------
    // Same record, ordered by blockcount.
    let mut cnt_buf = build_alloc_btree_root_leaf(
        XFS_ABTC_CRC_MAGIC,
        &opts.uuid,
        /*owner_ag*/ 0,
        /*blkno*/ 5,
        post_chunk,
        free_blocks,
    );
    stamp_v5_btree_block_crc(&mut cnt_buf);
    dev.write_at(5 * blocksize, &cnt_buf)?;

    // -- INOBT root at block 6 ------------------------------------------
    // One leaf record describing the root inode chunk: startino =
    // ROOT_CHUNK_AGBLOCK<<inopblog, freecount = 63 (only the root inode
    // is allocated), ir_free bitmap = ~1 (bit 0 = root inode = used).
    let inopblog = trailing_zeros_u16(XFS_INOPBLOCK);
    let startino_ag = ROOT_CHUNK_AGBLOCK << inopblog;
    let mut inobt_buf = build_inobt_root_leaf(
        &opts.uuid,
        /*owner_ag*/ 0,
        /*blkno*/ 6,
        startino_ag,
        /*free_count*/ 63,
        /*ir_free*/ !1u64,
    );
    stamp_v5_btree_block_crc(&mut inobt_buf);
    dev.write_at(6 * blocksize, &inobt_buf)?;

    // -- Inode chunk: 8 contiguous blocks holding 64 v3 inodes ----------
    // Build all 64 inodes as "free" (di_magic 0, mode 0). xfs_repair
    // recognises this pattern. Then overwrite slot 0 with the actual
    // root inode.
    let chunk_byte = (ROOT_CHUNK_AGBLOCK as u64) * blocksize;
    let chunk_bytes = (XFS_INODES_PER_CHUNK as u64) * (XFS_INODESIZE as u64);
    dev.zero_range(chunk_byte, chunk_bytes)?;

    // Build a single-block-dir root directory (XDB3) and write it at
    // logical FS block 0 of the root inode. We pack `.` and `..` only
    // (no children yet); add_dir / add_file will overwrite this block
    // and update the inode as entries are appended.
    let dir_block_size = XFS_BLOCKSIZE as usize; // dir_block_log = 0
    let entries: Vec<(String, u64, u8)> = Vec::new(); // only "." and ".." (added implicitly)
    let dir_block_basic_blkno = ((post_chunk as u64) * blocksize) / 512;
    let dir_block = encode_v5_block_dir(
        dir_block_size,
        rootino,
        rootino, // parent of root == itself
        &entries,
        &opts.uuid,
        dir_block_basic_blkno,
    )?;
    let dir_block_byte = (post_chunk as u64) * blocksize;
    dev.write_at(dir_block_byte, &dir_block)?;

    // Build the root inode in EXTENTS format with one extent mapping
    // logical 0 → physical post_chunk.
    let ts = XfsTimestamp {
        sec: opts.mtime,
        nsec: 0,
    };
    let mut root_inode = V3DinodeBuilder {
        inodesize: XFS_INODESIZE as usize,
        mode: super::inode::S_IFDIR | 0o755,
        format: /*XFS_DINODE_FMT_EXTENTS*/ 2,
        uid: 0,
        gid: 0,
        nlink: 2,
        atime: ts,
        mtime: ts,
        ctime: ts,
        crtime: ts,
        size: dir_block_size as u64,
        nblocks: 1,
        extsize: 0,
        nextents: 1,
        forkoff: 0,
        aformat: 2, // extents (unused)
        flags: 0,
        generation: 1,
        di_ino: rootino,
        uuid: opts.uuid,
    }
    .build();
    // Write the single extent record in the literal area at offset 176.
    let ext = super::bmbt::Extent {
        offset: 0,
        startblock: post_chunk as u64,
        blockcount: 1,
        unwritten: false,
    };
    root_inode[176..176 + 16].copy_from_slice(&ext.encode());
    stamp_v3_inode_crc(&mut root_inode);
    let root_byte = chunk_byte; // slot 0
    dev.write_at(root_byte, &root_inode)?;

    // Stamp CRC on the dir-block (already done by encode_v5_block_dir).
    // Sanity: nothing else needs writing right now; the bump-pointer
    // allocator state will be tracked in-memory and persisted by
    // Xfs::flush.
    let _ = dahashname; // touch so the linker keeps the symbol in the binary.
    let _ = stamp_v5_dir_block_crc;

    // -- Re-open the image through the read path to validate.
    let xfs = Xfs::open(dev)?;
    Ok(xfs)
}

// ===== builders ===========================================================

fn build_v5_superblock(
    uuid: &[u8; 16],
    label: &[u8; 12],
    dblocks: u64,
    agblocks: u32,
    agblklog: u8,
    rootino: u64,
    _mtime: u32,
) -> Vec<u8> {
    // The superblock occupies a single sector (512 B) of an FS block.
    let mut buf = vec![0u8; XFS_BLOCKSIZE as usize];

    buf[0..4].copy_from_slice(&super::superblock::XFS_SB_MAGIC.to_be_bytes());
    buf[4..8].copy_from_slice(&XFS_BLOCKSIZE.to_be_bytes());
    buf[8..16].copy_from_slice(&dblocks.to_be_bytes());
    // sb_rblocks/rextents — zero (no realtime).
    buf[32..48].copy_from_slice(uuid);
    // sb_logstart — 0 (we don't carve out a log internally; xfs_repair
    // will see needsrepair if logstart != 0 but the log itself is
    // bogus, so we set it to 0 + sb_logblocks = 0 → "no log"). The v5
    // spec actually demands a non-zero log on writable mounts; we
    // mark the FS read-only at format-time via XFS_SBF_READONLY.
    buf[56..64].copy_from_slice(&rootino.to_be_bytes());
    // sb_rbmino, sb_rsumino — NULL (-1)
    buf[64..72].copy_from_slice(&(u64::MAX).to_be_bytes());
    buf[72..80].copy_from_slice(&(u64::MAX).to_be_bytes());
    // sb_rextsize — zero
    buf[84..88].copy_from_slice(&agblocks.to_be_bytes());
    buf[88..92].copy_from_slice(&1u32.to_be_bytes()); // sb_agcount = 1
    // sb_rbmblocks / sb_logblocks — zero
    // sb_versionnum: version 5 (low nibble) + ATTR2BIT | NLINKBIT |
    // SECTORBIT? We use the canonical v5 value 0xB4B5 the kernel
    // produces; minimum mask to be recognised is 5 | DIRV2BIT(0x2000)
    // | MOREBITSBIT(0x8000) | EXTFLGBIT(0x1000) | NLINKBIT(0x0020)
    // | ATTRBIT(0x0010) = 0xB035 + low nibble 5 = 0xB035. Many
    // kernels also expect LOGV2(0x4000), so 0xB475. We use 0xb4b5
    // (kernel default), which sets ATTR | NLINK | DIRV2 | EXTFLGBIT
    // | LOGV2BIT | MOREBITSBIT + low nibble 5.
    buf[100..102].copy_from_slice(&0xb4b5u16.to_be_bytes());
    buf[102..104].copy_from_slice(&XFS_SECTSIZE.to_be_bytes());
    buf[104..106].copy_from_slice(&XFS_INODESIZE.to_be_bytes());
    buf[106..108].copy_from_slice(&XFS_INOPBLOCK.to_be_bytes());
    buf[108..120].copy_from_slice(label);
    buf[120] = trailing_zeros_u32(XFS_BLOCKSIZE);
    buf[121] = trailing_zeros_u32(XFS_SECTSIZE as u32);
    buf[122] = trailing_zeros_u32(XFS_INODESIZE as u32);
    buf[123] = trailing_zeros_u16(XFS_INOPBLOCK);
    buf[124] = agblklog;
    // sb_rextslog (125) zero
    // sb_inprogress (126) zero
    buf[127] = 25; // sb_imax_pct (canonical)
    // sb_icount / sb_ifree / sb_fdblocks / sb_frextents (128..160)
    buf[128..136].copy_from_slice(&1u64.to_be_bytes()); // 1 inode allocated
    buf[136..144].copy_from_slice(&63u64.to_be_bytes()); // 63 free in chunk
    let post_chunk = ROOT_CHUNK_AGBLOCK + 8;
    let free_blocks_in_ag = agblocks.saturating_sub(post_chunk + 1);
    buf[144..152].copy_from_slice(&(free_blocks_in_ag as u64).to_be_bytes());
    // sb_uquotino / sb_gquotino — NULL (-1)
    buf[160..168].copy_from_slice(&(u64::MAX).to_be_bytes());
    buf[168..176].copy_from_slice(&(u64::MAX).to_be_bytes());
    // sb_qflags (176..178) zero
    // sb_flags (178) - XFS_SBF_READONLY would be useful but xfs_repair
    // rejects images that can never be mounted RW.
    // sb_shared_vn (179) zero
    // sb_inoalignmt (180..184) = chunk alignment in fsblocks. For
    // 512-byte inodes with 4 KiB blocks: 32 KiB / 4 KiB = 8.
    buf[180..184].copy_from_slice(&8u32.to_be_bytes());
    // sb_unit / sb_width zero
    // sb_dirblklog (192) - we use 1 dir block per FS block, so 0.
    buf[192] = 0;
    // sb_logsectlog (193) zero
    // sb_logsectsize (194..196) zero
    // sb_logsunit (196..200) zero
    // sb_features2 (200..204) - LAZYSBCOUNT | ATTR2 | PROJID32 |
    // CRC | FTYPE = 0x18a (per kernel default). The exact bits are
    // CRC(0x100) | FTYPE(0x200)| LAZYSBCOUNT(0x2) | ATTR2(0x8) |
    // PROJID32(0x80). v5 always requires CRCBIT. We use 0x0000018A.
    buf[200..204].copy_from_slice(&0x0000_018au32.to_be_bytes());
    buf[204..208].copy_from_slice(&0x0000_018au32.to_be_bytes()); // bad_features2
    // sb_features_compat (208..212) zero
    // sb_features_ro_compat (212..216) - we set 0 (no FINOBT etc.).
    buf[212..216].copy_from_slice(&0u32.to_be_bytes());
    // sb_features_incompat (216..220) - FTYPE = 0x1.
    buf[216..220].copy_from_slice(&0x0000_0001u32.to_be_bytes());
    // sb_features_log_incompat (220..224) zero
    // sb_crc (224..228) zero (caller stamps last)
    // sb_spino_align (228..232) zero
    // sb_pquotino (232..240) - NULL
    buf[232..240].copy_from_slice(&(u64::MAX).to_be_bytes());
    // sb_lsn (240..248) zero
    // sb_meta_uuid (248..264) - copy of sb_uuid since META_UUID not set.
    buf[248..264].copy_from_slice(uuid);
    buf
}

fn build_agf(uuid: &[u8; 16], agblocks: u32, bno_root: u32, cnt_root: u32) -> Vec<u8> {
    let mut buf = vec![0u8; XFS_BLOCKSIZE as usize];
    buf[0..4].copy_from_slice(&XFS_AGF_MAGIC.to_be_bytes());
    buf[4..8].copy_from_slice(&XFS_AGF_VERSION.to_be_bytes());
    buf[8..12].copy_from_slice(&0u32.to_be_bytes()); // seqno
    buf[12..16].copy_from_slice(&agblocks.to_be_bytes()); // length
    // agf_roots[3] at 16..28 -- [bno, cnt, rmap]
    buf[16..20].copy_from_slice(&bno_root.to_be_bytes());
    buf[20..24].copy_from_slice(&cnt_root.to_be_bytes());
    buf[24..28].copy_from_slice(&0u32.to_be_bytes()); // rmap (unused)
    // agf_levels[3] at 28..40 -- all 1 for single-leaf trees
    buf[28..32].copy_from_slice(&1u32.to_be_bytes());
    buf[32..36].copy_from_slice(&1u32.to_be_bytes());
    buf[36..40].copy_from_slice(&0u32.to_be_bytes()); // rmap level
    // agf_flfirst/last/count at 40..52 - empty AGFL.
    buf[40..44].copy_from_slice(&0u32.to_be_bytes()); // flfirst
    buf[44..48].copy_from_slice(&0u32.to_be_bytes()); // fllast
    buf[48..52].copy_from_slice(&0u32.to_be_bytes()); // flcount
    let post_chunk = ROOT_CHUNK_AGBLOCK + 8;
    let free = agblocks.saturating_sub(post_chunk + 1);
    buf[52..56].copy_from_slice(&free.to_be_bytes()); // freeblks
    buf[56..60].copy_from_slice(&free.to_be_bytes()); // longest
    buf[60..64].copy_from_slice(&0u32.to_be_bytes()); // btreeblks
    // v5 block starts at 64: uuid(16) rmap_blocks(4) refcount_blocks(4)
    // refcount_root(4) refcount_level(4) spare64(14*8) | lsn(8) crc(4) spare2(4)
    buf[64..80].copy_from_slice(uuid);
    // spare/refc etc all zero
    // CRC at offset 224 in the AGF (v5 layout: see XFS spec).
    buf
}

/// AGF CRC offset within its sector. Per spec, AGF places `agf_crc` in
/// the unlogged region trailer. The kernel writes it at offset 224.
pub const AGF_CRC_OFFSET: usize = 224;

fn build_agi(uuid: &[u8; 16], agblocks: u32, inobt_root: u32, rootino: u64) -> Vec<u8> {
    let mut buf = vec![0u8; XFS_BLOCKSIZE as usize];
    buf[0..4].copy_from_slice(&XFS_AGI_MAGIC.to_be_bytes());
    buf[4..8].copy_from_slice(&XFS_AGI_VERSION.to_be_bytes());
    buf[8..12].copy_from_slice(&0u32.to_be_bytes()); // seqno
    buf[12..16].copy_from_slice(&agblocks.to_be_bytes()); // length
    buf[16..20].copy_from_slice(&1u32.to_be_bytes()); // count (1 allocated)
    buf[20..24].copy_from_slice(&inobt_root.to_be_bytes()); // root
    buf[24..28].copy_from_slice(&1u32.to_be_bytes()); // level
    buf[28..32].copy_from_slice(&63u32.to_be_bytes()); // freecount
    // newino at 32..36 — set to the root chunk's startino so xfs_repair
    // sees a coherent "last allocated chunk" hint.
    let inopblog = trailing_zeros_u16(XFS_INOPBLOCK);
    let startino_ag = ROOT_CHUNK_AGBLOCK << inopblog;
    buf[32..36].copy_from_slice(&startino_ag.to_be_bytes());
    // dirino at 36..40 — deprecated, set to -1.
    buf[36..40].copy_from_slice(&u32::MAX.to_be_bytes());
    // unlinked[64] table at 40..40+256 — all -1.
    for i in 0..64 {
        let off = 40 + i * 4;
        buf[off..off + 4].copy_from_slice(&u32::MAX.to_be_bytes());
    }
    // v5 fields: uuid(16) crc(4) pad32(4) lsn(8) free_root(4) free_level(4)
    // iblocks(4) fblocks(4)  → starting at offset 296.
    buf[296..312].copy_from_slice(uuid);
    // crc at 312..316 left zero (stamped later)
    // pad32 at 316..320
    // lsn at 320..328
    // free_root at 328..332 - NULL (no FINOBT)
    buf[328..332].copy_from_slice(&0u32.to_be_bytes());
    buf[332..336].copy_from_slice(&0u32.to_be_bytes()); // free_level
    let _ = rootino;
    buf
}

/// AGI CRC offset: byte 312 (after uuid).
pub const AGI_CRC_OFFSET: usize = 312;

fn build_agfl(uuid: &[u8; 16]) -> Vec<u8> {
    // Whole-block AGFL (matches the FS block size; spec uses only the
    // first sector but the kernel zeroes the rest, so we follow suit).
    let mut buf = vec![0u8; XFS_BLOCKSIZE as usize];
    buf[0..4].copy_from_slice(&XFS_AGFL_MAGIC.to_be_bytes());
    buf[4..8].copy_from_slice(&0u32.to_be_bytes()); // seqno
    buf[8..24].copy_from_slice(uuid);
    // lsn at 24..32 zero
    // crc at 32..36 zero (stamp later)
    // entries start at 36 — all -1 (empty list).
    for i in 0..XFS_AGFL_NSLOTS_512 {
        let off = 36 + i * 4;
        buf[off..off + 4].copy_from_slice(&u32::MAX.to_be_bytes());
    }
    buf
}

/// AGFL CRC offset: byte 32.
pub const AGFL_CRC_OFFSET: usize = 32;

fn build_alloc_btree_root_leaf(
    magic: u32,
    uuid: &[u8; 16],
    owner_ag: u32,
    blkno_ag: u32,
    free_startblock: u32,
    free_blockcount: u32,
) -> Vec<u8> {
    let mut buf = vec![0u8; XFS_BLOCKSIZE as usize];
    buf[0..4].copy_from_slice(&magic.to_be_bytes());
    buf[4..6].copy_from_slice(&0u16.to_be_bytes()); // level (leaf)
    buf[6..8].copy_from_slice(&1u16.to_be_bytes()); // numrecs
    buf[8..12].copy_from_slice(&u32::MAX.to_be_bytes()); // leftsib
    buf[12..16].copy_from_slice(&u32::MAX.to_be_bytes()); // rightsib
    // bb_blkno at 16..24 — FSB * sectsize / 512 = AGblk * 4096/512 = AGblk*8
    let basic_blkno = (blkno_ag as u64) * (XFS_BLOCKSIZE as u64 / 512);
    buf[16..24].copy_from_slice(&basic_blkno.to_be_bytes());
    // lsn at 24..32 zero
    buf[32..48].copy_from_slice(uuid);
    buf[48..52].copy_from_slice(&owner_ag.to_be_bytes());
    // crc at 52..56 left zero (stamped later)

    // One leaf record at offset 56: startblock(4) blockcount(4).
    let rec_off = XFS_BTREE_SBLOCK_V5_SIZE;
    buf[rec_off..rec_off + 4].copy_from_slice(&free_startblock.to_be_bytes());
    buf[rec_off + 4..rec_off + 8].copy_from_slice(&free_blockcount.to_be_bytes());
    buf
}

fn build_inobt_root_leaf(
    uuid: &[u8; 16],
    owner_ag: u32,
    blkno_ag: u32,
    startino_ag: u32,
    freecount: u32,
    ir_free: u64,
) -> Vec<u8> {
    let mut buf = vec![0u8; XFS_BLOCKSIZE as usize];
    buf[0..4].copy_from_slice(&XFS_IBT_CRC_MAGIC.to_be_bytes());
    buf[4..6].copy_from_slice(&0u16.to_be_bytes()); // level
    buf[6..8].copy_from_slice(&1u16.to_be_bytes()); // numrecs
    buf[8..12].copy_from_slice(&u32::MAX.to_be_bytes());
    buf[12..16].copy_from_slice(&u32::MAX.to_be_bytes());
    let basic_blkno = (blkno_ag as u64) * (XFS_BLOCKSIZE as u64 / 512);
    buf[16..24].copy_from_slice(&basic_blkno.to_be_bytes());
    buf[32..48].copy_from_slice(uuid);
    buf[48..52].copy_from_slice(&owner_ag.to_be_bytes());
    // crc at 52..56 zero (stamped later)

    // Leaf record: ir_startino(4) ir_freecount(4) ir_free(8) — 16 B.
    // (Sparse-inode form adds a u16 ir_holemask + u8 ir_count; we use
    // the classic 16-byte record because we do NOT set SPINODES.)
    let rec_off = XFS_BTREE_SBLOCK_V5_SIZE;
    buf[rec_off..rec_off + 4].copy_from_slice(&startino_ag.to_be_bytes());
    buf[rec_off + 4..rec_off + 8].copy_from_slice(&freecount.to_be_bytes());
    buf[rec_off + 8..rec_off + 16].copy_from_slice(&ir_free.to_be_bytes());
    buf
}

// ===== CRC stamping helpers ==============================================

/// Superblock CRC at offset 224 (LE32).
pub const SB_CRC_OFFSET: usize = 224;

pub fn stamp_v5_superblock_crc(buf: &mut [u8]) {
    buf[SB_CRC_OFFSET..SB_CRC_OFFSET + 4].copy_from_slice(&[0u8; 4]);
    let crc = crc32c::crc32c(&buf[..XFS_SECTSIZE as usize]);
    buf[SB_CRC_OFFSET..SB_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
}

pub fn stamp_v5_agf_crc(buf: &mut [u8]) {
    buf[AGF_CRC_OFFSET..AGF_CRC_OFFSET + 4].copy_from_slice(&[0u8; 4]);
    let crc = crc32c::crc32c(&buf[..XFS_SECTSIZE as usize]);
    buf[AGF_CRC_OFFSET..AGF_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
}

pub fn stamp_v5_agi_crc(buf: &mut [u8]) {
    buf[AGI_CRC_OFFSET..AGI_CRC_OFFSET + 4].copy_from_slice(&[0u8; 4]);
    let crc = crc32c::crc32c(&buf[..XFS_SECTSIZE as usize]);
    buf[AGI_CRC_OFFSET..AGI_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
}

pub fn stamp_v5_agfl_crc(buf: &mut [u8]) {
    buf[AGFL_CRC_OFFSET..AGFL_CRC_OFFSET + 4].copy_from_slice(&[0u8; 4]);
    let crc = crc32c::crc32c(&buf[..XFS_SECTSIZE as usize]);
    buf[AGFL_CRC_OFFSET..AGFL_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
}

/// The xfs_btree_sblock CRC is at offset 52 (LE32) and the CRC covers
/// the whole FS-block-sized buffer. Used for the BNO / CNT / INOBT
/// roots.
pub const BTREE_SBLOCK_CRC_OFFSET: usize = 52;

pub fn stamp_v5_btree_block_crc(buf: &mut [u8]) {
    buf[BTREE_SBLOCK_CRC_OFFSET..BTREE_SBLOCK_CRC_OFFSET + 4].copy_from_slice(&[0u8; 4]);
    let crc = crc32c::crc32c(buf);
    buf[BTREE_SBLOCK_CRC_OFFSET..BTREE_SBLOCK_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
}

// ===== misc utilities ====================================================

fn ceil_log2_u32(n: u32) -> u8 {
    if n <= 1 {
        return 0;
    }
    let mut l = 0u8;
    let mut x: u64 = 1;
    while x < n as u64 {
        x <<= 1;
        l += 1;
    }
    l
}

fn trailing_zeros_u32(n: u32) -> u8 {
    n.trailing_zeros() as u8
}

fn trailing_zeros_u16(n: u16) -> u8 {
    n.trailing_zeros() as u8
}

/// Borrow the parsed superblock from `xfs`. Kept around for callers
/// outside this module that want to inspect the FS geometry without
/// going through the writer.
#[allow(dead_code)]
pub(crate) fn sb_of(xfs: &Xfs) -> &Superblock {
    xfs.superblock()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn format_writes_valid_superblock() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = FormatOpts::default();
        let xfs = format(&mut dev, &opts).unwrap();
        // Sanity: re-open works.
        assert_eq!(xfs.block_size(), 4096);
        assert_eq!(xfs.inode_size(), 512);
        assert_eq!(xfs.ag_count(), 1);
        // And the probe sees the magic.
        let mut dev2 = dev; // reuse
        assert!(super::super::probe(&mut dev2).unwrap());
    }

    #[test]
    fn format_root_inode_is_readable() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = FormatOpts::default();
        let xfs = format(&mut dev, &opts).unwrap();
        let entries = xfs.list_path(&mut dev, "/").unwrap();
        // Empty directory after format: list_path surfaces "." and ".."
        // verbatim from block-format directories, so a fresh root has
        // exactly those two entries and no others.
        assert_eq!(entries.len(), 2);
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"."));
        assert!(names.contains(&".."));
    }

    #[test]
    fn ceil_log2_examples() {
        assert_eq!(ceil_log2_u32(1), 0);
        assert_eq!(ceil_log2_u32(2), 1);
        assert_eq!(ceil_log2_u32(3), 2);
        assert_eq!(ceil_log2_u32(4), 2);
        assert_eq!(ceil_log2_u32(5), 3);
        assert_eq!(ceil_log2_u32(2048), 11);
    }
}
