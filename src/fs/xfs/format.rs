//! XFS v5 image formatter.
//!
//! Lays down a fresh XFS v5 filesystem on a [`BlockDevice`], ready to
//! be populated through [`Xfs::add_file`] / [`Xfs::add_dir`] /
//! [`Xfs::add_symlink`] / [`Xfs::add_device`]. Multi-AG aware: devices
//! larger than [`MULTI_AG_THRESHOLD_BYTES`] (256 MiB) get one
//! 256 MiB AG per 65 536 FS blocks. The formatter writes a 2 MiB
//! "empty clean" internal log in AG 0 so the kernel will mount the
//! image read/write (see [`super::journal`]). Out of scope: realtime
//! sub-volume, reverse-mapping B+tree, reference-count B+tree, sparse
//! inodes, FINOBT.
//!
//! ## On-disk geometry (each AG)
//!
//! AG headers live in the first four 512-byte SECTORS of AG-block 0:
//!
//! ```text
//!   Sector 0 (byte 0)    Superblock           (XFSB)
//!   Sector 1 (byte 512)  AGF                  (XAGF, one sector)
//!   Sector 2 (byte 1024) AGI                  (XAGI, one sector)
//!   Sector 3 (byte 1536) AGFL                 (XAFL, v5 header + free-list u32s)
//! ```
//!
//! AG-blocks 1..6 are reserved for the AG B+tree roots:
//!
//! ```text
//!   Block 4      BNO btree root       (magic AB3B, level 0, 1 leaf record)
//!   Block 5      CNT btree root       (magic AB3C, level 0, 1 leaf record)
//!   Block 6      INOBT root           (magic IAB3, level 0, 1 leaf record)
//! ```
//!
//! In AG 0 only:
//!
//! ```text
//!   Block 8..15      Root inode chunk (64 v3 inodes, 32 KiB)
//!   Block 16         Root dir block   (XDB3, "." + "..")
//!   Block 17..528    Internal journal log (DEFAULT_LOG_BLOCKS = 512)
//!   Block 529..      Free pool (bump-pointer allocator)
//! ```
//!
//! Defaults: 4 KiB FS blocks, 512-byte v5 inodes (the v5 minimum), AG
//! width = 256 MiB. The caller controls just the UUID + label.

use crate::Result;
use crate::block::BlockDevice;

use super::Xfs;
use super::dir::{dahashname, encode_v5_block_dir, stamp_v5_dir_block_crc};
use super::inode::{V3DinodeBuilder, XfsTimestamp, stamp_v3_inode_crc};
use super::journal::{DEFAULT_LOG_BLOCKS, write_empty_log};
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

/// Reserved per-AG metadata blocks. SB/AGF/AGI/AGFL live in sectors
/// 0..3 of AG-block 0; BNO/CNT/INOBT live at AG-blocks 4/5/6. Block 7
/// is unused (AGFL reservation slack).
pub const AG0_METADATA_BLOCKS: u32 = 7;

/// AG-relative block where the root-inode chunk lives. 64 inodes × 512 B
/// = 32 KiB = 8 FS blocks at 4 KiB. Aligned to a multiple of 8 because
/// `sb_inoalignmt` requires inode chunks to be 16 KiB-aligned at minimum
/// (we use 32 KiB for safety). mkfs.xfs puts the root chunk at AG-block
/// 8 for v5 images of this geometry; we match that so xfs_repair's
/// "calculated root inode" heuristic agrees with our layout.
pub const ROOT_CHUNK_AGBLOCK: u32 = 8;

/// AG-relative block where the first root-directory data block lives.
/// Sits immediately after the root inode chunk so its FSB encodes
/// cleanly inside the inode's single-extent literal area.
pub const ROOT_DIR_AGBLOCK: u32 = ROOT_CHUNK_AGBLOCK + 8;

/// AG-relative block where the internal journal log starts in AG 0.
/// Sits immediately after the root dir block (i.e. at
/// ROOT_CHUNK_AGBLOCK + 8 + 1 = 17).
pub const LOG_AGBLOCK: u32 = ROOT_DIR_AGBLOCK + 1;

/// Threshold above which [`format()`] starts creating multiple
/// allocation groups. Devices > 256 MiB get one AG per 65 536 FS
/// blocks (256 MiB at 4 KiB). Smaller devices stay single-AG.
pub const MULTI_AG_THRESHOLD_BYTES: u64 = 256 * 1024 * 1024;
/// FS blocks per AG when [`format()`] picks `agcount` automatically.
pub const DEFAULT_AGBLOCKS_PER_AG: u32 = 65_536;

/// Pick the number of allocation groups to lay down on `dev_bytes`.
/// Returns `1` for devices smaller than [`MULTI_AG_THRESHOLD_BYTES`];
/// otherwise picks `max(1, dblocks / DEFAULT_AGBLOCKS_PER_AG)`.
pub fn choose_agcount(dev_bytes: u64) -> u32 {
    if dev_bytes <= MULTI_AG_THRESHOLD_BYTES {
        return 1;
    }
    let dblocks = dev_bytes / (XFS_BLOCKSIZE as u64);
    let count = dblocks / (DEFAULT_AGBLOCKS_PER_AG as u64);
    count.clamp(1, u32::MAX as u64) as u32
}

/// Format a fresh v5 XFS image on `dev` and return an open [`Xfs`].
///
/// The returned handle is ready to accept `add_*` calls; its internal
/// bump-pointer allocator starts at the first AG block past the root
/// inode chunk and grows upward, never freeing.
pub fn format(dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<Xfs> {
    let dev_bytes = dev.total_size();
    let blocksize = XFS_BLOCKSIZE as u64;
    // We need at minimum: the static metadata, the root inode chunk,
    // the log, plus one free block for the root dir block.
    let log_blocks_u64 = DEFAULT_LOG_BLOCKS as u64;
    let min_blocks = LOG_AGBLOCK as u64 + log_blocks_u64 + 1;
    if dev_bytes < blocksize * min_blocks {
        return Err(crate::Error::InvalidArgument(format!(
            "xfs: device of {dev_bytes} bytes is too small to format a fresh XFS image",
        )));
    }
    let total_blocks = (dev_bytes / blocksize) as u32;
    let agcount = choose_agcount(dev_bytes);
    // `agblocks` is the per-AG block count quoted in the superblock.
    // Every AG is exactly this big except possibly the last, which the
    // kernel notices via `agf_length`. The kernel demands that every AG
    // start at a multiple of `agblocks` from byte 0.
    let agblocks = if agcount == 1 {
        total_blocks
    } else {
        // Round-up division: cover `total_blocks` with `agcount` AGs.
        total_blocks.div_ceil(agcount)
    };
    // log2(agblocks), rounded up — used to pack FSBs.
    let agblklog = ceil_log2_u32(agblocks);
    let dblocks = total_blocks as u64;

    // Zero the entire metadata region of AG 0 + the log + a generous
    // head of "future allocations". The rest of every AG (including
    // the metadata sectors of AG 1..agcount-1) gets zeroed on demand
    // as we write each AG below.
    let metadata_end = (LOG_AGBLOCK as u64 + log_blocks_u64 + 1) * blocksize;
    dev.zero_range(0, metadata_end.min(dev_bytes))?;

    let rootino = (ROOT_CHUNK_AGBLOCK as u64) * (XFS_INOPBLOCK as u64);
    // Internal log starts at AG 0 block LOG_AGBLOCK. The on-disk FSB is
    // (ag << agblklog) | agblk = LOG_AGBLOCK (ag = 0).
    let logstart_fsb = LOG_AGBLOCK as u64;
    let logblocks = DEFAULT_LOG_BLOCKS;
    // In AG 0 the free area begins right after the log; everywhere else
    // it begins right after the static metadata.
    let free_pool_ag0 = LOG_AGBLOCK + logblocks;
    let free_pool_other = AG0_METADATA_BLOCKS;

    // -- Build + write the primary superblock -----------------------------
    let sb_buf = build_v5_superblock(
        &opts.uuid,
        &opts.label,
        dblocks,
        agcount,
        agblocks,
        agblklog,
        rootino,
        logstart_fsb,
        logblocks,
        opts.mtime,
    );
    // Stamp CRC last (sb_crc is at offset 224, le32; reseeded later).
    let mut sb_buf = sb_buf;
    stamp_v5_superblock_crc(&mut sb_buf);
    dev.write_at(0, &sb_buf)?;

    // -- Write metadata headers for every AG. -----------------------------
    for ag in 0..agcount {
        let ag_byte = (ag as u64) * (agblocks as u64) * blocksize;
        // The last AG can be short; its `agf_length` should reflect the
        // actual block count we have on disk.
        let this_ag_blocks = if ag == agcount - 1 {
            (total_blocks.saturating_sub(ag * agblocks)).max(1)
        } else {
            agblocks
        };
        let (post_chunk, allocated_inodes, free_inodes_in_ag) = if ag == 0 {
            // Three slots pre-allocated by the formatter (root, rbmino,
            // rsumino) and 61 free in the root inode chunk.
            (free_pool_ag0, 3u32, 61u32)
        } else {
            (free_pool_other, 0u32, 0u32)
        };
        let free_blocks = this_ag_blocks.saturating_sub(post_chunk);

        // AG 1+ still needs the static blocks zeroed so the magic
        // checks below land on a clean slate.
        if ag != 0 {
            let zero_end = ((AG0_METADATA_BLOCKS + 1) as u64) * blocksize;
            dev.zero_range(ag_byte, zero_end.min(dev_bytes - ag_byte))?;
        }

        // Secondary superblocks live at AG-relative block 0 of every
        // AG. The kernel uses them for `xfs_repair` fallback.
        if ag != 0 {
            let mut sb_copy = sb_buf.clone();
            // Re-stamp CRC over the unmodified copy (it already has the
            // primary CRC; the SB content for secondaries is identical
            // except some kernels zero counters — we keep them, since
            // the kernel re-reads the primary at mount).
            stamp_v5_superblock_crc(&mut sb_copy);
            dev.write_at(ag_byte, &sb_copy)?;
        }

        // -- AGF at AG sector 1 (byte 512) -------------------------------
        // The kernel + xfs_db read AG headers at sector-aligned offsets,
        // NOT FS-block-aligned. AG-block 0 holds all four headers packed
        // into its first four sectors (SB/AGF/AGI/AGFL); AG-block 1
        // onward is the btree-root area.
        let mut agf_buf = build_agf(
            &opts.uuid,
            ag,
            this_ag_blocks,
            free_blocks,
            /*bno_root*/ 4,
            /*cnt_root*/ 5,
        );
        stamp_v5_agf_crc(&mut agf_buf);
        dev.write_at(ag_byte + (XFS_SECTSIZE as u64), &agf_buf)?;

        // -- AGI at AG sector 2 (byte 1024) -----------------------------
        let mut agi_buf = build_agi(
            &opts.uuid,
            ag,
            this_ag_blocks,
            allocated_inodes,
            free_inodes_in_ag,
            /*inobt_root*/ 6,
            if ag == 0 {
                ROOT_CHUNK_AGBLOCK << trailing_zeros_u16(XFS_INOPBLOCK)
            } else {
                0
            },
            ag == 0,
        );
        stamp_v5_agi_crc(&mut agi_buf);
        dev.write_at(ag_byte + 2 * (XFS_SECTSIZE as u64), &agi_buf)?;

        // -- AGFL at AG sector 3 (byte 1536) ----------------------------
        let mut agfl_buf = build_agfl(&opts.uuid, ag);
        stamp_v5_agfl_crc(&mut agfl_buf);
        dev.write_at(ag_byte + 3 * (XFS_SECTSIZE as u64), &agfl_buf)?;

        // -- BNO btree root at AG-block 4 -------------------------------
        let mut bno_buf = build_alloc_btree_root_leaf(
            XFS_ABTB_CRC_MAGIC,
            &opts.uuid,
            /*owner_ag*/ ag,
            agblocks,
            /*blkno*/ 4,
            post_chunk,
            free_blocks,
        );
        stamp_v5_btree_block_crc(&mut bno_buf);
        dev.write_at(ag_byte + 4 * blocksize, &bno_buf)?;

        // -- CNT btree root at AG-block 5 -------------------------------
        let mut cnt_buf = build_alloc_btree_root_leaf(
            XFS_ABTC_CRC_MAGIC,
            &opts.uuid,
            /*owner_ag*/ ag,
            agblocks,
            /*blkno*/ 5,
            post_chunk,
            free_blocks,
        );
        stamp_v5_btree_block_crc(&mut cnt_buf);
        dev.write_at(ag_byte + 5 * blocksize, &cnt_buf)?;

        // -- INOBT root at AG-block 6 ----------------------------------
        let inopblog = trailing_zeros_u16(XFS_INOPBLOCK);
        let (startino_ag, freecount, ir_free, numrecs) = if ag == 0 {
            // Slots 0 (root), 1 (rbmino), 2 (rsumino) are allocated;
            // every other slot is free.
            let ir_free = !0b111u64;
            (ROOT_CHUNK_AGBLOCK << inopblog, 61u32, ir_free, 1u16)
        } else {
            // No inode chunks in this AG yet — empty INOBT root.
            (0u32, 0u32, 0u64, 0u16)
        };
        let mut inobt_buf = build_inobt_root_leaf(
            &opts.uuid,
            /*owner_ag*/ ag,
            agblocks,
            /*blkno*/ 6,
            startino_ag,
            freecount,
            ir_free,
            numrecs,
        );
        stamp_v5_btree_block_crc(&mut inobt_buf);
        dev.write_at(ag_byte + 6 * blocksize, &inobt_buf)?;
    }

    // -- Inode chunk: 8 contiguous blocks holding 64 v3 inodes (AG 0) ---
    // Build all 64 inodes as "free" (di_magic 0, mode 0). xfs_repair
    // recognises this pattern. Then overwrite slot 0 with the actual
    // root inode.
    let chunk_byte = (ROOT_CHUNK_AGBLOCK as u64) * blocksize;
    let chunk_bytes = (XFS_INODES_PER_CHUNK as u64) * (XFS_INODESIZE as u64);
    dev.zero_range(chunk_byte, chunk_bytes)?;

    // -- Log area (AG 0, starting at LOG_AGBLOCK) -----------------------
    let log_byte_off = (LOG_AGBLOCK as u64) * blocksize;
    let log_bytes = log_blocks_u64 * blocksize;
    // Zero the log first (caller relies on tail blocks being zero so
    // the kernel sees them as `cycle = 0` past the head).
    dev.zero_range(log_byte_off, log_bytes)?;
    write_empty_log(dev, log_byte_off, log_bytes, &opts.uuid)?;

    // Build a single-block-dir root directory (XDB3) and write it at
    // logical FS block 0 of the root inode. We pack `.` and `..` only
    // (no children yet); add_dir / add_file will overwrite this block
    // and update the inode as entries are appended.
    let dir_block_size = XFS_BLOCKSIZE as usize; // dir_block_log = 0
    let entries: Vec<(String, u64, u8)> = Vec::new(); // only "." and ".." (added implicitly)
    let dir_block_basic_blkno = ((ROOT_DIR_AGBLOCK as u64) * blocksize) / 512;
    let dir_block = encode_v5_block_dir(
        dir_block_size,
        rootino,
        rootino, // parent of root == itself
        &entries,
        &opts.uuid,
        dir_block_basic_blkno,
    )?;
    let dir_block_byte = (ROOT_DIR_AGBLOCK as u64) * blocksize;
    dev.write_at(dir_block_byte, &dir_block)?;

    // Build the root inode in EXTENTS format with one extent mapping
    // logical 0 → physical post_chunk_ag0.
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
        startblock: ROOT_DIR_AGBLOCK as u64,
        blockcount: 1,
        unwritten: false,
    };
    root_inode[176..176 + 16].copy_from_slice(&ext.encode());
    stamp_v3_inode_crc(&mut root_inode);
    let root_byte = chunk_byte; // slot 0
    dev.write_at(root_byte, &root_inode)?;

    // -- rbmino (slot 1) + rsumino (slot 2): empty real-time bitmap +
    // summary inodes. Both are S_IFREG, size 0. They have to exist
    // (allocated state in INOBT, CRC-stamped inode body) because
    // xfs_repair refuses to mount a v5 filesystem where sb_rbmino /
    // sb_rsumino name un-allocated inode slots.
    for (slot, ino_num) in [(1u32, rootino + 1), (2u32, rootino + 2)] {
        let mut buf = V3DinodeBuilder {
            inodesize: XFS_INODESIZE as usize,
            mode: super::inode::S_IFREG | 0o600,
            format: /*EXTENTS*/ 2,
            uid: 0,
            gid: 0,
            nlink: 1,
            atime: ts,
            mtime: ts,
            ctime: ts,
            crtime: ts,
            size: 0,
            nblocks: 0,
            extsize: 0,
            nextents: 0,
            forkoff: 0,
            aformat: 2,
            flags: 0,
            generation: 1,
            di_ino: ino_num,
            uuid: opts.uuid,
        }
        .build();
        stamp_v3_inode_crc(&mut buf);
        let byte = chunk_byte + (slot as u64) * (XFS_INODESIZE as u64);
        dev.write_at(byte, &buf)?;
    }

    // -- Free inode slots (3..63): mkfs.xfs stamps every free slot
    //    with a "null" v3 inode (di_magic = IN, version = 3, mode 0,
    //    di_next_unlinked = -1, valid CRC). xfs_repair validates the
    //    CRC even on free slots, so we follow the same convention.
    for slot in 3u32..64u32 {
        let mut buf = vec![0u8; XFS_INODESIZE as usize];
        buf[0..2].copy_from_slice(&super::inode::XFS_DINODE_MAGIC.to_be_bytes());
        buf[4] = 3; // di_version
        // di_next_unlinked at 96..100 = NULL (-1)
        buf[96..100].copy_from_slice(&u32::MAX.to_be_bytes());
        // di_ino at 152..160
        let ino_num = rootino + (slot as u64);
        buf[152..160].copy_from_slice(&ino_num.to_be_bytes());
        buf[160..176].copy_from_slice(&opts.uuid);
        stamp_v3_inode_crc(&mut buf);
        let byte = chunk_byte + (slot as u64) * (XFS_INODESIZE as u64);
        dev.write_at(byte, &buf)?;
    }

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

#[allow(clippy::too_many_arguments)]
fn build_v5_superblock(
    uuid: &[u8; 16],
    label: &[u8; 12],
    dblocks: u64,
    agcount: u32,
    agblocks: u32,
    agblklog: u8,
    rootino: u64,
    logstart_fsb: u64,
    logblocks: u32,
    _mtime: u32,
) -> Vec<u8> {
    // The superblock occupies a single sector (512 B) of an FS block.
    let mut buf = vec![0u8; XFS_BLOCKSIZE as usize];

    buf[0..4].copy_from_slice(&super::superblock::XFS_SB_MAGIC.to_be_bytes());
    buf[4..8].copy_from_slice(&XFS_BLOCKSIZE.to_be_bytes());
    buf[8..16].copy_from_slice(&dblocks.to_be_bytes());
    // sb_rblocks/rextents — zero (no realtime).
    buf[32..48].copy_from_slice(uuid);
    // sb_logstart — internal log starts at this FSB (0 means no
    // internal log, but we always carve one out today).
    buf[48..56].copy_from_slice(&logstart_fsb.to_be_bytes());
    buf[56..64].copy_from_slice(&rootino.to_be_bytes());
    // sb_rbmino, sb_rsumino — XFS demands these inode numbers be the
    // next two slots after the root inode, even when realtime is
    // disabled (rblocks = 0). The inodes themselves need not actually
    // be allocated; xfs_db / kernel only validate that the number
    // values are inside the inode-space (ag/blk/slot decoding).
    buf[64..72].copy_from_slice(&(rootino + 1).to_be_bytes());
    buf[72..80].copy_from_slice(&(rootino + 2).to_be_bytes());
    // sb_rextsize — 1 (in FS blocks). xfs_repair's "geometry sanity
    // check" rejects rextsize=0 even when realtime is unused.
    buf[80..84].copy_from_slice(&1u32.to_be_bytes());
    // The two inode slots adjacent to rootino must report as
    // allocated; we mark them used in the INOBT bitmap below by
    // tweaking ir_free, and we lay down empty (zero-size) v3 inodes
    // at slots 1 and 2 of the root chunk so kernel verifiers find a
    // consistent picture.
    buf[84..88].copy_from_slice(&agblocks.to_be_bytes());
    buf[88..92].copy_from_slice(&agcount.to_be_bytes());
    // sb_rbmblocks — zero
    // sb_logblocks — internal log size in FS blocks
    buf[96..100].copy_from_slice(&logblocks.to_be_bytes());
    // sb_versionnum: v5 + the flag set mkfs.xfs writes today
    // (0xB4A5). Bits: MOREBITS(0x8000) | LOGV2BIT(0x4000) | DIRV2BIT
    // (0x2000) | EXTFLGBIT(0x1000) | SECTORBIT(0x0400) | ALIGNBIT
    // (0x0080) | NLINKBIT(0x0020) + low nibble 5. ATTRBIT (0x0010)
    // is intentionally NOT set on v5 — xattrs go through ATTR2 in
    // features2 instead. Match mkfs exactly so its verifier accepts
    // our image.
    buf[100..102].copy_from_slice(&0xb4a5u16.to_be_bytes());
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
    // sb_icount counts inodes provisioned (chunks × 64), not just
    // those in use. xfs_repair derives icount the same way and would
    // flag a mismatch otherwise.
    buf[128..136].copy_from_slice(&64u64.to_be_bytes());
    buf[136..144].copy_from_slice(&61u64.to_be_bytes());
    // Total free data blocks: AG 0 has (agblocks - LOG_END) free
    // (everything past the log is free); the remaining AGs have
    // (agblocks - AG0_METADATA_BLOCKS) each.
    let ag0_free = agblocks.saturating_sub(LOG_AGBLOCK + logblocks) as u64;
    let other_free = agblocks.saturating_sub(AG0_METADATA_BLOCKS) as u64;
    let total_free = ag0_free + other_free * ((agcount as u64).saturating_sub(1));
    buf[144..152].copy_from_slice(&total_free.to_be_bytes());
    // sb_uquotino / sb_gquotino — zero when quotas are disabled, as
    // mkfs.xfs writes. (Older docs say -1, but the verifier wants 0.)
    buf[160..168].copy_from_slice(&0u64.to_be_bytes());
    buf[168..176].copy_from_slice(&0u64.to_be_bytes());
    // sb_qflags (176..178) zero
    // sb_flags (178) - XFS_SBF_READONLY would be useful but xfs_repair
    // rejects images that can never be mounted RW.
    // sb_shared_vn (179) zero
    // sb_inoalignmt (180..184) = inode chunk alignment in FS blocks.
    // xfs_repair calculates the expected value as
    // `XFS_INODE_BIG_CLUSTER_SIZE / blocksize` where
    // XFS_INODE_BIG_CLUSTER_SIZE = 16 KiB. For our 4 KiB blocks that's
    // 4 — NOT the chunk size (which is 32 KiB / 8 blocks). Any other
    // value triggers "inconsistent inode alignment" in xfs_repair.
    buf[180..184].copy_from_slice(&4u32.to_be_bytes());
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
    // sb_pquotino (232..240) — zero when projid quotas are disabled.
    buf[232..240].copy_from_slice(&0u64.to_be_bytes());
    // sb_lsn (240..248) zero
    // sb_meta_uuid (248..264) - copy of sb_uuid since META_UUID not set.
    buf[248..264].copy_from_slice(uuid);
    buf
}

fn build_agf(
    uuid: &[u8; 16],
    ag: u32,
    ag_length_blocks: u32,
    free_blocks: u32,
    bno_root: u32,
    cnt_root: u32,
) -> Vec<u8> {
    let mut buf = vec![0u8; XFS_SECTSIZE as usize];
    buf[0..4].copy_from_slice(&XFS_AGF_MAGIC.to_be_bytes());
    buf[4..8].copy_from_slice(&XFS_AGF_VERSION.to_be_bytes());
    buf[8..12].copy_from_slice(&ag.to_be_bytes()); // seqno
    buf[12..16].copy_from_slice(&ag_length_blocks.to_be_bytes()); // length
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
    buf[52..56].copy_from_slice(&free_blocks.to_be_bytes()); // freeblks
    buf[56..60].copy_from_slice(&free_blocks.to_be_bytes()); // longest
    buf[60..64].copy_from_slice(&0u32.to_be_bytes()); // btreeblks
    // v5 block starts at 64: uuid(16) rmap_blocks(4) refcount_blocks(4)
    // refcount_root(4) refcount_level(4) spare64(14*8) | lsn(8) crc(4) spare2(4)
    buf[64..80].copy_from_slice(uuid);
    // spare/refc etc all zero
    // CRC at offset 224 in the AGF (v5 layout: see XFS spec).
    buf
}

/// AGF CRC offset within its sector. Per the v5 `struct xfs_agf`
/// layout:
/// `[..208]` user fields, `[208..216]` agf_lsn (u64), `[216..220]`
/// agf_crc (le32). mkfs.xfs lays it down at byte 216 of the AGF
/// sector and we follow suit.
pub const AGF_CRC_OFFSET: usize = 216;

#[allow(clippy::too_many_arguments)]
fn build_agi(
    uuid: &[u8; 16],
    ag: u32,
    ag_length_blocks: u32,
    allocated_inodes: u32,
    free_inodes: u32,
    inobt_root: u32,
    newino_hint: u32,
    has_chunks: bool,
) -> Vec<u8> {
    let mut buf = vec![0u8; XFS_SECTSIZE as usize];
    buf[0..4].copy_from_slice(&XFS_AGI_MAGIC.to_be_bytes());
    buf[4..8].copy_from_slice(&XFS_AGI_VERSION.to_be_bytes());
    buf[8..12].copy_from_slice(&ag.to_be_bytes()); // seqno
    buf[12..16].copy_from_slice(&ag_length_blocks.to_be_bytes()); // length
    buf[16..20].copy_from_slice(&(allocated_inodes + free_inodes).to_be_bytes()); // count
    buf[20..24].copy_from_slice(&inobt_root.to_be_bytes()); // root
    // INOBT level is 1 (single leaf) when there's a root chunk; we keep
    // it at 1 even when empty because the AGI's `root` still points to a
    // valid (zero-record) leaf block.
    let _ = has_chunks;
    buf[24..28].copy_from_slice(&1u32.to_be_bytes()); // level
    buf[28..32].copy_from_slice(&free_inodes.to_be_bytes()); // freecount
    // newino at 32..36 — set to the chunk hint (or -1 if no chunks).
    if has_chunks {
        buf[32..36].copy_from_slice(&newino_hint.to_be_bytes());
    } else {
        buf[32..36].copy_from_slice(&u32::MAX.to_be_bytes());
    }
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
    buf
}

/// AGI CRC offset: byte 312 (after uuid).
pub const AGI_CRC_OFFSET: usize = 312;

fn build_agfl(uuid: &[u8; 16], ag: u32) -> Vec<u8> {
    // Sector-sized AGFL (the spec uses one sector for the AG free
    // list; the AGFL slots fill the remaining space after the 36-byte
    // v5 header).
    let mut buf = vec![0u8; XFS_SECTSIZE as usize];
    buf[0..4].copy_from_slice(&XFS_AGFL_MAGIC.to_be_bytes());
    buf[4..8].copy_from_slice(&ag.to_be_bytes()); // seqno
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

#[allow(clippy::too_many_arguments)]
fn build_alloc_btree_root_leaf(
    magic: u32,
    uuid: &[u8; 16],
    owner_ag: u32,
    agblocks: u32,
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
    // bb_blkno at 16..24 — ABSOLUTE basic block number (= device byte
    // offset / 512). Computed as
    // `(ag * agblocks + blkno_ag) * (blocksize / 512)`.
    let basic_blkno =
        ((owner_ag as u64) * (agblocks as u64) + blkno_ag as u64) * (XFS_BLOCKSIZE as u64 / 512);
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

#[allow(clippy::too_many_arguments)]
fn build_inobt_root_leaf(
    uuid: &[u8; 16],
    owner_ag: u32,
    agblocks: u32,
    blkno_ag: u32,
    startino_ag: u32,
    freecount: u32,
    ir_free: u64,
    numrecs: u16,
) -> Vec<u8> {
    let mut buf = vec![0u8; XFS_BLOCKSIZE as usize];
    buf[0..4].copy_from_slice(&XFS_IBT_CRC_MAGIC.to_be_bytes());
    buf[4..6].copy_from_slice(&0u16.to_be_bytes()); // level
    buf[6..8].copy_from_slice(&numrecs.to_be_bytes());
    buf[8..12].copy_from_slice(&u32::MAX.to_be_bytes());
    buf[12..16].copy_from_slice(&u32::MAX.to_be_bytes());
    let basic_blkno =
        ((owner_ag as u64) * (agblocks as u64) + blkno_ag as u64) * (XFS_BLOCKSIZE as u64 / 512);
    buf[16..24].copy_from_slice(&basic_blkno.to_be_bytes());
    buf[32..48].copy_from_slice(uuid);
    buf[48..52].copy_from_slice(&owner_ag.to_be_bytes());
    // crc at 52..56 zero (stamped later)

    // Leaf record: ir_startino(4) ir_freecount(4) ir_free(8) — 16 B.
    // (Sparse-inode form adds a u16 ir_holemask + u8 ir_count; we use
    // the classic 16-byte record because we do NOT set SPINODES.)
    if numrecs > 0 {
        let rec_off = XFS_BTREE_SBLOCK_V5_SIZE;
        buf[rec_off..rec_off + 4].copy_from_slice(&startino_ag.to_be_bytes());
        buf[rec_off + 4..rec_off + 8].copy_from_slice(&freecount.to_be_bytes());
        buf[rec_off + 8..rec_off + 16].copy_from_slice(&ir_free.to_be_bytes());
    }
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

    #[test]
    fn choose_agcount_thresholds() {
        // Devices ≤256 MiB stay single-AG.
        assert_eq!(choose_agcount(0), 1);
        assert_eq!(choose_agcount(64 * 1024 * 1024), 1);
        assert_eq!(choose_agcount(256 * 1024 * 1024), 1);
        // Above the threshold: one AG per 65 536 FS blocks (256 MiB).
        assert_eq!(choose_agcount(512 * 1024 * 1024), 2);
        assert_eq!(choose_agcount(1024 * 1024 * 1024), 4);
    }

    #[test]
    fn format_writes_journal_stub() {
        let mut dev = MemoryBackend::new(64 * 1024 * 1024);
        let opts = FormatOpts::default();
        let xfs = format(&mut dev, &opts).unwrap();
        // sb_logstart and sb_logblocks must be non-zero after format.
        let mut sb = [0u8; 512];
        dev.read_at(0, &mut sb).unwrap();
        let logstart = u64::from_be_bytes(sb[48..56].try_into().unwrap());
        let logblocks = u32::from_be_bytes(sb[96..100].try_into().unwrap());
        assert!(logstart > 0);
        assert!(logblocks >= super::super::journal::DEFAULT_LOG_BLOCKS);
        // The first 4 bytes at sb_logstart's byte offset must be the
        // log-record magic 0xFEEDBABE.
        let log_byte = logstart * 4096;
        let mut hdr = [0u8; 4];
        dev.read_at(log_byte, &mut hdr).unwrap();
        assert_eq!(u32::from_be_bytes(hdr), 0xFEED_BABE);
        let _ = xfs;
    }
}
