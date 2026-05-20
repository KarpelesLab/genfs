//! XFS internal journal — clean-unmount stub plus a single-transaction
//! inode-update Path A writer + replay.
//!
//! Two paths coexist in this module:
//!
//! 1. **Clean-log stub** ([`write_empty_log`]) — lays down the
//!    "freshly unmounted" record at log block 0. Used by [`format`]
//!    after creating a brand-new image, and after a successful
//!    checkpoint to restore the log to its clean state.
//!
//! 2. **Path A: real transactions** ([`write_inode_update_transaction`]
//!    / [`replay_log`]) — writes a write-ahead log record describing
//!    a single inode update, followed by a commit op. A subsequent
//!    crash (i.e. the inode never reaches disk) is detected by
//!    [`replay_log`], which re-applies the inode bytes from the
//!    record payload, then restamps the unmount record so the log is
//!    clean again.
//!
//! The kernel's replay code expects host-endian payloads inside log
//! items. We choose **little-endian** unconditionally (the only host
//! byte order this crate targets), and stamp `h_fmt =
//! XLOG_FMT_LINUX_LE`. Cross-endian replay is not supported.
//!
//! ## On-disk layout (per the XFS PDF "Journaling Log" section and the
//! kernel's `xlog_rec_header` layout)
//!
//! Each log "basic block" is 512 bytes. The first 4 bytes of every BB
//! are reserved for cycle-stamping (the kernel overwrites them with
//! the current cycle when it writes the log); the original 4 bytes are
//! preserved in `h_cycle_data[]` of the log-record header.
//!
//! Record header at BB 0 (big-endian on disk):
//!
//! ```text
//!     0    4   h_magicno    = 0xFEEDBABE
//!     4    4   h_cycle      = 1
//!     8    4   h_version    = 2 (LOGV2)
//!    12    4   h_len        = bytes of payload after header (≥ BB size)
//!    16    8   h_lsn        = (cycle << 32) | block        (cycle=1, block=0)
//!    24    8   h_tail_lsn   = same as h_lsn for a clean log
//!    32    4   h_crc        = 0 (not validated for empty/clean log)
//!    36    4   h_prev_block = 0xFFFFFFFF (none)
//!    40    4   h_num_logops = 1
//!    44  256   h_cycle_data[64] — preserved-first-4-bytes per BB
//!   300    4   h_fmt        = 1 (XLOG_FMT_LINUX_LE)
//!   304   16   h_fs_uuid
//!   320    4   h_size       = 32768
//! ```
//!
//! Total header = 324 bytes; we round h_len up to one BB (512).
//!
//! At BB 1 we lay down an XLOG_UNMOUNT_TYPE op:
//!
//! ```text
//!   xlog_op_header (12 bytes, big-endian on disk):
//!     0    4   oh_tid       (any, we use 1)
//!     4    4   oh_len       = 8 (unmount payload size)
//!     8    1   oh_clientid  = XFS_LOG (0xAA) -- "log client"
//!     9    1   oh_flags     = XLOG_COMMIT_TRANS | XLOG_UNMOUNT_TRANS (0x20)
//!    10    2   oh_res2      = 0
//!   --- payload (8 bytes, little-endian) ---
//!    12    2   um_magic     = XLOG_UNMOUNT_TYPE (0x556e = "Un")
//!    14    6   pad
//! ```
//!
//! The kernel only validates the unmount magic; the rest of the BB is
//! zero. The first 4 bytes of BB 1 are overwritten with the cycle
//! stamp (1), and the original first 4 bytes (the oh_tid) are stored
//! in `h_cycle_data[0]` of the record header.
//!
//! ## Limitations
//!
//! - We do NOT compute `h_crc`. The kernel's policy is to log a
//!   `Corruption` warning if the CRC is wrong, but to still accept the
//!   log when `h_num_logops == 1` and the only op is an unmount
//!   record. (Real mounts succeed in practice.) A future revision can
//!   add crc32c stamping.
//! - We always write `h_size = 32768`, `h_version = 2`, `h_fmt = 1`
//!   (little-endian Linux) — the same values mkfs.xfs emits today.
//! - We never wrap-around. The log is single-pass.

use crate::Result;
use crate::block::BlockDevice;

/// XFS log basic-block size (always 512, independent of `sb_blocksize`).
pub const BBSIZE: u64 = 512;

/// `XLOG_HEADER_MAGIC_NUM` — first 4 bytes of every log record header.
pub const XLOG_HEADER_MAGIC_NUM: u32 = 0xFEED_BABE;

/// `XLOG_VERSION_2` — v2 log layout (LOGV2BIT must be set in sb_versionnum).
pub const XLOG_VERSION_2: u32 = 2;

/// `XLOG_FMT_LINUX_LE` — log format byte. mkfs.xfs writes this on any
/// little-endian host (which is everything we care about today).
pub const XLOG_FMT_LINUX_LE: u32 = 1;

/// Default in-memory iclog buffer size that mkfs.xfs records in
/// `h_size`. The kernel will not log a warning at any of {16k, 32k,
/// 64k, 128k, 256k}.
pub const XLOG_DEFAULT_H_SIZE: u32 = 32 * 1024;

/// `XLOG_UNMOUNT_TYPE` — 2-byte magic that immediately follows the op
/// header. Stored little-endian on disk regardless of the rest of the
/// log because `h_fmt = XLOG_FMT_LINUX_LE`.
pub const XLOG_UNMOUNT_TYPE: u16 = 0x556e; // "Un"

/// `XFS_TRANSACTION` clientid that mkfs.xfs writes for the unmount op.
pub const XFS_LOG_CLIENTID: u8 = 0xAA;

/// `XLOG_COMMIT_TRANS | XLOG_UNMOUNT_TRANS` flags.
pub const XLOG_OP_UNMOUNT_FLAGS: u8 = 0x20;

/// Default log size in FS blocks chosen by [`format`] when laying out a
/// fresh image. 512 blocks at 4 KiB = 2 MiB — well below mkfs.xfs's
/// recommended 10 MiB minimum but enough that the kernel will accept
/// the image as cleanly unmounted. Real production images should be
/// formatted with a larger log.
pub const DEFAULT_LOG_BLOCKS: u32 = 512;

/// Write the "empty unmount" log content into `dev` starting at
/// `log_byte_off` and covering `log_bytes` bytes. The region MUST be
/// pre-zeroed (we only stamp the bytes we know about). `uuid` is the
/// volume's `sb_uuid`.
pub fn write_empty_log(
    dev: &mut dyn BlockDevice,
    log_byte_off: u64,
    log_bytes: u64,
    uuid: &[u8; 16],
) -> Result<()> {
    if log_bytes < 2 * BBSIZE {
        return Err(crate::Error::InvalidArgument(format!(
            "xfs: log {log_bytes} bytes too small (need at least 1024)"
        )));
    }
    // Zero the first 2 BBs (record header + unmount op) — the caller is
    // responsible for the rest. We touch nothing past BB 1.
    let mut hdr_bb = vec![0u8; BBSIZE as usize];
    let mut op_bb = vec![0u8; BBSIZE as usize];

    // --- record header at BB 0 -----------------------------------------
    hdr_bb[0..4].copy_from_slice(&XLOG_HEADER_MAGIC_NUM.to_be_bytes());
    hdr_bb[4..8].copy_from_slice(&1u32.to_be_bytes()); // h_cycle
    hdr_bb[8..12].copy_from_slice(&XLOG_VERSION_2.to_be_bytes());
    // h_len: bytes of payload (after this 512-B header) — we write 1
    // BB of payload (the unmount op + its data fits in one BB).
    hdr_bb[12..16].copy_from_slice(&(BBSIZE as u32).to_be_bytes());
    // h_lsn = cycle<<32 | block. Both zero for cycle 1 block 0 packed:
    let h_lsn = 1u64 << 32;
    hdr_bb[16..24].copy_from_slice(&h_lsn.to_be_bytes());
    // h_tail_lsn = h_lsn (log is clean — head == tail).
    hdr_bb[24..32].copy_from_slice(&h_lsn.to_be_bytes());
    // h_crc at [32..36] — left zero (kernel logs a warning, accepts mount).
    hdr_bb[36..40].copy_from_slice(&u32::MAX.to_be_bytes()); // h_prev_block
    hdr_bb[40..44].copy_from_slice(&1u32.to_be_bytes()); // h_num_logops
    // h_cycle_data[0..64] at [44..300]: each slot holds the original
    // first 4 bytes of the corresponding BB inside this record. We only
    // record one record covering BB 1 (the unmount op). Its first 4
    // bytes are the op_header.oh_tid which we set to 1 below.
    hdr_bb[44..48].copy_from_slice(&1u32.to_be_bytes());
    // Remaining h_cycle_data entries [48..300] left zero (the rest of
    // the record's BBs are zero-initialised so their cycle_data is also
    // zero).
    hdr_bb[300..304].copy_from_slice(&XLOG_FMT_LINUX_LE.to_be_bytes());
    hdr_bb[304..320].copy_from_slice(uuid);
    hdr_bb[320..324].copy_from_slice(&XLOG_DEFAULT_H_SIZE.to_be_bytes());

    dev.write_at(log_byte_off, &hdr_bb)?;

    // --- unmount op header + payload at BB 1 ---------------------------
    //
    // After cycle-stamping by the kernel, bytes [0..4] become the cycle
    // (== 1). We write the cycle stamp here directly so a reader that
    // never went through the kernel sees a consistent picture; the
    // *original* oh_tid (= 1) is preserved in hdr.h_cycle_data[0]
    // above.
    op_bb[0..4].copy_from_slice(&1u32.to_be_bytes()); // cycle stamp
    op_bb[4..8].copy_from_slice(&8u32.to_be_bytes()); // oh_len = 8 (unmount payload)
    op_bb[8] = XFS_LOG_CLIENTID;
    op_bb[9] = XLOG_OP_UNMOUNT_FLAGS;
    op_bb[10..12].copy_from_slice(&0u16.to_be_bytes()); // oh_res2
    // payload at [12..]: 2 bytes XLOG_UNMOUNT_TYPE (little-endian per
    // h_fmt), the rest of the op is zero.
    op_bb[12..14].copy_from_slice(&XLOG_UNMOUNT_TYPE.to_le_bytes());

    dev.write_at(log_byte_off + BBSIZE, &op_bb)?;
    Ok(())
}

// ======================================================================
// Path A: real log transactions (write-ahead + replay).
// ======================================================================
//
// The XFS log carries variable-length records. Each record starts with
// an `xlog_rec_header_t` (the 324-byte struct already used above for the
// unmount stub), followed by one or more `xlog_op_header_t` operations
// concatenated back-to-back. The op header looks like:
//
// ```text
//   xlog_op_header_t (12 bytes, big-endian on disk):
//     0    4   oh_tid           transaction id
//     4    4   oh_len           number of payload bytes that follow
//     8    1   oh_clientid      XFS_TRANSACTION (0x69)
//     9    1   oh_flags         XLOG_START_TRANS / XLOG_COMMIT_TRANS / 0
//    10    2   oh_res2          padding
//     (payload of `oh_len` bytes, host-endian)
// ```
//
// The transaction we write describes a single inode update:
//
//   op 1 — xfs_trans_header  (XLOG_START_TRANS flag, 16-byte payload)
//   op 2 — xfs_inode_log_format_64 (56-byte payload, ilf_size = 2)
//   op 3 — full inode buffer copy (XFS_INODESIZE = 256 bytes, host LE)
//   op 4 — commit op (XLOG_COMMIT_TRANS flag, no payload)
//
// The total payload, including its op headers, must fit in one BB so
// that we keep the cycle-stamping book-keeping simple. 4 × 12 + 16 +
// 56 + 256 = 376 bytes, well below the 512-byte BB limit.

/// `oh_clientid` for ordinary metadata transactions (per the XFS
/// "Log Operations" section).
pub const XFS_TRANSACTION_CLIENTID: u8 = 0x69;

/// `oh_flags` — XLOG_START_TRANS.
pub const XLOG_START_TRANS: u8 = 0x01;

/// `oh_flags` — XLOG_COMMIT_TRANS.
pub const XLOG_COMMIT_TRANS: u8 = 0x02;

/// Magic for an `xfs_trans_header` payload (host-endian "TRAN").
pub const XFS_TRANS_HEADER_MAGIC: u32 = 0x5452_414e;

/// Inode-log-item magic for `xfs_inode_log_format_64.ilf_type`.
/// Stored in **host** byte order per the on-disk spec.
pub const XFS_LI_INODE: u16 = 0x123b;

/// `ilf_fields` — log the inode core.
pub const XFS_ILOG_CORE: u32 = 0x0001;
/// `ilf_fields` — log the data fork extent list.
pub const XFS_ILOG_DEXT: u32 = 0x0004;

/// Bytes of one `xlog_op_header`.
const OP_HDR_BYTES: usize = 12;

/// Bytes of one `xfs_trans_header` payload (16).
const TRANS_HDR_BYTES: usize = 16;

/// Bytes of one `xfs_inode_log_format_64` payload (56).
const INODE_LOG_FORMAT_BYTES: usize = 56;

/// Encode + write a single-inode-update transaction starting at log
/// byte offset `log_byte_off`. The record header is laid down at BB 0
/// (overwriting the unmount stub); the ops + their payloads follow at
/// BB 1.
///
/// Parameters:
///
/// - `tid`: transaction id (any non-zero u32).
/// - `cycle`: log cycle number. Must be > 0; the writer also uses it
///   as the cycle stamp for every BB it touches.
/// - `inode_ino`: absolute inode number being updated.
/// - `inode_disk_byte`: byte offset on the device where the inode core
///   lives (replay copies the payload back here).
/// - `inode_bytes`: full inode buffer (XFS_INODESIZE bytes) to log.
/// - `uuid`: `sb_uuid` (stamped into the record header).
///
/// After the call, the log is in the "dirty / pending replay" state.
/// A subsequent [`replay_log`] call applies the inode bytes back, then
/// rewrites the unmount stub via [`write_empty_log`].
#[allow(clippy::too_many_arguments)]
pub fn write_inode_update_transaction(
    dev: &mut dyn BlockDevice,
    log_byte_off: u64,
    log_bytes: u64,
    tid: u32,
    cycle: u32,
    inode_ino: u64,
    inode_disk_byte: u64,
    inode_bytes: &[u8],
    uuid: &[u8; 16],
) -> Result<()> {
    // Total payload size (concatenated ops + their headers):
    //   4 op headers (one each for start/inode-format/inode-buffer/commit)
    //   + xfs_trans_header (16) + inode_log_format (56) + inode bytes.
    let payload_len =
        4 * OP_HDR_BYTES + TRANS_HDR_BYTES + INODE_LOG_FORMAT_BYTES + inode_bytes.len();
    // BB-align the payload (the record header's h_len is in bytes but
    // every BB needs its own cycle-stamp slot).
    let payload_bbs = (payload_len as u64).div_ceil(BBSIZE) as usize;
    let payload_capacity = payload_bbs * BBSIZE as usize;
    let log_bb_capacity = log_bytes / BBSIZE;
    // 1 BB for the record header + payload_bbs BBs for ops.
    if log_bb_capacity < (1 + payload_bbs as u64) {
        return Err(crate::Error::InvalidArgument(format!(
            "xfs: log {log_bytes} bytes too small for transaction ({payload_bbs}+1 BBs needed)"
        )));
    }
    if payload_bbs > 64 {
        // h_cycle_data has 64 slots in the v2 layout we emit.
        return Err(crate::Error::Unsupported(format!(
            "xfs: log transaction payload {payload_bbs} BBs exceeds h_cycle_data[] slot count"
        )));
    }

    // Build a contiguous payload buffer covering payload_bbs BBs.
    let mut payload = vec![0u8; payload_capacity];
    let mut cursor: usize = 0;

    // ---- op 1: start-trans + xfs_trans_header payload --------------
    write_op_header(
        &mut payload[cursor..cursor + OP_HDR_BYTES],
        tid,
        TRANS_HDR_BYTES as u32,
        XLOG_START_TRANS,
    );
    cursor += OP_HDR_BYTES;
    // xfs_trans_header (host-endian per spec):
    //   th_magic   (u32) = 0x5452414e
    //   th_type    (u32) = XFS_TRANS_FSYNC_TS (39) — any plausible value
    //   th_tid     (i32) = tid
    //   th_num_items(u32) = 1 (one log item)
    payload[cursor..cursor + 4].copy_from_slice(&XFS_TRANS_HEADER_MAGIC.to_le_bytes());
    payload[cursor + 4..cursor + 8].copy_from_slice(&39u32.to_le_bytes());
    payload[cursor + 8..cursor + 12].copy_from_slice(&tid.to_le_bytes());
    payload[cursor + 12..cursor + 16].copy_from_slice(&1u32.to_le_bytes());
    cursor += TRANS_HDR_BYTES;

    // ---- op 2: inode-log-format payload (56 bytes, host-endian) ----
    write_op_header(
        &mut payload[cursor..cursor + OP_HDR_BYTES],
        tid,
        INODE_LOG_FORMAT_BYTES as u32,
        0,
    );
    cursor += OP_HDR_BYTES;
    //   ilf_type      (u16) = 0x123b
    //   ilf_size      (u16) = 2 (this op + the inode buffer op)
    //   ilf_fields    (u32) = XFS_ILOG_CORE | XFS_ILOG_DEXT
    //   ilf_asize     (u16) = 0
    //   ilf_dsize     (u16) = 0
    //   ilf_pad       (u32) = 0
    //   ilf_ino       (u64) = inode_ino
    //   ilf_u         (16 bytes) = inode_disk_byte (we repurpose the
    //                  union to record where to replay to)
    //   ilf_blkno     (i64) = inode_disk_byte / 512 (sector address)
    //   ilf_len       (i32) = ceil(inode_bytes.len() / 512)
    //   ilf_boffset   (i32) = inode_disk_byte % 512
    payload[cursor..cursor + 2].copy_from_slice(&XFS_LI_INODE.to_le_bytes());
    payload[cursor + 2..cursor + 4].copy_from_slice(&2u16.to_le_bytes());
    payload[cursor + 4..cursor + 8].copy_from_slice(&(XFS_ILOG_CORE | XFS_ILOG_DEXT).to_le_bytes());
    payload[cursor + 8..cursor + 10].copy_from_slice(&0u16.to_le_bytes());
    payload[cursor + 10..cursor + 12].copy_from_slice(&0u16.to_le_bytes());
    payload[cursor + 12..cursor + 16].copy_from_slice(&0u32.to_le_bytes());
    payload[cursor + 16..cursor + 24].copy_from_slice(&inode_ino.to_le_bytes());
    // ilf_u (16 bytes): stash the absolute disk-byte target here. The
    // kernel uses it as a UUID, but our own replay reads it back as
    // u64 + u64 (target byte + 0 pad).
    payload[cursor + 24..cursor + 32].copy_from_slice(&inode_disk_byte.to_le_bytes());
    payload[cursor + 32..cursor + 40].copy_from_slice(&0u64.to_le_bytes());
    let blkno = (inode_disk_byte / 512) as i64;
    payload[cursor + 40..cursor + 48].copy_from_slice(&blkno.to_le_bytes());
    let len = inode_bytes.len().div_ceil(512) as i32;
    payload[cursor + 48..cursor + 52].copy_from_slice(&len.to_le_bytes());
    let boffset = (inode_disk_byte % 512) as i32;
    payload[cursor + 52..cursor + 56].copy_from_slice(&boffset.to_le_bytes());
    cursor += INODE_LOG_FORMAT_BYTES;

    // ---- op 3: inode buffer payload (host-endian raw bytes) --------
    write_op_header(
        &mut payload[cursor..cursor + OP_HDR_BYTES],
        tid,
        inode_bytes.len() as u32,
        0,
    );
    cursor += OP_HDR_BYTES;
    payload[cursor..cursor + inode_bytes.len()].copy_from_slice(inode_bytes);
    cursor += inode_bytes.len();

    // ---- op 4: commit-trans (no payload) ----------------------------
    write_op_header(
        &mut payload[cursor..cursor + OP_HDR_BYTES],
        tid,
        0,
        XLOG_COMMIT_TRANS,
    );
    cursor += OP_HDR_BYTES;
    debug_assert_eq!(cursor, payload_len);

    // ---- Cycle stamping ----
    // Capture the original first 4 bytes of each payload BB into
    // h_cycle_data[i], then stamp each BB's first 4 bytes with `cycle`.
    let mut h_cycle_data = [0u32; 64];
    for (i, slot) in h_cycle_data.iter_mut().take(payload_bbs).enumerate() {
        let off = i * BBSIZE as usize;
        *slot = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap());
        payload[off..off + 4].copy_from_slice(&cycle.to_be_bytes());
    }

    // ---- Record header (BB 0) ----
    let mut hdr_bb = vec![0u8; BBSIZE as usize];
    hdr_bb[0..4].copy_from_slice(&XLOG_HEADER_MAGIC_NUM.to_be_bytes());
    hdr_bb[4..8].copy_from_slice(&cycle.to_be_bytes());
    hdr_bb[8..12].copy_from_slice(&XLOG_VERSION_2.to_be_bytes());
    // h_len: bytes of payload (BB-aligned).
    hdr_bb[12..16].copy_from_slice(&(payload_capacity as u32).to_be_bytes());
    // h_lsn = (cycle << 32) | block_offset_in_log. Block = 0 (BB 0).
    let h_lsn = (cycle as u64) << 32;
    hdr_bb[16..24].copy_from_slice(&h_lsn.to_be_bytes());
    // h_tail_lsn: oldest record needing replay. We always commit at
    // BB 0 in a single-pass log, so equal to h_lsn.
    hdr_bb[24..32].copy_from_slice(&h_lsn.to_be_bytes());
    // h_crc left zero.
    hdr_bb[36..40].copy_from_slice(&u32::MAX.to_be_bytes()); // h_prev_block
    hdr_bb[40..44].copy_from_slice(&4u32.to_be_bytes()); // h_num_logops = 4
    // h_cycle_data[0..64]: original first 4 bytes per BB.
    for (i, v) in h_cycle_data.iter().enumerate() {
        let off = 44 + i * 4;
        hdr_bb[off..off + 4].copy_from_slice(&v.to_be_bytes());
    }
    hdr_bb[300..304].copy_from_slice(&XLOG_FMT_LINUX_LE.to_be_bytes());
    hdr_bb[304..320].copy_from_slice(uuid);
    hdr_bb[320..324].copy_from_slice(&XLOG_DEFAULT_H_SIZE.to_be_bytes());

    // Persist.
    dev.write_at(log_byte_off, &hdr_bb)?;
    dev.write_at(log_byte_off + BBSIZE, &payload)?;
    Ok(())
}

/// Write a single 12-byte big-endian op header into `buf`.
fn write_op_header(buf: &mut [u8], oh_tid: u32, oh_len: u32, oh_flags: u8) {
    buf[0..4].copy_from_slice(&oh_tid.to_be_bytes());
    buf[4..8].copy_from_slice(&oh_len.to_be_bytes());
    buf[8] = XFS_TRANSACTION_CLIENTID;
    buf[9] = oh_flags;
    buf[10..12].copy_from_slice(&0u16.to_be_bytes());
}

/// Result of [`replay_log`]: what happened during the scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayOutcome {
    /// Log was already clean (unmount record at BB 0). No work done.
    AlreadyClean,
    /// A committed inode-update transaction was found and applied;
    /// the log has been restamped clean.
    Replayed,
    /// A record header was present at BB 0 but no commit op followed —
    /// the transaction is torn / partial; we discarded it and
    /// restamped the log clean.
    PartialDiscarded,
}

/// Scan the log starting at `log_byte_off` for a single committed
/// inode-update transaction; if found, write the logged inode bytes
/// back to their disk address, then re-stamp the unmount record so
/// the log parses as clean again.
///
/// This intentionally handles only the shape of transaction emitted
/// by [`write_inode_update_transaction`] above. Foreign log content
/// (a real kernel log) is rejected with `Error::Unsupported`.
pub fn replay_log(
    dev: &mut dyn BlockDevice,
    log_byte_off: u64,
    log_bytes: u64,
    uuid: &[u8; 16],
) -> Result<ReplayOutcome> {
    if log_bytes < 2 * BBSIZE {
        return Err(crate::Error::InvalidArgument(format!(
            "xfs: log {log_bytes} bytes too small for replay scan"
        )));
    }
    let mut hdr = vec![0u8; BBSIZE as usize];
    dev.read_at(log_byte_off, &mut hdr)?;
    let magic = u32::from_be_bytes(hdr[0..4].try_into().unwrap());
    if magic != XLOG_HEADER_MAGIC_NUM {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: log header magic {magic:#010x} != {XLOG_HEADER_MAGIC_NUM:#010x}"
        )));
    }
    let num_logops = u32::from_be_bytes(hdr[40..44].try_into().unwrap());
    let h_len = u32::from_be_bytes(hdr[12..16].try_into().unwrap());

    // The "clean / unmount" record has h_num_logops == 1. Anything
    // else is a (potentially) dirty record.
    if num_logops == 1 {
        // Confirm via BB 1 that this is the unmount op and exit.
        let mut bb1 = vec![0u8; BBSIZE as usize];
        dev.read_at(log_byte_off + BBSIZE, &mut bb1)?;
        if bb1[8] == XFS_LOG_CLIENTID && (bb1[9] & XLOG_OP_UNMOUNT_FLAGS) != 0 {
            return Ok(ReplayOutcome::AlreadyClean);
        }
        return Err(crate::Error::InvalidImage(
            "xfs: log num_logops==1 but op is not unmount".into(),
        ));
    }

    if num_logops != 4 {
        return Err(crate::Error::Unsupported(format!(
            "xfs: log replay only handles 4-op inode-update records (found {num_logops})"
        )));
    }
    let h_len = h_len as usize;
    if h_len % BBSIZE as usize != 0 || h_len == 0 {
        return Err(crate::Error::Unsupported(format!(
            "xfs: log replay only handles BB-aligned records (h_len = {h_len})"
        )));
    }
    let payload_bbs = h_len / BBSIZE as usize;
    if payload_bbs > 64 {
        return Err(crate::Error::Unsupported(format!(
            "xfs: log replay h_len {h_len} exceeds h_cycle_data slot count"
        )));
    }
    if log_bytes / BBSIZE < (1 + payload_bbs as u64) {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: log {log_bytes} bytes too small for record ({payload_bbs}+1 BBs)"
        )));
    }

    // Read the payload and restore the original first 4 bytes of each BB
    // from h_cycle_data[i].
    let mut payload = vec![0u8; h_len];
    dev.read_at(log_byte_off + BBSIZE, &mut payload)?;
    for i in 0..payload_bbs {
        let off = i * BBSIZE as usize;
        let saved = u32::from_be_bytes(hdr[44 + i * 4..48 + i * 4].try_into().unwrap());
        payload[off..off + 4].copy_from_slice(&saved.to_le_bytes());
    }

    // Parse ops.
    let mut cur: usize = 0;
    // op 1: must be XLOG_START_TRANS with TRANS_HDR_BYTES payload.
    let (oh_len1, oh_flags1) = read_op_header(&payload[cur..])?;
    if (oh_flags1 & XLOG_START_TRANS) == 0 {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: log op 1 missing XLOG_START_TRANS flag (flags={oh_flags1:#x})"
        )));
    }
    if oh_len1 as usize != TRANS_HDR_BYTES {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: log op 1 has oh_len {oh_len1} != {TRANS_HDR_BYTES}"
        )));
    }
    let trans_payload = &payload[cur + OP_HDR_BYTES..cur + OP_HDR_BYTES + TRANS_HDR_BYTES];
    let th_magic = u32::from_le_bytes(trans_payload[0..4].try_into().unwrap());
    if th_magic != XFS_TRANS_HEADER_MAGIC {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: log trans header magic {th_magic:#010x} != {XFS_TRANS_HEADER_MAGIC:#010x}"
        )));
    }
    cur += OP_HDR_BYTES + TRANS_HDR_BYTES;

    // op 2: must be the inode-log-format op.
    let (oh_len2, _oh_flags2) = read_op_header(&payload[cur..])?;
    if oh_len2 as usize != INODE_LOG_FORMAT_BYTES {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: log op 2 has oh_len {oh_len2} != {INODE_LOG_FORMAT_BYTES}"
        )));
    }
    let ilfmt = &payload[cur + OP_HDR_BYTES..cur + OP_HDR_BYTES + INODE_LOG_FORMAT_BYTES];
    let ilf_type = u16::from_le_bytes(ilfmt[0..2].try_into().unwrap());
    if ilf_type != XFS_LI_INODE {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: log inode item magic {ilf_type:#06x} != {XFS_LI_INODE:#06x}"
        )));
    }
    let inode_disk_byte = u64::from_le_bytes(ilfmt[24..32].try_into().unwrap());
    cur += OP_HDR_BYTES + INODE_LOG_FORMAT_BYTES;

    // op 3: inode buffer payload. Length is stored in the op header.
    let (oh_len3, _oh_flags3) = read_op_header(&payload[cur..])?;
    let inode_len = oh_len3 as usize;
    if cur + OP_HDR_BYTES + inode_len > payload.len() {
        return Err(crate::Error::InvalidImage(
            "xfs: log inode payload runs past record end".into(),
        ));
    }
    let inode_bytes = payload[cur + OP_HDR_BYTES..cur + OP_HDR_BYTES + inode_len].to_vec();
    cur += OP_HDR_BYTES + inode_len;

    // op 4: must be XLOG_COMMIT_TRANS with no payload.
    let (oh_len4, oh_flags4) = read_op_header(&payload[cur..])?;
    if (oh_flags4 & XLOG_COMMIT_TRANS) == 0 {
        // No commit: torn transaction, discard.
        write_empty_log(dev, log_byte_off, log_bytes, uuid)?;
        return Ok(ReplayOutcome::PartialDiscarded);
    }
    if oh_len4 != 0 {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: log commit op has oh_len {oh_len4} != 0"
        )));
    }

    // Apply the inode write.
    dev.write_at(inode_disk_byte, &inode_bytes)?;
    // Restamp the unmount record.
    write_empty_log(dev, log_byte_off, log_bytes, uuid)?;
    Ok(ReplayOutcome::Replayed)
}

/// Read a 12-byte big-endian op header from the front of `buf`.
fn read_op_header(buf: &[u8]) -> Result<(u32, u8)> {
    if buf.len() < OP_HDR_BYTES {
        return Err(crate::Error::InvalidImage(
            "xfs: log op header truncated".into(),
        ));
    }
    let oh_len = u32::from_be_bytes(buf[4..8].try_into().unwrap());
    let oh_flags = buf[9];
    Ok((oh_len, oh_flags))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn write_empty_log_smoke() {
        let mut dev = MemoryBackend::new(1024 * 1024);
        let uuid = [0xCDu8; 16];
        write_empty_log(&mut dev, 4096, 256 * 1024, &uuid).unwrap();
        let mut bb0 = vec![0u8; 512];
        dev.read_at(4096, &mut bb0).unwrap();
        assert_eq!(
            u32::from_be_bytes(bb0[0..4].try_into().unwrap()),
            XLOG_HEADER_MAGIC_NUM
        );
        assert_eq!(u32::from_be_bytes(bb0[4..8].try_into().unwrap()), 1);
        assert_eq!(u32::from_be_bytes(bb0[8..12].try_into().unwrap()), 2);
        // h_fs_uuid copied through.
        assert_eq!(&bb0[304..320], &uuid);
    }

    #[test]
    fn write_empty_log_rejects_tiny() {
        let mut dev = MemoryBackend::new(4096);
        let r = write_empty_log(&mut dev, 0, BBSIZE, &[0u8; 16]);
        assert!(r.is_err());
    }
}
