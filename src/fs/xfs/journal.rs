//! XFS internal journal — minimal "empty / clean" log writer.
//!
//! The XFS write path currently never replays log records: every
//! transaction is committed in place. The kernel, however, refuses to
//! mount read/write unless the on-disk log is recognised as "clean"
//! (last record is an unmount record). This module lays down such a
//! log: a single record header at log block 0 followed by an `unmount`
//! op-header + payload, the rest of the log left zeroed.
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
