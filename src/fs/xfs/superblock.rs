//! XFS superblock — on-disk layout + parser.
//!
//! The XFS superblock lives at byte offset 0 of allocation-group 0 (and is
//! mirrored to every other AG, though we only read AG0's copy here). All
//! multi-byte fields are **big-endian**. The fields we decode are documented
//! in the public XFS filesystem-structure reference; bytes we don't need
//! (UUIDs, log/realtime metadata, projection limits, scrub state, etc.) are
//! left as raw slices or skipped.
//!
//! ```text
//!   off  len  field                  notes
//!     0    4  sb_magicnum            "XFSB"
//!     4    4  sb_blocksize           bytes per FS block (powers of two)
//!     8    8  sb_dblocks             total FS blocks
//!    16    8  sb_rblocks             realtime blocks (we reject != 0 reads)
//!    24    8  sb_rextents
//!    32   16  sb_uuid
//!    48    8  sb_logstart
//!    56    8  sb_rootino             root inode number
//!    64    8  sb_rbmino
//!    72    8  sb_rsumino
//!    80    4  sb_rextsize
//!    84    4  sb_agblocks            blocks per AG
//!    88    4  sb_agcount
//!    92    4  sb_rbmblocks
//!    96    4  sb_logblocks
//!   100    2  sb_versionnum
//!   102    2  sb_sectsize
//!   104    2  sb_inodesize           bytes per inode (256 or 512)
//!   106    2  sb_inopblock           inodes per FS block
//!   108   12  sb_fname               volume label
//!   120    1  sb_blocklog            log2(sb_blocksize)
//!   121    1  sb_sectlog
//!   122    1  sb_inodelog            log2(sb_inodesize)
//!   123    1  sb_inopblog            log2(sb_inopblock)
//!   124    1  sb_agblklog            ceil(log2(sb_agblocks))
//!   125    1  sb_rextslog
//!   126    1  sb_inprogress
//!   127    1  sb_imax_pct
//!   128    8  sb_icount
//!   136    8  sb_ifree
//!   144    8  sb_fdblocks
//!   152    8  sb_frextents
//!   ...
//!   200    4  sb_features2           (v4 onward)
//!   204    4  sb_bad_features2
//!   208    4  sb_features_compat     (v5)
//!   212    4  sb_features_ro_compat  (v5)
//!   216    4  sb_features_incompat   (v5)
//!   220    4  sb_features_log_incompat
//!   224    4  sb_crc
//!   228    4  sb_spino_align
//!   232    8  sb_pquotino
//!   240    8  sb_lsn
//!   248   16  sb_meta_uuid
//! ```
//!
//! We only validate the magic / version / "looks sane" subset; structural
//! consistency is checked again as we descend into AG headers and inodes.

use crate::Result;

/// Superblock magic: ASCII "XFSB".
pub const XFS_SB_MAGIC: u32 = 0x5846_5342;

/// Bits of `sb_versionnum`. The low 4 bits are the format version (4 = v4,
/// 5 = v5 / "CRC" / aka "modern XFS"). Higher bits gate optional features.
pub const XFS_SB_VERSION_NUMBITS: u16 = 0x000f;
pub const XFS_SB_VERSION_5: u16 = 5;
pub const XFS_SB_VERSION_4: u16 = 4;

/// Minimum and maximum block size XFS allows (512 .. 65536). We accept the
/// full range; the kernel pegs it to the host page size at mount, but for
/// read-only inspection we don't care.
pub const XFS_MIN_BLOCKSIZE: u32 = 512;
pub const XFS_MAX_BLOCKSIZE: u32 = 65_536;

/// Decoded superblock — only the fields the reader actually needs. The raw
/// uuid is kept for diagnostics.
#[derive(Debug, Clone)]
pub struct Superblock {
    pub magic: u32,
    pub blocksize: u32,
    pub dblocks: u64,
    pub rblocks: u64,
    pub uuid: [u8; 16],
    /// `sb_logstart` — first FSB of the internal log (0 = external).
    pub logstart: u64,
    pub rootino: u64,
    pub agblocks: u32,
    pub agcount: u32,
    /// `sb_logblocks` — FS-block count of the internal log.
    pub logblocks: u32,
    pub versionnum: u16,
    pub sectsize: u16,
    pub inodesize: u16,
    pub inopblock: u16,
    pub blocklog: u8,
    pub sectlog: u8,
    pub inodelog: u8,
    pub inopblog: u8,
    pub agblklog: u8,
    pub dirblklog: u8,
    pub features2: u32,
    pub features_compat: u32,
    pub features_ro_compat: u32,
    pub features_incompat: u32,
    pub features_log_incompat: u32,
}

impl Superblock {
    /// Decode a superblock from at least 264 bytes (enough to cover all the
    /// v5 fields we read). Validates magic, version, and basic geometric
    /// consistency. `buf` may be longer; trailing bytes are ignored.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < 264 {
            return Err(crate::Error::InvalidImage(
                "xfs: superblock buffer too small".into(),
            ));
        }
        let magic = u32::from_be_bytes(buf[0..4].try_into().unwrap());
        if magic != XFS_SB_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: bad superblock magic {magic:#010x} (expected XFSB)"
            )));
        }
        let blocksize = u32::from_be_bytes(buf[4..8].try_into().unwrap());
        if !(XFS_MIN_BLOCKSIZE..=XFS_MAX_BLOCKSIZE).contains(&blocksize)
            || !blocksize.is_power_of_two()
        {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: bad blocksize {blocksize}"
            )));
        }
        let dblocks = u64::from_be_bytes(buf[8..16].try_into().unwrap());
        let rblocks = u64::from_be_bytes(buf[16..24].try_into().unwrap());
        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(&buf[32..48]);
        let logstart = u64::from_be_bytes(buf[48..56].try_into().unwrap());
        let rootino = u64::from_be_bytes(buf[56..64].try_into().unwrap());
        let agblocks = u32::from_be_bytes(buf[84..88].try_into().unwrap());
        let agcount = u32::from_be_bytes(buf[88..92].try_into().unwrap());
        let logblocks = u32::from_be_bytes(buf[96..100].try_into().unwrap());
        let versionnum = u16::from_be_bytes(buf[100..102].try_into().unwrap());
        let sectsize = u16::from_be_bytes(buf[102..104].try_into().unwrap());
        let inodesize = u16::from_be_bytes(buf[104..106].try_into().unwrap());
        let inopblock = u16::from_be_bytes(buf[106..108].try_into().unwrap());
        let blocklog = buf[120];
        let sectlog = buf[121];
        let inodelog = buf[122];
        let inopblog = buf[123];
        let agblklog = buf[124];
        // sb_rextslog is at 125; sb_inprogress 126; sb_imax_pct 127.
        // sb_icount/ifree/fdblocks/frextents follow at 128..160.
        // Optional v4-onward fields:
        let features2 = u32::from_be_bytes(buf[200..204].try_into().unwrap());
        // v5 feature words. On v4 these bytes still exist on disk (zero) so
        // it's safe to read them; they are only meaningful for v5.
        let features_compat = u32::from_be_bytes(buf[208..212].try_into().unwrap());
        let features_ro_compat = u32::from_be_bytes(buf[212..216].try_into().unwrap());
        let features_incompat = u32::from_be_bytes(buf[216..220].try_into().unwrap());
        let features_log_incompat = u32::from_be_bytes(buf[220..224].try_into().unwrap());
        // sb_dirblklog lives further in (offset 192 in v5 image), but its
        // value is identical to `dir_block_log` derived from the directory
        // version. We read it from the on-disk layout: offset 192 in v4+.
        let dirblklog = buf[192];

        // Range-check every log/shift field BEFORE using it in a shift, so a
        // malicious image can't trigger a shift-overflow panic. The shifts
        // below all use `checked_shl`; a `None` result means the log field is
        // inconsistent with its companion size field.
        if blocklog >= 32 {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: bad blocklog {blocklog} (must be < 32)"
            )));
        }
        if inodelog >= 16 {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: bad inodelog {inodelog} (must be < 16)"
            )));
        }
        if inopblog >= 16 {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: bad inopblog {inopblog} (must be < 16)"
            )));
        }
        if agblklog >= 64 {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: bad agblklog {agblklog} (must be < 64)"
            )));
        }
        // Cap the multi-block directory-block shift so `dir_block_size`
        // (`blocksize << dirblklog`) stays within a few MiB and never
        // overflows. XFS allows dir blocks up to 64 KiB; require the product
        // to fit in u32 and stay <= 8 MiB.
        const MAX_DIR_BLOCK_SIZE: u32 = 8 * 1024 * 1024;
        match blocksize.checked_shl(dirblklog as u32) {
            Some(dbs) if dbs <= MAX_DIR_BLOCK_SIZE => {}
            _ => {
                return Err(crate::Error::InvalidImage(format!(
                    "xfs: bad dirblklog {dirblklog} (blocksize {blocksize} << dirblklog oversized)"
                )));
            }
        }

        // Sanity-check the log fields where we depend on them.
        if 1u32.checked_shl(blocklog as u32) != Some(blocksize) {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: blocklog {blocklog} disagrees with blocksize {blocksize}"
            )));
        }
        if inodesize == 0 || !inodesize.is_power_of_two() {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: bad inodesize {inodesize}"
            )));
        }
        if 1u16.checked_shl(inodelog as u32) != Some(inodesize) {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: inodelog {inodelog} disagrees with inodesize {inodesize}"
            )));
        }
        if inopblock == 0 || 1u16.checked_shl(inopblog as u32) != Some(inopblock) {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: inopblog {inopblog} disagrees with inopblock {inopblock}"
            )));
        }
        if agcount == 0 {
            return Err(crate::Error::InvalidImage("xfs: agcount is 0".into()));
        }
        if agblocks == 0 {
            return Err(crate::Error::InvalidImage("xfs: agblocks is 0".into()));
        }
        // `agblklog` must be large enough to address every block in an AG:
        // `1 << agblklog >= agblocks`. (agblklog < 64 was checked above, so
        // the shift cannot overflow here.)
        if (1u64 << agblklog as u32) < agblocks as u64 {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: agblklog {agblklog} too small for agblocks {agblocks}"
            )));
        }
        let version = versionnum & XFS_SB_VERSION_NUMBITS;
        if version != XFS_SB_VERSION_4 && version != XFS_SB_VERSION_5 {
            return Err(crate::Error::Unsupported(format!(
                "xfs: unsupported sb_versionnum {versionnum:#06x} (low nibble = {version})"
            )));
        }

        Ok(Self {
            magic,
            blocksize,
            dblocks,
            rblocks,
            uuid,
            logstart,
            rootino,
            agblocks,
            agcount,
            logblocks,
            versionnum,
            sectsize,
            inodesize,
            inopblock,
            blocklog,
            sectlog,
            inodelog,
            inopblog,
            agblklog,
            dirblklog,
            features2,
            features_compat,
            features_ro_compat,
            features_incompat,
            features_log_incompat,
        })
    }

    /// True iff this is a v5 (CRC) superblock.
    pub fn is_v5(&self) -> bool {
        (self.versionnum & XFS_SB_VERSION_NUMBITS) == XFS_SB_VERSION_5
    }

    /// Total bytes claimed by the volume — `sb_dblocks * sb_blocksize`.
    pub fn total_bytes(&self) -> u64 {
        self.dblocks.saturating_mul(self.blocksize as u64)
    }

    /// Directory block size in bytes. XFS directory blocks may be larger
    /// than FS blocks (multi-block directory blocks); the multiplier is
    /// `1 << sb_dirblklog`.
    pub fn dir_block_size(&self) -> u32 {
        // `Superblock::decode` validates `dirblklog` so this shift cannot
        // overflow for a successfully-decoded superblock; saturate defensively
        // rather than panic if called on a hand-built value.
        self.blocksize
            .checked_shl(self.dirblklog as u32)
            .unwrap_or(u32::MAX)
    }

    /// Byte offset on the device where the internal log begins. Translates
    /// `sb_logstart` (an FSB packed `(ag << agblklog) | agblk`) to a
    /// byte address using the same scheme as inode/extent addressing.
    /// Returns 0 if there is no internal log (`logstart == 0`).
    pub fn logstart_byte_offset(&self) -> u64 {
        if self.logstart == 0 {
            return 0;
        }
        let ag = self.logstart >> self.agblklog as u32;
        let agblk = self.logstart & ((1u64 << self.agblklog as u32) - 1);
        ag * (self.agblocks as u64) * (self.blocksize as u64) + agblk * (self.blocksize as u64)
    }

    /// Byte size of the internal log: `sb_logblocks * sb_blocksize`.
    pub fn log_bytes(&self) -> u64 {
        (self.logblocks as u64) * (self.blocksize as u64)
    }
}

/// Build a synthetic superblock buffer with the given key fields and zeros
/// elsewhere. Used by the integration-style tests in `mod.rs`; declared at
/// crate level (not under `cfg(test)`) so cross-submodule test code can see
/// it. Lives in this file because it mirrors the layout documented above.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn synth_sb_for_tests(
    blocksize: u32,
    dblocks: u64,
    agblocks: u32,
    agcount: u32,
    inodesize: u16,
    inopblock: u16,
    rootino: u64,
    version: u16,
) -> Vec<u8> {
    let mut buf = vec![0u8; 264];
    buf[0..4].copy_from_slice(&XFS_SB_MAGIC.to_be_bytes());
    buf[4..8].copy_from_slice(&blocksize.to_be_bytes());
    buf[8..16].copy_from_slice(&dblocks.to_be_bytes());
    buf[56..64].copy_from_slice(&rootino.to_be_bytes());
    buf[84..88].copy_from_slice(&agblocks.to_be_bytes());
    buf[88..92].copy_from_slice(&agcount.to_be_bytes());
    buf[100..102].copy_from_slice(&version.to_be_bytes());
    buf[102..104].copy_from_slice(&(512u16).to_be_bytes());
    buf[104..106].copy_from_slice(&inodesize.to_be_bytes());
    buf[106..108].copy_from_slice(&inopblock.to_be_bytes());
    buf[120] = blocksize.trailing_zeros() as u8;
    buf[121] = 9; // sectlog = log2(512)
    buf[122] = inodesize.trailing_zeros() as u8;
    buf[123] = inopblock.trailing_zeros() as u8;
    buf[124] = next_pow2_log_for_tests(agblocks);
    buf
}

#[cfg(test)]
fn next_pow2_log_for_tests(n: u32) -> u8 {
    let mut l = 0u8;
    let mut x: u64 = 1;
    while x < n as u64 {
        x <<= 1;
        l += 1;
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn synth(
        blocksize: u32,
        dblocks: u64,
        agblocks: u32,
        agcount: u32,
        inodesize: u16,
        inopblock: u16,
        rootino: u64,
        version: u16,
    ) -> Vec<u8> {
        super::synth_sb_for_tests(
            blocksize, dblocks, agblocks, agcount, inodesize, inopblock, rootino, version,
        )
    }

    #[test]
    fn decode_minimal() {
        // 4 KiB blocks, 32 MiB volume, 4 AGs, 512-byte inodes, 8 inodes/block.
        let buf = synth(4096, 8192, 2048, 4, 512, 8, 128, XFS_SB_VERSION_5);
        let sb = Superblock::decode(&buf).unwrap();
        assert!(sb.is_v5());
        assert_eq!(sb.blocksize, 4096);
        assert_eq!(sb.dblocks, 8192);
        assert_eq!(sb.agblocks, 2048);
        assert_eq!(sb.agcount, 4);
        assert_eq!(sb.inodesize, 512);
        assert_eq!(sb.inopblock, 8);
        assert_eq!(sb.rootino, 128);
        assert_eq!(sb.total_bytes(), 8192 * 4096);
        assert_eq!(sb.blocklog, 12);
        assert_eq!(sb.inodelog, 9);
        assert_eq!(sb.inopblog, 3);
        // ceil(log2(2048)) = 11.
        assert_eq!(sb.agblklog, 11);
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut buf = synth(4096, 8192, 2048, 4, 512, 8, 128, XFS_SB_VERSION_5);
        buf[0] = 0;
        assert!(matches!(
            Superblock::decode(&buf),
            Err(crate::Error::InvalidImage(_))
        ));
    }

    #[test]
    fn decode_rejects_inconsistent_blocklog() {
        let mut buf = synth(4096, 8192, 2048, 4, 512, 8, 128, XFS_SB_VERSION_5);
        buf[120] = 11; // 1<<11 = 2048, not 4096
        assert!(matches!(
            Superblock::decode(&buf),
            Err(crate::Error::InvalidImage(_))
        ));
    }

    #[test]
    fn decode_rejects_oversized_dirblklog() {
        // dirblklog is at byte 192; a huge value would overflow
        // `blocksize << dirblklog` or yield an enormous dir-block alloc.
        let mut buf = synth(4096, 8192, 2048, 4, 512, 8, 128, XFS_SB_VERSION_5);
        buf[192] = 40; // 4096 << 40 overflows u32 / is wildly oversized
        assert!(matches!(
            Superblock::decode(&buf),
            Err(crate::Error::InvalidImage(_))
        ));
    }

    #[test]
    fn decode_rejects_huge_agblklog() {
        // agblklog >= 64 would overflow `1u64 << agblklog`.
        let mut buf = synth(4096, 8192, 2048, 4, 512, 8, 128, XFS_SB_VERSION_5);
        buf[124] = 64;
        assert!(matches!(
            Superblock::decode(&buf),
            Err(crate::Error::InvalidImage(_))
        ));
    }

    #[test]
    fn decode_rejects_agblklog_too_small_for_agblocks() {
        // 1<<agblklog must be >= agblocks (2048 here needs agblklog >= 11).
        let mut buf = synth(4096, 8192, 2048, 4, 512, 8, 128, XFS_SB_VERSION_5);
        buf[124] = 5; // 1<<5 = 32 < 2048
        assert!(matches!(
            Superblock::decode(&buf),
            Err(crate::Error::InvalidImage(_))
        ));
    }

    #[test]
    fn decode_rejects_out_of_range_blocklog() {
        // blocklog >= 32 must be rejected before any shift is attempted.
        let mut buf = synth(4096, 8192, 2048, 4, 512, 8, 128, XFS_SB_VERSION_5);
        buf[120] = 200;
        assert!(matches!(
            Superblock::decode(&buf),
            Err(crate::Error::InvalidImage(_))
        ));
    }

    #[test]
    fn decode_rejects_v3() {
        let buf = synth(4096, 8192, 2048, 4, 512, 8, 128, 3);
        assert!(matches!(
            Superblock::decode(&buf),
            Err(crate::Error::Unsupported(_))
        ));
    }
}
