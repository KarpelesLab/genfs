//! XFS symbolic-link target decoding.
//!
//! Symlinks come in two flavours:
//!
//! - **Local (inline).** `di_format == LOCAL` and the target bytes live
//!   directly in the inode's literal area. The first `di_size` bytes are
//!   the target path; no header, no NUL terminator.
//!
//! - **Remote (extent).** `di_format == EXTENTS` and the target sits in one
//!   or more disk blocks pointed to by the extent list. For v4 (no CRC)
//!   the block content is just raw target bytes. For v5 (CRC) each block
//!   starts with a 56-byte `xfs_dsymlink_hdr` (magic `XSLM` =
//!   0x58_53_4C_4D), and the target spans the remaining bytes of the
//!   block(s).
//!
//! The kernel caps symlink targets at `PATH_MAX` (4096); we don't enforce
//! that and simply return whatever `di_size` claims, after sanity checks.

use crate::Result;
use crate::block::BlockDevice;

use super::bmbt::{BmbtLayout, Extent};

/// Magic at the start of a v5 symlink data block ("XSLM").
pub const XFS_SYMLINK_MAGIC: u32 = 0x5853_4C4D;

/// Maximum symlink target length the kernel allows (`PATH_MAX`). A target
/// claiming more than this is malformed and we refuse to allocate for it.
pub const XFS_SYMLINK_MAXLEN: u64 = 4096;

/// Size of the v5 symlink-block header.
pub const XFS_SYMLINK_HDR_SIZE: usize = 56;

/// Decode a **local-format** (inline) symlink: the literal area's first
/// `size` bytes are the target.
pub fn decode_local(lit: &[u8], size: u64) -> Result<String> {
    let n = size as usize;
    if n > lit.len() {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: local symlink size {n} exceeds literal area {}",
            lit.len()
        )));
    }
    let bytes = &lit[..n];
    let s = std::str::from_utf8(bytes)
        .map_err(|_| crate::Error::InvalidImage("xfs: non-UTF-8 inline symlink target".into()))?
        .to_string();
    Ok(s)
}

/// Read a **remote** (extent-list) symlink target by reading each extent
/// block in logical order, stripping the v5 header if present, and
/// concatenating the remainder until `size` bytes are gathered.
pub fn decode_remote(
    dev: &mut dyn BlockDevice,
    layout: &BmbtLayout,
    extents: &[Extent],
    size: u64,
) -> Result<String> {
    if size == 0 {
        return Ok(String::new());
    }
    // Cap the claimed target length BEFORE allocating so a malicious inode
    // with a huge `di_size` can't drive an unbounded `Vec::with_capacity`.
    // The kernel limits symlink targets to PATH_MAX (4096).
    if size > XFS_SYMLINK_MAXLEN {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: remote symlink size {size} exceeds PATH_MAX {XFS_SYMLINK_MAXLEN}"
        )));
    }
    let bs = layout.blocksize as u64;
    let agblklog = layout.agblklog as u32;
    let agblocks = layout.agblocks as u64;
    // Also bound `size` by the bytes the extent list can actually back, so we
    // never reserve more than the data on disk could supply.
    let backed_bytes: u64 = extents
        .iter()
        .map(|e| (e.blockcount as u64).saturating_mul(bs))
        .fold(0u64, |a, b| a.saturating_add(b));
    if size > backed_bytes {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: remote symlink size {size} exceeds {backed_bytes} bytes backed by extents"
        )));
    }
    // Reserve a modest amount up front and grow as bytes arrive, rather than
    // trusting `size` for the initial capacity.
    let mut out = Vec::with_capacity((size as usize).min(bs as usize));
    let mut remaining = size as usize;
    // We assume extents are sorted by logical offset (XFS invariant). Walk
    // them in array order.
    for ext in extents {
        if remaining == 0 {
            break;
        }
        if ext.unwritten {
            return Err(crate::Error::Unsupported(
                "xfs: unwritten extents in symlink target".into(),
            ));
        }
        for blkidx in 0..ext.blockcount as u64 {
            if remaining == 0 {
                break;
            }
            let fsb = ext.startblock + blkidx;
            let ag = fsb >> agblklog;
            let agblk = fsb & ((1u64 << agblklog) - 1);
            let byte_off = ag * agblocks * bs + agblk * bs;
            let mut block = vec![0u8; bs as usize];
            dev.read_at(byte_off, &mut block)?;
            let payload: &[u8] = if layout.is_v5 {
                if block.len() < XFS_SYMLINK_HDR_SIZE {
                    return Err(crate::Error::InvalidImage(
                        "xfs: v5 symlink block shorter than header".into(),
                    ));
                }
                let magic = u32::from_be_bytes(block[0..4].try_into().unwrap());
                if magic != XFS_SYMLINK_MAGIC {
                    return Err(crate::Error::InvalidImage(format!(
                        "xfs: v5 symlink block magic {magic:#010x}, want {XFS_SYMLINK_MAGIC:#010x}"
                    )));
                }
                &block[XFS_SYMLINK_HDR_SIZE..]
            } else {
                &block[..]
            };
            let take = remaining.min(payload.len());
            out.extend_from_slice(&payload[..take]);
            remaining -= take;
        }
    }
    if remaining != 0 {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: symlink target ran short by {remaining} bytes"
        )));
    }
    let s = String::from_utf8(out)
        .map_err(|_| crate::Error::InvalidImage("xfs: non-UTF-8 remote symlink target".into()))?;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn local_decode_basic() {
        let lit = b"/etc/hostname\0\0\0\0";
        // size = 13 ("/etc/hostname"); decoder doesn't trim padding, it
        // honours the inode size exactly.
        let s = decode_local(lit, 13).unwrap();
        assert_eq!(s, "/etc/hostname");
    }

    #[test]
    fn local_decode_oversize_errors() {
        let lit = b"abc";
        assert!(decode_local(lit, 100).is_err());
    }

    #[test]
    fn remote_v5_round_trip() {
        // 4 KiB blocks, 256 blocks/AG, agblklog=8. Allocate FSB 5.
        let layout = BmbtLayout {
            blocksize: 4096,
            agblocks: 256,
            agblklog: 8,
            is_v5: true,
        };
        let total = 256u64 * 4096 * 2;
        let mut dev = MemoryBackend::new(total);
        // Build a single-block symlink target: header + "/usr/lib/foo".
        let target = "/usr/lib/foo";
        let mut block = vec![0u8; 4096];
        block[0..4].copy_from_slice(&XFS_SYMLINK_MAGIC.to_be_bytes());
        block[XFS_SYMLINK_HDR_SIZE..XFS_SYMLINK_HDR_SIZE + target.len()]
            .copy_from_slice(target.as_bytes());
        // FSB 5 = AG 0, agblk 5 ⇒ byte 5 * 4096.
        dev.write_at(5 * 4096, &block).unwrap();
        let extents = vec![Extent {
            offset: 0,
            startblock: 5,
            blockcount: 1,
            unwritten: false,
        }];
        let s = decode_remote(&mut dev, &layout, &extents, target.len() as u64).unwrap();
        assert_eq!(s, target);
    }

    #[test]
    fn remote_v4_no_header() {
        let layout = BmbtLayout {
            blocksize: 512,
            agblocks: 64,
            agblklog: 6,
            is_v5: false,
        };
        let total = 64u64 * 512 * 2;
        let mut dev = MemoryBackend::new(total);
        let target = "../relative/path";
        let mut block = vec![0u8; 512];
        block[..target.len()].copy_from_slice(target.as_bytes());
        // FSB 3 = byte 3 * 512.
        dev.write_at(3 * 512, &block).unwrap();
        let extents = vec![Extent {
            offset: 0,
            startblock: 3,
            blockcount: 1,
            unwritten: false,
        }];
        let s = decode_remote(&mut dev, &layout, &extents, target.len() as u64).unwrap();
        assert_eq!(s, target);
    }

    #[test]
    fn remote_rejects_oversized_size() {
        // A `di_size` far above PATH_MAX must be rejected before any large
        // allocation, regardless of what the extents could back.
        let layout = BmbtLayout {
            blocksize: 4096,
            agblocks: 256,
            agblklog: 8,
            is_v5: true,
        };
        let mut dev = MemoryBackend::new(256 * 4096);
        let extents = vec![Extent {
            offset: 0,
            startblock: 5,
            blockcount: 1,
            unwritten: false,
        }];
        let r = decode_remote(&mut dev, &layout, &extents, 1 << 40);
        assert!(matches!(r, Err(crate::Error::InvalidImage(_))));
    }

    #[test]
    fn remote_rejects_size_exceeding_extents() {
        // size within PATH_MAX but larger than the single 512-byte block the
        // extents can supply.
        let layout = BmbtLayout {
            blocksize: 512,
            agblocks: 64,
            agblklog: 6,
            is_v5: false,
        };
        let mut dev = MemoryBackend::new(64 * 512 * 2);
        let extents = vec![Extent {
            offset: 0,
            startblock: 3,
            blockcount: 1,
            unwritten: false,
        }];
        // 4000 <= PATH_MAX but > 512 bytes backed by the one extent block.
        let r = decode_remote(&mut dev, &layout, &extents, 4000);
        assert!(matches!(r, Err(crate::Error::InvalidImage(_))));
    }

    #[test]
    fn remote_v5_bad_magic_errors() {
        let layout = BmbtLayout {
            blocksize: 4096,
            agblocks: 256,
            agblklog: 8,
            is_v5: true,
        };
        let mut dev = MemoryBackend::new(256 * 4096);
        dev.write_at(5 * 4096, &vec![0u8; 4096]).unwrap();
        let extents = vec![Extent {
            offset: 0,
            startblock: 5,
            blockcount: 1,
            unwritten: false,
        }];
        let r = decode_remote(&mut dev, &layout, &extents, 10);
        assert!(matches!(r, Err(crate::Error::InvalidImage(_))));
    }
}
