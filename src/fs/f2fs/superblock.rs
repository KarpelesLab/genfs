//! F2FS superblock parsing.
//!
//! Layout reference: [kernel docs](https://docs.kernel.org/filesystems/f2fs.html)
//! and the FAST '15 paper "F2FS: A New File System for Flash Storage".
//! Two physical copies live one F2FS block apart, at byte offsets
//! `1024` and `1024 + 0x1000`. Both copies share the same fields.
//!
//! Only the fields the read driver actually uses are decoded here.

use crate::Result;
use crate::block::BlockDevice;

/// F2FS superblock magic: bytes `0x10 0x20 0xF5 0xF2` on disk
/// (little-endian `u32`).
pub const F2FS_MAGIC: u32 = 0xF2F5_2010;
/// Byte offset of the primary superblock copy.
pub const SB_OFFSET_PRIMARY: u64 = 1024;
/// Byte offset of the backup superblock copy.
pub const SB_OFFSET_BACKUP: u64 = 1024 + 0x1000;

/// Reserved inode numbers — small values are not real on-disk nodes.
/// Conventionally NAT entries `0`, `1`, `2` are the "meta" pseudo-inode,
/// the node-tracking pseudo-inode, and the root directory.
pub const F2FS_ROOT_INO_DEFAULT: u32 = 3;

/// Decoded superblock fields relevant to a read driver.
///
/// All field offsets used by [`Superblock::decode`] follow the publicly
/// documented F2FS on-disk format (kernel docs + FAST '15 paper).
#[derive(Debug, Clone)]
pub struct Superblock {
    pub magic: u32,
    pub major_ver: u16,
    pub minor_ver: u16,
    /// log2 of sector size in bytes (always `9` = 512 B).
    pub log_sectorsize: u32,
    /// log2 of FS block size in bytes (always `12` = 4 KiB).
    pub log_blocksize: u32,
    /// log2 of blocks per segment (typically `9` → 512 blocks → 2 MiB segments).
    pub log_blocks_per_seg: u32,
    /// Segments per section (typically `1`).
    pub segs_per_sec: u32,
    /// Sections per zone (typically `1`).
    pub secs_per_zone: u32,
    /// Total blocks in the volume.
    pub block_count: u64,
    /// Total segments in the volume.
    pub segment_count: u32,
    /// Segment count reserved for each meta region.
    pub segment_count_ckpt: u32,
    pub segment_count_sit: u32,
    pub segment_count_nat: u32,
    pub segment_count_ssa: u32,
    pub segment_count_main: u32,
    /// Start segment / block addresses (block-addressed within the volume).
    pub segment0_blkaddr: u32,
    pub cp_blkaddr: u32,
    pub sit_blkaddr: u32,
    pub nat_blkaddr: u32,
    pub ssa_blkaddr: u32,
    pub main_blkaddr: u32,
    /// Reserved inode numbers.
    pub root_ino: u32,
    pub node_ino: u32,
    pub meta_ino: u32,
    /// Number of `cp_payload` extra blocks following the CP header
    /// (carries the SIT/NAT bitmap overflow when sets are large).
    pub cp_payload: u32,
    /// 16-bit UTF-16LE volume label, NUL-terminated.
    pub volume_name: String,
}

impl Superblock {
    /// Decode at least the first `0x400` bytes of an SB copy. Returns
    /// `None` when the magic doesn't match.
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 0x400 {
            return None;
        }
        let r32 = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().ok().unwrap());
        let r16 = |o: usize| u16::from_le_bytes(buf[o..o + 2].try_into().ok().unwrap());
        let r64 = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().ok().unwrap());

        let magic = r32(0x00);
        if magic != F2FS_MAGIC {
            return None;
        }
        let major_ver = r16(0x04);
        let minor_ver = r16(0x06);
        // Field offsets per `include/linux/f2fs_fs.h` / kernel docs.
        // Every offset from log_blocksize onward was previously off by
        // +4 bytes; mkfs.f2fs / fsck.f2fs reject those images even
        // though our reader could round-trip them via the same wrong
        // offsets. Aligned to the canonical layout now.
        let log_sectorsize = r32(0x08);
        // 0x0C log_sectors_per_block (unused — we only support 4 KiB blocks)
        let log_blocksize = r32(0x10);
        let log_blocks_per_seg = r32(0x14);

        // Validate the log-shift geometry before any field derived from it
        // is used (block_size = 1 << log_blocksize feeds NAT/SIT geometry
        // and every shift below). An out-of-range shift would overflow or
        // produce nonsensical addresses; F2FS fixes these values, so an
        // image with anything else is malformed. Reject early.
        if log_blocksize != 12 {
            return None;
        }
        if log_sectorsize != 9 {
            return None;
        }
        if !(1..=10).contains(&log_blocks_per_seg) {
            return None;
        }
        let segs_per_sec = r32(0x18);
        let secs_per_zone = r32(0x1C);
        // 0x20 checksum_offset
        let block_count = r64(0x24);
        let _section_count = r32(0x2C);
        let segment_count = r32(0x30);
        let segment_count_ckpt = r32(0x34);
        let segment_count_sit = r32(0x38);
        let segment_count_nat = r32(0x3C);
        let segment_count_ssa = r32(0x40);
        let segment_count_main = r32(0x44);
        let segment0_blkaddr = r32(0x48);
        let cp_blkaddr = r32(0x4C);
        let sit_blkaddr = r32(0x50);
        let nat_blkaddr = r32(0x54);
        let ssa_blkaddr = r32(0x58);
        let main_blkaddr = r32(0x5C);
        let root_ino = r32(0x60);
        let node_ino = r32(0x64);
        let meta_ino = r32(0x68);

        // The 16-byte uuid sits at 0x6C..0x7C; volume_name at 0x7C spans
        // 512 UTF-16LE code units (1024 bytes). We read at most 64 bytes
        // for the human-readable label.
        let name_off = 0x7C;
        let volume_name = if buf.len() >= name_off + 64 {
            utf16_lossy_until_nul(&buf[name_off..name_off + 64])
        } else {
            String::new()
        };

        // cp_payload sits in the trailing area of the SB (after volume
        // name and extension list). 0x3FC is the well-known offset used
        // by mkfs.f2fs to stash this value; if the read region is too
        // short we default to 0 (the common case for small images).
        let cp_payload = if buf.len() >= 0x400 { r32(0x3F8) } else { 0 };

        Some(Self {
            magic,
            major_ver,
            minor_ver,
            log_sectorsize,
            log_blocksize,
            log_blocks_per_seg,
            segs_per_sec,
            secs_per_zone,
            block_count,
            segment_count,
            segment_count_ckpt,
            segment_count_sit,
            segment_count_nat,
            segment_count_ssa,
            segment_count_main,
            segment0_blkaddr,
            cp_blkaddr,
            sit_blkaddr,
            nat_blkaddr,
            ssa_blkaddr,
            main_blkaddr,
            root_ino,
            node_ino,
            meta_ino,
            cp_payload,
            volume_name,
        })
    }

    /// FS block size in bytes (always 4096 on F2FS).
    #[inline]
    pub fn block_size(&self) -> u32 {
        1u32 << self.log_blocksize
    }

    /// Blocks per segment.
    #[inline]
    pub fn blocks_per_seg(&self) -> u32 {
        1u32 << self.log_blocks_per_seg
    }
}

fn utf16_lossy_until_nul(bytes: &[u8]) -> String {
    let mut units: Vec<u16> = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let unit = u16::from_le_bytes([chunk[0], chunk[1]]);
        if unit == 0 {
            break;
        }
        units.push(unit);
    }
    String::from_utf16_lossy(&units)
}

/// Load whichever superblock copy validates. Errors `InvalidImage` if
/// neither passes.
pub fn load(dev: &mut dyn BlockDevice) -> Result<Superblock> {
    if dev.total_size() < SB_OFFSET_BACKUP + 0x400 {
        return Err(crate::Error::InvalidImage(
            "f2fs: device too small to hold a superblock".into(),
        ));
    }
    let mut buf = vec![0u8; 0x400];
    dev.read_at(SB_OFFSET_PRIMARY, &mut buf)?;
    if let Some(sb) = Superblock::decode(&buf) {
        return Ok(sb);
    }
    dev.read_at(SB_OFFSET_BACKUP, &mut buf)?;
    if let Some(sb) = Superblock::decode(&buf) {
        return Ok(sb);
    }
    Err(crate::Error::InvalidImage(
        "f2fs: superblock magic not found in either primary or backup slot".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal, valid superblock buffer (magic + canonical log shifts).
    fn valid_sb() -> Vec<u8> {
        let mut buf = vec![0u8; 0x400];
        buf[0x00..0x04].copy_from_slice(&F2FS_MAGIC.to_le_bytes());
        buf[0x08..0x0C].copy_from_slice(&9u32.to_le_bytes()); // log_sectorsize
        buf[0x10..0x14].copy_from_slice(&12u32.to_le_bytes()); // log_blocksize
        buf[0x14..0x18].copy_from_slice(&9u32.to_le_bytes()); // log_blocks_per_seg
        buf
    }

    #[test]
    fn decode_accepts_canonical_geometry() {
        assert!(Superblock::decode(&valid_sb()).is_some());
    }

    #[test]
    fn decode_rejects_bad_log_blocksize() {
        let mut buf = valid_sb();
        buf[0x10..0x14].copy_from_slice(&13u32.to_le_bytes());
        assert!(Superblock::decode(&buf).is_none());
    }

    #[test]
    fn decode_rejects_bad_log_sectorsize() {
        let mut buf = valid_sb();
        buf[0x08..0x0C].copy_from_slice(&10u32.to_le_bytes());
        assert!(Superblock::decode(&buf).is_none());
    }

    #[test]
    fn decode_rejects_insane_log_blocks_per_seg() {
        let mut buf = valid_sb();
        // A value of 31 would make `1 << log_blocks_per_seg` overflow geometry.
        buf[0x14..0x18].copy_from_slice(&31u32.to_le_bytes());
        assert!(Superblock::decode(&buf).is_none());
        // Zero is also rejected.
        buf[0x14..0x18].copy_from_slice(&0u32.to_le_bytes());
        assert!(Superblock::decode(&buf).is_none());
    }
}
