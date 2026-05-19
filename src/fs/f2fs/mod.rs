//! F2FS — Flash-Friendly File System. Scaffold module.
//!
//! ## Status
//!
//! Detection only. The on-disk superblock is parsed enough to identify
//! the volume and surface a useful `Unsupported` error for any
//! list / read / write operation. Real read-only support (segment-aware
//! NAT/SIT walk, dnodes, inline-data) lands in a follow-up.
//!
//! ## Reference
//!
//! - Linux kernel docs: <https://docs.kernel.org/filesystems/f2fs.html>
//! - "F2FS: A New File System for Flash Storage" (USENIX FAST '15)
//!
//! The two superblock copies live at byte offsets `1024` and
//! `1024 + 0x1000`; both start with the 4-byte magic `0xF2F52010`
//! (stored little-endian). See `include/linux/f2fs_fs.h` in the public
//! Linux source tree for the field layout (mirrored in the docs link
//! above). This module deliberately does **not** read or import that
//! header — the field offsets we rely on (magic, version, log_blocksize,
//! block_count, volume_name) come from the kernel documentation.
//!
//! ## Future write support
//!
//! F2FS is log-structured: every metadata update appends a new node /
//! segment rather than rewriting the original. A fresh-image writer is
//! tractable (lay down two checkpoints, an empty NAT, a single root
//! inode in the data area) but more involved than the FAT/ext writers
//! because of the checkpoint pack invariants. Not in scope yet.

use crate::Result;
use crate::block::BlockDevice;

/// F2FS superblock magic, little-endian: `0x10 0x20 0xF5 0xF2` on disk.
const F2FS_MAGIC: u32 = 0xF2F5_2010;
/// First superblock copy lives 1 KiB into the device, just like ext.
const SB_OFFSET_PRIMARY: u64 = 1024;
/// Second superblock copy lives one F2FS block (4 KiB) after the first.
const SB_OFFSET_BACKUP: u64 = 1024 + 0x1000;

/// What we parse out of either superblock copy for detection / info.
#[derive(Debug, Clone)]
pub struct Superblock {
    pub magic: u32,
    pub major_ver: u16,
    pub minor_ver: u16,
    /// log2 of sector size in bytes (always 9 = 512 B).
    pub log_sectorsize: u32,
    /// log2 of FS block size in bytes (always 12 = 4 KiB).
    pub log_blocksize: u32,
    /// Total blocks in the volume.
    pub block_count: u64,
    /// 16-bit big-endian UTF-16 volume name; decoded to UTF-8 lossily.
    pub volume_name: String,
}

impl Superblock {
    /// Decode the 512-byte head of an F2FS superblock copy. Returns
    /// `None` if the magic doesn't match.
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 0x200 {
            return None;
        }
        let magic = u32::from_le_bytes(buf[0..4].try_into().ok()?);
        if magic != F2FS_MAGIC {
            return None;
        }
        let major_ver = u16::from_le_bytes(buf[4..6].try_into().ok()?);
        let minor_ver = u16::from_le_bytes(buf[6..8].try_into().ok()?);
        let log_sectorsize = u32::from_le_bytes(buf[8..12].try_into().ok()?);
        // skip log_sectors_per_block (12..16) and log_blocksize_to_segment (16..20)
        let log_blocksize = u32::from_le_bytes(buf[20..24].try_into().ok()?);
        // log_blocks_per_seg (24..28), segs_per_sec (28..32), secs_per_zone (32..36),
        // checksum_offset (36..40), block_count is at 40..48
        let block_count = u64::from_le_bytes(buf[40..48].try_into().ok()?);
        // The volume name is at offset 0x1f0 in the on-disk SB, a UTF-16LE
        // null-terminated string of up to 512 code units. We read at most
        // 32 chars (64 bytes) for the short-name view; the full read happens
        // when we implement the actual driver.
        let name_off = 0x1f0;
        let name = if buf.len() >= name_off + 64 {
            utf16_lossy_until_nul(&buf[name_off..name_off + 64])
        } else {
            String::new()
        };
        Some(Self {
            magic,
            major_ver,
            minor_ver,
            log_sectorsize,
            log_blocksize,
            block_count,
            volume_name: name,
        })
    }
}

/// Probe for either F2FS superblock copy.
pub fn probe(dev: &mut dyn BlockDevice) -> Result<bool> {
    if dev.total_size() < SB_OFFSET_BACKUP + 4 {
        return Ok(false);
    }
    let mut head = [0u8; 4];
    dev.read_at(SB_OFFSET_PRIMARY, &mut head)?;
    if u32::from_le_bytes(head) == F2FS_MAGIC {
        return Ok(true);
    }
    dev.read_at(SB_OFFSET_BACKUP, &mut head)?;
    Ok(u32::from_le_bytes(head) == F2FS_MAGIC)
}

/// Open the volume far enough to extract the superblock for `info`.
/// All other operations are unsupported in this scaffold.
pub struct F2fs {
    sb: Superblock,
}

impl F2fs {
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        // The volume name lives at offset 0x1f0; we read a full 1 KiB so
        // the name region is included in the slice handed to `decode`.
        if dev.total_size() < SB_OFFSET_BACKUP + 0x400 {
            return Err(crate::Error::InvalidImage(
                "f2fs: device too small to hold a superblock".into(),
            ));
        }
        let mut buf = [0u8; 0x400];
        dev.read_at(SB_OFFSET_PRIMARY, &mut buf)?;
        if let Some(sb) = Superblock::decode(&buf) {
            return Ok(Self { sb });
        }
        dev.read_at(SB_OFFSET_BACKUP, &mut buf)?;
        if let Some(sb) = Superblock::decode(&buf) {
            return Ok(Self { sb });
        }
        Err(crate::Error::InvalidImage(
            "f2fs: superblock magic not found in either primary or backup slot".into(),
        ))
    }

    pub fn total_bytes(&self) -> u64 {
        self.sb.block_count << self.sb.log_blocksize
    }

    pub fn block_size(&self) -> u32 {
        1u32 << self.sb.log_blocksize
    }

    pub fn volume_name(&self) -> &str {
        &self.sb.volume_name
    }

    pub fn superblock(&self) -> &Superblock {
        &self.sb
    }

    pub fn list_path(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &str,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        Err(unsupported())
    }
}

fn unsupported() -> crate::Error {
    crate::Error::Unsupported("f2fs: read support is not implemented yet (scaffold only)".into())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal SB buffer that just sets the magic and a few fields,
    /// enough for `decode` to succeed.
    fn fake_sb(block_count: u64, name: &str) -> Vec<u8> {
        let mut v = vec![0u8; 0x400];
        v[0..4].copy_from_slice(&F2FS_MAGIC.to_le_bytes());
        v[4..6].copy_from_slice(&1u16.to_le_bytes()); // major
        v[6..8].copy_from_slice(&15u16.to_le_bytes()); // minor
        v[8..12].copy_from_slice(&9u32.to_le_bytes()); // log_sectorsize
        v[20..24].copy_from_slice(&12u32.to_le_bytes()); // log_blocksize = 4 KiB
        v[40..48].copy_from_slice(&block_count.to_le_bytes());
        let utf16: Vec<u16> = name.encode_utf16().collect();
        let off = 0x1f0;
        for (i, c) in utf16.iter().enumerate().take(31) {
            v[off + i * 2..off + i * 2 + 2].copy_from_slice(&c.to_le_bytes());
        }
        v
    }

    #[test]
    fn decode_recognises_magic_and_fields() {
        let buf = fake_sb(0x100, "myvol");
        let sb = Superblock::decode(&buf).expect("should decode");
        assert_eq!(sb.magic, F2FS_MAGIC);
        assert_eq!(sb.block_count, 0x100);
        assert_eq!(sb.volume_name, "myvol");
        assert_eq!(sb.log_blocksize, 12);
    }

    #[test]
    fn decode_rejects_wrong_magic() {
        let mut buf = fake_sb(8, "x");
        buf[0..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(Superblock::decode(&buf).is_none());
    }

    #[test]
    fn probe_detects_primary_copy() {
        use crate::block::MemoryBackend;
        let mut dev = MemoryBackend::new(64 * 1024);
        let sb = fake_sb(8, "vol");
        dev.write_at(SB_OFFSET_PRIMARY, &sb).unwrap();
        assert!(probe(&mut dev).unwrap());
    }

    #[test]
    fn probe_detects_backup_copy() {
        use crate::block::MemoryBackend;
        let mut dev = MemoryBackend::new(64 * 1024);
        let sb = fake_sb(8, "vol");
        dev.write_at(SB_OFFSET_BACKUP, &sb).unwrap();
        assert!(probe(&mut dev).unwrap());
    }

    #[test]
    fn open_reports_geometry() {
        use crate::block::MemoryBackend;
        let mut dev = MemoryBackend::new(64 * 1024);
        let sb = fake_sb(8, "demo");
        dev.write_at(SB_OFFSET_PRIMARY, &sb).unwrap();
        let f = F2fs::open(&mut dev).unwrap();
        assert_eq!(f.block_size(), 4096);
        assert_eq!(f.total_bytes(), 8 * 4096);
        assert_eq!(f.volume_name(), "demo");
    }
}
