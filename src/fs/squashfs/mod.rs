//! SquashFS — read-only compressed archive filesystem. Scaffold module.
//!
//! ## Status
//!
//! Detection + superblock parse only. List / read returns `Unsupported`;
//! the inode-table and directory-table walks land in a follow-up, and a
//! decompressor abstraction (zstd / xz / lz4 / lzma / gzip) is wired
//! in at the same time.
//!
//! ## Reference
//!
//! - SquashFS on-disk layout, kernel docs:
//!   <https://docs.kernel.org/filesystems/squashfs.html>
//! - On-disk header reference (Mountains): squashfs source ships the
//!   complete struct definitions; we re-derive them from the docs.
//!
//! ## Versioning
//!
//! Modern SquashFS images use major version 4 (minor 0); older 3.x
//! images appear in long-tail kernels. v1 scaffold only accepts v4 —
//! a v3 image opens with an `Unsupported` error naming the version.
//!
//! ## Compression
//!
//! SquashFS supports many algorithms; the superblock's `compression`
//! field picks one and the optional "compression options" block right
//! after the header carries algorithm-specific parameters. The scaffold
//! records which algorithm is in use but does not decompress anything.

use crate::Result;
use crate::block::BlockDevice;

/// SquashFS magic, little-endian: `hsqs` reversed on disk = `0x73717368`.
const SQUASHFS_MAGIC: u32 = 0x7371_7368;

/// Compression scheme advertised in the superblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    Gzip,
    Lzma,
    Lzo,
    Xz,
    Lz4,
    Zstd,
    Unknown(u16),
}

impl Compression {
    fn from_id(id: u16) -> Self {
        match id {
            1 => Self::Gzip,
            2 => Self::Lzma,
            3 => Self::Lzo,
            4 => Self::Xz,
            5 => Self::Lz4,
            6 => Self::Zstd,
            other => Self::Unknown(other),
        }
    }
}

/// Decoded prefix of the SquashFS superblock that we care about for
/// detection and `info`.
#[derive(Debug, Clone)]
pub struct Superblock {
    pub magic: u32,
    pub inode_count: u32,
    /// Last modification time in Unix seconds.
    pub mkfs_time: u32,
    /// Block size in bytes (power-of-two, 4 KiB … 1 MiB).
    pub block_size: u32,
    /// Number of fragment entries in the fragment table.
    pub fragment_count: u32,
    pub compression: Compression,
    /// log2(block_size), redundant with `block_size`.
    pub block_log: u16,
    pub flags: u16,
    /// IDs in the id lookup table (uid+gid entries).
    pub id_count: u16,
    pub major: u16,
    pub minor: u16,
    /// Inode reference (block,offset) for the root directory inode.
    pub root_inode: u64,
    pub bytes_used: u64,
}

impl Superblock {
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 96 {
            return None;
        }
        let magic = u32::from_le_bytes(buf[0..4].try_into().ok()?);
        if magic != SQUASHFS_MAGIC {
            return None;
        }
        let inode_count = u32::from_le_bytes(buf[4..8].try_into().ok()?);
        let mkfs_time = u32::from_le_bytes(buf[8..12].try_into().ok()?);
        let block_size = u32::from_le_bytes(buf[12..16].try_into().ok()?);
        let fragment_count = u32::from_le_bytes(buf[16..20].try_into().ok()?);
        let compression = Compression::from_id(u16::from_le_bytes(buf[20..22].try_into().ok()?));
        let block_log = u16::from_le_bytes(buf[22..24].try_into().ok()?);
        let flags = u16::from_le_bytes(buf[24..26].try_into().ok()?);
        let id_count = u16::from_le_bytes(buf[26..28].try_into().ok()?);
        let major = u16::from_le_bytes(buf[28..30].try_into().ok()?);
        let minor = u16::from_le_bytes(buf[30..32].try_into().ok()?);
        let root_inode = u64::from_le_bytes(buf[32..40].try_into().ok()?);
        let bytes_used = u64::from_le_bytes(buf[40..48].try_into().ok()?);
        Some(Self {
            magic,
            inode_count,
            mkfs_time,
            block_size,
            fragment_count,
            compression,
            block_log,
            flags,
            id_count,
            major,
            minor,
            root_inode,
            bytes_used,
        })
    }
}

pub fn probe(dev: &mut dyn BlockDevice) -> Result<bool> {
    if dev.total_size() < 4 {
        return Ok(false);
    }
    let mut head = [0u8; 4];
    dev.read_at(0, &mut head)?;
    Ok(u32::from_le_bytes(head) == SQUASHFS_MAGIC)
}

#[derive(Debug)]
pub struct Squashfs {
    sb: Superblock,
}

impl Squashfs {
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        if dev.total_size() < 96 {
            return Err(crate::Error::InvalidImage(
                "squashfs: device too small to hold a superblock".into(),
            ));
        }
        let mut buf = [0u8; 96];
        dev.read_at(0, &mut buf)?;
        let sb = Superblock::decode(&buf).ok_or_else(|| {
            crate::Error::InvalidImage("squashfs: superblock magic mismatch".into())
        })?;
        if sb.major != 4 {
            return Err(crate::Error::Unsupported(format!(
                "squashfs: only version 4.x is supported (got {}.{})",
                sb.major, sb.minor
            )));
        }
        Ok(Self { sb })
    }

    pub fn total_bytes(&self) -> u64 {
        self.sb.bytes_used
    }

    pub fn block_size(&self) -> u32 {
        self.sb.block_size
    }

    pub fn compression(&self) -> Compression {
        self.sb.compression
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
    crate::Error::Unsupported(
        "squashfs: read support is not implemented yet (scaffold only)".into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_sb(major: u16, comp: u16) -> Vec<u8> {
        let mut v = vec![0u8; 128];
        v[0..4].copy_from_slice(&SQUASHFS_MAGIC.to_le_bytes());
        v[4..8].copy_from_slice(&3u32.to_le_bytes()); // inodes
        v[8..12].copy_from_slice(&0u32.to_le_bytes());
        v[12..16].copy_from_slice(&131072u32.to_le_bytes()); // 128 KiB
        v[16..20].copy_from_slice(&0u32.to_le_bytes());
        v[20..22].copy_from_slice(&comp.to_le_bytes());
        v[22..24].copy_from_slice(&17u16.to_le_bytes());
        v[28..30].copy_from_slice(&major.to_le_bytes());
        v[30..32].copy_from_slice(&0u16.to_le_bytes());
        v[40..48].copy_from_slice(&512u64.to_le_bytes()); // bytes_used
        v
    }

    #[test]
    fn decode_recognises_zstd() {
        let v = fake_sb(4, 6);
        let sb = Superblock::decode(&v).unwrap();
        assert_eq!(sb.compression, Compression::Zstd);
        assert_eq!(sb.block_size, 131072);
    }

    #[test]
    fn open_rejects_v3() {
        use crate::block::MemoryBackend;
        let mut dev = MemoryBackend::new(4096);
        dev.write_at(0, &fake_sb(3, 1)).unwrap();
        let err = Squashfs::open(&mut dev).unwrap_err();
        match err {
            crate::Error::Unsupported(_) => {}
            _ => panic!("expected Unsupported, got {err:?}"),
        }
    }

    #[test]
    fn open_accepts_v4() {
        use crate::block::MemoryBackend;
        let mut dev = MemoryBackend::new(4096);
        dev.write_at(0, &fake_sb(4, 6)).unwrap();
        let s = Squashfs::open(&mut dev).unwrap();
        assert_eq!(s.compression(), Compression::Zstd);
    }

    #[test]
    fn probe_matches_magic() {
        use crate::block::MemoryBackend;
        let mut dev = MemoryBackend::new(4096);
        dev.write_at(0, &SQUASHFS_MAGIC.to_le_bytes()).unwrap();
        assert!(probe(&mut dev).unwrap());
    }
}
