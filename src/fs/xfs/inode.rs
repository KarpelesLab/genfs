//! XFS dinode — the on-disk inode structure (`xfs_dinode_core` + fork data).
//!
//! Layout (big-endian throughout):
//!
//! ```text
//!   off  len  field             notes
//!     0    2  di_magic          "IN" (0x494e)
//!     2    2  di_mode           S_IF* | perm bits
//!     4    1  di_version        1, 2, or 3 (v3 == v5 CRC inodes)
//!     5    1  di_format         data-fork format (see XFS_DINODE_FMT_*)
//!     6    2  di_onlink         legacy nlink (v1/v2 only)
//!     8    4  di_uid
//!    12    4  di_gid
//!    16    4  di_nlink          v2+ link count
//!    20    2  di_projid         v2+
//!    22    2  di_projid_hi
//!    24    8  di_pad
//!    32    2  di_flushiter
//!    32    8  (v3) di_changecount
//!    32   16  di_atime/mtime overlap (see below)
//!    32   32  di_atime/mtime/ctime  v2:  3*8-byte timestamps   sec(BE u32) + nsec(BE u32)
//!    56    8  di_size           size in bytes
//!    64    8  di_nblocks        blocks consumed
//!    72    4  di_extsize        preferred extent size
//!    76    4  di_nextents       number of extents in data fork
//!    80    2  di_anextents      number of extents in attribute fork
//!    82    1  di_forkoff        offset in 8-byte words of attribute fork
//!    83    1  di_aformat
//!    84    4  di_dmevmask
//!    88    2  di_dmstate
//!    90    2  di_flags
//!    92    4  di_gen
//!
//!   v3 extension (di_version == 3):
//!    96    4  di_next_unlinked
//!   100    4  di_crc             stored little-endian
//!   104    8  di_changecount
//!   112    8  di_lsn
//!   120    8  di_flags2
//!   128    4  di_cowextsize
//!   132   12  di_pad2
//!   144    8  di_crtime          creation time (sec + nsec)
//!   152    8  di_ino             self-reference
//!   160   16  di_uuid            volume meta UUID
//!   ^-- end of v3 core (176 bytes); fork starts at di_literal_area
//! ```
//!
//! Two literal-area sizes are common:
//! - v2 (256-byte) inode: core = 96 bytes (no v3 extension), literal area = 160 bytes.
//! - v3 (512-byte) inode: core = 176 bytes, literal area = 336 bytes.
//!
//! `di_forkoff` (when non-zero) measures, in 8-byte units, where the
//! attribute fork begins **inside the literal area**. When zero, the
//! attribute fork doesn't exist (or uses the secondary inode-attr format).

use crate::Result;

/// Inode magic: ASCII "IN" big-endian.
pub const XFS_DINODE_MAGIC: u16 = 0x494e;

/// Data-fork formats (`di_format`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiFormat {
    /// Special inode (no fork data, e.g. character devices).
    Dev,
    /// Inline data: shortform directory, symlink target, or zero-length file.
    Local,
    /// Extent list packed into the literal area (the common case).
    Extents,
    /// On-disk B+tree root in the literal area; leaves elsewhere.
    Btree,
    /// Unknown format byte we encountered (recorded so the caller can
    /// produce a useful error).
    Unknown(u8),
}

impl DiFormat {
    fn from_byte(b: u8) -> Self {
        match b {
            0 => Self::Dev,
            1 => Self::Local,
            2 => Self::Extents,
            3 => Self::Btree,
            other => Self::Unknown(other),
        }
    }
}

/// POSIX file-type bits.
pub const S_IFMT: u16 = 0o170_000;
pub const S_IFIFO: u16 = 0o010_000;
pub const S_IFCHR: u16 = 0o020_000;
pub const S_IFDIR: u16 = 0o040_000;
pub const S_IFBLK: u16 = 0o060_000;
pub const S_IFREG: u16 = 0o100_000;
pub const S_IFLNK: u16 = 0o120_000;
pub const S_IFSOCK: u16 = 0o140_000;

/// A timestamp the way XFS v3 stores it on disk: 32-bit big-endian seconds
/// followed by 32-bit big-endian nanoseconds. XFS v5 with the BIGTIME
/// feature reinterprets this as a single 64-bit count; we don't enable that
/// path here.
#[derive(Debug, Clone, Copy, Default)]
pub struct XfsTimestamp {
    pub sec: u32,
    pub nsec: u32,
}

impl XfsTimestamp {
    pub fn decode(buf: &[u8]) -> Self {
        Self {
            sec: u32::from_be_bytes(buf[0..4].try_into().unwrap()),
            nsec: u32::from_be_bytes(buf[4..8].try_into().unwrap()),
        }
    }
}

/// Decoded inode core. The literal-area bytes (where the fork lives) are
/// returned alongside as a slice into the caller's buffer.
#[derive(Debug, Clone)]
pub struct DinodeCore {
    pub magic: u16,
    pub mode: u16,
    pub version: u8,
    pub format: DiFormat,
    pub uid: u32,
    pub gid: u32,
    pub nlink: u32,
    pub atime: XfsTimestamp,
    pub mtime: XfsTimestamp,
    pub ctime: XfsTimestamp,
    pub size: u64,
    pub nblocks: u64,
    pub nextents: u32,
    pub forkoff: u8,
    pub flags: u16,
    pub generation: u32,
    /// v3 (CRC) only: self-reference inode number.
    pub di_ino: Option<u64>,
    /// Byte offset within the inode where the data-fork literal area begins.
    /// 96 for v2 inodes, 176 for v3 inodes.
    pub literal_offset: usize,
}

impl DinodeCore {
    /// Decode the core. `buf` must be at least `inodesize` bytes.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < 96 {
            return Err(crate::Error::InvalidImage(
                "xfs: inode buffer too small".into(),
            ));
        }
        let magic = u16::from_be_bytes(buf[0..2].try_into().unwrap());
        if magic != XFS_DINODE_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: bad inode magic {magic:#06x} (expected IN)"
            )));
        }
        let mode = u16::from_be_bytes(buf[2..4].try_into().unwrap());
        let version = buf[4];
        let format = DiFormat::from_byte(buf[5]);
        let uid = u32::from_be_bytes(buf[8..12].try_into().unwrap());
        let gid = u32::from_be_bytes(buf[12..16].try_into().unwrap());
        let nlink = u32::from_be_bytes(buf[16..20].try_into().unwrap());
        let atime = XfsTimestamp::decode(&buf[32..40]);
        let mtime = XfsTimestamp::decode(&buf[40..48]);
        let ctime = XfsTimestamp::decode(&buf[48..56]);
        let size = u64::from_be_bytes(buf[56..64].try_into().unwrap());
        let nblocks = u64::from_be_bytes(buf[64..72].try_into().unwrap());
        let nextents = u32::from_be_bytes(buf[76..80].try_into().unwrap());
        let forkoff = buf[82];
        let flags = u16::from_be_bytes(buf[90..92].try_into().unwrap());
        let generation = u32::from_be_bytes(buf[92..96].try_into().unwrap());

        let (literal_offset, di_ino) = if version >= 3 {
            if buf.len() < 176 {
                return Err(crate::Error::InvalidImage(
                    "xfs: v3 inode buffer too small for core".into(),
                ));
            }
            let ino = u64::from_be_bytes(buf[152..160].try_into().unwrap());
            (176, Some(ino))
        } else {
            (96, None)
        };

        if let DiFormat::Unknown(b) = format {
            return Err(crate::Error::Unsupported(format!(
                "xfs: unknown di_format {b}"
            )));
        }

        Ok(Self {
            magic,
            mode,
            version,
            format,
            uid,
            gid,
            nlink,
            atime,
            mtime,
            ctime,
            size,
            nblocks,
            nextents,
            forkoff,
            flags,
            generation,
            di_ino,
            literal_offset,
        })
    }

    /// Slice into `buf` covering the literal area (data fork prefix) of
    /// length `lit_len`. `lit_len` is `inodesize - literal_offset` when the
    /// attribute fork is absent; otherwise it's `forkoff * 8` bytes.
    pub fn literal_area<'a>(&self, buf: &'a [u8], inodesize: usize) -> &'a [u8] {
        let end = if self.forkoff == 0 {
            inodesize
        } else {
            self.literal_offset + (self.forkoff as usize) * 8
        };
        &buf[self.literal_offset..end.min(buf.len())]
    }

    pub fn is_dir(&self) -> bool {
        (self.mode & S_IFMT) == S_IFDIR
    }
    pub fn is_reg(&self) -> bool {
        (self.mode & S_IFMT) == S_IFREG
    }
    pub fn is_symlink(&self) -> bool {
        (self.mode & S_IFMT) == S_IFLNK
    }
}

/// Byte offset of `di_crc` inside a v3 dinode core. CRC32C is computed
/// over the entire on-disk inode (every byte the inode occupies) with
/// this 4-byte field zeroed, then stored as little-endian (`__le32`).
pub const V3_CRC_OFFSET: usize = 100;

/// On-disk byte size of the v3 dinode core (including the v3 extension
/// up to but not including the literal area).
pub const V3_CORE_SIZE: usize = 176;

/// Build a v3 (CRC) dinode. Lays out an `inodesize`-byte buffer with
/// every field except `di_crc`. Use [`stamp_v3_inode_crc`] **after**
/// writing the literal-area (fork) bytes so the checksum covers them.
#[derive(Debug, Clone)]
pub struct V3DinodeBuilder {
    pub inodesize: usize,
    pub mode: u16,
    pub format: u8,
    pub uid: u32,
    pub gid: u32,
    pub nlink: u32,
    pub atime: XfsTimestamp,
    pub mtime: XfsTimestamp,
    pub ctime: XfsTimestamp,
    pub crtime: XfsTimestamp,
    pub size: u64,
    pub nblocks: u64,
    pub extsize: u32,
    pub nextents: u32,
    pub forkoff: u8,
    pub aformat: u8,
    pub flags: u16,
    pub generation: u32,
    pub di_ino: u64,
    pub uuid: [u8; 16],
    /// `di_flags2` (v3 extension, offset 120..128). Carries the
    /// REFLINK / BIGTIME / NREXT64 bits. Most callers leave this `0`;
    /// the `clone_file` writer sets `XFS_DIFLAG2_REFLINK` on both
    /// the source and destination inodes.
    pub flags2: u64,
}

/// `di_flags2` bit: file has shared extents (clone / reflink). XFS
/// kernels require this on every inode that participates in a clone;
/// `xfs_repair` flags the volume corrupt if the bit is unset but the
/// extent is in the REFCNTBT.
pub const XFS_DIFLAG2_REFLINK: u64 = 0x2;

impl V3DinodeBuilder {
    /// Allocate the on-disk inode buffer and stamp every field except
    /// `di_crc`. The literal area starts at byte 176 and is the caller's
    /// to populate.
    pub fn build(&self) -> Vec<u8> {
        let mut buf = vec![0u8; self.inodesize];
        buf[0..2].copy_from_slice(&XFS_DINODE_MAGIC.to_be_bytes());
        buf[2..4].copy_from_slice(&self.mode.to_be_bytes());
        buf[4] = 3;
        buf[5] = self.format;
        // di_onlink (legacy) — v3 zero
        buf[8..12].copy_from_slice(&self.uid.to_be_bytes());
        buf[12..16].copy_from_slice(&self.gid.to_be_bytes());
        buf[16..20].copy_from_slice(&self.nlink.to_be_bytes());
        buf[32..40].copy_from_slice(&encode_ts(self.atime));
        buf[40..48].copy_from_slice(&encode_ts(self.mtime));
        buf[48..56].copy_from_slice(&encode_ts(self.ctime));
        buf[56..64].copy_from_slice(&self.size.to_be_bytes());
        buf[64..72].copy_from_slice(&self.nblocks.to_be_bytes());
        buf[72..76].copy_from_slice(&self.extsize.to_be_bytes());
        buf[76..80].copy_from_slice(&self.nextents.to_be_bytes());
        buf[82] = self.forkoff;
        buf[83] = self.aformat;
        buf[90..92].copy_from_slice(&self.flags.to_be_bytes());
        buf[92..96].copy_from_slice(&self.generation.to_be_bytes());
        // di_next_unlinked at 96..100 = NULL (-1)
        buf[96..100].copy_from_slice(&u32::MAX.to_be_bytes());
        // di_crc at 100..104 — caller's responsibility via stamp_v3_inode_crc
        // di_flags2 at 120..128 — REFLINK / BIGTIME / NREXT64 etc.
        buf[120..128].copy_from_slice(&self.flags2.to_be_bytes());
        buf[144..152].copy_from_slice(&encode_ts(self.crtime));
        buf[152..160].copy_from_slice(&self.di_ino.to_be_bytes());
        buf[160..176].copy_from_slice(&self.uuid);
        buf
    }
}

fn encode_ts(ts: XfsTimestamp) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0..4].copy_from_slice(&ts.sec.to_be_bytes());
    out[4..8].copy_from_slice(&ts.nsec.to_be_bytes());
    out
}

/// Compute and store the v3 dinode CRC. CRC32C is taken over the full
/// inode buffer with the 4-byte `di_crc` field at offset 100 zeroed,
/// then written back as little-endian.
pub fn stamp_v3_inode_crc(buf: &mut [u8]) {
    buf[V3_CRC_OFFSET..V3_CRC_OFFSET + 4].copy_from_slice(&[0u8; 4]);
    let crc = crc32c::crc32c(buf);
    buf[V3_CRC_OFFSET..V3_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_v3_inode(size: u64, format: u8, mode: u16, lit_payload: &[u8]) -> Vec<u8> {
        // 512-byte inode.
        let mut buf = vec![0u8; 512];
        buf[0..2].copy_from_slice(&XFS_DINODE_MAGIC.to_be_bytes());
        buf[2..4].copy_from_slice(&mode.to_be_bytes());
        buf[4] = 3; // v3
        buf[5] = format;
        buf[16..20].copy_from_slice(&1u32.to_be_bytes()); // nlink
        buf[56..64].copy_from_slice(&size.to_be_bytes());
        // di_ino self ref
        buf[152..160].copy_from_slice(&128u64.to_be_bytes());
        // Literal area at 176; copy payload (capped to 336 bytes).
        let n = lit_payload.len().min(512 - 176);
        buf[176..176 + n].copy_from_slice(&lit_payload[..n]);
        buf
    }

    #[test]
    fn decode_v3_local_dir() {
        let buf = synth_v3_inode(0, 1, S_IFDIR | 0o755, &[0xAA; 64]);
        let core = DinodeCore::decode(&buf).unwrap();
        assert_eq!(core.magic, XFS_DINODE_MAGIC);
        assert_eq!(core.version, 3);
        assert_eq!(core.format, DiFormat::Local);
        assert!(core.is_dir());
        assert_eq!(core.literal_offset, 176);
        assert_eq!(core.di_ino, Some(128));
        let lit = core.literal_area(&buf, 512);
        assert_eq!(lit.len(), 512 - 176);
        assert!(lit.iter().take(64).all(|&b| b == 0xAA));
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut buf = synth_v3_inode(0, 1, S_IFDIR | 0o755, &[]);
        buf[0] = 0;
        assert!(matches!(
            DinodeCore::decode(&buf),
            Err(crate::Error::InvalidImage(_))
        ));
    }

    #[test]
    fn forkoff_caps_literal_area() {
        let mut buf = synth_v3_inode(0, 1, S_IFREG | 0o644, &[0; 8]);
        // forkoff = 4 words = 32 bytes of data fork inside the literal area.
        buf[82] = 4;
        let core = DinodeCore::decode(&buf).unwrap();
        let lit = core.literal_area(&buf, 512);
        assert_eq!(lit.len(), 32);
    }
}
