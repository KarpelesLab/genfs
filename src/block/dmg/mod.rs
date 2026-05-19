//! Apple Disk Image (DMG) — scaffold.
//!
//! ## Status
//!
//! Detection + koly-trailer parse only. `DmgBackend::open` validates
//! the trailer and exposes [`BlockDevice::total_size`] (= `sector_count
//! * 512`), but every `read_at` returns
//! [`crate::Error::Unsupported`] until the chunk decoder is built.
//!
//! The real reader will decompress per-chunk UDIF blocks (raw, ADC,
//! zlib, bzip2, LZFSE, LZMA) into the requested byte range. Writers
//! are out of scope for v1.
//!
//! ## Background
//!
//! UDIF (Universal Disk Image Format) is the modern .dmg layout. The
//! file is laid out as:
//!
//! ```text
//!   [ data-fork bytes (compressed chunks) ]
//!   [ XML / binary-plist resource fork    ]   ← table of `mish` blocks
//!   [ 512-byte koly trailer at file end   ]   ← magic, offsets, sector count
//! ```
//!
//! All multi-byte fields in the koly trailer are **big-endian**. We
//! anchor detection on the `koly` magic at `file_size - 512`. Older
//! NDIF (`disk`-magic) images are explicitly out of scope.
//!
//! ## References
//!
//! Apple has never published a formal UDIF spec. The layout in this
//! module follows two public reverse-engineering write-ups:
//!
//! - Jonathan Levin, *DMG file structure* (newosxbook.com).
//! - The `libdmg-hfsplus` project's `dmg.h`, particularly the
//!   `UDIFResourceFile` struct.
//!
//! All offsets and field meanings are public spec; no Apple
//! source / SDK was consulted.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::Result;
use crate::block::BlockDevice;

/// `koly` magic — first four bytes of the 512-byte UDIF trailer.
pub const KOLY_MAGIC: u32 = 0x6B6F_6C79; // "koly"

/// Total size of the trailer (one sector).
pub const KOLY_SIZE: u64 = 512;

/// Fields from the 512-byte UDIF trailer. Multi-byte values are
/// big-endian on disk; this struct holds them in host order after
/// `decode`. Only the fields the reader actually needs are surfaced;
/// the trailer's per-segment / checksum payloads are decoded into raw
/// byte arrays so a caller that wants them can still inspect them.
#[derive(Debug, Clone)]
pub struct KolyTrailer {
    /// `udif_signature` — must equal [`KOLY_MAGIC`].
    pub signature: u32,
    /// `udif_version` — 4 for current images.
    pub version: u32,
    /// `header_size` — typically 512 (the trailer is its own header).
    pub header_size: u32,
    /// `flags` — implementation-defined; carry through verbatim.
    pub flags: u32,
    /// `running_data_fork_offset` — start of the data fork inside this
    /// segment. Zero for single-segment images.
    pub running_data_fork_offset: u64,
    /// `data_fork_offset` — absolute byte offset of the compressed
    /// chunk data within the file.
    pub data_fork_offset: u64,
    /// `data_fork_length` — bytes of compressed data.
    pub data_fork_length: u64,
    /// `rsrc_fork_offset` — absolute byte offset of the legacy NDIF
    /// resource fork. Usually 0 on UDIF.
    pub rsrc_fork_offset: u64,
    /// `rsrc_fork_length`.
    pub rsrc_fork_length: u64,
    /// 1-based segment index of this file in a multi-segment image.
    pub segment_number: u32,
    /// Total segment count.
    pub segment_count: u32,
    /// Stable 16-byte segment identifier shared across all segments of
    /// a multi-segment image.
    pub segment_id: [u8; 16],
    /// CRC over the data fork: `data_checksum_type` (1 = CRC32) +
    /// `data_checksum_size` + `data_checksum[]`.
    pub data_checksum_type: u32,
    pub data_checksum_size: u32,
    pub data_checksum: [u8; 128],
    /// Byte offset + length of the XML / binary-plist resource that
    /// carries the per-chunk `mish` table.
    pub xml_offset: u64,
    pub xml_length: u64,
    /// Master image checksum: same shape as the data checksum.
    pub master_checksum_type: u32,
    pub master_checksum_size: u32,
    pub master_checksum: [u8; 128],
    /// `image_variant` — implementation-defined.
    pub image_variant: u32,
    /// `sector_count` — virtual size in 512-byte sectors. Total virtual
    /// bytes are `sector_count * 512`.
    pub sector_count: u64,
}

impl KolyTrailer {
    /// Decode the 512-byte trailer from a buffer whose first 512 bytes
    /// are the koly block. Returns `Err` if the magic doesn't match or
    /// the version isn't 4 (older NDIF / future versions out of scope).
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < KOLY_SIZE as usize {
            return Err(crate::Error::InvalidImage(
                "dmg: trailer slice shorter than 512 bytes".into(),
            ));
        }
        let signature = u32::from_be_bytes(buf[0x000..0x004].try_into().unwrap());
        if signature != KOLY_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "dmg: koly magic mismatch (got {signature:#010x})"
            )));
        }
        let version = u32::from_be_bytes(buf[0x004..0x008].try_into().unwrap());
        if version != 4 {
            return Err(crate::Error::Unsupported(format!(
                "dmg: koly version {version} not supported (only v4)"
            )));
        }
        let mut segment_id = [0u8; 16];
        segment_id.copy_from_slice(&buf[0x040..0x050]);
        let mut data_checksum = [0u8; 128];
        data_checksum.copy_from_slice(&buf[0x058..0x0D8]);
        let mut master_checksum = [0u8; 128];
        master_checksum.copy_from_slice(&buf[0x168..0x1E8]);
        Ok(Self {
            signature,
            version,
            header_size: u32::from_be_bytes(buf[0x008..0x00C].try_into().unwrap()),
            flags: u32::from_be_bytes(buf[0x00C..0x010].try_into().unwrap()),
            running_data_fork_offset: u64::from_be_bytes(buf[0x010..0x018].try_into().unwrap()),
            data_fork_offset: u64::from_be_bytes(buf[0x018..0x020].try_into().unwrap()),
            data_fork_length: u64::from_be_bytes(buf[0x020..0x028].try_into().unwrap()),
            rsrc_fork_offset: u64::from_be_bytes(buf[0x028..0x030].try_into().unwrap()),
            rsrc_fork_length: u64::from_be_bytes(buf[0x030..0x038].try_into().unwrap()),
            segment_number: u32::from_be_bytes(buf[0x038..0x03C].try_into().unwrap()),
            segment_count: u32::from_be_bytes(buf[0x03C..0x040].try_into().unwrap()),
            segment_id,
            data_checksum_type: u32::from_be_bytes(buf[0x050..0x054].try_into().unwrap()),
            data_checksum_size: u32::from_be_bytes(buf[0x054..0x058].try_into().unwrap()),
            data_checksum,
            xml_offset: u64::from_be_bytes(buf[0x0D8..0x0E0].try_into().unwrap()),
            xml_length: u64::from_be_bytes(buf[0x0E0..0x0E8].try_into().unwrap()),
            master_checksum_type: u32::from_be_bytes(buf[0x160..0x164].try_into().unwrap()),
            master_checksum_size: u32::from_be_bytes(buf[0x164..0x168].try_into().unwrap()),
            master_checksum,
            image_variant: u32::from_be_bytes(buf[0x1E8..0x1EC].try_into().unwrap()),
            sector_count: u64::from_be_bytes(buf[0x1EC..0x1F4].try_into().unwrap()),
        })
    }
}

/// Cheap detector: probe a file's last 512 bytes for the koly trailer.
/// Returns `Ok(true)` only when the magic matches; any I/O failure or
/// size-too-small condition returns `Ok(false)` so callers can fall
/// through to other backends.
pub fn probe(path: &Path) -> Result<bool> {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return Ok(false),
    };
    if meta.len() < KOLY_SIZE {
        return Ok(false);
    }
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };
    let trailer_offset = meta.len() - KOLY_SIZE;
    f.seek(SeekFrom::Start(trailer_offset))?;
    let mut head = [0u8; 4];
    if f.read_exact(&mut head).is_err() {
        return Ok(false);
    }
    Ok(u32::from_be_bytes(head) == KOLY_MAGIC)
}

/// Read-only DMG backend.
///
/// Today this opens the file, parses the koly trailer, and exposes the
/// virtual size via [`BlockDevice::total_size`]. `read_at` returns
/// [`crate::Error::Unsupported`] until the chunk decoder lands — at
/// which point this scaffold will look the same to callers, just with
/// real bytes coming out.
#[derive(Debug)]
pub struct DmgBackend {
    /// The backing `.dmg` file. Held by the scaffold so a future
    /// chunk-decoder pass can `pread` at `data_fork_offset`. The
    /// `dead_code` allow is temporary; remove once the decoder lands.
    #[allow(dead_code)]
    file: File,
    trailer: KolyTrailer,
    /// Cached virtual size in bytes (`sector_count * 512`).
    virtual_size: u64,
    /// Position of the implicit `Seek` cursor — kept so the `Seek`
    /// impl works the way callers expect from a `BlockDevice`.
    cursor: u64,
}

impl DmgBackend {
    /// Open a DMG file. Validates the koly trailer, the version field
    /// (must be 4), and that the trailer's sector_count fits in i64.
    /// Does not yet load the resource-fork chunk table.
    pub fn open(path: &Path) -> Result<Self> {
        let meta = std::fs::metadata(path)?;
        if meta.len() < KOLY_SIZE {
            return Err(crate::Error::InvalidImage(
                "dmg: file smaller than the 512-byte koly trailer".into(),
            ));
        }
        let mut file = File::open(path)?;
        let trailer_offset = meta.len() - KOLY_SIZE;
        file.seek(SeekFrom::Start(trailer_offset))?;
        let mut buf = vec![0u8; KOLY_SIZE as usize];
        file.read_exact(&mut buf)?;
        let trailer = KolyTrailer::decode(&buf)?;
        if trailer.segment_count > 1 {
            return Err(crate::Error::Unsupported(format!(
                "dmg: multi-segment images not supported (segment_count = {})",
                trailer.segment_count
            )));
        }
        let virtual_size = trailer
            .sector_count
            .checked_mul(512)
            .ok_or_else(|| crate::Error::InvalidImage("dmg: sector_count overflows u64".into()))?;
        Ok(Self {
            file,
            trailer,
            virtual_size,
            cursor: 0,
        })
    }

    /// Borrow the decoded trailer for diagnostics.
    pub fn trailer(&self) -> &KolyTrailer {
        &self.trailer
    }
}

impl BlockDevice for DmgBackend {
    fn block_size(&self) -> u32 {
        512
    }

    fn total_size(&self) -> u64 {
        self.virtual_size
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }

    fn read_at(&mut self, _offset: u64, _buf: &mut [u8]) -> Result<()> {
        Err(crate::Error::Unsupported(
            "dmg: chunk decompression is not implemented yet — this scaffold \
             only parses the koly trailer + reports virtual size"
                .into(),
        ))
    }

    fn write_at(&mut self, _offset: u64, _buf: &[u8]) -> Result<()> {
        Err(crate::Error::Unsupported(
            "dmg: read-only container; writes are out of scope".into(),
        ))
    }
}

impl Read for DmgBackend {
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::other(
            "dmg: chunk decompression is not implemented yet",
        ))
    }
}

impl Write for DmgBackend {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("dmg: read-only container"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for DmgBackend {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let total = self.virtual_size;
        let new = match pos {
            SeekFrom::Start(o) => o,
            SeekFrom::Current(d) => (self.cursor as i64).saturating_add(d).max(0) as u64,
            SeekFrom::End(d) => (total as i64).saturating_add(d).max(0) as u64,
        };
        self.cursor = new;
        Ok(new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal koly trailer with the given sector_count. Big-endian
    /// fields; everything else stays zero.
    fn fake_trailer(sector_count: u64, version: u32) -> Vec<u8> {
        let mut v = vec![0u8; KOLY_SIZE as usize];
        v[0x000..0x004].copy_from_slice(&KOLY_MAGIC.to_be_bytes());
        v[0x004..0x008].copy_from_slice(&version.to_be_bytes());
        v[0x008..0x00C].copy_from_slice(&512u32.to_be_bytes());
        v[0x1EC..0x1F4].copy_from_slice(&sector_count.to_be_bytes());
        v
    }

    #[test]
    fn decode_recognises_valid_trailer() {
        let buf = fake_trailer(2048, 4);
        let t = KolyTrailer::decode(&buf).unwrap();
        assert_eq!(t.signature, KOLY_MAGIC);
        assert_eq!(t.version, 4);
        assert_eq!(t.header_size, 512);
        assert_eq!(t.sector_count, 2048);
    }

    #[test]
    fn decode_rejects_wrong_magic() {
        let mut buf = fake_trailer(0, 4);
        buf[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        let err = KolyTrailer::decode(&buf).unwrap_err();
        match err {
            crate::Error::InvalidImage(_) => {}
            _ => panic!("expected InvalidImage, got {err:?}"),
        }
    }

    #[test]
    fn decode_rejects_unknown_version() {
        let buf = fake_trailer(0, 3);
        let err = KolyTrailer::decode(&buf).unwrap_err();
        match err {
            crate::Error::Unsupported(_) => {}
            _ => panic!("expected Unsupported, got {err:?}"),
        }
    }

    #[test]
    fn probe_matches_trailing_koly() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("img.dmg");
        // 8 KiB of zero data + 512-byte trailer.
        let mut content = vec![0u8; 8192];
        content.extend_from_slice(&fake_trailer(16, 4));
        std::fs::write(&p, &content).unwrap();
        assert!(probe(&p).unwrap());
    }

    #[test]
    fn probe_misses_when_no_trailer() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("not.dmg");
        std::fs::write(&p, vec![0u8; 8192]).unwrap();
        assert!(!probe(&p).unwrap());
    }

    #[test]
    fn open_reports_virtual_size() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("img.dmg");
        let mut content = vec![0u8; 8192];
        content.extend_from_slice(&fake_trailer(2048, 4));
        std::fs::write(&p, &content).unwrap();
        let dmg = DmgBackend::open(&p).unwrap();
        assert_eq!(dmg.total_size(), 2048 * 512);
        assert_eq!(dmg.block_size(), 512);
        assert_eq!(dmg.trailer().sector_count, 2048);
    }

    #[test]
    fn open_rejects_multi_segment() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("img.dmg");
        let mut t = fake_trailer(0, 4);
        // segment_count at 0x03C
        t[0x03C..0x040].copy_from_slice(&3u32.to_be_bytes());
        let mut content = vec![0u8; 8192];
        content.extend_from_slice(&t);
        std::fs::write(&p, &content).unwrap();
        let err = DmgBackend::open(&p).unwrap_err();
        match err {
            crate::Error::Unsupported(_) => {}
            _ => panic!("expected Unsupported, got {err:?}"),
        }
    }

    #[test]
    fn read_at_returns_unsupported() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("img.dmg");
        let mut content = vec![0u8; 8192];
        content.extend_from_slice(&fake_trailer(16, 4));
        std::fs::write(&p, &content).unwrap();
        let mut dmg = DmgBackend::open(&p).unwrap();
        let mut buf = [0u8; 16];
        let err = dmg.read_at(0, &mut buf).unwrap_err();
        match err {
            crate::Error::Unsupported(_) => {}
            _ => panic!("expected Unsupported, got {err:?}"),
        }
    }
}
