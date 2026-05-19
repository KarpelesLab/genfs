//! NTFS — Microsoft's NT File System. Scaffold module.
//!
//! ## Status
//!
//! Detection + BPB parse only. The MFT, attribute decoding, and
//! directory-index B-tree walks are TBD. Write support is further out
//! still; when it lands, NTFS-targeted writes preserve the source's
//! Windows-native attributes (DOS flags, ADS, security descriptors)
//! byte-for-byte rather than translating them.
//!
//! ## Reference
//!
//! - Microsoft "[MS-FSCC] File System Control Codes" — the public
//!   reference for NTFS file information classes:
//!   <https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-fscc/>
//! - Linux kernel "NTFS3" docs:
//!   <https://docs.kernel.org/filesystems/ntfs3.html>
//! - "NTFS Documentation" by Richard Russon and Yuval Fledel (the
//!   long-standing community reference document).
//!
//! Where the public docs leave gaps we cross-check against the on-disk
//! behaviour of a fresh `mkntfs` image rather than reading any kernel /
//! ntfs-3g source.
//!
//! ## Attribute model — non-Unix metadata
//!
//! NTFS metadata doesn't map cleanly onto POSIX. The shape we adopt for
//! cross-FS conversion is:
//!
//! | NTFS concept                              | xattr key                                | Notes                                                    |
//! |-------------------------------------------|------------------------------------------|----------------------------------------------------------|
//! | `$STANDARD_INFORMATION.file_attributes`   | `user.ntfs.dos_attrs`                    | 32-bit LE: READONLY/HIDDEN/SYSTEM/ARCHIVE/COMPRESSED/etc |
//! | Object ID GUID (`$OBJECT_ID`)             | `user.ntfs.object_id`                    | 16 bytes raw GUID                                        |
//! | Reparse point tag + data (`$REPARSE_POINT`)| `user.ntfs.reparse`                     | Tag (LE u32) prepended to raw reparse data               |
//! | Alternate Data Streams (named `$DATA`)    | `user.ntfs.ads.<name>`                   | Per-stream xattr; binary stream contents                 |
//! | `$SECURITY_DESCRIPTOR` (raw NT SD blob)   | `system.ntfs_security`                   | Self-relative SD blob; consumers who understand it       |
//! |                                           |                                          | can decode to SDDL                                       |
//! | Short (8.3) filename                      | `user.ntfs.short_name`                   | UTF-16LE per `$FILE_NAME` with namespace=DOS             |
//! | Last-write / creation / change / access   | inode timestamps + `user.ntfs.times.raw` | The latter holds all four NT-FILETIME (100 ns) values    |
//! |                                           |                                          | so NTFS→NTFS keeps sub-second precision intact           |
//! | Hard-link namespace (`$FILE_NAME` per link)| (preserved by tree shape)                | We replicate the link, copying its own dos_attrs etc.    |
//!
//! Why xattrs: every FS we target (ext, XFS, exFAT-with-extensions,
//! tar/PAX) either supports `user.*` xattrs natively or has a documented
//! way to carry them. Round-tripping through tar (PAX `SCHILY.xattr.*`
//! records) keeps the metadata intact across tools.
//!
//! Why `system.ntfs_security` (not `user.*`): security descriptors are
//! the bit closest to POSIX ACLs, and `system.*` is the conventional
//! namespace for kernel-validated metadata. ext4's ACL EAs already live
//! there. tar's PAX format carries `system.*` xattrs as well.
//!
//! ### NTFS → NTFS round-trip guarantee
//!
//! When the source and destination FS are both NTFS, the writer takes
//! the **raw attribute byte stream** from the source's MFT record and
//! re-emits it verbatim, rather than going through the xattr indirection.
//! This is what makes the "NTFS to NTFS results in the exact same
//! attributes" promise tractable — we don't lose precision converting to
//! and from a lossy intermediate representation. The xattr mapping above
//! is reserved for the cross-FS case.
//!
//! ## Layout overview (for the future implementation)
//!
//! NTFS is centred on the Master File Table (MFT), itself a regular
//! file whose record-0 entry describes its own layout. Key structures:
//!
//! ```text
//!   sector 0 .. cluster 0:  Boot sector (BPB + extended NTFS fields)
//!   logical cluster (lcn_mft):       MFT (file 0)
//!   logical cluster (lcn_mft_mirr):  MFT mirror (first 4 records)
//!   …                                user files (records 24+)
//! ```
//!
//! Every metadata read goes through the MFT: open MFT record 5 to walk
//! the root directory, follow indexes (`$INDEX_ROOT` / `$INDEX_ALLOCATION`)
//! to descend, read `$DATA` attribute (resident or run-list) for file
//! contents. The scaffold does none of this yet.

use crate::Result;
use crate::block::BlockDevice;

/// First 8 bytes of an NTFS boot sector: `0xEB 0x52 0x90` jump
/// instruction followed by the OEM ID `"NTFS    "`. The jump-instruction
/// prefix is not unique across FAT/exFAT/NTFS, so we anchor on the OEM
/// ID at offset 3.
const NTFS_OEM: &[u8; 8] = b"NTFS    ";

/// Decoded NTFS boot-sector / BPB fields. Multi-byte fields are LE.
#[derive(Debug, Clone)]
pub struct BootSector {
    pub bytes_per_sector: u16,
    pub sectors_per_cluster: u8,
    /// `total_sectors` from BPB: total sectors in volume - 1.
    pub total_sectors: u64,
    /// Logical cluster number of the MFT's first record.
    pub mft_lcn: u64,
    /// Logical cluster number of the MFT mirror's first record.
    pub mft_mirr_lcn: u64,
    /// `clusters_per_mft_record`. Positive: clusters per record. Negative
    /// (interpreted as signed i8): `record_size = 1 << -value` bytes
    /// (used when records are smaller than a cluster, e.g. 1024 bytes).
    pub clusters_per_mft_record: i8,
    pub clusters_per_index_record: i8,
    pub volume_serial: u64,
}

impl BootSector {
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 80 || &buf[3..11] != NTFS_OEM {
            return None;
        }
        let bytes_per_sector = u16::from_le_bytes(buf[11..13].try_into().ok()?);
        let sectors_per_cluster = buf[13];
        // BPB has zeros for reserved/old fields; total_sectors lives at 0x28.
        let total_sectors = u64::from_le_bytes(buf[0x28..0x30].try_into().ok()?);
        let mft_lcn = u64::from_le_bytes(buf[0x30..0x38].try_into().ok()?);
        let mft_mirr_lcn = u64::from_le_bytes(buf[0x38..0x40].try_into().ok()?);
        let clusters_per_mft_record = buf[0x40] as i8;
        let clusters_per_index_record = buf[0x44] as i8;
        let volume_serial = u64::from_le_bytes(buf[0x48..0x50].try_into().ok()?);
        Some(Self {
            bytes_per_sector,
            sectors_per_cluster,
            total_sectors,
            mft_lcn,
            mft_mirr_lcn,
            clusters_per_mft_record,
            clusters_per_index_record,
            volume_serial,
        })
    }

    /// Resolve the MFT record size in bytes from the BPB field. Either
    /// `clusters_per_mft_record * cluster_size` (positive) or
    /// `1 << -value` bytes (negative; the canonical 1024-byte case).
    pub fn mft_record_size(&self) -> u32 {
        let v = self.clusters_per_mft_record;
        if v >= 0 {
            (v as u32) * u32::from(self.bytes_per_sector) * u32::from(self.sectors_per_cluster)
        } else {
            1u32 << (-(v as i32))
        }
    }

    /// Cluster size = sector size * sectors-per-cluster.
    pub fn cluster_size(&self) -> u32 {
        u32::from(self.bytes_per_sector) * u32::from(self.sectors_per_cluster)
    }
}

pub fn probe(dev: &mut dyn BlockDevice) -> Result<bool> {
    if dev.total_size() < 11 {
        return Ok(false);
    }
    let mut head = [0u8; 11];
    dev.read_at(0, &mut head)?;
    Ok(&head[3..11] == NTFS_OEM)
}

pub struct Ntfs {
    boot: BootSector,
}

impl Ntfs {
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        if dev.total_size() < 512 {
            return Err(crate::Error::InvalidImage(
                "ntfs: device too small to hold a boot sector".into(),
            ));
        }
        let mut buf = [0u8; 512];
        dev.read_at(0, &mut buf)?;
        let boot = BootSector::decode(&buf).ok_or_else(|| {
            crate::Error::InvalidImage("ntfs: boot sector OEM ID is not 'NTFS    '".into())
        })?;
        Ok(Self { boot })
    }

    pub fn total_bytes(&self) -> u64 {
        self.boot.total_sectors * u64::from(self.boot.bytes_per_sector)
    }

    pub fn cluster_size(&self) -> u32 {
        self.boot.cluster_size()
    }

    pub fn bytes_per_sector(&self) -> u16 {
        self.boot.bytes_per_sector
    }

    pub fn sectors_per_cluster(&self) -> u8 {
        self.boot.sectors_per_cluster
    }

    pub fn mft_record_size(&self) -> u32 {
        self.boot.mft_record_size()
    }

    pub fn volume_serial(&self) -> u64 {
        self.boot.volume_serial
    }

    pub fn boot_sector(&self) -> &BootSector {
        &self.boot
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
    crate::Error::Unsupported("ntfs: read support is not implemented yet (scaffold only)".into())
}

/// Names for the xattr namespace this driver will use when round-tripping
/// NTFS metadata through other filesystems. Kept as constants so callers
/// (and future tests) can refer to them rather than hard-coding the strings.
pub mod xattr_keys {
    /// $STANDARD_INFORMATION.file_attributes (32-bit LE).
    pub const DOS_ATTRS: &str = "user.ntfs.dos_attrs";
    /// $OBJECT_ID GUID (16 raw bytes).
    pub const OBJECT_ID: &str = "user.ntfs.object_id";
    /// Reparse-point tag (LE u32) followed by raw reparse data.
    pub const REPARSE: &str = "user.ntfs.reparse";
    /// Alternate Data Streams; full key is `user.ntfs.ads.<name>`.
    pub const ADS_PREFIX: &str = "user.ntfs.ads.";
    /// Self-relative NT SECURITY_DESCRIPTOR blob.
    pub const SECURITY: &str = "system.ntfs_security";
    /// Short 8.3 filename (UTF-16LE from a $FILE_NAME with namespace=DOS).
    pub const SHORT_NAME: &str = "user.ntfs.short_name";
    /// Raw NT-FILETIME quadruple (create, modify, change, access) at 100 ns
    /// granularity, 4 × 8 = 32 bytes LE.
    pub const TIMES_RAW: &str = "user.ntfs.times.raw";
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_boot(bps: u16, spc: u8, mft_lcn: u64, mft_rec: i8) -> Vec<u8> {
        let mut v = vec![0u8; 512];
        v[0..3].copy_from_slice(&[0xEB, 0x52, 0x90]);
        v[3..11].copy_from_slice(NTFS_OEM);
        v[11..13].copy_from_slice(&bps.to_le_bytes());
        v[13] = spc;
        v[0x28..0x30].copy_from_slice(&1024u64.to_le_bytes());
        v[0x30..0x38].copy_from_slice(&mft_lcn.to_le_bytes());
        v[0x38..0x40].copy_from_slice(&(mft_lcn + 1).to_le_bytes());
        v[0x40] = mft_rec as u8;
        v[0x44] = 1;
        v[0x48..0x50].copy_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
        v
    }

    #[test]
    fn decode_recognises_oem_id() {
        let buf = fake_boot(512, 8, 4, -10); // 1024-byte MFT records
        let bs = BootSector::decode(&buf).unwrap();
        assert_eq!(bs.bytes_per_sector, 512);
        assert_eq!(bs.sectors_per_cluster, 8);
        assert_eq!(bs.mft_record_size(), 1024);
        assert_eq!(bs.cluster_size(), 4096);
    }

    #[test]
    fn decode_handles_positive_clusters_per_mft_record() {
        let buf = fake_boot(512, 8, 4, 2); // 2 clusters * 4096 = 8192-byte records
        let bs = BootSector::decode(&buf).unwrap();
        assert_eq!(bs.mft_record_size(), 8192);
    }

    #[test]
    fn decode_rejects_wrong_oem() {
        let mut buf = fake_boot(512, 8, 4, -10);
        buf[3..11].copy_from_slice(b"EXFAT   ");
        assert!(BootSector::decode(&buf).is_none());
    }

    #[test]
    fn probe_detects_ntfs() {
        use crate::block::MemoryBackend;
        let mut dev = MemoryBackend::new(4096);
        dev.write_at(0, &fake_boot(512, 8, 4, -10)).unwrap();
        assert!(probe(&mut dev).unwrap());
    }

    #[test]
    fn list_path_returns_unsupported_in_scaffold() {
        use crate::block::MemoryBackend;
        let mut dev = MemoryBackend::new(4096);
        dev.write_at(0, &fake_boot(512, 8, 4, -10)).unwrap();
        let mut ntfs = Ntfs::open(&mut dev).unwrap();
        let err = ntfs.list_path(&mut dev, "/").unwrap_err();
        match err {
            crate::Error::Unsupported(_) => {}
            _ => panic!("expected Unsupported"),
        }
    }
}
