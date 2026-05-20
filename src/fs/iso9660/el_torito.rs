//! El Torito Bootable CD-ROM Format (Phoenix / IBM, 1995).
//!
//! Layout:
//!
//! ```text
//! Boot Record (LBA 17, type 0)
//!   bytes 0x47..4B   le u32   catalog_lba           ← parsed by vd.rs
//!
//! Boot Catalog (one sector at catalog_lba)
//!   bytes 0..32      Validation Entry
//!                    header_id (0x01) | platform | reserved | id_string | checksum | 0x55 0xAA
//!   bytes 32..64     Default boot entry (Section Entry)
//!                    boot_indicator (0x88=bootable) | media_type | load_segment | system_type
//!                    | unused | sector_count | load_rba | reserved
//!   bytes 64+        Optional Section Header + Section Entries (additional boot images)
//! ```
//!
//! We parse the validation + default entry and capture every section
//! that follows, so the writer can recreate them on repack. No
//! verification of the boot-image checksum is required by spec.

use crate::Result;
use crate::block::BlockDevice;

use super::SECTOR_SIZE;

/// One El Torito boot entry. The default entry plus any section entries
/// share the same on-disk shape (32 bytes); we surface a tagged version.
#[derive(Debug, Clone)]
pub struct BootEntry {
    /// True for `boot_indicator == 0x88`.
    pub bootable: bool,
    /// `media_type` byte (0 = no emulation, 1 = 1.2 MB floppy, 2 = 1.44 MB,
    /// 3 = 2.88 MB, 4 = hard disk).
    pub media_type: u8,
    /// Real-mode load segment (0 = default 7C0h).
    pub load_segment: u16,
    /// `system_type` byte from the partition table for hard-disk emul.
    pub system_type: u8,
    /// Number of virtual / emulated sectors loaded by the BIOS.
    pub sector_count: u16,
    /// LBA of the boot image data.
    pub load_rba: u32,
    /// Platform: 0 = x86, 1 = PPC, 2 = Mac, 0xEF = EFI. Set on the
    /// validation entry / section header; copied here for convenience.
    pub platform: u8,
}

/// Parsed El Torito catalog. Reproduced verbatim by the writer when a
/// source ISO is repacked into a new ISO.
#[derive(Debug, Clone)]
pub struct BootCatalog {
    pub default_entry: BootEntry,
    pub additional: Vec<BootEntry>,
    pub id_string: String,
}

impl BootCatalog {
    /// Read the boot catalog at `catalog_lba` and decode it.
    pub fn load(dev: &mut dyn BlockDevice, catalog_lba: u32) -> Result<Self> {
        let mut buf = vec![0u8; SECTOR_SIZE as usize];
        dev.read_at(u64::from(catalog_lba) * u64::from(SECTOR_SIZE), &mut buf)?;

        // Validation entry: bytes 0..32.
        if buf[0] != 0x01 {
            return Err(crate::Error::InvalidImage(
                "el-torito: validation entry missing header_id 0x01".into(),
            ));
        }
        let platform = buf[1];
        let id_string = String::from_utf8_lossy(&buf[4..28])
            .trim_end_matches(['\0', ' '])
            .to_string();
        if buf[30] != 0x55 || buf[31] != 0xAA {
            return Err(crate::Error::InvalidImage(
                "el-torito: validation entry missing 0x55AA terminator".into(),
            ));
        }

        // Default entry: bytes 32..64.
        let default_entry = decode_entry(&buf[32..64], platform)?;

        // Walk additional sections (each header is 32 bytes, indicator
        // 0x90 = "more sections follow", 0x91 = "final section").
        let mut additional = Vec::new();
        let mut cursor = 64usize;
        let mut current_platform;
        while cursor + 32 <= buf.len() {
            let hdr = &buf[cursor..cursor + 32];
            match hdr[0] {
                0x90 | 0x91 => {
                    current_platform = hdr[1];
                    let n_entries = u16::from_le_bytes([hdr[2], hdr[3]]) as usize;
                    let final_section = hdr[0] == 0x91;
                    cursor += 32;
                    for _ in 0..n_entries {
                        if cursor + 32 > buf.len() {
                            break;
                        }
                        let ent = decode_entry(&buf[cursor..cursor + 32], current_platform).ok();
                        if let Some(e) = ent {
                            additional.push(e);
                        }
                        cursor += 32;
                    }
                    if final_section {
                        break;
                    }
                }
                _ => break, // anything else is end-of-catalog padding
            }
        }

        Ok(Self {
            default_entry,
            additional,
            id_string,
        })
    }
}

fn decode_entry(buf: &[u8], platform: u8) -> Result<BootEntry> {
    if buf.len() < 32 {
        return Err(crate::Error::InvalidImage(
            "el-torito: short section entry".into(),
        ));
    }
    Ok(BootEntry {
        bootable: buf[0] == 0x88,
        media_type: buf[1],
        load_segment: u16::from_le_bytes([buf[2], buf[3]]),
        system_type: buf[4],
        sector_count: u16::from_le_bytes([buf[6], buf[7]]),
        load_rba: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
        platform,
    })
}
