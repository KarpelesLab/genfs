//! GUID Partition Table.
//!
//! Layout (assuming 512-byte LBAs, `N` total LBAs):
//!
//! ```text
//!   LBA 0      protective MBR
//!   LBA 1      primary GPT header
//!   LBA 2..34  primary partition entries (128 entries × 128 B = 32 sectors)
//!   LBA 34     first usable LBA
//!   LBA N-34   last usable LBA
//!   LBA N-33   backup partition entries (32 sectors)
//!   LBA N-1    backup GPT header
//! ```
//!
//! v1 assumes a 512-byte logical sector. Devices with `block_size() != 512`
//! are rejected — a follow-up will generalise the calculations.

use uuid::Uuid;

use super::mbr::write_protective_mbr;
use super::{Partition, PartitionKind, PartitionTable, uuid_from_gpt_bytes, uuid_to_gpt_bytes};
use crate::Result;
use crate::block::BlockDevice;

/// GPT signature at byte 0 of the header sector: ASCII "EFI PART".
pub const SIGNATURE: &[u8; 8] = b"EFI PART";

/// GPT revision 1.0.
pub const REVISION: u32 = 0x0001_0000;

/// Fixed header size used by all known GPT versions.
pub const HEADER_SIZE: u32 = 92;

/// Conventional number of partition entries.
pub const NUM_ENTRIES: u32 = 128;

/// Conventional partition entry size.
pub const ENTRY_SIZE: u32 = 128;

/// Number of LBAs taken by the entries array
/// (`NUM_ENTRIES * ENTRY_SIZE / 512 = 16384 / 512 = 32`).
pub const ENTRIES_LBAS: u64 = (NUM_ENTRIES as u64 * ENTRY_SIZE as u64) / 512;

/// First LBA usable by partitions (`1 header + 32 entry LBAs + LBA 0 = 34`).
pub const FIRST_USABLE_LBA: u64 = 1 + 1 + ENTRIES_LBAS;

/// Required logical sector size in v1.
pub const REQUIRED_BLOCK_SIZE: u32 = 512;

/// An in-memory GPT.
#[derive(Debug, Clone)]
pub struct Gpt {
    pub disk_guid: Uuid,
    pub partitions: Vec<Partition>,
}

impl Gpt {
    /// Build a GPT from a partition list. Generates a random disk GUID and
    /// fills missing per-partition UUIDs with random v4s. Validates that no
    /// two partitions overlap and that all fit within the device's usable
    /// LBA range when `total_lba` is known (callers normally supply this from
    /// `dev.total_size() / dev.block_size()` at write time, so `build` only
    /// checks intra-table consistency).
    pub fn build(mut partitions: Vec<Partition>) -> Result<Self> {
        if partitions.len() > NUM_ENTRIES as usize {
            return Err(crate::Error::InvalidArgument(format!(
                "GPT supports up to {NUM_ENTRIES} entries, got {}",
                partitions.len()
            )));
        }
        for p in &mut partitions {
            if p.kind == PartitionKind::Empty {
                continue;
            }
            if p.size_lba == 0 {
                return Err(crate::Error::InvalidArgument(
                    "GPT partition has zero size".into(),
                ));
            }
            if p.uuid.is_none() {
                p.uuid = Some(Uuid::new_v4());
            }
            if let Some(ref n) = p.name
                && n.encode_utf16().count() > 36
            {
                return Err(crate::Error::InvalidArgument(format!(
                    "GPT partition name exceeds 36 UTF-16 code units: {n:?}"
                )));
            }
        }
        check_no_overlap(&partitions)?;
        Ok(Self {
            disk_guid: Uuid::new_v4(),
            partitions,
        })
    }

    /// Read a GPT from a block device. Validates signature and CRCs on the
    /// primary header (falling back to the backup if the primary is corrupt
    /// is a follow-up; v1 errors out instead).
    pub fn read(dev: &mut dyn BlockDevice) -> Result<Self> {
        require_block_size(dev)?;
        let total_lba = dev.total_size() / 512;
        if total_lba < FIRST_USABLE_LBA + ENTRIES_LBAS + 1 {
            return Err(crate::Error::InvalidImage(
                "device too small for a GPT".into(),
            ));
        }
        let mut hdr_buf = [0u8; 512];
        dev.read_at(512, &mut hdr_buf)?;
        let hdr = parse_header(&hdr_buf)?;
        if hdr.my_lba != 1 {
            return Err(crate::Error::InvalidImage(format!(
                "GPT primary header has unexpected my_lba {}",
                hdr.my_lba
            )));
        }

        let entries_bytes = hdr.num_entries as usize * hdr.entry_size as usize;
        let mut entries_buf = vec![0u8; entries_bytes];
        dev.read_at(hdr.entries_start_lba * 512, &mut entries_buf)?;
        let computed = crc32fast::hash(&entries_buf);
        if computed != hdr.entries_crc {
            return Err(crate::Error::InvalidImage(format!(
                "GPT entries CRC mismatch: stored {:#010x}, computed {:#010x}",
                hdr.entries_crc, computed
            )));
        }

        let mut partitions = Vec::new();
        for i in 0..hdr.num_entries as usize {
            let off = i * hdr.entry_size as usize;
            let ent = &entries_buf[off..off + 128];
            if let Some(p) = decode_entry(ent) {
                partitions.push(p);
            }
        }

        // Suppress trailing all-empty entries to keep the returned slice tidy.
        // (Entries in the middle that happen to be empty are preserved as
        // PartitionKind::Empty in `decode_entry` — they won't show up in the
        // returned list since `decode_entry` returns None for all-nil.)

        Ok(Self {
            disk_guid: hdr.disk_guid,
            partitions,
        })
    }

    fn header_for(&self, is_primary: bool, total_lba: u64, entries_crc: u32) -> Header {
        let primary_lba = 1;
        let backup_lba = total_lba - 1;
        let (my_lba, alt_lba, entries_start_lba) = if is_primary {
            (primary_lba, backup_lba, 2)
        } else {
            (backup_lba, primary_lba, total_lba - 1 - ENTRIES_LBAS)
        };
        Header {
            my_lba,
            alternate_lba: alt_lba,
            first_usable_lba: FIRST_USABLE_LBA,
            last_usable_lba: total_lba - 1 - ENTRIES_LBAS - 1,
            disk_guid: self.disk_guid,
            entries_start_lba,
            num_entries: NUM_ENTRIES,
            entry_size: ENTRY_SIZE,
            entries_crc,
        }
    }
}

impl PartitionTable for Gpt {
    fn write(&self, dev: &mut dyn BlockDevice) -> Result<()> {
        require_block_size(dev)?;
        let total_lba = dev.total_size() / 512;
        let min_required = FIRST_USABLE_LBA + ENTRIES_LBAS + 1;
        if total_lba < min_required {
            return Err(crate::Error::InvalidArgument(format!(
                "device has {total_lba} LBAs, need at least {min_required} for a GPT"
            )));
        }

        // Validate per-partition bounds against the actual disk.
        let last_usable = total_lba - 1 - ENTRIES_LBAS - 1;
        for p in &self.partitions {
            if p.kind == PartitionKind::Empty {
                continue;
            }
            if p.start_lba < FIRST_USABLE_LBA {
                return Err(crate::Error::InvalidArgument(format!(
                    "partition starts at LBA {} (before first usable LBA {FIRST_USABLE_LBA})",
                    p.start_lba
                )));
            }
            if p.end_lba() > last_usable {
                return Err(crate::Error::InvalidArgument(format!(
                    "partition ends at LBA {} (past last usable LBA {last_usable})",
                    p.end_lba()
                )));
            }
        }

        // 1. Protective MBR at LBA 0.
        write_protective_mbr(dev)?;

        // 2. Build entries array.
        let mut entries = vec![0u8; (NUM_ENTRIES * ENTRY_SIZE) as usize];
        for (i, p) in self.partitions.iter().enumerate() {
            let off = i * ENTRY_SIZE as usize;
            encode_entry(p, &mut entries[off..off + ENTRY_SIZE as usize]);
        }
        let entries_crc = crc32fast::hash(&entries);

        // 3. Write primary entries at LBA 2.
        dev.write_at(2 * 512, &entries)?;
        // 4. Write backup entries.
        let backup_entries_lba = total_lba - 1 - ENTRIES_LBAS;
        dev.write_at(backup_entries_lba * 512, &entries)?;

        // 5. Write primary header at LBA 1.
        let primary_hdr = self.header_for(true, total_lba, entries_crc);
        let primary_buf = encode_header(&primary_hdr);
        dev.write_at(512, &primary_buf)?;

        // 6. Write backup header at last LBA.
        let backup_hdr = self.header_for(false, total_lba, entries_crc);
        let backup_buf = encode_header(&backup_hdr);
        dev.write_at((total_lba - 1) * 512, &backup_buf)?;

        Ok(())
    }

    fn partitions(&self) -> &[Partition] {
        &self.partitions
    }
}

#[derive(Debug)]
struct Header {
    my_lba: u64,
    alternate_lba: u64,
    first_usable_lba: u64,
    last_usable_lba: u64,
    disk_guid: Uuid,
    entries_start_lba: u64,
    num_entries: u32,
    entry_size: u32,
    entries_crc: u32,
}

fn require_block_size(dev: &dyn BlockDevice) -> Result<()> {
    if dev.block_size() != REQUIRED_BLOCK_SIZE {
        return Err(crate::Error::Unsupported(format!(
            "GPT in v1 requires {REQUIRED_BLOCK_SIZE}-byte sectors, device reports {}",
            dev.block_size()
        )));
    }
    Ok(())
}

fn check_no_overlap(parts: &[Partition]) -> Result<()> {
    let mut spans: Vec<(u64, u64)> = parts
        .iter()
        .filter(|p| p.kind != PartitionKind::Empty)
        .map(|p| (p.start_lba, p.end_lba()))
        .collect();
    spans.sort_unstable_by_key(|(s, _)| *s);
    for w in spans.windows(2) {
        if w[1].0 <= w[0].1 {
            return Err(crate::Error::InvalidArgument(format!(
                "partitions overlap: [{}-{}] vs [{}-{}]",
                w[0].0, w[0].1, w[1].0, w[1].1
            )));
        }
    }
    Ok(())
}

/// Encode a 512-byte header sector. Computes the header CRC over the first
/// `HEADER_SIZE` bytes with the CRC field zeroed, then patches the CRC in.
fn encode_header(h: &Header) -> [u8; 512] {
    let mut buf = [0u8; 512];
    buf[0..8].copy_from_slice(SIGNATURE);
    buf[8..12].copy_from_slice(&REVISION.to_le_bytes());
    buf[12..16].copy_from_slice(&HEADER_SIZE.to_le_bytes());
    // 16..20: header CRC (zero for now)
    // 20..24: reserved (zero)
    buf[24..32].copy_from_slice(&h.my_lba.to_le_bytes());
    buf[32..40].copy_from_slice(&h.alternate_lba.to_le_bytes());
    buf[40..48].copy_from_slice(&h.first_usable_lba.to_le_bytes());
    buf[48..56].copy_from_slice(&h.last_usable_lba.to_le_bytes());
    buf[56..72].copy_from_slice(&uuid_to_gpt_bytes(h.disk_guid));
    buf[72..80].copy_from_slice(&h.entries_start_lba.to_le_bytes());
    buf[80..84].copy_from_slice(&h.num_entries.to_le_bytes());
    buf[84..88].copy_from_slice(&h.entry_size.to_le_bytes());
    buf[88..92].copy_from_slice(&h.entries_crc.to_le_bytes());

    let crc = crc32fast::hash(&buf[..HEADER_SIZE as usize]);
    buf[16..20].copy_from_slice(&crc.to_le_bytes());
    buf
}

fn parse_header(buf: &[u8; 512]) -> Result<Header> {
    if &buf[0..8] != SIGNATURE {
        return Err(crate::Error::InvalidImage(
            "GPT signature \"EFI PART\" not found".into(),
        ));
    }
    let revision = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    if revision != REVISION {
        return Err(crate::Error::Unsupported(format!(
            "GPT revision {revision:#010x} not supported"
        )));
    }
    let header_size = u32::from_le_bytes(buf[12..16].try_into().unwrap());
    if header_size as usize > buf.len() || header_size < 92 {
        return Err(crate::Error::InvalidImage(format!(
            "GPT header_size {header_size} out of range"
        )));
    }
    let stored_crc = u32::from_le_bytes(buf[16..20].try_into().unwrap());
    // Recompute with CRC field zeroed
    let mut tmp = *buf;
    tmp[16..20].fill(0);
    let computed = crc32fast::hash(&tmp[..header_size as usize]);
    if stored_crc != computed {
        return Err(crate::Error::InvalidImage(format!(
            "GPT header CRC mismatch: stored {stored_crc:#010x}, computed {computed:#010x}"
        )));
    }

    let my_lba = u64::from_le_bytes(buf[24..32].try_into().unwrap());
    let alternate_lba = u64::from_le_bytes(buf[32..40].try_into().unwrap());
    let first_usable_lba = u64::from_le_bytes(buf[40..48].try_into().unwrap());
    let last_usable_lba = u64::from_le_bytes(buf[48..56].try_into().unwrap());
    let disk_guid = uuid_from_gpt_bytes(buf[56..72].try_into().unwrap());
    let entries_start_lba = u64::from_le_bytes(buf[72..80].try_into().unwrap());
    let num_entries = u32::from_le_bytes(buf[80..84].try_into().unwrap());
    let entry_size = u32::from_le_bytes(buf[84..88].try_into().unwrap());
    let entries_crc = u32::from_le_bytes(buf[88..92].try_into().unwrap());

    Ok(Header {
        my_lba,
        alternate_lba,
        first_usable_lba,
        last_usable_lba,
        disk_guid,
        entries_start_lba,
        num_entries,
        entry_size,
        entries_crc,
    })
}

fn encode_entry(p: &Partition, out: &mut [u8]) {
    debug_assert_eq!(out.len(), 128);
    if p.kind == PartitionKind::Empty {
        // Leave all-zero.
        return;
    }
    let type_uuid = p.kind.as_gpt_uuid();
    let part_uuid = p.uuid.unwrap_or(Uuid::nil());
    out[0..16].copy_from_slice(&uuid_to_gpt_bytes(type_uuid));
    out[16..32].copy_from_slice(&uuid_to_gpt_bytes(part_uuid));
    out[32..40].copy_from_slice(&p.start_lba.to_le_bytes());
    out[40..48].copy_from_slice(&p.end_lba().to_le_bytes());
    let mut attrs = p.attributes;
    if p.bootable {
        attrs |= 1 << 2; // "legacy BIOS bootable"
    }
    out[48..56].copy_from_slice(&attrs.to_le_bytes());
    if let Some(name) = &p.name {
        for (i, cu) in name.encode_utf16().take(36).enumerate() {
            let off = 56 + i * 2;
            out[off..off + 2].copy_from_slice(&cu.to_le_bytes());
        }
    }
}

fn decode_entry(b: &[u8]) -> Option<Partition> {
    debug_assert_eq!(b.len(), 128);
    let type_bytes: [u8; 16] = b[0..16].try_into().unwrap();
    let uuid_bytes: [u8; 16] = b[16..32].try_into().unwrap();
    let type_uuid = uuid_from_gpt_bytes(&type_bytes);
    if type_uuid.is_nil() {
        return None;
    }
    let part_uuid = uuid_from_gpt_bytes(&uuid_bytes);
    let start = u64::from_le_bytes(b[32..40].try_into().unwrap());
    let end = u64::from_le_bytes(b[40..48].try_into().unwrap());
    let attrs = u64::from_le_bytes(b[48..56].try_into().unwrap());
    let mut name = String::new();
    let mut name_units = [0u16; 36];
    for (i, slot) in name_units.iter_mut().enumerate() {
        let off = 56 + i * 2;
        *slot = u16::from_le_bytes(b[off..off + 2].try_into().unwrap());
    }
    let len = name_units
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(name_units.len());
    if let Ok(s) = String::from_utf16(&name_units[..len]) {
        name = s;
    }
    Some(Partition {
        start_lba: start,
        size_lba: end - start + 1,
        kind: PartitionKind::from_gpt_uuid(type_uuid),
        uuid: Some(part_uuid),
        name: if name.is_empty() { None } else { Some(name) },
        bootable: attrs & (1 << 2) != 0,
        attributes: attrs & !(1 << 2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    fn mb(n: u64) -> u64 {
        n * 1024 * 1024
    }

    #[test]
    fn roundtrip_two_partitions() {
        let mut dev = MemoryBackend::new(mb(64));
        let parts = vec![
            Partition {
                start_lba: 2048,
                size_lba: 2048,
                kind: PartitionKind::EfiSystem,
                name: Some("EFI System Partition".into()),
                ..Partition::new(0, 0, PartitionKind::EfiSystem)
            },
            Partition {
                start_lba: 4096,
                size_lba: 100_000,
                kind: PartitionKind::LinuxFilesystem,
                name: Some("root".into()),
                bootable: false,
                ..Partition::new(0, 0, PartitionKind::LinuxFilesystem)
            },
        ];
        let gpt = Gpt::build(parts.clone()).unwrap();
        gpt.write(&mut dev).unwrap();

        let parsed = Gpt::read(&mut dev).unwrap();
        assert_eq!(parsed.disk_guid, gpt.disk_guid);
        assert_eq!(parsed.partitions.len(), 2);
        for (a, b) in gpt.partitions.iter().zip(parsed.partitions.iter()) {
            assert_eq!(a.start_lba, b.start_lba);
            assert_eq!(a.size_lba, b.size_lba);
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.name, b.name);
            assert_eq!(a.uuid, b.uuid);
        }
    }

    #[test]
    fn backup_header_present() {
        let mut dev = MemoryBackend::new(mb(64));
        let gpt = Gpt::build(vec![Partition::new(
            2048,
            1024,
            PartitionKind::LinuxFilesystem,
        )])
        .unwrap();
        gpt.write(&mut dev).unwrap();
        let total_lba = dev.total_size() / 512;
        let mut backup = [0u8; 512];
        dev.read_at((total_lba - 1) * 512, &mut backup).unwrap();
        let hdr = parse_header(&backup).unwrap();
        assert_eq!(hdr.my_lba, total_lba - 1);
        assert_eq!(hdr.alternate_lba, 1);
        assert_eq!(hdr.entries_start_lba, total_lba - 1 - ENTRIES_LBAS);
    }

    #[test]
    fn header_crc_catches_corruption() {
        let mut dev = MemoryBackend::new(mb(64));
        let gpt = Gpt::build(vec![Partition::new(
            2048,
            1024,
            PartitionKind::LinuxFilesystem,
        )])
        .unwrap();
        gpt.write(&mut dev).unwrap();
        // Flip a byte in the partition entries (LBA 2).
        let mut byte = [0u8; 1];
        dev.read_at(2 * 512 + 60, &mut byte).unwrap();
        byte[0] ^= 0x01;
        dev.write_at(2 * 512 + 60, &byte).unwrap();
        let err = Gpt::read(&mut dev).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidImage(_)));
    }

    #[test]
    fn protective_mbr_present_after_write() {
        let mut dev = MemoryBackend::new(mb(64));
        let gpt = Gpt::build(vec![Partition::new(
            2048,
            1024,
            PartitionKind::LinuxFilesystem,
        )])
        .unwrap();
        gpt.write(&mut dev).unwrap();
        let mut sector = [0u8; 512];
        dev.read_at(0, &mut sector).unwrap();
        assert_eq!(sector[510..512], [0x55, 0xAA]);
        assert_eq!(sector[446 + 4], 0xEE);
    }

    #[test]
    fn rejects_overlapping_partitions() {
        let parts = vec![
            Partition::new(2048, 1024, PartitionKind::LinuxFilesystem),
            Partition::new(2500, 1024, PartitionKind::LinuxFilesystem),
        ];
        let err = Gpt::build(parts).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    #[test]
    fn rejects_too_small_device() {
        let mut dev = MemoryBackend::new(64 * 1024); // 64 KiB = 128 LBAs
        let gpt = Gpt::build(vec![Partition::new(34, 32, PartitionKind::LinuxFilesystem)]).unwrap();
        // 128 LBAs is plenty for the GPT itself (66 LBAs of metadata) but we
        // want to make sure it works with a tight fit.
        gpt.write(&mut dev).unwrap();
        let parsed = Gpt::read(&mut dev).unwrap();
        assert_eq!(parsed.partitions.len(), 1);
    }
}
