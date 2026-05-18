//! Master Boot Record partition table.
//!
//! v1 limitations:
//! - 4 primary partitions only — no extended/logical chain.
//! - LBA addressing only; CHS fields are written with the conventional
//!   "exceeds CHS" placeholder `FE FF FF` so legacy tools recognise the
//!   entry as LBA-only.
//! - `start_lba` and `size_lba` must each fit in 32 bits.

use std::io::SeekFrom;

use super::{Partition, PartitionKind, PartitionTable};
use crate::Result;
use crate::block::BlockDevice;

/// MBR sector size — always 512 bytes regardless of the device's logical
/// sector size (the on-disk layout is fixed).
pub const MBR_SECTOR: usize = 512;

/// Byte offset of the first partition entry within the boot sector.
pub const PART_ENTRY_OFFSET: usize = 446;

/// Size of each partition entry.
pub const PART_ENTRY_SIZE: usize = 16;

/// Boot signature at offset 510..512.
pub const SIGNATURE: [u8; 2] = [0x55, 0xAA];

/// CHS placeholder written when LBA addressing is used. The first byte (head)
/// is 0xFE which tools interpret as "see the LBA fields instead". Conventional
/// value, matches what `fdisk` writes.
const CHS_LBA: [u8; 3] = [0xFE, 0xFF, 0xFF];

/// In-memory MBR. Exactly 4 entries; trailing unused entries have
/// `kind == PartitionKind::Empty`.
#[derive(Debug, Clone)]
pub struct Mbr {
    partitions: Vec<Partition>,
}

impl Mbr {
    /// Build a new MBR from a list of partitions. Up to 4 entries; pads with
    /// empty entries internally.
    pub fn new(partitions: Vec<Partition>) -> Result<Self> {
        if partitions.len() > 4 {
            return Err(crate::Error::InvalidArgument(format!(
                "MBR supports up to 4 primary partitions, got {}",
                partitions.len()
            )));
        }
        for p in &partitions {
            check_fits(p)?;
        }
        Ok(Self { partitions })
    }

    /// Read an MBR from a block device. Returns
    /// [`Error::InvalidImage`](crate::Error::InvalidImage) if the
    /// signature is missing.
    pub fn read(dev: &mut dyn BlockDevice) -> Result<Self> {
        if dev.total_size() < MBR_SECTOR as u64 {
            return Err(crate::Error::InvalidImage(
                "device too small to hold an MBR".into(),
            ));
        }
        let mut sector = [0u8; MBR_SECTOR];
        dev.seek(SeekFrom::Start(0))?;
        dev.read_exact(&mut sector)?;
        if sector[510..512] != SIGNATURE {
            return Err(crate::Error::InvalidImage(
                "MBR signature 0x55AA not found".into(),
            ));
        }
        let mut parts = Vec::new();
        for i in 0..4 {
            let off = PART_ENTRY_OFFSET + i * PART_ENTRY_SIZE;
            let entry = &sector[off..off + PART_ENTRY_SIZE];
            let p = decode_entry(entry);
            // Strip trailing empties so partitions() returns only what was
            // actually used; preserve gaps in the middle by stopping at the
            // last non-empty.
            parts.push(p);
        }
        while parts.last().is_some_and(|p| p.kind == PartitionKind::Empty && p.size_lba == 0) {
            parts.pop();
        }
        Ok(Self { partitions: parts })
    }
}

impl PartitionTable for Mbr {
    fn write(&self, dev: &mut dyn BlockDevice) -> Result<()> {
        if dev.total_size() < MBR_SECTOR as u64 {
            return Err(crate::Error::InvalidArgument(
                "device too small to hold an MBR".into(),
            ));
        }
        let mut sector = [0u8; MBR_SECTOR];
        for (i, p) in self.partitions.iter().enumerate() {
            check_fits(p)?;
            let off = PART_ENTRY_OFFSET + i * PART_ENTRY_SIZE;
            encode_entry(p, &mut sector[off..off + PART_ENTRY_SIZE]);
        }
        sector[510] = SIGNATURE[0];
        sector[511] = SIGNATURE[1];
        dev.write_at(0, &sector)?;
        Ok(())
    }

    fn partitions(&self) -> &[Partition] {
        &self.partitions
    }
}

fn check_fits(p: &Partition) -> Result<()> {
    if p.kind == PartitionKind::Empty {
        return Ok(());
    }
    if p.start_lba > u32::MAX as u64 {
        return Err(crate::Error::Unsupported(format!(
            "MBR cannot address start_lba {} (> 2^32 sectors)",
            p.start_lba
        )));
    }
    if p.size_lba > u32::MAX as u64 {
        return Err(crate::Error::Unsupported(format!(
            "MBR cannot address size_lba {} (> 2^32 sectors)",
            p.size_lba
        )));
    }
    Ok(())
}

fn encode_entry(p: &Partition, out: &mut [u8]) {
    debug_assert_eq!(out.len(), PART_ENTRY_SIZE);
    out[0] = if p.bootable { 0x80 } else { 0x00 };
    out[1..4].copy_from_slice(&CHS_LBA);
    out[4] = p.kind.as_mbr_byte();
    out[5..8].copy_from_slice(&CHS_LBA);
    out[8..12].copy_from_slice(&(p.start_lba as u32).to_le_bytes());
    out[12..16].copy_from_slice(&(p.size_lba as u32).to_le_bytes());
    if p.kind == PartitionKind::Empty {
        out[1..4].copy_from_slice(&[0, 0, 0]);
        out[5..8].copy_from_slice(&[0, 0, 0]);
        out[8..12].copy_from_slice(&[0, 0, 0, 0]);
        out[12..16].copy_from_slice(&[0, 0, 0, 0]);
    }
}

fn decode_entry(b: &[u8]) -> Partition {
    debug_assert_eq!(b.len(), PART_ENTRY_SIZE);
    let bootable = b[0] == 0x80;
    let kind = PartitionKind::from_mbr_byte(b[4]);
    let start_lba = u32::from_le_bytes(b[8..12].try_into().unwrap()) as u64;
    let size_lba = u32::from_le_bytes(b[12..16].try_into().unwrap()) as u64;
    Partition {
        start_lba,
        size_lba,
        kind,
        uuid: None,
        name: None,
        bootable,
        attributes: 0,
    }
}

/// Write a protective MBR — a single partition of type 0xEE covering the
/// whole disk minus the MBR sector itself. Required ahead of a GPT.
///
/// Exposed for use by [`super::gpt::Gpt`]; not normally called directly.
pub(crate) fn write_protective_mbr(dev: &mut dyn BlockDevice) -> Result<()> {
    let total_lba = dev.total_size() / u64::from(dev.block_size());
    if total_lba < 2 {
        return Err(crate::Error::InvalidArgument(
            "device too small for protective MBR + GPT".into(),
        ));
    }
    let size_lba_field = (total_lba - 1).min(u64::from(u32::MAX)) as u32;
    let mut sector = [0u8; MBR_SECTOR];
    let off = PART_ENTRY_OFFSET;
    sector[off] = 0x00;
    sector[off + 1..off + 4].copy_from_slice(&[0x00, 0x02, 0x00]);
    sector[off + 4] = 0xEE;
    sector[off + 5..off + 8].copy_from_slice(&[0xFF, 0xFF, 0xFF]);
    sector[off + 8..off + 12].copy_from_slice(&1u32.to_le_bytes());
    sector[off + 12..off + 16].copy_from_slice(&size_lba_field.to_le_bytes());
    sector[510] = SIGNATURE[0];
    sector[511] = SIGNATURE[1];
    dev.write_at(0, &sector)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    fn mb(n: u64) -> u64 {
        n * 1024 * 1024
    }

    #[test]
    fn roundtrip_three_partitions() {
        let mut dev = MemoryBackend::new(mb(64));
        let parts = vec![
            Partition {
                start_lba: 2048,
                size_lba: 20480,
                kind: PartitionKind::LinuxFilesystem,
                bootable: true,
                ..Partition::new(0, 0, PartitionKind::LinuxFilesystem)
            },
            Partition {
                start_lba: 22528,
                size_lba: 4096,
                kind: PartitionKind::LinuxSwap,
                ..Partition::new(0, 0, PartitionKind::LinuxSwap)
            },
            Partition {
                start_lba: 26624,
                size_lba: 8192,
                kind: PartitionKind::Fat32,
                ..Partition::new(0, 0, PartitionKind::Fat32)
            },
        ];
        let mbr = Mbr::new(parts.clone()).unwrap();
        mbr.write(&mut dev).unwrap();

        let parsed = Mbr::read(&mut dev).unwrap();
        assert_eq!(parsed.partitions().len(), 3);
        for (a, b) in parts.iter().zip(parsed.partitions().iter()) {
            assert_eq!(a.start_lba, b.start_lba);
            assert_eq!(a.size_lba, b.size_lba);
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.bootable, b.bootable);
        }
    }

    #[test]
    fn signature_present_after_write() {
        let mut dev = MemoryBackend::new(mb(8));
        let mbr = Mbr::new(vec![Partition::new(2048, 1024, PartitionKind::LinuxFilesystem)]).unwrap();
        mbr.write(&mut dev).unwrap();
        let mut sig = [0u8; 2];
        dev.read_at(510, &mut sig).unwrap();
        assert_eq!(sig, SIGNATURE);
    }

    #[test]
    fn rejects_too_many_partitions() {
        let mut parts = Vec::new();
        for i in 0..5 {
            parts.push(Partition::new(i * 1024 + 2048, 512, PartitionKind::LinuxFilesystem));
        }
        let err = Mbr::new(parts).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    #[test]
    fn rejects_partition_past_2tb() {
        let p = Partition::new((u32::MAX as u64) + 1, 1024, PartitionKind::LinuxFilesystem);
        let err = Mbr::new(vec![p]).unwrap_err();
        assert!(matches!(err, crate::Error::Unsupported(_)));
    }

    #[test]
    fn missing_signature_rejected_on_read() {
        let mut dev = MemoryBackend::new(MBR_SECTOR as u64);
        // No write — sector is all zero, no signature.
        let err = Mbr::read(&mut dev).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidImage(_)));
    }

    #[test]
    fn protective_mbr_has_ee_type_and_size_clamped() {
        let mut dev = MemoryBackend::new(mb(8));
        write_protective_mbr(&mut dev).unwrap();
        let mut sector = [0u8; MBR_SECTOR];
        dev.read_at(0, &mut sector).unwrap();
        assert_eq!(sector[510..512], SIGNATURE);
        assert_eq!(sector[PART_ENTRY_OFFSET + 4], 0xEE);
        let start = u32::from_le_bytes(
            sector[PART_ENTRY_OFFSET + 8..PART_ENTRY_OFFSET + 12]
                .try_into()
                .unwrap(),
        );
        assert_eq!(start, 1);
        // 8 MiB / 512 = 16384 sectors → size field = 16383
        let size = u32::from_le_bytes(
            sector[PART_ENTRY_OFFSET + 12..PART_ENTRY_OFFSET + 16]
                .try_into()
                .unwrap(),
        );
        assert_eq!(size, 16383);
    }
}
