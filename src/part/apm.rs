//! Apple Partition Map (APM).
//!
//! The classic Macintosh / PowerPC partitioning scheme, also emitted by Toast
//! and Disk Copy `.toast`/`.img` images. Layout:
//!
//! * **Block 0** — the *Driver Descriptor Map* (DDM), signature `ER` (0x4552).
//!   It records the device block size (`sbBlkSize`) and count, plus an optional
//!   driver list. APM itself addresses partitions in **512-byte blocks**
//!   regardless of `sbBlkSize` (a `.toast` advertises 2048 in the DDM but the
//!   partition map entries are still 512-byte-block relative).
//! * **Block 1..** — the partition map: one `PartitionMapEntry`-shaped 512-byte
//!   record per partition, each signed `PM` (0x504D). The first entry's
//!   `pmMapBlkCnt` gives the number of map entries (one of which is usually the
//!   self-describing `Apple_partition_map` entry).
//!
//! This reader is **read-only**: [`PartitionTable::write`] returns
//! [`crate::Error::Unsupported`]. fstool surfaces APM the same way it surfaces
//! GPT/MBR — `info` lists the partitions and `path:N` slices partition *N*.

use crate::Result;
use crate::block::BlockDevice;

use super::{Partition, PartitionKind, PartitionTable};

/// APM partition map entries are always relative to 512-byte blocks, even when
/// the DDM advertises a larger device block size.
const APM_BLOCK: u64 = 512;

/// Driver Descriptor Map signature, at byte 0 of block 0.
const DDM_SIG: &[u8; 2] = b"ER";
/// Partition map entry signature, at byte 0 of each map block.
const PM_SIG: &[u8; 2] = b"PM";

/// Upper bound on partition map entries we will parse, to cap work on a
/// malformed `pmMapBlkCnt`.
const MAX_ENTRIES: u32 = 256;

/// A parsed Apple Partition Map.
#[derive(Debug, Clone)]
pub struct Apm {
    partitions: Vec<Partition>,
}

impl Apm {
    /// True if `dev` carries an Apple Partition Map: a `ER` Driver Descriptor
    /// Map at block 0 *and* a `PM` partition map entry at block 1. Requiring
    /// both avoids misreading a stray `ER`/`PM` in a bare filesystem.
    pub fn probe(dev: &mut dyn BlockDevice) -> bool {
        if dev.total_size() < 1024 {
            return false;
        }
        let mut head = [0u8; 2];
        if dev.read_at(0, &mut head).is_err() || &head != DDM_SIG {
            return false;
        }
        if dev.read_at(APM_BLOCK, &mut head).is_err() || &head != PM_SIG {
            return false;
        }
        true
    }

    /// Parse the partition map from `dev`.
    ///
    /// Returns [`crate::Error::InvalidImage`] if the partition map entry at
    /// block 1 is missing its `PM` signature.
    pub fn read(dev: &mut dyn BlockDevice) -> Result<Self> {
        let total = dev.total_size();
        let mut first = [0u8; 512];
        dev.read_at(APM_BLOCK, &mut first)?;
        if &first[0..2] != PM_SIG {
            return Err(crate::Error::InvalidImage(
                "apm: no PM signature at block 1".into(),
            ));
        }
        // pmMapBlkCnt (u32-BE @4) — number of entries in the map. Cap it.
        let map_blk_cnt = be32(&first, 4).min(MAX_ENTRIES);

        let mut partitions = Vec::new();
        for i in 0..map_blk_cnt {
            let off = (u64::from(i) + 1) * APM_BLOCK;
            if off + 512 > total {
                break;
            }
            let mut e = [0u8; 512];
            dev.read_at(off, &mut e)?;
            if &e[0..2] != PM_SIG {
                break;
            }
            // pmPyPartStart (u32-BE @8): physical start block of the partition.
            // pmPartBlkCnt  (u32-BE @12): partition length in blocks.
            let start = u64::from(be32(&e, 8));
            let count = u64::from(be32(&e, 12));
            // pmPartName (@16, 32B) and pmPartType (@48, 32B) are NUL-padded
            // ASCII (e.g. "Apple_HFS", "Apple_partition_map", "Apple_Free").
            let name = c_str(&e[16..48]);
            let ptype = c_str(&e[48..80]);

            let mut part = Partition::new(start, count, PartitionKind::Apm(ptype));
            if !name.is_empty() {
                part.name = Some(name);
            }
            partitions.push(part);
        }

        Ok(Apm { partitions })
    }
}

impl PartitionTable for Apm {
    fn write(&self, _dev: &mut dyn BlockDevice) -> Result<()> {
        Err(crate::Error::Unsupported(
            "apm: writing an Apple Partition Map is not supported".into(),
        ))
    }

    fn partitions(&self) -> &[Partition] {
        &self.partitions
    }
}

/// Read a big-endian u32 at `off` within `b`. Caller guarantees `off + 4 <= len`.
fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Decode a NUL-padded ASCII field, trimming at the first NUL. Non-ASCII bytes
/// are kept via lossy UTF-8 (APM type/name fields are ASCII in practice).
fn c_str(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    /// Build a minimal APM disk: DDM at block 0, then `entries` partition map
    /// blocks. Each entry is (pyPartStart, partBlkCnt, name, type).
    fn build_apm(entries: &[(u32, u32, &str, &str)]) -> MemoryBackend {
        let n = entries.len() as u32;
        // Enough blocks to cover the map plus the partitions' declared extents.
        let max_end = entries
            .iter()
            .map(|(s, c, _, _)| u64::from(*s) + u64::from(*c))
            .max()
            .unwrap_or(0);
        let total = (max_end.max(u64::from(n) + 1)) * APM_BLOCK;
        let mut dev = MemoryBackend::new(total);

        // DDM (block 0): "ER", sbBlkSize=512, sbBlkCount.
        let mut ddm = [0u8; 512];
        ddm[0..2].copy_from_slice(DDM_SIG);
        ddm[2..4].copy_from_slice(&512u16.to_be_bytes());
        ddm[4..8].copy_from_slice(&((total / APM_BLOCK) as u32).to_be_bytes());
        dev.write_at(0, &ddm).unwrap();

        for (i, (start, count, name, ptype)) in entries.iter().enumerate() {
            let mut e = [0u8; 512];
            e[0..2].copy_from_slice(PM_SIG);
            e[4..8].copy_from_slice(&n.to_be_bytes()); // pmMapBlkCnt
            e[8..12].copy_from_slice(&start.to_be_bytes());
            e[12..16].copy_from_slice(&count.to_be_bytes());
            let nb = name.as_bytes();
            e[16..16 + nb.len()].copy_from_slice(nb);
            let tb = ptype.as_bytes();
            e[48..48 + tb.len()].copy_from_slice(tb);
            dev.write_at((i as u64 + 1) * APM_BLOCK, &e).unwrap();
        }
        dev
    }

    #[test]
    fn probe_and_parse_three_partitions() {
        let mut dev = build_apm(&[
            (1, 63, "Apple", "Apple_partition_map"),
            (64, 800, "MacOS", "Apple_HFS"),
            (864, 100, "Extra", "Apple_Free"),
        ]);
        assert!(Apm::probe(&mut dev));

        let apm = Apm::read(&mut dev).unwrap();
        let parts = apm.partitions();
        assert_eq!(parts.len(), 3);

        assert_eq!(parts[1].start_lba, 64);
        assert_eq!(parts[1].size_lba, 800);
        assert_eq!(parts[1].name.as_deref(), Some("MacOS"));
        assert_eq!(parts[1].kind, PartitionKind::Apm("Apple_HFS".to_string()));
    }

    #[test]
    fn probe_rejects_bare_volume() {
        // No DDM/PM signatures.
        let mut dev = MemoryBackend::new(2048);
        assert!(!Apm::probe(&mut dev));
        assert!(Apm::read(&mut dev).is_err());
    }

    #[test]
    fn slice_targets_the_hfs_partition() {
        let mut dev = build_apm(&[
            (1, 63, "Apple", "Apple_partition_map"),
            (64, 8, "MacOS", "Apple_HFS"),
        ]);
        let apm = Apm::read(&mut dev).unwrap();
        // Partition index 1 → bytes [64*512 .. 72*512).
        let sliced = super::super::slice_partition(&apm, &mut dev, 1).unwrap();
        assert_eq!(sliced.total_size(), 8 * 512);
    }

    #[test]
    fn write_is_unsupported() {
        let mut dev = build_apm(&[(1, 63, "Apple", "Apple_partition_map")]);
        let apm = Apm::read(&mut dev).unwrap();
        let mut out = MemoryBackend::new(4096);
        assert!(apm.write(&mut out).is_err());
    }
}
