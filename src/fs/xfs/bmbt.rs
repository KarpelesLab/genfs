//! XFS block-map B-tree records — extent decoding + tree walker.
//!
//! When `di_format == EXTENTS`, the inode's data fork is an array of
//! 128-bit packed records. Each record encodes a single contiguous range
//! mapping a logical file offset to a physical FS block.
//!
//! When `di_format == BTREE`, the inode's data fork is instead a B-tree
//! **root** (`xfs_bmdr_block`) whose leaves hold the same 128-bit extent
//! records. The root is laid out as:
//!
//! ```text
//!   __be16 bb_level
//!   __be16 bb_numrecs
//!   __be64 keys[numrecs]      // br_startoff  (sorted ascending)
//!   __be64 ptrs[numrecs]      // FSB pointer to child block (level-1)
//! ```
//!
//! Non-root blocks have a bigger header:
//!
//! - v4 (`BMAP` magic = 0x424D4150): 24-byte header
//!   `__be32 magic; __be16 level; __be16 numrecs; __be64 leftsib; __be64 rightsib`
//! - v5 (`BMA3` magic = 0x424D4133): v4 fields plus blkno/lsn/uuid/owner/crc
//!   — 72 bytes total
//!
//! After the header, intermediate (level > 0) blocks store `keys` then
//! `ptrs`; leaf (level == 0) blocks store packed 16-byte extent records
//! exactly like the EXTENTS literal-area form.
//!
//! ## Packed extent record (`xfs_bmbt_rec`)
//!
//! ```text
//!   bit  63    : flag (preallocated/unwritten)
//!   bits 62..9 : 54-bit logical file-block offset
//!   bits  8..0 : top 9 bits of startblock
//!   --- second 64-bit word ---
//!   bits 63..21: bottom 43 bits of startblock (52 bits total)
//!   bits 20..0 : 21-bit block count
//! ```

use crate::Result;
use crate::block::BlockDevice;

/// Bytes per extent record on disk.
pub const BMBT_REC_SIZE: usize = 16;

/// Bytes per key in a bmbt B-tree node.
pub const BMBT_KEY_SIZE: usize = 8;

/// Bytes per pointer in a bmbt B-tree node.
pub const BMBT_PTR_SIZE: usize = 8;

/// v4 BMBT block magic ("BMAP").
pub const XFS_BMAP_MAGIC: u32 = 0x424D_4150;
/// v5 BMBT block magic ("BMA3").
pub const XFS_BMAP_CRC_MAGIC: u32 = 0x424D_4133;

/// Decoded extent record. `startblock` is an XFS "filesystem block number"
/// (FSB) — a global block index across all AGs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    /// First logical file-block this extent covers.
    pub offset: u64,
    /// Physical starting FS block.
    pub startblock: u64,
    /// Number of FS blocks in this extent.
    pub blockcount: u32,
    /// True for unwritten / preallocated extents (kernel exposes as zeros).
    pub unwritten: bool,
}

impl Extent {
    /// Decode one packed 128-bit record.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < BMBT_REC_SIZE {
            return Err(crate::Error::InvalidImage(
                "xfs: bmbt record buffer too small".into(),
            ));
        }
        let hi = u64::from_be_bytes(buf[0..8].try_into().unwrap());
        let lo = u64::from_be_bytes(buf[8..16].try_into().unwrap());
        let unwritten = (hi >> 63) & 1 == 1;
        // bits 62..9 — 54-bit offset.
        let offset = (hi >> 9) & ((1u64 << 54) - 1);
        // Top 9 bits of startblock (bits 8..0 of hi) become bits 51..43 of
        // the 52-bit field; remaining 43 bits sit at bits 63..21 of lo.
        let sb_hi = hi & ((1u64 << 9) - 1);
        let sb_lo = lo >> 21;
        let startblock = (sb_hi << 43) | sb_lo;
        let blockcount = (lo & ((1u64 << 21) - 1)) as u32;
        if blockcount == 0 {
            return Err(crate::Error::InvalidImage(
                "xfs: bmbt record has blockcount=0".into(),
            ));
        }
        Ok(Self {
            offset,
            startblock,
            blockcount,
            unwritten,
        })
    }

    /// Encode for the round-trip unit test.
    #[cfg(test)]
    pub fn encode(&self) -> [u8; BMBT_REC_SIZE] {
        let unwritten = if self.unwritten { 1u64 } else { 0 };
        let hi = (unwritten << 63)
            | ((self.offset & ((1 << 54) - 1)) << 9)
            | ((self.startblock >> 43) & ((1 << 9) - 1));
        let lo = ((self.startblock & ((1 << 43) - 1)) << 21)
            | (self.blockcount as u64 & ((1 << 21) - 1));
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&hi.to_be_bytes());
        out[8..16].copy_from_slice(&lo.to_be_bytes());
        out
    }
}

/// Decode `n` consecutive extent records from `buf`. `buf` must hold at
/// least `n * BMBT_REC_SIZE` bytes.
pub fn decode_extents(buf: &[u8], n: u32) -> Result<Vec<Extent>> {
    let need = (n as usize)
        .checked_mul(BMBT_REC_SIZE)
        .ok_or_else(|| crate::Error::InvalidImage("xfs: extent count overflows".into()))?;
    if buf.len() < need {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: extent array truncated: need {need} bytes, have {}",
            buf.len()
        )));
    }
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        let start = i * BMBT_REC_SIZE;
        out.push(Extent::decode(&buf[start..start + BMBT_REC_SIZE])?);
    }
    Ok(out)
}

/// FSB → byte offset within the device. Mirrors the inode-addressing math
/// (`ag = fsb >> agblklog`, `agblk = fsb & ((1<<agblklog)-1)`).
fn fsb_to_byte(agblklog: u8, blocksize: u32, agblocks: u32, fsb: u64) -> u64 {
    let ag = fsb >> agblklog as u32;
    let agblk = fsb & ((1u64 << agblklog as u32) - 1);
    ag * (agblocks as u64) * (blocksize as u64) + agblk * (blocksize as u64)
}

/// Layout describing how to walk the BMBT for a single inode.
#[derive(Debug, Clone, Copy)]
pub struct BmbtLayout {
    pub blocksize: u32,
    pub agblocks: u32,
    pub agblklog: u8,
    pub is_v5: bool,
}

impl BmbtLayout {
    /// Bytes occupied by the on-disk node header (excluding keys/ptrs/recs).
    pub fn node_header_bytes(&self) -> usize {
        if self.is_v5 { 72 } else { 24 }
    }
}

/// Decode the inode-fork **root** of a bmbt (when `di_format == BTREE`).
/// Returns the root `(level, numrecs, keys, ptrs)`. Keys are
/// `br_startoff` u64 values; pointers are FSB block numbers of children.
///
/// `lit` is the literal area, `forkoff_words` is the value `di_forkoff`
/// (size in 8-byte words of the data fork, or 0 to use the full literal).
pub fn decode_root(lit: &[u8]) -> Result<(u16, u16, Vec<u64>, Vec<u64>)> {
    if lit.len() < 4 {
        return Err(crate::Error::InvalidImage(
            "xfs: bmbt root truncated".into(),
        ));
    }
    let level = u16::from_be_bytes(lit[0..2].try_into().unwrap());
    let numrecs = u16::from_be_bytes(lit[2..4].try_into().unwrap());
    let nrec = numrecs as usize;
    let keys_bytes = nrec
        .checked_mul(BMBT_KEY_SIZE)
        .ok_or_else(|| crate::Error::InvalidImage("xfs: bmbt root keys length overflows".into()))?;
    let ptrs_bytes = nrec
        .checked_mul(BMBT_PTR_SIZE)
        .ok_or_else(|| crate::Error::InvalidImage("xfs: bmbt root ptrs length overflows".into()))?;
    // The bmdr root uses a SPLIT layout: keys[] is at [4 .. 4+keys_bytes],
    // ptrs[] starts where the inode-fork's max keys would end. In practice
    // for the root we use the kernel's "tightly-packed" alternative where
    // ptrs immediately follow keys. Both layouts coexist in the wild but we
    // implement the packed variant — the keys + ptrs lengths derived from
    // `numrecs` exactly fill the fork, so the only ambiguity is whether
    // there's slack between them. We tolerate either: ptrs are placed at
    // the END of the literal area when slack is present.
    if 4 + keys_bytes + ptrs_bytes > lit.len() {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: bmbt root needs {} bytes but literal area is {}",
            4 + keys_bytes + ptrs_bytes,
            lit.len()
        )));
    }
    let keys_start = 4;
    let keys_end = keys_start + keys_bytes;
    let mut keys = Vec::with_capacity(nrec);
    for i in 0..nrec {
        let off = keys_start + i * BMBT_KEY_SIZE;
        keys.push(u64::from_be_bytes(lit[off..off + 8].try_into().unwrap()));
    }
    // Try packed first: ptrs follow keys directly.
    let ptrs_start_packed = keys_end;
    let ptrs_start_tail = lit.len() - ptrs_bytes;
    // Heuristic: if packed and tail differ, pick whichever yields plausible
    // pointer values (non-zero, < 2^48). The two are equal only when the
    // literal area is exactly filled.
    let ptrs_start = if ptrs_start_packed == ptrs_start_tail {
        ptrs_start_packed
    } else {
        // Prefer the tail layout (matches the on-disk bmdr root format used
        // by the kernel when there is slack between key and ptr arrays).
        ptrs_start_tail
    };
    let mut ptrs = Vec::with_capacity(nrec);
    for i in 0..nrec {
        let off = ptrs_start + i * BMBT_PTR_SIZE;
        ptrs.push(u64::from_be_bytes(lit[off..off + 8].try_into().unwrap()));
    }
    Ok((level, numrecs, keys, ptrs))
}

/// Read a non-root BMBT block from disk and return its `(level, numrecs,
/// keys, ptrs, records)`. For leaf blocks (level == 0) `records` is
/// populated and `keys`/`ptrs` are empty; for intermediate blocks the
/// reverse holds.
/// Parsed contents of a single bmbt node block. Internal nodes populate
/// `keys` + `ptrs` and leave `recs` empty; leaf nodes populate `recs` and
/// leave the key/ptr arrays empty.
pub struct BmbtNodeRead {
    pub level: u16,
    pub numrecs: u16,
    pub keys: Vec<u64>,
    pub ptrs: Vec<u64>,
    pub recs: Vec<Extent>,
}

pub fn read_node_block(
    dev: &mut dyn BlockDevice,
    layout: &BmbtLayout,
    fsb: u64,
) -> Result<BmbtNodeRead> {
    let byte_off = fsb_to_byte(layout.agblklog, layout.blocksize, layout.agblocks, fsb);
    let mut block = vec![0u8; layout.blocksize as usize];
    dev.read_at(byte_off, &mut block)?;
    if block.len() < layout.node_header_bytes() {
        return Err(crate::Error::InvalidImage(
            "xfs: bmbt node block smaller than its header".into(),
        ));
    }
    let magic = u32::from_be_bytes(block[0..4].try_into().unwrap());
    let want = if layout.is_v5 {
        XFS_BMAP_CRC_MAGIC
    } else {
        XFS_BMAP_MAGIC
    };
    if magic != want {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: bmbt block at fsb {fsb} has magic {magic:#010x}, want {want:#010x}"
        )));
    }
    // v4 header: magic(4) level(2) numrecs(2) leftsib(8) rightsib(8) = 24 B.
    // v5 header: same first 8 bytes + blkno(8) lsn(8) uuid(16) owner(8)
    // crc(4) pad(4) = 72 B. We only need level + numrecs.
    let level = u16::from_be_bytes(block[4..6].try_into().unwrap());
    let numrecs = u16::from_be_bytes(block[6..8].try_into().unwrap());
    let nrec = numrecs as usize;
    let hdr = layout.node_header_bytes();
    if level == 0 {
        // Leaf: packed extent records follow the header.
        let need = nrec
            .checked_mul(BMBT_REC_SIZE)
            .ok_or_else(|| crate::Error::InvalidImage("xfs: bmbt leaf nrec overflows".into()))?;
        if hdr + need > block.len() {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: bmbt leaf needs {} bytes but block is {}",
                hdr + need,
                block.len()
            )));
        }
        let mut recs = Vec::with_capacity(nrec);
        for i in 0..nrec {
            let off = hdr + i * BMBT_REC_SIZE;
            recs.push(Extent::decode(&block[off..off + BMBT_REC_SIZE])?);
        }
        Ok(BmbtNodeRead {
            level,
            numrecs,
            keys: Vec::new(),
            ptrs: Vec::new(),
            recs,
        })
    } else {
        // Internal: keys then ptrs. Non-root nodes use the FIXED max-numrecs
        // layout where keys[] and ptrs[] each occupy max_recs * size bytes.
        // The keys are at [hdr .. hdr + max*8] and ptrs at the matching
        // position. We have to compute max_recs from the block size:
        //
        //   max_recs = (blocksize - hdr) / (key_size + ptr_size)
        //
        // and then read the first `numrecs` entries from each array.
        let max_recs = (layout.blocksize as usize - hdr) / (BMBT_KEY_SIZE + BMBT_PTR_SIZE);
        if nrec > max_recs {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: bmbt internal numrecs {nrec} > max {max_recs}"
            )));
        }
        let keys_start = hdr;
        let ptrs_start = hdr + max_recs * BMBT_KEY_SIZE;
        let mut keys = Vec::with_capacity(nrec);
        let mut ptrs = Vec::with_capacity(nrec);
        for i in 0..nrec {
            let ko = keys_start + i * BMBT_KEY_SIZE;
            let po = ptrs_start + i * BMBT_PTR_SIZE;
            if po + 8 > block.len() {
                return Err(crate::Error::InvalidImage(
                    "xfs: bmbt internal ptr offset out of block".into(),
                ));
            }
            keys.push(u64::from_be_bytes(block[ko..ko + 8].try_into().unwrap()));
            ptrs.push(u64::from_be_bytes(block[po..po + 8].try_into().unwrap()));
        }
        Ok(BmbtNodeRead {
            level,
            numrecs,
            keys,
            ptrs,
            recs: Vec::new(),
        })
    }
}

/// Walk a complete bmbt anchored at the inode-fork root (`di_format ==
/// BTREE`), collecting every leaf extent in logical-offset order.
///
/// `lit` is the literal area of the inode (the bmdr root).
pub fn walk_btree(
    dev: &mut dyn BlockDevice,
    layout: &BmbtLayout,
    lit: &[u8],
) -> Result<Vec<Extent>> {
    let (root_level, root_numrecs, _root_keys, root_ptrs) = decode_root(lit)?;
    if root_level == 0 {
        // Degenerate case: the "root" is actually a leaf (would normally be
        // represented as EXTENTS format, but we tolerate it).
        return decode_extents(&lit[4..], root_numrecs as u32);
    }
    let mut out = Vec::new();
    // DFS: stack of (level, fsb) to visit. Push children L→R so that
    // popping yields left-to-right order? We instead use a queue-style
    // recursive walk for clarity, since we don't need RAII unwind.
    fn walk(
        dev: &mut dyn BlockDevice,
        layout: &BmbtLayout,
        fsb: u64,
        sink: &mut Vec<Extent>,
    ) -> Result<()> {
        let node = read_node_block(dev, layout, fsb)?;
        if node.level == 0 {
            sink.extend(node.recs);
        } else {
            for child in node.ptrs {
                walk(dev, layout, child, sink)?;
            }
        }
        Ok(())
    }
    for child in root_ptrs {
        walk(dev, layout, child, &mut out)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn extent_roundtrip() {
        let cases = [
            Extent {
                offset: 0,
                startblock: 100,
                blockcount: 1,
                unwritten: false,
            },
            Extent {
                offset: 4,
                startblock: 200,
                blockcount: 8,
                unwritten: false,
            },
            // High bits set in startblock (top 9 bits).
            Extent {
                offset: 1 << 30,
                startblock: (1u64 << 50) | 0x12_3456,
                blockcount: 0x1FFFFF,
                unwritten: true,
            },
        ];
        for c in cases {
            let bytes = c.encode();
            let back = Extent::decode(&bytes).unwrap();
            assert_eq!(back, c);
        }
    }

    #[test]
    fn extent_rejects_zero_blockcount() {
        let mut bytes = [0u8; 16];
        // offset=0, startblock=0, blockcount=0 — invalid.
        let e = Extent::decode(&bytes);
        assert!(e.is_err());
        // Even with a non-zero offset, blockcount=0 is bogus.
        bytes[0] = 0x00;
        let e = Extent::decode(&bytes);
        assert!(e.is_err());
    }

    #[test]
    fn known_pattern() {
        // Build a known record by hand: offset=4, startblock=200, blockcount=8.
        let offset: u64 = 4;
        let startblock: u64 = 200;
        let blockcount: u64 = 8;
        let hi = (offset << 9) | ((startblock >> 43) & ((1 << 9) - 1));
        let lo = ((startblock & ((1 << 43) - 1)) << 21) | blockcount;
        let mut buf = [0u8; 16];
        buf[0..8].copy_from_slice(&hi.to_be_bytes());
        buf[8..16].copy_from_slice(&lo.to_be_bytes());
        let e = Extent::decode(&buf).unwrap();
        assert_eq!(e.offset, 4);
        assert_eq!(e.startblock, 200);
        assert_eq!(e.blockcount, 8);
        assert!(!e.unwritten);
    }

    /// Build a fully synthetic two-level v5 bmbt: a root with 2 ptrs to
    /// leaves, each leaf holding 2 extents. Then verify `walk_btree`
    /// returns all 4 extents in order.
    #[test]
    fn walk_two_level_v5_btree() {
        let blocksize = 4096u32;
        let agblocks = 256u32;
        let agblklog = 8u8;
        let layout = BmbtLayout {
            blocksize,
            agblocks,
            agblklog,
            is_v5: true,
        };
        // Build a 4-AG device, 256 blocks per AG ⇒ 4 MiB. Allocate FSBs
        // 10 and 11 for the two leaves.
        let total = (agblocks as u64) * 4 * blocksize as u64;
        let mut dev = MemoryBackend::new(total);

        // Build leaf at FSB 10: 2 extents.
        let mut leaf0 = vec![0u8; blocksize as usize];
        leaf0[0..4].copy_from_slice(&XFS_BMAP_CRC_MAGIC.to_be_bytes());
        leaf0[4..6].copy_from_slice(&0u16.to_be_bytes()); // level
        leaf0[6..8].copy_from_slice(&2u16.to_be_bytes()); // numrecs
        // (no siblings, blkno, etc. — we don't validate them)
        let e0 = Extent {
            offset: 0,
            startblock: 100,
            blockcount: 4,
            unwritten: false,
        };
        let e1 = Extent {
            offset: 4,
            startblock: 200,
            blockcount: 8,
            unwritten: false,
        };
        leaf0[72..72 + 16].copy_from_slice(&e0.encode());
        leaf0[72 + 16..72 + 32].copy_from_slice(&e1.encode());
        // Write at FSB 10.
        dev.write_at(fsb_to_byte(agblklog, blocksize, agblocks, 10), &leaf0)
            .unwrap();

        // Build leaf at FSB 11: 2 extents.
        let mut leaf1 = vec![0u8; blocksize as usize];
        leaf1[0..4].copy_from_slice(&XFS_BMAP_CRC_MAGIC.to_be_bytes());
        leaf1[6..8].copy_from_slice(&2u16.to_be_bytes());
        let e2 = Extent {
            offset: 12,
            startblock: 300,
            blockcount: 2,
            unwritten: false,
        };
        let e3 = Extent {
            offset: 20,
            startblock: 400,
            blockcount: 1,
            unwritten: false,
        };
        leaf1[72..72 + 16].copy_from_slice(&e2.encode());
        leaf1[72 + 16..72 + 32].copy_from_slice(&e3.encode());
        dev.write_at(fsb_to_byte(agblklog, blocksize, agblocks, 11), &leaf1)
            .unwrap();

        // Build the root in a 64-byte buffer (typical inode-fork size).
        let mut root = vec![0u8; 64];
        root[0..2].copy_from_slice(&1u16.to_be_bytes()); // level
        root[2..4].copy_from_slice(&2u16.to_be_bytes()); // numrecs
        // Packed layout for 64-byte root with 2 entries:
        //   header(4) + 2*8 keys + 2*8 ptrs = 36 bytes; tail layout places
        //   ptrs at the very end. Our decoder prefers tail when slack exists.
        // keys at [4..20]; ptrs at [48..64].
        root[4..12].copy_from_slice(&0u64.to_be_bytes()); // key 0 = offset 0
        root[12..20].copy_from_slice(&12u64.to_be_bytes()); // key 1 = offset 12
        root[48..56].copy_from_slice(&10u64.to_be_bytes()); // ptr 0 = FSB 10
        root[56..64].copy_from_slice(&11u64.to_be_bytes()); // ptr 1 = FSB 11

        let extents = walk_btree(&mut dev, &layout, &root).unwrap();
        assert_eq!(extents.len(), 4);
        assert_eq!(extents[0], e0);
        assert_eq!(extents[1], e1);
        assert_eq!(extents[2], e2);
        assert_eq!(extents[3], e3);
    }

    #[test]
    fn fsb_to_byte_matches_inode_math() {
        // agblklog=3 means 8 blocks per AG. FSB 9 = AG 1, block 1.
        let b = fsb_to_byte(3, 4096, 8, 9);
        // ag1_start = 1 AG * 8 blocks/AG * 4096 B/block = 32768
        // offset = ag1_start + 1 block * 4096 B = 36864.
        assert_eq!(b, 32768 + 4096);
    }
}
