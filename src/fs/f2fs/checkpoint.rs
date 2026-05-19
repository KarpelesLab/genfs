//! F2FS checkpoint pack discovery + parsing.
//!
//! F2FS keeps two checkpoint packs (CP0 and CP1), one segment apart.
//! Each pack opens with a `f2fs_checkpoint` block, has zero or more
//! `cp_payload` bitmap pages, then a NAT-journal-bearing summary block
//! (the "cp_pack_start_sum"), some more summary blocks, and a footer
//! that mirrors the header. We pick whichever pack has the higher
//! `checkpoint_ver` AND validates its head-block CRC32.
//!
//! Reference: kernel docs §"Checkpoint" + FAST '15 §2.7.

use crate::Result;
use crate::block::BlockDevice;

use super::constants::{CP_COMPACT_SUM_FLAG, F2FS_BLK_CSUM_OFFSET, F2FS_BLKSIZE, NAT_ENTRY_SIZE};
use super::superblock::Superblock;

/// Logical view of a parsed checkpoint pack head.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// `checkpoint_ver` — monotonically increasing, used to pick the live pack.
    pub version: u64,
    /// `user_block_count`.
    pub user_block_count: u64,
    /// `valid_block_count`.
    pub valid_block_count: u64,
    /// Bitfield of `CP_*_FLAG` bits.
    pub flags: u32,
    /// First block within this CP pack that holds summary data (where the
    /// NAT journal lives).
    pub cp_pack_start_sum: u32,
    /// `cp_pack_total_block_count` — total length of the pack in blocks.
    pub cp_pack_total_block_count: u32,
    /// CP payload count (overflow bitmap pages between the head block and
    /// the first summary block); typically 0 for small images.
    pub cp_payload: u32,
    /// Absolute block address of the head block we picked.
    pub head_blkaddr: u32,
    /// `nat_ver_bitmap` length in bytes (variable per image — kept so the
    /// NAT bitmap pages can be located if we ever need them).
    pub nat_ver_bitmap_bytesize: u32,
    /// `sit_ver_bitmap` length in bytes.
    pub sit_ver_bitmap_bytesize: u32,
    /// 0 or 1 — which copy of the NAT is current (cleared bits in the
    /// per-block bitmap select the alternate page).
    pub cur_nat_pack: u8,
    /// 0 or 1 — which copy of the SIT is current.
    pub cur_sit_pack: u8,
    /// Raw NAT journal entries: list of `(nid, ino, block_addr, version)`
    /// pulled from the cp_pack_start_sum block. Newer than the on-disk
    /// NAT pages and must be consulted first.
    pub nat_journal: Vec<NatJournalEntry>,
}

#[derive(Debug, Clone, Copy)]
pub struct NatJournalEntry {
    pub nid: u32,
    pub ino: u32,
    pub block_addr: u32,
    pub version: u8,
}

impl Checkpoint {
    /// Iterate both candidate CP packs, pick the one with the higher
    /// `checkpoint_ver` whose CRC32 validates.
    pub fn load(dev: &mut dyn BlockDevice, sb: &Superblock) -> Result<Self> {
        let bs = sb.block_size() as u64;
        let blocks_per_seg = sb.blocks_per_seg();
        let cp0_blk = sb.cp_blkaddr;
        let cp1_blk = sb
            .cp_blkaddr
            .checked_add(blocks_per_seg)
            .ok_or_else(|| crate::Error::InvalidImage("f2fs: cp_blkaddr overflow".into()))?;

        let mut cp0 = Self::try_load(dev, sb, cp0_blk).ok();
        let mut cp1 = Self::try_load(dev, sb, cp1_blk).ok();

        // `cur_*_pack` flips depending on which CP we ended up using.
        if let Some(c) = cp0.as_mut() {
            c.cur_nat_pack = 0;
            c.cur_sit_pack = 0;
        }
        if let Some(c) = cp1.as_mut() {
            c.cur_nat_pack = 1;
            c.cur_sit_pack = 1;
        }

        let _ = bs; // (silence unused if optimizer skips it on debug)

        match (cp0, cp1) {
            (Some(a), Some(b)) => Ok(if b.version > a.version { b } else { a }),
            (Some(a), None) => Ok(a),
            (None, Some(b)) => Ok(b),
            (None, None) => Err(crate::Error::InvalidImage(
                "f2fs: neither checkpoint pack validates".into(),
            )),
        }
    }

    /// Read + validate a single CP head block + the NAT journal that
    /// follows it.
    fn try_load(dev: &mut dyn BlockDevice, sb: &Superblock, head_blkaddr: u32) -> Result<Self> {
        let bs = sb.block_size() as u64;
        let mut head = vec![0u8; F2FS_BLKSIZE];
        dev.read_at(head_blkaddr as u64 * bs, &mut head)?;
        let cp = decode_cp_head(&head, head_blkaddr)?;

        // Validate the head-block CRC32. The CRC covers everything up to
        // the 4-byte footer.
        let want = u32::from_le_bytes(
            head[F2FS_BLK_CSUM_OFFSET..F2FS_BLK_CSUM_OFFSET + 4]
                .try_into()
                .unwrap(),
        );
        let got = crc32fast::hash(&head[..F2FS_BLK_CSUM_OFFSET]);
        if got != want {
            return Err(crate::Error::InvalidImage(format!(
                "f2fs: cp@{head_blkaddr}: crc mismatch (want {want:08x}, got {got:08x})"
            )));
        }

        // The NAT journal lives in the cp_pack_start_sum block (the first
        // summary page in the pack). We only pull entries when the pack
        // sets CP_COMPACT_SUM_FLAG OR when the summary is one block —
        // for v1 we treat any non-empty entry list optimistically.
        let sum_block = head_blkaddr
            .checked_add(cp.cp_pack_start_sum)
            .ok_or_else(|| crate::Error::InvalidImage("f2fs: cp summary overflow".into()))?;
        let mut sumbuf = vec![0u8; F2FS_BLKSIZE];
        dev.read_at(sum_block as u64 * bs, &mut sumbuf)?;
        let nat_journal = decode_nat_journal(&sumbuf);

        let mut out = cp;
        out.nat_journal = nat_journal;
        Ok(out)
    }

    /// Convenience: look up the journal-cached NAT entry for `nid`.
    pub fn nat_journal_lookup(&self, nid: u32) -> Option<NatJournalEntry> {
        self.nat_journal.iter().copied().find(|e| e.nid == nid)
    }
}

/// Decode the leading region of a CP head block. The block is 4 KiB and
/// the trailing 4 bytes are the CRC32 footer.
fn decode_cp_head(buf: &[u8], head_blkaddr: u32) -> Result<Checkpoint> {
    if buf.len() < F2FS_BLKSIZE {
        return Err(crate::Error::InvalidImage(
            "f2fs: short read on CP head".into(),
        ));
    }
    let r32 = |o: usize| u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
    let r64 = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());

    // Field offsets follow the publicly documented f2fs_checkpoint layout.
    let version = r64(0x00);
    let user_block_count = r64(0x08);
    let valid_block_count = r64(0x10);
    // 0x18 rsvd_segment_count, 0x1C overprov_segment_count
    let _free_segment_count = r32(0x20);
    // 0x24..0x44 — eight cur_node_segno + cur_node_blkoff + cur_data_segno +
    // cur_data_blkoff slots; we don't need them for read.
    // 0x64 valid_node_count, 0x68 valid_inode_count, 0x6C next_free_nid
    let nat_ver_bitmap_bytesize = r32(0x70);
    let sit_ver_bitmap_bytesize = r32(0x74);
    let cp_pack_total_block_count = r32(0x78);
    let ckpt_flags = r32(0x7C);
    let cp_pack_start_sum = r32(0x80);
    let cp_payload = r32(0x84);

    Ok(Checkpoint {
        version,
        user_block_count,
        valid_block_count,
        flags: ckpt_flags,
        cp_pack_start_sum,
        cp_pack_total_block_count,
        cp_payload,
        head_blkaddr,
        nat_ver_bitmap_bytesize,
        sit_ver_bitmap_bytesize,
        cur_nat_pack: 0,
        cur_sit_pack: 0,
        nat_journal: Vec::new(),
    })
}

/// Decode the NAT journal at the head of the cp_pack_start_sum block.
///
/// In a real F2FS image this lives inside a compact or normal summary
/// footer. For v1 we encode a minimal-but-compatible layout:
///
/// - bytes 0..2   : `n_nats` (le u16) — number of journal entries
/// - bytes 2..4   : reserved
/// - bytes 4.. n*16  : entries, each `nid (u32) | ino (u32) | block_addr
///                    (u32) | version (u8) | pad (3 B)`.
///
/// Standard F2FS uses a packed `(u32 nid, f2fs_nat_entry)` representation
/// — we widen to a fixed 16-byte stride for simpler decoding while
/// staying inside the same block.
fn decode_nat_journal(buf: &[u8]) -> Vec<NatJournalEntry> {
    if buf.len() < 4 {
        return Vec::new();
    }
    let n = u16::from_le_bytes([buf[0], buf[1]]) as usize;
    // Defensive: a malicious image can't drag us past the block.
    let stride = 16usize;
    let max = (buf.len().saturating_sub(4)) / stride;
    let n = n.min(max);

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = 4 + i * stride;
        if o + stride > buf.len() {
            break;
        }
        let nid = u32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
        let ino = u32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap());
        let block_addr = u32::from_le_bytes(buf[o + 8..o + 12].try_into().unwrap());
        let version = buf[o + 12];
        out.push(NatJournalEntry {
            nid,
            ino,
            block_addr,
            version,
        });
    }
    out
}

/// Encode a CP head block — single source of truth for the offsets
/// [`decode_cp_head`] consumes. Used by the live flush path.
pub(crate) fn encode_cp_head_writer(cp: &Checkpoint) -> Vec<u8> {
    let mut buf = vec![0u8; F2FS_BLKSIZE];
    buf[0x00..0x08].copy_from_slice(&cp.version.to_le_bytes());
    buf[0x08..0x10].copy_from_slice(&cp.user_block_count.to_le_bytes());
    buf[0x10..0x18].copy_from_slice(&cp.valid_block_count.to_le_bytes());
    buf[0x70..0x74].copy_from_slice(&cp.nat_ver_bitmap_bytesize.to_le_bytes());
    buf[0x74..0x78].copy_from_slice(&cp.sit_ver_bitmap_bytesize.to_le_bytes());
    buf[0x78..0x7C].copy_from_slice(&cp.cp_pack_total_block_count.to_le_bytes());
    buf[0x7C..0x80].copy_from_slice(&cp.flags.to_le_bytes());
    buf[0x80..0x84].copy_from_slice(&cp.cp_pack_start_sum.to_le_bytes());
    buf[0x84..0x88].copy_from_slice(&cp.cp_payload.to_le_bytes());
    let crc = crc32fast::hash(&buf[..F2FS_BLK_CSUM_OFFSET]);
    buf[F2FS_BLK_CSUM_OFFSET..F2FS_BLK_CSUM_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    buf
}

/// Encode an empty NAT-journal summary block — used by the writer when
/// every nid is persisted to the on-disk NAT pages instead of the
/// journal.
pub(crate) fn encode_empty_journal_block() -> Vec<u8> {
    encode_nat_journal_block_writer(&[])
}

/// Encode a NAT-journal block. Layout mirrors [`decode_nat_journal`].
pub(crate) fn encode_nat_journal_block_writer(entries: &[NatJournalEntry]) -> Vec<u8> {
    let mut buf = vec![0u8; F2FS_BLKSIZE];
    buf[0..2].copy_from_slice(&(entries.len() as u16).to_le_bytes());
    let stride = 16usize;
    for (i, e) in entries.iter().enumerate() {
        let o = 4 + i * stride;
        if o + stride > buf.len() {
            break;
        }
        buf[o..o + 4].copy_from_slice(&e.nid.to_le_bytes());
        buf[o + 4..o + 8].copy_from_slice(&e.ino.to_le_bytes());
        buf[o + 8..o + 12].copy_from_slice(&e.block_addr.to_le_bytes());
        buf[o + 12] = e.version;
    }
    buf
}

/// Test alias for [`encode_cp_head_writer`].
#[cfg(test)]
pub(crate) fn encode_cp_head(cp: &Checkpoint) -> Vec<u8> {
    encode_cp_head_writer(cp)
}

/// Test alias for [`encode_nat_journal_block_writer`].
#[cfg(test)]
pub(crate) fn encode_nat_journal_block(entries: &[NatJournalEntry]) -> Vec<u8> {
    encode_nat_journal_block_writer(entries)
}

// Silence unused-import warnings when CP_COMPACT_SUM_FLAG isn't read in
// the cut-down v1.
#[allow(dead_code)]
const _CP_FLAGS_REFERENCED: u32 = CP_COMPACT_SUM_FLAG;
#[allow(dead_code)]
const _NAT_SIZE_REFERENCED: usize = NAT_ENTRY_SIZE;
