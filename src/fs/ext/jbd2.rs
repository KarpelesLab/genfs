//! JBD2 (ext3/4 journal) on-disk format and transaction commit/replay.
//!
//! genfs writes only the simplest flavour of JBD2 transactions:
//!
//! - Journal feature flags are all clear (no `INCOMPAT_64BIT`, no
//!   `INCOMPAT_CSUM_V2/V3`). Tags are the classic 8- (SAME_UUID) or
//!   24-byte (with UUID) `journal_block_tag_s` records, no per-tag
//!   checksum, no descriptor-block tail, no commit-block checksum.
//! - All journaled blocks share the same UUID; the descriptor block
//!   leaves the per-tag UUID set on the first tag only and flips
//!   `JBD2_FLAG_SAME_UUID` on the rest. Since we always use the
//!   filesystem's own UUID (which is also written into the journal
//!   superblock), the kernel/e2fsck accept the transaction.
//! - The commit block is the 32-byte header form: nothing past
//!   `h_commit_nsec` is used.
//!
//! ## Why JBD2 fields are big-endian
//!
//! ext4 metadata is little-endian, but JBD2 was designed to be portable
//! across SPARC mounts (which historically wrote big-endian); the kernel
//! converts every field through `be32_to_cpu` / `be64_to_cpu`. Our
//! encode/decode helpers follow suit.
//!
//! ## Layout summary (all offsets relative to the start of the block)
//!
//! Journal header (12 B), shared prefix of every block type:
//!
//! ```text
//!   0..4    h_magic       = 0xC03B_3998 (BE)
//!   4..8    h_blocktype   = 1=descriptor, 2=commit, 3=SB v1, 4=SB v2,
//!                           5=revocation (BE)
//!   8..12   h_sequence    = transaction id (BE)
//! ```
//!
//! Descriptor block tag (non-CSUM_V3, non-64BIT):
//!
//! ```text
//!   0..4    t_blocknr (low 32 bits) (BE)
//!   4..6    t_checksum (BE, zero when no CSUM_V2)
//!   6..8    t_flags (BE; bit 0=ESCAPE, bit 1=SAME_UUID, bit 3=LAST_TAG)
//!   8..24   tag UUID (omitted when SAME_UUID is set)
//! ```
//!
//! Commit block:
//!
//! ```text
//!   0..12   journal_header
//!   12..16  h_chksum_{type,size,padding[2]}
//!   16..48  h_chksum[8] (zero unless commit-block checksum requested)
//!   48..56  h_commit_sec (BE u64)
//!   56..60  h_commit_nsec (BE u32)
//! ```
//!
//! References: <https://docs.kernel.org/filesystems/ext4/journal.html>

use crate::Result;
use crate::block::BlockDevice;

/// JBD2 magic at offset 0 of every journal block (BE).
pub const JBD2_MAGIC: u32 = 0xC03B_3998;

/// `h_blocktype` constants.
pub const JBD2_DESCRIPTOR_BLOCK: u32 = 1;
pub const JBD2_COMMIT_BLOCK: u32 = 2;
pub const JBD2_SUPERBLOCK_V1: u32 = 3;
pub const JBD2_SUPERBLOCK_V2: u32 = 4;

/// Descriptor-tag flag bits.
pub const JBD2_FLAG_ESCAPE: u16 = 0x1;
pub const JBD2_FLAG_SAME_UUID: u16 = 0x2;
pub const JBD2_FLAG_LAST_TAG: u16 = 0x8;

/// Journal SB field offsets (big-endian on disk).
pub const JSB_OFF_BLOCKSIZE: usize = 12;
pub const JSB_OFF_MAXLEN: usize = 16;
pub const JSB_OFF_FIRST: usize = 20;
pub const JSB_OFF_SEQUENCE: usize = 24;
pub const JSB_OFF_START: usize = 28;
pub const JSB_OFF_FEATURE_INCOMPAT: usize = 40;
pub const JSB_OFF_UUID: usize = 48;

/// Decoded view of the parts of the journal superblock we care about.
#[derive(Debug, Clone, Copy)]
pub struct JournalSuperblock {
    pub blocksize: u32,
    pub maxlen: u32,
    pub first: u32,
    pub sequence: u32,
    pub start: u32,
    pub feature_incompat: u32,
    pub uuid: [u8; 16],
}

impl JournalSuperblock {
    /// Parse a journal-SB block. Validates the magic and the SB blocktype.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < 64 {
            return Err(crate::Error::InvalidImage(
                "ext: journal SB block shorter than 64 bytes".into(),
            ));
        }
        let magic = u32::from_be_bytes(buf[0..4].try_into().unwrap());
        if magic != JBD2_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "ext: bad JBD2 magic {magic:#010x} on journal SB block"
            )));
        }
        let blocktype = u32::from_be_bytes(buf[4..8].try_into().unwrap());
        if blocktype != JBD2_SUPERBLOCK_V1 && blocktype != JBD2_SUPERBLOCK_V2 {
            return Err(crate::Error::InvalidImage(format!(
                "ext: journal SB block has blocktype {blocktype} (expected v1=3 or v2=4)"
            )));
        }
        let mut uuid = [0u8; 16];
        uuid.copy_from_slice(&buf[JSB_OFF_UUID..JSB_OFF_UUID + 16]);
        Ok(Self {
            blocksize: u32::from_be_bytes(
                buf[JSB_OFF_BLOCKSIZE..JSB_OFF_BLOCKSIZE + 4]
                    .try_into()
                    .unwrap(),
            ),
            maxlen: u32::from_be_bytes(buf[JSB_OFF_MAXLEN..JSB_OFF_MAXLEN + 4].try_into().unwrap()),
            first: u32::from_be_bytes(buf[JSB_OFF_FIRST..JSB_OFF_FIRST + 4].try_into().unwrap()),
            sequence: u32::from_be_bytes(
                buf[JSB_OFF_SEQUENCE..JSB_OFF_SEQUENCE + 4]
                    .try_into()
                    .unwrap(),
            ),
            start: u32::from_be_bytes(buf[JSB_OFF_START..JSB_OFF_START + 4].try_into().unwrap()),
            feature_incompat: u32::from_be_bytes(
                buf[JSB_OFF_FEATURE_INCOMPAT..JSB_OFF_FEATURE_INCOMPAT + 4]
                    .try_into()
                    .unwrap(),
            ),
            uuid,
        })
    }
}

/// Encode a 12-byte journal block header.
pub fn encode_header(blocktype: u32, sequence: u32) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0..4].copy_from_slice(&JBD2_MAGIC.to_be_bytes());
    out[4..8].copy_from_slice(&blocktype.to_be_bytes());
    out[8..12].copy_from_slice(&sequence.to_be_bytes());
    out
}

/// One block to be journaled: the destination filesystem block number
/// and a snapshot of its post-commit contents.
#[derive(Debug, Clone)]
pub struct JournalBlock {
    pub fs_block: u32,
    pub bytes: Vec<u8>,
}

/// Build the descriptor block bytes (`block_size` long) listing one
/// chunk of a (potentially multi-descriptor) transaction.
///
/// `is_first_descriptor` is true when this is the very first descriptor
/// in the transaction — its first tag carries the 16-byte UUID payload
/// and clears `SAME_UUID`. Continuation descriptors set `SAME_UUID`
/// on every tag, dropping the 16-byte overhead.
///
/// `is_last_descriptor` is true when this is the final descriptor in
/// the transaction — its last tag gets `LAST_TAG` so the reader knows
/// to expect a commit block after this descriptor's data payloads.
/// Continuation descriptors don't set `LAST_TAG`; the reader probes
/// the next block to decide whether to keep walking or finalise.
///
/// Tag capacity is `(bs - 12 - 16) / 8` for the first descriptor and
/// `(bs - 12) / 8` for continuations; see [`descriptor_tag_capacity`].
pub fn encode_descriptor_block(
    block_size: u32,
    sequence: u32,
    blocks: &[JournalBlock],
    uuid: &[u8; 16],
    is_first_descriptor: bool,
    is_last_descriptor: bool,
) -> Vec<u8> {
    let mut out = vec![0u8; block_size as usize];
    out[..12].copy_from_slice(&encode_header(JBD2_DESCRIPTOR_BLOCK, sequence));
    let mut off = 12usize;
    for (i, jb) in blocks.iter().enumerate() {
        let is_very_first_tag = is_first_descriptor && i == 0;
        let is_very_last_tag = is_last_descriptor && i + 1 == blocks.len();
        let mut flags: u16 = 0;
        if !is_very_first_tag {
            flags |= JBD2_FLAG_SAME_UUID;
        }
        if is_very_last_tag {
            flags |= JBD2_FLAG_LAST_TAG;
        }
        // t_blocknr (low 32 bits)
        out[off..off + 4].copy_from_slice(&jb.fs_block.to_be_bytes());
        // t_checksum (low 16, BE) — zero, no CSUM_V2
        out[off + 4..off + 6].copy_from_slice(&0u16.to_be_bytes());
        // t_flags (BE)
        out[off + 6..off + 8].copy_from_slice(&flags.to_be_bytes());
        off += 8;
        if is_very_first_tag {
            out[off..off + 16].copy_from_slice(uuid);
            off += 16;
        }
    }
    out
}

/// Number of tags one descriptor block can hold at `block_size`. The
/// first descriptor in a transaction loses 16 bytes to the UUID
/// payload after its first tag; continuation descriptors don't.
pub fn descriptor_tag_capacity(block_size: u32, is_first_descriptor: bool) -> usize {
    let header = 12usize;
    let uuid_overhead = if is_first_descriptor { 16 } else { 0 };
    (block_size as usize - header - uuid_overhead) / 8
}

/// Build the commit block bytes (`block_size` long). Without any
/// `INCOMPAT_CSUM_*` feature the checksum bytes are left zero — the
/// kernel ignores them when the feature flag is clear. `commit_sec` /
/// `commit_nsec` carry a best-effort wall-clock timestamp for log dumps.
pub fn encode_commit_block(
    block_size: u32,
    sequence: u32,
    commit_sec: u64,
    commit_nsec: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; block_size as usize];
    out[..12].copy_from_slice(&encode_header(JBD2_COMMIT_BLOCK, sequence));
    // 12..14: h_chksum_type / h_chksum_size — zero when no commit csum
    // 14..16: h_padding[2] — zero
    // 16..48: h_chksum[8] (32 bytes) — zero
    out[48..56].copy_from_slice(&commit_sec.to_be_bytes());
    out[56..60].copy_from_slice(&commit_nsec.to_be_bytes());
    out
}

/// Update the journal SB's `s_sequence` field (BE u32 at offset 24).
/// Caller is responsible for writing the buffer back.
pub fn set_sequence(buf: &mut [u8], sequence: u32) {
    buf[JSB_OFF_SEQUENCE..JSB_OFF_SEQUENCE + 4].copy_from_slice(&sequence.to_be_bytes());
}

/// Update the journal SB's `s_start` field (BE u32 at offset 28). A
/// non-zero value marks the journal as having work to replay starting at
/// that block; zero is the clean-shutdown sentinel.
pub fn set_start(buf: &mut [u8], start: u32) {
    buf[JSB_OFF_START..JSB_OFF_START + 4].copy_from_slice(&start.to_be_bytes());
}

/// Decode one classic (non-V3, non-64BIT) tag from `buf`. Returns
/// `(t_blocknr, t_flags, tag_size_in_bytes_including_uuid)`.
///
/// `is_first` controls whether we read a 16-byte UUID after the 8-byte
/// header: per the kernel docs, the UUID is present "unless SAME_UUID is
/// set". The first tag in a descriptor block always carries the UUID
/// (it's the seed of the SAME_UUID chain); subsequent tags carry one
/// only when their SAME_UUID flag is clear.
pub fn decode_tag(buf: &[u8], is_first: bool) -> Result<(u32, u16, usize)> {
    if buf.len() < 8 {
        return Err(crate::Error::InvalidImage(
            "ext: journal descriptor tag past end of block".into(),
        ));
    }
    let blocknr = u32::from_be_bytes(buf[0..4].try_into().unwrap());
    // t_checksum (low 16 bits, BE) at 4..6 — ignored without CSUM_V2.
    let flags = u16::from_be_bytes(buf[6..8].try_into().unwrap());
    let has_uuid = is_first || (flags & JBD2_FLAG_SAME_UUID) == 0;
    let size = if has_uuid { 24 } else { 8 };
    if buf.len() < size {
        return Err(crate::Error::InvalidImage(
            "ext: journal descriptor tag uuid past end of block".into(),
        ));
    }
    Ok((blocknr, flags, size))
}

/// Read journal-relative block `idx` and return its bytes. Maps through
/// the journal inode's block tree via [`crate::fs::ext::Ext::file_block`].
pub(crate) fn read_journal_block(
    ext: &super::Ext,
    dev: &mut dyn BlockDevice,
    journal_inode: &super::Inode,
    idx: u32,
) -> Result<Vec<u8>> {
    let phys = ext.file_block(dev, journal_inode, idx)?;
    if phys == 0 {
        return Err(crate::Error::InvalidImage(format!(
            "ext: journal block {idx} unmapped"
        )));
    }
    let bs = ext.layout.block_size as usize;
    let mut buf = vec![0u8; bs];
    dev.read_at(phys as u64 * bs as u64, &mut buf)?;
    Ok(buf)
}

/// Write journal-relative block `idx` from `bytes`.
pub(crate) fn write_journal_block(
    ext: &super::Ext,
    dev: &mut dyn BlockDevice,
    journal_inode: &super::Inode,
    idx: u32,
    bytes: &[u8],
) -> Result<()> {
    let phys = ext.file_block(dev, journal_inode, idx)?;
    if phys == 0 {
        return Err(crate::Error::InvalidImage(format!(
            "ext: journal block {idx} unmapped"
        )));
    }
    let bs = ext.layout.block_size as u64;
    dev.write_at(phys as u64 * bs, bytes)?;
    Ok(())
}

/// Replay any committed-but-not-checkpointed transactions in the journal.
/// Walks the log starting at `s_start`, transaction by transaction, and
/// applies each transaction's data blocks to their target FS locations.
/// On clean exit (`s_start == 0`) this is a no-op.
///
/// Returns `true` if any work was replayed (caller may need to refresh
/// in-memory bitmaps from disk).
pub(crate) fn replay_journal(ext: &super::Ext, dev: &mut dyn BlockDevice) -> Result<bool> {
    let jino = ext.sb.journal_inum;
    if jino == 0 {
        return Ok(false);
    }
    let journal_inode = ext.read_inode(dev, jino)?;
    let bs = ext.layout.block_size;
    let jsb_buf = read_journal_block(ext, dev, &journal_inode, 0)?;
    let jsb = JournalSuperblock::decode(&jsb_buf)?;
    if jsb.start == 0 {
        return Ok(false);
    }
    if jsb.blocksize != bs {
        return Err(crate::Error::InvalidImage(format!(
            "ext: journal blocksize {} != FS blocksize {bs}",
            jsb.blocksize
        )));
    }

    let mut idx = jsb.start;
    let mut expected_tid = jsb.sequence;
    let mut replayed = false;
    'transactions: loop {
        let blk = read_journal_block(ext, dev, &journal_inode, idx)?;
        let magic = u32::from_be_bytes(blk[0..4].try_into().unwrap());
        if magic != JBD2_MAGIC {
            // Not a JBD2-tagged block — end of log.
            break;
        }
        let blocktype = u32::from_be_bytes(blk[4..8].try_into().unwrap());
        let tid = u32::from_be_bytes(blk[8..12].try_into().unwrap());
        if tid != expected_tid {
            // Sequence number mismatch — log ends here (stale data from
            // an older transaction reused log space).
            break;
        }
        if blocktype != JBD2_DESCRIPTOR_BLOCK {
            // Either a stray commit (no data) or an unknown block; bail
            // out of replay rather than guessing.
            break;
        }

        // Walk descriptor blocks for this transaction until LAST_TAG
        // appears or the next block stops being a descriptor. Apply
        // every tag's data payload to its FS-home block as we go.
        let mut current_desc = blk;
        loop {
            let (data_targets, _) = parse_descriptor_tags(&current_desc, bs)?;
            let saw_last_tag = data_targets
                .iter()
                .any(|t| t.flags & JBD2_FLAG_LAST_TAG != 0);
            idx = ring_next(idx, &jsb);
            for tag in &data_targets {
                let mut payload = read_journal_block(ext, dev, &journal_inode, idx)?;
                if tag.flags & JBD2_FLAG_ESCAPE != 0 {
                    payload[0..4].copy_from_slice(&JBD2_MAGIC.to_be_bytes());
                }
                dev.write_at(tag.fs_block as u64 * bs as u64, &payload)?;
                idx = ring_next(idx, &jsb);
            }
            if saw_last_tag {
                break;
            }
            // Peek the next block: if it's another descriptor with the
            // same tid, keep going; if it's a commit, finalise; if
            // neither, bail.
            let peek = read_journal_block(ext, dev, &journal_inode, idx)?;
            let peek_magic = u32::from_be_bytes(peek[0..4].try_into().unwrap());
            if peek_magic != JBD2_MAGIC {
                break 'transactions;
            }
            let peek_type = u32::from_be_bytes(peek[4..8].try_into().unwrap());
            let peek_tid = u32::from_be_bytes(peek[8..12].try_into().unwrap());
            if peek_tid != tid {
                break 'transactions;
            }
            if peek_type == JBD2_COMMIT_BLOCK {
                break;
            }
            if peek_type != JBD2_DESCRIPTOR_BLOCK {
                break 'transactions;
            }
            current_desc = peek;
        }

        // Next block must be a commit block with the same tid.
        let commit_buf = read_journal_block(ext, dev, &journal_inode, idx)?;
        let cmagic = u32::from_be_bytes(commit_buf[0..4].try_into().unwrap());
        let ctype = u32::from_be_bytes(commit_buf[4..8].try_into().unwrap());
        let ctid = u32::from_be_bytes(commit_buf[8..12].try_into().unwrap());
        let commit_seen =
            cmagic == JBD2_MAGIC && ctype == JBD2_COMMIT_BLOCK && ctid == tid;
        idx = ring_next(idx, &jsb);

        if !commit_seen {
            // Descriptor sequence without a matching commit — partial
            // transaction, don't apply it (replay is atomic).
            break;
        }

        replayed = true;
        expected_tid = expected_tid.wrapping_add(1);
    }

    if replayed {
        // Mark the journal clean: s_start = 0, s_sequence = next-expected
        // tid (so the next mutation reuses a fresh sequence). Clear the
        // FS-level INCOMPAT_RECOVER if it was set (we have, in fact, done
        // the recovery).
        let mut jsb_new = jsb_buf.clone();
        set_start(&mut jsb_new, 0);
        set_sequence(&mut jsb_new, expected_tid);
        write_journal_block(ext, dev, &journal_inode, 0, &jsb_new)?;
    }
    Ok(replayed)
}

/// Compute the next journal ring index. `idx` wraps from `maxlen - 1` back
/// to `first` (block 0 is the SB; usable log is `first..maxlen`).
pub(crate) fn ring_next(idx: u32, jsb: &JournalSuperblock) -> u32 {
    let next = idx + 1;
    if next >= jsb.maxlen { jsb.first } else { next }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ParsedTag {
    pub fs_block: u32,
    pub flags: u16,
}

/// Parse a descriptor block's tag array into `(tags, total_count)`.
pub(crate) fn parse_descriptor_tags(
    buf: &[u8],
    block_size: u32,
) -> Result<(Vec<ParsedTag>, usize)> {
    let mut out = Vec::new();
    let mut off = 12usize;
    let mut first = true;
    while off + 8 <= block_size as usize {
        let (fs_block, flags, sz) = decode_tag(&buf[off..], first)?;
        if fs_block == 0 && flags == 0 && first {
            // Empty descriptor — bail.
            break;
        }
        out.push(ParsedTag { fs_block, flags });
        off += sz;
        first = false;
        if flags & JBD2_FLAG_LAST_TAG != 0 {
            break;
        }
    }
    let count = out.len();
    Ok((out, count))
}

/// Write a fresh transaction into the journal: descriptor, data payload
/// blocks, commit. Updates the in-memory `jsb` view (the caller is
/// responsible for stamping the new `s_sequence` / `s_start` into the
/// on-disk journal SB at the right moment).
///
/// `jsb_buf` is the live journal-superblock block (read+modified+written
/// here). `start_idx` is the journal block index where this transaction
/// begins; on return the caller knows it lands at `start_idx` and
/// occupies `1 + blocks.len() + 1` journal blocks.
///
/// Returns the journal block index immediately past the commit block —
/// where the next transaction would start.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_transaction(
    ext: &super::Ext,
    dev: &mut dyn BlockDevice,
    journal_inode: &super::Inode,
    jsb_buf: &mut [u8],
    jsb: &JournalSuperblock,
    start_idx: u32,
    tid: u32,
    blocks: &[JournalBlock],
    commit_sec: u64,
    commit_nsec: u32,
) -> Result<u32> {
    let bs = ext.layout.block_size;
    let first_cap = descriptor_tag_capacity(bs, true);
    let next_cap = descriptor_tag_capacity(bs, false);

    // Total journal blocks: one descriptor per chunk + every data
    // block + one trailing commit. First chunk holds up to `first_cap`
    // tags (it carries the UUID payload), subsequent chunks up to
    // `next_cap` each.
    let n_descs = if blocks.is_empty() {
        1
    } else if blocks.len() <= first_cap {
        1
    } else {
        1 + (blocks.len() - first_cap).div_ceil(next_cap)
    };
    let need = (n_descs + blocks.len() + 1) as u32;
    let avail = jsb.maxlen.saturating_sub(jsb.first);
    if need > avail {
        return Err(crate::Error::Unsupported(format!(
            "ext: journal too small ({} blocks, transaction needs {need})",
            jsb.maxlen
        )));
    }

    let mut idx = start_idx;
    let mut chunk_start = 0usize;
    let mut is_first_desc = true;
    while chunk_start < blocks.len().max(1) {
        let cap = if is_first_desc { first_cap } else { next_cap };
        let chunk_end = (chunk_start + cap).min(blocks.len());
        let chunk = if blocks.is_empty() {
            &[][..]
        } else {
            &blocks[chunk_start..chunk_end]
        };
        let is_last_desc = chunk_end == blocks.len();

        let desc = encode_descriptor_block(
            bs,
            tid,
            chunk,
            &jsb.uuid,
            is_first_desc,
            is_last_desc,
        );
        write_journal_block(ext, dev, journal_inode, idx, &desc)?;
        idx = ring_next(idx, jsb);

        for jb in chunk {
            debug_assert_eq!(jb.bytes.len(), bs as usize, "journal payload wrong size");
            write_journal_block(ext, dev, journal_inode, idx, &jb.bytes)?;
            idx = ring_next(idx, jsb);
        }

        chunk_start = chunk_end;
        is_first_desc = false;
        if blocks.is_empty() {
            // Special case: an empty transaction emits one empty
            // descriptor followed by the commit. Break to skip the
            // outer loop's bounds bump (which would underflow).
            break;
        }
    }

    // Commit.
    let commit = encode_commit_block(bs, tid, commit_sec, commit_nsec);
    write_journal_block(ext, dev, journal_inode, idx, &commit)?;
    let after = ring_next(idx, jsb);

    // Bump the in-memory copy of the journal SB. Caller writes it back
    // at the right moment (after the commit block hits disk).
    set_start(jsb_buf, start_idx);
    set_sequence(jsb_buf, tid);
    Ok(after)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let h = encode_header(JBD2_COMMIT_BLOCK, 0x1234_5678);
        assert_eq!(u32::from_be_bytes(h[0..4].try_into().unwrap()), JBD2_MAGIC);
        assert_eq!(
            u32::from_be_bytes(h[4..8].try_into().unwrap()),
            JBD2_COMMIT_BLOCK
        );
        assert_eq!(
            u32::from_be_bytes(h[8..12].try_into().unwrap()),
            0x1234_5678
        );
    }

    #[test]
    fn descriptor_layout() {
        let blocks = vec![
            JournalBlock {
                fs_block: 100,
                bytes: vec![0; 1024],
            },
            JournalBlock {
                fs_block: 200,
                bytes: vec![0; 1024],
            },
        ];
        let uuid = [0xAA; 16];
        let buf = encode_descriptor_block(1024, 7, &blocks, &uuid, true, true);
        // Header.
        assert_eq!(
            u32::from_be_bytes(buf[0..4].try_into().unwrap()),
            JBD2_MAGIC
        );
        assert_eq!(
            u32::from_be_bytes(buf[4..8].try_into().unwrap()),
            JBD2_DESCRIPTOR_BLOCK
        );
        assert_eq!(u32::from_be_bytes(buf[8..12].try_into().unwrap()), 7);
        // Tag 0: block 100, no SAME_UUID, no LAST_TAG, UUID embedded.
        assert_eq!(u32::from_be_bytes(buf[12..16].try_into().unwrap()), 100);
        let flags0 = u16::from_be_bytes(buf[18..20].try_into().unwrap());
        assert_eq!(flags0 & JBD2_FLAG_SAME_UUID, 0);
        assert_eq!(flags0 & JBD2_FLAG_LAST_TAG, 0);
        assert_eq!(&buf[20..36], &uuid);
        // Tag 1 starts at offset 36 (12 hdr + 24 tag0). LAST_TAG + SAME_UUID set.
        assert_eq!(u32::from_be_bytes(buf[36..40].try_into().unwrap()), 200);
        let flags1 = u16::from_be_bytes(buf[42..44].try_into().unwrap());
        assert!(flags1 & JBD2_FLAG_SAME_UUID != 0);
        assert!(flags1 & JBD2_FLAG_LAST_TAG != 0);
    }

    #[test]
    fn descriptor_round_trip_parses() {
        let blocks = vec![
            JournalBlock {
                fs_block: 100,
                bytes: vec![0; 1024],
            },
            JournalBlock {
                fs_block: 200,
                bytes: vec![0; 1024],
            },
            JournalBlock {
                fs_block: 300,
                bytes: vec![0; 1024],
            },
        ];
        let uuid = [0x42; 16];
        let buf = encode_descriptor_block(1024, 9, &blocks, &uuid, true, true);
        let (tags, n) = parse_descriptor_tags(&buf, 1024).unwrap();
        assert_eq!(n, 3);
        assert_eq!(tags[0].fs_block, 100);
        assert_eq!(tags[1].fs_block, 200);
        assert_eq!(tags[2].fs_block, 300);
        assert!(tags[2].flags & JBD2_FLAG_LAST_TAG != 0);
    }

    #[test]
    fn commit_layout() {
        let buf = encode_commit_block(1024, 42, 1_234_567, 890);
        assert_eq!(
            u32::from_be_bytes(buf[0..4].try_into().unwrap()),
            JBD2_MAGIC
        );
        assert_eq!(
            u32::from_be_bytes(buf[4..8].try_into().unwrap()),
            JBD2_COMMIT_BLOCK
        );
        assert_eq!(u32::from_be_bytes(buf[8..12].try_into().unwrap()), 42);
        // commit_sec at 48..56 (BE u64), commit_nsec at 56..60 (BE u32).
        assert_eq!(
            u64::from_be_bytes(buf[48..56].try_into().unwrap()),
            1_234_567
        );
        assert_eq!(u32::from_be_bytes(buf[56..60].try_into().unwrap()), 890);
    }

    #[test]
    fn ring_next_wraps() {
        let jsb = JournalSuperblock {
            blocksize: 1024,
            maxlen: 10,
            first: 1,
            sequence: 1,
            start: 0,
            feature_incompat: 0,
            uuid: [0; 16],
        };
        assert_eq!(ring_next(1, &jsb), 2);
        assert_eq!(ring_next(8, &jsb), 9);
        assert_eq!(ring_next(9, &jsb), 1);
    }
}
