//! HFS+ journal — Path A "real transactions".
//!
//! Apple TN1150 specifies a `JournalInfoBlock` pointing at a circular
//! journal buffer whose first sector is a [`JournalHeader`] (magic
//! `"JNLx"`). Between `start` and `end` lies a sequence of transactions;
//! each transaction is one or more *block lists*, where a block list
//! consists of a `block_list_header_t` + an array of `block_info_t`
//! entries describing the disk blocks being committed, followed by the
//! block data itself.
//!
//! ## On-disk layout (per block list, packed big-endian)
//!
//! ```text
//! 0  2  max_blocks
//! 2  2  num_blocks            (includes the sentinel info[0])
//! 4  4  bytes_used            (header + info_array + data)
//! 8  4  checksum              (zeroed when computing)
//! 12 4  flags / padding
//! 16 .. block_info[num_blocks] {
//!         u64 bnum            (sector number on disk, sector = 512 B)
//!         u32 bsize           (bytes of data; multiple of 512)
//!         u32 next            (in-memory only; written zero)
//!       }
//! ```
//!
//! The first `block_info` slot is a sentinel: its `bnum` is the
//! transaction's total byte size (`bytes_used` above) and its `bsize`
//! is 0. The block_info array is followed by [`BLHDR_SIZE`] padding so
//! that the concatenated block data starts at a fixed offset from the
//! block-list header.
//!
//! ## Endianness
//!
//! Every multi-byte field inside the journal is big-endian, matching the
//! `JOURNAL_HEADER_ENDIAN = 0x12345678` constant we encode in the
//! journal header. A LE-host reader detects the byte-swap via that
//! field and adjusts accordingly.
//!
//! ## Replay
//!
//! [`replay`] walks the circular buffer from `start` to `end`, copies
//! each described data chunk to its target sector, and then advances
//! `start := end` on disk. Idempotent — replaying a clean journal
//! (start == end) is a no-op.
//!
//! ## Crash safety
//!
//! [`JournalLog::commit`] writes the transaction body + advances `end`
//! BEFORE applying the in-place writes. A crash between those two
//! phases leaves a valid journal entry that the next [`replay`] will
//! re-apply, restoring the file system to the post-commit state.

use crate::Result;
use crate::block::BlockDevice;

use super::writer::{JOURNAL_HEADER_ENDIAN, JOURNAL_HEADER_MAGIC, VOL_ATTR_JOURNALED};

/// Hardcoded sector size used for `bnum` in the on-disk block_info_t.
/// macOS uses 512 universally; HFS+ block sizes are larger but the
/// journal always describes I/O in 512-byte units.
pub const JOURNAL_SECTOR: u64 = 512;

/// Size of one block-list-header region (header + info array, before
/// the block data). 4096 bytes gives us up to (4096 / 16) - 1 = 255
/// blocks per list, which is more than enough for a single sync.
pub const BLHDR_SIZE: u32 = 4096;

/// Journal-header size in bytes (one 512-byte sector).
pub const JHDR_SIZE: u32 = 512;

/// Number of bytes occupied by one block_info entry on disk.
const BINFO_SIZE: usize = 16;

/// The 16-byte fixed prefix of a `block_list_header_t` (before the
/// `block_info` array).
const BLHDR_FIXED_SIZE: usize = 16;

/// Per-block pending write. `dev_off` is the byte offset on the
/// underlying volume; `data` is its replacement contents.
#[derive(Debug, Clone)]
pub(crate) struct PendingBlock {
    pub dev_off: u64,
    pub data: Vec<u8>,
}

/// In-memory journal log. Constructed from the on-disk journal-info
/// block; collects pending writes via [`JournalLog::add`] and emits
/// one transaction per [`JournalLog::commit`].
pub(crate) struct JournalLog {
    /// Byte offset of the journal buffer on the volume.
    pub buf_off: u64,
    /// Size of the journal buffer (the circular ring).
    pub buf_size: u64,
    /// Current `start` field from the on-disk header.
    pub start: u64,
    /// Current `end` field from the on-disk header.
    pub end: u64,
    /// Pending writes accumulated since the last commit. Coalesced
    /// by `dev_off` on commit so the latest data wins.
    pending: Vec<PendingBlock>,
}

impl JournalLog {
    /// Read the volume's journal-info block and journal header. Returns
    /// `Ok(None)` if the volume is not journaled (or the JIB pointer
    /// is zero — defensive fallback).
    pub fn load(
        dev: &mut dyn BlockDevice,
        vh: &super::volume_header::VolumeHeader,
    ) -> Result<Option<Self>> {
        if vh.attributes & VOL_ATTR_JOURNALED == 0 {
            return Ok(None);
        }
        let info_block = vh.journal_info_block;
        if info_block == 0 {
            return Ok(None);
        }
        let bs = u64::from(vh.block_size);
        let info_off = u64::from(info_block) * bs;
        let mut info = [0u8; 52];
        dev.read_at(info_off, &mut info)?;
        let buf_off = u64::from_be_bytes(info[36..44].try_into().unwrap());
        let buf_size = u64::from_be_bytes(info[44..52].try_into().unwrap());
        if buf_off == 0 || buf_size == 0 {
            return Ok(None);
        }
        let mut hdr = [0u8; 24];
        dev.read_at(buf_off, &mut hdr)?;
        let magic = u32::from_be_bytes(hdr[0..4].try_into().unwrap());
        let endian = u32::from_be_bytes(hdr[4..8].try_into().unwrap());
        if magic != JOURNAL_HEADER_MAGIC || endian != JOURNAL_HEADER_ENDIAN {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+ journal: unrecognised header magic/endian \
                 ({magic:#010x}/{endian:#010x})"
            )));
        }
        let start = u64::from_be_bytes(hdr[8..16].try_into().unwrap());
        let end = u64::from_be_bytes(hdr[16..24].try_into().unwrap());
        Ok(Some(Self {
            buf_off,
            buf_size,
            start,
            end,
            pending: Vec::new(),
        }))
    }

    /// True iff there are unreplayed transactions on disk.
    pub fn is_dirty(&self) -> bool {
        self.start != self.end
    }

    /// Queue a pending write. If `data.len()` is not a multiple of
    /// [`JOURNAL_SECTOR`], the recorded buffer is padded with zeros to
    /// the next sector boundary on commit.
    pub fn add(&mut self, dev_off: u64, data: Vec<u8>) {
        // Coalesce identical-range writes so the latest one wins.
        for slot in self.pending.iter_mut() {
            if slot.dev_off == dev_off && slot.data.len() == data.len() {
                slot.data = data;
                return;
            }
        }
        self.pending.push(PendingBlock { dev_off, data });
    }

    /// Search pending writes for the byte at `dev_off`. Used by the
    /// file handle to serve reads of bytes we've buffered but not yet
    /// committed. Returns the slice (and its start offset) of the
    /// pending block that contains `dev_off`, if any.
    pub fn lookup(&self, dev_off: u64) -> Option<(u64, &[u8])> {
        for p in self.pending.iter().rev() {
            let end = p.dev_off + p.data.len() as u64;
            if dev_off >= p.dev_off && dev_off < end {
                return Some((p.dev_off, &p.data));
            }
        }
        None
    }

    /// Commit the pending writes to disk through the journal. Steps:
    ///   1. Round each block to a whole number of [`JOURNAL_SECTOR`]s.
    ///   2. Build a block-list transaction at offset `end` in the
    ///      circular buffer. Update on-disk `end` (header rewrite).
    ///   3. Apply each block to its target dev offset.
    ///   4. Advance on-disk `start := end`, header rewrite.
    ///
    /// The order is critical: after step 2 a crash leaves a complete
    /// journal entry that the next [`replay`] will redo. Between steps
    /// 3 and 4 a crash also leaves the journal claiming unreplayed
    /// work — replay is idempotent.
    pub fn commit(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        // 1. Lay out the transaction.
        let entries = std::mem::take(&mut self.pending);
        let mut padded: Vec<(u64, Vec<u8>)> = Vec::with_capacity(entries.len());
        for e in entries {
            let pad_len = (e.data.len() as u64).div_ceil(JOURNAL_SECTOR) * JOURNAL_SECTOR;
            let mut buf = e.data;
            buf.resize(pad_len as usize, 0);
            padded.push((e.dev_off, buf));
        }

        // 2. Encode the block list.
        let num_info: u16 = u16::try_from(padded.len() + 1).map_err(|_| {
            crate::Error::Unsupported(
                "hfs+ journal: too many blocks in one transaction (>65534)".into(),
            )
        })?;
        let data_total: u32 = padded
            .iter()
            .map(|(_, d)| d.len() as u32)
            .fold(0u32, |a, b| a.saturating_add(b));
        let bytes_used: u32 = BLHDR_SIZE.checked_add(data_total).ok_or_else(|| {
            crate::Error::Unsupported("hfs+ journal: transaction overflows u32".into())
        })?;
        if u64::from(bytes_used) > self.buf_size - u64::from(JHDR_SIZE) {
            return Err(crate::Error::Unsupported(
                "hfs+ journal: transaction larger than journal buffer".into(),
            ));
        }

        let mut tx = vec![0u8; bytes_used as usize];
        // block_list_header_t (16 B, the first 4 are followed by info[0])
        tx[0..2].copy_from_slice(&num_info.to_be_bytes());
        tx[2..4].copy_from_slice(&num_info.to_be_bytes());
        tx[4..8].copy_from_slice(&bytes_used.to_be_bytes());
        // 8..12 checksum (filled at end)
        // 12..16 flags / padding stay zero

        let info_base = BLHDR_FIXED_SIZE;
        // Slot 0 sentinel.
        tx[info_base..info_base + 8].copy_from_slice(&u64::from(bytes_used).to_be_bytes());
        // bsize / next zero.

        // Slots 1..num_info describe the data blocks.
        for (i, (dev_off, data)) in padded.iter().enumerate() {
            let slot = info_base + (i + 1) * BINFO_SIZE;
            let sector = dev_off / JOURNAL_SECTOR;
            let bsize = data.len() as u32;
            tx[slot..slot + 8].copy_from_slice(&sector.to_be_bytes());
            tx[slot + 8..slot + 12].copy_from_slice(&bsize.to_be_bytes());
        }

        // 3. Concatenated block data starts at BLHDR_SIZE.
        let mut cursor = BLHDR_SIZE as usize;
        for (_off, data) in &padded {
            tx[cursor..cursor + data.len()].copy_from_slice(data);
            cursor += data.len();
        }

        // 4. Checksum over the 16-byte block_list_header.
        let csum = crc32_reflected(&tx[..BLHDR_FIXED_SIZE]);
        tx[8..12].copy_from_slice(&csum.to_be_bytes());

        // 5. Write the transaction into the ring at `end`.
        let usable = self.buf_size - u64::from(JHDR_SIZE);
        let tx_off_in_ring = self.end - u64::from(JHDR_SIZE);
        let new_end;
        if tx_off_in_ring + u64::from(bytes_used) > usable {
            // Won't fit before the wrap point. Restart at JHDR_SIZE.
            new_end = u64::from(JHDR_SIZE) + u64::from(bytes_used);
            self.write_at_buffer(dev, u64::from(JHDR_SIZE), &tx)?;
        } else {
            new_end = self.end + u64::from(bytes_used);
            self.write_at_buffer(dev, tx_off_in_ring + u64::from(JHDR_SIZE), &tx)?;
        }

        // 6. Persist `end` on disk. From this point on a crash leaves
        //    a valid transaction the next replay will apply.
        write_journal_header(dev, self.buf_off, self.start, new_end, self.buf_size)?;
        dev.sync()?;
        self.end = new_end;

        // 7. Apply the actual block writes in place.
        for (dev_off, data) in &padded {
            dev.write_at(*dev_off, data)?;
        }
        dev.sync()?;

        // 8. Advance `start := end` to mark the transaction replayed.
        write_journal_header(dev, self.buf_off, self.end, self.end, self.buf_size)?;
        dev.sync()?;
        self.start = self.end;

        Ok(())
    }

    /// Write `data` into the journal buffer at a journal-buffer-relative
    /// `ring_off` (i.e. absolute disk offset `buf_off + ring_off`).
    fn write_at_buffer(&self, dev: &mut dyn BlockDevice, ring_off: u64, data: &[u8]) -> Result<()> {
        let abs = self.buf_off + ring_off;
        if abs + data.len() as u64 > self.buf_off + self.buf_size {
            return Err(crate::Error::Unsupported(
                "hfs+ journal: write would overflow buffer".into(),
            ));
        }
        dev.write_at(abs, data)
    }
}

/// Walk `[start, end)` of the journal buffer and apply every block
/// described by every transaction in that range, then on-disk advance
/// `start := end`. Idempotent when `start == end` (no-op).
pub(crate) fn replay(
    dev: &mut dyn BlockDevice,
    vh: &super::volume_header::VolumeHeader,
) -> Result<()> {
    let Some(log) = JournalLog::load(dev, vh)? else {
        return Ok(());
    };
    if !log.is_dirty() {
        return Ok(());
    }
    let mut cursor = log.start;
    let end = log.end;
    while cursor != end {
        let mut hdr_fixed = [0u8; BLHDR_FIXED_SIZE];
        dev.read_at(log.buf_off + cursor, &mut hdr_fixed)?;
        let num_blocks = u16::from_be_bytes(hdr_fixed[2..4].try_into().unwrap()) as usize;
        let bytes_used = u32::from_be_bytes(hdr_fixed[4..8].try_into().unwrap());
        if num_blocks == 0 || bytes_used < BLHDR_SIZE {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+ journal: malformed block list (num={num_blocks}, bytes={bytes_used})"
            )));
        }
        let info_bytes = num_blocks * BINFO_SIZE;
        let mut info = vec![0u8; info_bytes];
        dev.read_at(log.buf_off + cursor + BLHDR_FIXED_SIZE as u64, &mut info)?;
        let mut data_cursor = log.buf_off + cursor + u64::from(BLHDR_SIZE);
        for i in 1..num_blocks {
            let slot = i * BINFO_SIZE;
            let sector = u64::from_be_bytes(info[slot..slot + 8].try_into().unwrap());
            let bsize = u32::from_be_bytes(info[slot + 8..slot + 12].try_into().unwrap()) as usize;
            let mut data = vec![0u8; bsize];
            dev.read_at(data_cursor, &mut data)?;
            let target = sector * JOURNAL_SECTOR;
            dev.write_at(target, &data)?;
            data_cursor += bsize as u64;
        }
        cursor += u64::from(bytes_used);
        // Wrap point.
        if cursor >= log.buf_size {
            cursor = u64::from(JHDR_SIZE);
        }
    }
    write_journal_header(dev, log.buf_off, end, end, log.buf_size)?;
    dev.sync()?;
    Ok(())
}

/// Write a fresh 512-byte journal header at the start of the journal
/// buffer, carrying the supplied `start` / `end` / `size` fields.
pub(crate) fn write_journal_header(
    dev: &mut dyn BlockDevice,
    buf_off: u64,
    start: u64,
    end: u64,
    buf_size: u64,
) -> Result<()> {
    let mut b = [0u8; JHDR_SIZE as usize];
    b[0..4].copy_from_slice(&JOURNAL_HEADER_MAGIC.to_be_bytes());
    b[4..8].copy_from_slice(&JOURNAL_HEADER_ENDIAN.to_be_bytes());
    b[8..16].copy_from_slice(&start.to_be_bytes());
    b[16..24].copy_from_slice(&end.to_be_bytes());
    b[24..32].copy_from_slice(&buf_size.to_be_bytes());
    b[32..36].copy_from_slice(&BLHDR_SIZE.to_be_bytes());
    b[36..40].copy_from_slice(&0u32.to_be_bytes());
    b[40..44].copy_from_slice(&JHDR_SIZE.to_be_bytes());
    let csum = crc32_reflected(&b);
    b[36..40].copy_from_slice(&csum.to_be_bytes());
    dev.write_at(buf_off, &b)?;
    Ok(())
}

/// CRC-32 (reflected poly 0xEDB88320, init 0xFFFFFFFF, no final XOR).
/// Matches Apple's journal-header / block-list-header checksum.
fn crc32_reflected(buf: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in buf {
        let mut c = (crc ^ u32::from(byte)) & 0xff;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
        }
        crc = (crc >> 8) ^ c;
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    /// Build a minimal in-memory journal buffer (no surrounding HFS+
    /// volume) and verify a single transaction round-trips: commit
    /// records the data, advances start = end, and the target block
    /// has the expected bytes.
    #[test]
    fn journal_commit_writes_data_and_advances_start() {
        let mut dev = MemoryBackend::new(64 * 1024);
        let buf_off: u64 = 4096;
        let buf_size: u64 = 16 * 1024;
        write_journal_header(
            &mut dev,
            buf_off,
            u64::from(JHDR_SIZE),
            u64::from(JHDR_SIZE),
            buf_size,
        )
        .unwrap();
        let mut log = JournalLog {
            buf_off,
            buf_size,
            start: u64::from(JHDR_SIZE),
            end: u64::from(JHDR_SIZE),
            pending: Vec::new(),
        };
        log.add(32 * 1024, vec![0xAB; 512]);
        log.commit(&mut dev).unwrap();
        let mut got = [0u8; 512];
        dev.read_at(32 * 1024, &mut got).unwrap();
        assert!(got.iter().all(|&b| b == 0xAB));
        assert_eq!(log.start, log.end);
    }

    /// A two-step crash simulation: write a transaction (advancing
    /// `end` and applying the writes) but stop before advancing `start`,
    /// then verify replay re-applies the writes idempotently.
    #[test]
    fn replay_reapplies_pending_transaction() {
        let mut dev = MemoryBackend::new(128 * 1024);
        let buf_off: u64 = 8192;
        let buf_size: u64 = 16 * 1024;
        write_journal_header(
            &mut dev,
            buf_off,
            u64::from(JHDR_SIZE),
            u64::from(JHDR_SIZE),
            buf_size,
        )
        .unwrap();
        let mut log = JournalLog {
            buf_off,
            buf_size,
            start: u64::from(JHDR_SIZE),
            end: u64::from(JHDR_SIZE),
            pending: Vec::new(),
        };
        log.add(32 * 1024, vec![0xCD; 512]);
        log.commit(&mut dev).unwrap();
        // Now corrupt the target block so we can verify replay
        // restores it, and rewind `start` so the journal looks dirty.
        dev.write_at(32 * 1024, &[0u8; 512]).unwrap();
        write_journal_header(&mut dev, buf_off, u64::from(JHDR_SIZE), log.end, buf_size).unwrap();
        // Replay needs a VolumeHeader to look up the JIB. Build a
        // minimal one — only the fields replay actually consults
        // (`attributes`, `block_size`, `journal_info_block`) need to
        // be meaningful.
        use crate::fs::hfs_plus::volume_header::{
            ExtentDescriptor, FORK_EXTENT_COUNT, ForkData, VolumeHeader,
        };
        let blank_fork = ForkData {
            logical_size: 0,
            clump_size: 0,
            total_blocks: 0,
            extents: [ExtentDescriptor::default(); FORK_EXTENT_COUNT],
        };
        let vh = VolumeHeader {
            signature: *b"H+",
            version: 4,
            attributes: VOL_ATTR_JOURNALED,
            journal_info_block: 1, // points at block 1; we'll stamp it
            block_size: 4096,
            total_blocks: 16,
            free_blocks: 0,
            next_catalog_id: 16,
            allocation_file: blank_fork,
            extents_file: blank_fork,
            catalog_file: blank_fork,
            attributes_file: blank_fork,
            startup_file: blank_fork,
        };
        let info_off = u64::from(vh.journal_info_block) * u64::from(vh.block_size);
        let mut info = [0u8; 52];
        info[0..4].copy_from_slice(&2u32.to_be_bytes());
        info[36..44].copy_from_slice(&buf_off.to_be_bytes());
        info[44..52].copy_from_slice(&buf_size.to_be_bytes());
        dev.write_at(info_off, &info).unwrap();

        replay(&mut dev, &vh).unwrap();
        let mut got = [0u8; 512];
        dev.read_at(32 * 1024, &mut got).unwrap();
        assert!(got.iter().all(|&b| b == 0xCD), "replay restored data");
    }
}
