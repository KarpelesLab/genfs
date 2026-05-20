//! HFS+ in-place read/write file handle.
//!
//! Backs [`crate::fs::Filesystem::open_file_rw`] for HFS+ volumes.
//! Every `Write::write` is eager: the relevant allocation block(s) are
//! read, patched, written back, and the catalog file record's data
//! fork is updated in the in-memory writer. The actual catalog B-tree
//! rewrite happens on `sync()` (or `Drop`), which delegates to
//! [`super::HfsPlus::flush`].
//!
//! ## Journal handling — Path B (clean-shutdown bypass)
//!
//! On open, the journal header is read (if the volume carries one) and
//! verified clean (`start == end`). All mutations are then performed
//! in-place, bypassing the journal entirely. After flush, the journal
//! is still clean — the kernel/fsck will conclude there is nothing to
//! replay. If the journal is dirty on open, we refuse with
//! [`crate::Error::Unsupported`]: the caller should mount the image
//! with a tool that replays first.
//!
//! ## Lifetime
//!
//! The handle holds `&'a mut HfsPlus` and `&'a mut dyn BlockDevice`
//! for its full lifetime, matching the trait signature.

use std::io::{self, Read, Seek, SeekFrom, Write};

use super::HfsPlus;
use super::catalog::{REC_FILE, UniStr, mode};
use super::extents::FORK_DATA;
use super::volume_header::{ExtentDescriptor, FORK_DATA_SIZE, FORK_EXTENT_COUNT, ForkData};
use super::writer::OwnedKey;
use crate::Result;
use crate::block::BlockDevice;
use crate::fs::FileHandle;

/// A read/write file handle on an HFS+ regular file. Reads / writes
/// translate fork-relative byte offsets into allocation-block I/O via
/// the file's extent list (inline + extents-overflow). Growth allocates
/// new runs from the volume bitmap.
pub struct HfsPlusFileHandle<'a> {
    fs: &'a mut HfsPlus,
    dev: &'a mut dyn BlockDevice,
    /// OwnedKey of the catalog entry that owns this file. Used to look
    /// the body up in the in-memory writer on every metadata refresh.
    cat_key: OwnedKey,
    /// CNID of the file we're editing. Needed to drive overflow-record
    /// updates.
    file_id: u32,
    /// The file's allocation runs, in fork order. Inline (first 8) plus
    /// any spilled into extents-overflow are merged into a single flat
    /// list for simple addressing during reads / writes.
    runs: Vec<ExtentDescriptor>,
    /// Logical length in bytes (after any pending writes).
    file_size: u64,
    /// Read / write cursor.
    pos: u64,
    /// True once mutations have made the on-disk catalog stale. Cleared
    /// on `sync` / `Drop`.
    dirty: bool,
}

impl<'a> HfsPlusFileHandle<'a> {
    /// Construct a handle for the file at `cat_key`. Caller has already
    /// resolved the entry to a non-hardlink non-symlink regular file
    /// and assembled its merged run list.
    pub(super) fn new(
        fs: &'a mut HfsPlus,
        dev: &'a mut dyn BlockDevice,
        cat_key: OwnedKey,
        file_id: u32,
        runs: Vec<ExtentDescriptor>,
        file_size: u64,
    ) -> Self {
        Self {
            fs,
            dev,
            cat_key,
            file_id,
            runs,
            file_size,
            pos: 0,
            dirty: false,
        }
    }

    /// Allocation block size in bytes.
    fn block_size(&self) -> u64 {
        u64::from(self.fs.volume_header.block_size)
    }

    /// Translate a fork-relative byte offset to (device byte offset,
    /// bytes available in this run). Returns `None` past EOF (i.e. past
    /// the last allocated block).
    fn locate(&self, fork_off: u64) -> Option<(u64, u64)> {
        let bs = self.block_size();
        let mut cursor: u64 = 0;
        for run in &self.runs {
            let run_bytes = u64::from(run.block_count) * bs;
            if fork_off < cursor + run_bytes {
                let into_run = fork_off - cursor;
                let dev_off = u64::from(run.start_block) * bs + into_run;
                let remaining = run_bytes - into_run;
                return Some((dev_off, remaining));
            }
            cursor += run_bytes;
        }
        None
    }

    /// Total bytes covered by the current run list.
    fn allocated_bytes(&self) -> u64 {
        let bs = self.block_size();
        self.runs
            .iter()
            .map(|r| u64::from(r.block_count) * bs)
            .sum()
    }

    /// Total allocation blocks across the run list.
    fn total_blocks(&self) -> u32 {
        self.runs
            .iter()
            .map(|r| r.block_count)
            .fold(0u32, |a, b| a.saturating_add(b))
    }

    /// Read up to `buf.len()` bytes from the current cursor; never past EOF.
    fn read_inner(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.file_size || buf.is_empty() {
            return Ok(0);
        }
        let remaining_in_file = self.file_size - self.pos;
        let want = (buf.len() as u64).min(remaining_in_file) as usize;
        let mut done: usize = 0;
        while done < want {
            let (dev_off, in_run) = self.locate(self.pos + done as u64).ok_or_else(|| {
                io::Error::other(format!(
                    "hfs+: file_size {} extends past allocated runs",
                    self.file_size
                ))
            })?;
            let chunk = (want - done).min(in_run as usize);
            self.dev
                .read_at(dev_off, &mut buf[done..done + chunk])
                .map_err(io::Error::other)?;
            done += chunk;
        }
        self.pos += done as u64;
        Ok(done)
    }

    /// Append new runs to cover at least `needed_bytes` bytes total
    /// across the run list. Newly-allocated blocks are zeroed on disk
    /// so reads of unwritten holes return zeros.
    fn ensure_capacity(&mut self, needed_bytes: u64) -> Result<()> {
        let bs = self.block_size();
        let have_bytes = self.allocated_bytes();
        if needed_bytes <= have_bytes {
            return Ok(());
        }
        let extra_bytes = needed_bytes - have_bytes;
        let mut extra_blocks = u32::try_from(extra_bytes.div_ceil(bs)).map_err(|_| {
            crate::Error::InvalidArgument(
                "hfs+ handle: capacity grow overflows u32 block count".into(),
            )
        })?;
        let writer = self.fs.writer.as_mut().ok_or_else(|| {
            crate::Error::InvalidArgument("hfs+: volume opened read-only".into())
        })?;
        while extra_blocks > 0 {
            let run = writer.allocate_largest_run(extra_blocks)?;
            // Zero the freshly-allocated run on disk so reads of holes
            // return zero. (A block freed by `remove` and re-handed-out
            // may still carry stale bytes.)
            let zero = vec![0u8; (u64::from(run.block_count) * bs) as usize];
            let off = u64::from(run.start_block) * bs;
            self.dev.write_at(off, &zero)?;
            extra_blocks -= run.block_count;
            self.runs.push(run);
        }
        Ok(())
    }

    /// Shrink the run list so it covers no more than `cap_bytes` bytes,
    /// freeing the trailing blocks. The blocks-actually-needed count is
    /// always rounded up so a partial-block tail stays allocated.
    fn shrink_to(&mut self, cap_bytes: u64) -> Result<()> {
        let bs = self.block_size();
        let needed_blocks_u64 = cap_bytes.div_ceil(bs);
        let needed_blocks = u32::try_from(needed_blocks_u64).map_err(|_| {
            crate::Error::InvalidArgument("hfs+ handle: shrink-target overflows u32 blocks".into())
        })?;
        let mut have: u32 = self.total_blocks();
        let writer = self.fs.writer.as_mut().ok_or_else(|| {
            crate::Error::InvalidArgument("hfs+: volume opened read-only".into())
        })?;
        while have > needed_blocks {
            let last = self
                .runs
                .last_mut()
                .ok_or_else(|| crate::Error::InvalidImage("hfs+ handle: run list empty".into()))?;
            let surplus = have - needed_blocks;
            if last.block_count <= surplus {
                let dead = *last;
                writer.free(dead.start_block, dead.block_count);
                have -= dead.block_count;
                self.runs.pop();
            } else {
                // Trim the tail of this run.
                let new_count = last.block_count - surplus;
                let freed_start = last.start_block + new_count;
                writer.free(freed_start, surplus);
                last.block_count = new_count;
                have = needed_blocks;
            }
        }
        Ok(())
    }

    /// Write `data` at fork-relative offset `off`. The runs must already
    /// cover `off + data.len()` bytes.
    fn write_into_runs(&mut self, off: u64, data: &[u8]) -> io::Result<()> {
        let mut done = 0usize;
        while done < data.len() {
            let (dev_off, in_run) = self.locate(off + done as u64).ok_or_else(|| {
                io::Error::other("hfs+: write past allocated capacity (internal)")
            })?;
            let chunk = (data.len() - done).min(in_run as usize);
            self.dev
                .write_at(dev_off, &data[done..done + chunk])
                .map_err(io::Error::other)?;
            done += chunk;
        }
        Ok(())
    }

    /// Zero the bytes in `[start, end)`. The runs must already cover
    /// that range.
    fn zero_range(&mut self, start: u64, end: u64) -> io::Result<()> {
        if end <= start {
            return Ok(());
        }
        const Z: [u8; 4096] = [0u8; 4096];
        let mut pos = start;
        while pos < end {
            let want = (end - pos).min(Z.len() as u64) as usize;
            self.write_into_runs(pos, &Z[..want])?;
            pos += want as u64;
        }
        Ok(())
    }

    /// Internal write path. Extends the runs as needed, zeros any gap
    /// between old EOF and the cursor, writes the bytes, updates the
    /// in-memory size and marks dirty.
    fn write_inner(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let new_end = self.pos + buf.len() as u64;
        let gap_start = self.file_size;
        let gap_end = self.pos.min(new_end);
        self.ensure_capacity(new_end)
            .map_err(|e| io::Error::other(e.to_string()))?;
        if gap_end > gap_start {
            self.zero_range(gap_start, gap_end)?;
        }
        self.write_into_runs(self.pos, buf)?;
        self.pos += buf.len() as u64;
        if self.pos > self.file_size {
            self.file_size = self.pos;
        }
        self.dirty = true;
        Ok(buf.len())
    }

    /// Grow or shrink the file to exactly `new_len` bytes. Marks dirty.
    fn set_len_inner(&mut self, new_len: u64) -> Result<()> {
        if new_len > self.file_size {
            let old_len = self.file_size;
            self.ensure_capacity(new_len)?;
            self.file_size = new_len;
            self.zero_range(old_len, new_len).map_err(crate::Error::Io)?;
        } else if new_len < self.file_size {
            self.shrink_to(new_len)?;
            self.file_size = new_len;
            if self.pos > self.file_size {
                self.pos = self.file_size;
            }
        }
        self.dirty = true;
        Ok(())
    }

    /// Re-encode the catalog file body to reflect the current run list
    /// and `file_size`. Inline extents go straight into the file
    /// record's data fork; spilled extents go into
    /// `writer.overflow_extents` keyed by `(FORK_DATA, file_id, ...)`.
    fn refresh_catalog_body(&mut self) -> Result<()> {
        // Build the new inline-fork descriptor up front so we don't
        // hold both an immutable view of `self.runs` and a mutable
        // borrow of `self.fs.writer` simultaneously.
        let mut inline = [ExtentDescriptor::default(); FORK_EXTENT_COUNT];
        let inline_count = self.runs.len().min(FORK_EXTENT_COUNT);
        for (slot, ext) in inline.iter_mut().zip(self.runs.iter().take(inline_count)) {
            *slot = *ext;
        }
        let total_blocks = self.total_blocks();
        let logical_size = self.file_size;
        let file_id = self.file_id;
        let extra_runs: Vec<ExtentDescriptor> = if self.runs.len() > FORK_EXTENT_COUNT {
            self.runs[FORK_EXTENT_COUNT..].to_vec()
        } else {
            Vec::new()
        };
        let writer = self.fs.writer.as_mut().ok_or_else(|| {
            crate::Error::InvalidArgument("hfs+: volume opened read-only".into())
        })?;
        let new_fork = ForkData {
            logical_size,
            clump_size: writer.block_size,
            total_blocks,
            extents: inline,
        };

        // 1. Patch the file record's data-fork bytes inside the
        //    encoded body. Data fork starts at offset 88, length
        //    FORK_DATA_SIZE (= 80).
        let body = writer
            .catalog
            .get_mut(&self.cat_key)
            .ok_or_else(|| crate::Error::InvalidImage("hfs+ handle: catalog entry vanished".into()))?;
        if body.len() < 88 + FORK_DATA_SIZE {
            return Err(crate::Error::InvalidImage(
                "hfs+ handle: short catalog file body".into(),
            ));
        }
        if body.len() < 2 || i16::from_be_bytes([body[0], body[1]]) != REC_FILE {
            return Err(crate::Error::InvalidImage(
                "hfs+ handle: catalog body is not a file record".into(),
            ));
        }
        let enc = encode_fork_array(&new_fork);
        body[88..88 + FORK_DATA_SIZE].copy_from_slice(&enc);

        // 2. Drop any existing overflow records for this file's data
        //    fork; we'll re-emit them from the current run list.
        let keys: Vec<(u8, u32, u32)> = writer
            .overflow_extents
            .range((FORK_DATA, file_id, 0)..=(FORK_DATA, file_id, u32::MAX))
            .map(|(k, _)| *k)
            .collect();
        for k in keys {
            writer.overflow_extents.remove(&k);
        }

        // 3. If we have more than 8 runs, lay the remainder into
        //    extents-overflow records (8 per record), keyed by the
        //    fork-block where each record starts.
        if !extra_runs.is_empty() {
            let mut start_block: u32 = inline
                .iter()
                .map(|e| e.block_count)
                .fold(0u32, |a, b| a.saturating_add(b));
            for chunk in extra_runs.chunks(FORK_EXTENT_COUNT) {
                let mut group = [ExtentDescriptor::default(); FORK_EXTENT_COUNT];
                for (slot, ext) in group.iter_mut().zip(chunk.iter()) {
                    *slot = *ext;
                }
                writer
                    .overflow_extents
                    .insert((FORK_DATA, file_id, start_block), group);
                for ext in chunk {
                    start_block = start_block.saturating_add(ext.block_count);
                }
            }
        }
        Ok(())
    }

    /// Persist this handle's changes. Updates the catalog body bytes,
    /// then calls `HfsPlus::flush` to rewrite the catalog tree, the
    /// extents-overflow tree, the allocation bitmap, and the volume
    /// header. The journal stub (if present) is left untouched — it's
    /// still clean because all mutations went straight to the live
    /// metadata.
    fn sync_inner(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        self.refresh_catalog_body()?;
        self.fs.flush(self.dev)?;
        self.dirty = false;
        Ok(())
    }
}

impl<'a> Read for HfsPlusFileHandle<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_inner(buf)
    }
}

impl<'a> Write for HfsPlusFileHandle<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write_inner(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.sync_inner().map_err(|e| io::Error::other(e.to_string()))
    }
}

impl<'a> Seek for HfsPlusFileHandle<'a> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos: i128 = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(d) => self.file_size as i128 + d as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
        };
        if new_pos < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "hfs+: seek to negative offset",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

impl<'a> FileHandle for HfsPlusFileHandle<'a> {
    fn len(&self) -> u64 {
        self.file_size
    }

    fn set_len(&mut self, new_len: u64) -> Result<()> {
        self.set_len_inner(new_len)
    }

    fn sync(&mut self) -> Result<()> {
        self.sync_inner()
    }
}

impl<'a> Drop for HfsPlusFileHandle<'a> {
    fn drop(&mut self) {
        // Best-effort: persist on drop so the file is durable even when
        // the caller forgets to `sync`. Drop can't return errors, so we
        // swallow them. Tests should `sync()` explicitly to surface I/O
        // failures.
        if self.dirty {
            let _ = self.refresh_catalog_body();
            let _ = self.fs.flush(self.dev);
            self.dirty = false;
        }
    }
}

/// Encode an `HFSPlusForkData` (80 bytes) into a byte array. Mirrors
/// the writer's `fork_to_array` helper (kept private there) — duplicated
/// here so the handle stays decoupled from writer internals.
fn encode_fork_array(fork: &ForkData) -> [u8; FORK_DATA_SIZE] {
    let mut out = [0u8; FORK_DATA_SIZE];
    out[0..8].copy_from_slice(&fork.logical_size.to_be_bytes());
    out[8..12].copy_from_slice(&fork.clump_size.to_be_bytes());
    out[12..16].copy_from_slice(&fork.total_blocks.to_be_bytes());
    for (i, ext) in fork.extents.iter().enumerate() {
        let off = 16 + i * 8;
        out[off..off + 4].copy_from_slice(&ext.start_block.to_be_bytes());
        out[off + 4..off + 8].copy_from_slice(&ext.block_count.to_be_bytes());
    }
    out
}

/// Open `path` for read+write, honoring the requested flags. Refuses
/// (with [`crate::Error::Unsupported`]) on a journaled volume whose
/// journal still has work to replay — the caller should mount the image
/// with a tool that performs the replay first.
pub(super) fn open_file_rw<'a>(
    fs: &'a mut HfsPlus,
    dev: &'a mut dyn BlockDevice,
    path: &std::path::Path,
    flags: crate::fs::OpenFlags,
    meta: Option<crate::fs::FileMeta>,
) -> Result<Box<dyn FileHandle + 'a>> {
    let path_str = path
        .to_str()
        .ok_or_else(|| crate::Error::InvalidArgument(format!("hfs+: non-UTF-8 path {path:?}")))?;

    // Path-B journal check: if the volume is journaled, the on-disk
    // journal MUST be clean (start == end). Otherwise we'd be writing
    // into a volume that still has pending transactions to replay,
    // which would corrupt the next-replay outcome.
    ensure_journal_clean(fs, dev)?;

    // Resolve to an existing file, optionally creating it. We can't
    // borrow `fs` through the lookup machinery while we also need a
    // mutable borrow for create, so do them in distinct phases.
    let resolved = lookup_file_for_rw(fs, path_str);
    let (parent_id, name) = split_parent_and_name(fs, path_str)?;
    let owned_key = OwnedKey {
        parent_id,
        name: name.clone(),
    };

    let resolved = match resolved {
        Ok(()) => Some(()),
        Err(_) if flags.create => None,
        Err(e) => return Err(e),
    };

    if resolved.is_none() {
        // create=true path. Insert an empty regular file via the
        // existing writer machinery.
        let meta = meta.ok_or_else(|| {
            crate::Error::InvalidArgument(
                "hfs+: open_file_rw with create=true requires meta".into(),
            )
        })?;
        let mut empty = std::io::Cursor::new(Vec::<u8>::new());
        fs.create_file(dev, path_str, &mut empty, 0, meta.mode, meta.uid, meta.gid)?;
    }

    // Read the catalog body and assemble the merged run list.
    let (file_id, file_size, runs) = {
        let writer = fs.writer.as_ref().ok_or_else(|| {
            crate::Error::InvalidArgument("hfs+: volume opened read-only".into())
        })?;
        let body = writer.catalog.get(&owned_key).ok_or_else(|| {
            crate::Error::InvalidArgument(format!(
                "hfs+: no entry {:?} under CNID {parent_id}",
                name.to_string_lossy()
            ))
        })?;
        if body.len() < 88 + FORK_DATA_SIZE
            || i16::from_be_bytes([body[0], body[1]]) != REC_FILE
        {
            return Err(crate::Error::InvalidArgument(format!(
                "hfs+: {path_str} is not a regular file"
            )));
        }
        // Verify the file is a plain regular (not hlnk, not slnk).
        let file_type = &body[48..52];
        let creator = &body[52..56];
        if file_type == b"hlnk" && creator == b"hfs+" {
            return Err(crate::Error::Unsupported(format!(
                "hfs+: {path_str} is a hard link; open_file_rw on the iNode is not implemented"
            )));
        }
        if file_type == b"slnk" && creator == b"rhap" {
            return Err(crate::Error::InvalidArgument(format!(
                "hfs+: {path_str} is a symlink, not a regular file"
            )));
        }
        // mode_bits: file_mode lives inside BSDInfo at body[32..48], with
        // file_mode at byte 42..44.
        let file_mode = u16::from_be_bytes([body[42], body[43]]);
        let mode_bits = file_mode & mode::S_IFMT;
        if mode_bits != 0 && mode_bits != mode::S_IFREG {
            return Err(crate::Error::Unsupported(format!(
                "hfs+: {path_str} is not a regular file (mode {file_mode:#06o})"
            )));
        }
        let file_id = u32::from_be_bytes(body[8..12].try_into().unwrap());

        let mut fork_buf = [0u8; FORK_DATA_SIZE];
        fork_buf.copy_from_slice(&body[88..88 + FORK_DATA_SIZE]);
        let fork = ForkData::decode(&fork_buf);

        // Merged run list: inline extents first, then overflow groups
        // in fork-block order.
        let mut runs: Vec<ExtentDescriptor> = Vec::new();
        for ext in &fork.extents {
            if ext.block_count == 0 {
                continue;
            }
            runs.push(*ext);
        }
        for ((fork_type, fid, _start), group) in writer
            .overflow_extents
            .range((FORK_DATA, file_id, 0)..=(FORK_DATA, file_id, u32::MAX))
        {
            debug_assert_eq!((*fork_type, *fid), (FORK_DATA, file_id));
            for ext in group {
                if ext.block_count == 0 {
                    continue;
                }
                runs.push(*ext);
            }
        }

        (file_id, fork.logical_size, runs)
    };

    let mut handle = HfsPlusFileHandle::new(fs, dev, owned_key, file_id, runs, file_size);
    if flags.truncate && handle.file_size > 0 {
        handle.set_len_inner(0)?;
    }
    if flags.append {
        handle.pos = handle.file_size;
    }
    Ok(Box::new(handle))
}

/// Verify the volume's journal (if any) reports zero pending work.
/// Returns Ok on unjournaled volumes and on journaled volumes whose
/// `start == end`. Returns [`crate::Error::Unsupported`] otherwise.
fn ensure_journal_clean(fs: &HfsPlus, dev: &mut dyn BlockDevice) -> Result<()> {
    use super::writer::{JOURNAL_HEADER_ENDIAN, JOURNAL_HEADER_MAGIC, VOL_ATTR_JOURNALED};

    if fs.volume_header.attributes & VOL_ATTR_JOURNALED == 0 {
        return Ok(());
    }
    let info_block = fs.volume_header.journal_info_block;
    if info_block == 0 {
        return Err(crate::Error::InvalidImage(
            "hfs+: volume has the journaled bit set but no JournalInfoBlock".into(),
        ));
    }
    let bs = u64::from(fs.volume_header.block_size);
    let info_off = u64::from(info_block) * bs;
    // The JournalInfoBlock fits in a single allocation block; we only
    // need the first 52 bytes (flags + device_signature + offset + size).
    let mut info = [0u8; 52];
    dev.read_at(info_off, &mut info)?;
    let jbuf_off = u64::from_be_bytes(info[36..44].try_into().unwrap());
    let jbuf_size = u64::from_be_bytes(info[44..52].try_into().unwrap());
    if jbuf_off == 0 || jbuf_size == 0 {
        return Err(crate::Error::InvalidImage(
            "hfs+: JournalInfoBlock has zero buffer offset or size".into(),
        ));
    }

    // The journal header lives at the start of the journal buffer. We
    // only need the first 24 bytes: magic + endian + start + end.
    let mut hdr = [0u8; 24];
    dev.read_at(jbuf_off, &mut hdr)?;
    let magic = u32::from_be_bytes(hdr[0..4].try_into().unwrap());
    let endian = u32::from_be_bytes(hdr[4..8].try_into().unwrap());
    if magic != JOURNAL_HEADER_MAGIC || endian != JOURNAL_HEADER_ENDIAN {
        // Could be a native-endian journal header from a tool that
        // wrote it in the host's byte order. Try LE.
        let magic_le = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        let endian_le = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        if magic_le != JOURNAL_HEADER_MAGIC || endian_le != JOURNAL_HEADER_ENDIAN {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: journal header has unrecognised magic/endian \
                 ({magic:#010x}/{endian:#010x})"
            )));
        }
        // Native-LE encoding: start, end are at the same offsets but
        // little-endian.
        let start = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
        let end = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
        if start != end {
            return Err(crate::Error::Unsupported(format!(
                "hfs+: journal has unreplayed work (start={start:#x} != end={end:#x}); \
                 mount and unmount with a journal-aware tool first"
            )));
        }
        return Ok(());
    }
    let start = u64::from_be_bytes(hdr[8..16].try_into().unwrap());
    let end = u64::from_be_bytes(hdr[16..24].try_into().unwrap());
    if start != end {
        return Err(crate::Error::Unsupported(format!(
            "hfs+: journal has unreplayed work (start={start:#x} != end={end:#x}); \
             mount and unmount with a journal-aware tool first"
        )));
    }
    Ok(())
}

/// Probe the writer's in-memory catalog for a regular file at
/// `path_str`. Returns Ok only when the entry exists.
fn lookup_file_for_rw(fs: &HfsPlus, path_str: &str) -> Result<()> {
    let (parent_id, name) = split_parent_and_name(fs, path_str)?;
    let writer = fs
        .writer
        .as_ref()
        .ok_or_else(|| crate::Error::InvalidArgument("hfs+: volume opened read-only".into()))?;
    let key = OwnedKey {
        parent_id,
        name: name.clone(),
    };
    if writer.catalog.contains_key(&key) {
        Ok(())
    } else {
        Err(crate::Error::InvalidArgument(format!(
            "hfs+: no entry {:?} under CNID {parent_id}",
            name.to_string_lossy()
        )))
    }
}

/// Walk `path_str` and return the parent CNID + final-name component,
/// resolving every intermediate against the writer's in-memory catalog.
fn split_parent_and_name(fs: &HfsPlus, path_str: &str) -> Result<(u32, UniStr)> {
    let parts: Vec<&str> = path_str
        .split('/')
        .filter(|p| !p.is_empty() && *p != ".")
        .collect();
    if parts.is_empty() {
        return Err(crate::Error::InvalidArgument(
            "hfs+: cannot open the root path".into(),
        ));
    }
    let (last, prefix) = parts.split_last().unwrap();
    let writer = fs
        .writer
        .as_ref()
        .ok_or_else(|| crate::Error::InvalidArgument("hfs+: volume opened read-only".into()))?;
    let mut cnid = super::catalog::ROOT_FOLDER_ID;
    for part in prefix {
        let name = UniStr::from_str_lossy(part);
        let (_, child_cnid, rec_type) = writer.lookup(cnid, &name).ok_or_else(|| {
            crate::Error::InvalidArgument(format!(
                "hfs+: parent component {part:?} does not exist"
            ))
        })?;
        if rec_type != super::catalog::REC_FOLDER {
            return Err(crate::Error::InvalidArgument(format!(
                "hfs+: component {part:?} is not a directory"
            )));
        }
        cnid = child_cnid;
    }
    Ok((cnid, UniStr::from_str_lossy(last)))
}
