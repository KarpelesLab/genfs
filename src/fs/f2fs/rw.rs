//! F2FS in-place file handle — [`crate::fs::FileHandle`] for
//! `F2fs::open_file_rw`.
//!
//! ## Log-structured write semantics
//!
//! F2FS never overwrites a data block — every modified block is
//! reallocated. The handle therefore buffers writes in memory and
//! commits them lazily:
//!
//! - `Write::write` updates an in-memory `dirty` map keyed by logical
//!   block index. No device I/O.
//! - `Read::read` consults the dirty map first; for any byte not in the
//!   buffer, it walks the writer's in-memory pointer tree (post-
//!   `place_data_block` placements visible there) and falls back to
//!   the on-disk block.
//! - `sync()` walks every dirty block, allocates a fresh data block
//!   via `write::Writer::alloc_data_block`, writes the merged
//!   bytes there, repoints the inode's pointer tree at the new block,
//!   and emits a fresh checkpoint by calling [`super::F2fs::flush`].
//!   The old block address is simply abandoned (the next checkpoint
//!   doesn't list it, matching log-structured invalidation).
//! - `Drop` calls `sync()` and ignores its result — best effort, as
//!   `Drop` can't surface errors.
//!
//! ## Inline-data fast path
//!
//! Files whose total length stays at or below
//! [`super::write::MAX_INLINE_DATA`] live entirely in the inode's
//! literal area (the `F2FS_INLINE_DATA` bit). The handle special-cases
//! this layout: reads/writes mutate `InodeRec::inline_payload`
//! directly, and `sync()` skips the data-block allocator. If a write
//! grows the file past `MAX_INLINE_DATA`, we transparently "de-inline":
//! clear the bit, drop the literal payload into block 0, and continue
//! as a regular block-stored file.
//!
//! ## Constraints
//!
//! This module only works on a handle obtained through
//! [`super::F2fs::format`] — i.e. one with a live `Writer`. Re-opened
//! images (`F2fs::open`) don't have the writer state needed to allocate
//! new blocks or re-emit a checkpoint, so `open_file_rw` rejects them
//! with `Unsupported`.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::{FileHandle, FileMeta, OpenFlags};

use super::F2fs;
use super::constants::{F2FS_BLKSIZE, F2FS_DATA_EXIST, F2FS_INLINE_DATA, NEW_ADDR, NULL_ADDR};
use super::write::MAX_INLINE_DATA;

/// One full 4 KiB dirty block (logical block index → bytes).
type DirtyMap = BTreeMap<u64, Vec<u8>>;

/// In-place read+write handle for a single regular file. Holds mutable
/// borrows of the FS state and block device. Pending writes are kept
/// in `dirty`; `sync()` allocates fresh data blocks and emits a fresh
/// checkpoint via the underlying writer.
pub struct F2fsFileHandle<'a> {
    fs: &'a mut F2fs,
    dev: &'a mut dyn BlockDevice,
    nid: u32,
    /// Logical size of the file. Reflects any `set_len` / `Write::write`
    /// past EOF — the on-disk inode picks this value up at `sync()`.
    size: u64,
    /// Cursor position for the `Read + Write + Seek` impls.
    pos: u64,
    /// Per-block dirty buffers. Sparse — keys are logical block indices,
    /// values are 4 KiB `Vec<u8>` snapshots of the block-after-write
    /// (read-modify-write semantics on the first touch). Bytes past
    /// the file's current size in a dirty block are zeroed at sync()
    /// time.
    dirty: DirtyMap,
    /// True if the file is currently stored inline (`F2FS_INLINE_DATA`).
    /// Updated lazily — `de_inline_if_needed()` flips this when a write
    /// pushes the file past `MAX_INLINE_DATA`.
    inline: bool,
    /// Working copy of the inline payload. Mirrors
    /// `InodeRec::inline_payload` while `inline == true`; flushed back
    /// to the inode in `sync()`.
    inline_buf: Vec<u8>,
    /// Set on `sync()` success; prevents `Drop` from double-flushing.
    synced: bool,
}

impl<'a> F2fsFileHandle<'a> {
    /// Open `path` for in-place reads + writes.
    ///
    /// This is the implementation backing
    /// [`crate::fs::Filesystem::open_file_rw`] for F2FS.
    pub(crate) fn open(
        fs: &'a mut F2fs,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
        flags: OpenFlags,
        meta: Option<FileMeta>,
    ) -> Result<Self> {
        // Writer must be live — open_file_rw is fresh-image only.
        if fs.writer.is_none() {
            return Err(crate::Error::Unsupported(
                "f2fs: open_file_rw requires a writable handle (use F2fs::format)".into(),
            ));
        }

        // Try to resolve via the writer's in-memory tree (covers
        // anything added in this session). Fall back to creating
        // a fresh file if `flags.create` is set and resolution fails.
        let existing_nid = {
            let w = fs.writer.as_ref().expect("checked above");
            resolve_in_writer(w, path)?
        };

        let nid = match existing_nid {
            Some(n) => n,
            None => {
                if !flags.create {
                    return Err(crate::Error::InvalidArgument(format!(
                        "f2fs: open_file_rw: not found: {path:?}"
                    )));
                }
                let m = meta.ok_or_else(|| {
                    crate::Error::InvalidArgument(
                        "f2fs: open_file_rw create=true requires meta".into(),
                    )
                })?;
                // Create as empty file via the writer's normal path.
                fs.writer_mut()?
                    .add_file(dev, path, crate::fs::FileSource::Zero(0), m)?
            }
        };

        let (inline, inline_buf, mut size) = {
            let w = fs.writer.as_ref().expect("writer live");
            let ino = w
                .inodes
                .get(&nid)
                .ok_or_else(|| crate::Error::InvalidImage(format!("f2fs: ghost nid {nid}")))?;
            let inline = ino.inline_flags & F2FS_INLINE_DATA != 0;
            (inline, ino.inline_payload.clone(), ino.size)
        };

        let mut me = Self {
            fs,
            dev,
            nid,
            size,
            pos: 0,
            dirty: BTreeMap::new(),
            inline,
            inline_buf,
            synced: false,
        };

        if flags.truncate {
            me.set_len_internal(0)?;
            size = 0;
        }
        if flags.append {
            me.pos = size;
        }

        Ok(me)
    }

    /// Read up to `out.len()` bytes from `pos` into `out`. Honours the
    /// dirty buffer first, then the writer's in-memory pointer tree,
    /// then the on-disk block.
    fn read_at(&mut self, pos: u64, out: &mut [u8]) -> std::io::Result<usize> {
        if pos >= self.size {
            return Ok(0);
        }
        let bs = F2FS_BLKSIZE as u64;
        let remaining_in_file = (self.size - pos) as usize;
        let n_want = out.len().min(remaining_in_file);
        if n_want == 0 {
            return Ok(0);
        }

        // Inline path: serve directly from inline_buf (with implicit
        // zero-padding past the buffer length).
        if self.inline {
            let start = pos as usize;
            let end = (start + n_want).min(self.inline_buf.len());
            if start >= end {
                // All requested bytes are past the materialised payload.
                out[..n_want].fill(0);
                return Ok(n_want);
            }
            let n_copy = end - start;
            out[..n_copy].copy_from_slice(&self.inline_buf[start..end]);
            if n_copy < n_want {
                out[n_copy..n_want].fill(0);
            }
            return Ok(n_want);
        }

        // Block path: walk one block, possibly less than n_want.
        let block_idx = pos / bs;
        let in_block_off = (pos % bs) as usize;
        let n = n_want.min(F2FS_BLKSIZE - in_block_off);

        let block_bytes = self.fetch_block(block_idx).map_err(std::io::Error::other)?;
        out[..n].copy_from_slice(&block_bytes[in_block_off..in_block_off + n]);
        Ok(n)
    }

    /// Resolve the 4 KiB block at `block_idx` — checking the dirty
    /// buffer first, then the writer's in-memory tree, then disk.
    fn fetch_block(&mut self, block_idx: u64) -> Result<Vec<u8>> {
        if let Some(b) = self.dirty.get(&block_idx) {
            return Ok(b.clone());
        }
        self.read_block_on_disk(block_idx)
    }

    fn read_block_on_disk(&mut self, block_idx: u64) -> Result<Vec<u8>> {
        let w = self.fs.writer.as_ref().expect("writer live");
        let phys = w.current_data_block(self.nid, block_idx)?;
        let bs = F2FS_BLKSIZE as u64;
        if phys == NULL_ADDR || phys == NEW_ADDR {
            return Ok(vec![0u8; F2FS_BLKSIZE]);
        }
        let mut buf = vec![0u8; F2FS_BLKSIZE];
        self.dev.read_at(phys as u64 * bs, &mut buf)?;
        Ok(buf)
    }

    /// Apply `data` at `pos` in the file. Grows the file when writing
    /// past the current EOF.
    fn write_at(&mut self, pos: u64, data: &[u8]) -> std::io::Result<usize> {
        if data.is_empty() {
            return Ok(0);
        }
        // If the new write would push past MAX_INLINE_DATA, de-inline first.
        if self.inline && (pos + data.len() as u64) as usize > MAX_INLINE_DATA {
            self.de_inline().map_err(std::io::Error::other)?;
        }

        if self.inline {
            let end = (pos + data.len() as u64) as usize;
            if self.inline_buf.len() < end {
                self.inline_buf.resize(end, 0);
            }
            self.inline_buf[pos as usize..end].copy_from_slice(data);
            let new_size = self.size.max(end as u64);
            self.size = new_size;
            return Ok(data.len());
        }

        // Block path: stage each block we touch into `dirty`. We may
        // need to materialise the block first (read-modify-write).
        let bs = F2FS_BLKSIZE as u64;
        let mut written = 0usize;
        let total = data.len();
        let mut cur_pos = pos;
        while written < total {
            let block_idx = cur_pos / bs;
            let in_block_off = (cur_pos % bs) as usize;
            let chunk = (F2FS_BLKSIZE - in_block_off).min(total - written);
            // Materialise the block if not already dirty.
            if !self.dirty.contains_key(&block_idx) {
                let on_disk = self
                    .read_block_on_disk(block_idx)
                    .map_err(std::io::Error::other)?;
                self.dirty.insert(block_idx, on_disk);
            }
            let block = self.dirty.get_mut(&block_idx).expect("just inserted");
            block[in_block_off..in_block_off + chunk]
                .copy_from_slice(&data[written..written + chunk]);
            written += chunk;
            cur_pos += chunk as u64;
        }
        let new_size = self.size.max(pos + data.len() as u64);
        self.size = new_size;
        Ok(data.len())
    }

    /// Switch a file from inline to block-stored. Called when a write
    /// would push the file past `MAX_INLINE_DATA`.
    fn de_inline(&mut self) -> Result<()> {
        if !self.inline {
            return Ok(());
        }
        // Stash the materialised inline_buf into a dirty block 0 so
        // sync() picks it up like any other block-stored data. Inflate
        // to 4 KiB with zero padding.
        let mut block0 = vec![0u8; F2FS_BLKSIZE];
        let n = self.inline_buf.len().min(F2FS_BLKSIZE);
        block0[..n].copy_from_slice(&self.inline_buf[..n]);
        self.dirty.insert(0, block0);
        // Any bytes past block 0 from the inline buffer (there shouldn't
        // be any — inline_buf ≤ MAX_INLINE_DATA < F2FS_BLKSIZE) stay in
        // block 0. We don't preallocate further blocks; subsequent
        // writes past block 0 will materialise their own blocks on
        // demand via the normal path.
        self.inline = false;
        self.inline_buf.clear();
        // Update the inode record so sync() emits a non-inline inode.
        let w = self.fs.writer.as_mut().expect("writer live");
        let ino = w
            .inodes
            .get_mut(&self.nid)
            .ok_or_else(|| crate::Error::InvalidImage("f2fs: ghost nid".into()))?;
        ino.inline_flags &= !F2FS_INLINE_DATA;
        ino.inline_flags &= !F2FS_DATA_EXIST;
        ino.inline_payload.clear();
        Ok(())
    }

    /// `set_len` implementation: grow or shrink the file. Growing
    /// extends with zero bytes; shrinking truncates dirty buffers
    /// and updates the size, leaving allocated blocks past the new
    /// end to be ignored by readers (size is the source of truth).
    fn set_len_internal(&mut self, new_len: u64) -> Result<()> {
        if new_len == self.size {
            return Ok(());
        }
        if new_len < self.size {
            // Shrink. Discard any dirty buffer wholly past the new end.
            let bs = F2FS_BLKSIZE as u64;
            self.dirty.retain(|&idx, _| idx * bs < new_len);
            // Trim the final partial dirty block.
            let last_idx = new_len / bs;
            let last_off = (new_len % bs) as usize;
            if last_off != 0 {
                if let Some(b) = self.dirty.get_mut(&last_idx) {
                    for byte in b.iter_mut().skip(last_off) {
                        *byte = 0;
                    }
                }
            }
            if self.inline {
                let n = (new_len as usize).min(self.inline_buf.len());
                self.inline_buf.truncate(n);
            }
            self.size = new_len;
            if self.pos > self.size {
                self.pos = self.size;
            }
            return Ok(());
        }
        // Grow. Extend with zero bytes — represented as a size bump
        // (any blocks past the previous EOF read back as zero blocks
        // via NULL_ADDR until they're explicitly written).
        if self.inline {
            // If we'd stay inline, just extend inline_buf with zeros.
            if (new_len as usize) <= MAX_INLINE_DATA {
                self.inline_buf.resize(new_len as usize, 0);
                self.size = new_len;
                return Ok(());
            }
            // Otherwise, de-inline and fall through to the block grow.
            self.de_inline()?;
        }
        // Block path: just bump size. Reads past EOF return zero; sync
        // never has to materialise the gap blocks.
        self.size = new_len;
        Ok(())
    }

    /// Write all pending dirty bytes + the inline payload to disk,
    /// allocate fresh data blocks, repoint the inode, and emit a fresh
    /// checkpoint.
    fn sync_internal(&mut self) -> Result<()> {
        if self.synced {
            return Ok(());
        }
        let bs = F2FS_BLKSIZE as u64;

        // 1) Update inode metadata (size, inline payload) from our
        //    working copy. We do this even if `dirty` is empty, so
        //    metadata-only changes (truncate, set_len) survive sync.
        {
            let w = self.fs.writer.as_mut().expect("writer live");
            let ino = w
                .inodes
                .get_mut(&self.nid)
                .ok_or_else(|| crate::Error::InvalidImage("f2fs: ghost nid".into()))?;
            if self.inline {
                ino.inline_flags |= F2FS_INLINE_DATA | F2FS_DATA_EXIST;
                ino.inline_payload = self.inline_buf.clone();
                // Inline-stored files claim no main-area data blocks.
                ino.blocks = 0;
                // Wipe stale i_addr / i_nid pointers — they'd
                // be reinterpreted as the inline payload area otherwise.
                ino.i_addr = [0; super::constants::ADDRS_PER_INODE];
                ino.i_nid = [0; super::constants::NIDS_PER_INODE];
            }
            ino.size = self.size;
        }

        // 2) For each dirty block, allocate a fresh main-area block,
        //    write the buffer, and repoint the pointer tree.
        if !self.inline {
            // Snapshot the keys to avoid borrowing self.dirty across
            // the writer mutation.
            let dirty_indices: Vec<u64> = self.dirty.keys().copied().collect();
            for idx in dirty_indices {
                // Skip dirty blocks that lie wholly past the (possibly
                // shrunk) end-of-file — there's nothing for the reader
                // to ever see there.
                if idx * bs >= self.size {
                    continue;
                }
                let mut block = self.dirty.remove(&idx).expect("just listed");
                // Zero any tail bytes past the file's logical end so
                // we don't leak undefined memory across a short write.
                let file_end_in_block = if (idx + 1) * bs <= self.size {
                    F2FS_BLKSIZE
                } else {
                    (self.size - idx * bs) as usize
                };
                for byte in block.iter_mut().skip(file_end_in_block) {
                    *byte = 0;
                }
                // Allocate + write the new physical block.
                let phys = {
                    let w = self.fs.writer.as_mut().expect("writer live");
                    w.alloc_data_block()?
                };
                self.dev.write_at(phys as u64 * bs, &block)?;
                // Repoint the inode's pointer tree at the new block.
                let w = self.fs.writer.as_mut().expect("writer live");
                w.place_data_block(self.nid, idx, phys)?;
            }

            // Recompute i_blocks for the inode: count of non-zero
            // i_addr slots (data blocks). The writer's `flush()`
            // already recomputes the authoritative value (including
            // the inode block + dnode/indirect overhead), but seeding
            // this with the data-block count keeps the invariant that
            // an interim `read_inode` call sees a sane figure.
            let w = self.fs.writer.as_mut().expect("writer live");
            let ino = w
                .inodes
                .get_mut(&self.nid)
                .ok_or_else(|| crate::Error::InvalidImage("f2fs: ghost nid".into()))?;
            let mut count = 0u64;
            for a in ino.i_addr.iter() {
                if *a != 0 {
                    count += 1;
                }
            }
            ino.blocks = count;
        } else {
            // We may still have stale entries in `dirty` (e.g. if we
            // round-tripped through de_inline and back). Discard them —
            // the inline payload is the source of truth.
            self.dirty.clear();
        }

        // 3) Emit a fresh checkpoint so the on-disk image is consistent
        //    and re-openable. This is what makes the "checkpoint per
        //    sync" model work.
        self.fs.flush(self.dev)?;
        self.synced = true;
        Ok(())
    }
}

impl<'a> Read for F2fsFileHandle<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.read_at(self.pos, buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl<'a> Write for F2fsFileHandle<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // Each new write makes prior `sync` results stale — re-arm the
        // flag so the next sync (whether explicit or via Drop) commits.
        self.synced = false;
        let n = self.write_at(self.pos, buf)?;
        self.pos += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // std::io::Write::flush isn't required to push bytes durably —
        // we keep this a no-op and let `sync()` (or `Drop`) do the
        // checkpoint emission. Otherwise every `BufWriter::flush` would
        // trigger a CP, which is the antithesis of the "lazy
        // checkpoint" design.
        Ok(())
    }
}

impl<'a> Seek for F2fsFileHandle<'a> {
    fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
        let new_pos: i64 = match from {
            SeekFrom::Start(n) => n as i64,
            SeekFrom::End(off) => self.size as i64 + off,
            SeekFrom::Current(off) => self.pos as i64 + off,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "f2fs: seek before start",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

impl<'a> FileHandle for F2fsFileHandle<'a> {
    fn len(&self) -> u64 {
        self.size
    }

    fn set_len(&mut self, new_len: u64) -> Result<()> {
        self.synced = false;
        self.set_len_internal(new_len)
    }

    fn sync(&mut self) -> Result<()> {
        self.sync_internal()
    }
}

impl<'a> Drop for F2fsFileHandle<'a> {
    fn drop(&mut self) {
        // Best-effort sync on drop. We can't surface errors, so we
        // swallow them and rely on the caller to have called `sync()`
        // explicitly if they need to observe failure.
        let _ = self.sync_internal();
    }
}

/// Resolve a posix-style path to its nid via the writer's in-memory
/// tree. Returns `None` for a path whose leaf doesn't exist (the parent
/// must still resolve — otherwise we error).
fn resolve_in_writer(w: &super::write::Writer, path: &std::path::Path) -> Result<Option<u32>> {
    let s = path
        .to_str()
        .ok_or_else(|| crate::Error::InvalidArgument("f2fs: non-UTF-8 path".into()))?;
    if s == "/" || s.is_empty() {
        return Ok(Some(3));
    }
    let parts: Vec<&str> = s
        .trim_matches('/')
        .split('/')
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        return Err(crate::Error::InvalidArgument(format!(
            "f2fs: empty path {s:?}"
        )));
    }
    let mut cur = 3u32;
    for comp in &parts[..parts.len() - 1] {
        let kids = w.children.get(&cur).ok_or_else(|| {
            crate::Error::InvalidArgument(format!("f2fs: parent of {s:?} not a known dir"))
        })?;
        let found = kids
            .iter()
            .find(|d| d.name == comp.as_bytes())
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!("f2fs: no such dir {comp:?} in {s:?}"))
            })?;
        cur = found.ino;
    }
    let leaf = parts[parts.len() - 1].as_bytes();
    let existing = w
        .children
        .get(&cur)
        .and_then(|kids| kids.iter().find(|d| d.name == leaf).map(|d| d.ino));
    Ok(existing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;
    use crate::fs::{FileMeta, FileSource, Filesystem, OpenFlags};
    use std::io::SeekFrom;
    use std::path::Path;

    /// Fresh formatted F2FS volume on a 4 MiB MemoryBackend.
    fn fresh_fs() -> (MemoryBackend, super::F2fs) {
        let mut dev = MemoryBackend::new(4 * 1024 * 1024);
        let opts = super::super::FormatOpts {
            log_blocks_per_seg: 2,
            ..super::super::FormatOpts::default()
        };
        let fs = super::F2fs::format(&mut dev, &opts).unwrap();
        (dev, fs)
    }

    /// Slurp the whole file `/<name>` from a fresh `F2fs::open(&dev)`
    /// and return its bytes.
    fn read_all_from_reopened(dev: &mut MemoryBackend, name: &str) -> Vec<u8> {
        let mut fs2 = super::F2fs::open(dev).unwrap();
        let mut r = fs2.open_file_reader(dev, &format!("/{name}")).unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        out
    }

    /// Format → create file → open_file_rw → seek+write → drop → flush →
    /// reopen → confirm new bytes survived the checkpoint.
    #[test]
    fn open_file_rw_partial_write_round_trip() {
        let (mut dev, mut fs) = fresh_fs();
        // Inline file (≤ MAX_INLINE_DATA) created up-front.
        let initial = b"hello world, this is a test payload!".to_vec();
        fs.create_file(
            &mut dev,
            Path::new("/note.txt"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(initial.clone())),
                len: initial.len() as u64,
            },
            FileMeta::default(),
        )
        .unwrap();

        {
            let mut h = fs
                .open_file_rw(&mut dev, Path::new("/note.txt"), OpenFlags::default(), None)
                .unwrap();
            assert_eq!(h.len(), initial.len() as u64);
            h.seek(SeekFrom::Start(7)).unwrap();
            h.write_all(b"WORLD").unwrap();
            // sync inside drop.
        }
        // Final fs.flush is implicit in sync(), but call it for safety.
        fs.flush(&mut dev).unwrap();

        let got = read_all_from_reopened(&mut dev, "note.txt");
        let mut expected = initial.clone();
        expected[7..12].copy_from_slice(b"WORLD");
        assert_eq!(got, expected);
    }

    /// Writing past EOF must extend the file's logical size.
    #[test]
    fn open_file_rw_extends_file() {
        let (mut dev, mut fs) = fresh_fs();
        let initial = b"short".to_vec();
        fs.create_file(
            &mut dev,
            Path::new("/grow.bin"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(initial.clone())),
                len: initial.len() as u64,
            },
            FileMeta::default(),
        )
        .unwrap();

        {
            let mut h = fs
                .open_file_rw(&mut dev, Path::new("/grow.bin"), OpenFlags::default(), None)
                .unwrap();
            h.seek(SeekFrom::End(0)).unwrap();
            h.write_all(b"-extended-tail").unwrap();
            h.sync().unwrap();
            assert_eq!(h.len(), (initial.len() + "-extended-tail".len()) as u64);
        }
        fs.flush(&mut dev).unwrap();

        let got = read_all_from_reopened(&mut dev, "grow.bin");
        assert_eq!(got, b"short-extended-tail");
    }

    /// `set_len` must grow with zeros and shrink (with persistence).
    #[test]
    fn open_file_rw_set_len_grow_and_shrink() {
        let (mut dev, mut fs) = fresh_fs();
        let initial = b"abcdefghij".to_vec(); // 10 bytes
        fs.create_file(
            &mut dev,
            Path::new("/sz.bin"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(initial.clone())),
                len: initial.len() as u64,
            },
            FileMeta::default(),
        )
        .unwrap();

        // Grow to 20 bytes.
        {
            let mut h = fs
                .open_file_rw(&mut dev, Path::new("/sz.bin"), OpenFlags::default(), None)
                .unwrap();
            h.set_len(20).unwrap();
            assert_eq!(h.len(), 20);
            h.sync().unwrap();
        }
        fs.flush(&mut dev).unwrap();
        let got = read_all_from_reopened(&mut dev, "sz.bin");
        let mut expected = initial.clone();
        expected.resize(20, 0);
        assert_eq!(got, expected);

        // Shrink to 4 bytes.
        {
            let mut h = fs
                .open_file_rw(&mut dev, Path::new("/sz.bin"), OpenFlags::default(), None)
                .unwrap();
            h.set_len(4).unwrap();
            assert_eq!(h.len(), 4);
            h.sync().unwrap();
        }
        fs.flush(&mut dev).unwrap();
        let got = read_all_from_reopened(&mut dev, "sz.bin");
        assert_eq!(got, b"abcd");
    }

    /// `OpenFlags::append` places the cursor at EOF on open.
    #[test]
    fn open_file_rw_append() {
        let (mut dev, mut fs) = fresh_fs();
        fs.create_file(
            &mut dev,
            Path::new("/log.txt"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"line1\n".to_vec())),
                len: 6,
            },
            FileMeta::default(),
        )
        .unwrap();

        {
            let mut h = fs
                .open_file_rw(
                    &mut dev,
                    Path::new("/log.txt"),
                    OpenFlags {
                        append: true,
                        ..OpenFlags::default()
                    },
                    None,
                )
                .unwrap();
            h.write_all(b"line2\n").unwrap();
            h.sync().unwrap();
        }
        fs.flush(&mut dev).unwrap();

        let got = read_all_from_reopened(&mut dev, "log.txt");
        assert_eq!(got, b"line1\nline2\n");
    }

    /// `OpenFlags::create` makes a new file if absent.
    #[test]
    fn open_file_rw_create_new() {
        let (mut dev, mut fs) = fresh_fs();
        {
            let mut h = fs
                .open_file_rw(
                    &mut dev,
                    Path::new("/fresh.bin"),
                    OpenFlags {
                        create: true,
                        ..OpenFlags::default()
                    },
                    Some(FileMeta::default()),
                )
                .unwrap();
            assert_eq!(h.len(), 0);
            h.write_all(b"freshly-minted").unwrap();
            h.sync().unwrap();
        }
        fs.flush(&mut dev).unwrap();
        let got = read_all_from_reopened(&mut dev, "fresh.bin");
        assert_eq!(got, b"freshly-minted");
    }

    /// Exercise the explicit per-handle checkpoint path: the bytes must
    /// be visible to a brand-new `F2fs::open(&dev)` after `sync()` even
    /// without a subsequent `fs.flush(&mut dev)`.
    #[test]
    fn open_file_rw_writes_are_persisted_via_checkpoint() {
        let (mut dev, mut fs) = fresh_fs();
        fs.create_file(
            &mut dev,
            Path::new("/cp.bin"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"AAAA".to_vec())),
                len: 4,
            },
            FileMeta::default(),
        )
        .unwrap();

        // Touch the file via the rw handle and sync — sync is what emits
        // the new checkpoint. We deliberately don't call `fs.flush` after.
        {
            let mut h = fs
                .open_file_rw(&mut dev, Path::new("/cp.bin"), OpenFlags::default(), None)
                .unwrap();
            h.seek(SeekFrom::Start(1)).unwrap();
            h.write_all(b"ZZ").unwrap();
            h.sync().unwrap();
        }

        // Re-open from scratch and verify the modified bytes are there —
        // this is only possible if sync() really emitted a checkpoint.
        let mut fs2 = super::F2fs::open(&mut dev).unwrap();
        let mut r = fs2.open_file_reader(&mut dev, "/cp.bin").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, b"AZZA");
    }

    /// Write enough bytes to push a previously-inline file past
    /// MAX_INLINE_DATA so the de-inline code path runs.
    #[test]
    fn open_file_rw_de_inlines_when_growing_past_threshold() {
        let (mut dev, mut fs) = fresh_fs();
        // Start as a tiny inline file.
        fs.create_file(
            &mut dev,
            Path::new("/grow.bin"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"tiny".to_vec())),
                len: 4,
            },
            FileMeta::default(),
        )
        .unwrap();

        // Grow past MAX_INLINE_DATA.
        let big_size = MAX_INLINE_DATA + F2FS_BLKSIZE; // 1 inline-area + 1 block
        let mut tail = vec![0u8; big_size - 4];
        for (i, b) in tail.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(7);
        }
        {
            let mut h = fs
                .open_file_rw(&mut dev, Path::new("/grow.bin"), OpenFlags::default(), None)
                .unwrap();
            h.seek(SeekFrom::Start(4)).unwrap();
            h.write_all(&tail).unwrap();
            h.sync().unwrap();
            assert_eq!(h.len(), big_size as u64);
        }
        fs.flush(&mut dev).unwrap();

        let got = read_all_from_reopened(&mut dev, "grow.bin");
        let mut expected = b"tiny".to_vec();
        expected.extend_from_slice(&tail);
        assert_eq!(got.len(), expected.len());
        assert_eq!(got, expected);
    }
}
