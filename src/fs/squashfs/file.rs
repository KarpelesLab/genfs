//! SquashFS regular-file streaming reader.
//!
//! Holds a borrow of the block device and walks the file's block list one
//! block at a time, decoding the size word's "uncompressed" bit. The
//! current block's payload sits in a `Vec<u8>` buffer; the next block is
//! fetched lazily when the buffer drains.
//!
//! Compressed data and compressed fragment blocks return
//! [`crate::Error::Unsupported`] on first access — the error is surfaced
//! through `io::Read::read()` via `io::Error::other`.

use std::io::{self, Read, Seek, SeekFrom};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::FileReadHandle;
use crate::fs::squashfs::Compression;
use crate::fs::squashfs::fragment::{self, FragmentEntry};
use crate::fs::squashfs::inode::FileInode;
use crate::fs::squashfs::metablock::compression_to_algo;

/// Streaming reader for a regular SquashFS file. Stays within one block
/// (or fragment-tail) buffer at a time; never loads the entire file.
pub struct FileReader<'a> {
    dev: &'a mut dyn BlockDevice,
    /// Compression algorithm from the superblock (used only for error
    /// messages when we hit a compressed block).
    compression: Compression,
    fragment_table_start: u64,
    fragment_count: u32,
    /// Disk offset where the file's first data block starts.
    next_disk_offset: u64,
    /// Block sizes copied out of the inode.
    block_sizes: Vec<u32>,
    /// Index of the next block in `block_sizes` to read.
    next_block_idx: usize,
    /// Full block size from the superblock; sparse zero blocks fill this.
    full_block_size: u32,
    /// Bytes remaining for the *whole* file. Decrements as we hand bytes
    /// to the caller; once zero, we report EOF.
    remaining: u64,
    /// Currently-buffered block (decompressed or copied raw). Empty when
    /// no block is staged.
    buf: Vec<u8>,
    /// Byte offset into `buf` of the next byte to return.
    buf_pos: usize,
    /// Optional fragment descriptor: fetched once when we reach the tail.
    fragment_index: u32,
    fragment_offset: u32,
    /// Have we already fetched the fragment block into `buf`?
    fragment_consumed: bool,
}

impl<'a> FileReader<'a> {
    pub fn new(
        dev: &'a mut dyn BlockDevice,
        inode: &FileInode,
        compression: Compression,
        fragment_table_start: u64,
        fragment_count: u32,
        full_block_size: u32,
    ) -> Self {
        Self {
            dev,
            compression,
            fragment_table_start,
            fragment_count,
            next_disk_offset: inode.blocks_start,
            block_sizes: inode.block_sizes.clone(),
            next_block_idx: 0,
            full_block_size,
            remaining: inode.file_size,
            buf: Vec::new(),
            buf_pos: 0,
            fragment_index: inode.fragment_index,
            fragment_offset: inode.fragment_offset,
            fragment_consumed: false,
        }
    }

    fn has_fragment(&self) -> bool {
        self.fragment_index != 0xFFFF_FFFF
    }

    /// Refill `self.buf` with the next block (full data block or fragment
    /// tail). Returns `Ok(true)` if a block was staged, `Ok(false)` if we
    /// are out of blocks/tail. Errors propagate compression-related
    /// `Unsupported` cleanly.
    fn refill(&mut self) -> Result<bool> {
        // Done?
        if self.remaining == 0 {
            return Ok(false);
        }
        // Try the next full block.
        if self.next_block_idx < self.block_sizes.len() {
            let size_word = self.block_sizes[self.next_block_idx];
            let uncompressed = size_word & 0x0100_0000 != 0;
            let on_disk = size_word & 0x00FF_FFFF;
            self.buf_pos = 0;
            if on_disk == 0 {
                // Sparse block — entire block is zero, not on disk.
                let take = (self.full_block_size as u64).min(self.remaining) as usize;
                self.buf = vec![0u8; take];
            } else {
                let mut payload = vec![0u8; on_disk as usize];
                self.dev.read_at(self.next_disk_offset, &mut payload)?;
                let decoded = if uncompressed {
                    payload
                } else {
                    // Non-tail blocks decompress to exactly `full_block_size`;
                    // the tail block decompresses to whatever's left of the
                    // file. minilzo treats this as an exact size, so the
                    // remaining-vs-full distinction matters for LZO.
                    let expected = self.remaining.min(self.full_block_size as u64) as usize;
                    let algo = compression_to_algo(self.compression).ok_or_else(|| {
                        crate::Error::InvalidImage(format!(
                            "squashfs: unknown compressor id {}",
                            compression_label(self.compression)
                        ))
                    })?;
                    crate::compression::decompress(algo, &payload, expected)?
                };
                // Trim the final block to actual file size if needed.
                let take = (decoded.len() as u64).min(self.remaining) as usize;
                let mut trimmed = decoded;
                trimmed.truncate(take);
                self.buf = trimmed;
            }
            self.next_disk_offset = self.next_disk_offset.saturating_add(on_disk as u64);
            self.next_block_idx += 1;
            return Ok(true);
        }
        // Out of full blocks — fall back to fragment tail, if any.
        if self.has_fragment() && !self.fragment_consumed {
            self.fragment_consumed = true;
            let entry: FragmentEntry = fragment::read_fragment(
                self.dev,
                self.fragment_table_start,
                self.fragment_count,
                self.compression,
                self.fragment_index,
            )?;
            // Read the fragment block, decompressing if needed, then slice
            // out our tail.
            let frag_size = entry.on_disk_size() as usize;
            let mut frag_buf = vec![0u8; frag_size];
            self.dev.read_at(entry.start, &mut frag_buf)?;
            let frag_buf = if entry.is_uncompressed() {
                frag_buf
            } else {
                let algo = compression_to_algo(self.compression).ok_or_else(|| {
                    crate::Error::InvalidImage(format!(
                        "squashfs: unknown compressor id {}",
                        compression_label(self.compression)
                    ))
                })?;
                // Fragment blocks pack multiple file tails together; their
                // uncompressed size is at most one FS block. We can't know
                // it exactly from on-disk metadata, so we pass `full_block_size`
                // as the cap. For LZO that becomes an exact-length request,
                // which works for mksquashfs-produced images.
                crate::compression::decompress(algo, &frag_buf, self.full_block_size as usize)?
            };
            let off = self.fragment_offset as usize;
            let take = self.remaining as usize;
            if off.checked_add(take).is_none_or(|end| end > frag_buf.len()) {
                return Err(crate::Error::InvalidImage(format!(
                    "squashfs: fragment tail [{off}..{}] exceeds fragment block size {}",
                    off + take,
                    frag_buf.len()
                )));
            }
            self.buf = frag_buf[off..off + take].to_vec();
            self.buf_pos = 0;
            return Ok(true);
        }
        Ok(false)
    }
}

impl<'a> Read for FileReader<'a> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if self.buf_pos >= self.buf.len() {
            match self.refill() {
                Ok(true) => {}
                Ok(false) => return Ok(0),
                Err(e) => return Err(io::Error::other(format!("{e}"))),
            }
        }
        let avail = self.buf.len() - self.buf_pos;
        let n = avail.min(out.len());
        out[..n].copy_from_slice(&self.buf[self.buf_pos..self.buf_pos + n]);
        self.buf_pos += n;
        self.remaining = self.remaining.saturating_sub(n as u64);
        Ok(n)
    }
}

fn compression_label(c: Compression) -> &'static str {
    match c {
        Compression::Gzip => "gzip",
        Compression::Lzma => "lzma",
        Compression::Lzo => "lzo",
        Compression::Xz => "xz",
        Compression::Lz4 => "lz4",
        Compression::Zstd => "zstd",
        Compression::Unknown(_) => "unknown",
    }
}

/// Random-access (`Read + Seek + len`) handle for a SquashFS file.
///
/// Walks the same block list as [`FileReader`] but caches the
/// most-recently-decompressed block so backward seeks within a
/// block (or sequential reads after a small seek) don't pay the
/// decompression cost twice. Seeks across blocks invalidate the
/// cache; the next read fetches + decompresses the new block.
///
/// Memory footprint: at most one decompressed block (typically
/// 128 KiB) plus block-size table + on-disk offset prefix sum.
pub struct SquashfsFileReadHandle<'a> {
    dev: &'a mut dyn BlockDevice,
    compression: Compression,
    fragment_table_start: u64,
    fragment_count: u32,
    /// Disk offset of the first data block. Prefix-summed against
    /// `block_disk_offsets` so block N starts at
    /// `blocks_start + block_disk_offsets[N]`.
    blocks_start: u64,
    /// `block_sizes[N]` — the on-disk size word (high bit = uncompressed).
    block_sizes: Vec<u32>,
    /// Cumulative on-disk byte offsets, len = block_sizes.len() + 1.
    block_disk_offsets: Vec<u64>,
    full_block_size: u32,
    /// Total logical file length.
    len: u64,
    /// Current logical position in the file.
    pos: u64,
    /// Cached decompressed block. `cached_block_idx == usize::MAX`
    /// means "no block cached yet."
    cached_block_idx: usize,
    cached_buf: Vec<u8>,
    /// Fragment metadata. `fragment_index == 0xFFFF_FFFF` means
    /// "no fragment tail."
    fragment_index: u32,
    fragment_offset: u32,
    /// Cached decompressed fragment-tail bytes (just this file's slice).
    /// Allocated lazily; empty before first use.
    cached_fragment_tail: Vec<u8>,
    cached_fragment_loaded: bool,
}

impl<'a> SquashfsFileReadHandle<'a> {
    pub fn new(
        dev: &'a mut dyn BlockDevice,
        inode: &FileInode,
        compression: Compression,
        fragment_table_start: u64,
        fragment_count: u32,
        full_block_size: u32,
    ) -> Self {
        // Prefix-sum the on-disk block sizes so any block's disk
        // offset is a single index away.
        let mut block_disk_offsets = Vec::with_capacity(inode.block_sizes.len() + 1);
        block_disk_offsets.push(0u64);
        let mut acc = 0u64;
        for &sz in &inode.block_sizes {
            let on_disk = sz & 0x00FF_FFFF;
            acc = acc.saturating_add(u64::from(on_disk));
            block_disk_offsets.push(acc);
        }
        Self {
            dev,
            compression,
            fragment_table_start,
            fragment_count,
            blocks_start: inode.blocks_start,
            block_sizes: inode.block_sizes.clone(),
            block_disk_offsets,
            full_block_size,
            len: inode.file_size,
            pos: 0,
            cached_block_idx: usize::MAX,
            cached_buf: Vec::new(),
            fragment_index: inode.fragment_index,
            fragment_offset: inode.fragment_offset,
            cached_fragment_tail: Vec::new(),
            cached_fragment_loaded: false,
        }
    }

    fn has_fragment(&self) -> bool {
        self.fragment_index != 0xFFFF_FFFF
    }

    /// Logical position where the fragment tail begins.
    fn fragment_start_pos(&self) -> u64 {
        u64::from(self.full_block_size) * self.block_sizes.len() as u64
    }

    /// Ensure block N is decoded into `cached_buf`. Idempotent when
    /// the cache already holds the requested block.
    fn ensure_block_cached(&mut self, idx: usize) -> Result<()> {
        if self.cached_block_idx == idx {
            return Ok(());
        }
        let size_word = self.block_sizes[idx];
        let uncompressed = size_word & 0x0100_0000 != 0;
        let on_disk = size_word & 0x00FF_FFFF;
        let disk_offset = self.blocks_start + self.block_disk_offsets[idx];
        let logical_start = u64::from(self.full_block_size) * idx as u64;
        let logical_end = (logical_start + u64::from(self.full_block_size)).min(self.len);
        let block_logical_len = (logical_end - logical_start) as usize;
        if on_disk == 0 {
            self.cached_buf = vec![0u8; block_logical_len];
        } else {
            let mut payload = vec![0u8; on_disk as usize];
            self.dev.read_at(disk_offset, &mut payload)?;
            let decoded = if uncompressed {
                payload
            } else {
                let algo = compression_to_algo(self.compression).ok_or_else(|| {
                    crate::Error::InvalidImage(format!(
                        "squashfs: unknown compressor id {}",
                        compression_label(self.compression)
                    ))
                })?;
                crate::compression::decompress(algo, &payload, block_logical_len)?
            };
            let mut trimmed = decoded;
            trimmed.truncate(block_logical_len);
            self.cached_buf = trimmed;
        }
        self.cached_block_idx = idx;
        Ok(())
    }

    /// Ensure the fragment tail's slice is cached.
    fn ensure_fragment_cached(&mut self) -> Result<()> {
        if self.cached_fragment_loaded {
            return Ok(());
        }
        let frag_start_pos = self.fragment_start_pos();
        let tail_len = (self.len - frag_start_pos) as usize;
        let entry: FragmentEntry = fragment::read_fragment(
            self.dev,
            self.fragment_table_start,
            self.fragment_count,
            self.compression,
            self.fragment_index,
        )?;
        let mut frag_buf = vec![0u8; entry.on_disk_size() as usize];
        self.dev.read_at(entry.start, &mut frag_buf)?;
        let frag_buf = if entry.is_uncompressed() {
            frag_buf
        } else {
            let algo = compression_to_algo(self.compression).ok_or_else(|| {
                crate::Error::InvalidImage(format!(
                    "squashfs: unknown compressor id {}",
                    compression_label(self.compression)
                ))
            })?;
            crate::compression::decompress(algo, &frag_buf, self.full_block_size as usize)?
        };
        let off = self.fragment_offset as usize;
        if off.checked_add(tail_len).is_none_or(|end| end > frag_buf.len()) {
            return Err(crate::Error::InvalidImage(format!(
                "squashfs: fragment tail [{off}..{}] exceeds fragment block size {}",
                off + tail_len,
                frag_buf.len()
            )));
        }
        self.cached_fragment_tail = frag_buf[off..off + tail_len].to_vec();
        self.cached_fragment_loaded = true;
        Ok(())
    }
}

impl<'a> Read for SquashfsFileReadHandle<'a> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() || self.pos >= self.len {
            return Ok(0);
        }
        let frag_start = self.fragment_start_pos();
        if self.pos < frag_start || !self.has_fragment() {
            // Reading from a data block. Compute block index + offset
            // within the (decompressed) block, refill cache if needed.
            let block_idx = (self.pos / u64::from(self.full_block_size)) as usize;
            if block_idx >= self.block_sizes.len() {
                return Ok(0);
            }
            self.ensure_block_cached(block_idx)
                .map_err(|e| io::Error::other(format!("{e}")))?;
            let block_logical_start = u64::from(self.full_block_size) * block_idx as u64;
            let off_in_block = (self.pos - block_logical_start) as usize;
            let avail = self.cached_buf.len() - off_in_block;
            let n = avail.min(out.len());
            out[..n].copy_from_slice(&self.cached_buf[off_in_block..off_in_block + n]);
            self.pos += n as u64;
            Ok(n)
        } else {
            self.ensure_fragment_cached()
                .map_err(|e| io::Error::other(format!("{e}")))?;
            let off_in_tail = (self.pos - frag_start) as usize;
            let avail = self.cached_fragment_tail.len() - off_in_tail;
            let n = avail.min(out.len());
            out[..n].copy_from_slice(
                &self.cached_fragment_tail[off_in_tail..off_in_tail + n],
            );
            self.pos += n as u64;
            Ok(n)
        }
    }
}

impl<'a> Seek for SquashfsFileReadHandle<'a> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let target_i128 = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
            SeekFrom::End(d) => self.len as i128 + d as i128,
        };
        if target_i128 < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "squashfs: seek to negative offset",
            ));
        }
        // Clamp past-EOF to `len` so a subsequent read returns 0.
        let target = (target_i128 as u128).min(self.len as u128) as u64;
        self.pos = target;
        Ok(self.pos)
    }
}

impl<'a> FileReadHandle for SquashfsFileReadHandle<'a> {
    fn len(&self) -> u64 {
        self.len
    }
}
