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

use std::io::{self, Read};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::squashfs::Compression;
use crate::fs::squashfs::fragment::{self, FragmentEntry};
use crate::fs::squashfs::inode::FileInode;

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
                if !uncompressed {
                    return Err(crate::Error::Unsupported(format!(
                        "squashfs: {} decompression requires a feature flag, not built",
                        compression_label(self.compression)
                    )));
                }
                let mut payload = vec![0u8; on_disk as usize];
                self.dev.read_at(self.next_disk_offset, &mut payload)?;
                // Trim the final block to actual file size if needed.
                let take = (payload.len() as u64).min(self.remaining) as usize;
                payload.truncate(take);
                self.buf = payload;
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
            if !entry.is_uncompressed() {
                return Err(crate::Error::Unsupported(format!(
                    "squashfs: {} decompression requires a feature flag, not built",
                    compression_label(self.compression)
                )));
            }
            // Read the fragment block, slice out our tail.
            let frag_size = entry.on_disk_size() as usize;
            let mut frag_buf = vec![0u8; frag_size];
            self.dev.read_at(entry.start, &mut frag_buf)?;
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
