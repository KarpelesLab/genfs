//! GRF write-side helpers — `add_file`, `remove`, `flush`.
//!
//! Mutation model: the on-disk GRF carries data first (each file
//! lives at an absolute offset `HEADER_SIZE + entry.pos`) and the
//! table last (at `HEADER_SIZE + header.table_offset`). Adding a
//! file appends to the data region; removing a file marks bytes
//! wasted but leaves them on disk until a repack. Flush rewrites
//! the table at the current `data_end` and updates the header.
//!
//! This module never re-encrypts file bodies — libgrf's writer
//! clears `MIXCRYPT` / `DES` on the way out, and so do we. Each new
//! file lands as `GRF_FLAG_FILE` only, body zlib-compressed.

use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::FileSource;
use crate::fs::grf::header::Header;
use crate::fs::grf::table::{self, Entry, GRF_FLAG_FILE};
use crate::fs::grf::{Grf, HEADER_SIZE};

/// 4-byte alignment used by GRF on each file's compressed bytes.
const DATA_ALIGN: u32 = 4;

/// Append a file at the current `data_end`, compressing with zlib.
/// On disk the entry lands as `GRF_FLAG_FILE` only — no per-file
/// encryption is ever written by this implementation.
pub(super) fn add_file(
    grf: &mut Grf,
    dev: &mut dyn BlockDevice,
    key: String,
    src: FileSource,
) -> Result<()> {
    // Load the source bytes into memory. GRF entries are zlib blobs
    // that must be deflated as a single stream, so we can't avoid
    // buffering — keep this realistic by capping at 4 GiB minus
    // a margin so the u32 fields don't overflow.
    let plain = read_source(src)?;
    if plain.len() > (u32::MAX as usize - 1024) {
        return Err(crate::Error::Unsupported(
            "grf: file body larger than 4 GiB - 1 KiB".into(),
        ));
    }

    // If an entry already exists at `key`, replace it: the old
    // body's bytes become wasted space (we don't try to reclaim them
    // in place — that would require a free-list, see Limitations).
    if let Some(old) = grf.entries.remove(&key) {
        grf.wasted_space = grf
            .wasted_space
            .saturating_add(u64::from(old.len_aligned));
    }

    let compressed =
        crate::compression::compress(crate::compression::Algo::Zlib, &plain)?;
    let len = compressed.len() as u32;
    let len_aligned = len.div_ceil(DATA_ALIGN) * DATA_ALIGN;

    // Position relative to end-of-header; data_end is absolute.
    let pos_u64 = grf.data_end - HEADER_SIZE as u64;
    if pos_u64 > u32::MAX as u64 {
        return Err(crate::Error::Unsupported(
            "grf: archive grown past 4 GiB; repack into a new archive".into(),
        ));
    }
    let pos = pos_u64 as u32;

    // Write the compressed bytes (followed by alignment padding) at
    // `data_end`. We zero-pad to len_aligned so any stale tail from
    // a previous lifetime of this archive doesn't leak through.
    dev.write_at(grf.data_end, &compressed)?;
    if len_aligned > len {
        let pad = vec![0u8; (len_aligned - len) as usize];
        dev.write_at(grf.data_end + len as u64, &pad)?;
    }
    grf.data_end += u64::from(len_aligned);

    grf.entries.insert(
        key.clone(),
        Entry {
            name: key,
            size: plain.len() as u32,
            len,
            len_aligned,
            pos,
            flags: GRF_FLAG_FILE,
        },
    );
    grf.dirty = true;
    Ok(())
}

/// Unlink an entry by name. The file's data bytes stay on disk but
/// stop being referenced; `wasted_space` accumulates the bytes that
/// could be reclaimed by a repack.
pub(super) fn remove(grf: &mut Grf, key: &str) -> Result<()> {
    let removed = grf.entries.remove(key).ok_or_else(|| {
        crate::Error::InvalidArgument(format!("grf: no entry at {key:?}"))
    })?;
    grf.wasted_space = grf
        .wasted_space
        .saturating_add(u64::from(removed.len_aligned));
    grf.dirty = true;
    Ok(())
}

/// Re-serialise the file table, compress it, write it at
/// `data_end`, then rewrite the header. Truncates the device to the
/// new end-of-table.
pub(super) fn flush(grf: &mut Grf, dev: &mut dyn BlockDevice) -> Result<()> {
    if !grf.dirty && !grf.fresh {
        return Ok(());
    }

    // The writer only emits v0x200 today.
    if grf.version != 0x200 {
        return Err(crate::Error::Unsupported(format!(
            "grf: writer only emits v0x200; in-memory archive is v{:#x}",
            grf.version
        )));
    }

    // Serialise entries in pos-sorted order. libgrf does the same;
    // it keeps the table cache-friendly when reading.
    let mut by_pos: Vec<_> = grf.entries.values().cloned().collect();
    by_pos.sort_by_key(|e| e.pos);

    let raw_table = table::encode_v200(&by_pos);
    let compressed =
        crate::compression::compress(crate::compression::Algo::Zlib, &raw_table)?;

    // Table sits at data_end. Write the 8-byte posinfo header,
    // then the compressed table bytes.
    let table_abs = grf.data_end;
    let mut posinfo = [0u8; 8];
    posinfo[0..4].copy_from_slice(&(compressed.len() as u32).to_le_bytes());
    posinfo[4..8].copy_from_slice(&(raw_table.len() as u32).to_le_bytes());
    dev.write_at(table_abs, &posinfo)?;
    dev.write_at(table_abs + 8, &compressed)?;

    // BlockDevice has no `truncate` — the device's logical size is
    // fixed at creation. The space past the table stays as whatever
    // bytes were there (zeros for a fresh image; possibly garbage
    // for a reopened one). libgrf truncates the OS file; the
    // fstool-level callers (image creation, repack pipeline) size
    // the destination so the tail is benign zero.
    let _new_end = table_abs + 8 + compressed.len() as u64;

    // Rewrite the header with the new table offset + filecount.
    let table_offset = (table_abs - HEADER_SIZE as u64) as u32;
    let head = Header {
        encrypted_header: grf.encrypted_header,
        table_offset,
        seed: grf.seed,
        filecount: grf.entries.len() as u32,
        version: grf.version,
    };
    let head_bytes = head.encode();
    dev.write_at(0, &head_bytes)?;

    grf.table_offset = table_offset;
    grf.dirty = false;
    grf.fresh = false;
    dev.sync()?;
    Ok(())
}

/// Drain a [`FileSource`] into a single byte vector. GRF requires
/// the entire body to be in memory for a single deflate stream, so
/// we can't stream this — the trade-off is documented in the type
/// rather than hidden in the call sites.
fn read_source(src: FileSource) -> Result<Vec<u8>> {
    let (mut reader, len) = src.open()?;
    // Seek to the start in case the caller passed a `Reader` that
    // had already been read from (defensive — most won't have).
    let _ = reader.seek(SeekFrom::Start(0));
    let mut out = Vec::with_capacity(len as usize);
    reader.read_to_end(&mut out)?;
    Ok(out)
}
