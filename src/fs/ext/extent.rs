//! ext4 extent tree encoding.
//!
//! An ext4 inode that uses extents stores an extent header at the start of
//! `i_block[..]` (12 bytes) followed by extent records. The 60-byte `i_block`
//! array fits the header + up to 4 leaf extents. Inodes that need more
//! extents put an extent index in `i_block` pointing at leaf blocks on disk
//! (v1 only emits depth-0 trees — up to 4 extents per inode).
//!
//! References (UAPI / on-disk constants, not GPL code):
//! - `linux/include/uapi/linux/types.h`-style layout of the extent structures
//! - `Documentation/filesystems/ext4/blockmap.rst` (kernel docs, public)
//!
//! ## Layout (all little-endian)
//!
//! ```text
//!   extent_header (12 B)
//!     0..2   eh_magic       = 0xF30A
//!     2..4   eh_entries     = N (valid extents)
//!     4..6   eh_max         = capacity (4 in i_block, more in leaf blocks)
//!     6..8   eh_depth       = 0 → leaf entries, > 0 → index entries
//!     8..12  eh_generation  = 0 (unused)
//!
//!   extent (leaf, 12 B)
//!     0..4   ee_block       = first logical block in the file
//!     4..6   ee_len         = number of blocks (≤ 32768 for initialized)
//!     6..8   ee_start_hi    = high 16 bits of physical block number
//!     8..12  ee_start_lo    = low  32 bits of physical block number
//! ```

/// Magic at the start of an extent header.
pub const EXT4_EXT_MAGIC: u16 = 0xF30A;

/// Maximum number of leaf extents that fit alongside the header in the
/// 60-byte `i_block` array.
pub const MAX_EXTENTS_IN_INODE: usize = 4;

/// Maximum number of index entries that fit alongside the header in the
/// 60-byte `i_block` array. Same value as the leaf cap; both records are
/// 12 bytes.
pub const MAX_INDICES_IN_INODE: usize = 4;

/// Maximum `ee_len` for an "initialized" extent. Values in `32768..=65535`
/// are interpreted as "uninitialized" (zero-filled on read), which we
/// don't emit — so callers must split runs longer than this.
pub const MAX_LEN_PER_EXTENT: u16 = 32_768;

/// One contiguous run in a file: `len` blocks starting at logical block
/// `logical` mapped to physical block `physical`.
#[derive(Debug, Clone, Copy)]
pub struct ExtentRun {
    pub logical: u32,
    pub len: u16,
    pub physical: u64,
}

/// One internal-node entry in the extent tree. Points at a child node
/// (another idx block at depth > 1, or a leaf block at depth == 1).
#[derive(Debug, Clone, Copy)]
pub struct ExtentIdx {
    /// First logical block covered by the subtree rooted at this index.
    pub block: u32,
    /// Physical block holding the child node (extent_header + entries).
    pub leaf: u64,
}

/// Number of extent records (leaf or idx) that fit in a single
/// `block_size`-byte tree block. Used by the reader as an upper bound on
/// `eh_entries`, so it must reflect the WITHOUT-csum capacity (a
/// `metadata_csum`-on writer emits a smaller `eh_max`, which fits
/// inside this bound).
pub fn entries_per_leaf_block(block_size: u32) -> usize {
    ((block_size as usize) - 12) / 12
}

/// Same as [`entries_per_leaf_block`] but reserves the trailing 4-byte
/// `ext4_extent_tail` (the CRC32C) when `csum_tail` is set. The writer
/// emits this value as `eh_max` on each leaf block.
pub fn entries_per_leaf_block_capped(block_size: u32, csum_tail: bool) -> usize {
    let header = 12usize;
    let tail = if csum_tail { 4 } else { 0 };
    ((block_size as usize) - header - tail) / 12
}

/// Encode the 12-byte extent header.
pub fn encode_header(entries: u16, max: u16, depth: u16) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0..2].copy_from_slice(&EXT4_EXT_MAGIC.to_le_bytes());
    out[2..4].copy_from_slice(&entries.to_le_bytes());
    out[4..6].copy_from_slice(&max.to_le_bytes());
    out[6..8].copy_from_slice(&depth.to_le_bytes());
    // 8..12: eh_generation — leave zero
    out
}

/// Encode one 12-byte leaf-extent record.
pub fn encode_leaf(run: ExtentRun) -> [u8; 12] {
    assert!(
        run.len <= MAX_LEN_PER_EXTENT,
        "extent length {} exceeds initialized cap {}",
        run.len,
        MAX_LEN_PER_EXTENT
    );
    let mut out = [0u8; 12];
    out[0..4].copy_from_slice(&run.logical.to_le_bytes());
    out[4..6].copy_from_slice(&run.len.to_le_bytes());
    out[6..8].copy_from_slice(&((run.physical >> 32) as u16).to_le_bytes());
    out[8..12].copy_from_slice(&(run.physical as u32).to_le_bytes());
    out
}

/// Collapse a list of physical block numbers (indexed by logical block)
/// into the minimum-length sequence of contiguous extents.
///
/// A `0` entry is a hole (sparse file): it produces no extent, and the
/// logical gap it leaves naturally starts a fresh extent for the next
/// present block.
pub fn coalesce(data_blocks: &[u32]) -> Vec<ExtentRun> {
    let mut out: Vec<ExtentRun> = Vec::new();
    for (i, &phys) in data_blocks.iter().enumerate() {
        if phys == 0 {
            continue; // hole — no extent covers this logical block
        }
        let logical = i as u32;
        if let Some(last) = out.last_mut() {
            // Same run if physically contiguous AND fits within ee_len cap.
            let next_phys_in_run = last.physical + last.len as u64;
            if next_phys_in_run == phys as u64
                && last.len < MAX_LEN_PER_EXTENT
                && (last.logical + last.len as u32) == logical
            {
                last.len += 1;
                continue;
            }
        }
        out.push(ExtentRun {
            logical,
            len: 1,
            physical: phys as u64,
        });
    }
    out
}

/// Pack an extent header + a list of leaf extents into a 60-byte
/// `i_block` slice. Returns `Err` if more than [`MAX_EXTENTS_IN_INODE`]
/// extents are supplied (depth-0 cap; multi-level trees are deferred).
pub fn pack_into_iblock(runs: &[ExtentRun]) -> crate::Result<[u8; 60]> {
    if runs.len() > MAX_EXTENTS_IN_INODE {
        return Err(crate::Error::Unsupported(format!(
            "ext4: file requires {} extents, max {} per depth-0 tree (multi-level trees not yet implemented)",
            runs.len(),
            MAX_EXTENTS_IN_INODE
        )));
    }
    let mut out = [0u8; 60];
    let hdr = encode_header(runs.len() as u16, MAX_EXTENTS_IN_INODE as u16, 0);
    out[0..12].copy_from_slice(&hdr);
    for (i, run) in runs.iter().enumerate() {
        let off = 12 + i * 12;
        out[off..off + 12].copy_from_slice(&encode_leaf(*run));
    }
    Ok(out)
}

/// Encode one 12-byte index record (`extent_idx`).
///
/// Layout:
///
/// ```text
///   0..4   ei_block       = first logical block covered by the subtree
///   4..8   ei_leaf_lo     = low  32 bits of physical block holding the child node
///   8..10  ei_leaf_hi     = high 16 bits of physical block holding the child node
///   10..12 ei_unused      = 0
/// ```
pub fn encode_idx(idx: ExtentIdx) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0..4].copy_from_slice(&idx.block.to_le_bytes());
    out[4..8].copy_from_slice(&(idx.leaf as u32).to_le_bytes());
    out[8..10].copy_from_slice(&((idx.leaf >> 32) as u16).to_le_bytes());
    // 10..12 zero
    out
}

/// Decode one 12-byte index record.
pub fn decode_idx(buf: &[u8]) -> ExtentIdx {
    let block = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let leaf_lo = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as u64;
    let leaf_hi = u16::from_le_bytes(buf[8..10].try_into().unwrap()) as u64;
    ExtentIdx {
        block,
        leaf: (leaf_hi << 32) | leaf_lo,
    }
}

/// Pack an extent header + a list of index entries into a 60-byte
/// `i_block` slice. Returns `Err` if more than [`MAX_INDICES_IN_INODE`]
/// idx entries are supplied (i.e. we'd need depth > 1).
pub fn pack_idx_into_iblock(indices: &[ExtentIdx]) -> crate::Result<[u8; 60]> {
    if indices.len() > MAX_INDICES_IN_INODE {
        return Err(crate::Error::Unsupported(format!(
            "ext4: depth-1 tree requires {} idx entries, max {} inline (depth > 1 not implemented)",
            indices.len(),
            MAX_INDICES_IN_INODE
        )));
    }
    let mut out = [0u8; 60];
    let hdr = encode_header(indices.len() as u16, MAX_INDICES_IN_INODE as u16, 1);
    out[0..12].copy_from_slice(&hdr);
    for (i, idx) in indices.iter().enumerate() {
        let off = 12 + i * 12;
        out[off..off + 12].copy_from_slice(&encode_idx(*idx));
    }
    Ok(out)
}

/// Encode a full leaf block (header + leaf extents) into a `bs`-byte
/// buffer. Used when writing a leaf block to disk in a depth-1 tree.
/// When `csum_tail` is set, the trailing 4 bytes are reserved for the
/// `ext4_extent_tail` CRC32C (stamped at flush time, not here) and
/// `eh_max` is capped to fit.
pub fn encode_leaf_block(
    runs: &[ExtentRun],
    block_size: u32,
    csum_tail: bool,
) -> crate::Result<Vec<u8>> {
    let max = entries_per_leaf_block_capped(block_size, csum_tail);
    if runs.len() > max {
        return Err(crate::Error::Unsupported(format!(
            "ext4: leaf block would need {} entries, max {} per {}-byte block (csum_tail={csum_tail})",
            runs.len(),
            max,
            block_size,
        )));
    }
    let mut out = vec![0u8; block_size as usize];
    let hdr = encode_header(runs.len() as u16, max as u16, 0);
    out[0..12].copy_from_slice(&hdr);
    for (i, r) in runs.iter().enumerate() {
        let off = 12 + i * 12;
        out[off..off + 12].copy_from_slice(&encode_leaf(*r));
    }
    Ok(out)
}

/// Pack `runs` into a depth-1 extent tree: an idx-node header + index
/// entries in the 60-byte `i_block` view, plus one or more leaf-block
/// images. The caller supplies `leaf_phys_blocks` — one physical block
/// number per leaf — typically allocated immediately before this call.
///
/// Returns `(i_block_bytes, leaf_images)`. `leaf_images[i]` is the byte
/// payload that should be staged at `leaf_phys_blocks[i]`.
///
/// Errors if `runs` would need more leaf blocks than `MAX_INDICES_IN_INODE`
/// (4) — depth > 1 is intentionally not supported here; with 4 leaves of
/// `entries_per_leaf_block_capped(bs, csum) * 32768` blocks each, depth-1
/// already covers more than any realistic single-file address space.
pub fn pack_depth1(
    runs: &[ExtentRun],
    block_size: u32,
    csum_tail: bool,
    leaf_phys_blocks: &[u32],
) -> crate::Result<([u8; 60], Vec<Vec<u8>>)> {
    let per_leaf = entries_per_leaf_block_capped(block_size, csum_tail);
    let need_leaves = runs.len().div_ceil(per_leaf);
    if need_leaves > MAX_INDICES_IN_INODE {
        return Err(crate::Error::Unsupported(format!(
            "ext4: depth-1 tree needs {need_leaves} leaf blocks, max {} (depth>1 not supported)",
            MAX_INDICES_IN_INODE
        )));
    }
    if leaf_phys_blocks.len() != need_leaves {
        return Err(crate::Error::InvalidArgument(format!(
            "pack_depth1: caller supplied {} leaf blocks, need {need_leaves}",
            leaf_phys_blocks.len()
        )));
    }
    let mut indices = Vec::with_capacity(need_leaves);
    let mut leaf_images = Vec::with_capacity(need_leaves);
    for (chunk_idx, chunk) in runs.chunks(per_leaf).enumerate() {
        indices.push(ExtentIdx {
            block: chunk[0].logical,
            leaf: leaf_phys_blocks[chunk_idx] as u64,
        });
        leaf_images.push(encode_leaf_block(chunk, block_size, csum_tail)?);
    }
    let i_block = pack_idx_into_iblock(&indices)?;
    Ok((i_block, leaf_images))
}

/// Decode a leaf block's runs. The header must claim depth == 0; any
/// other value is an internal-node block and the caller should walk the
/// tree further (not yet implemented for depth > 1).
pub fn decode_leaf_block(buf: &[u8]) -> crate::Result<(ExtentHeader, Vec<ExtentRun>)> {
    let header = decode_header(&buf[..12])?;
    if header.depth != 0 {
        return Err(crate::Error::Unsupported(format!(
            "ext4: nested extent block has depth {} (only depth 0 leaves supported)",
            header.depth
        )));
    }
    let max = entries_per_leaf_block(buf.len() as u32);
    if header.entries as usize > max {
        return Err(crate::Error::InvalidImage(format!(
            "ext4: leaf block claims {} entries, max {}",
            header.entries, max
        )));
    }
    let mut runs = Vec::with_capacity(header.entries as usize);
    for i in 0..header.entries as usize {
        let off = 12 + i * 12;
        runs.push(decode_leaf(&buf[off..off + 12]));
    }
    Ok((header, runs))
}

/// Decode the index entries from a 60-byte `i_block` view that represents
/// a depth-1 (or deeper) extent tree. Caller must verify
/// `header.depth >= 1` before treating `indices` as idx entries.
pub fn decode_idx_iblock(buf: &[u8; 60]) -> crate::Result<(ExtentHeader, Vec<ExtentIdx>)> {
    let header = decode_header(&buf[..12])?;
    if header.entries as usize > MAX_INDICES_IN_INODE {
        return Err(crate::Error::InvalidImage(format!(
            "ext4: inline extent header claims {} idx entries, max is {}",
            header.entries, MAX_INDICES_IN_INODE
        )));
    }
    let mut indices = Vec::with_capacity(header.entries as usize);
    for i in 0..header.entries as usize {
        let off = 12 + i * 12;
        indices.push(decode_idx(&buf[off..off + 12]));
    }
    Ok((header, indices))
}

/// Convert the 15 `u32` slots of an inode's `i_block` array into the
/// 60-byte view used by extent-tree encoders / decoders.
pub fn iblock_to_bytes(slots: &[u32; super::constants::N_BLOCKS]) -> [u8; 60] {
    let mut out = [0u8; 60];
    for (i, slot) in slots.iter().enumerate() {
        let off = i * 4;
        out[off..off + 4].copy_from_slice(&slot.to_le_bytes());
    }
    out
}

/// Inverse of [`iblock_to_bytes`]. Repacks a 60-byte extent-tree view
/// back into the 15-slot `i_block` array.
pub fn bytes_to_iblock(bytes: &[u8; 60]) -> [u32; super::constants::N_BLOCKS] {
    let mut out = [0u32; super::constants::N_BLOCKS];
    for (i, slot) in out.iter_mut().enumerate() {
        let off = i * 4;
        *slot = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
    }
    out
}

/// Decoded extent-tree header.
#[derive(Debug, Clone, Copy)]
pub struct ExtentHeader {
    pub entries: u16,
    pub max: u16,
    pub depth: u16,
}

/// Decode the 12-byte header at the start of an extent block / `i_block`.
/// Returns `Err` if the magic doesn't match.
pub fn decode_header(buf: &[u8]) -> crate::Result<ExtentHeader> {
    if buf.len() < 12 {
        return Err(crate::Error::InvalidImage(
            "ext4: extent header buffer too small".into(),
        ));
    }
    let magic = u16::from_le_bytes(buf[0..2].try_into().unwrap());
    if magic != EXT4_EXT_MAGIC {
        return Err(crate::Error::InvalidImage(format!(
            "ext4: extent header magic {magic:#06x} != {:#06x}",
            EXT4_EXT_MAGIC
        )));
    }
    Ok(ExtentHeader {
        entries: u16::from_le_bytes(buf[2..4].try_into().unwrap()),
        max: u16::from_le_bytes(buf[4..6].try_into().unwrap()),
        depth: u16::from_le_bytes(buf[6..8].try_into().unwrap()),
    })
}

/// Decode one 12-byte leaf-extent record at `buf`.
pub fn decode_leaf(buf: &[u8]) -> ExtentRun {
    let ee_block = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let ee_len = u16::from_le_bytes(buf[4..6].try_into().unwrap());
    let ee_start_hi = u16::from_le_bytes(buf[6..8].try_into().unwrap()) as u64;
    let ee_start_lo = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as u64;
    ExtentRun {
        logical: ee_block,
        len: ee_len,
        physical: (ee_start_hi << 32) | ee_start_lo,
    }
}

/// Decode a depth-0 extent tree from a 60-byte `i_block` view. Returns
/// `(header, runs)`. Caller must verify `header.depth == 0` before
/// treating `runs` as leaf entries.
pub fn decode_depth0_iblock(buf: &[u8; 60]) -> crate::Result<(ExtentHeader, Vec<ExtentRun>)> {
    let header = decode_header(&buf[..12])?;
    if header.entries as usize > MAX_EXTENTS_IN_INODE {
        return Err(crate::Error::InvalidImage(format!(
            "ext4: inline extent header claims {} entries, max is {}",
            header.entries, MAX_EXTENTS_IN_INODE
        )));
    }
    let mut runs = Vec::with_capacity(header.entries as usize);
    if header.depth == 0 {
        for i in 0..header.entries as usize {
            let off = 12 + i * 12;
            runs.push(decode_leaf(&buf[off..off + 12]));
        }
    }
    Ok((header, runs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_layout() {
        let h = encode_header(2, 4, 0);
        assert_eq!(&h[0..2], &EXT4_EXT_MAGIC.to_le_bytes());
        assert_eq!(u16::from_le_bytes(h[2..4].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(h[4..6].try_into().unwrap()), 4);
        assert_eq!(u16::from_le_bytes(h[6..8].try_into().unwrap()), 0);
    }

    #[test]
    fn leaf_layout() {
        let leaf = encode_leaf(ExtentRun {
            logical: 0,
            len: 12,
            physical: 0x1_2345_6789,
        });
        assert_eq!(u32::from_le_bytes(leaf[0..4].try_into().unwrap()), 0);
        assert_eq!(u16::from_le_bytes(leaf[4..6].try_into().unwrap()), 12);
        // Physical 0x1_2345_6789 = hi 0x0001, lo 0x2345_6789
        assert_eq!(u16::from_le_bytes(leaf[6..8].try_into().unwrap()), 0x0001);
        assert_eq!(
            u32::from_le_bytes(leaf[8..12].try_into().unwrap()),
            0x2345_6789
        );
    }

    #[test]
    fn coalesce_contiguous() {
        let blocks: Vec<u32> = (100..112).collect(); // 12 contiguous blocks
        let runs = coalesce(&blocks);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].logical, 0);
        assert_eq!(runs[0].len, 12);
        assert_eq!(runs[0].physical, 100);
    }

    #[test]
    fn coalesce_with_gap() {
        // 100..103, gap, 200..202
        let blocks = vec![100, 101, 102, 200, 201];
        let runs = coalesce(&blocks);
        assert_eq!(runs.len(), 2);
        assert_eq!(
            (runs[0].logical, runs[0].len, runs[0].physical),
            (0, 3, 100)
        );
        assert_eq!(
            (runs[1].logical, runs[1].len, runs[1].physical),
            (3, 2, 200)
        );
    }

    #[test]
    fn pack_rejects_too_many_extents() {
        let runs: Vec<_> = (0..5)
            .map(|i| ExtentRun {
                logical: i * 10,
                len: 1,
                physical: 1000 + i as u64 * 10,
            })
            .collect();
        let err = pack_into_iblock(&runs).unwrap_err();
        assert!(matches!(err, crate::Error::Unsupported(_)));
    }

    #[test]
    fn pack_roundtrip_one_extent() {
        let runs = vec![ExtentRun {
            logical: 0,
            len: 12,
            physical: 100,
        }];
        let packed = pack_into_iblock(&runs).unwrap();
        // Header at 0..12
        assert_eq!(&packed[0..2], &EXT4_EXT_MAGIC.to_le_bytes());
        // 1 entry
        assert_eq!(u16::from_le_bytes(packed[2..4].try_into().unwrap()), 1);
        // Max 4
        assert_eq!(u16::from_le_bytes(packed[4..6].try_into().unwrap()), 4);
        // Depth 0
        assert_eq!(u16::from_le_bytes(packed[6..8].try_into().unwrap()), 0);
        // Leaf at 12..24
        assert_eq!(u32::from_le_bytes(packed[12..16].try_into().unwrap()), 0);
        assert_eq!(u16::from_le_bytes(packed[16..18].try_into().unwrap()), 12);
        assert_eq!(u32::from_le_bytes(packed[20..24].try_into().unwrap()), 100);
    }
}
