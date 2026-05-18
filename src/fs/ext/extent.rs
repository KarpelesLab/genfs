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

/// Collapse a sorted list of `(logical, physical)` mappings into the
/// minimum-length sequence of contiguous extents (same as how genext2fs
/// / mke2fs / kernel would).
///
/// Input must be sorted by `logical`; physical blocks are read straight
/// through.
pub fn coalesce(data_blocks: &[u32]) -> Vec<ExtentRun> {
    let mut out: Vec<ExtentRun> = Vec::new();
    for (i, &phys) in data_blocks.iter().enumerate() {
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
