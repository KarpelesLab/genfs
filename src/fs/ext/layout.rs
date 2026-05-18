//! Geometry computation for an ext2 filesystem.
//!
//! Given the high-level inputs (`block_size`, `blocks_count`, `inodes_count`),
//! this module decides:
//!
//! - the value of `first_data_block` (1 for 1 KiB blocks, 0 otherwise);
//! - `blocks_per_group` and how many groups the filesystem will have;
//! - `inodes_per_group` and how many blocks each group's inode table occupies;
//! - for each group, the absolute block numbers of its bitmap, inode bitmap,
//!   inode table, and the first / last block of its data area.
//!
//! All of this is a pure function so we can unit-test the layout decisions
//! against known-good genext2fs / mke2fs outputs.

use super::constants::{GROUP_DESC_SIZE, INODE_SIZE_DYNAMIC};

/// Result of [`plan`]. Holds the choices the writer needs to lay out an
/// ext2 image; one [`GroupLayout`] per block group.
#[derive(Debug, Clone)]
pub struct Layout {
    pub block_size: u32,
    pub blocks_count: u32,
    pub inodes_count: u32,
    pub first_data_block: u32,
    pub blocks_per_group: u32,
    pub inodes_per_group: u32,
    pub inode_size: u16,
    /// Number of blocks the inode table occupies in each group.
    pub inode_table_blocks: u32,
    /// Number of GDT blocks (ceil(num_groups * 32 / block_size)).
    pub gdt_blocks: u32,
    /// One entry per group, in order.
    pub groups: Vec<GroupLayout>,
}

impl Layout {
    /// Total number of groups.
    pub fn num_groups(&self) -> u32 {
        self.groups.len() as u32
    }
}

/// Per-group layout (absolute block numbers).
#[derive(Debug, Clone, Copy)]
pub struct GroupLayout {
    /// First block (absolute) of this group.
    pub start_block: u32,
    /// Last block (absolute, inclusive) of this group.
    pub end_block: u32,
    /// Whether this group holds a superblock + GDT copy. For v1 every group
    /// gets a copy (no SPARSE_SUPER).
    pub has_superblock: bool,
    /// Absolute block number of this group's block bitmap.
    pub block_bitmap: u32,
    /// Absolute block number of this group's inode bitmap.
    pub inode_bitmap: u32,
    /// Absolute block number of the first block of this group's inode table.
    pub inode_table: u32,
    /// First absolute data block in this group (after all metadata).
    pub data_start: u32,
    /// Number of metadata blocks occupied in this group (superblock + GDT +
    /// bitmaps + inode table).
    pub meta_blocks: u32,
}

/// Compute a layout for `(block_size, blocks_count, inodes_count)`.
///
/// Returns [`crate::Error::InvalidArgument`] if the requested geometry cannot fit
/// the metadata overhead.
pub fn plan(block_size: u32, blocks_count: u32, inodes_count: u32) -> crate::Result<Layout> {
    if !block_size.is_power_of_two() || block_size < 1024 {
        return Err(crate::Error::InvalidArgument(format!(
            "ext: block_size must be a power of two ≥ 1024, got {block_size}"
        )));
    }
    if blocks_count < 32 {
        return Err(crate::Error::InvalidArgument(format!(
            "ext: blocks_count {blocks_count} too small"
        )));
    }
    if inodes_count < 11 {
        return Err(crate::Error::InvalidArgument(format!(
            "ext: inodes_count must include the reserved range (≥ 11), got {inodes_count}"
        )));
    }

    // first_data_block: where the bitmap-tracked region begins.
    // For 1 KiB blocks the boot block 0 is outside the filesystem proper; for
    // larger block sizes block 0 contains both the boot region and the
    // superblock (the SB lives at offset 1024 of block 0).
    let first_data_block: u32 = if block_size == 1024 { 1 } else { 0 };

    // The bitmap covers up to 8 * block_size blocks. genext2fs reports
    // blocks_per_group = min(8 * block_size, blocks_count) — note: NOT
    // subtracting first_data_block. The bit past the actual end of the disk
    // is marked used as a sentinel.
    let max_per_group = 8 * block_size;
    let blocks_per_group = max_per_group.min(blocks_count);

    // blocks_per_group MUST be a multiple of 8: the block bitmap is checked
    // byte-aligned per group, and e2fsck rejects a non-byte-aligned group
    // size ("Padding at end of block bitmap is not set"). The multi-group
    // case uses 8*block_size which is always a multiple of 8; the small
    // single-group case uses blocks_count directly, so blocks_count must be
    // a multiple of 8 there.
    if blocks_per_group % 8 != 0 {
        return Err(crate::Error::InvalidArgument(format!(
            "ext: blocks_count {blocks_count} must be a multiple of 8 for a \
             single-group filesystem (blocks_per_group must be byte-aligned)"
        )));
    }

    // Group 0 covers [first_data_block, first_data_block + blocks_per_group)
    // intersected with [0, blocks_count). Subsequent groups follow.
    let group_input_blocks = blocks_count - first_data_block;
    let num_groups = group_input_blocks.div_ceil(blocks_per_group);

    // Round inodes_count up so each group has a whole multiple of 8 inodes
    // (the bitmap needs full bytes).
    let mut inodes_per_group = inodes_count.div_ceil(num_groups);
    // Bitmap fits 8 * block_size inodes max per group.
    let max_inodes_per_group = 8 * block_size;
    if inodes_per_group > max_inodes_per_group {
        return Err(crate::Error::InvalidArgument(format!(
            "ext: too many inodes per group ({inodes_per_group}, max {max_inodes_per_group})"
        )));
    }
    // Round up to multiple of 8 (genext2fs uses this).
    inodes_per_group = inodes_per_group.div_ceil(8) * 8;
    // Recompute total inodes_count from per_group * num_groups so the
    // bitmaps are exactly full.
    let inodes_count = inodes_per_group * num_groups;

    let inode_size = INODE_SIZE_DYNAMIC;
    let inode_table_bytes = inodes_per_group as u64 * inode_size as u64;
    let inode_table_blocks = inode_table_bytes.div_ceil(block_size as u64) as u32;

    // GDT size: 32 bytes per descriptor.
    let gdt_bytes = num_groups as u64 * GROUP_DESC_SIZE as u64;
    let gdt_blocks = gdt_bytes.div_ceil(block_size as u64) as u32;

    let mut groups = Vec::with_capacity(num_groups as usize);
    for g in 0..num_groups {
        let start = first_data_block + g * blocks_per_group;
        let nominal_end = start + blocks_per_group - 1;
        let end = nominal_end.min(blocks_count - 1);

        // Every group gets a superblock + GDT copy in v1 (no SPARSE_SUPER).
        let has_sb = true;

        // Block layout within the group:
        //   [SB? + GDT?] -> block bitmap -> inode bitmap -> inode table -> data
        // The superblock copy occupies one block. For group 0 with 1 KiB
        // blocks the superblock is at block 1 (first_data_block).
        // For groups other than 0, the SB copy starts at the group's first
        // block.
        let mut next = start;
        if has_sb {
            // SB block + GDT blocks
            next += 1 + gdt_blocks;
        }
        let block_bitmap = next;
        let inode_bitmap = next + 1;
        let inode_table = next + 2;
        let data_start = inode_table + inode_table_blocks;
        let meta_blocks = data_start - start;

        groups.push(GroupLayout {
            start_block: start,
            end_block: end,
            has_superblock: has_sb,
            block_bitmap,
            inode_bitmap,
            inode_table,
            data_start,
            meta_blocks,
        });
    }

    Ok(Layout {
        block_size,
        blocks_count,
        inodes_count,
        first_data_block,
        blocks_per_group,
        inodes_per_group,
        inode_size,
        inode_table_blocks,
        gdt_blocks,
        groups,
    })
}

/// Build a [`Layout`] from a parsed [`super::superblock::Superblock`]. Used
/// by [`super::Ext::open`] to reconstruct the geometry of an existing image
/// without re-running the planner's defaulting heuristics.
pub fn from_superblock(sb: &super::superblock::Superblock) -> crate::Result<Layout> {
    let block_size = sb.block_size();
    if !block_size.is_power_of_two() || block_size < 1024 {
        return Err(crate::Error::InvalidImage(format!(
            "ext: bad block_size {block_size}"
        )));
    }
    let group_input_blocks = sb.blocks_count - sb.first_data_block;
    let num_groups = group_input_blocks.div_ceil(sb.blocks_per_group);

    let inode_table_blocks =
        (sb.inodes_per_group as u64 * sb.inode_size as u64).div_ceil(block_size as u64) as u32;
    let gdt_blocks =
        (num_groups as u64 * GROUP_DESC_SIZE as u64).div_ceil(block_size as u64) as u32;

    let mut groups = Vec::with_capacity(num_groups as usize);
    for g in 0..num_groups {
        let start = sb.first_data_block + g * sb.blocks_per_group;
        let nominal_end = start + sb.blocks_per_group - 1;
        let end = nominal_end.min(sb.blocks_count - 1);
        // For an existing image we don't yet know which groups have a
        // superblock copy (depends on SPARSE_SUPER); for v1 we assume every
        // group has one when reading too. The descriptor's bitmap/table
        // pointers (which we read from disk separately) are the source of
        // truth for actual positions.
        let has_sb = true;
        let meta_first = start + if has_sb { 1 + gdt_blocks } else { 0 };
        let block_bitmap = meta_first;
        let inode_bitmap = meta_first + 1;
        let inode_table = meta_first + 2;
        let data_start = inode_table + inode_table_blocks;
        let meta_blocks = data_start - start;
        groups.push(GroupLayout {
            start_block: start,
            end_block: end,
            has_superblock: has_sb,
            block_bitmap,
            inode_bitmap,
            inode_table,
            data_start,
            meta_blocks,
        });
    }

    Ok(Layout {
        block_size,
        blocks_count: sb.blocks_count,
        inodes_count: sb.inodes_count,
        first_data_block: sb.first_data_block,
        blocks_per_group: sb.blocks_per_group,
        inodes_per_group: sb.inodes_per_group,
        inode_size: sb.inode_size,
        inode_table_blocks,
        gdt_blocks,
        groups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Matches the genext2fs reference: -B 1024 -b 1024 with auto inode count.
    #[test]
    fn single_group_1kib() {
        let layout = plan(1024, 1024, 16).unwrap();
        assert_eq!(layout.first_data_block, 1);
        // genext2fs sets blocks_per_group = blocks_count for a short
        // single-group FS (NOT blocks_count - first_data_block).
        assert_eq!(layout.blocks_per_group, 1024);
        assert_eq!(layout.num_groups(), 1);
        assert_eq!(layout.inodes_per_group, 16);
        assert_eq!(layout.inode_size, 128);
        assert_eq!(layout.inode_table_blocks, 2);
        assert_eq!(layout.gdt_blocks, 1);

        let g0 = &layout.groups[0];
        assert_eq!(g0.start_block, 1);
        assert_eq!(g0.end_block, 1023);
        assert!(g0.has_superblock);
        assert_eq!(g0.block_bitmap, 3); // SB(1) + GDT(2) → bitmap at 3
        assert_eq!(g0.inode_bitmap, 4);
        assert_eq!(g0.inode_table, 5);
        assert_eq!(g0.data_start, 7); // inode table = 2 blocks, ends at 6
        assert_eq!(g0.meta_blocks, 6);
    }

    #[test]
    fn multiple_groups_1kib() {
        // 32 MiB FS with 1 KiB blocks → 4 groups of 8192 blocks each.
        let layout = plan(1024, 32 * 1024, 256).unwrap();
        assert_eq!(layout.num_groups(), 4);
        assert_eq!(layout.blocks_per_group, 8192);

        let g0 = &layout.groups[0];
        let g1 = &layout.groups[1];
        assert_eq!(g0.start_block, 1);
        assert_eq!(g0.end_block, 8192);
        assert_eq!(g1.start_block, 8193);
        assert_eq!(g1.end_block, 16384);
    }

    #[test]
    fn block_size_4096() {
        // 4 KiB blocks: first_data_block = 0; SB lives at offset 1024 of
        // block 0; GDT starts at block 1.
        let layout = plan(4096, 1024, 64).unwrap();
        assert_eq!(layout.first_data_block, 0);
        // 1024 blocks < max (32768), so blocks_per_group = blocks_count.
        assert_eq!(layout.blocks_per_group, 1024);
        assert_eq!(layout.num_groups(), 1);
    }

    #[test]
    fn rejects_invalid_block_size() {
        assert!(plan(512, 1024, 16).is_err());
        assert!(plan(3000, 1024, 16).is_err()); // not a power of two
    }

    #[test]
    fn rejects_too_few_blocks() {
        assert!(plan(1024, 8, 16).is_err());
    }

    #[test]
    fn inodes_round_up_to_multiple_of_8() {
        let layout = plan(1024, 1024, 11).unwrap();
        assert_eq!(layout.inodes_per_group % 8, 0);
        assert!(layout.inodes_per_group >= 11);
    }
}
