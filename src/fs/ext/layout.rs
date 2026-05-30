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

use super::constants::{GROUP_DESC_SIZE, GROUP_DESC_SIZE_64, INODE_SIZE_DYNAMIC};

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
    /// On-disk size of each group descriptor: 32 (classic) or 64
    /// (`INCOMPAT_64BIT`). The writer emits 64 when
    /// `FormatOpts::use_64bit` is set; the reader picks up whichever
    /// size the on-disk `s_desc_size` advertises.
    pub desc_size: usize,
    /// Number of blocks the inode table occupies in each group.
    pub inode_table_blocks: u32,
    /// Number of GDT blocks (ceil(num_groups * desc_size / block_size)).
    pub gdt_blocks: u32,
    /// Base-2 logarithm of the number of groups per "flex unit" when the
    /// `INCOMPAT_FLEX_BG` feature is active. 0 means flex_bg is disabled.
    /// When non-zero, the first group of each flex unit packs the block
    /// bitmaps, inode bitmaps, and inode tables of all
    /// `2^log_groups_per_flex` groups in that unit contiguously; the
    /// remaining groups in the unit hold only data (and optional SB+GDT
    /// backups per `sparse_super`).
    pub log_groups_per_flex: u8,
    /// One entry per group, in order.
    pub groups: Vec<GroupLayout>,
}

impl Layout {
    /// Total number of groups.
    pub fn num_groups(&self) -> u32 {
        self.groups.len() as u32
    }

    /// Number of groups per flex unit, or 1 when flex_bg is disabled.
    pub fn flex_size(&self) -> u32 {
        if self.log_groups_per_flex == 0 {
            1
        } else {
            1u32 << self.log_groups_per_flex
        }
    }
}

/// Per-group layout (absolute block numbers).
#[derive(Debug, Clone, Copy)]
pub struct GroupLayout {
    /// First block (absolute) of this group.
    pub start_block: u32,
    /// Last block (absolute, inclusive) of this group.
    pub end_block: u32,
    /// Whether this group holds a superblock + GDT image (primary in
    /// group 0, backup elsewhere). The defaulting depends on
    /// [`SparseSuperMode`]: every group under `All`, the sparse rule
    /// under `Classic`, or just the two listed groups (plus group 0)
    /// under `Two`.
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

/// Whether group `g` (out of `num_groups`) holds a superblock + GDT
/// backup when `RO_COMPAT_SPARSE_SUPER` is active. The rule (from the
/// ext kernel docs): groups 0 and 1 always; otherwise only groups whose
/// number is a power of 3, 5, or 7.
pub fn group_has_sparse_super(g: u32) -> bool {
    if g <= 1 {
        return true;
    }
    is_power_of(g, 3) || is_power_of(g, 5) || is_power_of(g, 7)
}

fn is_power_of(mut n: u32, base: u32) -> bool {
    if n == 0 {
        return false;
    }
    while n.is_multiple_of(base) {
        n /= base;
    }
    n == 1
}

/// Compute a layout for `(block_size, blocks_count, inodes_count)`.
///
/// Returns [`crate::Error::InvalidArgument`] if the requested geometry cannot fit
/// the metadata overhead.
pub fn plan(block_size: u32, blocks_count: u32, inodes_count: u32) -> crate::Result<Layout> {
    plan_with(block_size, blocks_count, inodes_count, false)
}

/// Compute a layout, optionally enabling `RO_COMPAT_SPARSE_SUPER` — groups
/// 0, 1, and powers-of-3/5/7 hold backups; the rest skip them.
pub fn plan_with(
    block_size: u32,
    blocks_count: u32,
    inodes_count: u32,
    sparse_super: bool,
) -> crate::Result<Layout> {
    plan_full(block_size, blocks_count, inodes_count, sparse_super, 0)
}

/// Selects which (if any) groups carry SB+GDT backups, beyond the classic
/// "every group" / `RO_COMPAT_SPARSE_SUPER` rules.
#[derive(Debug, Clone, Copy, Default)]
pub enum SparseSuperMode {
    /// Backups in every group (classic ext2).
    #[default]
    All,
    /// `RO_COMPAT_SPARSE_SUPER`: groups 0, 1, and powers of 3/5/7.
    Classic,
    /// `sparse_super2`: backups in exactly the two groups listed
    /// (typically `[group 1, last group]`).
    Two([u32; 2]),
}

impl SparseSuperMode {
    /// Whether group `g` carries a SB+GDT image (primary or backup). Group
    /// 0 always does — it holds the *primary* superblock + GDT, regardless
    /// of `sparse_super` / `sparse_super2` settings. Other groups only
    /// carry the SB+GDT if the rules say so.
    pub fn group_has_backup(self, g: u32) -> bool {
        if g == 0 {
            return true;
        }
        match self {
            SparseSuperMode::All => true,
            SparseSuperMode::Classic => group_has_sparse_super(g),
            SparseSuperMode::Two([a, b]) => g == a || g == b,
        }
    }
}

/// Full layout planner with optional `INCOMPAT_FLEX_BG` packing.
///
/// `log_groups_per_flex == 0` disables flex_bg (classic per-group metadata
/// placement). Otherwise `2^log_groups_per_flex` groups form a flex unit;
/// the first group of each unit packs the bitmaps and inode tables of
/// every group in the unit contiguously, while the rest of the unit holds
/// only data (and, optionally, SB+GDT backups per `sparse_super`). The
/// on-disk format caps `log_groups_per_flex` at 5 (32 groups per unit).
pub fn plan_full(
    block_size: u32,
    blocks_count: u32,
    inodes_count: u32,
    sparse_super: bool,
    log_groups_per_flex: u8,
) -> crate::Result<Layout> {
    let mode = if sparse_super {
        SparseSuperMode::Classic
    } else {
        SparseSuperMode::All
    };
    plan_layout(
        block_size,
        blocks_count,
        inodes_count,
        mode,
        log_groups_per_flex,
        false,
    )
}

/// Layout planner with full opt-in feature surface: choice of sparse-super
/// mode, optional `INCOMPAT_FLEX_BG` packing, and optional `INCOMPAT_64BIT`
/// (which only affects `desc_size`/`gdt_blocks` — the on-disk layout itself
/// is otherwise unchanged for sub-2³² block filesystems).
pub fn plan_layout(
    block_size: u32,
    blocks_count: u32,
    inodes_count: u32,
    sparse_super_mode: SparseSuperMode,
    log_groups_per_flex: u8,
    use_64bit: bool,
) -> crate::Result<Layout> {
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
    if log_groups_per_flex > 5 {
        return Err(crate::Error::InvalidArgument(format!(
            "ext: log_groups_per_flex {log_groups_per_flex} > 5 (max flex unit = 32 groups)"
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
    if !blocks_per_group.is_multiple_of(8) {
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

    // GDT size: 32 bytes per descriptor in the classic layout, 64 with
    // `INCOMPAT_64BIT`. The wider descriptor pulls in the upper-half
    // bitmap/itable pointers and the bg_checksum_hi field.
    let desc_size = if use_64bit {
        GROUP_DESC_SIZE_64
    } else {
        GROUP_DESC_SIZE
    };
    let gdt_bytes = num_groups as u64 * desc_size as u64;
    let gdt_blocks = gdt_bytes.div_ceil(block_size as u64) as u32;

    // Cap the flex unit so the first group of each unit can hold the
    // packed bitmaps + inode tables of all `flex_size` members. With
    // small blocks and many inodes the per-group inode table is large, so
    // a nominal `log_groups_per_flex` of 4 (16 groups) would overflow the
    // first group; shrink it to the largest power of two that fits.
    let mut log_groups_per_flex = log_groups_per_flex;
    if log_groups_per_flex != 0 {
        // Worst case the unit's first group also carries a SB + GDT backup.
        let reserve = 1 + gdt_blocks;
        let per_member = 2 + inode_table_blocks; // block bitmap + inode bitmap + inode table
        let budget = blocks_per_group.saturating_sub(reserve);
        let max_members = (budget / per_member.max(1)).max(1);
        while (1u32 << log_groups_per_flex) > max_members && log_groups_per_flex > 0 {
            log_groups_per_flex -= 1;
        }
    }

    let flex_size: u32 = if log_groups_per_flex == 0 {
        1
    } else {
        1u32 << log_groups_per_flex
    };

    let mut groups: Vec<GroupLayout> = Vec::with_capacity(num_groups as usize);
    for g in 0..num_groups {
        let start = first_data_block + g * blocks_per_group;
        let nominal_end = start + blocks_per_group - 1;
        let end = nominal_end.min(blocks_count - 1);

        // Whether this group carries a SB+GDT backup. The classic ext2
        // case is "every group"; `sparse_super` keeps only groups 0, 1, and
        // powers of 3/5/7; `sparse_super2` keeps only the two listed groups.
        let has_sb = sparse_super_mode.group_has_backup(g);
        let sb_gdt_blocks: u32 = if has_sb { 1 + gdt_blocks } else { 0 };
        let local_meta_start = start + sb_gdt_blocks;

        let (block_bitmap, inode_bitmap, inode_table, data_start, meta_blocks);

        if log_groups_per_flex == 0 {
            // Classic per-group layout: SB?+GDT? -> bbm -> ibm -> itable -> data.
            block_bitmap = local_meta_start;
            inode_bitmap = local_meta_start + 1;
            inode_table = local_meta_start + 2;
            data_start = inode_table + inode_table_blocks;
            meta_blocks = data_start - start;
        } else {
            // flex_bg: per-group bitmap + inode-table live in flex_first.
            let flex_first = (g / flex_size) * flex_size;
            let pos_in_flex = g - flex_first;
            // Resolve the packed-region base for this flex unit. When g is
            // itself flex_first the first-of-flex layout isn't in `groups`
            // yet, so derive it from local state; otherwise look it up.
            let (first_start, first_has_sb) = if pos_in_flex == 0 {
                (start, has_sb)
            } else {
                let prev = &groups[flex_first as usize];
                (prev.start_block, prev.has_superblock)
            };
            let packed_base = first_start + if first_has_sb { 1 + gdt_blocks } else { 0 };
            // Layout inside `flex_first`:
            //   [SB?+GDT?] bbm[0..flex_size] ibm[0..flex_size] itable[0..flex_size] data
            let bbm_base = packed_base;
            let ibm_base = bbm_base + flex_size;
            let table_base = ibm_base + flex_size;
            block_bitmap = bbm_base + pos_in_flex;
            inode_bitmap = ibm_base + pos_in_flex;
            inode_table = table_base + pos_in_flex * inode_table_blocks;

            if pos_in_flex == 0 {
                // First-of-flex group owns the packed area.
                let packed_end = table_base + flex_size * inode_table_blocks;
                data_start = packed_end;
                meta_blocks = data_start - start;
            } else {
                // Non-first member: only its own SB+GDT backup (if any)
                // sits at the start; everything else is data.
                data_start = local_meta_start;
                meta_blocks = sb_gdt_blocks;
            }
        }

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

    // Sanity check: with flex_bg the packed metadata of each flex unit
    // must fit within the first group's address range. Otherwise e2fsck
    // would reject the image (and the writer would overflow the bitmap).
    if log_groups_per_flex != 0 {
        for first in (0..num_groups).step_by(flex_size as usize) {
            let g0 = &groups[first as usize];
            if g0.data_start > g0.end_block + 1 {
                return Err(crate::Error::InvalidArgument(format!(
                    "ext: flex_bg metadata for unit starting at group {} \
                     ({} blocks) exceeds its capacity ({} blocks); try a \
                     smaller log_groups_per_flex.",
                    first,
                    g0.data_start - g0.start_block,
                    g0.end_block + 1 - g0.start_block,
                )));
            }
        }
    }

    Ok(Layout {
        block_size,
        blocks_count,
        inodes_count,
        first_data_block,
        blocks_per_group,
        inodes_per_group,
        inode_size,
        desc_size,
        inode_table_blocks,
        gdt_blocks,
        log_groups_per_flex,
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
    let desc_size = sb.group_desc_size();
    let gdt_blocks = (num_groups as u64 * desc_size as u64).div_ceil(block_size as u64) as u32;

    // `sparse_super2` (compat 0x200) takes precedence over `sparse_super`:
    // it pins backups to the two listed groups regardless. Otherwise fall
    // back to the classic sparse-super rule, or "every group" if neither.
    let sparse_super2_on = sb.feature_compat & super::constants::feature::COMPAT_SPARSE_SUPER2 != 0;
    let sparse_super_on =
        sb.feature_ro_compat & super::constants::feature::RO_COMPAT_SPARSE_SUPER != 0;
    let sparse_super_mode = if sparse_super2_on {
        SparseSuperMode::Two(sb.backup_bgs)
    } else if sparse_super_on {
        SparseSuperMode::Classic
    } else {
        SparseSuperMode::All
    };
    let flex_bg_on = sb.feature_incompat & super::constants::feature::INCOMPAT_FLEX_BG != 0;
    let log_groups_per_flex = if flex_bg_on {
        sb.log_groups_per_flex
    } else {
        0
    };
    let flex_size: u32 = if log_groups_per_flex == 0 {
        1
    } else {
        1u32 << log_groups_per_flex
    };

    let mut groups: Vec<GroupLayout> = Vec::with_capacity(num_groups as usize);
    for g in 0..num_groups {
        let start = sb.first_data_block + g * sb.blocks_per_group;
        let nominal_end = start + sb.blocks_per_group - 1;
        let end = nominal_end.min(sb.blocks_count - 1);
        // Whether this group is expected to carry a SB+GDT backup. The
        // on-disk descriptor pointers (read by `Ext::open` and patched into
        // `layout.groups`) remain the source of truth for actual block
        // locations; `has_superblock` only controls where we *write* SB+GDT
        // backups on flush.
        let has_sb = sparse_super_mode.group_has_backup(g);
        let sb_gdt_blocks: u32 = if has_sb { 1 + gdt_blocks } else { 0 };
        let local_meta_start = start + sb_gdt_blocks;

        let (block_bitmap, inode_bitmap, inode_table, data_start, meta_blocks);
        if log_groups_per_flex == 0 {
            block_bitmap = local_meta_start;
            inode_bitmap = local_meta_start + 1;
            inode_table = local_meta_start + 2;
            data_start = inode_table + inode_table_blocks;
            meta_blocks = data_start - start;
        } else {
            let flex_first = (g / flex_size) * flex_size;
            let pos_in_flex = g - flex_first;
            let (first_start, first_has_sb) = if pos_in_flex == 0 {
                (start, has_sb)
            } else {
                let prev = &groups[flex_first as usize];
                (prev.start_block, prev.has_superblock)
            };
            let packed_base = first_start + if first_has_sb { 1 + gdt_blocks } else { 0 };
            let bbm_base = packed_base;
            let ibm_base = bbm_base + flex_size;
            let table_base = ibm_base + flex_size;
            block_bitmap = bbm_base + pos_in_flex;
            inode_bitmap = ibm_base + pos_in_flex;
            inode_table = table_base + pos_in_flex * inode_table_blocks;
            if pos_in_flex == 0 {
                let packed_end = table_base + flex_size * inode_table_blocks;
                data_start = packed_end;
                meta_blocks = data_start - start;
            } else {
                data_start = local_meta_start;
                meta_blocks = sb_gdt_blocks;
            }
        }

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
        desc_size,
        inode_table_blocks,
        gdt_blocks,
        log_groups_per_flex,
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

    #[test]
    fn flex_bg_packs_metadata_into_first_group() {
        // bs=4096, 4 groups of 32768 blocks; log_groups_per_flex = 2 →
        // one flex unit covers all 4 groups.
        let blocks_per_group = 32768u32;
        let total_blocks = 4 * blocks_per_group; // first_data_block = 0 for bs=4096
        let layout = plan_full(4096, total_blocks, 1024, false, 2).unwrap();
        assert_eq!(layout.num_groups(), 4);
        assert_eq!(layout.log_groups_per_flex, 2);
        assert_eq!(layout.flex_size(), 4);

        let g0 = &layout.groups[0];
        let g1 = &layout.groups[1];
        let g2 = &layout.groups[2];
        let g3 = &layout.groups[3];

        // Group 0 owns the packed bitmaps + inode tables for all 4 groups.
        let gdt = layout.gdt_blocks;
        let packed_base = 1 + gdt; // SB + GDT for bs=4096
        assert_eq!(g0.block_bitmap, packed_base);
        assert_eq!(g1.block_bitmap, packed_base + 1);
        assert_eq!(g2.block_bitmap, packed_base + 2);
        assert_eq!(g3.block_bitmap, packed_base + 3);
        assert_eq!(g0.inode_bitmap, packed_base + 4);
        assert_eq!(g1.inode_bitmap, packed_base + 5);
        assert_eq!(g2.inode_bitmap, packed_base + 6);
        assert_eq!(g3.inode_bitmap, packed_base + 7);
        let table_base = packed_base + 8;
        assert_eq!(g0.inode_table, table_base);
        assert_eq!(g1.inode_table, table_base + layout.inode_table_blocks);
        assert_eq!(g2.inode_table, table_base + 2 * layout.inode_table_blocks);
        assert_eq!(g3.inode_table, table_base + 3 * layout.inode_table_blocks);

        // sparse_super off ⇒ every group has SB+GDT in its own range; the
        // non-first flex member then has data_start = start + 1 + gdt.
        assert_eq!(g1.meta_blocks, 1 + gdt);
        assert_eq!(g1.data_start, g1.start_block + 1 + gdt);
        assert_eq!(g2.meta_blocks, 1 + gdt);
        assert_eq!(g3.meta_blocks, 1 + gdt);
    }

    #[test]
    fn flex_bg_with_sparse_super_skips_backups() {
        // With sparse_super, group 2 of flex unit 0 has no SB+GDT — only
        // its bitmap+table live elsewhere (in group 0); its data_start
        // equals its start_block. Group 3 IS a power of 3, so it carries
        // a backup.
        let blocks_per_group = 32768u32;
        let total_blocks = 4 * blocks_per_group;
        let layout = plan_full(4096, total_blocks, 1024, true, 2).unwrap();
        let g2 = &layout.groups[2];
        let g3 = &layout.groups[3];
        assert!(!g2.has_superblock);
        assert!(g3.has_superblock);
        assert_eq!(g2.meta_blocks, 0);
        assert_eq!(g2.data_start, g2.start_block);
        assert_eq!(g3.meta_blocks, 1 + layout.gdt_blocks);
    }

    #[test]
    fn flex_bg_rejects_too_large_log() {
        // 2^6 = 64, exceeds the spec cap of 32.
        let err = plan_full(4096, 32 * 1024, 256, false, 6).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    #[test]
    fn flex_bg_off_matches_classic_plan() {
        // log_groups_per_flex = 0 → identical to plan_with(.., false).
        let a = plan_full(1024, 32 * 1024, 256, false, 0).unwrap();
        let b = plan_with(1024, 32 * 1024, 256, false).unwrap();
        assert_eq!(a.num_groups(), b.num_groups());
        for i in 0..a.groups.len() {
            assert_eq!(a.groups[i].block_bitmap, b.groups[i].block_bitmap);
            assert_eq!(a.groups[i].inode_bitmap, b.groups[i].inode_bitmap);
            assert_eq!(a.groups[i].inode_table, b.groups[i].inode_table);
            assert_eq!(a.groups[i].data_start, b.groups[i].data_start);
            assert_eq!(a.groups[i].meta_blocks, b.groups[i].meta_blocks);
        }
    }

    #[test]
    fn use_64bit_widens_desc_size_and_gdt() {
        // INCOMPAT_64BIT moves the on-disk descriptor from 32 → 64 bytes,
        // which doubles the GDT footprint in blocks.
        let a = plan_layout(4096, 4 * 32768, 1024, SparseSuperMode::All, 0, false).unwrap();
        let b = plan_layout(4096, 4 * 32768, 1024, SparseSuperMode::All, 0, true).unwrap();
        assert_eq!(a.desc_size, GROUP_DESC_SIZE);
        assert_eq!(b.desc_size, GROUP_DESC_SIZE_64);
        // Both planners agree on num_groups; GDT byte count doubles.
        assert_eq!(a.num_groups(), b.num_groups());
    }

    #[test]
    fn sparse_super_mode_two_keeps_only_listed_groups() {
        // SparseSuperMode::Two([1, 3]) keeps backups in group 0 (primary),
        // 1, and 3. Group 2 must skip the SB+GDT prefix and start data
        // right at its first block.
        let layout = plan_layout(
            4096,
            4 * 32768,
            1024,
            SparseSuperMode::Two([1, 3]),
            0,
            false,
        )
        .unwrap();
        assert!(layout.groups[0].has_superblock);
        assert!(layout.groups[1].has_superblock);
        assert!(!layout.groups[2].has_superblock);
        assert!(layout.groups[3].has_superblock);
        // Even without SB+GDT, group 2 still owns its own bitmap +
        // inode-table prefix. The invariant we *can* check is that
        // group 2 saves exactly `1 + gdt_blocks` of metadata vs group 1
        // (which carries the backup).
        let sb_gdt = 1 + layout.gdt_blocks;
        assert_eq!(
            layout.groups[1].meta_blocks - layout.groups[2].meta_blocks,
            sb_gdt,
            "group 2 saves exactly 1 + gdt_blocks of metadata vs a group with a backup",
        );
    }

    #[test]
    fn sparse_super_mode_group_zero_is_always_primary() {
        // Even when group 0 isn't in the [a, b] backup list, group 0
        // always carries the primary SB+GDT.
        let mode = SparseSuperMode::Two([5, 9]);
        assert!(mode.group_has_backup(0));
        assert!(!mode.group_has_backup(2));
        assert!(mode.group_has_backup(5));
        assert!(mode.group_has_backup(9));
    }
}
