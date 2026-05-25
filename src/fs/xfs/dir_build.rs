//! XFS v5 multi-block directory builders (writer side).
//!
//! The block-format directory writer ([`super::dir::encode_v5_block_dir`])
//! packs every entry plus an inline hash array into a single directory
//! block, which caps a directory at a few hundred entries. This module
//! builds the **leaf** and **node** formats the kernel uses for larger
//! directories, laid out across the directory's segmented logical address
//! space:
//!
//! - **data** blocks at logical dir-block offsets `[0, LEAF_FIRSTDB)` hold
//!   the entries (magic `XDD3`, `xfs_dir3_blk_hdr` 48 B + `bestfree[3]`).
//! - **leaf** space at `[LEAF_FIRSTDB, FREE_FIRSTDB)` holds the hash index:
//!   one `XDLF` leaf1 block (leaf format), or one `XDN3` da-node root over
//!   many `XDLN` leafn blocks (node format).
//! - **free** space at `[FREE_FIRSTDB, …)` holds `XDF3` blocks carrying the
//!   `bests` array (one `__be16` per data block) for node format.
//!
//! Leaf/node blocks use the 56-byte `xfs_da3_blkinfo` header (16-bit magic
//! at offset 8, CRC at offset 12); data/free blocks use the 48-byte
//! `xfs_dir3_blk_hdr` (32-bit magic at offset 0, CRC at offset 4). The
//! `dataptr` stored in each leaf entry is `(db * dirblksize + off) / 8`.

use crate::Result;

use super::dir::{
    V5_DATA_HDR_SIZE, XFS_DIR2_DATA_FREE_TAG, XFS_DIR2_LEAF_FIRSTDB_BYTES, XFS_DIR3_DATA_MAGIC,
    dahashname, stamp_v5_dir_block_crc,
};

/// v5 da-block magics (16-bit, stored at offset 8 of `xfs_da3_blkinfo`).
pub const XFS_DIR3_LEAF1_MAGIC: u16 = 0x3df1;
pub const XFS_DIR3_LEAFN_MAGIC: u16 = 0x3dff;
pub const XFS_DA3_NODE_MAGIC: u16 = 0x3ebe;
/// v5 free-block magic (32-bit "XDF3", `xfs_dir3_blk_hdr` style).
pub const XFS_DIR3_FREE_MAGIC: u32 = 0x5844_4633;

/// `xfs_da3_blkinfo` size and CRC offset (leaf / node blocks).
const DA3_CRC_OFFSET: usize = 12;
/// `xfs_dir3_leaf_hdr` / `xfs_da3_node_hdr` size: 56 B blkinfo + 8 B.
const LEAF_HDR_SIZE: usize = 64;
/// `xfs_dir3_free_hdr` size: 48 B blk-hdr + firstdb/nvalid/nused/pad.
const FREE_HDR_SIZE: usize = 64;
/// `xfs_dir2_leaf_tail` size (`__be32 bestcount`) at the end of a leaf1.
const LEAF_TAIL_SIZE: usize = 4;

/// Which on-disk format a set of entries was laid out into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirFormat {
    /// Single block-format directory (caller uses `encode_v5_block_dir`).
    Block,
    /// Data blocks + one `XDLF` leaf1 block.
    Leaf,
    /// Data blocks + `XDN3` node root over `XDLN` leafn blocks + `XDF3`.
    Node,
}

/// Layout decision for a directory: how entries partition into data
/// blocks, the sorted leaf (hash, dataptr) index, per-data-block bests,
/// and — for node format — how the index splits across leafn blocks.
#[derive(Debug, Clone)]
pub struct DirLayout {
    pub format: DirFormat,
    /// Entries assigned to each data block (logical db == index).
    pub data_blocks: Vec<Vec<(String, u64, u8)>>,
    /// bestfree[0].length (free slack) of each data block.
    pub bests: Vec<u16>,
    /// Sorted (hashval, dataptr) over every entry.
    pub leaf_ents: Vec<(u32, u32)>,
    /// Node format: number of leaf entries in each leafn block.
    pub leafn_counts: Vec<usize>,
}

impl DirLayout {
    pub fn n_data(&self) -> usize {
        self.data_blocks.len()
    }
    /// Number of blocks in the leaf address space: 1 (leaf1) or
    /// `1 + leafn_count` (node: root + leaves).
    pub fn n_leafspace(&self) -> usize {
        match self.format {
            DirFormat::Leaf => 1,
            DirFormat::Node => 1 + self.leafn_counts.len(),
            DirFormat::Block => 0,
        }
    }
    /// Number of free-space blocks (node format only).
    pub fn n_free(&self) -> usize {
        match self.format {
            DirFormat::Node => 1,
            _ => 0,
        }
    }
}

/// Size of one packed data-entry record (8-byte aligned).
fn entry_record_len(namelen: usize) -> usize {
    // inumber(8) + namelen(1) + name + ftype(1) + tag(2), padded to 8.
    let raw = 8 + 1 + namelen + 1 + 2;
    (raw + 7) & !7
}

/// Plan the on-disk layout for `entries` (which must already include the
/// synthetic `.` and `..` records, in that order at the front). Computes
/// the data-block partition, the sorted hash index, per-block bests, and
/// the leaf-vs-node decision. Pure — no allocation or byte emission.
pub fn plan_layout(entries: &[(String, u64, u8)], dir_block_size: usize) -> Result<DirLayout> {
    // Does everything fit a single block-format block? That block holds the
    // header, the packed records, the inline leaf array (`count * 8`) and
    // the 8-byte block tail. If so the caller keeps using the cheaper
    // `encode_v5_block_dir` path.
    let count = entries.len();
    let records: usize = entries.iter().map(|e| entry_record_len(e.0.len())).sum();
    if V5_DATA_HDR_SIZE + records + count * 8 + 8 <= dir_block_size {
        return Ok(DirLayout {
            format: DirFormat::Block,
            data_blocks: vec![entries.to_vec()],
            bests: Vec::new(),
            leaf_ents: Vec::new(),
            leafn_counts: Vec::new(),
        });
    }

    // Partition entries into data blocks (greedy fill). Track each entry's
    // in-block byte offset so we can form its dataptr = (db*bsize+off)/8.
    let mut data_blocks: Vec<Vec<(String, u64, u8)>> = Vec::new();
    let mut bests: Vec<u16> = Vec::new();
    let mut leaf_ents: Vec<(u32, u32)> = Vec::new();

    let mut cur: Vec<(String, u64, u8)> = Vec::new();
    let mut pos = V5_DATA_HDR_SIZE;
    let db_ptr_base = |db: usize, off: usize| -> u32 { ((db * dir_block_size + off) / 8) as u32 };

    for ent in entries {
        let namelen = ent.0.len();
        if namelen == 0 || namelen > 255 {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: bad dir entry namelen {namelen}"
            )));
        }
        let rec = entry_record_len(namelen);
        if pos + rec > dir_block_size {
            // Close current data block (trailing slack becomes bestfree).
            let slack = dir_block_size - pos;
            bests.push(slack as u16);
            data_blocks.push(std::mem::take(&mut cur));
            pos = V5_DATA_HDR_SIZE;
        }
        let db = data_blocks.len();
        leaf_ents.push((dahashname(ent.0.as_bytes()), db_ptr_base(db, pos)));
        cur.push(ent.clone());
        pos += rec;
    }
    // Flush the final data block.
    let slack = dir_block_size - pos;
    bests.push(slack as u16);
    data_blocks.push(cur);

    // Sort the hash index (ascending hashval, ties by dataptr).
    leaf_ents.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Leaf format if every hash entry plus the bests array + tail fit one
    // leaf1 block; otherwise node format.
    let n_data = data_blocks.len();
    let leaf1_capacity = dir_block_size - LEAF_HDR_SIZE - LEAF_TAIL_SIZE - n_data * 2;
    let format;
    let mut leafn_counts = Vec::new();
    if leaf_ents.len() * 8 <= leaf1_capacity {
        format = DirFormat::Leaf;
    } else {
        format = DirFormat::Node;
        let per_leaf = (dir_block_size - LEAF_HDR_SIZE) / 8;
        let mut remaining = leaf_ents.len();
        while remaining > 0 {
            let take = remaining.min(per_leaf);
            leafn_counts.push(take);
            remaining -= take;
        }
    }

    Ok(DirLayout {
        format,
        data_blocks,
        bests,
        leaf_ents,
        leafn_counts,
    })
}

/// Logical dir-block index where the leaf address space starts.
pub fn leaf_firstdb(dir_block_size: usize) -> u64 {
    XFS_DIR2_LEAF_FIRSTDB_BYTES / dir_block_size as u64
}

/// Logical dir-block index where the free address space starts (64 GiB).
pub fn free_firstdb(dir_block_size: usize) -> u64 {
    2 * XFS_DIR2_LEAF_FIRSTDB_BYTES / dir_block_size as u64
}

/// Build one `XDD3` data block holding `entries` at logical block
/// `db_index`. Identical packing to [`plan_layout`].
pub fn build_data_block(
    entries: &[(String, u64, u8)],
    dir_block_size: usize,
    owner: u64,
    uuid: &[u8; 16],
    blkno_bbs: u64,
) -> Result<Vec<u8>> {
    let mut block = vec![0u8; dir_block_size];
    block[0..4].copy_from_slice(&XFS_DIR3_DATA_MAGIC.to_be_bytes());
    block[8..16].copy_from_slice(&blkno_bbs.to_be_bytes());
    block[24..40].copy_from_slice(uuid);
    block[40..48].copy_from_slice(&owner.to_be_bytes());

    let mut pos = V5_DATA_HDR_SIZE;
    for (name, inum, ft) in entries {
        let namelen = name.len();
        let padded = entry_record_len(namelen);
        if pos + padded > dir_block_size {
            return Err(crate::Error::InvalidArgument(
                "xfs: build_data_block: entries overflow block".into(),
            ));
        }
        block[pos..pos + 8].copy_from_slice(&inum.to_be_bytes());
        block[pos + 8] = namelen as u8;
        block[pos + 9..pos + 9 + namelen].copy_from_slice(name.as_bytes());
        block[pos + 9 + namelen] = *ft;
        let tag = (pos as u16).to_be_bytes();
        block[pos + padded - 2..pos + padded].copy_from_slice(&tag);
        pos += padded;
    }
    if pos < dir_block_size {
        let slack = dir_block_size - pos;
        block[pos..pos + 2].copy_from_slice(&XFS_DIR2_DATA_FREE_TAG.to_be_bytes());
        block[pos + 2..pos + 4].copy_from_slice(&(slack as u16).to_be_bytes());
        let tag_off = pos + slack - 2;
        block[tag_off..tag_off + 2].copy_from_slice(&(pos as u16).to_be_bytes());
        block[48..50].copy_from_slice(&(pos as u16).to_be_bytes());
        block[50..52].copy_from_slice(&(slack as u16).to_be_bytes());
    }
    stamp_v5_dir_block_crc(&mut block);
    Ok(block)
}

/// Stamp the v5 `xfs_da3_blkinfo` CRC (offset 12), then return.
fn stamp_da3_crc(block: &mut [u8]) {
    block[DA3_CRC_OFFSET..DA3_CRC_OFFSET + 4].copy_from_slice(&[0u8; 4]);
    let crc = crc32c::crc32c(block);
    block[DA3_CRC_OFFSET..DA3_CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
}

/// Write a 56-byte `xfs_da3_blkinfo` header at the front of `block`.
fn write_da3_blkinfo(
    block: &mut [u8],
    magic: u16,
    forw: u32,
    back: u32,
    blkno_bbs: u64,
    owner: u64,
    uuid: &[u8; 16],
) {
    block[0..4].copy_from_slice(&forw.to_be_bytes());
    block[4..8].copy_from_slice(&back.to_be_bytes());
    block[8..10].copy_from_slice(&magic.to_be_bytes());
    // pad [10..12], crc [12..16] (stamped later)
    block[16..24].copy_from_slice(&blkno_bbs.to_be_bytes());
    // lsn [24..32] zero
    block[32..48].copy_from_slice(uuid);
    block[48..56].copy_from_slice(&owner.to_be_bytes());
}

/// Build the single `XDLF` leaf1 block: header + sorted `ents[]` + trailing
/// `bests[]` (one `__be16` per data block) + `xfs_dir2_leaf_tail`.
pub fn build_leaf1_block(
    ents: &[(u32, u32)],
    bests: &[u16],
    dir_block_size: usize,
    owner: u64,
    uuid: &[u8; 16],
    blkno_bbs: u64,
) -> Result<Vec<u8>> {
    let mut block = vec![0u8; dir_block_size];
    write_da3_blkinfo(
        &mut block,
        XFS_DIR3_LEAF1_MAGIC,
        0,
        0,
        blkno_bbs,
        owner,
        uuid,
    );
    block[56..58].copy_from_slice(&(ents.len() as u16).to_be_bytes()); // count
    // stale [58..60] = 0, pad [60..64] = 0
    let ents_end = LEAF_HDR_SIZE + ents.len() * 8;
    let bests_start = dir_block_size - LEAF_TAIL_SIZE - bests.len() * 2;
    if ents_end > bests_start {
        return Err(crate::Error::InvalidArgument(
            "xfs: build_leaf1_block: index does not fit one leaf block".into(),
        ));
    }
    let mut p = LEAF_HDR_SIZE;
    for (hash, addr) in ents {
        block[p..p + 4].copy_from_slice(&hash.to_be_bytes());
        block[p + 4..p + 8].copy_from_slice(&addr.to_be_bytes());
        p += 8;
    }
    // bests[] grow up from bests_start; leaf_tail.bestcount at the very end.
    let mut bp = bests_start;
    for b in bests {
        block[bp..bp + 2].copy_from_slice(&b.to_be_bytes());
        bp += 2;
    }
    block[dir_block_size - 4..dir_block_size].copy_from_slice(&(bests.len() as u32).to_be_bytes());
    stamp_da3_crc(&mut block);
    Ok(block)
}

/// Build one `XDLN` leafn block holding `ents[]` (no tail/bests). `forw` /
/// `back` chain sibling leafn blocks by logical da-block address.
pub fn build_leafn_block(
    ents: &[(u32, u32)],
    dir_block_size: usize,
    owner: u64,
    uuid: &[u8; 16],
    blkno_bbs: u64,
    forw: u32,
    back: u32,
) -> Result<Vec<u8>> {
    let mut block = vec![0u8; dir_block_size];
    write_da3_blkinfo(
        &mut block,
        XFS_DIR3_LEAFN_MAGIC,
        forw,
        back,
        blkno_bbs,
        owner,
        uuid,
    );
    block[56..58].copy_from_slice(&(ents.len() as u16).to_be_bytes());
    let ents_end = LEAF_HDR_SIZE + ents.len() * 8;
    if ents_end > dir_block_size {
        return Err(crate::Error::InvalidArgument(
            "xfs: build_leafn_block: too many entries".into(),
        ));
    }
    let mut p = LEAF_HDR_SIZE;
    for (hash, addr) in ents {
        block[p..p + 4].copy_from_slice(&hash.to_be_bytes());
        block[p + 4..p + 8].copy_from_slice(&addr.to_be_bytes());
        p += 8;
    }
    stamp_da3_crc(&mut block);
    Ok(block)
}

/// Build the `XDN3` da-node root over the leafn blocks. `children` is
/// `(hashval, before)` per leafn: `hashval` is that leaf's highest hash,
/// `before` its logical da-block address. `level` is 1 (above the leaves).
pub fn build_da_node_block(
    children: &[(u32, u32)],
    dir_block_size: usize,
    owner: u64,
    uuid: &[u8; 16],
    blkno_bbs: u64,
) -> Result<Vec<u8>> {
    let mut block = vec![0u8; dir_block_size];
    write_da3_blkinfo(&mut block, XFS_DA3_NODE_MAGIC, 0, 0, blkno_bbs, owner, uuid);
    block[56..58].copy_from_slice(&(children.len() as u16).to_be_bytes()); // count
    block[58..60].copy_from_slice(&1u16.to_be_bytes()); // level
    // pad [60..64]
    let ents_end = LEAF_HDR_SIZE + children.len() * 8;
    if ents_end > dir_block_size {
        return Err(crate::Error::InvalidArgument(
            "xfs: build_da_node_block: too many children for one node".into(),
        ));
    }
    let mut p = LEAF_HDR_SIZE;
    for (hash, before) in children {
        block[p..p + 4].copy_from_slice(&hash.to_be_bytes());
        block[p + 4..p + 8].copy_from_slice(&before.to_be_bytes());
        p += 8;
    }
    stamp_da3_crc(&mut block);
    Ok(block)
}

/// Build one `XDF3` free block carrying `bests` (one `__be16` per data
/// block), covering data blocks starting at `firstdb`.
pub fn build_free_block(
    bests: &[u16],
    firstdb: u32,
    dir_block_size: usize,
    owner: u64,
    uuid: &[u8; 16],
    blkno_bbs: u64,
) -> Result<Vec<u8>> {
    let mut block = vec![0u8; dir_block_size];
    block[0..4].copy_from_slice(&XFS_DIR3_FREE_MAGIC.to_be_bytes());
    block[8..16].copy_from_slice(&blkno_bbs.to_be_bytes());
    block[24..40].copy_from_slice(uuid);
    block[40..48].copy_from_slice(&owner.to_be_bytes());
    block[48..52].copy_from_slice(&firstdb.to_be_bytes());
    block[52..56].copy_from_slice(&(bests.len() as u32).to_be_bytes()); // nvalid
    // `nused` counts bests slots that refer to a live data block — every
    // entry except the NULLDATAOFF (0xffff) holes left by removed blocks.
    // A *full* data block has best == 0 but is still live, so it counts.
    let nused = bests.iter().filter(|&&b| b != 0xffff).count() as u32;
    block[56..60].copy_from_slice(&nused.to_be_bytes());
    // pad [60..64]
    let ents_end = FREE_HDR_SIZE + bests.len() * 2;
    if ents_end > dir_block_size {
        return Err(crate::Error::InvalidArgument(
            "xfs: build_free_block: too many bests for one free block".into(),
        ));
    }
    let mut p = FREE_HDR_SIZE;
    for b in bests {
        block[p..p + 2].copy_from_slice(&b.to_be_bytes());
        p += 2;
    }
    // Free blocks use the `xfs_dir3_blk_hdr` CRC at offset 4 (not da3's
    // offset 12, which lies inside blkno here).
    stamp_v5_dir_block_crc(&mut block);
    Ok(block)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::xfs::dir::{XFS_DIR3_DATA_MAGIC, XFS_DIR3_FT_DIR, XFS_DIR3_FT_REG_FILE};

    fn entries(n: usize) -> Vec<(String, u64, u8)> {
        let mut v = vec![
            (".".to_string(), 64, XFS_DIR3_FT_DIR),
            ("..".to_string(), 64, XFS_DIR3_FT_DIR),
        ];
        for i in 0..n {
            v.push((format!("file{i:05}"), 1000 + i as u64, XFS_DIR3_FT_REG_FILE));
        }
        v
    }

    #[test]
    fn plan_layout_picks_format_by_size() {
        // Tiny → block; mid → leaf; large → node.
        assert_eq!(
            plan_layout(&entries(10), 4096).unwrap().format,
            DirFormat::Block
        );
        assert_eq!(
            plan_layout(&entries(300), 4096).unwrap().format,
            DirFormat::Leaf
        );
        assert_eq!(
            plan_layout(&entries(5000), 4096).unwrap().format,
            DirFormat::Node
        );
    }

    #[test]
    fn plan_layout_leaf_index_is_sorted_and_complete() {
        let ents = entries(300);
        let plan = plan_layout(&ents, 4096).unwrap();
        assert_eq!(
            plan.leaf_ents.len(),
            ents.len(),
            "one hash entry per record"
        );
        for w in plan.leaf_ents.windows(2) {
            assert!(w[0].0 <= w[1].0, "leaf hash index must be ascending");
        }
        // Every entry packed into some data block, none lost.
        let packed: usize = plan.data_blocks.iter().map(|b| b.len()).sum();
        assert_eq!(packed, ents.len());
    }

    #[test]
    fn node_leafn_split_covers_every_entry() {
        let plan = plan_layout(&entries(5000), 4096).unwrap();
        assert_eq!(plan.format, DirFormat::Node);
        let total: usize = plan.leafn_counts.iter().sum();
        assert_eq!(
            total,
            plan.leaf_ents.len(),
            "leafn split must cover all hashes"
        );
        // Each leafn slice fits one block.
        let per = (4096 - LEAF_HDR_SIZE) / 8;
        assert!(plan.leafn_counts.iter().all(|&c| c <= per));
    }

    fn crc_at(block: &[u8], off: usize) -> bool {
        let stored = u32::from_le_bytes(block[off..off + 4].try_into().unwrap());
        let mut tmp = block.to_vec();
        tmp[off..off + 4].copy_from_slice(&[0u8; 4]);
        crc32c::crc32c(&tmp) == stored
    }

    #[test]
    fn built_blocks_have_correct_magic_and_crc() {
        let uuid = [0xABu8; 16];
        // data block (dir3_blk_hdr: magic@0, crc@4)
        let data = build_data_block(&entries(50), 4096, 64, &uuid, 0).unwrap();
        assert_eq!(
            u32::from_be_bytes(data[0..4].try_into().unwrap()),
            XFS_DIR3_DATA_MAGIC
        );
        assert!(crc_at(&data, 4), "data block CRC");

        let leaf = build_leaf1_block(&[(1, 8), (2, 16)], &[10, 20], 4096, 64, &uuid, 0).unwrap();
        assert_eq!(
            u16::from_be_bytes(leaf[8..10].try_into().unwrap()),
            XFS_DIR3_LEAF1_MAGIC
        );
        assert!(crc_at(&leaf, DA3_CRC_OFFSET), "leaf1 da3 CRC");

        let leafn = build_leafn_block(&[(1, 8)], 4096, 64, &uuid, 0, 0, 0).unwrap();
        assert_eq!(
            u16::from_be_bytes(leafn[8..10].try_into().unwrap()),
            XFS_DIR3_LEAFN_MAGIC
        );
        assert!(crc_at(&leafn, DA3_CRC_OFFSET), "leafn da3 CRC");

        let node = build_da_node_block(&[(5, 0x800001)], 4096, 64, &uuid, 0).unwrap();
        assert_eq!(
            u16::from_be_bytes(node[8..10].try_into().unwrap()),
            XFS_DA3_NODE_MAGIC
        );
        assert_eq!(
            u16::from_be_bytes(node[58..60].try_into().unwrap()),
            1,
            "node level"
        );
        assert!(crc_at(&node, DA3_CRC_OFFSET), "node da3 CRC");

        // free block (dir3_blk_hdr: magic@0, crc@4). A full data block has
        // best 0 but still counts in nused (only 0xffff is excluded).
        let free = build_free_block(&[0, 24, 0xffff], 0, 4096, 64, &uuid, 0).unwrap();
        assert_eq!(
            u32::from_be_bytes(free[0..4].try_into().unwrap()),
            XFS_DIR3_FREE_MAGIC
        );
        assert_eq!(
            u32::from_be_bytes(free[52..56].try_into().unwrap()),
            3,
            "nvalid"
        );
        assert_eq!(
            u32::from_be_bytes(free[56..60].try_into().unwrap()),
            2,
            "nused (excl 0xffff)"
        );
        assert!(crc_at(&free, 4), "free block CRC");
    }
}
