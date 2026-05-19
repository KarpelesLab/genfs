//! HFS+ B-tree node walking — TN1150 "B-Trees".
//!
//! Both the catalog file and the extents-overflow file are B-trees
//! that share the same on-disk node format. A B-tree file is a
//! sequence of fixed-size *nodes*; node 0 is the *header node* which
//! carries a `BTHeaderRec` describing the rest of the tree. Other
//! nodes are *index* (`kind = 0`), *leaf* (`kind = -1`), *map*
//! (`kind = 2`), or — only when present — *header* (`kind = 1`).
//!
//! ## Node layout (TN1150 "BTNodeDescriptor")
//!
//! ```text
//! offset  size  field
//! 0       4     fLink         (forward sibling node)
//! 4       4     bLink         (back sibling node)
//! 8       1     kind          (BTreeNodeKind)
//! 9       1     height        (1 for leaves; grows upward)
//! 10      2     numRecords
//! 12      2     reserved
//! 14      ...   record area (variable)
//! ...     ...   record-offset table at the END of the node:
//!                 node[node_size - 2*(i+1) .. node_size - 2*i] = offset[i]
//! ```
//!
//! Record `i` occupies bytes `[offset[i], offset[i+1])`; `offset[n]`
//! (where `n = numRecords`) is the start of the free space and bounds
//! the last record. There are `numRecords + 1` offsets total.
//!
//! All multi-byte integers are big-endian.

use crate::Result;
use crate::block::BlockDevice;

use super::volume_header::{ExtentDescriptor, ForkData};

/// Leaf node kind, per TN1150 `kBTLeafNode = -1` (`0xFF`).
pub const KIND_LEAF: i8 = -1;
/// Index node kind, per TN1150 `kBTIndexNode = 0`.
pub const KIND_INDEX: i8 = 0;
/// Header node kind, per TN1150 `kBTHeaderNode = 1`.
pub const KIND_HEADER: i8 = 1;

/// Fixed on-disk encoded size of `BTNodeDescriptor`.
pub const NODE_DESCRIPTOR_SIZE: usize = 14;

/// Fixed on-disk encoded size of `BTHeaderRec`.
pub const HEADER_REC_SIZE: usize = 106;

/// Decoded B-tree node descriptor.
#[derive(Debug, Clone, Copy)]
pub struct NodeDescriptor {
    pub f_link: u32,
    pub b_link: u32,
    pub kind: i8,
    pub height: u8,
    pub num_records: u16,
}

impl NodeDescriptor {
    /// Decode the 14-byte BTNodeDescriptor at the start of every node.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < NODE_DESCRIPTOR_SIZE {
            return Err(crate::Error::InvalidImage(
                "hfs+: short node descriptor".into(),
            ));
        }
        Ok(Self {
            f_link: u32::from_be_bytes(buf[0..4].try_into().unwrap()),
            b_link: u32::from_be_bytes(buf[4..8].try_into().unwrap()),
            kind: buf[8] as i8,
            height: buf[9],
            num_records: u16::from_be_bytes(buf[10..12].try_into().unwrap()),
        })
    }
}

/// Decoded BTHeaderRec — TN1150 "B-Tree Header Node". Stored as the
/// first record of node 0 of a B-tree file.
///
/// ```text
/// offset  size  field
/// 0       2     treeDepth
/// 2       4     rootNode
/// 6       4     leafRecords
/// 10      4     firstLeafNode
/// 14      4     lastLeafNode
/// 18      2     nodeSize
/// 20      2     maxKeyLength
/// 22      4     totalNodes
/// 26      4     freeNodes
/// 30      2     reserved1
/// 32      4     clumpSize
/// 36      1     btreeType
/// 37      1     keyCompareType
/// 38      4     attributes
/// 42      ...   reserved (16 u32 words)
/// ```
#[derive(Debug, Clone, Copy)]
pub struct BTreeHeader {
    pub tree_depth: u16,
    pub root_node: u32,
    pub leaf_records: u32,
    pub first_leaf_node: u32,
    pub last_leaf_node: u32,
    pub node_size: u16,
    pub max_key_length: u16,
    pub total_nodes: u32,
    pub free_nodes: u32,
    pub clump_size: u32,
    pub btree_type: u8,
    pub key_compare_type: u8,
    pub attributes: u32,
}

impl BTreeHeader {
    /// Decode the BTHeaderRec record (first record of node 0).
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_REC_SIZE {
            return Err(crate::Error::InvalidImage("hfs+: short BTHeaderRec".into()));
        }
        let h = Self {
            tree_depth: u16::from_be_bytes(buf[0..2].try_into().unwrap()),
            root_node: u32::from_be_bytes(buf[2..6].try_into().unwrap()),
            leaf_records: u32::from_be_bytes(buf[6..10].try_into().unwrap()),
            first_leaf_node: u32::from_be_bytes(buf[10..14].try_into().unwrap()),
            last_leaf_node: u32::from_be_bytes(buf[14..18].try_into().unwrap()),
            node_size: u16::from_be_bytes(buf[18..20].try_into().unwrap()),
            max_key_length: u16::from_be_bytes(buf[20..22].try_into().unwrap()),
            total_nodes: u32::from_be_bytes(buf[22..26].try_into().unwrap()),
            free_nodes: u32::from_be_bytes(buf[26..30].try_into().unwrap()),
            clump_size: u32::from_be_bytes(buf[32..36].try_into().unwrap()),
            btree_type: buf[36],
            key_compare_type: buf[37],
            attributes: u32::from_be_bytes(buf[38..42].try_into().unwrap()),
        };
        if h.node_size == 0 || !h.node_size.is_power_of_two() {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: BTHeaderRec node_size {} is not a positive power of two",
                h.node_size
            )));
        }
        Ok(h)
    }
}

/// Reader that maps a fork's allocation-block extents to absolute
/// device byte offsets. Reading is restricted to the inline (≤ 8)
/// extents; if the fork requires the extents-overflow file, the
/// caller MUST extend `extents` with the full list before constructing
/// this reader.
pub struct ForkReader {
    /// Volume-relative base offset (0 unless we one day support
    /// embedded partitions).
    pub base_offset: u64,
    /// Allocation block size, in bytes.
    pub block_size: u32,
    /// Extents to read through, in order. Empty extents (count = 0)
    /// terminate the list.
    pub extents: Vec<ExtentDescriptor>,
    /// File length in bytes; reads past this return zero-pad if the
    /// extents cover more, but normally the caller stays inside.
    pub logical_size: u64,
}

impl ForkReader {
    /// Build a fork reader from an inline `ForkData` record. Returns
    /// `Err(Unsupported)` if the fork would require the extents-overflow
    /// file (i.e. its `total_blocks` exceeds the sum of inline extent
    /// block counts) — for v1 we report this cleanly rather than walking
    /// the overflow tree.
    pub fn from_inline(fork: &ForkData, block_size: u32, what: &str) -> Result<Self> {
        if u64::from(fork.total_blocks) > fork.inline_blocks() {
            return Err(crate::Error::Unsupported(format!(
                "hfs+: {what} fork has {} allocation blocks but only \
                 {} inline; extents-overflow file is not supported in v1",
                fork.total_blocks,
                fork.inline_blocks()
            )));
        }
        let extents: Vec<ExtentDescriptor> = fork
            .extents
            .iter()
            .copied()
            .filter(|e| e.block_count != 0)
            .collect();
        Ok(Self {
            base_offset: 0,
            block_size,
            extents,
            logical_size: fork.logical_size,
        })
    }

    /// Build a fork reader from an inline `ForkData` plus an
    /// already-collected list of overflow extents pulled from the
    /// extents-overflow B-tree. The combined extent list must cover
    /// the fork's `total_blocks`, otherwise `Err(InvalidImage)` is
    /// returned.
    pub fn from_inline_plus_overflow(
        fork: &ForkData,
        overflow: &[ExtentDescriptor],
        block_size: u32,
        what: &str,
    ) -> Result<Self> {
        let mut extents: Vec<ExtentDescriptor> = fork
            .extents
            .iter()
            .copied()
            .filter(|e| e.block_count != 0)
            .collect();
        extents.extend(overflow.iter().copied().filter(|e| e.block_count != 0));
        let covered: u64 = extents.iter().map(|e| u64::from(e.block_count)).sum();
        if covered < u64::from(fork.total_blocks) {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: {what} fork still missing extents after overflow walk \
                 (covered {covered} blocks, expected {})",
                fork.total_blocks
            )));
        }
        Ok(Self {
            base_offset: 0,
            block_size,
            extents,
            logical_size: fork.logical_size,
        })
    }

    /// Translate a fork-relative byte offset to the absolute device
    /// offset by walking the extent list. Returns `None` if `offset`
    /// is past the last extent.
    fn translate(&self, offset: u64) -> Option<(u64, u64)> {
        let mut walked: u64 = 0;
        for ext in &self.extents {
            let len = u64::from(ext.block_count) * u64::from(self.block_size);
            if offset < walked + len {
                let within = offset - walked;
                let device = self.base_offset
                    + u64::from(ext.start_block) * u64::from(self.block_size)
                    + within;
                let avail = len - within;
                return Some((device, avail));
            }
            walked += len;
        }
        None
    }

    /// Read `buf.len()` bytes starting at `offset` into the fork.
    pub fn read(&self, dev: &mut dyn BlockDevice, offset: u64, buf: &mut [u8]) -> Result<()> {
        let mut pos = offset;
        let mut written = 0usize;
        while written < buf.len() {
            let (dev_off, avail) = self.translate(pos).ok_or_else(|| {
                crate::Error::InvalidImage(format!(
                    "hfs+: fork read past mapped extents (offset {pos})"
                ))
            })?;
            let want = (buf.len() - written).min(avail as usize);
            dev.read_at(dev_off, &mut buf[written..written + want])?;
            pos += want as u64;
            written += want;
        }
        Ok(())
    }
}

/// Read a single B-tree node into a freshly allocated buffer.
pub fn read_node(
    dev: &mut dyn BlockDevice,
    fork: &ForkReader,
    node_idx: u32,
    node_size: u32,
) -> Result<Vec<u8>> {
    let off = u64::from(node_idx) * u64::from(node_size);
    let mut buf = vec![0u8; node_size as usize];
    fork.read(dev, off, &mut buf)?;
    Ok(buf)
}

/// Decode the `numRecords + 1` record offsets at the end of `node`.
///
/// The table grows backwards from the end of the node: `offset[i]`
/// lives at `node[node_size - 2*(i+1) .. node_size - 2*i]`. We return
/// the offsets in ascending record index so `record_bytes(node, &offsets, i)`
/// is `&node[offsets[i]..offsets[i+1]]`.
pub fn record_offsets(node: &[u8], num_records: u16) -> Result<Vec<u16>> {
    let n = num_records as usize;
    if node.len() < (n + 1) * 2 {
        return Err(crate::Error::InvalidImage(
            "hfs+: node too small to hold its record-offset table".into(),
        ));
    }
    let total = node.len();
    let mut offs = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let off_at = total - 2 * (i + 1);
        let val = u16::from_be_bytes([node[off_at], node[off_at + 1]]);
        offs.push(val);
    }
    // Validate: offsets must be monotonic ascending and within bounds
    // (the last one bounds the free space and so MAY equal total - 2*(n+1),
    // i.e. the start of the offset table itself).
    let max_data = total - 2 * (n + 1);
    let mut prev: u16 = NODE_DESCRIPTOR_SIZE as u16;
    for (i, &o) in offs.iter().enumerate() {
        if (o as usize) < NODE_DESCRIPTOR_SIZE || (o as usize) > total {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: record offset[{i}] = {o} is out of range (node size {total})"
            )));
        }
        if o < prev {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: record offsets not ascending at index {i} ({o} < {prev})"
            )));
        }
        prev = o;
    }
    if (offs[n] as usize) > max_data + 2 {
        // Allow the final offset to touch the offset table boundary; reject
        // anything strictly past it.
        return Err(crate::Error::InvalidImage(format!(
            "hfs+: final record offset {} exceeds free-space boundary {}",
            offs[n], max_data
        )));
    }
    Ok(offs)
}

/// Borrow the bytes of record `i` from a fully-loaded node, given the
/// decoded offset table.
pub fn record_bytes<'a>(node: &'a [u8], offs: &[u16], i: usize) -> &'a [u8] {
    let start = offs[i] as usize;
    let end = offs[i + 1] as usize;
    &node[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_node(records: &[&[u8]]) -> Vec<u8> {
        // 512-byte node with NODE_DESCRIPTOR_SIZE header, records,
        // padding, then offset table.
        let node_size = 512usize;
        let mut node = vec![0u8; node_size];
        // Descriptor: kind = leaf, height = 1, numRecords = records.len()
        node[8] = KIND_LEAF as u8;
        node[9] = 1;
        node[10..12].copy_from_slice(&(records.len() as u16).to_be_bytes());

        // Pack records sequentially after the descriptor.
        let mut cursor = NODE_DESCRIPTOR_SIZE;
        let mut offsets = Vec::with_capacity(records.len() + 1);
        for r in records {
            offsets.push(cursor as u16);
            node[cursor..cursor + r.len()].copy_from_slice(r);
            cursor += r.len();
        }
        offsets.push(cursor as u16);

        // Offset table at the END of the node, growing backwards.
        for (i, &o) in offsets.iter().enumerate() {
            let pos = node_size - 2 * (i + 1);
            node[pos..pos + 2].copy_from_slice(&o.to_be_bytes());
        }
        node
    }

    #[test]
    fn decode_node_descriptor() {
        let mut buf = [0u8; NODE_DESCRIPTOR_SIZE];
        buf[0..4].copy_from_slice(&42u32.to_be_bytes());
        buf[4..8].copy_from_slice(&0u32.to_be_bytes());
        buf[8] = KIND_INDEX as u8;
        buf[9] = 2;
        buf[10..12].copy_from_slice(&7u16.to_be_bytes());
        let d = NodeDescriptor::decode(&buf).unwrap();
        assert_eq!(d.f_link, 42);
        assert_eq!(d.kind, KIND_INDEX);
        assert_eq!(d.height, 2);
        assert_eq!(d.num_records, 7);
    }

    #[test]
    fn record_offset_table_round_trip() {
        let node = synth_node(&[b"aa", b"bbbb", b"cccccc"]);
        let nd = NodeDescriptor::decode(&node).unwrap();
        assert_eq!(nd.num_records, 3);
        let offs = record_offsets(&node, nd.num_records).unwrap();
        assert_eq!(offs.len(), 4);
        // first record begins right after the descriptor
        assert_eq!(offs[0], NODE_DESCRIPTOR_SIZE as u16);
        assert_eq!(record_bytes(&node, &offs, 0), b"aa");
        assert_eq!(record_bytes(&node, &offs, 1), b"bbbb");
        assert_eq!(record_bytes(&node, &offs, 2), b"cccccc");
    }

    #[test]
    fn record_offsets_rejects_descending() {
        let node_size = 256usize;
        let mut node = vec![0u8; node_size];
        node[10..12].copy_from_slice(&2u16.to_be_bytes());
        // Offsets reverse order on purpose.
        node[node_size - 2..node_size].copy_from_slice(&30u16.to_be_bytes());
        node[node_size - 4..node_size - 2].copy_from_slice(&50u16.to_be_bytes());
        node[node_size - 6..node_size - 4].copy_from_slice(&20u16.to_be_bytes());
        assert!(record_offsets(&node, 2).is_err());
    }

    #[test]
    fn fork_reader_translates_across_extents() {
        // block_size = 100, extents: [{start=2,count=3}, {start=10,count=2}].
        let fr = ForkReader {
            base_offset: 0,
            block_size: 100,
            extents: vec![
                ExtentDescriptor {
                    start_block: 2,
                    block_count: 3,
                },
                ExtentDescriptor {
                    start_block: 10,
                    block_count: 2,
                },
            ],
            logical_size: 500,
        };
        // offset 0 -> device 200, avail 300
        let (d, a) = fr.translate(0).unwrap();
        assert_eq!(d, 200);
        assert_eq!(a, 300);
        // offset 250 -> device 450, avail 50
        let (d, a) = fr.translate(250).unwrap();
        assert_eq!(d, 450);
        assert_eq!(a, 50);
        // offset 300 -> jumps to second extent: device 1000, avail 200
        let (d, a) = fr.translate(300).unwrap();
        assert_eq!(d, 1000);
        assert_eq!(a, 200);
        // offset 600 -> past end, None
        assert!(fr.translate(600).is_none());
    }
}
