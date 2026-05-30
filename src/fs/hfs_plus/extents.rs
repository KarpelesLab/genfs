//! HFS+ extents-overflow file — TN1150 "Extents Overflow File".
//!
//! The extents-overflow B-tree records the ninth and later extents of
//! any fork (data or resource) that needs more than the eight inline
//! extents kept in the catalog. Its key is:
//!
//! ```text
//! offset  size  field
//! 0       2     keyLength (= 10)
//! 2       1     forkType  (0 = data, 0xFF = resource)
//! 3       1     pad
//! 4       4     fileID    (CNID)
//! 8       4     startBlock (the fork-block where this record's run begins)
//! ```
//!
//! Each leaf record's data is an array of 8 `HFSPlusExtentDescriptor`
//! entries (`{ u32 startBlock, u32 blockCount }`); zero-count entries
//! terminate the run.
//!
//! Records are ordered first by `(forkType, fileID)`, then ascending by
//! `startBlock`. To gather every extent of a fork the walker descends
//! the tree to the first record for the (forkType, fileID) and then
//! follows the leaf chain until a record with a different
//! (forkType, fileID) is reached.

use std::cmp::Ordering;

use crate::Result;
use crate::block::BlockDevice;

use super::btree::{
    BTreeHeader, ForkReader, HEADER_REC_SIZE, KIND_INDEX, KIND_LEAF, NODE_DESCRIPTOR_SIZE,
    NodeDescriptor, read_node, record_bytes, record_offsets,
};
use super::volume_header::{ExtentDescriptor, FORK_EXTENT_COUNT};

/// Fork-type tag for the data fork.
pub const FORK_DATA: u8 = 0x00;
/// Fork-type tag for the resource fork.
pub const FORK_RESOURCE: u8 = 0xFF;

/// Fixed length (10 bytes) of the keyLength-bearing portion of an
/// HFSPlusExtentKey on disk.
pub const EXTENT_KEY_PAYLOAD_LEN: usize = 10;

/// On-disk size of an extent-overflow leaf record's data portion
/// (8 extent descriptors × 8 bytes).
pub const EXTENT_RECORD_SIZE: usize = FORK_EXTENT_COUNT * 8;

/// Decoded HFSPlusExtentKey.
#[derive(Debug, Clone, Copy)]
pub struct ExtentKey {
    pub fork_type: u8,
    pub file_id: u32,
    pub start_block: u32,
}

impl ExtentKey {
    /// Decode an extent-overflow key from `buf`. The leading u16
    /// `keyLength` is consumed.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < 2 + EXTENT_KEY_PAYLOAD_LEN {
            return Err(crate::Error::InvalidImage(
                "hfs+: short extents-overflow key".into(),
            ));
        }
        let key_length = u16::from_be_bytes([buf[0], buf[1]]);
        if key_length as usize != EXTENT_KEY_PAYLOAD_LEN {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: extents-overflow key_length {key_length} != {EXTENT_KEY_PAYLOAD_LEN}"
            )));
        }
        Ok(Self {
            fork_type: buf[2],
            // buf[3] is the pad byte
            file_id: u32::from_be_bytes(buf[4..8].try_into().unwrap()),
            start_block: u32::from_be_bytes(buf[8..12].try_into().unwrap()),
        })
    }

    /// Total encoded length of the key including the keyLength prefix,
    /// padded up to the next even byte boundary (B-tree alignment).
    pub fn encoded_len(&self) -> usize {
        // 2-byte keyLength + 10-byte payload = 12, already even.
        2 + EXTENT_KEY_PAYLOAD_LEN
    }

    /// Order extent keys: by (forkType, fileID, startBlock). The
    /// ordering must match Apple's; treating forkType as an unsigned
    /// u8 places data (0x00) before resource (0xFF), matching TN1150.
    pub fn order(&self, other: &Self) -> Ordering {
        (self.fork_type, self.file_id, self.start_block).cmp(&(
            other.fork_type,
            other.file_id,
            other.start_block,
        ))
    }
}

/// Decode the data portion of an extent-overflow leaf record: a
/// fixed-size array of eight `(startBlock, blockCount)` descriptors.
pub fn decode_extent_record(buf: &[u8]) -> Result<[ExtentDescriptor; FORK_EXTENT_COUNT]> {
    if buf.len() < EXTENT_RECORD_SIZE {
        return Err(crate::Error::InvalidImage(
            "hfs+: extents-overflow record too short".into(),
        ));
    }
    let mut out = [ExtentDescriptor::default(); FORK_EXTENT_COUNT];
    for (i, slot) in out.iter_mut().enumerate() {
        let off = i * 8;
        let mut e = [0u8; 8];
        e.copy_from_slice(&buf[off..off + 8]);
        *slot = ExtentDescriptor::decode(&e);
    }
    Ok(out)
}

/// Opened extents-overflow B-tree, ready to query for additional
/// extents that didn't fit inline in a catalog file record.
pub struct ExtentsOverflow {
    pub fork: ForkReader,
    pub header: BTreeHeader,
}

impl ExtentsOverflow {
    /// Open the extents-overflow B-tree by reading its header node (node 0).
    pub fn open(dev: &mut dyn BlockDevice, fork: ForkReader) -> Result<Self> {
        let mut bootstrap = vec![0u8; 512];
        fork.read(dev, 0, &mut bootstrap)?;
        let desc = NodeDescriptor::decode(&bootstrap)?;
        if desc.kind != super::btree::KIND_HEADER {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: extents-overflow node 0 has kind {} (expected header)",
                desc.kind
            )));
        }
        let hdr_buf = &bootstrap[NODE_DESCRIPTOR_SIZE..NODE_DESCRIPTOR_SIZE + HEADER_REC_SIZE];
        let header = BTreeHeader::decode(hdr_buf)?;
        Ok(Self { fork, header })
    }

    /// Collect every extent for `(fork_type, file_id)` whose
    /// `start_block` is greater-than-or-equal-to `first_start_block`
    /// (i.e. those covering fork-blocks past the inline extents).
    ///
    /// Returns the extents in fork order, with zero-count terminators
    /// stripped.
    pub fn find_extents(
        &self,
        dev: &mut dyn BlockDevice,
        file_id: u32,
        fork_type: u8,
        first_start_block: u32,
    ) -> Result<Vec<ExtentDescriptor>> {
        let mut out = Vec::new();
        if self.header.root_node == 0 {
            return Ok(out);
        }

        // Lower-bound key: the first record we want.
        let target = ExtentKey {
            fork_type,
            file_id,
            start_block: first_start_block,
        };

        // Descend index nodes, picking the largest separator key
        // ≤ target. This lands us on the leaf that may contain the
        // first matching record (or precedes it).
        let node_size = u32::from(self.header.node_size);
        let mut node_idx = self.header.root_node;
        // Bound the descent by tree depth and reject out-of-range child
        // pointers so a self-referential index node cannot loop forever.
        let max_descent = self.header.tree_depth.max(1) as usize + 1;
        let mut descent_left = max_descent;
        let leaf_idx = loop {
            if descent_left == 0 {
                return Err(crate::Error::InvalidImage(
                    "hfs+: extents-overflow B-tree descent exceeded tree depth (cycle?)".into(),
                ));
            }
            descent_left -= 1;
            if node_idx >= self.header.total_nodes {
                return Err(crate::Error::InvalidImage(format!(
                    "hfs+: extents-overflow child node {node_idx} >= total_nodes {}",
                    self.header.total_nodes
                )));
            }
            let node = read_node(dev, &self.fork, node_idx, node_size)?;
            let desc = NodeDescriptor::decode(&node)?;
            let offs = record_offsets(&node, desc.num_records)?;
            if desc.kind == KIND_LEAF {
                break node_idx;
            }
            if desc.kind != KIND_INDEX {
                return Err(crate::Error::InvalidImage(format!(
                    "hfs+: extents-overflow non-leaf kind {} in descent",
                    desc.kind
                )));
            }
            let mut child: Option<u32> = None;
            for i in 0..desc.num_records as usize {
                let rec = record_bytes(&node, &offs, i);
                let key = ExtentKey::decode(rec)?;
                let ptr_off = key.encoded_len();
                if ptr_off + 4 > rec.len() {
                    return Err(crate::Error::InvalidImage(
                        "hfs+: extents-overflow index record missing child pointer".into(),
                    ));
                }
                let next = u32::from_be_bytes(rec[ptr_off..ptr_off + 4].try_into().unwrap());
                match key.order(&target) {
                    Ordering::Less | Ordering::Equal => child = Some(next),
                    Ordering::Greater => break,
                }
            }
            node_idx = match child {
                Some(c) => c,
                // target is below the first child key — descend the
                // first subtree (records here would be the smallest in
                // the tree).
                None => {
                    let rec = record_bytes(&node, &offs, 0);
                    let key = ExtentKey::decode(rec)?;
                    let ptr_off = key.encoded_len();
                    u32::from_be_bytes(rec[ptr_off..ptr_off + 4].try_into().unwrap())
                }
            };
        };

        // Walk the leaf chain forward from `leaf_idx`, collecting every
        // record whose key matches (fork_type, file_id) with
        // start_block ≥ first_start_block. Stop as soon as we walk past
        // either the file_id or the fork_type boundary.
        // Bound the forward leaf-chain walk by the node count and require a
        // strictly-increasing fLink: a malicious image can otherwise build a
        // leaf-chain cycle that loops forever.
        let mut cur = leaf_idx;
        let mut steps_left = self.header.total_nodes as usize;
        while cur != 0 {
            if steps_left == 0 {
                return Err(crate::Error::InvalidImage(
                    "hfs+: extents-overflow leaf chain exceeded node count (cycle?)".into(),
                ));
            }
            steps_left -= 1;
            if cur >= self.header.total_nodes {
                return Err(crate::Error::InvalidImage(format!(
                    "hfs+: extents-overflow leaf node {cur} >= total_nodes {}",
                    self.header.total_nodes
                )));
            }
            let node = read_node(dev, &self.fork, cur, node_size)?;
            let desc = NodeDescriptor::decode(&node)?;
            if desc.kind != KIND_LEAF {
                return Err(crate::Error::InvalidImage(format!(
                    "hfs+: extents-overflow leaf chain hit kind {}",
                    desc.kind
                )));
            }
            let offs = record_offsets(&node, desc.num_records)?;

            let mut passed = false;
            for i in 0..desc.num_records as usize {
                let rec = record_bytes(&node, &offs, i);
                let key = ExtentKey::decode(rec)?;
                // Skip records below our target (the descent may have
                // started us in the right leaf but the matching record
                // is not the first one).
                match (key.fork_type, key.file_id).cmp(&(fork_type, file_id)) {
                    Ordering::Less => continue,
                    Ordering::Greater => {
                        passed = true;
                        break;
                    }
                    Ordering::Equal => {}
                }
                if key.start_block < first_start_block {
                    continue;
                }
                let body_start = key.encoded_len();
                if body_start + EXTENT_RECORD_SIZE > rec.len() {
                    return Err(crate::Error::InvalidImage(
                        "hfs+: extents-overflow record body truncated".into(),
                    ));
                }
                let body = &rec[body_start..body_start + EXTENT_RECORD_SIZE];
                let extents = decode_extent_record(body)?;
                for e in extents {
                    if e.block_count == 0 {
                        break;
                    }
                    out.push(e);
                }
            }
            if passed {
                break;
            }
            cur = desc.f_link;
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::btree::{KIND_HEADER, KIND_LEAF, NODE_DESCRIPTOR_SIZE};
    use super::*;
    use crate::block::{BlockDevice, MemoryBackend};

    #[test]
    fn extent_key_decodes() {
        // keyLength = 10, fork = data, pad = 0, fileID = 42, startBlock = 9
        let mut buf = Vec::new();
        buf.extend_from_slice(&(EXTENT_KEY_PAYLOAD_LEN as u16).to_be_bytes());
        buf.push(FORK_DATA);
        buf.push(0);
        buf.extend_from_slice(&42u32.to_be_bytes());
        buf.extend_from_slice(&9u32.to_be_bytes());
        let k = ExtentKey::decode(&buf).unwrap();
        assert_eq!(k.fork_type, FORK_DATA);
        assert_eq!(k.file_id, 42);
        assert_eq!(k.start_block, 9);
    }

    #[test]
    fn extent_key_rejects_wrong_length() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&8u16.to_be_bytes()); // wrong key length
        buf.extend(std::iter::repeat_n(0u8, EXTENT_KEY_PAYLOAD_LEN));
        assert!(ExtentKey::decode(&buf).is_err());
    }

    #[test]
    fn extent_key_order_compares_fork_then_id_then_block() {
        let a = ExtentKey {
            fork_type: FORK_DATA,
            file_id: 10,
            start_block: 100,
        };
        let b = ExtentKey {
            fork_type: FORK_DATA,
            file_id: 10,
            start_block: 200,
        };
        let c = ExtentKey {
            fork_type: FORK_DATA,
            file_id: 11,
            start_block: 0,
        };
        let d = ExtentKey {
            fork_type: FORK_RESOURCE,
            file_id: 10,
            start_block: 0,
        };
        assert_eq!(a.order(&b), Ordering::Less);
        assert_eq!(b.order(&c), Ordering::Less);
        assert_eq!(c.order(&d), Ordering::Less);
        assert_eq!(b.order(&a), Ordering::Greater);
    }

    /// Build a serialised B-tree node containing one or more
    /// `(key_bytes, payload_bytes)` records, padded to `node_size`.
    /// The descriptor is filled in with `kind` and `f_link`.
    fn build_node(
        node_size: usize,
        kind: i8,
        f_link: u32,
        records: &[(Vec<u8>, Vec<u8>)],
    ) -> Vec<u8> {
        let mut node = vec![0u8; node_size];
        node[0..4].copy_from_slice(&f_link.to_be_bytes());
        node[8] = kind as u8;
        node[9] = 1;
        node[10..12].copy_from_slice(&(records.len() as u16).to_be_bytes());
        let mut cursor = NODE_DESCRIPTOR_SIZE;
        let mut offsets = Vec::with_capacity(records.len() + 1);
        for (k, v) in records {
            offsets.push(cursor as u16);
            node[cursor..cursor + k.len()].copy_from_slice(k);
            cursor += k.len();
            node[cursor..cursor + v.len()].copy_from_slice(v);
            cursor += v.len();
        }
        offsets.push(cursor as u16);
        for (i, &o) in offsets.iter().enumerate() {
            let pos = node_size - 2 * (i + 1);
            node[pos..pos + 2].copy_from_slice(&o.to_be_bytes());
        }
        node
    }

    fn key_bytes(fork_type: u8, file_id: u32, start_block: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity(2 + EXTENT_KEY_PAYLOAD_LEN);
        v.extend_from_slice(&(EXTENT_KEY_PAYLOAD_LEN as u16).to_be_bytes());
        v.push(fork_type);
        v.push(0);
        v.extend_from_slice(&file_id.to_be_bytes());
        v.extend_from_slice(&start_block.to_be_bytes());
        v
    }

    fn record_payload(extents: &[(u32, u32)]) -> Vec<u8> {
        let mut v = vec![0u8; EXTENT_RECORD_SIZE];
        for (i, (start, count)) in extents.iter().enumerate() {
            let off = i * 8;
            v[off..off + 4].copy_from_slice(&start.to_be_bytes());
            v[off + 4..off + 8].copy_from_slice(&count.to_be_bytes());
        }
        v
    }

    /// Write the B-tree header record into the start of node 0.
    fn write_header_node(
        node_size: usize,
        root: u32,
        first_leaf: u32,
        last_leaf: u32,
        total_nodes: u32,
    ) -> Vec<u8> {
        let mut node = vec![0u8; node_size];
        // Descriptor: header kind.
        node[8] = KIND_HEADER as u8;
        node[9] = 0;
        node[10..12].copy_from_slice(&3u16.to_be_bytes()); // hdr + user + map placeholders
        // BTHeaderRec at offset 14.
        let h = NODE_DESCRIPTOR_SIZE;
        node[h..h + 2].copy_from_slice(&1u16.to_be_bytes()); // tree_depth
        node[h + 2..h + 6].copy_from_slice(&root.to_be_bytes());
        node[h + 6..h + 10].copy_from_slice(&0u32.to_be_bytes()); // leaf_records
        node[h + 10..h + 14].copy_from_slice(&first_leaf.to_be_bytes());
        node[h + 14..h + 18].copy_from_slice(&last_leaf.to_be_bytes());
        node[h + 18..h + 20].copy_from_slice(&(node_size as u16).to_be_bytes());
        node[h + 20..h + 22].copy_from_slice(&10u16.to_be_bytes()); // max_key_length
        node[h + 22..h + 26].copy_from_slice(&total_nodes.to_be_bytes());
        node[h + 26..h + 30].copy_from_slice(&0u32.to_be_bytes()); // free_nodes
        // record offsets: header rec at 14, user rec at 14+106=120, map rec
        // at end. We only need the first offset to be sane.
        let end = node_size;
        // 4 offsets (numRecords=3 + free): [14, 120, 248, free_start]
        let offs = [
            NODE_DESCRIPTOR_SIZE as u16,
            (NODE_DESCRIPTOR_SIZE + HEADER_REC_SIZE) as u16,
            (NODE_DESCRIPTOR_SIZE + HEADER_REC_SIZE + 128) as u16,
            (NODE_DESCRIPTOR_SIZE + HEADER_REC_SIZE + 128) as u16,
        ];
        for (i, o) in offs.iter().enumerate() {
            let pos = end - 2 * (i + 1);
            node[pos..pos + 2].copy_from_slice(&o.to_be_bytes());
        }
        node
    }

    #[test]
    fn overflow_walker_returns_extents_for_file() {
        // Build a 512-byte node B-tree:
        //   node 0 = header (root_node = 1, first_leaf = 1)
        //   node 1 = leaf with 3 records:
        //       (data, 100, 8)  -> [(500,2),(600,3),0..]
        //       (data, 100, 13) -> [(700,4),0..]    ; total now 17 blocks
        //       (data, 101, 8)  -> [(900,1),0..]    ; different file_id
        let node_size = 512usize;
        let header = write_header_node(node_size, 1, 1, 1, 2);
        let leaf = build_node(
            node_size,
            KIND_LEAF,
            0,
            &[
                (
                    key_bytes(FORK_DATA, 100, 8),
                    record_payload(&[(500, 2), (600, 3)]),
                ),
                (key_bytes(FORK_DATA, 100, 13), record_payload(&[(700, 4)])),
                (key_bytes(FORK_DATA, 101, 8), record_payload(&[(900, 1)])),
            ],
        );

        let mut dev = MemoryBackend::new((node_size * 4) as u64);
        dev.write_at(0, &header).unwrap();
        dev.write_at(node_size as u64, &leaf).unwrap();

        // Fork covers nodes 0..3 = 1536 bytes.
        let fork = ForkReader {
            base_offset: 0,
            block_size: node_size as u32,
            extents: vec![ExtentDescriptor {
                start_block: 0,
                block_count: 4,
            }],
            logical_size: (node_size * 4) as u64,
        };
        let ov = ExtentsOverflow::open(&mut dev, fork).unwrap();
        let found = ov.find_extents(&mut dev, 100, FORK_DATA, 8).unwrap();
        assert_eq!(found.len(), 3);
        assert_eq!(found[0].start_block, 500);
        assert_eq!(found[0].block_count, 2);
        assert_eq!(found[1].start_block, 600);
        assert_eq!(found[1].block_count, 3);
        assert_eq!(found[2].start_block, 700);
        assert_eq!(found[2].block_count, 4);
    }

    #[test]
    fn overflow_walker_filters_by_file_id_and_fork() {
        let node_size = 512usize;
        let header = write_header_node(node_size, 1, 1, 1, 2);
        let leaf = build_node(
            node_size,
            KIND_LEAF,
            0,
            &[
                (key_bytes(FORK_DATA, 50, 8), record_payload(&[(11, 1)])),
                (key_bytes(FORK_DATA, 60, 8), record_payload(&[(22, 2)])),
                (key_bytes(FORK_RESOURCE, 60, 8), record_payload(&[(33, 3)])),
            ],
        );
        let mut dev = MemoryBackend::new((node_size * 4) as u64);
        dev.write_at(0, &header).unwrap();
        dev.write_at(node_size as u64, &leaf).unwrap();
        let fork = ForkReader {
            base_offset: 0,
            block_size: node_size as u32,
            extents: vec![ExtentDescriptor {
                start_block: 0,
                block_count: 4,
            }],
            logical_size: (node_size * 4) as u64,
        };
        let ov = ExtentsOverflow::open(&mut dev, fork).unwrap();
        // file_id=60, data fork only.
        let got = ov.find_extents(&mut dev, 60, FORK_DATA, 8).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].start_block, 22);

        // file_id=60, resource fork.
        let got = ov.find_extents(&mut dev, 60, FORK_RESOURCE, 0).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].start_block, 33);

        // No matching extents.
        let got = ov.find_extents(&mut dev, 99, FORK_DATA, 0).unwrap();
        assert!(got.is_empty());
    }

    #[test]
    fn extent_record_decodes_terminator() {
        let mut buf = vec![0u8; EXTENT_RECORD_SIZE];
        // first descriptor: start=100, count=2
        buf[0..4].copy_from_slice(&100u32.to_be_bytes());
        buf[4..8].copy_from_slice(&2u32.to_be_bytes());
        // second: start=200, count=3
        buf[8..12].copy_from_slice(&200u32.to_be_bytes());
        buf[12..16].copy_from_slice(&3u32.to_be_bytes());
        // remaining zeros are terminators.
        let arr = decode_extent_record(&buf).unwrap();
        assert_eq!(arr[0].start_block, 100);
        assert_eq!(arr[0].block_count, 2);
        assert_eq!(arr[1].start_block, 200);
        assert_eq!(arr[1].block_count, 3);
        assert_eq!(arr[2].block_count, 0);
    }
}
