//! HFS+ attributes file — TN1150 "Attributes File".
//!
//! The attributes B-tree stores extended-attribute key/value pairs for
//! files and folders. We only need read-side support: looking up a
//! single attribute (such as `com.apple.decmpfs`) by `(fileID, name)`
//! and returning either its inline bytes or a fork descriptor.
//!
//! ## Key layout (HFSPlusAttrKey, TN1150 + Darwin `hfs_format.h`)
//!
//! ```text
//! offset  size  field
//! 0       2     keyLength       (bytes following this u16)
//! 2       2     pad             (reserved, 0)
//! 4       4     fileID          (HFSCatalogNodeID)
//! 8       4     startBlock      (alloc-block offset within attribute fork; 0 for inline)
//! 12      2     attrNameLength  (number of UTF-16 code units; max 127)
//! 14      ...   attrName        (UTF-16 BE, no NUL)
//! ```
//!
//! Comparison order is `(fileID, attrName, startBlock)`. The name is
//! folded case-insensitively on plain HFS+ and compared byte-exact on
//! HFSX, matching the catalog key rules.
//!
//! ## Record layout (first 4 bytes select the record type)
//!
//! ```text
//! 0x00000010  HFSPlusAttrInlineData : recordType(4) + reserved(4) + attrSize(4) + data[attrSize]
//! 0x00000020  HFSPlusAttrForkData   : recordType(4) + reserved(4) + HFSPlusForkData(80)
//! 0x00000030  HFSPlusAttrExtents    : recordType(4) + reserved(4) + 8 × ExtentDescriptor
//! ```
//!
//! For decmpfs the value is small (≤ 3802 bytes is the inline cap in
//! practice) and always lands in an `Inline` record; the resource fork
//! holds the bulk compressed payload separately in the *catalog* file's
//! own `resourceFork` field, not via `HFSPlusAttrForkData`.

use std::cmp::Ordering;

use crate::Result;
use crate::block::BlockDevice;

use super::btree::{
    BTreeHeader, ForkReader, HEADER_REC_SIZE, KIND_INDEX, KIND_LEAF, NODE_DESCRIPTOR_SIZE,
    NodeDescriptor, read_node, record_bytes, record_offsets,
};
use super::catalog::{UniStr, compare_unistr};
use super::volume_header::ForkData;

/// `kHFSPlusAttrInlineData` — value stored inline in the leaf record.
pub const REC_INLINE_DATA: u32 = 0x10;
/// `kHFSPlusAttrForkData` — value stored in its own fork.
pub const REC_FORK_DATA: u32 = 0x20;
/// `kHFSPlusAttrExtents` — overflow extents for a fork-data attribute.
pub const REC_EXTENTS: u32 = 0x30;

/// A decoded attribute key.
#[derive(Debug, Clone)]
pub struct AttrKey {
    pub file_id: u32,
    pub start_block: u32,
    pub name: UniStr,
    /// Total encoded length of the key including the leading
    /// `keyLength` u16, padded up to a 2-byte boundary.
    pub encoded_len: usize,
}

impl AttrKey {
    /// Decode an attribute key from `buf`.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < 14 {
            return Err(crate::Error::InvalidImage(
                "hfs+: short attribute key".into(),
            ));
        }
        let key_length = u16::from_be_bytes([buf[0], buf[1]]) as usize;
        // pad at [2..4] is reserved
        let file_id = u32::from_be_bytes(buf[4..8].try_into().unwrap());
        let start_block = u32::from_be_bytes(buf[8..12].try_into().unwrap());
        let name_len = u16::from_be_bytes([buf[12], buf[13]]) as usize;
        if name_len > 127 {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: attribute name length {name_len} exceeds 127"
            )));
        }
        // On-disk total encoded length = keyLength(2) + pad(2) + fileID(4)
        // + startBlock(4) + nameLength(2) + name(2*N) = 14 + 2*N.
        let payload_len = 14 + 2 * name_len;
        // The on-disk key_length field covers everything *after* the
        // keyLength u16 itself: pad(2) + fileID(4) + startBlock(4) +
        // nameLength(2) + name(2*N) = 12 + 2*N bytes.
        if 12 + 2 * name_len != key_length {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: attribute key_length {key_length} disagrees with parsed name \
                 ({name_len} UTF-16 code units)"
            )));
        }
        if buf.len() < payload_len {
            return Err(crate::Error::InvalidImage(
                "hfs+: attribute key truncated".into(),
            ));
        }
        let mut code_units = Vec::with_capacity(name_len);
        for i in 0..name_len {
            let off = 14 + 2 * i;
            code_units.push(u16::from_be_bytes([buf[off], buf[off + 1]]));
        }
        let used = payload_len;
        let encoded_len = if used % 2 == 0 { used } else { used + 1 };
        Ok(Self {
            file_id,
            start_block,
            name: UniStr { code_units },
            encoded_len,
        })
    }

    /// Compare two attribute keys. Order is `(fileID, name, startBlock)`.
    pub fn compare(&self, other: &AttrKey, case_sensitive: bool) -> Ordering {
        match self.file_id.cmp(&other.file_id) {
            Ordering::Equal => match compare_unistr(&self.name, &other.name, case_sensitive) {
                Ordering::Equal => self.start_block.cmp(&other.start_block),
                o => o,
            },
            o => o,
        }
    }
}

/// One leaf record from the attributes B-tree.
#[derive(Debug, Clone)]
pub enum AttrRecord {
    /// Value stored inline in the record (the common case for small
    /// xattrs like decmpfs headers ≤ ~3.8 KiB).
    Inline {
        /// The attribute value bytes, in declaration order.
        data: Vec<u8>,
    },
    /// Value stored in its own fork; the embedded `ForkData` describes
    /// up to eight inline extents. Overflow extents (record type 0x30)
    /// are not currently followed — they're rare for xattrs.
    Fork {
        /// Fork-data record describing the attribute's value extents.
        fork: ForkData,
    },
}

impl AttrRecord {
    /// Decode the body bytes of an attribute leaf record (the part
    /// after the key).
    pub fn decode(body: &[u8]) -> Result<Self> {
        if body.len() < 4 {
            return Err(crate::Error::InvalidImage(
                "hfs+: short attribute record body".into(),
            ));
        }
        let rec_type = u32::from_be_bytes(body[0..4].try_into().unwrap());
        match rec_type {
            REC_INLINE_DATA => {
                if body.len() < 12 {
                    return Err(crate::Error::InvalidImage(
                        "hfs+: short HFSPlusAttrInlineData".into(),
                    ));
                }
                let attr_size = u32::from_be_bytes(body[8..12].try_into().unwrap()) as usize;
                if body.len() < 12 + attr_size {
                    return Err(crate::Error::InvalidImage(format!(
                        "hfs+: HFSPlusAttrInlineData attrSize {attr_size} exceeds record body ({} bytes)",
                        body.len() - 12
                    )));
                }
                Ok(Self::Inline {
                    data: body[12..12 + attr_size].to_vec(),
                })
            }
            REC_FORK_DATA => {
                if body.len() < 8 + 80 {
                    return Err(crate::Error::InvalidImage(
                        "hfs+: short HFSPlusAttrForkData".into(),
                    ));
                }
                let mut tmp = [0u8; 80];
                tmp.copy_from_slice(&body[8..88]);
                Ok(Self::Fork {
                    fork: ForkData::decode(&tmp),
                })
            }
            REC_EXTENTS => Err(crate::Error::Unsupported(
                "hfs+: attribute overflow-extents record (0x30) not supported".into(),
            )),
            other => Err(crate::Error::InvalidImage(format!(
                "hfs+: unknown attribute record type {other:#010x}"
            ))),
        }
    }
}

/// Opened, ready-to-query attributes B-tree.
pub struct Attributes {
    pub fork: ForkReader,
    pub header: BTreeHeader,
    pub case_sensitive: bool,
}

impl Attributes {
    /// Open the attributes B-tree by reading its header node (node 0).
    /// `fork` is the attributes file's data fork.
    pub fn open(dev: &mut dyn BlockDevice, fork: ForkReader, case_sensitive: bool) -> Result<Self> {
        let mut bootstrap = vec![0u8; 512];
        fork.read(dev, 0, &mut bootstrap)?;
        let probe_desc = NodeDescriptor::decode(&bootstrap)?;
        if probe_desc.kind != super::btree::KIND_HEADER {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: attributes node 0 has kind {} (expected header)",
                probe_desc.kind
            )));
        }
        let hdr_buf = &bootstrap[NODE_DESCRIPTOR_SIZE..NODE_DESCRIPTOR_SIZE + HEADER_REC_SIZE];
        let header = BTreeHeader::decode(hdr_buf)?;
        Ok(Self {
            fork,
            header,
            case_sensitive,
        })
    }

    /// Look up an attribute named `name` on file `file_id`. Returns
    /// `None` if no record exists. The lookup uses `startBlock = 0`
    /// (the inline / first-segment record).
    pub fn lookup(
        &self,
        dev: &mut dyn BlockDevice,
        file_id: u32,
        name: &str,
    ) -> Result<Option<AttrRecord>> {
        let wanted = AttrKey {
            file_id,
            start_block: 0,
            name: UniStr::from_str_lossy(name),
            encoded_len: 0,
        };
        let node_size = u32::from(self.header.node_size);
        let mut node_idx = self.header.root_node;
        if node_idx == 0 {
            return Ok(None);
        }
        loop {
            let node = read_node(dev, &self.fork, node_idx, node_size)?;
            let desc = NodeDescriptor::decode(&node)?;
            let offs = record_offsets(&node, desc.num_records)?;
            if desc.kind == KIND_LEAF {
                for i in 0..desc.num_records as usize {
                    let rec = record_bytes(&node, &offs, i);
                    let key = AttrKey::decode(rec)?;
                    match key.compare(&wanted, self.case_sensitive) {
                        Ordering::Equal => {
                            let body_start = align2(key.encoded_len);
                            if body_start > rec.len() {
                                return Err(crate::Error::InvalidImage(
                                    "hfs+: attribute key overruns record".into(),
                                ));
                            }
                            return Ok(Some(AttrRecord::decode(&rec[body_start..])?));
                        }
                        Ordering::Greater => return Ok(None),
                        Ordering::Less => continue,
                    }
                }
                return Ok(None);
            } else if desc.kind == KIND_INDEX {
                let mut child: Option<u32> = None;
                for i in 0..desc.num_records as usize {
                    let rec = record_bytes(&node, &offs, i);
                    let key = AttrKey::decode(rec)?;
                    let pointer_off = align2(key.encoded_len);
                    if pointer_off + 4 > rec.len() {
                        return Err(crate::Error::InvalidImage(
                            "hfs+: attributes index record missing child pointer".into(),
                        ));
                    }
                    let next =
                        u32::from_be_bytes(rec[pointer_off..pointer_off + 4].try_into().unwrap());
                    match key.compare(&wanted, self.case_sensitive) {
                        Ordering::Less | Ordering::Equal => child = Some(next),
                        Ordering::Greater => break,
                    }
                }
                node_idx = match child {
                    Some(c) => c,
                    None => return Ok(None),
                };
            } else {
                return Err(crate::Error::InvalidImage(format!(
                    "hfs+: unexpected B-tree node kind {} in attributes traversal",
                    desc.kind
                )));
            }
        }
    }
}

fn align2(n: usize) -> usize {
    n + (n & 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_inline_record(name: &str, value: &[u8]) -> Vec<u8> {
        // Build a (key, inline-record) pair encoded as one leaf-record
        // byte slice, suitable for AttrKey::decode + AttrRecord::decode.
        let name_units: Vec<u16> = name.encode_utf16().collect();
        let key_payload_len = 12 + 2 * name_units.len();
        let mut out = Vec::new();
        out.extend_from_slice(&(key_payload_len as u16).to_be_bytes());
        out.extend_from_slice(&0u16.to_be_bytes()); // pad
        out.extend_from_slice(&42u32.to_be_bytes()); // file_id
        out.extend_from_slice(&0u32.to_be_bytes()); // start_block
        out.extend_from_slice(&(name_units.len() as u16).to_be_bytes());
        for u in &name_units {
            out.extend_from_slice(&u.to_be_bytes());
        }
        // Pad to 2-byte alignment (already aligned: 14 + 2*N is even)
        // Inline record body
        out.extend_from_slice(&REC_INLINE_DATA.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes()); // reserved
        out.extend_from_slice(&(value.len() as u32).to_be_bytes()); // attrSize
        out.extend_from_slice(value);
        out
    }

    #[test]
    fn attr_key_round_trip() {
        let buf = make_inline_record("com.apple.decmpfs", &[0u8; 32]);
        let key = AttrKey::decode(&buf).unwrap();
        assert_eq!(key.file_id, 42);
        assert_eq!(key.start_block, 0);
        assert_eq!(key.name.to_string_lossy(), "com.apple.decmpfs");
        // 2 (keyLength) + 4 (pad+fileID? no — pad=2 + fileID=4) etc.
        // = 2 + 2 + 4 + 4 + 2 + 2*17 = 48
        assert_eq!(key.encoded_len, 48);
    }

    #[test]
    fn attr_inline_record_decode() {
        let payload = b"hello world".to_vec();
        let buf = make_inline_record("foo", &payload);
        let key = AttrKey::decode(&buf).unwrap();
        let body = &buf[align2(key.encoded_len)..];
        match AttrRecord::decode(body).unwrap() {
            AttrRecord::Inline { data } => assert_eq!(data, payload),
            other => panic!("expected Inline, got {other:?}"),
        }
    }

    #[test]
    fn attr_key_compare_orders_by_file_id_then_name() {
        let a = AttrKey {
            file_id: 1,
            start_block: 0,
            name: UniStr::from_str_lossy("zzz"),
            encoded_len: 0,
        };
        let b = AttrKey {
            file_id: 2,
            start_block: 0,
            name: UniStr::from_str_lossy("aaa"),
            encoded_len: 0,
        };
        assert_eq!(a.compare(&b, false), Ordering::Less);

        let c = AttrKey {
            file_id: 2,
            start_block: 0,
            name: UniStr::from_str_lossy("bbb"),
            encoded_len: 0,
        };
        assert_eq!(b.compare(&c, false), Ordering::Less);
    }
}
