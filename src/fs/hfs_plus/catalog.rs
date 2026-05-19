//! HFS+ catalog file — TN1150 "Catalog File".
//!
//! The catalog B-tree is keyed by `(parentID, nodeName)` and holds
//! four kinds of leaf records:
//!
//! ```text
//! 0x0001  HFSPlusCatalogFolder        (folder data)
//! 0x0002  HFSPlusCatalogFile          (file data, inc. dataFork+resourceFork)
//! 0x0003  HFSPlusCatalogThread        (folder thread: CNID -> parent + name)
//! 0x0004  HFSPlusCatalogFileThread    (file thread: same purpose)
//! ```
//!
//! Catalog records are aligned to 2 bytes within a node; key length is
//! big-endian and the key's payload is `parentID (u32 BE)` followed by
//! an `HFSUniStr255` (u16 BE character count + UTF-16 BE chars).
//!
//! The root *folder* has CNID `kHFSRootFolderID = 2`; the parent CNID
//! of the root is `kHFSRootParentID = 1`. The root's name in the
//! catalog thread record is the volume name.

use std::cmp::Ordering;

use crate::Result;
use crate::block::BlockDevice;

use super::btree::{
    BTreeHeader, ForkReader, KIND_INDEX, KIND_LEAF, NodeDescriptor, NODE_DESCRIPTOR_SIZE,
    HEADER_REC_SIZE, read_node, record_bytes, record_offsets,
};
use super::volume_header::ForkData;

/// `kHFSRootParentID` — the synthetic parent CNID of the volume root.
pub const ROOT_PARENT_ID: u32 = 1;
/// `kHFSRootFolderID` — CNID of the volume root folder itself.
pub const ROOT_FOLDER_ID: u32 = 2;

/// `kHFSPlusFolderRecord`.
pub const REC_FOLDER: i16 = 0x0001;
/// `kHFSPlusFileRecord`.
pub const REC_FILE: i16 = 0x0002;
/// `kHFSPlusFolderThreadRecord`.
pub const REC_FOLDER_THREAD: i16 = 0x0003;
/// `kHFSPlusFileThreadRecord`.
pub const REC_FILE_THREAD: i16 = 0x0004;

/// File-type modes used in `HFSPlusBSDInfo.fileMode`. Match the
/// standard POSIX `S_IF*` values.
pub mod mode {
    pub const S_IFMT: u16 = 0o170000;
    pub const S_IFIFO: u16 = 0o010000;
    pub const S_IFCHR: u16 = 0o020000;
    pub const S_IFDIR: u16 = 0o040000;
    pub const S_IFBLK: u16 = 0o060000;
    pub const S_IFREG: u16 = 0o100000;
    pub const S_IFLNK: u16 = 0o120000;
    pub const S_IFSOCK: u16 = 0o140000;
}

/// An HFSUniStr255 decoded to native UTF-16 code units.
///
/// HFS+ stores the length as a u16 (BE) followed by exactly that many
/// UTF-16 code units (also BE). We decode the slice into a `Vec<u16>`
/// for case-insensitive comparison via the simple HFS+ rules.
#[derive(Debug, Clone, Default)]
pub struct UniStr {
    pub code_units: Vec<u16>,
}

impl UniStr {
    /// Decode an HFSUniStr255 from `buf`, returning the parsed string and the
    /// number of bytes consumed (2 + 2*length).
    pub fn decode(buf: &[u8]) -> Result<(Self, usize)> {
        if buf.len() < 2 {
            return Err(crate::Error::InvalidImage(
                "hfs+: short HFSUniStr255 length".into(),
            ));
        }
        let n = u16::from_be_bytes([buf[0], buf[1]]) as usize;
        if n > 255 {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: HFSUniStr255 length {n} exceeds 255"
            )));
        }
        let bytes_needed = 2 + 2 * n;
        if buf.len() < bytes_needed {
            return Err(crate::Error::InvalidImage(
                "hfs+: HFSUniStr255 truncated".into(),
            ));
        }
        let mut code_units = Vec::with_capacity(n);
        for i in 0..n {
            let off = 2 + 2 * i;
            code_units.push(u16::from_be_bytes([buf[off], buf[off + 1]]));
        }
        Ok((Self { code_units }, bytes_needed))
    }

    /// Build a `UniStr` from a Rust `&str` (UTF-16 code units, no
    /// normalisation). The host text MUST already be in the form HFS+
    /// uses (NFD); we don't normalise here — callers seeking a path
    /// component should match exactly what the disk stores.
    pub fn from_str_lossy(s: &str) -> Self {
        Self {
            code_units: s.encode_utf16().collect(),
        }
    }

    /// Lossy UTF-16 -> String conversion for display.
    pub fn to_string_lossy(&self) -> String {
        String::from_utf16_lossy(&self.code_units)
    }
}

/// Case-insensitive folding of a single Basic-Multilingual-Plane code unit,
/// using the simplified ASCII rule plus the common Latin-1 letters. The
/// real HFS+ folding table is far larger (TN1150 "Case-Insensitive String
/// Comparison Algorithm") — we approximate it adequately for ASCII path
/// components, which is the only case the path-resolver is asked to handle.
fn fold_case(c: u16) -> u16 {
    match c {
        0x41..=0x5A => c + 0x20,
        // Latin-1 uppercase A-O with diacritics and OE-equivalent
        0xC0..=0xD6 => c + 0x20,
        // Latin-1 uppercase O-Y with diacritics
        0xD8..=0xDE => c + 0x20,
        _ => c,
    }
}

/// Lexicographic compare of two HFSUniStr255 values using case-insensitive
/// folding for the "H+" variant. For "HX" we'd compare verbatim; the
/// catalog driver hands a flag.
pub fn compare_unistr(a: &UniStr, b: &UniStr, case_sensitive: bool) -> Ordering {
    let n = a.code_units.len().min(b.code_units.len());
    for i in 0..n {
        let (ai, bi) = if case_sensitive {
            (a.code_units[i], b.code_units[i])
        } else {
            (fold_case(a.code_units[i]), fold_case(b.code_units[i]))
        };
        match ai.cmp(&bi) {
            Ordering::Equal => continue,
            o => return o,
        }
    }
    a.code_units.len().cmp(&b.code_units.len())
}

/// A decoded catalog key.
///
/// ```text
/// offset  size  field
/// 0       2     keyLength       (length of bytes AFTER this u16)
/// 2       4     parentID        (HFSCatalogNodeID)
/// 6       2+2*N nodeName        (HFSUniStr255)
/// ```
#[derive(Debug, Clone)]
pub struct CatalogKey {
    pub parent_id: u32,
    pub name: UniStr,
    /// Total encoded length of the key including the key_length prefix
    /// (= keyLength + 2). Needed to skip past the key to the data.
    pub encoded_len: usize,
}

impl CatalogKey {
    /// Decode a catalog key from `buf`. The returned `encoded_len` is
    /// the field at offset 0 plus 2, padded up to an even byte count
    /// (records and their keys are 2-byte aligned per TN1150).
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < 6 {
            return Err(crate::Error::InvalidImage(
                "hfs+: short catalog key".into(),
            ));
        }
        let key_length = u16::from_be_bytes([buf[0], buf[1]]) as usize;
        let parent_id = u32::from_be_bytes(buf[2..6].try_into().unwrap());
        let (name, name_len) = UniStr::decode(&buf[6..])?;
        let used = 2 + key_length;
        // The on-disk key_length already covers parentID + nodeName;
        // sanity check it against what we just parsed.
        if 4 + name_len != key_length {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: catalog key_length {key_length} disagrees with parsed name \
                 (parent + {name_len} name bytes)"
            )));
        }
        let encoded_len = if used % 2 == 0 { used } else { used + 1 };
        Ok(Self {
            parent_id,
            name,
            encoded_len,
        })
    }

    /// Compare against another key. Catalog keys order by `parentID`
    /// first, then `nodeName` using the HFS+ case-folding rules (unless
    /// `case_sensitive`).
    pub fn compare(&self, other: &CatalogKey, case_sensitive: bool) -> Ordering {
        match self.parent_id.cmp(&other.parent_id) {
            Ordering::Equal => compare_unistr(&self.name, &other.name, case_sensitive),
            o => o,
        }
    }
}

/// POSIX-ish file metadata embedded in folder and file records.
///
/// HFSPlusBSDInfo (16 bytes):
/// ```text
/// offset  size  field
/// 0       4     ownerID
/// 4       4     groupID
/// 8       1     adminFlags
/// 9       1     ownerFlags
/// 10      2     fileMode
/// 12      4     special (rdev for char/block, link count for hard links)
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BsdInfo {
    pub owner_id: u32,
    pub group_id: u32,
    pub admin_flags: u8,
    pub owner_flags: u8,
    pub file_mode: u16,
    pub special: u32,
}

impl BsdInfo {
    fn decode(buf: &[u8]) -> Self {
        Self {
            owner_id: u32::from_be_bytes(buf[0..4].try_into().unwrap()),
            group_id: u32::from_be_bytes(buf[4..8].try_into().unwrap()),
            admin_flags: buf[8],
            owner_flags: buf[9],
            file_mode: u16::from_be_bytes(buf[10..12].try_into().unwrap()),
            special: u32::from_be_bytes(buf[12..16].try_into().unwrap()),
        }
    }
}

/// HFSPlusCatalogFolder — a folder leaf record (kHFSPlusFolderRecord).
///
/// ```text
/// offset  size  field
/// 0       2     recordType (= 0x0001)
/// 2       2     flags
/// 4       4     valence (number of immediate children)
/// 8       4     folderID (CNID of this folder)
/// 12      4     createDate
/// 16      4     contentModDate
/// 20      4     attributeModDate
/// 24      4     accessDate
/// 28      4     backupDate
/// 32      16    HFSPlusBSDInfo
/// 48      16    FolderInfo (Finder)
/// 64      16    ExtendedFolderInfo (Finder)
/// 80      4     textEncoding
/// 84      4     reserved
/// ```
#[derive(Debug, Clone)]
pub struct CatalogFolder {
    pub folder_id: u32,
    pub valence: u32,
    pub bsd: BsdInfo,
}

/// HFSPlusCatalogFile — a file leaf record (kHFSPlusFileRecord).
///
/// ```text
/// offset  size  field
/// 0       2     recordType (= 0x0002)
/// 2       2     flags
/// 4       4     reserved1
/// 8       4     fileID
/// 12-31         dates (5x 4 bytes)
/// 32      16    HFSPlusBSDInfo
/// 48      16    FileInfo
/// 64      16    ExtendedFileInfo
/// 80      4     textEncoding
/// 84      4     reserved2
/// 88      80    HFSPlusForkData dataFork
/// 168     80    HFSPlusForkData resourceFork
/// ```
#[derive(Debug, Clone)]
pub struct CatalogFile {
    pub file_id: u32,
    pub bsd: BsdInfo,
    pub data_fork: ForkData,
    pub resource_fork: ForkData,
}

/// Either thread record (folder or file) maps a CNID back to its
/// parent + name.
#[derive(Debug, Clone)]
pub struct CatalogThread {
    pub parent_id: u32,
    pub name: UniStr,
}

/// A decoded catalog leaf record body (after the key has been parsed).
#[derive(Debug, Clone)]
pub enum CatalogRecord {
    Folder(CatalogFolder),
    File(CatalogFile),
    Thread(CatalogThread),
}

impl CatalogRecord {
    /// Decode the body bytes of a catalog leaf record (excluding the
    /// preceding catalog key).
    pub fn decode(body: &[u8]) -> Result<Self> {
        if body.len() < 2 {
            return Err(crate::Error::InvalidImage(
                "hfs+: short catalog record body".into(),
            ));
        }
        let rec_type = i16::from_be_bytes([body[0], body[1]]);
        match rec_type {
            REC_FOLDER => Self::decode_folder(body),
            REC_FILE => Self::decode_file(body),
            REC_FOLDER_THREAD | REC_FILE_THREAD => Self::decode_thread(body),
            other => Err(crate::Error::InvalidImage(format!(
                "hfs+: unknown catalog record type {other:#06x}"
            ))),
        }
    }

    fn decode_folder(body: &[u8]) -> Result<Self> {
        if body.len() < 88 {
            return Err(crate::Error::InvalidImage(
                "hfs+: short HFSPlusCatalogFolder".into(),
            ));
        }
        let valence = u32::from_be_bytes(body[4..8].try_into().unwrap());
        let folder_id = u32::from_be_bytes(body[8..12].try_into().unwrap());
        let bsd = BsdInfo::decode(&body[32..48]);
        Ok(Self::Folder(CatalogFolder {
            folder_id,
            valence,
            bsd,
        }))
    }

    fn decode_file(body: &[u8]) -> Result<Self> {
        // recordType(2) + flags(2) + reserved1(4) + fileID(4) + 5*date(20)
        //   = 32, then BSDInfo(16) = 48, FinderInfo(32) = 80, textEncoding(4) +
        //   reserved2(4) = 88, dataFork(80) = 168, resourceFork(80) = 248.
        if body.len() < 248 {
            return Err(crate::Error::InvalidImage(
                "hfs+: short HFSPlusCatalogFile".into(),
            ));
        }
        let file_id = u32::from_be_bytes(body[8..12].try_into().unwrap());
        let bsd = BsdInfo::decode(&body[32..48]);
        let mut fbuf = [0u8; 80];
        fbuf.copy_from_slice(&body[88..168]);
        let data_fork = ForkData::decode(&fbuf);
        fbuf.copy_from_slice(&body[168..248]);
        let resource_fork = ForkData::decode(&fbuf);
        Ok(Self::File(CatalogFile {
            file_id,
            bsd,
            data_fork,
            resource_fork,
        }))
    }

    fn decode_thread(body: &[u8]) -> Result<Self> {
        // recordType(2) + reserved(2) + parentID(4) + HFSUniStr255
        if body.len() < 8 {
            return Err(crate::Error::InvalidImage(
                "hfs+: short catalog thread record".into(),
            ));
        }
        let parent_id = u32::from_be_bytes(body[4..8].try_into().unwrap());
        let (name, _) = UniStr::decode(&body[8..])?;
        Ok(Self::Thread(CatalogThread { parent_id, name }))
    }
}

/// Opened, ready-to-query catalog file.
pub struct Catalog {
    pub fork: ForkReader,
    pub header: BTreeHeader,
    pub case_sensitive: bool,
}

impl Catalog {
    /// Open the catalog by reading its B-tree header node (node 0).
    pub fn open(
        dev: &mut dyn BlockDevice,
        fork: ForkReader,
        case_sensitive: bool,
    ) -> Result<Self> {
        // Read just enough of node 0 to discover the real node size.
        // The header record always begins at byte 14 of node 0, immediately
        // after the BTNodeDescriptor.
        let mut bootstrap = vec![0u8; 512];
        fork.read(dev, 0, &mut bootstrap)?;
        let probe_desc = NodeDescriptor::decode(&bootstrap)?;
        if probe_desc.kind != super::btree::KIND_HEADER {
            return Err(crate::Error::InvalidImage(format!(
                "hfs+: catalog node 0 has kind {} (expected header)",
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

    /// Look up the record with exactly the given key. Returns the
    /// decoded record body, or `None` if no exact match exists.
    pub fn lookup(
        &self,
        dev: &mut dyn BlockDevice,
        wanted: &CatalogKey,
    ) -> Result<Option<CatalogRecord>> {
        let node_size = u32::from(self.header.node_size);
        let mut node_idx = self.header.root_node;
        if node_idx == 0 {
            // Empty tree.
            return Ok(None);
        }

        // Descend from root to leaf.
        loop {
            let node = read_node(dev, &self.fork, node_idx, node_size)?;
            let desc = NodeDescriptor::decode(&node)?;
            let offs = record_offsets(&node, desc.num_records)?;

            if desc.kind == KIND_LEAF {
                // Linear scan within the leaf — leaves are small.
                for i in 0..desc.num_records as usize {
                    let rec = record_bytes(&node, &offs, i);
                    let key = CatalogKey::decode(rec)?;
                    match key.compare(wanted, self.case_sensitive) {
                        Ordering::Equal => {
                            let body_start = align2(key.encoded_len);
                            if body_start > rec.len() {
                                return Err(crate::Error::InvalidImage(
                                    "hfs+: catalog record key overruns its slot".into(),
                                ));
                            }
                            let body = &rec[body_start..];
                            return Ok(Some(CatalogRecord::decode(body)?));
                        }
                        Ordering::Greater => return Ok(None),
                        Ordering::Less => continue,
                    }
                }
                return Ok(None);
            } else if desc.kind == KIND_INDEX {
                // Pick the largest key that is ≤ wanted; that subtree
                // is where wanted (if anywhere) lives.
                let mut child: Option<u32> = None;
                for i in 0..desc.num_records as usize {
                    let rec = record_bytes(&node, &offs, i);
                    let key = CatalogKey::decode(rec)?;
                    let pointer_off = align2(key.encoded_len);
                    if pointer_off + 4 > rec.len() {
                        return Err(crate::Error::InvalidImage(
                            "hfs+: index record missing child pointer".into(),
                        ));
                    }
                    let next = u32::from_be_bytes(rec[pointer_off..pointer_off + 4].try_into().unwrap());
                    match key.compare(wanted, self.case_sensitive) {
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
                    "hfs+: unexpected B-tree node kind {} in catalog traversal",
                    desc.kind
                )));
            }
        }
    }
}

/// Round `n` up to the next even value (catalog records are 2-byte aligned).
fn align2(n: usize) -> usize {
    n + (n & 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unistr_decode_round_trip() {
        // "ABcd" = 4 code units.
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u16.to_be_bytes());
        for c in "ABcd".encode_utf16() {
            buf.extend_from_slice(&c.to_be_bytes());
        }
        let (s, n) = UniStr::decode(&buf).unwrap();
        assert_eq!(n, 10);
        assert_eq!(s.code_units, vec![0x41, 0x42, 0x63, 0x64]);
        assert_eq!(s.to_string_lossy(), "ABcd");
    }

    #[test]
    fn unistr_decode_truncated_errors() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&3u16.to_be_bytes());
        buf.extend_from_slice(&0x41u16.to_be_bytes());
        // Only one character supplied; need three.
        assert!(UniStr::decode(&buf).is_err());
    }

    #[test]
    fn compare_unistr_case_insensitive_default() {
        let a = UniStr::from_str_lossy("Hello");
        let b = UniStr::from_str_lossy("hello");
        assert_eq!(compare_unistr(&a, &b, false), Ordering::Equal);
        // Case-sensitive: 'H' (0x48) < 'h' (0x68).
        assert_eq!(compare_unistr(&a, &b, true), Ordering::Less);
    }

    #[test]
    fn catalog_key_compare_orders_by_parent_then_name() {
        let k1 = CatalogKey {
            parent_id: 1,
            name: UniStr::from_str_lossy("z"),
            encoded_len: 0,
        };
        let k2 = CatalogKey {
            parent_id: 2,
            name: UniStr::from_str_lossy("a"),
            encoded_len: 0,
        };
        assert_eq!(k1.compare(&k2, false), Ordering::Less);

        let k3 = CatalogKey {
            parent_id: 2,
            name: UniStr::from_str_lossy("b"),
            encoded_len: 0,
        };
        // Same parent: name decides.
        assert_eq!(k2.compare(&k3, false), Ordering::Less);
    }

    #[test]
    fn catalog_key_decode_round_trip() {
        // Synthesise a key with parentID=42, name="hi" (2 code units).
        let name_bytes_len = 2 + 2 * 2; // length u16 + 2 code units
        let key_payload_len = 4 + name_bytes_len; // parentID + name
        let mut buf = Vec::new();
        buf.extend_from_slice(&(key_payload_len as u16).to_be_bytes());
        buf.extend_from_slice(&42u32.to_be_bytes());
        buf.extend_from_slice(&2u16.to_be_bytes());
        for c in "hi".encode_utf16() {
            buf.extend_from_slice(&c.to_be_bytes());
        }
        let key = CatalogKey::decode(&buf).unwrap();
        assert_eq!(key.parent_id, 42);
        assert_eq!(key.name.to_string_lossy(), "hi");
        // encoded_len = 2 (length field) + 4 (parentID) + 2 (count) + 4 (chars) = 12 (even).
        assert_eq!(key.encoded_len, 12);
    }
}
