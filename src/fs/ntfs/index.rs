//! NTFS directory-index decoding.
//!
//! Directory entries are stored as a B+-tree of $FILE_NAME-keyed records.
//! The root lives in the resident $INDEX_ROOT attribute; deeper nodes
//! live in $INDEX_ALLOCATION-backed `INDX` blocks. Each "index entry"
//! has a fixed 16-byte header followed by an optional key (here, a
//! $FILE_NAME payload) and an optional 8-byte VCN trailer pointing at a
//! child block.
//!
//! Reference: NTFS Documentation (Russon & Fledel), §INDX.

use crate::Result;

use super::attribute::FileName;

pub(crate) const INDX_RECORD_MAGIC: &[u8; 4] = b"INDX";

/// Index entry header flags.
pub const ENTRY_FLAG_HAS_CHILD: u32 = 0x01;
pub const ENTRY_FLAG_LAST: u32 = 0x02;

/// Decoded $INDEX_ROOT header.
#[derive(Debug, Clone)]
pub struct IndexRootHeader {
    /// Type code of the indexed attribute (0x30 = $FILE_NAME).
    pub indexed_attr_type: u32,
    /// Collation rule (Unicode, etc.).
    pub collation_rule: u32,
    /// Bytes per $INDEX_ALLOCATION block.
    pub index_block_size: u32,
    /// Clusters per index block (negative = power of two of bytes).
    pub clusters_per_index_block: i8,
    /// Offset of the first index entry, relative to the start of the
    /// _index header_ (not the root attribute).
    pub first_entry_offset: u32,
    /// Total bytes in use within the index header.
    pub bytes_in_use: u32,
    /// Total bytes allocated for the index header.
    pub bytes_allocated: u32,
    /// `INDEX_LARGE` flag (0x01) — a non-resident $INDEX_ALLOCATION exists.
    pub flags: u8,
    /// Offset within the root attribute value where the index header
    /// starts (i.e. where `first_entry_offset` is measured from).
    pub header_offset: usize,
}

impl IndexRootHeader {
    pub fn parse(value: &[u8]) -> Result<Self> {
        if value.len() < 32 {
            return Err(crate::Error::InvalidImage(
                "ntfs: $INDEX_ROOT too short".into(),
            ));
        }
        let indexed_attr_type = u32::from_le_bytes(value[0..4].try_into().unwrap());
        let collation_rule = u32::from_le_bytes(value[4..8].try_into().unwrap());
        let index_block_size = u32::from_le_bytes(value[8..12].try_into().unwrap());
        let clusters_per_index_block = value[12] as i8;
        // 13..16 padding
        // Index header begins at offset 16:
        let header_offset = 16usize;
        let first_entry_offset =
            u32::from_le_bytes(value[header_offset..header_offset + 4].try_into().unwrap());
        let bytes_in_use = u32::from_le_bytes(
            value[header_offset + 4..header_offset + 8]
                .try_into()
                .unwrap(),
        );
        let bytes_allocated = u32::from_le_bytes(
            value[header_offset + 8..header_offset + 12]
                .try_into()
                .unwrap(),
        );
        let flags = value[header_offset + 12];
        Ok(Self {
            indexed_attr_type,
            collation_rule,
            index_block_size,
            clusters_per_index_block,
            first_entry_offset,
            bytes_in_use,
            bytes_allocated,
            flags,
            header_offset,
        })
    }

    pub fn has_index_allocation(&self) -> bool {
        self.flags & 0x01 != 0
    }
}

/// One decoded entry from an index node.
#[derive(Debug, Clone)]
pub struct IndexEntry {
    /// MFT reference of the file this entry points at (0 for the last
    /// node-terminator entry).
    pub file_ref: u64,
    /// Decoded $FILE_NAME key, if this is a real (non-terminator) entry.
    pub file_name: Option<FileName>,
    /// VCN of the child index block, if `flags & HAS_CHILD`.
    pub child_vcn: Option<u64>,
}

/// Walk an index node (root or allocation block) starting at `entries_start`
/// within `buf`. Calls `visit` for each real entry; terminator stops the
/// walk. Returns the list of child VCNs (in order) so the caller can
/// descend.
pub fn walk_index_node(
    buf: &[u8],
    entries_start: usize,
    bytes_in_use: usize,
    mut visit: impl FnMut(&IndexEntry),
) -> Result<Vec<u64>> {
    let end = entries_start
        .checked_add(bytes_in_use)
        .ok_or_else(|| crate::Error::InvalidImage("ntfs: index walk overflow".into()))?;
    if end > buf.len() {
        return Err(crate::Error::InvalidImage(
            "ntfs: index node oversteps buffer".into(),
        ));
    }
    let mut cursor = entries_start;
    let mut children = Vec::new();
    while cursor + 16 <= end {
        let file_ref = u64::from_le_bytes(buf[cursor..cursor + 8].try_into().unwrap());
        let entry_len =
            u16::from_le_bytes(buf[cursor + 8..cursor + 10].try_into().unwrap()) as usize;
        let key_len =
            u16::from_le_bytes(buf[cursor + 10..cursor + 12].try_into().unwrap()) as usize;
        let flags = u32::from_le_bytes(buf[cursor + 12..cursor + 16].try_into().unwrap());

        if entry_len < 16 || cursor + entry_len > end {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: index entry length {entry_len} oversteps node"
            )));
        }

        let mut child_vcn = None;
        if flags & ENTRY_FLAG_HAS_CHILD != 0 {
            if entry_len < 8 {
                return Err(crate::Error::InvalidImage(
                    "ntfs: index entry with child but too short".into(),
                ));
            }
            let vcn_off = cursor + entry_len - 8;
            child_vcn = Some(u64::from_le_bytes(
                buf[vcn_off..vcn_off + 8].try_into().unwrap(),
            ));
        }

        let is_last = flags & ENTRY_FLAG_LAST != 0;
        let file_name = if !is_last && key_len > 0 {
            let key_start = cursor + 16;
            let key_end = key_start + key_len;
            if key_end > cursor + entry_len {
                return Err(crate::Error::InvalidImage(
                    "ntfs: index entry key oversteps entry".into(),
                ));
            }
            Some(FileName::parse(&buf[key_start..key_end])?)
        } else {
            None
        };

        let entry = IndexEntry {
            file_ref,
            file_name,
            child_vcn,
        };

        // Visit children in the in-order traversal: descend into the
        // child (if any) BEFORE emitting the entry — except that for
        // a B+-tree of NTFS the leaf-vs-internal distinction is per
        // entry. We collect children in entry order; the caller does the
        // recursion. The IS_LAST entry has no key but may still have a
        // child pointer (the rightmost subtree).
        if !is_last {
            visit(&entry);
        }
        if let Some(vcn) = entry.child_vcn {
            children.push(vcn);
        }

        cursor += entry_len;
        if is_last {
            break;
        }
    }
    Ok(children)
}

/// Decoded $INDEX_ALLOCATION block header (after USA fixup applied).
#[derive(Debug, Clone)]
pub struct IndexBlockHeader {
    /// Offset of the first entry relative to the _index header_ start
    /// (offset 0x18 inside the INDX block).
    pub first_entry_offset: u32,
    pub bytes_in_use: u32,
    pub bytes_allocated: u32,
    pub flags: u8,
}

impl IndexBlockHeader {
    /// Parse the INDX block's inner index header. The fixed `INDX` part
    /// occupies bytes 0..0x18; the index header itself starts at 0x18.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < 0x28 {
            return Err(crate::Error::InvalidImage(
                "ntfs: INDX block too small".into(),
            ));
        }
        if &buf[0..4] != INDX_RECORD_MAGIC {
            return Err(crate::Error::InvalidImage(format!(
                "ntfs: bad INDX magic {:02x?}",
                &buf[0..4]
            )));
        }
        let header_off = 0x18usize;
        let first_entry_offset =
            u32::from_le_bytes(buf[header_off..header_off + 4].try_into().unwrap());
        let bytes_in_use =
            u32::from_le_bytes(buf[header_off + 4..header_off + 8].try_into().unwrap());
        let bytes_allocated =
            u32::from_le_bytes(buf[header_off + 8..header_off + 12].try_into().unwrap());
        let flags = buf[header_off + 12];
        Ok(Self {
            first_entry_offset,
            bytes_in_use,
            bytes_allocated,
            flags,
        })
    }

    /// Offset within the block where the entry stream begins, in absolute
    /// terms (relative to the INDX block start).
    pub fn entries_start(&self) -> usize {
        0x18 + self.first_entry_offset as usize
    }

    /// Absolute end of the in-use entry stream.
    pub fn entries_end(&self) -> usize {
        0x18 + self.first_entry_offset as usize + self.bytes_in_use as usize
            - self.first_entry_offset as usize
    }

    /// Total bytes the walker should scan starting at `entries_start()`.
    pub fn entries_byte_len(&self) -> usize {
        // `bytes_in_use` is measured from the index header start (offset
        // 0x18 in the INDX block). It includes the 16-byte header **plus**
        // any padding / USA bytes that sit between the header and the
        // first entry. Subtract `first_entry_offset` to get the entries-
        // stream length exactly.
        (self.bytes_in_use as usize).saturating_sub(self.first_entry_offset as usize)
    }
}
