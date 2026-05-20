//! XFS extended-attribute leaf block (one-block "leaf form").
//!
//! When the shortform attribute area inside the inode overflows, XFS
//! promotes the attr fork to a separate leaf block referenced via the
//! inode's attribute-fork extent list (`di_aformat = EXTENTS`,
//! `di_anextents = 1`). The leaf block holds every (name, value) pair
//! in one allocation; a second overflow would push to *node* form
//! (multiple leaves indexed by an upper dabtree node), which is out of
//! scope for v1.
//!
//! On-disk layout for the v5 (CRC-enabled) variant — bytes are
//! big-endian unless noted:
//!
//! ```text
//!   xfs_da3_blkinfo (56 B):
//!      0   4   forw       __be32  forward sibling block (0 for the single leaf)
//!      4   4   back       __be32  backward sibling  (0)
//!      8   2   magic      __be16  XFS_ATTR3_LEAF_MAGIC = 0x3bee
//!     10   2   pad        __be16  zero
//!     12   4   crc        __le32  CRC32C of the block with this field zeroed
//!     16   8   blkno      __be64  basic-blocksize-units (512-B) block address
//!     24   8   lsn        __be64  log sequence number (zero on freshly-laid leaves)
//!     32  16   uuid       16 B   superblock metadata UUID
//!     48   8   owner      __be64  inode number that owns this leaf
//!
//!   xfs_attr3_leaf_hdr extension:
//!     56   2   count      __be16  number of attr entries
//!     58   2   usedbytes  __be16  bytes of name/value records consumed
//!     60   2   firstused  __be16  byte offset of the earliest name/value record
//!     62   1   holes      __u8    1 if the block needs compaction (we never set)
//!     63   1   pad1       __u8    alignment
//!     64  12   freemap[3] 3 × (base:__be16, size:__be16)
//!     76   4   pad2       __be32  alignment
//!     80   ──  end of header; entries[] begin here
//!
//!   xfs_attr_leaf_entry × count (8 B each), sorted by hashval ascending:
//!      0   4   hashval    __be32  xfs_da_hashname(name suffix)
//!      4   2   nameidx    __be16  byte offset of the name/value record
//!      6   1   flags      __u8    XFS_ATTR_LOCAL | XFS_ATTR_ROOT | XFS_ATTR_SECURE
//!      7   1   pad2       __u8    zero
//!
//!   xfs_attr_leaf_name_local (when XFS_ATTR_LOCAL is set), per record:
//!      0   2   valuelen   __be16
//!      2   1   namelen    __u8
//!      3   ──  name bytes followed immediately by value bytes
//!                (no terminator; aligned to 32-bit afterwards)
//! ```
//!
//! Name/value records grow downward from the end of the block; the
//! entries array grows upward from the header. `firstused` points at
//! the lowest occupied byte in the name/value region, so the free
//! region is the range `header_end + count*8 .. firstused`.
//!
//! Sources: `xfs.org`'s public XFS Filesystem Structure spec
//! (sections 8.2 "Leaf Attributes", 6.3 "Leaf Directories" for the
//! `xfs_da_blkinfo` header). The v5 `xfs_da3_blkinfo` extension and the
//! `xfs_attr3_leaf_hdr` pad2 are documented in the same spec under
//! the v3 CRC structures.

use crate::Result;

/// v5 / v3 attribute-leaf magic ("3" + "BEE" → 0x3bee).
pub const XFS_ATTR3_LEAF_MAGIC: u16 = 0x3bee;

/// v4 (legacy, no-CRC) attribute-leaf magic.
pub const XFS_ATTR_LEAF_MAGIC: u16 = 0xfbee;

/// Size of the v5 attribute-leaf header (xfs_da3_blkinfo + leaf hdr).
pub const XFS_ATTR3_LEAF_HDR_SIZE: usize = 80;

/// Size of the v4 attribute-leaf header.
pub const XFS_ATTR_LEAF_HDR_SIZE: usize = 32;

/// Size of one xfs_attr_leaf_entry record.
pub const XFS_ATTR_LEAF_ENTRY_SIZE: usize = 8;

/// Number of freespace slots tracked in the leaf header.
pub const XFS_ATTR_LEAF_MAPSIZE: usize = 3;

/// Byte offset of `crc` inside the v5 attr-leaf header (within `info`).
pub const XFS_ATTR3_LEAF_CRC_OFF: usize = 12;

// --- entry flag bits -------------------------------------------------
//
// These match the public XFS spec / xfs_db column ordering
// `[hashval,nameidx,incomplete,root,secure,local]` for the leaf flag
// byte. The shortform-area flag byte uses a *different* convention
// (see `super::xattr`) — keep the two mappings separate.

/// Leaf-entry flag: value stored inline in this same leaf block.
pub const XFS_ATTR_LOCAL: u8 = 0x01;
/// Leaf-entry flag: name belongs to the trusted ("root") namespace.
pub const XFS_ATTR_ROOT: u8 = 0x02;
/// Leaf-entry flag: name belongs to the security namespace.
pub const XFS_ATTR_SECURE: u8 = 0x04;
/// Leaf-entry flag: attribute is in the middle of being added/removed.
pub const XFS_ATTR_INCOMPLETE: u8 = 0x80;

/// Map a userland xattr name (`"user.foo"`, `"trusted.bar"`,
/// `"security.selinux"`) to (suffix, leaf-flag-byte).
pub fn leaf_name_to_disk(name: &str) -> (String, u8) {
    if let Some(rest) = name.strip_prefix("user.") {
        (rest.to_string(), 0)
    } else if let Some(rest) = name.strip_prefix("trusted.") {
        (rest.to_string(), XFS_ATTR_ROOT)
    } else if let Some(rest) = name.strip_prefix("security.") {
        (rest.to_string(), XFS_ATTR_SECURE)
    } else {
        (name.to_string(), 0)
    }
}

/// Inverse of [`leaf_name_to_disk`].
pub fn leaf_name_from_disk(suffix: &str, flags: u8) -> String {
    if flags & XFS_ATTR_ROOT != 0 {
        format!("trusted.{suffix}")
    } else if flags & XFS_ATTR_SECURE != 0 {
        format!("security.{suffix}")
    } else {
        format!("user.{suffix}")
    }
}

/// XFS directory / xattr name hash — same algorithm as
/// `super::dir::dahashname`, re-exposed here to avoid an upward
/// dependency on the `dir` module from the leaf decoder's tests.
pub fn dahashname(name: &[u8]) -> u32 {
    super::dir::dahashname(name)
}

/// Local-record size in bytes for a (name, value) pair. The record is
/// `valuelen(2) + namelen(1) + name + value`, then 32-bit aligned.
fn local_record_size(namelen: usize, valuelen: usize) -> usize {
    let raw = 2 + 1 + namelen + valuelen;
    (raw + 3) & !3
}

/// Total bytes the entries[] array consumes for `n` entries.
fn entries_size(n: usize) -> usize {
    n * XFS_ATTR_LEAF_ENTRY_SIZE
}

/// Compute the **minimum** v5 leaf-block size that holds these
/// attributes plus header + entries[]. Useful for sizing decisions
/// before allocation. The result is rounded up to the next multiple of
/// 8 to honour the 64-bit alignment requirement of `firstused`.
pub fn min_leaf_block_size(attrs: &[(String, Vec<u8>)]) -> usize {
    let mut bytes = XFS_ATTR3_LEAF_HDR_SIZE + entries_size(attrs.len());
    for (name, value) in attrs {
        let (suffix, _flags) = leaf_name_to_disk(name);
        bytes += local_record_size(suffix.len(), value.len());
    }
    (bytes + 7) & !7
}

/// Encode a v5 leaf attribute block for the given attributes, padding
/// out to `block_size`. `owner` is the inode that owns this block;
/// `uuid` is the superblock metadata UUID; `blkno` is the basic-512B
/// block address on disk (`device_byte / 512`). CRC is stamped at the
/// end.
pub fn encode_v5_leaf(
    attrs: &[(String, Vec<u8>)],
    block_size: usize,
    owner: u64,
    uuid: &[u8; 16],
    blkno: u64,
) -> Result<Vec<u8>> {
    if block_size < min_leaf_block_size(attrs) {
        return Err(crate::Error::InvalidArgument(format!(
            "xfs: leaf xattr block size {block_size} too small for {} attrs",
            attrs.len()
        )));
    }
    if attrs.len() > u16::MAX as usize {
        return Err(crate::Error::InvalidArgument(
            "xfs: too many xattrs for one leaf block".into(),
        ));
    }
    let mut block = vec![0u8; block_size];

    // xfs_da3_blkinfo — forw/back = 0, magic = 0x3bee, pad = 0, crc
    // stamped last, blkno, lsn = 0, uuid, owner.
    block[8..10].copy_from_slice(&XFS_ATTR3_LEAF_MAGIC.to_be_bytes());
    block[16..24].copy_from_slice(&blkno.to_be_bytes());
    block[32..48].copy_from_slice(uuid);
    block[48..56].copy_from_slice(&owner.to_be_bytes());

    // We lay name/value records out *in entries[] order* — i.e. by
    // ascending hashval — at the bottom of the block. `firstused`
    // becomes the lowest record offset. Records are 32-bit aligned;
    // the entire occupied tail of the block is contiguous (no holes).
    let count = attrs.len();
    let entries_start = XFS_ATTR3_LEAF_HDR_SIZE;
    let entries_end = entries_start + entries_size(count);

    // Pre-compute per-entry (suffix, flags, hashval, record_size).
    let mut records: Vec<(String, u8, u32, Vec<u8>)> = Vec::with_capacity(count);
    for (name, value) in attrs {
        let (suffix, flags) = leaf_name_to_disk(name);
        let hash = dahashname(suffix.as_bytes());
        if suffix.len() > u8::MAX as usize {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: leaf xattr name {name:?} suffix > 255 bytes"
            )));
        }
        if value.len() > u16::MAX as usize {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: leaf xattr value for {name:?} > 65535 bytes"
            )));
        }
        records.push((suffix, flags, hash, value.clone()));
    }
    // Sort by ascending hashval — required by XFS for binary lookup.
    // Ties are broken by name comparison so the order is deterministic.
    records.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));

    // Place name/value records, descending from end-of-block. We track
    // the next free byte; each record's `nameidx` is its starting
    // offset.
    let mut next_end = block_size;
    let mut nameidxs: Vec<u16> = Vec::with_capacity(count);
    let mut usedbytes_total: u16 = 0;
    for (suffix, _flags, _hash, value) in &records {
        let rsize = local_record_size(suffix.len(), value.len());
        if next_end < entries_end + rsize {
            return Err(crate::Error::InvalidArgument(
                "xfs: leaf xattr block too small (entries+records collide)".into(),
            ));
        }
        next_end -= rsize;
        let off = next_end;
        // Lay out the local record.
        block[off..off + 2].copy_from_slice(&(value.len() as u16).to_be_bytes());
        block[off + 2] = suffix.len() as u8;
        let name_off = off + 3;
        block[name_off..name_off + suffix.len()].copy_from_slice(suffix.as_bytes());
        let val_off = name_off + suffix.len();
        block[val_off..val_off + value.len()].copy_from_slice(value);
        // Bytes between val_off+value.len() and off+rsize remain zero
        // (alignment padding).
        nameidxs.push(off as u16);
        usedbytes_total = usedbytes_total
            .checked_add(rsize as u16)
            .ok_or_else(|| crate::Error::InvalidArgument("xfs: leaf usedbytes overflow".into()))?;
    }

    // Stamp the entries[] array.
    for (i, (_suffix, flags, hash, _value)) in records.iter().enumerate() {
        let e_off = entries_start + i * XFS_ATTR_LEAF_ENTRY_SIZE;
        block[e_off..e_off + 4].copy_from_slice(&hash.to_be_bytes());
        block[e_off + 4..e_off + 6].copy_from_slice(&nameidxs[i].to_be_bytes());
        block[e_off + 6] = XFS_ATTR_LOCAL | *flags;
        // pad2 at e_off+7 stays zero.
    }

    // Header fields after info{}.
    block[56..58].copy_from_slice(&(count as u16).to_be_bytes());
    block[58..60].copy_from_slice(&usedbytes_total.to_be_bytes());
    block[60..62].copy_from_slice(&(next_end as u16).to_be_bytes());
    // holes = 0, pad1 = 0 already.
    // freemap[0] = the gap between entries[] and the lowest record.
    let free_base = entries_end as u16;
    let free_size = (next_end - entries_end) as u16;
    block[64..66].copy_from_slice(&free_base.to_be_bytes());
    block[66..68].copy_from_slice(&free_size.to_be_bytes());
    // freemap[1] and freemap[2] stay zero.
    // pad2 (76..80) stays zero.

    stamp_v5_leaf_crc(&mut block);
    Ok(block)
}

/// Compute and store the CRC32C of a v5 attr-leaf block in place.
pub fn stamp_v5_leaf_crc(block: &mut [u8]) {
    block[XFS_ATTR3_LEAF_CRC_OFF..XFS_ATTR3_LEAF_CRC_OFF + 4].copy_from_slice(&[0u8; 4]);
    let crc = crc32c::crc32c(block);
    block[XFS_ATTR3_LEAF_CRC_OFF..XFS_ATTR3_LEAF_CRC_OFF + 4].copy_from_slice(&crc.to_le_bytes());
}

/// Decode a v5 (or v4) attribute leaf block into a flat
/// `HashMap<userland_name, value>`. Local-format entries only — remote
/// (out-of-line) value blocks aren't supported and surface
/// `Error::Unsupported`. Incomplete entries are skipped (they're in the
/// middle of a kernel-side transaction and should never be exposed).
pub fn decode_leaf(block: &[u8]) -> Result<std::collections::HashMap<String, Vec<u8>>> {
    if block.len() < 12 {
        return Err(crate::Error::InvalidImage(
            "xfs: attr leaf block too small for blkinfo".into(),
        ));
    }
    let magic = u16::from_be_bytes(block[8..10].try_into().unwrap());
    let (hdr_size, is_v5) = match magic {
        XFS_ATTR3_LEAF_MAGIC => (XFS_ATTR3_LEAF_HDR_SIZE, true),
        XFS_ATTR_LEAF_MAGIC => (XFS_ATTR_LEAF_HDR_SIZE, false),
        other => {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: bad attr-leaf magic {other:#06x}"
            )));
        }
    };
    if block.len() < hdr_size {
        return Err(crate::Error::InvalidImage(
            "xfs: attr leaf block shorter than its header".into(),
        ));
    }
    // count / usedbytes / firstused live in the same relative order in
    // both header variants; their absolute offsets differ.
    let (count_off, _usedbytes_off, _firstused_off) = if is_v5 {
        (56usize, 58usize, 60usize)
    } else {
        (12usize, 14usize, 16usize)
    };
    let count = u16::from_be_bytes(block[count_off..count_off + 2].try_into().unwrap()) as usize;
    let entries_start = hdr_size;
    let entries_end = entries_start + count * XFS_ATTR_LEAF_ENTRY_SIZE;
    if entries_end > block.len() {
        return Err(crate::Error::InvalidImage(
            "xfs: attr leaf entries[] runs past end of block".into(),
        ));
    }

    let mut out = std::collections::HashMap::with_capacity(count);
    for i in 0..count {
        let e_off = entries_start + i * XFS_ATTR_LEAF_ENTRY_SIZE;
        let nameidx = u16::from_be_bytes(block[e_off + 4..e_off + 6].try_into().unwrap()) as usize;
        let flags = block[e_off + 6];
        if flags & XFS_ATTR_INCOMPLETE != 0 {
            // Mid-transaction entry; don't expose.
            continue;
        }
        if flags & XFS_ATTR_LOCAL == 0 {
            return Err(crate::Error::Unsupported(
                "xfs: remote-value xattrs in leaf form not supported".into(),
            ));
        }
        if nameidx + 3 > block.len() {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: attr leaf entry {i} nameidx {nameidx} past block end"
            )));
        }
        let valuelen = u16::from_be_bytes(block[nameidx..nameidx + 2].try_into().unwrap()) as usize;
        let namelen = block[nameidx + 2] as usize;
        let name_start = nameidx + 3;
        let name_end = name_start + namelen;
        let val_end = name_end + valuelen;
        if val_end > block.len() {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: attr leaf entry {i} name/value runs past block end"
            )));
        }
        let suffix = std::str::from_utf8(&block[name_start..name_end])
            .map_err(|_| crate::Error::InvalidImage("xfs: non-UTF-8 leaf xattr name".into()))?;
        // Mask out LOCAL/INCOMPLETE so leaf_name_from_disk only sees
        // namespace bits.
        let ns_flags = flags & (XFS_ATTR_ROOT | XFS_ATTR_SECURE);
        let full_name = leaf_name_from_disk(suffix, ns_flags);
        let value = block[name_end..val_end].to_vec();
        out.insert(full_name, value);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_two_attrs() {
        let attrs = vec![
            ("user.mime_type".to_string(), b"text/plain".to_vec()),
            ("trusted.foo".to_string(), b"bar".to_vec()),
        ];
        let uuid = [0xAB; 16];
        let block = encode_v5_leaf(&attrs, 4096, 128, &uuid, 8).unwrap();
        let decoded = decode_leaf(&block).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.get("user.mime_type"), Some(&b"text/plain".to_vec()));
        assert_eq!(decoded.get("trusted.foo"), Some(&b"bar".to_vec()));
    }

    #[test]
    fn round_trip_many_small_attrs() {
        // 16 attrs — well past what shortform's 256-byte literal area
        // would fit, exercising the leaf path.
        let mut attrs = Vec::new();
        for i in 0..16 {
            attrs.push((format!("user.k{i}"), format!("v{i}").into_bytes()));
        }
        let block = encode_v5_leaf(&attrs, 4096, 200, &[0u8; 16], 16).unwrap();
        let decoded = decode_leaf(&block).unwrap();
        assert_eq!(decoded.len(), 16);
        for i in 0..16 {
            assert_eq!(
                decoded.get(&format!("user.k{i}")),
                Some(&format!("v{i}").into_bytes())
            );
        }
    }

    #[test]
    fn round_trip_empty_value() {
        let attrs = vec![("user.flag".to_string(), Vec::new())];
        let block = encode_v5_leaf(&attrs, 4096, 1, &[0; 16], 1).unwrap();
        let decoded = decode_leaf(&block).unwrap();
        assert_eq!(decoded.get("user.flag"), Some(&Vec::new()));
    }

    #[test]
    fn round_trip_all_three_namespaces() {
        let attrs = vec![
            ("user.a".to_string(), b"u".to_vec()),
            ("trusted.b".to_string(), b"t".to_vec()),
            ("security.c".to_string(), b"s".to_vec()),
        ];
        let block = encode_v5_leaf(&attrs, 4096, 1, &[0; 16], 1).unwrap();
        let decoded = decode_leaf(&block).unwrap();
        assert_eq!(decoded.get("user.a"), Some(&b"u".to_vec()));
        assert_eq!(decoded.get("trusted.b"), Some(&b"t".to_vec()));
        assert_eq!(decoded.get("security.c"), Some(&b"s".to_vec()));
    }

    #[test]
    fn reject_oversize_block() {
        let attrs = vec![("user.k".to_string(), vec![0u8; 8000])];
        // 256-byte block can't hold an 8 KiB value.
        assert!(encode_v5_leaf(&attrs, 256, 1, &[0; 16], 1).is_err());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut block = vec![0u8; 4096];
        block[8..10].copy_from_slice(&0xdeadu16.to_be_bytes());
        let r = decode_leaf(&block);
        assert!(matches!(r, Err(crate::Error::InvalidImage(_))));
    }

    #[test]
    fn skips_incomplete_entries() {
        // Hand-build a block with one good + one incomplete entry.
        let attrs = vec![
            ("user.good".to_string(), b"g".to_vec()),
            ("user.bad".to_string(), b"b".to_vec()),
        ];
        let mut block = encode_v5_leaf(&attrs, 4096, 1, &[0; 16], 1).unwrap();
        // Find the entry whose suffix == "bad" and stamp INCOMPLETE.
        let count = u16::from_be_bytes(block[56..58].try_into().unwrap()) as usize;
        for i in 0..count {
            let e_off = XFS_ATTR3_LEAF_HDR_SIZE + i * XFS_ATTR_LEAF_ENTRY_SIZE;
            let nameidx =
                u16::from_be_bytes(block[e_off + 4..e_off + 6].try_into().unwrap()) as usize;
            let namelen = block[nameidx + 2] as usize;
            let name = &block[nameidx + 3..nameidx + 3 + namelen];
            if name == b"bad" {
                block[e_off + 6] |= XFS_ATTR_INCOMPLETE;
            }
        }
        // CRC is now stale — but `decode_leaf` doesn't verify it, so the
        // decode still works. We'd re-stamp before persisting.
        let decoded = decode_leaf(&block).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded.get("user.good"), Some(&b"g".to_vec()));
        assert!(!decoded.contains_key("user.bad"));
    }

    #[test]
    fn hashval_ascending_in_entries() {
        let attrs = vec![
            ("user.zzz".to_string(), b"1".to_vec()),
            ("user.aaa".to_string(), b"2".to_vec()),
            ("user.mmm".to_string(), b"3".to_vec()),
        ];
        let block = encode_v5_leaf(&attrs, 4096, 1, &[0; 16], 1).unwrap();
        let count = u16::from_be_bytes(block[56..58].try_into().unwrap()) as usize;
        let mut prev = 0u32;
        for i in 0..count {
            let e_off = XFS_ATTR3_LEAF_HDR_SIZE + i * XFS_ATTR_LEAF_ENTRY_SIZE;
            let h = u32::from_be_bytes(block[e_off..e_off + 4].try_into().unwrap());
            assert!(h >= prev, "hashvals must be non-decreasing in entries[]");
            prev = h;
        }
    }
}
