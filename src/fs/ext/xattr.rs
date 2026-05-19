//! Extended attributes (xattrs) for ext2/3/4.
//!
//! ext stores per-inode xattrs in an external "xattr block" pointed at
//! by `inode.file_acl`. Inode-body xattrs (post-128-byte extended inode
//! area) and EA inodes (values stored in their own inode) are NOT
//! supported in this v1 — the block format covers ~1 KiB to 1 MiB of
//! xattr payload, which fits virtually every real use case (SELinux
//! labels, capabilities, POSIX ACLs, small user-namespace metadata).
//!
//! Layout — one filesystem block:
//!
//! ```text
//!   0   4  h_magic     = 0xEA020000
//!   4   4  h_refcount  = 1 (we never share xattr blocks)
//!   8   4  h_blocks    = 1
//!  12   4  h_hash      = chained hash of all entries
//!  16   4  h_checksum  = CRC32C(uuid + block_nr + block-with-csum-zeroed)
//!  20  12  h_reserved  = 0
//!  32     entries (each padded to 4 bytes), terminated by a 4-byte zero word
//!  ...    values stored backwards from the end of the block
//! ```
//!
//! Entry header (16 bytes):
//!
//! ```text
//!   0  1  e_name_len
//!   1  1  e_name_index   (namespace prefix code; the on-disk name omits the prefix)
//!   2  2  e_value_offs   (byte offset within the block)
//!   4  4  e_value_inum   (0 for inline values — we only do inline)
//!   8  4  e_value_size
//!  12  4  e_value_hash
//!  16  ?  e_name[e_name_len]   (padded to next 4-byte boundary)
//! ```
//!
//! Namespaces are coded as integers so common prefixes ("user.",
//! "trusted.", "security.") don't get written on disk for every entry:
//!
//! | code | full prefix                  |
//! |------|------------------------------|
//! | 1    | `user.`                      |
//! | 2    | `system.posix_acl_access` (no suffix) |
//! | 3    | `system.posix_acl_default` (no suffix) |
//! | 4    | `trusted.`                   |
//! | 6    | `security.`                  |
//! | 7    | `system.`                    |
//! | 0    | none — raw name              |

use crate::Result;

/// "EA02 0000" — the ext xattr block magic, little-endian.
pub const MAGIC: u32 = 0xEA02_0000;
/// 32 bytes of header before the first entry.
pub const HEADER_SIZE: usize = 32;
/// 16 bytes per entry header (before the variable-length name).
pub const ENTRY_HEADER_SIZE: usize = 16;
/// On-disk entries are padded to a 4-byte boundary.
pub const PAD: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xattr {
    /// Full attribute name including the namespace prefix
    /// (e.g. `"user.something"`, `"security.selinux"`).
    pub name: String,
    pub value: Vec<u8>,
}

impl Xattr {
    pub fn new(name: impl Into<String>, value: impl Into<Vec<u8>>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// Map a full xattr name to its on-disk `(name_index, suffix)` pair.
/// Names that don't match a known prefix get `name_index = 0` and the
/// full name as the suffix.
pub fn name_index_and_suffix(full_name: &str) -> (u8, &str) {
    if full_name == "system.posix_acl_access" {
        return (2, "");
    }
    if full_name == "system.posix_acl_default" {
        return (3, "");
    }
    if let Some(s) = full_name.strip_prefix("user.") {
        return (1, s);
    }
    if let Some(s) = full_name.strip_prefix("trusted.") {
        return (4, s);
    }
    if let Some(s) = full_name.strip_prefix("security.") {
        return (6, s);
    }
    if let Some(s) = full_name.strip_prefix("system.") {
        return (7, s);
    }
    (0, full_name)
}

/// Reverse [`name_index_and_suffix`] — reconstruct a full xattr name
/// from an on-disk `(name_index, suffix)` pair.
pub fn rejoin_name(name_index: u8, suffix: &str) -> String {
    let prefix = match name_index {
        1 => "user.",
        2 => "system.posix_acl_access",
        3 => "system.posix_acl_default",
        4 => "trusted.",
        6 => "security.",
        7 => "system.",
        _ => "",
    };
    // Indices 2 and 3 are full names with no suffix.
    if matches!(name_index, 2 | 3) {
        prefix.to_string()
    } else {
        format!("{prefix}{suffix}")
    }
}

const NAME_HASH_SHIFT: u32 = 5;
const VALUE_HASH_SHIFT: u32 = 16;
const BLOCK_HASH_SHIFT: u32 = 16;

/// Kernel-compatible per-entry hash: the entry's name (with namespace
/// prefix stripped, per the on-disk format) chained with its value
/// processed as 32-bit words.
pub fn entry_hash(name: &str, value: &[u8]) -> u32 {
    let mut h = 0u32;
    for &b in name.as_bytes() {
        h = (h << NAME_HASH_SHIFT) ^ (h >> (32 - NAME_HASH_SHIFT)) ^ (b as u32);
    }
    // Values are hashed as little-endian 32-bit words. The trailing
    // partial word is zero-padded — the kernel does the same.
    for chunk in value.chunks(4) {
        let mut w = [0u8; 4];
        w[..chunk.len()].copy_from_slice(chunk);
        let word = u32::from_le_bytes(w);
        h = (h << VALUE_HASH_SHIFT) ^ (h >> (32 - VALUE_HASH_SHIFT)) ^ word;
    }
    h
}

/// Block hash = chained per-entry hashes (kernel's `ext4_xattr_rehash`).
fn block_hash(entry_hashes: &[u32]) -> u32 {
    let mut h = 0u32;
    for &e in entry_hashes {
        h = (h << BLOCK_HASH_SHIFT) ^ (h >> (32 - BLOCK_HASH_SHIFT)) ^ e;
    }
    h
}

/// Encode `xattrs` into a single filesystem block of size `block_size`.
/// The returned buffer is fully populated (entries forward from the
/// header, values backward from the end). The caller is responsible for
/// stamping the CRC32C checksum at `[16..20]` after writing — call
/// [`stamp_checksum`] with the volume UUID and the block number.
pub fn encode_block(xattrs: &[Xattr], block_size: usize) -> Result<Vec<u8>> {
    if block_size < HEADER_SIZE + ENTRY_HEADER_SIZE + 4 {
        return Err(crate::Error::InvalidArgument(format!(
            "ext: xattr block_size {block_size} too small"
        )));
    }
    let mut block = vec![0u8; block_size];
    // Header
    block[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    block[4..8].copy_from_slice(&1u32.to_le_bytes()); // refcount
    block[8..12].copy_from_slice(&1u32.to_le_bytes()); // blocks

    // Sort by (name_index ASC, suffix ASC) — the kernel inserts in this
    // order and some validators (e2fsck) flag out-of-order entries.
    let mut sorted: Vec<(u8, &str, &Xattr)> = xattrs
        .iter()
        .map(|x| {
            let (idx, suffix) = name_index_and_suffix(&x.name);
            (idx, suffix, x)
        })
        .collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)));

    let mut entry_pos = HEADER_SIZE;
    let mut value_pos = block_size;
    let mut hashes: Vec<u32> = Vec::with_capacity(sorted.len());

    for (idx, suffix, xa) in sorted {
        let name_bytes = suffix.as_bytes();
        let name_len = name_bytes.len();
        if name_len > 255 {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: xattr name {:?} exceeds 255 bytes",
                xa.name
            )));
        }
        let entry_size = ENTRY_HEADER_SIZE + name_len;
        let entry_pad = entry_size.next_multiple_of(PAD);
        let value_size = xa.value.len();
        let value_pad = value_size.next_multiple_of(PAD);

        // Account for the 4-byte zero end-marker after the last entry.
        // Rearranged to avoid underflow when value_pad > value_pos.
        if entry_pos + entry_pad + 4 + value_pad > value_pos {
            return Err(crate::Error::InvalidArgument(format!(
                "ext: xattrs don't fit in a single {block_size}-byte block"
            )));
        }

        let new_value_off = value_pos - value_pad;
        // Entry header
        block[entry_pos] = name_len as u8;
        block[entry_pos + 1] = idx;
        block[entry_pos + 2..entry_pos + 4].copy_from_slice(&(new_value_off as u16).to_le_bytes());
        // e_value_inum at +4..+8 stays 0 (inline value).
        block[entry_pos + 8..entry_pos + 12].copy_from_slice(&(value_size as u32).to_le_bytes());
        let h = entry_hash(suffix, &xa.value);
        block[entry_pos + 12..entry_pos + 16].copy_from_slice(&h.to_le_bytes());
        // Name
        block[entry_pos + 16..entry_pos + 16 + name_len].copy_from_slice(name_bytes);
        // (padding bytes are already zero)

        // Value
        block[new_value_off..new_value_off + value_size].copy_from_slice(&xa.value);

        hashes.push(h);
        entry_pos += entry_pad;
        value_pos = new_value_off;
    }
    // 4-byte zero end marker is already in place because the block was
    // zero-initialised and we never wrote into [entry_pos..entry_pos+4].

    let bh = block_hash(&hashes);
    block[12..16].copy_from_slice(&bh.to_le_bytes());
    Ok(block)
}

/// Decode inline xattrs from the extended-inode area (`inode_size > 128`).
///
/// `inline_area` is the slice starting at byte `128 + i_extra_isize`
/// within the inode — i.e. starting at the 4-byte magic. ext4 stores
/// `value_offs` in each entry relative to "the byte right after the
/// 4-byte magic", which differs from the block-xattr convention where
/// `value_offs` is from the block start. Returns an empty vec if the
/// magic isn't present (no inline xattrs).
pub fn decode_inline(inline_area: &[u8]) -> Result<Vec<Xattr>> {
    if inline_area.len() < 4 {
        return Ok(Vec::new());
    }
    let magic = u32::from_le_bytes(inline_area[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Ok(Vec::new());
    }
    // Inside the inline area, `value_offs` is measured from the start
    // of the entry array — i.e. payload[0..] — not from the magic.
    let payload = &inline_area[4..];
    decode_payload(payload, payload)
}

/// Decode an xattr block into a list of [`Xattr`]s. Validates the magic
/// but does NOT validate the per-block CRC32C — the caller is expected
/// to do that against the volume UUID + block number when
/// `metadata_csum` is active.
pub fn decode_block(block: &[u8]) -> Result<Vec<Xattr>> {
    if block.len() < HEADER_SIZE {
        return Err(crate::Error::InvalidImage(
            "ext: xattr block shorter than header".into(),
        ));
    }
    let magic = u32::from_le_bytes(block[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Err(crate::Error::InvalidImage(format!(
            "ext: bad xattr block magic {magic:#010x} (expected {MAGIC:#010x})"
        )));
    }
    // Block xattrs: `value_offs` is measured from the start of the
    // block (so the value lookup base IS the block itself).
    decode_payload(&block[HEADER_SIZE..], block)
}

/// Shared core: walk entries in `entries` (terminated by a zero-byte
/// name_len) and resolve each entry's value via `value_base[value_offs..]`.
fn decode_payload(entries: &[u8], value_base: &[u8]) -> Result<Vec<Xattr>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + ENTRY_HEADER_SIZE <= entries.len() {
        let name_len = entries[pos] as usize;
        if name_len == 0 {
            break;
        }
        let name_index = entries[pos + 1];
        let value_offs = u16::from_le_bytes(entries[pos + 2..pos + 4].try_into().unwrap()) as usize;
        let value_inum = u32::from_le_bytes(entries[pos + 4..pos + 8].try_into().unwrap());
        let value_size =
            u32::from_le_bytes(entries[pos + 8..pos + 12].try_into().unwrap()) as usize;
        if value_inum != 0 {
            return Err(crate::Error::Unsupported(
                "ext: EA-inode xattr values are not supported".into(),
            ));
        }
        if pos + ENTRY_HEADER_SIZE + name_len > entries.len() {
            return Err(crate::Error::InvalidImage(
                "ext: xattr entry header runs past end of payload".into(),
            ));
        }
        let name_bytes = &entries[pos + ENTRY_HEADER_SIZE..pos + ENTRY_HEADER_SIZE + name_len];
        let suffix = std::str::from_utf8(name_bytes)
            .map_err(|_| crate::Error::InvalidImage("ext: xattr name is not valid UTF-8".into()))?;
        if value_offs + value_size > value_base.len() {
            return Err(crate::Error::InvalidImage(
                "ext: xattr value range out of bounds".into(),
            ));
        }
        let value = value_base[value_offs..value_offs + value_size].to_vec();
        out.push(Xattr {
            name: rejoin_name(name_index, suffix),
            value,
        });
        pos += (ENTRY_HEADER_SIZE + name_len).next_multiple_of(PAD);
    }
    Ok(out)
}

/// CRC32C the xattr block per ext4's rules: seed → block number as
/// LE u64 (always 8 bytes; the kernel uses `__le64 dsk_block_nr`
/// regardless of `INCOMPAT_64BIT`) → block bytes with the `h_checksum`
/// field zeroed. Stamps the result into `block[16..20]`.
pub fn stamp_checksum(block: &mut [u8], seed: u32, block_number: u64) {
    block[16..20].copy_from_slice(&0u32.to_le_bytes());
    let c = super::csum::raw_update(seed, &block_number.to_le_bytes());
    let c = super::csum::raw_update(c, block);
    block[16..20].copy_from_slice(&c.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_user_xattrs() {
        let xs = vec![
            Xattr::new("user.foo", b"foo-value".to_vec()),
            Xattr::new(
                "security.selinux",
                b"system_u:object_r:unlabeled_t:s0\0".to_vec(),
            ),
            Xattr::new("trusted.opaque", vec![0u8, 1, 2, 3]),
        ];
        let block = encode_block(&xs, 4096).unwrap();
        let back = decode_block(&block).unwrap();
        // Decoded order is sorted (index ASC, suffix ASC).
        let mut want: Vec<Xattr> = xs.clone();
        want.sort_by(|a, b| {
            let (ai, asuf) = name_index_and_suffix(&a.name);
            let (bi, bsuf) = name_index_and_suffix(&b.name);
            ai.cmp(&bi).then(asuf.cmp(bsuf))
        });
        assert_eq!(back, want);
    }

    #[test]
    fn rejects_too_big_for_block() {
        // 100 KiB of value can't fit in a 4 KiB block.
        let xs = vec![Xattr::new("user.big", vec![0u8; 100 * 1024])];
        assert!(encode_block(&xs, 4096).is_err());
    }

    #[test]
    fn name_index_maps_known_prefixes() {
        assert_eq!(name_index_and_suffix("user.foo"), (1, "foo"));
        assert_eq!(name_index_and_suffix("trusted.bar"), (4, "bar"));
        assert_eq!(name_index_and_suffix("security.x"), (6, "x"));
        assert_eq!(name_index_and_suffix("system.posix_acl_access"), (2, ""));
        assert_eq!(name_index_and_suffix("weird"), (0, "weird"));
    }
}
