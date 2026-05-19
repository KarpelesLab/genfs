//! XFS extended-attribute fork (shortform / "local" form only).
//!
//! When an inode has at least one xattr and the encoded form fits in the
//! attribute-fork half of the inode literal area, XFS records the
//! attribute fork as `di_aformat = LOCAL` (1) and stores everything
//! inline. The on-disk layout (from the XFS PDF, "Shortform Attributes"
//! section) is:
//!
//! ```text
//!   xfs_attr_sf_hdr:
//!     __be16 totsize     // total bytes including header
//!     __u8   count       // number of entries
//!     __u8   padding     // alignment
//!
//!   xfs_attr_sf_entry × count:
//!     __u8   namelen
//!     __u8   valuelen
//!     __u8   flags       // 0 = user, ATTR_ROOT(0x02) = trusted, ATTR_SECURE(0x01) = security
//!     __u8   nameval[namelen + valuelen]
//! ```
//!
//! Names and values are stored exactly as-is — no NULL terminator. The
//! flag byte encodes the on-disk namespace. We map the Linux-userland
//! prefixes (`user.X`, `trusted.X`, `security.X`) to/from these flags.
//!
//! Spill to leaf / node attribute blocks (when the xattrs don't fit
//! inline) is intentionally out of scope for v1; `add_xattr` returns
//! `Error::Unsupported` in that case.
//!
//! Reference: <https://mirrors.edge.kernel.org/pub/linux/utils/fs/xfs/docs/xfs_filesystem_structure.pdf>
//! section "Shortform Attributes".
//!
//! This module is read+write: the encoder is used by `Xfs::add_xattr`,
//! the decoder by `Xfs::read_xattrs`.

use std::collections::HashMap;

use crate::Result;

/// Flag bit on a shortform entry — XFS_ATTR_ROOT (trusted namespace).
pub const XFS_ATTR_ROOT: u8 = 0x02;
/// Flag bit on a shortform entry — XFS_ATTR_SECURE (security namespace).
pub const XFS_ATTR_SECURE: u8 = 0x01;

/// Decoded shortform attribute entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortformAttr {
    /// Fully-qualified xattr name as seen by userland, e.g.
    /// `"user.mime_type"` or `"trusted.glusterfs.id"`.
    pub name: String,
    pub value: Vec<u8>,
}

/// Translate a userland xattr name (`"user.foo"`, `"trusted.bar"`,
/// `"security.selinux"`) into the on-disk shortform suffix + flags byte.
/// Names without one of the recognised prefixes default to the user
/// namespace.
pub fn name_to_disk(name: &str) -> (String, u8) {
    if let Some(rest) = name.strip_prefix("user.") {
        (rest.to_string(), 0)
    } else if let Some(rest) = name.strip_prefix("trusted.") {
        (rest.to_string(), XFS_ATTR_ROOT)
    } else if let Some(rest) = name.strip_prefix("security.") {
        (rest.to_string(), XFS_ATTR_SECURE)
    } else {
        // No recognised prefix — treat the whole string as user.* on
        // disk by stripping nothing. Keep the original full name in the
        // suffix so a later read returns the same bytes verbatim.
        (name.to_string(), 0)
    }
}

/// Inverse of [`name_to_disk`] — turn a (suffix, flags) pair from disk
/// back into a userland xattr name. Suffixes whose flag byte names no
/// known namespace are surfaced as `user.<suffix>`.
pub fn name_from_disk(suffix: &str, flags: u8) -> String {
    if flags & XFS_ATTR_ROOT != 0 {
        format!("trusted.{suffix}")
    } else if flags & XFS_ATTR_SECURE != 0 {
        format!("security.{suffix}")
    } else {
        format!("user.{suffix}")
    }
}

/// Compute the on-disk byte size of a shortform xattr area containing
/// the given (name, value) pairs. Includes the 4-byte header.
pub fn shortform_size(attrs: &[(String, Vec<u8>)]) -> usize {
    let mut total = 4usize; // hdr: totsize(2) + count(1) + pad(1)
    for (name, value) in attrs {
        let (suffix, _flags) = name_to_disk(name);
        total += 3 + suffix.len() + value.len();
    }
    total
}

/// Encode a shortform xattr area for the given attributes. Returns the
/// raw bytes ready to be placed at `inode_literal[forkoff*8..]`. Fails
/// if any single name or value exceeds 255 bytes (the shortform format
/// uses 8-bit length fields).
pub fn encode_shortform(attrs: &[(String, Vec<u8>)]) -> Result<Vec<u8>> {
    let total = shortform_size(attrs);
    if total > u16::MAX as usize {
        return Err(crate::Error::InvalidArgument(format!(
            "xfs: shortform xattr area {total} > 65535"
        )));
    }
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&(total as u16).to_be_bytes());
    buf.push(attrs.len() as u8);
    buf.push(0u8); // padding
    for (name, value) in attrs {
        let (suffix, flags) = name_to_disk(name);
        if suffix.len() > 255 {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: xattr name {name:?} suffix > 255 bytes"
            )));
        }
        if value.len() > 255 {
            return Err(crate::Error::InvalidArgument(format!(
                "xfs: xattr value for {name:?} > 255 bytes (shortform limit)"
            )));
        }
        buf.push(suffix.len() as u8);
        buf.push(value.len() as u8);
        buf.push(flags);
        buf.extend_from_slice(suffix.as_bytes());
        buf.extend_from_slice(value);
    }
    Ok(buf)
}

/// Decode a shortform xattr area into a HashMap keyed by the userland
/// xattr name. Tolerates a buffer that is longer than `totsize` (we
/// only walk `totsize` bytes); rejects buffers shorter than `totsize`
/// or with truncated entry data.
pub fn decode_shortform(buf: &[u8]) -> Result<HashMap<String, Vec<u8>>> {
    if buf.len() < 4 {
        return Err(crate::Error::InvalidImage(
            "xfs: shortform xattr header truncated".into(),
        ));
    }
    let totsize = u16::from_be_bytes(buf[0..2].try_into().unwrap()) as usize;
    let count = buf[2] as usize;
    // padding at buf[3] — ignore
    if totsize > buf.len() {
        return Err(crate::Error::InvalidImage(format!(
            "xfs: shortform xattr totsize {totsize} > buffer {}",
            buf.len()
        )));
    }
    let mut out = HashMap::with_capacity(count);
    let mut pos = 4usize;
    for _ in 0..count {
        if pos + 3 > totsize {
            return Err(crate::Error::InvalidImage(
                "xfs: shortform xattr entry header truncated".into(),
            ));
        }
        let namelen = buf[pos] as usize;
        let valuelen = buf[pos + 1] as usize;
        let flags = buf[pos + 2];
        let name_start = pos + 3;
        let name_end = name_start + namelen;
        let val_end = name_end + valuelen;
        if val_end > totsize {
            return Err(crate::Error::InvalidImage(format!(
                "xfs: shortform xattr entry overshoots totsize {totsize} at {val_end}"
            )));
        }
        let suffix = std::str::from_utf8(&buf[name_start..name_end]).map_err(|_| {
            crate::Error::InvalidImage("xfs: non-UTF-8 shortform xattr name".into())
        })?;
        let full_name = name_from_disk(suffix, flags);
        let value = buf[name_end..val_end].to_vec();
        out.insert(full_name, value);
        pos = val_end;
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
            ("trusted.glusterfs".to_string(), b"abc".to_vec()),
        ];
        let encoded = encode_shortform(&attrs).unwrap();
        // Header totsize matches our buffer length.
        let totsize = u16::from_be_bytes(encoded[0..2].try_into().unwrap()) as usize;
        assert_eq!(totsize, encoded.len());
        let decoded = decode_shortform(&encoded).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.get("user.mime_type"), Some(&b"text/plain".to_vec()));
        assert_eq!(decoded.get("trusted.glusterfs"), Some(&b"abc".to_vec()));
    }

    #[test]
    fn empty_attr_value() {
        let attrs = vec![("user.flag".to_string(), Vec::new())];
        let encoded = encode_shortform(&attrs).unwrap();
        let decoded = decode_shortform(&encoded).unwrap();
        assert_eq!(decoded.get("user.flag"), Some(&Vec::new()));
    }

    #[test]
    fn reject_oversize_value() {
        let attrs = vec![("user.big".to_string(), vec![0u8; 256])];
        assert!(encode_shortform(&attrs).is_err());
    }

    #[test]
    fn name_namespace_round_trip() {
        for n in ["user.foo", "trusted.bar", "security.selinux"] {
            let (suffix, flags) = name_to_disk(n);
            assert_eq!(name_from_disk(&suffix, flags), n);
        }
    }
}
