//! PAX extended-header records.
//!
//! A PAX header is a regular tar entry with `typeflag = 'x'` whose body
//! holds a sequence of length-prefixed `keyword=value\n` records:
//!
//! ```text
//!   "%d %s=%s\n"
//! ```
//!
//! where `%d` is the byte length of the *whole* record (including the
//! length digits, the space, and the trailing newline).
//!
//! Records relevant to fstool:
//!
//! - `path`             — supersedes ustar's `name`+`prefix` for paths > 100 bytes
//! - `linkpath`         — supersedes `linkname` for symlink targets > 100 bytes
//! - `size`             — supersedes ustar's 12-octal-digit size (≤ 8 GiB)
//! - `mtime`            — high-precision modification time (we use seconds-only)
//! - `SCHILY.xattr.<n>` — xattr `n` with arbitrary binary value bytes
//!
//! The xattr key prefix `SCHILY.xattr.` is the de-facto standard
//! introduced by Schily's `star`. GNU tar, libarchive, and bsdtar all
//! emit and consume it.

use crate::Result;
use crate::fs::ext::xattr::Xattr;

pub const KEY_PATH: &str = "path";
pub const KEY_LINKPATH: &str = "linkpath";
pub const KEY_SIZE: &str = "size";
pub const KEY_MTIME: &str = "mtime";
pub const XATTR_PREFIX: &str = "SCHILY.xattr.";

/// One PAX record. `value` is raw bytes — xattr values may be binary.
#[derive(Debug, Clone)]
pub struct Record {
    pub key: String,
    pub value: Vec<u8>,
}

/// Encode a list of PAX records into the body bytes of an `x`-typeflag
/// entry. Each record contributes one length-prefixed line.
pub fn encode_records(records: &[Record]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in records {
        // Each record's length includes its own digits, so we have to
        // solve `len_digits + 1 (space) + key.len + 1 (=) + value.len
        // + 1 (newline) == len`. The number of digits depends on len,
        // so iterate: pick a digit count, compute resulting len, and
        // verify it matches.
        let key_eq_val_nl = r.key.len() + 1 /* '=' */ + r.value.len() + 1 /* \n */;
        // Lower bound: try 1 digit, then 2, ... until the size stabilises.
        let mut digits = 1usize;
        let total = loop {
            let candidate = digits + 1 /* space */ + key_eq_val_nl;
            let needed_digits = decimal_width(candidate);
            if needed_digits <= digits {
                break candidate;
            }
            digits = needed_digits;
        };
        out.extend_from_slice(format!("{total}").as_bytes());
        out.push(b' ');
        out.extend_from_slice(r.key.as_bytes());
        out.push(b'=');
        out.extend_from_slice(&r.value);
        out.push(b'\n');
    }
    out
}

/// Decode the body of an `x` PAX header into individual records.
pub fn decode_records(body: &[u8]) -> Result<Vec<Record>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < body.len() {
        // Parse length prefix (digits up to the first space).
        let space = body[pos..].iter().position(|&b| b == b' ').ok_or_else(|| {
            crate::Error::InvalidImage("tar: PAX record missing length space".into())
        })?;
        let len_str = std::str::from_utf8(&body[pos..pos + space])
            .map_err(|_| crate::Error::InvalidImage("tar: PAX length not ASCII".into()))?;
        let len: usize = len_str
            .parse()
            .map_err(|_| crate::Error::InvalidImage(format!("tar: bad PAX length {len_str:?}")))?;
        if pos + len > body.len() {
            return Err(crate::Error::InvalidImage(
                "tar: PAX record length runs past end".into(),
            ));
        }
        if body[pos + len - 1] != b'\n' {
            return Err(crate::Error::InvalidImage(
                "tar: PAX record doesn't end with newline".into(),
            ));
        }
        // After the space, the rest of the record (minus the newline) is
        // `key=value`.
        let kv = &body[pos + space + 1..pos + len - 1];
        let eq = kv
            .iter()
            .position(|&b| b == b'=')
            .ok_or_else(|| crate::Error::InvalidImage("tar: PAX record missing '='".into()))?;
        let key = std::str::from_utf8(&kv[..eq])
            .map_err(|_| crate::Error::InvalidImage("tar: PAX key not UTF-8".into()))?
            .to_string();
        let value = kv[eq + 1..].to_vec();
        out.push(Record { key, value });
        pos += len;
    }
    Ok(out)
}

/// Build the PAX records that need to ride alongside a tar entry given
/// its `(path, linkpath, size, mtime, xattrs)`. Returns an empty vec
/// when every field fits in the plain ustar header.
pub fn records_for_entry(
    path: &str,
    linkpath: Option<&str>,
    size_octal_overflows: bool,
    xattrs: &[Xattr],
) -> Vec<Record> {
    let mut out = Vec::new();
    if path.len() > super::header::NAME_LEN || !path_fits_ustar(path) || !path.is_ascii() {
        out.push(Record {
            key: KEY_PATH.into(),
            value: path.as_bytes().to_vec(),
        });
    }
    if let Some(t) = linkpath
        && (t.len() > super::header::LINKNAME_LEN || !t.is_ascii())
    {
        out.push(Record {
            key: KEY_LINKPATH.into(),
            value: t.as_bytes().to_vec(),
        });
    }
    if size_octal_overflows {
        // Caller already knows the size; let it pass through unchanged.
        // The actual size record is appended by the caller because it
        // owns the value.
    }
    for x in xattrs {
        out.push(Record {
            key: format!("{XATTR_PREFIX}{}", x.name),
            value: x.value.clone(),
        });
    }
    out
}

/// True when `path` fits in `name` (≤ 100 bytes) or splits cleanly into
/// `prefix` (≤ 155) + `name` (≤ 100) at a `/` boundary. False → must
/// emit a PAX `path` record.
pub fn path_fits_ustar(path: &str) -> bool {
    if path.len() <= super::header::NAME_LEN {
        return true;
    }
    if path.len() > super::header::PREFIX_LEN + 1 + super::header::NAME_LEN {
        return false;
    }
    // Find a `/` such that prefix.len() ≤ 155 and (suffix) ≤ 100.
    for (i, b) in path.bytes().enumerate() {
        if b == b'/'
            && i <= super::header::PREFIX_LEN
            && path.len() - i - 1 <= super::header::NAME_LEN
        {
            return true;
        }
    }
    false
}

fn decimal_width(n: usize) -> usize {
    if n < 10 {
        1
    } else if n < 100 {
        2
    } else if n < 1000 {
        3
    } else if n < 10_000 {
        4
    } else if n < 100_000 {
        5
    } else {
        n.to_string().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_simple_records() {
        let recs = vec![
            Record {
                key: "path".into(),
                value: b"some/very/long/path/that/does/not/fit/in/100/bytes/of/ustar/name/field/even/with/a/prefix/split/at/all".to_vec(),
            },
            Record {
                key: "SCHILY.xattr.user.foo".into(),
                value: b"hello".to_vec(),
            },
        ];
        let body = encode_records(&recs);
        let decoded = decode_records(&body).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].key, "path");
        assert_eq!(decoded[1].key, "SCHILY.xattr.user.foo");
        assert_eq!(decoded[1].value, b"hello");
    }

    #[test]
    fn record_length_self_consistent() {
        let recs = vec![Record {
            key: "k".into(),
            value: b"v".to_vec(),
        }];
        let body = encode_records(&recs);
        // "5 k=v\n" — len = 5 because: "5" + " " + "k=v" + "\n" = 1+1+3+1 = 6
        // Wait: 6, so digits=1 gives total=1+1+3+1=6; recomputing width(6)=1 → stable.
        // Format: "6 k=v\n"
        assert_eq!(&body[..], b"6 k=v\n");
    }

    #[test]
    fn path_fits_ustar_logic() {
        assert!(path_fits_ustar("short"));
        assert!(path_fits_ustar(&"a".repeat(100)));
        assert!(!path_fits_ustar(&"a".repeat(101))); // > 100, no slash to split
        // Splittable: 50 chars + "/" + 50 chars → 101 total, splits at index 50
        let p = format!("{}/{}", "a".repeat(50), "b".repeat(50));
        assert!(path_fits_ustar(&p));
    }

    #[test]
    fn xattr_records_emitted() {
        let xs = [Xattr::new("user.foo", b"bar".to_vec())];
        let recs = records_for_entry("hello", None, false, &xs);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].key, "SCHILY.xattr.user.foo");
        assert_eq!(recs[0].value, b"bar");
    }
}
