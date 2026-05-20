//! Minimal base64 decoder used to recover the mish-block payloads
//! embedded under each `<key>blkx</key>` `<data>` element in the
//! UDIF resource-fork plist.
//!
//! Scope is intentionally narrow — DMG plists are produced by
//! Apple's `hdiutil` and always use the standard alphabet (RFC 4648,
//! `'+'` / `'/'` / `'='` padding). We accept whitespace anywhere
//! (CR / LF / TAB / SP) because the plist pretty-prints the base64
//! payload across many lines, and we reject any other byte.
//!
//! Pulling in the `base64` crate just for this hot-path would be
//! gratuitous: the decoder is ~40 lines of straight-line code and
//! has no cross-cutting requirements with the rest of the crate.
//! If we later need base64 elsewhere (qcow2 backing-file URIs, etc.)
//! this will graduate to a shared helper.
//!
//! Cross-check: the `decodes_known_vectors` test below pins us to
//! the RFC 4648 reference vectors so any drift in alphabet or
//! padding handling is caught locally — no need to round-trip
//! through a real DMG fixture for unit-level confidence.

use crate::Result;

/// Standard base64 alphabet. Index = 6-bit value, value = ASCII byte.
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Decode a base64 string. Whitespace (space, tab, CR, LF) inside the
/// input is silently skipped — Apple's plist serialiser splits long
/// base64 payloads across many indented lines. Anything else that
/// isn't an alphabet byte or `=` padding is a hard error.
pub fn decode(input: &str) -> Result<Vec<u8>> {
    // Build a reverse lookup table once per call. 256 bytes; cheap.
    // 0xFF marks "not in alphabet" — `=` is also 0xFF, but we treat it
    // separately below.
    let mut lookup = [0xFFu8; 256];
    for (i, &b) in ALPHABET.iter().enumerate() {
        lookup[b as usize] = i as u8;
    }

    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    // 24-bit accumulator + count of 6-bit groups buffered (0..4).
    let mut acc: u32 = 0;
    let mut groups: u32 = 0;
    let mut pad: u32 = 0;

    for &c in input.as_bytes() {
        match c {
            b' ' | b'\t' | b'\r' | b'\n' => continue,
            b'=' => {
                pad += 1;
                if pad > 2 {
                    return Err(crate::Error::InvalidImage(
                        "dmg: base64 has more than two '=' padding bytes".into(),
                    ));
                }
                // Padding still contributes a (zero) sextet to the
                // accumulator so the group bookkeeping stays in sync;
                // we trim the unused output bytes below.
                acc <<= 6;
                groups += 1;
            }
            _ => {
                if pad > 0 {
                    return Err(crate::Error::InvalidImage(
                        "dmg: base64 has non-padding bytes after '='".into(),
                    ));
                }
                let v = lookup[c as usize];
                if v == 0xFF {
                    return Err(crate::Error::InvalidImage(format!(
                        "dmg: base64 contains invalid byte {c:#x}"
                    )));
                }
                acc = (acc << 6) | (v as u32);
                groups += 1;
            }
        }
        if groups == 4 {
            out.push(((acc >> 16) & 0xFF) as u8);
            out.push(((acc >> 8) & 0xFF) as u8);
            out.push((acc & 0xFF) as u8);
            acc = 0;
            groups = 0;
        }
    }

    if groups != 0 {
        return Err(crate::Error::InvalidImage(
            "dmg: base64 input length not a multiple of 4 after stripping whitespace".into(),
        ));
    }

    // Trim padding bytes from the tail. `pad` is 1 or 2 means we
    // accidentally emitted 1 or 2 zero bytes — drop them.
    for _ in 0..pad {
        out.pop();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_known_vectors() {
        // RFC 4648 §10 reference vectors.
        assert_eq!(decode("").unwrap(), b"");
        assert_eq!(decode("Zg==").unwrap(), b"f");
        assert_eq!(decode("Zm8=").unwrap(), b"fo");
        assert_eq!(decode("Zm9v").unwrap(), b"foo");
        assert_eq!(decode("Zm9vYg==").unwrap(), b"foob");
        assert_eq!(decode("Zm9vYmE=").unwrap(), b"fooba");
        assert_eq!(decode("Zm9vYmFy").unwrap(), b"foobar");
    }

    #[test]
    fn tolerates_whitespace_and_line_breaks() {
        let s = "Zm9v\n\tYmFy\r\n";
        assert_eq!(decode(s).unwrap(), b"foobar");
    }

    #[test]
    fn rejects_invalid_bytes() {
        assert!(decode("Zm9v!").is_err());
        assert!(decode("Zm===").is_err()); // 3 padding bytes
        assert!(decode("Zm9vA").is_err()); // wrong length
    }
}
