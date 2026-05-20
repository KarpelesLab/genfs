//! Joliet UCS-2 BE name codec.
//!
//! Joliet (Microsoft TR-IMSA-1999) stores volume identifier and file
//! names as UCS-2 big-endian. Names also drop the `;version` suffix in
//! Joliet identifiers, so we strip it here for symmetry with the 8.3
//! handling on the PVD side.

/// Decode a UCS-2 BE byte slice into a `String`. Trailing 0x00 / 0x20
/// (space) words are stripped — that's the Joliet identifier padding
/// convention. The `;1`-style version suffix some tools emit even on
/// Joliet identifiers is also stripped.
pub fn ucs2_be_to_string(buf: &[u8]) -> String {
    let mut units: Vec<u16> = buf
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect();
    // Strip trailing padding (0x0000 or 0x0020 — Joliet uses spaces for
    // volume identifier padding, NULs for name padding).
    while let Some(&last) = units.last() {
        if last == 0 || last == 0x20 {
            units.pop();
        } else {
            break;
        }
    }
    let raw = String::from_utf16_lossy(&units);
    // `;NN`-style version suffix.
    match raw.rsplit_once(';') {
        Some((stem, _ver)) => stem.to_string(),
        None => raw,
    }
}

/// Encode a Rust string as UCS-2 BE bytes (writer-side).
#[allow(dead_code)]
pub fn string_to_ucs2_be(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() * 2);
    for u in s.encode_utf16() {
        out.extend_from_slice(&u.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ucs2_decode_strips_padding_and_version() {
        let mut bytes = string_to_ucs2_be("README");
        bytes.extend_from_slice(&[0x00, b';']);
        bytes.extend_from_slice(&[0x00, b'1']);
        // Pad to 32 bytes with NULs.
        while bytes.len() < 32 {
            bytes.push(0);
        }
        assert_eq!(ucs2_be_to_string(&bytes), "README");
    }

    #[test]
    fn round_trip_ascii() {
        let s = "Hello";
        assert_eq!(ucs2_be_to_string(&string_to_ucs2_be(s)), s);
    }

    #[test]
    fn round_trip_non_ascii() {
        let s = "café";
        let enc = string_to_ucs2_be(s);
        assert_eq!(ucs2_be_to_string(&enc), s);
    }
}
