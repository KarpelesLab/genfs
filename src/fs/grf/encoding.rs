//! CP949 ↔ UTF-8 helpers for GRF filenames.
//!
//! Ragnarok Online's `.grf` archives store filenames in CP949 — the
//! Microsoft Korean codepage, a superset of EUC-KR with extra
//! private-use mappings. Filenames seen on disk are CP949 bytes; the
//! fstool `Filesystem` trait surface is UTF-8 strings. Conversion
//! happens once at the GRF boundary (during table parse / write).
//!
//! We use `encoding_rs::EUC_KR`. The WHATWG `euc-kr` label is
//! intentionally defined to be CP949 for web-compatibility — the spec
//! mapping is the union we want.

use std::borrow::Cow;

use encoding_rs::EUC_KR;

/// Decode a CP949 byte slice into UTF-8. Stray bytes that don't map
/// (corruption, truncation) are replaced with U+FFFD — we don't
/// surface decoding errors because GRFs in the wild ship occasional
/// junk filenames and refusing the whole archive helps nobody.
pub fn cp949_to_utf8(bytes: &[u8]) -> Cow<'_, str> {
    let (decoded, _, _had_errors) = EUC_KR.decode(bytes);
    decoded
}

/// Encode a UTF-8 string back to CP949 bytes. Characters outside
/// CP949 are emitted as numeric character references (`&#NNNN;`)
/// because that's what `encoding_rs` does for unmappable codepoints
/// — practical for round-trip with our own reader since we'll just
/// decode them back to the references. Filenames are normally ASCII
/// or Hangul, so this path is rare.
pub fn utf8_to_cp949(s: &str) -> Vec<u8> {
    let (encoded, _, _had_errors) = EUC_KR.encode(s);
    encoded.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_round_trip() {
        let plain = b"data/info.txt";
        let utf = cp949_to_utf8(plain);
        assert_eq!(utf, "data/info.txt");
        let back = utf8_to_cp949(&utf);
        assert_eq!(back, plain);
    }

    #[test]
    fn hangul_round_trip() {
        let utf = "한국어.txt";
        let bytes = utf8_to_cp949(utf);
        // CP949 encodes each Hangul syllable as 2 bytes; UTF-8 uses 3
        // bytes per Hangul codepoint. Three syllables + ".txt" =
        // 6 + 4 = 10 bytes in CP949 vs 13 bytes in UTF-8.
        assert_eq!(bytes.len(), 10);
        let back = cp949_to_utf8(&bytes);
        assert_eq!(back, utf);
    }
}
