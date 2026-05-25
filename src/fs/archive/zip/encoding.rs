//! ZIP filename encoding, following the technique in the project's
//! "universal zip" note.
//!
//! - **Write:** set general-purpose bit 11 (UTF-8 / EFS) *only* when a
//!   name carries genuine multibyte UTF-8; pure-ASCII names are left
//!   unflagged so every reader since 1990 agrees on them.
//! - **Read:** trust the bytes as UTF-8 when bit 11 is set; otherwise
//!   auto-detect with an ordered candidate list (ASCII → UTF-8 →
//!   Shift-JIS → EUC-JP → ISO-8859-15) so a name written by Japanese
//!   Windows (Shift-JIS, no flag) still decodes correctly.

/// Whether `name` must carry the UTF-8 language-encoding flag (bit 11).
/// True only for names that aren't pure ASCII (Rust strings are already
/// valid UTF-8, so a non-ASCII name is genuine multibyte UTF-8).
pub fn needs_utf8_flag(name: &str) -> bool {
    !name.is_ascii()
}

/// Decode a stored filename to UTF-8. `utf8_flag` is general-purpose
/// bit 11 from the entry.
pub fn decode_name(bytes: &[u8], utf8_flag: bool) -> String {
    if utf8_flag {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    // No marker: infer. ASCII and strict UTF-8 are unambiguous.
    if bytes.is_ascii() {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    if let Ok(s) = std::str::from_utf8(bytes) {
        return s.to_string();
    }
    // Japanese legacy encodings, accepted only on a clean decode.
    for enc in [encoding_rs::SHIFT_JIS, encoding_rs::EUC_JP] {
        let (cow, _, had_errors) = enc.decode(bytes);
        if !had_errors {
            return cow.into_owned();
        }
    }
    // ISO-8859-15 maps every byte, so this always succeeds.
    let (cow, _, _) = encoding_rs::ISO_8859_15.decode(bytes);
    cow.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_is_unflagged() {
        assert!(!needs_utf8_flag("hello.txt"));
        assert!(needs_utf8_flag("日本語.txt"));
    }

    #[test]
    fn utf8_flag_trusts_bytes() {
        let n = "café.txt";
        assert_eq!(decode_name(n.as_bytes(), true), n);
    }

    #[test]
    fn shift_jis_without_flag_round_trips() {
        // "ソ" (U+30BD) in Shift-JIS is 0x83 0x5C — invalid UTF-8, so the
        // detector must fall through to Shift-JIS.
        let (sjis, _, err) = encoding_rs::SHIFT_JIS.encode("ソ.txt");
        assert!(!err);
        assert_eq!(decode_name(&sjis, false), "ソ.txt");
    }

    #[test]
    fn ascii_without_flag_is_verbatim() {
        assert_eq!(decode_name(b"plain.txt", false), "plain.txt");
    }
}
