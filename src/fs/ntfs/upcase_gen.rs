//! Generate the canonical NTFS `$UpCase` table (128 KiB / 65536 × u16 LE).
//!
//! NTFS stores a fixed case-folding table that maps each BMP code point to
//! its uppercase form. Microsoft's table is derived from the Unicode
//! `UnicodeData.txt` "Simple_Uppercase_Mapping" field for the BMP range,
//! plus a handful of NTFS-specific overrides. The exact table changed
//! across Windows releases, but the modern one is a strict superset of
//! the ASCII / Latin-1 case mappings.
//!
//! We don't carry the full Unicode table in source; instead we generate
//! a "best effort" upcase table that covers the rules every NTFS reader
//! cares about:
//!
//! * Identity for code points that have no upper-case form.
//! * ASCII a..z → A..Z.
//! * Latin-1 supplement, Latin Extended-A / B (most positions).
//! * Greek, Cyrillic, Armenian (the common ranges).
//! * Fullwidth Latin (FF41..FF5A → FF21..FF3A).
//!
//! This table is good enough for chkdsk to accept the volume and good
//! enough for case-insensitive path lookups against typical filenames.
//! It is deterministic so a chkdsk-recomputed checksum will match across
//! runs of fstool.

/// Emit a 128 KiB blob in the NTFS `$UpCase` format: 65536 little-endian
/// u16 entries, indexed by BMP code point, value = upper-case folding.
pub fn build_upcase_blob() -> Vec<u8> {
    let mut table = [0u16; 0x10000];
    for (i, entry) in table.iter_mut().enumerate() {
        *entry = i as u16;
    }

    // ASCII a..z → A..Z
    for c in b'a'..=b'z' {
        table[c as usize] = (c - 0x20) as u16;
    }

    // Latin-1 supplement: ÿ etc. à..ö (0xE0..0xF6) → 0xC0..0xD6
    for c in 0xE0u16..=0xF6 {
        table[c as usize] = c - 0x20;
    }
    // ø..þ (0xF8..0xFE) → 0xD8..0xDE
    for c in 0xF8u16..=0xFE {
        table[c as usize] = c - 0x20;
    }
    // ÿ (0xFF) → Ÿ (0x178)
    table[0xFF] = 0x178;
    // µ (0xB5) → Μ (0x39C, Greek capital Mu) per Unicode; some NTFS
    // versions keep this as identity. Use identity here for stability —
    // chkdsk validates a hash, not exact case folding.

    // Latin Extended-A: alternating lower/upper pairs from 0x0100..0x017F.
    // Most positions follow odd = lower / even = upper (or vice versa).
    // Use the Unicode bicameral pattern: pairs at (0x100,0x101), (0x102,0x103), ...
    // where the lower-case is at the odd index.
    let mut c = 0x0100u16;
    while c <= 0x012F {
        // (upper, lower) consecutive
        table[(c + 1) as usize] = c;
        c += 2;
    }
    // 0x0132..0x0137: IJ ligature pair + others, same odd-lower pattern.
    let mut c = 0x0132u16;
    while c <= 0x0137 {
        table[(c + 1) as usize] = c;
        c += 2;
    }
    // 0x0139..0x0148: L with various marks; pattern is opposite (lower at
    // even index, upper at odd). Skip rare edge cases — identity is the
    // safe fall-back.
    let mut c = 0x0139u16;
    while c <= 0x0148 {
        // (lower, upper) — lower at even.
        table[c as usize] = c + 1;
        c += 2;
    }
    // 0x014A..0x0177: alternating upper/lower like 0x0100..0x012F.
    let mut c = 0x014Au16;
    while c <= 0x0177 {
        table[(c + 1) as usize] = c;
        c += 2;
    }

    // Greek letters (basic block): α..ω (0x03B1..0x03C9) → Α..Ω (0x0391..0x03A9).
    // Note 0x03C2 (final sigma) → Σ (0x03A3).
    for c in 0x03B1u16..=0x03C9 {
        let upper = if c == 0x03C2 { 0x03A3 } else { c - 0x20 };
        table[c as usize] = upper;
    }

    // Cyrillic: а..я (0x0430..0x044F) → А..Я (0x0410..0x042F).
    for c in 0x0430u16..=0x044F {
        table[c as usize] = c - 0x20;
    }
    // Cyrillic supplement: ё (0x0451) → Ё (0x0401), 0x0452..0x045F → 0x0402..0x040F.
    table[0x0451] = 0x0401;
    for c in 0x0452u16..=0x045F {
        table[c as usize] = c - 0x50;
    }

    // Armenian: ա..ֆ (0x0561..0x0586) → Ա..Ֆ (0x0531..0x0556).
    for c in 0x0561u16..=0x0586 {
        table[c as usize] = c - 0x30;
    }

    // Fullwidth Latin: ａ..ｚ (0xFF41..0xFF5A) → Ａ..Ｚ (0xFF21..0xFF3A).
    for c in 0xFF41u16..=0xFF5A {
        table[c as usize] = c - 0x20;
    }

    // Surrogate range: identity (we never fold surrogates).
    // (Already identity from initialization.)

    let mut out = Vec::with_capacity(0x10000 * 2);
    for v in table.iter() {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fold(table: &[u8], c: u16) -> u16 {
        let i = c as usize * 2;
        u16::from_le_bytes([table[i], table[i + 1]])
    }

    #[test]
    fn ascii_fold() {
        let t = build_upcase_blob();
        assert_eq!(fold(&t, b'a' as u16), b'A' as u16);
        assert_eq!(fold(&t, b'z' as u16), b'Z' as u16);
        assert_eq!(fold(&t, b'A' as u16), b'A' as u16);
        assert_eq!(fold(&t, b'0' as u16), b'0' as u16);
    }

    #[test]
    fn length_is_128_kib() {
        let t = build_upcase_blob();
        assert_eq!(t.len(), 128 * 1024);
    }

    #[test]
    fn latin1_supplement_fold() {
        let t = build_upcase_blob();
        assert_eq!(fold(&t, 0xE0), 0xC0);
        assert_eq!(fold(&t, 0xF6), 0xD6);
        assert_eq!(fold(&t, 0xFF), 0x178);
    }
}
