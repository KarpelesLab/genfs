//! MacRoman → Unicode decoding for classic HFS filenames.
//!
//! Classic HFS stores names as MacRoman (8-bit) Pascal strings. Bytes
//! `0x00–0x7F` are ASCII; `0x80–0xFF` map to the Unicode code points below
//! (the standard Apple MacRoman table).

/// Unicode code points for MacRoman bytes `0x80..=0xFF`.
#[rustfmt::skip]
const HIGH: [char; 128] = [
    'Ä','Å','Ç','É','Ñ','Ö','Ü','á','à','â','ä','ã','å','ç','é','è',
    'ê','ë','í','ì','î','ï','ñ','ó','ò','ô','ö','õ','ú','ù','û','ü',
    '†','°','¢','£','§','•','¶','ß','®','©','™','´','¨','≠','Æ','Ø',
    '∞','±','≤','≥','¥','µ','∂','∑','∏','π','∫','ª','º','Ω','æ','ø',
    '¿','¡','¬','√','ƒ','≈','∆','«','»','…','\u{00A0}','À','Ã','Õ','Œ','œ',
    '–','—','“','”','‘','’','÷','◊','ÿ','Ÿ','⁄','€','‹','›','ﬁ','ﬂ',
    '‡','·','‚','„','‰','Â','Ê','Á','Ë','È','Í','Î','Ï','Ì','Ó','Ô',
    '\u{F8FF}','Ò','Ú','Û','Ù','ı','ˆ','˜','¯','˘','˙','˚','¸','˝','˛','ˇ',
];

/// Decode a MacRoman byte string to a UTF-8 `String`.
pub fn decode(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&b| {
            if b < 0x80 {
                b as char
            } else {
                HIGH[(b - 0x80) as usize]
            }
        })
        .collect()
}

/// Case-insensitive equality used for path-component matching. Folds ASCII
/// letters (sufficient for the System-disk style names this reader targets);
/// other characters compare exactly after MacRoman decoding.
pub fn eq_ignore_case(a: &str, b: &str) -> bool {
    let fold = |c: char| {
        if c.is_ascii_uppercase() {
            c.to_ascii_lowercase()
        } else {
            c
        }
    };
    a.chars().map(fold).eq(b.chars().map(fold))
}
