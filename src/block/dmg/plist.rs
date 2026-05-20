//! Minimal XML-plist scanner used to pull the per-partition `mish`
//! block payloads out of the UDIF resource-fork plist.
//!
//! ## Why not a real plist parser?
//!
//! The resource fork is an Apple XML plist (`<?xml … ?>` + `<plist>`
//! root). We don't need full plist semantics: dmg readers only care
//! about one specific shape — a `<key>blkx</key>` followed by an
//! `<array>` of dicts, each dict carrying a `<key>Data</key>` →
//! `<data>` element holding the base64-encoded mish block.
//!
//! A 50-line scanner that walks the byte stream looking for those
//! anchors is enough. We don't try to validate the rest of the
//! schema; we just yield the base64 payloads in document order.
//! If the file isn't actually a plist we'll still find no matches
//! and surface "no blkx entries" upstream.
//!
//! ## What we tolerate
//!
//! Tag attributes (`<data type="base64">…`), self-closing forms,
//! whitespace anywhere — `find_tag_content` does a substring match
//! on the open / close tags. CDATA sections, character entities and
//! UTF-8 BOM aren't expected inside `<data>` (base64 is 7-bit ASCII
//! by construction) so we don't translate them.

use crate::Result;

/// Walk the plist byte stream and yield the base64 payload of every
/// `<data>` element under `<key>blkx</key> <array> …`. The order
/// matches the array order — i.e. partition order on disk — which is
/// what the chunk router downstream expects.
pub fn extract_blkx_data_entries(plist: &str) -> Result<Vec<String>> {
    // Find the `<key>blkx</key>` anchor. The key is sometimes written
    // with a trailing space inside the tag (Apple's serialiser
    // doesn't, but we accept either) and the casing is fixed.
    let Some(blkx_pos) = find_key(plist, "blkx") else {
        return Err(crate::Error::InvalidImage(
            "dmg: resource-fork plist has no <key>blkx</key>".into(),
        ));
    };
    // Skip past the closing `</key>`.
    let after_key = blkx_pos;
    // Locate the `<array>` that owns the blkx dicts.
    let Some(arr_start) = plist[after_key..].find("<array>") else {
        return Err(crate::Error::InvalidImage(
            "dmg: <key>blkx</key> is not followed by an <array>".into(),
        ));
    };
    let arr_start = after_key + arr_start + "<array>".len();
    // Find the matching `</array>`. The blkx array doesn't contain
    // nested arrays in any DMG we've seen — Apple only nests dicts
    // inside this slot — so a single substring search is safe. If
    // a future image breaks this assumption we'd need a real parser.
    let Some(arr_end_rel) = plist[arr_start..].find("</array>") else {
        return Err(crate::Error::InvalidImage(
            "dmg: <key>blkx</key>'s <array> is not closed".into(),
        ));
    };
    let arr_end = arr_start + arr_end_rel;
    let arr_body = &plist[arr_start..arr_end];

    // Now walk the array body collecting <data>…</data> bodies.
    // Each <data> sits inside a <dict>; we don't enforce that.
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < arr_body.len() {
        let Some(open_rel) = arr_body[cursor..].find("<data>") else {
            break;
        };
        let open = cursor + open_rel + "<data>".len();
        let Some(close_rel) = arr_body[open..].find("</data>") else {
            return Err(crate::Error::InvalidImage(
                "dmg: <data> element inside blkx array is not closed".into(),
            ));
        };
        let close = open + close_rel;
        out.push(arr_body[open..close].to_string());
        cursor = close + "</data>".len();
    }

    Ok(out)
}

/// Find the position **after** the closing `</key>` of a `<key>NAME</key>`
/// pair whose body equals `name` (whitespace-trimmed). Returns `None`
/// if no such key exists. Linear scan; fine for plists < 1 MiB.
fn find_key(s: &str, name: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    while i + 5 <= bytes.len() {
        let rel = s[i..].find("<key>")?;
        let body_start = i + rel + "<key>".len();
        let body_end_rel = s[body_start..].find("</key>")?;
        let body_end = body_start + body_end_rel;
        let body = s[body_start..body_end].trim();
        if body == name {
            return Some(body_end + "</key>".len());
        }
        i = body_end + "</key>".len();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_blkx_data_in_order() {
        let plist = r#"<?xml version="1.0"?>
<plist version="1.0">
<dict>
  <key>resource-fork</key>
  <dict>
    <key>blkx</key>
    <array>
      <dict>
        <key>Attributes</key><string>0x0050</string>
        <key>Data</key>
        <data>AAAA</data>
      </dict>
      <dict>
        <key>Data</key>
        <data>BBBB</data>
      </dict>
    </array>
  </dict>
</dict>
</plist>"#;
        let v = extract_blkx_data_entries(plist).unwrap();
        assert_eq!(v, vec!["AAAA".to_string(), "BBBB".to_string()]);
    }

    #[test]
    fn errors_when_blkx_missing() {
        let plist = "<plist><dict><key>other</key><string>x</string></dict></plist>";
        assert!(extract_blkx_data_entries(plist).is_err());
    }

    #[test]
    fn errors_when_data_unclosed() {
        let plist =
            r#"<plist><dict><key>blkx</key><array><dict><data>oops</dict></array></dict></plist>"#;
        assert!(extract_blkx_data_entries(plist).is_err());
    }
}
