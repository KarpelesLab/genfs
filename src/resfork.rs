//! Macintosh **resource-fork** parser + decoders for a few common types.
//!
//! Classic Mac files carry a *resource fork* alongside the data fork: a
//! container of typed, numbered "resources" (`ICN#` icons, `vers` version
//! strings, `DITL` dialog item lists, `STR `/`STR#` strings, `CODE`, …) — the
//! things you'd open in ResEdit. This module is filesystem-agnostic: hand it the
//! raw resource-fork bytes and it yields a typed inventory plus optional
//! human-readable summaries.
//!
//! On-disk layout (big-endian throughout), per *Inside Macintosh: More
//! Macintosh Toolbox*:
//! * 16-byte header: `dataOff`, `mapOff`, `dataLen`, `mapLen`.
//! * Resource data at `dataOff`: each resource is `len: u32` followed by `len`
//!   bytes.
//! * Resource map at `mapOff`: a 28-byte header whose last two `u16`s are the
//!   offsets (from the map start) to the *type list* and *name list*. The type
//!   list is `count-1: u16` then 8-byte entries (`OSType`, `count-1: u16`,
//!   `refListOff: u16` from the type-list start). Each 12-byte reference entry is
//!   `id: i16`, `nameOff: u16` (`0xFFFF` = unnamed, else from name-list start),
//!   `attrs: u8`, `dataOff: u24` (from the data area), `handle: u32`. Names are
//!   Pascal/MacRoman strings in the name list.

use crate::macroman;
use crate::{Error, Result};

/// Cap on parsed type entries — bounds work on a malformed map.
const MAX_TYPES: usize = 8192;
/// Cap on resources within a single type.
const MAX_RES_PER_TYPE: usize = 65536;

/// One resource within a [`ResourceType`].
#[derive(Debug, Clone)]
pub struct Resource {
    /// Resource ID (signed; system resources are often negative).
    pub id: i16,
    /// Optional MacRoman name.
    pub name: Option<String>,
    /// Resource attribute flags (`resSysHeap`, `resPurgeable`, …).
    pub attrs: u8,
    /// Absolute offset of the resource's payload in the fork buffer.
    data_pos: usize,
    /// Payload length in bytes.
    pub len: usize,
}

/// All resources sharing one four-byte `OSType`.
#[derive(Debug, Clone)]
pub struct ResourceType {
    /// The four-byte type code (e.g. `*b"vers"`, `*b"ICN#"`, `*b"STR "`).
    pub ostype: [u8; 4],
    /// Resources of this type, in map order.
    pub items: Vec<Resource>,
}

/// A parsed resource fork: owns its bytes and an index of types → resources.
#[derive(Debug, Clone)]
pub struct ResourceFork {
    bytes: Vec<u8>,
    types: Vec<ResourceType>,
}

impl ResourceFork {
    /// Parse `bytes` (the whole resource fork). Returns
    /// [`Error::InvalidImage`] on any structurally inconsistent offset/length.
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        let types = parse_map(&bytes)?;
        Ok(Self { bytes, types })
    }

    /// The parsed types, in map order.
    pub fn types(&self) -> &[ResourceType] {
        &self.types
    }

    /// Total resource count across all types.
    pub fn total(&self) -> usize {
        self.types.iter().map(|t| t.items.len()).sum()
    }

    /// Raw payload of resource (`ostype`, `id`), if present.
    pub fn resource_bytes(&self, ostype: &[u8; 4], id: i16) -> Option<&[u8]> {
        let t = self.types.iter().find(|t| &t.ostype == ostype)?;
        let r = t.items.iter().find(|r| r.id == id)?;
        self.bytes.get(r.data_pos..r.data_pos + r.len)
    }

    /// Payload of a [`Resource`] obtained from [`Self::types`].
    pub fn bytes_of(&self, r: &Resource) -> &[u8] {
        &self.bytes[r.data_pos..r.data_pos + r.len]
    }
}

fn parse_map(b: &[u8]) -> Result<Vec<ResourceType>> {
    let bad = || Error::InvalidImage("resfork: malformed resource fork".into());
    if b.len() < 16 {
        return Err(bad());
    }
    let data_off = be32(b, 0) as usize;
    let map_off = be32(b, 4) as usize;
    let map_len = be32(b, 12) as usize;
    let map_end = map_off.checked_add(map_len).ok_or_else(bad)?;
    if map_off < 16 || map_len < 28 || map_end > b.len() {
        return Err(bad());
    }
    let map = &b[map_off..map_end];

    let tlist_off = be16(map, 24) as usize;
    let namelist_off = be16(map, 26) as usize;
    if tlist_off + 2 > map.len() {
        return Err(bad());
    }
    let raw = be16(map, tlist_off);
    let num_types = if raw == 0xFFFF { 0 } else { raw as usize + 1 };
    if num_types > MAX_TYPES {
        return Err(bad());
    }

    let mut types = Vec::with_capacity(num_types);
    for i in 0..num_types {
        let te = tlist_off + 2 + i * 8;
        if te + 8 > map.len() {
            return Err(bad());
        }
        let ostype = [map[te], map[te + 1], map[te + 2], map[te + 3]];
        let rawc = be16(map, te + 4);
        let count = if rawc == 0xFFFF { 0 } else { rawc as usize + 1 };
        let ref_off = be16(map, te + 6) as usize;
        if count > MAX_RES_PER_TYPE {
            return Err(bad());
        }

        let mut items = Vec::with_capacity(count.min(1024));
        for j in 0..count {
            let re = tlist_off + ref_off + j * 12;
            if re + 12 > map.len() {
                return Err(bad());
            }
            let id = be16(map, re) as i16;
            let name_off = be16(map, re + 2);
            let attrs = map[re + 4];
            let res_data_off = ((map[re + 5] as usize) << 16)
                | ((map[re + 6] as usize) << 8)
                | map[re + 7] as usize;

            let name = (name_off != 0xFFFF)
                .then(|| {
                    let np = namelist_off + name_off as usize;
                    let nlen = *map.get(np)? as usize;
                    let s = map.get(np + 1..np + 1 + nlen)?;
                    Some(macroman::decode(s))
                })
                .flatten();

            let dp = data_off.checked_add(res_data_off).ok_or_else(bad)?;
            if dp + 4 > b.len() {
                return Err(bad());
            }
            let len = be32(b, dp) as usize;
            let data_pos = dp + 4;
            if data_pos.checked_add(len).is_none_or(|end| end > b.len()) {
                return Err(bad());
            }
            items.push(Resource {
                id,
                name,
                attrs,
                data_pos,
                len,
            });
        }
        types.push(ResourceType { ostype, items });
    }
    Ok(types)
}

/// Render an `OSType` for display: printable bytes as-is (MacRoman), others as
/// `.`. Trailing spaces are kept (e.g. `STR `).
pub fn ostype_str(t: &[u8; 4]) -> String {
    t.iter()
        .map(|&c| {
            if (0x20..0x7f).contains(&c) {
                c as char
            } else {
                '.'
            }
        })
        .collect()
}

/// A short, human-readable summary of a resource's contents for a curated set
/// of well-known types, or `None` for types we don't decode.
pub fn decode_summary(ostype: &[u8; 4], data: &[u8]) -> Option<String> {
    match ostype {
        b"vers" => decode_vers(data),
        b"STR " => decode_str(data),
        b"STR#" => decode_strlist(data),
        b"TEXT" => Some(decode_text(data)),
        b"ICN#" => Some("32×32 1-bit icon (+ mask)".into()),
        b"ICON" => Some("32×32 1-bit icon".into()),
        b"DITL" => decode_ditl(data),
        _ => None,
    }
}

/// Collapse control bytes to spaces so a one-line summary stays on one line.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if (c as u32) < 0x20 { ' ' } else { c })
        .collect()
}

fn pascal_at(d: &[u8], at: usize) -> Option<(String, usize)> {
    let len = *d.get(at)? as usize;
    let s = d.get(at + 1..at + 1 + len)?;
    Some((macroman::decode(s), at + 1 + len))
}

fn decode_vers(d: &[u8]) -> Option<String> {
    // major,minor,stage,prerel (4) + regionCode (2) + shortVer (pascal) +
    // longVer (pascal). The long string is the user-facing one.
    let (short, after) = pascal_at(d, 6)?;
    let long = pascal_at(d, after).map(|(s, _)| s).unwrap_or_default();
    let pick = if long.is_empty() { short } else { long };
    Some(format!("\"{}\"", sanitize(&pick)))
}

fn decode_str(d: &[u8]) -> Option<String> {
    let (s, _) = pascal_at(d, 0)?;
    Some(format!("\"{}\"", sanitize(&s)))
}

fn decode_strlist(d: &[u8]) -> Option<String> {
    if d.len() < 2 {
        return None;
    }
    let n = be16(d, 0) as usize;
    let mut p = 2;
    let mut shown = Vec::new();
    for _ in 0..n.min(4) {
        match pascal_at(d, p) {
            Some((s, next)) => {
                shown.push(format!("\"{}\"", sanitize(&s)));
                p = next;
            }
            None => break,
        }
    }
    let more = if n > shown.len() { ", …" } else { "" };
    Some(format!("{n} strings: {}{more}", shown.join(", ")))
}

fn decode_text(d: &[u8]) -> String {
    let head = &d[..d.len().min(256)];
    let s = sanitize(&macroman::decode(head));
    let preview: String = s.chars().take(48).collect();
    let ell = if d.len() > 48 { "…" } else { "" };
    format!("\"{preview}{ell}\" ({} bytes)", d.len())
}

fn decode_ditl(d: &[u8]) -> Option<String> {
    if d.len() < 2 {
        return None;
    }
    let n = be16(d, 0) as i32 + 1;
    Some(format!("{} items", n.max(0)))
}

#[inline]
fn be16(b: &[u8], o: usize) -> u16 {
    u16::from_be_bytes([b[o], b[o + 1]])
}
#[inline]
fn be32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny but valid resource fork with two types:
    /// `STR ` id 0 named "greeting" → Pascal "Hi", and `vers` id 1 → version
    /// with long string "1.0, test".
    fn sample() -> Vec<u8> {
        // --- data area ---
        let mut data = Vec::new();
        // STR  id 0: Pascal "Hi"
        let str_payload = [0x02, b'H', b'i'];
        let str_data_off = data.len() as u32; // 0
        data.extend_from_slice(&(str_payload.len() as u32).to_be_bytes());
        data.extend_from_slice(&str_payload);
        // vers id 1: 1,0,0,0, region 0, short "1.0", long "1.0, test"
        let mut vers = vec![0x01, 0x00, 0x00, 0x00, 0x00, 0x00];
        vers.push(3);
        vers.extend_from_slice(b"1.0");
        vers.push(9);
        vers.extend_from_slice(b"1.0, test");
        let vers_data_off = data.len() as u32;
        data.extend_from_slice(&(vers.len() as u32).to_be_bytes());
        data.extend_from_slice(&vers);

        // --- map ---
        let mut map = vec![0u8; 28]; // header (reserved) + tlist/namelist offsets
        let tlist_off = 28u16;
        // type list: count-1 = 1 (two types), then two 8-byte entries
        let mut tlist = Vec::new();
        tlist.extend_from_slice(&1u16.to_be_bytes()); // numTypes - 1
        // ref lists will follow the type list; compute offsets from tlist start.
        let type_region = 2 + 2 * 8; // count word + 2 entries
        let str_refoff = type_region as u16; // 18
        let vers_refoff = (type_region + 12) as u16; // 30
        // STR  entry
        tlist.extend_from_slice(b"STR ");
        tlist.extend_from_slice(&0u16.to_be_bytes()); // count-1 = 0
        tlist.extend_from_slice(&str_refoff.to_be_bytes());
        // vers entry
        tlist.extend_from_slice(b"vers");
        tlist.extend_from_slice(&0u16.to_be_bytes());
        tlist.extend_from_slice(&vers_refoff.to_be_bytes());
        // ref lists
        // STR  id 0, name offset 0, attrs 0, dataOff
        tlist.extend_from_slice(&0i16.to_be_bytes());
        tlist.extend_from_slice(&0u16.to_be_bytes()); // name at name-list+0
        tlist.push(0); // attrs
        tlist.extend_from_slice(&str_data_off.to_be_bytes()[1..]); // u24
        tlist.extend_from_slice(&0u32.to_be_bytes()); // handle
        // vers id 1, no name
        tlist.extend_from_slice(&1i16.to_be_bytes());
        tlist.extend_from_slice(&0xFFFFu16.to_be_bytes());
        tlist.push(0);
        tlist.extend_from_slice(&vers_data_off.to_be_bytes()[1..]);
        tlist.extend_from_slice(&0u32.to_be_bytes());

        let namelist_off = tlist_off as usize + tlist.len();
        // name list: "greeting"
        let mut names = Vec::new();
        names.push(8u8);
        names.extend_from_slice(b"greeting");

        map.extend_from_slice(&tlist);
        map.extend_from_slice(&names);
        // fill the two offset words in the map header
        map[24..26].copy_from_slice(&tlist_off.to_be_bytes());
        map[26..28].copy_from_slice(&(namelist_off as u16).to_be_bytes());

        // --- assemble fork ---
        let data_off = 16u32;
        let map_off = 16 + data.len() as u32;
        let mut fork = Vec::new();
        fork.extend_from_slice(&data_off.to_be_bytes());
        fork.extend_from_slice(&map_off.to_be_bytes());
        fork.extend_from_slice(&(data.len() as u32).to_be_bytes());
        fork.extend_from_slice(&(map.len() as u32).to_be_bytes());
        fork.extend_from_slice(&data);
        fork.extend_from_slice(&map);
        fork
    }

    #[test]
    fn parses_inventory_names_and_payloads() {
        let rf = ResourceFork::parse(sample()).unwrap();
        assert_eq!(rf.total(), 2);

        let str_t = rf.types().iter().find(|t| &t.ostype == b"STR ").unwrap();
        assert_eq!(str_t.items.len(), 1);
        assert_eq!(str_t.items[0].id, 0);
        assert_eq!(str_t.items[0].name.as_deref(), Some("greeting"));

        // Payload lookups.
        assert_eq!(rf.resource_bytes(b"STR ", 0).unwrap(), &[0x02, b'H', b'i']);
        assert!(rf.resource_bytes(b"vers", 1).is_some());
        assert!(rf.resource_bytes(b"vers", 99).is_none());
    }

    #[test]
    fn decodes_common_types() {
        let rf = ResourceFork::parse(sample()).unwrap();
        let str_data = rf.resource_bytes(b"STR ", 0).unwrap();
        assert_eq!(decode_summary(b"STR ", str_data).as_deref(), Some("\"Hi\""));

        let vers_data = rf.resource_bytes(b"vers", 1).unwrap();
        assert_eq!(
            decode_summary(b"vers", vers_data).as_deref(),
            Some("\"1.0, test\"")
        );

        assert_eq!(ostype_str(b"STR "), "STR ");
        assert!(decode_summary(b"CODE", &[0u8; 4]).is_none());
    }

    #[test]
    fn truncated_fork_errors() {
        assert!(ResourceFork::parse(vec![0u8; 8]).is_err());
        // Valid header but map past end.
        let mut b = vec![0u8; 16];
        b[4..8].copy_from_slice(&999u32.to_be_bytes()); // mapOff way past end
        assert!(ResourceFork::parse(b).is_err());
    }
}
