//! SquashFS extended-attributes table.
//!
//! Two pieces:
//!
//! - **Lookup table**: an array of 16-byte entries, one per "xattr set",
//!   stored in metablocks. An inode's `xattr_index` field is an index
//!   into this array. Each entry says where the key/value list for that
//!   set lives (a 64-bit metablock reference, with the same encoding as
//!   the inode reference: `(meta_block_loc << 16) | offset_in_block`),
//!   how many K/V pairs it contains, and its total uncompressed size.
//! - **K/V data**: a sequence of metablocks holding key/value records.
//!   Each record is `u16 type | u16 name_size | name[] | u32 value_size | value[]`.
//!   The `type` is one of 0=`user.`, 1=`trusted.`, 2=`security.`,
//!   plus bit `0x0100` to indicate "out-of-line" values (we don't emit
//!   those when writing).
//!
//! The superblock's `xattr_id_table_start` points at a small uncompressed
//! header: `u64 kv_start | u32 count | u32 unused | u64[] locations`,
//! where `kv_start` is the absolute offset of the first K/V metablock
//! and `locations` describes the lookup-table metablocks.
//!
//! When there are no xattrs, the superblock field is `u64::MAX`.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::squashfs::Compression;
use crate::fs::squashfs::metablock::{encode_metablock, read_metablock};

/// Recognised key prefixes in the SquashFS xattr table.
pub const XATTR_TYPE_USER: u16 = 0;
pub const XATTR_TYPE_TRUSTED: u16 = 1;
pub const XATTR_TYPE_SECURITY: u16 = 2;
const XATTR_TYPE_MASK: u16 = 0xFF;
const XATTR_FLAG_OOL: u16 = 0x0100;

/// One key/value pair as returned to the caller. The key is the full
/// dotted name (e.g. `"user.color"`, `"security.selinux"`).
#[derive(Debug, Clone)]
pub struct Xattr {
    pub key: String,
    pub value: Vec<u8>,
}

/// Cached, on-demand xattr reader. Mirrors [`super::idtable::IdTable`].
#[derive(Debug, Default)]
pub struct XattrReader {
    loaded: bool,
    /// Absolute disk offset of the first K/V metablock.
    kv_start: u64,
    /// Decoded lookup entries: (xattr_ref, count, size).
    lookup: Vec<XattrId>,
}

#[derive(Debug, Clone, Copy)]
struct XattrId {
    xattr_ref: u64,
    count: u32,
    _size: u32,
}

impl XattrReader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the lookup table from disk. Idempotent.
    pub fn ensure_loaded(
        &mut self,
        dev: &mut dyn BlockDevice,
        xattr_table_start: u64,
        compression: Compression,
    ) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        self.loaded = true;
        if xattr_table_start == u64::MAX {
            return Ok(());
        }
        // Read the 16-byte header.
        let mut head = [0u8; 16];
        dev.read_at(xattr_table_start, &mut head)?;
        let kv_start = u64::from_le_bytes(head[0..8].try_into().unwrap());
        let count = u32::from_le_bytes(head[8..12].try_into().unwrap());
        // bytes 12..16 ignored (unused per spec).
        if count == 0 {
            self.kv_start = kv_start;
            return Ok(());
        }
        let total_bytes = count as usize * 16;
        let metablock_count = total_bytes.div_ceil(8192);
        // Locations array immediately follows the 16-byte header.
        let mut locs = vec![0u8; metablock_count * 8];
        dev.read_at(xattr_table_start + 16, &mut locs)?;
        let mut entries = Vec::with_capacity(count as usize);
        let mut remaining = total_bytes;
        for i in 0..metablock_count {
            let off = i * 8;
            let mb_disk = u64::from_le_bytes(locs[off..off + 8].try_into().unwrap());
            let mb = read_metablock(dev, mb_disk, compression)?;
            let want = remaining.min(8192);
            if mb.data.len() < want {
                return Err(crate::Error::InvalidImage(format!(
                    "squashfs: xattr lookup metablock {i} too short"
                )));
            }
            for j in 0..(want / 16) {
                let p = j * 16;
                entries.push(XattrId {
                    xattr_ref: u64::from_le_bytes(mb.data[p..p + 8].try_into().unwrap()),
                    count: u32::from_le_bytes(mb.data[p + 8..p + 12].try_into().unwrap()),
                    _size: u32::from_le_bytes(mb.data[p + 12..p + 16].try_into().unwrap()),
                });
            }
            remaining -= want;
        }
        self.kv_start = kv_start;
        self.lookup = entries;
        Ok(())
    }

    /// Fetch all K/V pairs for the xattr set at `idx`. Returns an empty
    /// list if `idx` is out of range or `u32::MAX` (the sentinel for "no
    /// xattrs"), which mirrors what mksquashfs writes.
    pub fn fetch(
        &self,
        dev: &mut dyn BlockDevice,
        idx: u32,
        compression: Compression,
    ) -> Result<Vec<Xattr>> {
        if idx == u32::MAX || idx as usize >= self.lookup.len() {
            return Ok(Vec::new());
        }
        let entry = self.lookup[idx as usize];
        // Decode xattr_ref: high 48 bits = metablock disk offset relative to
        // kv_start, low 16 bits = uncompressed offset within that block.
        let meta_block_rel = entry.xattr_ref >> 16;
        let in_block_offset = (entry.xattr_ref & 0xFFFF) as usize;
        // Walk K/V records. Crossing metablock boundaries is allowed.
        let mut out = Vec::with_capacity(entry.count as usize);
        let mut mb_rel = meta_block_rel;
        let mut offset = in_block_offset;
        for _ in 0..entry.count {
            let (kv, nb, no) = read_kv_record(dev, self.kv_start, mb_rel, offset, compression)?;
            mb_rel = nb;
            offset = no;
            out.push(kv);
        }
        Ok(out)
    }
}

/// Read a single key/value record starting at `(mb_rel, offset)` within
/// the K/V metablock stream anchored at `kv_start`. Returns the parsed
/// record plus the new cursor.
fn read_kv_record(
    dev: &mut dyn BlockDevice,
    kv_start: u64,
    mut mb_rel: u64,
    mut offset: usize,
    compression: Compression,
) -> Result<(Xattr, u64, usize)> {
    use crate::fs::squashfs::metablock::MetadataReader;
    let mut mr = MetadataReader::new(kv_start, compression);
    // Key header: u16 type, u16 name_size.
    let (head, nb, no) = mr.read(dev, mb_rel, offset, 4)?;
    mb_rel = nb;
    offset = no;
    let raw_type = u16::from_le_bytes(head[0..2].try_into().unwrap());
    let name_size = u16::from_le_bytes(head[2..4].try_into().unwrap()) as usize;
    let (name_bytes, nb, no) = mr.read(dev, mb_rel, offset, name_size)?;
    mb_rel = nb;
    offset = no;
    let key_prefix = prefix_for_type(raw_type & XATTR_TYPE_MASK)?;
    let name_str = std::str::from_utf8(&name_bytes)
        .map_err(|e| crate::Error::InvalidImage(format!("squashfs: xattr name not utf-8: {e}")))?;
    let key = format!("{}{}", key_prefix, name_str);
    // Value header: u32 size, then bytes.
    let (vh, nb, no) = mr.read(dev, mb_rel, offset, 4)?;
    mb_rel = nb;
    offset = no;
    let v_size = u32::from_le_bytes(vh[0..4].try_into().unwrap()) as usize;
    let (mut v_bytes, nb, no) = mr.read(dev, mb_rel, offset, v_size)?;
    mb_rel = nb;
    offset = no;
    // Out-of-line values: v_bytes is a u64 reference. Follow it.
    if raw_type & XATTR_FLAG_OOL != 0 && v_size == 8 {
        let oref = u64::from_le_bytes(v_bytes.as_slice().try_into().unwrap());
        let ref_block = oref >> 16;
        let ref_offset = (oref & 0xFFFF) as usize;
        let mut mr2 = MetadataReader::new(kv_start, compression);
        let (vh2, nb2, no2) = mr2.read(dev, ref_block, ref_offset, 4)?;
        let real_size = u32::from_le_bytes(vh2[0..4].try_into().unwrap()) as usize;
        let (real_bytes, _, _) = mr2.read(dev, nb2, no2, real_size)?;
        v_bytes = real_bytes;
    }
    Ok((
        Xattr {
            key,
            value: v_bytes,
        },
        mb_rel,
        offset,
    ))
}

fn prefix_for_type(t: u16) -> Result<&'static str> {
    match t {
        XATTR_TYPE_USER => Ok("user."),
        XATTR_TYPE_TRUSTED => Ok("trusted."),
        XATTR_TYPE_SECURITY => Ok("security."),
        other => Err(crate::Error::InvalidImage(format!(
            "squashfs: unknown xattr type {other}"
        ))),
    }
}

fn type_for_prefix(key: &str) -> Option<(u16, &str)> {
    if let Some(rest) = key.strip_prefix("user.") {
        Some((XATTR_TYPE_USER, rest))
    } else if let Some(rest) = key.strip_prefix("trusted.") {
        Some((XATTR_TYPE_TRUSTED, rest))
    } else if let Some(rest) = key.strip_prefix("security.") {
        Some((XATTR_TYPE_SECURITY, rest))
    } else {
        None
    }
}

/// A pre-dedup xattr set as supplied by the caller during writing.
pub type XattrSet = Vec<Xattr>;

/// Build the on-disk xattr table from a list of unique xattr sets.
/// Returns `(disk_payload, header_offset_within_payload)`. The caller
/// writes `disk_payload` at absolute offset `B`; the superblock's
/// `xattr_id_table_start` becomes `B + header_offset`.
///
/// Out-of-line values are never emitted — we always inline. The table is
/// laid out as:
///
/// 1. K/V metablocks.
/// 2. Lookup-table metablocks (16-byte entries).
/// 3. 16-byte header (`kv_start`, `count`, `unused`) + `u64[]` lookup-table
///    metablock locations.
///
/// All offsets stored on disk are absolute, so the caller supplies `base`
/// = the absolute byte offset where `disk_payload` will live. Returns the
/// header offset relative to `base`.
pub fn encode_xattr_table(
    sets: &[XattrSet],
    base: u64,
    compression: Compression,
) -> Result<(Vec<u8>, u64)> {
    // 1) Serialise the K/V stream into a single uncompressed byte buffer,
    //    recording per-set `(uncompressed_byte_start, count)`.
    let mut kv_raw = Vec::new();
    let mut per_set: Vec<(u32, u32)> = Vec::with_capacity(sets.len()); // (uncompressed_offset, count)
    for set in sets {
        let off = kv_raw.len() as u32;
        per_set.push((off, set.len() as u32));
        for kv in set {
            let (ty, rest) = type_for_prefix(&kv.key).ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "squashfs: xattr key {:?} has unknown namespace (expected user./trusted./security.)",
                    kv.key
                ))
            })?;
            kv_raw.extend_from_slice(&ty.to_le_bytes());
            kv_raw.extend_from_slice(&(rest.len() as u16).to_le_bytes());
            kv_raw.extend_from_slice(rest.as_bytes());
            kv_raw.extend_from_slice(&(kv.value.len() as u32).to_le_bytes());
            kv_raw.extend_from_slice(&kv.value);
        }
    }
    // 2) Chunk the K/V byte stream into 8 KiB metablocks. Record each
    //    block's *absolute* disk offset (we know `base`) and its
    //    on-disk size so we can convert uncompressed offsets into
    //    xattr_ref values.
    let mut out = Vec::new();
    let mut kv_block_offsets_abs: Vec<u64> = Vec::new();
    let mut kv_block_disk_sizes: Vec<u32> = Vec::new();
    {
        let mut pos = 0usize;
        while pos < kv_raw.len() {
            let end = (pos + 8192).min(kv_raw.len());
            let mb = encode_metablock(&kv_raw[pos..end], compression)?;
            kv_block_offsets_abs.push(base + out.len() as u64);
            kv_block_disk_sizes.push(mb.len() as u32);
            out.extend_from_slice(&mb);
            pos = end;
        }
    }
    let kv_start_abs = if kv_block_offsets_abs.is_empty() {
        // No xattrs — the header still needs a kv_start; use the end of
        // the (empty) K/V section.
        base + out.len() as u64
    } else {
        kv_block_offsets_abs[0]
    };
    // Helper: turn an uncompressed K/V byte offset into a
    // (meta_block_offset_relative_to_kv_start << 16) | offset_in_block
    // xattr_ref. The block boundaries are at 8 KiB increments in the
    // uncompressed stream.
    let xattr_ref_from_uncompressed = |u_off: u32| -> u64 {
        let mb_idx = (u_off as usize) / 8192;
        let in_off = (u_off as usize) % 8192;
        // Block offset relative to kv_start = sum of disk sizes of prior blocks.
        let rel: u64 = kv_block_disk_sizes[..mb_idx]
            .iter()
            .map(|&n| n as u64)
            .sum();
        (rel << 16) | (in_off as u64)
    };

    // 3) Build lookup-table entries (16 bytes each). Each set's `size` is
    //    the total uncompressed byte length it consumes in the K/V stream
    //    (i.e. the difference between the next set's offset and ours,
    //    or kv_raw.len() for the last set).
    let mut lookup_raw = Vec::with_capacity(per_set.len() * 16);
    for (i, &(u_off, count)) in per_set.iter().enumerate() {
        let next = if i + 1 < per_set.len() {
            per_set[i + 1].0
        } else {
            kv_raw.len() as u32
        };
        let size = next - u_off;
        let xref = if kv_block_offsets_abs.is_empty() {
            0
        } else {
            xattr_ref_from_uncompressed(u_off)
        };
        lookup_raw.extend_from_slice(&xref.to_le_bytes());
        lookup_raw.extend_from_slice(&count.to_le_bytes());
        lookup_raw.extend_from_slice(&size.to_le_bytes());
    }

    // 4) Chunk lookup-table bytes into metablocks. Record absolute offsets.
    let mut lookup_block_offsets_abs: Vec<u64> = Vec::new();
    {
        let mut pos = 0usize;
        while pos < lookup_raw.len() {
            let end = (pos + 8192).min(lookup_raw.len());
            let mb = encode_metablock(&lookup_raw[pos..end], compression)?;
            lookup_block_offsets_abs.push(base + out.len() as u64);
            out.extend_from_slice(&mb);
            pos = end;
        }
    }

    // 5) Header + lookup-block location array.
    let header_offset = out.len() as u64;
    out.extend_from_slice(&kv_start_abs.to_le_bytes());
    out.extend_from_slice(&(per_set.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // unused
    for l in &lookup_block_offsets_abs {
        out.extend_from_slice(&l.to_le_bytes());
    }
    Ok((out, header_offset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    #[test]
    fn round_trip_simple_xattrs() {
        // One inode with two xattrs.
        let set: XattrSet = vec![
            Xattr {
                key: "user.color".into(),
                value: b"orange".to_vec(),
            },
            Xattr {
                key: "security.selinux".into(),
                value: b"unconfined_u".to_vec(),
            },
        ];
        let base = 200u64;
        let (payload, hdr_off) =
            encode_xattr_table(std::slice::from_ref(&set), base, Compression::Unknown(0)).unwrap();
        let mut dev = MemoryBackend::new(base + payload.len() as u64 + 64);
        dev.write_at(base, &payload).unwrap();
        let mut r = XattrReader::new();
        r.ensure_loaded(&mut dev, base + hdr_off, Compression::Unknown(0))
            .unwrap();
        let read_set = r.fetch(&mut dev, 0, Compression::Unknown(0)).unwrap();
        assert_eq!(read_set.len(), 2);
        assert_eq!(read_set[0].key, "user.color");
        assert_eq!(read_set[0].value, b"orange");
        assert_eq!(read_set[1].key, "security.selinux");
        assert_eq!(read_set[1].value, b"unconfined_u");
    }

    #[test]
    fn empty_xattr_table_decodes_zero_count() {
        let base = 0u64;
        let (payload, hdr_off) = encode_xattr_table(&[], base, Compression::Unknown(0)).unwrap();
        let mut dev = MemoryBackend::new(payload.len() as u64 + 64);
        dev.write_at(0, &payload).unwrap();
        let mut r = XattrReader::new();
        r.ensure_loaded(&mut dev, hdr_off, Compression::Unknown(0))
            .unwrap();
        let empty = r.fetch(&mut dev, 0, Compression::Unknown(0)).unwrap();
        assert_eq!(empty.len(), 0);
    }
}
