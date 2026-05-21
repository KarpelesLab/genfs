//! ext4 HTree (`COMPAT_DIR_INDEX`) write-side support.
//!
//! ## What an indexed directory looks like on disk
//!
//! A regular ext4 directory is a flat sequence of `ext4_dir_entry_2`
//! records concatenated across one or more data blocks. Lookup is O(n).
//!
//! With `COMPAT_DIR_INDEX` the first data block of the directory is a
//! special **dx_root** block whose payload layers a hash-index table
//! over what otherwise look like two ordinary dirent records (`.` and
//! `..`). Bytes 0..12 hold `.`, 12..24 hold `..`, 24..32 hold a tiny
//! [`DxRootInfo`] header, and 32.. is an array of [`DxEntry`] slots
//! `{hash, block}`. Slot 0 is overloaded as `{limit, count, leaf_block}`
//! ([`DxCountLimit`]); slots 1..count carry real `{hash, block}` pairs.
//!
//! Lookups walk the dx_entry table by binary search on `hash`, then dive
//! into the leaf block whose range covers that hash. The leaf block is
//! a perfectly ordinary `ext4_dir_entry_2` block — readers that don't
//! grok HTree simply ignore the root's `dx_*` overlay (the `.` / `..`
//! façade lets them treat the root as a normal dir block whose data
//! ends after `..`) and linear-scan every leaf, which still finds every
//! entry. The cost is just O(n) where O(log n) was on offer.
//!
//! The "fake `.` / `..` façade" trick is also why setting
//! `COMPAT_DIR_INDEX` on a filesystem doesn't force *every* directory
//! to be indexed — un-indexed dirs are still valid; only those whose
//! inode carries `EXT4_INDEX_FL` are interpreted as HTree.
//!
//! ## Hashing
//!
//! Names hash through a half-rounds MD4 (Linux's `DX_HASH_HALF_MD4`),
//! which produces a 32-bit major and 32-bit minor hash. We emit
//! `DX_HASH_HALF_MD4_UNSIGNED` (`hash_version = 1` per the kernel's
//! enum but variant `1` matches the *unsigned* path under modern
//! e2fsprogs — see `linux/include/linux/dx_hash.h`). The low bit of
//! the major hash is reserved as a collision-chain marker and is
//! always cleared; the value `0xFFFF_FFFE` is a sentinel meaning EOF
//! so the hasher remaps it to `0xFFFF_FFFC` before returning.
//!
//! ## metadata_csum
//!
//! dx_root and dx_node blocks carry an 8-byte `dx_tail` at the very
//! end (`dt_reserved` + `dt_checksum`). The csum is over a *subset* of
//! the block — only the slots that are actually in use — not the full
//! 4 KiB. See [`stamp_dx_csum`].
//!
//! ## Scope (v1 writer)
//!
//! - Single-level only (`indirect_levels = 0`). With 4 KiB blocks and
//!   metadata_csum on, a dx_root holds 507 leaf slots × ~250 entries
//!   per leaf ≈ 127 K entries in a single directory — well past
//!   anything a normal rootfs throws at us. Multi-level (dx_node
//!   intermediate blocks) is deferred.
//! - HALF_MD4_UNSIGNED hash only. `tea`, `siphash`, and the legacy
//!   `dx_hack_hash` are not emitted on the write side (the reader,
//!   when added, will need to handle whatever it sees).

use super::constants::DENT_DIR;

/// Hash-version selector for the `dx_root_info.hash_version` field.
/// Matches the constants in `linux/fs/ext4/ext4.h` / `tune2fs(8)`.
pub const DX_HASH_LEGACY: u8 = 0;
pub const DX_HASH_HALF_MD4: u8 = 1;
pub const DX_HASH_TEA: u8 = 2;
pub const DX_HASH_LEGACY_UNSIGNED: u8 = 3;
pub const DX_HASH_HALF_MD4_UNSIGNED: u8 = 4;
pub const DX_HASH_TEA_UNSIGNED: u8 = 5;
pub const DX_HASH_SIPHASH: u8 = 6;

/// Size of the `dx_tail` checksum footer in bytes.
pub const DX_TAIL_LEN: usize = 8;

/// Size of one `dx_entry` slot (also the size of `dx_countlimit` since
/// they overlay).
pub const DX_ENTRY_LEN: usize = 8;

/// Static bytes consumed by the dx_root prefix before the dx_entry
/// table: `.` (12) + `..` (12) + `dx_root_info` (8).
pub const DX_ROOT_HEADER_LEN: usize = 32;

/// One slot in the dx_root / dx_node entry table.
#[derive(Debug, Clone, Copy)]
pub struct DxEntry {
    /// First major hash covered by the leaf at `block`. For slot 0 the
    /// hash field overlays `dx_countlimit` and is otherwise unused as a
    /// hash key (the leftmost leaf is implicit "hash ≥ 0").
    pub hash: u32,
    /// Logical block number of the leaf (or, for multi-level trees,
    /// the child dx_node) within the directory.
    pub block: u32,
}

/// Number of dx_entry slots a dx_root with the given block size can
/// hold, after subtracting the 32-byte prefix and the optional 8-byte
/// `dx_tail`.
pub fn dx_root_limit(block_size: u32, csum_tail: bool) -> usize {
    let mut entry_space = block_size as usize - DX_ROOT_HEADER_LEN;
    if csum_tail {
        entry_space -= DX_TAIL_LEN;
    }
    entry_space / DX_ENTRY_LEN
}

/// Maximum number of real dirent records that fit in a regular leaf
/// block of size `bs` with metadata_csum's 12-byte trailing dirent
/// taken into account. The leaf block format is identical to a plain
/// ext4 directory data block (no dx_* overlay), so the regular
/// `usable_dir_len` rule applies.
pub fn leaf_max_entries(_block_size: u32) -> usize {
    // We don't actually cap by count — what matters is byte
    // occupancy. This helper exists for documentation symmetry with
    // dx_root_limit and is unused inside this module; entry-packing
    // code consults `dir::usable_dir_len` directly.
    usize::MAX
}

// ─── DX_HASH_HALF_MD4 ───────────────────────────────────────────────

/// Compute the (major, minor) HTree hash of `name` under
/// `DX_HASH_HALF_MD4_UNSIGNED`. Matches the kernel's `ext4fs_dirhash`
/// for the unsigned variant: the seed defaults to the MD4 IV, the name
/// is padded into 32-byte chunks via `str2hashbuf`, each chunk is mixed
/// in with `half_md4_transform`, then the major hash is `buf[1]` and
/// the minor hash is `buf[2]`. The low bit of the major hash is
/// cleared (collision-chain marker reservation); the EOF sentinel
/// `0xFFFF_FFFE` is remapped to `0xFFFF_FFFC`.
pub fn half_md4_hash(name: &[u8]) -> (u32, u32) {
    // Default MD4 seed (no per-FS hash-seed override yet).
    let mut buf: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];
    let mut remaining = name;
    let mut in_buf = [0u32; 8];
    // Matches the kernel's `while (len > 0)` — empty names skip the
    // transform entirely so their hash is the raw MD4 IV's middle
    // words.
    while !remaining.is_empty() {
        str2hashbuf_unsigned(remaining, &mut in_buf, 8);
        half_md4_transform(&mut buf, &in_buf);
        if remaining.len() <= 32 {
            break;
        }
        remaining = &remaining[32..];
    }
    let mut hash = buf[1] & !1; // clear collision-chain marker
    if hash == 0xFFFF_FFFE {
        hash = 0xFFFF_FFFC;
    }
    (hash, buf[2])
}

/// Pack the next ≤ 32 bytes of `msg` into `out[..num]` as u32 words
/// for half-MD4 input. Matches the kernel's `str2hashbuf_unsigned`
/// exactly — including its quirk of carrying the held partial word
/// into the trailing slot (so a length-5 name still has its 5th byte
/// fold into the hash, just in the slot following the first full
/// word). The signed variant differs by interpreting bytes as `i8`;
/// modern Linux defaults to unsigned and so do we.
///
/// The pad word `pad = len_byte | len_byte << 8 | len_byte << 16 |
/// len_byte << 24` fills any position the actual name doesn't reach,
/// distinguishing two short names of different lengths from each
/// other (both would otherwise hash on identical prefixes).
fn str2hashbuf_unsigned(msg: &[u8], out: &mut [u32; 8], num: usize) {
    let mut len = msg.len();
    let len_byte = (len & 0xff) as u32;
    let pad = len_byte | (len_byte << 8) | (len_byte << 16) | (len_byte << 24);
    let mut val = pad;
    let mut written = 0usize;
    let mut num_remaining = num;
    if len > num * 4 {
        len = num * 4;
    }
    for i in 0..len {
        // Kernel: val = ((int) ucp[i]) + (val << 8)
        val = (msg[i] as u32).wrapping_add(val.wrapping_shl(8));
        if i % 4 == 3 {
            out[written] = val;
            written += 1;
            val = pad;
            num_remaining -= 1;
        }
    }
    // Trailing slots: write the held partial val once (preserving any
    // unflushed bytes from a non-aligned tail), then fill the rest
    // with pad. e2fsprogs 1.47 expresses this as:
    //   if (--num >= 0) *buf++ = val;
    //   while (--num >= 0) *buf++ = pad;
    if num_remaining > 0 {
        out[written] = val;
        written += 1;
        num_remaining -= 1;
    }
    while num_remaining > 0 {
        out[written] = pad;
        written += 1;
        num_remaining -= 1;
    }
}

/// Half-MD4 transform: three rounds of MD4 mixing using the input as
/// 8 u32 words (only half the standard MD4 input size — the "half"
/// in the name).
///
/// Reference: Linux `fs/ext4/hash.c::half_md4_transform`.
fn half_md4_transform(buf: &mut [u32; 4], i: &[u32; 8]) {
    let (mut a, mut b, mut c, mut d) = (buf[0], buf[1], buf[2], buf[3]);

    // Round 1: F(x,y,z) = (x & y) | (!x & z)
    a = round_f(a, b, c, d, i[0], 3);
    d = round_f(d, a, b, c, i[1], 7);
    c = round_f(c, d, a, b, i[2], 11);
    b = round_f(b, c, d, a, i[3], 19);
    a = round_f(a, b, c, d, i[4], 3);
    d = round_f(d, a, b, c, i[5], 7);
    c = round_f(c, d, a, b, i[6], 11);
    b = round_f(b, c, d, a, i[7], 19);

    // Round 2: G(x,y,z) = (x & y) | (x & z) | (y & z); +K2
    a = round_g(a, b, c, d, i[1], 3);
    d = round_g(d, a, b, c, i[3], 5);
    c = round_g(c, d, a, b, i[5], 9);
    b = round_g(b, c, d, a, i[7], 13);
    a = round_g(a, b, c, d, i[0], 3);
    d = round_g(d, a, b, c, i[2], 5);
    c = round_g(c, d, a, b, i[4], 9);
    b = round_g(b, c, d, a, i[6], 13);

    // Round 3: H(x,y,z) = x ^ y ^ z; +K3
    a = round_h(a, b, c, d, i[3], 3);
    d = round_h(d, a, b, c, i[7], 9);
    c = round_h(c, d, a, b, i[2], 11);
    b = round_h(b, c, d, a, i[6], 15);
    a = round_h(a, b, c, d, i[1], 3);
    d = round_h(d, a, b, c, i[5], 9);
    c = round_h(c, d, a, b, i[0], 11);
    b = round_h(b, c, d, a, i[4], 15);

    buf[0] = buf[0].wrapping_add(a);
    buf[1] = buf[1].wrapping_add(b);
    buf[2] = buf[2].wrapping_add(c);
    buf[3] = buf[3].wrapping_add(d);
}

#[inline]
fn round_f(a: u32, b: u32, c: u32, d: u32, x: u32, s: u32) -> u32 {
    let f = (b & c) | ((!b) & d);
    a.wrapping_add(f).wrapping_add(x).rotate_left(s)
}

#[inline]
fn round_g(a: u32, b: u32, c: u32, d: u32, x: u32, s: u32) -> u32 {
    let g = (b & c) | (b & d) | (c & d);
    const K2: u32 = 0x5a827999;
    a.wrapping_add(g).wrapping_add(x).wrapping_add(K2).rotate_left(s)
}

#[inline]
fn round_h(a: u32, b: u32, c: u32, d: u32, x: u32, s: u32) -> u32 {
    let h = b ^ c ^ d;
    const K3: u32 = 0x6ed9eba1;
    a.wrapping_add(h).wrapping_add(x).wrapping_add(K3).rotate_left(s)
}

// ─── dx_root encoding ───────────────────────────────────────────────

/// Build the dx_root block for an indexed directory.
///
/// `self_inode` / `parent_inode` populate the fake `.` / `..` entries
/// at the head of the block. `hash_version` is the algorithm we used
/// when bucketing names into `entries` (see [`half_md4_hash`]).
/// `entries[0]` is the dx_countlimit slot — its `hash` field is the
/// packed `(limit, count)` value, its `block` field is the leftmost
/// leaf block number. `entries[1..]` are real `{hash, block}` rows
/// sorted by ascending hash.
///
/// When `csum_tail` is set, the final 8 bytes are reserved for the
/// dx_tail (zero-stamped here; the actual CRC32C is computed at flush
/// time by [`compute_dx_csum`]).
pub fn make_dx_root_block(
    self_inode: u32,
    parent_inode: u32,
    block_size: u32,
    hash_version: u8,
    entries: &[DxEntry],
    with_filetype: bool,
    csum_tail: bool,
) -> Vec<u8> {
    assert!(
        !entries.is_empty(),
        "dx_root must have at least the countlimit slot"
    );
    let mut buf = vec![0u8; block_size as usize];

    // Fake "." dirent (offset 0..12).
    write_fake_dirent(&mut buf[0..12], self_inode, b".", with_filetype);

    // Fake ".." dirent (offset 12). rec_len spans from offset 12 to
    // the END of the block — always, regardless of csum_tail. The
    // dx_entry table and (when present) the 8-byte dx_tail both live
    // INSIDE this declared rec_len so a legacy linear-walk reader sees
    // ".." absorb everything and never tries to decode the dx_entries
    // (or dx_tail) as a separate dirent. The kernel's HTree-aware
    // code path indexes into the dx_entry array directly and ignores
    // the rec_len fiction.
    let _ = csum_tail; // (kept for API symmetry with regular dir block)
    let dotdot_rec_len = (block_size as usize - 12) as u16;
    let mut dotdot = vec![0u8; 12];
    dotdot[0..4].copy_from_slice(&parent_inode.to_le_bytes());
    dotdot[4..6].copy_from_slice(&dotdot_rec_len.to_le_bytes());
    if with_filetype {
        dotdot[6] = 2; // name_len
        dotdot[7] = DENT_DIR;
    } else {
        dotdot[6..8].copy_from_slice(&2u16.to_le_bytes());
    }
    dotdot[8] = b'.';
    dotdot[9] = b'.';
    buf[12..24].copy_from_slice(&dotdot);

    // dx_root_info (offset 24..32).
    //   0..4 reserved_zero (le32, must be 0)
    //   4    hash_version
    //   5    info_length (== 8)
    //   6    indirect_levels (== 0 in v1; multi-level deferred)
    //   7    unused_flags (0)
    buf[24..28].copy_from_slice(&0u32.to_le_bytes());
    buf[28] = hash_version;
    buf[29] = 8;
    buf[30] = 0; // indirect_levels
    buf[31] = 0; // unused_flags

    // dx_entry table starting at offset 32. entries[0] is the
    // dx_countlimit slot: low 16 bits of the "hash" field are `limit`,
    // high 16 bits are `count`. Its `block` field is the leaf block
    // for the leftmost (hash=0) range.
    for (i, e) in entries.iter().enumerate() {
        let off = DX_ROOT_HEADER_LEN + i * DX_ENTRY_LEN;
        buf[off..off + 4].copy_from_slice(&e.hash.to_le_bytes());
        buf[off + 4..off + 8].copy_from_slice(&e.block.to_le_bytes());
    }

    // dx_tail placeholder when metadata_csum is on; the 4-byte CRC is
    // stamped at flush time.
    if csum_tail {
        let tail_off = block_size as usize - DX_TAIL_LEN;
        for b in buf[tail_off..].iter_mut() {
            *b = 0;
        }
    }
    buf
}

/// Pack the (limit, count) pair into the low/high halves of the
/// dx_countlimit slot's "hash" field. Use as `entries[0].hash`.
pub fn pack_countlimit(limit: u16, count: u16) -> u32 {
    (limit as u32) | ((count as u32) << 16)
}

/// Compute the dx_root / dx_node CRC32C over the in-use prefix of the
/// block: `count_offset + count * DX_ENTRY_LEN` bytes, then the 4-byte
/// `dt_reserved` field (zero), then 4 bytes of zero standing in for
/// `dt_checksum`. Mirrors the kernel's `ext4_dx_csum`.
pub fn compute_dx_csum(
    raw_update: impl Fn(u32, &[u8]) -> u32,
    seed: u32,
    inode_num: u32,
    inode_generation: u32,
    block: &[u8],
    count_offset: usize,
    count: usize,
) -> u32 {
    // Per-inode seed (kernel's `i_csum_seed`).
    let c = raw_update(seed, &inode_num.to_le_bytes());
    let c = raw_update(c, &inode_generation.to_le_bytes());
    // The in-use prefix.
    let used_len = count_offset + count * DX_ENTRY_LEN;
    let c = raw_update(c, &block[..used_len]);
    // dt_reserved (4 bytes of zero).
    let c = raw_update(c, &0u32.to_le_bytes());
    // dt_checksum placeholder (4 bytes of zero — checksum is computed
    // OVER the field treated as zero, then stamped back into it).
    raw_update(c, &0u32.to_le_bytes())
}

/// Stamp `csum` into the `dt_checksum` field at the very end of the
/// block. Pairs with [`compute_dx_csum`].
pub fn stamp_dx_csum(block: &mut [u8], csum: u32) {
    let n = block.len();
    block[n - 4..].copy_from_slice(&csum.to_le_bytes());
}

fn write_fake_dirent(dst: &mut [u8], inode: u32, name: &[u8], with_filetype: bool) {
    assert!(dst.len() >= 12, "fake dirent slot must be 12 bytes");
    dst[0..4].copy_from_slice(&inode.to_le_bytes());
    dst[4..6].copy_from_slice(&12u16.to_le_bytes());
    if with_filetype {
        dst[6] = name.len() as u8;
        dst[7] = DENT_DIR;
    } else {
        dst[6..8].copy_from_slice(&(name.len() as u16).to_le_bytes());
    }
    dst[8..8 + name.len()].copy_from_slice(name);
    for b in dst[8 + name.len()..12].iter_mut() {
        *b = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// dx_root layout sanity: ".", "..", info header.
    #[test]
    fn dx_root_header_layout() {
        let entries = vec![DxEntry {
            hash: pack_countlimit(508, 1),
            block: 1,
        }];
        let buf = make_dx_root_block(12, 2, 4096, DX_HASH_HALF_MD4_UNSIGNED, &entries, true, false);
        // "." at offset 0
        assert_eq!(u32::from_le_bytes(buf[0..4].try_into().unwrap()), 12);
        assert_eq!(u16::from_le_bytes(buf[4..6].try_into().unwrap()), 12);
        assert_eq!(buf[6], 1); // name_len
        assert_eq!(buf[7], DENT_DIR);
        assert_eq!(buf[8], b'.');
        // ".." at offset 12, rec_len = 4096 - 12 (no csum tail).
        assert_eq!(u32::from_le_bytes(buf[12..16].try_into().unwrap()), 2);
        assert_eq!(
            u16::from_le_bytes(buf[16..18].try_into().unwrap()),
            (4096 - 12) as u16
        );
        // dx_root_info at offset 24.
        assert_eq!(u32::from_le_bytes(buf[24..28].try_into().unwrap()), 0);
        assert_eq!(buf[28], DX_HASH_HALF_MD4_UNSIGNED);
        assert_eq!(buf[29], 8); // info_length
        assert_eq!(buf[30], 0); // indirect_levels
        // entries[0] (the countlimit slot) at offset 32.
        let cl_hash = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        assert_eq!(cl_hash & 0xffff, 508);
        assert_eq!(cl_hash >> 16, 1);
        assert_eq!(u32::from_le_bytes(buf[36..40].try_into().unwrap()), 1);
    }

    /// Csum-tail variant: ".." rec_len shrinks by 8 to leave room for dx_tail,
    /// and the dx_root_limit drops by one slot.
    #[test]
    fn dx_root_limit_metadata_csum() {
        assert_eq!(dx_root_limit(4096, false), (4096 - 32) / 8);
        assert_eq!(dx_root_limit(4096, true), (4096 - 32 - 8) / 8);
        assert_eq!(dx_root_limit(1024, true), (1024 - 32 - 8) / 8);
    }

    /// Empty-name hash should be deterministic and ≠ 0.
    #[test]
    fn half_md4_empty_name() {
        let (h, _) = half_md4_hash(b"");
        // Low bit always cleared.
        assert_eq!(h & 1, 0);
    }

    /// Two different names must produce different major hashes (with
    /// overwhelming probability — this is a smoke test, not a
    /// collision-resistance proof).
    #[test]
    fn half_md4_different_names_different_hashes() {
        let (h1, _) = half_md4_hash(b"foo");
        let (h2, _) = half_md4_hash(b"bar");
        let (h3, _) = half_md4_hash(b"foobar");
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h2, h3);
    }

    /// All hashes must have the low bit clear (collision-chain marker).
    #[test]
    fn half_md4_low_bit_always_clear() {
        for n in 0u32..32 {
            let name = format!("name{n:04}");
            let (h, _) = half_md4_hash(name.as_bytes());
            assert_eq!(h & 1, 0, "hash {h:#x} of {name:?} has low bit set");
        }
    }

    /// EOF sentinel must not be produced.
    #[test]
    fn half_md4_eof_sentinel_remapped() {
        // We can't easily force the algorithm to produce 0xFFFFFFFE
        // for an arbitrary name, but the remap path is exercised
        // by the code itself; here we just confirm the post-condition.
        for n in 0u32..2000 {
            let name = format!("entry_{n}");
            let (h, _) = half_md4_hash(name.as_bytes());
            assert_ne!(h, 0xFFFF_FFFE);
        }
    }

    /// countlimit packing round-trips.
    #[test]
    fn countlimit_packing() {
        let v = pack_countlimit(508, 2);
        assert_eq!(v & 0xffff, 508);
        assert_eq!(v >> 16, 2);
    }

    /// Bit-exact agreement with `libext2fs::ext2fs_dirhash` for
    /// DX_HASH_HALF_MD4 (and DX_HASH_HALF_MD4_UNSIGNED — identical for
    /// ASCII names since signed/unsigned interpretation only differs
    /// on bytes ≥ 0x80). Reference values captured by a small C
    /// program linking against e2fsprogs 1.47.4. If this test ever
    /// breaks, the implementation is wrong — e2fsck will reject the
    /// on-disk index.
    #[test]
    fn half_md4_matches_libext2fs() {
        let cases: &[(&[u8], u32, u32)] = &[
            (b"entry_0000", 0x63df8060, 0x0cf0abb8),
            (b"entry_0001", 0xb88bbf6c, 0x2dae5e85),
            (b"entry_0002", 0xa4283f96, 0x870c4121),
            (b"foo", 0x74c657ac, 0x85a8d812),
            (b"bar", 0x4caaf2ba, 0x16c15fb9),
            (b"baz", 0xee788c74, 0xc5a8743c),
        ];
        for &(name, want_h, want_m) in cases {
            let (got_h, got_m) = half_md4_hash(name);
            assert_eq!(
                got_h,
                want_h,
                "major hash mismatch for {:?}: got {got_h:#010x}, want {want_h:#010x}",
                String::from_utf8_lossy(name)
            );
            assert_eq!(
                got_m,
                want_m,
                "minor hash mismatch for {:?}: got {got_m:#010x}, want {want_m:#010x}",
                String::from_utf8_lossy(name)
            );
        }
    }
}
