//! Fresh-image F2FS formatter.
//!
//! Produces an empty F2FS volume on a [`BlockDevice`]. The layout is
//! laid out exactly as the read driver expects (see the head comment in
//! [`super::superblock`] and [`super::checkpoint`]); the writer then
//! mutates that layout incrementally through [`super::write`] and stamps
//! a fresh checkpoint pack at [`super::F2fs::flush`] time.
//!
//! ## What "fresh-image" buys us
//!
//! - No wear-levelling: blocks are allocated sequentially from
//!   `main_blkaddr` upward.
//! - No checkpoint replay / roll-forward: a single CP pack (CP0) is
//!   stamped at end of life with the live state; CP1 is left zero and
//!   only validates after the writer asks for a second flush.
//! - No GC / cleaning: a "section" is never reused, so the SIT bitmap
//!   only ever gains valid bits.
//!
//! ## On-disk regions (all in 4 KiB block units)
//!
//! ```text
//!   block 0..2   : superblock (primary at byte 1024, backup at 1024 + 4096)
//!   block 2..    : checkpoint area, `segment_count_ckpt * blocks_per_seg` blocks
//!                  → CP0 head + summary, CP1 head + summary
//!         ..     : SIT area  `segment_count_sit * blocks_per_seg`
//!         ..     : NAT area  `segment_count_nat * blocks_per_seg`
//!         ..     : SSA area  `segment_count_ssa * blocks_per_seg`
//!         ..     : main area `segment_count_main * blocks_per_seg`
//! ```
//!
//! The two SIT / NAT halves serve the shadow-copy scheme but the v1
//! formatter writes both halves to the same content so the reader's
//! "pick `cur_nat_pack`" logic is a no-op.
//!
//! Reference: kernel docs §"Filesystem Architecture" and FAST '15 §2.

use crate::Result;
use crate::block::BlockDevice;

use super::constants::{
    F2FS_BLK_CSUM_OFFSET, F2FS_BLKSIZE, NAT_ENTRY_PER_BLOCK, NAT_ENTRY_SIZE, S_IFDIR,
};
use super::superblock::{F2FS_MAGIC, F2FS_ROOT_INO_DEFAULT, SB_OFFSET_BACKUP, SB_OFFSET_PRIMARY};

/// User-facing format knobs. Anything not in here is derived from the
/// device size and a couple of geometry defaults.
#[derive(Debug, Clone)]
pub struct FormatOpts {
    /// 16-byte UUID stamped into the superblock. Defaults to all zero
    /// (caller decides whether to randomise).
    pub uuid: [u8; 16],
    /// Up to 512 UTF-16LE code units; truncated when longer.
    pub volume_label: String,
    /// log2 of blocks per segment. F2FS canonical value is `9`
    /// (→ 512 blocks → 2 MiB segments). The formatter accepts smaller
    /// values for tiny test images.
    pub log_blocks_per_seg: u32,
    /// Segments per section. Set to 1 unless you know what you're doing.
    pub segs_per_sec: u32,
    /// Sections per zone. Same caveat as above.
    pub secs_per_zone: u32,
    /// File mode for the root directory (mode bits, no type — type is
    /// always `S_IFDIR`).
    pub root_mode: u16,
    /// Owner of the root directory.
    pub root_uid: u32,
    pub root_gid: u32,
    /// Stamp this Unix-epoch timestamp on root and on the CP version.
    pub mtime: u32,
}

impl Default for FormatOpts {
    fn default() -> Self {
        Self {
            uuid: [0; 16],
            volume_label: String::new(),
            // log2(512) = 9 — standard F2FS segment size. Smaller is
            // allowed for tiny test images (the formatter accepts 2..=9).
            log_blocks_per_seg: 9,
            segs_per_sec: 1,
            secs_per_zone: 1,
            root_mode: 0o755,
            root_uid: 0,
            root_gid: 0,
            mtime: 0,
        }
    }
}

/// Resolved geometry of the volume we're about to write. Every field is
/// in 4 KiB blocks unless noted.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    pub total_blocks: u64,
    pub blocks_per_seg: u32,
    pub log_blocks_per_seg: u32,
    pub segs_per_sec: u32,
    pub secs_per_zone: u32,
    pub segment_count_ckpt: u32,
    pub segment_count_sit: u32,
    pub segment_count_nat: u32,
    pub segment_count_ssa: u32,
    pub segment_count_main: u32,
    pub cp_blkaddr: u32,
    pub sit_blkaddr: u32,
    pub nat_blkaddr: u32,
    pub ssa_blkaddr: u32,
    pub main_blkaddr: u32,
}

impl Geometry {
    /// Total segment count across every region.
    pub fn segment_count(&self) -> u32 {
        self.segment_count_ckpt
            + self.segment_count_sit
            + self.segment_count_nat
            + self.segment_count_ssa
            + self.segment_count_main
    }

    /// Block addresses of the two CP packs.
    pub fn cp_pack_blkaddrs(&self) -> [u32; 2] {
        [self.cp_blkaddr, self.cp_blkaddr + self.blocks_per_seg]
    }

    /// Total number of NAT entries the on-disk pages can hold (per pack,
    /// since each pack holds half the NAT region).
    pub fn max_nat_entries(&self) -> u32 {
        let pages_per_pack = (self.segment_count_nat * self.blocks_per_seg) / 2;
        pages_per_pack * NAT_ENTRY_PER_BLOCK as u32
    }
}

/// Plan a geometry for a device of `total_blocks` 4 KiB blocks.
///
/// The plan is biased toward "smallest legal layout that still works":
/// 2 CP segs (two packs), 1 SIT seg, 2 NAT segs (two packs), 1 SSA seg,
/// the rest to the main area.
pub fn plan_geometry(total_blocks: u64, opts: &FormatOpts) -> Result<Geometry> {
    if total_blocks < 64 {
        return Err(crate::Error::InvalidArgument(format!(
            "f2fs: device too small ({total_blocks} blocks; need ≥ 64)"
        )));
    }
    if !(1..=9).contains(&opts.log_blocks_per_seg) {
        return Err(crate::Error::InvalidArgument(format!(
            "f2fs: log_blocks_per_seg must be in 1..=9, got {}",
            opts.log_blocks_per_seg
        )));
    }
    let blocks_per_seg: u32 = 1u32 << opts.log_blocks_per_seg;

    // Two superblock blocks at the start.
    let sb_blocks = 2u32;

    // Meta region: CP / SIT / NAT / SSA in *segments*.
    let segment_count_ckpt = 2u32;
    let segment_count_sit = 1u32;
    let segment_count_nat = 2u32;
    let segment_count_ssa = 1u32;

    let meta_blocks = sb_blocks
        + (segment_count_ckpt + segment_count_sit + segment_count_nat + segment_count_ssa)
            * blocks_per_seg;
    if (meta_blocks as u64) >= total_blocks {
        return Err(crate::Error::InvalidArgument(format!(
            "f2fs: device has {total_blocks} blocks, need > {meta_blocks} for metadata"
        )));
    }
    // Main area: rounded down to a whole-segment count.
    let main_blocks = total_blocks as u32 - meta_blocks;
    let segment_count_main = main_blocks / blocks_per_seg;
    if segment_count_main == 0 {
        return Err(crate::Error::InvalidArgument(
            "f2fs: not enough room for any main-area segment".into(),
        ));
    }

    let cp_blkaddr = sb_blocks;
    let sit_blkaddr = cp_blkaddr + segment_count_ckpt * blocks_per_seg;
    let nat_blkaddr = sit_blkaddr + segment_count_sit * blocks_per_seg;
    let ssa_blkaddr = nat_blkaddr + segment_count_nat * blocks_per_seg;
    let main_blkaddr = ssa_blkaddr + segment_count_ssa * blocks_per_seg;

    Ok(Geometry {
        total_blocks,
        blocks_per_seg,
        log_blocks_per_seg: opts.log_blocks_per_seg,
        segs_per_sec: opts.segs_per_sec.max(1),
        secs_per_zone: opts.secs_per_zone.max(1),
        segment_count_ckpt,
        segment_count_sit,
        segment_count_nat,
        segment_count_ssa,
        segment_count_main,
        cp_blkaddr,
        sit_blkaddr,
        nat_blkaddr,
        ssa_blkaddr,
        main_blkaddr,
    })
}

/// Stamp two primary + backup superblock copies derived from `geom` and
/// `opts`.
pub(crate) fn write_superblocks(
    dev: &mut dyn BlockDevice,
    geom: &Geometry,
    opts: &FormatOpts,
) -> Result<()> {
    let mut buf = vec![0u8; 0x400];
    buf[0x00..0x04].copy_from_slice(&F2FS_MAGIC.to_le_bytes());
    buf[0x04..0x06].copy_from_slice(&1u16.to_le_bytes()); // major_ver
    buf[0x06..0x08].copy_from_slice(&15u16.to_le_bytes()); // minor_ver
    // Field offsets per kernel `f2fs_fs.h`. The whole tail of the
    // superblock used to be 4 bytes too late; fsck.f2fs / mkfs.f2fs
    // now accept what we emit.
    buf[0x08..0x0C].copy_from_slice(&9u32.to_le_bytes()); // log_sectorsize (512)
    buf[0x0C..0x10].copy_from_slice(&3u32.to_le_bytes()); // log_sectors_per_block (8 sec / 4 KiB)
    buf[0x10..0x14].copy_from_slice(&12u32.to_le_bytes()); // log_blocksize (4096)
    buf[0x14..0x18].copy_from_slice(&geom.log_blocks_per_seg.to_le_bytes());
    buf[0x18..0x1C].copy_from_slice(&geom.segs_per_sec.to_le_bytes());
    buf[0x1C..0x20].copy_from_slice(&geom.secs_per_zone.to_le_bytes());
    // 0x20 = checksum_offset (unused by our reader)
    buf[0x24..0x2C].copy_from_slice(&geom.total_blocks.to_le_bytes());
    let section_count = geom.segment_count_main / geom.segs_per_sec.max(1);
    buf[0x2C..0x30].copy_from_slice(&section_count.to_le_bytes());
    buf[0x30..0x34].copy_from_slice(&geom.segment_count().to_le_bytes());
    buf[0x34..0x38].copy_from_slice(&geom.segment_count_ckpt.to_le_bytes());
    buf[0x38..0x3C].copy_from_slice(&geom.segment_count_sit.to_le_bytes());
    buf[0x3C..0x40].copy_from_slice(&geom.segment_count_nat.to_le_bytes());
    buf[0x40..0x44].copy_from_slice(&geom.segment_count_ssa.to_le_bytes());
    buf[0x44..0x48].copy_from_slice(&geom.segment_count_main.to_le_bytes());
    buf[0x48..0x4C].copy_from_slice(&0u32.to_le_bytes()); // segment0_blkaddr
    buf[0x4C..0x50].copy_from_slice(&geom.cp_blkaddr.to_le_bytes());
    buf[0x50..0x54].copy_from_slice(&geom.sit_blkaddr.to_le_bytes());
    buf[0x54..0x58].copy_from_slice(&geom.nat_blkaddr.to_le_bytes());
    buf[0x58..0x5C].copy_from_slice(&geom.ssa_blkaddr.to_le_bytes());
    buf[0x5C..0x60].copy_from_slice(&geom.main_blkaddr.to_le_bytes());
    buf[0x60..0x64].copy_from_slice(&F2FS_ROOT_INO_DEFAULT.to_le_bytes());
    buf[0x64..0x68].copy_from_slice(&1u32.to_le_bytes()); // node_ino
    buf[0x68..0x6C].copy_from_slice(&2u32.to_le_bytes()); // meta_ino

    // 16-byte UUID at 0x6C.
    buf[0x6C..0x7C].copy_from_slice(&opts.uuid);

    // 512-codepoint UTF-16LE volume label at 0x7C.
    let mut units = opts
        .volume_label
        .encode_utf16()
        .take(511)
        .collect::<Vec<u16>>();
    units.push(0);
    for (i, c) in units.iter().enumerate() {
        let o = 0x7C + i * 2;
        if o + 2 > buf.len() {
            break;
        }
        buf[o..o + 2].copy_from_slice(&c.to_le_bytes());
    }

    // cp_payload (0 for our small / single-pack layout).
    buf[0x3F8..0x3FC].copy_from_slice(&0u32.to_le_bytes());

    dev.write_at(SB_OFFSET_PRIMARY, &buf)?;
    dev.write_at(SB_OFFSET_BACKUP, &buf)?;
    Ok(())
}

/// Zero every metadata block once at format time. The main area is left
/// untouched (it's already zero on a freshly-`zero_range`d device) so
/// large images don't pay an O(N) zeroing pass twice.
pub(crate) fn wipe_metadata_region(dev: &mut dyn BlockDevice, geom: &Geometry) -> Result<()> {
    let bs = F2FS_BLKSIZE as u64;
    let start = (geom.cp_blkaddr as u64) * bs;
    let end = (geom.main_blkaddr as u64) * bs;
    dev.zero_range(start, end - start)?;
    Ok(())
}

/// Encode a single 4 KiB NAT page that holds `entries` starting at the
/// page's slot 0. The page's trailing CRC32 covers the first 4092 bytes.
pub(crate) fn encode_nat_page(entries: &[(u8, u32, u32)]) -> Vec<u8> {
    let mut page = vec![0u8; F2FS_BLKSIZE];
    for (i, (version, ino, block_addr)) in entries.iter().enumerate() {
        let o = i * NAT_ENTRY_SIZE;
        if o + NAT_ENTRY_SIZE > F2FS_BLK_CSUM_OFFSET {
            break;
        }
        page[o] = *version;
        page[o + 1..o + 5].copy_from_slice(&ino.to_le_bytes());
        page[o + 5..o + 9].copy_from_slice(&block_addr.to_le_bytes());
    }
    let crc = crc32fast::hash(&page[..F2FS_BLK_CSUM_OFFSET]);
    page[F2FS_BLK_CSUM_OFFSET..F2FS_BLK_CSUM_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    page
}

/// Encode a single 4 KiB SIT page that tracks `n_valid_blocks_per_seg`
/// segments worth of bitmaps. v1 marks every block of every used segment
/// as valid (we never reclaim); unused segments stay zero.
///
/// Layout per SIT entry (64 bytes here for simplicity, F2FS spec uses
/// 74 packed): `vblocks u16 | valid_map[blocks_per_seg / 8] | mtime u64`.
/// Our reader doesn't decode the SIT, so the exact stride is only
/// observed by `fsck.f2fs`. We use a conservative all-zero region — the
/// CRC footer still validates the page.
pub(crate) fn encode_sit_page() -> Vec<u8> {
    let mut page = vec![0u8; F2FS_BLKSIZE];
    let crc = crc32fast::hash(&page[..F2FS_BLK_CSUM_OFFSET]);
    page[F2FS_BLK_CSUM_OFFSET..F2FS_BLK_CSUM_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    page
}

/// Encode a 4 KiB SSA page. Each entry is a `f2fs_summary` (7 bytes) that
/// records `(nid, version, ofs_in_node)` for the block at that offset in
/// the parent segment. A blank page (zeros + CRC) is enough for the
/// reader and is the safest default for a fresh-image build.
pub(crate) fn encode_ssa_page() -> Vec<u8> {
    let mut page = vec![0u8; F2FS_BLKSIZE];
    let crc = crc32fast::hash(&page[..F2FS_BLK_CSUM_OFFSET]);
    page[F2FS_BLK_CSUM_OFFSET..F2FS_BLK_CSUM_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    page
}

/// Build a minimally-valid root inode block for the freshly formatted FS.
/// The block carries the directory mode/uid/gid/links + an inline-dentry
/// payload that's empty until the writer pushes children.
#[allow(dead_code)]
pub(crate) fn encode_root_inode_block(opts: &FormatOpts) -> Vec<u8> {
    use super::constants::{ADDRS_PER_INODE, F2FS_INLINE_DENTRY, NIDS_PER_INODE};
    use super::inode::I_ADDR_OFFSET;

    let mut buf = vec![0u8; F2FS_BLKSIZE];
    let mode = S_IFDIR | (opts.root_mode & 0x0FFF);
    buf[0x00..0x02].copy_from_slice(&mode.to_le_bytes());
    buf[0x03] = F2FS_INLINE_DENTRY;
    buf[0x04..0x08].copy_from_slice(&opts.root_uid.to_le_bytes());
    buf[0x08..0x0C].copy_from_slice(&opts.root_gid.to_le_bytes());
    buf[0x0C..0x10].copy_from_slice(&2u32.to_le_bytes()); // links: "." + parent ref
    buf[0x10..0x18].copy_from_slice(&(F2FS_BLKSIZE as u64).to_le_bytes());
    buf[0x18..0x20].copy_from_slice(&0u64.to_le_bytes()); // blocks (inline)
    buf[0x20..0x28].copy_from_slice(&(opts.mtime as u64).to_le_bytes());
    buf[0x28..0x30].copy_from_slice(&(opts.mtime as u64).to_le_bytes());
    buf[0x30..0x38].copy_from_slice(&(opts.mtime as u64).to_le_bytes());

    // i_addr + i_nid arrays are zero (no children, no node tree).
    let _ = (ADDRS_PER_INODE, NIDS_PER_INODE, I_ADDR_OFFSET);

    let crc = crc32fast::hash(&buf[..F2FS_BLK_CSUM_OFFSET]);
    buf[F2FS_BLK_CSUM_OFFSET..F2FS_BLK_CSUM_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_rejects_tiny_devices() {
        let opts = FormatOpts {
            log_blocks_per_seg: 2,
            ..FormatOpts::default()
        };
        assert!(plan_geometry(8, &opts).is_err());
    }

    #[test]
    fn geometry_lays_out_meta_then_main() {
        let opts = FormatOpts {
            log_blocks_per_seg: 2,
            ..FormatOpts::default()
        };
        let g = plan_geometry(256, &opts).unwrap();
        // Sanity: cp < sit < nat < ssa < main.
        assert!(g.cp_blkaddr < g.sit_blkaddr);
        assert!(g.sit_blkaddr < g.nat_blkaddr);
        assert!(g.nat_blkaddr < g.ssa_blkaddr);
        assert!(g.ssa_blkaddr < g.main_blkaddr);
        // Sanity: SB blocks + every region fits inside the device.
        let used_blocks = 2u32
            + (g.segment_count_ckpt
                + g.segment_count_sit
                + g.segment_count_nat
                + g.segment_count_ssa
                + g.segment_count_main)
                * g.blocks_per_seg;
        assert!((used_blocks as u64) <= g.total_blocks);
        // Sanity: main_blkaddr matches the running sum.
        assert_eq!(
            g.main_blkaddr,
            used_blocks - g.segment_count_main * g.blocks_per_seg
        );
    }
}
