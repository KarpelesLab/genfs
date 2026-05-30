//! exFAT volume formatter — emits the boot region (main + backup), an
//! empty FAT, the Allocation Bitmap, the Up-case Table, and a single
//! cluster for the root directory.
//!
//! ## Boot region (12 sectors, mirrored)
//!
//! ```text
//!   0       Main Boot Sector              (parsed by BootSector::decode)
//!   1..=8   Extended Boot Sectors         (trailing 0xAA55 signature)
//!   9       OEM Parameters                (all zero — none configured)
//!   10      Reserved                      (all zero)
//!   11      Main Boot Checksum            (32-bit checksum repeated)
//!   12..=23 Backup of sectors 0..=11
//! ```
//!
//! ## Boot checksum
//!
//! Per Microsoft's exFAT specification (BootChecksum) the 32-bit checksum
//! is computed over the contents of the 11 boot sectors that precede the
//! checksum sector, byte by byte, *skipping* offsets 106, 107, and 112 of
//! the main boot sector (these are VolumeFlags and PercentInUse — the
//! mount-time mutable fields).
//!
//! ```text
//!     checksum = ROR1(checksum) + byte    (32-bit, wraps)
//! ```
//!
//! Sector 11 is filled with the checksum repeated `bytes_per_sector / 4`
//! times in little-endian form.

use crate::Result;
use crate::block::BlockDevice;

use super::dir::{ATTR_DIRECTORY, ENTRY_SIZE};

/// Knobs for [`super::Exfat::format`]. All defaults yield a usable
/// volume; only `bytes_per_sector_shift`, `sectors_per_cluster_shift`,
/// and `volume_label` are typically of interest.
#[derive(Debug, Clone)]
pub struct FormatOpts {
    /// log2 of the sector size in bytes. Must be in 9..=12 (512..=4096).
    pub bytes_per_sector_shift: u8,
    /// log2 of the cluster size in sectors. The product
    /// `bytes_per_sector_shift + sectors_per_cluster_shift` must be <= 25.
    pub sectors_per_cluster_shift: u8,
    /// Volume serial number. Defaults to a deterministic constant so test
    /// images are reproducible; production callers should override.
    pub volume_serial_number: u32,
    /// Volume label. Empty string → no VolumeLabel entry in the root.
    pub volume_label: String,
}

impl Default for FormatOpts {
    fn default() -> Self {
        Self {
            // 512 B sectors, 4 KiB clusters: minimal viable defaults.
            bytes_per_sector_shift: 9,
            sectors_per_cluster_shift: 3,
            volume_serial_number: 0xCAFE_F00D,
            volume_label: String::new(),
        }
    }
}

impl FormatOpts {
    /// Pull exFAT-specific keys out of an
    /// [`OptionMap`](crate::format_opts::OptionMap). Recognised keys:
    ///
    /// - `bytes_per_sector_shift` (u8, 9..=12)
    /// - `sectors_per_cluster_shift` (u8)
    /// - `volume_serial_number` (u32, decimal or `0x…`)
    /// - `volume_label` (string, up to 11 UTF-16 code units; longer is
    ///   silently truncated by the formatter)
    pub fn apply_options(&mut self, map: &mut crate::format_opts::OptionMap) -> crate::Result<()> {
        if let Some(v) = map.take_u8("bytes_per_sector_shift")? {
            self.bytes_per_sector_shift = v;
        }
        if let Some(v) = map.take_u8("sectors_per_cluster_shift")? {
            self.sectors_per_cluster_shift = v;
        }
        if let Some(v) = map.take_u32("volume_serial_number")? {
            self.volume_serial_number = v;
        }
        if let Some(v) = map.take_str("volume_label") {
            self.volume_label = v;
        }
        Ok(())
    }
}

/// Geometry that the formatter computed from the device size and options.
/// Owned by [`super::Exfat`] after format so the writer doesn't have to
/// re-derive things.
#[derive(Debug, Clone)]
pub struct Geometry {
    pub bytes_per_sector: u32,
    pub sectors_per_cluster: u32,
    pub bytes_per_cluster: u32,
    pub volume_length_sectors: u64,
    pub fat_offset_sectors: u32,
    pub fat_length_sectors: u32,
    pub cluster_heap_offset_sectors: u32,
    pub cluster_count: u32,
    pub first_cluster_of_root_directory: u32,
    pub volume_serial_number: u32,
}

impl Geometry {
    pub fn fat_byte_offset(&self) -> u64 {
        self.fat_offset_sectors as u64 * self.bytes_per_sector as u64
    }

    pub fn fat_byte_length(&self) -> u64 {
        self.fat_length_sectors as u64 * self.bytes_per_sector as u64
    }

    pub fn cluster_heap_byte_offset(&self) -> u64 {
        self.cluster_heap_offset_sectors as u64 * self.bytes_per_sector as u64
    }

    pub fn cluster_byte_offset(&self, cluster: u32) -> u64 {
        self.cluster_heap_byte_offset() + (cluster as u64 - 2) * self.bytes_per_cluster as u64
    }
}

/// Compute volume geometry for a device of `total_bytes` bytes under `opts`.
///
/// Layout (in sectors):
///
/// ```text
///   0..24                     boot region (main + backup, 12 sectors each)
///   fat_offset..fat_offset+N  FAT (32-bit entries; we run a single FAT)
///   cluster_heap_offset..end  data clusters
/// ```
///
/// We reserve at least 24 sectors of head (boot region + slack), align
/// the FAT to a sector boundary, then align the cluster heap to a
/// **cluster** boundary so cluster reads/writes are sector-aligned.
pub fn compute_geometry(total_bytes: u64, opts: &FormatOpts) -> Result<Geometry> {
    if !(9..=12).contains(&opts.bytes_per_sector_shift) {
        return Err(crate::Error::InvalidArgument(format!(
            "exfat: bytes_per_sector_shift {} out of range (9..=12)",
            opts.bytes_per_sector_shift
        )));
    }
    if opts.bytes_per_sector_shift as u32 + opts.sectors_per_cluster_shift as u32 > 25 {
        return Err(crate::Error::InvalidArgument(format!(
            "exfat: bytes_per_sector_shift + sectors_per_cluster_shift = {} exceeds 25",
            opts.bytes_per_sector_shift as u32 + opts.sectors_per_cluster_shift as u32
        )));
    }
    let bytes_per_sector = 1u32 << opts.bytes_per_sector_shift;
    let sectors_per_cluster = 1u32 << opts.sectors_per_cluster_shift;
    let bytes_per_cluster = bytes_per_sector << opts.sectors_per_cluster_shift;
    if total_bytes < (1u64 << 20) {
        return Err(crate::Error::InvalidArgument(format!(
            "exfat: volume too small ({total_bytes} bytes; need at least 1 MiB)"
        )));
    }
    if !total_bytes.is_multiple_of(bytes_per_sector as u64) {
        return Err(crate::Error::InvalidArgument(format!(
            "exfat: total_bytes {total_bytes} is not a multiple of sector size {bytes_per_sector}"
        )));
    }
    let volume_length_sectors = total_bytes / bytes_per_sector as u64;
    if volume_length_sectors > u64::MAX / 2 {
        return Err(crate::Error::InvalidArgument(
            "exfat: VolumeLength sectors out of range".into(),
        ));
    }

    // Reserve 24 sectors for the boot region (main + backup), put the FAT
    // immediately after. The spec requires fat_offset >= 24.
    let fat_offset_sectors: u32 = 32; // a small margin past sector 23
    // First estimate the cluster count assuming a FAT that covers every
    // cluster in 4-byte entries. Solve:
    //   total = fat_offset + fat_len + align_to_cluster(cluster_count * spc)
    //   fat_len_bytes = (cluster_count + 2) * 4
    // We iterate twice — the FAT length depends on cluster_count, which
    // depends on cluster heap size, which depends on FAT length.
    let mut cluster_count_est: u64 = (volume_length_sectors / sectors_per_cluster as u64).max(1);
    let mut fat_length_sectors: u32 = 0;
    let mut cluster_heap_offset_sectors: u32 = 0;
    for _ in 0..8 {
        let fat_bytes = (cluster_count_est + 2).checked_mul(4).ok_or_else(|| {
            crate::Error::InvalidArgument("exfat: cluster count overflows FAT size".into())
        })?;
        let fat_secs = fat_bytes.div_ceil(bytes_per_sector as u64).max(1) as u32;
        // Cluster heap must be cluster-aligned within the volume.
        let after_fat = fat_offset_sectors as u64 + fat_secs as u64;
        let cluster_heap =
            after_fat.div_ceil(sectors_per_cluster as u64) * sectors_per_cluster as u64;
        if cluster_heap >= volume_length_sectors {
            return Err(crate::Error::InvalidArgument(
                "exfat: no space left for cluster heap".into(),
            ));
        }
        let data_sectors = volume_length_sectors - cluster_heap;
        let new_cluster_count = data_sectors / sectors_per_cluster as u64;
        if new_cluster_count < 1 {
            return Err(crate::Error::InvalidArgument(
                "exfat: device too small for one data cluster".into(),
            ));
        }
        fat_length_sectors = fat_secs;
        cluster_heap_offset_sectors = u32::try_from(cluster_heap).map_err(|_| {
            crate::Error::InvalidArgument("exfat: cluster heap offset overflows u32".into())
        })?;
        if new_cluster_count == cluster_count_est {
            break;
        }
        cluster_count_est = new_cluster_count;
    }
    let cluster_count = u32::try_from(cluster_count_est)
        .map_err(|_| crate::Error::InvalidArgument("exfat: cluster_count exceeds u32".into()))?;
    if cluster_count < 5 {
        // We need at least clusters for Bitmap + Upcase + Root + a bit of
        // slack to put a user file in. Fail loud rather than producing a
        // unusable volume.
        return Err(crate::Error::InvalidArgument(format!(
            "exfat: only {cluster_count} clusters available; need >= 5"
        )));
    }

    // We hand-pick the root cluster: 2 = bitmap, 3 = up-case, 4 = root.
    Ok(Geometry {
        bytes_per_sector,
        sectors_per_cluster,
        bytes_per_cluster,
        volume_length_sectors,
        fat_offset_sectors,
        fat_length_sectors,
        cluster_heap_offset_sectors,
        cluster_count,
        first_cluster_of_root_directory: 4,
        volume_serial_number: opts.volume_serial_number,
    })
}

/// Build the Main Boot Sector image (sector 0 of the boot region) for `geom`.
pub fn make_main_boot_sector(geom: &Geometry, sector_size: usize) -> Vec<u8> {
    let mut b = vec![0u8; sector_size];
    b[0..3].copy_from_slice(&[0xEB, 0x76, 0x90]);
    b[3..11].copy_from_slice(b"EXFAT   ");
    // 11..64 MustBeZero — already zero.
    b[64..72].copy_from_slice(&0u64.to_le_bytes()); // PartitionOffset
    b[72..80].copy_from_slice(&geom.volume_length_sectors.to_le_bytes());
    b[80..84].copy_from_slice(&geom.fat_offset_sectors.to_le_bytes());
    b[84..88].copy_from_slice(&geom.fat_length_sectors.to_le_bytes());
    b[88..92].copy_from_slice(&geom.cluster_heap_offset_sectors.to_le_bytes());
    b[92..96].copy_from_slice(&geom.cluster_count.to_le_bytes());
    b[96..100].copy_from_slice(&geom.first_cluster_of_root_directory.to_le_bytes());
    b[100..104].copy_from_slice(&geom.volume_serial_number.to_le_bytes());
    b[104..106].copy_from_slice(&0x0100u16.to_le_bytes()); // revision 1.0
    b[106..108].copy_from_slice(&0u16.to_le_bytes()); // VolumeFlags
    b[108] = log2_u32(geom.bytes_per_sector);
    b[109] = log2_u32(geom.sectors_per_cluster);
    b[110] = 1; // NumberOfFats
    b[111] = 0x80; // DriveSelect
    b[112] = 0; // PercentInUse (0xFF = unknown; 0 == 0% in use)
    // 113..119 Reserved (already zero)
    // 120..510 BootCode (already zero — bootable code is optional)
    b[sector_size - 2] = 0x55;
    b[sector_size - 1] = 0xAA;
    b
}

/// log2 of `n` (must be a power of 2). Panics in debug if not.
fn log2_u32(n: u32) -> u8 {
    debug_assert!(n.is_power_of_two());
    n.trailing_zeros() as u8
}

/// Build the extended boot sectors (8 sectors, all-zero except the
/// trailing `0xAA55` signature in the last 4 bytes of each).
pub fn make_extended_boot_sector(sector_size: usize) -> Vec<u8> {
    let mut b = vec![0u8; sector_size];
    // ExtendedBootSignature in the last 4 bytes is 0xAA550000 LE.
    let n = sector_size;
    b[n - 4..n].copy_from_slice(&0xAA55_0000u32.to_le_bytes());
    b
}

/// Compute the 32-bit BootChecksum over the first 11 sectors of the boot
/// region. Skips offsets 106, 107, 112 of sector 0 (VolumeFlags +
/// PercentInUse).
pub fn boot_checksum(sectors_0_to_10: &[u8], sector_size: usize) -> u32 {
    debug_assert_eq!(sectors_0_to_10.len(), 11 * sector_size);
    let mut sum: u32 = 0;
    for (i, &b) in sectors_0_to_10.iter().enumerate() {
        // Skip MainBootSector[106], [107], [112].
        if i == 106 || i == 107 || i == 112 {
            continue;
        }
        sum = sum.rotate_right(1).wrapping_add(b as u32);
    }
    sum
}

/// Fill sector 11 with `checksum` repeated bytes_per_sector/4 times in LE.
pub fn make_checksum_sector(checksum: u32, sector_size: usize) -> Vec<u8> {
    let mut out = vec![0u8; sector_size];
    for chunk in out.chunks_exact_mut(4) {
        chunk.copy_from_slice(&checksum.to_le_bytes());
    }
    out
}

/// Write the 12-sector boot region (sectors 0..=11) at `start_offset`
/// (in bytes). Returns the BootChecksum so caller can mirror to the
/// backup region.
pub fn write_boot_region(
    dev: &mut dyn BlockDevice,
    geom: &Geometry,
    start_offset: u64,
) -> Result<u32> {
    let ss = geom.bytes_per_sector as usize;
    let main = make_main_boot_sector(geom, ss);
    let ext = make_extended_boot_sector(ss);
    // OEM Parameters (sector 9): all zero is valid (no parameters).
    let oem = vec![0u8; ss];
    // Reserved (sector 10): all zero.
    let reserved = vec![0u8; ss];

    // Concatenate sectors 0..=10 to compute the checksum.
    let mut first_eleven = Vec::with_capacity(11 * ss);
    first_eleven.extend_from_slice(&main);
    for _ in 0..8 {
        first_eleven.extend_from_slice(&ext);
    }
    first_eleven.extend_from_slice(&oem);
    first_eleven.extend_from_slice(&reserved);
    let checksum = boot_checksum(&first_eleven, ss);
    let csum_sector = make_checksum_sector(checksum, ss);

    // Write 12 sectors starting at `start_offset`.
    dev.write_at(start_offset, &first_eleven)?;
    dev.write_at(start_offset + 11 * ss as u64, &csum_sector)?;
    Ok(checksum)
}

/// Emit a simple Up-case Table: lowercase ASCII a..z → A..Z, every other
/// BMP code unit identity. Returns the uncompressed table bytes along
/// with its 32-bit table checksum (per the rolling checksum in `upcase.rs`).
pub fn make_ascii_upcase_table() -> (Vec<u8>, u32) {
    // We emit the first 128 explicit slots (covers ASCII), then an
    // identity-run for the remainder of the BMP would balloon the table
    // unnecessarily — readers default to identity beyond the populated
    // range. We mirror what the reader's `Upcase::ascii()` produces so
    // round-tripping is exact.
    let mut bytes = Vec::with_capacity(0x80 * 2);
    for i in 0..0x80u16 {
        let c = i as u8;
        let v = if c.is_ascii_lowercase() {
            (c - b'a' + b'A') as u16
        } else {
            i
        };
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let csum = super::upcase::table_checksum(&bytes);
    (bytes, csum)
}

/// Build an Allocation Bitmap directory-entry slot (32 bytes).
pub fn make_bitmap_entry(first_cluster: u32, data_length: u64) -> [u8; ENTRY_SIZE] {
    let mut e = [0u8; ENTRY_SIZE];
    e[0] = super::dir::ENTRY_ALLOCATION_BITMAP;
    e[1] = 0; // BitmapFlags: bit 0 = which-bitmap (0 = first).
    // 2..20 reserved
    e[20..24].copy_from_slice(&first_cluster.to_le_bytes());
    e[24..32].copy_from_slice(&data_length.to_le_bytes());
    e
}

/// Build an Up-case Table directory-entry slot (32 bytes).
pub fn make_upcase_entry(checksum: u32, first_cluster: u32, data_length: u64) -> [u8; ENTRY_SIZE] {
    let mut e = [0u8; ENTRY_SIZE];
    e[0] = super::dir::ENTRY_UPCASE_TABLE;
    e[4..8].copy_from_slice(&checksum.to_le_bytes());
    e[20..24].copy_from_slice(&first_cluster.to_le_bytes());
    e[24..32].copy_from_slice(&data_length.to_le_bytes());
    e
}

/// Build a VolumeLabel directory-entry slot (32 bytes). Truncates to 11
/// UTF-16 code units per spec.
pub fn make_volume_label_entry(label: &str) -> [u8; ENTRY_SIZE] {
    let mut units: Vec<u16> = label.encode_utf16().collect();
    units.truncate(11);
    let mut e = [0u8; ENTRY_SIZE];
    e[0] = super::dir::ENTRY_VOLUME_LABEL;
    e[1] = units.len() as u8;
    for (i, u) in units.iter().enumerate() {
        let off = 2 + i * 2;
        e[off..off + 2].copy_from_slice(&u.to_le_bytes());
    }
    e
}

/// Build a FileDirectoryEntry + StreamExtension + FileName entry set for
/// a file or directory with the given metadata. The SetChecksum is
/// computed and written into the primary entry. Returns the assembled
/// bytes (multiple of 32).
#[allow(clippy::too_many_arguments)]
pub fn make_file_entry_set(
    name: &str,
    is_directory: bool,
    secondary_flags: u8,
    first_cluster: u32,
    data_length: u64,
    valid_data_length: u64,
    create_timestamp: u32,
    name_hash: u16,
) -> Vec<u8> {
    let name_units: Vec<u16> = name.encode_utf16().collect();
    let n_name_entries = name_units.len().div_ceil(15).max(1);
    let secondary_count = (1 + n_name_entries) as u8;
    let attr = if is_directory { ATTR_DIRECTORY } else { 0 };

    let mut primary = [0u8; ENTRY_SIZE];
    primary[0] = super::dir::ENTRY_FILE;
    primary[1] = secondary_count;
    // 2..4 SetChecksum — filled below.
    primary[4..6].copy_from_slice(&attr.to_le_bytes());
    primary[8..12].copy_from_slice(&create_timestamp.to_le_bytes());
    primary[12..16].copy_from_slice(&create_timestamp.to_le_bytes());
    primary[16..20].copy_from_slice(&create_timestamp.to_le_bytes());

    let mut stream = [0u8; ENTRY_SIZE];
    stream[0] = super::dir::ENTRY_STREAM_EXTENSION;
    stream[1] = secondary_flags;
    stream[3] = name_units.len() as u8;
    stream[4..6].copy_from_slice(&name_hash.to_le_bytes());
    stream[8..16].copy_from_slice(&valid_data_length.to_le_bytes());
    stream[20..24].copy_from_slice(&first_cluster.to_le_bytes());
    stream[24..32].copy_from_slice(&data_length.to_le_bytes());

    let mut out = Vec::with_capacity((1 + secondary_count as usize) * ENTRY_SIZE);
    out.extend_from_slice(&primary);
    out.extend_from_slice(&stream);
    for chunk in name_units.chunks(15) {
        let mut e = [0u8; ENTRY_SIZE];
        e[0] = super::dir::ENTRY_FILE_NAME;
        for (i, &u) in chunk.iter().enumerate() {
            let off = 2 + i * 2;
            e[off..off + 2].copy_from_slice(&u.to_le_bytes());
        }
        out.extend_from_slice(&e);
    }
    if name_units.is_empty() {
        // Guarantee at least one FileName entry so secondary_count == 2.
        let e = [0u8; ENTRY_SIZE];
        out.extend_from_slice(&e);
        let _ = e;
    }

    // Compute and patch SetChecksum.
    let csum = super::dir::set_checksum(&out);
    out[2..4].copy_from_slice(&csum.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boot_checksum_skip_fields() {
        // Two single-sector buffers that differ only in the skipped
        // offsets must produce the same checksum.
        let ss = 512;
        let mut a = vec![0u8; 11 * ss];
        a[0..3].copy_from_slice(&[0xEB, 0x76, 0x90]);
        a[3..11].copy_from_slice(b"EXFAT   ");
        let mut b = a.clone();
        b[106] = 0xFF;
        b[107] = 0xFF;
        b[112] = 0xAB;
        assert_eq!(boot_checksum(&a, ss), boot_checksum(&b, ss));
    }

    #[test]
    fn geometry_fits_simple_volume() {
        // 4 MiB volume, 512 B sectors, 4 KiB clusters.
        let opts = FormatOpts::default();
        let g = compute_geometry(4 * 1024 * 1024, &opts).unwrap();
        assert_eq!(g.bytes_per_sector, 512);
        assert_eq!(g.bytes_per_cluster, 4096);
        assert!(g.fat_offset_sectors >= 24);
        assert!(g.cluster_heap_offset_sectors as u64 > g.fat_offset_sectors as u64);
        // Cluster heap must be cluster-aligned within the volume.
        assert_eq!(
            g.cluster_heap_offset_sectors as u32 % g.sectors_per_cluster,
            0
        );
        assert!(g.cluster_count >= 5);
    }

    #[test]
    fn ascii_upcase_round_trip() {
        let (bytes, _) = make_ascii_upcase_table();
        let uc = super::super::upcase::Upcase::decode(&bytes, bytes.len() as u64).unwrap();
        assert_eq!(uc.up(b'a' as u16), b'A' as u16);
        assert_eq!(uc.up(b'Z' as u16), b'Z' as u16);
    }
}
