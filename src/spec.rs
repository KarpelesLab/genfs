//! TOML image-specification schema and the `build` driver.
//!
//! A spec describes *what to put in an image* declaratively. Two shapes:
//!
//! 1. **Bare filesystem** — a top-level `[filesystem]` table. The whole
//!    image is one ext2/3/4 filesystem with no partition table. This is
//!    the `genext2fs` replacement.
//!
//!    ```toml
//!    [filesystem]
//!    type = "ext4"
//!    source = "./rootfs"
//!    block_size = 1024
//!    rootdevs = "minimal"
//!    ```
//!
//! 2. **Partitioned disk image** — an `[image]` table plus `[[partitions]]`
//!    entries. Partitions are laid out 1 MiB-aligned in declaration order;
//!    exactly one may use `size = "remaining"` (and it must be last). Each
//!    partition optionally carries a nested `[partitions.filesystem]` table
//!    whose ext filesystem is formatted to fill the partition.
//!
//! Sizes accept a human suffix: `512`, `4KiB`, `256MiB`, `1GiB`, `2GB`
//! (decimal `KB/MB/GB` and binary `KiB/MiB/GiB`; bare numbers are bytes).

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::ext::{Ext, FsKind};
use crate::fs::rootdevs::RootDevs;

/// Top-level parsed spec. Exactly one of `filesystem` / `image` must be set.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// Bare-filesystem mode: the entire image is this one filesystem.
    pub filesystem: Option<FilesystemSpec>,
    /// Partitioned-disk mode header. Requires `partitions` to be non-empty.
    pub image: Option<ImageSpec>,
    /// Partition entries (only meaningful alongside `image`).
    #[serde(default)]
    pub partitions: Vec<PartitionSpec>,
}

/// Disk-level options for a partitioned image.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageSpec {
    /// Total disk size, e.g. `"1GiB"`.
    pub size: String,
    /// `"gpt"` or `"mbr"`.
    pub partition_table: String,
}

/// One partition in a [`Spec::partitions`] list.
///
/// The optional filesystem goes in a nested `[partitions.filesystem]`
/// table (not flattened) so the partition's own `type` key doesn't collide
/// with the filesystem's `type` key:
///
/// ```toml
/// [[partitions]]
/// name = "root"
/// type = "linux"
/// size = "remaining"
///
/// [partitions.filesystem]
/// type = "ext4"
/// source = "./rootfs"
/// ```
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionSpec {
    pub name: Option<String>,
    /// Partition type: `"esp"`, `"linux"`, `"linux-swap"`, `"bios-boot"`,
    /// `"fat"`, or a raw `"0x83"` / GPT UUID string.
    #[serde(rename = "type")]
    pub kind: String,
    /// Partition size, e.g. `"256MiB"` or `"remaining"`.
    pub size: String,
    /// Filesystem to create inside this partition (optional — a partition
    /// can be left raw).
    pub filesystem: Option<FilesystemSpec>,
}

/// Filesystem configuration. Used both as the top-level `[filesystem]`
/// table (bare-FS mode) and as the nested `[partitions.filesystem]` table.
///
/// Marked `#[non_exhaustive]` because the legacy flat tunables (`block_size`,
/// `volume_label`, `mtime`, …) are progressively being replaced by the
/// free-form [`Self::options`] table — new FS-specific knobs land there
/// without needing a new flat field, but the door is kept open for either
/// route. External crates should construct `FilesystemSpec` only through
/// `toml::from_str` (or one of the [`Spec`] parse helpers), not via a
/// struct literal.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct FilesystemSpec {
    /// `"ext2"`, `"ext3"`, `"ext4"`, or `"fat32"`.
    #[serde(rename = "type")]
    pub fs_type: String,
    /// Host directory whose contents become the filesystem's tree.
    /// Omitted → an empty filesystem (just `/` and `/lost+found`).
    pub source: Option<PathBuf>,
    /// FS block size in bytes (1024 / 2048 / 4096). Default 1024. (ext only.)
    pub block_size: Option<u32>,
    /// Journal size in blocks (ext3/ext4 only). Default 1024.
    pub journal_blocks: Option<u32>,
    /// `"none"`, `"minimal"`, or `"standard"` — pre-populate `/dev`. (ext only.)
    pub rootdevs: Option<String>,
    /// Volume label (≤ 16 bytes for ext, ≤ 11 bytes for FAT32).
    pub volume_label: Option<String>,
    /// Modification timestamp baked into every inode (seconds since epoch).
    /// Default 0 for reproducible output. (ext only.)
    pub mtime: Option<u32>,
    /// When true, regular files are written sparsely — all-zero blocks
    /// become holes instead of consuming data blocks. Default false. (ext only.)
    pub sparse: Option<bool>,
    /// Explicit filesystem size, e.g. `"64MiB"`. Required for FAT32 in
    /// bare-FS mode (FAT32 has a ~33 MiB minimum and no streaming auto-size).
    /// Ignored when the FS sits in a partition (the partition size wins).
    pub size: Option<String>,
    /// FAT32 volume ID / serial number. Default 0 for reproducible output.
    pub volume_id: Option<u32>,
    /// Filesystem-specific options, as a free-form TOML table. Keyed by
    /// the same names accepted by the CLI's `-O key=val` flag and by
    /// each backend's `FormatOpts::apply_options`. Values must be
    /// scalars (string / int / bool / float). The legacy flat fields
    /// above are still honoured and pre-loaded into the same bag —
    /// `options` lets you reach knobs that don't have a dedicated flat
    /// field (e.g. squashfs `compression`, ntfs `bytes_per_sector`,
    /// hfs+ `journaled`).
    pub options: Option<toml::Table>,
}

impl Spec {
    /// Parse a spec from TOML text.
    pub fn parse(toml_text: &str) -> Result<Self> {
        let spec: Spec = toml::from_str(toml_text)
            .map_err(|e| crate::Error::InvalidArgument(format!("spec: invalid TOML: {e}")))?;
        spec.validate()?;
        Ok(spec)
    }

    /// Parse a spec from a file on disk.
    pub fn parse_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text)
    }

    fn validate(&self) -> Result<()> {
        match (&self.filesystem, &self.image) {
            (Some(_), Some(_)) => Err(crate::Error::InvalidArgument(
                "spec: cannot set both [filesystem] and [image] — pick bare-FS or partitioned"
                    .into(),
            )),
            (None, None) => Err(crate::Error::InvalidArgument(
                "spec: must set either [filesystem] (bare FS) or [image] + [[partitions]]".into(),
            )),
            (Some(_), None) => {
                if !self.partitions.is_empty() {
                    return Err(crate::Error::InvalidArgument(
                        "spec: [[partitions]] is meaningless without [image]".into(),
                    ));
                }
                Ok(())
            }
            (None, Some(_)) => {
                if self.partitions.is_empty() {
                    return Err(crate::Error::InvalidArgument(
                        "spec: [image] requires at least one [[partitions]] entry".into(),
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Build the image described by `spec` into the file at `output`.
pub fn build(spec: &Spec, output: &Path) -> Result<()> {
    if let Some(fs) = &spec.filesystem {
        build_bare_fs(fs, output)
    } else if let Some(image) = &spec.image {
        build_partitioned(image, &spec.partitions, output)
    } else {
        // validate() guarantees one of the two is set.
        unreachable!("Spec::validate ensures filesystem xor image")
    }
}

fn build_bare_fs(fs: &FilesystemSpec, output: &Path) -> Result<()> {
    match fs.fs_type.to_ascii_lowercase().as_str() {
        "ext2" | "ext3" | "ext4" => build_bare_ext(fs, output),
        "fat32" | "vfat" => build_bare_fat32(fs, output),
        "hfsplus" | "hfs+" => build_bare_via_trait::<crate::fs::hfs_plus::HfsPlus>(
            fs,
            output,
            hfs_plus_format_opts(fs)?,
        ),
        "ntfs" => build_bare_via_trait::<crate::fs::ntfs::Ntfs>(fs, output, ntfs_format_opts(fs)?),
        "f2fs" => build_bare_via_trait::<crate::fs::f2fs::F2fs>(fs, output, f2fs_format_opts(fs)?),
        "squashfs" => build_bare_via_trait::<crate::fs::squashfs::Squashfs>(
            fs,
            output,
            squashfs_format_opts(fs)?,
        ),
        "xfs" => build_bare_via_trait::<crate::fs::xfs::Xfs>(fs, output, xfs_format_opts(fs)?),
        "iso" | "iso9660" => build_bare_via_trait::<crate::fs::iso9660::Iso9660>(
            fs,
            output,
            iso9660_format_opts(fs)?,
        ),
        "grf" => build_bare_via_trait::<crate::fs::grf::Grf>(fs, output, grf_format_opts(fs)?),
        "zip" => {
            build_bare_via_trait::<crate::fs::archive::zip::ZipFs>(fs, output, zip_format_opts(fs)?)
        }
        "cpio" => build_bare_via_trait::<crate::fs::archive::cpio::CpioFs>(
            fs,
            output,
            cpio_format_opts(fs)?,
        ),
        "ar" => {
            build_bare_via_trait::<crate::fs::archive::ar::ArFs>(fs, output, ar_format_opts(fs)?)
        }
        other => Err(crate::Error::InvalidArgument(format!(
            "spec: unknown filesystem type {other:?}"
        ))),
    }
}

fn zip_format_opts(fs: &FilesystemSpec) -> Result<crate::fs::archive::zip::ZipFormatOpts> {
    let mut bag = options_bag_for(fs)?;
    // `volume_label` is meaningless for archives; drop it so check_empty
    // doesn't reject a spec that set it generically.
    let _ = bag.take_str("volume_label");
    let mut opts = crate::fs::archive::zip::ZipFormatOpts::default();
    opts.apply_options(&mut bag)?;
    bag.check_empty("zip")?;
    Ok(opts)
}

fn cpio_format_opts(fs: &FilesystemSpec) -> Result<crate::fs::archive::cpio::CpioFormatOpts> {
    let mut bag = options_bag_for(fs)?;
    let _ = bag.take_str("volume_label");
    let mut opts = crate::fs::archive::cpio::CpioFormatOpts;
    opts.apply_options(&mut bag)?;
    bag.check_empty("cpio")?;
    Ok(opts)
}

fn ar_format_opts(fs: &FilesystemSpec) -> Result<crate::fs::archive::ar::ArFormatOpts> {
    let mut bag = options_bag_for(fs)?;
    let _ = bag.take_str("volume_label");
    let mut opts = crate::fs::archive::ar::ArFormatOpts;
    opts.apply_options(&mut bag)?;
    bag.check_empty("ar")?;
    Ok(opts)
}

/// Build a fresh [`OptionMap`] from the FS spec for use by a backend's
/// `FormatOpts::apply_options`. Pre-loads `volume_label` from the flat
/// legacy field (other flat fields are FS-specific and live on the
/// respective helpers), then merges the optional `[filesystem.options]`
/// table on top. Caller is responsible for `check_empty` after taking
/// the keys it recognises.
fn options_bag_for(fs: &FilesystemSpec) -> Result<crate::format_opts::OptionMap> {
    let mut bag = crate::format_opts::OptionMap::new();
    if let Some(label) = &fs.volume_label {
        bag.insert("volume_label", label);
    }
    if let Some(table) = &fs.options {
        bag.merge_toml(table)?;
    }
    Ok(bag)
}

fn grf_format_opts(fs: &FilesystemSpec) -> Result<crate::fs::grf::FormatOpts> {
    let mut bag = options_bag_for(fs)?;
    let mut opts = crate::fs::grf::FormatOpts::default();
    opts.apply_options(&mut bag)?;
    bag.check_empty("grf")?;
    Ok(opts)
}

fn iso9660_format_opts(fs: &FilesystemSpec) -> Result<crate::fs::iso9660::FormatOpts> {
    let mut bag = options_bag_for(fs)?;
    let mut opts = crate::fs::iso9660::FormatOpts {
        volume_id: "FSTOOL".into(),
        application_id: "fstool".into(),
        ..crate::fs::iso9660::FormatOpts::default()
    };
    opts.apply_options(&mut bag)?;
    bag.check_empty("iso9660")?;
    Ok(opts)
}

/// Generic "create the backing file, format `F`, populate from
/// `source`, flush" pipeline. Used for every writable FS whose
/// destination size doesn't have to be derived from the source tree
/// (we accept an explicit `size` in TOML, defaulting to 256 MiB).
fn build_bare_via_trait<F: crate::fs::FilesystemFactory>(
    fs: &FilesystemSpec,
    output: &Path,
    opts: F::FormatOpts,
) -> Result<()> {
    let bytes = match fs.size.as_deref() {
        Some(s) => parse_size(s)?,
        None => 256 * 1024 * 1024,
    };
    let mut dev = crate::block::create_image(output, bytes, &crate::block::CreateOpts::default())?;
    let mut fs_obj = F::format(dev.as_mut(), &opts)?;
    if let Some(src) = &fs.source {
        let source = source_from_spec(src)?;
        crate::repack::populate_fs_from_source(dev.as_mut(), &mut fs_obj, &source)?;
    }
    fs_obj.flush(dev.as_mut())?;
    dev.sync()?;
    // Archive writers (zip/cpio/ar) report their exact length; truncate
    // the over-provisioned sparse file down to it. Filesystem images
    // return `None` and keep their provisioned size.
    let archive_len = crate::fs::Filesystem::image_len(&fs_obj);
    drop(fs_obj);
    drop(dev);
    if let Some(len) = archive_len {
        truncate_archive_file(output, len)?;
    }
    Ok(())
}

/// Shrink an over-provisioned archive output file to its true length.
/// No-op for block devices and qcow2 containers.
fn truncate_archive_file(output: &Path, len: u64) -> Result<()> {
    if crate::block::file::is_block_device(output)
        || output
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("qcow2"))
    {
        return Ok(());
    }
    let f = std::fs::OpenOptions::new().write(true).open(output)?;
    f.set_len(len)?;
    Ok(())
}

/// In-partition variant of [`build_bare_via_trait`]: takes an already
/// pre-sliced `dev` (just one partition's view) and formats `F` into
/// it, optionally populating from `source`.
fn format_in_partition_via_trait<F: crate::fs::FilesystemFactory>(
    dev: &mut dyn BlockDevice,
    fs: &FilesystemSpec,
    opts: F::FormatOpts,
) -> Result<()> {
    let mut fs_obj = F::format(dev, &opts)?;
    if let Some(src) = &fs.source {
        let source = source_from_spec(src)?;
        crate::repack::populate_fs_from_source(dev, &mut fs_obj, &source)?;
    }
    fs_obj.flush(dev)?;
    Ok(())
}

fn hfs_plus_format_opts(fs: &FilesystemSpec) -> Result<crate::fs::hfs_plus::FormatOpts> {
    let mut bag = options_bag_for(fs)?;
    let mut opts = crate::fs::hfs_plus::FormatOpts {
        volume_name: "Untitled".to_string(),
        ..crate::fs::hfs_plus::FormatOpts::default()
    };
    opts.apply_options(&mut bag)?;
    bag.check_empty("hfs+")?;
    Ok(opts)
}

fn ntfs_format_opts(fs: &FilesystemSpec) -> Result<crate::fs::ntfs::format::FormatOpts> {
    let mut bag = options_bag_for(fs)?;
    let mut opts = crate::fs::ntfs::format::FormatOpts::default();
    opts.apply_options(&mut bag)?;
    bag.check_empty("ntfs")?;
    Ok(opts)
}

fn f2fs_format_opts(fs: &FilesystemSpec) -> Result<crate::fs::f2fs::FormatOpts> {
    let mut bag = options_bag_for(fs)?;
    let mut opts = crate::fs::f2fs::FormatOpts::default();
    opts.apply_options(&mut bag)?;
    bag.check_empty("f2fs")?;
    Ok(opts)
}

fn squashfs_format_opts(fs: &FilesystemSpec) -> Result<crate::fs::squashfs::FormatOpts> {
    let mut bag = options_bag_for(fs)?;
    let mut opts = crate::fs::squashfs::FormatOpts {
        block_size: fs
            .block_size
            .unwrap_or(crate::fs::squashfs::DEFAULT_BLOCK_SIZE),
        ..crate::fs::squashfs::FormatOpts::default()
    };
    opts.apply_options(&mut bag)?;
    bag.check_empty("squashfs")?;
    Ok(opts)
}

fn xfs_format_opts(fs: &FilesystemSpec) -> Result<crate::fs::xfs::format::FormatOpts> {
    let mut bag = options_bag_for(fs)?;
    let mut opts = crate::fs::xfs::format::FormatOpts::default();
    opts.apply_options(&mut bag)?;
    bag.check_empty("xfs")?;
    Ok(opts)
}

fn build_bare_ext(fs: &FilesystemSpec, output: &Path) -> Result<()> {
    let kind = parse_fs_kind(&fs.fs_type)?;
    let block_size = fs.block_size.unwrap_or(1024);
    let opts = ext_format_opts(fs, kind, block_size, None)?;
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = crate::block::create_image(output, size, &crate::block::CreateOpts::default())?;
    format_ext_into(dev.as_mut(), fs, &opts)?;
    dev.sync()?;
    Ok(())
}

fn build_bare_fat32(fs: &FilesystemSpec, output: &Path) -> Result<()> {
    // Sizing precedence: explicit `size` → use as-is. Otherwise size
    // from the source (directory / tar archive / image walk). FAT32
    // has a ~33 MiB minimum we have to honour; the sizing helper
    // already floors at that.
    let bytes = match (fs.size.as_deref(), &fs.source) {
        (Some(s), _) => parse_size(s)?,
        (None, Some(src)) => {
            let source = source_from_spec(src)?;
            crate::repack::fat32_min_bytes_for_source(&source)?
        }
        (None, None) => {
            return Err(crate::Error::InvalidArgument(
                "spec: FAT32 needs either `size` or `source` (no streaming auto-size; minimum ~33 MiB)".into(),
            ));
        }
    };
    let total_sectors: u32 = (bytes / SECTOR).try_into().map_err(|_| {
        crate::Error::InvalidArgument(
            "spec: FAT32 image size doesn't fit in a u32 sector count".into(),
        )
    })?;
    let label = fat32_volume_label(fs.volume_label.as_deref());
    let volume_id = fs.volume_id.unwrap_or(0);
    let mut dev = crate::block::create_image(output, bytes, &crate::block::CreateOpts::default())?;
    format_fat32_into(dev.as_mut(), fs, total_sectors, volume_id, label)?;
    dev.sync()?;
    Ok(())
}

fn format_fat32_into(
    dev: &mut dyn BlockDevice,
    fs: &FilesystemSpec,
    total_sectors: u32,
    volume_id: u32,
    label: [u8; 11],
) -> Result<()> {
    use crate::fs::fat::Fat32;
    let opts = crate::fs::fat::FatFormatOpts {
        total_sectors,
        volume_id,
        volume_label: label,
    };
    let mut fat = Fat32::format(dev, &opts)?;
    if let Some(src) = &fs.source {
        let source = source_from_spec(src)?;
        crate::repack::populate_fat32_from_source(dev, &mut fat, &source)?;
    }
    fat.flush(dev)?;
    Ok(())
}

/// Convert a `FilesystemSpec.source` value into a [`crate::repack::Source`].
/// The path is interpreted the same way [`crate::repack::Source::detect`]
/// interprets a CLI argument: a directory becomes `HostDir`, a tar
/// archive (by extension) becomes `TarArchive`, and any other string
/// — including the `disk.img:N` partition selector — falls through
/// to `Image`.
fn source_from_spec(src: &Path) -> Result<crate::repack::Source> {
    let s = src.to_str().ok_or_else(|| {
        crate::Error::InvalidArgument(format!("spec: source path {src:?} is not valid UTF-8"))
    })?;
    crate::repack::Source::detect(s)
}

/// Space-pad and truncate `label` to exactly 11 bytes for FAT32. ASCII only;
/// non-ASCII bytes are replaced with `_` (FAT32 short labels are OEM-encoded;
/// we don't try to translate code pages).
fn fat32_volume_label(label: Option<&str>) -> [u8; 11] {
    let mut out = [b' '; 11];
    let Some(s) = label else {
        return *b"NO NAME    ";
    };
    let upper = s.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    for (i, &b) in bytes.iter().take(11).enumerate() {
        out[i] = if b.is_ascii() && b >= 0x20 && b != 0x7F {
            b
        } else {
            b'_'
        };
    }
    out
}

/// Build the [`crate::fs::ext::FormatOpts`] for a [`FilesystemSpec`].
/// `blocks_count_override` forces a specific block count (used to make a
/// partition's filesystem fill the partition exactly); `None` auto-sizes
/// from the source tree.
fn ext_format_opts(
    fs: &FilesystemSpec,
    kind: FsKind,
    block_size: u32,
    blocks_count_override: Option<u32>,
) -> Result<crate::fs::ext::FormatOpts> {
    let rootdevs = parse_rootdevs(fs.rootdevs.as_deref())?;
    let mtime = fs.mtime.unwrap_or(0);

    let mut plan = match &fs.source {
        Some(src) => {
            let source = source_from_spec(src)?;
            crate::repack::ext_build_plan_for_source(&source, block_size, kind)?
        }
        None => crate::fs::ext::BuildPlan::new(block_size, kind),
    };
    if let Some(j) = fs.journal_blocks {
        plan.journal_blocks = j;
    }
    for _ in 0..rootdevs_entry_count(rootdevs) {
        plan.add_device();
    }
    if rootdevs != RootDevs::None {
        plan.add_dir(); // the /dev directory itself
    }

    let mut opts = plan.to_format_opts();
    opts.mtime = mtime;
    opts.sparse = fs.sparse.unwrap_or(false);
    if let Some(label) = &fs.volume_label {
        let bytes = label.as_bytes();
        let n = bytes.len().min(16);
        opts.volume_label[..n].copy_from_slice(&bytes[..n]);
    }
    if let Some(bc) = blocks_count_override {
        if bc < opts.blocks_count {
            return Err(crate::Error::InvalidArgument(format!(
                "spec: partition holds {bc} blocks but its contents need at least {}",
                opts.blocks_count
            )));
        }
        opts.blocks_count = bc;
        // The auto-sized inode count was computed for the source tree only.
        // When filling a (potentially much larger) partition, scale up to
        // mke2fs's default density of one inode per 16 KiB so the partition
        // isn't inode-starved.
        let by_density = (bc as u64 * block_size as u64 / 16_384) as u32;
        opts.inodes_count = opts.inodes_count.max(by_density);
    }
    Ok(opts)
}

/// Format + populate an ext filesystem into `dev` from `fs`.
fn format_ext_into(
    dev: &mut dyn BlockDevice,
    fs: &FilesystemSpec,
    opts: &crate::fs::ext::FormatOpts,
) -> Result<()> {
    let rootdevs = parse_rootdevs(fs.rootdevs.as_deref())?;
    // Spec builds always run on a device that was just created (raw
    // file / qcow2 / sliced partition of one), which reads back as
    // zero — skip `format_with`'s full-device zero pass.
    let opts = crate::fs::ext::FormatOpts {
        prezeroed: true,
        ..opts.clone()
    };
    let mut ext = Ext::format_with(dev, &opts)?;
    if let Some(src) = &fs.source {
        let source = source_from_spec(src)?;
        crate::repack::populate_ext_from_source(dev, &mut ext, &source)?;
    }
    if rootdevs != RootDevs::None {
        ext.populate_rootdevs(dev, rootdevs, 0, 0, opts.mtime)?;
    }
    ext.flush(dev)?;
    Ok(())
}

/// Logical sector size for partition-table geometry. Both MBR and our v1
/// GPT writer assume 512-byte sectors.
const SECTOR: u64 = 512;
/// Partition-start alignment: 1 MiB, the modern convention.
const ALIGN_LBA: u64 = 2048;

fn build_partitioned(image: &ImageSpec, partitions: &[PartitionSpec], output: &Path) -> Result<()> {
    use crate::part::{Gpt, Mbr, Partition, PartitionTable, slice_partition};

    let total_bytes = parse_size(&image.size)?;
    let total_lba = total_bytes / SECTOR;
    let table = image.partition_table.to_ascii_lowercase();

    // Reserve space for the partition table's own metadata.
    //   GPT: LBA 0 protective MBR + 1..33 primary + last 33 backup.
    //   MBR: just LBA 0.
    let (first_free, last_usable) = match table.as_str() {
        "gpt" => (ALIGN_LBA, total_lba.saturating_sub(34)),
        "mbr" => (ALIGN_LBA, total_lba.saturating_sub(1)),
        other => {
            return Err(crate::Error::InvalidArgument(format!(
                "spec: unknown partition_table {other:?} (want gpt or mbr)"
            )));
        }
    };
    if table == "mbr" && partitions.len() > 4 {
        return Err(crate::Error::InvalidArgument(
            "spec: MBR supports at most 4 partitions".into(),
        ));
    }

    // Lay out partitions sequentially. Exactly one may use size
    // "remaining", and it must be the last entry.
    let remaining_idx = partitions
        .iter()
        .position(|p| p.size.eq_ignore_ascii_case("remaining"));
    if let Some(idx) = remaining_idx
        && idx != partitions.len() - 1
    {
        return Err(crate::Error::InvalidArgument(
            "spec: size = \"remaining\" is only allowed on the last partition".into(),
        ));
    }

    let mut placed: Vec<Partition> = Vec::with_capacity(partitions.len());
    let mut cursor = first_free;
    for p in partitions {
        let start = cursor.div_ceil(ALIGN_LBA) * ALIGN_LBA;
        let size_lba = if p.size.eq_ignore_ascii_case("remaining") {
            if last_usable < start {
                return Err(crate::Error::InvalidArgument(
                    "spec: no space left for the \"remaining\" partition".into(),
                ));
            }
            last_usable + 1 - start
        } else {
            let bytes = parse_size(&p.size)?;
            bytes / SECTOR
        };
        if size_lba == 0 {
            return Err(crate::Error::InvalidArgument(format!(
                "spec: partition {:?} has zero size",
                p.name.as_deref().unwrap_or("?")
            )));
        }
        if start + size_lba - 1 > last_usable {
            return Err(crate::Error::InvalidArgument(format!(
                "spec: partition {:?} (LBA {}..{}) overflows the {}-LBA disk",
                p.name.as_deref().unwrap_or("?"),
                start,
                start + size_lba - 1,
                total_lba
            )));
        }
        let kind = parse_partition_kind(&p.kind)?;
        let mut part = Partition::new(start, size_lba, kind);
        part.name = p.name.clone();
        placed.push(part);
        cursor = start + size_lba;
    }

    // Create the backing file and write the partition table.
    let mut dev =
        crate::block::create_image(output, total_bytes, &crate::block::CreateOpts::default())?;
    match table.as_str() {
        "gpt" => {
            let gpt = Gpt::build(placed.clone())?;
            gpt.write(dev.as_mut())?;
        }
        "mbr" => {
            let mbr = Mbr::new(placed.clone())?;
            mbr.write(dev.as_mut())?;
        }
        _ => unreachable!(),
    }

    // Rebuild a PartitionTable trait object once so slice_partition can
    // compute each partition's byte range. (The Gpt/Mbr built above were
    // consumed by write(); rebuilding from `placed` is cheap.)
    let table_obj: Box<dyn PartitionTable> = match table.as_str() {
        "gpt" => Box::new(Gpt::build(placed.clone())?),
        "mbr" => Box::new(Mbr::new(placed.clone())?),
        _ => unreachable!(),
    };

    // Format + populate each partition that carries a filesystem.
    for (i, p) in partitions.iter().enumerate() {
        let Some(fs) = &p.filesystem else {
            continue;
        };
        let part_bytes = placed[i].size_lba * SECTOR;
        let mut slice = slice_partition(table_obj.as_ref(), dev.as_mut(), i)?;
        match fs.fs_type.to_ascii_lowercase().as_str() {
            "ext2" | "ext3" | "ext4" => {
                let kind = parse_fs_kind(&fs.fs_type)?;
                let block_size = fs.block_size.unwrap_or(1024);
                // Fill the partition: blocks_count = partition_bytes / fs_block_size,
                // rounded DOWN to a multiple of 8 (the byte-aligned-group invariant).
                let blocks = ((part_bytes / block_size as u64) / 8 * 8) as u32;
                let opts = ext_format_opts(fs, kind, block_size, Some(blocks))?;
                format_ext_into(&mut slice, fs, &opts)?;
            }
            "fat32" | "vfat" => {
                let total_sectors: u32 = (part_bytes / SECTOR).try_into().map_err(|_| {
                    crate::Error::InvalidArgument(
                        "spec: FAT32 partition size doesn't fit in a u32 sector count".into(),
                    )
                })?;
                let label = fat32_volume_label(fs.volume_label.as_deref());
                let volume_id = fs.volume_id.unwrap_or(0);
                format_fat32_into(&mut slice, fs, total_sectors, volume_id, label)?;
            }
            "hfsplus" | "hfs+" => {
                format_in_partition_via_trait::<crate::fs::hfs_plus::HfsPlus>(
                    &mut slice,
                    fs,
                    hfs_plus_format_opts(fs)?,
                )?;
            }
            "ntfs" => {
                format_in_partition_via_trait::<crate::fs::ntfs::Ntfs>(
                    &mut slice,
                    fs,
                    ntfs_format_opts(fs)?,
                )?;
            }
            "f2fs" => {
                format_in_partition_via_trait::<crate::fs::f2fs::F2fs>(
                    &mut slice,
                    fs,
                    f2fs_format_opts(fs)?,
                )?;
            }
            "squashfs" => {
                format_in_partition_via_trait::<crate::fs::squashfs::Squashfs>(
                    &mut slice,
                    fs,
                    squashfs_format_opts(fs)?,
                )?;
            }
            "xfs" => {
                format_in_partition_via_trait::<crate::fs::xfs::Xfs>(
                    &mut slice,
                    fs,
                    xfs_format_opts(fs)?,
                )?;
            }
            other => {
                return Err(crate::Error::InvalidArgument(format!(
                    "spec: unknown filesystem type {other:?}"
                )));
            }
        }
    }

    dev.sync()?;
    Ok(())
}

/// Map a partition-type string to a [`crate::part::PartitionKind`].
fn parse_partition_kind(s: &str) -> Result<crate::part::PartitionKind> {
    use crate::part::PartitionKind;
    let lower = s.to_ascii_lowercase();
    Ok(match lower.as_str() {
        "esp" | "efi" | "efi-system" => PartitionKind::EfiSystem,
        "linux" | "linux-filesystem" => PartitionKind::LinuxFilesystem,
        "linux-swap" | "swap" => PartitionKind::LinuxSwap,
        "bios-boot" | "bios" => PartitionKind::BiosBoot,
        "fat" | "fat32" => PartitionKind::Fat32,
        "msdata" | "microsoft-basic-data" | "basic-data" => PartitionKind::MicrosoftBasicData,
        "empty" => PartitionKind::Empty,
        other => {
            // Accept a raw MBR byte ("0x83") or a GPT type UUID.
            if let Some(hex) = other.strip_prefix("0x") {
                let b = u8::from_str_radix(hex, 16).map_err(|_| {
                    crate::Error::InvalidArgument(format!("spec: bad MBR type byte {other:?}"))
                })?;
                PartitionKind::from_mbr_byte(b)
            } else if let Ok(uuid) = uuid::Uuid::parse_str(other) {
                PartitionKind::from_gpt_uuid(uuid)
            } else {
                return Err(crate::Error::InvalidArgument(format!(
                    "spec: unknown partition type {s:?}"
                )));
            }
        }
    })
}

fn parse_fs_kind(s: &str) -> Result<FsKind> {
    match s.to_ascii_lowercase().as_str() {
        "ext2" => Ok(FsKind::Ext2),
        "ext3" => Ok(FsKind::Ext3),
        "ext4" => Ok(FsKind::Ext4),
        other => Err(crate::Error::InvalidArgument(format!(
            "spec: parse_fs_kind only handles ext2/3/4 (got {other:?})"
        ))),
    }
}

fn parse_rootdevs(s: Option<&str>) -> Result<RootDevs> {
    match s.map(|x| x.to_ascii_lowercase()) {
        None => Ok(RootDevs::None),
        Some(x) => match x.as_str() {
            "none" => Ok(RootDevs::None),
            "minimal" => Ok(RootDevs::Minimal),
            "standard" => Ok(RootDevs::Standard),
            other => Err(crate::Error::InvalidArgument(format!(
                "spec: unknown rootdevs value {other:?} (want none/minimal/standard)"
            ))),
        },
    }
}

fn rootdevs_entry_count(kind: RootDevs) -> usize {
    crate::fs::rootdevs::device_table(kind).len()
}

/// Parse a human-readable size string into bytes. Accepts a bare integer
/// (bytes), or a number followed by a unit: `KB`/`MB`/`GB`/`TB` (decimal,
/// ×1000) or `KiB`/`MiB`/`GiB`/`TiB` (binary, ×1024). Case-insensitive.
pub fn parse_size(s: &str) -> Result<u64> {
    let s = s.trim();
    let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (num, unit) = s.split_at(split);
    let value: u64 = num
        .parse()
        .map_err(|_| crate::Error::InvalidArgument(format!("spec: bad size {s:?}")))?;
    let mult: u64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "kb" => 1_000,
        "mb" => 1_000_000,
        "gb" => 1_000_000_000,
        "tb" => 1_000_000_000_000,
        "kib" | "k" => 1 << 10,
        "mib" | "m" => 1 << 20,
        "gib" | "g" => 1 << 30,
        "tib" | "t" => 1 << 40,
        other => {
            return Err(crate::Error::InvalidArgument(format!(
                "spec: unknown size unit {other:?}"
            )));
        }
    };
    value
        .checked_mul(mult)
        .ok_or_else(|| crate::Error::InvalidArgument(format!("spec: size {s:?} overflows u64")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_parsing() {
        assert_eq!(parse_size("0").unwrap(), 0);
        assert_eq!(parse_size("512").unwrap(), 512);
        assert_eq!(parse_size("1KiB").unwrap(), 1024);
        assert_eq!(parse_size("4 KiB").unwrap(), 4096);
        assert_eq!(parse_size("256MiB").unwrap(), 256 * 1024 * 1024);
        assert_eq!(parse_size("1GiB").unwrap(), 1 << 30);
        assert_eq!(parse_size("2GB").unwrap(), 2_000_000_000);
        assert_eq!(parse_size("1m").unwrap(), 1 << 20);
        assert!(parse_size("12parsecs").is_err());
        assert!(parse_size("").is_err());
    }

    #[test]
    fn bare_fs_spec_parses() {
        let toml = r#"
            [filesystem]
            type = "ext4"
            source = "./rootfs"
            block_size = 4096
            rootdevs = "minimal"
        "#;
        let spec = Spec::parse(toml).unwrap();
        let fs = spec.filesystem.unwrap();
        assert_eq!(fs.fs_type, "ext4");
        assert_eq!(fs.block_size, Some(4096));
        assert_eq!(fs.rootdevs.as_deref(), Some("minimal"));
        assert!(spec.image.is_none());
    }

    #[test]
    fn rejects_both_filesystem_and_image() {
        let toml = r#"
            [filesystem]
            type = "ext2"

            [image]
            size = "1GiB"
            partition_table = "gpt"
        "#;
        assert!(Spec::parse(toml).is_err());
    }

    #[test]
    fn rejects_empty_spec() {
        assert!(Spec::parse("").is_err());
    }

    #[test]
    fn image_requires_partitions() {
        let toml = r#"
            [image]
            size = "1GiB"
            partition_table = "gpt"
        "#;
        assert!(Spec::parse(toml).is_err());
    }

    #[test]
    fn partitioned_spec_with_nested_filesystem_parses() {
        let toml = r#"
            [image]
            size = "64MiB"
            partition_table = "gpt"

            [[partitions]]
            name = "EFI"
            type = "esp"
            size = "16MiB"

            [[partitions]]
            name = "root"
            type = "linux"
            size = "remaining"

            [partitions.filesystem]
            type = "ext4"
            source = "./rootfs"
            block_size = 4096
        "#;
        let spec = Spec::parse(toml).unwrap();
        assert_eq!(spec.partitions.len(), 2);
        assert_eq!(spec.partitions[0].kind, "esp");
        assert!(spec.partitions[0].filesystem.is_none());
        let root_fs = spec.partitions[1].filesystem.as_ref().unwrap();
        assert_eq!(root_fs.fs_type, "ext4");
        assert_eq!(root_fs.block_size, Some(4096));
    }

    #[test]
    fn options_table_parses_and_flows_into_format_opts() {
        // `[filesystem.options]` is a free-form scalar table; the
        // backend's `apply_options` consumes the keys it knows. Here
        // we feed squashfs `compression` + `block_size` and verify
        // both land on the resulting FormatOpts.
        let toml = r#"
            [filesystem]
            type = "squashfs"
            source = "./rootfs"

            [filesystem.options]
            compression = "gzip"
            block_size = 65536
        "#;
        let spec = Spec::parse(toml).unwrap();
        let fs = spec.filesystem.as_ref().unwrap();
        let opts = squashfs_format_opts(fs).unwrap();
        assert_eq!(opts.block_size, 65536);
        assert!(matches!(
            opts.compression,
            crate::fs::squashfs::Compression::Gzip
        ));
    }

    #[test]
    fn options_table_rejects_unknown_keys() {
        // hfs+'s `apply_options` doesn't know `widgetsize`, so the
        // overall spec parse-and-format step should fail with a clear
        // error citing the unrecognised key.
        let toml = r#"
            [filesystem]
            type = "hfs+"

            [filesystem.options]
            widgetsize = 42
        "#;
        let spec = Spec::parse(toml).unwrap();
        let fs = spec.filesystem.as_ref().unwrap();
        let err = hfs_plus_format_opts(fs).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("hfs+"), "msg: {msg}");
        assert!(msg.contains("widgetsize"), "msg: {msg}");
    }

    #[test]
    fn remaining_must_be_last_partition() {
        let toml = r#"
            [image]
            size = "64MiB"
            partition_table = "gpt"

            [[partitions]]
            name = "a"
            type = "linux"
            size = "remaining"

            [[partitions]]
            name = "b"
            type = "linux"
            size = "16MiB"
        "#;
        let spec = Spec::parse(toml).unwrap();
        let err = build(&spec, std::path::Path::new("/tmp/fstool-test-unused.img")).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidArgument(_)));
    }

    #[test]
    fn partition_kind_strings() {
        use crate::part::PartitionKind;
        assert_eq!(
            parse_partition_kind("esp").unwrap(),
            PartitionKind::EfiSystem
        );
        assert_eq!(
            parse_partition_kind("linux").unwrap(),
            PartitionKind::LinuxFilesystem
        );
        assert_eq!(
            parse_partition_kind("0x83").unwrap(),
            PartitionKind::LinuxFilesystem
        );
        assert!(parse_partition_kind("nonsense").is_err());
    }

    /// End-to-end test of `source = "*.tar"` inside a bare-FS spec.
    /// The tar file is built in-memory, the spec is parsed, and the
    /// built image is re-opened to verify the tar's contents landed
    /// inside the ext4 root. Exercises `repack::Source::detect →
    /// TarArchive → populate_ext_from_source` in the spec path.
    #[test]
    fn build_bare_ext4_from_tar_source() {
        // Stage a tar archive in a tempfile.
        let tmp_dir = std::env::temp_dir();
        let pid = std::process::id();
        let tar_path = tmp_dir.join(format!("fstool-spec-src-{pid}.tar"));
        let _ = std::fs::remove_file(&tar_path);
        write_minimal_tar(&tar_path);

        let img = tmp_dir.join(format!("fstool-spec-tar-{pid}.img"));
        let _ = std::fs::remove_file(&img);
        let toml = format!(
            r#"
                [filesystem]
                type = "ext4"
                source = {src:?}
                block_size = 4096
            "#,
            src = tar_path.to_string_lossy(),
        );
        let spec = Spec::parse(&toml).unwrap();
        build(&spec, &img).unwrap();

        // Re-open the image and confirm the tar entries are present
        // under root.
        let mut dev = crate::block::FileBackend::open(&img).unwrap();
        let ext = crate::fs::ext::Ext::open(&mut dev).unwrap();
        let entries = ext.list_inode(&mut dev, 2).unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"hello.txt"),
            "expected hello.txt in root, got {names:?}"
        );
        drop(ext);
        drop(dev);
        let _ = std::fs::remove_file(&tar_path);
        let _ = std::fs::remove_file(&img);
    }

    /// Same idea, but the source is an *existing ext4 image* instead
    /// of a tar archive. Goes through `repack::Source::detect →
    /// Image → AnyFs walker → copy_into_ext`.
    #[test]
    fn build_bare_ext4_from_existing_image_source() {
        let tmp_dir = std::env::temp_dir();
        let pid = std::process::id();
        let donor = tmp_dir.join(format!("fstool-spec-donor-{pid}.img"));
        let dst = tmp_dir.join(format!("fstool-spec-img-{pid}.img"));
        let _ = std::fs::remove_file(&donor);
        let _ = std::fs::remove_file(&dst);

        // Build the donor with `fstool` itself: a 4 MiB ext4 image
        // with one file at /greeting.txt.
        {
            let host_src = tmp_dir.join(format!("fstool-spec-donor-src-{pid}"));
            let _ = std::fs::remove_dir_all(&host_src);
            std::fs::create_dir_all(&host_src).unwrap();
            std::fs::write(host_src.join("greeting.txt"), b"hi from donor\n").unwrap();
            let mut plan = crate::fs::ext::BuildPlan::new(4096, crate::fs::ext::FsKind::Ext4);
            plan.scan_host_path(&host_src).unwrap();
            let opts = plan.to_format_opts();
            let sz = opts.blocks_count as u64 * opts.block_size as u64;
            let mut d = crate::block::FileBackend::create(&donor, sz).unwrap();
            let mut e = crate::fs::ext::Ext::format_with(&mut d, &opts).unwrap();
            e.populate_from_host_dir(&mut d, 2, &host_src).unwrap();
            e.flush(&mut d).unwrap();
            d.sync().unwrap();
            let _ = std::fs::remove_dir_all(&host_src);
        }

        let toml = format!(
            r#"
                [filesystem]
                type = "ext4"
                source = {src:?}
                block_size = 4096
            "#,
            src = donor.to_string_lossy(),
        );
        let spec = Spec::parse(&toml).unwrap();
        build(&spec, &dst).unwrap();

        let mut dev = crate::block::FileBackend::open(&dst).unwrap();
        let ext = crate::fs::ext::Ext::open(&mut dev).unwrap();
        let entries = ext.list_inode(&mut dev, 2).unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"greeting.txt"),
            "expected greeting.txt in root, got {names:?}"
        );
        drop(ext);
        drop(dev);
        let _ = std::fs::remove_file(&donor);
        let _ = std::fs::remove_file(&dst);
    }

    /// Build a minimal ustar archive at `path` with one regular file
    /// `hello.txt` containing `"hi\n"`. Hand-rolled so the test
    /// doesn't depend on host `tar`.
    fn write_minimal_tar(path: &std::path::Path) {
        use std::io::Write;
        // 512-byte ustar header + one 512-byte body block + two
        // 512-byte zero terminators.
        let mut hdr = [0u8; 512];
        let name = b"hello.txt";
        hdr[..name.len()].copy_from_slice(name);
        // mode = 0644 (NUL-terminated octal in 8 bytes)
        hdr[100..107].copy_from_slice(b"0000644");
        // uid / gid
        hdr[108..115].copy_from_slice(b"0000000");
        hdr[116..123].copy_from_slice(b"0000000");
        // size = 3 (octal "0000003")
        hdr[124..135].copy_from_slice(b"00000000003");
        // mtime = 0
        hdr[136..147].copy_from_slice(b"00000000000");
        // typeflag = '0' (regular)
        hdr[156] = b'0';
        // magic + version
        hdr[257..263].copy_from_slice(b"ustar\0");
        hdr[263..265].copy_from_slice(b"00");
        // checksum: spaces first, then octal sum.
        for b in &mut hdr[148..156] {
            *b = b' ';
        }
        let sum: u32 = hdr.iter().map(|&b| b as u32).sum();
        let cksum = format!("{sum:06o}\0 ");
        hdr[148..148 + cksum.len()].copy_from_slice(cksum.as_bytes());

        let body = b"hi\n";
        let mut body_block = [0u8; 512];
        body_block[..body.len()].copy_from_slice(body);

        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&hdr).unwrap();
        f.write_all(&body_block).unwrap();
        f.write_all(&[0u8; 1024]).unwrap();
    }
}
