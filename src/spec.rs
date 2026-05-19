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
use crate::block::{BlockDevice, FileBackend};
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
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilesystemSpec {
    /// `"ext2"`, `"ext3"`, or `"ext4"`. (FAT32 is post-v1.)
    #[serde(rename = "type")]
    pub fs_type: String,
    /// Host directory whose contents become the filesystem's tree.
    /// Omitted → an empty filesystem (just `/` and `/lost+found`).
    pub source: Option<PathBuf>,
    /// FS block size in bytes (1024 / 2048 / 4096). Default 1024.
    pub block_size: Option<u32>,
    /// Journal size in blocks (ext3/ext4 only). Default 1024.
    pub journal_blocks: Option<u32>,
    /// `"none"`, `"minimal"`, or `"standard"` — pre-populate `/dev`.
    pub rootdevs: Option<String>,
    /// Volume label (≤ 16 bytes).
    pub volume_label: Option<String>,
    /// Modification timestamp baked into every inode (seconds since epoch).
    /// Default 0 for reproducible output.
    pub mtime: Option<u32>,
    /// When true, regular files are written sparsely — all-zero blocks
    /// become holes instead of consuming data blocks. Default false.
    pub sparse: Option<bool>,
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
    let kind = parse_fs_kind(&fs.fs_type)?;
    let block_size = fs.block_size.unwrap_or(1024);
    let opts = ext_format_opts(fs, kind, block_size, None)?;
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(output, size)?;
    format_ext_into(&mut dev, fs, &opts)?;
    dev.sync()?;
    Ok(())
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

    let mut plan = crate::fs::ext::BuildPlan::new(block_size, kind);
    if let Some(j) = fs.journal_blocks {
        plan.journal_blocks = j;
    }
    if let Some(src) = &fs.source {
        plan.scan_host_path(src)?;
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
    let mut ext = Ext::format_with(dev, opts)?;
    if let Some(src) = &fs.source {
        ext.populate_from_host_dir(dev, 2, src)?;
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
    let mut dev = FileBackend::create(output, total_bytes)?;
    match table.as_str() {
        "gpt" => {
            let gpt = Gpt::build(placed.clone())?;
            gpt.write(&mut dev)?;
        }
        "mbr" => {
            let mbr = Mbr::new(placed.clone())?;
            mbr.write(&mut dev)?;
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
        let kind = parse_fs_kind(&fs.fs_type)?;
        let block_size = fs.block_size.unwrap_or(1024);
        let part_bytes = placed[i].size_lba * SECTOR;
        // Fill the partition: blocks_count = partition_bytes / fs_block_size,
        // rounded DOWN to a multiple of 8 (the byte-aligned-group invariant).
        let blocks = ((part_bytes / block_size as u64) / 8 * 8) as u32;
        let opts = ext_format_opts(fs, kind, block_size, Some(blocks))?;
        let mut slice = slice_partition(table_obj.as_ref(), &mut dev, i)?;
        format_ext_into(&mut slice, fs, &opts)?;
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
        "fat32" | "vfat" => Err(crate::Error::Unsupported(
            "spec: FAT32 is not implemented yet (post-v1)".into(),
        )),
        other => Err(crate::Error::InvalidArgument(format!(
            "spec: unknown filesystem type {other:?}"
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
}
