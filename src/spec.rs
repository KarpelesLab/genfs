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
//!    entries. Each partition optionally carries a filesystem. (Landing in
//!    a follow-up; the schema is parsed today but `build` rejects it with
//!    a clear "not yet implemented" error.)
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
/// table (bare-FS mode) and flattened into each `[[partitions]]` entry.
#[derive(Debug, Deserialize)]
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
    } else {
        // image + partitions path
        Err(crate::Error::Unsupported(
            "spec: partitioned disk images are not yet implemented (bare [filesystem] only)".into(),
        ))
    }
}

fn build_bare_fs(fs: &FilesystemSpec, output: &Path) -> Result<()> {
    let kind = parse_fs_kind(&fs.fs_type)?;
    let block_size = fs.block_size.unwrap_or(1024);
    let rootdevs = parse_rootdevs(fs.rootdevs.as_deref())?;
    let mtime = fs.mtime.unwrap_or(0);

    // Size the filesystem from the source tree (plus rootdevs + journal).
    let mut plan = crate::fs::ext::BuildPlan::new(block_size, kind);
    if let Some(j) = fs.journal_blocks {
        plan.journal_blocks = j;
    }
    if let Some(src) = &fs.source {
        plan.scan_host_path(src)?;
    }
    // Account for the /dev tree if requested.
    for _ in 0..rootdevs_entry_count(rootdevs) {
        plan.add_device();
    }
    if rootdevs != RootDevs::None {
        plan.add_dir(); // the /dev directory itself
    }

    let mut opts = plan.to_format_opts();
    opts.mtime = mtime;
    if let Some(label) = &fs.volume_label {
        let bytes = label.as_bytes();
        let n = bytes.len().min(16);
        opts.volume_label[..n].copy_from_slice(&bytes[..n]);
    }

    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(output, size)?;
    let mut ext = Ext::format_with(&mut dev, &opts)?;
    if let Some(src) = &fs.source {
        ext.populate_from_host_dir(&mut dev, 2, src)?;
    }
    if rootdevs != RootDevs::None {
        ext.populate_rootdevs(&mut dev, rootdevs, 0, 0, mtime)?;
    }
    ext.flush(&mut dev)?;
    dev.sync()?;
    Ok(())
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
        // build() rejects the partitioned path for now.
        assert!(matches!(
            build(&spec, std::path::Path::new("/dev/null")),
            Err(crate::Error::Unsupported(_))
        ));
    }
}
