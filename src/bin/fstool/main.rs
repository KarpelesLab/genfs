//! fstool CLI — thin wrapper over the library.
//!
//! Subcommands:
//!
//! - `fstool create` — build a bare filesystem image of any supported
//!   type from an optional host directory. Pick the filesystem with
//!   `--type/-t`; pass FS-specific knobs with `-O key=val,key=val`.
//! - `fstool ls`     — list the contents of a directory inside an image.
//! - `fstool cat`    — print a regular file's contents to stdout.
//! - `fstool info`   — show a one-screen summary of an existing image.
//! - `fstool repack` — rewrite an image into a possibly-different
//!   filesystem at a different size; merge multiple sources.
//! - `fstool build`  — drive a TOML spec.
//! - `fstool add` / `fstool rm` / `fstool shell` — in-place mutation
//!   on supported filesystems.

mod shell;

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use fstool::block::{BlockDevice, FileBackend};
use fstool::format_opts::OptionMap;
use fstool::fs::ext::{Ext, FsKind};

#[derive(Parser, Debug)]
#[command(
    name = "fstool",
    version,
    about = "Build and inspect disk-image filesystems (ext2/3/4, MBR, GPT)."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Create a fresh filesystem image of the chosen type, optionally
    /// populated from a host directory tree. Replaces the older
    /// `ext-build` / `fat-build` commands.
    ///
    /// Pass FS-specific knobs through `-O key=val,key=val` (repeatable).
    /// The recognised keys are documented next to each backend's
    /// `FormatOpts::apply_options`; unknown keys are rejected with a
    /// clear error citing the FS type.
    Create {
        /// Filesystem type to format: ext2 / ext3 / ext4 / fat32 / vfat /
        /// exfat / hfs+ / hfsplus / ntfs / f2fs / squashfs / xfs / iso /
        /// iso9660 / grf / apfs.
        #[arg(short = 't', long = "type", value_name = "TYPE")]
        fs_type: String,
        /// Optional source directory on the host. Omit to create an
        /// empty filesystem (subject to per-FS minimum sizes).
        #[arg(value_name = "SRC_DIR")]
        src_dir: Option<PathBuf>,
        /// Output image file or block device. Block devices are formatted
        /// to their full capacity; regular files are auto-sized when the
        /// FS supports it (ext / fat32 / iso / grf) or default to a
        /// per-FS minimum otherwise. Explicit `--size` wins.
        #[arg(short = 'o', long = "output", value_name = "IMAGE")]
        output: PathBuf,
        /// Image size, e.g. "64MiB" or "1GiB". Ignored when OUTPUT is a
        /// block device (the device's capacity wins).
        #[arg(long, value_name = "SIZE")]
        size: Option<String>,
        /// Shortcut for `-O volume_label=…`. Per-FS truncation /
        /// case-folding rules apply (FAT32 upper-cases to 11 bytes,
        /// HFS+ stores UTF-8, …).
        #[arg(long)]
        label: Option<String>,
        /// Required when OUTPUT is a block device — refuses to format
        /// a real device without an explicit opt-in.
        #[arg(long)]
        force: bool,
        /// qcow2 cluster size (only honoured when OUTPUT ends in
        /// `.qcow2` / `.qcow` / `.q2`). Accepts `64KiB`, `1MiB`, or a
        /// bare byte count; must be a power of two ≥ 512. Default 64 KiB.
        #[arg(long, value_name = "SIZE", default_value = "64KiB")]
        cluster_size: String,
        /// FS-specific options as `key=val[,key=val]…`. Repeatable.
        /// Examples: `-O block_size=4096,sparse=true` (ext),
        /// `-O volume_id=0xCAFEBABE` (fat32),
        /// `-O compression=zstd,block_size=128KiB` (squashfs).
        #[arg(short = 'O', long = "options", value_name = "KEY=VAL", action = clap::ArgAction::Append)]
        options: Vec<String>,
    },

    /// List a directory inside an image. One entry per line:
    /// `<inode>\t<kind>\t<name>`. To target a partition, append `:N`
    /// (1-indexed) to the image path: `disk.img:2`.
    Ls {
        /// Image path, optionally with `:N` to select partition N.
        #[arg(value_name = "IMAGE[:N]")]
        image: String,
        /// Path inside the image to list. Defaults to `/`.
        #[arg(value_name = "PATH", default_value = "/")]
        path: String,
    },

    /// Print the contents of a regular file from inside an image to stdout.
    Cat {
        /// Image path, optionally with `:N` to select partition N.
        #[arg(value_name = "IMAGE[:N]")]
        image: String,
        /// Path inside the image to read.
        #[arg(value_name = "PATH")]
        path: String,
    },

    /// One-screen summary of an existing image. On a partitioned image
    /// (no `:N`), prints the partition table; with `:N`, prints the
    /// filesystem summary for that partition.
    Info {
        /// Image path, optionally with `:N` to select partition N.
        #[arg(value_name = "IMAGE[:N]")]
        image: String,
    },

    /// Build an image from a TOML spec file. Bare-filesystem specs are
    /// supported today; partitioned-disk specs land in a follow-up.
    Build {
        /// Path to the TOML spec file.
        #[arg(value_name = "SPEC")]
        spec: PathBuf,
        /// Output image file.
        #[arg(short = 'o', long = "output", value_name = "IMAGE")]
        output: PathBuf,
    },

    /// Copy a host file or directory into an existing image. The
    /// destination's parent directory must already exist in the image.
    Add {
        /// Image path, optionally with `:N` to select partition N.
        #[arg(value_name = "IMAGE[:N]")]
        image: String,
        /// Host file or directory to copy in.
        #[arg(value_name = "HOST_SRC")]
        host_src: PathBuf,
        /// Absolute destination path inside the image.
        #[arg(value_name = "FS_DEST")]
        fs_dest: String,
    },

    /// Remove a file, symlink, device node, or empty directory from an
    /// existing image.
    Rm {
        /// Image path, optionally with `:N` to select partition N.
        #[arg(value_name = "IMAGE[:N]")]
        image: String,
        /// Absolute path inside the image to remove.
        #[arg(value_name = "FS_PATH")]
        fs_path: String,
    },

    /// Open an interactive shell over an image. Maintains a virtual cwd
    /// and reads commands from stdin; type `help` once inside for the
    /// command list, or `quit` (or EOF) to leave.
    Shell {
        /// Image path, optionally with `:N` to select partition N.
        #[arg(value_name = "IMAGE[:N]")]
        image: String,
    },

    /// Convert an image between container formats (raw ↔ qcow2). Streams
    /// every byte from SRC to DST; no filesystem awareness, so this
    /// works on partitioned disks just as well as bare filesystems. Use
    /// `repack` instead if you want to shrink the image — `convert` can
    /// only grow it.
    Convert {
        /// Source image (raw, qcow2, or block device).
        #[arg(value_name = "SRC")]
        src: PathBuf,
        /// Destination image. Format is picked from the extension
        /// (`.qcow2` / `.qcow` / `.q2` → qcow2; otherwise raw).
        #[arg(value_name = "DST")]
        dst: PathBuf,
        /// Destination size. Defaults to the source's virtual size. May
        /// be larger (grows the image with all-zero tail) but not smaller.
        #[arg(long, value_name = "SIZE")]
        size: Option<String>,
        /// qcow2 cluster size for the destination, when DST is a qcow2.
        #[arg(long, value_name = "SIZE", default_value = "64KiB")]
        cluster_size: String,
    },

    /// Repack an image into a fresh filesystem at a (possibly different)
    /// size. Walks SRC's filesystem, stages each file into a host
    /// tempdir, then formats DST from scratch. Use this when you need to
    /// shrink an image — `convert` can only do byte copies, while
    /// `repack` actually rewrites the filesystem.
    Repack {
        /// One or more sources followed by a destination. With one
        /// source, repacks normally. With two or more, the sources are
        /// merged bottom→top before repacking: later layers override
        /// earlier ones, and tar-OCI / overlayfs whiteouts are honoured.
        /// Globs like `repack data* out.tar` work as long as the shell
        /// expands them in source-then-destination order.
        #[arg(value_name = "PATH", num_args = 2.., required = true)]
        paths: Vec<String>,
        /// Destination size. Default: same as source's filesystem size.
        /// Mutually exclusive with `--shrink`.
        #[arg(long, value_name = "SIZE", conflicts_with = "shrink")]
        size: Option<String>,
        /// Auto-size the destination to the minimum that fits the
        /// staged content (uses BuildPlan for ext; the FAT32 minimum
        /// of ~33 MiB still applies).
        #[arg(long)]
        shrink: bool,
        /// Destination filesystem type. Defaults to the source's type
        /// (ext2/3/4 → matching ext flavour; FAT32 → FAT32). Pass an
        /// explicit value to convert filesystem types (loses
        /// per-inode metadata on ext → FAT32).
        #[arg(long, value_name = "TYPE")]
        fs_type: Option<String>,
        /// ext block size (1024/2048/4096); ignored for FAT32 output.
        #[arg(long, default_value_t = 1024)]
        block_size: u32,
        /// qcow2 cluster size for the destination, when DST is a qcow2.
        #[arg(long, value_name = "SIZE", default_value = "64KiB")]
        cluster_size: String,
    },
    /// Mount an ext{2,3,4} image at a host mountpoint via FUSE.
    /// Only available when fstool is built with `--features fuse`.
    /// The mount stays attached until `umount` on the mountpoint
    /// (or the process exits — `AutoUnmount` is on by default).
    #[cfg(feature = "fuse")]
    Mount {
        /// Image to mount. Plain file or `path:N` partition selector.
        #[arg(value_name = "IMAGE")]
        image: String,
        /// Host directory to mount the image under. Must already exist
        /// and be empty (or close to it — your kernel decides).
        #[arg(value_name = "MOUNTPOINT")]
        mountpoint: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("fstool: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> fstool::Result<()> {
    match cli.command {
        Command::Create {
            fs_type,
            src_dir,
            output,
            size,
            label,
            force,
            cluster_size,
            options,
        } => create_cmd(CreateArgs {
            fs_type: &fs_type,
            src_dir: src_dir.as_deref(),
            output: &output,
            size: size.as_deref(),
            label: label.as_deref(),
            force,
            cluster_size: &cluster_size,
            options: &options,
        }),
        Command::Ls { image, path } => ls(&image, &path),
        Command::Cat { image, path } => cat(&image, &path),
        Command::Info { image } => info(&image),
        Command::Build { spec, output } => build(&spec, &output),
        Command::Add {
            image,
            host_src,
            fs_dest,
        } => add(&image, &host_src, &fs_dest),
        Command::Rm { image, fs_path } => rm(&image, &fs_path),
        Command::Shell { image } => shell_cmd(&image),
        Command::Convert {
            src,
            dst,
            size,
            cluster_size,
        } => convert_cmd(&src, &dst, size.as_deref(), &cluster_size),
        Command::Repack {
            mut paths,
            size,
            shrink,
            fs_type,
            block_size,
            cluster_size,
        } => {
            let dst_str = paths.pop().expect("clap enforces num_args >= 2");
            let dst = PathBuf::from(dst_str);
            let srcs = paths;
            // Install a progress sink for the duration of the repack;
            // it emits a refreshing status line in TTY mode and stays
            // quiet for pipes/logs.
            fstool::repack::enter(fstool::repack::Progress::auto());
            let res = repack_cmd(
                &srcs,
                &dst,
                size.as_deref(),
                shrink,
                fs_type.as_deref(),
                block_size,
                &cluster_size,
            );
            fstool::repack::leave();
            res
        }
        #[cfg(feature = "fuse")]
        Command::Mount { image, mountpoint } => mount_cmd(&image, &mountpoint),
    }
}

#[cfg(feature = "fuse")]
fn mount_cmd(image: &str, mountpoint: &std::path::Path) -> fstool::Result<()> {
    // FUSE adapter wants ownership of both `Ext` and a `Box<dyn
    // BlockDevice>`. Partition selectors (`image:N`) and qcow2 sources
    // go through `inspect::with_target_device` which doesn't give us
    // ownership; for v1 we accept plain raw images only.
    if image.contains(':') {
        return Err(fstool::Error::Unsupported(
            "fstool mount: partitioned-image selectors (`path:N`) and qcow2 sources \
             are not wired through the FUSE adapter yet — open a raw .img directly"
                .into(),
        ));
    }
    let path = std::path::Path::new(image);
    let mut dev: Box<dyn fstool::block::BlockDevice> =
        Box::new(fstool::block::FileBackend::open(path)?);
    let mut ext = fstool::fs::ext::Ext::open(dev.as_mut())?;
    // Replay any pending journal so the mounted view matches what the
    // kernel would see on first mount of an unclean shutdown.
    let _ = ext.replay_pending_journal(dev.as_mut())?;
    let fs = fstool::fuse_adapter::FstoolFs::new(ext, dev);
    eprintln!(
        "fstool: mounted {image} at {} (umount to detach)",
        mountpoint.display()
    );
    fs.mount(mountpoint, "fstool").map_err(fstool::Error::Io)?;
    Ok(())
}

fn convert_cmd(
    src: &std::path::Path,
    dst: &std::path::Path,
    size_arg: Option<&str>,
    cluster_size: &str,
) -> fstool::Result<()> {
    let cluster_size = parse_cluster_size(cluster_size)?;
    let mut src_dev = fstool::block::open_image(src)?;
    let src_size = src_dev.total_size();
    let dst_size = match size_arg {
        None => src_size,
        Some(s) => {
            let want = fstool::spec::parse_size(s)?;
            if want < src_size {
                return Err(fstool::Error::InvalidArgument(format!(
                    "convert: --size {want} is smaller than source's {src_size}; use `repack --shrink` instead"
                )));
            }
            want
        }
    };
    let mut dst_dev =
        fstool::block::create_image(dst, dst_size, &fstool::block::CreateOpts { cluster_size })?;
    // 1 MiB copy buffer. Reads from sparse regions return zeros; on the
    // qcow2 side those become unallocated clusters (no on-disk cost).
    let mut buf = vec![0u8; 1024 * 1024];
    let mut copied = 0u64;
    while copied < src_size {
        let n = (src_size - copied).min(buf.len() as u64) as usize;
        src_dev.read_at(copied, &mut buf[..n])?;
        // Skip all-zero chunks on the write side so sparse output stays sparse.
        if !buf[..n].iter().all(|&b| b == 0) {
            dst_dev.write_at(copied, &buf[..n])?;
        }
        copied += n as u64;
    }
    dst_dev.sync()?;
    eprintln!(
        "converted {} → {} ({} → {} bytes)",
        src.display(),
        dst.display(),
        src_size,
        dst_size
    );
    Ok(())
}

fn repack_cmd(
    srcs: &[String],
    dst: &std::path::Path,
    size_arg: Option<&str>,
    shrink: bool,
    fs_type_override: Option<&str>,
    block_size: u32,
    cluster_size: &str,
) -> fstool::Result<()> {
    let qcow2_cluster_size = parse_cluster_size(cluster_size)?;

    if srcs.is_empty() {
        return Err(fstool::Error::InvalidArgument(
            "repack: at least one source is required".into(),
        ));
    }

    // Multi-source: flatten all layers to a single uncompressed tar in
    // a tempfile, then recurse with that tempfile as the sole source.
    // The tempfile lives until the inner repack completes; the
    // recursive call inherits the same `dst` and options, so the merged
    // tar drives the rest of the pipeline exactly as a single tar
    // source would.
    if srcs.len() > 1 {
        let layers: Vec<fstool::repack::Source> = srcs
            .iter()
            .map(|s| fstool::repack::Source::detect(s))
            .collect::<fstool::Result<_>>()?;
        let tmp = fstool::merge::flatten_to_tempfile(&layers)?;
        let merged = tmp.path().to_string_lossy().into_owned();
        let res = repack_cmd(
            std::slice::from_ref(&merged),
            dst,
            size_arg,
            shrink,
            fs_type_override,
            block_size,
            cluster_size,
        );
        drop(tmp);
        return res;
    }

    let src = srcs[0].as_str();
    let src_target = fstool::inspect::Target::parse(src);

    // Compressed-tar source: stream the archive directly into the
    // destination without spooling through a tempfile. Only the
    // tar → tar combination is fully wired here; for other targets
    // the function returns an actionable error.
    if let Some(algo) = tar_input_codec(src) {
        return repack_from_tar_stream(src, algo, dst, fs_type_override);
    }

    // Open the source once and walk it; the source FS stays open across
    // the destination build so we stream each file straight through
    // without ever touching the host filesystem.
    fstool::inspect::with_target_device(&src_target, |src_dev| {
        let mut src_fs = fstool::inspect::AnyFs::open(src_dev)?;
        // For ext sources with INCOMPAT_RECOVER / s_start != 0,
        // replay the journal onto the source so we read the
        // post-recovery state (anything still pending in the log
        // would otherwise be lost from the repack output).
        if let fstool::inspect::AnyFs::Ext(ext) = &mut src_fs {
            let _ = ext.replay_pending_journal(src_dev)?;
        }
        let source_kind = src_fs.kind_string();
        let src_total = src_dev.total_size();

        // Compute the destination size + a BuildPlan-shaped sketch by
        // walking the source FS, no host I/O involved.
        //
        // Default destination FS: explicit --fs-type, else infer from
        // the dst extension (.tar → tar), else preserve the source kind.
        // Tar output, optionally with a codec extension chain
        // (`.tar.gz` / `.tar.zst` / `.tgz` etc.). The chain is parsed
        // here so the writer below can pick a streaming compressor.
        let dst_tar_codec = tar_output_codec(dst);
        let target_fs_str = fs_type_override
            .map(|s| s.to_string())
            .or_else(|| {
                dst.extension()
                    .and_then(|s| s.to_str())
                    .filter(|e| e.eq_ignore_ascii_case("tar"))
                    .map(|_| "tar".to_string())
            })
            .or_else(|| dst_tar_codec.map(|_| "tar".to_string()))
            .unwrap_or_else(|| {
                // Default destination FS when nothing else specifies one:
                // preserve the source FS, unless the source is tar (in
                // which case picking "tar" would just round-trip the
                // archive — almost never what the user wants). Default to
                // ext4 in that case; the user can override with --fs-type.
                if source_kind == "tar" {
                    "ext4".into()
                } else {
                    source_kind.to_string()
                }
            });
        let lower = target_fs_str.to_ascii_lowercase();
        let dst_size = match (size_arg, shrink) {
            (Some(s), _) => fstool::spec::parse_size(s)?,
            (None, true) => match lower.as_str() {
                "ext2" | "ext3" | "ext4" => {
                    let plan = build_ext_plan(src_dev, &mut src_fs, block_size, &lower)?;
                    plan.blocks_count() as u64 * plan.block_size as u64
                }
                "fat32" | "vfat" => {
                    let bytes = sum_source_file_bytes(src_dev, &mut src_fs)?;
                    let needed = bytes
                        .saturating_mul(2)
                        .max(fstool::fs::fat::MIN_FAT32_CLUSTERS as u64 * 1024);
                    needed.div_ceil(512) * 512
                }
                "tar" => tar_size_upper_bound(src_dev, &src_fs)?,
                "iso" | "iso9660" => {
                    // ISO writer needs ~32 MiB headroom for a small tree.
                    // Real sizing happens during flush; we just want enough
                    // backing image to write into.
                    let bytes = sum_source_file_bytes(src_dev, &mut src_fs)?;
                    bytes.saturating_add(32 * 1024 * 1024)
                }
                "grf" => {
                    // GRF stores zlib-compressed bodies — sum_source
                    // gives an upper bound (uncompressed). Add 64 KiB
                    // headroom for the header + table.
                    let bytes = sum_source_file_bytes(src_dev, &mut src_fs)?;
                    bytes.saturating_add(64 * 1024)
                }
                other => {
                    return Err(fstool::Error::InvalidArgument(format!(
                        "repack: unknown --fs-type {other:?}"
                    )));
                }
            },
            // For tar, "explicit size" doesn't really apply since the
            // archive grows to whatever fits. Without --shrink either
            // we still upper-bound the destination from the source.
            (None, false) => match lower.as_str() {
                "tar" => tar_size_upper_bound(src_dev, &src_fs)?,
                "iso" | "iso9660" => {
                    // ISO writer needs enough room for descriptors,
                    // path tables, dir records, and file data. Use a
                    // generous upper bound — the writer leaves the
                    // unused tail of the backing file alone.
                    let bytes = sum_source_file_bytes(src_dev, &mut src_fs).unwrap_or(0);
                    bytes.saturating_add(32 * 1024 * 1024)
                }
                "grf" => {
                    let bytes = sum_source_file_bytes(src_dev, &mut src_fs).unwrap_or(0);
                    bytes.saturating_add(64 * 1024)
                }
                _ => src_total,
            },
        };

        // Tar output is special: a tar archive is sequential, not a
        // pre-sized block device. We open `dst` directly as a `Write`
        // (optionally codec-wrapped for `.tar.gz` / `.tar.zst` / etc.)
        // and stream every entry through a `TarStreamWriter`. No
        // tempfile, no `set_len`, no `create_image`.
        if lower == "tar" {
            let written = repack_into_tar_streaming(src_dev, &src_fs, dst, dst_tar_codec)?;
            if let Some(algo) = dst_tar_codec {
                eprintln!(
                    "repacked {src} → {} (fs: {source_kind} → tar.{}, {written} bytes plain)",
                    dst.display(),
                    algo.name()
                );
            } else {
                eprintln!(
                    "repacked {src} → {} (fs: {source_kind} → tar, {written} bytes)",
                    dst.display()
                );
            }
            return Ok(());
        }
        let mut dst_dev = fstool::block::create_image(
            dst,
            dst_size,
            &fstool::block::CreateOpts {
                cluster_size: qcow2_cluster_size,
            },
        )?;
        match lower.as_str() {
            "ext2" | "ext3" | "ext4" => {
                let plan = build_ext_plan(src_dev, &mut src_fs, block_size, &lower)?;
                let mut opts = plan.to_format_opts();
                // Preserve sparse-file extent: the source reader emits
                // zero bytes wherever the source has a hole, and the
                // destination writer turns all-zero blocks into holes
                // when sparse is on. End result: holes round-trip
                // through repack instead of being inflated to dense
                // zero blocks. Semantically transparent for non-sparse
                // files (their blocks aren't all-zero so nothing
                // changes).
                opts.sparse = true;
                // Grow to fill the destination if the user requested an
                // explicit size larger than the auto-min.
                let plan_size = opts.blocks_count as u64 * opts.block_size as u64;
                if dst_size > plan_size {
                    let max = (dst_size / opts.block_size as u64) as u32;
                    opts.blocks_count = (max / 8) * 8;
                    let by_density =
                        (opts.blocks_count as u64 * opts.block_size as u64 / 16_384) as u32;
                    opts.inodes_count = opts.inodes_count.max(by_density);
                }
                let mut dst_ext = fstool::fs::ext::Ext::format_with(dst_dev.as_mut(), &opts)?;
                copy_into_ext(src_dev, &src_fs, dst_dev.as_mut(), &mut dst_ext)?;
                dst_ext.flush(dst_dev.as_mut())?;
            }
            "fat32" | "vfat" => {
                let total_sectors: u32 = (dst_size / 512).try_into().map_err(|_| {
                    fstool::Error::InvalidArgument(
                        "repack: FAT32 size doesn't fit in a u32 sector count".into(),
                    )
                })?;
                let label = *b"REPACKED   ";
                let opts = fstool::fs::fat::FatFormatOpts {
                    total_sectors,
                    volume_id: 0,
                    volume_label: label,
                };
                let mut dst_fat = fstool::fs::fat::Fat32::format(dst_dev.as_mut(), &opts)?;
                copy_into_fat32(src_dev, &src_fs, dst_dev.as_mut(), &mut dst_fat)?;
                dst_fat.flush(dst_dev.as_mut())?;
            }
            "hfsplus" | "hfs+" => {
                repack_via_trait::<fstool::fs::hfs_plus::HfsPlus>(
                    dst_dev.as_mut(),
                    &fstool::fs::hfs_plus::FormatOpts::default(),
                    src,
                )?;
            }
            "ntfs" => {
                repack_via_trait::<fstool::fs::ntfs::Ntfs>(
                    dst_dev.as_mut(),
                    &fstool::fs::ntfs::format::FormatOpts::default(),
                    src,
                )?;
            }
            "f2fs" => {
                repack_via_trait::<fstool::fs::f2fs::F2fs>(
                    dst_dev.as_mut(),
                    &fstool::fs::f2fs::FormatOpts::default(),
                    src,
                )?;
            }
            "squashfs" => {
                repack_via_trait::<fstool::fs::squashfs::Squashfs>(
                    dst_dev.as_mut(),
                    &fstool::fs::squashfs::FormatOpts::default(),
                    src,
                )?;
            }
            "xfs" => {
                repack_via_trait::<fstool::fs::xfs::Xfs>(
                    dst_dev.as_mut(),
                    &fstool::fs::xfs::format::FormatOpts::default(),
                    src,
                )?;
            }
            "iso" | "iso9660" => {
                let opts = fstool::fs::iso9660::FormatOpts {
                    volume_id: "FSTOOL".into(),
                    application_id: "fstool".into(),
                    ..fstool::fs::iso9660::FormatOpts::default()
                };
                repack_via_trait::<fstool::fs::iso9660::Iso9660>(dst_dev.as_mut(), &opts, src)?;
            }
            "grf" => {
                let opts = fstool::fs::grf::FormatOpts::default();
                repack_via_trait::<fstool::fs::grf::Grf>(dst_dev.as_mut(), &opts, src)?;
            }
            other => {
                return Err(fstool::Error::InvalidArgument(format!(
                    "repack: unknown --fs-type {other:?}"
                )));
            }
        }
        dst_dev.sync()?;
        eprintln!(
            "repacked {src} → {} (fs: {source_kind} → {target_fs_str}, {dst_size} bytes)",
            dst.display()
        );
        Ok(())
    })
}

/// Generic repack pipeline driven by [`fstool::fs::FilesystemFactory`].
/// Formats `F` onto `dst_dev`, walks the source `src` (passed as the
/// CLI's `disk.img:N` / tar / dir spec) through
/// [`fstool::repack::Source::detect`], and replays every entry into
/// `F` via the trait. The destination is flushed before this returns.
fn repack_via_trait<F: fstool::fs::FilesystemFactory>(
    dst_dev: &mut dyn fstool::block::BlockDevice,
    opts: &F::FormatOpts,
    src: &str,
) -> fstool::Result<()> {
    let mut dst = F::format(dst_dev, opts)?;
    let source = fstool::repack::Source::detect(src)?;
    fstool::repack::populate_fs_from_source(dst_dev, &mut dst, &source)?;
    dst.flush(dst_dev)?;
    Ok(())
}

/// Walk the source filesystem and recreate every entry inside the
/// destination ext. Preserves mode, uid/gid, mtime, atime, ctime; copies
/// symlinks and device nodes verbatim when the source is ext (FAT
/// source has none of those).
fn copy_into_ext(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src_fs: &fstool::inspect::AnyFs,
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::ext::Ext,
) -> fstool::Result<()> {
    use fstool::fs::FileMeta;
    use fstool::inspect::AnyFs;
    match src_fs {
        AnyFs::Ext(src_ext) => copy_ext_dir(src_dev, src_ext, 2, dst_dev, dst, 2),
        AnyFs::Fat32(src_fat) => {
            copy_fat_dir_into_ext(src_dev, src_fat, "/", dst_dev, dst, 2, &FileMeta::default())
        }
        AnyFs::Tar(src_tar) => copy_tar_into_ext(src_dev, src_tar, dst_dev, dst),
        _ => Err(unsupported_repack_src(src_fs)),
    }
}

/// Walk the source filesystem and recreate every entry inside the
/// destination FAT32. FAT can't represent symlinks / device nodes /
/// per-file permissions — those are dropped (with a stderr note when
/// the source had them).
fn copy_into_fat32(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src_fs: &fstool::inspect::AnyFs,
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::fat::Fat32,
) -> fstool::Result<()> {
    use fstool::inspect::AnyFs;
    match src_fs {
        AnyFs::Ext(src_ext) => copy_ext_dir_into_fat(src_dev, src_ext, 2, "/", dst_dev, dst),
        AnyFs::Fat32(src_fat) => copy_fat_dir_into_fat(src_dev, src_fat, "/", dst_dev, dst),
        AnyFs::Tar(src_tar) => copy_tar_into_fat(src_dev, src_tar, dst_dev, dst),
        _ => Err(unsupported_repack_src(src_fs)),
    }
}

/// Repack-source error for the four read-only FSes (xfs/exfat/hfs+/apfs)
/// — they're inspectable via ls/cat/info but not yet wired into the
/// FS-to-FS copy walkers.
/// If `path` looks like a compressed tar (`.tar.gz`, `.tar.zst`,
/// `.tar.xz`, `.tgz`, `.txz`, `.tar.lz4`, `.tar.lzma`, `.tar.lzo`),
/// return the codec to use; otherwise `None`. Used by repack to pick a
/// streaming compressor for the output file.
fn tar_output_codec(path: &std::path::Path) -> Option<fstool::compression::Algo> {
    let s = path.to_string_lossy().to_ascii_lowercase();
    if s.ends_with(".tgz") {
        return Some(fstool::compression::Algo::Gzip);
    }
    if s.ends_with(".txz") {
        return Some(fstool::compression::Algo::Xz);
    }
    if !s.contains(".tar.") {
        // Bare `.gz` / `.zst` etc. without a `.tar.` prefix isn't tar.
        return None;
    }
    fstool::compression::Algo::from_extension(path)
}

/// `Some(algo)` when `path` points at a compressed tar archive that
/// should be stream-walked rather than decompressed-to-tempfile.
/// `None` for plain `.tar` (the regular BlockDevice path handles it
/// fine) and for non-tar files.
fn tar_input_codec(path: &str) -> Option<fstool::compression::Algo> {
    // Strip any `:N` partition selector — tar archives don't have
    // partitions, but the parsing helper allows the form.
    let p = std::path::Path::new(path.split(':').next().unwrap_or(path));
    tar_output_codec(p)
}

/// Repack from a streaming compressed-tar source.
///
/// Three destinations are wired:
/// - **tar / tar.<algo>**: one decompression pass, transcoded straight
///   into the output writer. Hard links are now materialised properly
///   thanks to [`TarStreamIndex`] (a tiny upfront pre-walk builds an
///   index so the writer can re-decompress the prefix needed to copy
///   each link target's bytes).
/// - **ext{2,3,4} / fat32**: two decompression passes. The first walk
///   sizes the destination (entry counts + byte totals); the second
///   replays the entries into the freshly-formatted destination via
///   [`TarStreamIndex::open_body`], which seeks a fresh decoder to
///   each regular file's body offset. The destination FS isn't
///   append-only, which is why the size has to be known up front.
///   This is the deliberate streaming-invariant-honouring alternative
///   to spooling the whole archive through a tempfile.
fn repack_from_tar_stream(
    src: &str,
    algo: fstool::compression::Algo,
    dst: &std::path::Path,
    fs_type_override: Option<&str>,
) -> fstool::Result<()> {
    use fstool::fs::tar::{EntryKind as TarKind, TarEntryMeta, TarStreamWriter};
    let dst_codec = tar_output_codec(dst);
    let target_fs_str = fs_type_override
        .map(|s| s.to_ascii_lowercase())
        .or_else(|| {
            dst.extension()
                .and_then(|s| s.to_str())
                .filter(|e| e.eq_ignore_ascii_case("tar"))
                .map(|_| "tar".to_string())
        })
        .or_else(|| dst_codec.map(|_| "tar".to_string()))
        .unwrap_or_else(|| "tar".to_string());
    let target_lower = target_fs_str.to_ascii_lowercase();

    if target_lower != "tar" {
        // Non-tar destination: two-pass build (size, format, replay).
        return repack_from_tar_stream_into_fs(src, algo, dst, &target_lower);
    }

    // Build an in-memory index over the source so hard links can be
    // resolved. The index walk is one full decompression pass; the
    // transcoding pump below is the second.
    let index = build_tar_stream_index(src, algo)?;

    let file = std::fs::File::create(dst)?;
    let buffered: Box<dyn std::io::Write> =
        Box::new(std::io::BufWriter::with_capacity(64 * 1024, file));
    let inner: Box<dyn std::io::Write> = match dst_codec {
        Some(a) => fstool::compression::make_writer(a, buffered)?,
        None => buffered,
    };
    let mut writer = TarStreamWriter::new(inner);

    let mut reader = open_tar_stream_reader(src, Some(algo))?;
    while let Some(mut ent) = reader.next_entry()? {
        let entry = ent.entry.clone();
        let meta = TarEntryMeta {
            mode: entry.mode,
            uid: entry.uid,
            gid: entry.gid,
            mtime: entry.mtime,
            uname: String::new(),
            gname: String::new(),
        };
        match entry.kind {
            TarKind::Regular => {
                writer.add_file(&entry.path, &mut ent, entry.size, meta, &entry.xattrs)?;
            }
            TarKind::Dir => writer.add_dir(&entry.path, meta, &entry.xattrs)?,
            TarKind::Symlink => {
                let target = entry.link_target.as_deref().unwrap_or("");
                writer.add_symlink(&entry.path, target, meta, &entry.xattrs)?;
            }
            TarKind::HardLink => {
                // Materialise the link target's body via the index: a
                // fresh decompression stream is opened and skipped
                // forward to the target's body offset, yielding the
                // bytes through a bounded Read.
                let mut body = index.open_body(&entry.path, || open_decoded_stream(src, algo))?;
                let size = body.remaining();
                writer.add_file(&entry.path, &mut body, size, meta, &entry.xattrs)?;
            }
            TarKind::CharDev => {
                writer.add_device(
                    &entry.path,
                    fstool::fs::DeviceKind::Char,
                    entry.device_major,
                    entry.device_minor,
                    meta,
                    &entry.xattrs,
                )?;
            }
            TarKind::BlockDev => {
                writer.add_device(
                    &entry.path,
                    fstool::fs::DeviceKind::Block,
                    entry.device_major,
                    entry.device_minor,
                    meta,
                    &entry.xattrs,
                )?;
            }
            TarKind::Fifo => {
                writer.add_device(
                    &entry.path,
                    fstool::fs::DeviceKind::Fifo,
                    0,
                    0,
                    meta,
                    &entry.xattrs,
                )?;
            }
        }
    }
    writer.finish()?;
    let written = writer.bytes_written();
    drop(writer);
    if let Some(a) = dst_codec {
        eprintln!(
            "repacked {src} → {} (fs: tar.{} → tar.{}, {written} bytes plain)",
            dst.display(),
            algo.name(),
            a.name()
        );
    } else {
        eprintln!(
            "repacked {src} → {} (fs: tar.{} → tar, {written} bytes)",
            dst.display(),
            algo.name()
        );
    }
    Ok(())
}

/// Open `src` as a freshly-decoded `Read` positioned at the
/// decompressed stream's byte 0. Boxed so it composes with the
/// existing helpers; callers feed this to [`TarStreamIndex::open_body`]
/// to seek to a specific entry's body offset.
fn open_decoded_stream(
    src: &str,
    algo: fstool::compression::Algo,
) -> fstool::Result<Box<dyn std::io::Read>> {
    let p = std::path::Path::new(src.split(':').next().unwrap_or(src));
    let file = std::fs::File::open(p)?;
    let buffered: Box<dyn std::io::Read> =
        Box::new(std::io::BufReader::with_capacity(64 * 1024, file));
    fstool::compression::make_reader(algo, buffered)
}

/// Single-pass walk that builds a [`TarStreamIndex`] for a compressed
/// tar source. Bodies are NOT consumed: the underlying reader skips
/// past each body's bytes during `next_entry`, so the only buffered
/// data is the per-entry metadata.
fn build_tar_stream_index(
    src: &str,
    algo: fstool::compression::Algo,
) -> fstool::Result<fstool::fs::tar::TarStreamIndex> {
    let reader = open_tar_stream_reader(src, Some(algo))?;
    fstool::fs::tar::TarStreamIndex::build_from(reader)
}

/// Two-pass repack from a compressed tar into a pre-sized destination
/// (ext{2,3,4} or fat32). Pass 1 walks the archive to build an index
/// (which also gives us entry counts + total file bytes for sizing);
/// Pass 2 replays the entries into the freshly-formatted destination,
/// re-decompressing the source per regular-file body via the index's
/// seek-by-offset helper.
///
/// The two-pass cost is the price of honouring the streaming
/// invariant: no whole-archive tempfile, no decompressed RAM spool.
/// Worst case (lots of small files) the second pass re-decompresses
/// nearly the entire archive; in practice tar archives are seek-light
/// enough that the index drives a single linear progression through
/// the source.
fn repack_from_tar_stream_into_fs(
    src: &str,
    algo: fstool::compression::Algo,
    dst: &std::path::Path,
    target_lower: &str,
) -> fstool::Result<()> {
    use fstool::fs::tar::EntryKind as TarKind;
    // ── Pass 1: walk + index + sizing aggregates ──
    let index = build_tar_stream_index(src, algo)?;
    let (size_estimate, total_files, total_dirs, total_symlinks, total_devices, total_bytes) =
        size_from_tar_index(&index, target_lower)?;
    let _ = (
        total_files,
        total_dirs,
        total_symlinks,
        total_devices,
        total_bytes,
    ); // counts kept for reporting only

    let mut dst_dev =
        fstool::block::create_image(dst, size_estimate, &fstool::block::CreateOpts::default())?;

    // ── Pass 2: format + replay ──
    match target_lower {
        "ext2" | "ext3" | "ext4" => {
            let kind = match target_lower {
                "ext2" => fstool::fs::ext::FsKind::Ext2,
                "ext3" => fstool::fs::ext::FsKind::Ext3,
                _ => fstool::fs::ext::FsKind::Ext4,
            };
            let mut plan = fstool::fs::ext::BuildPlan::new(4096, kind);
            for ix in index.entries() {
                match ix.entry.kind {
                    TarKind::Regular | TarKind::HardLink => plan.add_file(ix.entry.size),
                    TarKind::Dir => plan.add_dir(),
                    TarKind::Symlink => plan.add_symlink(
                        ix.entry
                            .link_target
                            .as_deref()
                            .map(|s| s.len())
                            .unwrap_or(0),
                    ),
                    TarKind::CharDev | TarKind::BlockDev | TarKind::Fifo => plan.add_device(),
                }
            }
            let mut opts = plan.to_format_opts();
            // Same sparse-preserve rationale as the FS-to-ext repack
            // path: tar archives can carry long zero runs (sparse-form
            // entries unpacked dense, or just zero-filled files). The
            // writer's all-zero-block check turns them back into
            // holes on the destination.
            opts.sparse = true;
            let mut dst_ext = fstool::fs::ext::Ext::format_with(dst_dev.as_mut(), &opts)?;
            replay_tar_index_into_ext(src, algo, &index, dst_dev.as_mut(), &mut dst_ext)?;
            dst_ext.flush(dst_dev.as_mut())?;
        }
        "fat32" | "vfat" => {
            let total_sectors: u32 = (size_estimate / 512).try_into().map_err(|_| {
                fstool::Error::InvalidArgument(
                    "repack: FAT32 size doesn't fit in a u32 sector count".into(),
                )
            })?;
            let fat_opts = fstool::fs::fat::FatFormatOpts {
                total_sectors,
                volume_id: 0,
                volume_label: *b"REPACKED   ",
            };
            let mut dst_fat = fstool::fs::fat::Fat32::format(dst_dev.as_mut(), &fat_opts)?;
            replay_tar_index_into_fat(src, algo, &index, dst_dev.as_mut(), &mut dst_fat)?;
            dst_fat.flush(dst_dev.as_mut())?;
        }
        other => {
            return Err(fstool::Error::Unsupported(format!(
                "repack: streaming a `.tar.{}` source into a {other} destination is not yet wired",
                algo.name()
            )));
        }
    }
    dst_dev.sync()?;
    eprintln!(
        "repacked {src} → {} (fs: tar.{} → {target_lower}, ~{size_estimate} bytes; two-pass)",
        dst.display(),
        algo.name(),
    );
    Ok(())
}

/// Aggregate the size-relevant counters from a built [`TarStreamIndex`]
/// and return `(size_estimate, files, dirs, symlinks, devices, bytes)`.
/// `target_lower` tunes the size estimate per destination FS.
fn size_from_tar_index(
    index: &fstool::fs::tar::TarStreamIndex,
    target_lower: &str,
) -> fstool::Result<(u64, u64, u64, u64, u64, u64)> {
    use fstool::fs::tar::EntryKind as TarKind;
    let mut files = 0u64;
    let mut dirs = 0u64;
    let mut symlinks = 0u64;
    let mut devices = 0u64;
    let mut bytes = 0u64;
    for ix in index.entries() {
        match ix.entry.kind {
            TarKind::Regular => {
                files += 1;
                bytes += ix.entry.size;
            }
            TarKind::HardLink => {
                files += 1;
                bytes += ix.entry.size;
            }
            TarKind::Dir => dirs += 1,
            TarKind::Symlink => symlinks += 1,
            TarKind::CharDev | TarKind::BlockDev | TarKind::Fifo => devices += 1,
        }
    }
    let size_estimate = match target_lower {
        "ext2" | "ext3" | "ext4" => {
            // Conservative ext sizing: file bytes + dir/inode overhead.
            // We give 4 KiB per inode + 1 MiB structural pad; min 8 MiB.
            let inodes = files + dirs + symlinks + devices + 16;
            let raw = bytes + inodes * 4096 + 1024 * 1024;
            raw.max(8 * 1024 * 1024).div_ceil(4096) * 4096
        }
        "fat32" | "vfat" => {
            // FAT32 needs at least MIN_FAT32_CLUSTERS clusters of 1 KiB
            // overhead per cluster. Double the byte total to leave room
            // for cluster fragmentation + FAT tables + dir entries.
            let needed = bytes
                .saturating_mul(2)
                .max(fstool::fs::fat::MIN_FAT32_CLUSTERS as u64 * 1024);
            needed.div_ceil(512) * 512
        }
        _ => bytes + 16 * 1024 * 1024,
    };
    Ok((size_estimate, files, dirs, symlinks, devices, bytes))
}

/// Pass 2 (ext): replay the indexed entries into a freshly-formatted
/// ext destination, re-decompressing per regular file via
/// `TarStreamIndex::open_body`.
fn replay_tar_index_into_ext(
    src: &str,
    algo: fstool::compression::Algo,
    index: &fstool::fs::tar::TarStreamIndex,
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::ext::Ext,
) -> fstool::Result<()> {
    use fstool::fs::tar::EntryKind as TarKind;
    use fstool::fs::{DeviceKind, FileMeta};
    let mut path_to_ino: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    path_to_ino.insert("/".into(), 2);

    for ix in index.entries() {
        let e = &ix.entry;
        let parent_path = parent_of(&e.path);
        let parent_ino = ensure_ext_dir(dst_dev, dst, &mut path_to_ino, &parent_path)?;
        let leaf = leaf_of(&e.path);
        let meta = FileMeta {
            mode: e.mode & 0o7777,
            uid: e.uid,
            gid: e.gid,
            mtime: e.mtime as u32,
            atime: e.mtime as u32,
            ctime: e.mtime as u32,
        };
        let new_ino = match e.kind {
            TarKind::Regular => {
                let mut body = index.open_body(&e.path, || open_decoded_stream(src, algo))?;
                dst.add_file_to_streaming(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    &mut body,
                    e.size,
                    meta,
                )?
            }
            TarKind::HardLink => {
                // Materialise the linked target's content. Preserving
                // ext hard-link semantics across FS types is out of
                // scope; we copy the bytes instead.
                let mut body = index.open_body(&e.path, || open_decoded_stream(src, algo))?;
                let len = body.remaining();
                dst.add_file_to_streaming(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    &mut body,
                    len,
                    meta,
                )?
            }
            TarKind::Dir => ensure_ext_dir(dst_dev, dst, &mut path_to_ino, &e.path)?,
            TarKind::Symlink => {
                let target = e.link_target.as_deref().unwrap_or("");
                dst.add_symlink_to(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    target.as_bytes(),
                    meta,
                )?
            }
            TarKind::CharDev => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Char,
                e.device_major,
                e.device_minor,
                meta,
            )?,
            TarKind::BlockDev => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Block,
                e.device_major,
                e.device_minor,
                meta,
            )?,
            TarKind::Fifo => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Fifo,
                0,
                0,
                meta,
            )?,
        };
        if matches!(e.kind, TarKind::Dir) {
            path_to_ino.insert(e.path.clone(), new_ino);
        }
        if !e.xattrs.is_empty() {
            dst.set_xattrs(dst_dev, new_ino, &e.xattrs)?;
        }
    }
    Ok(())
}

/// Pass 2 (FAT32): same as the ext replay, minus the metadata FAT
/// can't carry. Entries that aren't regular / dir / hard-link are
/// dropped with a stderr note.
fn replay_tar_index_into_fat(
    src: &str,
    algo: fstool::compression::Algo,
    index: &fstool::fs::tar::TarStreamIndex,
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::fat::Fat32,
) -> fstool::Result<()> {
    use fstool::fs::tar::EntryKind as TarKind;
    let mut made_dirs: std::collections::HashSet<String> =
        std::collections::HashSet::from(["/".into()]);
    for ix in index.entries() {
        let e = &ix.entry;
        let parent = parent_of(&e.path);
        ensure_fat_dir(dst_dev, dst, &mut made_dirs, &parent)?;
        match e.kind {
            TarKind::Regular => {
                let mut body = index.open_body(&e.path, || open_decoded_stream(src, algo))?;
                dst.add_file_from_reader(dst_dev, &e.path, &mut body, e.size)?;
            }
            TarKind::HardLink => {
                let mut body = index.open_body(&e.path, || open_decoded_stream(src, algo))?;
                let len = body.remaining();
                dst.add_file_from_reader(dst_dev, &e.path, &mut body, len)?;
            }
            TarKind::Dir => {
                ensure_fat_dir(dst_dev, dst, &mut made_dirs, &e.path)?;
            }
            _ => {
                eprintln!(
                    "repack: dropping {:?} — FAT32 can't represent {:?}",
                    e.path, e.kind
                );
            }
        }
    }
    Ok(())
}

/// Open a (possibly codec-wrapped) tar archive as a streaming reader.
fn open_tar_stream_reader(
    path: &str,
    algo: Option<fstool::compression::Algo>,
) -> fstool::Result<fstool::fs::tar::TarStreamReader<Box<dyn std::io::Read>>> {
    let p = std::path::Path::new(path.split(':').next().unwrap_or(path));
    let file = std::fs::File::open(p)?;
    let buffered: Box<dyn std::io::Read> =
        Box::new(std::io::BufReader::with_capacity(64 * 1024, file));
    let inner: Box<dyn std::io::Read> = match algo {
        Some(a) => fstool::compression::make_reader(a, buffered)?,
        None => buffered,
    };
    Ok(fstool::fs::tar::TarStreamReader::new(inner))
}

/// Normalise an absolute tar path: starts with '/', no trailing '/'
/// (root is '/'). Matches `Tar::normalise_path`'s output.
fn normalise_tar_path(p: &str) -> String {
    let trimmed = p.trim_end_matches('/');
    if trimmed.is_empty() {
        return "/".into();
    }
    if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{trimmed}")
    }
}

fn parent_of_tar_path(p: &str) -> String {
    match p.rfind('/') {
        Some(0) => "/".to_string(),
        Some(i) => p[..i].to_string(),
        None => "/".to_string(),
    }
}

fn leaf_of_tar_path(p: &str) -> &str {
    match p.rfind('/') {
        Some(i) => &p[i + 1..],
        None => p,
    }
}

/// `ls` for a `.tar.<algo>` (or `.tar`): builds a [`TarStreamIndex`]
/// once, then prints the entries that live directly under `path`.
/// The index is the same one used by `cat`/`info`/`repack`, so the
/// one-pass-decompression cost is amortised whenever a single CLI
/// invocation needs multiple lookups.
fn ls_tar_stream(
    image: &str,
    path: &str,
    algo: Option<fstool::compression::Algo>,
) -> fstool::Result<()> {
    use fstool::fs::tar::EntryKind as TarKind;
    let index = open_tar_stream_index(image, algo)?;
    let want = normalise_tar_path(path);
    let mut children: Vec<fstool::fs::DirEntry> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (idx, ix) in index.entries().iter().enumerate() {
        let p = &ix.entry.path;
        let parent = parent_of_tar_path(p);
        if parent != want {
            continue;
        }
        let leaf = leaf_of_tar_path(p);
        if !seen.insert(leaf.to_string()) {
            continue;
        }
        let dirent_kind = match ix.entry.kind {
            TarKind::Dir => fstool::fs::EntryKind::Dir,
            TarKind::Regular | TarKind::HardLink => fstool::fs::EntryKind::Regular,
            TarKind::Symlink => fstool::fs::EntryKind::Symlink,
            TarKind::CharDev => fstool::fs::EntryKind::Char,
            TarKind::BlockDev => fstool::fs::EntryKind::Block,
            TarKind::Fifo => fstool::fs::EntryKind::Fifo,
        };
        let size = if matches!(dirent_kind, fstool::fs::EntryKind::Regular) {
            ix.entry.size
        } else {
            0
        };
        children.push(fstool::fs::DirEntry {
            name: leaf.to_string(),
            inode: idx as u32 + 1,
            kind: dirent_kind,
            size,
        });
    }
    if children.is_empty() && want != "/" {
        let exists_as_dir = index.entries().iter().any(|ix| ix.entry.path == want);
        let has_descendants = index.entries().iter().any(|ix| {
            let p = &ix.entry.path;
            p.starts_with(&want) && p.len() > want.len() && p.as_bytes()[want.len()] == b'/'
        });
        if !exists_as_dir && !has_descendants {
            return Err(fstool::Error::InvalidArgument(format!(
                "tar: no such directory {want:?}"
            )));
        }
    }
    let mut out = std::io::stdout().lock();
    for e in &children {
        let _ = writeln!(out, "{}\t{:?}\t{}", e.inode, e.kind, e.name);
    }
    Ok(())
}

/// `cat` for a `.tar.<algo>`: build an index, look up the entry, and
/// open a bounded body reader that re-decompresses the source up to
/// the entry's body offset. Hard links resolve transparently via the
/// index (the link target's body bytes are returned).
fn cat_tar_stream(
    image: &str,
    path: &str,
    algo: Option<fstool::compression::Algo>,
) -> fstool::Result<()> {
    let index = open_tar_stream_index(image, algo)?;
    let want = normalise_tar_path(path);
    let mut body = match algo {
        Some(a) => index.open_body(&want, || open_decoded_stream(image, a))?,
        None => index.open_body(&want, || open_decoded_stream_plain(image))?,
    };
    let mut out = std::io::stdout().lock();
    std::io::copy(&mut body, &mut out)?;
    Ok(())
}

/// `info` for a `.tar.<algo>`: drive the same indexer used by `ls`/
/// `cat`/`repack`, then summarise the entry counts and root listing.
fn info_tar_stream(image: &str, algo: Option<fstool::compression::Algo>) -> fstool::Result<()> {
    use fstool::fs::tar::EntryKind as TarKind;
    let index = open_tar_stream_index(image, algo)?;
    let mut total = 0usize;
    let mut files = 0usize;
    let mut dirs = 0usize;
    let mut symlinks = 0usize;
    let mut devices = 0usize;
    let mut content_bytes = 0u64;
    let mut total_xattrs = 0usize;
    let mut top_entries: Vec<(String, fstool::fs::EntryKind, u32)> = Vec::new();
    let mut seen_top: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut idx = 0u32;
    for ix in index.entries() {
        total += 1;
        idx += 1;
        match ix.entry.kind {
            TarKind::Regular | TarKind::HardLink => {
                files += 1;
                content_bytes += ix.entry.size;
            }
            TarKind::Dir => dirs += 1,
            TarKind::Symlink => symlinks += 1,
            TarKind::CharDev | TarKind::BlockDev | TarKind::Fifo => devices += 1,
        }
        total_xattrs += ix.entry.xattrs.len();
        let parent = parent_of_tar_path(&ix.entry.path);
        if parent == "/" {
            let leaf = leaf_of_tar_path(&ix.entry.path).to_string();
            if seen_top.insert(leaf.clone()) {
                let dk = match ix.entry.kind {
                    TarKind::Dir => fstool::fs::EntryKind::Dir,
                    TarKind::Regular | TarKind::HardLink => fstool::fs::EntryKind::Regular,
                    TarKind::Symlink => fstool::fs::EntryKind::Symlink,
                    TarKind::CharDev => fstool::fs::EntryKind::Char,
                    TarKind::BlockDev => fstool::fs::EntryKind::Block,
                    TarKind::Fifo => fstool::fs::EntryKind::Fifo,
                };
                top_entries.push((leaf, dk, idx));
            }
        }
    }
    println!("fs kind:           tar (streaming)");
    println!("entries:           {total}");
    println!("  files:           {files}");
    println!("  directories:     {dirs}");
    println!("  symlinks:        {symlinks}");
    println!("  devices/fifos:   {devices}");
    println!("file content:      {content_bytes} bytes");
    println!("xattrs total:      {total_xattrs}");
    println!();
    println!("/ listing:");
    for (name, kind, ino) in &top_entries {
        println!("  {:>10}  {:?}  {}", ino, kind, name);
    }
    Ok(())
}

/// Open the tar source (optionally codec-wrapped) and build a
/// random-access index over it. Shared entry point for the
/// streaming-tar inspector commands.
fn open_tar_stream_index(
    image: &str,
    algo: Option<fstool::compression::Algo>,
) -> fstool::Result<fstool::fs::tar::TarStreamIndex> {
    let reader = open_tar_stream_reader(image, algo)?;
    fstool::fs::tar::TarStreamIndex::build_from(reader)
}

/// Open a plain (uncompressed) tar source as a boxed `Read`. Returned
/// from the `factory` closure passed to `TarStreamIndex::open_body`
/// when the source isn't codec-wrapped.
fn open_decoded_stream_plain(image: &str) -> fstool::Result<Box<dyn std::io::Read>> {
    let p = std::path::Path::new(image.split(':').next().unwrap_or(image));
    let file = std::fs::File::open(p)?;
    Ok(Box::new(std::io::BufReader::with_capacity(64 * 1024, file)))
}

fn unsupported_repack_src(src_fs: &fstool::inspect::AnyFs) -> fstool::Error {
    fstool::Error::Unsupported(format!(
        "repack: {} source is not yet wired into the FS-to-FS copy path (it's inspectable via `ls`/`cat`/`info` but can't yet be a repack source)",
        src_fs.kind_string()
    ))
}

// ─── ext → ext (full metadata preservation) ─────────────────────────────

fn copy_ext_dir(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::ext::Ext,
    src_ino: u32,
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::ext::Ext,
    dst_ino: u32,
) -> fstool::Result<()> {
    let mut link_map = std::collections::HashMap::new();
    copy_ext_dir_at(
        src_dev,
        src,
        src_ino,
        dst_dev,
        dst,
        dst_ino,
        "",
        &mut link_map,
    )
}

fn copy_ext_dir_at(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::ext::Ext,
    src_ino: u32,
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::ext::Ext,
    dst_ino: u32,
    parent_path: &str,
    link_map: &mut std::collections::HashMap<u32, u32>,
) -> fstool::Result<()> {
    use fstool::fs::ext::dir as ext_dir;
    use fstool::fs::ext::inode::decode_devnum;
    use fstool::fs::{DeviceKind, FileMeta};

    let entries = src.list_inode(src_dev, src_ino)?;
    for e in entries {
        if e.name == "." || e.name == ".." || (src_ino == 2 && e.name == "lost+found") {
            continue;
        }
        let inode = src.read_inode(src_dev, e.inode)?;
        let meta = FileMeta {
            mode: inode.mode & 0o7777,
            uid: inode.uid as u32,
            gid: inode.gid as u32,
            mtime: inode.mtime,
            atime: inode.atime,
            ctime: inode.ctime,
        };
        let name = e.name.as_bytes();
        let mode_type = inode.mode & fstool::fs::ext::constants::S_IFMT;
        // Read source xattrs once per entry; preserve them across the
        // create + (optional) recursion.
        let xattrs = src.read_xattrs(src_dev, e.inode)?;
        let mut entry_path = String::with_capacity(parent_path.len() + 1 + e.name.len());
        entry_path.push_str(parent_path);
        entry_path.push('/');
        entry_path.push_str(&e.name);
        fstool::repack::note(&entry_path);

        // Hard-link short circuit: subsequent encounters of a
        // multi-link source inode reuse the dst inode that was
        // created the first time, instead of duplicating the file
        // body. Skipped for directories (POSIX disallows dir links).
        if inode.links_count > 1
            && mode_type != fstool::fs::ext::constants::S_IFDIR
            && let Some(&existing_dst) = link_map.get(&e.inode)
        {
            dst.add_link_to(dst_dev, dst_ino, name, existing_dst)?;
            continue;
        }

        let new_ino = match mode_type {
            t if t == fstool::fs::ext::constants::S_IFREG => {
                let mut reader = src.open_file_reader(src_dev, e.inode)?;
                dst.add_file_to_streaming(
                    dst_dev,
                    dst_ino,
                    name,
                    &mut reader,
                    inode.size as u64,
                    meta,
                )?
            }
            t if t == fstool::fs::ext::constants::S_IFDIR => {
                // For ext4 destinations with enough children, emit an
                // HTree (DIR_INDEX) directory; otherwise pre-size the
                // dir with one contiguous extent and use a plain
                // unindexed layout.
                let child_entries = src.list_inode(src_dev, e.inode).unwrap_or_default();
                let real_children: Vec<&fstool::fs::DirEntry> = child_entries
                    .iter()
                    .filter(|c| c.name != "." && c.name != "..")
                    .collect();
                let use_htree = matches!(dst.kind, fstool::fs::ext::FsKind::Ext4)
                    && real_children.len() >= 250;
                let child_ino = if use_htree {
                    let names: Vec<&[u8]> =
                        real_children.iter().map(|c| c.name.as_bytes()).collect();
                    dst.add_dir_indexed(dst_dev, dst_ino, name, meta, &names)?
                } else {
                    let mut bytes: usize = 24;
                    for c in &real_children {
                        bytes += ext_dir::min_rec_len(c.name.len());
                    }
                    let usable = ext_dir::usable_dir_len(
                        dst.layout.block_size,
                        dst.has_metadata_csum_pub(),
                    );
                    let child_blocks = bytes.div_ceil(usable).max(1) as u32;
                    dst.add_dir_to_with_blocks(dst_dev, dst_ino, name, meta, child_blocks)?
                };
                copy_ext_dir_at(
                    src_dev, src, e.inode, dst_dev, dst, child_ino, &entry_path, link_map,
                )?;
                child_ino
            }
            t if t == fstool::fs::ext::constants::S_IFLNK => {
                let target = src.read_symlink_target(src_dev, e.inode)?;
                dst.add_symlink_to(dst_dev, dst_ino, name, target.as_bytes(), meta)?
            }
            t if t == fstool::fs::ext::constants::S_IFCHR => {
                let (major, minor) = decode_devnum(inode.block[0]);
                dst.add_device_to(dst_dev, dst_ino, name, DeviceKind::Char, major, minor, meta)?
            }
            t if t == fstool::fs::ext::constants::S_IFBLK => {
                let (major, minor) = decode_devnum(inode.block[0]);
                dst.add_device_to(
                    dst_dev,
                    dst_ino,
                    name,
                    DeviceKind::Block,
                    major,
                    minor,
                    meta,
                )?
            }
            t if t == fstool::fs::ext::constants::S_IFIFO => {
                dst.add_device_to(dst_dev, dst_ino, name, DeviceKind::Fifo, 0, 0, meta)?
            }
            t if t == fstool::fs::ext::constants::S_IFSOCK => {
                dst.add_device_to(dst_dev, dst_ino, name, DeviceKind::Socket, 0, 0, meta)?
            }
            _ => {
                eprintln!(
                    "repack: skipping inode {} ({:?}) — unknown mode {:#o}",
                    e.inode, e.name, inode.mode
                );
                continue;
            }
        };
        if !xattrs.is_empty() {
            dst.set_xattrs(dst_dev, new_ino, &xattrs)?;
        }
        // Record src→dst for multi-link non-directory inodes so the
        // hard-link short circuit above catches the next occurrence.
        if inode.links_count > 1 && mode_type != fstool::fs::ext::constants::S_IFDIR {
            link_map.insert(e.inode, new_ino);
        }
    }
    Ok(())
}

// ─── FAT32 → FAT32 ──────────────────────────────────────────────────────

fn copy_fat_dir_into_fat(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::fat::Fat32,
    src_path: &str,
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::fat::Fat32,
) -> fstool::Result<()> {
    use fstool::fs::EntryKind;
    let entries = src.list_path(src_dev, src_path)?;
    for e in entries {
        let child = join_fs_path(src_path, &e.name);
        match e.kind {
            EntryKind::Dir => {
                dst.add_dir(dst_dev, &child)?;
                copy_fat_dir_into_fat(src_dev, src, &child, dst_dev, dst)?;
            }
            EntryKind::Regular => {
                // Resolve the source entry to get its actual file_size.
                let (entry, _) = src.resolve_entry(src_dev, &child)?;
                let mut reader = src.open_file_reader(src_dev, &child)?;
                dst.add_file_from_reader(dst_dev, &child, &mut reader, entry.file_size as u64)?;
            }
            _ => {} // FAT can't carry anything else
        }
    }
    Ok(())
}

// ─── ext → FAT32 (drops metadata FAT can't store) ───────────────────────

fn copy_ext_dir_into_fat(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::ext::Ext,
    src_ino: u32,
    cur_path: &str,
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::fat::Fat32,
) -> fstool::Result<()> {
    let entries = src.list_inode(src_dev, src_ino)?;
    for e in entries {
        if e.name == "." || e.name == ".." || (src_ino == 2 && e.name == "lost+found") {
            continue;
        }
        let inode = src.read_inode(src_dev, e.inode)?;
        let mode_type = inode.mode & fstool::fs::ext::constants::S_IFMT;
        let child = join_fs_path(cur_path, &e.name);
        match mode_type {
            t if t == fstool::fs::ext::constants::S_IFREG => {
                let mut reader = src.open_file_reader(src_dev, e.inode)?;
                dst.add_file_from_reader(dst_dev, &child, &mut reader, inode.size as u64)?;
            }
            t if t == fstool::fs::ext::constants::S_IFDIR => {
                dst.add_dir(dst_dev, &child)?;
                copy_ext_dir_into_fat(src_dev, src, e.inode, &child, dst_dev, dst)?;
            }
            t if t == fstool::fs::ext::constants::S_IFLNK
                || t == fstool::fs::ext::constants::S_IFCHR
                || t == fstool::fs::ext::constants::S_IFBLK
                || t == fstool::fs::ext::constants::S_IFIFO
                || t == fstool::fs::ext::constants::S_IFSOCK =>
            {
                eprintln!(
                    "repack: dropping {child:?} ({:?}) — FAT32 can't represent it",
                    fstool_mode_kind(mode_type)
                );
            }
            _ => {}
        }
    }
    Ok(())
}

// ─── FAT32 → ext ────────────────────────────────────────────────────────

fn copy_fat_dir_into_ext(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::fat::Fat32,
    src_path: &str,
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::ext::Ext,
    dst_ino: u32,
    meta: &fstool::fs::FileMeta,
) -> fstool::Result<()> {
    use fstool::fs::EntryKind;
    let entries = src.list_path(src_dev, src_path)?;
    for e in entries {
        let child = join_fs_path(src_path, &e.name);
        match e.kind {
            EntryKind::Dir => {
                let new_ino = dst.add_dir_to(dst_dev, dst_ino, e.name.as_bytes(), *meta)?;
                copy_fat_dir_into_ext(src_dev, src, &child, dst_dev, dst, new_ino, meta)?;
            }
            EntryKind::Regular => {
                let (entry, _) = src.resolve_entry(src_dev, &child)?;
                let mut reader = src.open_file_reader(src_dev, &child)?;
                dst.add_file_to_streaming(
                    dst_dev,
                    dst_ino,
                    e.name.as_bytes(),
                    &mut reader,
                    entry.file_size as u64,
                    *meta,
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

// ─── Tar → ext ──────────────────────────────────────────────────────────

/// Replay a tar archive's entries into a fresh ext destination.
/// Preserves mode, uid/gid, mtime, symlinks, device nodes, and xattrs.
fn copy_tar_into_ext(
    src_dev: &mut dyn fstool::block::BlockDevice,
    tar: &fstool::fs::tar::Tar,
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::ext::Ext,
) -> fstool::Result<()> {
    use fstool::fs::tar::EntryKind;
    use fstool::fs::{DeviceKind, FileMeta};
    // Map every absolute path in the tar to its destination inode,
    // creating ancestor dirs on demand so an entry can land before its
    // parent dir appears in the archive.
    let mut path_to_ino: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    path_to_ino.insert("/".into(), 2);

    let entries: Vec<fstool::fs::tar::Entry> = tar.entries().to_vec();
    for e in entries {
        let parent_path = parent_of(&e.path);
        let parent_ino = ensure_ext_dir(dst_dev, dst, &mut path_to_ino, &parent_path)?;
        let leaf = leaf_of(&e.path);
        let meta = FileMeta {
            mode: e.mode & 0o7777,
            uid: e.uid,
            gid: e.gid,
            mtime: e.mtime as u32,
            atime: e.mtime as u32,
            ctime: e.mtime as u32,
        };
        let new_ino = match e.kind {
            EntryKind::Regular => {
                let mut reader = tar.open_file_reader(src_dev, &e.path)?;
                dst.add_file_to_streaming(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    &mut reader,
                    e.size,
                    meta,
                )?
            }
            EntryKind::Dir => {
                // ensure_ext_dir already creates it if missing; we just
                // need its inode.
                ensure_ext_dir(dst_dev, dst, &mut path_to_ino, &e.path)?
            }
            EntryKind::Symlink => {
                let target = e.link_target.as_deref().unwrap_or("");
                dst.add_symlink_to(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    target.as_bytes(),
                    meta,
                )?
            }
            EntryKind::HardLink => {
                // Materialise the link target's content again. Preserves
                // file content across the conversion at the cost of a
                // copy; preserving the link itself across FS types is
                // out of scope.
                let target = e.link_target.as_deref().unwrap_or("");
                let abs_target = if target.starts_with('/') {
                    target.to_string()
                } else {
                    format!("/{target}")
                };
                let target_entry = tar.lookup(&abs_target).ok_or_else(|| {
                    fstool::Error::InvalidImage(format!(
                        "tar: hard link {:?} → {abs_target:?} (target missing)",
                        e.path
                    ))
                })?;
                let mut reader = tar.open_file_reader(src_dev, &abs_target)?;
                dst.add_file_to_streaming(
                    dst_dev,
                    parent_ino,
                    leaf.as_bytes(),
                    &mut reader,
                    target_entry.size,
                    meta,
                )?
            }
            EntryKind::CharDev => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Char,
                e.device_major,
                e.device_minor,
                meta,
            )?,
            EntryKind::BlockDev => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Block,
                e.device_major,
                e.device_minor,
                meta,
            )?,
            EntryKind::Fifo => dst.add_device_to(
                dst_dev,
                parent_ino,
                leaf.as_bytes(),
                DeviceKind::Fifo,
                0,
                0,
                meta,
            )?,
        };
        if matches!(e.kind, EntryKind::Dir) {
            path_to_ino.insert(e.path.clone(), new_ino);
        }
        if !e.xattrs.is_empty() {
            dst.set_xattrs(dst_dev, new_ino, &e.xattrs)?;
        }
    }
    Ok(())
}

fn ensure_ext_dir(
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::ext::Ext,
    path_to_ino: &mut std::collections::HashMap<String, u32>,
    path: &str,
) -> fstool::Result<u32> {
    use fstool::fs::FileMeta;
    if let Some(&ino) = path_to_ino.get(path) {
        return Ok(ino);
    }
    let parent = parent_of(path);
    let parent_ino = ensure_ext_dir(dst_dev, dst, path_to_ino, &parent)?;
    let leaf = leaf_of(path);
    let meta = FileMeta {
        mode: 0o755,
        ..FileMeta::default()
    };
    let ino = dst.add_dir_to(dst_dev, parent_ino, leaf.as_bytes(), meta)?;
    path_to_ino.insert(path.to_string(), ino);
    Ok(ino)
}

// ─── Tar → FAT32 ────────────────────────────────────────────────────────

fn copy_tar_into_fat(
    src_dev: &mut dyn fstool::block::BlockDevice,
    tar: &fstool::fs::tar::Tar,
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::fat::Fat32,
) -> fstool::Result<()> {
    use fstool::fs::tar::EntryKind;
    let mut made_dirs: std::collections::HashSet<String> =
        std::collections::HashSet::from(["/".into()]);
    let entries: Vec<fstool::fs::tar::Entry> = tar.entries().to_vec();
    for e in entries {
        let parent = parent_of(&e.path);
        ensure_fat_dir(dst_dev, dst, &mut made_dirs, &parent)?;
        match e.kind {
            EntryKind::Regular => {
                let mut reader = tar.open_file_reader(src_dev, &e.path)?;
                dst.add_file_from_reader(dst_dev, &e.path, &mut reader, e.size)?;
            }
            EntryKind::Dir => {
                ensure_fat_dir(dst_dev, dst, &mut made_dirs, &e.path)?;
            }
            EntryKind::HardLink => {
                let target = e.link_target.as_deref().unwrap_or("");
                let abs_target = if target.starts_with('/') {
                    target.to_string()
                } else {
                    format!("/{target}")
                };
                if let Some(target_entry) = tar.lookup(&abs_target) {
                    let mut reader = tar.open_file_reader(src_dev, &abs_target)?;
                    dst.add_file_from_reader(dst_dev, &e.path, &mut reader, target_entry.size)?;
                }
            }
            _ => {
                eprintln!(
                    "repack: dropping {:?} — FAT32 can't represent {:?}",
                    e.path, e.kind
                );
            }
        }
    }
    Ok(())
}

fn ensure_fat_dir(
    dst_dev: &mut dyn fstool::block::BlockDevice,
    dst: &mut fstool::fs::fat::Fat32,
    made: &mut std::collections::HashSet<String>,
    path: &str,
) -> fstool::Result<()> {
    if made.contains(path) {
        return Ok(());
    }
    let parent = parent_of(path);
    ensure_fat_dir(dst_dev, dst, made, &parent)?;
    dst.add_dir(dst_dev, path)?;
    made.insert(path.to_string());
    Ok(())
}

fn parent_of(path: &str) -> String {
    let p = path.trim_end_matches('/');
    match p.rfind('/') {
        Some(0) | None => "/".into(),
        Some(i) => p[..i].into(),
    }
}

fn leaf_of(path: &str) -> &str {
    let p = path.trim_end_matches('/');
    p.rsplit('/').next().unwrap_or(p)
}

fn join_fs_path(parent: &str, leaf: &str) -> String {
    if parent.ends_with('/') {
        format!("{parent}{leaf}")
    } else {
        format!("{parent}/{leaf}")
    }
}

fn fstool_mode_kind(mode_type: u16) -> &'static str {
    use fstool::fs::ext::constants::*;
    match mode_type {
        t if t == S_IFLNK => "symlink",
        t if t == S_IFCHR => "char-device",
        t if t == S_IFBLK => "block-device",
        t if t == S_IFIFO => "fifo",
        t if t == S_IFSOCK => "socket",
        _ => "other",
    }
}

// ─── shrink sizing ───────────────────────────────────────────────────────

/// Build a BuildPlan that reflects the source filesystem's content,
/// without touching the host filesystem. Trait-driven via
/// [`fstool::repack::scan_into_build_plan`] — no per-FS arms.
fn build_ext_plan(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src_fs: &mut fstool::inspect::AnyFs,
    block_size: u32,
    fs_kind_str: &str,
) -> fstool::Result<fstool::fs::ext::BuildPlan> {
    use fstool::fs::ext::{BuildPlan, FsKind};
    let kind = match fs_kind_str {
        "ext2" => FsKind::Ext2,
        "ext3" => FsKind::Ext3,
        "ext4" => FsKind::Ext4,
        other => {
            return Err(fstool::Error::InvalidArgument(format!(
                "repack: unknown ext kind {other:?}"
            )));
        }
    };
    let mut plan = BuildPlan::new(block_size, kind);
    fstool::repack::build_ext_plan_through_trait(src_dev, src_fs, &mut plan)?;
    Ok(plan)
}

/// Sum the size of every regular file in the source filesystem — used
/// by FAT32 / ISO shrink sizing. Trait-driven walk via
/// [`fstool::inspect::AnyFs::total_file_bytes`].
fn sum_source_file_bytes(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src_fs: &mut fstool::inspect::AnyFs,
) -> fstool::Result<u64> {
    src_fs.total_file_bytes(src_dev)
}

/// Upper-bound the size of a tar archive built from `src_fs`. Walks the
/// source once, accumulating header + content + worst-case PAX overhead
/// for each entry, plus a 1 KiB safety pad. The actual archive almost
/// always comes out smaller; the destination file is truncated to the
/// real length after the write.
fn tar_size_upper_bound(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src_fs: &fstool::inspect::AnyFs,
) -> fstool::Result<u64> {
    use fstool::inspect::AnyFs;
    let mut total: u64 = 0;
    // Conservative per-entry header allowance: 512 (ustar) + 2 * 512
    // (one PAX header + body) + a generous xattr payload buffer.
    let per_entry_overhead = |xattr_bytes: u64| 1536 + xattr_bytes + 512;
    match src_fs {
        AnyFs::Ext(src_ext) => {
            tar_size_walk_ext(src_dev, src_ext, 2, &mut total, &per_entry_overhead)?;
        }
        AnyFs::Fat32(src_fat) => {
            tar_size_walk_fat(src_dev, src_fat, "/", &mut total, &per_entry_overhead)?;
        }
        AnyFs::Tar(src_tar) => {
            for e in src_tar.entries() {
                let xb: u64 = e
                    .xattrs
                    .iter()
                    .map(|x| (x.name.len() + x.value.len()) as u64)
                    .sum();
                let content = if matches!(e.kind, fstool::fs::tar::EntryKind::Regular) {
                    (e.size + 511) & !511
                } else {
                    0
                };
                total += per_entry_overhead(xb) + content;
            }
        }
        AnyFs::Grf(src_grf) => {
            // GRF stores file *sizes* uncompressed in the entry
            // table — no need to inflate to size the output tar.
            // No xattrs, so the per-entry overhead is constant.
            for entry in src_grf.entries.values() {
                let content = (u64::from(entry.size) + 511) & !511;
                total += per_entry_overhead(0) + content;
            }
        }
        _ => return Err(unsupported_repack_src(src_fs)),
    }
    // Two zero blocks for EOF + 1 KiB pad.
    Ok(total + 1024 + 1024)
}

fn tar_size_walk_ext(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::ext::Ext,
    src_ino: u32,
    total: &mut u64,
    overhead: &dyn Fn(u64) -> u64,
) -> fstool::Result<()> {
    for e in src.list_inode(src_dev, src_ino)? {
        if e.name == "." || e.name == ".." || (src_ino == 2 && e.name == "lost+found") {
            continue;
        }
        let inode = src.read_inode(src_dev, e.inode)?;
        let xattrs = src.read_xattrs(src_dev, e.inode)?;
        let xb: u64 = xattrs
            .iter()
            .map(|x| (x.name.len() + x.value.len()) as u64)
            .sum();
        let mode_type = inode.mode & fstool::fs::ext::constants::S_IFMT;
        let content = if mode_type == fstool::fs::ext::constants::S_IFREG {
            ((inode.size as u64) + 511) & !511
        } else {
            0
        };
        *total += overhead(xb) + content;
        if mode_type == fstool::fs::ext::constants::S_IFDIR {
            tar_size_walk_ext(src_dev, src, e.inode, total, overhead)?;
        }
    }
    Ok(())
}

fn tar_size_walk_fat(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::fat::Fat32,
    src_path: &str,
    total: &mut u64,
    overhead: &dyn Fn(u64) -> u64,
) -> fstool::Result<()> {
    use fstool::fs::EntryKind;
    for e in src.list_path(src_dev, src_path)? {
        let child = join_fs_path(src_path, &e.name);
        match e.kind {
            EntryKind::Dir => {
                *total += overhead(0);
                tar_size_walk_fat(src_dev, src, &child, total, overhead)?;
            }
            EntryKind::Regular => {
                let (entry, _) = src.resolve_entry(src_dev, &child)?;
                *total += overhead(0) + (((entry.file_size as u64) + 511) & !511);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Walk every entry in `src_fs` and emit it as a tar archive directly
/// to `dst` (optionally codec-wrapped). Returns the number of plain
/// (uncompressed) bytes written through the writer; the on-disk file
/// is whatever the codec produces.
///
/// Streaming output path: the destination file is opened once, wrapped
/// in a compressing writer if needed, and never seeked. No tempfile.
fn repack_into_tar_streaming(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src_fs: &fstool::inspect::AnyFs,
    dst: &std::path::Path,
    codec: Option<fstool::compression::Algo>,
) -> fstool::Result<u64> {
    use fstool::fs::tar::TarStreamWriter;
    let file = std::fs::File::create(dst)?;
    let buffered: Box<dyn std::io::Write> =
        Box::new(std::io::BufWriter::with_capacity(64 * 1024, file));
    let inner: Box<dyn std::io::Write> = match codec {
        Some(algo) => fstool::compression::make_writer(algo, buffered)?,
        None => buffered,
    };
    let mut writer = TarStreamWriter::new(inner);
    repack_walk_into_sink(src_dev, src_fs, &mut writer)?;
    writer.finish()?;
    let written = writer.bytes_written();
    drop(writer);
    Ok(written)
}

/// Walk every entry in `src_fs` and emit it through the given
/// [`TarSink`]. Used by the streaming output path; the BlockDevice-
/// backed `TarWriter` is no longer wired into the CLI for tar output.
fn repack_walk_into_sink(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src_fs: &fstool::inspect::AnyFs,
    sink: &mut dyn fstool::fs::tar::TarSink,
) -> fstool::Result<()> {
    use fstool::inspect::AnyFs;
    match src_fs {
        AnyFs::Ext(src_ext) => tar_walk_ext(src_dev, src_ext, 2, "", sink),
        AnyFs::Fat32(src_fat) => tar_walk_fat(src_dev, src_fat, "/", sink),
        AnyFs::Tar(src_tar) => tar_replay_tar(src_dev, src_tar, sink),
        AnyFs::Grf(src_grf) => tar_walk_grf(src_dev, src_grf, sink),
        _ => Err(unsupported_repack_src(src_fs)),
    }
}

fn tar_walk_grf(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::grf::Grf,
    writer: &mut dyn fstool::fs::tar::TarSink,
) -> fstool::Result<()> {
    use fstool::fs::tar::TarEntryMeta;
    // GRF entries are flat archives — no POSIX metadata, no
    // directory entries (parents are implicit from `/`-separated
    // path components). We synthesise tar dir entries for each
    // unique parent so the output is well-formed.
    let mut emitted_dirs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let meta = TarEntryMeta {
        mode: 0o644,
        uid: 0,
        gid: 0,
        mtime: 0,
        uname: String::new(),
        gname: String::new(),
    };
    let dir_meta = TarEntryMeta {
        mode: 0o755,
        ..meta.clone()
    };
    for (name, entry) in &src.entries {
        let path = format!("/{name}");
        fstool::repack::note(&path);
        // Emit each prefix directory exactly once.
        let mut acc = String::new();
        for part in name.split('/').collect::<Vec<_>>().split_last().unwrap().1 {
            if !acc.is_empty() {
                acc.push('/');
            }
            acc.push_str(part);
            if emitted_dirs.insert(acc.clone()) {
                writer.add_dir(&format!("/{acc}"), dir_meta.clone(), &[])?;
            }
        }
        let bytes = src.read_entry(src_dev, entry)?;
        let len = bytes.len() as u64;
        let mut reader = std::io::Cursor::new(bytes);
        writer.add_file(&path, &mut reader, len, meta.clone(), &[])?;
    }
    Ok(())
}

fn tar_walk_ext(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::ext::Ext,
    src_ino: u32,
    prefix: &str,
    writer: &mut dyn fstool::fs::tar::TarSink,
) -> fstool::Result<()> {
    use fstool::fs::DeviceKind;
    use fstool::fs::ext::constants::*;
    use fstool::fs::ext::inode::decode_devnum;
    use fstool::fs::tar::TarEntryMeta;
    for e in src.list_inode(src_dev, src_ino)? {
        if e.name == "." || e.name == ".." || (src_ino == 2 && e.name == "lost+found") {
            continue;
        }
        let inode = src.read_inode(src_dev, e.inode)?;
        let xattrs = src.read_xattrs(src_dev, e.inode)?;
        let path = if prefix.is_empty() {
            format!("/{}", e.name)
        } else {
            format!("{prefix}/{}", e.name)
        };
        fstool::repack::note(&path);
        let meta = TarEntryMeta {
            mode: inode.mode & 0o7777,
            uid: inode.uid as u32,
            gid: inode.gid as u32,
            mtime: inode.mtime as u64,
            uname: String::new(),
            gname: String::new(),
        };
        let mode_type = inode.mode & S_IFMT;
        match mode_type {
            t if t == S_IFREG => {
                let mut reader = src.open_file_reader(src_dev, e.inode)?;
                writer.add_file(&path, &mut reader, inode.size as u64, meta, &xattrs)?;
            }
            t if t == S_IFDIR => {
                writer.add_dir(&path, meta, &xattrs)?;
                tar_walk_ext(src_dev, src, e.inode, &path, writer)?;
            }
            t if t == S_IFLNK => {
                let target = src.read_symlink_target(src_dev, e.inode)?;
                writer.add_symlink(&path, &target, meta, &xattrs)?;
            }
            t if t == S_IFCHR => {
                let (maj, min) = decode_devnum(inode.block[0]);
                writer.add_device(&path, DeviceKind::Char, maj, min, meta, &xattrs)?;
            }
            t if t == S_IFBLK => {
                let (maj, min) = decode_devnum(inode.block[0]);
                writer.add_device(&path, DeviceKind::Block, maj, min, meta, &xattrs)?;
            }
            t if t == S_IFIFO => {
                writer.add_device(&path, DeviceKind::Fifo, 0, 0, meta, &xattrs)?;
            }
            _ => {
                eprintln!(
                    "repack: skipping {path:?} — unsupported mode {:#o}",
                    inode.mode
                );
            }
        }
    }
    Ok(())
}

fn tar_walk_fat(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::fat::Fat32,
    src_path: &str,
    writer: &mut dyn fstool::fs::tar::TarSink,
) -> fstool::Result<()> {
    use fstool::fs::EntryKind;
    use fstool::fs::tar::TarEntryMeta;
    for e in src.list_path(src_dev, src_path)? {
        let child = join_fs_path(src_path, &e.name);
        fstool::repack::note(&child);
        let meta = TarEntryMeta::default();
        match e.kind {
            EntryKind::Dir => {
                writer.add_dir(&child, meta, &[])?;
                tar_walk_fat(src_dev, src, &child, writer)?;
            }
            EntryKind::Regular => {
                let (entry, _) = src.resolve_entry(src_dev, &child)?;
                let mut reader = src.open_file_reader(src_dev, &child)?;
                writer.add_file(&child, &mut reader, entry.file_size as u64, meta, &[])?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn tar_replay_tar(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::tar::Tar,
    writer: &mut dyn fstool::fs::tar::TarSink,
) -> fstool::Result<()> {
    use fstool::fs::DeviceKind;
    use fstool::fs::tar::{EntryKind, TarEntryMeta};
    let entries: Vec<fstool::fs::tar::Entry> = src.entries().to_vec();
    for e in entries {
        fstool::repack::note(&e.path);
        let meta = TarEntryMeta {
            mode: e.mode,
            uid: e.uid,
            gid: e.gid,
            mtime: e.mtime,
            uname: String::new(),
            gname: String::new(),
        };
        match e.kind {
            EntryKind::Regular => {
                let mut reader = src.open_file_reader(src_dev, &e.path)?;
                writer.add_file(&e.path, &mut reader, e.size, meta, &e.xattrs)?;
            }
            EntryKind::Dir => writer.add_dir(&e.path, meta, &e.xattrs)?,
            EntryKind::Symlink => {
                let target = e.link_target.as_deref().unwrap_or("");
                writer.add_symlink(&e.path, target, meta, &e.xattrs)?;
            }
            EntryKind::HardLink => {
                // Preserve content for the link's apparent file.
                let target = e.link_target.as_deref().unwrap_or("");
                let abs = if target.starts_with('/') {
                    target.to_string()
                } else {
                    format!("/{target}")
                };
                if let Some(target_entry) = src.lookup(&abs) {
                    let mut reader = src.open_file_reader(src_dev, &abs)?;
                    writer.add_file(&e.path, &mut reader, target_entry.size, meta, &e.xattrs)?;
                }
            }
            EntryKind::CharDev => {
                writer.add_device(
                    &e.path,
                    DeviceKind::Char,
                    e.device_major,
                    e.device_minor,
                    meta,
                    &e.xattrs,
                )?;
            }
            EntryKind::BlockDev => {
                writer.add_device(
                    &e.path,
                    DeviceKind::Block,
                    e.device_major,
                    e.device_minor,
                    meta,
                    &e.xattrs,
                )?;
            }
            EntryKind::Fifo => {
                writer.add_device(&e.path, DeviceKind::Fifo, 0, 0, meta, &e.xattrs)?;
            }
        }
    }
    Ok(())
}

fn shell_cmd(image: &str) -> fstool::Result<()> {
    let target = fstool::inspect::Target::parse(image);
    fstool::inspect::with_target_device(&target, |dev| {
        let fs = fstool::inspect::AnyFs::open(dev)?;
        let mut sh = shell::Shell::new(fs);
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        sh.run(dev, stdin.lock(), stdout.lock())?;
        dev.sync()?;
        Ok(())
    })
}

fn rm(image: &str, fs_path: &str) -> fstool::Result<()> {
    let target = fstool::inspect::Target::parse(image);
    fstool::inspect::with_target_device(&target, |dev| {
        let mut fs = fstool::inspect::AnyFs::open(dev)?;
        fs.remove(dev, fs_path)?;
        fs.flush(dev)?;
        dev.sync()?;
        eprintln!("removed {fs_path}");
        Ok(())
    })
}

fn add(image: &str, host_src: &std::path::Path, fs_dest: &str) -> fstool::Result<()> {
    let meta = std::fs::symlink_metadata(host_src)?;
    let target = fstool::inspect::Target::parse(image);
    fstool::inspect::with_target_device(&target, |dev| {
        let mut fs = fstool::inspect::AnyFs::open(dev)?;
        if meta.is_dir() {
            fs.add_dir_tree(dev, fs_dest, host_src)?;
        } else if meta.is_file() {
            fs.add_file(dev, fs_dest, host_src)?;
        } else {
            return Err(fstool::Error::InvalidArgument(format!(
                "add: {} is neither a regular file nor a directory",
                host_src.display()
            )));
        }
        fs.flush(dev)?;
        dev.sync()?;
        eprintln!("added {} → {fs_dest}", host_src.display());
        Ok(())
    })
}

fn build(spec_path: &std::path::Path, output: &std::path::Path) -> fstool::Result<()> {
    let spec = fstool::spec::Spec::parse_file(spec_path)?;
    fstool::spec::build(&spec, output)?;
    eprintln!("built {} from {}", output.display(), spec_path.display());
    Ok(())
}

/// Group of flag values plumbed into [`create_cmd`]. Keeps the
/// callsite in `run` tidy and avoids a 10-parameter function signature.
struct CreateArgs<'a> {
    fs_type: &'a str,
    src_dir: Option<&'a std::path::Path>,
    output: &'a std::path::Path,
    size: Option<&'a str>,
    label: Option<&'a str>,
    force: bool,
    cluster_size: &'a str,
    options: &'a [String],
}

/// `fstool create` — format a fresh filesystem image and optionally
/// populate it from a host directory tree. The fs-type chooses the
/// formatter; `-O key=val[,key=val]…` ferries FS-specific knobs through
/// the [`OptionMap`] surface.
fn create_cmd(args: CreateArgs<'_>) -> fstool::Result<()> {
    use fstool::block::file::is_block_device;

    let fs_type = args.fs_type.to_ascii_lowercase();
    let is_device = is_block_device(args.output);
    require_force_for_device(args.output, is_device, args.force)?;
    let qcow2_cluster_size = parse_cluster_size(args.cluster_size)?;

    // Build the option bag. `--label` is a shortcut for the standard
    // `volume_label` key so the same flag works for every FS that
    // accepts a label (even when the underlying field has a different
    // name — each backend's `apply_options` translates).
    let mut opts = OptionMap::new();
    for o in args.options {
        opts.merge_cli(o)?;
    }
    if let Some(label) = args.label {
        opts.insert("volume_label", label);
    }

    // Resolve an optional source. The destination size is derived from
    // it where the FS supports auto-sizing; otherwise --size wins.
    let source = args
        .src_dir
        .map(|p| fstool::repack::Source::HostDir(p.to_path_buf()));

    // Per-FS dispatch. Each arm consumes the relevant keys from `opts`,
    // calls `check_empty`, sizes the destination, then formats +
    // populates.
    match fs_type.as_str() {
        "ext2" | "ext3" | "ext4" => create_ext(
            &fs_type,
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
        )?,
        "fat32" | "vfat" => create_fat32(
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
        )?,
        "exfat" => create_exfat(
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
        )?,
        "hfs+" | "hfsplus" => create_via_factory::<fstool::fs::hfs_plus::HfsPlus>(
            "hfs+",
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
            fstool::fs::hfs_plus::FormatOpts::default(),
            |o, m| o.apply_options(m),
            DEFAULT_MIN_SIZE,
        )?,
        "ntfs" => create_via_factory::<fstool::fs::ntfs::Ntfs>(
            "ntfs",
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
            fstool::fs::ntfs::format::FormatOpts::default(),
            |o, m| o.apply_options(m),
            16 * 1024 * 1024,
        )?,
        "f2fs" => create_via_factory::<fstool::fs::f2fs::F2fs>(
            "f2fs",
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
            fstool::fs::f2fs::FormatOpts::default(),
            |o, m| o.apply_options(m),
            64 * 1024 * 1024,
        )?,
        "squashfs" => create_via_factory::<fstool::fs::squashfs::Squashfs>(
            "squashfs",
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
            fstool::fs::squashfs::FormatOpts::default(),
            |o, m| o.apply_options(m),
            DEFAULT_MIN_SIZE,
        )?,
        "xfs" => create_via_factory::<fstool::fs::xfs::Xfs>(
            "xfs",
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
            fstool::fs::xfs::format::FormatOpts::default(),
            |o, m| o.apply_options(m),
            16 * 1024 * 1024,
        )?,
        "iso" | "iso9660" => create_via_factory::<fstool::fs::iso9660::Iso9660>(
            "iso9660",
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
            fstool::fs::iso9660::FormatOpts::default(),
            |o, m| o.apply_options(m),
            DEFAULT_MIN_SIZE,
        )?,
        "grf" => create_via_factory::<fstool::fs::grf::Grf>(
            "grf",
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
            fstool::fs::grf::FormatOpts::default(),
            |o, m| o.apply_options(m),
            DEFAULT_MIN_SIZE,
        )?,
        "apfs" => create_via_factory::<fstool::fs::apfs::Apfs>(
            "apfs",
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
            fstool::fs::apfs::ApfsFormatOpts::default(),
            |o, m| o.apply_options(m),
            DEFAULT_MIN_SIZE,
        )?,
        other => {
            return Err(fstool::Error::InvalidArgument(format!(
                "create: unknown --type {other:?} (try ext4, fat32, exfat, hfs+, ntfs, \
                 f2fs, squashfs, xfs, iso, grf, apfs)"
            )));
        }
    }
    Ok(())
}

/// Conservative default `--size` for filesystems that don't auto-size
/// today. 1 MiB is enough for an empty image; populated images bump up
/// via the per-arm minimum.
const DEFAULT_MIN_SIZE: u64 = 1024 * 1024;

/// ext2 / ext3 / ext4 arm of `create`. Honours `block_size` from the
/// option bag (default 1024), then uses [`BuildPlan`] to auto-size
/// against the source. Falls back to a minimum-format on an empty FS.
fn create_ext(
    fs_kind: &str,
    source: Option<&fstool::repack::Source>,
    output: &std::path::Path,
    size_arg: Option<&str>,
    mut bag: OptionMap,
    is_device: bool,
    qcow2_cluster_size: u32,
) -> fstool::Result<()> {
    use fstool::fs::ext::BuildPlan;

    let kind = match fs_kind {
        "ext2" => FsKind::Ext2,
        "ext3" => FsKind::Ext3,
        "ext4" => FsKind::Ext4,
        _ => unreachable!(),
    };
    // Block size is a `create_ext`-shaped knob: BuildPlan needs it
    // before it can run, so we peel it off the bag here instead of
    // letting the generic per-FS `apply_options` consume it.
    let block_size = bag
        .take_u32("block_size")?
        .unwrap_or(1024);

    let mut plan = BuildPlan::new(block_size, kind);
    if let Some(src) = source {
        match src {
            fstool::repack::Source::HostDir(p) => plan.scan_host_path(p)?,
            _ => {
                return Err(fstool::Error::InvalidArgument(
                    "create: ext only accepts a host directory as source".into(),
                ));
            }
        }
    }
    let mut format_opts = plan.to_format_opts();
    format_opts.apply_options(&mut bag)?;
    bag.check_empty(fs_kind)?;

    // Sizing: explicit --size beats the plan; on a block device the
    // device's capacity wins.
    let plan_size = format_opts.blocks_count as u64 * format_opts.block_size as u64;
    let want_size = match size_arg {
        Some(s) => fstool::spec::parse_size(s)?,
        None => plan_size,
    };
    let mut dev = fstool::block::create_image(
        output,
        want_size,
        &fstool::block::CreateOpts {
            cluster_size: qcow2_cluster_size,
        },
    )?;
    let actual_size = dev.total_size();
    if actual_size > plan_size {
        // Grow the FS to fill the device (block-device or larger
        // explicit --size). blocks_count stays multiple-of-8 for the
        // bitmap; inode density tracks mke2fs's 1-inode-per-16-KiB rule.
        let max_blocks_u64 = actual_size / format_opts.block_size as u64;
        let max_blocks = u32::try_from(max_blocks_u64).unwrap_or(u32::MAX);
        format_opts.blocks_count = (max_blocks / 8) * 8;
        let by_density =
            (format_opts.blocks_count as u64 * format_opts.block_size as u64 / 16_384) as u32;
        format_opts.inodes_count = format_opts.inodes_count.max(by_density);
    }
    let final_size = format_opts.blocks_count as u64 * format_opts.block_size as u64;

    let mut ext = Ext::format_with(dev.as_mut(), &format_opts)?;
    if let Some(src) = source {
        fstool::repack::populate_ext_from_source(dev.as_mut(), &mut ext, src)?;
    }
    ext.flush(dev.as_mut())?;
    dev.sync()?;
    eprintln!(
        "wrote {} ({} bytes, {fs_kind}{}{}, {} inodes, {} blocks)",
        output.display(),
        final_size,
        if format_opts.sparse { ", sparse" } else { "" },
        if is_device { ", block device" } else { "" },
        format_opts.inodes_count,
        format_opts.blocks_count
    );
    Ok(())
}

/// FAT32 arm of `create`. The 65 525-cluster floor means even an empty
/// image needs ~33 MiB, so the device capacity (or an explicit --size)
/// is mandatory.
fn create_fat32(
    source: Option<&fstool::repack::Source>,
    output: &std::path::Path,
    size_arg: Option<&str>,
    mut bag: OptionMap,
    is_device: bool,
    qcow2_cluster_size: u32,
) -> fstool::Result<()> {
    use fstool::fs::fat::Fat32;

    // Defaults match the old `fat-build` behaviour: 11-byte ASCII
    // label, 0 volume id. Keep the upper-casing + non-printable
    // scrubbing so legacy CLI usage stays bit-identical.
    let label_str = bag.take_str("volume_label").unwrap_or_else(|| "NO NAME".into());
    let label_bytes = fat32_label_bytes(&label_str);
    let volume_id = bag.take_u32("volume_id")?.unwrap_or(0);
    bag.check_empty("fat32")?;

    let bytes = if is_device {
        let dev = FileBackend::open(output)?;
        dev.total_size()
    } else {
        let s = size_arg.ok_or_else(|| {
            fstool::Error::InvalidArgument(
                "create: --size is required for fat32 when OUTPUT is a regular file \
                 (FAT32 needs ≥ ~33 MiB)".into(),
            )
        })?;
        fstool::spec::parse_size(s)?
    };
    let total_sectors: u32 = (bytes / 512).try_into().map_err(|_| {
        fstool::Error::InvalidArgument(
            "create: FAT32 device size doesn't fit in a u32 sector count".into(),
        )
    })?;
    let mut dev = fstool::block::create_image(
        output,
        bytes,
        &fstool::block::CreateOpts {
            cluster_size: qcow2_cluster_size,
        },
    )?;
    let fat_opts = fstool::fs::fat::FatFormatOpts {
        total_sectors,
        volume_id,
        volume_label: label_bytes,
    };
    let mut fat = Fat32::format(dev.as_mut(), &fat_opts)?;
    if let Some(src) = source {
        fstool::repack::populate_fat32_from_source(dev.as_mut(), &mut fat, src)?;
    }
    fat.flush(dev.as_mut())?;
    dev.sync()?;
    eprintln!(
        "wrote {} ({} bytes, fat32{}, label {:?})",
        output.display(),
        bytes,
        if is_device { ", block device" } else { "" },
        label_str
    );
    Ok(())
}

/// exFAT arm of `create`. Routes through [`populate_fs_from_source_dyn`]
/// since exFAT doesn't implement [`FilesystemFactory`] yet.
fn create_exfat(
    source: Option<&fstool::repack::Source>,
    output: &std::path::Path,
    size_arg: Option<&str>,
    mut bag: OptionMap,
    is_device: bool,
    qcow2_cluster_size: u32,
) -> fstool::Result<()> {
    use fstool::fs::exfat::{Exfat, FormatOpts as ExFormat};

    let mut format_opts = ExFormat::default();
    format_opts.apply_options(&mut bag)?;
    bag.check_empty("exfat")?;

    let bytes = resolve_size_for_dev(output, size_arg, is_device, 16 * 1024 * 1024)?;
    let mut dev = fstool::block::create_image(
        output,
        bytes,
        &fstool::block::CreateOpts {
            cluster_size: qcow2_cluster_size,
        },
    )?;
    let mut exfat = Exfat::format(dev.as_mut(), &format_opts)?;
    if let Some(src) = source {
        fstool::repack::populate_fs_from_source_dyn(dev.as_mut(), &mut exfat, src)?;
    }
    exfat.flush(dev.as_mut())?;
    dev.sync()?;
    eprintln!(
        "wrote {} ({} bytes, exfat{})",
        output.display(),
        bytes,
        if is_device { ", block device" } else { "" }
    );
    Ok(())
}

/// Format + populate any FS that implements [`FilesystemFactory`].
/// `apply` is the per-FS `FormatOpts::apply_options` closure (avoids
/// adding the method to the trait surface).
fn create_via_factory<F>(
    label: &str,
    source: Option<&fstool::repack::Source>,
    output: &std::path::Path,
    size_arg: Option<&str>,
    mut bag: OptionMap,
    is_device: bool,
    qcow2_cluster_size: u32,
    mut format_opts: F::FormatOpts,
    apply: impl FnOnce(&mut F::FormatOpts, &mut OptionMap) -> fstool::Result<()>,
    default_min_size: u64,
) -> fstool::Result<()>
where
    F: fstool::fs::FilesystemFactory,
{
    apply(&mut format_opts, &mut bag)?;
    bag.check_empty(label)?;

    let bytes = resolve_size_for_dev(output, size_arg, is_device, default_min_size)?;
    let mut dev = fstool::block::create_image(
        output,
        bytes,
        &fstool::block::CreateOpts {
            cluster_size: qcow2_cluster_size,
        },
    )?;
    let mut fs = F::format(dev.as_mut(), &format_opts)?;
    if let Some(src) = source {
        fstool::repack::populate_fs_from_source(dev.as_mut(), &mut fs, src)?;
    }
    fs.flush(dev.as_mut())?;
    dev.sync()?;
    eprintln!(
        "wrote {} ({} bytes, {label}{})",
        output.display(),
        bytes,
        if is_device { ", block device" } else { "" }
    );
    Ok(())
}

/// Pick a size for the destination image. Block devices use their
/// capacity; regular files honour --size if given, otherwise fall back
/// to the FS's per-arm minimum (which exists because every FS but ext
/// has a non-trivial floor — FAT32 ~33 MiB, NTFS / XFS / F2FS several
/// MiB, …).
fn resolve_size_for_dev(
    output: &std::path::Path,
    size_arg: Option<&str>,
    is_device: bool,
    default_min: u64,
) -> fstool::Result<u64> {
    if is_device {
        let dev = FileBackend::open(output)?;
        return Ok(dev.total_size());
    }
    match size_arg {
        Some(s) => fstool::spec::parse_size(s),
        None => Ok(default_min),
    }
}

/// Pack a label into the 11-byte FAT32 short-label slot: ASCII upper-case,
/// space-padded, non-printable bytes replaced with `_`.
fn fat32_label_bytes(label: &str) -> [u8; 11] {
    let mut out = [b' '; 11];
    let upper = label.to_ascii_uppercase();
    for (i, &b) in upper.as_bytes().iter().take(11).enumerate() {
        out[i] = if b.is_ascii() && b >= 0x20 && b != 0x7F {
            b
        } else {
            b'_'
        };
    }
    out
}

/// Parse the `--cluster-size` flag's value into a u32 byte count.
/// Errors if not a power of two or below the 512-byte minimum.
fn parse_cluster_size(s: &str) -> fstool::Result<u32> {
    let bytes = fstool::spec::parse_size(s)?;
    if !bytes.is_power_of_two() {
        return Err(fstool::Error::InvalidArgument(format!(
            "--cluster-size {s} must be a power of two"
        )));
    }
    if bytes < 512 || bytes > u32::MAX as u64 {
        return Err(fstool::Error::InvalidArgument(format!(
            "--cluster-size {s} out of range (512..=u32::MAX)"
        )));
    }
    Ok(bytes as u32)
}

/// Refuse to format a block device without --force; emit a clear message
/// pointing at the flag.
fn require_force_for_device(
    output: &std::path::Path,
    is_device: bool,
    force: bool,
) -> fstool::Result<()> {
    if is_device && !force {
        return Err(fstool::Error::InvalidArgument(format!(
            "refusing to format block device {} without --force",
            output.display()
        )));
    }
    Ok(())
}

fn ls(image: &str, path: &str) -> fstool::Result<()> {
    // Compressed-tar archives stream-walk per invocation; bypass the
    // tempfile-spooling BlockDevice path entirely.
    if let Some(algo) = tar_input_codec(image) {
        return ls_tar_stream(image, path, Some(algo));
    }
    let target = fstool::inspect::Target::parse(image);
    fstool::inspect::with_target_device(&target, |dev| {
        let mut fs = fstool::inspect::AnyFs::open(dev)?;
        let entries = fs.list(dev, path)?;
        let mut out = std::io::stdout().lock();
        for e in &entries {
            let _ = writeln!(out, "{}\t{:?}\t{}", e.inode, e.kind, e.name);
        }
        Ok(())
    })
}

fn cat(image: &str, path: &str) -> fstool::Result<()> {
    if let Some(algo) = tar_input_codec(image) {
        return cat_tar_stream(image, path, Some(algo));
    }
    let target = fstool::inspect::Target::parse(image);
    fstool::inspect::with_target_device(&target, |dev| {
        let mut fs = fstool::inspect::AnyFs::open(dev)?;
        let mut out = std::io::stdout().lock();
        fs.copy_file_to(dev, path, &mut out)?;
        Ok(())
    })
}

fn info(image: &str) -> fstool::Result<()> {
    if let Some(algo) = tar_input_codec(image) {
        return info_tar_stream(image, Some(algo));
    }
    let target = fstool::inspect::Target::parse(image);
    // If the user gave a bare `disk.img` and it carries a partition
    // table, print the table instead of trying to open it as a single
    // filesystem (which would fail).
    if target.partition.is_none() {
        let mut disk = fstool::block::open_image(&target.path)?;
        if let Some(table) = fstool::inspect::detect_partition_table(disk.as_mut())? {
            print_partition_table(&target.path, disk.total_size(), &table);
            return Ok(());
        }
        // No table — fall through to the bare-FS info below using the
        // already-opened disk.
        let mut fs = fstool::inspect::AnyFs::open(disk.as_mut())?;
        print_fs_info(disk.as_mut(), &mut fs);
        return Ok(());
    }
    fstool::inspect::with_target_device(&target, |dev| {
        let mut fs = fstool::inspect::AnyFs::open(dev)?;
        print_fs_info(dev, &mut fs);
        Ok(())
    })
}

fn print_partition_table(
    path: &std::path::Path,
    total_bytes: u64,
    table: &fstool::inspect::DetectedTable,
) {
    println!("image:             {}", path.display());
    println!("size:              {total_bytes} bytes");
    println!("partition table:   {}", table.label());
    let parts = table.partitions();
    println!("partitions:        {}", parts.len());
    if parts.is_empty() {
        return;
    }
    println!();
    println!("  N  start (LBA)     end (LBA)         size       kind                    name");
    for (i, p) in parts.iter().enumerate() {
        let end = p.start_lba + p.size_lba - 1;
        let bytes = p.size_lba * 512;
        let name = p.name.as_deref().unwrap_or("");
        println!(
            "  {:>2}  {:>11}  {:>13}  {:>10}  {:<22}  {}",
            i + 1,
            p.start_lba,
            end,
            human_size(bytes),
            format!("{:?}", p.kind),
            name
        );
    }
}

fn print_tar_info(tar: &fstool::fs::tar::Tar) {
    let entries = tar.entries();
    let mut files = 0usize;
    let mut dirs = 0usize;
    let mut symlinks = 0usize;
    let mut devices = 0usize;
    let mut content_bytes = 0u64;
    let mut total_xattrs = 0usize;
    for e in entries {
        match e.kind {
            fstool::fs::tar::EntryKind::Regular | fstool::fs::tar::EntryKind::HardLink => {
                files += 1;
                content_bytes += e.size;
            }
            fstool::fs::tar::EntryKind::Dir => dirs += 1,
            fstool::fs::tar::EntryKind::Symlink => symlinks += 1,
            fstool::fs::tar::EntryKind::CharDev
            | fstool::fs::tar::EntryKind::BlockDev
            | fstool::fs::tar::EntryKind::Fifo => devices += 1,
        }
        total_xattrs += e.xattrs.len();
    }
    println!("entries:           {}", entries.len());
    println!("  files:           {files}");
    println!("  directories:     {dirs}");
    println!("  symlinks:        {symlinks}");
    println!("  devices/fifos:   {devices}");
    println!("file content:      {content_bytes} bytes");
    println!("xattrs total:      {total_xattrs}");
}

fn human_size(b: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    if b >= GIB {
        format!("{:.2} GiB", b as f64 / GIB as f64)
    } else if b >= MIB {
        format!("{:.2} MiB", b as f64 / MIB as f64)
    } else if b >= KIB {
        format!("{:.2} KiB", b as f64 / KIB as f64)
    } else {
        format!("{b} B")
    }
}

fn print_fs_info(dev: &mut dyn fstool::block::BlockDevice, fs: &mut fstool::inspect::AnyFs) {
    println!("fs kind:           {}", fs.kind_string());
    match fs {
        fstool::inspect::AnyFs::Ext(ext) => print_ext_info(ext),
        fstool::inspect::AnyFs::Fat32(fat) => print_fat_info(fat),
        fstool::inspect::AnyFs::Tar(tar) => print_tar_info(tar),
        fstool::inspect::AnyFs::Xfs(xfs) => print_xfs_info(xfs),
        fstool::inspect::AnyFs::Exfat(exfat) => print_exfat_info(exfat),
        fstool::inspect::AnyFs::HfsPlus(hfs) => print_hfs_plus_info(hfs),
        fstool::inspect::AnyFs::Apfs(apfs) => print_apfs_info(apfs),
        fstool::inspect::AnyFs::Ntfs(ntfs) => print_ntfs_info(ntfs),
        fstool::inspect::AnyFs::F2fs(f2) => print_f2fs_info(f2),
        fstool::inspect::AnyFs::Squashfs(sq) => print_squashfs_info(sq),
        fstool::inspect::AnyFs::Iso9660(iso) => print_iso9660_info(iso),
        fstool::inspect::AnyFs::Grf(grf) => print_grf_info(grf),
    }
    println!();
    println!("/ listing:");
    match fs.list(dev, "/") {
        Ok(entries) => {
            for e in &entries {
                println!("  {:>10}  {:?}  {}", e.inode, e.kind, e.name);
            }
        }
        Err(e) => println!("  (couldn't list /: {e})"),
    }
}

fn print_ext_info(ext: &Ext) {
    let sb = &ext.sb;
    println!("block size:        {}", sb.block_size());
    println!("blocks total:      {}", sb.blocks_count);
    println!("blocks free:       {}", sb.free_blocks_count);
    println!("inodes total:      {}", sb.inodes_count);
    println!("inodes free:       {}", sb.free_inodes_count);
    println!("blocks per group:  {}", sb.blocks_per_group);
    println!("inodes per group:  {}", sb.inodes_per_group);
    println!("groups:            {}", ext.layout.num_groups());
    println!("first data block:  {}", sb.first_data_block);
    println!("revision:          {}", sb.rev_level);
    println!("first inode:       {}", sb.first_ino);
    println!("inode size:        {}", sb.inode_size);
    println!(
        "feature flags:     compat={:#010x}  incompat={:#010x}  ro_compat={:#010x}",
        sb.feature_compat, sb.feature_incompat, sb.feature_ro_compat
    );
    println!("uuid:              {}", format_uuid(&sb.uuid));
}

fn print_fat_info(fat: &fstool::fs::fat::Fat32) {
    let b = fat.boot_sector();
    let label = std::str::from_utf8(&b.volume_label)
        .unwrap_or("?")
        .trim_end();
    println!("sector size:       {}", b.bytes_per_sector);
    println!("sectors / cluster: {}", b.sectors_per_cluster);
    println!("total sectors:     {}", b.total_sectors);
    println!("FAT size (sect):   {}", b.fat_size);
    println!("# of FATs:         {}", b.num_fats);
    println!("reserved sectors:  {}", b.reserved_sector_count);
    println!("data clusters:     {}", b.cluster_count());
    println!("root cluster:      {}", b.root_cluster);
    println!("volume ID:         {:#010x}", b.volume_id);
    println!("volume label:      {label:?}");
}

fn print_xfs_info(xfs: &fstool::fs::xfs::Xfs) {
    println!("total bytes:       {}", xfs.total_bytes());
    println!("block size:        {}", xfs.block_size());
    println!("inode size:        {}", xfs.inode_size());
    println!("AG count:          {}", xfs.ag_count());
}

fn print_exfat_info(exfat: &fstool::fs::exfat::Exfat) {
    println!("total bytes:       {}", exfat.total_bytes());
    println!("cluster size:      {}", exfat.cluster_size());
    println!("sectors / cluster: {}", exfat.sectors_per_cluster());
    println!("root cluster:      {}", exfat.root_directory_cluster());
    println!("volume label:      {:?}", exfat.volume_label());
}

fn print_hfs_plus_info(hfs: &fstool::fs::hfs_plus::HfsPlus) {
    println!("total bytes:       {}", hfs.total_bytes());
    println!("block size:        {}", hfs.block_size());
    println!("volume name:       {:?}", hfs.volume_name());
}

fn print_apfs_info(apfs: &fstool::fs::apfs::Apfs) {
    println!("total bytes:       {}", apfs.total_bytes());
    println!("block size:        {}", apfs.block_size());
    println!("volume name:       {:?}", apfs.volume_name());
}

fn print_ntfs_info(ntfs: &fstool::fs::ntfs::Ntfs) {
    println!("total bytes:       {}", ntfs.total_bytes());
    println!("cluster size:      {}", ntfs.cluster_size());
    println!("bytes / sector:    {}", ntfs.bytes_per_sector());
    println!("sectors / cluster: {}", ntfs.sectors_per_cluster());
    println!("MFT record size:   {}", ntfs.mft_record_size());
    println!("volume serial:     {:#018x}", ntfs.volume_serial());
    println!("note:              read support is scaffold-only");
}

fn print_f2fs_info(f2: &fstool::fs::f2fs::F2fs) {
    let sb = f2.superblock();
    println!("total bytes:       {}", f2.total_bytes());
    println!("block size:        {}", f2.block_size());
    println!("version:           {}.{}", sb.major_ver, sb.minor_ver);
    println!("block count:       {}", sb.block_count);
    println!("volume name:       {:?}", f2.volume_name());
    println!("note:              read support is scaffold-only");
}

fn print_squashfs_info(sq: &fstool::fs::squashfs::Squashfs) {
    let sb = sq.superblock();
    println!("total bytes:       {}", sq.total_bytes());
    println!("block size:        {}", sq.block_size());
    println!("compression:       {:?}", sq.compression());
    println!("version:           {}.{}", sb.major, sb.minor);
    println!("inode count:       {}", sb.inode_count);
    println!("note:              read support is scaffold-only");
}

fn print_iso9660_info(iso: &fstool::fs::iso9660::Iso9660) {
    println!("volume id:         {}", iso.volume_id());
    println!("system id:         {}", iso.pvd.system_id);
    println!(
        "volume space (B):  {}",
        u64::from(iso.pvd.volume_space_size) * 2048
    );
    println!("logical block:     {}", iso.pvd.logical_block_size);
    println!("joliet:            {}", iso.joliet.is_some());
    println!("rock ridge:        {}", iso.rock_ridge);
    println!("bootable:          {}", iso.boot.is_some());
    if let Some(boot) = iso.boot.as_ref() {
        println!("  boot platform:   0x{:02x}", boot.default_entry.platform);
        println!("  boot lba:        {}", boot.default_entry.load_rba);
        println!("  boot sectors:    {}", boot.default_entry.sector_count);
    }
}

fn print_grf_info(grf: &fstool::fs::grf::Grf) {
    println!("grf version:       {:#x}", grf.version);
    println!("table offset:      {}", grf.table_offset);
    println!("seed:              {}", grf.seed);
    println!("encrypted header:  {}", grf.encrypted_header);
    println!("file count:        {}", grf.entries.len());
    println!("wasted space (B):  {}", grf.wasted_space());
}

fn format_uuid(bytes: &[u8; 16]) -> String {
    let mut s = String::with_capacity(36);
    for (i, b) in bytes.iter().enumerate() {
        s.push_str(&format!("{b:02x}"));
        if matches!(i, 3 | 5 | 7 | 9) {
            s.push('-');
        }
    }
    s
}
