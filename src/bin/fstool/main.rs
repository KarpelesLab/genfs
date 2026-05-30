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
use fstool::path_style::{self, PathStyle};

#[derive(Parser, Debug)]
#[command(
    name = "fstool",
    version,
    about = "Build and inspect disk-image filesystems (ext2/3/4, MBR, GPT)."
)]
struct Cli {
    /// How filesystem paths are spelled. `unix` (default): `/` separates
    /// everywhere, and a literal `/` inside an HFS/HFS+ name shows as `:`.
    /// `native`: the filesystem's own separator (`:` for HFS/HFS+, `\` for
    /// FAT/exFAT/NTFS, `/` elsewhere) with real filenames preserved.
    #[arg(
        long = "path-style",
        value_enum,
        default_value_t = fstool::path_style::PathStyle::Unix,
        global = true
    )]
    path_style: fstool::path_style::PathStyle,

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
        /// Recurse into subdirectories (like `ls -R`): each directory is
        /// printed under a `path:` header.
        #[arg(short = 'R', long)]
        recursive: bool,
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

    /// Analyze a source and report its contents (file / dir / symlink /
    /// device counts, total file bytes, ext inode estimate) plus the
    /// recommended destination image size per filesystem type — the same
    /// metrics used to size a `--shrink` repack. The source may be a
    /// filesystem image (optionally `:N` for a partition), a tar /
    /// `.tar.gz` archive, or a host directory.
    Analyze {
        /// Source: image[:N], tar / tar.gz, or host directory.
        #[arg(value_name = "SOURCE")]
        source: String,
        /// Report the recommended size only for this destination fs type
        /// (default: every type that takes a content-fit size).
        #[arg(long, value_name = "TYPE")]
        fs_type: Option<String>,
        /// ext block size used for the ext size estimate.
        #[arg(long, default_value_t = 4096)]
        block_size: u32,
        /// Emit machine-readable JSON instead of text.
        #[arg(long)]
        json: bool,
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
        /// Open the image strictly read-only. The underlying file is
        /// opened `O_RDONLY` (any write fails at the syscall), the
        /// shell skips its in-place-mutable check (lets you browse
        /// tar / tar.gz / ISO / SquashFS too), and `put` / `rm` /
        /// `mkdir` refuse with a clear error.
        #[arg(long)]
        ro: bool,
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
        Command::Ls {
            image,
            path,
            recursive,
        } => ls(&image, &path, recursive, cli.path_style),
        Command::Cat { image, path } => cat(&image, &path, cli.path_style),
        Command::Info { image } => info(&image),
        Command::Analyze {
            source,
            fs_type,
            block_size,
            json,
        } => analyze_cmd(&source, fs_type.as_deref(), block_size, json),
        Command::Build { spec, output } => build(&spec, &output),
        Command::Add {
            image,
            host_src,
            fs_dest,
        } => add(&image, &host_src, &fs_dest, cli.path_style),
        Command::Rm { image, fs_path } => rm(&image, &fs_path, cli.path_style),
        Command::Shell { image, ro } => shell_cmd(&image, ro, cli.path_style),
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
    use fstool::fs::Filesystem;
    use fstool::inspect::FsKind;

    // FUSE adapter wants ownership of both the filesystem handle and
    // the backing block device. Partition selectors (`image:N`) and
    // qcow2 sources go through `inspect::with_target_device` which
    // doesn't give us ownership; for v1 we accept plain raw images
    // only.
    if image.contains(':') {
        return Err(fstool::Error::Unsupported(
            "fstool mount: partitioned-image selectors (`path:N`) and qcow2 sources \
             are not wired through the FUSE adapter yet — open a raw .img directly"
                .into(),
        ));
    }
    let path = std::path::Path::new(image);
    let mut dev: Box<dyn fstool::block::BlockDevice + Send> =
        Box::new(fstool::block::FileBackend::open(path)?);
    let kind = fstool::inspect::detect_fs(dev.as_mut())?;
    let (fs, fs_name): (Box<dyn Filesystem + Send>, &'static str) = match kind {
        FsKind::Ext => {
            let mut ext = fstool::fs::ext::Ext::open(dev.as_mut())?;
            // Replay any pending journal so the mounted view matches
            // what the kernel would see on first mount of an unclean
            // shutdown.
            let _ = ext.replay_pending_journal(dev.as_mut())?;
            (Box::new(ext), "ext")
        }
        FsKind::Fat32 => (
            Box::new(fstool::fs::fat::Fat32::open(dev.as_mut())?),
            "fat32",
        ),
        FsKind::Exfat => (
            Box::new(fstool::fs::exfat::Exfat::open(dev.as_mut())?),
            "exfat",
        ),
        FsKind::Xfs => (Box::new(fstool::fs::xfs::Xfs::open(dev.as_mut())?), "xfs"),
        FsKind::HfsPlus => (
            Box::new(fstool::fs::hfs_plus::HfsPlus::open(dev.as_mut())?),
            "hfs+",
        ),
        FsKind::Apfs => (
            Box::new(fstool::fs::apfs::Apfs::open(dev.as_mut())?),
            "apfs",
        ),
        FsKind::Ntfs => (
            Box::new(fstool::fs::ntfs::Ntfs::open(dev.as_mut())?),
            "ntfs",
        ),
        FsKind::F2fs => (
            Box::new(fstool::fs::f2fs::F2fs::open(dev.as_mut())?),
            "f2fs",
        ),
        FsKind::Squashfs => (
            Box::new(fstool::fs::squashfs::Squashfs::open(dev.as_mut())?),
            "squashfs",
        ),
        FsKind::Iso9660 => (
            Box::new(fstool::fs::iso9660::Iso9660::open(dev.as_mut())?),
            "iso9660",
        ),
        FsKind::Tar => (Box::new(fstool::fs::tar::Tar::open(dev.as_mut())?), "tar"),
        FsKind::Grf => (
            Box::new(fstool::fs::grf::Grf::open_dev(dev.as_mut())?),
            "grf",
        ),
        FsKind::Zip => (
            Box::new(fstool::fs::archive::zip::ZipFs::open(dev.as_mut())?),
            "zip",
        ),
        FsKind::Cpio => (
            Box::new(fstool::fs::archive::cpio::CpioFs::open(dev.as_mut())?),
            "cpio",
        ),
        FsKind::Ar => (
            Box::new(fstool::fs::archive::ar::ArFs::open(dev.as_mut())?),
            "ar",
        ),
        FsKind::SevenZ => (
            Box::new(fstool::fs::archive::sevenz::SevenZFs::open(dev.as_mut())?),
            "7z",
        ),
        FsKind::Rar => (
            Box::new(fstool::fs::archive::rar::RarFs::open(dev.as_mut())?),
            "rar",
        ),
        FsKind::Arc => (
            Box::new(fstool::fs::archive::arc::ArcFs::open(dev.as_mut())?),
            "arc",
        ),
        FsKind::Lha => (
            Box::new(fstool::fs::archive::lha::LhaFs::open(dev.as_mut())?),
            "lha",
        ),
        FsKind::Lzx => (
            Box::new(fstool::fs::archive::lzx::LzxFs::open(dev.as_mut())?),
            "lzx",
        ),
        FsKind::Cab => (
            Box::new(fstool::fs::archive::cab::CabFs::open(dev.as_mut())?),
            "cab",
        ),
        FsKind::Sit => (
            Box::new(fstool::fs::archive::sit::SitFs::open(dev.as_mut())?),
            "sit",
        ),
        // FsKind is #[non_exhaustive]; new variants added in the
        // future error out here instead of silently falling through.
        _ => {
            return Err(fstool::Error::Unsupported(format!(
                "fstool mount: filesystem {kind:?} is not wired through the FUSE adapter yet"
            )));
        }
    };
    let cap = fs.mutation_capability();
    let ro_suffix = if cap.supports_add_remove() {
        ""
    } else {
        " (read-only)"
    };
    // The CLI opts into `allow_other` so multi-user access works
    // when the operator has enabled `user_allow_other` in
    // `/etc/fuse.conf`; the integration tests leave it off so they
    // pass on stock setups.
    let adapter = fstool::fuse_adapter::FstoolFs::new(fs, dev, fs_name).allow_other(true);
    eprintln!(
        "fstool: mounted {image} as {fs_name}{ro_suffix} at {} (umount to detach)",
        mountpoint.display()
    );
    adapter.mount(mountpoint).map_err(fstool::Error::Io)?;
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
    use fstool::repack::RepackSink;
    let qcow2_cluster_size = parse_cluster_size(cluster_size)?;

    if srcs.is_empty() {
        return Err(fstool::Error::InvalidArgument(
            "repack: at least one source is required".into(),
        ));
    }

    // Multi-source: fold the layers into an in-memory `MergeModel`
    // (metadata only — no file bodies in RAM) and drive the destination
    // straight from it. Each tar layer is read **forward only**, in the
    // order entries appear in it, so no random access (and no tempfile)
    // is ever needed for the merged source.
    if srcs.len() > 1 {
        let layers: Vec<fstool::repack::Source> = srcs
            .iter()
            .map(|s| fstool::repack::Source::detect(s))
            .collect::<fstool::Result<_>>()?;
        return repack_layered_to_dst(
            layers,
            dst,
            size_arg,
            shrink,
            fs_type_override,
            block_size,
            qcow2_cluster_size,
        );
    }

    // Streaming fast path: a compressed-tar source never needs the
    // decompress-to-tempfile — decode on the fly and walk forward into the
    // destination (a second decode pass handles sizing for ext / `--shrink`
    // / default). Every destination whose writer streams each file as it
    // arrives takes this branch, including the deferred archive formats
    // (SquashFS / ISO 9660 / GRF), which now write data incrementally. Only
    // tar→tar output (a pure sequential re-mux) falls through.
    // The source is taken as a tar stream when it has a recognised
    // compressed-tar extension (codec = `Some(algo)`) OR a plain `.tar`
    // extension (codec = `None`). The streaming path handles both, so a
    // plain tar no longer falls through to the random-access `Tar::open`
    // — which was eating ~17 % of the W1s ext4 profile parsing entry
    // headers up front.
    let raw_src = srcs[0].as_str();
    let codec_opt: Option<Option<fstool::compression::Algo>> =
        if let Some(algo) = tar_input_codec(raw_src) {
            Some(Some(algo))
        } else {
            let bare = raw_src.split(':').next().unwrap_or(raw_src);
            let p = std::path::Path::new(bare);
            if p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("tar"))
            {
                Some(None)
            } else {
                None
            }
        };
    if let Some(codec) = codec_opt {
        let raw = srcs[0].as_str();
        let tar_path = std::path::PathBuf::from(raw.split(':').next().unwrap_or(raw));
        // Resolve the destination FS the same way the main path does;
        // a tar source defaults to ext4 when nothing else specifies one.
        let target_fs = fs_type_override
            .map(|s| s.to_string())
            .or_else(|| {
                dst.extension()
                    .and_then(|s| s.to_str())
                    .filter(|e| e.eq_ignore_ascii_case("tar"))
                    .map(|_| "tar".to_string())
            })
            .or_else(|| tar_output_codec(dst).map(|_| "tar".to_string()))
            .unwrap_or_else(|| "ext4".to_string());
        let lower = target_fs.to_ascii_lowercase();
        // tar → tar is a pure forward re-mux: decode the source on the fly
        // and re-emit through the tar stream sink, no tempfile.
        if lower == "tar" {
            return repack_tar_stream_to_tar(&tar_path, codec, dst, tar_output_codec(dst));
        }
        if matches!(
            lower.as_str(),
            "ext2"
                | "ext3"
                | "ext4"
                | "fat32"
                | "vfat"
                | "xfs"
                | "hfsplus"
                | "hfs+"
                | "ntfs"
                | "f2fs"
                | "squashfs"
                | "iso"
                | "iso9660"
                | "grf"
                | "zip"
                | "cpio"
        ) {
            return repack_tar_stream_to_fs(
                &tar_path,
                codec,
                dst,
                &lower,
                size_arg,
                shrink,
                block_size,
                qcow2_cluster_size,
            );
        }
        // All streamable destinations are handled above; nothing falls
        // through for a compressed-tar source.
    }

    // Compressed-tar source: decompress once to a tempfile and treat
    // that plain `.tar` as the source — the unified walker then handles
    // it like any other image, for every destination.
    let _decompressed; // keeps the tempfile alive for the call
    let src_owned: String = if let Some(algo) = tar_input_codec(srcs[0].as_str()) {
        let raw = srcs[0].as_str();
        let path = std::path::Path::new(raw.split(':').next().unwrap_or(raw));
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or(raw);
        // Decompressing a multi-GB archive is the dominant silent cost
        // before files start streaming — announce it.
        fstool::repack::phase(&format!("decompressing {name} …"));
        let tmp = fstool::compression::decompress_to_tempfile(path, algo)?;
        let p = tmp.path().to_string_lossy().into_owned();
        _decompressed = Some(tmp);
        p
    } else {
        _decompressed = None;
        srcs[0].clone()
    };
    let src = src_owned.as_str();
    let src_target = fstool::inspect::Target::parse(src);

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
                // Fixed-size block filesystems: the analyze API owns the
                // content-fit sizing (ext via BuildPlan, fat32 via the
                // byte heuristic).
                "ext2" | "ext3" | "ext4" | "fat32" | "vfat" => {
                    fstool::analyze::analyze_fs(&mut src_fs, src_dev, block_size)?
                        .recommended_size(&lower)
                        .expect("block fs has a recommended size")
                }
                // Tar output streams to a file; no pre-sized device, so
                // the destination size is unused.
                "tar" => 0,
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
                "zip" | "cpio" | "ar" => {
                    // Archive writers stream into a sparse, over-sized
                    // file that is truncated to the real length after
                    // flush. ×2 + 16 MiB covers per-entry headers even
                    // for a tree of many tiny files.
                    let bytes = sum_source_file_bytes(src_dev, &mut src_fs)?;
                    bytes.saturating_mul(2).saturating_add(16 * 1024 * 1024)
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
                // Tar output streams to a file; no pre-sized device, so
                // the destination size is unused.
                "tar" => 0,
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
                "zip" | "cpio" | "ar" => {
                    let bytes = sum_source_file_bytes(src_dev, &mut src_fs).unwrap_or(0);
                    bytes.saturating_mul(2).saturating_add(16 * 1024 * 1024)
                }
                _ => src_total,
            },
        };

        // Tar output is special: a tar archive is sequential, written to
        // a `Write` (optionally codec-wrapped) rather than a pre-sized
        // block device. This is the one stream-vs-non-stream branch.
        if lower == "tar" {
            let file = std::fs::File::create(dst)?;
            let buffered: Box<dyn std::io::Write> =
                Box::new(std::io::BufWriter::with_capacity(64 * 1024, file));
            let inner = match dst_tar_codec {
                Some(algo) => fstool::compression::make_writer(algo, buffered)?,
                None => buffered,
            };
            let mut sink = fstool::repack::TarStreamSink::new(inner);
            fstool::repack::walk_anyfs(&mut src_fs, src_dev, &mut sink)?;
            sink.finish()?;
            let written = sink.bytes_written();
            match dst_tar_codec {
                Some(algo) => eprintln!(
                    "repacked {src} → {} (fs: {source_kind} → tar.{}, {written} bytes plain)",
                    dst.display(),
                    algo.name()
                ),
                None => eprintln!(
                    "repacked {src} → {} (fs: {source_kind} → tar, {written} bytes)",
                    dst.display()
                ),
            }
            return Ok(());
        }

        // Creating + formatting the destination (writing group
        // descriptors, bitmaps, inode tables, journal for ext; the FAT
        // tables for FAT32; …) runs before the first file streams, so
        // announce it — on a multi-GB image this is several seconds.
        fstool::repack::phase(&format!(
            "formatting {target_fs_str} destination ({}) …",
            human_size(dst_size)
        ));
        let mut dst_dev = fstool::block::create_image(
            dst,
            dst_size,
            &fstool::block::CreateOpts {
                cluster_size: qcow2_cluster_size,
            },
        )?;
        // `Some(len)` for archive writers (zip/cpio/ar) so we truncate
        // the over-provisioned file; `None` for fixed-size FS images.
        let archive_len: Option<u64> = match lower.as_str() {
            "ext2" | "ext3" | "ext4" => {
                let kind = match lower.as_str() {
                    "ext2" => fstool::fs::ext::FsKind::Ext2,
                    "ext3" => fstool::fs::ext::FsKind::Ext3,
                    _ => fstool::fs::ext::FsKind::Ext4,
                };
                let mut opts = fstool::analyze::analyze_fs(&mut src_fs, src_dev, block_size)?
                    .ext_format_opts(kind);
                // Sparse: the source reader emits zeros over holes and
                // the ext writer re-sparsifies all-zero blocks, so holes
                // round-trip.
                opts.sparse = true;
                // dst_dev came from `block::create_image` above — a
                // fresh sparse file or qcow2, already all-zero. Tell
                // the formatter so it skips its upfront full-device
                // zero pass.
                opts.prezeroed = true;
                let plan_size = opts.blocks_count as u64 * opts.block_size as u64;
                if dst_size > plan_size {
                    let max = (dst_size / opts.block_size as u64) as u32;
                    opts.blocks_count = (max / 8) * 8;
                    let by_density =
                        (opts.blocks_count as u64 * opts.block_size as u64 / 16_384) as u32;
                    opts.inodes_count = opts.inodes_count.max(by_density);
                }
                let mut dst_ext = fstool::fs::ext::Ext::format_with(dst_dev.as_mut(), &opts)?;
                {
                    let mut sink = fstool::repack::FsSink::new(&mut dst_ext, dst_dev.as_mut());
                    fstool::repack::walk_anyfs(&mut src_fs, src_dev, &mut sink)?;
                }
                dst_ext.flush(dst_dev.as_mut())?;
                None
            }
            "fat32" | "vfat" => {
                let total_sectors: u32 = (dst_size / 512).try_into().map_err(|_| {
                    fstool::Error::InvalidArgument(
                        "repack: FAT32 size doesn't fit in a u32 sector count".into(),
                    )
                })?;
                let opts = fstool::fs::fat::FatFormatOpts {
                    total_sectors,
                    volume_id: 0,
                    volume_label: *b"REPACKED   ",
                };
                let mut dst_fat = fstool::fs::fat::Fat32::format(dst_dev.as_mut(), &opts)?;
                {
                    let mut sink =
                        fstool::repack::FsSink::new(&mut dst_fat, dst_dev.as_mut()).lossy();
                    fstool::repack::walk_anyfs(&mut src_fs, src_dev, &mut sink)?;
                }
                dst_fat.flush(dst_dev.as_mut())?;
                None
            }
            "hfsplus" | "hfs+" => repack_via_trait::<fstool::fs::hfs_plus::HfsPlus>(
                dst_dev.as_mut(),
                &fstool::fs::hfs_plus::FormatOpts::default(),
                &mut src_fs,
                src_dev,
                false,
            )?,
            "ntfs" => repack_via_trait::<fstool::fs::ntfs::Ntfs>(
                dst_dev.as_mut(),
                &fstool::fs::ntfs::format::FormatOpts::default(),
                &mut src_fs,
                src_dev,
                false,
            )?,
            "f2fs" => repack_via_trait::<fstool::fs::f2fs::F2fs>(
                dst_dev.as_mut(),
                &fstool::fs::f2fs::FormatOpts::default(),
                &mut src_fs,
                src_dev,
                false,
            )?,
            "squashfs" => repack_via_trait::<fstool::fs::squashfs::Squashfs>(
                dst_dev.as_mut(),
                &fstool::fs::squashfs::FormatOpts::default(),
                &mut src_fs,
                src_dev,
                false,
            )?,
            "xfs" => repack_via_trait::<fstool::fs::xfs::Xfs>(
                dst_dev.as_mut(),
                &fstool::fs::xfs::format::FormatOpts::default(),
                &mut src_fs,
                src_dev,
                false,
            )?,
            "iso" | "iso9660" => {
                let opts = fstool::fs::iso9660::FormatOpts {
                    volume_id: "FSTOOL".into(),
                    application_id: "fstool".into(),
                    ..fstool::fs::iso9660::FormatOpts::default()
                };
                repack_via_trait::<fstool::fs::iso9660::Iso9660>(
                    dst_dev.as_mut(),
                    &opts,
                    &mut src_fs,
                    src_dev,
                    false,
                )?
            }
            "grf" => repack_via_trait::<fstool::fs::grf::Grf>(
                dst_dev.as_mut(),
                &fstool::fs::grf::FormatOpts::default(),
                &mut src_fs,
                src_dev,
                false,
            )?,
            "zip" => repack_via_trait::<fstool::fs::archive::zip::ZipFs>(
                dst_dev.as_mut(),
                &fstool::fs::archive::zip::ZipFormatOpts::default(),
                &mut src_fs,
                src_dev,
                true,
            )?,
            "cpio" => repack_via_trait::<fstool::fs::archive::cpio::CpioFs>(
                dst_dev.as_mut(),
                &fstool::fs::archive::cpio::CpioFormatOpts,
                &mut src_fs,
                src_dev,
                false,
            )?,
            "ar" => repack_via_trait::<fstool::fs::archive::ar::ArFs>(
                dst_dev.as_mut(),
                &fstool::fs::archive::ar::ArFormatOpts,
                &mut src_fs,
                src_dev,
                true,
            )?,
            other => {
                return Err(fstool::Error::InvalidArgument(format!(
                    "repack: unknown --fs-type {other:?}"
                )));
            }
        };
        dst_dev.sync()?;
        let report_size = match archive_len {
            Some(len) => {
                drop(dst_dev);
                truncate_output_file(dst, len)?;
                len
            }
            None => dst_size,
        };
        eprintln!(
            "repacked {src} → {} (fs: {source_kind} → {target_fs_str}, {report_size} bytes)",
            dst.display()
        );
        Ok(())
    })
}

/// Generic repack pipeline driven by [`fstool::fs::FilesystemFactory`].
/// Formats `F` onto `dst_dev`, then walks the already-open source
/// (`src_fs` + `src_dev`) into it through the unified [`fstool::repack`]
/// sink. `lossy` drops entries the destination can't represent
/// (symlinks/devices/xattrs) rather than erroring. The destination is
/// flushed before returning; the `Option<u64>` is the exact archive
/// length for stream-style writers (zip/cpio/ar), `None` for sized
/// filesystem images.
fn repack_via_trait<F: fstool::fs::FilesystemFactory>(
    dst_dev: &mut dyn fstool::block::BlockDevice,
    opts: &F::FormatOpts,
    src_fs: &mut fstool::inspect::AnyFs,
    src_dev: &mut dyn fstool::block::BlockDevice,
    lossy: bool,
) -> fstool::Result<Option<u64>> {
    let mut dst = F::format(dst_dev, opts)?;
    {
        let mut sink = fstool::repack::FsSink::new(&mut dst, dst_dev);
        if lossy {
            sink = sink.lossy();
        }
        fstool::repack::walk_anyfs(src_fs, src_dev, &mut sink)?;
    }
    dst.flush(dst_dev)?;
    Ok(fstool::fs::Filesystem::image_len(&dst))
}

/// Streaming counterpart of [`repack_via_trait`]: format `dst_dev` and
/// populate it by walking the compressed tar at `tar_path` forward
/// (decoded on the fly via [`fstool::repack::open_tar_stream`]) — no
/// tempfile, no random access.
fn repack_stream_via_trait<F: fstool::fs::FilesystemFactory>(
    dst_dev: &mut dyn fstool::block::BlockDevice,
    opts: &F::FormatOpts,
    tar_path: &std::path::Path,
    codec: Option<fstool::compression::Algo>,
    lossy: bool,
) -> fstool::Result<Option<u64>> {
    let mut dst = F::format(dst_dev, opts)?;
    {
        let mut sink = fstool::repack::FsSink::new(&mut dst, dst_dev);
        if lossy {
            sink = sink.lossy();
        }
        let mut reader = fstool::repack::open_tar_stream(tar_path, codec)?;
        fstool::repack::walk_tar_stream(&mut reader, &mut sink)?;
    }
    dst.flush(dst_dev)?;
    Ok(fstool::fs::Filesystem::image_len(&dst))
}

/// Repack a compressed-tar source into a freshly-formatted block-device
/// filesystem **without decompressing to a tempfile**. The archive is
/// decoded on the fly; sizing (for ext, `--shrink`, or a defaulted
/// size) runs a first streaming pass into a counting sink, then the
/// write pass re-opens the source and decodes from byte 0 again. Only
/// reached for the block-FS targets gated by the caller.
/// Re-mux a compressed-tar source into a tar destination by streaming:
/// decode the source archive on the fly and re-emit each entry through the
/// tar stream sink, optionally re-compressing the output. No tempfile and
/// no random access — a tar is only ever walked forward.
fn repack_tar_stream_to_tar(
    tar_path: &std::path::Path,
    src_codec: Option<fstool::compression::Algo>,
    dst: &std::path::Path,
    dst_tar_codec: Option<fstool::compression::Algo>,
) -> fstool::Result<()> {
    use fstool::repack::RepackSink;
    let file = std::fs::File::create(dst)?;
    let buffered: Box<dyn std::io::Write> =
        Box::new(std::io::BufWriter::with_capacity(64 * 1024, file));
    let inner = match dst_tar_codec {
        Some(algo) => fstool::compression::make_writer(algo, buffered)?,
        None => buffered,
    };
    let mut sink = fstool::repack::TarStreamSink::new(inner);
    let mut reader = fstool::repack::open_tar_stream(tar_path, src_codec)?;
    fstool::repack::walk_tar_stream(&mut reader, &mut sink)?;
    sink.finish()?;
    let written = sink.bytes_written();
    match dst_tar_codec {
        Some(algo) => eprintln!(
            "repacked {} → {} (fs: tar → tar.{}, {written} bytes plain)",
            tar_path.display(),
            dst.display(),
            algo.name()
        ),
        None => eprintln!(
            "repacked {} → {} (fs: tar → tar, {written} bytes)",
            tar_path.display(),
            dst.display()
        ),
    }
    Ok(())
}

/// Drive a multi-source repack from an in-memory [`fstool::merge::MergeModel`]
/// straight into the destination — no flatten-to-tempfile step. The model
/// is built once and reused for both sizing and the walk; tar layers are
/// read forward (in the order entries appear), host layers are random-access.
#[allow(clippy::too_many_arguments)]
fn repack_layered_to_dst(
    layers: Vec<fstool::repack::Source>,
    dst: &std::path::Path,
    size_arg: Option<&str>,
    shrink: bool,
    fs_type_override: Option<&str>,
    block_size: u32,
    qcow2_cluster_size: u32,
) -> fstool::Result<()> {
    use fstool::merge::MergeModel;
    use fstool::repack::{FsSink, RepackSink, TarStreamSink};
    let _ = shrink; // sizing always derives from the model when no explicit size

    // Pass 1 — build the metadata-only model. No file body is read here.
    let model = MergeModel::build(&layers)?;
    let analysis = model.analysis(block_size);

    // Resolve destination fs-type. With a layered source there's no source
    // "kind" to preserve — default to ext4 unless the dst extension says
    // tar.
    let dst_tar_codec = tar_output_codec(dst);
    let target_fs = fs_type_override
        .map(|s| s.to_string())
        .or_else(|| {
            dst.extension()
                .and_then(|s| s.to_str())
                .filter(|e| e.eq_ignore_ascii_case("tar"))
                .map(|_| "tar".to_string())
        })
        .or_else(|| dst_tar_codec.map(|_| "tar".to_string()))
        .unwrap_or_else(|| "ext4".to_string());
    let lower = target_fs.to_ascii_lowercase();

    let explicit = match size_arg {
        Some(s) => Some(fstool::spec::parse_size(s)?),
        None => None,
    };
    let dst_size = if let Some(sz) = explicit {
        sz
    } else if lower == "tar" {
        0
    } else if let Some(sz) = analysis.recommended_size(&lower) {
        sz
    } else {
        // Over-provision: archive writers truncate via `image_len()` after
        // flush; block FSes fill the device.
        analysis
            .total_file_bytes
            .saturating_mul(2)
            .saturating_add(64 * 1024 * 1024)
    };

    // tar output is sequential — write straight through a `TarStreamSink`,
    // optionally codec-wrapped. No backing device.
    if lower == "tar" {
        let file = std::fs::File::create(dst)?;
        let buffered: Box<dyn std::io::Write> =
            Box::new(std::io::BufWriter::with_capacity(64 * 1024, file));
        let inner = match dst_tar_codec {
            Some(algo) => fstool::compression::make_writer(algo, buffered)?,
            None => buffered,
        };
        let mut sink = TarStreamSink::new(inner);
        model.walk_into_sink(&layers, &mut sink)?;
        sink.finish()?;
        let written = sink.bytes_written();
        match dst_tar_codec {
            Some(algo) => eprintln!(
                "repacked layered → {} (fs: layered → tar.{}, {written} bytes plain)",
                dst.display(),
                algo.name()
            ),
            None => eprintln!(
                "repacked layered → {} (fs: layered → tar, {written} bytes)",
                dst.display()
            ),
        }
        return Ok(());
    }

    fstool::repack::phase(&format!(
        "formatting {target_fs} destination ({}) …",
        human_size(dst_size)
    ));
    // Totals from the merge model power the copy-phase progress bar.
    let total_entries = analysis.files + analysis.dirs + analysis.symlinks + analysis.devices;
    fstool::repack::set_total(total_entries, analysis.total_file_bytes);
    let mut dst_dev = fstool::block::create_image(
        dst,
        dst_size,
        &fstool::block::CreateOpts {
            cluster_size: qcow2_cluster_size,
        },
    )?;

    let archive_len: Option<u64> = match lower.as_str() {
        "ext2" | "ext3" | "ext4" => {
            use fstool::fs::ext::{Ext, FsKind};
            let kind = match lower.as_str() {
                "ext2" => FsKind::Ext2,
                "ext3" => FsKind::Ext3,
                _ => FsKind::Ext4,
            };
            let mut opts = analysis.ext_format_opts(kind);
            opts.sparse = true;
            // dst_dev is a fresh image from `create_image` above.
            opts.prezeroed = true;
            let plan_size = opts.blocks_count as u64 * opts.block_size as u64;
            if dst_size > plan_size {
                let max = (dst_size / opts.block_size as u64) as u32;
                opts.blocks_count = (max / 8) * 8;
                let by_density =
                    (opts.blocks_count as u64 * opts.block_size as u64 / 16_384) as u32;
                opts.inodes_count = opts.inodes_count.max(by_density);
            }
            let mut dst = Ext::format_with(dst_dev.as_mut(), &opts)?;
            {
                let mut sink = FsSink::new(&mut dst, dst_dev.as_mut());
                model.walk_into_sink(&layers, &mut sink)?;
            }
            dst.flush(dst_dev.as_mut())?;
            None
        }
        "fat32" | "vfat" => {
            let total_sectors: u32 = (dst_size / 512).try_into().map_err(|_| {
                fstool::Error::InvalidArgument(
                    "repack: FAT32 size doesn't fit in a u32 sector count".into(),
                )
            })?;
            let opts = fstool::fs::fat::FatFormatOpts {
                total_sectors,
                volume_id: 0,
                volume_label: *b"REPACKED   ",
            };
            let mut dst = fstool::fs::fat::Fat32::format(dst_dev.as_mut(), &opts)?;
            {
                let mut sink = FsSink::new(&mut dst, dst_dev.as_mut()).lossy();
                model.walk_into_sink(&layers, &mut sink)?;
            }
            dst.flush(dst_dev.as_mut())?;
            None
        }
        "xfs" => repack_layered_via_trait::<fstool::fs::xfs::Xfs>(
            dst_dev.as_mut(),
            &fstool::fs::xfs::format::FormatOpts::default(),
            &model,
            &layers,
            false,
        )?,
        "hfsplus" | "hfs+" => repack_layered_via_trait::<fstool::fs::hfs_plus::HfsPlus>(
            dst_dev.as_mut(),
            &fstool::fs::hfs_plus::FormatOpts::default(),
            &model,
            &layers,
            false,
        )?,
        "ntfs" => repack_layered_via_trait::<fstool::fs::ntfs::Ntfs>(
            dst_dev.as_mut(),
            &fstool::fs::ntfs::format::FormatOpts::default(),
            &model,
            &layers,
            false,
        )?,
        "f2fs" => repack_layered_via_trait::<fstool::fs::f2fs::F2fs>(
            dst_dev.as_mut(),
            &fstool::fs::f2fs::FormatOpts::default(),
            &model,
            &layers,
            false,
        )?,
        "squashfs" => repack_layered_via_trait::<fstool::fs::squashfs::Squashfs>(
            dst_dev.as_mut(),
            &fstool::fs::squashfs::FormatOpts::default(),
            &model,
            &layers,
            false,
        )?,
        "iso" | "iso9660" => {
            let opts = fstool::fs::iso9660::FormatOpts {
                volume_id: "FSTOOL".into(),
                application_id: "fstool".into(),
                ..fstool::fs::iso9660::FormatOpts::default()
            };
            repack_layered_via_trait::<fstool::fs::iso9660::Iso9660>(
                dst_dev.as_mut(),
                &opts,
                &model,
                &layers,
                false,
            )?
        }
        "grf" => repack_layered_via_trait::<fstool::fs::grf::Grf>(
            dst_dev.as_mut(),
            &fstool::fs::grf::FormatOpts::default(),
            &model,
            &layers,
            false,
        )?,
        "zip" => repack_layered_via_trait::<fstool::fs::archive::zip::ZipFs>(
            dst_dev.as_mut(),
            &fstool::fs::archive::zip::ZipFormatOpts::default(),
            &model,
            &layers,
            true,
        )?,
        "cpio" => repack_layered_via_trait::<fstool::fs::archive::cpio::CpioFs>(
            dst_dev.as_mut(),
            &fstool::fs::archive::cpio::CpioFormatOpts,
            &model,
            &layers,
            false,
        )?,
        other => {
            return Err(fstool::Error::InvalidArgument(format!(
                "repack: unknown --fs-type {other:?}"
            )));
        }
    };
    dst_dev.sync()?;
    let report_size = match archive_len {
        Some(len) => {
            drop(dst_dev);
            truncate_output_file(dst, len)?;
            len
        }
        None => {
            drop(dst_dev);
            dst_size
        }
    };
    eprintln!(
        "repacked layered → {} (fs: layered → {lower}, {report_size} bytes)",
        dst.display()
    );
    Ok(())
}

/// Generic trait-based variant of [`repack_layered_to_dst`]'s dispatch
/// arms: format `F` on `dst_dev`, walk the in-memory model into an
/// `FsSink`, flush. The model is built once by the caller and reused so
/// the per-layer tar index is scanned exactly once.
fn repack_layered_via_trait<F: fstool::fs::FilesystemFactory>(
    dst_dev: &mut dyn fstool::block::BlockDevice,
    opts: &F::FormatOpts,
    model: &fstool::merge::MergeModel,
    layers: &[fstool::repack::Source],
    lossy: bool,
) -> fstool::Result<Option<u64>> {
    let mut dst = F::format(dst_dev, opts)?;
    {
        let mut sink = fstool::repack::FsSink::new(&mut dst, dst_dev);
        if lossy {
            sink = sink.lossy();
        }
        model.walk_into_sink(layers, &mut sink)?;
    }
    dst.flush(dst_dev)?;
    Ok(fstool::fs::Filesystem::image_len(&dst))
}

#[allow(clippy::too_many_arguments)]
fn repack_tar_stream_to_fs(
    tar_path: &std::path::Path,
    codec: Option<fstool::compression::Algo>,
    dst: &std::path::Path,
    lower: &str,
    size_arg: Option<&str>,
    shrink: bool,
    block_size: u32,
    qcow2_cluster_size: u32,
) -> fstool::Result<()> {
    use fstool::fs::ext::{Ext, FsKind};
    let _ = shrink; // sizing always uses a pass when no explicit size

    let explicit = match size_arg {
        Some(s) => Some(fstool::spec::parse_size(s)?),
        None => None,
    };

    // One streaming content scan yields everything sizing needs: the ext
    // FormatOpts (inode count, group layout) and the file-byte total for
    // the FAT/other heuristics. (Replaces the former PlanSink/ByteSumSink
    // passes with a single `analyze` pass.)
    let analysis = fstool::analyze::analyze_source(
        &fstool::repack::Source::TarArchive {
            path: tar_path.to_path_buf(),
            codec,
        },
        block_size,
    )?;

    // ext needs full FormatOpts regardless of whether the size is explicit.
    let mut ext_opts = match lower {
        "ext2" | "ext3" | "ext4" => {
            let kind = match lower {
                "ext2" => FsKind::Ext2,
                "ext3" => FsKind::Ext3,
                _ => FsKind::Ext4,
            };
            let mut opts = analysis.ext_format_opts(kind);
            opts.sparse = true;
            // dst_dev is created fresh below via `create_image`, so the
            // formatter can skip its full-device zero pass.
            opts.prezeroed = true;
            Some(opts)
        }
        _ => None,
    };

    // Destination size: explicit wins; ext derives from its plan; fat32
    // uses the analyze recommendation; the remaining self-sizing block
    // FSes (xfs/hfs+/ntfs/f2fs) get a generous upper bound.
    let dst_size = if let Some(sz) = explicit {
        sz
    } else if let Some(opts) = &ext_opts {
        opts.blocks_count as u64 * opts.block_size as u64
    } else if let Some(sz) = analysis.recommended_size(lower) {
        sz
    } else {
        analysis
            .total_file_bytes
            .saturating_mul(2)
            .saturating_add(64 * 1024 * 1024)
    };

    // Grow the ext plan's image to an explicitly-requested larger size
    // (mirrors the random-access path's adjustment).
    if let Some(opts) = ext_opts.as_mut() {
        let plan_size = opts.blocks_count as u64 * opts.block_size as u64;
        if dst_size > plan_size {
            let max = (dst_size / opts.block_size as u64) as u32;
            opts.blocks_count = (max / 8) * 8;
            let by_density = (opts.blocks_count as u64 * opts.block_size as u64 / 16_384) as u32;
            opts.inodes_count = opts.inodes_count.max(by_density);
        }
    }

    fstool::repack::phase(&format!(
        "formatting {lower} destination ({}) …",
        human_size(dst_size)
    ));
    // Hand the totals from the analyze pass to the progress sink so the
    // copy phase renders a bar instead of a filename ticker.
    let total_entries = analysis.files + analysis.dirs + analysis.symlinks + analysis.devices;
    fstool::repack::set_total(total_entries, analysis.total_file_bytes);
    let mut dst_dev = fstool::block::create_image(
        dst,
        dst_size,
        &fstool::block::CreateOpts {
            cluster_size: qcow2_cluster_size,
        },
    )?;

    // `Some(len)` for the deferred archive formats (squashfs/iso/grf) so
    // the over-provisioned backing file is truncated to its real length;
    // `None` for the fixed-size block filesystems.
    let archive_len: Option<u64> = match lower {
        "ext2" | "ext3" | "ext4" => {
            let opts = ext_opts.expect("ext opts computed above");
            let mut dst_ext = Ext::format_with(dst_dev.as_mut(), &opts)?;
            {
                let mut sink = fstool::repack::FsSink::new(&mut dst_ext, dst_dev.as_mut());
                let mut reader = fstool::repack::open_tar_stream(tar_path, codec)?;
                fstool::repack::walk_tar_stream(&mut reader, &mut sink)?;
            }
            dst_ext.flush(dst_dev.as_mut())?;
            None
        }
        "fat32" | "vfat" => {
            let total_sectors: u32 = (dst_size / 512).try_into().map_err(|_| {
                fstool::Error::InvalidArgument(
                    "repack: FAT32 size doesn't fit in a u32 sector count".into(),
                )
            })?;
            let opts = fstool::fs::fat::FatFormatOpts {
                total_sectors,
                volume_id: 0,
                volume_label: *b"REPACKED   ",
            };
            let mut dst_fat = fstool::fs::fat::Fat32::format(dst_dev.as_mut(), &opts)?;
            {
                let mut sink = fstool::repack::FsSink::new(&mut dst_fat, dst_dev.as_mut()).lossy();
                let mut reader = fstool::repack::open_tar_stream(tar_path, codec)?;
                fstool::repack::walk_tar_stream(&mut reader, &mut sink)?;
            }
            dst_fat.flush(dst_dev.as_mut())?;
            None
        }
        "xfs" => repack_stream_via_trait::<fstool::fs::xfs::Xfs>(
            dst_dev.as_mut(),
            &fstool::fs::xfs::format::FormatOpts::default(),
            tar_path,
            codec,
            false,
        )?,
        "hfsplus" | "hfs+" => repack_stream_via_trait::<fstool::fs::hfs_plus::HfsPlus>(
            dst_dev.as_mut(),
            &fstool::fs::hfs_plus::FormatOpts::default(),
            tar_path,
            codec,
            false,
        )?,
        "ntfs" => repack_stream_via_trait::<fstool::fs::ntfs::Ntfs>(
            dst_dev.as_mut(),
            &fstool::fs::ntfs::format::FormatOpts::default(),
            tar_path,
            codec,
            false,
        )?,
        "f2fs" => repack_stream_via_trait::<fstool::fs::f2fs::F2fs>(
            dst_dev.as_mut(),
            &fstool::fs::f2fs::FormatOpts::default(),
            tar_path,
            codec,
            false,
        )?,
        "squashfs" => repack_stream_via_trait::<fstool::fs::squashfs::Squashfs>(
            dst_dev.as_mut(),
            &fstool::fs::squashfs::FormatOpts::default(),
            tar_path,
            codec,
            false,
        )?,
        "iso" | "iso9660" => {
            let opts = fstool::fs::iso9660::FormatOpts {
                volume_id: "FSTOOL".into(),
                application_id: "fstool".into(),
                ..fstool::fs::iso9660::FormatOpts::default()
            };
            repack_stream_via_trait::<fstool::fs::iso9660::Iso9660>(
                dst_dev.as_mut(),
                &opts,
                tar_path,
                codec,
                false,
            )?
        }
        "grf" => repack_stream_via_trait::<fstool::fs::grf::Grf>(
            dst_dev.as_mut(),
            &fstool::fs::grf::FormatOpts::default(),
            tar_path,
            codec,
            false,
        )?,
        "zip" => repack_stream_via_trait::<fstool::fs::archive::zip::ZipFs>(
            dst_dev.as_mut(),
            &fstool::fs::archive::zip::ZipFormatOpts::default(),
            tar_path,
            codec,
            true,
        )?,
        "cpio" => repack_stream_via_trait::<fstool::fs::archive::cpio::CpioFs>(
            dst_dev.as_mut(),
            &fstool::fs::archive::cpio::CpioFormatOpts,
            tar_path,
            codec,
            false,
        )?,
        other => unreachable!("repack_tar_stream_to_fs reached for ungated fs {other:?}"),
    };
    dst_dev.sync()?;
    let report_size = match archive_len {
        Some(len) => {
            drop(dst_dev);
            truncate_output_file(dst, len)?;
            len
        }
        None => {
            drop(dst_dev);
            dst_size
        }
    };

    eprintln!(
        "repacked {} → {} (fs: tar → {lower}, {report_size} bytes)",
        tar_path.display(),
        dst.display()
    );
    Ok(())
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

/// Open a (possibly codec-wrapped) tar archive as a streaming reader.
fn open_tar_stream_reader(
    path: &str,
    algo: Option<fstool::compression::Algo>,
) -> fstool::Result<fstool::fs::tar::TarStreamReader<Box<dyn std::io::Read>>> {
    // Only strip a `:N` partition suffix where N is purely numeric — a
    // Windows path like `C:\foo\src.tar` must keep its drive-letter colon.
    let p = match path.rsplit_once(':') {
        Some((head, tail)) if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => {
            std::path::Path::new(head)
        }
        _ => std::path::Path::new(path),
    };
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
    recursive: bool,
) -> fstool::Result<()> {
    let index = open_tar_stream_index(image, algo)?;
    let want = normalise_tar_path(path);
    let mut out = std::io::stdout().lock();
    if recursive {
        ls_tar_recursive(&index, &want, &mut out)?;
    } else {
        let children = tar_children(&index, &want);
        if children.is_empty() && want != "/" {
            tar_require_dir(&index, &want)?;
        }
        for e in &children {
            let _ = writeln!(out, "{}\t{:?}\t{}", e.inode, e.kind, e.name);
        }
    }
    Ok(())
}

/// The entries living directly under `want` in a tar stream index, deduped by
/// leaf name (the tar may carry both an explicit dir entry and its members).
fn tar_children(index: &fstool::fs::tar::TarStreamIndex, want: &str) -> Vec<fstool::fs::DirEntry> {
    use fstool::fs::tar::EntryKind as TarKind;
    let mut children: Vec<fstool::fs::DirEntry> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (idx, ix) in index.entries().iter().enumerate() {
        let p = &ix.entry.path;
        if parent_of_tar_path(p) != want {
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
    children
}

/// Error unless `want` exists as a directory (or has descendants) in the index.
fn tar_require_dir(index: &fstool::fs::tar::TarStreamIndex, want: &str) -> fstool::Result<()> {
    let exists_as_dir = index.entries().iter().any(|ix| ix.entry.path == want);
    let has_descendants = index.entries().iter().any(|ix| {
        let p = &ix.entry.path;
        p.starts_with(want) && p.len() > want.len() && p.as_bytes()[want.len()] == b'/'
    });
    if !exists_as_dir && !has_descendants {
        return Err(fstool::Error::InvalidArgument(format!(
            "tar: no such directory {want:?}"
        )));
    }
    Ok(())
}

/// Recursive (`ls -R`) walk of a tar stream index: print `want` under a header,
/// then descend into each subdirectory in listing order.
fn ls_tar_recursive(
    index: &fstool::fs::tar::TarStreamIndex,
    want: &str,
    out: &mut impl std::io::Write,
) -> fstool::Result<()> {
    if want != "/" {
        tar_require_dir(index, want)?;
    }
    let children = tar_children(index, want);
    writeln!(out, "{want}:")?;
    for e in &children {
        writeln!(out, "{}\t{:?}\t{}", e.inode, e.kind, e.name)?;
    }
    writeln!(out)?;
    for e in &children {
        if e.kind == fstool::fs::EntryKind::Dir && is_descendable(&e.name) {
            let child = if want == "/" {
                format!("/{}", e.name)
            } else {
                format!("{want}/{}", e.name)
            };
            // A tar built from `.` carries a root self-entry whose parent is
            // also `/`; never recurse back into the directory we're listing.
            if child != want {
                ls_tar_recursive(index, &child, out)?;
            }
        }
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

// ─── ext → ext (full metadata preservation) ─────────────────────────────

// ─── FAT32 → FAT32 ──────────────────────────────────────────────────────

// ─── ext → FAT32 (drops metadata FAT can't store) ───────────────────────

// ─── FAT32 → ext ────────────────────────────────────────────────────────

// ─── Tar → ext ──────────────────────────────────────────────────────────

// ─── Tar → FAT32 ────────────────────────────────────────────────────────

// ─── shrink sizing ───────────────────────────────────────────────────────

/// Sum the size of every regular file in the source filesystem — used
/// by FAT32 / ISO shrink sizing. Trait-driven walk via
/// [`fstool::inspect::AnyFs::total_file_bytes`].
fn sum_source_file_bytes(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src_fs: &mut fstool::inspect::AnyFs,
) -> fstool::Result<u64> {
    src_fs.total_file_bytes(src_dev)
}

fn shell_cmd(image: &str, ro: bool, style: PathStyle) -> fstool::Result<()> {
    let target = fstool::inspect::Target::parse(image);
    if ro {
        // Read-only shell: open the underlying file O_RDONLY (any
        // write through the BlockDevice fails with PermissionDenied),
        // use AnyFs::open instead of open_writable (so APFS doesn't
        // pay the Write-state spaceman re-parse + journaled FSes
        // don't even consider a replay write), and skip both the
        // compressed-source rejection and the mutability capability
        // gate — compressed and streaming/immutable sources are all
        // legal browse targets. Shell::dispatch will refuse put /
        // rm / mkdir itself.
        return fstool::inspect::with_target_device_read_only(&target, |dev| {
            let fs = fstool::inspect::AnyFs::open(dev)?;
            let mut sh = shell::Shell::new_read_only(fs, style);
            run_shell(&mut sh, dev)?;
            // No dev.sync() — read-only.
            Ok(())
        });
    }
    // Mutating shell: refuse a compressed source up-front (with_target_device
    // would happily decompress to a tempfile and any put/rm against the
    // shell would land on that tempfile, silently lost on exit) and
    // refuse non-mutable filesystems.
    fstool::inspect::reject_compressed_for_mutation(&target)?;
    fstool::inspect::with_target_device(&target, |dev| {
        let fs = fstool::inspect::AnyFs::open_writable(dev)?;
        // Shell exists for in-place mutation (`put` / `rm`). A
        // filesystem that can't support those — tar, ISO 9660,
        // SquashFS, etc. — has nothing useful to offer here; point
        // the user at `fstool ls` / `fstool cat` for read-only
        // browsing of those formats, or pass `--ro` to allow shell
        // browsing without any write capability.
        let cap = fs.mutation_capability();
        if !cap.supports_add_remove() {
            return Err(fstool::Error::InvalidArgument(format!(
                "shell: {} is {} ({}) — shell requires an in-place mutable filesystem; \
                 use `fstool shell --ro` for read-only browsing, or `fstool ls` / \
                 `fstool cat` for one-shot reads",
                target.path.display(),
                fs.kind_string(),
                match cap {
                    fstool::fs::MutationCapability::Streaming => "streaming",
                    fstool::fs::MutationCapability::Immutable => "immutable",
                    _ => unreachable!("supports_add_remove() was false"),
                },
            )));
        }
        let mut sh = shell::Shell::new(fs, style);
        run_shell(&mut sh, dev)?;
        dev.sync()?;
        Ok(())
    })
}

/// Drive a [`shell::Shell`] to completion. On an interactive TTY (and with the
/// `readline` feature on) this uses `rustyline` for line editing + history;
/// otherwise — piped stdin, or the feature disabled — it falls back to the
/// plain line-buffered reader so scripted input stays deterministic.
fn run_shell(
    sh: &mut shell::Shell,
    dev: &mut dyn fstool::block::BlockDevice,
) -> fstool::Result<()> {
    #[cfg(feature = "readline")]
    {
        use std::io::IsTerminal;
        if std::io::stdin().is_terminal() {
            return sh.run_interactive(dev);
        }
    }
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    sh.run(dev, stdin.lock(), stdout.lock())
}

fn rm(image: &str, fs_path: &str, style: PathStyle) -> fstool::Result<()> {
    let target = fstool::inspect::Target::parse(image);
    fstool::inspect::reject_compressed_for_mutation(&target)?;
    fstool::inspect::with_target_device(&target, |dev| {
        let mut fs = fstool::inspect::AnyFs::open_writable(dev)?;
        let cpath = path_style::to_canonical(fs_path, fs.kind(), style);
        fs.remove(dev, &cpath)?;
        fs.flush(dev)?;
        dev.sync()?;
        eprintln!("removed {fs_path}");
        Ok(())
    })
}

fn add(
    image: &str,
    host_src: &std::path::Path,
    fs_dest: &str,
    style: PathStyle,
) -> fstool::Result<()> {
    let meta = std::fs::symlink_metadata(host_src)?;
    let target = fstool::inspect::Target::parse(image);
    fstool::inspect::reject_compressed_for_mutation(&target)?;
    fstool::inspect::with_target_device(&target, |dev| {
        let mut fs = fstool::inspect::AnyFs::open_writable(dev)?;
        // Only the in-image destination is style-translated; `host_src` is a
        // real host path and must be left untouched.
        let dest = path_style::to_canonical(fs_dest, fs.kind(), style);
        if meta.is_dir() {
            fs.add_dir_tree(dev, &dest, host_src)?;
        } else if meta.is_file() {
            fs.add_file(dev, &dest, host_src)?;
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
        "zip" => create_via_factory::<fstool::fs::archive::zip::ZipFs>(
            "zip",
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
            fstool::fs::archive::zip::ZipFormatOpts::default(),
            |o, m| o.apply_options(m),
            DEFAULT_MIN_SIZE,
        )?,
        "cpio" => create_via_factory::<fstool::fs::archive::cpio::CpioFs>(
            "cpio",
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
            fstool::fs::archive::cpio::CpioFormatOpts,
            |o, m| o.apply_options(m),
            DEFAULT_MIN_SIZE,
        )?,
        "ar" => create_via_factory::<fstool::fs::archive::ar::ArFs>(
            "ar",
            source.as_ref(),
            args.output,
            args.size,
            opts,
            is_device,
            qcow2_cluster_size,
            fstool::fs::archive::ar::ArFormatOpts,
            |o, m| o.apply_options(m),
            DEFAULT_MIN_SIZE,
        )?,
        other => {
            return Err(fstool::Error::InvalidArgument(format!(
                "create: unknown --type {other:?} (try ext4, fat32, exfat, hfs+, ntfs, \
                 f2fs, squashfs, xfs, iso, grf, apfs, zip, cpio, ar)"
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
    let block_size = bag.take_u32("block_size")?.unwrap_or(1024);

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

    // `dev` was just created by `create_image` — a freshly-`set_len`'d
    // sparse raw file or a fresh qcow2 — so it reads back as zero. Skip
    // the formatter's full-device zero pass.
    format_opts.prezeroed = true;
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
    let label_str = bag
        .take_str("volume_label")
        .unwrap_or_else(|| "NO NAME".into());
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
                 (FAT32 needs ≥ ~33 MiB)"
                    .into(),
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
#[allow(clippy::too_many_arguments)] // generic dispatcher — each arg is a distinct knob
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

    // When the user supplied a source but no `--size`, walk it once to
    // pick a size that fits. Without this, a host-dir source always got
    // the 1 MiB `default_min_size` and the writer overran the device on
    // anything larger than empty. We over-provision (`total_file_bytes ×
    // 2 + 64 MiB`) and let the deferred-archive backends (squashfs / iso
    // / grf) truncate via `image_len` after flush; fixed-size FSes just
    // get a comfortable image.
    let auto_size = if !is_device
        && size_arg.is_none()
        && let Some(src) = source
    {
        let analysis = fstool::analyze::analyze_source(src, 1024)?;
        Some(
            analysis
                .total_file_bytes
                .saturating_mul(2)
                .saturating_add(64 * 1024 * 1024)
                .max(default_min_size),
        )
    } else {
        None
    };
    let bytes = match auto_size {
        Some(b) => b,
        None => resolve_size_for_dev(output, size_arg, is_device, default_min_size)?,
    };
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
    // Archive writers fill a generously-sized sparse file; truncate it
    // to the true archive length (filesystem images return `None`).
    let archive_len = fstool::fs::Filesystem::image_len(&fs);
    drop(fs);
    drop(dev);
    let final_len = match archive_len {
        Some(len) if !is_device => {
            truncate_output_file(output, len)?;
            len
        }
        _ => bytes,
    };
    eprintln!(
        "wrote {} ({} bytes, {label}{})",
        output.display(),
        final_len,
        if is_device { ", block device" } else { "" }
    );
    Ok(())
}

/// Truncate an over-provisioned archive output file to its true length.
/// No-op for block devices and qcow2 containers — archives target plain
/// files, and the provisioned sparse tail is only zeros.
fn truncate_output_file(output: &std::path::Path, len: u64) -> fstool::Result<()> {
    use fstool::block::file::is_block_device;
    if is_block_device(output)
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

fn ls(image: &str, path: &str, recursive: bool, style: PathStyle) -> fstool::Result<()> {
    // Compressed-tar archives stream-walk per invocation; bypass the
    // tempfile-spooling BlockDevice path entirely. Tar's separator is `/`, so
    // path-style is a no-op there — pass the path through unchanged.
    if let Some(algo) = tar_input_codec(image) {
        return ls_tar_stream(image, path, Some(algo), recursive);
    }
    let target = fstool::inspect::Target::parse(image);
    fstool::inspect::with_target_device(&target, |dev| {
        let mut fs = fstool::inspect::AnyFs::open(dev)?;
        let kind = fs.kind();
        let cpath = path_style::to_canonical(path, kind, style);
        let mut out = std::io::stdout().lock();
        if recursive {
            ls_recursive(&mut fs, dev, &cpath, kind, style, &mut out)?;
        } else {
            let entries = fs.list(dev, &cpath)?;
            for e in &entries {
                let _ = writeln!(
                    out,
                    "{}\t{:?}\t{}",
                    e.inode,
                    e.kind,
                    path_style::display_name(&e.name, kind, style)
                );
            }
        }
        Ok(())
    })
}

/// Depth-first `ls -R` over an opened filesystem: print the directory under a
/// `path:` header, then recurse into each child directory in listing order.
/// Only `EntryKind::Dir` children are descended (symlinks are never followed,
/// so there are no loops to guard against); `.`/`..` are skipped defensively.
fn ls_recursive(
    fs: &mut fstool::inspect::AnyFs,
    dev: &mut dyn fstool::block::BlockDevice,
    dir: &str,
    kind: fstool::inspect::FsKind,
    style: PathStyle,
    out: &mut impl std::io::Write,
) -> fstool::Result<()> {
    let entries = fs.list(dev, dir)?;
    // `dir` and `e.name` are canonical; translate ONLY for display. The
    // recursion below must keep using the canonical values, or a name with a
    // separator-collision (e.g. HFS "A/ROSE Includes") would be mis-resolved.
    writeln!(out, "{}:", path_style::display_path(dir, kind, style))?;
    for e in &entries {
        writeln!(
            out,
            "{}\t{:?}\t{}",
            e.inode,
            e.kind,
            path_style::display_name(&e.name, kind, style)
        )?;
    }
    writeln!(out)?;
    for e in &entries {
        if e.kind == fstool::fs::EntryKind::Dir && is_descendable(&e.name) {
            let child = join_image_path(dir, &e.name);
            // Guard against a directory that lists itself as a child (some
            // containers carry a `.`/`/` self-entry) — that would recurse
            // forever.
            if child != dir {
                ls_recursive(fs, dev, &child, kind, style, out)?;
            }
        }
    }
    Ok(())
}

/// Whether a child entry name is a real subdirectory worth descending into:
/// not empty, not the `.`/`..` self/parent links.
fn is_descendable(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".."
}

/// Join an image-internal directory path with a child name, normalising the
/// slash so the root (`/`) doesn't produce a doubled `//`.
fn join_image_path(dir: &str, name: &str) -> String {
    if dir.ends_with('/') {
        format!("{dir}{name}")
    } else {
        format!("{dir}/{name}")
    }
}

fn cat(image: &str, path: &str, style: PathStyle) -> fstool::Result<()> {
    if let Some(algo) = tar_input_codec(image) {
        return cat_tar_stream(image, path, Some(algo));
    }
    let target = fstool::inspect::Target::parse(image);
    fstool::inspect::with_target_device(&target, |dev| {
        let mut fs = fstool::inspect::AnyFs::open(dev)?;
        let cpath = path_style::to_canonical(path, fs.kind(), style);
        let mut out = std::io::stdout().lock();
        fs.copy_file_to(dev, &cpath, &mut out)?;
        Ok(())
    })
}

fn analyze_cmd(
    source: &str,
    fs_type: Option<&str>,
    block_size: u32,
    json: bool,
) -> fstool::Result<()> {
    let src = fstool::repack::Source::detect(source)?;
    let analysis = fstool::analyze::analyze_source(&src, block_size)?;
    let fs_types: Vec<&str> = match fs_type {
        Some(t) => vec![t],
        None => fstool::analyze::SIZED_FS_TYPES.to_vec(),
    };
    let report = analysis.report(&fs_types);

    if json {
        let s = serde_json::to_string_pretty(&report)
            .map_err(|e| fstool::Error::Io(std::io::Error::other(e)))?;
        println!("{s}");
        return Ok(());
    }

    println!("source:    {source}");
    println!("files:     {}", report.files);
    println!("dirs:      {}", report.dirs);
    println!("symlinks:  {}", report.symlinks);
    println!("devices:   {}", report.devices);
    println!("hardlinks: {}", report.hardlinks);
    println!(
        "file data: {} ({} bytes)",
        human_size(report.total_file_bytes),
        report.total_file_bytes
    );
    println!("inodes:    {}", report.inode_count);
    println!(
        "recommended image size (ext block size {}):",
        report.block_size
    );
    if report.recommended_size.is_empty() {
        println!("  (none — destination grows/truncates; no fixed size needed)");
    } else {
        for (t, sz) in &report.recommended_size {
            println!("  {t:<8} {} ({sz} bytes)", human_size(*sz));
        }
    }
    Ok(())
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
        fstool::inspect::AnyFs::Hfs(hfs) => print_hfs_info(hfs),
        fstool::inspect::AnyFs::Apfs(apfs) => print_apfs_info(apfs),
        fstool::inspect::AnyFs::Ntfs(ntfs) => print_ntfs_info(ntfs),
        fstool::inspect::AnyFs::F2fs(f2) => print_f2fs_info(f2),
        fstool::inspect::AnyFs::Squashfs(sq) => print_squashfs_info(sq),
        fstool::inspect::AnyFs::Iso9660(iso) => print_iso9660_info(iso),
        fstool::inspect::AnyFs::Grf(grf) => print_grf_info(grf),
        // Archive backends carry no extra summary beyond the kind line
        // above; the `/ listing` below covers their contents.
        fstool::inspect::AnyFs::Archive(..) => {}
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

fn print_hfs_info(hfs: &fstool::fs::hfs::Hfs) {
    println!("volume name:       {:?}", hfs.volume_name);
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
