//! fstool CLI — thin wrapper over the library.
//!
//! Subcommands (P5 first cut):
//!
//! - `fstool ext-build` — build a bare ext2/3/4 image from a host directory.
//! - `fstool ls`        — list the contents of a directory inside an image.
//! - `fstool cat`       — print a regular file's contents to stdout.
//! - `fstool info`      — show a one-screen summary of an existing image.
//!
//! Full TOML-spec-driven `fstool build` and `fstool add` / `fstool rm` land
//! in P5b / P5c — see the project README.

mod shell;

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

use fstool::block::{BlockDevice, FileBackend};
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
    /// Build a bare ext2 / ext3 / ext4 image from a host directory.
    /// (Genext2fs-replacement mode — no partition table.)
    ExtBuild {
        /// Source directory on the host filesystem.
        #[arg(value_name = "SRC_DIR")]
        src_dir: PathBuf,
        /// Output image file or block device. Block devices are formatted
        /// to their full capacity; regular files are auto-sized to the
        /// source tree.
        #[arg(short = 'o', long = "output", value_name = "IMAGE")]
        output: PathBuf,
        /// Which ext flavour to write.
        #[arg(long, value_enum, default_value_t = ExtKindArg::Ext2)]
        kind: ExtKindArg,
        /// Block size in bytes (1024, 2048, or 4096).
        #[arg(long, default_value_t = 1024)]
        block_size: u32,
        /// Write files sparsely: all-zero blocks become holes.
        #[arg(long)]
        sparse: bool,
        /// Required when OUTPUT is a block device — refuses to format
        /// a real device without an explicit opt-in.
        #[arg(long)]
        force: bool,
        /// qcow2 cluster size (only honoured when OUTPUT ends in
        /// `.qcow2` / `.qcow` / `.q2`). Accepts `64KiB`, `1MiB`, or a
        /// bare byte count; must be a power of two ≥ 512. Default 64 KiB.
        #[arg(long, value_name = "SIZE", default_value = "64KiB")]
        cluster_size: String,
    },

    /// Build a bare FAT32 image from a host directory. FAT32 has no
    /// streaming auto-size (it needs ≥ 65525 clusters → ~33 MiB minimum),
    /// so `--size` is required for regular-file output; block-device
    /// output uses the device's full capacity instead.
    FatBuild {
        /// Source directory on the host filesystem.
        #[arg(value_name = "SRC_DIR")]
        src_dir: PathBuf,
        /// Output image file or block device.
        #[arg(short = 'o', long = "output", value_name = "IMAGE")]
        output: PathBuf,
        /// Image size, e.g. "64MiB" or "1GiB". Ignored when OUTPUT is a
        /// block device (the device's capacity wins).
        #[arg(long, value_name = "SIZE")]
        size: Option<String>,
        /// Volume label (up to 11 ASCII bytes, upper-cased).
        #[arg(long, default_value = "NO NAME")]
        label: String,
        /// Volume ID / serial number. Default 0 for reproducible output.
        #[arg(long, default_value_t = 0)]
        volume_id: u32,
        /// Required when OUTPUT is a block device — refuses to format
        /// a real device without an explicit opt-in.
        #[arg(long)]
        force: bool,
        /// qcow2 cluster size (only honoured when OUTPUT is a qcow2
        /// path). Default 64 KiB.
        #[arg(long, value_name = "SIZE", default_value = "64KiB")]
        cluster_size: String,
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
        /// Source image, optionally with `:N` to select a partition.
        #[arg(value_name = "SRC[:N]")]
        src: String,
        /// Destination image (raw or qcow2 per extension).
        #[arg(value_name = "DST")]
        dst: PathBuf,
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
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ExtKindArg {
    Ext2,
    Ext3,
    Ext4,
}

impl From<ExtKindArg> for FsKind {
    fn from(a: ExtKindArg) -> Self {
        match a {
            ExtKindArg::Ext2 => FsKind::Ext2,
            ExtKindArg::Ext3 => FsKind::Ext3,
            ExtKindArg::Ext4 => FsKind::Ext4,
        }
    }
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
        Command::ExtBuild {
            src_dir,
            output,
            kind,
            block_size,
            sparse,
            force,
            cluster_size,
        } => ext_build(
            &src_dir,
            &output,
            kind.into(),
            block_size,
            sparse,
            force,
            &cluster_size,
        ),
        Command::FatBuild {
            src_dir,
            output,
            size,
            label,
            volume_id,
            force,
            cluster_size,
        } => fat_build(
            &src_dir,
            &output,
            size.as_deref(),
            &label,
            volume_id,
            force,
            &cluster_size,
        ),
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
            src,
            dst,
            size,
            shrink,
            fs_type,
            block_size,
            cluster_size,
        } => repack_cmd(
            &src,
            &dst,
            size.as_deref(),
            shrink,
            fs_type.as_deref(),
            block_size,
            &cluster_size,
        ),
    }
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
    src: &str,
    dst: &std::path::Path,
    size_arg: Option<&str>,
    shrink: bool,
    fs_type_override: Option<&str>,
    block_size: u32,
    cluster_size: &str,
) -> fstool::Result<()> {
    let qcow2_cluster_size = parse_cluster_size(cluster_size)?;
    let src_target = fstool::inspect::Target::parse(src);

    // Open the source once and walk it; the source FS stays open across
    // the destination build so we stream each file straight through
    // without ever touching the host filesystem.
    fstool::inspect::with_target_device(&src_target, |src_dev| {
        let src_fs = fstool::inspect::AnyFs::open(src_dev)?;
        let source_kind = src_fs.kind_string();
        let src_total = src_dev.total_size();

        // Compute the destination size + a BuildPlan-shaped sketch by
        // walking the source FS, no host I/O involved.
        let target_fs_str = fs_type_override.unwrap_or(source_kind).to_string();
        let lower = target_fs_str.to_ascii_lowercase();
        let dst_size = match (size_arg, shrink) {
            (Some(s), _) => fstool::spec::parse_size(s)?,
            (None, true) => match lower.as_str() {
                "ext2" | "ext3" | "ext4" => {
                    let plan = build_ext_plan(src_dev, &src_fs, block_size, &lower)?;
                    plan.blocks_count() as u64 * plan.block_size as u64
                }
                "fat32" | "vfat" => {
                    let bytes = sum_source_file_bytes(src_dev, &src_fs)?;
                    let needed = bytes
                        .saturating_mul(2)
                        .max(fstool::fs::fat::MIN_FAT32_CLUSTERS as u64 * 1024);
                    needed.div_ceil(512) * 512
                }
                other => {
                    return Err(fstool::Error::InvalidArgument(format!(
                        "repack: unknown --fs-type {other:?}"
                    )));
                }
            },
            (None, false) => src_total,
        };

        // Format the destination.
        let mut dst_dev = fstool::block::create_image(
            dst,
            dst_size,
            &fstool::block::CreateOpts {
                cluster_size: qcow2_cluster_size,
            },
        )?;
        match lower.as_str() {
            "ext2" | "ext3" | "ext4" => {
                let plan = build_ext_plan(src_dev, &src_fs, block_size, &lower)?;
                let mut opts = plan.to_format_opts();
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
    }
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
        match mode_type {
            t if t == fstool::fs::ext::constants::S_IFREG => {
                // Stream cluster-by-cluster: read the source file's bytes
                // into the destination ext's per-block buffer via a
                // borrowed Read — no host filesystem involvement, no
                // 'static-lifetime wrappers.
                let mut reader = src.open_file_reader(src_dev, e.inode)?;
                dst.add_file_to_streaming(
                    dst_dev,
                    dst_ino,
                    name,
                    &mut reader,
                    inode.size as u64,
                    meta,
                )?;
            }
            t if t == fstool::fs::ext::constants::S_IFDIR => {
                let new_ino = dst.add_dir_to(dst_dev, dst_ino, name, meta)?;
                copy_ext_dir(src_dev, src, e.inode, dst_dev, dst, new_ino)?;
            }
            t if t == fstool::fs::ext::constants::S_IFLNK => {
                let target = src.read_symlink_target(src_dev, e.inode)?;
                dst.add_symlink_to(dst_dev, dst_ino, name, target.as_bytes(), meta)?;
            }
            t if t == fstool::fs::ext::constants::S_IFCHR => {
                let (major, minor) = decode_devnum(inode.block[0]);
                dst.add_device_to(dst_dev, dst_ino, name, DeviceKind::Char, major, minor, meta)?;
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
                )?;
            }
            t if t == fstool::fs::ext::constants::S_IFIFO => {
                dst.add_device_to(dst_dev, dst_ino, name, DeviceKind::Fifo, 0, 0, meta)?;
            }
            t if t == fstool::fs::ext::constants::S_IFSOCK => {
                dst.add_device_to(dst_dev, dst_ino, name, DeviceKind::Socket, 0, 0, meta)?;
            }
            _ => {
                eprintln!(
                    "repack: skipping inode {} ({:?}) — unknown mode {:#o}",
                    e.inode, e.name, inode.mode
                );
            }
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
/// without touching the host filesystem.
fn build_ext_plan(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src_fs: &fstool::inspect::AnyFs,
    block_size: u32,
    fs_kind_str: &str,
) -> fstool::Result<fstool::fs::ext::BuildPlan> {
    use fstool::fs::ext::{BuildPlan, FsKind};
    use fstool::inspect::AnyFs;
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
    match src_fs {
        AnyFs::Ext(src_ext) => walk_ext_for_plan(src_dev, src_ext, 2, &mut plan)?,
        AnyFs::Fat32(src_fat) => walk_fat_for_plan(src_dev, src_fat, "/", &mut plan)?,
    }
    Ok(plan)
}

fn walk_ext_for_plan(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::ext::Ext,
    src_ino: u32,
    plan: &mut fstool::fs::ext::BuildPlan,
) -> fstool::Result<()> {
    let entries = src.list_inode(src_dev, src_ino)?;
    for e in entries {
        if e.name == "." || e.name == ".." || (src_ino == 2 && e.name == "lost+found") {
            continue;
        }
        let inode = src.read_inode(src_dev, e.inode)?;
        let mode_type = inode.mode & fstool::fs::ext::constants::S_IFMT;
        match mode_type {
            t if t == fstool::fs::ext::constants::S_IFREG => plan.add_file(inode.size as u64),
            t if t == fstool::fs::ext::constants::S_IFDIR => {
                plan.add_dir();
                walk_ext_for_plan(src_dev, src, e.inode, plan)?;
            }
            t if t == fstool::fs::ext::constants::S_IFLNK => plan.add_symlink(inode.size as usize),
            t if t
                == fstool::fs::ext::constants::S_IFCHR
                    | fstool::fs::ext::constants::S_IFBLK
                    | fstool::fs::ext::constants::S_IFIFO
                    | fstool::fs::ext::constants::S_IFSOCK =>
            {
                plan.add_device()
            }
            _ => {}
        }
    }
    Ok(())
}

fn walk_fat_for_plan(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::fat::Fat32,
    src_path: &str,
    plan: &mut fstool::fs::ext::BuildPlan,
) -> fstool::Result<()> {
    use fstool::fs::EntryKind;
    let entries = src.list_path(src_dev, src_path)?;
    for e in entries {
        let child = join_fs_path(src_path, &e.name);
        match e.kind {
            EntryKind::Dir => {
                plan.add_dir();
                walk_fat_for_plan(src_dev, src, &child, plan)?;
            }
            EntryKind::Regular => {
                let (entry, _) = src.resolve_entry(src_dev, &child)?;
                plan.add_file(entry.file_size as u64);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Sum the size of every regular file in the source filesystem — used
/// by FAT32 shrink sizing.
fn sum_source_file_bytes(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src_fs: &fstool::inspect::AnyFs,
) -> fstool::Result<u64> {
    use fstool::inspect::AnyFs;
    match src_fs {
        AnyFs::Ext(src_ext) => sum_ext_file_bytes(src_dev, src_ext, 2),
        AnyFs::Fat32(src_fat) => sum_fat_file_bytes(src_dev, src_fat, "/"),
    }
}

fn sum_ext_file_bytes(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::ext::Ext,
    src_ino: u32,
) -> fstool::Result<u64> {
    let mut total = 0u64;
    for e in src.list_inode(src_dev, src_ino)? {
        if e.name == "." || e.name == ".." || (src_ino == 2 && e.name == "lost+found") {
            continue;
        }
        let inode = src.read_inode(src_dev, e.inode)?;
        let mode_type = inode.mode & fstool::fs::ext::constants::S_IFMT;
        if mode_type == fstool::fs::ext::constants::S_IFREG {
            total += inode.size as u64;
        } else if mode_type == fstool::fs::ext::constants::S_IFDIR {
            total += sum_ext_file_bytes(src_dev, src, e.inode)?;
        }
    }
    Ok(total)
}

fn sum_fat_file_bytes(
    src_dev: &mut dyn fstool::block::BlockDevice,
    src: &fstool::fs::fat::Fat32,
    src_path: &str,
) -> fstool::Result<u64> {
    use fstool::fs::EntryKind;
    let mut total = 0u64;
    for e in src.list_path(src_dev, src_path)? {
        let child = join_fs_path(src_path, &e.name);
        match e.kind {
            EntryKind::Regular => {
                let (entry, _) = src.resolve_entry(src_dev, &child)?;
                total += entry.file_size as u64;
            }
            EntryKind::Dir => total += sum_fat_file_bytes(src_dev, src, &child)?,
            _ => {}
        }
    }
    Ok(total)
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

fn fat_build(
    src_dir: &std::path::Path,
    output: &std::path::Path,
    size: Option<&str>,
    label: &str,
    volume_id: u32,
    force: bool,
    cluster_size: &str,
) -> fstool::Result<()> {
    use fstool::block::file::is_block_device;
    use fstool::fs::fat::Fat32;

    let qcow2_cluster_size = parse_cluster_size(cluster_size)?;
    let is_device = is_block_device(output);
    require_force_for_device(output, is_device, force)?;

    let bytes = if is_device {
        // Capacity comes from the device; --size is ignored.
        let dev = FileBackend::open(output)?;
        dev.total_size()
    } else {
        let s = size.ok_or_else(|| {
            fstool::Error::InvalidArgument(
                "fat-build: --size is required when OUTPUT is a regular file".into(),
            )
        })?;
        fstool::spec::parse_size(s)?
    };
    let total_sectors: u32 = (bytes / 512).try_into().map_err(|_| {
        fstool::Error::InvalidArgument(
            "fat-build: device size doesn't fit in a u32 sector count".into(),
        )
    })?;
    let label_bytes = fat32_label_bytes(label);
    let mut dev = fstool::block::create_image(
        output,
        bytes,
        &fstool::block::CreateOpts {
            cluster_size: qcow2_cluster_size,
        },
    )?;
    Fat32::build_from_host_dir(dev.as_mut(), total_sectors, src_dir, volume_id, label_bytes)?;
    dev.sync()?;
    eprintln!(
        "wrote {} ({} bytes, fat32{}, label {:?})",
        output.display(),
        bytes,
        if is_device { ", block device" } else { "" },
        label
    );
    Ok(())
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

fn ext_build(
    src_dir: &std::path::Path,
    output: &std::path::Path,
    kind: FsKind,
    block_size: u32,
    sparse: bool,
    force: bool,
    cluster_size: &str,
) -> fstool::Result<()> {
    use fstool::block::file::is_block_device;
    use fstool::fs::ext::BuildPlan;

    let qcow2_cluster_size = parse_cluster_size(cluster_size)?;
    let is_device = is_block_device(output);
    require_force_for_device(output, is_device, force)?;

    let mut plan = BuildPlan::new(block_size, kind);
    plan.scan_host_path(src_dir)?;
    let mut opts = plan.to_format_opts();
    opts.sparse = sparse;
    let plan_size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = fstool::block::create_image(
        output,
        plan_size,
        &fstool::block::CreateOpts {
            cluster_size: qcow2_cluster_size,
        },
    )?;

    // On a block device the actual capacity is fixed by the hardware —
    // expand the FS to fill it instead of leaving most of the device
    // unused. blocks_count must stay a multiple of 8 for the block bitmap
    // to be byte-aligned, and inode density gets scaled to mke2fs's
    // 1-inode-per-16-KiB convention. (qcow2 always reports its virtual
    // size as plan_size, so this branch only triggers for block devices.)
    let actual_size = dev.total_size();
    if actual_size > plan_size {
        let max_blocks_u64 = actual_size / opts.block_size as u64;
        let max_blocks = u32::try_from(max_blocks_u64).unwrap_or(u32::MAX);
        opts.blocks_count = (max_blocks / 8) * 8;
        let by_density = (opts.blocks_count as u64 * opts.block_size as u64 / 16_384) as u32;
        opts.inodes_count = opts.inodes_count.max(by_density);
    }
    let final_size = opts.blocks_count as u64 * opts.block_size as u64;

    let mut ext = Ext::format_with(dev.as_mut(), &opts)?;
    ext.populate_from_host_dir(dev.as_mut(), 2, src_dir)?;
    ext.flush(dev.as_mut())?;
    dev.sync()?;
    eprintln!(
        "wrote {} ({} bytes, {:?}{}{}, {} inodes, {} blocks)",
        output.display(),
        final_size,
        kind,
        if sparse { ", sparse" } else { "" },
        if is_device { ", block device" } else { "" },
        opts.inodes_count,
        opts.blocks_count
    );
    Ok(())
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
    let target = fstool::inspect::Target::parse(image);
    fstool::inspect::with_target_device(&target, |dev| {
        let fs = fstool::inspect::AnyFs::open(dev)?;
        let entries = fs.list(dev, path)?;
        let mut out = std::io::stdout().lock();
        for e in &entries {
            let _ = writeln!(out, "{}\t{:?}\t{}", e.inode, e.kind, e.name);
        }
        Ok(())
    })
}

fn cat(image: &str, path: &str) -> fstool::Result<()> {
    let target = fstool::inspect::Target::parse(image);
    fstool::inspect::with_target_device(&target, |dev| {
        let fs = fstool::inspect::AnyFs::open(dev)?;
        let mut out = std::io::stdout().lock();
        fs.copy_file_to(dev, path, &mut out)?;
        Ok(())
    })
}

fn info(image: &str) -> fstool::Result<()> {
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
        let fs = fstool::inspect::AnyFs::open(disk.as_mut())?;
        print_fs_info(disk.as_mut(), &fs);
        return Ok(());
    }
    fstool::inspect::with_target_device(&target, |dev| {
        let fs = fstool::inspect::AnyFs::open(dev)?;
        print_fs_info(dev, &fs);
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

fn print_fs_info(dev: &mut dyn fstool::block::BlockDevice, fs: &fstool::inspect::AnyFs) {
    println!("fs kind:           {}", fs.kind_string());
    match fs {
        fstool::inspect::AnyFs::Ext(ext) => print_ext_info(ext),
        fstool::inspect::AnyFs::Fat32(fat) => print_fat_info(fat),
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
