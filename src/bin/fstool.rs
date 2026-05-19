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
        /// Output image file. Will be created (truncating any existing file).
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
    },

    /// Build a bare FAT32 image from a host directory. FAT32 has no
    /// streaming auto-size (it needs ≥ 65525 clusters → ~33 MiB minimum),
    /// so `--size` is required.
    FatBuild {
        /// Source directory on the host filesystem.
        #[arg(value_name = "SRC_DIR")]
        src_dir: PathBuf,
        /// Output image file. Will be created (truncating any existing file).
        #[arg(short = 'o', long = "output", value_name = "IMAGE")]
        output: PathBuf,
        /// Image size, e.g. "64MiB" or "1GiB".
        #[arg(long, value_name = "SIZE")]
        size: String,
        /// Volume label (up to 11 ASCII bytes, upper-cased).
        #[arg(long, default_value = "NO NAME")]
        label: String,
        /// Volume ID / serial number. Default 0 for reproducible output.
        #[arg(long, default_value_t = 0)]
        volume_id: u32,
    },

    /// List a directory inside an image. One entry per line:
    /// `<inode>\t<kind>\t<name>`.
    Ls {
        /// Path to the image file.
        #[arg(value_name = "IMAGE")]
        image: PathBuf,
        /// Path inside the image to list. Defaults to `/`.
        #[arg(value_name = "PATH", default_value = "/")]
        path: String,
    },

    /// Print the contents of a regular file from inside an image to stdout.
    Cat {
        /// Path to the image file.
        #[arg(value_name = "IMAGE")]
        image: PathBuf,
        /// Path inside the image to read.
        #[arg(value_name = "PATH")]
        path: String,
    },

    /// One-screen summary of an existing image: detected FS kind, block
    /// size, total blocks, used/free counts, and a top-level listing.
    Info {
        /// Path to the image file.
        #[arg(value_name = "IMAGE")]
        image: PathBuf,
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
        /// Path to the image file (modified in place).
        #[arg(value_name = "IMAGE")]
        image: PathBuf,
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
        /// Path to the image file (modified in place).
        #[arg(value_name = "IMAGE")]
        image: PathBuf,
        /// Absolute path inside the image to remove.
        #[arg(value_name = "FS_PATH")]
        fs_path: String,
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
        } => ext_build(&src_dir, &output, kind.into(), block_size, sparse),
        Command::FatBuild {
            src_dir,
            output,
            size,
            label,
            volume_id,
        } => fat_build(&src_dir, &output, &size, &label, volume_id),
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
    }
}

fn rm(image: &std::path::Path, fs_path: &str) -> fstool::Result<()> {
    let mut dev = FileBackend::open(image)?;
    let mut ext = Ext::open(&mut dev)?;
    ext.remove_path(&mut dev, fs_path)?;
    ext.flush(&mut dev)?;
    dev.sync()?;
    eprintln!("removed {fs_path}");
    Ok(())
}

fn add(image: &std::path::Path, host_src: &std::path::Path, fs_dest: &str) -> fstool::Result<()> {
    use fstool::fs::{FileMeta, FileSource, Filesystem};
    use std::os::unix::fs::PermissionsExt;

    let meta = std::fs::symlink_metadata(host_src)?;
    let mut dev = FileBackend::open(image)?;
    let mut ext = Ext::open(&mut dev)?;
    let dest = std::path::Path::new(fs_dest);

    if meta.is_dir() {
        let fmeta = FileMeta {
            mode: (meta.permissions().mode() & 0o7777) as u16,
            ..FileMeta::default()
        };
        ext.create_dir(&mut dev, dest, fmeta)?;
        // Resolve the new directory's inode and recurse the host tree into it.
        let dir_ino = ext.path_to_inode(&mut dev, fs_dest)?;
        ext.populate_from_host_dir(&mut dev, dir_ino, host_src)?;
    } else if meta.is_file() {
        let fmeta = FileMeta {
            mode: (meta.permissions().mode() & 0o7777) as u16,
            ..FileMeta::default()
        };
        ext.create_file(
            &mut dev,
            dest,
            FileSource::HostPath(host_src.to_path_buf()),
            fmeta,
        )?;
    } else {
        return Err(fstool::Error::InvalidArgument(format!(
            "add: {} is neither a regular file nor a directory",
            host_src.display()
        )));
    }
    ext.flush(&mut dev)?;
    dev.sync()?;
    eprintln!("added {} → {fs_dest}", host_src.display());
    Ok(())
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
    size: &str,
    label: &str,
    volume_id: u32,
) -> fstool::Result<()> {
    use fstool::fs::fat::Fat32;
    let bytes = fstool::spec::parse_size(size)?;
    let total_sectors: u32 = (bytes / 512).try_into().map_err(|_| {
        fstool::Error::InvalidArgument("fat-build: --size doesn't fit in a u32 sector count".into())
    })?;
    let label_bytes = fat32_label_bytes(label);
    let mut dev = FileBackend::create(output, bytes)?;
    Fat32::build_from_host_dir(&mut dev, total_sectors, src_dir, volume_id, label_bytes)?;
    dev.sync()?;
    eprintln!(
        "wrote {} ({} bytes, fat32, label {:?})",
        output.display(),
        bytes,
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
) -> fstool::Result<()> {
    use fstool::fs::ext::BuildPlan;
    let mut plan = BuildPlan::new(block_size, kind);
    plan.scan_host_path(src_dir)?;
    let mut opts = plan.to_format_opts();
    opts.sparse = sparse;
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(output, size)?;
    let mut ext = Ext::format_with(&mut dev, &opts)?;
    ext.populate_from_host_dir(&mut dev, 2, src_dir)?;
    ext.flush(&mut dev)?;
    dev.sync()?;
    eprintln!(
        "wrote {} ({} bytes, {:?}{}, {} inodes, {} blocks)",
        output.display(),
        size,
        kind,
        if sparse { ", sparse" } else { "" },
        opts.inodes_count,
        opts.blocks_count
    );
    Ok(())
}

fn ls(image: &std::path::Path, path: &str) -> fstool::Result<()> {
    let (mut dev, fs) = fstool::inspect::open_image_file(image)?;
    let entries = fs.list(&mut dev, path)?;
    let mut out = std::io::stdout().lock();
    for e in &entries {
        let _ = writeln!(out, "{}\t{:?}\t{}", e.inode, e.kind, e.name);
    }
    Ok(())
}

fn cat(image: &std::path::Path, path: &str) -> fstool::Result<()> {
    let (mut dev, fs) = fstool::inspect::open_image_file(image)?;
    let mut out = std::io::stdout().lock();
    fs.copy_file_to(&mut dev, path, &mut out)?;
    Ok(())
}

fn info(image: &std::path::Path) -> fstool::Result<()> {
    let (mut dev, fs) = fstool::inspect::open_image_file(image)?;
    println!("fs kind:           {}", fs.kind_string());
    match &fs {
        fstool::inspect::AnyFs::Ext(ext) => print_ext_info(ext),
        fstool::inspect::AnyFs::Fat32(fat) => print_fat_info(fat),
    }
    println!();
    println!("/ listing:");
    let entries = fs.list(&mut dev, "/")?;
    for e in &entries {
        println!("  {:>10}  {:?}  {}", e.inode, e.kind, e.name);
    }
    Ok(())
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
