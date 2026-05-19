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
        } => ext_build(&src_dir, &output, kind.into(), block_size, sparse, force),
        Command::FatBuild {
            src_dir,
            output,
            size,
            label,
            volume_id,
            force,
        } => fat_build(&src_dir, &output, size.as_deref(), &label, volume_id, force),
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
    }
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
) -> fstool::Result<()> {
    use fstool::block::file::is_block_device;
    use fstool::fs::fat::Fat32;

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
    let mut dev = FileBackend::create(output, bytes)?;
    Fat32::build_from_host_dir(&mut dev, total_sectors, src_dir, volume_id, label_bytes)?;
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
) -> fstool::Result<()> {
    use fstool::block::file::is_block_device;
    use fstool::fs::ext::BuildPlan;

    let is_device = is_block_device(output);
    require_force_for_device(output, is_device, force)?;

    let mut plan = BuildPlan::new(block_size, kind);
    plan.scan_host_path(src_dir)?;
    let mut opts = plan.to_format_opts();
    opts.sparse = sparse;
    let plan_size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(output, plan_size)?;

    // On a block device the actual capacity is fixed by the hardware —
    // expand the FS to fill it instead of leaving most of the device
    // unused. blocks_count must stay a multiple of 8 for the block bitmap
    // to be byte-aligned, and inode density gets scaled to mke2fs's
    // 1-inode-per-16-KiB convention.
    let actual_size = dev.total_size();
    if actual_size > plan_size {
        let max_blocks_u64 = actual_size / opts.block_size as u64;
        let max_blocks = u32::try_from(max_blocks_u64).unwrap_or(u32::MAX);
        opts.blocks_count = (max_blocks / 8) * 8;
        let by_density = (opts.blocks_count as u64 * opts.block_size as u64 / 16_384) as u32;
        opts.inodes_count = opts.inodes_count.max(by_density);
    }
    let final_size = opts.blocks_count as u64 * opts.block_size as u64;

    let mut ext = Ext::format_with(&mut dev, &opts)?;
    ext.populate_from_host_dir(&mut dev, 2, src_dir)?;
    ext.flush(&mut dev)?;
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
        let mut disk = FileBackend::open(&target.path)?;
        if let Some(table) = fstool::inspect::detect_partition_table(&mut disk)? {
            print_partition_table(&target.path, disk.total_size(), &table);
            return Ok(());
        }
        // No table — fall through to the bare-FS info below using the
        // already-opened disk.
        let fs = fstool::inspect::AnyFs::open(&mut disk)?;
        print_fs_info(&mut disk, &fs);
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
