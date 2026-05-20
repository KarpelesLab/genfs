//! External validation: produce ext2 images and check them with `e2fsck`
//! (e2fsprogs) and `debugfs`. Each test skips silently when the required
//! tool isn't on PATH so `cargo test` still passes on minimal CI images.
//!
//! Whole file is Unix-only: the source-tree fixtures use POSIX-style
//! symlinks + permission bits, and the cross-validation tools
//! (`e2fsck`, `debugfs`, `mke2fs`, `truncate`) only exist on
//! Unix-like systems.

#![cfg(unix)]

use std::io::Write;
use std::path::Path;
use std::process::Command;

use fstool::block::FileBackend;
use fstool::fs::ext::{Ext, FormatOpts};
use fstool::fs::rootdevs::RootDevs;
use fstool::fs::{DeviceKind, FileMeta, FileSource};
use tempfile::NamedTempFile;

fn which(tool: &str) -> Option<std::path::PathBuf> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {tool}"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let p = s.trim();
    if p.is_empty() { None } else { Some(p.into()) }
}

fn build_empty_ext2(path: &Path, opts: &FormatOpts) {
    let mut dev = FileBackend::create(path, opts.blocks_count as u64 * opts.block_size as u64)
        .expect("create image");
    Ext::format_with(&mut dev, opts).expect("format ext2");
    use fstool::block::BlockDevice;
    dev.sync().expect("sync");
    drop(dev);
}

#[test]
fn empty_ext2_passes_e2fsck() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts::default();
    build_empty_ext2(tmp.path(), &opts);

    let out = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "e2fsck failed (exit {:?}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code()
    );
}

#[test]
fn empty_ext2_lists_root_via_debugfs() {
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };
    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts::default();
    build_empty_ext2(tmp.path(), &opts);

    let out = Command::new("debugfs")
        .arg("-R")
        .arg("ls /")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "debugfs failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("lost+found"),
        "expected lost+found in root listing:\n{stdout}"
    );
}

/// Build an image with a small populated tree, then validate with e2fsck
/// and confirm the entries are visible via debugfs.
#[test]
fn populated_ext2_passes_e2fsck_and_debugfs() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };

    let tmp = NamedTempFile::new().unwrap();
    // We add 6 user entries + lost+found + reserved 1..10 + root = 18 inodes,
    // so default inodes_count=16 isn't enough.
    let opts = FormatOpts {
        inodes_count: 64,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;

    use fstool::block::BlockDevice;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    let mut ext = Ext::format_with(&mut dev, &opts).unwrap();

    // /hello.txt
    let mut src_file = NamedTempFile::new().unwrap();
    src_file
        .as_file_mut()
        .write_all(b"hello, fstool\n")
        .unwrap();
    let src_path = src_file.path().to_path_buf();
    ext.add_file_to(
        &mut dev,
        2, // root
        b"hello.txt",
        FileSource::HostPath(src_path),
        FileMeta {
            mode: 0o644,
            mtime: 0,
            ..Default::default()
        },
    )
    .unwrap();

    // /etc directory + /etc/conf file inside it
    let etc_ino = ext
        .add_dir_to(&mut dev, 2, b"etc", FileMeta::with_mode(0o755))
        .unwrap();
    let mut conf_src = NamedTempFile::new().unwrap();
    conf_src.as_file_mut().write_all(b"answer=42\n").unwrap();
    ext.add_file_to(
        &mut dev,
        etc_ino,
        b"conf",
        FileSource::HostPath(conf_src.path().to_path_buf()),
        FileMeta::with_mode(0o644),
    )
    .unwrap();

    // /bin -> /usr/bin (fast symlink, target < 60 bytes)
    ext.add_symlink_to(&mut dev, 2, b"bin", b"/usr/bin", FileMeta::with_mode(0o777))
        .unwrap();

    // /dev/null (char device 1,3) — needs /dev dir first
    let dev_ino = ext
        .add_dir_to(&mut dev, 2, b"dev", FileMeta::with_mode(0o755))
        .unwrap();
    ext.add_device_to(
        &mut dev,
        dev_ino,
        b"null",
        DeviceKind::Char,
        1,
        3,
        FileMeta::with_mode(0o666),
    )
    .unwrap();

    ext.flush(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);

    // e2fsck must be clean.
    let out = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "e2fsck failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // debugfs must list our entries.
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("ls /")
        .arg(tmp.path())
        .output()
        .unwrap();
    let listing = String::from_utf8_lossy(&out.stdout);
    for entry in ["hello.txt", "etc", "bin", "dev"] {
        assert!(listing.contains(entry), "missing /{entry} in:\n{listing}");
    }

    // /etc/conf
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("ls /etc")
        .arg(tmp.path())
        .output()
        .unwrap();
    let listing = String::from_utf8_lossy(&out.stdout);
    assert!(listing.contains("conf"), "missing conf:\n{listing}");

    // File content of /hello.txt via debugfs cat
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("cat /hello.txt")
        .arg(tmp.path())
        .output()
        .unwrap();
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(
        body.contains("hello, fstool"),
        "hello.txt content mismatch:\n{body}"
    );

    // Symlink target
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("stat /bin")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stat = String::from_utf8_lossy(&out.stdout);
    assert!(
        stat.contains("Fast symlink") || stat.contains("/usr/bin"),
        "symlink not recognised:\n{stat}"
    );
}

/// Build an image with the Standard root-dev set, verify e2fsck passes and
/// debugfs sees every expected node.
#[test]
fn ext2_with_standard_rootdevs_passes_e2fsck() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };

    use fstool::block::BlockDevice;
    let tmp = NamedTempFile::new().unwrap();
    // Standard set is 71 nodes + /dev dir + lost+found + root + 10 reserved
    // → ~84 inodes. Round up to 128 to leave headroom.
    let opts = FormatOpts {
        blocks_count: 4096,
        inodes_count: 128,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    let mut ext = Ext::format_with(&mut dev, &opts).unwrap();
    ext.populate_rootdevs(&mut dev, RootDevs::Standard, 0, 0, 0)
        .unwrap();
    ext.flush(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);

    // e2fsck must be clean.
    let out = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "e2fsck failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // Spot-check via debugfs ls /dev for both essentials and standard-only
    // nodes (a serial port, an IDE disk partition, a SCSI disk partition).
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("ls /dev")
        .arg(tmp.path())
        .output()
        .unwrap();
    let listing = String::from_utf8_lossy(&out.stdout);
    for entry in [
        "console", "null", "zero", "ptmx", "tty", "fuse", "random", "urandom", "tty0", "tty15",
        "ttyS0", "ttyS3", "kmsg", "mem", "port", "hda", "hda4", "hdd", "sda", "sda1", "sdd4",
    ] {
        assert!(
            listing.contains(entry),
            "missing /dev/{entry} in:\n{listing}"
        );
    }

    // Verify a specific device has the right major/minor (stat /dev/null).
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("stat /dev/null")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stat = String::from_utf8_lossy(&out.stdout);
    assert!(
        stat.contains("Device major/minor number: 01:03")
            || stat.contains("Major: 1") && stat.contains("Minor: 3"),
        "wrong device numbers for /dev/null:\n{stat}"
    );
}

/// Build an image, drop the Ext handle, reopen it, and verify the reader
/// API sees the same tree we wrote.
#[test]
fn ext2_open_lists_and_reads_what_was_written() {
    use fstool::block::BlockDevice;
    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts {
        inodes_count: 64,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    {
        let mut ext = Ext::format_with(&mut dev, &opts).unwrap();
        let mut src = NamedTempFile::new().unwrap();
        src.as_file_mut()
            .write_all(b"the quick brown fox\n")
            .unwrap();
        ext.add_file_to(
            &mut dev,
            2,
            b"fox.txt",
            FileSource::HostPath(src.path().to_path_buf()),
            FileMeta::with_mode(0o644),
        )
        .unwrap();
        ext.add_dir_to(&mut dev, 2, b"etc", FileMeta::with_mode(0o755))
            .unwrap();
        ext.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }

    // Reopen.
    let ext = Ext::open(&mut dev).unwrap();
    let entries = ext.list_inode(&mut dev, 2).unwrap();
    let names: std::collections::HashSet<_> = entries.iter().map(|e| e.name.clone()).collect();
    for n in ["lost+found", "fox.txt", "etc"] {
        assert!(names.contains(n), "missing {n}: {names:?}");
    }
    // Resolve a path.
    let fox = ext.path_to_inode(&mut dev, "/fox.txt").unwrap();
    let mut reader = ext.open_file_reader(&mut dev, fox).unwrap();
    let mut content = Vec::new();
    use std::io::Read as _;
    reader.read_to_end(&mut content).unwrap();
    assert_eq!(content, b"the quick brown fox\n");
}

/// Drive the Filesystem trait against Ext.
#[test]
fn ext2_via_filesystem_trait() {
    use fstool::block::BlockDevice;
    use fstool::fs::{Filesystem, FilesystemFactory};
    use std::io::Read;
    use std::path::Path;

    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts {
        inodes_count: 64,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    let mut ext = <Ext as FilesystemFactory>::format(&mut dev, &opts).unwrap();

    let mut src = NamedTempFile::new().unwrap();
    src.as_file_mut()
        .write_all(b"trait-impl content\n")
        .unwrap();

    ext.create_dir(&mut dev, Path::new("/etc"), FileMeta::with_mode(0o755))
        .unwrap();
    ext.create_file(
        &mut dev,
        Path::new("/etc/conf"),
        FileSource::HostPath(src.path().to_path_buf()),
        FileMeta::with_mode(0o644),
    )
    .unwrap();
    ext.create_symlink(
        &mut dev,
        Path::new("/conf"),
        Path::new("/etc/conf"),
        FileMeta::with_mode(0o777),
    )
    .unwrap();
    ext.flush(&mut dev).unwrap();
    dev.sync().unwrap();

    let entries = ext.list(&mut dev, Path::new("/etc")).unwrap();
    let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"conf"));

    let mut reader = ext.read_file(&mut dev, Path::new("/etc/conf")).unwrap();
    let mut body = Vec::new();
    reader.read_to_end(&mut body).unwrap();
    assert_eq!(body, b"trait-impl content\n");
}

/// ext2 sparse files: holes are represented as zero block pointers in the
/// direct + indirect block map. A mostly-zero file occupies few blocks and
/// the indirect block for an all-hole range isn't allocated at all.
#[test]
fn ext2_sparse_file_uses_holes() {
    use fstool::block::BlockDevice;
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };

    // 200 KiB mostly-zero file (well past the 12 KiB direct-block range at
    // 1 KiB blocks, so it exercises the single-indirect hole path).
    let mut body = vec![b'X'; 1024];
    body.extend(std::iter::repeat_n(0u8, 198 * 1024));
    body.extend(std::iter::repeat_n(b'Y', 1024));

    let srcfile = NamedTempFile::new().unwrap();
    std::fs::write(srcfile.path(), &body).unwrap();

    let opts = FormatOpts {
        blocks_count: 8192,
        inodes_count: 64,
        sparse: true,
        ..FormatOpts::default()
    };
    let tmp = NamedTempFile::new().unwrap();
    let mut dev = FileBackend::create(
        tmp.path(),
        opts.blocks_count as u64 * opts.block_size as u64,
    )
    .unwrap();
    let mut ext = Ext::format_with(&mut dev, &opts).unwrap();
    ext.add_file_to(
        &mut dev,
        2,
        b"holey",
        FileSource::HostPath(srcfile.path().to_path_buf()),
        FileMeta::with_mode(0o644),
    )
    .unwrap();
    ext.flush(&mut dev).unwrap();
    dev.sync().unwrap();

    // Content round-trips.
    let ino = ext.path_to_inode(&mut dev, "/holey").unwrap();
    let mut got = Vec::new();
    use std::io::Read;
    ext.open_file_reader(&mut dev, ino)
        .unwrap()
        .read_to_end(&mut got)
        .unwrap();
    assert_eq!(got, body);

    let inode = ext.read_inode(&mut dev, ino).unwrap();
    // Only the 2 KiB of real data should be allocated (+ at most one
    // indirect block) — nowhere near the dense 200 blocks.
    assert!(
        inode.blocks_512 < 32,
        "sparse ext2 file used {} sectors",
        inode.blocks_512
    );
    drop(dev);

    let out = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "e2fsck failed on sparse ext2:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
}

/// Open an ext2 image created by an external `mke2fs`, add a file via
/// fstool, flush, and verify the result is fsck-clean and debugfs can read
/// the file we wrote. This is the "modify someone else's image" path —
/// exercises Ext::open + add_file_to + flush against a third-party writer.
#[test]
fn fstool_can_modify_a_mke2fs_image() {
    let Some(_) = which("mke2fs") else {
        eprintln!("skipping: mke2fs not installed");
        return;
    };
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };

    let tmp = NamedTempFile::new().unwrap();
    // 8 MiB image. `-O none` disables every feature so the layout matches
    // our classic ext2 reader; once 64bit / metadata_csum / sparse_super /
    // flex_bg land we'll drop the -O none and exercise the full surface.
    let out = Command::new("mke2fs")
        .args([
            "-F",
            "-t",
            "ext2",
            "-b",
            "1024",
            "-L",
            "",
            "-U",
            "00000000-0000-0000-0000-000000000000",
            "-E",
            "nodiscard",
            "-O",
            "none",
            "-N",
            "64",
        ])
        .arg(tmp.path())
        .arg("8192")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "mke2fs failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Open with fstool, add a file under /, flush.
    use fstool::block::BlockDevice;
    let mut dev = FileBackend::open(tmp.path()).unwrap();
    let mut ext = Ext::open(&mut dev).unwrap();
    let mut src = NamedTempFile::new().unwrap();
    src.as_file_mut()
        .write_all(b"injected by fstool\n")
        .unwrap();
    ext.add_file_to(
        &mut dev,
        2,
        b"injected.txt",
        FileSource::HostPath(src.path().to_path_buf()),
        FileMeta::with_mode(0o644),
    )
    .unwrap();
    ext.flush(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);

    // e2fsck clean.
    let out = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "e2fsck failed after fstool modify:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // debugfs should see the new file with the right contents.
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("cat /injected.txt")
        .arg(tmp.path())
        .output()
        .unwrap();
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(
        body.contains("injected by fstool"),
        "injected file body wrong:\n{body}"
    );
}

/// Drive build_from_host_dir + BuildPlan auto-sizing end-to-end against
/// a small fixture tree on the host.
#[test]
fn ext2_build_from_host_dir_auto_size() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };

    use fstool::block::BlockDevice;
    use fstool::fs::ext::{Ext, FsKind};

    let tmpdir = tempfile::tempdir().unwrap();
    let src = tmpdir.path();
    std::fs::create_dir_all(src.join("etc")).unwrap();
    std::fs::create_dir_all(src.join("usr/bin")).unwrap();
    std::fs::write(src.join("hello.txt"), b"hello, world\n").unwrap();
    std::fs::write(src.join("etc/conf"), b"answer = 42\n").unwrap();
    std::os::unix::fs::symlink("/usr/bin", src.join("bin")).unwrap();
    std::os::unix::fs::symlink(
        "/very/long/path/that/exceeds/sixty/characters/for/sure/yes/indeed",
        src.join("slowlink"),
    )
    .unwrap();

    let tmp = NamedTempFile::new().unwrap();
    // Probe size needed by the plan.
    let mut plan = fstool::fs::ext::BuildPlan::new(1024, FsKind::Ext2);
    plan.scan_host_path(src).unwrap();
    let opts = plan.to_format_opts();

    // Pre-allocate the device file at the planned size.
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    Ext::build_from_host_dir(&mut dev, src, FsKind::Ext2, 1024).unwrap();
    dev.sync().unwrap();
    drop(dev);

    // Must be e2fsck-clean.
    let out = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "e2fsck failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    // Inspect via debugfs.
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("ls /")
        .arg(tmp.path())
        .output()
        .unwrap();
    let listing = String::from_utf8_lossy(&out.stdout);
    for entry in ["hello.txt", "etc", "usr", "bin", "slowlink"] {
        assert!(listing.contains(entry), "missing /{entry}: {listing}");
    }

    // /etc/conf body
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("cat /etc/conf")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("answer = 42"),
        "/etc/conf body wrong"
    );
}

#[test]
fn empty_ext2_dumpe2fs_clean() {
    let Some(_) = which("dumpe2fs") else {
        eprintln!("skipping: dumpe2fs not installed");
        return;
    };
    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts::default();
    build_empty_ext2(tmp.path(), &opts);

    let out = Command::new("dumpe2fs")
        .arg("-h")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "dumpe2fs failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // Sanity-check a few fields.
    assert!(
        stdout.contains("Filesystem magic number:  0xEF53"),
        "missing magic line:\n{stdout}"
    );
    assert!(stdout.contains("Block size:               1024"));
    assert!(stdout.contains("Inode count:              16"));
    assert!(stdout.contains("Filesystem state:         clean"));
}
