//! External validation: produce ext2 images and check them with `e2fsck`
//! (e2fsprogs) and `debugfs`. Each test skips silently when the required
//! tool isn't on PATH so `cargo test` still passes on minimal CI images.

use std::io::Write;
use std::path::Path;
use std::process::Command;

use genfs::block::FileBackend;
use genfs::fs::ext::{Ext, FormatOpts};
use genfs::fs::{DeviceKind, FileMeta, FileSource};
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
    use genfs::block::BlockDevice;
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

    use genfs::block::BlockDevice;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    let mut ext = Ext::format_with(&mut dev, &opts).unwrap();

    // /hello.txt
    let mut src_file = NamedTempFile::new().unwrap();
    src_file.as_file_mut().write_all(b"hello, genfs\n").unwrap();
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
        body.contains("hello, genfs"),
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
