//! External validation: produce FAT32 images and check them with
//! `fsck.vfat` (dosfstools) and `mdir` / `mtype` (mtools). Each test skips
//! silently when the required tool isn't on PATH.

use std::path::Path;
use std::process::Command;

use fstool::block::FileBackend;
use fstool::fs::fat::{Fat32, FatFormatOpts};
use tempfile::{NamedTempFile, TempDir};

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

fn format_empty(path: &Path, mib: u32) {
    let total_sectors = mib * 1024 * 1024 / 512;
    let bytes = total_sectors as u64 * 512;
    let mut dev = FileBackend::create(path, bytes).expect("create image");
    let opts = FatFormatOpts {
        total_sectors,
        volume_id: 0xCAFE_F00D,
        volume_label: *b"FSTOOL     ",
    };
    Fat32::format(&mut dev, &opts).expect("format fat32");
    use fstool::block::BlockDevice;
    dev.sync().expect("sync");
}

#[test]
fn empty_fat32_passes_fsck_vfat() {
    let Some(_) = which("fsck.vfat") else {
        eprintln!("skipping: fsck.vfat not installed");
        return;
    };
    let tmp = NamedTempFile::new().unwrap();
    format_empty(tmp.path(), 64);

    let out = Command::new("fsck.vfat")
        .args(["-n", "-v"])
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "fsck.vfat failed (exit {:?}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code()
    );
}

#[test]
fn build_from_host_dir_passes_fsck_vfat() {
    let Some(_) = which("fsck.vfat") else {
        eprintln!("skipping: fsck.vfat not installed");
        return;
    };
    let src = TempDir::new().unwrap();
    std::fs::write(src.path().join("hello.txt"), b"hello, fat32\n").unwrap();
    std::fs::create_dir(src.path().join("docs")).unwrap();
    std::fs::write(
        src.path().join("docs").join("README.md"),
        b"# Long Name File\n",
    )
    .unwrap();

    let tmp = NamedTempFile::new().unwrap();
    let total_sectors = 64 * 1024 * 1024 / 512;
    {
        use fstool::block::BlockDevice;
        let mut dev = FileBackend::create(tmp.path(), total_sectors as u64 * 512).unwrap();
        Fat32::build_from_host_dir(
            &mut dev,
            total_sectors,
            src.path(),
            0xCAFE_F00D,
            *b"FSTOOL     ",
        )
        .expect("build fat32");
        dev.sync().unwrap();
    }

    let out = Command::new("fsck.vfat")
        .args(["-n", "-v"])
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "fsck.vfat failed (exit {:?}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code()
    );
}

#[test]
fn host_dir_contents_visible_via_mtools() {
    let Some(_) = which("mdir") else {
        eprintln!("skipping: mtools not installed");
        return;
    };
    let Some(_) = which("mtype") else {
        eprintln!("skipping: mtools (mtype) not installed");
        return;
    };

    let src = TempDir::new().unwrap();
    std::fs::write(src.path().join("hello.txt"), b"hello, fat32\n").unwrap();
    std::fs::create_dir(src.path().join("docs")).unwrap();
    std::fs::write(src.path().join("docs").join("README.md"), b"long-name\n").unwrap();

    let tmp = NamedTempFile::new().unwrap();
    let total_sectors = 64 * 1024 * 1024 / 512;
    {
        use fstool::block::BlockDevice;
        let mut dev = FileBackend::create(tmp.path(), total_sectors as u64 * 512).unwrap();
        Fat32::build_from_host_dir(
            &mut dev,
            total_sectors,
            src.path(),
            0xCAFE_F00D,
            *b"FSTOOL     ",
        )
        .unwrap();
        dev.sync().unwrap();
    }

    // mtools needs a drive letter -> file mapping; pass via MTOOLSRC env
    // pointing to a config file naming the image as drive ::.
    let cfg = src.path().join("mtoolsrc");
    std::fs::write(
        &cfg,
        format!("drive +: file=\"{}\"\n", tmp.path().display()),
    )
    .unwrap();

    let out = Command::new("mdir")
        .env("MTOOLSRC", &cfg)
        .args(["-i", &tmp.path().display().to_string(), "::/"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "mdir failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("hello"),
        "mdir output missing hello.txt:\n{stdout}"
    );
    assert!(
        stdout.contains("docs"),
        "mdir output missing docs/:\n{stdout}"
    );

    // Verify a file's contents via mtype.
    let out = Command::new("mtype")
        .args(["-i", &tmp.path().display().to_string(), "::/hello.txt"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "mtype failed:\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"hello, fat32\n");
}
