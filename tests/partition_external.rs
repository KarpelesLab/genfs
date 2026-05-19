//! Cross-check our MBR/GPT writers against system tools (`sgdisk`, `fdisk`).
//!
//! Each test silently skips if the corresponding tool is missing from PATH —
//! that way `cargo test` still passes on minimal CI images while opportunistically
//! validating against the real tools when they're available.

use std::path::Path;
use std::process::Command;

use fstool::block::{BlockDevice, FileBackend};
use fstool::part::{Gpt, Mbr, Partition, PartitionKind, PartitionTable};
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

fn write_gpt_image(path: &Path, size: u64) {
    let mut dev = FileBackend::create(path, size).unwrap();
    let parts = vec![
        Partition {
            start_lba: 2048,
            size_lba: 2048,
            kind: PartitionKind::EfiSystem,
            name: Some("EFI System".into()),
            ..Partition::new(0, 0, PartitionKind::EfiSystem)
        },
        Partition {
            start_lba: 4096,
            size_lba: size / 512 - 4096 - 34,
            kind: PartitionKind::LinuxFilesystem,
            name: Some("root".into()),
            ..Partition::new(0, 0, PartitionKind::LinuxFilesystem)
        },
    ];
    let gpt = Gpt::build(parts).unwrap();
    gpt.write(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);
}

fn write_mbr_image(path: &Path, size: u64) {
    let mut dev = FileBackend::create(path, size).unwrap();
    let parts = vec![
        Partition {
            start_lba: 2048,
            size_lba: 20480,
            kind: PartitionKind::LinuxFilesystem,
            bootable: true,
            ..Partition::new(0, 0, PartitionKind::LinuxFilesystem)
        },
        Partition {
            start_lba: 22528,
            size_lba: 4096,
            kind: PartitionKind::LinuxSwap,
            ..Partition::new(0, 0, PartitionKind::LinuxSwap)
        },
    ];
    let mbr = Mbr::new(parts).unwrap();
    mbr.write(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);
}

#[test]
fn gpt_validates_with_sgdisk() {
    let Some(_) = which("sgdisk") else {
        eprintln!("skipping: sgdisk not installed");
        return;
    };

    let tmp = NamedTempFile::new().unwrap();
    write_gpt_image(tmp.path(), 64 * 1024 * 1024);

    // -p prints the partition table; non-zero exit means the GPT is broken.
    let out = Command::new("sgdisk")
        .arg("-p")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "sgdisk -p failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // Should list two partitions named EFI System and root.
    assert!(
        stdout.contains("EFI System"),
        "missing EFI System in sgdisk output:\n{stdout}"
    );
    assert!(
        stdout.contains("root"),
        "missing root in sgdisk output:\n{stdout}"
    );

    // -v verifies all CRCs and structures.
    let out = Command::new("sgdisk")
        .arg("-v")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "sgdisk -v failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // sgdisk -v on a clean GPT prints "No problems found."
    assert!(
        stdout.contains("No problems found") || stdout.contains("no problems found"),
        "sgdisk -v didn't report a clean image:\n{stdout}\n{stderr}"
    );
}

#[test]
fn mbr_validates_with_fdisk() {
    // macOS ships its own `fdisk(8)` with completely different
    // command-line syntax (and no `-l` flag) — the test only makes sense
    // against util-linux fdisk.
    if !cfg!(target_os = "linux") {
        eprintln!("skipping: util-linux fdisk only exists on Linux");
        return;
    }
    let Some(_) = which("fdisk") else {
        eprintln!("skipping: fdisk not installed");
        return;
    };

    let tmp = NamedTempFile::new().unwrap();
    write_mbr_image(tmp.path(), 16 * 1024 * 1024);

    let out = Command::new("fdisk")
        .arg("-l")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "fdisk -l failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // Should report a DOS disklabel with our two partitions.
    assert!(
        stdout.contains("Disklabel type: dos") || stdout.contains("type: dos"),
        "fdisk did not detect a DOS partition table:\n{stdout}"
    );
    // Linux type ("83") and Linux swap ("82") should both appear.
    let lower = stdout.to_lowercase();
    assert!(
        lower.contains("linux"),
        "missing 'Linux' partition type:\n{stdout}"
    );
    assert!(
        lower.contains("swap") || lower.contains(" 82 "),
        "missing swap partition:\n{stdout}"
    );
}
