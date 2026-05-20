//! End-to-end validation of the TOML spec `build` path.

use std::process::Command;

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

#[test]
fn build_bare_ext4_from_spec() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };

    // Source tree.
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(srcdir.path().join("etc")).unwrap();
    std::fs::write(srcdir.path().join("readme.txt"), b"spec-built image\n").unwrap();
    std::fs::write(srcdir.path().join("etc/app.conf"), b"mode=on\n").unwrap();

    // Spec referencing that tree.
    let spec_text = format!(
        r#"
        [filesystem]
        type = "ext4"
        source = "{}"
        block_size = 1024
        rootdevs = "minimal"
        volume_label = "specimg"
        "#,
        srcdir.path().display()
    );
    let spec = fstool::spec::Spec::parse(&spec_text).unwrap();

    let out = NamedTempFile::new().unwrap();
    fstool::spec::build(&spec, out.path()).unwrap();

    // e2fsck clean.
    let res = Command::new("e2fsck")
        .arg("-fn")
        .arg(out.path())
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "e2fsck failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&res.stdout),
        String::from_utf8_lossy(&res.stderr),
    );

    // debugfs: source tree present, /dev populated, file body intact.
    let listing = Command::new("debugfs")
        .arg("-R")
        .arg("ls /")
        .arg(out.path())
        .output()
        .unwrap();
    let root = String::from_utf8_lossy(&listing.stdout);
    for e in ["readme.txt", "etc", "dev"] {
        assert!(root.contains(e), "missing /{e}:\n{root}");
    }

    let devs = Command::new("debugfs")
        .arg("-R")
        .arg("ls /dev")
        .arg(out.path())
        .output()
        .unwrap();
    let dev = String::from_utf8_lossy(&devs.stdout);
    for n in ["console", "null", "zero", "urandom"] {
        assert!(dev.contains(n), "missing /dev/{n}:\n{dev}");
    }

    let body = Command::new("debugfs")
        .arg("-R")
        .arg("cat /etc/app.conf")
        .arg(out.path())
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&body.stdout).contains("mode=on"),
        "/etc/app.conf body wrong"
    );
}

/// Build a partitioned GPT disk image with an ESP + an ext4 root, then
/// validate the GPT with sgdisk and the root filesystem with e2fsck.
#[test]
fn build_partitioned_gpt_disk_from_spec() {
    let Some(_) = which("sgdisk") else {
        eprintln!("skipping: sgdisk not installed");
        return;
    };
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };

    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("hello"), b"in a partition\n").unwrap();

    let spec_text = format!(
        r#"
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
        source = "{}"
        block_size = 1024
        "#,
        srcdir.path().display()
    );
    let spec = fstool::spec::Spec::parse(&spec_text).unwrap();
    let out = NamedTempFile::new().unwrap();
    fstool::spec::build(&spec, out.path()).unwrap();

    // GPT must validate.
    let v = Command::new("sgdisk")
        .arg("-v")
        .arg(out.path())
        .output()
        .unwrap();
    let vout = String::from_utf8_lossy(&v.stdout);
    assert!(v.status.success(), "sgdisk -v failed: {vout}");
    assert!(
        vout.contains("No problems found"),
        "sgdisk -v not clean:\n{vout}"
    );

    // sgdisk -p must list both partitions.
    let p = Command::new("sgdisk")
        .arg("-p")
        .arg(out.path())
        .output()
        .unwrap();
    let pout = String::from_utf8_lossy(&p.stdout);
    assert!(pout.contains("EFI"), "missing EFI partition:\n{pout}");
    assert!(pout.contains("root"), "missing root partition:\n{pout}");

    // Carve out the root partition (entry 2) and e2fsck it. Parse the
    // start/end sectors from sgdisk -p's last data line.
    let root_line = pout
        .lines()
        .find(|l| l.trim_start().starts_with("2 "))
        .expect("partition 2 line");
    let nums: Vec<u64> = root_line
        .split_whitespace()
        .skip(1)
        .take(2)
        .map(|s| s.parse().unwrap())
        .collect();
    let (start, end) = (nums[0], nums[1]);

    let part = NamedTempFile::new().unwrap();
    let dd = Command::new("dd")
        .arg(format!("if={}", out.path().display()))
        .arg(format!("of={}", part.path().display()))
        .arg("bs=512")
        .arg(format!("skip={start}"))
        .arg(format!("count={}", end - start + 1))
        .arg("status=none")
        .output()
        .unwrap();
    assert!(dd.status.success(), "dd failed");

    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(part.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck on root partition failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&fsck.stdout),
        String::from_utf8_lossy(&fsck.stderr),
    );
}

#[test]
fn build_bare_fat32_from_spec() {
    let Some(_) = which("fsck.vfat") else {
        eprintln!("skipping: fsck.vfat not installed");
        return;
    };
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("hello.txt"), b"from spec\n").unwrap();

    let spec_text = format!(
        r#"
        [filesystem]
        type = "fat32"
        source = "{}"
        size = "64MiB"
        volume_label = "specfat"
        volume_id = 0xDEADBEEF
        "#,
        srcdir.path().display()
    );
    let spec = fstool::spec::Spec::parse(&spec_text).unwrap();
    let out = NamedTempFile::new().unwrap();
    fstool::spec::build(&spec, out.path()).unwrap();

    let res = Command::new("fsck.vfat")
        .args(["-n", "-v"])
        .arg(out.path())
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "fsck.vfat failed:\n{}",
        String::from_utf8_lossy(&res.stdout)
    );
}

#[test]
fn fat32_bare_requires_explicit_size() {
    let spec = fstool::spec::Spec::parse(
        r#"
        [filesystem]
        type = "fat32"
        "#,
    )
    .unwrap();
    let out = NamedTempFile::new().unwrap();
    let err = fstool::spec::build(&spec, out.path()).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("FAT32 needs either `size` or `source`"),
        "unexpected error: {msg}"
    );
}

#[test]
fn build_empty_ext2_from_spec() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    // No `source` → empty filesystem.
    let spec = fstool::spec::Spec::parse(
        r#"
        [filesystem]
        type = "ext2"
        block_size = 1024
        "#,
    )
    .unwrap();
    let out = NamedTempFile::new().unwrap();
    fstool::spec::build(&spec, out.path()).unwrap();

    let res = Command::new("e2fsck")
        .arg("-fn")
        .arg(out.path())
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "e2fsck failed:\n{}",
        String::from_utf8_lossy(&res.stdout)
    );
}
