//! Exercises the `fstool` binary end to end via its CLI.

use std::process::Command;

use tempfile::NamedTempFile;

/// Path to the freshly-built `fstool` binary (provided by Cargo for
/// integration tests).
const FSTOOL: &str = env!("CARGO_BIN_EXE_fstool");

fn which(tool: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {tool}"))
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// build (bare ext4 spec) → ls → cat → add → cat the added file.
#[test]
fn cli_build_ls_cat_add_roundtrip() {
    if !which("e2fsck") {
        eprintln!("skipping: e2fsck not installed");
        return;
    }

    // Source tree + a spare-capacity spec (extra inodes via a bigger tree
    // is awkward; instead we test `add` against the headroom a fresh image
    // happens to have).
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("one.txt"), b"first\n").unwrap();

    let spec = NamedTempFile::new().unwrap();
    std::fs::write(
        spec.path(),
        format!(
            "[filesystem]\ntype = \"ext4\"\nsource = \"{}\"\nblock_size = 1024\n",
            srcdir.path().display()
        ),
    )
    .unwrap();

    let img = NamedTempFile::new().unwrap();

    // build
    let out = Command::new(FSTOOL)
        .arg("build")
        .arg(spec.path())
        .arg("-o")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // ls /
    let out = Command::new(FSTOOL)
        .args(["ls"])
        .arg(img.path())
        .arg("/")
        .output()
        .unwrap();
    assert!(out.status.success());
    let listing = String::from_utf8_lossy(&out.stdout);
    assert!(
        listing.contains("one.txt"),
        "ls missing one.txt:\n{listing}"
    );

    // cat /one.txt
    let out = Command::new(FSTOOL)
        .args(["cat"])
        .arg(img.path())
        .arg("/one.txt")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout, b"first\n");

    // add a host file
    let extra = NamedTempFile::new().unwrap();
    std::fs::write(extra.path(), b"added via cli\n").unwrap();
    let out = Command::new(FSTOOL)
        .arg("add")
        .arg(img.path())
        .arg(extra.path())
        .arg("/two.txt")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "add failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // e2fsck must still be clean after the modification.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck failed after add:\n{}",
        String::from_utf8_lossy(&fsck.stdout)
    );

    // cat the added file
    let out = Command::new(FSTOOL)
        .args(["cat"])
        .arg(img.path())
        .arg("/two.txt")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout, b"added via cli\n");
}

/// build → rm a file → rm an empty dir → e2fsck clean → non-empty dir
/// rejected.
#[test]
fn cli_rm_file_and_empty_dir() {
    if !which("e2fsck") {
        eprintln!("skipping: e2fsck not installed");
        return;
    }

    // Source tree: a file, an empty dir, and a non-empty dir.
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("doomed.txt"), b"bye\n").unwrap();
    std::fs::create_dir(srcdir.path().join("emptydir")).unwrap();
    std::fs::create_dir(srcdir.path().join("fulldir")).unwrap();
    std::fs::write(srcdir.path().join("fulldir/keep"), b"k\n").unwrap();

    let img = NamedTempFile::new().unwrap();
    let out = Command::new(FSTOOL)
        .args(["ext-build", "--kind", "ext4"])
        .arg(srcdir.path())
        .arg("-o")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(out.status.success());

    // rm a regular file.
    let out = Command::new(FSTOOL)
        .arg("rm")
        .arg(img.path())
        .arg("/doomed.txt")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "rm file failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // rm an empty directory.
    let out = Command::new(FSTOOL)
        .arg("rm")
        .arg(img.path())
        .arg("/emptydir")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "rm empty dir failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // rm a non-empty directory must fail.
    let out = Command::new(FSTOOL)
        .arg("rm")
        .arg(img.path())
        .arg("/fulldir")
        .output()
        .unwrap();
    assert!(!out.status.success(), "rm non-empty dir should have failed");

    // e2fsck clean after the removals.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck failed after rm:\n{}",
        String::from_utf8_lossy(&fsck.stdout)
    );

    // The removed entries are gone; the kept ones remain.
    let out = Command::new(FSTOOL)
        .args(["ls"])
        .arg(img.path())
        .arg("/")
        .output()
        .unwrap();
    let listing = String::from_utf8_lossy(&out.stdout);
    assert!(!listing.contains("doomed.txt"), "doomed.txt still present");
    assert!(!listing.contains("emptydir"), "emptydir still present");
    assert!(listing.contains("fulldir"), "fulldir wrongly removed");
}

/// `fstool info` reports the expected filesystem summary.
#[test]
fn cli_info_reports_ext4() {
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("x"), b"y\n").unwrap();
    let img = NamedTempFile::new().unwrap();

    let out = Command::new(FSTOOL)
        .args(["ext-build", "--kind", "ext4"])
        .arg(srcdir.path())
        .arg("-o")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(out.status.success());

    let out = Command::new(FSTOOL)
        .arg("info")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(out.status.success());
    let info = String::from_utf8_lossy(&out.stdout);
    assert!(info.contains("ext4"), "info missing kind:\n{info}");
    assert!(
        info.contains("block size"),
        "info missing block size:\n{info}"
    );
}

/// `fstool fat-build` → `ls` → `cat` → `info` on a FAT32 image. Exercises
/// the unified inspection dispatch (the CLI doesn't know it's FAT32).
#[test]
fn cli_fat32_build_ls_cat_info_roundtrip() {
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("short.txt"), b"short body\n").unwrap();
    std::fs::create_dir(srcdir.path().join("nest")).unwrap();
    std::fs::write(
        srcdir.path().join("nest/A Long Name.md"),
        b"long-name body\n",
    )
    .unwrap();

    let img = NamedTempFile::new().unwrap();
    let out = Command::new(FSTOOL)
        .args(["fat-build", "--size", "64MiB", "--label", "CLIFAT"])
        .arg(srcdir.path())
        .arg("-o")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "fat-build failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // info names the FS as fat32.
    let out = Command::new(FSTOOL)
        .arg("info")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(out.status.success());
    let info = String::from_utf8_lossy(&out.stdout);
    assert!(info.contains("fat32"), "info missing fat32:\n{info}");
    assert!(info.contains("CLIFAT"), "info missing label:\n{info}");

    // ls /
    let out = Command::new(FSTOOL)
        .args(["ls"])
        .arg(img.path())
        .arg("/")
        .output()
        .unwrap();
    assert!(out.status.success());
    let listing = String::from_utf8_lossy(&out.stdout);
    assert!(listing.contains("short.txt"));
    assert!(listing.contains("nest"));

    // ls a subdirectory.
    let out = Command::new(FSTOOL)
        .args(["ls"])
        .arg(img.path())
        .arg("/nest")
        .output()
        .unwrap();
    assert!(out.status.success());
    let nest = String::from_utf8_lossy(&out.stdout);
    assert!(
        nest.contains("A Long Name.md"),
        "long-name entry missing from /nest:\n{nest}"
    );

    // cat the deep long-named file.
    let out = Command::new(FSTOOL)
        .args(["cat"])
        .arg(img.path())
        .arg("/nest/A Long Name.md")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cat failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"long-name body\n");
}
