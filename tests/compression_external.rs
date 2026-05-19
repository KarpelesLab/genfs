//! End-to-end test for the compression integration: build an ext2,
//! repack it to a `.tar.gz`, decompress that archive through fstool
//! again, and verify the round-trip works.
//!
//! Skips when external tools fstool depends on aren't available, but
//! the basic gzip path (pure-Rust flate2) always runs.

#![cfg(unix)]

use std::process::Command;

use tempfile::NamedTempFile;

const FSTOOL: &str = env!("CARGO_BIN_EXE_fstool");

fn which(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .ok()
        .is_some_and(|o| o.status.success())
}

#[test]
fn repack_into_tar_gz_then_inspect() {
    if !which("mke2fs") {
        eprintln!("skipping: mke2fs not installed");
        return;
    }

    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("hello.txt"), b"hello compressed\n").unwrap();
    std::fs::create_dir(srcdir.path().join("sub")).unwrap();
    std::fs::write(srcdir.path().join("sub/nested.txt"), b"nested body\n").unwrap();

    // Build a bare ext2 from the source tree.
    let src_img = NamedTempFile::new().unwrap();
    let out = Command::new(FSTOOL)
        .args([
            "ext-build",
            "--kind",
            "ext2",
            srcdir.path().to_str().unwrap(),
            "-o",
            src_img.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "ext-build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Repack into a gzip-compressed tar.
    let tarball = tempfile::Builder::new()
        .suffix(".tar.gz")
        .tempfile()
        .unwrap();
    let out = Command::new(FSTOOL)
        .args([
            "repack",
            src_img.path().to_str().unwrap(),
            tarball.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "repack ext2 → tar.gz failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // Confirm the file actually starts with the gzip magic.
    let magic = std::fs::read(tarball.path()).unwrap();
    assert!(
        magic.len() >= 2 && magic[0] == 0x1f && magic[1] == 0x8b,
        "expected gzip magic at start of {}",
        tarball.path().display()
    );

    // Now have fstool inspect the .tar.gz — it should auto-decompress and
    // walk the entries.
    let out = Command::new(FSTOOL)
        .args(["ls", tarball.path().to_str().unwrap(), "/"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "fstool ls on .tar.gz failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let listing = String::from_utf8_lossy(&out.stdout);
    assert!(
        listing.contains("hello.txt"),
        "expected hello.txt in listing:\n{listing}"
    );

    // And `cat` should stream the file contents back out.
    let out = Command::new(FSTOOL)
        .args(["cat", tarball.path().to_str().unwrap(), "/hello.txt"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout, b"hello compressed\n");
}
