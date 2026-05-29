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
            "create",
            "-t",
            "ext2",
            srcdir.path().to_str().unwrap(),
            "-o",
            src_img.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "create failed: {}",
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

/// Repack an ext2 image into a `.tar.lz4` and confirm the canonical LZ4
/// Frame (compcol) is both readable by the system `lz4` CLI and round-trips
/// back through fstool — exercising the `compcol::lz4::frame` path.
#[test]
fn repack_into_tar_lz4_interops_with_lz4_cli() {
    if !which("mke2fs") {
        eprintln!("skipping: mke2fs not installed");
        return;
    }

    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("hello.txt"), b"hello lz4 frame\n").unwrap();

    let src_img = NamedTempFile::new().unwrap();
    let out = Command::new(FSTOOL)
        .args([
            "create", "-t", "ext2",
            srcdir.path().to_str().unwrap(),
            "-o", src_img.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "create: {}", String::from_utf8_lossy(&out.stderr));

    let tarball = tempfile::Builder::new().suffix(".tar.lz4").tempfile().unwrap();
    let out = Command::new(FSTOOL)
        .args([
            "repack",
            src_img.path().to_str().unwrap(),
            tarball.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "repack →.tar.lz4: {}", String::from_utf8_lossy(&out.stderr));

    // Canonical LZ4 Frame magic 0x184D2204 (little-endian).
    let bytes = std::fs::read(tarball.path()).unwrap();
    assert_eq!(&bytes[0..4], &[0x04, 0x22, 0x4d, 0x18], "expected LZ4 frame magic");

    // The system `lz4` CLI must accept the frame (integrity check).
    if which("lz4") {
        let t = Command::new("lz4").arg("-t").arg(tarball.path()).output().unwrap();
        assert!(
            t.status.success(),
            "lz4 -t rejected fstool's frame: {}",
            String::from_utf8_lossy(&t.stderr)
        );
    }

    // And fstool reads its own .tar.lz4 back.
    let out = Command::new(FSTOOL)
        .args(["cat", tarball.path().to_str().unwrap(), "/hello.txt"])
        .output()
        .unwrap();
    assert!(out.status.success(), "cat .tar.lz4: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.stdout, b"hello lz4 frame\n");
}

/// Pipe `input` through `xz <args>` (stdin → stdout) and return stdout.
fn xz_pipe(args: &[&str], input: &[u8]) -> Vec<u8> {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new("xz")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn xz");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input)
        .expect("write to xz stdin");
    let out = child.wait_with_output().expect("wait xz");
    assert!(
        out.status.success(),
        "xz {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// Cross-tool interop: fstool's compcol-backed `.lzma` (alone) and `.xz`
/// codecs must interoperate with the system `xz` CLI in both directions.
/// This is the lzma/xz compatibility check called out during the compcol
/// migration — SquashFS's legacy LZMA compressor and `.tar.lzma` rely on
/// byte-compatible alone-format framing.
#[test]
fn lzma_and_xz_interoperate_with_xz_cli() {
    use fstool::compression::{Algo, compress, decompress};

    if !which("xz") {
        eprintln!("skipping: xz not installed");
        return;
    }

    // A payload with both repetition (so LZMA actually matches) and
    // variation (so it isn't a degenerate all-same block).
    let mut payload = Vec::new();
    for i in 0..4000u32 {
        payload.extend_from_slice(format!("line {i:05} the quick brown fox жжж\n").as_bytes());
    }

    for (algo, fmt) in [(Algo::Lzma, "lzma"), (Algo::Xz, "xz")] {
        // fstool encode → xz CLI decode.
        let enc = compress(algo, &payload).expect("fstool compress");
        let via_cli = xz_pipe(&["--format", fmt, "-dc"], &enc);
        assert_eq!(
            via_cli, payload,
            "{fmt}: system xz could not decode fstool output"
        );

        // xz CLI encode → fstool decode.
        let cli_enc = xz_pipe(&["--format", fmt, "-c"], &payload);
        let via_fstool = decompress(algo, &cli_enc, payload.len()).expect("fstool decompress");
        assert_eq!(
            via_fstool, payload,
            "{fmt}: fstool could not decode system xz output"
        );
    }
}
