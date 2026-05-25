//! End-to-end validation of the archive backends (zip / cpio / ar)
//! through the `fstool` CLI, plus cross-checks against the system
//! `unzip` / `cpio` / `ar` / `zip` tools when they are installed.
//!
//! Tests that need an external tool print a `skipping: …` line and
//! return success when it is absent, so the suite stays green on
//! minimal runners. The CLI round-trips (create → ls/cat) need no
//! external tool — they exercise fstool's own reader against its own
//! writer, including the post-flush truncation to the true archive
//! length.

use std::path::Path;
use std::process::Command;

const FSTOOL: &str = env!("CARGO_BIN_EXE_fstool");

fn which(tool: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {tool}"))
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn run(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(FSTOOL)
        .args(args)
        .output()
        .expect("spawn fstool");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Stage a small tree: a top-level file, a binary blob, and a nested file.
fn stage_tree(root: &Path) {
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::write(root.join("greeting.txt"), b"hello world\n").unwrap();
    std::fs::write(root.join("blob.bin"), vec![0x42u8; 5000]).unwrap();
    std::fs::write(root.join("sub/deep.txt"), b"nested file contents\n").unwrap();
}

#[test]
fn zip_create_and_self_round_trip() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    stage_tree(&src);
    let out = work.path().join("out.zip");

    let (ok, _, err) = run(&[
        "create",
        "-t",
        "zip",
        src.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(ok, "create zip failed: {err}");

    // info reports the kind.
    let (ok, info, _) = run(&["info", out.to_str().unwrap()]);
    assert!(ok);
    assert!(info.contains("zip"), "info missing kind: {info}");

    // ls root + nested.
    let (ok, root, _) = run(&["ls", out.to_str().unwrap(), "/"]);
    assert!(ok);
    assert!(root.contains("greeting.txt") && root.contains("blob.bin") && root.contains("sub"));
    let (ok, sub, _) = run(&["ls", out.to_str().unwrap(), "/sub"]);
    assert!(ok);
    assert!(sub.contains("deep.txt"), "nested listing wrong: {sub}");

    // cat matches source bytes.
    let (ok, cat, _) = run(&["cat", out.to_str().unwrap(), "/sub/deep.txt"]);
    assert!(ok);
    assert_eq!(cat, "nested file contents\n");

    // The output must be truncated to the exact archive length (a zero
    // tail would break the EOCD-at-EOF invariant our reader relies on).
    let provisioned = std::fs::metadata(&out).unwrap().len();
    assert!(
        provisioned < 1_000_000,
        "zip not truncated: {provisioned} bytes"
    );
}

#[test]
fn zip_cross_check_with_unzip() {
    if !which("unzip") {
        eprintln!("skipping: unzip not installed");
        return;
    }
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    stage_tree(&src);
    let out = work.path().join("out.zip");
    let (ok, _, err) = run(&[
        "create",
        "-t",
        "zip",
        src.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(ok, "create zip failed: {err}");

    // `unzip -t` validates the archive structure end-to-end.
    let status = Command::new("unzip")
        .arg("-t")
        .arg(&out)
        .status()
        .expect("spawn unzip");
    assert!(status.success(), "unzip -t rejected fstool's zip");

    // Extract and diff against the source tree.
    let dest = work.path().join("unz");
    std::fs::create_dir_all(&dest).unwrap();
    let status = Command::new("unzip")
        .arg("-q")
        .arg(&out)
        .arg("-d")
        .arg(&dest)
        .status()
        .unwrap();
    assert!(status.success());
    assert_eq!(
        std::fs::read(dest.join("sub/deep.txt")).unwrap(),
        b"nested file contents\n"
    );
    assert_eq!(
        std::fs::read(dest.join("blob.bin")).unwrap(),
        vec![0x42u8; 5000]
    );
}

#[test]
fn read_zip_made_by_system_zip() {
    if !which("zip") {
        eprintln!("skipping: zip not installed");
        return;
    }
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("ascii.txt"), b"deflate me deflate me deflate me\n").unwrap();
    // A non-ASCII (UTF-8) name exercises the encoding path.
    std::fs::write(src.join("café.txt"), b"unicode name\n").unwrap();
    let out = work.path().join("sys.zip");
    // -r recurse, default DEFLATE.
    let status = Command::new("zip")
        .arg("-r")
        .arg("-q")
        .arg(&out)
        .arg(".")
        .current_dir(&src)
        .status()
        .expect("spawn zip");
    assert!(status.success(), "system zip failed");

    let (ok, root, err) = run(&["ls", out.to_str().unwrap(), "/"]);
    assert!(ok, "fstool ls on system zip failed: {err}");
    assert!(root.contains("ascii.txt"), "missing ascii entry: {root}");
    assert!(
        root.contains("café.txt"),
        "missing/garbled unicode entry: {root}"
    );

    let (ok, cat, _) = run(&["cat", out.to_str().unwrap(), "/ascii.txt"]);
    assert!(ok, "fstool cat (deflate) failed");
    assert_eq!(cat, "deflate me deflate me deflate me\n");
}

#[test]
fn cpio_create_and_round_trip() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    stage_tree(&src);
    let out = work.path().join("out.cpio");
    let (ok, _, err) = run(&[
        "create",
        "-t",
        "cpio",
        src.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(ok, "create cpio failed: {err}");

    let (ok, root, _) = run(&["ls", out.to_str().unwrap(), "/"]);
    assert!(ok && root.contains("greeting.txt") && root.contains("sub"));
    let (ok, cat, _) = run(&["cat", out.to_str().unwrap(), "/sub/deep.txt"]);
    assert!(ok);
    assert_eq!(cat, "nested file contents\n");

    if which("cpio") {
        let dest = work.path().join("cpx");
        std::fs::create_dir_all(&dest).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        let mut child = Command::new("cpio")
            .args(["-idm", "--quiet"])
            .current_dir(&dest)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .expect("spawn cpio");
        use std::io::Write;
        child.stdin.take().unwrap().write_all(&bytes).unwrap();
        assert!(child.wait().unwrap().success(), "system cpio -i failed");
        assert_eq!(
            std::fs::read(dest.join("sub/deep.txt")).unwrap(),
            b"nested file contents\n"
        );
    } else {
        eprintln!("skipping cpio cross-check: cpio not installed");
    }
}

#[test]
fn ar_create_and_round_trip() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("flat");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("short.o"), b"alpha\n").unwrap();
    // A name > 15 chars exercises the GNU long-name table.
    std::fs::write(src.join("a_long_member_name.txt"), b"beta contents\n").unwrap();
    let out = work.path().join("out.a");
    let (ok, _, err) = run(&[
        "create",
        "-t",
        "ar",
        src.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(ok, "create ar failed: {err}");

    let (ok, root, _) = run(&["ls", out.to_str().unwrap(), "/"]);
    assert!(ok, "ar ls failed");
    assert!(
        root.contains("short.o") && root.contains("a_long_member_name.txt"),
        "ar listing: {root}"
    );
    let (ok, cat, _) = run(&["cat", out.to_str().unwrap(), "/a_long_member_name.txt"]);
    assert!(ok);
    assert_eq!(cat, "beta contents\n");

    if which("ar") {
        let listing = Command::new("ar").arg("t").arg(&out).output().unwrap();
        assert!(listing.status.success());
        let names = String::from_utf8_lossy(&listing.stdout);
        // fstool emits GNU-format `ar` (long names indexed through the
        // `//` table). GNU `ar` on Linux decodes that and prints the
        // full long name; BSD `ar` on macOS doesn't follow the GNU
        // long-name lookup, so it prints `/0` (the raw table-offset
        // marker) in place of the long name. We assert what each
        // dialect actually emits: the short name on every flavour,
        // the decoded long name only where the system `ar` knows how
        // to decode it.
        assert!(
            names.contains("short.o"),
            "system ar t didn't list short.o: {names}"
        );
        if cfg!(target_os = "linux") {
            assert!(
                names.contains("a_long_member_name.txt"),
                "GNU ar t: {names}"
            );
        }
    } else {
        eprintln!("skipping ar cross-check: ar not installed");
    }
}

#[test]
fn ar_rejects_nested_paths() {
    // `ar` is flat; a source tree with a subdirectory must fail clearly
    // rather than produce a malformed archive.
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    stage_tree(&src); // has sub/deep.txt
    let out = work.path().join("bad.a");
    let (ok, _, err) = run(&[
        "create",
        "-t",
        "ar",
        src.to_str().unwrap(),
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(!ok, "ar create should reject a nested tree");
    assert!(
        err.contains("flat archive") || err.contains("subdirectory"),
        "unhelpful error: {err}"
    );
}

#[test]
fn cross_format_repack_zip_to_cpio() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    stage_tree(&src);
    let zip = work.path().join("a.zip");
    assert!(
        run(&[
            "create",
            "-t",
            "zip",
            src.to_str().unwrap(),
            "-o",
            zip.to_str().unwrap()
        ])
        .0
    );

    let cpio = work.path().join("b.cpio");
    let (ok, _, err) = run(&[
        "repack",
        zip.to_str().unwrap(),
        cpio.to_str().unwrap(),
        "--fs-type",
        "cpio",
    ]);
    assert!(ok, "repack zip->cpio failed: {err}");
    let (ok, cat, _) = run(&["cat", cpio.to_str().unwrap(), "/sub/deep.txt"]);
    assert!(ok);
    assert_eq!(cat, "nested file contents\n");
}

#[test]
fn scaffold_formats_detect_but_decline() {
    // A fake 7z header: detection must name it, but reads must return a
    // clean "detection-only" Unsupported rather than panicking.
    let work = tempfile::tempdir().unwrap();
    let f = work.path().join("fake.7z");
    let mut bytes = b"7z\xBC\xAF\x27\x1C\x00\x04".to_vec();
    bytes.extend(std::iter::repeat_n(0u8, 200));
    std::fs::write(&f, &bytes).unwrap();

    let (ok, info, _) = run(&["info", f.to_str().unwrap()]);
    // `info` prints the kind heading even though listing fails.
    assert!(
        info.contains("7z"),
        "scaffold not detected: {info} (ok={ok})"
    );

    let (ok, _, err) = run(&["ls", f.to_str().unwrap(), "/"]);
    assert!(!ok, "scaffold ls should fail");
    assert!(
        err.contains("detection-only") || err.contains("not implemented"),
        "error: {err}"
    );
}

#[test]
fn detect_fs_recognises_every_archive_magic() {
    use fstool::block::{BlockDevice, MemoryBackend};
    use fstool::inspect::{FsKind, detect_fs};

    // (header bytes, offset, expected kind). Headers are written into a
    // small zeroed device — detection only inspects the first sector.
    let cases: &[(&[u8], usize, FsKind)] = &[
        (b"PK\x03\x04", 0, FsKind::Zip),
        (b"070701", 0, FsKind::Cpio),
        (b"070707", 0, FsKind::Cpio),
        (b"!<arch>\n", 0, FsKind::Ar),
        (b"7z\xBC\xAF\x27\x1C", 0, FsKind::SevenZ),
        (b"Rar!\x1A\x07\x00", 0, FsKind::Rar),
        (b"Rar!\x1A\x07\x01\x00", 0, FsKind::Rar),
        (b"MSCF", 0, FsKind::Cab),
        // LHA: header-size + checksum byte, then "-lh5-" at offset 2.
        (b"\x20\x00-lh5-", 0, FsKind::Lha),
        (b"LZX\x00", 0, FsKind::Lzx),
        (b"SIT!", 0, FsKind::Sit),
        (b"StuffIt (c)1997", 0, FsKind::Sit),
        // ARC: 0x1A + method byte, heuristic, checked last.
        (b"\x1A\x08", 0, FsKind::Arc),
    ];

    for (bytes, off, want) in cases {
        let mut dev = MemoryBackend::new(4096);
        dev.write_at(*off as u64, bytes).unwrap();
        let got = detect_fs(&mut dev).unwrap_or_else(|e| panic!("detect {want:?} failed: {e}"));
        assert_eq!(got, *want, "magic {bytes:?} → {got:?}, expected {want:?}");
    }
}
