//! End-to-end validation of the GRF reader + writer through the
//! `fstool` CLI. No external tools needed — the test drives `fstool
//! repack` to build a GRF from a tar, then reopens it via `ls` /
//! `cat` and checks the bytes round-trip.

use std::process::Command;

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

fn fstool() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target");
    p.push(if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    });
    p.push("fstool");
    p
}

fn ensure_built() {
    if fstool().exists() {
        return;
    }
    let status = Command::new("cargo")
        .args(["build", "--bin", "fstool"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("spawn cargo build");
    assert!(status.success(), "cargo build fstool failed");
}

#[test]
fn build_grf_from_tar_and_round_trip() {
    if which("tar").is_none() {
        eprintln!("skipping: tar not installed");
        return;
    }
    ensure_built();

    let work = tempfile::tempdir().unwrap();
    let stage = work.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    std::fs::write(stage.join("a.txt"), b"hello, world!\n").unwrap();
    std::fs::write(stage.join("b.bin"), vec![0x42u8; 4096]).unwrap();

    let tar_path = work.path().join("src.tar");
    let status = Command::new("tar")
        .arg("cf")
        .arg(&tar_path)
        .arg("a.txt")
        .arg("b.bin")
        .current_dir(&stage)
        .status()
        .expect("spawn tar");
    assert!(status.success(), "tar cf failed");

    let grf_path = work.path().join("out.grf");
    let status = Command::new(fstool())
        .arg("repack")
        .arg(&tar_path)
        .arg(&grf_path)
        .arg("--fs-type")
        .arg("grf")
        .status()
        .expect("spawn fstool repack");
    assert!(status.success(), "fstool repack tar → grf failed");

    // The GRF magic should land at offset 0.
    let head = std::fs::read(&grf_path).unwrap();
    assert!(
        head.len() >= 16 && &head[0..16] == b"Master of Magic\0",
        "expected GRF magic at offset 0"
    );

    // `ls /` should show both files.
    let out = Command::new(fstool())
        .arg("ls")
        .arg(&grf_path)
        .arg("/")
        .output()
        .expect("spawn fstool ls");
    assert!(out.status.success(), "fstool ls failed");
    let listing = String::from_utf8(out.stdout).unwrap();
    assert!(listing.contains("a.txt"), "ls didn't list a.txt: {listing}");
    assert!(listing.contains("b.bin"), "ls didn't list b.bin: {listing}");

    // `cat /a.txt` should reproduce the source bytes verbatim.
    let out = Command::new(fstool())
        .arg("cat")
        .arg(&grf_path)
        .arg("/a.txt")
        .output()
        .expect("spawn fstool cat");
    assert!(out.status.success(), "fstool cat failed");
    assert_eq!(out.stdout, b"hello, world!\n");

    // `info` should report version 0x200 and 2 files.
    let out = Command::new(fstool())
        .arg("info")
        .arg(&grf_path)
        .output()
        .expect("spawn fstool info");
    let info = String::from_utf8(out.stdout).unwrap();
    assert!(info.contains("0x200"), "info missing version: {info}");
    assert!(info.contains("file count:        2"), "info missing count: {info}");
}

#[test]
fn cp949_filename_round_trips_through_cli() {
    if which("tar").is_none() {
        eprintln!("skipping: tar not installed");
        return;
    }
    ensure_built();

    let work = tempfile::tempdir().unwrap();
    let stage = work.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    let korean_name = "한글.txt";
    std::fs::write(stage.join(korean_name), b"korean text\n").unwrap();

    let tar_path = work.path().join("src.tar");
    let status = Command::new("tar")
        .arg("cf")
        .arg(&tar_path)
        .arg(korean_name)
        .current_dir(&stage)
        .status()
        .expect("spawn tar");
    assert!(status.success(), "tar cf failed");

    let grf_path = work.path().join("out.grf");
    let status = Command::new(fstool())
        .arg("repack")
        .arg(&tar_path)
        .arg(&grf_path)
        .arg("--fs-type")
        .arg("grf")
        .status()
        .expect("spawn fstool repack");
    assert!(status.success(), "fstool repack failed");

    let out = Command::new(fstool())
        .arg("ls")
        .arg(&grf_path)
        .arg("/")
        .output()
        .expect("spawn fstool ls");
    let listing = String::from_utf8(out.stdout).unwrap();
    assert!(
        listing.contains(korean_name),
        "Hangul name lost in round trip: {listing}"
    );

    let out = Command::new(fstool())
        .arg("cat")
        .arg(&grf_path)
        .arg(format!("/{korean_name}"))
        .output()
        .expect("spawn fstool cat");
    assert_eq!(out.stdout, b"korean text\n");
}

#[test]
fn grf_round_trip_to_tar_and_back() {
    if which("tar").is_none() {
        eprintln!("skipping: tar not installed");
        return;
    }
    ensure_built();

    let work = tempfile::tempdir().unwrap();
    let stage = work.path().join("stage");
    std::fs::create_dir_all(&stage).unwrap();
    std::fs::write(stage.join("file.txt"), b"round trip data").unwrap();

    let src_tar = work.path().join("src.tar");
    Command::new("tar")
        .arg("cf")
        .arg(&src_tar)
        .arg("file.txt")
        .current_dir(&stage)
        .status()
        .expect("spawn tar");

    // tar → grf
    let grf = work.path().join("out.grf");
    Command::new(fstool())
        .arg("repack")
        .arg(&src_tar)
        .arg(&grf)
        .arg("--fs-type")
        .arg("grf")
        .status()
        .expect("spawn fstool tar→grf");

    // grf → tar
    let out_tar = work.path().join("out.tar");
    Command::new(fstool())
        .arg("repack")
        .arg(&grf)
        .arg(&out_tar)
        .status()
        .expect("spawn fstool grf→tar");

    // Confirm the file came through.
    let out = Command::new("tar")
        .args(["xOf"])
        .arg(&out_tar)
        .arg("/file.txt")
        .output()
        .expect("spawn tar xOf");
    let body = if out.status.success() && !out.stdout.is_empty() {
        out.stdout
    } else {
        // fstool's tar writer sometimes prefixes a leading `/` on
        // entries; try the unprefixed form too.
        let out = Command::new("tar")
            .args(["xOf"])
            .arg(&out_tar)
            .arg("file.txt")
            .output()
            .expect("spawn tar xOf");
        assert!(out.status.success(), "tar xOf file.txt failed");
        out.stdout
    };
    assert_eq!(body, b"round trip data");
}
