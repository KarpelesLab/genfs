//! End-to-end validation of `Source::Layered` flattening + whiteouts.
//!
//! Drives the public `fstool repack` CLI with `+`-separated layered
//! source specs. Verifies that:
//!
//! - upper-layer files override lower-layer files of the same path,
//! - `.wh.<name>` tombstones delete the named entry from the merge,
//! - `.wh..wh..opq` opaque markers wipe lower-layer children before
//!   the upper layer's own children land.

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

fn make_tar(dir: &std::path::Path, members: &[(&str, &[u8])]) -> std::path::PathBuf {
    let stage = tempfile::tempdir().unwrap();
    for (name, body) in members {
        let p = stage.path().join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&p, body).unwrap();
    }
    let tar_path = dir.join(format!(
        "layer-{}.tar",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let args: Vec<String> = members
        .iter()
        .map(|(name, _)| name.to_string())
        .collect();
    let mut cmd = Command::new("tar");
    cmd.arg("cf")
        .arg(&tar_path)
        .current_dir(stage.path())
        .args(&args);
    let status = cmd.status().expect("spawn tar");
    assert!(status.success(), "tar cf failed");
    tar_path
}

fn list_tar(path: &std::path::Path) -> Vec<String> {
    let out = Command::new("tar")
        .arg("tf")
        .arg(path)
        .output()
        .expect("spawn tar tf");
    assert!(out.status.success(), "tar tf failed");
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|s| {
            s.trim_end_matches('/')
                .trim_start_matches('/')
                .to_string()
        })
        .collect()
}

fn extract_tar_file(path: &std::path::Path, member: &str) -> Vec<u8> {
    // fstool's tar writer prefixes a leading `/` on each entry; pass
    // both the prefixed and unprefixed candidate so the test reads
    // through whichever GNU tar accepts.
    let candidate = format!("/{member}");
    let out = Command::new("tar")
        .arg("xOf")
        .arg(path)
        .arg(&candidate)
        .output()
        .expect("spawn tar xOf");
    if out.status.success() && !out.stdout.is_empty() {
        return out.stdout;
    }
    let out = Command::new("tar")
        .arg("xOf")
        .arg(path)
        .arg(member)
        .output()
        .expect("spawn tar xOf");
    assert!(out.status.success(), "tar xOf {member} failed");
    out.stdout
}

#[test]
fn merge_two_tars_override_and_whiteout() {
    if which("tar").is_none() {
        eprintln!("skipping: tar not installed");
        return;
    }
    ensure_built();
    let work = tempfile::tempdir().unwrap();

    // base: etc/conf=v1, etc/keep=k, etc/sub/x=x1
    let base = make_tar(
        work.path(),
        &[
            ("etc/conf", b"v1"),
            ("etc/keep", b"k"),
            ("etc/sub/x", b"x1"),
        ],
    );
    // top: etc/conf=v2 (override), etc/.wh.keep (delete keep), etc/new=n
    let top = make_tar(
        work.path(),
        &[
            ("etc/conf", b"v2"),
            ("etc/.wh.keep", b""),
            ("etc/new", b"n"),
        ],
    );

    let out_tar = work.path().join("merged.tar");
    let spec = format!("{}+{}", base.display(), top.display());

    let status = Command::new(fstool())
        .arg("repack")
        .arg(&spec)
        .arg(&out_tar)
        .status()
        .expect("spawn fstool repack");
    assert!(status.success(), "fstool repack failed");

    let members = list_tar(&out_tar);
    assert!(members.iter().any(|m| m == "etc/conf"), "etc/conf present");
    assert!(members.iter().any(|m| m == "etc/new"), "etc/new present");
    assert!(members.iter().any(|m| m == "etc/sub/x"), "etc/sub/x kept");
    assert!(
        !members.iter().any(|m| m == "etc/keep"),
        "etc/keep should be deleted by .wh.keep"
    );
    assert!(
        !members.iter().any(|m| m.contains(".wh.")),
        "no .wh.* tombstones leak through"
    );

    let conf = extract_tar_file(&out_tar, "etc/conf");
    assert_eq!(conf, b"v2", "etc/conf must come from top layer");
}

#[test]
fn merge_opaque_dir_drops_lower_children() {
    if which("tar").is_none() {
        eprintln!("skipping: tar not installed");
        return;
    }
    ensure_built();
    let work = tempfile::tempdir().unwrap();

    // base: etc/a=A, etc/b=B
    let base = make_tar(work.path(), &[("etc/a", b"A"), ("etc/b", b"B")]);
    // top: etc/.wh..wh..opq (opaque), etc/c=C
    let top = make_tar(
        work.path(),
        &[("etc/.wh..wh..opq", b""), ("etc/c", b"C")],
    );

    let out_tar = work.path().join("merged.tar");
    let spec = format!("{}+{}", base.display(), top.display());

    let status = Command::new(fstool())
        .arg("repack")
        .arg(&spec)
        .arg(&out_tar)
        .status()
        .expect("spawn fstool repack");
    assert!(status.success(), "fstool repack failed");

    let members = list_tar(&out_tar);
    assert!(members.iter().any(|m| m == "etc/c"), "etc/c kept");
    assert!(
        !members.iter().any(|m| m == "etc/a"),
        "etc/a wiped by opaque marker"
    );
    assert!(
        !members.iter().any(|m| m == "etc/b"),
        "etc/b wiped by opaque marker"
    );
    assert!(
        !members.iter().any(|m| m.contains(".wh.")),
        "no .wh.* tombstones leak through"
    );
}
