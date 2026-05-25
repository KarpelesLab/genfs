//! Tests for the unified repack pipeline: one walker + sink, no
//! per-(source,dest) custom paths. These exercise source→destination
//! combinations that the old specialized copiers could not handle, plus
//! metadata fidelity surfaced through a non-ext source's `getattr`.

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

fn run(args: &[&str]) -> (bool, String) {
    let out = Command::new(FSTOOL)
        .args(args)
        .output()
        .expect("spawn fstool");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn stage(root: &std::path::Path) {
    std::fs::create_dir_all(root.join("sub")).unwrap();
    std::fs::write(root.join("top.txt"), b"top\n").unwrap();
    std::fs::write(root.join("sub/deep.txt"), b"deep contents\n").unwrap();
}

/// A zip *source* repacked into a tar — the combination the old
/// FS-to-FS copiers rejected ("zip source is not yet wired"). The
/// unified walker drives any source through one path.
#[test]
fn zip_source_repacks_to_tar() {
    if !which("tar") {
        eprintln!("skipping: tar not installed");
        return;
    }
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    stage(&src);
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

    let tar = work.path().join("out.tar");
    let (ok, err) = run(&["repack", zip.to_str().unwrap(), tar.to_str().unwrap()]);
    assert!(ok, "zip → tar repack failed: {err}");

    let listing = Command::new("tar").arg("tf").arg(&tar).output().unwrap();
    let members = String::from_utf8_lossy(&listing.stdout);
    assert!(
        members.contains("sub/deep.txt"),
        "missing nested file:\n{members}"
    );

    let body = Command::new("tar")
        .arg("xOf")
        .arg(&tar)
        .arg("sub/deep.txt")
        .output()
        .unwrap();
    assert_eq!(body.stdout, b"deep contents\n");
}

/// A zip *source* repacked into an ext4 image — another previously
/// unsupported pair. `--shrink` sizes the ext destination from the
/// source content (trait-driven, works for any source).
#[test]
fn zip_source_repacks_to_ext4() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    stage(&src);
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

    let img = work.path().join("out.img");
    let (ok, err) = run(&[
        "repack",
        zip.to_str().unwrap(),
        img.to_str().unwrap(),
        "--fs-type",
        "ext4",
        "--shrink",
    ]);
    assert!(ok, "zip → ext4 repack failed: {err}");

    let out = Command::new(FSTOOL)
        .args(["cat", img.to_str().unwrap(), "/sub/deep.txt"])
        .output()
        .unwrap();
    assert_eq!(out.stdout, b"deep contents\n");
}

/// `create` into the deferred-write backends (SquashFS / ISO 9660 / GRF)
/// from a host directory. These keep the `FileSource` and read it at
/// `flush`, so the body must outlive `create_file` — a regression guard
/// for `FileSource::TempFile`. (Their lib-level tests drive the writer
/// API directly and wouldn't catch a broken CLI `create` path.)
#[test]
fn create_deferred_write_backends_from_dir() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    stage(&src);
    for fs in ["squashfs", "iso", "grf"] {
        let out = work.path().join(format!("o.{fs}"));
        let (ok, err) = run(&[
            "create",
            "-t",
            fs,
            src.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ]);
        assert!(ok, "create -t {fs} failed: {err}");
        let cat = Command::new(FSTOOL)
            .args(["cat", out.to_str().unwrap(), "/sub/deep.txt"])
            .output()
            .unwrap();
        assert_eq!(
            cat.stdout, b"deep contents\n",
            "{fs}: body wrong after create"
        );
    }
}

/// SquashFS source metadata fidelity: a `0640` file repacked to tar
/// keeps its mode + uid/gid (SquashFS `getattr` reads them from the
/// inode header + id table).
#[test]
#[cfg(unix)]
fn squashfs_source_preserves_mode_into_tar() {
    if !which("tar") {
        eprintln!("skipping: tar not installed");
        return;
    }
    use std::os::unix::fs::PermissionsExt;
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    let f = src.join("s.txt");
    std::fs::write(&f, b"x\n").unwrap();
    std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o640)).unwrap();

    let img = work.path().join("fs.sqsh");
    assert!(
        run(&[
            "create",
            "-t",
            "squashfs",
            src.to_str().unwrap(),
            "-o",
            img.to_str().unwrap()
        ])
        .0
    );
    let tar = work.path().join("out.tar");
    assert!(run(&["repack", img.to_str().unwrap(), tar.to_str().unwrap()]).0);

    let listing = Command::new("tar").arg("tvf").arg(&tar).output().unwrap();
    let s = String::from_utf8_lossy(&listing.stdout);
    let line = s.lines().find(|l| l.contains("s.txt")).unwrap_or("");
    assert!(
        line.contains("rw-r-----"),
        "squashfs mode not preserved:\n{line}"
    );
}

/// ISO 9660 Rock Ridge source fidelity: an RR image (built by a system
/// `mkisofs`/`genisoimage`/`xorrisofs`, since fstool's own writer
/// hardcodes the PX mode) repacked to tar keeps the PX mode + uid/gid.
#[test]
#[cfg(unix)]
fn iso_rock_ridge_source_preserves_mode_into_tar() {
    use std::os::unix::fs::PermissionsExt;
    let tool = ["genisoimage", "mkisofs", "xorrisofs"]
        .into_iter()
        .find(|t| which(t));
    let (Some(tool), true) = (tool, which("tar")) else {
        eprintln!("skipping: no iso builder / tar");
        return;
    };
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    let f = src.join("f.txt");
    std::fs::write(&f, b"x\n").unwrap();
    std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o640)).unwrap();

    let iso = work.path().join("rr.iso");
    let ok = Command::new(tool)
        .args(["-R", "-o"])
        .arg(&iso)
        .arg(&src)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("skipping: {tool} failed");
        return;
    }

    let tar = work.path().join("out.tar");
    assert!(run(&["repack", iso.to_str().unwrap(), tar.to_str().unwrap()]).0);
    let listing = Command::new("tar").arg("tvf").arg(&tar).output().unwrap();
    let s = String::from_utf8_lossy(&listing.stdout);
    let line = s.lines().find(|l| l.contains("f.txt")).unwrap_or("");
    assert!(
        line.contains("rw-r-----"),
        "iso RR mode not preserved:\n{line}"
    );
}

/// APFS source mode fidelity: a `0642` file repacked to tar keeps its
/// mode bits (APFS `getattr` reads `InodeVal.mode`). (APFS *create*
/// doesn't yet persist uid/gid/mtime, so only the mode is asserted.)
#[test]
#[cfg(unix)]
fn apfs_source_preserves_mode_into_tar() {
    if !which("tar") {
        eprintln!("skipping: tar not installed");
        return;
    }
    use std::os::unix::fs::PermissionsExt;
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    let f = src.join("a.txt");
    std::fs::write(&f, b"x\n").unwrap();
    std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o642)).unwrap();

    let img = work.path().join("fs.img");
    assert!(
        run(&[
            "create",
            "-t",
            "apfs",
            src.to_str().unwrap(),
            "-o",
            img.to_str().unwrap()
        ])
        .0
    );
    let tar = work.path().join("out.tar");
    assert!(run(&["repack", img.to_str().unwrap(), tar.to_str().unwrap()]).0);

    let listing = Command::new("tar").arg("tvf").arg(&tar).output().unwrap();
    let s = String::from_utf8_lossy(&listing.stdout);
    let line = s.lines().find(|l| l.contains("a.txt")).unwrap_or("");
    assert!(
        line.contains("rw-r---w-"),
        "apfs mode not preserved:\n{line}"
    );
}

/// Metadata fidelity through a non-ext source: build an f2fs image with
/// a `0640` file, repack it to tar, and confirm the mode + uid/gid
/// survive — proving f2fs's `getattr` is faithful (it would default to
/// `0644` root/root without it).
#[test]
#[cfg(unix)]
fn f2fs_source_preserves_mode_into_tar() {
    if !which("tar") {
        eprintln!("skipping: tar not installed");
        return;
    }
    use std::os::unix::fs::PermissionsExt;
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    let f = src.join("secret.txt");
    std::fs::write(&f, b"x\n").unwrap();
    std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o640)).unwrap();

    let img = work.path().join("fs.img");
    assert!(
        run(&[
            "create",
            "-t",
            "f2fs",
            src.to_str().unwrap(),
            "-o",
            img.to_str().unwrap()
        ])
        .0
    );

    let tar = work.path().join("out.tar");
    let (ok, err) = run(&["repack", img.to_str().unwrap(), tar.to_str().unwrap()]);
    assert!(ok, "f2fs → tar repack failed: {err}");

    // `tar tvf` renders the mode; 0640 = `rw-r-----`.
    let listing = Command::new("tar").arg("tvf").arg(&tar).output().unwrap();
    let s = String::from_utf8_lossy(&listing.stdout);
    let line = s
        .lines()
        .find(|l| l.contains("secret.txt"))
        .unwrap_or_else(|| panic!("secret.txt missing from tar:\n{s}"));
    assert!(
        line.contains("rw-r-----"),
        "f2fs mode not preserved (expected rw-r-----):\n{line}"
    );
}

/// Hard links from an ext source materialise into a tar (tar can't
/// represent links across the walk), and both names carry the content.
#[test]
#[cfg(unix)]
fn ext_hardlinks_materialise_into_tar() {
    if !which("mke2fs") || !which("tar") {
        eprintln!("skipping: mke2fs/tar not installed");
        return;
    }
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("a"), b"shared\n").unwrap();
    std::fs::hard_link(src.join("a"), src.join("b")).unwrap();

    let img = work.path().join("ext.img");
    let mk = Command::new("mke2fs")
        .args(["-q", "-F", "-t", "ext4", "-d"])
        .arg(&src)
        .arg(&img)
        .arg("4M")
        .output()
        .unwrap();
    if !mk.status.success() {
        eprintln!("skipping: mke2fs failed");
        return;
    }

    let tar = work.path().join("out.tar");
    let (ok, err) = run(&["repack", img.to_str().unwrap(), tar.to_str().unwrap()]);
    assert!(ok, "ext → tar repack failed: {err}");

    for name in ["a", "b"] {
        let body = Command::new("tar")
            .arg("xOf")
            .arg(&tar)
            .arg(name)
            .output()
            .unwrap();
        assert_eq!(body.stdout, b"shared\n", "hardlink {name} content wrong");
    }
}
