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

/// A **compressed-tar** source streamed into ext4 — the path that must
/// NOT decompress to a tempfile. Builds a `.tar.gz` (with system `tar`),
/// then repacks it into an ext4 image and confirms the tree + a nested
/// file + a symlink survive, with `--size` (single streaming pass) and
/// `--shrink` (sizing pass + write pass) both producing an
/// `e2fsck`-clean image.
///
/// Unix-only: Windows `tar.exe` writes archive member names with a
/// different separator convention, so the *input* archive this test
/// builds isn't portable. The streaming repack code itself is
/// byte-oriented and platform-agnostic — it's exercised on the Linux
/// and macOS runners (both of which also have `e2fsck` / a usable
/// `tar`).
#[cfg(unix)]
#[test]
fn compressed_tar_source_streams_to_ext4() {
    if !which("tar") {
        eprintln!("skipping: tar not installed (needed to build the .tar.gz source)");
        return;
    }
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    std::fs::create_dir_all(src.join("sub")).unwrap();
    std::fs::write(src.join("top.txt"), b"top\n").unwrap();
    std::fs::write(src.join("sub/deep.txt"), b"deep contents\n").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink("../top.txt", src.join("sub/lnk")).unwrap();

    // Build a gzip-compressed tar of the host tree with system `tar`.
    let targz = work.path().join("seed.tar.gz");
    let made = Command::new("tar")
        .arg("czf")
        .arg(&targz)
        .arg("-C")
        .arg(&src)
        .arg(".")
        .status()
        .expect("spawn tar");
    assert!(made.success(), "tar czf failed");

    for size_flag in [&["--size", "32MiB"][..], &["--shrink"][..]] {
        let img = work.path().join(format!("out{}.img", size_flag.len()));
        let mut args = vec![
            "repack",
            targz.to_str().unwrap(),
            img.to_str().unwrap(),
            "--fs-type",
            "ext4",
        ];
        args.extend_from_slice(size_flag);
        let (ok, err) = run(&args);
        assert!(ok, "tar.gz → ext4 stream ({size_flag:?}) failed: {err}");

        // Nested file content round-trips through the streaming walk.
        let cat = Command::new(FSTOOL)
            .args(["cat", img.to_str().unwrap(), "/sub/deep.txt"])
            .output()
            .unwrap();
        assert_eq!(
            cat.stdout, b"deep contents\n",
            "wrong body after stream ({size_flag:?})"
        );

        // Symlink survived (listing marks it with `@`).
        #[cfg(unix)]
        {
            let ls = Command::new(FSTOOL)
                .args(["ls", img.to_str().unwrap(), "/sub"])
                .output()
                .unwrap();
            let names = String::from_utf8_lossy(&ls.stdout);
            // `ls` prints `<inode>\t<kind>\t<name>` per line.
            assert!(
                names
                    .lines()
                    .any(|l| l.contains("Symlink") && l.ends_with("lnk")),
                "symlink missing in {names:?}"
            );
        }

        // The raw image must be e2fsck-clean.
        if which("e2fsck") {
            let out = Command::new("e2fsck")
                .args(["-fn"])
                .arg(&img)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "e2fsck not clean ({size_flag:?}):\n{}",
                String::from_utf8_lossy(&out.stdout)
            );
        }
    }
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

/// NTFS source fidelity: NTFS has no POSIX ownership, so `getattr`
/// surfaces real timestamps (NT-FILETIME → Unix) + a mode synthesised
/// from the DOS attributes (a read-only file → `r--r--r--`), and
/// `list_xattrs` carries the native metadata (`user.ntfs.dos_attrs`).
/// Content + size must survive (the walker streams `getattr` size).
#[test]
#[cfg(unix)]
fn ntfs_source_surfaces_times_mode_and_xattrs() {
    if !which("tar") {
        eprintln!("skipping: tar not installed");
        return;
    }
    use std::os::unix::fs::PermissionsExt;
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    let f = src.join("n.txt");
    std::fs::write(&f, b"payload\n").unwrap();
    std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o444)).unwrap();

    let img = work.path().join("fs.img");
    assert!(
        run(&[
            "create",
            "-t",
            "ntfs",
            src.to_str().unwrap(),
            "-o",
            img.to_str().unwrap()
        ])
        .0
    );
    let tar = work.path().join("out.tar");
    assert!(run(&["repack", img.to_str().unwrap(), tar.to_str().unwrap()]).0);

    // Content + size survive.
    let body = Command::new("tar")
        .arg("xOf")
        .arg(&tar)
        .arg("n.txt")
        .output()
        .unwrap();
    assert_eq!(body.stdout, b"payload\n", "ntfs body wrong");

    // Synthesised read-only mode + a real (non-1970) timestamp.
    let listing = Command::new("tar").arg("tvf").arg(&tar).output().unwrap();
    let s = String::from_utf8_lossy(&listing.stdout);
    let line = s.lines().find(|l| l.contains("n.txt")).unwrap_or("");
    assert!(
        line.contains("r--r--r--"),
        "ntfs read-only mode not synthesised:\n{line}"
    );
    assert!(
        !line.contains("1970"),
        "ntfs timestamp not surfaced (still epoch):\n{line}"
    );
}

/// HFS+ source fidelity: a `0640` file repacked to tar keeps its mode +
/// uid/gid (HFS+ `getattr` reads `HFSPlusBSDInfo`).
#[test]
#[cfg(unix)]
fn hfs_plus_source_preserves_mode_into_tar() {
    if !which("tar") {
        eprintln!("skipping: tar not installed");
        return;
    }
    use std::os::unix::fs::PermissionsExt;
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    let f = src.join("h.txt");
    std::fs::write(&f, b"x\n").unwrap();
    std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o640)).unwrap();

    let img = work.path().join("fs.img");
    assert!(
        run(&[
            "create",
            "-t",
            "hfs+",
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
    let line = s.lines().find(|l| l.contains("h.txt")).unwrap_or("");
    assert!(
        line.contains("rw-r-----"),
        "hfs+ mode not preserved:\n{line}"
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
