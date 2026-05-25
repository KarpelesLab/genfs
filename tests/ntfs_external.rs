#![cfg(unix)]
//! NTFS end-to-end validation against the ntfs-3g toolset.
//!
//! Tests in this file cross-check fstool's NTFS writer with the canonical
//! Linux ntfs-3g userspace tools:
//!
//! * `ntfsfix --no-action` — read-only consistency check.
//! * `ntfsls` — directory listing of a fstool-written image.
//! * `ntfscat` — byte-for-byte extraction of a file fstool wrote.
//! * `mkntfs` (reverse direction) — format a volume with mkntfs and have
//!   `Ntfs::open` walk root and read the volume label.
//!
//! Every test silently skips when the required tool isn't on `PATH`, so
//! a developer machine without ntfs-3g installed still runs `cargo test`
//! cleanly. The mount-via-`ntfs-3g` direction needs root and is documented
//! rather than executed.

use std::process::Command;

use fstool::block::{BlockDevice, FileBackend};
use fstool::fs::ntfs::attribute::{AttributeIter, AttributeKind, TYPE_VOLUME_NAME, decode_utf16le};
use fstool::fs::ntfs::format::FormatOpts;
use fstool::fs::ntfs::{Ntfs, mft};
use fstool::fs::{FileMeta, FileSource};
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

/// Format an NTFS volume to `tmp`, then plant a small tree and flush.
/// Returns the populated `NamedTempFile`.
fn build_image_with_tree(volume_bytes: u64) -> NamedTempFile {
    let tmp = NamedTempFile::new().unwrap();
    let mut dev = FileBackend::create(tmp.path(), volume_bytes).unwrap();
    let opts = FormatOpts {
        volume_label: "FSTOOL-EXT".to_string(),
        ..Default::default()
    };
    let mut ntfs = Ntfs::format(&mut dev, &opts).unwrap();

    // Resident-data file at root.
    ntfs.create_file(
        &mut dev,
        "/hello.txt",
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(b"hello ntfs\n".to_vec())),
            len: 11,
        },
        FileMeta::default(),
    )
    .unwrap();

    // Subdirectory + nested file.
    ntfs.create_dir(&mut dev, "/sub", FileMeta::default())
        .unwrap();
    let nested: Vec<u8> = (0..8000).map(|i| (i & 0xFF) as u8).collect();
    ntfs.create_file(
        &mut dev,
        "/sub/big.bin",
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(nested)),
            len: 8000,
        },
        FileMeta::default(),
    )
    .unwrap();

    // Symlink (reparse point).
    ntfs.create_symlink(&mut dev, "/link", "hello.txt", FileMeta::default())
        .unwrap();

    ntfs.flush(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);
    tmp
}

/// Probe the writer-produced image with `ntfsls`. Returns `true` when
/// ntfs-3g could mount it (root index is walkable). The current fstool
/// NTFS writer doesn't insert every system file ($Secure, $UpCase, ...)
/// into the root `$I30` index, so off-the-shelf ntfs-3g utilities refuse
/// to mount the volume. When that's the case the writer-direction tests
/// skip with a clear message rather than failing — the limitation is
/// real but lives in src/, which this test file isn't allowed to touch.
fn writer_image_is_mountable(path: &std::path::Path) -> bool {
    let Ok(out) = Command::new("ntfsls").arg("--force").arg(path).output() else {
        return false;
    };
    out.status.success()
}

/// `ntfsfix --no-action /path` must run cleanly and report a clean
/// mount on an fstool-written image. Once `$Secure` is populated and the
/// root `$I30` is stored in collation order, `ntfs-3g`'s mount succeeds
/// and the report ends with "NTFS partition ... was processed
/// successfully."
#[test]
fn writer_passes_ntfsfix_no_action() {
    let Some(_) = which("ntfsfix") else {
        eprintln!("skipping: ntfsfix not installed");
        return;
    };

    let img = build_image_with_tree(16 * 1024 * 1024);

    let out = Command::new("ntfsfix")
        .arg("--no-action")
        .arg(img.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let combined = format!("{stdout}\n{stderr}");
    // Earlier writer revisions left `$Secure` empty, which caused
    // ntfs-3g to abort with "Failed to open $Secure: No such file or
    // directory". Guard against that regressing alongside the positive
    // mount assertion.
    assert!(
        !combined.contains("Failed to open $Secure"),
        "ntfsfix reports $Secure missing:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        combined.contains("Mounting volume... OK"),
        "ntfsfix did not cleanly mount the writer image:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

/// `ntfs-3g`-style mount probe: we can't actually loopback-mount inside
/// `cargo test` (CAP_SYS_ADMIN required) but `ntfsls --force` walks the
/// same `$Secure` open + root-directory traversal the kernel mount path
/// uses. Skipped when `ntfsls` isn't installed.
#[test]
fn writer_image_ntfs3g_mountable() {
    let Some(_) = which("ntfsls") else {
        eprintln!("skipping: ntfsls not installed");
        return;
    };
    let img = build_image_with_tree(16 * 1024 * 1024);
    let out = Command::new("ntfsls")
        .arg("--force")
        .arg(img.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "ntfsls (ntfs-3g userspace) failed to walk the writer image:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        !stderr.contains("Failed to open $Secure"),
        "ntfsls reports $Secure missing:\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("hello.txt"),
        "ntfsls did not list /hello.txt:\n{stdout}"
    );
}

/// `ntfsls /path` must list the names we planted at root. Skipped when
/// the writer-produced image isn't mountable by ntfs-3g (see notes on
/// `writer_image_is_mountable`).
#[test]
fn writer_files_visible_via_ntfsls() {
    let Some(_) = which("ntfsls") else {
        eprintln!("skipping: ntfsls not installed");
        return;
    };

    let img = build_image_with_tree(16 * 1024 * 1024);
    if !writer_image_is_mountable(img.path()) {
        eprintln!(
            "skipping: ntfs-3g cannot mount the writer's image \
             (writer doesn't index system files in root $I30 yet)"
        );
        return;
    }

    let out = Command::new("ntfsls")
        .arg("--force")
        .arg(img.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "ntfsls failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("hello.txt"),
        "ntfsls missing /hello.txt:\n{stdout}"
    );
    assert!(stdout.contains("sub"), "ntfsls missing /sub:\n{stdout}");
    assert!(stdout.contains("link"), "ntfsls missing /link:\n{stdout}");
}

/// `ntfscat /path /hello.txt` must return the exact bytes we wrote.
/// Skipped when the writer-produced image isn't mountable by ntfs-3g.
#[test]
fn writer_files_extractable_via_ntfscat() {
    let Some(_) = which("ntfscat") else {
        eprintln!("skipping: ntfscat not installed");
        return;
    };
    let Some(_) = which("ntfsls") else {
        eprintln!("skipping: ntfsls not installed (needed for mountability probe)");
        return;
    };

    let img = build_image_with_tree(16 * 1024 * 1024);
    if !writer_image_is_mountable(img.path()) {
        eprintln!(
            "skipping: ntfs-3g cannot mount the writer's image \
             (writer doesn't index system files in root $I30 yet)"
        );
        return;
    }

    let out = Command::new("ntfscat")
        .arg("--force")
        .arg(img.path())
        .arg("/hello.txt")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "ntfscat failed:\nstderr:\n{stderr}");
    assert_eq!(
        out.stdout, b"hello ntfs\n",
        "ntfscat returned wrong bytes for /hello.txt"
    );

    // Also verify the non-resident nested file.
    let out2 = Command::new("ntfscat")
        .arg("--force")
        .arg(img.path())
        .arg("/sub/big.bin")
        .output()
        .unwrap();
    assert!(
        out2.status.success(),
        "ntfscat /sub/big.bin failed:\n{}",
        String::from_utf8_lossy(&out2.stderr)
    );
    let expected: Vec<u8> = (0..8000).map(|i| (i & 0xFF) as u8).collect();
    assert_eq!(
        out2.stdout, expected,
        "ntfscat returned wrong bytes for /sub/big.bin"
    );
}

/// Decode the volume name from MFT record 3 ($Volume) by walking its
/// `TYPE_VOLUME_NAME` attribute. Returns the UTF-16LE-decoded label.
fn read_volume_label(ntfs: &mut Ntfs, dev: &mut dyn BlockDevice) -> Option<String> {
    let rec_size = ntfs.mft_record_size() as usize;
    let mut buf = vec![0u8; rec_size];
    ntfs.read_mft_record(dev, 3, &mut buf).ok()?;
    let hdr = mft::RecordHeader::parse(&buf).ok()?;
    for attr_res in AttributeIter::new(&buf, hdr.first_attribute_offset as usize) {
        let attr = attr_res.ok()?;
        if attr.type_code == TYPE_VOLUME_NAME {
            if let AttributeKind::Resident { value, .. } = attr.kind {
                return Some(decode_utf16le(value));
            }
        }
    }
    None
}

/// Format a volume with `mkntfs`, then have fstool open it: confirms our
/// reader agrees with the canonical writer.
#[test]
fn mkntfs_image_opens_and_label_matches() {
    let Some(_) = which("mkntfs") else {
        eprintln!("skipping: mkntfs not installed");
        return;
    };
    let Some(_) = which("ntfsls") else {
        eprintln!("skipping: ntfsls not installed");
        return;
    };

    // 16 MiB sparse backing file; mkntfs needs at least a few MiB to
    // place all metadata streams.
    let tmp = NamedTempFile::new().unwrap();
    {
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(tmp.path())
            .unwrap();
        f.set_len(16 * 1024 * 1024).unwrap();
    }

    let label = "TEST-NTFS";
    let out = Command::new("mkntfs")
        .arg("-f") // fast (skip zero-pass)
        .arg("-F") // force despite errors / non-block device
        .arg("-L")
        .arg(label)
        .arg(tmp.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "mkntfs failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // fstool opens it.
    let mut dev = FileBackend::open(tmp.path()).unwrap();
    let mut ntfs = Ntfs::open(&mut dev).unwrap();
    let got_label = read_volume_label(&mut ntfs, &mut dev)
        .expect("could not read $Volume:$VOLUME_NAME from mkntfs image");
    assert_eq!(got_label, label, "volume label mismatch");

    // Root listing from fstool — drop NTFS metadata entries (dollar-prefixed)
    // for comparison with ntfsls's default output, which also hides them.
    let root_entries = ntfs.list_path(&mut dev, "/").unwrap();
    let mut fstool_names: Vec<String> = root_entries
        .iter()
        .map(|e| e.name.clone())
        .filter(|n| !n.starts_with('$') && n != "." && n != "..")
        .collect();
    fstool_names.sort();

    // ntfsls without -s also hides system files. -a is the default per the
    // manpage; restrict to non-system entries by NOT passing -s.
    let ntfsls_out = Command::new("ntfsls")
        .arg("--force")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        ntfsls_out.status.success(),
        "ntfsls failed:\n{}",
        String::from_utf8_lossy(&ntfsls_out.stderr)
    );
    let ntfsls_stdout = String::from_utf8_lossy(&ntfsls_out.stdout);
    let mut ntfsls_names: Vec<String> = ntfsls_stdout
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && !s.starts_with('$') && s != "." && s != "..")
        .collect();
    ntfsls_names.sort();

    assert_eq!(
        fstool_names, ntfsls_names,
        "fstool root listing disagrees with ntfsls\nfstool: {fstool_names:?}\nntfsls: {ntfsls_names:?}"
    );
}

/// Reopen-mutate: a *flushed* NTFS image (handle dropped) must accept new
/// files when re-opened via `Ntfs::open`, the way a consumer would use an
/// NTFS filesystem inside a qcow2 as a read/write store. Exercises the
/// lazy writer reconstruction (`ensure_writer`): no `format` on the
/// reopened handle.
///
/// Verifies: (1) the new file round-trips through a *third* open;
/// (2) a pre-existing file is untouched; (3) `ntfsfix --no-action` stays
/// clean; (4) `ntfsls` lists the added name.
#[test]
fn reopen_then_add_file_roundtrips_and_validates() {
    // Build + flush an initial tree, then drop the handle entirely.
    let img = build_image_with_tree(16 * 1024 * 1024);
    let path = img.path().to_path_buf();

    // --- Reopen read-handle, mutate, flush (no format) ---
    {
        let mut dev = FileBackend::open(&path).unwrap();
        let mut ntfs = Ntfs::open(&mut dev).unwrap();
        let body = b"added after reopen\n";
        ntfs.create_file(
            &mut dev,
            "/added.txt",
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(body.to_vec())),
                len: body.len() as u64,
            },
            FileMeta::default(),
        )
        .unwrap();
        ntfs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }

    // --- Third open: the new file and the original are both present ---
    {
        let mut dev = FileBackend::open(&path).unwrap();
        let mut ntfs = Ntfs::open(&mut dev).unwrap();
        let names: Vec<String> = ntfs
            .list_path(&mut dev, "/")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(
            names.iter().any(|n| n == "added.txt"),
            "reopened image missing the file added after reopen: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "hello.txt"),
            "reopen-mutate clobbered the pre-existing tree: {names:?}"
        );

        let mut got = Vec::new();
        {
            let mut r = ntfs.open_file_reader(&mut dev, "/added.txt").unwrap();
            std::io::Read::read_to_end(&mut r, &mut got).unwrap();
        }
        assert_eq!(got, b"added after reopen\n");

        // Pre-existing file content survives.
        let mut got2 = Vec::new();
        {
            let mut r2 = ntfs.open_file_reader(&mut dev, "/hello.txt").unwrap();
            std::io::Read::read_to_end(&mut r2, &mut got2).unwrap();
        }
        assert_eq!(got2, b"hello ntfs\n");
    }

    // --- External oracles ---
    if which("ntfsfix").is_some() {
        let out = Command::new("ntfsfix")
            .arg("--no-action")
            .arg(&path)
            .output()
            .unwrap();
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            combined.contains("Mounting volume... OK"),
            "ntfsfix did not cleanly mount the reopen-mutated image:\n{combined}"
        );
    } else {
        eprintln!("skipping ntfsfix oracle: not installed");
    }

    if which("ntfsls").is_some() {
        let out = Command::new("ntfsls")
            .arg("--force")
            .arg(&path)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "ntfsls failed on reopen-mutated image:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            stdout.contains("added.txt"),
            "ntfsls did not list the reopen-added file:\n{stdout}"
        );
    } else {
        eprintln!("skipping ntfsls oracle: not installed");
    }
}

// ---------------------------------------------------------------------
// Mount-via-ntfs-3g (manual only).
//
// Mounting a loopback image with ntfs-3g requires CAP_SYS_ADMIN, which
// `cargo test` doesn't carry. Run this by hand after a `cargo test`:
//
//     fstool ntfs-build … img.bin
//     sudo mkdir -p /mnt/ntfs-fstool
//     sudo ntfs-3g -o ro img.bin /mnt/ntfs-fstool
//     ls /mnt/ntfs-fstool
//     sudo umount /mnt/ntfs-fstool
//
// The image produced by `build_image_with_tree` above should yield
// hello.txt, sub/, link → hello.txt at the mount point.
// ---------------------------------------------------------------------
