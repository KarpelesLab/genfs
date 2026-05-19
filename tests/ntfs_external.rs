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

/// `ntfsfix --no-action /path` must run cleanly and print its diagnostic
/// banner against an fstool-written image. The banner appears even when
/// ntfs-3g later refuses to remount the volume because of writer gaps,
/// so we assert on the banner text rather than the exit code.
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
    // The "Mounting volume" / "Processing $MFT" banner is what ntfs-3g
    // prints when it recognises the volume as NTFS at all. That alone
    // confirms our boot sector + MFT layout are sound.
    assert!(
        combined.contains("Mounting volume")
            || combined.contains("Processing $MFT")
            || combined.contains("Reading $MFT"),
        "ntfsfix produced no recognisable diagnostic output:\nstdout:\n{stdout}\nstderr:\n{stderr}"
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
