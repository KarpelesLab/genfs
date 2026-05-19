#![cfg(unix)]
//! HFS+ end-to-end native-tool validation.
//!
//! Round-trips fstool-built images through `fsck.hfs` / `fsck.hfsplus`
//! (the Linux `hfsprogs` package ships the latter spelling), and
//! optionally lets `newfs_hfsplus` format an image that fstool then
//! reads back. Every test skips with a printed reason when the
//! required native tool is missing, mirroring the policy used by
//! `tests/ext4_external.rs`.

use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;

use fstool::block::{BlockDevice, FileBackend};
use fstool::fs::hfs_plus::{FormatOpts, HfsPlus};
use tempfile::NamedTempFile;

/// Volume size used by every writer test. 8 MiB is small enough to keep
/// the tests fast but large enough that the default writer layout
/// (allocation file + catalog file + extents-overflow + optional
/// journal stub) fits with plenty of free blocks left over.
const VOL_BYTES: u64 = 8 * 1024 * 1024;

/// Look for `tool` on `$PATH` the same way the ext4 tests do — invoking
/// `command -v` keeps us identical to `tests/ext4_external.rs` without
/// pulling in a new crate.
fn which(tool: &str) -> Option<PathBuf> {
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

/// Locate an `fsck.hfs`-equivalent binary. Linux ships the tool from
/// the `hfsprogs` package as `fsck.hfsplus`; macOS ships it as
/// `fsck.hfs`. Either spelling is acceptable — return the first one
/// we find. Also gate on the tool actually being runnable (some
/// distros ship a broken stub that errors on `-V`).
fn find_fsck_hfs() -> Option<(PathBuf, &'static str)> {
    for (name, label) in [("fsck.hfs", "fsck.hfs"), ("fsck.hfsplus", "fsck.hfsplus")] {
        if let Some(p) = which(name) {
            // Quick sanity check: ask the binary to print its version /
            // banner. We don't care if the exit code is non-zero (some
            // builds return 1 for "no operation"); we only want to
            // confirm the binary is loadable.
            if Command::new(&p).arg("-V").output().is_ok() {
                return Some((p, label));
            }
        }
    }
    None
}

/// Build an empty fstool-formatted HFS+ image in `tmp` and return the
/// (still-open) device + a handle on the formatted volume.
fn fresh_image(tmp: &NamedTempFile, opts: &FormatOpts) -> (FileBackend, HfsPlus) {
    let mut dev = FileBackend::create(tmp.path(), VOL_BYTES).unwrap();
    let hfs = HfsPlus::format(&mut dev, opts).unwrap();
    (dev, hfs)
}

/// Run `fsck.hfs(plus) -nf <image>` and assert it exits with status 0.
/// `-n` answers "no" to all repair prompts; `-f` forces a check even
/// when the volume already looks clean. The combined flag is what
/// Apple's tool, the macports port, and `hfsprogs` all support.
fn assert_fsck_clean(fsck: &std::path::Path, label: &str, image: &std::path::Path) {
    let out = Command::new(fsck).arg("-nf").arg(image).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "{label} -nf failed on {}:\nstatus: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        image.display(),
        out.status
    );
    // A clean check from either tool prints lines like "The volume
    // appears to be OK". Make sure we did NOT get any of the loud
    // failure markers fsck.hfs(plus) emits when it finds damage.
    let combined = format!("{stdout}\n{stderr}");
    for bad in [
        "Invalid",
        "INVALID",
        "corrupt",
        "CORRUPT",
        "** Repairs are needed",
        "The volume needs to be repaired",
        "could not be verified",
    ] {
        assert!(
            !combined.contains(bad),
            "{label} reported `{bad}` on {}:\n{combined}",
            image.display()
        );
    }
}

/// Fully-loaded writer test: format, populate with a directory tree
/// covering every entry kind we support (regular file, nested dir,
/// symlink, hard link), flush, then validate with `fsck.hfs(-plus)`.
#[test]
fn writer_image_passes_fsck_hfs() {
    let Some((fsck, label)) = find_fsck_hfs() else {
        eprintln!("skipping: fsck.hfs / fsck.hfsplus not installed");
        return;
    };

    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts {
        volume_name: "FstoolHFS".into(),
        ..FormatOpts::default()
    };
    let (mut dev, mut hfs) = fresh_image(&tmp, &opts);

    // /etc + /etc/conf
    hfs.create_dir(&mut dev, "/etc", 0o755, 0, 0).unwrap();
    let body = b"x=1\n";
    let mut src = Cursor::new(&body[..]);
    hfs.create_file(
        &mut dev,
        "/etc/conf",
        &mut src,
        body.len() as u64,
        0o644,
        0,
        0,
    )
    .unwrap();

    // /readme — bigger file that will exercise several extents
    let big: Vec<u8> = (0..16 * 1024).map(|i| (i & 0xFF) as u8).collect();
    let mut src = Cursor::new(&big[..]);
    hfs.create_file(&mut dev, "/readme", &mut src, big.len() as u64, 0o644, 0, 0)
        .unwrap();

    // /link -> etc/conf  (symlink target is a relative path)
    hfs.create_symlink(&mut dev, "/link", "etc/conf", 0o777, 0, 0)
        .unwrap();

    // /alias hardlinks /readme. Promotion moves /readme into the
    // HFS+ private-data directory as iNode<N>; both names then become
    // hlnk indirect-node entries pointing at that iNode.
    hfs.create_hardlink(&mut dev, "/readme", "/alias").unwrap();

    hfs.flush(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);

    assert_fsck_clean(&fsck, label, tmp.path());
}

/// Format with the journal stub enabled, then ensure `fsck.hfs(-plus)`
/// still considers the volume clean — proves our journal info block +
/// header are coherent enough for the tool to accept them.
#[test]
fn writer_journaled_image_passes_fsck_hfs() {
    let Some((fsck, label)) = find_fsck_hfs() else {
        eprintln!("skipping: fsck.hfs / fsck.hfsplus not installed");
        return;
    };

    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts {
        volume_name: "FstoolJrnl".into(),
        journaled: true,
        ..FormatOpts::default()
    };
    let (mut dev, mut hfs) = fresh_image(&tmp, &opts);

    // Sprinkle a single regular file so the catalog has something to
    // checksum beyond the root thread record.
    let body = b"journaled hello\n";
    let mut src = Cursor::new(&body[..]);
    hfs.create_file(
        &mut dev,
        "/hello.txt",
        &mut src,
        body.len() as u64,
        0o644,
        0,
        0,
    )
    .unwrap();

    hfs.flush(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);

    assert_fsck_clean(&fsck, label, tmp.path());
}

/// If a native HFS+ formatter is available, format with it and make
/// sure fstool's reader can mount the result + sees an empty root.
/// This is opportunistic — skipped silently when no tool is present.
#[test]
fn newfs_hfsplus_image_opens_via_fstool() {
    let Some(newfs) = which("newfs_hfsplus").or_else(|| which("newfs_hfs")) else {
        eprintln!("skipping: newfs_hfsplus / newfs_hfs not installed");
        return;
    };

    let tmp = NamedTempFile::new().unwrap();
    // newfs needs the file to exist with the right length first; it
    // doesn't grow the backing file itself.
    std::fs::File::create(tmp.path())
        .and_then(|f| f.set_len(VOL_BYTES))
        .unwrap();

    // -v sets the volume label. Both newfs_hfsplus (Linux/hfsprogs)
    // and newfs_hfs (macOS) accept it.
    let out = Command::new(&newfs)
        .arg("-v")
        .arg("ExtHFS")
        .arg(tmp.path())
        .output()
        .unwrap();
    if !out.status.success() {
        // Some hfsprogs builds refuse to operate on plain files. Skip
        // rather than fail the test — we have no way to provide a
        // loop device from inside `cargo test`.
        eprintln!(
            "skipping: {} refused to format image: {}",
            newfs.display(),
            String::from_utf8_lossy(&out.stderr)
        );
        return;
    }

    let mut dev = FileBackend::open(tmp.path()).unwrap();
    let hfs = HfsPlus::open(&mut dev).expect("fstool failed to open newfs image");
    let entries = hfs.list_path(&mut dev, "/").unwrap();
    assert!(
        entries.is_empty(),
        "freshly-formatted root should be empty, got: {entries:?}"
    );
}
