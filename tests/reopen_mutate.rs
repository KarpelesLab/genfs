#![cfg(unix)]
//! Cross-backend reopen-mutate sweep.
//!
//! The consumer contract under test: *create an image, drop the handle,
//! then reopen it through the generic [`fstool::inspect::open`] entry
//! point and keep writing* — i.e. use the filesystem as a real
//! read/write store (e.g. inside a qcow2). This is the path a caller hits
//! when they don't keep the post-`format` handle alive.
//!
//! For each writable backend we:
//!   1. format + seed `/seed.txt` (backend-specific opts), flush, drop;
//!   2. reopen via `inspect::open` (no `format`), add `/added.txt`, flush,
//!      drop — this is the step that broke on NTFS (writer state was only
//!      built by `format`) until lazy reconstruction landed;
//!   3. reopen a third time and confirm BOTH files are present with the
//!      right contents (the seed survived, the addition persisted);
//!   4. cross-validate with the native checker when one is installed.
//!
//! `e2fsck` / `fsck.fat` / `xfs_repair` / `fsck.hfsplus` are used where
//! present; backends with no checker on this box fall back to the
//! fstool-vs-fstool self-check from step 3. Each fsck step silently skips
//! when its tool isn't on `PATH`.
//!
//! F2FS is intentionally excluded from the positive sweep: its writer is
//! a whole-filesystem in-memory serializer (no incremental on-disk
//! mutation), so a reopened handle is read-only by design. That contract
//! is pinned by `f2fs_reopen_is_read_only` below.

use std::io::{Cursor, Read};
use std::path::Path;
use std::process::Command;

use fstool::block::{BlockDevice, FileBackend};
use fstool::fs::{FileMeta, FileSource, Filesystem};
use fstool::inspect;
use tempfile::NamedTempFile;

const SEED: &[u8] = b"seed body - present before reopen\n";
const ADDED: &[u8] = b"added through inspect::open after reopen\n";

fn which(tool: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {tool}"))
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn src(body: &[u8]) -> FileSource {
    FileSource::Reader {
        reader: Box::new(Cursor::new(body.to_vec())),
        len: body.len() as u64,
    }
}

fn read_file(fs: &mut dyn Filesystem, dev: &mut dyn BlockDevice, path: &str) -> Vec<u8> {
    let mut r = fs.read_file(dev, Path::new(path)).unwrap();
    let mut v = Vec::new();
    r.read_to_end(&mut v).unwrap();
    v
}

fn root_names(fs: &mut dyn Filesystem, dev: &mut dyn BlockDevice) -> Vec<String> {
    let mut n: Vec<String> = fs
        .list(dev, Path::new("/"))
        .unwrap()
        .into_iter()
        .map(|e| e.name)
        .collect();
    n.sort();
    n
}

/// Phases 2 + 3 of the contract, generic over the backend: reopen through
/// `inspect::open`, add `/added.txt`, flush, then reopen once more and
/// assert both files survive with the right bytes.
fn reopen_add_then_verify(path: &Path) {
    // Phase 2: reopen (no format) and add a file.
    {
        let mut dev = FileBackend::open(path).unwrap();
        let mut fs = inspect::open(&mut dev).unwrap();
        fs.create_file(
            &mut dev,
            Path::new("/added.txt"),
            src(ADDED),
            FileMeta::default(),
        )
        .expect("reopened handle must accept a new file");
        fs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }
    // Phase 3: reopen again and verify.
    {
        let mut dev = FileBackend::open(path).unwrap();
        let mut fs = inspect::open(&mut dev).unwrap();
        let names = root_names(&mut *fs, &mut dev);
        assert!(
            names.iter().any(|n| n == "added.txt"),
            "added file missing after reopen: {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "seed.txt"),
            "pre-existing seed clobbered by reopen-mutate: {names:?}"
        );
        assert_eq!(
            read_file(&mut *fs, &mut dev, "/added.txt"),
            ADDED,
            "added file content mismatch"
        );
        assert_eq!(
            read_file(&mut *fs, &mut dev, "/seed.txt"),
            SEED,
            "seed file content corrupted by reopen-mutate"
        );
    }
}

// ----------------------------------------------------------------------
// ext4
// ----------------------------------------------------------------------
#[test]
fn ext4_reopen_mutate() {
    use fstool::fs::ext::{Ext, FormatOpts, FsKind};
    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts {
        kind: FsKind::Ext4,
        blocks_count: 16384,
        inodes_count: 256,
        journal_blocks: 1024,
        sparse: true,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    {
        let mut dev = FileBackend::create(tmp.path(), size).unwrap();
        let mut fs: Box<dyn Filesystem> = Box::new(Ext::format_with(&mut dev, &opts).unwrap());
        fs.create_file(
            &mut dev,
            Path::new("/seed.txt"),
            src(SEED),
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }
    reopen_add_then_verify(tmp.path());

    if which("e2fsck") {
        let out = Command::new("e2fsck")
            .args(["-fn"])
            .arg(tmp.path())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "e2fsck not clean after reopen-mutate:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    } else {
        eprintln!("skipping e2fsck oracle: not installed");
    }
}

// ----------------------------------------------------------------------
// FAT32
// ----------------------------------------------------------------------
#[test]
fn fat32_reopen_mutate() {
    use fstool::fs::fat::{Fat32, FatFormatOpts};
    let tmp = NamedTempFile::new().unwrap();
    let total_sectors = 64 * 1024 * 1024 / 512; // 64 MiB
    {
        let mut dev = FileBackend::create(tmp.path(), total_sectors as u64 * 512).unwrap();
        let opts = FatFormatOpts {
            total_sectors,
            volume_id: 0xCAFE_F00D,
            volume_label: *b"FSTOOL     ",
        };
        let mut fs: Box<dyn Filesystem> = Box::new(Fat32::format(&mut dev, &opts).unwrap());
        fs.create_file(
            &mut dev,
            Path::new("/seed.txt"),
            src(SEED),
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }
    reopen_add_then_verify(tmp.path());

    if which("fsck.fat") {
        let out = Command::new("fsck.fat")
            .arg("-n")
            .arg(tmp.path())
            .output()
            .unwrap();
        // fsck.fat exits 0 when clean; -n answers "no" to any repair prompt.
        assert!(
            out.status.success(),
            "fsck.fat not clean after reopen-mutate:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    } else {
        eprintln!("skipping fsck.fat oracle: not installed");
    }
}

// ----------------------------------------------------------------------
// exFAT (no fsck.exfat here → self-check only)
// ----------------------------------------------------------------------
#[test]
fn exfat_reopen_mutate() {
    use fstool::fs::exfat::Exfat;
    use fstool::fs::exfat::format::FormatOpts;
    let tmp = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(tmp.path(), 64 * 1024 * 1024).unwrap();
        let opts = FormatOpts {
            bytes_per_sector_shift: 9,
            sectors_per_cluster_shift: 3,
            volume_serial_number: 0xCAFE_F00D,
            volume_label: "FSTOOL".to_string(),
        };
        let mut fs: Box<dyn Filesystem> = Box::new(Exfat::format(&mut dev, &opts).unwrap());
        fs.create_file(
            &mut dev,
            Path::new("/seed.txt"),
            src(SEED),
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }
    reopen_add_then_verify(tmp.path());

    if which("fsck.exfat") {
        let out = Command::new("fsck.exfat")
            .arg("-n")
            .arg(tmp.path())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "fsck.exfat not clean after reopen-mutate:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    } else {
        eprintln!("skipping fsck.exfat oracle: not installed (self-check only)");
    }
}

// ----------------------------------------------------------------------
// XFS (resume_writes reconstructs on first write)
// ----------------------------------------------------------------------
#[test]
fn xfs_reopen_mutate() {
    use fstool::fs::xfs::{self, FormatOpts};
    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts {
        uuid: [0x42u8; 16],
        ..Default::default()
    };
    {
        let mut dev = FileBackend::create(tmp.path(), 64 * 1024 * 1024).unwrap();
        let mut x = xfs::format(&mut dev, &opts).unwrap();
        x.begin_writes(opts.uuid);
        let mut fs: Box<dyn Filesystem> = Box::new(x);
        fs.create_file(
            &mut dev,
            Path::new("/seed.txt"),
            src(SEED),
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }
    reopen_add_then_verify(tmp.path());

    if which("xfs_repair") {
        let out = Command::new("xfs_repair")
            .args(["-n", "-o", "force_geometry"])
            .arg(tmp.path())
            .output()
            .unwrap();
        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert_ne!(
            out.status.code(),
            Some(2),
            "xfs_repair reports dirty log after reopen-mutate:\n{combined}"
        );
        assert!(
            combined.contains("No modify flag set"),
            "xfs_repair did not run to completion after reopen-mutate:\n{combined}"
        );
    } else {
        eprintln!("skipping xfs_repair oracle: not installed");
    }
}

// ----------------------------------------------------------------------
// HFS+ (open() reconstructs writer state)
// ----------------------------------------------------------------------
#[test]
fn hfs_plus_reopen_mutate() {
    use fstool::fs::hfs_plus::{FormatOpts, HfsPlus};
    let tmp = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(tmp.path(), 16 * 1024 * 1024).unwrap();
        let opts = FormatOpts {
            volume_name: "FstoolHFS".into(),
            ..FormatOpts::default()
        };
        let mut fs: Box<dyn Filesystem> = Box::new(HfsPlus::format(&mut dev, &opts).unwrap());
        fs.create_file(
            &mut dev,
            Path::new("/seed.txt"),
            src(SEED),
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }
    reopen_add_then_verify(tmp.path());

    if which("fsck.hfsplus") {
        let out = Command::new("fsck.hfsplus")
            .args(["-n"])
            .arg(tmp.path())
            .output()
            .unwrap();
        // fsck.hfsplus exit codes vary; require it didn't report it could
        // not be repaired (exit 8) and that it ran.
        assert_ne!(
            out.status.code(),
            Some(8),
            "fsck.hfsplus reports unrepairable volume after reopen-mutate:\n{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    } else {
        eprintln!("skipping fsck.hfsplus oracle: not installed");
    }
}

// ----------------------------------------------------------------------
// F2FS — build-once contract.
//
// F2FS's writer accumulates the whole filesystem in memory and serializes
// it from scratch at flush (NAT both halves, fresh-image cursegs), so it
// can't take incremental mutations on a reopened image. Reopening yields a
// read-only handle; a write attempt must fail cleanly (Unsupported), not
// corrupt the image. This test pins that contract so a future change that
// silently breaks it (or quietly "succeeds" and corrupts) is caught.
// ----------------------------------------------------------------------
#[test]
fn f2fs_reopen_is_read_only() {
    use fstool::fs::f2fs::{F2fs, FormatOpts};
    let tmp = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(tmp.path(), 64 * 1024 * 1024).unwrap();
        let opts = FormatOpts::default();
        let mut fs: Box<dyn Filesystem> = Box::new(F2fs::format(&mut dev, &opts).unwrap());
        fs.create_file(
            &mut dev,
            Path::new("/seed.txt"),
            src(SEED),
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }

    // Reopen and confirm the seed is readable (read path works fine).
    {
        let mut dev = FileBackend::open(tmp.path()).unwrap();
        let mut fs = inspect::open(&mut dev).unwrap();
        assert_eq!(read_file(&mut *fs, &mut dev, "/seed.txt"), SEED);

        // A direct trait write on the reopened (read-only) handle must
        // fail cleanly with Unsupported — not corrupt the image.
        let err = fs
            .create_file(
                &mut dev,
                Path::new("/added.txt"),
                src(ADDED),
                FileMeta::default(),
            )
            .expect_err("f2fs reopen-write must be rejected (build-once backend)");
        assert!(
            matches!(err, fstool::Error::Unsupported(_)),
            "expected Unsupported on f2fs reopen-write, got: {err:?}"
        );
    }

    // The generic AnyFs guard should reject `add` up front with a typed
    // Immutable error (writer-aware mutation_capability), rather than
    // letting the call descend into the writer.
    {
        let host = NamedTempFile::new().unwrap();
        std::fs::write(host.path(), ADDED).unwrap();
        let mut dev = FileBackend::open(tmp.path()).unwrap();
        let mut any = inspect::AnyFs::open(&mut dev).unwrap();
        let err = any
            .add_file(&mut dev, "/added.txt", host.path())
            .expect_err("AnyFs::add on a reopened f2fs image must be rejected");
        assert!(
            matches!(err, fstool::Error::Immutable { .. }),
            "expected typed Immutable error from the AnyFs guard, got: {err:?}"
        );
    }
}
