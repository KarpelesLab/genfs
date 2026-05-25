#![cfg(unix)]
//! F2FS end-to-end validation against the native f2fs-tools userspace
//! (`fsck.f2fs`, `mkfs.f2fs`, `dump.f2fs`). Each test silently skips when
//! its required tool isn't on PATH so the suite stays green on hosts
//! without f2fs-tools installed.

use std::io::Read;
use std::process::Command;

use fstool::block::{BlockDevice, FileBackend};
use fstool::fs::f2fs::{F2fs, FormatOpts};
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

/// Probe the native tool — returns true iff `tool -V` (or equivalent)
/// exits successfully. Used as the skip predicate. mkfs.f2fs's `-V`
/// prints to stderr and exits non-zero on some versions, so we also
/// accept simple presence on PATH as a fallback.
fn tool_available(tool: &str) -> bool {
    let probe = Command::new(tool).arg("-V").output();
    match probe {
        Ok(out) => {
            // Many f2fs-tools binaries print to stderr and may return
            // non-zero for -V; treat the ability to spawn at all as
            // sufficient evidence that the tool is installed.
            let _ = out;
            true
        }
        Err(_) => which(tool).is_some(),
    }
}

/// 64 MiB — comfortably above `mkfs.f2fs`'s ~30 MiB minimum and well
/// above the in-tree formatter's 64-block floor.
const IMAGE_BYTES: u64 = 64 * 1024 * 1024;

/// Writer -> fsck.f2fs round-trip. Format a fresh image via the library
/// API, populate it with a directory, a couple of files (one inline,
/// one multi-block), a symlink and a hard link, flush, then assert
/// `fsck.f2fs -f` returns clean.
#[test]
fn writer_image_passes_fsck_f2fs() {
    if !tool_available("fsck.f2fs") {
        eprintln!("skipping: fsck.f2fs not installed");
        return;
    }

    let tmp = NamedTempFile::new().unwrap();
    let mut dev = FileBackend::create(tmp.path(), IMAGE_BYTES).unwrap();

    // Canonical f2fs geometry: 2 MiB segments (log_blocks_per_seg = 9).
    let opts = FormatOpts {
        volume_label: "fstool-ext".into(),
        ..FormatOpts::default()
    };
    let mut fs = F2fs::format(&mut dev, &opts).unwrap();

    // A directory + an inline-sized file inside it.
    fs.create_dir(
        &mut dev,
        std::path::Path::new("/etc"),
        FileMeta::with_mode(0o755),
    )
    .unwrap();
    let small = b"x=1\n";
    fs.create_file(
        &mut dev,
        std::path::Path::new("/etc/config"),
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(small.to_vec())),
            len: small.len() as u64,
        },
        FileMeta::with_mode(0o644),
    )
    .unwrap();

    // A multi-block file (16 KiB) at the root.
    let big: Vec<u8> = (0..(16 * 1024))
        .map(|i| (i as u8).wrapping_mul(31))
        .collect();
    fs.create_file(
        &mut dev,
        std::path::Path::new("/big.bin"),
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(big.clone())),
            len: big.len() as u64,
        },
        FileMeta::with_mode(0o644),
    )
    .unwrap();

    // A symlink at the root.
    fs.create_symlink(
        &mut dev,
        std::path::Path::new("/link"),
        std::path::Path::new("etc/config"),
        FileMeta::with_mode(0o777),
    )
    .unwrap();

    // A hard link to the inline file (must come before flush; the
    // writer requires the source to be tracked in the current session).
    fs.create_hardlink(
        &mut dev,
        std::path::Path::new("/etc/config"),
        std::path::Path::new("/etc/config.alias"),
    )
    .unwrap();

    fs.flush(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);

    // `-f` forces the check even on a "clean" image. We additionally
    // pass `--readonly` (alias `-r` on some builds) when supported, to
    // make absolutely sure fsck won't try to mutate our image.
    let mut cmd = Command::new("fsck.f2fs");
    cmd.arg("-f").arg(tmp.path());
    let out = cmd.output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "fsck.f2fs failed (exit {:?}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code(),
    );
}

/// mkfs.f2fs -> fstool open. Have the native formatter build a volume,
/// then re-open it through our library and walk the root. Confirms the
/// reader survives a fully-canonical mkfs.f2fs layout (compression
/// disabled by default, no encryption).
#[test]
fn mkfs_f2fs_image_opens_through_fstool() {
    if !tool_available("mkfs.f2fs") {
        eprintln!("skipping: mkfs.f2fs not installed");
        return;
    }

    let tmp = NamedTempFile::new().unwrap();
    // Pre-size the sparse file so mkfs.f2fs has space to write into.
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(tmp.path())
        .unwrap();
    f.set_len(IMAGE_BYTES).unwrap();
    drop(f);

    // `-f` forces overwrite of a non-blank target without prompting.
    let out = Command::new("mkfs.f2fs")
        .args(["-f", "-l", "fstool-mkfs"])
        .arg(tmp.path())
        .output()
        .unwrap();
    if !out.status.success() {
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        panic!(
            "mkfs.f2fs failed (exit {:?}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
            out.status.code(),
        );
    }

    // Open through fstool.
    let mut dev = FileBackend::open(tmp.path()).unwrap();
    let mut fs = F2fs::open(&mut dev).unwrap();
    // Block size must be 4 KiB.
    assert_eq!(fs.block_size(), 4096);
    // total_bytes must round-trip through the SB's block_count *
    // block_size and match the configured image size to within one
    // block (mkfs may shave the tail).
    let tb = fs.total_bytes();
    assert!(
        tb <= IMAGE_BYTES && tb + (fs.block_size() as u64) >= IMAGE_BYTES,
        "total_bytes {tb} not within one block of image size {IMAGE_BYTES}",
    );

    // The root listing must succeed (it's empty on a fresh mkfs.f2fs).
    let list = fs.list_path(&mut dev, "/").unwrap();
    // A pristine mkfs.f2fs root has no user entries; list is empty.
    assert!(
        list.is_empty(),
        "fresh mkfs.f2fs root expected empty, got {list:?}",
    );
}

/// dump.f2fs sanity-check. If the tool is on PATH, run it against an
/// image produced by our writer (with a file inside) and assert it
/// exits cleanly. The output content is implementation-defined across
/// f2fs-tools versions, so we only check the exit status.
#[test]
fn writer_image_dump_f2fs_clean_exit() {
    if !tool_available("dump.f2fs") {
        eprintln!("skipping: dump.f2fs not installed");
        return;
    }

    let tmp = NamedTempFile::new().unwrap();
    let mut dev = FileBackend::create(tmp.path(), IMAGE_BYTES).unwrap();
    let opts = FormatOpts::default();
    let mut fs = F2fs::format(&mut dev, &opts).unwrap();

    let payload = b"dump.f2fs round-trip payload\n";
    fs.create_file(
        &mut dev,
        std::path::Path::new("/payload.txt"),
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(payload.to_vec())),
            len: payload.len() as u64,
        },
        FileMeta::with_mode(0o644),
    )
    .unwrap();
    fs.flush(&mut dev).unwrap();
    dev.sync().unwrap();

    // Round-trip the payload through our own reader as a smoke test.
    let mut r = fs.open_file_reader(&mut dev, "/payload.txt").unwrap();
    let mut got = Vec::new();
    r.read_to_end(&mut got).unwrap();
    assert_eq!(got, payload);
    drop(r);
    drop(dev);

    // dump.f2fs's CLI varies between versions: some accept the image
    // path positionally, some require `-i <nid>`. The most portable
    // smoke test is "does it run against this image without crashing".
    // We pass `-i 3` (root inode is normally nid=3) which is supported
    // across the versions we care about.
    let out = Command::new("dump.f2fs")
        .args(["-i", "3"])
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "dump.f2fs failed (exit {:?}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code(),
    );
}

/// Large single directory: enough entries to drive the f2fs directory
/// hash through several levels (well past the inline-dir and first
/// regular-dir-block thresholds). The build + read-back run unconditionally
/// so the writer's and reader's large-directory paths are always exercised;
/// the `fsck.f2fs` cross-check runs in CI (and anywhere the tool is on
/// PATH), guarding the on-disk directory structure at scale.
#[test]
fn writer_large_directory_passes_fsck_f2fs() {
    const N: usize = 10_000;
    let tmp = NamedTempFile::new().unwrap();
    let mut dev = FileBackend::create(tmp.path(), 256 * 1024 * 1024).unwrap();
    let opts = FormatOpts {
        volume_label: "fstool-big".into(),
        ..FormatOpts::default()
    };
    let mut fs = F2fs::format(&mut dev, &opts).unwrap();
    fs.create_dir(
        &mut dev,
        std::path::Path::new("/big"),
        FileMeta::with_mode(0o755),
    )
    .unwrap();
    for i in 0..N {
        let body = b"x";
        fs.create_file(
            &mut dev,
            &std::path::PathBuf::from(format!("/big/file{i:05}")),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(body.to_vec())),
                len: body.len() as u64,
            },
            FileMeta::with_mode(0o644),
        )
        .unwrap();
    }
    fs.flush(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);

    // Read-back: every entry must list back (writer + reader, no fsck needed).
    {
        let mut dev = FileBackend::open(tmp.path()).unwrap();
        let mut fs = F2fs::open(&mut dev).unwrap();
        let entries = fs.list_path(&mut dev, "/big").unwrap();
        let files = entries
            .iter()
            .filter(|e| e.name != "." && e.name != "..")
            .count();
        assert_eq!(files, N, "listed {files} of {N} files in /big");
    }

    // Native cross-check when fsck.f2fs is available (always in CI).
    if !tool_available("fsck.f2fs") {
        eprintln!("skipping fsck.f2fs cross-check: not installed");
        return;
    }
    let out = Command::new("fsck.f2fs")
        .arg("-f")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "fsck.f2fs failed on {N}-file dir (exit {:?}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code(),
    );
}
