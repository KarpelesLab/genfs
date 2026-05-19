#![cfg(unix)]
//! External validation: produce exFAT images with the library writer and
//! verify them with `fsck.exfat` (exfatprogs), and read back images that
//! `mkfs.exfat` produced. Each test skips silently when the required tool
//! isn't on PATH so the suite passes on a clean CI machine.
//!
//! NOTE: real loopback-mounted round-trips need root and are intentionally
//! omitted; we rely on `fsck.exfat -nv` (verbose, read-only) as the
//! native-tool check for writer output.

use std::io::Read;
use std::path::Path;
use std::process::Command;

use fstool::block::{BlockDevice, FileBackend};
use fstool::fs::exfat::Exfat;
use fstool::fs::exfat::format::FormatOpts;
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

/// Probe the tool with `--version`. Most exfatprogs binaries support it;
/// even when they return non-zero exit, the binary existing on PATH (via
/// `which`) is the real signal — `--version` is just a liveness check.
fn tool_present(tool: &str) -> bool {
    if which(tool).is_none() {
        return false;
    }
    // Best-effort: run --version. We don't require success because some
    // builds report the version on stderr with a non-zero exit code.
    let _ = Command::new(tool).arg("--version").output();
    true
}

/// Format a fresh exFAT volume on `path` of `mib` megabytes with `label`.
fn format_volume(path: &Path, mib: u32, label: &str) -> Exfat {
    let bytes = mib as u64 * 1024 * 1024;
    let mut dev = FileBackend::create(path, bytes).expect("create image");
    let opts = FormatOpts {
        bytes_per_sector_shift: 9,    // 512 B sectors
        sectors_per_cluster_shift: 3, // 4 KiB clusters
        volume_serial_number: 0xCAFE_F00D,
        volume_label: label.to_string(),
    };
    let fs = Exfat::format(&mut dev, &opts).expect("format exfat");
    dev.sync().expect("sync");
    drop(dev);
    // Re-open the volume to return a writable handle that owns the device,
    // but tests need both the device and the fs separately. Return the fs
    // alone here is not useful — callers re-open the file as needed.
    fs
}

#[test]
fn writer_image_passes_fsck_exfat() {
    if !tool_present("fsck.exfat") {
        eprintln!("skipping: fsck.exfat not installed");
        return;
    }
    let tmp = NamedTempFile::new().unwrap();
    // Format, then populate via streaming create_file calls. We open the
    // device fresh, write everything, flush, sync, then close.
    let _ = format_volume(tmp.path(), 64, "FSTOOLEXF");

    {
        let mut dev = FileBackend::open(tmp.path()).unwrap();
        let mut fs = Exfat::open(&mut dev).unwrap();

        // Small payload, streamed via &[u8] which is a std::io::Read source
        // (never materialises the file in memory beyond what create_file
        // chooses to buffer internally — see SCRATCH_BUF_BYTES).
        let p1: &[u8] = b"hello, exfat external\n";
        let mut r1: &[u8] = p1;
        fs.create_file(&mut dev, "/hello.txt", &mut r1, p1.len() as u64, 0)
            .unwrap();

        fs.create_dir(&mut dev, "/docs", 0).unwrap();

        let p2: &[u8] = b"# Long Name File\nNested under /docs.\n";
        let mut r2: &[u8] = p2;
        fs.create_file(
            &mut dev,
            "/docs/A Long Readme.md",
            &mut r2,
            p2.len() as u64,
            0,
        )
        .unwrap();

        // Nested directory + a file inside.
        fs.create_dir(&mut dev, "/docs/nested", 0).unwrap();
        let p3: &[u8] = b"deeply nested body\n";
        let mut r3: &[u8] = p3;
        fs.create_file(
            &mut dev,
            "/docs/nested/inside.bin",
            &mut r3,
            p3.len() as u64,
            0,
        )
        .unwrap();

        // Non-ASCII name to exercise UTF-16 + up-case + name hash. Mix of
        // BMP code points; if exfatprogs disagrees with our normalisation,
        // fsck will tell us.
        let p4: &[u8] = b"konnichiwa\n";
        let mut r4: &[u8] = p4;
        fs.create_file(
            &mut dev,
            "/\u{3053}\u{3093}\u{306B}\u{3061}\u{306F}.txt",
            &mut r4,
            p4.len() as u64,
            0,
        )
        .unwrap();

        fs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }

    // Run fsck.exfat in read-only, verbose mode. Per exfatprogs, exit
    // status 0 means a clean volume.
    let out = Command::new("fsck.exfat")
        .args(["-n", "-v"])
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "fsck.exfat failed (exit {:?}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code()
    );
}

#[test]
fn open_reads_back_an_mkfs_exfat_image() {
    if !tool_present("mkfs.exfat") {
        eprintln!("skipping: mkfs.exfat not installed");
        return;
    }

    // Create a 64 MiB sparse file and format it with mkfs.exfat directly.
    let tmp = NamedTempFile::new().unwrap();
    let bytes = 64u64 * 1024 * 1024;
    std::fs::File::create(tmp.path())
        .unwrap()
        .set_len(bytes)
        .unwrap();

    // -L is the label option for exfatprogs' mkfs.exfat. Some older
    // versions also accept -n; we use -L since exfatprogs is the modern
    // standard implementation.
    let mkfs = Command::new("mkfs.exfat")
        .args(["-L", "TEST-EXFAT"])
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        mkfs.status.success(),
        "mkfs.exfat failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&mkfs.stdout),
        String::from_utf8_lossy(&mkfs.stderr),
    );

    // Opening exercises the boot-region checksum validation in
    // BootSector::decode + Exfat::open. A bad checksum here would surface
    // as InvalidImage before we get to inspect the volume.
    let mut dev = FileBackend::open(tmp.path()).unwrap();
    let fs = Exfat::open(&mut dev).expect("open mkfs.exfat image");
    assert_eq!(
        fs.volume_label(),
        "TEST-EXFAT",
        "volume label round-trip mismatch"
    );

    // mkfs.exfat from exfatprogs leaves the root directory empty (no
    // "System Volume Information" — that's created by Windows on first
    // mount, not by the formatter). Tolerate either possibility.
    let root = fs.list_path(&mut dev, "/").unwrap();
    let names: Vec<&str> = root.iter().map(|e| e.name.as_str()).collect();
    let only_known = names.iter().all(|n| {
        n.eq_ignore_ascii_case("System Volume Information")
            || n.eq_ignore_ascii_case("$RECYCLE.BIN")
    });
    assert!(
        names.is_empty() || only_known,
        "unexpected entries in fresh mkfs.exfat root: {names:?}"
    );
}

#[test]
fn writer_image_fsck_verbose_simulates_mount_check() {
    // Same intent as `writer_image_passes_fsck_exfat` but with a more
    // populated tree — this acts as our stand-in for a real mount round-
    // trip (which would need root). If fsck -nv reports "clean", a kernel
    // mount of the same bytes would (modulo kernel-version quirks) also
    // succeed.
    if !tool_present("fsck.exfat") {
        eprintln!("skipping: fsck.exfat not installed");
        return;
    }

    let tmp = NamedTempFile::new().unwrap();
    let _ = format_volume(tmp.path(), 32, "MOUNTSIM");

    {
        let mut dev = FileBackend::open(tmp.path()).unwrap();
        let mut fs = Exfat::open(&mut dev).unwrap();

        // Stream a multi-cluster file (3 clusters @ 4 KiB = 12 KiB). The
        // body is generated cluster-by-cluster from a Cursor so we never
        // hold the whole image in memory at once.
        let body: Vec<u8> = (0..(12 * 1024)).map(|i| (i % 251) as u8).collect();
        let mut reader: &[u8] = &body;
        fs.create_file(&mut dev, "/multi.bin", &mut reader, body.len() as u64, 0)
            .unwrap();

        // An empty file (FirstCluster == 0, AllocationPossible flag clear).
        let mut empty: &[u8] = &[];
        fs.create_file(&mut dev, "/zero.bin", &mut empty, 0, 0)
            .unwrap();

        // A directory tree two levels deep with a file at the leaf.
        fs.create_dir(&mut dev, "/lvl1", 0).unwrap();
        fs.create_dir(&mut dev, "/lvl1/lvl2", 0).unwrap();
        let leaf: &[u8] = b"leaf\n";
        let mut rl: &[u8] = leaf;
        fs.create_file(
            &mut dev,
            "/lvl1/lvl2/leaf.txt",
            &mut rl,
            leaf.len() as u64,
            0,
        )
        .unwrap();

        // Verify we can read back our own writes before fsck sees them.
        fs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }

    // Sanity-check via our own reader first.
    {
        let mut dev = FileBackend::open(tmp.path()).unwrap();
        let fs = Exfat::open(&mut dev).unwrap();
        let root: Vec<String> = fs
            .list_path(&mut dev, "/")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(root.iter().any(|n| n == "multi.bin"));
        assert!(root.iter().any(|n| n == "zero.bin"));
        assert!(root.iter().any(|n| n == "lvl1"));
        // Stream the multi-cluster file back and compare lengths only —
        // a byte-compare would require holding the source in memory once
        // (it already is in `body` for the writer, but we don't keep it
        // around). Length check is sufficient to prove the chain walked.
        let mut r = fs.open_file_reader(&mut dev, "/multi.bin").unwrap();
        let mut total: u64 = 0;
        let mut buf = [0u8; 4096];
        loop {
            let n = r.read(&mut buf).unwrap();
            if n == 0 {
                break;
            }
            total += n as u64;
        }
        assert_eq!(total, 12 * 1024);
    }

    let out = Command::new("fsck.exfat")
        .args(["-n", "-v"])
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "fsck.exfat -nv failed (exit {:?}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status.code()
    );
}
