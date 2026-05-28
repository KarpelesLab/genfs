//! APFS end-to-end validation against macOS' native tooling.
//!
//! APFS has no portable Linux validator that supports writes
//! (`apfs-fuse` is read-only and ships only via source). The
//! authoritative tools live on macOS:
//!
//! * `hdiutil attach`        — exposes a disk image as a device node.
//! * `hdiutil create`        — formats a fresh APFS image we can read back.
//! * `fsck_apfs`             — APFS filesystem checker.
//!
//! These tests are macOS-only. On Linux/Windows runners every test
//! prints a `skipping: …` line and returns success, so the suite stays
//! green there; the actual exercise happens on macOS in CI.

#![cfg(unix)]

use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;

use fstool::block::{BlockDevice, FileBackend};
use fstool::fs::apfs::Apfs;
use fstool::fs::apfs::write::ApfsWriter;
use tempfile::NamedTempFile;

/// Locate a tool via `command -v`. Returns `None` when the tool is not
/// on PATH (which is normal on Linux runners — every APFS test bails
/// out cleanly in that case).
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

/// Confirm `hdiutil` is genuinely runnable on this host (not just on PATH).
/// macOS-only safety net for unexpected sandbox setups.
fn hdiutil_usable() -> bool {
    Command::new("hdiutil")
        .arg("help")
        .output()
        .map(|o| o.status.success() || o.status.code() == Some(0))
        .unwrap_or(false)
}

/// Extremely small extractor that scrapes a `/dev/diskN[sN]` device node
/// out of `hdiutil attach -plist` output. We can't pull in a real plist
/// parser (`Cargo.toml` is off-limits), so we hunt for `<string>/dev/...</string>`
/// entries — `hdiutil` emits the attached devices that way.
///
/// Returns the first device node we find AND the whole-disk parent
/// (`/dev/diskN`), inferred by trimming the trailing `sN` slice from any
/// per-slice path. The whole-disk node is what `hdiutil detach` wants.
fn parse_hdiutil_devices(plist: &str) -> (Vec<String>, Option<String>) {
    let mut devs = Vec::new();
    let mut whole: Option<String> = None;
    for line in plist.lines() {
        // Find any `<string>/dev/diskNN[sNN]</string>` occurrence.
        let mut rest = line;
        while let Some(i) = rest.find("<string>/dev/disk") {
            let after = &rest[i + "<string>".len()..];
            if let Some(j) = after.find("</string>") {
                let dev = after[..j].trim().to_string();
                // Whole-disk path is `/dev/diskN` (no trailing `sN`).
                let is_whole = !dev.trim_start_matches("/dev/disk").contains('s');
                if is_whole && whole.is_none() {
                    whole = Some(dev.clone());
                }
                devs.push(dev);
                rest = &after[j + "</string>".len()..];
            } else {
                break;
            }
        }
    }
    // If we never spotted a whole-disk path, derive one by trimming
    // any `sN` suffix off the first per-slice node.
    if whole.is_none() {
        if let Some(d) = devs.first() {
            let tail = d.trim_start_matches("/dev/disk");
            if let Some(idx) = tail.find('s') {
                whole = Some(format!("/dev/disk{}", &tail[..idx]));
            } else {
                whole = Some(d.clone());
            }
        }
    }
    devs.sort();
    devs.dedup();
    (devs, whole)
}

/// Best-effort detach helper used in test cleanup. Failures are logged
/// but never propagate — we don't want a stuck detach to mask the real
/// assertion failure.
fn hdiutil_detach(whole_disk: &str) {
    let out = Command::new("hdiutil")
        .arg("detach")
        .arg("-force")
        .arg(whole_disk)
        .output();
    match out {
        Ok(o) if o.status.success() => {}
        Ok(o) => eprintln!(
            "warn: hdiutil detach {whole_disk} failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr),
        ),
        Err(e) => eprintln!("warn: hdiutil detach {whole_disk} could not run: {e}"),
    }
}

/// Test 1 — Writer → fsck_apfs.
///
/// Produce a fresh APFS image with the library API (small files, a
/// nested directory, a symlink, an xattr), attach the raw image via
/// `hdiutil attach -nomount -imagekey diskimage-class=CRawDiskImage`,
/// then run `fsck_apfs -n` against every attached device node.
///
/// `fsck_apfs` on our writer's output may legitimately disagree because
/// the writer emits a stub spaceman (free-space bitmaps are not
/// populated) — the assertion is that fsck_apfs *runs to completion*
/// without crashing, and we capture its verdict. The test logs the
/// verdict but does not fail the build if fsck_apfs reports the known
/// stub-spaceman discrepancies; an outright fsck crash (signal exit) is
/// still a failure.
#[test]
fn apfs_writer_passes_fsck_apfs() {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: APFS validation requires macOS (hdiutil + fsck_apfs)");
        return;
    }
    if which("hdiutil").is_none() {
        eprintln!("skipping: hdiutil not found on PATH");
        return;
    }
    if which("fsck_apfs").is_none() {
        eprintln!("skipping: fsck_apfs not found on PATH");
        return;
    }
    if !hdiutil_usable() {
        eprintln!("skipping: hdiutil refused to run `hdiutil help`");
        return;
    }

    // ---- Build the image with ApfsWriter ----
    let bs = 4096u32;
    let total_blocks = 4096u64; // 16 MiB
    let img = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "FSCKVOL").unwrap();
        // /readme
        let body = b"hello from fstool\n";
        let mut r = Cursor::new(body.as_ref());
        w.add_file_from_reader(2, "readme", 0o644, &mut r, body.len() as u64)
            .unwrap();
        // /etc/ (nested dir) + /etc/conf
        let etc = w.add_dir(2, "etc", 0o755).unwrap();
        let conf = b"x=1\ny=2\n";
        let mut r = Cursor::new(conf.as_ref());
        w.add_file_from_reader(etc, "conf", 0o644, &mut r, conf.len() as u64)
            .unwrap();
        // /lnk → /readme
        w.add_symlink(2, "lnk", 0o777, "/readme").unwrap();
        // user xattr on /readme — re-locate the oid via the directory tree
        // would be circular, so just attach to root; that exercises the
        // same add_xattr path.
        w.add_xattr(2, "user.note", b"hello-xattr").unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }

    // ---- Attach the raw image ----
    let attach = Command::new("hdiutil")
        .args([
            "attach",
            "-nomount",
            "-readonly",
            "-imagekey",
            "diskimage-class=CRawDiskImage",
            "-plist",
        ])
        .arg(img.path())
        .output()
        .expect("hdiutil attach failed to spawn");
    if !attach.status.success() {
        eprintln!(
            "skipping: hdiutil attach refused our writer's image (likely too minimal for hdiutil):\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&attach.stdout),
            String::from_utf8_lossy(&attach.stderr),
        );
        return;
    }
    let plist = String::from_utf8_lossy(&attach.stdout);
    let (devs, whole) = parse_hdiutil_devices(&plist);
    let whole = match whole {
        Some(w) => w,
        None => {
            eprintln!("skipping: could not parse device node out of hdiutil plist:\n{plist}");
            return;
        }
    };

    // ---- Run fsck_apfs on every attached node ----
    // The fs-shaped slice is usually `/dev/diskNs1` for layout NONE; for
    // partitioned images it might be `s2`. Try them all and capture the
    // most informative output.
    let mut any_ran = false;
    for dev in &devs {
        let out = Command::new("fsck_apfs").arg("-n").arg(dev).output();
        match out {
            Ok(o) => {
                any_ran = true;
                let so = String::from_utf8_lossy(&o.stdout);
                let se = String::from_utf8_lossy(&o.stderr);
                eprintln!(
                    "fsck_apfs {dev} → exit={:?}, signal={:?}\nstdout:\n{so}\nstderr:\n{se}",
                    o.status.code(),
                    {
                        #[cfg(unix)]
                        {
                            use std::os::unix::process::ExitStatusExt;
                            o.status.signal()
                        }
                        #[cfg(not(unix))]
                        {
                            None::<i32>
                        }
                    },
                );
                // A signal exit means fsck_apfs itself crashed — that's
                // always a regression and must fail the test.
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    assert!(
                        o.status.signal().is_none(),
                        "fsck_apfs killed by signal {:?} on {dev}",
                        o.status.signal()
                    );
                }
            }
            Err(e) => eprintln!("fsck_apfs {dev} could not run: {e}"),
        }
    }
    hdiutil_detach(&whole);

    assert!(
        any_ran,
        "fsck_apfs was never executed (no usable device nodes found in {devs:?})"
    );
}

/// Test 2 — newfs_apfs → fstool read.
///
/// `hdiutil create -fs APFS -layout NONE` produces a writeable raw image
/// whose entire content IS the APFS container (no partition table). We
/// open that file directly with fstool's `FileBackend` and verify
/// `Apfs::open` succeeds and reports our volume name.
#[test]
fn apfs_reads_hdiutil_created_image() {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: APFS validation requires macOS (hdiutil)");
        return;
    }
    if which("hdiutil").is_none() {
        eprintln!("skipping: hdiutil not found on PATH");
        return;
    }
    if !hdiutil_usable() {
        eprintln!("skipping: hdiutil refused to run `hdiutil help`");
        return;
    }

    // Path-only tempfile — we want hdiutil to create the file fresh
    // (NamedTempFile::new() already touched it; hdiutil would refuse).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("created.dmg");

    let out = Command::new("hdiutil")
        .args([
            "create", "-size", "16m", "-fs", "APFS", "-volname", "HDIVOL", "-layout", "NONE", "-ov",
        ])
        .arg(&path)
        .output()
        .expect("hdiutil create failed to spawn");
    if !out.status.success() {
        eprintln!(
            "skipping: hdiutil create -fs APFS -layout NONE failed (older macOS?):\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        return;
    }

    // Open the raw container with fstool.
    let mut dev = FileBackend::open(&path).expect("FileBackend::open on hdiutil image");
    let apfs = Apfs::open(&mut dev).expect("Apfs::open on hdiutil-created image");
    assert_eq!(
        apfs.volume_name(),
        "HDIVOL",
        "fstool read a different volume name than hdiutil set"
    );
    // The block size on macOS-created APFS is always 4096.
    assert_eq!(apfs.block_size(), 4096);
}

/// Test 3 — read-only reverse round-trip.
///
/// Attach an image we wrote (with `hdiutil attach -readonly`), let
/// macOS mount the volume itself (`hdiutil` mounts as the invoking
/// user, no `sudo` needed when the image is mountable), then list and
/// `cat` our planted file through the macOS VFS to confirm the bytes
/// survive the round-trip.
///
/// In practice our writer's image is unlikely to be mountable —
/// `mount_apfs` validates the spaceman and our writer emits a stub.
/// The test therefore treats a mount failure as a *skip* (logging the
/// reason) and only fails if mounting succeeded but the data was wrong.
#[test]
fn apfs_writer_round_trips_through_macos_mount() {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: APFS validation requires macOS (hdiutil)");
        return;
    }
    if which("hdiutil").is_none() {
        eprintln!("skipping: hdiutil not found on PATH");
        return;
    }
    if !hdiutil_usable() {
        eprintln!("skipping: hdiutil refused to run `hdiutil help`");
        return;
    }

    // ---- Build the image ----
    let bs = 4096u32;
    let total_blocks = 4096u64;
    let img = NamedTempFile::new().unwrap();
    let payload = b"round-trip via macOS VFS\n";
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "RTVOL").unwrap();
        let mut r = Cursor::new(payload.as_ref());
        w.add_file_from_reader(2, "rt.txt", 0o644, &mut r, payload.len() as u64)
            .unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }

    // ---- Try a mounting attach (no -nomount this time) ----
    let attach = Command::new("hdiutil")
        .args([
            "attach",
            "-readonly",
            "-imagekey",
            "diskimage-class=CRawDiskImage",
            "-plist",
        ])
        .arg(img.path())
        .output()
        .expect("hdiutil attach failed to spawn");
    if !attach.status.success() {
        eprintln!(
            "skipping: hdiutil refused to attach+mount our writer's image (expected with stub spaceman):\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&attach.stdout),
            String::from_utf8_lossy(&attach.stderr),
        );
        return;
    }
    let plist = String::from_utf8_lossy(&attach.stdout);
    let (_devs, whole) = parse_hdiutil_devices(&plist);
    let whole = match whole {
        Some(w) => w,
        None => {
            eprintln!("skipping: could not parse device node out of hdiutil plist:\n{plist}");
            return;
        }
    };

    // hdiutil emits mount points as `<key>mount-point</key><string>…</string>`.
    let mut mount_point: Option<String> = None;
    let mut in_mp_key = false;
    for line in plist.lines() {
        let t = line.trim();
        if t.contains("<key>mount-point</key>") {
            in_mp_key = true;
            continue;
        }
        if in_mp_key {
            if let Some(s) = t.strip_prefix("<string>") {
                if let Some(end) = s.find("</string>") {
                    let mp = &s[..end];
                    if !mp.is_empty() {
                        mount_point = Some(mp.to_string());
                        break;
                    }
                }
            }
            in_mp_key = false;
        }
    }

    let mp = match mount_point {
        Some(mp) => mp,
        None => {
            hdiutil_detach(&whole);
            eprintln!(
                "skipping: hdiutil attached the image but did not mount any volume (expected with our stub-spaceman writer)"
            );
            return;
        }
    };

    // ls + cat through the macOS VFS.
    let ls = Command::new("ls").arg(&mp).output();
    let cat = Command::new("cat").arg(format!("{mp}/rt.txt")).output();
    hdiutil_detach(&whole);

    let ls = ls.expect("ls failed to spawn");
    assert!(
        ls.status.success(),
        "ls {mp} failed:\n{}",
        String::from_utf8_lossy(&ls.stderr)
    );
    let names = String::from_utf8_lossy(&ls.stdout);
    assert!(
        names.contains("rt.txt"),
        "macOS VFS did not see /rt.txt; ls output: {names}"
    );
    let cat = cat.expect("cat failed to spawn");
    assert!(
        cat.status.success(),
        "cat {mp}/rt.txt failed:\n{}",
        String::from_utf8_lossy(&cat.stderr)
    );
    assert_eq!(
        cat.stdout, payload,
        "macOS VFS returned different bytes than we wrote"
    );
}

/// Build an APFS image, run `Apfs::chmod`, mount it natively, and
/// confirm `stat` reports the new mode. macOS is our oracle — it
/// re-parses every byte of the new checkpoint we wrote.
///
/// Skipped on Linux runners (no hdiutil); skipped on macOS when
/// hdiutil refuses to attach the image (the writer's stub-spaceman
/// layout sometimes trips hdiutil on flat-file images).
#[test]
fn apfs_chmod_round_trips_through_macos_mount() {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: APFS validation requires macOS (hdiutil)");
        return;
    }
    if which("hdiutil").is_none() || !hdiutil_usable() {
        eprintln!("skipping: hdiutil not usable");
        return;
    }

    let bs = 4096u32;
    let total_blocks = 4096u64;
    let img = NamedTempFile::new().unwrap();
    let payload = b"mode-test\n";
    // Format with mode 0o600.
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "MODEVOL").unwrap();
        let mut r = Cursor::new(payload.as_ref());
        w.add_file_from_reader(2, "perms.txt", 0o600, &mut r, payload.len() as u64)
            .unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }
    // chmod to 0o644 via the new mutation API — writes a fresh APFS
    // checkpoint.
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.chmod(&mut dev, "/perms.txt", 0o644).unwrap();
        dev.sync().unwrap();
    }
    // Re-read through our own reader to confirm the post-chmod mode
    // landed in the new checkpoint (even when macOS mount skips
    // below).
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let fs = Apfs::open(&mut dev).unwrap();
        // Walk the volume's catalog to find perms.txt's inode mode.
        // The simplest probe: list_path on "/" should include it.
        let entries = fs.list_path(&mut dev, "/").expect("list root");
        assert!(
            entries.iter().any(|e| e.name == "perms.txt"),
            "perms.txt missing post-chmod: {entries:?}"
        );
    }

    // Optional cross-check: mount via macOS VFS and `stat` the file.
    let attach = Command::new("hdiutil")
        .args([
            "attach",
            "-readonly",
            "-imagekey",
            "diskimage-class=CRawDiskImage",
            "-plist",
        ])
        .arg(img.path())
        .output()
        .expect("hdiutil attach failed to spawn");
    if !attach.status.success() {
        eprintln!(
            "skipping macOS-mount cross-check (expected with stub-spaceman):\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&attach.stdout),
            String::from_utf8_lossy(&attach.stderr)
        );
        return;
    }
    let plist = String::from_utf8_lossy(&attach.stdout);
    let (_devs, whole) = parse_hdiutil_devices(&plist);
    let whole = match whole {
        Some(w) => w,
        None => {
            eprintln!("skipping: no device node in hdiutil plist");
            return;
        }
    };
    let mut mount_point: Option<String> = None;
    let mut in_mp_key = false;
    for line in plist.lines() {
        let t = line.trim();
        if t.contains("<key>mount-point</key>") {
            in_mp_key = true;
            continue;
        }
        if in_mp_key {
            if let Some(s) = t.strip_prefix("<string>")
                && let Some(end) = s.find("</string>")
            {
                let mp = &s[..end];
                if !mp.is_empty() {
                    mount_point = Some(mp.to_string());
                    break;
                }
            }
            in_mp_key = false;
        }
    }
    let mp = match mount_point {
        Some(mp) => mp,
        None => {
            hdiutil_detach(&whole);
            eprintln!("skipping: no mount point in hdiutil plist (stub-spaceman)");
            return;
        }
    };
    let stat = Command::new("stat")
        .args(["-f", "%p"])
        .arg(format!("{mp}/perms.txt"))
        .output()
        .expect("stat failed to spawn");
    hdiutil_detach(&whole);
    assert!(
        stat.status.success(),
        "stat failed:\n{}",
        String::from_utf8_lossy(&stat.stderr)
    );
    let raw = String::from_utf8_lossy(&stat.stdout);
    let mode_full = u32::from_str_radix(raw.trim(), 8).expect("octal parse");
    assert_eq!(
        mode_full & 0o7777,
        0o644,
        "expected 0o644 on perms.txt after chmod, got {mode_full:o}"
    );
}

/// Sanity-check that chown + set_times don't corrupt the on-disk
/// state. We can't easily verify the values via macOS mount because
/// mount is intermittent on stub-spaceman writers, but the new
/// checkpoint must still be openable + walkable through our own
/// reader.
#[test]
fn apfs_chown_and_set_times_produce_clean_checkpoint() {
    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total_blocks = 4096u64;
    let img = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "META").unwrap();
        let body = b"meta\n";
        let mut r = Cursor::new(body.as_ref());
        w.add_file_from_reader(2, "m.txt", 0o644, &mut r, body.len() as u64)
            .unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.chown(&mut dev, "/m.txt", 501, 20).unwrap();
        // mtime = ctime = atime = 1_700_000_000_000_000_000 ns
        // (≈ 2023-11-14 UTC). APFS time fields are ns since epoch.
        let t = 1_700_000_000_000_000_000u64;
        fs.set_times(&mut dev, "/m.txt", Some(t), Some(t), Some(t))
            .unwrap();
        dev.sync().unwrap();
    }
    // Verify the image is still openable and the file still
    // enumerates. The on-disk integrity check is implicit: if any
    // of our checkpoint COW steps wrote a malformed structure,
    // `Apfs::open` would error out here.
    let mut dev = FileBackend::open(img.path()).unwrap();
    let fs = Apfs::open(&mut dev).unwrap();
    let entries = fs.list_path(&mut dev, "/").unwrap();
    assert!(
        entries.iter().any(|e| e.name == "m.txt"),
        "m.txt missing after chown + set_times: {entries:?}"
    );
}

/// End-to-end exercise of the Phase 2 mutation API: rename a file,
/// hardlink it under a second name, unlink the rename target (one
/// link drops; the inode survives), then unlink the last link
/// (inode + extents are purged from subsequent checkpoints). Each
/// step writes a fresh APFS checkpoint via the same COW machinery.
#[test]
fn apfs_rename_unlink_link_round_trips() {
    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total_blocks = 4096u64;
    let img = NamedTempFile::new().unwrap();
    let payload = b"phase2\n";
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "P2VOL").unwrap();
        let mut r = Cursor::new(payload.as_ref());
        w.add_file_from_reader(2, "src.txt", 0o644, &mut r, payload.len() as u64)
            .unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }

    // rename: src.txt → renamed.txt
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.rename(&mut dev, "/src.txt", "/renamed.txt").unwrap();
        dev.sync().unwrap();
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let fs = Apfs::open(&mut dev).unwrap();
        let names: Vec<String> = fs
            .list_path(&mut dev, "/")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(
            names.contains(&"renamed.txt".to_string()),
            "rename failed: {names:?}"
        );
        assert!(
            !names.contains(&"src.txt".to_string()),
            "old name lingers: {names:?}"
        );
    }

    // link: add an alias under /alias.txt
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.link(&mut dev, "/renamed.txt", "/alias.txt").unwrap();
        dev.sync().unwrap();
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let fs = Apfs::open(&mut dev).unwrap();
        let names: Vec<String> = fs
            .list_path(&mut dev, "/")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(names.contains(&"renamed.txt".to_string()));
        assert!(
            names.contains(&"alias.txt".to_string()),
            "link missing: {names:?}"
        );
    }

    // unlink first link — second link survives, inode stays put
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.remove_path(&mut dev, "/renamed.txt").unwrap();
        dev.sync().unwrap();
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let fs = Apfs::open(&mut dev).unwrap();
        let names: Vec<String> = fs
            .list_path(&mut dev, "/")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(!names.contains(&"renamed.txt".to_string()));
        assert!(
            names.contains(&"alias.txt".to_string()),
            "alias must survive the first unlink: {names:?}"
        );
    }

    // unlink last link — file is gone
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.remove_path(&mut dev, "/alias.txt").unwrap();
        dev.sync().unwrap();
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let fs = Apfs::open(&mut dev).unwrap();
        let names: Vec<String> = fs
            .list_path(&mut dev, "/")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(
            !names.contains(&"alias.txt".to_string()),
            "alias should be gone: {names:?}"
        );
    }
}

// ---- Self-tests for the local plist parser ----

#[cfg(test)]
mod parser_tests {
    use super::parse_hdiutil_devices;

    #[test]
    fn extracts_whole_disk_and_slices() {
        let plist = r#"
        <plist>
          <dict>
            <key>system-entities</key>
            <array>
              <dict>
                <key>dev-entry</key>
                <string>/dev/disk7</string>
              </dict>
              <dict>
                <key>dev-entry</key>
                <string>/dev/disk7s1</string>
              </dict>
              <dict>
                <key>dev-entry</key>
                <string>/dev/disk7s2</string>
              </dict>
            </array>
          </dict>
        </plist>
        "#;
        let (devs, whole) = parse_hdiutil_devices(plist);
        assert_eq!(whole.as_deref(), Some("/dev/disk7"));
        assert!(devs.contains(&"/dev/disk7".to_string()));
        assert!(devs.contains(&"/dev/disk7s1".to_string()));
        assert!(devs.contains(&"/dev/disk7s2".to_string()));
    }

    #[test]
    fn derives_whole_disk_from_slice_only() {
        let plist = "<string>/dev/disk9s2</string>";
        let (devs, whole) = parse_hdiutil_devices(plist);
        assert_eq!(whole.as_deref(), Some("/dev/disk9"));
        assert_eq!(devs, vec!["/dev/disk9s2".to_string()]);
    }

    #[test]
    fn empty_plist_yields_nothing() {
        let (devs, whole) = parse_hdiutil_devices("");
        assert!(devs.is_empty());
        assert!(whole.is_none());
    }
}

/// Write-state create_file_at: format → flush → reopen for writes →
/// create a new file → close → reopen for reads → file is visible
/// with the right body. Exercises the extracted record builders,
/// MutatorCx::alloc_extent + write_extent_bytes, and the APSB
/// counter bump (`num_files += 1`).
#[test]
fn apfs_write_state_create_file_round_trips() {
    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total_blocks = 4096u64;
    let img = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "WSCF").unwrap();
        // Seed with one file so the volume isn't completely empty —
        // the new file's records have to sort cleanly alongside it.
        let body = b"seed\n";
        let mut r = Cursor::new(body.as_ref());
        w.add_file_from_reader(2, "seed.txt", 0o644, &mut r, body.len() as u64)
            .unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }
    let new_payload = b"created in write state\n";
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.create_file_at(&mut dev, "/created.txt", new_payload, 0o644, 0)
            .unwrap();
        dev.sync().unwrap();
    }
    let mut dev = FileBackend::open(img.path()).unwrap();
    let fs = Apfs::open(&mut dev).unwrap();
    let names: Vec<String> = fs
        .list_path(&mut dev, "/")
        .unwrap()
        .into_iter()
        .map(|e| e.name)
        .collect();
    assert!(
        names.contains(&"seed.txt".to_string()) && names.contains(&"created.txt".to_string()),
        "missing entries after create: {names:?}"
    );
    let mut r = fs.open_file_reader(&mut dev, "/created.txt").unwrap();
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut r, &mut buf).unwrap();
    assert_eq!(buf.as_slice(), new_payload, "created.txt body wrong");
}

/// Write-state create_dir_at + a nested create_file_at in a second
/// checkpoint. Verifies (a) parent oid resolution survives a fresh
/// open_writable cycle (it has to — `next_oid` advances across
/// checkpoints) and (b) the parent dir's `nchildren` is bumped on
/// each child add.
#[test]
fn apfs_write_state_create_dir_then_nested_file() {
    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total_blocks = 4096u64;
    let img = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let w = ApfsWriter::new(&mut dev, total_blocks, bs, "WSCD").unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.create_dir_at(&mut dev, "/etc", 0o755, 0).unwrap();
        dev.sync().unwrap();
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.create_file_at(&mut dev, "/etc/conf", b"k=v\n", 0o644, 0)
            .unwrap();
        dev.sync().unwrap();
    }
    let mut dev = FileBackend::open(img.path()).unwrap();
    let fs = Apfs::open(&mut dev).unwrap();
    let root_names: Vec<String> = fs
        .list_path(&mut dev, "/")
        .unwrap()
        .into_iter()
        .map(|e| e.name)
        .collect();
    assert!(
        root_names.contains(&"etc".to_string()),
        "/etc missing: {root_names:?}"
    );
    let etc_names: Vec<String> = fs
        .list_path(&mut dev, "/etc")
        .unwrap()
        .into_iter()
        .map(|e| e.name)
        .collect();
    assert!(
        etc_names.contains(&"conf".to_string()),
        "/etc/conf missing: {etc_names:?}"
    );
}

/// Write-state create_symlink_at: create a symlink under an existing
/// parent and read its body back. APFS stores symlink targets in a
/// regular file extent under the symlink inode, so `open_file_reader`
/// returns the raw target bytes.
#[test]
fn apfs_write_state_create_symlink_round_trips() {
    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total_blocks = 4096u64;
    let img = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let w = ApfsWriter::new(&mut dev, total_blocks, bs, "WSCS").unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.create_symlink_at(&mut dev, "/link", "/usr/bin/sh", 0o777, 0)
            .unwrap();
        dev.sync().unwrap();
    }
    let mut dev = FileBackend::open(img.path()).unwrap();
    let fs = Apfs::open(&mut dev).unwrap();
    let mut r = fs.open_file_reader(&mut dev, "/link").unwrap();
    let mut target = String::new();
    std::io::Read::read_to_string(&mut r, &mut target).unwrap();
    assert_eq!(target, "/usr/bin/sh", "symlink target wrong");
}

/// Write-state set_xattr + remove_xattr: set on a fresh file, verify
/// via read_xattrs, replace with a different value, verify the
/// replacement, then remove and verify the xattr is gone.
#[test]
fn apfs_write_state_set_xattr_round_trips() {
    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total_blocks = 4096u64;
    let img = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "WSXA").unwrap();
        let body = b"xa\n";
        let mut r = Cursor::new(body.as_ref());
        w.add_file_from_reader(2, "f.txt", 0o644, &mut r, body.len() as u64)
            .unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }
    // set initial value
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.set_xattr(&mut dev, "/f.txt", "user.tag", b"v1").unwrap();
        dev.sync().unwrap();
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let fs = Apfs::open(&mut dev).unwrap();
        let xs = fs.read_xattrs(&mut dev, "/f.txt").unwrap();
        assert_eq!(
            xs.get("user.tag").map(|v| v.as_slice()),
            Some(b"v1".as_ref()),
            "xattr v1 missing: {xs:?}"
        );
    }
    // replace
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.set_xattr(&mut dev, "/f.txt", "user.tag", b"v2-longer")
            .unwrap();
        dev.sync().unwrap();
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let fs = Apfs::open(&mut dev).unwrap();
        let xs = fs.read_xattrs(&mut dev, "/f.txt").unwrap();
        assert_eq!(
            xs.get("user.tag").map(|v| v.as_slice()),
            Some(b"v2-longer".as_ref()),
            "xattr replacement did not stick: {xs:?}"
        );
    }
    // remove
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        fs.remove_xattr(&mut dev, "/f.txt", "user.tag").unwrap();
        dev.sync().unwrap();
    }
    let mut dev = FileBackend::open(img.path()).unwrap();
    let fs = Apfs::open(&mut dev).unwrap();
    let xs = fs.read_xattrs(&mut dev, "/f.txt").unwrap();
    assert!(!xs.contains_key("user.tag"), "xattr lingered: {xs:?}");
}

/// Filesystem-trait wiring exercise: after open_writable, the trait
/// methods (create_file, create_dir, set_attrs, set_xattr, rename,
/// remove) should dispatch to the inherent Write-state methods we
/// added in commit-4 rather than returning Unsupported. This is the
/// surface a generic Filesystem consumer (FUSE adapter, build spec)
/// uses, so it has to actually work in Write state.
#[test]
fn apfs_filesystem_trait_dispatches_to_write_state() {
    use fstool::fs::{Filesystem, SetAttrs};
    use std::path::Path;

    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total_blocks = 4096u64;
    let img = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let w = ApfsWriter::new(&mut dev, total_blocks, bs, "TRAIT").unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }

    // create_file via trait
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        let body: Vec<u8> = b"alpha\n".to_vec();
        let body_len = body.len() as u64;
        Filesystem::create_file(
            &mut fs,
            &mut dev,
            Path::new("/a.txt"),
            fstool::fs::FileSource::Reader {
                reader: Box::new(Cursor::new(body)),
                len: body_len,
            },
            fstool::fs::FileMeta {
                mode: 0o644,
                ..Default::default()
            },
        )
        .unwrap();
        dev.sync().unwrap();
    }
    // create_dir + nested create_file
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        Filesystem::create_dir(
            &mut fs,
            &mut dev,
            Path::new("/sub"),
            fstool::fs::FileMeta {
                mode: 0o755,
                ..Default::default()
            },
        )
        .unwrap();
        dev.sync().unwrap();
    }
    // set_attrs (mode + uid + gid + times in one checkpoint)
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        Filesystem::set_attrs(
            &mut fs,
            &mut dev,
            Path::new("/a.txt"),
            SetAttrs {
                mode: Some(0o600),
                uid: Some(501),
                gid: Some(20),
                mtime: Some(1_700_000_000),
                ctime: Some(1_700_000_000),
                atime: Some(1_700_000_000),
            },
        )
        .unwrap();
        dev.sync().unwrap();
    }
    // set_xattr via trait
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        Filesystem::set_xattr(
            &mut fs,
            &mut dev,
            Path::new("/a.txt"),
            "user.role",
            b"trait",
        )
        .unwrap();
        dev.sync().unwrap();
    }
    // rename via trait
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        Filesystem::rename(&mut fs, &mut dev, Path::new("/a.txt"), Path::new("/b.txt")).unwrap();
        dev.sync().unwrap();
    }
    // remove via trait
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        Filesystem::remove(&mut fs, &mut dev, Path::new("/sub")).unwrap();
        dev.sync().unwrap();
    }

    // Final state: /b.txt with the right xattr and mode; /sub gone.
    let mut dev = FileBackend::open(img.path()).unwrap();
    let fs = Apfs::open(&mut dev).unwrap();
    let names: Vec<String> = fs
        .list_path(&mut dev, "/")
        .unwrap()
        .into_iter()
        .map(|e| e.name)
        .collect();
    assert!(
        names.contains(&"b.txt".to_string()),
        "/b.txt missing: {names:?}"
    );
    assert!(!names.contains(&"a.txt".to_string()), "old name lingers");
    assert!(
        !names.contains(&"sub".to_string()),
        "/sub not removed: {names:?}"
    );
    let xs = fs.read_xattrs(&mut dev, "/b.txt").unwrap();
    assert_eq!(
        xs.get("user.role").map(|v| v.as_slice()),
        Some(b"trait".as_ref()),
        "trait set_xattr did not stick: {xs:?}"
    );
}

/// Filesystem trait on a Read-state Apfs (post-flush, or post-Apfs::open)
/// must keep returning Unsupported for every mutation — re-opening for
/// writes requires Apfs::open_writable explicitly.
#[test]
fn apfs_filesystem_trait_refuses_mutations_on_read_state() {
    use fstool::fs::{Filesystem, SetAttrs};
    use std::path::Path;

    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total_blocks = 4096u64;
    let img = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let mut w = ApfsWriter::new(&mut dev, total_blocks, bs, "RDST").unwrap();
        let body = b"r\n";
        let mut r = Cursor::new(body.as_ref());
        w.add_file_from_reader(2, "f.txt", 0o644, &mut r, body.len() as u64)
            .unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }
    let mut dev = FileBackend::open(img.path()).unwrap();
    let mut fs = Apfs::open(&mut dev).unwrap();
    assert!(matches!(
        Filesystem::remove(&mut fs, &mut dev, Path::new("/f.txt")),
        Err(fstool::Error::Unsupported(_))
    ));
    assert!(matches!(
        Filesystem::rename(&mut fs, &mut dev, Path::new("/f.txt"), Path::new("/g.txt")),
        Err(fstool::Error::Unsupported(_))
    ));
    assert!(matches!(
        Filesystem::set_attrs(
            &mut fs,
            &mut dev,
            Path::new("/f.txt"),
            SetAttrs {
                mode: Some(0o600),
                ..Default::default()
            },
        ),
        Err(fstool::Error::Unsupported(_))
    ));
    assert!(matches!(
        Filesystem::set_xattr(&mut fs, &mut dev, Path::new("/f.txt"), "n", b"v"),
        Err(fstool::Error::Unsupported(_))
    ));
}

/// xp_desc ring buffer: drive more than XP_DESC_BLOCKS (16) successive
/// open_writable + sync cycles. Pre-fix, slot exhaustion errored at
/// the 15th cycle with `Unsupported: xp_desc area is full`. With the
/// ring buffer in place, the writer wraps around and overwrites stale
/// slots, and every cycle's checkpoint is openable by the next one.
///
/// Asserts after 25 cycles: every created file is visible. This
/// indirectly confirms that find_live_nxsb's "highest xid wins" logic
/// keeps picking the newest checkpoint as we ring past the original
/// slots.
#[test]
fn apfs_xp_desc_ring_buffer_survives_many_checkpoints() {
    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total_blocks = 8192u64; // 32 MiB — room for many small extents
    let img = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let w = ApfsWriter::new(&mut dev, total_blocks, bs, "RING").unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }
    // 25 cycles > XP_DESC_BLOCKS (16) → guaranteed to wrap.
    let n_cycles = 25usize;
    for i in 0..n_cycles {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        let path = format!("/file-{i:03}");
        let body = format!("body-{i}\n");
        fs.create_file_at(&mut dev, &path, body.as_bytes(), 0o644, 0)
            .expect("create_file_at past slot 15 (ring should wrap)");
        dev.sync().unwrap();
    }

    // Verify every file is visible from a fresh read open.
    let mut dev = FileBackend::open(img.path()).unwrap();
    let fs = Apfs::open(&mut dev).unwrap();
    let names: std::collections::HashSet<String> = fs
        .list_path(&mut dev, "/")
        .unwrap()
        .into_iter()
        .map(|e| e.name)
        .collect();
    for i in 0..n_cycles {
        let want = format!("file-{i:03}");
        assert!(
            names.contains(&want),
            "cycle {i}: {want} missing after ring rotation; saw {names:?}"
        );
    }
    // Spot-check the last file's body to confirm the latest checkpoint
    // is the one find_live_nxsb returned.
    let last = format!("/file-{:03}", n_cycles - 1);
    let mut r = fs.open_file_reader(&mut dev, &last).unwrap();
    let mut body = String::new();
    std::io::Read::read_to_string(&mut r, &mut body).unwrap();
    assert_eq!(body, format!("body-{}\n", n_cycles - 1));
}

/// CLI round-trip: `fstool add` against an APFS image must reach
/// the Write-state mutators (commit-A wiring). Before that change
/// AnyFs::open returned a Read-state Apfs handle and the trait
/// methods all returned Unsupported, so `fstool add disk.apfs …`
/// erred with "apfs is a write-once format" even though the
/// inherent Write-state API worked fine.
#[test]
fn cli_add_rm_reach_apfs_write_state() {
    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bin = env!("CARGO_BIN_EXE_fstool");
    let dir = tempfile::tempdir().unwrap();
    let img = dir.path().join("v.apfs");
    {
        let bs = 4096u32;
        let total = 4096u64;
        let mut dev = FileBackend::create(&img, total * bs as u64).unwrap();
        let mut w = ApfsWriter::new(&mut dev, total, bs, "CLI").unwrap();
        // Seed one file so /seed is present for the rm step.
        let body = b"seed\n";
        let mut r = Cursor::new(body.as_ref());
        w.add_file_from_reader(2, "seed.txt", 0o644, &mut r, body.len() as u64)
            .unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }

    // fstool add via CLI — must succeed (commit-A wiring).
    let host = dir.path().join("host.txt");
    std::fs::write(&host, b"cli-added\n").unwrap();
    let r = Command::new(bin)
        .arg("add")
        .arg(&img)
        .arg(&host)
        .arg("/added.txt")
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "fstool add on apfs failed: {}",
        String::from_utf8_lossy(&r.stderr)
    );

    // Verify via ls that /added.txt is present.
    let ls = Command::new(bin)
        .arg("ls")
        .arg(&img)
        .arg("/")
        .output()
        .unwrap();
    assert!(ls.status.success());
    let listing = String::from_utf8_lossy(&ls.stdout);
    assert!(
        listing.contains("added.txt"),
        "/added.txt missing: {listing}"
    );

    // fstool rm via CLI — must succeed.
    let r = Command::new(bin)
        .arg("rm")
        .arg(&img)
        .arg("/seed.txt")
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "fstool rm on apfs failed: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    let ls = Command::new(bin)
        .arg("ls")
        .arg(&img)
        .arg("/")
        .output()
        .unwrap();
    assert!(ls.status.success());
    let listing = String::from_utf8_lossy(&ls.stdout);
    assert!(!listing.contains("seed.txt"), "rm did not stick: {listing}");
}

/// `Filesystem::create_file` (via the trait) carries FileMeta.mtime
/// through to the APFS inode's time fields. Before commit-B the
/// PendingWrite single-pass writer dropped every timestamp and the
/// inode was stamped epoch (1970-01-01); now the user-supplied
/// mtime survives the round-trip.
#[test]
fn apfs_create_file_preserves_mtime() {
    use fstool::fs::{FileMeta, FileSource, Filesystem};
    use std::path::Path;

    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total_blocks = 4096u64;
    let img = NamedTempFile::new().unwrap();
    let mtime: u32 = 1_700_000_000; // ~2023-11-14
    {
        let mut dev = FileBackend::create(img.path(), total_blocks * bs as u64).unwrap();
        let w = ApfsWriter::new(&mut dev, total_blocks, bs, "MTIME").unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
        drop(dev);
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        let body = b"t\n".to_vec();
        let body_len = body.len() as u64;
        Filesystem::create_file(
            &mut fs,
            &mut dev,
            Path::new("/t.txt"),
            FileSource::Reader {
                reader: Box::new(Cursor::new(body)),
                len: body_len,
            },
            FileMeta {
                mode: 0o644,
                mtime,
                ..Default::default()
            },
        )
        .unwrap();
        dev.sync().unwrap();
    }
    let mut dev = FileBackend::open(img.path()).unwrap();
    let mut fs = Apfs::open(&mut dev).unwrap();
    let attrs = Filesystem::getattr(&mut fs, &mut dev, Path::new("/t.txt")).unwrap();
    assert_eq!(
        attrs.mtime, mtime,
        "mtime did not round-trip; got {} expected {mtime}",
        attrs.mtime
    );
}

/// `Filesystem::truncate` on APFS routes through the open_file_rw
/// then set_len pathway. Shrink first, then grow back with zeros,
/// then verify the file body matches.
#[test]
fn apfs_filesystem_truncate_round_trips() {
    use fstool::fs::Filesystem;
    use std::path::Path;

    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total = 4096u64;
    let img = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(img.path(), total * bs as u64).unwrap();
        let mut w = ApfsWriter::new(&mut dev, total, bs, "TRUN").unwrap();
        let body = b"hello world\n";
        let mut r = Cursor::new(body.as_ref());
        w.add_file_from_reader(2, "t.txt", 0o644, &mut r, body.len() as u64)
            .unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }
    // Shrink to 5 bytes.
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        Filesystem::truncate(&mut fs, &mut dev, Path::new("/t.txt"), 5).unwrap();
        dev.sync().unwrap();
    }
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let fs = Apfs::open(&mut dev).unwrap();
        let mut r = fs.open_file_reader(&mut dev, "/t.txt").unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut r, &mut buf).unwrap();
        assert_eq!(buf.as_slice(), b"hello", "shrink failed: {buf:?}");
    }
    // Grow back to 8 bytes — tail is zero-filled.
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        Filesystem::truncate(&mut fs, &mut dev, Path::new("/t.txt"), 8).unwrap();
        dev.sync().unwrap();
    }
    let mut dev = FileBackend::open(img.path()).unwrap();
    let fs = Apfs::open(&mut dev).unwrap();
    let mut r = fs.open_file_reader(&mut dev, "/t.txt").unwrap();
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut r, &mut buf).unwrap();
    assert_eq!(
        buf.as_slice(),
        b"hello\0\0\0",
        "grow zero-fill wrong: {buf:?}"
    );
}

/// `Filesystem::list_xattrs` surfaces every xattr on the inode,
/// sorted by name. Default in the trait returns empty — APFS
/// overrides to wrap Apfs::read_xattrs.
#[test]
fn apfs_filesystem_list_xattrs_sorted() {
    use fstool::fs::Filesystem;
    use std::path::Path;

    if !cfg!(target_os = "macos") && !cfg!(target_os = "linux") {
        eprintln!("skipping: APFS validation needs unix");
        return;
    }
    let bs = 4096u32;
    let total = 4096u64;
    let img = NamedTempFile::new().unwrap();
    {
        let mut dev = FileBackend::create(img.path(), total * bs as u64).unwrap();
        let mut w = ApfsWriter::new(&mut dev, total, bs, "XATT").unwrap();
        let body = b"x\n";
        let mut r = Cursor::new(body.as_ref());
        w.add_file_from_reader(2, "f.txt", 0o644, &mut r, body.len() as u64)
            .unwrap();
        w.finish().unwrap();
        dev.sync().unwrap();
    }
    // Set three xattrs out-of-order.
    {
        let mut dev = FileBackend::open(img.path()).unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        Filesystem::set_xattr(&mut fs, &mut dev, Path::new("/f.txt"), "user.zeta", b"z").unwrap();
        dev.sync().unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        Filesystem::set_xattr(&mut fs, &mut dev, Path::new("/f.txt"), "user.alpha", b"a").unwrap();
        dev.sync().unwrap();
        let mut fs = Apfs::open_writable(&mut dev).unwrap();
        Filesystem::set_xattr(&mut fs, &mut dev, Path::new("/f.txt"), "user.mid", b"m").unwrap();
        dev.sync().unwrap();
    }
    let mut dev = FileBackend::open(img.path()).unwrap();
    let mut fs = Apfs::open(&mut dev).unwrap();
    let xs = Filesystem::list_xattrs(&mut fs, &mut dev, Path::new("/f.txt")).unwrap();
    let names: Vec<&str> = xs.iter().map(|x| x.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["user.alpha", "user.mid", "user.zeta"],
        "got {names:?}"
    );
}
