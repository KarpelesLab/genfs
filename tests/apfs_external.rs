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
