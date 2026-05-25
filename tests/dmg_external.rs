#![cfg(unix)]
//! DMG end-to-end validation against macOS' `hdiutil`.
//!
//! The unit tests in `src/block/dmg/{mish,codec}.rs` exercise the
//! decoders against a fixed test asset. These tests close the loop:
//! they ask `hdiutil` to *write* a DMG in each supported compression
//! format (raw / zlib / bzip2 / LZFSE) and confirm fstool's
//! [`DmgBackend`] can read every virtual sector back. That guards
//! against `hdiutil` updates that introduce new chunk types, plist
//! layout changes, or trailer-field tweaks that the static asset
//! wouldn't catch.
//!
//! macOS-only. On Linux/Windows runners every test prints a
//! `skipping: …` line and returns success, matching
//! `tests/apfs_external.rs`.
//!
//! The format names are macOS's standard `hdiutil -format` codes:
//!
//! | code   | what it produces                                |
//! |--------|-------------------------------------------------|
//! | `UDRW` | UDIF read-write (no compression, raw chunks)    |
//! | `UDZO` | UDIF read-only zlib-compressed                  |
//! | `UDBZ` | UDIF read-only bzip2-compressed                 |
//! | `ULFO` | UDIF read-only LZFSE-compressed (newer macOS)  |
//!
//! Each format exercises a different code path in
//! `src/block/dmg/codec.rs`. ULFO additionally requires fstool's
//! `dmg-lzfse` feature (on by default) — if that's compiled out,
//! `DmgBackend::open` errors `Unsupported` and the test surfaces it
//! rather than silently passing.

use std::path::{Path, PathBuf};
use std::process::Command;

use fstool::block::{BlockDevice, DmgBackend};

/// Cheap `command -v` shim, identical in shape to the helpers in
/// `tests/apfs_external.rs` so the skip policy stays uniform.
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

/// Sanity-check that `hdiutil` actually runs (not just on PATH).
fn hdiutil_usable() -> bool {
    Command::new("hdiutil")
        .arg("help")
        .output()
        .map(|o| o.status.success() || o.status.code() == Some(0))
        .unwrap_or(false)
}

/// Returns `true` if the caller should skip — emits the reason on the
/// way out so CI logs explain the no-op.
fn skip_unless_macos_hdiutil() -> bool {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: DMG e2e requires macOS (hdiutil)");
        return true;
    }
    if which("hdiutil").is_none() {
        eprintln!("skipping: hdiutil not found on PATH");
        return true;
    }
    if !hdiutil_usable() {
        eprintln!("skipping: hdiutil refused to run `hdiutil help`");
        return true;
    }
    false
}

/// Ask `hdiutil` to create a single-volume HFS+ DMG of the given
/// `-format`. Returns the path on success, or `None` when `hdiutil`
/// refused — newer formats (`ULFO`, `ULMO`) need recent macOS and
/// some host configurations restrict creation; in either case we
/// gracefully degrade to a skip rather than fail the test.
fn create_hdiutil_dmg(dir: &Path, name: &str, format: &str) -> Option<PathBuf> {
    let path = dir.join(format!("{name}.dmg"));
    let out = Command::new("hdiutil")
        .args([
            "create",
            "-size",
            "8m",
            "-fs",
            "HFS+",
            "-layout",
            "NONE",
            "-volname",
            "FstoolDmgTest",
            "-format",
            format,
        ])
        .arg(&path)
        .output()
        .expect("hdiutil create failed to spawn");
    if !out.status.success() {
        eprintln!(
            "skipping {format}: hdiutil create refused:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        return None;
    }
    Some(path)
}

/// Shared body: format a DMG via hdiutil, open via DmgBackend, read
/// every virtual sector through the BlockDevice surface.
///
/// Doesn't byte-compare against a known plaintext (HFS+ writes
/// timestamps, UUIDs, etc. — the contents differ per `hdiutil` run),
/// but reading every sector forces each chunk through the
/// decompression path. A regression in `mish`/`codec` would surface
/// as a decode error, a CRC mismatch, or a hang.
fn dmg_read_every_sector(format: &str) {
    if skip_unless_macos_hdiutil() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let Some(path) = create_hdiutil_dmg(dir.path(), "rt", format) else {
        return;
    };

    let mut dmg = match DmgBackend::open(&path) {
        Ok(d) => d,
        Err(e) => {
            // ULFO might fail when fstool was built without dmg-lzfse;
            // surface the error rather than silently skipping so a CI
            // misconfig (feature disabled by accident) is visible.
            panic!("DmgBackend::open({format}) failed: {e}");
        }
    };
    assert!(
        dmg.total_size() > 0,
        "DMG {format} virtual size is 0 — koly trailer parse failed?"
    );
    assert!(
        dmg.chunk_count() > 0,
        "DMG {format} has zero chunks — mish parse failed?"
    );

    // Walk every virtual byte through the BlockDevice. 64 KiB stride
    // keeps the buffer cheap on small runners; the test image is 8 MiB
    // so total work is ≤128 calls.
    const STRIDE: usize = 64 * 1024;
    let size = dmg.total_size();
    let mut buf = vec![0u8; STRIDE];
    let mut off = 0u64;
    while off < size {
        let n = STRIDE.min((size - off) as usize);
        dmg.read_at(off, &mut buf[..n]).unwrap_or_else(|e| {
            panic!("DMG {format} read_at(off={off}, n={n}): {e}");
        });
        off += n as u64;
    }
}

#[test]
fn dmg_udrw_reads_every_sector() {
    dmg_read_every_sector("UDRW");
}

#[test]
fn dmg_udzo_zlib_reads_every_sector() {
    dmg_read_every_sector("UDZO");
}

#[test]
fn dmg_udbz_bzip2_reads_every_sector() {
    dmg_read_every_sector("UDBZ");
}

#[test]
fn dmg_ulfo_lzfse_reads_every_sector() {
    dmg_read_every_sector("ULFO");
}

/// Full-stack check: open the UDZO DMG, hand it to fstool's HFS+
/// reader, and confirm `list_path("/")` returns successfully. The
/// content of the empty volume varies a bit between macOS releases
/// (some include `.HFS+ Private Directory Data` etc.), so the
/// assertion is just "the catalog is readable" — not a specific name
/// list. This catches a class of regressions where the codec layer
/// returns garbage that happens to parse as catalog nodes.
#[test]
fn dmg_udzo_hfs_plus_walks_through_fstool() {
    if skip_unless_macos_hdiutil() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let Some(path) = create_hdiutil_dmg(dir.path(), "hfs", "UDZO") else {
        return;
    };
    let mut dmg = DmgBackend::open(&path).expect("DmgBackend::open");
    let hfs = fstool::fs::hfs_plus::HfsPlus::open(&mut dmg).expect("HfsPlus::open");
    let entries = hfs
        .list_path(&mut dmg, "/")
        .expect("list_path / on DMG-wrapped HFS+");
    eprintln!(
        "DMG-wrapped HFS+ root entries ({}): {:?}",
        entries.len(),
        entries.iter().map(|e| e.name.clone()).collect::<Vec<_>>()
    );
}
