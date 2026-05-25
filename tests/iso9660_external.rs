//! External validation of the ISO 9660 writer against native tooling.
//!
//! Currently focused on the Rock Ridge `SP` (System Use Sharing
//! Protocol) marker on the root's "." record: IEEE P1282 / SUSP §5.3
//! requires it so that conformant readers (e.g. `isoinfo -d`) recognise
//! that Rock Ridge extensions are present on the volume.
//!
//! Tests degrade gracefully when the native tools aren't installed.

use std::path::Path;
use std::process::Command;

use fstool::block::FileBackend;
use fstool::fs::iso9660::{FormatOpts, Iso9660Writer};
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

/// Build a tiny RR-enabled ISO image at `path` containing a single
/// directory and one regular file, sized large enough for the writer's
/// pass-1 layout (a few sectors of overhead).
fn build_tiny_iso(path: &Path) {
    let capacity: u64 = 4 * 1024 * 1024;
    let mut dev = FileBackend::create(path, capacity).unwrap();
    let opts = FormatOpts {
        volume_id: "SPCHECK".into(),
        joliet: true,
        rock_ridge: true,
        ..FormatOpts::default()
    };
    let mut w = Iso9660Writer::new(opts);
    w.add_dir(Path::new("/etc"), FileMeta::default()).unwrap();
    let body = b"hi\n".to_vec();
    let src = FileSource::Reader {
        reader: Box::new(std::io::Cursor::new(body.clone())),
        len: body.len() as u64,
    };
    w.add_file(&mut dev, Path::new("/etc/conf"), src, FileMeta::default())
        .unwrap();
    w.flush(&mut dev).unwrap();
}

#[test]
fn isoinfo_recognises_rock_ridge_sp_marker() {
    let Some(_) = which("isoinfo") else {
        eprintln!("skipping: isoinfo not installed");
        return;
    };

    let tmp = NamedTempFile::new().unwrap();
    build_tiny_iso(tmp.path());

    // `isoinfo -d -i <image>` dumps the volume descriptors. When the
    // SP entry is present on the root's "." record it prints either a
    // "Rock Ridge ... found" line, or — depending on isoinfo version —
    // a "SUSP signatures version 1 found" line. Without SP it prints
    // "NO SUSP/Rock Ridge present".
    let out = Command::new("isoinfo")
        .arg("-d")
        .arg("-i")
        .arg(tmp.path())
        .output()
        .expect("isoinfo failed to spawn");
    assert!(out.status.success(), "isoinfo failed: {:?}", out.status);
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let lower = combined.to_lowercase();
    let detected = (lower.contains("rock ridge") || lower.contains("susp signatures"))
        && !lower.contains("no susp")
        && !lower.contains("no rock ridge");
    assert!(
        detected,
        "isoinfo did not detect SUSP/Rock Ridge on the image. Output:\n{combined}"
    );
}
