//! ext3 (journal) end-to-end validation.

use std::process::Command;

use fstool::block::{BlockDevice, FileBackend};
use fstool::fs::ext::{Ext, FormatOpts, FsKind};
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

#[test]
fn ext3_passes_e2fsck_and_advertises_journal() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("dumpe2fs") else {
        eprintln!("skipping: dumpe2fs not installed");
        return;
    };

    let tmp = NamedTempFile::new().unwrap();
    // 8 MiB image; 1024-block journal at 1 KiB = 1 MiB (the JBD2 minimum).
    let opts = FormatOpts {
        kind: FsKind::Ext3,
        blocks_count: 8192,
        inodes_count: 64,
        journal_blocks: 1024,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    Ext::format_with(&mut dev, &opts).unwrap();
    dev.sync().unwrap();
    drop(dev);

    // e2fsck clean.
    let out = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "e2fsck failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // dumpe2fs must report the has_journal feature + journal inode 8.
    let out = Command::new("dumpe2fs")
        .arg("-h")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("has_journal"),
        "missing has_journal in:\n{stdout}"
    );
    assert!(
        stdout.contains("Journal inode:            8"),
        "missing Journal inode line in:\n{stdout}"
    );
}
