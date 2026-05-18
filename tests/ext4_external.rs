//! ext4 (journal + extent tree) end-to-end validation.

use std::io::Write;
use std::process::Command;

use fstool::block::{BlockDevice, FileBackend};
use fstool::fs::ext::{Ext, FormatOpts, FsKind};
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

#[test]
fn ext4_passes_e2fsck_and_advertises_features() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("dumpe2fs") else {
        eprintln!("skipping: dumpe2fs not installed");
        return;
    };
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };

    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts {
        kind: FsKind::Ext4,
        blocks_count: 8192,
        inodes_count: 64,
        journal_blocks: 1024,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    let mut ext = Ext::format_with(&mut dev, &opts).unwrap();

    // Plant a file to exercise the extent writer.
    let mut src = NamedTempFile::new().unwrap();
    src.as_file_mut()
        .write_all(b"the quick brown fox\n")
        .unwrap();
    ext.add_file_to(
        &mut dev,
        2,
        b"fox.txt",
        FileSource::HostPath(src.path().to_path_buf()),
        FileMeta::with_mode(0o644),
    )
    .unwrap();
    ext.flush(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);

    // e2fsck must be clean.
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

    // dumpe2fs must list the `extent` feature + the journal.
    let out = Command::new("dumpe2fs")
        .arg("-h")
        .arg(tmp.path())
        .output()
        .unwrap();
    let dump = String::from_utf8_lossy(&out.stdout);
    assert!(dump.contains("extent"), "missing `extent` feature:\n{dump}");
    assert!(dump.contains("has_journal"), "missing has_journal:\n{dump}");

    // debugfs `stat /fox.txt` must show an EXTENTS_FL flag and the extent
    // tree contents (not direct/indirect blocks).
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("stat /fox.txt")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stat = String::from_utf8_lossy(&out.stdout);
    assert!(
        stat.contains("EXTENTS") || stat.contains("Extents"),
        "expected extent-mode inode:\n{stat}"
    );

    // `debugfs cat` must return the file body.
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("cat /fox.txt")
        .arg(tmp.path())
        .output()
        .unwrap();
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(
        body.contains("the quick brown fox"),
        "wrong file body via debugfs:\n{body}"
    );
}

/// Round-trip the extent-encoded image through Ext::open + the streaming
/// reader, confirming our own reader resolves extents correctly.
#[test]
fn ext4_open_reads_extent_file() {
    use std::io::Read;
    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts {
        kind: FsKind::Ext4,
        blocks_count: 8192,
        inodes_count: 64,
        journal_blocks: 1024,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    {
        let mut ext = Ext::format_with(&mut dev, &opts).unwrap();
        let mut src = NamedTempFile::new().unwrap();
        src.as_file_mut()
            .write_all(b"extent-encoded payload\n")
            .unwrap();
        ext.add_file_to(
            &mut dev,
            2,
            b"payload.bin",
            FileSource::HostPath(src.path().to_path_buf()),
            FileMeta::with_mode(0o644),
        )
        .unwrap();
        ext.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }

    let ext = Ext::open(&mut dev).unwrap();
    assert_eq!(ext.kind, FsKind::Ext4);
    let ino = ext.path_to_inode(&mut dev, "/payload.bin").unwrap();
    let mut reader = ext.open_file_reader(&mut dev, ino).unwrap();
    let mut body = Vec::new();
    reader.read_to_end(&mut body).unwrap();
    assert_eq!(body, b"extent-encoded payload\n");
}
