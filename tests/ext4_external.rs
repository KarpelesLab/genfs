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

/// Read a *default* `mke2fs -t ext4` image — 64bit + flex_bg +
/// metadata_csum + extents + extra_isize all enabled. Confirms our reader
/// tolerates the modern feature set for inspection (ls / cat / info).
#[test]
fn read_default_mke2fs_ext4_image() {
    use std::io::Read;
    let Some(_) = which("mke2fs") else {
        eprintln!("skipping: mke2fs not installed");
        return;
    };

    // Source tree to embed.
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(srcdir.path().join("etc")).unwrap();
    std::fs::write(srcdir.path().join("readme"), b"default ext4\n").unwrap();
    std::fs::write(srcdir.path().join("etc/conf"), b"x=1\n").unwrap();

    let tmp = NamedTempFile::new().unwrap();
    let out = Command::new("mke2fs")
        .args([
            "-F",
            "-t",
            "ext4",
            "-b",
            "1024",
            "-L",
            "",
            "-U",
            "00000000-0000-0000-0000-000000000000",
            "-E",
            "nodiscard",
            "-d",
        ])
        .arg(srcdir.path())
        .arg(tmp.path())
        .arg("8192")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "mke2fs failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // fstool must open it and detect ext4.
    let mut dev = FileBackend::open(tmp.path()).unwrap();
    let ext = Ext::open(&mut dev).unwrap();
    assert_eq!(ext.kind, FsKind::Ext4);
    // 64-bit images use 64-byte group descriptors.
    assert_eq!(ext.sb.group_desc_size(), 64);

    // Root listing must include the embedded tree.
    let root = ext.list_inode(&mut dev, 2).unwrap();
    let names: std::collections::HashSet<_> = root.iter().map(|e| e.name.clone()).collect();
    assert!(names.contains("readme"), "missing /readme: {names:?}");
    assert!(names.contains("etc"), "missing /etc: {names:?}");

    // File contents come back byte-exact through the extent reader.
    let ino = ext.path_to_inode(&mut dev, "/readme").unwrap();
    let mut reader = ext.open_file_reader(&mut dev, ino).unwrap();
    let mut body = Vec::new();
    reader.read_to_end(&mut body).unwrap();
    assert_eq!(body, b"default ext4\n");

    let ino = ext.path_to_inode(&mut dev, "/etc/conf").unwrap();
    let mut reader = ext.open_file_reader(&mut dev, ino).unwrap();
    let mut body = Vec::new();
    reader.read_to_end(&mut body).unwrap();
    assert_eq!(body, b"x=1\n");
}

/// A mostly-zero file written with `sparse` set should occupy far fewer
/// blocks while still reading back identically, and stay e2fsck-clean.
#[test]
fn ext4_sparse_file_uses_holes() {
    use std::io::Read;
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };

    // 256 KiB: 4 KiB of data, 248 KiB of zeros, 4 KiB of data.
    let mut body = vec![b'A'; 4096];
    body.extend(std::iter::repeat_n(0u8, 248 * 1024));
    body.extend(std::iter::repeat_n(b'B', 4096));

    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("hole.bin"), &body).unwrap();

    let opts = FormatOpts {
        kind: FsKind::Ext4,
        blocks_count: 8192,
        inodes_count: 64,
        journal_blocks: 1024,
        sparse: true,
        ..FormatOpts::default()
    };
    let tmp = NamedTempFile::new().unwrap();
    let mut dev = FileBackend::create(
        tmp.path(),
        opts.blocks_count as u64 * opts.block_size as u64,
    )
    .unwrap();
    let mut ext = Ext::format_with(&mut dev, &opts).unwrap();
    ext.add_file_to(
        &mut dev,
        2,
        b"hole.bin",
        FileSource::HostPath(srcdir.path().join("hole.bin")),
        FileMeta::with_mode(0o644),
    )
    .unwrap();
    ext.flush(&mut dev).unwrap();
    dev.sync().unwrap();

    // The file's content must round-trip through our reader exactly.
    let ino = ext.path_to_inode(&mut dev, "/hole.bin").unwrap();
    let mut got = Vec::new();
    ext.open_file_reader(&mut dev, ino)
        .unwrap()
        .read_to_end(&mut got)
        .unwrap();
    assert_eq!(got, body, "sparse file content mismatch");

    // The inode should account for only the ~8 KiB of real data, not 256.
    let inode = ext.read_inode(&mut dev, ino).unwrap();
    // blocks_512 counts 512-byte sectors; 8 KiB = 16, full file = 512.
    assert!(
        inode.blocks_512 < 64,
        "sparse file used {} sectors, expected far fewer than the dense 512",
        inode.blocks_512
    );
    drop(dev);

    let out = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "e2fsck failed on sparse ext4:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
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
