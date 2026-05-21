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

/// With sparse_super, only groups 0, 1 and powers of 3/5/7 carry SB
/// backups. Builds a multi-group ext4 and checks via `dumpe2fs` that
/// the right groups are flagged with "Backup superblock".
#[test]
fn ext4_sparse_super_skips_non_backup_groups() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("dumpe2fs") else {
        eprintln!("skipping: dumpe2fs not installed");
        return;
    };

    // 4 groups (32 MiB at 1 KiB blocks).
    let opts = FormatOpts {
        kind: FsKind::Ext4,
        blocks_count: 32 * 1024,
        inodes_count: 64,
        journal_blocks: 1024,
        sparse_super: true,
        ..FormatOpts::default()
    };
    let tmp = NamedTempFile::new().unwrap();
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    Ext::format_with(&mut dev, &opts).unwrap();
    dev.sync().unwrap();
    drop(dev);

    // e2fsck must stay clean.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck failed on sparse_super image:\n{}",
        String::from_utf8_lossy(&fsck.stdout)
    );

    // dumpe2fs reports per-group metadata. With 4 groups: 0, 1, 3 are
    // backup; 2 is not (2 is not a power of 3/5/7). Group 3 IS (3 = 3^1).
    let dump = Command::new("dumpe2fs")
        .arg("-h")
        .arg(tmp.path())
        .output()
        .unwrap();
    let header = String::from_utf8_lossy(&dump.stdout);
    assert!(
        header.contains("sparse_super"),
        "sparse_super flag missing from dumpe2fs:\n{header}"
    );

    let dump = Command::new("dumpe2fs").arg(tmp.path()).output().unwrap();
    let body = String::from_utf8_lossy(&dump.stdout);
    // dumpe2fs lists each group's "Primary superblock" / "Backup
    // superblock" / no superblock at all.
    let mut g2_has_sb = false;
    let mut g3_has_sb = false;
    let mut current_group: Option<u32> = None;
    for line in body.lines() {
        if let Some(rest) = line.strip_prefix("Group ") {
            // "Group 2: (Blocks 16385-24576) ..."
            let num: u32 = rest
                .split_whitespace()
                .next()
                .unwrap()
                .trim_end_matches(':')
                .parse()
                .unwrap_or(0);
            current_group = Some(num);
        }
        if matches!(current_group, Some(2)) && line.contains("superblock at") {
            g2_has_sb = true;
        }
        if matches!(current_group, Some(3)) && line.contains("superblock at") {
            g3_has_sb = true;
        }
    }
    assert!(
        !g2_has_sb,
        "group 2 should NOT have a backup superblock with sparse_super:\n{body}"
    );
    assert!(
        g3_has_sb,
        "group 3 SHOULD have a backup superblock (3 is a power of 3):\n{body}"
    );
}

/// Add enough entries to a single directory that it spans multiple data
/// blocks. Exercises the writer's directory-growth path (per-block linear
/// fill, allocate-and-extend-extent on overflow). The output must pass
/// `e2fsck -fn` and `debugfs ls /bigdir` must list every entry.
#[test]
fn ext4_large_directory_spans_multiple_blocks() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };

    // 4 KiB blocks → ~120 short entries per dir block. 500 names guarantees
    // we cross the single-block cap, but stays well under the depth-0 extent
    // cap (4 contiguous-or-coalescing extents).
    let opts = FormatOpts {
        kind: FsKind::Ext4,
        block_size: 4096,
        blocks_count: 8192,
        inodes_count: 1024,
        journal_blocks: 1024,
        ..FormatOpts::default()
    };
    let tmp = NamedTempFile::new().unwrap();
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    let mut ext = Ext::format_with(&mut dev, &opts).unwrap();

    // Subdirectory to hold the burst.
    let bigdir = ext
        .add_dir_to(&mut dev, 2, b"bigdir", FileMeta::with_mode(0o755))
        .unwrap();

    // 500 zero-byte files with short, distinct names.
    let n = 500u32;
    for i in 0..n {
        let name = format!("f{i:04}");
        let mut src = NamedTempFile::new().unwrap();
        src.as_file_mut().write_all(b"").unwrap();
        ext.add_file_to(
            &mut dev,
            bigdir,
            name.as_bytes(),
            FileSource::HostPath(src.path().to_path_buf()),
            FileMeta::with_mode(0o644),
        )
        .unwrap();
    }
    ext.flush(&mut dev).unwrap();
    dev.sync().unwrap();

    // Confirm the inode's recorded size grew past one block.
    let bigdir_inode = ext.read_inode(&mut dev, bigdir).unwrap();
    assert!(
        bigdir_inode.size > opts.block_size,
        "expected multi-block dir, got size={} (one block is {})",
        bigdir_inode.size,
        opts.block_size
    );

    // Our own reader must see all 500 names.
    let entries = ext.list_inode(&mut dev, bigdir).unwrap();
    let names: std::collections::HashSet<_> = entries
        .iter()
        .map(|e| e.name.clone())
        .filter(|n| n != "." && n != "..")
        .collect();
    assert_eq!(
        names.len() as u32,
        n,
        "fstool ls miscounted: got {} expected {n}",
        names.len()
    );

    drop(dev);

    // e2fsck must stay clean.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck failed on multi-block dir image:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&fsck.stdout),
        String::from_utf8_lossy(&fsck.stderr)
    );

    // debugfs must agree on the entry count.
    let out = Command::new("debugfs")
        .arg("-R")
        .arg("ls -l /bigdir")
        .arg(tmp.path())
        .output()
        .unwrap();
    let listing = String::from_utf8_lossy(&out.stdout);
    let count = listing
        .lines()
        .filter(|l| {
            // Real entry rows start with "<inode>". Skip blank, ".", "..".
            let first = l.split_whitespace().next().unwrap_or("");
            first.parse::<u32>().is_ok()
                && !l.contains(" . ")
                && !l.ends_with(" .")
                && !l.contains(" .. ")
                && !l.ends_with(" ..")
        })
        .count();
    assert_eq!(
        count as u32, n,
        "debugfs counted {count} entries, expected {n}:\n{listing}"
    );
}

/// Force the extent tree past its depth-0 cap (4 inline leaves) by
/// interleaving directory growth with multi-block file writes. Each file
/// pushes the allocator forward several blocks, so each successive dir
/// block lands non-adjacent to the previous one → no coalescing → many
/// extents. The writer must promote depth-0 → depth-1, write a leaf
/// block with its `ext4_extent_tail` CRC32C, and pass e2fsck on the
/// metadata_csum-enabled output.
#[test]
fn ext4_fragmented_directory_promotes_to_depth1_extent_tree() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };

    let opts = FormatOpts {
        kind: FsKind::Ext4,
        block_size: 4096,
        blocks_count: 32 * 1024,
        inodes_count: 4096,
        journal_blocks: 1024,
        ..FormatOpts::default()
    };
    let tmp = NamedTempFile::new().unwrap();
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    let mut ext = Ext::format_with(&mut dev, &opts).unwrap();
    let bigdir = ext
        .add_dir_to(&mut dev, 2, b"frag", FileMeta::with_mode(0o755))
        .unwrap();

    // Make every file 16 KiB → 4 dir-block-sized regions are allocated
    // between each dir grow, fragmenting the dir's own block layout
    // enough to need more than 4 extents.
    let mut payload = Vec::with_capacity(16 * 1024);
    for i in 0..(16 * 1024) {
        payload.push((i & 0xff) as u8);
    }
    // ~250 entries fill one 4 KiB block; 2000 needs ~8 dir blocks, which
    // can't fit in 4 inline extents once each block is fragmented.
    let n = 2000u32;
    for i in 0..n {
        let name = format!("frag{i:04}");
        let mut src = NamedTempFile::new().unwrap();
        src.as_file_mut().write_all(&payload).unwrap();
        ext.add_file_to(
            &mut dev,
            bigdir,
            name.as_bytes(),
            FileSource::HostPath(src.path().to_path_buf()),
            FileMeta::with_mode(0o644),
        )
        .unwrap();
    }
    ext.flush(&mut dev).unwrap();
    dev.sync().unwrap();

    // Our reader must enumerate every entry — exercises both depth-0 and
    // depth-1 readback.
    let entries = ext.list_inode(&mut dev, bigdir).unwrap();
    let names: std::collections::HashSet<_> = entries
        .iter()
        .map(|e| e.name.clone())
        .filter(|n| n != "." && n != "..")
        .collect();
    assert_eq!(names.len() as u32, n, "fstool ls miscounted");

    // Confirm the inode's extent tree is now depth-1 (otherwise the test
    // wouldn't have exercised the new path). Decode the header directly.
    let frag = ext.read_inode(&mut dev, bigdir).unwrap();
    let bytes = {
        let mut out = [0u8; 60];
        for (i, slot) in frag.block.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&slot.to_le_bytes());
        }
        out
    };
    let magic = u16::from_le_bytes(bytes[0..2].try_into().unwrap());
    assert_eq!(magic, 0xF30A, "extent header magic missing");
    let depth = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
    assert_eq!(
        depth, 1,
        "expected /frag to use depth-1 extent tree (got depth={depth})"
    );

    drop(dev);

    // e2fsck stays clean — depth-1 leaves carry the `ext4_extent_tail`
    // CRC, which is what would fail here if the stamp path is wrong.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck failed on fragmented dir image:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&fsck.stdout),
        String::from_utf8_lossy(&fsck.stderr)
    );
}

/// Build an HTree-indexed directory (`EXT4_INDEX_FL`) and confirm
/// e2fsck accepts it, debugfs sees it as indexed, and our reader
/// enumerates every entry via the legacy linear-scan path that
/// dx_root's fake `.` / `..` façade is meant to support.
#[test]
fn ext4_indexed_directory_passes_e2fsck() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };
    use fstool::fs::ext::FormatOpts;

    let opts = FormatOpts {
        kind: FsKind::Ext4,
        block_size: 4096,
        blocks_count: 16 * 1024,
        inodes_count: 2048,
        journal_blocks: 1024,
        ..FormatOpts::default()
    };
    let tmp = NamedTempFile::new().unwrap();
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    let mut ext = Ext::format_with(&mut dev, &opts).unwrap();

    // 500 entries → ~3 leaves with our 87.5% target fill ratio.
    let names: Vec<String> = (0..500).map(|i| format!("entry_{i:04}")).collect();
    let name_bytes: Vec<&[u8]> = names.iter().map(|s| s.as_bytes()).collect();
    let bigdir = ext
        .add_dir_indexed(
            &mut dev,
            2,
            b"indexed",
            FileMeta::with_mode(0o755),
            &name_bytes,
        )
        .unwrap();

    // Now add the actual files. The router in add_entry_to_dir_block_for
    // hashes each name and lands it in the right leaf.
    for name in &name_bytes {
        let mut src = NamedTempFile::new().unwrap();
        src.as_file_mut().write_all(b"x\n").unwrap();
        ext.add_file_to(
            &mut dev,
            bigdir,
            name,
            FileSource::HostPath(src.path().to_path_buf()),
            FileMeta::with_mode(0o644),
        )
        .unwrap();
    }
    ext.flush(&mut dev).unwrap();
    dev.sync().unwrap();

    // Inode must carry EXT4_INDEX_FL (0x1000).
    let inode = ext.read_inode(&mut dev, bigdir).unwrap();
    assert!(
        inode.flags & 0x1000 != 0,
        "expected EXT4_INDEX_FL on /indexed inode, got flags={:#x}",
        inode.flags
    );

    // Our reader's linear scan must enumerate every name via the
    // `.`/`..` façade at the head of dx_root, even though it doesn't
    // understand the dx_entry table.
    let entries = ext.list_inode(&mut dev, bigdir).unwrap();
    let got: std::collections::HashSet<String> = entries
        .iter()
        .map(|e| e.name.clone())
        .filter(|n| n != "." && n != "..")
        .collect();
    assert_eq!(
        got.len(),
        names.len(),
        "fstool ls miscounted on indexed dir"
    );

    drop(dev);

    // e2fsck must accept the indexed dir. If half-MD4 doesn't match
    // the kernel's, or dx_root is malformed, this is where we find out.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck rejected indexed dir:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&fsck.stdout),
        String::from_utf8_lossy(&fsck.stderr)
    );

    // debugfs sees the directory as indexed (the htree_dump command
    // succeeds only on EXT4_INDEX_FL inodes).
    let dump = Command::new("debugfs")
        .arg("-R")
        .arg("htree_dump /indexed")
        .arg(tmp.path())
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&dump.stdout);
    assert!(
        out.contains("Number of entries") || out.contains("Hash Version") || out.contains("htree:"),
        "debugfs htree_dump didn't recognise /indexed:\n{out}"
    );
}

/// Build a source ext4 image via mke2fs that contains hard links, run
/// `fstool repack` against it, and confirm the destination preserves
/// the hardlink relationship (multiple names sharing one inode with
/// `links_count > 1`) instead of materialising each link as a
/// duplicated file body.
#[test]
fn ext4_repack_preserves_hardlinks() {
    let Some(_) = which("mke2fs") else {
        eprintln!("skipping: mke2fs not installed");
        return;
    };
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };

    // Source host tree with hardlinks. `ln a b` makes `b` a hardlink
    // to `a`; mke2fs's `-d` flag preserves these as ext4 hardlinks
    // (inode-shared dirents).
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(
        srcdir.path().join("primary"),
        b"shared bytes for the hardlink test\n",
    )
    .unwrap();
    std::fs::hard_link(
        srcdir.path().join("primary"),
        srcdir.path().join("alias_a"),
    )
    .unwrap();
    std::fs::hard_link(
        srcdir.path().join("primary"),
        srcdir.path().join("alias_b"),
    )
    .unwrap();

    let src = NamedTempFile::new().unwrap();
    let mk = Command::new("mke2fs")
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
        .arg(src.path())
        .arg("8192")
        .output()
        .unwrap();
    assert!(
        mk.status.success(),
        "mke2fs failed:\n{}",
        String::from_utf8_lossy(&mk.stderr)
    );

    // Sanity-check the source: the three names share one inode.
    let src_ext = {
        let mut dev = FileBackend::open(src.path()).unwrap();
        let ext = Ext::open(&mut dev).unwrap();
        let root = ext.list_inode(&mut dev, 2).unwrap();
        let mut shared_inos = std::collections::HashSet::new();
        for n in ["primary", "alias_a", "alias_b"] {
            let ino = root
                .iter()
                .find(|e| e.name == n)
                .map(|e| e.inode)
                .expect("primary/alias not found in source");
            shared_inos.insert(ino);
        }
        assert_eq!(
            shared_inos.len(),
            1,
            "expected one shared source inode, got {shared_inos:?}"
        );
        *shared_inos.iter().next().unwrap()
    };
    let _ = src_ext;

    // Run repack via fstool. The binary is built by the test harness.
    let dst = NamedTempFile::new().unwrap();
    let bin = std::path::PathBuf::from(env!("CARGO_BIN_EXE_fstool"));
    let out = Command::new(&bin)
        .args(["repack", "--shrink"])
        .arg(src.path())
        .arg(dst.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "fstool repack failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // Open the destination and confirm the three names share one
    // inode with links_count == 3.
    let mut dst_dev = FileBackend::open(dst.path()).unwrap();
    let dst_ext = Ext::open(&mut dst_dev).unwrap();
    let root = dst_ext.list_inode(&mut dst_dev, 2).unwrap();
    let mut dst_inos = std::collections::HashSet::new();
    for n in ["primary", "alias_a", "alias_b"] {
        let ino = root
            .iter()
            .find(|e| e.name == n)
            .map(|e| e.inode)
            .unwrap_or_else(|| panic!("destination missing {n}: {root:?}"));
        dst_inos.insert(ino);
    }
    assert_eq!(
        dst_inos.len(),
        1,
        "expected destination's three names to share one inode, got {dst_inos:?}"
    );
    let shared = *dst_inos.iter().next().unwrap();
    let shared_inode = dst_ext.read_inode(&mut dst_dev, shared).unwrap();
    assert_eq!(
        shared_inode.links_count, 3,
        "shared inode {shared} should have links_count=3, got {}",
        shared_inode.links_count
    );

    drop(dst_dev);

    // e2fsck must be clean on the destination.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(dst.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck rejected hardlink-preserving repack:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&fsck.stdout),
        String::from_utf8_lossy(&fsck.stderr)
    );
}

/// Build a source ext4 image whose biggest file is sparse (4 KiB of
/// real data, then a 240 KiB hole, then another 4 KiB of real data),
/// run `fstool repack --shrink`, and confirm the destination keeps
/// the hole instead of inflating the file to its full dense size.
#[test]
fn ext4_repack_preserves_sparse_files() {
    use std::io::Read;
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };

    // Source: build directly via fstool with sparse=true so the source
    // file's blocks_512 is small even though its logical size is 248 KiB.
    let opts = FormatOpts {
        kind: FsKind::Ext4,
        block_size: 4096,
        blocks_count: 16 * 1024,
        inodes_count: 64,
        journal_blocks: 1024,
        sparse: true,
        ..FormatOpts::default()
    };
    let src_tmp = NamedTempFile::new().unwrap();
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut src_dev = FileBackend::create(src_tmp.path(), size).unwrap();
    let mut src_ext = Ext::format_with(&mut src_dev, &opts).unwrap();

    let mut body = vec![b'A'; 4096];
    body.extend(std::iter::repeat_n(0u8, 240 * 1024));
    body.extend(std::iter::repeat_n(b'B', 4096));
    let payload = NamedTempFile::new().unwrap();
    std::fs::write(payload.path(), &body).unwrap();
    src_ext
        .add_file_to(
            &mut src_dev,
            2,
            b"sparse.bin",
            FileSource::HostPath(payload.path().to_path_buf()),
            FileMeta::with_mode(0o644),
        )
        .unwrap();
    src_ext.flush(&mut src_dev).unwrap();
    src_dev.sync().unwrap();
    drop(src_dev);

    // Confirm source's sparse.bin really has a small blocks_512.
    {
        let mut dev = FileBackend::open(src_tmp.path()).unwrap();
        let ext = Ext::open(&mut dev).unwrap();
        let ino = ext.path_to_inode(&mut dev, "/sparse.bin").unwrap();
        let inode = ext.read_inode(&mut dev, ino).unwrap();
        assert!(
            inode.blocks_512 < 64,
            "source sparse.bin used {} sectors, expected far fewer than dense ({})",
            inode.blocks_512,
            body.len() / 512
        );
    }

    // Repack via the CLI; the repack path now sets sparse=true on the
    // destination Ext.
    let dst = NamedTempFile::new().unwrap();
    let bin = std::path::PathBuf::from(env!("CARGO_BIN_EXE_fstool"));
    let out = Command::new(&bin)
        .args(["repack", "--shrink"])
        .arg(src_tmp.path())
        .arg(dst.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "fstool repack failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // Destination's sparse.bin must still be sparse: blocks_512 small,
    // file content byte-exact.
    let mut dst_dev = FileBackend::open(dst.path()).unwrap();
    let dst_ext = Ext::open(&mut dst_dev).unwrap();
    let ino = dst_ext.path_to_inode(&mut dst_dev, "/sparse.bin").unwrap();
    let inode = dst_ext.read_inode(&mut dst_dev, ino).unwrap();
    assert!(
        inode.blocks_512 < 64,
        "destination sparse.bin used {} sectors after repack, expected sparse layout",
        inode.blocks_512
    );
    let mut got = Vec::new();
    dst_ext
        .open_file_reader(&mut dst_dev, ino)
        .unwrap()
        .read_to_end(&mut got)
        .unwrap();
    assert_eq!(got, body, "sparse.bin content mismatch after repack");
    drop(dst_dev);

    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(dst.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck rejected sparse-preserving repack:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&fsck.stdout),
        String::from_utf8_lossy(&fsck.stderr)
    );
}
