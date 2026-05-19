#![cfg(unix)]
//! SquashFS end-to-end validation against the native `squashfs-tools`.
//!
//! Three scenarios:
//!
//! 1. fstool writer → `unsquashfs` (list + extract + byte-compare).
//! 2. `mksquashfs` → fstool reader (open + walk + byte-compare).
//! 3. Compression matrix: rebuild scenario 1 with every codec fstool
//!    supports at the configured feature set and confirm `unsquashfs -lc`
//!    accepts each.
//!
//! Every test gates on the presence of the relevant native tools and
//! returns early with `eprintln!("skipping: …")` when missing, so the
//! suite stays green on minimal CI images.

use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::Command;

use fstool::block::{BlockDevice, FileBackend};
use fstool::fs::DeviceKind;
use fstool::fs::squashfs::{Compression, EntryMeta, FormatOpts, Squashfs, Xattr};
use fstool::fs::{FileSource, ReadSeek};

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

/// Confirm a native tool advertises its version. Some distros stub the
/// binary, so we additionally require non-empty output.
fn tool_present(name: &str, version_arg: &str) -> bool {
    if which(name).is_none() {
        return false;
    }
    match Command::new(name).arg(version_arg).output() {
        Ok(out) => !out.stdout.is_empty() || !out.stderr.is_empty(),
        Err(_) => false,
    }
}

/// Truncate `path` to exactly `len` bytes — the writer addresses the
/// device through `write_at` so any sparse tail beyond the superblock's
/// `bytes_used` confuses `unsquashfs`.
fn trim_image_to(path: &Path, len: u64) {
    let f = fs::OpenOptions::new().write(true).open(path).unwrap();
    f.set_len(len).unwrap();
}

/// Allocate a fat, sparse FileBackend and run `builder` against it,
/// then trim the file down to the superblock's `bytes_used`.
fn build_image<F: FnOnce(&mut FileBackend, &mut Squashfs)>(
    path: &Path,
    compression: Compression,
    builder: F,
) -> u64 {
    // 16 MiB sparse — plenty for any of these fixture trees, even with
    // worst-case overhead from the metablock + fragment tables.
    let capacity: u64 = 16 * 1024 * 1024;
    let mut dev = FileBackend::create(path, capacity).unwrap();
    let mut sq = Squashfs::format(
        &mut dev,
        &FormatOpts {
            block_size: 4096,
            compression,
        },
    )
    .unwrap();
    builder(&mut dev, &mut sq);
    sq.flush(&mut dev).unwrap();
    let used = sq.total_bytes();
    dev.sync().unwrap();
    drop(dev);
    trim_image_to(path, used);
    used
}

/// Populate the writer with a fixed, richly-typed tree. Every test that
/// goes through the writer uses this so we exercise the same surface:
/// nested dirs, files, symlinks, hardlinks, every device-node kind, and
/// xattrs.
fn populate_rich_tree(dev: &mut FileBackend, sq: &mut Squashfs) {
    sq.create_dir(
        dev,
        "/etc",
        EntryMeta {
            mode: 0o755,
            uid: 0,
            gid: 0,
            mtime: 100,
        },
        Vec::new(),
    )
    .unwrap();
    sq.create_file(
        dev,
        "/etc/hosts",
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(b"127.0.0.1 localhost\n".to_vec()))
                as Box<dyn ReadSeek + Send>,
            len: 20,
        },
        EntryMeta {
            mode: 0o644,
            uid: 0,
            gid: 0,
            mtime: 200,
        },
        vec![Xattr {
            key: "user.kind".into(),
            value: b"file".to_vec(),
        }],
    )
    .unwrap();
    sq.create_file(
        dev,
        "/etc/greeting",
        FileSource::Reader {
            reader: Box::new(std::io::Cursor::new(b"hi there\n".to_vec()))
                as Box<dyn ReadSeek + Send>,
            len: 9,
        },
        EntryMeta {
            mode: 0o644,
            uid: 0,
            gid: 0,
            mtime: 201,
        },
        Vec::new(),
    )
    .unwrap();
    sq.create_hardlink(dev, "/etc/hosts", "/etc/hosts.bak")
        .unwrap();
    sq.create_symlink(
        dev,
        "/sym",
        "etc/hosts",
        EntryMeta {
            mode: 0o777,
            uid: 0,
            gid: 0,
            mtime: 300,
        },
        Vec::new(),
    )
    .unwrap();
    sq.create_device(
        dev,
        "/dev/null",
        DeviceKind::Char,
        1,
        3,
        EntryMeta {
            mode: 0o666,
            uid: 0,
            gid: 0,
            mtime: 400,
        },
        Vec::new(),
    )
    .unwrap();
    sq.create_device(
        dev,
        "/dev/sda",
        DeviceKind::Block,
        8,
        0,
        EntryMeta {
            mode: 0o600,
            uid: 0,
            gid: 0,
            mtime: 500,
        },
        Vec::new(),
    )
    .unwrap();
    sq.create_device(
        dev,
        "/run/fifo",
        DeviceKind::Fifo,
        0,
        0,
        EntryMeta {
            mode: 0o600,
            uid: 0,
            gid: 0,
            mtime: 600,
        },
        Vec::new(),
    )
    .unwrap();
    sq.create_device(
        dev,
        "/run/sock",
        DeviceKind::Socket,
        0,
        0,
        EntryMeta {
            mode: 0o600,
            uid: 0,
            gid: 0,
            mtime: 700,
        },
        vec![Xattr {
            key: "user.purpose".into(),
            value: b"unix-socket".to_vec(),
        }],
    )
    .unwrap();
}

/// Map a [`Compression`] to its `mksquashfs -comp` argument and the
/// matching cfg-feature flag. Returns `None` for codecs we don't know
/// how to drive from the command line.
fn codec_info(c: Compression) -> Option<(&'static str, bool)> {
    match c {
        Compression::Gzip => Some(("gzip", cfg!(feature = "gzip"))),
        Compression::Xz => Some(("xz", cfg!(feature = "xz"))),
        Compression::Lzma => Some(("lzma", cfg!(feature = "lzma"))),
        Compression::Lz4 => Some(("lz4", cfg!(feature = "lz4"))),
        Compression::Zstd => Some(("zstd", cfg!(feature = "zstd"))),
        Compression::Lzo => Some(("lzo", cfg!(feature = "lzo"))),
        _ => None,
    }
}

/// Cheap probe: ask `mksquashfs` whether it accepts this codec. Some
/// distro builds drop `lzo` or `lzma` for licensing reasons.
fn mksquashfs_supports_codec(codec: &str) -> bool {
    let out = match Command::new("mksquashfs")
        .args(["-help-comp", codec])
        .output()
    {
        Ok(o) => o,
        Err(_) => return false,
    };
    // mksquashfs prints "Compressor <codec>:" on success, and an error
    // (and non-zero status) when the codec is missing.
    out.status.success()
        && (String::from_utf8_lossy(&out.stdout)
            .to_lowercase()
            .contains(codec)
            || String::from_utf8_lossy(&out.stderr)
                .to_lowercase()
                .contains(codec))
}

// ---------------------------------------------------------------------------
// 1) Writer → unsquashfs.
// ---------------------------------------------------------------------------

/// fstool writes a rich tree, `unsquashfs -lc` lists it, then
/// `unsquashfs -d` extracts it and we diff each regular file byte-for-byte.
#[test]
fn writer_image_passes_unsquashfs_round_trip() {
    if !tool_present("unsquashfs", "-version") {
        eprintln!("skipping: unsquashfs not installed");
        return;
    }

    let workdir = tempfile::tempdir().unwrap();
    let img = workdir.path().join("rich.sqfs");

    // Pick whatever codec fstool was compiled with — gzip is the safest
    // bet (default feature) and is universally supported by squashfs-tools.
    let codec = if cfg!(feature = "gzip") {
        Compression::Gzip
    } else if cfg!(feature = "zstd") {
        Compression::Zstd
    } else {
        // No codec compiled in: use Unknown(0) which the writer emits as
        // uncompressed metablocks. unsquashfs handles uncompressed too.
        Compression::Unknown(0)
    };
    build_image(&img, codec, populate_rich_tree);

    // ---- `unsquashfs -lc` lists every file + empty dir. ----
    let out = Command::new("unsquashfs")
        .arg("-lc")
        .arg(&img)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "unsquashfs -lc failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let listing = String::from_utf8_lossy(&out.stdout);
    // -lc shows files + empty dirs only; our `/etc`, `/dev`, `/run` are
    // all non-empty so we focus on the leaf entries we wrote.
    for must_have in [
        "etc/hosts",
        "etc/hosts.bak",
        "etc/greeting",
        "sym",
        "dev/null",
        "dev/sda",
        "run/fifo",
        "run/sock",
    ] {
        assert!(
            listing.contains(must_have),
            "unsquashfs -lc missed {must_have:?}:\n{listing}"
        );
    }

    // ---- `unsquashfs -d` extracts the tree to a fresh directory. ----
    // Pass `-no-xattrs` and `-i` (ignore-errors) so the test works as a
    // non-root user: device-node creation and `system.*`-xattr writes
    // both need privileges that CI runners don't have. The regular
    // files / dirs / symlinks / hardlinks still get extracted, and the
    // assertions below only touch those.
    let extract = workdir.path().join("extract");
    let out = Command::new("unsquashfs")
        .arg("-no-xattrs")
        .arg("-ignore-errors")
        .arg("-d")
        .arg(&extract)
        .arg(&img)
        .output()
        .unwrap();
    if !out.status.success() {
        // Some unsquashfs versions don't have `-ignore-errors`; fall
        // back to a plain `-no-xattrs` extract and tolerate exit code
        // 1 since device-node failures still leave the regular files
        // behind.
        let _ = std::fs::remove_dir_all(&extract);
        let _ = Command::new("unsquashfs")
            .arg("-no-xattrs")
            .arg("-d")
            .arg(&extract)
            .arg(&img)
            .output();
    }
    assert!(
        extract.exists(),
        "unsquashfs -d produced no output dir:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    // Byte-perfect file content via extraction.
    assert_eq!(
        fs::read(extract.join("etc/hosts")).unwrap(),
        b"127.0.0.1 localhost\n"
    );
    assert_eq!(
        fs::read(extract.join("etc/greeting")).unwrap(),
        b"hi there\n"
    );
    // hardlink shares the same content (and ideally the same inode, but
    // we don't assert that — `unsquashfs` extracts hardlinks as separate
    // copies on some setups).
    assert_eq!(
        fs::read(extract.join("etc/hosts.bak")).unwrap(),
        b"127.0.0.1 localhost\n"
    );
    // Symlink target preserved.
    let link_target = fs::read_link(extract.join("sym")).unwrap();
    assert_eq!(link_target.to_string_lossy(), "etc/hosts");
    // Symlink resolves to the right content too.
    assert_eq!(
        fs::read(extract.join("sym")).unwrap(),
        b"127.0.0.1 localhost\n"
    );
}

// ---------------------------------------------------------------------------
// 2) mksquashfs → fstool reader.
// ---------------------------------------------------------------------------

/// Recursively walk a SquashFS root via `Squashfs::list_path` and verify
/// every regular file's content against the matching source file.
fn assert_fstool_tree_matches_disk(
    sq: &Squashfs,
    dev: &mut dyn BlockDevice,
    src_root: &Path,
    sq_path: &str,
) {
    let entries = sq.list_path(dev, sq_path).unwrap();
    let listed_names: HashSet<String> = entries.iter().map(|e| e.name.clone()).collect();
    let on_disk: HashSet<String> = fs::read_dir(src_root)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        listed_names, on_disk,
        "fstool listing mismatch at {sq_path:?}: fstool={listed_names:?} disk={on_disk:?}"
    );

    for e in entries {
        let child_sq = if sq_path == "/" {
            format!("/{}", e.name)
        } else {
            format!("{sq_path}/{}", e.name)
        };
        let child_disk = src_root.join(&e.name);
        let md = fs::symlink_metadata(&child_disk).unwrap();
        let ty = md.file_type();
        if ty.is_dir() {
            assert_eq!(
                e.kind,
                fstool::fs::EntryKind::Dir,
                "{child_sq:?} kind mismatch"
            );
            assert_fstool_tree_matches_disk(sq, dev, &child_disk, &child_sq);
        } else if ty.is_file() {
            assert_eq!(
                e.kind,
                fstool::fs::EntryKind::Regular,
                "{child_sq:?} kind mismatch"
            );
            let want = fs::read(&child_disk).unwrap();
            let mut got = Vec::new();
            sq.open_file_reader(dev, &child_sq)
                .unwrap()
                .read_to_end(&mut got)
                .unwrap();
            assert_eq!(got, want, "byte mismatch for {child_sq:?}");
        } else if ty.is_symlink() {
            assert_eq!(
                e.kind,
                fstool::fs::EntryKind::Symlink,
                "{child_sq:?} kind mismatch"
            );
            let want = fs::read_link(&child_disk)
                .unwrap()
                .to_string_lossy()
                .into_owned();
            let got = sq.read_symlink(dev, &child_sq).unwrap();
            assert_eq!(got, want, "symlink target mismatch for {child_sq:?}");
        }
    }
}

/// Build a small tree on disk, run `mksquashfs` (`-comp gzip -no-xattrs`),
/// then re-open through `Squashfs::open` and confirm every byte round-trips.
#[test]
fn mksquashfs_image_opens_with_fstool_gzip() {
    if !tool_present("mksquashfs", "-version") {
        eprintln!("skipping: mksquashfs not installed");
        return;
    }
    if !cfg!(feature = "gzip") {
        eprintln!("skipping: fstool built without gzip feature");
        return;
    }

    let workdir = tempfile::tempdir().unwrap();
    let src = workdir.path().join("srctree");
    fs::create_dir_all(src.join("dir1/dir2")).unwrap();
    fs::write(src.join("top.txt"), b"top-level content\n").unwrap();
    fs::write(src.join("dir1/mid.bin"), b"middle\x00\x01\x02bin\n").unwrap();
    // Larger-than-one-fragment file to exercise full-block reads.
    let big: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
    fs::write(src.join("dir1/dir2/big.bin"), &big).unwrap();
    std::os::unix::fs::symlink("../top.txt", src.join("dir1/link")).unwrap();

    let img = workdir.path().join("from-mksquashfs.sqfs");
    let out = Command::new("mksquashfs")
        .arg(&src)
        .arg(&img)
        .args(["-comp", "gzip", "-no-xattrs", "-noappend", "-quiet"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "mksquashfs failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let mut dev = FileBackend::open(&img).unwrap();
    let sq = Squashfs::open(&mut dev).unwrap();
    assert_eq!(sq.compression(), Compression::Gzip);
    assert_fstool_tree_matches_disk(&sq, &mut dev, &src, "/");
}

/// Same as the gzip flavour but with zstd, exercising fstool's zstd
/// decompressor against a real `mksquashfs` zstd image.
#[test]
fn mksquashfs_image_opens_with_fstool_zstd() {
    if !tool_present("mksquashfs", "-version") {
        eprintln!("skipping: mksquashfs not installed");
        return;
    }
    if !cfg!(feature = "zstd") {
        eprintln!("skipping: fstool built without zstd feature");
        return;
    }
    if !mksquashfs_supports_codec("zstd") {
        eprintln!("skipping: local mksquashfs has no zstd compressor");
        return;
    }

    let workdir = tempfile::tempdir().unwrap();
    let src = workdir.path().join("srctree");
    fs::create_dir_all(src.join("nested")).unwrap();
    fs::write(src.join("nested/a.txt"), b"zstd payload A\n").unwrap();
    fs::write(
        src.join("nested/b.txt"),
        b"zstd payload B with more bytes\n",
    )
    .unwrap();

    let img = workdir.path().join("from-mksquashfs-zstd.sqfs");
    let out = Command::new("mksquashfs")
        .arg(&src)
        .arg(&img)
        .args(["-comp", "zstd", "-no-xattrs", "-noappend", "-quiet"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "mksquashfs -comp zstd failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let mut dev = FileBackend::open(&img).unwrap();
    let sq = Squashfs::open(&mut dev).unwrap();
    assert_eq!(sq.compression(), Compression::Zstd);
    assert_fstool_tree_matches_disk(&sq, &mut dev, &src, "/");
}

// ---------------------------------------------------------------------------
// 3) Compression matrix: writer → unsquashfs -lc, one image per codec.
// ---------------------------------------------------------------------------

/// Drive the matrix from a single body so each codec gets identical
/// inputs. Skip any codec that wasn't compiled into fstool.
fn unsquashfs_accepts_codec(c: Compression) {
    if !tool_present("unsquashfs", "-version") {
        eprintln!("skipping: unsquashfs not installed");
        return;
    }
    let Some((name, enabled)) = codec_info(c) else {
        eprintln!("skipping: codec not driveable from tests");
        return;
    };
    if !enabled {
        eprintln!("skipping: fstool built without {name} feature");
        return;
    }

    let workdir = tempfile::tempdir().unwrap();
    let img = workdir.path().join(format!("rich-{name}.sqfs"));
    build_image(&img, c, populate_rich_tree);

    let out = Command::new("unsquashfs")
        .arg("-lc")
        .arg(&img)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "unsquashfs -lc rejected {name} image:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let listing = String::from_utf8_lossy(&out.stdout);
    // Sanity: at least one of our leaf entries shows up. Some local
    // unsquashfs builds may not link a given codec — when that happens
    // they exit non-zero, which the assert above already catches.
    assert!(
        listing.contains("etc/hosts"),
        "unsquashfs -lc listing for {name} missed etc/hosts:\n{listing}"
    );
}

#[test]
fn writer_image_unsquashfs_lc_gzip() {
    unsquashfs_accepts_codec(Compression::Gzip);
}

#[test]
fn writer_image_unsquashfs_lc_xz() {
    unsquashfs_accepts_codec(Compression::Xz);
}

#[test]
fn writer_image_unsquashfs_lc_lz4() {
    unsquashfs_accepts_codec(Compression::Lz4);
}

#[test]
fn writer_image_unsquashfs_lc_zstd() {
    unsquashfs_accepts_codec(Compression::Zstd);
}

// Mainline squashfs-tools v4 dropped legacy LZMA support; only LZMA2
// (under the `xz` compressor id) is current. unsquashfs builds without
// the legacy-LZMA codec can't read these images even though the
// writer's framing is spec-conformant. Ignored until we either drop
// the legacy LZMA writer or pin a build of unsquashfs that supports it.
#[test]
#[ignore]
fn writer_image_unsquashfs_lc_lzma() {
    unsquashfs_accepts_codec(Compression::Lzma);
}

#[test]
fn writer_image_unsquashfs_lc_lzo() {
    unsquashfs_accepts_codec(Compression::Lzo);
}
