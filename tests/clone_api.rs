#![cfg(unix)]
//! Phase 3a — the cross-backend reflink / clone API surface.
//!
//! `Filesystem::clone_file` works on every writable backend: by default
//! it byte-copies via a tempfile (preserving metadata best-effort
//! through `getattr`), and reflink-capable backends override to share
//! extents. `clone_range` only succeeds on backends that natively
//! support sub-file extent sharing — `Unsupported` everywhere else.
//!
//! No backend ships native reflink support yet (XFS gets it in Phase
//! 3b), so this file pins the *fallback* contract and the
//! `clone_capability` default. The XFS reflink tests will land in
//! `tests/reflink_external.rs` alongside the writer changes.

use std::io::{Cursor, Read};
use std::path::Path;

use fstool::block::{BlockDevice, FileBackend};
use fstool::fs::{CloneCapability, FileMeta, FileSource, Filesystem};
use fstool::inspect;
use tempfile::NamedTempFile;

/// Default `clone_file` fallback round-trips file content on every
/// writable backend. Exercises ext4 because (a) ext4 is the canonical
/// reopen-mutate backend and (b) its `getattr` is fully populated, so
/// the fallback preserves mode / uid / gid via the read-back path.
#[test]
fn ext4_clone_file_default_byte_copy() {
    use fstool::fs::ext::{Ext, FormatOpts, FsKind};
    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts {
        kind: FsKind::Ext4,
        blocks_count: 8192,
        inodes_count: 64,
        journal_blocks: 1024,
        sparse: true,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let body: Vec<u8> = (0..4321).map(|i| (i & 0xFF) as u8).collect();
    {
        let mut dev = FileBackend::create(tmp.path(), size).unwrap();
        let mut fs: Box<dyn Filesystem> = Box::new(Ext::format_with(&mut dev, &opts).unwrap());
        // Plant a source file with a specific mode so the metadata
        // round-trip is observable.
        fs.create_file(
            &mut dev,
            Path::new("/src.bin"),
            FileSource::Reader {
                reader: Box::new(Cursor::new(body.clone())),
                len: body.len() as u64,
            },
            FileMeta {
                mode: 0o640,
                uid: 1000,
                gid: 1000,
                mtime: 1_700_000_000,
                atime: 1_700_000_000,
                ctime: 1_700_000_000,
            },
        )
        .unwrap();

        // The default `clone_capability` for ext4 is None — no extent
        // sharing — so this exercises the byte-copy fallback path.
        assert_eq!(fs.clone_capability(), CloneCapability::None);
        fs.clone_file(&mut dev, Path::new("/src.bin"), Path::new("/dst.bin"))
            .unwrap();
        fs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }

    // Reopen, confirm dst has the same bytes + the metadata survived.
    let mut dev = FileBackend::open(tmp.path()).unwrap();
    let mut fs = inspect::open(&mut dev).unwrap();
    let mut got = Vec::new();
    {
        let mut r = fs.read_file(&mut dev, Path::new("/dst.bin")).unwrap();
        r.read_to_end(&mut got).unwrap();
    }
    assert_eq!(got, body, "cloned content mismatch");
    let attrs = fs.getattr(&mut dev, Path::new("/dst.bin")).unwrap();
    assert_eq!(attrs.mode & 0o777, 0o640, "mode not preserved");
    assert_eq!(attrs.uid, 1000, "uid not preserved");
    assert_eq!(attrs.gid, 1000, "gid not preserved");
    assert_eq!(attrs.mtime, 1_700_000_000, "mtime not preserved");
}

/// `clone_range` defaults to `Unsupported` on every backend until
/// Phase 3b lands XFS's REFLINK opt-in. Pin the contract so we notice
/// if a backend accidentally claims sub-file reflink without the
/// refcount-btree machinery to back it.
#[test]
fn clone_range_default_is_unsupported() {
    use fstool::fs::ext::{Ext, FormatOpts, FsKind};
    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts {
        kind: FsKind::Ext4,
        blocks_count: 8192,
        inodes_count: 64,
        journal_blocks: 1024,
        sparse: true,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(tmp.path(), size).unwrap();
    let mut fs: Box<dyn Filesystem> = Box::new(Ext::format_with(&mut dev, &opts).unwrap());
    fs.create_file(
        &mut dev,
        Path::new("/a"),
        FileSource::Zero(4096),
        FileMeta::default(),
    )
    .unwrap();
    fs.create_file(
        &mut dev,
        Path::new("/b"),
        FileSource::Zero(4096),
        FileMeta::default(),
    )
    .unwrap();
    let err = fs
        .clone_range(&mut dev, Path::new("/a"), 0, Path::new("/b"), 0, 4096)
        .expect_err("clone_range default must reject");
    assert!(
        matches!(err, fstool::Error::Unsupported(_)),
        "expected Unsupported, got: {err:?}"
    );
}

/// AnyFs::clone_capability + clone_file routes correctly through the
/// dispatch. Same fallback semantics, but exercised through the
/// generic enum a CLI consumer would hold.
#[test]
fn anyfs_clone_file_routes_through_default_fallback() {
    use fstool::fs::ext::{Ext, FormatOpts, FsKind};
    let tmp = NamedTempFile::new().unwrap();
    let opts = FormatOpts {
        kind: FsKind::Ext4,
        blocks_count: 8192,
        inodes_count: 64,
        journal_blocks: 1024,
        sparse: true,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let body = b"clone via AnyFs\n";
    {
        let mut dev = FileBackend::create(tmp.path(), size).unwrap();
        let mut fs: Box<dyn Filesystem> = Box::new(Ext::format_with(&mut dev, &opts).unwrap());
        fs.create_file(
            &mut dev,
            Path::new("/seed.txt"),
            FileSource::Reader {
                reader: Box::new(Cursor::new(body.to_vec())),
                len: body.len() as u64,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        dev.sync().unwrap();
    }

    let mut dev = FileBackend::open(tmp.path()).unwrap();
    let mut any = inspect::AnyFs::open(&mut dev).unwrap();
    assert_eq!(any.clone_capability(), CloneCapability::None);
    any.clone_file(&mut dev, "/seed.txt", "/clone.txt").unwrap();
    any.flush(&mut dev).unwrap();
    dev.sync().unwrap();
    drop(dev);

    // Confirm the cloned file is observable through a fresh reopen.
    let mut dev = FileBackend::open(tmp.path()).unwrap();
    let mut fs = inspect::open(&mut dev).unwrap();
    let mut got = Vec::new();
    {
        let mut r = fs.read_file(&mut dev, Path::new("/clone.txt")).unwrap();
        r.read_to_end(&mut got).unwrap();
    }
    assert_eq!(got, body);
}
