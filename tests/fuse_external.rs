//! External validation for the FUSE adapter — actually mount an
//! image on a background thread, then exercise the kernel
//! `read`/`getattr`/`readdir` paths through `std::fs` against the
//! mountpoint. This closes the "untested in motion" gap: the unit
//! tests in `src/fuse_adapter.rs` exercise the trait wiring, but
//! only this file proves the adapter actually answers the kernel
//! over `/dev/fuse`.
//!
//! Gated on:
//!
//! * `target_os = "linux"` — `fuser` 0.16 only builds the libfuse
//!   variant on Linux/macOS, and our CI runs Linux.
//! * `feature = "fuse"` — the adapter itself is opt-in.
//!
//! Skips silently when the host lacks usable FUSE: no `/dev/fuse`,
//! no `fusermount3` on `PATH`, or the user can't open the device
//! (which catches sandboxed CI runners that don't expose FUSE).

#![cfg(all(target_os = "linux", feature = "fuse"))]

use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use fstool::block::FileBackend;
use fstool::fs::ext::{Ext, FormatOpts};
use fstool::fs::{FileMeta, FileSource};
use fstool::fuse_adapter::FstoolFs;
use tempfile::{NamedTempFile, TempDir};

/// True iff the host has the bits we need to actually mount: the
/// `/dev/fuse` device must be openable for read/write, and
/// `fusermount3` must be on PATH (libfuse calls it to wire up the
/// mount). On a sandboxed CI runner either can be missing.
fn fuse_usable() -> bool {
    use std::fs::OpenOptions;
    if OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/fuse")
        .is_err()
    {
        return false;
    }
    Command::new("sh")
        .arg("-c")
        .arg("command -v fusermount3 || command -v fusermount")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build a small ext2 image at `path` populated with a few seeded
/// files we can verify against once mounted. Returns the
/// `(path, contents)` pairs we put in (relative to the mount
/// point) so the test can assert on them without re-hardcoding.
fn build_seed_image(path: &Path) -> Vec<(&'static str, &'static [u8])> {
    let opts = FormatOpts {
        inodes_count: 64,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(path, size).expect("create image");
    let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext2");

    let seeds: Vec<(&'static str, &'static [u8])> = vec![
        ("hello.txt", b"hello from fuse\n"),
        ("greeting.txt", b"konnichiwa\n"),
    ];

    for (name, body) in &seeds {
        let mut src = NamedTempFile::new().expect("seed src");
        src.as_file_mut().write_all(body).expect("write seed");
        ext.add_file_to(
            &mut dev,
            2, // root inode
            name.as_bytes(),
            FileSource::HostPath(src.path().to_path_buf()),
            FileMeta {
                mode: 0o644,
                mtime: 0,
                ..Default::default()
            },
        )
        .expect("add seed file");
        // Keep the temp file alive until after add_file_to copied
        // it — explicit drop here so it's obvious in the diff.
        drop(src);
    }

    // /etc/conf inside a subdir, so we exercise readdir on a
    // non-root path through FUSE.
    let etc_ino = ext
        .add_dir_to(&mut dev, 2, b"etc", FileMeta::with_mode(0o755))
        .expect("mkdir /etc");
    let mut conf_src = NamedTempFile::new().expect("conf src");
    conf_src
        .as_file_mut()
        .write_all(b"answer=42\n")
        .expect("write conf");
    ext.add_file_to(
        &mut dev,
        etc_ino,
        b"conf",
        FileSource::HostPath(conf_src.path().to_path_buf()),
        FileMeta::with_mode(0o644),
    )
    .expect("add /etc/conf");
    drop(conf_src);

    ext.flush(&mut dev).expect("flush ext");
    {
        use fstool::block::BlockDevice;
        dev.sync().expect("sync");
    }
    drop(dev);

    seeds
}

/// Wait for the mountpoint to actually carry the FUSE filesystem.
/// `spawn_mount2` returns once the kernel has accepted the mount,
/// but a brief window can exist before our adapter has answered
/// the first `lookup` — poll the readdir for up to 5 s.
fn wait_until_mounted(mountpoint: &Path) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if let Ok(mut rd) = std::fs::read_dir(mountpoint)
            && rd.next().is_some()
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("mountpoint never became readable within 5s");
}

#[test]
fn fuse_kernel_roundtrip_ext2() {
    if !fuse_usable() {
        eprintln!("skipping: /dev/fuse not usable or fusermount missing");
        return;
    }

    // Build the backing image as a NamedTempFile so it survives
    // for the whole test; the mountpoint is a TempDir.
    let img = NamedTempFile::new().expect("img tempfile");
    let seeds = build_seed_image(img.path());
    let mountdir = TempDir::new().expect("mount tempdir");
    let mountpoint = mountdir.path().to_path_buf();

    // Open the freshly-built image with its block device + ext FS
    // and hand both to the adapter. `Box<dyn ... + Send>` is the
    // bound the spawn-mount path requires.
    let dev: Box<dyn fstool::block::BlockDevice + Send> =
        Box::new(FileBackend::open(img.path()).expect("open image"));
    let mut dev = dev;
    let ext = Ext::open(dev.as_mut()).expect("open ext");
    let fs: Box<dyn fstool::fs::Filesystem + Send> = Box::new(ext);

    let session = FstoolFs::new(fs, dev, "ext")
        .spawn_mount(&mountpoint)
        .expect("spawn_mount");

    wait_until_mounted(&mountpoint);

    // Each seeded file should be readable byte-for-byte through
    // the kernel via `std::fs::read`.
    for (name, body) in &seeds {
        let path = mountpoint.join(name);
        let got = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("read {} via FUSE failed: {e}", path.display()));
        assert_eq!(&got[..], *body, "content mismatch for {}", path.display());
    }

    // readdir at the root should surface our two top-level files
    // plus the /etc directory we created.
    let names: std::collections::HashSet<String> = std::fs::read_dir(&mountpoint)
        .expect("readdir root")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    for want in ["hello.txt", "greeting.txt", "etc"] {
        assert!(
            names.contains(want),
            "missing {want} in root listing: {names:?}"
        );
    }

    // getattr through the kernel — metadata() goes through FUSE
    // GETATTR which exercises a different code path than READDIR.
    let meta = std::fs::metadata(mountpoint.join("hello.txt")).expect("stat hello.txt");
    assert!(meta.is_file(), "hello.txt not a regular file");
    assert_eq!(meta.len() as usize, b"hello from fuse\n".len());

    // Subdirectory readdir + read.
    let conf = std::fs::read(mountpoint.join("etc/conf")).expect("read /etc/conf");
    assert_eq!(&conf[..], b"answer=42\n");

    // Drop the session to unmount; the temp dir cleans up on
    // scope exit. Order matters — TempDir's drop tries to remove
    // the mountpoint, which fails if the FS is still mounted.
    drop(session);

    // Give the kernel a beat to actually tear down before TempDir
    // tries to rmdir; 200 ms is well over what the autounmount
    // path needs.
    std::thread::sleep(Duration::from_millis(200));
}
