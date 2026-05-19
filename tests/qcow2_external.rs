//! qcow2 backend validation against real `qemu-img`-produced images.
//! Each test skips silently when `qemu-img` isn't on PATH.

use std::io::Read as _;
use std::process::Command;

use fstool::block::{BlockDevice, Qcow2Backend};
use tempfile::NamedTempFile;

fn which(tool: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {tool}"))
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `qemu-img create -f qcow2 …` produces an empty image whose virtual
/// size and cluster_size we should parse correctly.
#[test]
fn opens_qemu_img_created_image() {
    if !which("qemu-img") {
        eprintln!("skipping: qemu-img not installed");
        return;
    }
    let tmp = NamedTempFile::new().unwrap();
    let out = Command::new("qemu-img")
        .args(["create", "-q", "-f", "qcow2"])
        .arg(tmp.path())
        .arg("64M")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "qemu-img create failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let back = Qcow2Backend::open(tmp.path()).unwrap();
    assert_eq!(back.total_size(), 64 * 1024 * 1024);
    assert_eq!(back.header().cluster_size(), 65536);
    // qemu-img defaults to v3 (`compat=1.1`).
    assert_eq!(back.header().version, 3);
}

/// Read-back invariant: write a pattern into a raw image, convert it to
/// qcow2 via qemu-img, and read it through Qcow2Backend. Bytes must
/// match the original pattern.
#[test]
fn read_back_pattern_via_qemu_img_convert() {
    if !which("qemu-img") {
        eprintln!("skipping: qemu-img not installed");
        return;
    }

    // Build a 4 MiB raw image with a known pattern at a few offsets.
    let raw = NamedTempFile::new().unwrap();
    {
        use std::io::Write as _;
        let mut f = std::fs::File::create(raw.path()).unwrap();
        f.set_len(4 * 1024 * 1024).unwrap();
        f.write_all(b"hello qcow2 reader\n").unwrap();
        // Pattern straddling a 64 KiB cluster boundary.
        use std::io::Seek as _;
        use std::io::SeekFrom;
        f.seek(SeekFrom::Start(65500)).unwrap();
        f.write_all(&[0xAB; 200]).unwrap();
        // Pattern in the middle of a cluster.
        f.seek(SeekFrom::Start(2 * 1024 * 1024)).unwrap();
        f.write_all(b"halfway through\n").unwrap();
        f.sync_all().unwrap();
    }

    // Convert raw → qcow2.
    let qcow = NamedTempFile::new().unwrap();
    let out = Command::new("qemu-img")
        .args(["convert", "-f", "raw", "-O", "qcow2"])
        .arg(raw.path())
        .arg(qcow.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "qemu-img convert failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Read through Qcow2Backend; bytes should match what we wrote.
    let mut back = Qcow2Backend::open(qcow.path()).unwrap();
    assert_eq!(back.total_size(), 4 * 1024 * 1024);

    let mut head = [0u8; 32];
    back.read_at(0, &mut head).unwrap();
    assert_eq!(&head[..19], b"hello qcow2 reader\n");

    let mut straddle = [0u8; 200];
    back.read_at(65500, &mut straddle).unwrap();
    assert!(straddle.iter().all(|&b| b == 0xAB));

    let mut mid = [0u8; 16];
    back.read_at(2 * 1024 * 1024, &mut mid).unwrap();
    assert_eq!(&mid, b"halfway through\n");

    // Unallocated tail reads as zeros.
    let mut tail = [0xffu8; 4096];
    back.read_at(3 * 1024 * 1024, &mut tail).unwrap();
    assert!(tail.iter().all(|&b| b == 0), "tail should be zero");

    // Stream the whole thing via Read.
    use std::io::Seek as _;
    use std::io::SeekFrom;
    back.seek(SeekFrom::Start(0)).unwrap();
    let mut all = Vec::new();
    back.read_to_end(&mut all).unwrap();
    assert_eq!(all.len(), 4 * 1024 * 1024);
    assert_eq!(&all[..19], b"hello qcow2 reader\n");
}

/// fstool::block::open_image dispatches to Qcow2Backend on qcow2 magic.
#[test]
fn open_image_dispatches_to_qcow2() {
    if !which("qemu-img") {
        eprintln!("skipping: qemu-img not installed");
        return;
    }
    let tmp = NamedTempFile::new().unwrap();
    Command::new("qemu-img")
        .args(["create", "-q", "-f", "qcow2"])
        .arg(tmp.path())
        .arg("32M")
        .output()
        .unwrap();

    let mut dev = fstool::block::open_image(tmp.path()).unwrap();
    assert_eq!(dev.total_size(), 32 * 1024 * 1024);
    // Read returns zeros.
    let mut buf = [0xffu8; 1024];
    dev.read_at(0, &mut buf).unwrap();
    assert!(buf.iter().all(|&b| b == 0));
}
