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

/// Qcow2Backend::create makes a fresh image that qemu-img validates.
#[test]
fn create_then_qemu_img_check() {
    if !which("qemu-img") {
        eprintln!("skipping: qemu-img not installed");
        return;
    }
    let tmp = NamedTempFile::new().unwrap();
    {
        let mut back = Qcow2Backend::create(tmp.path(), 64 * 1024 * 1024, 65536).unwrap();
        // Write a few patterns through the allocator.
        back.write_at(0, b"hello fresh qcow2\n").unwrap();
        back.write_at(1024 * 1024, &[0xCDu8; 128]).unwrap();
        back.write_at(63 * 1024 * 1024, &[0xEFu8; 4096]).unwrap();
        back.sync().unwrap();
    }

    // qemu-img info: parses as a real qcow2 v3 with the expected size.
    let info = Command::new("qemu-img")
        .args(["info", "--output=json"])
        .arg(tmp.path())
        .output()
        .unwrap();
    assert!(
        info.status.success(),
        "qemu-img info failed:\n{}",
        String::from_utf8_lossy(&info.stderr)
    );
    let s = String::from_utf8_lossy(&info.stdout);
    assert!(s.contains("\"virtual-size\": 67108864"), "info:\n{s}");
    assert!(s.contains("\"format\": \"qcow2\""), "info:\n{s}");

    // qemu-img check: structural validation.
    let check = Command::new("qemu-img")
        .arg("check")
        .arg(tmp.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&check.stdout);
    let stderr = String::from_utf8_lossy(&check.stderr);
    assert!(
        check.status.success(),
        "qemu-img check failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // Reopen through our reader and verify the patterns came back.
    let mut back = Qcow2Backend::open(tmp.path()).unwrap();
    let mut head = [0u8; 32];
    back.read_at(0, &mut head).unwrap();
    assert_eq!(&head[..18], b"hello fresh qcow2\n");
    let mut mid = [0u8; 128];
    back.read_at(1024 * 1024, &mut mid).unwrap();
    assert!(mid.iter().all(|&b| b == 0xCD));
    let mut tail = [0u8; 4096];
    back.read_at(63 * 1024 * 1024, &mut tail).unwrap();
    assert!(tail.iter().all(|&b| b == 0xEF));

    // Unallocated cluster reads as zeros.
    let mut zeros = [0xffu8; 1024];
    back.read_at(8 * 1024 * 1024, &mut zeros).unwrap();
    assert!(zeros.iter().all(|&b| b == 0));
}

/// `fstool ext-build src -o out.qcow2` produces a valid qcow2 carrying
/// an ext4 image. Verified with qemu-img check + (after convert-to-raw)
/// e2fsck.
#[test]
fn ext_build_into_qcow2() {
    if !which("qemu-img") || !which("e2fsck") {
        eprintln!("skipping: qemu-img or e2fsck missing");
        return;
    }

    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("hello"), b"in qcow2\n").unwrap();
    std::fs::create_dir(srcdir.path().join("etc")).unwrap();
    std::fs::write(srcdir.path().join("etc/conf"), b"k=v\n").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("disk.qcow2");
    let bin = env!("CARGO_BIN_EXE_fstool");
    let r = Command::new(bin)
        .args(["ext-build", "--kind", "ext4"])
        .arg(srcdir.path())
        .arg("-o")
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "ext-build failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );

    // qemu-img check on the qcow2.
    let chk = Command::new("qemu-img")
        .arg("check")
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        chk.status.success(),
        "qemu-img check failed:\n{}",
        String::from_utf8_lossy(&chk.stdout)
    );

    // Convert to raw and e2fsck.
    let raw = dir.path().join("disk.raw");
    let cv = Command::new("qemu-img")
        .args(["convert", "-O", "raw"])
        .arg(&out)
        .arg(&raw)
        .output()
        .unwrap();
    assert!(cv.status.success(), "qemu-img convert failed");
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(&raw)
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck on converted ext4 failed:\n{}",
        String::from_utf8_lossy(&fsck.stdout)
    );

    // fstool's own ls/cat works on the qcow2 directly.
    let ls = Command::new(bin)
        .arg("ls")
        .arg(&out)
        .arg("/")
        .output()
        .unwrap();
    assert!(ls.status.success());
    let s = String::from_utf8_lossy(&ls.stdout);
    assert!(s.contains("hello"));
    assert!(s.contains("etc"));

    let cat = Command::new(bin)
        .arg("cat")
        .arg(&out)
        .arg("/etc/conf")
        .output()
        .unwrap();
    assert!(cat.status.success());
    assert_eq!(cat.stdout, b"k=v\n");
}

/// `fstool build spec -o disk.qcow2` produces a GPT-partitioned qcow2
/// with two filesystems. The partition target syntax (`disk.qcow2:N`)
/// walks each partition cleanly.
#[test]
fn build_partitioned_qcow2() {
    if !which("qemu-img") {
        eprintln!("skipping: qemu-img not installed");
        return;
    }

    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("hello"), b"in partition 2\n").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let spec_path = dir.path().join("spec.toml");
    std::fs::write(
        &spec_path,
        format!(
            r#"
            [image]
            size = "128MiB"
            partition_table = "gpt"

            [[partitions]]
            name = "EFI"
            type = "esp"
            size = "48MiB"

            [partitions.filesystem]
            type = "fat32"
            volume_label = "EFI"

            [[partitions]]
            name = "root"
            type = "linux"
            size = "remaining"

            [partitions.filesystem]
            type = "ext4"
            source = "{}"
            block_size = 1024
            "#,
            srcdir.path().display()
        ),
    )
    .unwrap();

    let out = dir.path().join("disk.qcow2");
    let bin = env!("CARGO_BIN_EXE_fstool");
    let r = Command::new(bin)
        .arg("build")
        .arg(&spec_path)
        .arg("-o")
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "build failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );

    let chk = Command::new("qemu-img")
        .arg("check")
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        chk.status.success(),
        "qemu-img check failed:\n{}",
        String::from_utf8_lossy(&chk.stdout)
    );

    // info on the qcow2 lists the table.
    let info = Command::new(bin).arg("info").arg(&out).output().unwrap();
    assert!(info.status.success());
    let s = String::from_utf8_lossy(&info.stdout);
    assert!(s.contains("partition table:"));
    assert!(s.contains("EFI"));
    assert!(s.contains("root"));

    // :2 walks the ext4 partition.
    let mut p2 = std::ffi::OsString::from(&out);
    p2.push(":2");
    let ls = Command::new(bin)
        .arg("ls")
        .arg(&p2)
        .arg("/")
        .output()
        .unwrap();
    assert!(ls.status.success(), "ls :2 failed");
    let s = String::from_utf8_lossy(&ls.stdout);
    assert!(s.contains("hello"));
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
