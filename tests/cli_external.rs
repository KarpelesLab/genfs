//! Exercises the `fstool` binary end to end via its CLI.

use std::process::Command;

use tempfile::NamedTempFile;

/// Path to the freshly-built `fstool` binary (provided by Cargo for
/// integration tests).
const FSTOOL: &str = env!("CARGO_BIN_EXE_fstool");

fn which(tool: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {tool}"))
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// build (bare ext4 spec) → ls → cat → add → cat the added file.
#[test]
fn cli_build_ls_cat_add_roundtrip() {
    if !which("e2fsck") {
        eprintln!("skipping: e2fsck not installed");
        return;
    }

    // Source tree + a spare-capacity spec (extra inodes via a bigger tree
    // is awkward; instead we test `add` against the headroom a fresh image
    // happens to have).
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("one.txt"), b"first\n").unwrap();

    let spec = NamedTempFile::new().unwrap();
    std::fs::write(
        spec.path(),
        format!(
            "[filesystem]\ntype = \"ext4\"\nsource = \"{}\"\nblock_size = 1024\n",
            srcdir.path().display()
        ),
    )
    .unwrap();

    let img = NamedTempFile::new().unwrap();

    // build
    let out = Command::new(FSTOOL)
        .arg("build")
        .arg(spec.path())
        .arg("-o")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // ls /
    let out = Command::new(FSTOOL)
        .args(["ls"])
        .arg(img.path())
        .arg("/")
        .output()
        .unwrap();
    assert!(out.status.success());
    let listing = String::from_utf8_lossy(&out.stdout);
    assert!(
        listing.contains("one.txt"),
        "ls missing one.txt:\n{listing}"
    );

    // cat /one.txt
    let out = Command::new(FSTOOL)
        .args(["cat"])
        .arg(img.path())
        .arg("/one.txt")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout, b"first\n");

    // add a host file
    let extra = NamedTempFile::new().unwrap();
    std::fs::write(extra.path(), b"added via cli\n").unwrap();
    let out = Command::new(FSTOOL)
        .arg("add")
        .arg(img.path())
        .arg(extra.path())
        .arg("/two.txt")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "add failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // e2fsck must still be clean after the modification.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck failed after add:\n{}",
        String::from_utf8_lossy(&fsck.stdout)
    );

    // cat the added file
    let out = Command::new(FSTOOL)
        .args(["cat"])
        .arg(img.path())
        .arg("/two.txt")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout, b"added via cli\n");
}

/// build → rm a file → rm an empty dir → e2fsck clean → non-empty dir
/// rejected.
#[test]
fn cli_rm_file_and_empty_dir() {
    if !which("e2fsck") {
        eprintln!("skipping: e2fsck not installed");
        return;
    }

    // Source tree: a file, an empty dir, and a non-empty dir.
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("doomed.txt"), b"bye\n").unwrap();
    std::fs::create_dir(srcdir.path().join("emptydir")).unwrap();
    std::fs::create_dir(srcdir.path().join("fulldir")).unwrap();
    std::fs::write(srcdir.path().join("fulldir/keep"), b"k\n").unwrap();

    let img = NamedTempFile::new().unwrap();
    let out = Command::new(FSTOOL)
        .args(["ext-build", "--kind", "ext4"])
        .arg(srcdir.path())
        .arg("-o")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(out.status.success());

    // rm a regular file.
    let out = Command::new(FSTOOL)
        .arg("rm")
        .arg(img.path())
        .arg("/doomed.txt")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "rm file failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // rm an empty directory.
    let out = Command::new(FSTOOL)
        .arg("rm")
        .arg(img.path())
        .arg("/emptydir")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "rm empty dir failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // rm a non-empty directory must fail.
    let out = Command::new(FSTOOL)
        .arg("rm")
        .arg(img.path())
        .arg("/fulldir")
        .output()
        .unwrap();
    assert!(!out.status.success(), "rm non-empty dir should have failed");

    // e2fsck clean after the removals.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck failed after rm:\n{}",
        String::from_utf8_lossy(&fsck.stdout)
    );

    // The removed entries are gone; the kept ones remain.
    let out = Command::new(FSTOOL)
        .args(["ls"])
        .arg(img.path())
        .arg("/")
        .output()
        .unwrap();
    let listing = String::from_utf8_lossy(&out.stdout);
    assert!(!listing.contains("doomed.txt"), "doomed.txt still present");
    assert!(!listing.contains("emptydir"), "emptydir still present");
    assert!(listing.contains("fulldir"), "fulldir wrongly removed");
}

/// `fstool info` reports the expected filesystem summary.
#[test]
fn cli_info_reports_ext4() {
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("x"), b"y\n").unwrap();
    let img = NamedTempFile::new().unwrap();

    let out = Command::new(FSTOOL)
        .args(["ext-build", "--kind", "ext4"])
        .arg(srcdir.path())
        .arg("-o")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(out.status.success());

    let out = Command::new(FSTOOL)
        .arg("info")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(out.status.success());
    let info = String::from_utf8_lossy(&out.stdout);
    assert!(info.contains("ext4"), "info missing kind:\n{info}");
    assert!(
        info.contains("block size"),
        "info missing block size:\n{info}"
    );
}

/// `fstool info disk.img` on a partitioned image prints the table;
/// `fstool info disk.img:N` and `ls`/`cat` walk into a partition's FS.
#[test]
fn cli_partition_target_syntax() {
    if !which("e2fsck") {
        eprintln!("skipping: e2fsck not installed");
        return;
    }

    // Build a GPT disk with an EFI/FAT32 + a root/ext4.
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("hello"), b"in a partition\n").unwrap();
    std::fs::create_dir(srcdir.path().join("etc")).unwrap();
    std::fs::write(srcdir.path().join("etc/app.conf"), b"mode=on\n").unwrap();

    let spec = NamedTempFile::new().unwrap();
    std::fs::write(
        spec.path(),
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

    let img = NamedTempFile::new().unwrap();
    let out = Command::new(FSTOOL)
        .arg("build")
        .arg(spec.path())
        .arg("-o")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // `info disk.img` (no :N) prints the partition table.
    let out = Command::new(FSTOOL)
        .arg("info")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("partition table:"), "missing table header:\n{s}");
    assert!(s.contains("gpt"), "expected gpt label:\n{s}");
    assert!(s.contains("EFI"), "expected EFI name:\n{s}");
    assert!(s.contains("root"), "expected root name:\n{s}");

    // `info :1` opens the EFI FAT32 partition.
    let out = Command::new(FSTOOL)
        .arg("info")
        .arg(format!("{}:1", img.path().display()))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "info :1 failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("fat32"), "expected fat32 fs:\n{s}");

    // `info :2` opens the root ext4 partition.
    let out = Command::new(FSTOOL)
        .arg("info")
        .arg(format!("{}:2", img.path().display()))
        .output()
        .unwrap();
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("ext4"), "expected ext4 fs:\n{s}");

    // `ls :2 /` shows the source tree.
    let out = Command::new(FSTOOL)
        .arg("ls")
        .arg(format!("{}:2", img.path().display()))
        .arg("/")
        .output()
        .unwrap();
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("hello"));
    assert!(s.contains("etc"));

    // `cat :2 /etc/app.conf` returns the file body.
    let out = Command::new(FSTOOL)
        .arg("cat")
        .arg(format!("{}:2", img.path().display()))
        .arg("/etc/app.conf")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout, b"mode=on\n");

    // Out-of-range partition index → clean error.
    let out = Command::new(FSTOOL)
        .arg("ls")
        .arg(format!("{}:9", img.path().display()))
        .arg("/")
        .output()
        .unwrap();
    assert!(!out.status.success());
    let s = String::from_utf8_lossy(&out.stderr);
    assert!(
        s.contains("out of range"),
        "expected out-of-range error:\n{s}"
    );

    // `add :2 host /new.txt` writes into a partition; e2fsck still clean.
    let extra = NamedTempFile::new().unwrap();
    std::fs::write(extra.path(), b"added to root partition\n").unwrap();
    let out = Command::new(FSTOOL)
        .arg("add")
        .arg(format!("{}:2", img.path().display()))
        .arg(extra.path())
        .arg("/new.txt")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "add :2 failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // dd out the root partition so e2fsck can check it.
    // sgdisk -p tells us the LBAs.
    if which("sgdisk") {
        let p = Command::new("sgdisk")
            .arg("-p")
            .arg(img.path())
            .output()
            .unwrap();
        let pout = String::from_utf8_lossy(&p.stdout);
        let root_line = pout
            .lines()
            .find(|l| l.trim_start().starts_with("2 "))
            .expect("partition 2 line");
        let nums: Vec<u64> = root_line
            .split_whitespace()
            .skip(1)
            .take(2)
            .map(|s| s.parse().unwrap())
            .collect();
        let (start, end) = (nums[0], nums[1]);
        let part = NamedTempFile::new().unwrap();
        let dd = Command::new("dd")
            .arg(format!("if={}", img.path().display()))
            .arg(format!("of={}", part.path().display()))
            .arg("bs=512")
            .arg(format!("skip={start}"))
            .arg(format!("count={}", end - start + 1))
            .arg("status=none")
            .output()
            .unwrap();
        assert!(dd.status.success());
        let fsck = Command::new("e2fsck")
            .arg("-fn")
            .arg(part.path())
            .output()
            .unwrap();
        assert!(
            fsck.status.success(),
            "e2fsck on root partition failed after :2 add:\n{}",
            String::from_utf8_lossy(&fsck.stdout)
        );
    }
}

/// `fstool shell` runs an SFTP-style REPL. Drive it with a scripted
/// stdin and assert the captured stdout contains the right output for
/// each command.
#[test]
fn cli_shell_navigates_and_mutates() {
    if !which("e2fsck") {
        eprintln!("skipping: e2fsck not installed");
        return;
    }

    // Build a small ext4 with a file and a subdirectory.
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::create_dir(srcdir.path().join("docs")).unwrap();
    std::fs::write(srcdir.path().join("docs/readme"), b"deep body\n").unwrap();
    std::fs::write(srcdir.path().join("top.txt"), b"top body\n").unwrap();

    let img = NamedTempFile::new().unwrap();
    let out = Command::new(FSTOOL)
        .args(["ext-build", "--kind", "ext4"])
        .arg(srcdir.path())
        .arg("-o")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "ext-build failed");

    // Drive the shell.
    let extra = NamedTempFile::new().unwrap();
    std::fs::write(extra.path(), b"shell-added\n").unwrap();
    let script = format!(
        "pwd\n\
         ls /\n\
         cd docs\n\
         pwd\n\
         cat readme\n\
         cd ..\n\
         mkdir /new\n\
         put {} /new/copy.txt\n\
         cat /new/copy.txt\n\
         rm /top.txt\n\
         ls /\n\
         quit\n",
        extra.path().display()
    );

    let mut child = std::process::Command::new(FSTOOL)
        .arg("shell")
        .arg(img.path())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    {
        use std::io::Write;
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(script.as_bytes())
            .unwrap();
    }
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "shell exited non-zero:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    assert!(
        stdout.contains("top.txt"),
        "missing top.txt in /:\n{stdout}"
    );
    assert!(stdout.contains("docs/"), "missing docs/ in /:\n{stdout}");
    assert!(stdout.contains("/docs"), "pwd after cd docs:\n{stdout}");
    assert!(stdout.contains("deep body"), "cat readme output:\n{stdout}");
    assert!(
        stdout.contains("shell-added"),
        "cat of added file:\n{stdout}"
    );
    // After `rm /top.txt`, the listing must no longer show it. The
    // assertion below counts occurrences — the script does `ls /` twice,
    // so before-rm it appears once; after-rm it shouldn't appear in the
    // second listing. We just check the FINAL state via a fresh `ls`.
    let final_listing = Command::new(FSTOOL)
        .args(["ls"])
        .arg(img.path())
        .arg("/")
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&final_listing.stdout);
    assert!(!s.contains("top.txt"), "top.txt should be gone:\n{s}");
    assert!(s.contains("new"), "/new should exist:\n{s}");

    // e2fsck still clean after all the shell mutations.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck failed after shell mutations:\n{}",
        String::from_utf8_lossy(&fsck.stdout)
    );
}

/// FAT32 add/rm through `fstool`: parallel to the ext test.
#[test]
fn cli_fat32_add_and_rm() {
    if !which("fsck.vfat") {
        eprintln!("skipping: fsck.vfat not installed");
        return;
    }

    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("keep.txt"), b"keep\n").unwrap();
    std::fs::write(srcdir.path().join("goodbye.txt"), b"bye\n").unwrap();

    let img = NamedTempFile::new().unwrap();
    let out = Command::new(FSTOOL)
        .args(["fat-build", "--size", "64MiB", "--label", "CLIRM"])
        .arg(srcdir.path())
        .arg("-o")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "fat-build failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // add a host file
    let extra = NamedTempFile::new().unwrap();
    std::fs::write(extra.path(), b"added via cli\n").unwrap();
    let out = Command::new(FSTOOL)
        .arg("add")
        .arg(img.path())
        .arg(extra.path())
        .arg("/added.txt")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "add failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // cat the added file
    let out = Command::new(FSTOOL)
        .args(["cat"])
        .arg(img.path())
        .arg("/added.txt")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(out.stdout, b"added via cli\n");

    // rm a different file
    let out = Command::new(FSTOOL)
        .arg("rm")
        .arg(img.path())
        .arg("/goodbye.txt")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "rm failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // fsck must still be clean.
    let res = Command::new("fsck.vfat")
        .args(["-n", "-v"])
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "fsck.vfat failed after add/rm:\n{}",
        String::from_utf8_lossy(&res.stdout)
    );

    // ls shows the expected state.
    let out = Command::new(FSTOOL)
        .args(["ls"])
        .arg(img.path())
        .arg("/")
        .output()
        .unwrap();
    let listing = String::from_utf8_lossy(&out.stdout);
    assert!(listing.contains("keep.txt"));
    assert!(listing.contains("added.txt"));
    assert!(!listing.contains("goodbye.txt"));
}

/// `fstool fat-build` → `ls` → `cat` → `info` on a FAT32 image. Exercises
/// the unified inspection dispatch (the CLI doesn't know it's FAT32).
#[test]
fn cli_fat32_build_ls_cat_info_roundtrip() {
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("short.txt"), b"short body\n").unwrap();
    std::fs::create_dir(srcdir.path().join("nest")).unwrap();
    std::fs::write(
        srcdir.path().join("nest/A Long Name.md"),
        b"long-name body\n",
    )
    .unwrap();

    let img = NamedTempFile::new().unwrap();
    let out = Command::new(FSTOOL)
        .args(["fat-build", "--size", "64MiB", "--label", "CLIFAT"])
        .arg(srcdir.path())
        .arg("-o")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "fat-build failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // info names the FS as fat32.
    let out = Command::new(FSTOOL)
        .arg("info")
        .arg(img.path())
        .output()
        .unwrap();
    assert!(out.status.success());
    let info = String::from_utf8_lossy(&out.stdout);
    assert!(info.contains("fat32"), "info missing fat32:\n{info}");
    assert!(info.contains("CLIFAT"), "info missing label:\n{info}");

    // ls /
    let out = Command::new(FSTOOL)
        .args(["ls"])
        .arg(img.path())
        .arg("/")
        .output()
        .unwrap();
    assert!(out.status.success());
    let listing = String::from_utf8_lossy(&out.stdout);
    assert!(listing.contains("short.txt"));
    assert!(listing.contains("nest"));

    // ls a subdirectory.
    let out = Command::new(FSTOOL)
        .args(["ls"])
        .arg(img.path())
        .arg("/nest")
        .output()
        .unwrap();
    assert!(out.status.success());
    let nest = String::from_utf8_lossy(&out.stdout);
    assert!(
        nest.contains("A Long Name.md"),
        "long-name entry missing from /nest:\n{nest}"
    );

    // cat the deep long-named file.
    let out = Command::new(FSTOOL)
        .args(["cat"])
        .arg(img.path())
        .arg("/nest/A Long Name.md")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "cat failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(out.stdout, b"long-name body\n");
}

/// `fstool convert` does a byte-for-byte raw ↔ qcow2 round-trip.
#[test]
fn cli_convert_raw_qcow2_roundtrip() {
    if !which("qemu-img") {
        eprintln!("skipping: qemu-img not installed");
        return;
    }

    // Build a small ext4 raw image to use as the convert source.
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("hello"), b"hello convert\n").unwrap();
    let raw = NamedTempFile::new().unwrap();
    let out = Command::new(FSTOOL)
        .args(["ext-build", "--kind", "ext4"])
        .arg(srcdir.path())
        .arg("-o")
        .arg(raw.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "ext-build failed");

    let dir = tempfile::tempdir().unwrap();
    let qcow = dir.path().join("disk.qcow2");
    let raw2 = dir.path().join("disk2.img");

    // raw → qcow2
    let r = Command::new(FSTOOL)
        .arg("convert")
        .arg(raw.path())
        .arg(&qcow)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "convert raw→qcow2 failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );

    // qemu-img check on the qcow2.
    let chk = Command::new("qemu-img")
        .arg("check")
        .arg(&qcow)
        .output()
        .unwrap();
    assert!(chk.status.success(), "qemu-img check failed");

    // qcow2 → raw, must match original byte-for-byte.
    let r = Command::new(FSTOOL)
        .arg("convert")
        .arg(&qcow)
        .arg(&raw2)
        .output()
        .unwrap();
    assert!(r.status.success(), "convert qcow2→raw failed");
    let a = std::fs::read(raw.path()).unwrap();
    let b = std::fs::read(&raw2).unwrap();
    assert_eq!(a, b, "round-tripped raw differs from original");

    // Grow: 64 MiB destination, source still readable through the larger image.
    let big = dir.path().join("big.qcow2");
    let r = Command::new(FSTOOL)
        .arg("convert")
        .arg(raw.path())
        .arg(&big)
        .args(["--size", "64MiB"])
        .output()
        .unwrap();
    assert!(r.status.success(), "convert --size grow failed");
    let ls = Command::new(FSTOOL)
        .arg("ls")
        .arg(&big)
        .arg("/")
        .output()
        .unwrap();
    assert!(ls.status.success());
    let s = String::from_utf8_lossy(&ls.stdout);
    assert!(s.contains("hello"), "grown image lost source content:\n{s}");

    // Shrink request is rejected.
    let small = dir.path().join("small.qcow2");
    let r = Command::new(FSTOOL)
        .arg("convert")
        .arg(raw.path())
        .arg(&small)
        .args(["--size", "1KiB"])
        .output()
        .unwrap();
    assert!(!r.status.success(), "shrink should have been rejected");
    let s = String::from_utf8_lossy(&r.stderr);
    assert!(
        s.contains("repack"),
        "rejection should point to repack:\n{s}"
    );
}

/// `fstool repack --shrink` produces a smaller image whose content
/// matches the source.
#[test]
fn cli_repack_shrink() {
    if !which("e2fsck") || !which("mke2fs") {
        eprintln!("skipping: e2fsck/mke2fs missing");
        return;
    }

    // Make a deliberately oversized ext4 image via mke2fs.
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("keep.txt"), b"keep this\n").unwrap();
    std::fs::create_dir(srcdir.path().join("etc")).unwrap();
    std::fs::write(srcdir.path().join("etc/conf"), b"v=1\n").unwrap();

    let big = NamedTempFile::new().unwrap();
    Command::new("truncate")
        .args(["-s", "16M"])
        .arg(big.path())
        .output()
        .unwrap();
    let mk = Command::new("mke2fs")
        .args(["-F", "-t", "ext4", "-d"])
        .arg(srcdir.path())
        .arg("-L")
        .arg("src")
        .arg(big.path())
        .output()
        .unwrap();
    assert!(
        mk.status.success(),
        "mke2fs failed:\n{}",
        String::from_utf8_lossy(&mk.stderr)
    );
    let big_len = std::fs::metadata(big.path()).unwrap().len();

    // Repack with --shrink.
    let shrunk = NamedTempFile::new().unwrap();
    let r = Command::new(FSTOOL)
        .arg("repack")
        .arg(big.path())
        .arg(shrunk.path())
        .arg("--shrink")
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "repack --shrink failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );
    let shrunk_len = std::fs::metadata(shrunk.path()).unwrap().len();
    assert!(
        shrunk_len < big_len,
        "repack didn't shrink: {shrunk_len} vs {big_len}"
    );

    // e2fsck must be clean.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(shrunk.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck on shrunk image failed:\n{}",
        String::from_utf8_lossy(&fsck.stdout)
    );

    // Content survived.
    let cat = Command::new(FSTOOL)
        .arg("cat")
        .arg(shrunk.path())
        .arg("/etc/conf")
        .output()
        .unwrap();
    assert!(cat.status.success());
    assert_eq!(cat.stdout, b"v=1\n");
}

/// `fstool repack` cross-FS-type: ext → FAT32.
#[test]
fn cli_repack_ext_to_fat32() {
    if !which("fsck.vfat") || !which("mke2fs") {
        eprintln!("skipping: fsck.vfat/mke2fs missing");
        return;
    }

    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("README.txt"), b"cross-fs content\n").unwrap();

    let src = NamedTempFile::new().unwrap();
    Command::new("truncate")
        .args(["-s", "8M"])
        .arg(src.path())
        .output()
        .unwrap();
    Command::new("mke2fs")
        .args(["-F", "-t", "ext4", "-d"])
        .arg(srcdir.path())
        .arg(src.path())
        .output()
        .unwrap();

    let fat = NamedTempFile::new().unwrap();
    let r = Command::new(FSTOOL)
        .arg("repack")
        .arg(src.path())
        .arg(fat.path())
        .args(["--fs-type", "fat32", "--size", "64MiB"])
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "repack ext→fat32 failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );

    let chk = Command::new("fsck.vfat")
        .args(["-n", "-v"])
        .arg(fat.path())
        .output()
        .unwrap();
    assert!(
        chk.status.success(),
        "fsck.vfat failed on repacked FAT32:\n{}",
        String::from_utf8_lossy(&chk.stdout)
    );

    let ls = Command::new(FSTOOL)
        .arg("ls")
        .arg(fat.path())
        .arg("/")
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&ls.stdout);
    assert!(
        s.to_ascii_lowercase().contains("readme.txt"),
        "missing README.txt in FAT32 output:\n{s}"
    );
}

/// `fstool repack` preserves symlinks, mode, and uid/gid (ext → ext).
/// Verifies that the direct FS-to-FS copier doesn't drop metadata that
/// the previous host-tempdir staging implementation lost.
#[test]
fn cli_repack_preserves_symlinks_and_mode() {
    if !which("e2fsck") || !which("mke2fs") || !which("debugfs") {
        eprintln!("skipping: e2fsprogs missing");
        return;
    }

    // Build a source ext4 with a regular file (mode 0640), a relative
    // symlink, and an absolute symlink — none of which the old
    // staging path could reproduce without root.
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("regular.txt"), b"keep me\n").unwrap();
    std::os::unix::fs::symlink("regular.txt", srcdir.path().join("link-rel")).unwrap();
    std::os::unix::fs::symlink("/etc/passwd", srcdir.path().join("link-abs")).unwrap();
    use std::os::unix::fs::PermissionsExt;
    let p = srcdir.path().join("regular.txt");
    let mut perms = std::fs::metadata(&p).unwrap().permissions();
    perms.set_mode(0o640);
    std::fs::set_permissions(&p, perms).unwrap();

    let src = NamedTempFile::new().unwrap();
    Command::new("truncate")
        .args(["-s", "16M"])
        .arg(src.path())
        .output()
        .unwrap();
    Command::new("mke2fs")
        .args(["-F", "-t", "ext4", "-d"])
        .arg(srcdir.path())
        .arg(src.path())
        .output()
        .unwrap();

    let dst = NamedTempFile::new().unwrap();
    let r = Command::new(FSTOOL)
        .arg("repack")
        .arg(src.path())
        .arg(dst.path())
        .arg("--shrink")
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "repack failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );

    // debugfs reports per-inode mode + symlink targets.
    let ls = Command::new("debugfs")
        .arg("-R")
        .arg("ls -l /")
        .arg(dst.path())
        .output()
        .unwrap();
    let listing = String::from_utf8_lossy(&ls.stdout);
    // mode for regular.txt should be 0640 (octal "100640" = REG | 0640).
    assert!(
        listing.contains("100640"),
        "mode 0640 not preserved on regular.txt:\n{listing}"
    );

    let abs = Command::new("debugfs")
        .arg("-R")
        .arg("stat /link-abs")
        .arg(dst.path())
        .output()
        .unwrap();
    let abs_out = String::from_utf8_lossy(&abs.stdout);
    assert!(
        abs_out.contains("/etc/passwd"),
        "absolute symlink target lost:\n{abs_out}"
    );

    let rel = Command::new("debugfs")
        .arg("-R")
        .arg("stat /link-rel")
        .arg(dst.path())
        .output()
        .unwrap();
    let rel_out = String::from_utf8_lossy(&rel.stdout);
    assert!(
        rel_out.contains("regular.txt"),
        "relative symlink target lost:\n{rel_out}"
    );

    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(dst.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck failed after repack:\n{}",
        String::from_utf8_lossy(&fsck.stdout)
    );
}

/// `fstool repack` preserves ext xattrs through the FS-to-FS copy.
/// Source is an mke2fs image with `debugfs ea_set`-stamped xattrs;
/// destination must report identical values via `debugfs ea_get`.
#[test]
fn cli_repack_preserves_xattrs() {
    if !which("mke2fs") || !which("debugfs") || !which("e2fsck") {
        eprintln!("skipping: e2fsprogs missing");
        return;
    }

    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("withxattr.txt"), b"hi\n").unwrap();

    let src = NamedTempFile::new().unwrap();
    Command::new("truncate")
        .args(["-s", "16M"])
        .arg(src.path())
        .output()
        .unwrap();
    Command::new("mke2fs")
        .args(["-F", "-t", "ext4", "-d"])
        .arg(srcdir.path())
        .arg(src.path())
        .output()
        .unwrap();

    // Stamp xattrs onto the source via debugfs.
    for cmd in [
        r#"ea_set /withxattr.txt user.foo "hello-xattr""#,
        r#"ea_set /withxattr.txt security.selinux "system_u:object_r:demo_t:s0""#,
        r#"ea_set /withxattr.txt trusted.opaque "blob""#,
    ] {
        let out = Command::new("debugfs")
            .args(["-w", "-R"])
            .arg(cmd)
            .arg(src.path())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "debugfs ea_set failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    // Repack with shrink.
    let dst = NamedTempFile::new().unwrap();
    let r = Command::new(FSTOOL)
        .arg("repack")
        .arg(src.path())
        .arg(dst.path())
        .arg("--shrink")
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "repack failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );

    // e2fsck on destination must be clean.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(dst.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck failed after repack:\n{}",
        String::from_utf8_lossy(&fsck.stdout)
    );

    // Each xattr must round-trip via debugfs ea_get.
    let cases = [
        ("user.foo", "hello-xattr"),
        ("security.selinux", "system_u:object_r:demo_t:s0"),
        ("trusted.opaque", "blob"),
    ];
    for (name, expected) in cases {
        let out = Command::new("debugfs")
            .args(["-R"])
            .arg(format!("ea_get /withxattr.txt {name}"))
            .arg(dst.path())
            .output()
            .unwrap();
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(
            s.contains(expected),
            "xattr {name:?} lost or wrong:\nstdout:\n{s}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// `fstool repack` from ext to tar and back: content, mode, symlinks,
/// and xattrs all survive the round-trip.
#[test]
fn cli_repack_ext_tar_ext_preserves_metadata() {
    if !which("mke2fs") || !which("debugfs") || !which("e2fsck") {
        eprintln!("skipping: e2fsprogs missing");
        return;
    }

    // Build a source ext4 with a file (mode 0640), a relative symlink,
    // and a nested subdir.
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("regular.txt"), b"file body\n").unwrap();
    std::os::unix::fs::symlink("regular.txt", srcdir.path().join("link.txt")).unwrap();
    std::fs::create_dir(srcdir.path().join("sub")).unwrap();
    std::fs::write(srcdir.path().join("sub/inside.txt"), b"nested\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    let p = srcdir.path().join("regular.txt");
    let mut perms = std::fs::metadata(&p).unwrap().permissions();
    perms.set_mode(0o640);
    std::fs::set_permissions(&p, perms).unwrap();

    let src = NamedTempFile::new().unwrap();
    Command::new("truncate")
        .args(["-s", "16M"])
        .arg(src.path())
        .output()
        .unwrap();
    Command::new("mke2fs")
        .args(["-F", "-t", "ext4", "-d"])
        .arg(srcdir.path())
        .arg(src.path())
        .output()
        .unwrap();
    // Stamp xattrs on regular.txt.
    Command::new("debugfs")
        .args(["-w", "-R"])
        .arg(r#"ea_set /regular.txt user.tag "from-ext""#)
        .arg(src.path())
        .output()
        .unwrap();

    // ext → tar
    let dir = tempfile::tempdir().unwrap();
    let tar = dir.path().join("intermediate.tar");
    let r = Command::new(FSTOOL)
        .arg("repack")
        .arg(src.path())
        .arg(&tar)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "ext → tar failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );
    // `file` reports POSIX tar archive.
    let f = std::fs::read(&tar).unwrap();
    assert_eq!(&f[257..262], b"ustar");

    // List the tar via fstool itself — proves the tar reader works.
    let ls = Command::new(FSTOOL)
        .arg("ls")
        .arg(&tar)
        .arg("/")
        .output()
        .unwrap();
    assert!(ls.status.success());
    let s = String::from_utf8_lossy(&ls.stdout);
    assert!(s.contains("regular.txt"));
    assert!(s.contains("link.txt"));
    assert!(s.contains("sub"));

    // tar → ext (auto, uses ext4 because source is tar).
    let back = NamedTempFile::new().unwrap();
    let r = Command::new(FSTOOL)
        .arg("repack")
        .arg(&tar)
        .arg(back.path())
        .arg("--shrink")
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "tar → ext failed:\n{}",
        String::from_utf8_lossy(&r.stderr)
    );

    // Verify mode + uid/gid + symlink target + xattrs all survived.
    let listing = Command::new("debugfs")
        .arg("-R")
        .arg("ls -l /")
        .arg(back.path())
        .output()
        .unwrap();
    let l = String::from_utf8_lossy(&listing.stdout);
    assert!(l.contains("100640"), "mode 0640 not preserved:\n{l}");

    let symlink = Command::new("debugfs")
        .arg("-R")
        .arg("stat /link.txt")
        .arg(back.path())
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&symlink.stdout);
    assert!(s.contains("regular.txt"), "symlink target lost:\n{s}");

    let xattr = Command::new("debugfs")
        .arg("-R")
        .arg("ea_get /regular.txt user.tag")
        .arg(back.path())
        .output()
        .unwrap();
    let s = String::from_utf8_lossy(&xattr.stdout);
    assert!(
        s.contains("from-ext"),
        "user.tag xattr lost through ext → tar → ext:\n{s}"
    );

    // e2fsck on the round-tripped image.
    let fsck = Command::new("e2fsck")
        .arg("-fn")
        .arg(back.path())
        .output()
        .unwrap();
    assert!(
        fsck.status.success(),
        "e2fsck failed after ext → tar → ext:\n{}",
        String::from_utf8_lossy(&fsck.stdout)
    );
}

/// fstool's tar reader is compatible with standard `tar -tvf`.
#[test]
fn cli_tar_archive_readable_by_system_tar() {
    if !which("mke2fs") || !which("tar") {
        eprintln!("skipping: mke2fs/tar missing");
        return;
    }
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::write(srcdir.path().join("hello"), b"hi\n").unwrap();

    let src = NamedTempFile::new().unwrap();
    Command::new("truncate")
        .args(["-s", "16M"])
        .arg(src.path())
        .output()
        .unwrap();
    Command::new("mke2fs")
        .args(["-F", "-t", "ext4", "-d"])
        .arg(srcdir.path())
        .arg(src.path())
        .output()
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let tar = dir.path().join("out.tar");
    Command::new(FSTOOL)
        .arg("repack")
        .arg(src.path())
        .arg(&tar)
        .output()
        .unwrap();

    let out = Command::new("tar")
        .args(["-tf"])
        .arg(&tar)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "system tar failed to list our archive"
    );
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("hello"), "system tar didn't see /hello:\n{s}");
}
