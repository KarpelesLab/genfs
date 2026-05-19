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
