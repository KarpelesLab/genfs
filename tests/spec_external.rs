//! End-to-end validation of the TOML spec `build` path.

use std::process::Command;

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
fn build_bare_ext4_from_spec() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    let Some(_) = which("debugfs") else {
        eprintln!("skipping: debugfs not installed");
        return;
    };

    // Source tree.
    let srcdir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(srcdir.path().join("etc")).unwrap();
    std::fs::write(srcdir.path().join("readme.txt"), b"spec-built image\n").unwrap();
    std::fs::write(srcdir.path().join("etc/app.conf"), b"mode=on\n").unwrap();

    // Spec referencing that tree.
    let spec_text = format!(
        r#"
        [filesystem]
        type = "ext4"
        source = "{}"
        block_size = 1024
        rootdevs = "minimal"
        volume_label = "specimg"
        "#,
        srcdir.path().display()
    );
    let spec = fstool::spec::Spec::parse(&spec_text).unwrap();

    let out = NamedTempFile::new().unwrap();
    fstool::spec::build(&spec, out.path()).unwrap();

    // e2fsck clean.
    let res = Command::new("e2fsck")
        .arg("-fn")
        .arg(out.path())
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "e2fsck failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&res.stdout),
        String::from_utf8_lossy(&res.stderr),
    );

    // debugfs: source tree present, /dev populated, file body intact.
    let listing = Command::new("debugfs")
        .arg("-R")
        .arg("ls /")
        .arg(out.path())
        .output()
        .unwrap();
    let root = String::from_utf8_lossy(&listing.stdout);
    for e in ["readme.txt", "etc", "dev"] {
        assert!(root.contains(e), "missing /{e}:\n{root}");
    }

    let devs = Command::new("debugfs")
        .arg("-R")
        .arg("ls /dev")
        .arg(out.path())
        .output()
        .unwrap();
    let dev = String::from_utf8_lossy(&devs.stdout);
    for n in ["console", "null", "zero", "urandom"] {
        assert!(dev.contains(n), "missing /dev/{n}:\n{dev}");
    }

    let body = Command::new("debugfs")
        .arg("-R")
        .arg("cat /etc/app.conf")
        .arg(out.path())
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&body.stdout).contains("mode=on"),
        "/etc/app.conf body wrong"
    );
}

#[test]
fn build_empty_ext2_from_spec() {
    let Some(_) = which("e2fsck") else {
        eprintln!("skipping: e2fsck not installed");
        return;
    };
    // No `source` → empty filesystem.
    let spec = fstool::spec::Spec::parse(
        r#"
        [filesystem]
        type = "ext2"
        block_size = 1024
        "#,
    )
    .unwrap();
    let out = NamedTempFile::new().unwrap();
    fstool::spec::build(&spec, out.path()).unwrap();

    let res = Command::new("e2fsck")
        .arg("-fn")
        .arg(out.path())
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "e2fsck failed:\n{}",
        String::from_utf8_lossy(&res.stdout)
    );
}
