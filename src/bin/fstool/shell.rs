//! Interactive shell over an [`fstool::inspect::AnyFs`]: an SFTP-style
//! REPL for poking at an image without paying the open/parse cost on
//! every command. The shell maintains a virtual current directory inside
//! the image and resolves relative paths against it.
//!
//! Commands:
//!
//! ```text
//!   ls [PATH]           list a directory (default: cwd)
//!   pwd                 print the current directory
//!   cd [PATH]           change directory (no arg → /)
//!   cat PATH            print a file's contents to stdout
//!   put HOST [DEST]     copy a host file/dir into the image
//!                       (DEST defaults to the basename of HOST in cwd)
//!   rm PATH             remove a file or empty directory
//!   mkdir PATH          create an empty directory
//!   info [PATH]         no arg → image summary; with PATH → per-file
//!                       metadata (kind/mode/owner/size/blocks/nlink
//!                       /inode/atime/mtime/ctime/rdev) plus any
//!                       extended attributes (fs-specific properties
//!                       come through here: NTFS DOS attrs, ADS,
//!                       security descriptors; ext / squashfs xattrs;
//!                       HFS+ Finder info; …)
//!   help                list these commands
//!   quit | exit         leave
//! ```
//!
//! The shell is just a wrapper around [`fstool::inspect::AnyFs`]; every
//! command dispatches through that, so ext and FAT32 images both work.

use std::io::{BufRead, Write};
use std::path::Path;

use fstool::Result;
use fstool::block::BlockDevice;
use fstool::inspect::AnyFs;

/// An interactive shell over an opened image.
pub struct Shell {
    fs: AnyFs,
    /// Current working directory inside the image. Always absolute and
    /// normalised (no `.`/`..`/empty segments).
    cwd: String,
    /// True when the shell is in read-only mode (`fstool shell --ro`).
    /// `put` / `rm` / `mkdir` are refused at dispatch time and the
    /// underlying device is opened `O_RDONLY` by the caller.
    read_only: bool,
}

impl Shell {
    /// A new shell rooted at `/` over `fs`. The shell is mutating —
    /// `put` / `rm` / `mkdir` go through to the FS writer.
    pub fn new(fs: AnyFs) -> Self {
        Self {
            fs,
            cwd: "/".into(),
            read_only: false,
        }
    }

    /// A read-only shell over `fs`. `put` / `rm` / `mkdir` refuse
    /// with a clear error; only `ls` / `cat` / `cd` / `pwd` / `info`
    /// / `help` work. Intended for `fstool shell --ro` where the
    /// caller has opened the BlockDevice read-only (so even a
    /// missed gate fails at the syscall).
    pub fn new_read_only(fs: AnyFs) -> Self {
        Self {
            fs,
            cwd: "/".into(),
            read_only: true,
        }
    }

    /// Read commands from `input` line by line and execute each one against
    /// `dev`, writing prompts, results, and errors to `output`. Returns
    /// when the input stream reaches EOF or the user runs `quit` / `exit`.
    /// Errors from individual commands are reported and the loop continues —
    /// only I/O errors on the input or output streams propagate.
    pub fn run(
        &mut self,
        dev: &mut dyn BlockDevice,
        mut input: impl BufRead,
        mut output: impl Write,
    ) -> Result<()> {
        loop {
            write!(output, "fstool:{}> ", self.cwd)?;
            output.flush()?;
            let mut line = String::new();
            let n = input.read_line(&mut line)?;
            if n == 0 {
                writeln!(output)?; // newline so the next shell prompt isn't on our line
                break;
            }
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            match self.dispatch(dev, line, &mut output) {
                Ok(true) => break,
                Ok(false) => {}
                Err(e) => writeln!(output, "error: {e}")?,
            }
        }
        Ok(())
    }

    /// Interactive REPL with line editing and command history (↑/↓ to recall
    /// previous commands, Ctrl-A/E, Ctrl-R search, …) via `rustyline`. Used
    /// when stdin is a TTY; piped input still flows through [`Shell::run`],
    /// which keeps the deterministic, testable line-buffered path.
    ///
    /// History persists to `~/.fstool_history` between sessions. `Ctrl-C`
    /// abandons the current line and re-prompts; `Ctrl-D` at an empty prompt
    /// exits, matching a typical Unix shell.
    #[cfg(feature = "readline")]
    pub fn run_interactive(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        use rustyline::error::ReadlineError;

        let mut rl = rustyline::DefaultEditor::new()
            .map_err(|e| fstool::Error::Io(std::io::Error::other(e.to_string())))?;
        let history = history_path();
        if let Some(path) = history.as_ref() {
            // A missing history file on first run is fine.
            let _ = rl.load_history(path);
        }

        let mut output = std::io::stdout().lock();
        loop {
            let prompt = format!("fstool:{}> ", self.cwd);
            match rl.readline(&prompt) {
                Ok(line) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() || trimmed.starts_with('#') {
                        continue;
                    }
                    let _ = rl.add_history_entry(trimmed);
                    match self.dispatch(dev, trimmed, &mut output) {
                        Ok(true) => break,
                        Ok(false) => {}
                        Err(e) => writeln!(output, "error: {e}")?,
                    }
                }
                // Ctrl-C: drop the half-typed line and re-prompt.
                Err(ReadlineError::Interrupted) => continue,
                // Ctrl-D at the prompt: exit the shell.
                Err(ReadlineError::Eof) => break,
                Err(e) => return Err(fstool::Error::Io(std::io::Error::other(e.to_string()))),
            }
        }

        if let Some(path) = history.as_ref() {
            let _ = rl.save_history(path);
        }
        Ok(())
    }

    /// Dispatch one command line. Returns `Ok(true)` if the shell should
    /// exit (`quit` / `exit`), `Ok(false)` to continue, or an `Err` for
    /// the loop to print and recover from.
    fn dispatch(
        &mut self,
        dev: &mut dyn BlockDevice,
        line: &str,
        output: &mut impl Write,
    ) -> Result<bool> {
        let (cmd, rest) = split_cmd(line);
        match cmd {
            "quit" | "exit" | ":q" => Ok(true),
            "help" | "?" => {
                self.cmd_help(output)?;
                Ok(false)
            }
            "pwd" => {
                writeln!(output, "{}", self.cwd)?;
                Ok(false)
            }
            "ls" => {
                self.cmd_ls(dev, rest, output)?;
                Ok(false)
            }
            "cd" => {
                self.cmd_cd(dev, rest)?;
                Ok(false)
            }
            "cat" => {
                self.cmd_cat(dev, rest, output)?;
                Ok(false)
            }
            "put" => {
                self.require_writable("put")?;
                self.cmd_put(dev, rest, output)?;
                Ok(false)
            }
            "rm" => {
                self.require_writable("rm")?;
                self.cmd_rm(dev, rest, output)?;
                Ok(false)
            }
            "mkdir" => {
                self.require_writable("mkdir")?;
                self.cmd_mkdir(dev, rest, output)?;
                Ok(false)
            }
            "info" => {
                self.cmd_info(dev, rest, output)?;
                Ok(false)
            }
            "" => Ok(false),
            other => Err(fstool::Error::InvalidArgument(format!(
                "unknown command {other:?} (try `help`)"
            ))),
        }
    }

    /// Refuse a mutating command when the shell is in `--ro` mode.
    /// The underlying BlockDevice is also opened `O_RDONLY` so a
    /// missed gate would still fail at the syscall, but this gives
    /// the user a clean error rather than `PermissionDenied`.
    fn require_writable(&self, cmd: &str) -> Result<()> {
        if self.read_only {
            return Err(fstool::Error::InvalidArgument(format!(
                "{cmd}: shell is read-only (started with --ro); restart \
                 without --ro to mutate the image",
            )));
        }
        Ok(())
    }

    fn cmd_help(&self, output: &mut impl Write) -> Result<()> {
        let ro_note = if self.read_only {
            "\n(shell is read-only: put / rm / mkdir refuse — restart without --ro to mutate)\n"
        } else {
            ""
        };
        let body = format!(
            "ls [PATH]           list a directory (default: cwd)
pwd                 print the current directory
cd [PATH]           change directory (no arg → /)
cat PATH            print a file's contents to stdout
put HOST [DEST]     copy a host file or directory into the image
rm PATH             remove a file or empty directory
mkdir PATH          create an empty directory
info [PATH]         no arg → image summary; with PATH → file metadata
                    (kind/mode/owner/size/blocks/nlink/inode/atime/mtime
                    /ctime/rdev) plus any extended attributes
help | ?            print this help
quit | exit         leave{ro_note}\n"
        );
        output.write_all(body.as_bytes())?;
        Ok(())
    }

    fn cmd_ls(
        &mut self,
        dev: &mut dyn BlockDevice,
        arg: &str,
        output: &mut impl Write,
    ) -> Result<()> {
        let target = if arg.is_empty() {
            self.cwd.clone()
        } else {
            self.resolve(arg)
        };
        let entries = self.fs.list(dev, &target)?;
        for e in &entries {
            let suffix = match e.kind {
                fstool::fs::EntryKind::Dir => "/",
                fstool::fs::EntryKind::Symlink => "@",
                _ => "",
            };
            writeln!(output, "{}{}", e.name, suffix)?;
        }
        Ok(())
    }

    fn cmd_cd(&mut self, dev: &mut dyn BlockDevice, arg: &str) -> Result<()> {
        let target = if arg.is_empty() {
            "/".to_string()
        } else {
            self.resolve(arg)
        };
        // Verify it's actually a directory by listing it. Cheap, and gives
        // a useful error if the path is wrong.
        self.fs.list(dev, &target)?;
        self.cwd = target;
        Ok(())
    }

    fn cmd_cat(
        &mut self,
        dev: &mut dyn BlockDevice,
        arg: &str,
        output: &mut impl Write,
    ) -> Result<()> {
        if arg.is_empty() {
            return Err(fstool::Error::InvalidArgument(
                "cat: PATH is required".into(),
            ));
        }
        let path = self.resolve(arg);
        self.fs.copy_file_to(dev, &path, output)?;
        Ok(())
    }

    fn cmd_put(
        &mut self,
        dev: &mut dyn BlockDevice,
        arg: &str,
        output: &mut impl Write,
    ) -> Result<()> {
        let mut parts = arg.splitn(2, char::is_whitespace);
        let host_str = parts.next().unwrap_or("").trim();
        let dest_arg = parts.next().unwrap_or("").trim();
        if host_str.is_empty() {
            return Err(fstool::Error::InvalidArgument(
                "put: HOST is required".into(),
            ));
        }
        let host = Path::new(host_str);
        let meta = std::fs::symlink_metadata(host)?;
        let dest = if dest_arg.is_empty() {
            let leaf = host.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
                fstool::Error::InvalidArgument(
                    "put: HOST has no usable leaf name; specify DEST explicitly".into(),
                )
            })?;
            join(&self.cwd, leaf)
        } else {
            self.resolve(dest_arg)
        };
        if meta.is_dir() {
            self.fs.add_dir_tree(dev, &dest, host)?;
        } else if meta.is_file() {
            self.fs.add_file(dev, &dest, host)?;
        } else {
            return Err(fstool::Error::InvalidArgument(format!(
                "put: {} is neither a regular file nor a directory",
                host.display()
            )));
        }
        self.fs.flush(dev)?;
        dev.sync()?;
        writeln!(output, "put {} → {dest}", host.display())?;
        Ok(())
    }

    fn cmd_rm(
        &mut self,
        dev: &mut dyn BlockDevice,
        arg: &str,
        output: &mut impl Write,
    ) -> Result<()> {
        if arg.is_empty() {
            return Err(fstool::Error::InvalidArgument(
                "rm: PATH is required".into(),
            ));
        }
        let path = self.resolve(arg);
        if path == "/" {
            return Err(fstool::Error::InvalidArgument(
                "rm: refusing to remove /".into(),
            ));
        }
        self.fs.remove(dev, &path)?;
        self.fs.flush(dev)?;
        dev.sync()?;
        writeln!(output, "removed {path}")?;
        Ok(())
    }

    fn cmd_mkdir(
        &mut self,
        dev: &mut dyn BlockDevice,
        arg: &str,
        output: &mut impl Write,
    ) -> Result<()> {
        if arg.is_empty() {
            return Err(fstool::Error::InvalidArgument(
                "mkdir: PATH is required".into(),
            ));
        }
        let path = self.resolve(arg);
        self.fs.mkdir(dev, &path)?;
        self.fs.flush(dev)?;
        dev.sync()?;
        writeln!(output, "mkdir {path}")?;
        Ok(())
    }

    fn cmd_info(
        &mut self,
        dev: &mut dyn BlockDevice,
        arg: &str,
        output: &mut impl Write,
    ) -> Result<()> {
        // No path → image-level summary, unchanged behaviour.
        if arg.is_empty() {
            writeln!(output, "fs kind: {}", self.fs.kind_string())?;
            return Ok(());
        }

        // With a path → per-file metadata. `getattr` returns the
        // POSIX-ish fields every backend can answer; `list_xattrs`
        // surfaces fs-specific properties (NTFS DOS attrs, ADS, security
        // descriptors; ext / squashfs xattrs; HFS+ Finder info; …).
        let path = self.resolve(arg);
        let attrs = self.fs.getattr(dev, Path::new(&path))?;

        writeln!(output, "path:   {path}")?;
        writeln!(output, "kind:   {}", fmt_kind(attrs.kind))?;
        writeln!(
            output,
            "mode:   {:04o}  ({})",
            attrs.mode & 0o7777,
            fmt_mode(attrs.kind, attrs.mode)
        )?;
        writeln!(output, "owner:  {}:{}", attrs.uid, attrs.gid)?;
        writeln!(output, "size:   {} bytes", attrs.size)?;
        writeln!(output, "blocks: {}  (512-byte units)", attrs.blocks)?;
        writeln!(output, "nlink:  {}", attrs.nlink)?;
        writeln!(output, "inode:  {}", attrs.inode)?;
        writeln!(
            output,
            "atime:  {}  ({})",
            attrs.atime,
            fmt_unix_utc(attrs.atime)
        )?;
        writeln!(
            output,
            "mtime:  {}  ({})",
            attrs.mtime,
            fmt_unix_utc(attrs.mtime)
        )?;
        writeln!(
            output,
            "ctime:  {}  ({})",
            attrs.ctime,
            fmt_unix_utc(attrs.ctime)
        )?;
        match attrs.kind {
            fstool::fs::EntryKind::Char | fstool::fs::EntryKind::Block => {
                let (maj, min) = fstool::fs::ext::inode::decode_devnum(attrs.rdev);
                writeln!(
                    output,
                    "rdev:   {:#x}  (major {maj}, minor {min})",
                    attrs.rdev
                )?;
            }
            _ => writeln!(output, "rdev:   -")?,
        }

        // Symlinks: also surface the target. Backends that don't carry
        // symlinks (FAT/exFAT) error here; we just skip on error so
        // info on a non-symlink still works.
        if matches!(attrs.kind, fstool::fs::EntryKind::Symlink)
            && let Ok(tgt) = self.fs.read_symlink(dev, &path)
        {
            writeln!(output, "target: {tgt}")?;
        }

        // Extended attributes — fs-specific metadata in a generic shape.
        // Empty xattr lists are common (most ext images, most files),
        // so omit the section entirely when none.
        let xattrs = self.fs.list_xattrs(dev, Path::new(&path))?;
        if !xattrs.is_empty() {
            writeln!(output)?;
            writeln!(output, "xattrs ({}):", xattrs.len())?;
            for xa in &xattrs {
                writeln!(output, "  {:<28} = {}", xa.name, fmt_xattr_value(&xa.value))?;
            }
        }
        Ok(())
    }

    /// Resolve `path` against [`Self::cwd`]: absolute paths normalise as
    /// themselves; relative paths are joined onto cwd. Both then go
    /// through [`normalize_path`] to collapse `.`, `..`, and `//`.
    fn resolve(&self, path: &str) -> String {
        let combined = if path.starts_with('/') {
            path.to_string()
        } else {
            join(&self.cwd, path)
        };
        normalize_path(&combined)
    }
}

/// Where the interactive shell persists its command history. `~/.fstool_history`
/// on Unix (via `$HOME`), `%USERPROFILE%\.fstool_history` on Windows. Returns
/// `None` when neither home variable is set, in which case history is
/// session-only.
#[cfg(feature = "readline")]
fn history_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(std::path::Path::new(&home).join(".fstool_history"))
}

fn split_cmd(line: &str) -> (&str, &str) {
    match line.find(char::is_whitespace) {
        Some(i) => (&line[..i], line[i..].trim()),
        None => (line, ""),
    }
}

fn join(base: &str, rel: &str) -> String {
    if base.ends_with('/') {
        format!("{base}{rel}")
    } else {
        format!("{base}/{rel}")
    }
}

/// Collapse `.`, `..`, and empty segments into an absolute, normalised
/// path. `..` past root is a no-op.
pub fn normalize_path(p: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    if out.is_empty() {
        "/".into()
    } else {
        format!("/{}", out.join("/"))
    }
}

// ---------- formatting helpers for `cmd_info` ----------

/// Human-readable name for a [`fstool::fs::EntryKind`].
fn fmt_kind(kind: fstool::fs::EntryKind) -> &'static str {
    use fstool::fs::EntryKind;
    match kind {
        EntryKind::Regular => "regular file",
        EntryKind::Dir => "directory",
        EntryKind::Symlink => "symbolic link",
        EntryKind::Char => "character device",
        EntryKind::Block => "block device",
        EntryKind::Fifo => "fifo",
        EntryKind::Socket => "socket",
        EntryKind::Unknown => "unknown",
    }
}

/// Render POSIX permission bits in the `ls -l` shape — leading
/// type byte, three rwx triples, setuid/setgid/sticky overlays.
fn fmt_mode(kind: fstool::fs::EntryKind, mode: u16) -> String {
    use fstool::fs::EntryKind;
    let mut s = String::with_capacity(10);
    s.push(match kind {
        EntryKind::Regular => '-',
        EntryKind::Dir => 'd',
        EntryKind::Symlink => 'l',
        EntryKind::Char => 'c',
        EntryKind::Block => 'b',
        EntryKind::Fifo => 'p',
        EntryKind::Socket => 's',
        EntryKind::Unknown => '?',
    });
    for shift in [6u16, 3, 0] {
        let bits = (mode >> shift) & 0o7;
        s.push(if bits & 0o4 != 0 { 'r' } else { '-' });
        s.push(if bits & 0o2 != 0 { 'w' } else { '-' });
        s.push(if bits & 0o1 != 0 { 'x' } else { '-' });
    }
    // setuid/setgid/sticky overlay on the x slots.
    let bytes: Vec<u8> = s.bytes().collect();
    let mut bytes = bytes;
    if mode & 0o4000 != 0 {
        bytes[3] = if bytes[3] == b'x' { b's' } else { b'S' };
    }
    if mode & 0o2000 != 0 {
        bytes[6] = if bytes[6] == b'x' { b's' } else { b'S' };
    }
    if mode & 0o1000 != 0 {
        bytes[9] = if bytes[9] == b'x' { b't' } else { b'T' };
    }
    String::from_utf8(bytes).unwrap()
}

/// Format a Unix epoch second count as an ISO-8601 UTC string
/// (`YYYY-MM-DDTHH:MM:SSZ`). Uses Hinnant's `civil_from_days`
/// algorithm — no external date crate, valid for all positive `u32`
/// timestamps (up to year 2106).
fn fmt_unix_utc(t: u32) -> String {
    let total = t as i64;
    let days = total.div_euclid(86_400);
    let sod = total.rem_euclid(86_400) as u32;
    let z = days + 719_468;
    let era = if z >= 0 {
        z / 146_097
    } else {
        (z - 146_096) / 146_097
    };
    let doe = z - era * 146_097; // [0, 146097)
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    let h = sod / 3600;
    let mn = (sod / 60) % 60;
    let s = sod % 60;
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mn:02}:{s:02}Z")
}

/// Format an xattr value for display. Pure-ASCII (printable + tab/LF)
/// renders as a quoted string; otherwise prints the byte length plus a
/// hex preview of the first 16 bytes. Keeps single-line so the
/// `name = value` layout stays scannable.
fn fmt_xattr_value(value: &[u8]) -> String {
    let is_printable = !value.is_empty()
        && value
            .iter()
            .all(|&b| matches!(b, b'\t' | b'\n' | 0x20..=0x7e));
    if is_printable {
        // Strip a trailing newline so the line stays tight.
        let s = std::str::from_utf8(value).unwrap();
        let s = s.strip_suffix('\n').unwrap_or(s);
        return format!("{:?}", s);
    }
    let n = value.len();
    let preview_len = n.min(16);
    let mut hex = String::with_capacity(preview_len * 3);
    for (i, b) in value[..preview_len].iter().enumerate() {
        if i > 0 {
            hex.push(' ');
        }
        hex.push_str(&format!("{b:02x}"));
    }
    if n > preview_len {
        format!("<{n} bytes> {hex}…")
    } else {
        format!("<{n} bytes> {hex}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_roots() {
        assert_eq!(normalize_path("/"), "/");
        assert_eq!(normalize_path(""), "/");
        assert_eq!(normalize_path("///"), "/");
    }

    #[test]
    fn normalize_collapses_dotdot() {
        assert_eq!(normalize_path("/a/b/../c"), "/a/c");
        assert_eq!(normalize_path("/a/../.."), "/");
        assert_eq!(normalize_path("/./a/./b/"), "/a/b");
    }

    #[test]
    fn split_cmd_simple() {
        assert_eq!(split_cmd("ls"), ("ls", ""));
        assert_eq!(split_cmd("ls /etc"), ("ls", "/etc"));
        assert_eq!(split_cmd("put a b"), ("put", "a b"));
    }

    #[test]
    fn fmt_mode_renders_ls_l_layout() {
        use fstool::fs::EntryKind;
        assert_eq!(super::fmt_mode(EntryKind::Regular, 0o644), "-rw-r--r--");
        assert_eq!(super::fmt_mode(EntryKind::Dir, 0o755), "drwxr-xr-x");
        assert_eq!(super::fmt_mode(EntryKind::Symlink, 0o777), "lrwxrwxrwx");
        assert_eq!(super::fmt_mode(EntryKind::Char, 0o600), "crw-------");
        assert_eq!(super::fmt_mode(EntryKind::Block, 0o660), "brw-rw----");
        // Setuid / setgid / sticky overlays.
        assert_eq!(super::fmt_mode(EntryKind::Regular, 0o4755), "-rwsr-xr-x");
        assert_eq!(super::fmt_mode(EntryKind::Regular, 0o4644), "-rwSr--r--");
        assert_eq!(super::fmt_mode(EntryKind::Dir, 0o1755), "drwxr-xr-t");
    }

    #[test]
    fn fmt_unix_utc_known_epochs() {
        // The Unix epoch.
        assert_eq!(super::fmt_unix_utc(0), "1970-01-01T00:00:00Z");
        // 2001-09-09T01:46:40Z — the iconic 1e9 timestamp.
        assert_eq!(super::fmt_unix_utc(1_000_000_000), "2001-09-09T01:46:40Z");
        // 2023-11-14T22:13:20Z — the 1.7e9 mark.
        assert_eq!(super::fmt_unix_utc(1_700_000_000), "2023-11-14T22:13:20Z");
    }

    #[test]
    fn fmt_xattr_value_chooses_string_or_hex() {
        // Printable ASCII renders as a quoted string.
        assert_eq!(super::fmt_xattr_value(b"text/plain"), r#""text/plain""#);
        // Trailing newline gets stripped so the output stays single-line.
        assert_eq!(super::fmt_xattr_value(b"v1\n"), r#""v1""#);
        // Non-printable bytes fall back to <N bytes> + hex preview.
        let mut v = b"\x01\x00\x04\x80".to_vec();
        assert_eq!(super::fmt_xattr_value(&v), "<4 bytes> 01 00 04 80");
        // Long values truncate after 16 bytes with a `…` marker.
        v = (0u8..=31).collect();
        let s = super::fmt_xattr_value(&v);
        assert!(s.starts_with("<32 bytes> "), "{s}");
        assert!(s.ends_with('…'), "{s}");
    }
}
