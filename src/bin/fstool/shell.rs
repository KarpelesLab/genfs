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
//!   info                show image summary
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
}

impl Shell {
    /// A new shell rooted at `/` over `fs`.
    pub fn new(fs: AnyFs) -> Self {
        Self {
            fs,
            cwd: "/".into(),
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
                self.cmd_put(dev, rest, output)?;
                Ok(false)
            }
            "rm" => {
                self.cmd_rm(dev, rest, output)?;
                Ok(false)
            }
            "mkdir" => {
                self.cmd_mkdir(dev, rest, output)?;
                Ok(false)
            }
            "info" => {
                self.cmd_info(output)?;
                Ok(false)
            }
            "" => Ok(false),
            other => Err(fstool::Error::InvalidArgument(format!(
                "unknown command {other:?} (try `help`)"
            ))),
        }
    }

    fn cmd_help(&self, output: &mut impl Write) -> Result<()> {
        let body = "\
ls [PATH]           list a directory (default: cwd)
pwd                 print the current directory
cd [PATH]           change directory (no arg → /)
cat PATH            print a file's contents to stdout
put HOST [DEST]     copy a host file or directory into the image
rm PATH             remove a file or empty directory
mkdir PATH          create an empty directory
info                show image summary
help | ?            print this help
quit | exit         leave\n";
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

    fn cmd_info(&self, output: &mut impl Write) -> Result<()> {
        writeln!(output, "fs kind: {}", self.fs.kind_string())?;
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
}
