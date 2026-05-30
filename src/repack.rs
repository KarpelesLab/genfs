//! Repack — copy file trees from one source into a freshly-formatted
//! destination filesystem.
//!
//! Three kinds of sources, exposed as [`Source`]:
//!
//! * A **host directory** (`Source::HostDir`) — the original
//!   `fstool create` flow, walks a directory tree.
//! * A **tar archive** on disk (`Source::TarArchive`), with optional
//!   compression codec. Compressed archives go through a two-pass
//!   stream-index → replay flow; plain `.tar` falls through to the
//!   `Image` path (the regular tar reader sits on top of a
//!   [`crate::block::BlockDevice`]).
//! * An **existing image** (`Source::Image`) — a raw or qcow2 file,
//!   optionally with a `:N` partition selector. Walks the source FS
//!   through [`AnyFs`](crate::inspect::AnyFs) and copies entries
//!   straight through without host-filesystem intermediation.
//!
//! The two main entry points — [`populate_ext_from_source`] and
//! [`populate_fat32_from_source`] — take an already-formatted
//! destination filesystem and stream the chosen source's contents
//! into it. Auto-sizing helpers ([`ext_build_plan_for_source`],
//! [`fat32_min_bytes_for_source`]) let callers right-size the
//! destination geometry up-front.
//!
//! Used by both the `fstool repack` CLI command and by the
//! [`spec`](crate::spec) layer when a TOML `source = "..."` value
//! points at a tar / image instead of a directory.

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::Result;
use crate::block::BlockDevice;
use crate::compression::Algo;
use crate::fs::ext::xattr::Xattr;
use crate::fs::ext::{Ext, FsKind};
use crate::fs::tar::{TarEntryMeta, TarStreamWriter};
use crate::fs::{DeviceKind, FileMeta, Filesystem, XattrPair};

/// Progress reporter for a running repack. The CLI installs one with
/// [`enter`] before driving the copy path and tears it down with
/// [`leave`] afterward; the inner copy code calls [`note`] for each
/// entry it processes.
///
/// Output style adapts to the destination:
/// - TTY → refreshing single line (`\r` + ANSI clear-to-end-of-line),
///   throttled to one update per 200 ms.
/// - Otherwise → silent unless explicitly verbose, in which case a
///   fresh line lands every 500 entries.
pub struct Progress {
    files: u64,
    last_emit: Instant,
    last_path: String,
    is_tty: bool,
    verbose: bool,
    started: Instant,
    /// Total entry count for the current phase, when known. Set by
    /// [`set_total`] after the analyze pass; until then or on phases
    /// without a totals signal, falls back to the filename ticker.
    total_files: Option<u64>,
    /// Total file-body bytes for the current phase. Same lifecycle as
    /// `total_files`.
    total_bytes: Option<u64>,
    /// File-body bytes written so far in the current phase. Reset at
    /// every [`phase`] boundary.
    bytes_done: u64,
}

impl Progress {
    /// Construct with style auto-detected from stderr.
    pub fn auto() -> Self {
        let now = Instant::now();
        Self {
            files: 0,
            // Treat the first note as immediately due.
            last_emit: now - Duration::from_secs(1),
            last_path: String::new(),
            is_tty: std::io::stderr().is_terminal(),
            verbose: false,
            started: now,
            total_files: None,
            total_bytes: None,
            bytes_done: 0,
        }
    }

    /// Force on (line-per-tick), useful when stderr is captured to a
    /// log and the user still wants periodic progress.
    pub fn verbose() -> Self {
        let mut p = Self::auto();
        p.verbose = true;
        p
    }

    fn note_inner(&mut self, path: &str) {
        self.files += 1;
        self.last_path.clear();
        self.last_path.push_str(path);
        self.emit_status();
    }

    /// Add `n` bytes to the running body-bytes counter so a phase with
    /// known totals can render a byte-accurate progress bar. Triggers
    /// the same throttled display refresh as [`note_inner`] — useful for
    /// large files whose body is streamed between `note` calls.
    fn note_bytes_inner(&mut self, n: u64) {
        self.bytes_done = self.bytes_done.saturating_add(n);
        self.emit_status();
    }

    /// Render a status line (or progress bar) to stderr, throttled to ≤
    /// 5 Hz so a tight per-file loop doesn't spam.
    fn emit_status(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_emit) < Duration::from_millis(200) {
            return;
        }
        self.last_emit = now;
        // On a TTY, sample the terminal width per emit and let
        // `status_line` budget the path-ticker variant against it. Bar
        // mode is already fixed-width and ignores the hint.
        let cols = if self.is_tty { term_cols() } else { None };
        let line = self.status_line(cols);
        if self.is_tty {
            let mut err = std::io::stderr().lock();
            let _ = write!(err, "\r\x1b[Krepack: {line}");
            let _ = err.flush();
        } else if self.verbose && self.files.is_multiple_of(500) {
            eprintln!("repack: {line}");
        }
    }

    /// Build the "progress bar" or "N files | last_path" line for the
    /// current state. With totals (set after analyze) the line reads
    /// `[████░░░░░░░░] 38%  38247/100001 files  39.2/100.0 MiB`; without,
    /// it falls back to the `N files | path` ticker — and when `cols`
    /// is known, the path is truncated on the **left** (with an `…`
    /// marker) so the filename suffix stays visible on narrow terminals.
    fn status_line(&self, cols: Option<usize>) -> String {
        match (self.total_files, self.total_bytes) {
            (Some(tf), Some(tb)) if tf > 0 || tb > 0 => {
                let file_frac = if tf > 0 {
                    (self.files as f64 / tf as f64).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let byte_frac = if tb > 0 {
                    (self.bytes_done as f64 / tb as f64).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                // Weighted blend: bytes dominate when there's real data,
                // file count dominates for tiny/empty files. (Most repack
                // workloads are one or the other, so this is mostly
                // cosmetic.)
                let frac = if tb > 0 { byte_frac } else { file_frac };
                let bar = render_bar(24, frac);
                format!(
                    "{bar} {:3.0}%  {}/{} files  {}/{}",
                    frac * 100.0,
                    self.files,
                    tf,
                    human_bytes(self.bytes_done),
                    human_bytes(tb),
                )
            }
            _ => {
                let prefix = format!("{} files | ", self.files);
                // "repack: " (8) + prefix + path must fit inside `cols`.
                // Reserve 1 column so the cursor doesn't push us into a
                // wrap on tight terminals.
                let path = match cols {
                    Some(c) => {
                        let budget = c.saturating_sub(8 + prefix.chars().count() + 1);
                        truncate_left(&self.last_path, budget)
                    }
                    None => self.last_path.clone(),
                };
                format!("{prefix}{path}")
            }
        }
    }

    /// Emit a coarse phase marker (decompress / scan / format) on its
    /// own line. Unlike per-file [`note`], phases print in **every**
    /// mode — TTY and pipe/log alike — because they're the milestones
    /// that explain the otherwise-silent seconds before files start
    /// streaming (decompressing a big `.tar.gz`, formatting a multi-GB
    /// destination). There are only a handful per run, so they don't
    /// spam a captured log.
    fn phase_inner(&mut self, msg: &str) {
        // Reset the throttle so the first `note` after a phase isn't
        // swallowed by the 200 ms window.
        self.last_emit = Instant::now();
        if self.is_tty {
            let mut err = std::io::stderr().lock();
            // Clear any in-progress refreshing line first, then drop the
            // phase on its own line so it stays above the file counter.
            let _ = writeln!(err, "\r\x1b[Krepack: {msg}");
            let _ = err.flush();
        } else {
            eprintln!("repack: {msg}");
        }
    }

    fn finish_inner(&self) {
        let elapsed = self.started.elapsed();
        if self.is_tty {
            let mut err = std::io::stderr().lock();
            let _ = writeln!(
                err,
                "\r\x1b[Krepack: {} files in {:.1}s",
                self.files,
                elapsed.as_secs_f32()
            );
        } else if self.verbose || self.files > 0 {
            eprintln!(
                "repack: {} files in {:.1}s",
                self.files,
                elapsed.as_secs_f32()
            );
        }
    }
}

/// Sample the terminal's column count from stderr (file descriptor 2)
/// via the `TIOCGWINSZ` ioctl. Returns `None` when stderr isn't a TTY,
/// the ioctl fails, or the width comes back zero (which some terminals
/// report transiently during resizes). Unix-only; on other platforms
/// we just don't truncate.
#[cfg(unix)]
fn term_cols() -> Option<usize> {
    use std::os::fd::AsRawFd;
    let fd = std::io::stderr().as_raw_fd();
    // SAFETY: `winsize` is a POD layout populated by the kernel; we
    // pass a valid mutable pointer and check the return code before
    // reading the fields.
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws) };
    if rc != 0 || ws.ws_col == 0 {
        return None;
    }
    Some(ws.ws_col as usize)
}

#[cfg(not(unix))]
fn term_cols() -> Option<usize> {
    None
}

/// Truncate `path` on the **left** so the result fits in `budget` cells,
/// preserving the trailing characters (filename and any short tail of
/// its parent path). When trimmed, a leading `…` marks the elision.
/// Counts Unicode scalar values, which is a good-enough proxy for
/// terminal cells for the ASCII path component names we typically deal
/// with. (Wide-CJK paths may overshoot by a column or two.)
fn truncate_left(path: &str, budget: usize) -> String {
    if budget == 0 {
        return String::new();
    }
    // Fast path: already fits.
    if path.chars().count() <= budget {
        return path.to_string();
    }
    // Reserve one cell for the `…` marker.
    let keep = budget.saturating_sub(1);
    // Take the last `keep` chars, byte-safe.
    let (byte_start, _) = path
        .char_indices()
        .rev()
        .nth(keep.saturating_sub(1))
        .unwrap_or((path.len(), '\0'));
    let mut out = String::with_capacity(1 + path.len() - byte_start);
    out.push('…');
    out.push_str(&path[byte_start..]);
    out
}

/// Render a `[████░░░░]` bar of width `cells`. Unicode block characters
/// for the filled portion, light shade for the empty.
fn render_bar(cells: usize, frac: f64) -> String {
    let filled = (cells as f64 * frac).round() as usize;
    let filled = filled.min(cells);
    let mut s = String::with_capacity(cells + 2);
    s.push('[');
    for i in 0..cells {
        s.push(if i < filled { '█' } else { '░' });
    }
    s.push(']');
    s
}

/// Format `n` bytes as e.g. `47.5 MiB` / `1.2 GiB`. Compact and unitless
/// past 4 GiB to keep the progress line short.
fn human_bytes(n: u64) -> String {
    const KIB: f64 = 1024.0;
    let f = n as f64;
    if f < KIB {
        return format!("{n} B");
    }
    let mib = f / KIB / KIB;
    if mib < 1.0 {
        return format!("{:.1} KiB", f / KIB);
    }
    let gib = mib / KIB;
    if gib < 1.0 {
        return format!("{mib:.1} MiB");
    }
    format!("{gib:.2} GiB")
}

thread_local! {
    static ACTIVE_PROGRESS: std::cell::RefCell<Option<Progress>> =
        const { std::cell::RefCell::new(None) };
}

/// Install `p` as the active progress sink for this thread. Inner copy
/// code calls [`note`] without knowing whether progress is wired up;
/// the sink is per-thread so different concurrent repacks don't trample
/// each other.
pub fn enter(p: Progress) {
    ACTIVE_PROGRESS.with(|cell| *cell.borrow_mut() = Some(p));
}

/// Tear down the active progress sink, emitting its final summary.
/// No-op if `enter` wasn't called.
pub fn leave() {
    ACTIVE_PROGRESS.with(|cell| {
        if let Some(p) = cell.borrow().as_ref() {
            p.finish_inner();
        }
        *cell.borrow_mut() = None;
    });
}

/// Record one processed entry at `path`. No-op when no sink is active.
pub fn note(path: &str) {
    ACTIVE_PROGRESS.with(|cell| {
        if let Some(p) = cell.borrow_mut().as_mut() {
            p.note_inner(path);
        }
    });
}

/// Announce a coarse phase boundary (e.g. "decompressing …",
/// "scanning source …", "formatting … destination"). Prints on its own
/// line in both TTY and pipe/log modes and **resets** the running file
/// counter — every emitted phase is the start of a new pass, so the
/// user-visible "repack: N files" reflects the current phase rather
/// than summing across passes (without this a two-pass build double-
/// counts every entry, since both the analyze pass and the copy pass
/// walk the source). No-op when no sink is active.
pub fn phase(msg: &str) {
    ACTIVE_PROGRESS.with(|cell| {
        if let Some(p) = cell.borrow_mut().as_mut() {
            p.files = 0;
            p.last_path.clear();
            p.bytes_done = 0;
            // Totals are per-phase too; callers wanting a progress bar
            // on the next phase call `set_total` after `phase`.
            p.total_files = None;
            p.total_bytes = None;
            p.phase_inner(msg);
        }
    });
}

/// Record `n` bytes of file body written. Used by the streaming walkers
/// to drive the progress bar's byte fraction. No-op when no sink is
/// active.
pub fn note_bytes(n: u64) {
    ACTIVE_PROGRESS.with(|cell| {
        if let Some(p) = cell.borrow_mut().as_mut() {
            p.note_bytes_inner(n);
        }
    });
}

/// Tell the active progress sink the totals for the current phase so it
/// can render a percentage / bar. Call this **after** `phase("…")` (which
/// clears any previous totals) and **before** the copy walk starts. The
/// numbers come from the analyze pass: `files` = directory entries to
/// emit (regular + dir + symlink + device), `bytes` = total regular-file
/// body bytes. No-op when no sink is active.
pub fn set_total(files: u64, bytes: u64) {
    ACTIVE_PROGRESS.with(|cell| {
        if let Some(p) = cell.borrow_mut().as_mut() {
            p.total_files = Some(files);
            p.total_bytes = Some(bytes);
        }
    });
}

/// Where to draw a filesystem's contents from when building or
/// populating it. See module docs.
#[derive(Debug, Clone)]
pub enum Source {
    /// A directory on the host filesystem, walked recursively.
    HostDir(PathBuf),
    /// A tar archive on disk, plain or compressed.
    TarArchive { path: PathBuf, codec: Option<Algo> },
    /// An existing image, optionally with a `:N` partition selector.
    Image(crate::inspect::Target),
    /// Multiple sources stacked bottom→top. Upper layers override
    /// files of the same path; tombstones (`.wh.*` files, character
    /// device 0/0) delete paths from lower layers. See
    /// [`crate::merge`] for the fold semantics.
    Layered(Vec<Source>),
}

impl Source {
    /// Auto-detect what kind of source `spec` points at.
    ///
    /// * `a+b+c` (`+`-separated specs) → `Layered`, applied bottom→top.
    /// * An existing directory path → `HostDir`.
    /// * A recognised tar extension (`.tar`, `.tar.gz`, `.tgz`,
    ///   `.tar.xz`, `.txz`, `.tar.zst`, `.tar.lz4`, `.tar.lzma`,
    ///   `.tar.lzo`) → `TarArchive`.
    /// * Anything else, including a `path:N` partition selector
    ///   → `Image`. Parsed by [`crate::inspect::Target::parse`].
    pub fn detect(spec: &str) -> Result<Self> {
        // Layered: `a+b+c` (bottom=a, top=c). The single `+` separator
        // never collides with real paths in practice — `+` is rare in
        // filenames and `path:partition` syntax uses `:` not `+`.
        if spec.contains('+') {
            let parts: Vec<_> = spec
                .split('+')
                .filter(|p| !p.is_empty())
                .map(Self::detect)
                .collect::<Result<_>>()?;
            if parts.len() > 1 {
                return Ok(Self::Layered(parts));
            }
            if let Some(single) = parts.into_iter().next() {
                return Ok(single);
            }
        }
        // Strip the partition suffix (`:N`) only when `N` is purely
        // numeric — otherwise a Windows drive letter (`C:\foo.tar`)
        // gets mistaken for a `path:partition` spec and the rest of
        // the detection logic loses the file extension.
        let bare = match spec.rsplit_once(':') {
            Some((head, tail)) if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => {
                head
            }
            _ => spec,
        };
        let bare_path = Path::new(bare);
        if bare == spec
            && let Ok(meta) = std::fs::metadata(bare_path)
            && meta.is_dir()
        {
            return Ok(Self::HostDir(bare_path.to_path_buf()));
        }
        if let Some(codec) = tar_input_codec(spec) {
            return Ok(Self::TarArchive {
                path: bare_path.to_path_buf(),
                codec: Some(codec),
            });
        }
        if has_plain_tar_extension(bare_path) {
            return Ok(Self::TarArchive {
                path: bare_path.to_path_buf(),
                codec: None,
            });
        }
        Ok(Self::Image(crate::inspect::Target::parse(spec)))
    }
}

fn has_plain_tar_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("tar"))
}

// ======================================================================
// Unified repack: one source walker → one of two sinks. The only branch
// is streaming output (tar / compressed tar) vs a block-device-backed
// filesystem. There is NO per-(source,dest) special-casing — fidelity
// comes uniformly from the source's trait `getattr` / `list_xattrs` /
// `read_symlink` / `rdev`.
// ======================================================================

/// Per-entry metadata carried through [`RepackSink`]. A superset of both
/// [`FileMeta`] and [`TarEntryMeta`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RepackMeta {
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub mtime: u32,
    pub atime: u32,
    pub ctime: u32,
}

impl RepackMeta {
    /// Default metadata for a directory created on demand (e.g. a tar
    /// entry whose parent dir wasn't an explicit archive member):
    /// `0755`, root-owned, zero times.
    fn dir_default() -> Self {
        Self {
            mode: 0o755,
            uid: 0,
            gid: 0,
            mtime: 0,
            atime: 0,
            ctime: 0,
        }
    }

    fn to_file_meta(self) -> FileMeta {
        FileMeta {
            mode: self.mode,
            uid: self.uid,
            gid: self.gid,
            mtime: self.mtime,
            atime: self.atime,
            ctime: self.ctime,
        }
    }

    fn to_tar_meta(self) -> TarEntryMeta {
        TarEntryMeta {
            mode: self.mode,
            uid: self.uid,
            gid: self.gid,
            mtime: u64::from(self.mtime),
            uname: String::new(),
            gname: String::new(),
        }
    }
}

/// The destination side of a repack. A source walk drives exactly one of
/// these; the two implementations are an FS-backed sink ([`FsSink`]) and
/// a streaming-tar sink ([`TarStreamSink`]).
pub trait RepackSink {
    fn put_dir(&mut self, path: &str, meta: RepackMeta, xattrs: &[XattrPair]) -> Result<()>;
    fn put_file(
        &mut self,
        path: &str,
        body: &mut dyn Read,
        len: u64,
        meta: RepackMeta,
        xattrs: &[XattrPair],
    ) -> Result<()>;
    fn put_symlink(
        &mut self,
        path: &str,
        target: &str,
        meta: RepackMeta,
        xattrs: &[XattrPair],
    ) -> Result<()>;
    fn put_device(
        &mut self,
        path: &str,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: RepackMeta,
        xattrs: &[XattrPair],
    ) -> Result<()>;
    /// Create `path` as a hard link to the already-emitted `target`.
    /// Returns `Ok(false)` when the sink can't represent hard links —
    /// the walker then materialises the body with a fresh [`Self::put_file`].
    fn put_hardlink(
        &mut self,
        path: &str,
        target: &str,
        meta: RepackMeta,
        xattrs: &[XattrPair],
    ) -> Result<bool>;
    /// Materialise `path` as an independent copy of the already-written
    /// `target`. Used by the streaming tar walker when `put_hardlink`
    /// returns `false` (the destination can't represent a hard link)
    /// and the source body has already streamed past — so the copy is
    /// sourced from the destination, not the source. Default errors;
    /// only [`FsSink`] implements it.
    fn materialise_copy(
        &mut self,
        _path: &str,
        _target: &str,
        _meta: RepackMeta,
        _xattrs: &[XattrPair],
    ) -> Result<()> {
        Err(crate::Error::Unsupported(
            "repack: sink can't materialise a hard-link copy".into(),
        ))
    }
    /// Finalise the destination (flush / write the archive trailer).
    fn finish(&mut self) -> Result<()>;
}

/// Convert `XattrPair`s (the trait-surface xattr type) to the tar
/// writer's `Xattr` type — a plain field rename.
fn xattrs_to_tar(xattrs: &[XattrPair]) -> Vec<Xattr> {
    xattrs
        .iter()
        .map(|x| Xattr {
            name: x.name.clone(),
            value: x.value.clone(),
        })
        .collect()
}

/// Sink that writes into any block-device-backed [`Filesystem`] through
/// the trait. `lossy` (set for FAT/exFAT) drops symlinks / devices /
/// xattrs the destination can't represent instead of erroring.
pub struct FsSink<'a> {
    dst: &'a mut dyn Filesystem,
    dev: &'a mut dyn BlockDevice,
    lossy: bool,
}

impl<'a> FsSink<'a> {
    pub fn new(dst: &'a mut dyn Filesystem, dev: &'a mut dyn BlockDevice) -> Self {
        Self {
            dst,
            dev,
            lossy: false,
        }
    }

    /// Mark the destination as metadata-poor (FAT/exFAT): unrepresentable
    /// entries are dropped with a warning rather than failing the repack.
    pub fn lossy(mut self) -> Self {
        self.lossy = true;
        self
    }

    /// Apply xattrs to a just-created path in one batch, swallowing
    /// `Unsupported` (backends without xattr storage) — and, when
    /// `lossy`, swallowing any error.
    fn apply_xattrs(&mut self, path: &str, xattrs: &[XattrPair]) -> Result<()> {
        if xattrs.is_empty() {
            return Ok(());
        }
        match self.dst.set_xattrs(self.dev, Path::new(path), xattrs) {
            Ok(()) => Ok(()),
            Err(crate::Error::Unsupported(_)) => Ok(()),
            Err(e) if self.lossy => {
                eprintln!("repack: dropping xattrs on {path:?}: {e}");
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Run `op`, tolerating an `Unsupported`/error result on a lossy
    /// (FAT-like) destination by dropping the entry with a warning.
    fn lossy_ok(&self, path: &str, what: &str, r: Result<()>) -> Result<()> {
        match r {
            Ok(()) => Ok(()),
            Err(e) if self.lossy => {
                eprintln!("repack: dropping {what} {path:?} — destination can't represent it: {e}");
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

impl RepackSink for FsSink<'_> {
    fn put_dir(&mut self, path: &str, meta: RepackMeta, xattrs: &[XattrPair]) -> Result<()> {
        self.dst
            .create_dir(self.dev, Path::new(path), meta.to_file_meta())?;
        self.apply_xattrs(path, xattrs)
    }

    fn put_file(
        &mut self,
        path: &str,
        body: &mut dyn Read,
        len: u64,
        meta: RepackMeta,
        xattrs: &[XattrPair],
    ) -> Result<()> {
        self.dst.create_file_streaming(
            self.dev,
            Path::new(path),
            body,
            len,
            meta.to_file_meta(),
        )?;
        self.apply_xattrs(path, xattrs)
    }

    fn put_symlink(
        &mut self,
        path: &str,
        target: &str,
        meta: RepackMeta,
        xattrs: &[XattrPair],
    ) -> Result<()> {
        let r = self.dst.create_symlink(
            self.dev,
            Path::new(path),
            Path::new(target),
            meta.to_file_meta(),
        );
        self.lossy_ok(path, "symlink", r)?;
        self.apply_xattrs(path, xattrs)
    }

    fn put_device(
        &mut self,
        path: &str,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: RepackMeta,
        xattrs: &[XattrPair],
    ) -> Result<()> {
        let r = self.dst.create_device(
            self.dev,
            Path::new(path),
            kind,
            major,
            minor,
            meta.to_file_meta(),
        );
        self.lossy_ok(path, "device node", r)?;
        self.apply_xattrs(path, xattrs)
    }

    fn put_hardlink(
        &mut self,
        path: &str,
        target: &str,
        _meta: RepackMeta,
        _xattrs: &[XattrPair],
    ) -> Result<bool> {
        match self
            .dst
            .hardlink(self.dev, Path::new(target), Path::new(path))
        {
            Ok(()) => Ok(true),
            // Destination has no hardlink concept → caller materialises.
            Err(crate::Error::Unsupported(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn materialise_copy(
        &mut self,
        path: &str,
        target: &str,
        meta: RepackMeta,
        xattrs: &[XattrPair],
    ) -> Result<()> {
        // The streaming walker has no source body to replay, so copy the
        // already-written `target` back out of the destination FS and
        // re-create it at `path`. Buffers the body in memory — only the
        // rare "hard link into a destination that can't link"
        // (FAT/exFAT) path reaches here, where it's a clean fallback.
        let mut buf = Vec::new();
        {
            let mut r = self.dst.read_file(self.dev, Path::new(target))?;
            r.read_to_end(&mut buf).map_err(crate::Error::from)?;
        }
        let len = buf.len() as u64;
        let mut cur = std::io::Cursor::new(buf);
        self.dst.create_file_streaming(
            self.dev,
            Path::new(path),
            &mut cur,
            len,
            meta.to_file_meta(),
        )?;
        self.apply_xattrs(path, xattrs)
    }

    fn finish(&mut self) -> Result<()> {
        self.dst.flush(self.dev)
    }
}

/// Sink that streams a tar archive (optionally codec-wrapped) to a
/// `Write`. Hard links are materialised (tar copies the body).
pub struct TarStreamSink {
    writer: TarStreamWriter<Box<dyn Write>>,
}

impl TarStreamSink {
    pub fn new(inner: Box<dyn Write>) -> Self {
        Self {
            writer: TarStreamWriter::new(inner),
        }
    }

    /// Plain (uncompressed) bytes written so far — used by the CLI's
    /// completion message.
    pub fn bytes_written(&self) -> u64 {
        self.writer.bytes_written()
    }
}

/// Tar stores relative member names; the walker emits absolute paths.
fn tar_name(path: &str) -> &str {
    path.trim_start_matches('/')
}

impl RepackSink for TarStreamSink {
    fn put_dir(&mut self, path: &str, meta: RepackMeta, xattrs: &[XattrPair]) -> Result<()> {
        self.writer
            .add_dir(tar_name(path), meta.to_tar_meta(), &xattrs_to_tar(xattrs))
    }

    fn put_file(
        &mut self,
        path: &str,
        body: &mut dyn Read,
        len: u64,
        meta: RepackMeta,
        xattrs: &[XattrPair],
    ) -> Result<()> {
        self.writer.add_file(
            tar_name(path),
            body,
            len,
            meta.to_tar_meta(),
            &xattrs_to_tar(xattrs),
        )
    }

    fn put_symlink(
        &mut self,
        path: &str,
        target: &str,
        meta: RepackMeta,
        xattrs: &[XattrPair],
    ) -> Result<()> {
        self.writer.add_symlink(
            tar_name(path),
            target,
            meta.to_tar_meta(),
            &xattrs_to_tar(xattrs),
        )
    }

    fn put_device(
        &mut self,
        path: &str,
        kind: DeviceKind,
        major: u32,
        minor: u32,
        meta: RepackMeta,
        xattrs: &[XattrPair],
    ) -> Result<()> {
        self.writer.add_device(
            tar_name(path),
            kind,
            major,
            minor,
            meta.to_tar_meta(),
            &xattrs_to_tar(xattrs),
        )
    }

    fn put_hardlink(
        &mut self,
        _path: &str,
        _target: &str,
        _meta: RepackMeta,
        _xattrs: &[XattrPair],
    ) -> Result<bool> {
        // Tar materialises hard links: tell the walker to copy the body.
        Ok(false)
    }

    fn finish(&mut self) -> Result<()> {
        self.writer.finish()
    }
}

/// Walk `source` and replay every entry into `sink`. This is the single
/// repack copy path: there is no per-(source,dest) branching — `sink`
/// decides whether the output is a streaming tar or a block-device FS.
/// The caller is responsible for `sink.finish()` (or, for an FS sink,
/// flushing the destination) afterwards.
pub fn walk_source_into_sink(source: &Source, sink: &mut dyn RepackSink) -> Result<()> {
    match source {
        Source::HostDir(p) => walk_host_dir(p, sink),
        Source::TarArchive { path, codec } => {
            // Stream the tar forward, whether compressed or not — a tar
            // is sequential, never seeked, so `Tar::open`'s upfront
            // index build is wasted work (it dominated profiles before
            // this arm was unified).
            let mut reader = open_tar_stream(path, *codec)?;
            walk_tar_stream(&mut reader, sink)
        }
        Source::Image(target) => walk_image(target, sink),
        Source::Layered(layers) => {
            // Build the merged tree in memory (metadata only) and drive
            // the sink directly — no temp file, RAM bounded by metadata.
            let model = crate::merge::MergeModel::build(layers)?;
            model.walk_into_sink(layers, sink)
        }
    }
}

/// Split a Linux-encoded `rdev` into `(major, minor)` — the inverse of
/// the encoding ext (and tar's `getattr`) use.
fn split_rdev(rdev: u32) -> (u32, u32) {
    crate::fs::ext::inode::decode_devnum(rdev)
}

/// Open `target` as an image and walk its filesystem into `sink`.
fn walk_image(target: &crate::inspect::Target, sink: &mut dyn RepackSink) -> Result<()> {
    crate::inspect::with_target_device(target, |src_dev| {
        let mut src_fs = crate::inspect::AnyFs::open(src_dev)?;
        // Replay any pending ext journal so we read the post-recovery
        // state (anything still in the log would otherwise be lost).
        if let crate::inspect::AnyFs::Ext(ext) = &mut src_fs {
            let _ = ext.replay_pending_journal(src_dev)?;
        }
        walk_anyfs(&mut src_fs, src_dev, sink)
    })
}

/// DFS over an opened source filesystem, emitting each entry into `sink`
/// with full metadata (`getattr`), xattrs (`list_xattrs`), symlink
/// targets, and device numbers. Hard links are de-duplicated by
/// `(inode, nlink)`; if the sink can't represent a link it materialises
/// the body.
///
/// Public so the CLI can drive a source it has already opened (e.g. for
/// sizing) without re-opening it.
pub fn walk_anyfs(
    src_fs: &mut crate::inspect::AnyFs,
    src_dev: &mut dyn BlockDevice,
    sink: &mut dyn RepackSink,
) -> Result<()> {
    use crate::fs::EntryKind;
    let mut link_map: std::collections::HashMap<(u32, u32), String> =
        std::collections::HashMap::new();
    let mut stack: Vec<String> = vec!["/".to_string()];
    while let Some(dir) = stack.pop() {
        for e in src_fs.list(src_dev, &dir)? {
            if e.name == "." || e.name == ".." || e.name == "lost+found" {
                continue;
            }
            let child = join_fs_path(&dir, &e.name);
            note(&child);
            let child_path = Path::new(&child);
            let attrs = src_fs.getattr(src_dev, child_path)?;
            let xattrs = src_fs.list_xattrs(src_dev, child_path)?;
            let meta = RepackMeta {
                mode: attrs.mode,
                uid: attrs.uid,
                gid: attrs.gid,
                mtime: attrs.mtime,
                atime: attrs.atime,
                ctime: attrs.ctime,
            };
            match attrs.kind {
                EntryKind::Dir => {
                    sink.put_dir(&child, meta, &xattrs)?;
                    stack.push(child);
                }
                EntryKind::Regular => {
                    if attrs.inode != 0 && attrs.nlink > 1 {
                        let key = (attrs.inode, attrs.nlink);
                        if let Some(first) = link_map.get(&key) {
                            if sink.put_hardlink(&child, first, meta, &xattrs)? {
                                continue;
                            }
                            // else: sink can't link — fall through and copy.
                        } else {
                            link_map.insert(key, child.clone());
                        }
                    }
                    let mut body = src_fs.open_body_reader(src_dev, &child)?;
                    sink.put_file(&child, &mut *body, attrs.size, meta, &xattrs)?;
                    note_bytes(attrs.size);
                }
                EntryKind::Symlink => {
                    let target = src_fs.read_symlink(src_dev, &child)?;
                    sink.put_symlink(&child, &target, meta, &xattrs)?;
                }
                EntryKind::Char | EntryKind::Block | EntryKind::Fifo | EntryKind::Socket => {
                    let (major, minor) = split_rdev(attrs.rdev);
                    let kind = match attrs.kind {
                        EntryKind::Char => DeviceKind::Char,
                        EntryKind::Block => DeviceKind::Block,
                        EntryKind::Fifo => DeviceKind::Fifo,
                        EntryKind::Socket => DeviceKind::Socket,
                        _ => unreachable!(),
                    };
                    sink.put_device(&child, kind, major, minor, meta, &xattrs)?;
                }
                EntryKind::Unknown => {
                    eprintln!("repack: skipping unknown entry {child:?}");
                }
            }
        }
    }
    Ok(())
}

/// Open a (possibly compressed) tar at `path` as a forward-only byte
/// stream — `make_reader` decompresses on the fly, so no tempfile is
/// ever written. Each call re-opens from byte 0; callers that need a
/// sizing pre-pass simply call this twice (the "read twice" the
/// streaming repack trades for not staging to disk).
pub fn open_tar_stream(path: &Path, codec: Option<Algo>) -> Result<Box<dyn Read>> {
    let file = std::io::BufReader::with_capacity(64 * 1024, std::fs::File::open(path)?);
    match codec {
        Some(algo) => crate::compression::make_reader(algo, file),
        None => Ok(Box::new(file)),
    }
}

/// Ensure every ancestor directory of `path` exists in `sink`, creating
/// any that are missing with default `0755` metadata. `created` tracks
/// what's already there (seed it with `"/"`). GNU tar emits parent dirs
/// before their contents, so this normally only fills genuine gaps; a
/// real dir entry that arrives later still lands via `put_dir` with its
/// true metadata (it won't be in `created` yet).
fn ensure_parents(
    path: &str,
    created: &mut std::collections::HashSet<String>,
    sink: &mut dyn RepackSink,
) -> Result<()> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return Ok(());
    }
    let parts: Vec<&str> = trimmed.split('/').collect();
    // All ancestors, shallow → deep (exclude the leaf itself).
    let mut cur = String::new();
    for seg in &parts[..parts.len() - 1] {
        cur.push('/');
        cur.push_str(seg);
        if created.insert(cur.clone()) {
            sink.put_dir(&cur, RepackMeta::dir_default(), &[])?;
        }
    }
    Ok(())
}

/// Stream a (decompressed) tar `reader` straight into `sink` — the
/// non-random-access counterpart to [`walk_anyfs`]. Drives
/// `tar::stream::TarStreamReader`, which resolves PAX / GNU long-name +
/// long-link overrides and exposes each entry's body as a `Read`. Bodies are
/// streamed (never fully resident); parent directories are created on
/// demand; hard links the destination can't represent fall back to a
/// destination-sourced copy via [`RepackSink::materialise_copy`].
///
/// Used for compressed-tar sources so they never decompress to a
/// tempfile. Does **not** call `sink.finish()` — the caller does that
/// after any final bookkeeping (matching `walk_anyfs`).
pub fn walk_tar_stream(reader: &mut dyn Read, sink: &mut dyn RepackSink) -> Result<()> {
    use crate::fs::tar::EntryKind as TarKind;
    use crate::fs::tar::stream::TarStreamReader;

    // Collapse `.` and empty path segments (a `tar -C dir .` archive
    // emits `./`-prefixed members; the stream reader's `normalise_path`
    // only fixes leading/trailing slashes, leaving interior `.`s that
    // would otherwise create a bogus `/.` directory).
    fn collapse(p: &str) -> String {
        let mut out = String::new();
        for seg in p.split('/').filter(|s| !s.is_empty() && *s != ".") {
            out.push('/');
            out.push_str(seg);
        }
        if out.is_empty() { "/".to_string() } else { out }
    }

    let mut tsr = TarStreamReader::new(reader);
    let mut created: std::collections::HashSet<String> =
        std::collections::HashSet::from(["/".to_string()]);

    while let Some(mut se) = tsr.next_entry()? {
        // Snapshot metadata off the entry before borrowing `se` as the
        // body reader for `put_file`.
        let path = collapse(&se.entry.path);
        let kind = se.entry.kind;
        let size = se.entry.size;
        let link = se.entry.link_target.clone();
        let (dmaj, dmin) = (se.entry.device_major, se.entry.device_minor);
        let meta = RepackMeta {
            mode: se.entry.mode,
            uid: se.entry.uid,
            gid: se.entry.gid,
            mtime: se.entry.mtime as u32,
            atime: se.entry.mtime as u32,
            ctime: se.entry.mtime as u32,
        };
        let xattrs: Vec<XattrPair> = se
            .entry
            .xattrs
            .iter()
            .map(|x| XattrPair {
                name: x.name.clone(),
                value: x.value.clone(),
            })
            .collect();

        if path == "/" {
            // The archive's root entry (rare) — nothing to create.
            continue;
        }
        note(&path);
        ensure_parents(&path, &mut created, sink)?;

        match kind {
            TarKind::Dir => {
                sink.put_dir(&path, meta, &xattrs)?;
                created.insert(path);
            }
            TarKind::Regular => {
                sink.put_file(&path, &mut se, size, meta, &xattrs)?;
                note_bytes(size);
            }
            TarKind::Symlink => {
                let target = link.as_deref().unwrap_or("");
                sink.put_symlink(&path, target, meta, &xattrs)?;
            }
            TarKind::HardLink => {
                // tar link targets are archive-relative; resolve to the
                // same absolute, `.`-collapsed form the entry paths use.
                let raw = link.as_deref().unwrap_or("");
                let target = collapse(&crate::fs::tar::normalise_path(raw));
                if !sink.put_hardlink(&path, &target, meta, &xattrs)? {
                    sink.materialise_copy(&path, &target, meta, &xattrs)?;
                }
            }
            TarKind::CharDev => {
                sink.put_device(&path, DeviceKind::Char, dmaj, dmin, meta, &xattrs)?;
            }
            TarKind::BlockDev => {
                sink.put_device(&path, DeviceKind::Block, dmaj, dmin, meta, &xattrs)?;
            }
            TarKind::Fifo => {
                sink.put_device(&path, DeviceKind::Fifo, 0, 0, meta, &xattrs)?;
            }
        }
    }
    Ok(())
}

/// Recursively walk a host directory tree into `sink`, preserving host
/// metadata (mode/uid/gid/times), symlinks, and — on Unix — device
/// nodes and hard links.
fn walk_host_dir(root: &Path, sink: &mut dyn RepackSink) -> Result<()> {
    #[cfg(unix)]
    let mut link_map: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    let mut stack: Vec<(PathBuf, String)> = vec![(root.to_path_buf(), "/".to_string())];
    while let Some((dir, fs_dir)) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_str().ok_or_else(|| {
                crate::Error::InvalidArgument(format!("repack: non-UTF-8 host filename {name:?}"))
            })?;
            let dest = join_fs_path(&fs_dir, name_str);
            note(&dest);
            // `DirEntry::metadata` does not traverse symlinks.
            let meta = entry.metadata()?;
            let ft = meta.file_type();
            let rmeta = host_meta_to_repack(&meta);
            if ft.is_dir() {
                sink.put_dir(&dest, rmeta, &[])?;
                stack.push((entry.path(), dest));
            } else if ft.is_symlink() {
                let target = std::fs::read_link(entry.path())?;
                let t = target.to_str().ok_or_else(|| {
                    crate::Error::InvalidArgument("repack: non-UTF-8 symlink target".into())
                })?;
                sink.put_symlink(&dest, t, rmeta, &[])?;
            } else if ft.is_file() {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::MetadataExt;
                    if meta.nlink() > 1 {
                        let ino = meta.ino();
                        if let Some(first) = link_map.get(&ino) {
                            if sink.put_hardlink(&dest, first, rmeta, &[])? {
                                continue;
                            }
                        } else {
                            link_map.insert(ino, dest.clone());
                        }
                    }
                }
                let mut f = std::fs::File::open(entry.path())?;
                sink.put_file(&dest, &mut f, meta.len(), rmeta, &[])?;
                note_bytes(meta.len());
            } else {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::{FileTypeExt, MetadataExt};
                    let (kind, major, minor) = if ft.is_char_device() {
                        let (maj, min) = split_rdev(meta.rdev() as u32);
                        (Some(DeviceKind::Char), maj, min)
                    } else if ft.is_block_device() {
                        let (maj, min) = split_rdev(meta.rdev() as u32);
                        (Some(DeviceKind::Block), maj, min)
                    } else if ft.is_fifo() {
                        (Some(DeviceKind::Fifo), 0, 0)
                    } else if ft.is_socket() {
                        (Some(DeviceKind::Socket), 0, 0)
                    } else {
                        (None, 0, 0)
                    };
                    if let Some(k) = kind {
                        sink.put_device(&dest, k, major, minor, rmeta, &[])?;
                        continue;
                    }
                }
                eprintln!("repack: skipping unsupported host entry {dest:?}");
            }
        }
    }
    Ok(())
}

/// Host `Metadata` → [`RepackMeta`].
fn host_meta_to_repack(meta: &std::fs::Metadata) -> RepackMeta {
    let fm = host_meta_to_fs(meta);
    RepackMeta {
        mode: fm.mode,
        uid: fm.uid,
        gid: fm.gid,
        mtime: fm.mtime,
        atime: fm.atime,
        ctime: fm.ctime,
    }
}

/// Populate `dst` (a freshly formatted ext{2,3,4}) with the contents
/// of `source`. The destination is assumed to already exist.
pub fn populate_ext_from_source(
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut Ext,
    source: &Source,
) -> Result<()> {
    let mut sink = FsSink::new(dst, dst_dev);
    walk_source_into_sink(source, &mut sink)
}

/// Populate `dst` (a freshly formatted FAT32) with the contents of
/// `source`. The destination is assumed to already exist. FAT can't
/// hold symlinks / device nodes / POSIX metadata, so the sink runs in
/// lossy mode (drop-with-warning).
pub fn populate_fat32_from_source(
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut crate::fs::fat::Fat32,
    source: &Source,
) -> Result<()> {
    let mut sink = FsSink::new(dst, dst_dev).lossy();
    walk_source_into_sink(source, &mut sink)
}

/// Build a [`BuildPlan`](crate::fs::ext::BuildPlan) sized for the
/// source. Walks the source once and feeds entry counts + byte totals
/// into the plan; the resulting `to_format_opts()` is ready to drive
/// `Ext::format_with`.
pub fn ext_build_plan_for_source(
    source: &Source,
    block_size: u32,
    kind: FsKind,
) -> Result<crate::fs::ext::BuildPlan> {
    let mut plan = crate::fs::ext::BuildPlan::new(block_size, kind);
    match source {
        Source::HostDir(p) => plan.scan_host_path(p)?,
        Source::TarArchive {
            path,
            codec: Some(algo),
        } => {
            let spec = path.to_string_lossy().into_owned();
            let index = build_tar_stream_index(&spec, *algo)?;
            walk_tar_index_for_plan(&index, &mut plan);
        }
        Source::TarArchive { path, codec: None } => {
            let target = crate::inspect::Target::parse(&path.to_string_lossy());
            crate::inspect::with_target_device(&target, |src_dev| {
                let mut src_fs = crate::inspect::AnyFs::open(src_dev)?;
                build_ext_plan_inner(src_dev, &mut src_fs, &mut plan)
            })?;
        }
        Source::Image(target) => {
            crate::inspect::with_target_device(target, |src_dev| {
                let mut src_fs = crate::inspect::AnyFs::open(src_dev)?;
                build_ext_plan_inner(src_dev, &mut src_fs, &mut plan)
            })?;
        }
        Source::Layered(layers) => {
            // Plan from the in-memory model — no tempfile, RAM bounded
            // by tree metadata. `analysis(block_size).plan` walks the
            // model's nodes once and accumulates counts/byte totals
            // identical to what scanning a flattened tar would produce.
            let model = crate::merge::MergeModel::build(layers)?;
            plan = model.analysis(block_size).plan;
            plan.kind = kind;
        }
    }
    Ok(plan)
}

/// Populate any [`crate::fs::Filesystem`] from `source`, dispatching
/// through the trait for every entry create. Works for HFS+, NTFS,
/// F2FS, SquashFS, XFS — whatever implements the trait — though
/// ext/FAT32 callers should prefer [`populate_ext_from_source`] /
/// [`populate_fat32_from_source`] which preserve xattrs and use the
/// per-FS fast paths.
///
/// Tar-archive and existing-image sources route through an internal
/// helper that opens the source via
/// [`crate::inspect::AnyFs`] and replays entries through trait
/// methods.
pub fn populate_fs_from_source<F: crate::fs::Filesystem>(
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut F,
    source: &Source,
) -> Result<()> {
    populate_fs_from_source_dyn(dst_dev, dst, source)
}

/// Trait-object form of [`populate_fs_from_source`]. Used by code
/// paths (e.g. [`crate::inspect::AnyFs`] dispatch helpers) that have
/// a `&mut dyn Filesystem` rather than a known concrete type.
pub fn populate_fs_from_source_dyn(
    dst_dev: &mut dyn crate::block::BlockDevice,
    dst: &mut dyn crate::fs::Filesystem,
    source: &Source,
) -> Result<()> {
    // Lossy: a destination that can't represent a symlink / device /
    // xattr drops it with a warning rather than failing the repack
    // (matching the old generic path, which ignored device-create
    // errors). Entries the destination *can* store are still created.
    let mut sink = FsSink::new(dst, dst_dev).lossy();
    walk_source_into_sink(source, &mut sink)
}

/// Convert host `Metadata` into a public [`crate::fs::FileMeta`].
fn host_meta_to_fs(meta: &std::fs::Metadata) -> crate::fs::FileMeta {
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    #[cfg(unix)]
    {
        crate::fs::FileMeta {
            mode: (meta.mode() & 0o7777) as u16,
            uid: meta.uid(),
            gid: meta.gid(),
            mtime: meta.mtime() as u32,
            atime: meta.atime() as u32,
            ctime: meta.ctime() as u32,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = meta;
        crate::fs::FileMeta::default()
    }
}

/// Compute the minimum FAT32 byte capacity needed to fit `source`.
/// Bumps to the FAT32 cluster-count minimum + rounds up to a 512-byte
/// sector boundary.
pub fn fat32_min_bytes_for_source(source: &Source) -> Result<u64> {
    let bytes = match source {
        Source::HostDir(p) => sum_host_dir_bytes(p)?,
        Source::TarArchive {
            path,
            codec: Some(algo),
        } => {
            let spec = path.to_string_lossy().into_owned();
            let index = build_tar_stream_index(&spec, *algo)?;
            let (sz, _, _, _, _, _) = size_from_tar_index(&index, "fat32")?;
            return Ok(sz);
        }
        Source::TarArchive { path, codec: None } => {
            let target = crate::inspect::Target::parse(&path.to_string_lossy());
            let mut sum = 0u64;
            crate::inspect::with_target_device(&target, |src_dev| {
                let mut src_fs = crate::inspect::AnyFs::open(src_dev)?;
                sum = sum_source_file_bytes(src_dev, &mut src_fs)?;
                Ok(())
            })?;
            sum
        }
        Source::Image(target) => {
            let mut sum = 0u64;
            crate::inspect::with_target_device(target, |src_dev| {
                let mut src_fs = crate::inspect::AnyFs::open(src_dev)?;
                sum = sum_source_file_bytes(src_dev, &mut src_fs)?;
                Ok(())
            })?;
            sum
        }
        Source::Layered(layers) => {
            // Sum from the in-memory model — no tempfile, no body reads.
            let model = crate::merge::MergeModel::build(layers)?;
            // block_size doesn't affect the byte total; pass a sentinel.
            model.analysis(1024).total_file_bytes
        }
    };
    let needed = bytes
        .saturating_mul(2)
        .max(crate::fs::fat::MIN_FAT32_CLUSTERS as u64 * 1024);
    Ok(needed.div_ceil(512) * 512)
}

fn sum_host_dir_bytes(root: &Path) -> Result<u64> {
    let mut total = 0u64;
    let mut stack: Vec<PathBuf> = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            if meta.is_dir() {
                stack.push(entry.path());
            } else if meta.is_file() {
                total = total.saturating_add(meta.len());
            }
        }
    }
    Ok(total)
}

/// Drive a BuildPlan from any source filesystem through the
/// [`crate::fs::Filesystem`] trait — no per-FS arms. Walks via
/// [`crate::fs::Filesystem::list`], using `DirEntry::size` for files
/// and [`crate::fs::Filesystem::read_symlink`] for each symlink's
/// target length (so the long-symlink branch fires only when needed).
fn build_ext_plan_inner(
    src_dev: &mut dyn crate::block::BlockDevice,
    src_fs: &mut crate::inspect::AnyFs,
    plan: &mut crate::fs::ext::BuildPlan,
) -> Result<()> {
    build_ext_plan_through_trait(src_dev, src_fs, plan)
}

/// Public counterpart of `build_ext_plan_inner` for the binary
/// crate's `build_ext_plan`. Walks the source through the
/// [`crate::fs::Filesystem`] trait, no AnyFs match.
pub fn build_ext_plan_through_trait(
    src_dev: &mut dyn crate::block::BlockDevice,
    src_fs: &mut crate::inspect::AnyFs,
    plan: &mut crate::fs::ext::BuildPlan,
) -> Result<()> {
    src_fs.as_filesystem_dyn(|fs| scan_into_build_plan(src_dev, fs, plan))
}

/// Filesystem-trait-driven recursive scan that fills a BuildPlan.
/// Same shape as the old per-FS walkers; symlink target length is
/// resolved via `Filesystem::read_symlink` (FSes that don't carry
/// symlinks return `Unsupported`, which gets degraded to "assume
/// long symlink — one block").
pub(crate) fn scan_into_build_plan(
    dev: &mut dyn crate::block::BlockDevice,
    fs: &mut dyn crate::fs::Filesystem,
    plan: &mut crate::fs::ext::BuildPlan,
) -> Result<()> {
    use crate::fs::EntryKind;
    let mut stack: Vec<std::path::PathBuf> = vec![std::path::PathBuf::from("/")];
    while let Some(dir) = stack.pop() {
        let entries = fs.list(dev, &dir)?;
        for e in entries {
            if e.name == "." || e.name == ".." || e.name == "lost+found" {
                continue;
            }
            let child = dir.join(&e.name);
            match e.kind {
                EntryKind::Dir => {
                    plan.add_dir();
                    stack.push(child);
                }
                EntryKind::Regular => plan.add_file(e.size),
                EntryKind::Symlink => {
                    let len = fs
                        .read_symlink(dev, &child)
                        .map(|t| t.as_os_str().len())
                        .unwrap_or(usize::MAX);
                    plan.add_symlink(len);
                }
                EntryKind::Char | EntryKind::Block | EntryKind::Fifo | EntryKind::Socket => {
                    plan.add_device()
                }
                EntryKind::Unknown => {}
            }
        }
    }
    Ok(())
}

/// TarStreamIndex variant of [`walk_tar_for_plan`] — adds one entry
/// of each kind to the build plan for every record in the index.
fn walk_tar_index_for_plan(
    index: &crate::fs::tar::TarStreamIndex,
    plan: &mut crate::fs::ext::BuildPlan,
) {
    use crate::fs::tar::EntryKind as TarKind;
    for ix in index.entries() {
        match ix.entry.kind {
            TarKind::Regular | TarKind::HardLink => plan.add_file(ix.entry.size),
            TarKind::Dir => plan.add_dir(),
            TarKind::Symlink => plan.add_symlink(
                ix.entry
                    .link_target
                    .as_deref()
                    .map(|s| s.len())
                    .unwrap_or(0),
            ),
            TarKind::CharDev | TarKind::BlockDev | TarKind::Fifo => plan.add_device(),
        }
    }
}

// ----------------------------------------------------------------------
// Internal helpers (moved verbatim from src/bin/fstool/main.rs)
// ----------------------------------------------------------------------

/// Repack-source error for the four read-only FSes (xfs/exfat/hfs+/apfs)
/// — they're inspectable via ls/cat/info but not yet wired into the
/// FS-to-FS copy walkers.
/// If `path` looks like a compressed tar (`.tar.gz`, `.tar.zst`,
/// `.tar.xz`, `.tgz`, `.txz`, `.tar.lz4`, `.tar.lzma`, `.tar.lzo`),
/// return the codec to use; otherwise `None`. Used by repack to pick a
/// streaming compressor for the output file.
pub(crate) fn tar_output_codec(path: &std::path::Path) -> Option<crate::compression::Algo> {
    let s = path.to_string_lossy().to_ascii_lowercase();
    if s.ends_with(".tgz") {
        return Some(crate::compression::Algo::Gzip);
    }
    if s.ends_with(".txz") {
        return Some(crate::compression::Algo::Xz);
    }
    if !s.contains(".tar.") {
        // Bare `.gz` / `.zst` etc. without a `.tar.` prefix isn't tar.
        return None;
    }
    crate::compression::Algo::from_extension(path)
}

/// `Some(algo)` when `path` points at a compressed tar archive that
/// should be stream-walked rather than decompressed-to-tempfile.
/// `None` for plain `.tar` (the regular BlockDevice path handles it
/// fine) and for non-tar files.
pub(crate) fn tar_input_codec(path: &str) -> Option<crate::compression::Algo> {
    // Strip any `:N` partition selector — tar archives don't have
    // partitions, but the parsing helper allows the form.
    let p = std::path::Path::new(path.split(':').next().unwrap_or(path));
    tar_output_codec(p)
}

/// Single-pass walk that builds a [`TarStreamIndex`] for a compressed
/// tar source. Bodies are NOT consumed: the underlying reader skips
/// past each body's bytes during `next_entry`, so the only buffered
/// data is the per-entry metadata.
pub(crate) fn build_tar_stream_index(
    src: &str,
    algo: crate::compression::Algo,
) -> crate::Result<crate::fs::tar::TarStreamIndex> {
    let reader = open_tar_stream_reader(src, Some(algo))?;
    crate::fs::tar::TarStreamIndex::build_from(reader)
}

/// Aggregate the size-relevant counters from a built [`TarStreamIndex`]
/// and return `(size_estimate, files, dirs, symlinks, devices, bytes)`.
/// `target_lower` tunes the size estimate per destination FS.
pub(crate) fn size_from_tar_index(
    index: &crate::fs::tar::TarStreamIndex,
    target_lower: &str,
) -> crate::Result<(u64, u64, u64, u64, u64, u64)> {
    use crate::fs::tar::EntryKind as TarKind;
    let mut files = 0u64;
    let mut dirs = 0u64;
    let mut symlinks = 0u64;
    let mut devices = 0u64;
    let mut bytes = 0u64;
    for ix in index.entries() {
        match ix.entry.kind {
            TarKind::Regular => {
                files += 1;
                bytes += ix.entry.size;
            }
            TarKind::HardLink => {
                files += 1;
                bytes += ix.entry.size;
            }
            TarKind::Dir => dirs += 1,
            TarKind::Symlink => symlinks += 1,
            TarKind::CharDev | TarKind::BlockDev | TarKind::Fifo => devices += 1,
        }
    }
    let size_estimate = match target_lower {
        "ext2" | "ext3" | "ext4" => {
            // Conservative ext sizing: file bytes + dir/inode overhead.
            // We give 4 KiB per inode + 1 MiB structural pad; min 8 MiB.
            let inodes = files + dirs + symlinks + devices + 16;
            let raw = bytes + inodes * 4096 + 1024 * 1024;
            raw.max(8 * 1024 * 1024).div_ceil(4096) * 4096
        }
        "fat32" | "vfat" => {
            // FAT32 needs at least MIN_FAT32_CLUSTERS clusters of 1 KiB
            // overhead per cluster. Double the byte total to leave room
            // for cluster fragmentation + FAT tables + dir entries.
            let needed = bytes
                .saturating_mul(2)
                .max(crate::fs::fat::MIN_FAT32_CLUSTERS as u64 * 1024);
            needed.div_ceil(512) * 512
        }
        _ => bytes + 16 * 1024 * 1024,
    };
    Ok((size_estimate, files, dirs, symlinks, devices, bytes))
}

/// Open a (possibly codec-wrapped) tar archive as a streaming reader.
pub(crate) fn open_tar_stream_reader(
    path: &str,
    algo: Option<crate::compression::Algo>,
) -> crate::Result<crate::fs::tar::TarStreamReader<Box<dyn std::io::Read>>> {
    // Strip a `:N` partition selector only when `N` is purely numeric —
    // a Windows path like `C:\foo\src.tar` must not be split at the
    // drive-letter colon. (Tar sources don't actually carry a partition
    // suffix; this is defensive parity with `Source::detect`.)
    let p = match path.rsplit_once(':') {
        Some((head, tail)) if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => {
            std::path::Path::new(head)
        }
        _ => std::path::Path::new(path),
    };
    let file = std::fs::File::open(p)?;
    let buffered: Box<dyn std::io::Read> =
        Box::new(std::io::BufReader::with_capacity(64 * 1024, file));
    let inner: Box<dyn std::io::Read> = match algo {
        Some(a) => crate::compression::make_reader(a, buffered)?,
        None => buffered,
    };
    Ok(crate::fs::tar::TarStreamReader::new(inner))
}

/// Open the tar source (optionally codec-wrapped) and build a
/// random-access index over it. Shared entry point for the
/// streaming-tar inspector commands.
pub fn open_tar_stream_index(
    image: &str,
    algo: Option<crate::compression::Algo>,
) -> crate::Result<crate::fs::tar::TarStreamIndex> {
    let reader = open_tar_stream_reader(image, algo)?;
    crate::fs::tar::TarStreamIndex::build_from(reader)
}

// ─── ext → ext (full metadata preservation) ─────────────────────────────

// ─── FAT32 → FAT32 ──────────────────────────────────────────────────────

// ─── ext → FAT32 (drops metadata FAT can't store) ───────────────────────

// ─── FAT32 → ext ────────────────────────────────────────────────────────

// ─── Tar → ext ──────────────────────────────────────────────────────────

// ─── Tar → FAT32 ────────────────────────────────────────────────────────

pub(crate) fn join_fs_path(parent: &str, leaf: &str) -> String {
    if parent.ends_with('/') {
        format!("{parent}{leaf}")
    } else {
        format!("{parent}/{leaf}")
    }
}

// ─── shrink sizing ───────────────────────────────────────────────────────

/// Sum the size of every regular file in the source filesystem — used
/// by FAT32 shrink sizing.
pub(crate) fn sum_source_file_bytes(
    src_dev: &mut dyn crate::block::BlockDevice,
    src_fs: &mut crate::inspect::AnyFs,
) -> crate::Result<u64> {
    src_fs.total_file_bytes(src_dev)
}

#[cfg(test)]
mod progress_tests {
    use super::truncate_left;

    #[test]
    fn truncate_left_fits_returns_full() {
        assert_eq!(truncate_left("short.txt", 10), "short.txt");
        // Exactly the budget: no marker needed.
        assert_eq!(truncate_left("9chars.tx", 9), "9chars.tx");
    }

    #[test]
    fn truncate_left_keeps_trailing_filename() {
        // 20-byte path, budget 10 → "…" + 9 trailing chars.
        let p = "/var/log/path/to/deep/file.log";
        let out = truncate_left(p, 10);
        assert_eq!(out, "…/file.log");
        assert!(out.chars().count() <= 10);
    }

    #[test]
    fn truncate_left_zero_budget_empty() {
        assert_eq!(truncate_left("x", 0), "");
    }

    #[test]
    fn truncate_left_unicode_path() {
        // Multi-byte chars in the path: must split at a char boundary.
        let p = "/données/important/документ.txt";
        let out = truncate_left(p, 15);
        // First char should be the ellipsis, rest the trailing slice.
        assert!(out.starts_with('…'));
        assert!(out.chars().count() <= 15);
        // The filename suffix must survive.
        assert!(out.contains(".txt"));
    }
}

#[cfg(test)]
mod ticker_layout_tests {
    use super::Progress;

    fn make() -> Progress {
        let mut p = Progress::auto();
        p.is_tty = true;
        p.files = 42;
        p.last_path = "/very/deep/nested/dir/some-long-filename.bin".into();
        p
    }

    #[test]
    fn ticker_no_cols_keeps_full_path() {
        let p = make();
        let line = p.status_line(None);
        assert!(line.contains("/very/deep/nested/dir/some-long-filename.bin"));
    }

    #[test]
    fn ticker_narrow_pty_truncates_left() {
        let p = make();
        // 60-col terminal. "repack: " (8) + "42 files | " (11) + path budget = 60-1
        let line = p.status_line(Some(60));
        // Total visible (with "repack: " prefix and minus the 1-col cursor margin) must fit
        assert!(8 + line.chars().count() <= 60);
        // Left side trimmed → starts ".../some-long-filename.bin" suffix preserved.
        assert!(line.contains("filename.bin"));
        // Should contain the ellipsis marker introduced by truncate_left.
        assert!(line.contains('…'));
    }

    #[test]
    fn ticker_tight_pty_still_shows_basename_suffix() {
        let p = make();
        let line = p.status_line(Some(30));
        assert!(8 + line.chars().count() <= 30);
        // Even tight, the trailing characters survive (in particular `.bin`).
        assert!(line.contains(".bin"));
    }
}
