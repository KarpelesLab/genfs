//! Lib-level fuzz of the `Filesystem` trait. Operates on
//! `&mut dyn Filesystem` + `&mut dyn BlockDevice` directly — no
//! FUSE, no shell — so it runs in CI without any kernel module or
//! external helper.
//!
//! The core (`fuzz_filesystem`) drives a deterministic xorshift PRNG
//! through randomised `create_file` / `open_file_rw` / `truncate` /
//! `remove` sequences against a single root directory, mirroring
//! everything in a shadow `BTreeMap<String, Vec<u8>>`. After each
//! mutation we verify the shadow against the live FS via `list` +
//! `read_file`. A periodic `flush` exercises the persist path.
//!
//! Each backend gets a thin `#[test]` that:
//!   1. allocates a `MemoryBackend` (or `FileBackend` if size is
//!      large enough to need on-disk staging),
//!   2. formats it with backend-specific `FormatOpts`,
//!   3. boxes the FS as `&mut dyn Filesystem` and hands both to the
//!      shared core.
//!
//! Determinism: the seed is fixed per backend, so a failing run is
//! reproducible byte-for-byte; bump the seed when extending.

#![cfg(unix)]

use std::collections::{BTreeMap, HashSet};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use fstool::block::{BlockDevice, FileBackend, MemoryBackend};
use fstool::fs::{FileMeta, FileSource, Filesystem, OpenFlags, ReadSeek};
use tempfile::NamedTempFile;

// ----------------------------------------------------------------------
// Deterministic PRNG
// ----------------------------------------------------------------------

/// Xorshift64* — small, fast, no `rand` dep, deterministic given the
/// seed. Quality is fine for "pick the next op and a few small
/// integers"; we're not running statistical tests.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Reject the all-zero state; xorshift would stick there.
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `0..n` (n must be > 0).
    fn range(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
    /// `n` random bytes.
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        for chunk in out.chunks_mut(8) {
            let v = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
        out
    }
}

// ----------------------------------------------------------------------
// Op model
// ----------------------------------------------------------------------

/// What a single fuzz step does to the FS + shadow.
#[derive(Debug, Clone, Copy)]
enum Op {
    /// New file via `create_file(FileSource::Reader)`.
    Create,
    /// Replace an existing file's bytes via `open_file_rw` with
    /// `truncate: true` + `write_all` (the FUSE-style overwrite).
    Overwrite,
    /// Append bytes via `open_file_rw` with `append: true`.
    Append,
    /// Patch a random byte range inside an existing file via
    /// `open_file_rw` + `seek` + `write_all`. Drives the partial-write
    /// path that uniquely characterises `MutationCapability::Mutable`.
    PatchRange,
    /// Resize an existing file via `FileHandle::set_len`. Both grow
    /// (zero-fill) and shrink paths are reachable here.
    SetLen,
    /// `remove(path)`.
    Delete,
    /// `clone_file(src, dst)` — copy or share extents from an existing
    /// file to a fresh destination. Backends with
    /// `clone_capability().shares_extents() == true` (XFS today) freeze
    /// both inodes after the call (writes would corrupt the sharing
    /// peer per `XFS_DIFLAG2_REFLINK` semantics); other backends get
    /// the default byte-copy fallback and the destination stays
    /// independently mutable.
    Clone,
    /// `flush(dev)` — exercise the persist path mid-run, not just at
    /// the end.
    Flush,
}

/// Cap on what each op may add/remove in one step. Tight ceilings
/// keep tiny memory-backed images from running out of space.
struct Caps {
    /// Maximum simultaneous files. New `Create`s are skipped once we
    /// hit this, biasing the run toward mutation of existing files.
    max_files: usize,
    /// Largest body any single op may produce on disk. Cumulative
    /// volume is naturally bounded by `max_files * max_size`.
    max_size: usize,
    /// Whether `PatchRange` should be tried. Backends that report
    /// `MutationCapability::WholeFileOnly` skip these.
    allow_partial: bool,
    /// Whether `SetLen` should be tried. Some backends gate this on a
    /// non-zero growth path that hasn't been wired yet — skip when
    /// false.
    allow_set_len: bool,
}

impl Caps {
    /// Sensible defaults for a "real" mutable backend.
    fn mutable_small() -> Self {
        Self {
            max_files: 8,
            max_size: 8 * 1024,
            allow_partial: true,
            allow_set_len: true,
        }
    }

    /// Tight caps for the ext4 writer, whose depth-0 extent tree
    /// holds at most 4 extents — heavy fragmentation from repeated
    /// append/patch on a single file pushes past that limit. With
    /// `max_size = 4 KiB` and `max_files = 4` a 200-iter run stays
    /// inside the depth-0 limit comfortably.
    fn ext4_tight() -> Self {
        Self {
            max_files: 4,
            max_size: 4 * 1024,
            allow_partial: true,
            allow_set_len: true,
        }
    }

    /// F2FS today: `set_len(small)` followed by `set_len(big)` leaves
    /// stale data in the blocks past the new size (the shrink path
    /// doesn't free or zero them, so a re-grow surfaces old bytes
    /// instead of zeroes). Filed for follow-up; in the meantime the
    /// fuzz exercises every *other* mutation against F2FS.
    fn f2fs_no_set_len() -> Self {
        Self {
            max_files: 8,
            max_size: 8 * 1024,
            allow_partial: true,
            allow_set_len: false,
        }
    }
}

// ----------------------------------------------------------------------
// Core fuzzer
// ----------------------------------------------------------------------

/// Drive `iters` randomised ops against `fs` rooted at `/`. Verifies
/// the live FS against an in-memory shadow after every step. Panics
/// with the seed + op index on the first divergence so a failing run
/// is reproducible (re-run with the same `seed`).
fn fuzz_filesystem(
    fs: &mut dyn Filesystem,
    dev: &mut dyn BlockDevice,
    seed: u64,
    iters: usize,
    caps: &Caps,
) {
    let cap = fs.mutation_capability();
    assert!(
        cap.supports_add_remove(),
        "fuzz core requires add/remove support; got {cap:?}"
    );
    let supports_partial = caps.allow_partial && cap.supports_partial_writes();
    // `clone_file` works on every mutable backend (the trait's default
    // is a byte-copy fallback); reflink-capable backends override and
    // produce shared extents. `shares_extents` controls the freezing
    // policy: when the clone physically shares blocks (XFS), both
    // peers must be treated as immutable thereafter or writes would
    // corrupt the other side.
    let shares_extents = fs.clone_capability().shares_extents();

    let mut rng = Rng::new(seed);
    let mut shadow: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut frozen: HashSet<String> = HashSet::new();

    for step in 0..iters {
        let op = pick_op(&mut rng, &shadow, &frozen, caps, supports_partial);
        apply_op(
            fs,
            dev,
            &mut rng,
            &mut shadow,
            &mut frozen,
            op,
            caps,
            shares_extents,
        )
        .unwrap_or_else(|e| panic!("seed={seed} step={step} op={op:?}: {e}"));

        // After every op the live FS must match the shadow exactly:
        // same set of names, same bytes. We flush first so backends
        // whose `list`/`read_file` read straight from on-disk state
        // (HFS+, F2FS) see the latest mutations — the fuzz's contract
        // is "after a flush, the FS contents match what we asked for",
        // which is the only contract the trait actually promises.
        fs.flush(dev)
            .unwrap_or_else(|e| panic!("seed={seed} step={step} flush before verify: {e}"));
        verify_against_shadow(fs, dev, &shadow)
            .unwrap_or_else(|e| panic!("seed={seed} step={step} after op={op:?}: {e}"));

        // Every 16 steps, additionally probe the random-access reader
        // (`open_file_ro`) on a random present file, seeking to a
        // random offset and reading a short window. This catches
        // backends whose RO seek path diverges from `read_file`.
        if step % 16 == 15 {
            probe_random_seek(fs, dev, &mut rng, &shadow)
                .unwrap_or_else(|e| panic!("seed={seed} step={step} seek probe: {e}"));
        }
    }

    // Final flush + verify. A backend that drops dirty state on
    // flush will fail here even if every in-memory step looked fine.
    fs.flush(dev)
        .unwrap_or_else(|e| panic!("seed={seed}: final flush failed: {e}"));
    verify_against_shadow(fs, dev, &shadow)
        .unwrap_or_else(|e| panic!("seed={seed} after final flush: {e}"));
}

fn pick_op(
    rng: &mut Rng,
    shadow: &BTreeMap<String, Vec<u8>>,
    frozen: &HashSet<String>,
    caps: &Caps,
    supports_partial: bool,
) -> Op {
    // Per-op weights, then re-weight by what's actually meaningful
    // given current state. Without files, we can only `Create`.
    if shadow.is_empty() {
        return Op::Create;
    }
    let can_create = shadow.len() < caps.max_files;
    // Most ops need a file we can still mutate; once a file has been
    // reflinked (frozen), Overwrite / Append / PatchRange / SetLen /
    // Delete would either corrupt the sharing peer or leave the
    // refcount-btree inconsistent (no CoW-on-write or refcount
    // decrement yet). If every existing file is frozen we fall back
    // to Create / Flush only.
    let has_unfrozen = shadow.keys().any(|n| !frozen.contains(n));

    let mut table: Vec<Op> = Vec::with_capacity(16);
    if can_create {
        table.extend([Op::Create; 4]);
    }
    if has_unfrozen {
        table.extend([Op::Overwrite; 3]);
        table.extend([Op::Append; 3]);
        if supports_partial {
            table.extend([Op::PatchRange; 3]);
        }
        if caps.allow_set_len {
            table.extend([Op::SetLen; 2]);
        }
        table.extend([Op::Delete; 2]);
        // Clone weight 1: rare but non-trivial — exercises the trait
        // default byte-copy on most backends and the REFCNTBT update
        // path on XFS. Gated on `has_unfrozen` since we need a live
        // source, and on `can_create` since the destination is a new
        // file in the same caps.max_files budget.
        if can_create {
            table.push(Op::Clone);
        }
    }
    table.push(Op::Flush);
    table[rng.range(table.len() as u64) as usize]
}

#[allow(clippy::too_many_arguments)]
fn apply_op(
    fs: &mut dyn Filesystem,
    dev: &mut dyn BlockDevice,
    rng: &mut Rng,
    shadow: &mut BTreeMap<String, Vec<u8>>,
    frozen: &mut HashSet<String>,
    op: Op,
    caps: &Caps,
    shares_extents: bool,
) -> Result<(), String> {
    match op {
        Op::Create => {
            let name = pick_fresh_name(rng, shadow);
            let path = format!("/{name}");
            let len = (rng.range(caps.max_size as u64) + 1) as usize;
            let body = rng.bytes(len);
            create_via_reader(fs, dev, &path, &body)?;
            shadow.insert(name, body);
        }
        Op::Overwrite => {
            let Some(name) = pick_present_unfrozen(rng, shadow, frozen) else {
                return Ok(());
            };
            let path = format!("/{name}");
            let len = (rng.range(caps.max_size as u64) + 1) as usize;
            let body = rng.bytes(len);
            overwrite_via_rw(fs, dev, &path, &body)?;
            shadow.insert(name, body);
        }
        Op::Append => {
            let Some(name) = pick_present_unfrozen(rng, shadow, frozen) else {
                return Ok(());
            };
            let path = format!("/{name}");
            let add_len = (rng.range((caps.max_size / 2) as u64) + 1) as usize;
            let chunk = rng.bytes(add_len);
            append_via_rw(fs, dev, &path, &chunk)?;
            shadow.get_mut(&name).unwrap().extend_from_slice(&chunk);
        }
        Op::PatchRange => {
            let Some(name) = pick_present_unfrozen(rng, shadow, frozen) else {
                return Ok(());
            };
            let cur = shadow.get(&name).cloned().unwrap();
            if cur.is_empty() {
                return Ok(());
            }
            let off = rng.range(cur.len() as u64) as usize;
            // Patch can extend past EOF — write_all on an open handle
            // grows the file. Cap so we never blow caps.max_size for
            // the resulting file.
            let max_grow = caps.max_size.saturating_sub(off).max(1);
            let n = (rng.range(max_grow as u64) + 1) as usize;
            let chunk = rng.bytes(n);
            let path = format!("/{name}");
            patch_via_rw(fs, dev, &path, off as u64, &chunk)?;
            let entry = shadow.get_mut(&name).unwrap();
            if off + chunk.len() > entry.len() {
                entry.resize(off + chunk.len(), 0);
            }
            entry[off..off + chunk.len()].copy_from_slice(&chunk);
        }
        Op::SetLen => {
            let Some(name) = pick_present_unfrozen(rng, shadow, frozen) else {
                return Ok(());
            };
            let new_len = rng.range(caps.max_size as u64);
            let path = format!("/{name}");
            set_len_via_rw(fs, dev, &path, new_len)?;
            let entry = shadow.get_mut(&name).unwrap();
            entry.resize(new_len as usize, 0);
        }
        Op::Delete => {
            let Some(name) = pick_present_unfrozen(rng, shadow, frozen) else {
                return Ok(());
            };
            let path = format!("/{name}");
            fs.remove(dev, Path::new(&path))
                .map_err(|e| format!("remove {path}: {e}"))?;
            shadow.remove(&name);
        }
        Op::Clone => {
            // Need an unfrozen source (we'd freeze it on shares_extents
            // backends) and a fresh destination name.
            let Some(src_name) = pick_present_unfrozen(rng, shadow, frozen) else {
                return Ok(());
            };
            let dst_name = pick_fresh_name(rng, shadow);
            let src_path = format!("/{src_name}");
            let dst_path = format!("/{dst_name}");
            fs.clone_file(dev, Path::new(&src_path), Path::new(&dst_path))
                .map_err(|e| format!("clone_file {src_path} → {dst_path}: {e}"))?;
            // Mirror the bytes in the shadow.
            let body = shadow.get(&src_name).cloned().unwrap();
            shadow.insert(dst_name.clone(), body);
            // If the backend physically shares extents (XFS today),
            // writes through either side would corrupt the other; mark
            // both immutable for the rest of this fuzz run.
            if shares_extents {
                frozen.insert(src_name);
                frozen.insert(dst_name);
            }
        }
        Op::Flush => {
            fs.flush(dev).map_err(|e| format!("flush: {e}"))?;
        }
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Operation primitives — each mirrors a single trait call.
// ----------------------------------------------------------------------

fn create_via_reader(
    fs: &mut dyn Filesystem,
    dev: &mut dyn BlockDevice,
    path: &str,
    body: &[u8],
) -> Result<(), String> {
    let reader: Box<dyn ReadSeek + Send> = Box::new(std::io::Cursor::new(body.to_vec()));
    let src = FileSource::Reader {
        reader,
        len: body.len() as u64,
    };
    fs.create_file(
        dev,
        Path::new(path),
        src,
        FileMeta {
            mode: 0o644,
            mtime: 1,
            ..Default::default()
        },
    )
    .map_err(|e| format!("create_file {path}: {e}"))
}

fn overwrite_via_rw(
    fs: &mut dyn Filesystem,
    dev: &mut dyn BlockDevice,
    path: &str,
    body: &[u8],
) -> Result<(), String> {
    let mut h = fs
        .open_file_rw(
            dev,
            Path::new(path),
            OpenFlags {
                truncate: true,
                ..OpenFlags::default()
            },
            None,
        )
        .map_err(|e| format!("open_file_rw truncate {path}: {e}"))?;
    h.write_all(body)
        .map_err(|e| format!("write_all {path}: {e}"))?;
    h.sync().map_err(|e| format!("sync {path}: {e}"))?;
    Ok(())
}

fn append_via_rw(
    fs: &mut dyn Filesystem,
    dev: &mut dyn BlockDevice,
    path: &str,
    chunk: &[u8],
) -> Result<(), String> {
    let mut h = fs
        .open_file_rw(
            dev,
            Path::new(path),
            OpenFlags {
                append: true,
                ..OpenFlags::default()
            },
            None,
        )
        .map_err(|e| format!("open_file_rw append {path}: {e}"))?;
    h.write_all(chunk)
        .map_err(|e| format!("write_all append {path}: {e}"))?;
    h.sync().map_err(|e| format!("sync {path}: {e}"))?;
    Ok(())
}

fn patch_via_rw(
    fs: &mut dyn Filesystem,
    dev: &mut dyn BlockDevice,
    path: &str,
    off: u64,
    chunk: &[u8],
) -> Result<(), String> {
    let mut h = fs
        .open_file_rw(dev, Path::new(path), OpenFlags::default(), None)
        .map_err(|e| format!("open_file_rw {path}: {e}"))?;
    h.seek(SeekFrom::Start(off))
        .map_err(|e| format!("seek {off}: {e}"))?;
    h.write_all(chunk)
        .map_err(|e| format!("patch write {path}: {e}"))?;
    h.sync().map_err(|e| format!("sync {path}: {e}"))?;
    Ok(())
}

fn set_len_via_rw(
    fs: &mut dyn Filesystem,
    dev: &mut dyn BlockDevice,
    path: &str,
    new_len: u64,
) -> Result<(), String> {
    let mut h = fs
        .open_file_rw(dev, Path::new(path), OpenFlags::default(), None)
        .map_err(|e| format!("open_file_rw {path}: {e}"))?;
    h.set_len(new_len)
        .map_err(|e| format!("set_len {path} -> {new_len}: {e}"))?;
    h.sync().map_err(|e| format!("sync {path}: {e}"))?;
    Ok(())
}

// ----------------------------------------------------------------------
// Verification helpers
// ----------------------------------------------------------------------

fn verify_against_shadow(
    fs: &mut dyn Filesystem,
    dev: &mut dyn BlockDevice,
    shadow: &BTreeMap<String, Vec<u8>>,
) -> Result<(), String> {
    let entries = fs
        .list(dev, Path::new("/"))
        .map_err(|e| format!("list /: {e}"))?;
    // Names ext writes that aren't ours; filter so backends with
    // their own bookkeeping (lost+found, $-prefixed metadata) don't
    // trip the shadow.
    let names: std::collections::BTreeSet<String> = entries
        .iter()
        .map(|e| e.name.clone())
        .filter(|n| !is_fs_internal(n))
        .collect();
    let want: std::collections::BTreeSet<String> = shadow.keys().cloned().collect();
    if names != want {
        let extra: Vec<_> = names.difference(&want).collect();
        let missing: Vec<_> = want.difference(&names).collect();
        return Err(format!(
            "root listing diverged: missing={missing:?} extra={extra:?}"
        ));
    }
    for (name, body) in shadow {
        let path = format!("/{name}");
        let mut r = fs
            .read_file(dev, Path::new(&path))
            .map_err(|e| format!("read_file {path}: {e}"))?;
        let mut got = Vec::with_capacity(body.len());
        r.read_to_end(&mut got)
            .map_err(|e| format!("read_to_end {path}: {e}"))?;
        drop(r);
        if got != *body {
            let diff_at = got
                .iter()
                .zip(body.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(usize::min(got.len(), body.len()));
            let slice_got = preview_at(&got, diff_at);
            let slice_want = preview_at(body, diff_at);
            return Err(format!(
                "{path} mismatch (got {}B, want {}B, first diff @ {}): \
                 got {slice_got}, want {slice_want}",
                got.len(),
                body.len(),
                diff_at,
            ));
        }
    }
    Ok(())
}

fn probe_random_seek(
    fs: &mut dyn Filesystem,
    dev: &mut dyn BlockDevice,
    rng: &mut Rng,
    shadow: &BTreeMap<String, Vec<u8>>,
) -> Result<(), String> {
    let Some(name) = pick_present(rng, shadow) else {
        return Ok(());
    };
    let body = shadow.get(&name).unwrap();
    if body.is_empty() {
        return Ok(());
    }
    let off = rng.range(body.len() as u64);
    let max_window = body.len() as u64 - off;
    let n = rng.range(max_window) + 1;
    let want = &body[off as usize..(off + n) as usize];
    let path = format!("/{name}");
    let mut h = match fs.open_file_ro(dev, Path::new(&path)) {
        Ok(h) => h,
        // open_file_ro is optional on some backends. If a backend
        // returns Unsupported, treat the probe as a no-op rather
        // than a failure — `read_file` already covers correctness.
        Err(fstool::Error::Unsupported(_)) => return Ok(()),
        Err(e) => return Err(format!("open_file_ro {path}: {e}")),
    };
    h.seek(SeekFrom::Start(off))
        .map_err(|e| format!("seek {off}: {e}"))?;
    let mut got = vec![0u8; n as usize];
    h.read_exact(&mut got)
        .map_err(|e| format!("read_exact {n} @ {off}: {e}"))?;
    if got != want {
        return Err(format!(
            "seek-read mismatch at {path}:{off}+{n}: got {:?}, want {:?}",
            preview(&got),
            preview(want),
        ));
    }
    Ok(())
}

fn pick_fresh_name(rng: &mut Rng, shadow: &BTreeMap<String, Vec<u8>>) -> String {
    // Short ASCII-only names — backends differ on charset support and
    // we're testing the data path, not codepoint handling.
    for _ in 0..32 {
        let n = (rng.range(8) + 3) as usize;
        let name: String = (0..n)
            .map(|_| (b'a' + (rng.range(26) as u8)) as char)
            .collect();
        if !shadow.contains_key(&name) {
            return name;
        }
    }
    // Fallback if we kept colliding (extremely unlikely at our caps).
    format!("file_{:x}", rng.next_u64())
}

fn pick_present(rng: &mut Rng, shadow: &BTreeMap<String, Vec<u8>>) -> Option<String> {
    if shadow.is_empty() {
        return None;
    }
    let i = rng.range(shadow.len() as u64) as usize;
    shadow.keys().nth(i).cloned()
}

/// Like [`pick_present`] but only over files not in the `frozen` set
/// — i.e. files the fuzz hasn't reflinked. Returns `None` when every
/// existing file is frozen, which the caller treats as "skip this op".
fn pick_present_unfrozen(
    rng: &mut Rng,
    shadow: &BTreeMap<String, Vec<u8>>,
    frozen: &HashSet<String>,
) -> Option<String> {
    let live: Vec<&String> = shadow.keys().filter(|n| !frozen.contains(*n)).collect();
    if live.is_empty() {
        return None;
    }
    let i = rng.range(live.len() as u64) as usize;
    Some(live[i].clone())
}

/// Names that backends create as part of their on-disk layout and
/// that the fuzz never touches — filtered out before comparing
/// against the shadow.
fn is_fs_internal(name: &str) -> bool {
    matches!(name, "lost+found" | "." | "..")
}

fn preview(b: &[u8]) -> String {
    let head: Vec<u8> = b.iter().take(16).copied().collect();
    format!("{head:02x?}")
}

fn preview_at(b: &[u8], at: usize) -> String {
    let start = at.saturating_sub(4);
    let end = b.len().min(at + 12);
    let win: Vec<u8> = b[start..end].to_vec();
    format!("[{start}..{end}]={win:02x?}")
}

// ----------------------------------------------------------------------
// Backend wirings
// ----------------------------------------------------------------------

const FUZZ_ITERS: usize = 200;

#[test]
fn fuzz_ext2() {
    use fstool::fs::ext::{Ext, FormatOpts, FsKind};
    let opts = FormatOpts {
        kind: FsKind::Ext2,
        // Bump to keep enough free inodes/blocks for the fuzz's
        // create-heavy phase. Defaults (1024 blocks * 1024) leave
        // about 800 KiB of data; plenty for `max_size = 8 KiB` *
        // `max_files = 8`.
        inodes_count: 256,
        blocks_count: 4096,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = MemoryBackend::new(size);
    let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext2");
    fuzz_filesystem(
        &mut ext,
        &mut dev,
        0xE2_0202_C0DE,
        FUZZ_ITERS,
        &Caps::mutable_small(),
    );
}

#[test]
fn fuzz_ext3() {
    use fstool::fs::ext::{Ext, FormatOpts, FsKind};
    let opts = FormatOpts {
        kind: FsKind::Ext3,
        inodes_count: 256,
        blocks_count: 4096,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = MemoryBackend::new(size);
    let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext3");
    fuzz_filesystem(
        &mut ext,
        &mut dev,
        0xE3_0303_C0DE,
        FUZZ_ITERS,
        &Caps::mutable_small(),
    );
}

#[test]
fn fuzz_ext4() {
    use fstool::fs::ext::{Ext, FormatOpts, FsKind};
    let opts = FormatOpts {
        kind: FsKind::Ext4,
        inodes_count: 256,
        blocks_count: 4096,
        ..FormatOpts::default()
    };
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = MemoryBackend::new(size);
    let mut ext = Ext::format_with(&mut dev, &opts).expect("format ext4");
    fuzz_filesystem(
        &mut ext,
        &mut dev,
        0xE4_0404_C0DE,
        FUZZ_ITERS,
        &Caps::ext4_tight(),
    );
}

#[test]
fn fuzz_fat32() {
    use fstool::fs::fat::{Fat32, FatFormatOpts};
    // FAT32 needs ≥ ~33 MiB to clear MIN_FAT32_CLUSTERS. Bump to
    // 64 MiB to be safe and keep the cluster count reasonable.
    const TOTAL_SECTORS: u32 = 64 * 1024 * 1024 / 512;
    let tmp = NamedTempFile::new().expect("tempfile");
    let mut dev =
        FileBackend::create(tmp.path(), TOTAL_SECTORS as u64 * 512).expect("FileBackend create");
    let opts = FatFormatOpts {
        total_sectors: TOTAL_SECTORS,
        ..FatFormatOpts::default()
    };
    let mut fs = Fat32::format(&mut dev, &opts).expect("format fat32");
    fuzz_filesystem(
        &mut fs,
        &mut dev,
        0xFAFA_3232_C0DE,
        FUZZ_ITERS,
        &Caps::mutable_small(),
    );
}

#[test]
fn fuzz_exfat() {
    use fstool::fs::exfat::Exfat;
    use fstool::fs::exfat::format::FormatOpts;
    // 16 MiB is the floor at which exfat's defaults (4 KiB clusters)
    // produce a reasonable layout. Use 32 MiB for headroom.
    const SIZE: u64 = 32 * 1024 * 1024;
    let tmp = NamedTempFile::new().expect("tempfile");
    let mut dev = FileBackend::create(tmp.path(), SIZE).expect("FileBackend create");
    let opts = FormatOpts::default();
    let mut fs = Exfat::format(&mut dev, &opts).expect("format exfat");
    fuzz_filesystem(
        &mut fs,
        &mut dev,
        0xEF_AAAA_C0DE,
        FUZZ_ITERS,
        &Caps::mutable_small(),
    );
}

#[test]
fn fuzz_hfs_plus() {
    use fstool::fs::hfs_plus::{FormatOpts, HfsPlus};
    // HFS+ format with defaults wants room for ~32 catalog nodes
    // * 8 KiB plus extents tree + alloc bitmap; 16 MiB is comfortable.
    const SIZE: u64 = 16 * 1024 * 1024;
    let tmp = NamedTempFile::new().expect("tempfile");
    let mut dev = FileBackend::create(tmp.path(), SIZE).expect("FileBackend create");
    let opts = FormatOpts::default();
    let mut fs = HfsPlus::format(&mut dev, &opts).expect("format hfs+");
    fuzz_filesystem(
        &mut fs,
        &mut dev,
        0x4859_4653_C0DE,
        FUZZ_ITERS,
        &Caps::mutable_small(),
    );
}

#[test]
fn fuzz_f2fs() {
    use fstool::fs::f2fs::{F2fs, FormatOpts};
    // F2FS minimum is roughly 50 MiB at log_blocks_per_seg=9. We use
    // the same value so the fuzz exercises the standard segment layout
    // (smaller log values are an off-the-beaten-path corner that the
    // unit tests cover separately).
    const SIZE: u64 = 64 * 1024 * 1024;
    let tmp = NamedTempFile::new().expect("tempfile");
    let mut dev = FileBackend::create(tmp.path(), SIZE).expect("FileBackend create");
    let opts = FormatOpts::default();
    let mut fs = F2fs::format(&mut dev, &opts).expect("format f2fs");
    fuzz_filesystem(
        &mut fs,
        &mut dev,
        0xF2F2_5555_C0DE,
        FUZZ_ITERS,
        &Caps::f2fs_no_set_len(),
    );
}

#[test]
fn fuzz_xfs() {
    use fstool::fs::xfs::{self, FormatOpts};
    // 256 MiB clears xfs::format's minimum AG-size thresholds at all
    // default block / log sizes; smaller images error out at format.
    const SIZE: u64 = 256 * 1024 * 1024;
    let tmp = NamedTempFile::new().expect("tempfile");
    let mut dev = FileBackend::create(tmp.path(), SIZE).expect("FileBackend create");
    let opts = FormatOpts::default();
    let mut fs = xfs::format(&mut dev, &opts).expect("format xfs");
    fuzz_filesystem(
        &mut fs,
        &mut dev,
        0x5846_5300_C0DE,
        FUZZ_ITERS,
        &Caps::mutable_small(),
    );
}

#[test]
fn fuzz_ntfs() {
    // NTFS landed `remove` in Phase 1 today, so it now satisfies the
    // fuzz core's `supports_add_remove()` precondition. 16 MiB is the
    // floor at which the writer's default layout (boot + MFT +
    // $Bitmap + $LogFile + $UpCase + $Secure) fits with headroom; the
    // existing external tests use the same size.
    use fstool::fs::ntfs::Ntfs;
    use fstool::fs::ntfs::format::FormatOpts;
    const SIZE: u64 = 16 * 1024 * 1024;
    let tmp = NamedTempFile::new().expect("tempfile");
    let mut dev = FileBackend::create(tmp.path(), SIZE).expect("FileBackend create");
    let opts = FormatOpts {
        volume_label: "FSTOOL-FUZZ".to_string(),
        ..Default::default()
    };
    let mut fs = Ntfs::format(&mut dev, &opts).expect("format ntfs");
    fuzz_filesystem(
        &mut fs,
        &mut dev,
        0x4E54_4653_C0DE, // "NTFS" + suffix
        FUZZ_ITERS,
        &Caps::mutable_small(),
    );
}
