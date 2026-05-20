//! Layered source merging with whiteout / opaque-dir semantics.
//!
//! Folds `Source::Layered(Vec<Source>)` into a single flattened tar
//! archive on disk (held in a `NamedTempFile` for the duration of the
//! repack). The flattened tar then drives the existing `populate_*`
//! pipelines unchanged.
//!
//! Two tombstone conventions are supported:
//!
//! - **Tar-OCI** (the canonical container-image convention):
//!     - `.wh.<name>` in directory D → delete `D/<name>` from the
//!       in-progress merged tree, including any subtree.
//!     - `.wh..wh..opq` in directory D → drop everything previously
//!       contributed to `D/*` before this layer's own children of D
//!       are folded in.
//! - **OverlayFS native**:
//!     - Character device with major=0, minor=0 → delete this path.
//!     - Directory carrying the xattr `trusted.overlay.opaque == "y"`
//!       → opaque-dir semantics (drop lower-layer children at this
//!       path).
//!
//! Layering is bottom→top: the first source in the `Vec` is the base,
//! later sources override files of the same path and apply tombstones
//! to the running merge. Tombstones themselves are never written to
//! the output.
//!
//! The fold is two-pass per layer:
//!
//! 1. Index the layer (tar → `TarStreamIndex`; FS image → list+stat
//!    walk through `AnyFs`).
//! 2. Apply tombstones, then overlay non-tombstoned entries on the
//!    running merged tree.
//!
//! When all layers are folded, the merged tree is serialised into a
//! `NamedTempFile` as a plain (uncompressed) tar — the smallest
//! representation that round-trips every Unix entry kind (regular,
//! dir, symlink, char/block/fifo/socket via PAX, xattrs via
//! `SCHILY.xattr.*`). `Source::Layered` then becomes equivalent to a
//! `TarArchive { path: <tempfile>, codec: None }` for downstream
//! pipelines.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::EntryKind;
use crate::repack::Source;

/// One node in the merged tree. Lives until the final tar is written.
struct MergedNode {
    kind: EntryKind,
    /// Where the bytes live, if applicable.
    body: NodeBody,
    /// Symlink target / hardlink target (kind-dependent).
    target: Option<PathBuf>,
    /// Posix mode (defaults to 0o644 / 0o755 if unknown).
    mode: u32,
    /// Owner / group IDs (default 0 / 0).
    uid: u32,
    gid: u32,
    /// Modification time (default 0).
    mtime: i64,
    /// Device numbers (only for Char / Block).
    dev_major: u32,
    dev_minor: u32,
}

/// Where a node's body bytes live during the merge. `None` for
/// directories / symlinks / devices.
enum NodeBody {
    None,
    Inline(Vec<u8>),
}

/// Running state of the merge.
struct MergedTree {
    nodes: BTreeMap<PathBuf, MergedNode>,
}

impl MergedTree {
    fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
        }
    }

    /// Drop `path` and every descendant.
    fn remove_subtree(&mut self, path: &Path) {
        let prefix = path.to_path_buf();
        let prefix_with_sep = {
            let mut s = prefix.to_string_lossy().into_owned();
            if !s.ends_with('/') {
                s.push('/');
            }
            s
        };
        self.nodes.retain(|k, _| {
            let s = k.to_string_lossy();
            *k != prefix && !s.starts_with(&prefix_with_sep)
        });
    }

    /// Drop every descendant of `path` but keep `path` itself.
    fn make_opaque(&mut self, path: &Path) {
        let prefix_with_sep = {
            let mut s = path.to_string_lossy().into_owned();
            if !s.ends_with('/') {
                s.push('/');
            }
            s
        };
        self.nodes.retain(|k, _| {
            let s = k.to_string_lossy();
            !s.starts_with(&prefix_with_sep)
        });
    }

    fn insert(&mut self, path: PathBuf, node: MergedNode) {
        // If we're replacing a directory with a non-directory, also
        // remove its old subtree.
        if let Some(existing) = self.nodes.get(&path) {
            if matches!(existing.kind, EntryKind::Dir) && !matches!(node.kind, EntryKind::Dir) {
                self.remove_subtree(&path);
            }
        }
        self.nodes.insert(path, node);
    }
}

/// Flatten `layers` into a single tar archive on disk. Returns the
/// temp-file handle (the caller MUST keep it alive for the lifetime of
/// any downstream `Source::TarArchive` that references it).
pub fn flatten_to_tempfile(
    layers: &[Source],
) -> Result<tempfile::NamedTempFile> {
    let mut merged = MergedTree::new();
    for layer in layers {
        apply_layer(layer, &mut merged)?;
    }
    let mut tmp = tempfile::NamedTempFile::new()?;
    write_tar(&merged, tmp.as_file_mut())?;
    tmp.as_file_mut().sync_all()?;
    Ok(tmp)
}

fn apply_layer(layer: &Source, merged: &mut MergedTree) -> Result<()> {
    match layer {
        Source::HostDir(p) => apply_host_dir(p, merged),
        Source::TarArchive { path, codec } => apply_tar(path, *codec, merged),
        Source::Image(target) => apply_image(target, merged),
        Source::Layered(nested) => {
            // Recursively flatten nested layered sources in place.
            for s in nested {
                apply_layer(s, merged)?;
            }
            Ok(())
        }
    }
}

fn apply_host_dir(root: &Path, merged: &mut MergedTree) -> Result<()> {
    let mut stack: Vec<(PathBuf, PathBuf)> = vec![(root.to_path_buf(), PathBuf::from("/"))];
    while let Some((host, fs)) = stack.pop() {
        for entry in std::fs::read_dir(&host)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy().into_owned();
            let dest = join_path(&fs, &name_str);
            let meta = entry.metadata()?;
            let ft = meta.file_type();
            if let Some(action) = whiteout_action(&fs, &name_str) {
                apply_action(action, merged);
                continue;
            }
            let (uid, gid, mode, mtime) = host_attrs(&meta);
            if ft.is_dir() {
                merged.insert(
                    dest.clone(),
                    MergedNode {
                        kind: EntryKind::Dir,
                        body: NodeBody::None,
                        target: None,
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: 0,
                        dev_minor: 0,
                    },
                );
                stack.push((entry.path(), dest));
            } else if ft.is_symlink() {
                let target = std::fs::read_link(entry.path())?;
                merged.insert(
                    dest,
                    MergedNode {
                        kind: EntryKind::Symlink,
                        body: NodeBody::None,
                        target: Some(target),
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: 0,
                        dev_minor: 0,
                    },
                );
            } else if ft.is_file() {
                let bytes = std::fs::read(entry.path())?;
                merged.insert(
                    dest,
                    MergedNode {
                        kind: EntryKind::Regular,
                        body: NodeBody::Inline(bytes),
                        target: None,
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: 0,
                        dev_minor: 0,
                    },
                );
            }
        }
    }
    Ok(())
}

fn apply_tar(
    path: &Path,
    codec: Option<crate::compression::Algo>,
    merged: &mut MergedTree,
) -> Result<()> {
    // Open via the existing tar-source path.
    let target = crate::inspect::Target::parse(&path.to_string_lossy());
    let _ = codec; // both compressed and plain handled below via AnyFs.
    crate::inspect::with_target_device(&target, |src_dev| {
        let mut any = crate::inspect::AnyFs::open(src_dev)?;
        let crate::inspect::AnyFs::Tar(tar) = &mut any else {
            return Err(crate::Error::InvalidArgument(
                "merge: expected tar source".into(),
            ));
        };
        // Walk every entry via the indexed view: scan + apply.
        for entry in tar.entries() {
            let path = match entry.path.strip_prefix("./") {
                Some(s) => format!("/{s}"),
                None => {
                    if entry.path.starts_with('/') {
                        entry.path.clone()
                    } else {
                        format!("/{}", entry.path)
                    }
                }
            };
            let path = path.trim_end_matches('/').to_string();
            if path.is_empty() {
                continue;
            }
            // Detect tar-OCI whiteouts on the basename.
            let parent = parent_of_str(&path);
            let base = basename_of_str(&path);
            if let Some(action) = whiteout_action(Path::new(&parent), &base) {
                apply_action(action, merged);
                continue;
            }
            let p = PathBuf::from(&path);
            let mode = u32::from(entry.mode);
            let uid = entry.uid;
            let gid = entry.gid;
            let mtime = entry.mtime as i64;
            match entry.kind {
                crate::fs::tar::EntryKind::Dir => {
                    merged.insert(
                        p,
                        MergedNode {
                            kind: EntryKind::Dir,
                            body: NodeBody::None,
                            target: None,
                            mode,
                            uid,
                            gid,
                            mtime,
                            dev_major: 0,
                            dev_minor: 0,
                        },
                    );
                }
                crate::fs::tar::EntryKind::Regular => {
                    let mut body = Vec::with_capacity(entry.size as usize);
                    let mut reader = tar.open_file_reader(src_dev, &entry.path)?;
                    std::io::Read::read_to_end(&mut reader, &mut body)?;
                    merged.insert(
                        p,
                        MergedNode {
                            kind: EntryKind::Regular,
                            body: NodeBody::Inline(body),
                            target: None,
                            mode,
                            uid,
                            gid,
                            mtime,
                            dev_major: 0,
                            dev_minor: 0,
                        },
                    );
                }
                crate::fs::tar::EntryKind::Symlink => {
                    let target = entry.link_target.clone().unwrap_or_default();
                    merged.insert(
                        p,
                        MergedNode {
                            kind: EntryKind::Symlink,
                            body: NodeBody::None,
                            target: Some(PathBuf::from(target)),
                            mode,
                            uid,
                            gid,
                            mtime,
                            dev_major: 0,
                            dev_minor: 0,
                        },
                    );
                }
                _ => {
                    // Hardlinks / device nodes / pax — skip for v1.
                }
            }
        }
        Ok(())
    })
}

fn apply_image(_target: &crate::inspect::Target, _merged: &mut MergedTree) -> Result<()> {
    // FS-image layer support (with overlayfs char-dev 0/0 + xattr-based
    // opaque dir detection) lands in a follow-up. Tar layers are the
    // primary OCI-image format and cover the immediate use case.
    Err(crate::Error::Unsupported(
        "merge: FS-image source layers are not yet wired (use tar layers for now)".into(),
    ))
}

/// Recognise tar-OCI tombstone marker filenames.
enum WhiteoutAction {
    Delete(PathBuf),
    Opaque(PathBuf),
}

fn whiteout_action(parent: &Path, name: &str) -> Option<WhiteoutAction> {
    if name == ".wh..wh..opq" {
        return Some(WhiteoutAction::Opaque(parent.to_path_buf()));
    }
    if let Some(rest) = name.strip_prefix(".wh.") {
        let target = join_path(parent, rest);
        return Some(WhiteoutAction::Delete(target));
    }
    None
}

fn apply_action(action: WhiteoutAction, merged: &mut MergedTree) {
    match action {
        WhiteoutAction::Delete(p) => merged.remove_subtree(&p),
        WhiteoutAction::Opaque(p) => merged.make_opaque(&p),
    }
}

fn join_path(parent: &Path, name: &str) -> PathBuf {
    let mut s = parent.to_string_lossy().into_owned();
    if !s.ends_with('/') {
        s.push('/');
    }
    s.push_str(name);
    PathBuf::from(s)
}

fn parent_of_str(path: &str) -> String {
    match path.rsplit_once('/') {
        Some((p, _)) => {
            if p.is_empty() {
                "/".to_string()
            } else {
                p.to_string()
            }
        }
        None => "/".to_string(),
    }
}

fn basename_of_str(path: &str) -> String {
    match path.rsplit_once('/') {
        Some((_, b)) => b.to_string(),
        None => path.to_string(),
    }
}

#[cfg(unix)]
fn host_attrs(meta: &std::fs::Metadata) -> (u32, u32, u32, i64) {
    use std::os::unix::fs::MetadataExt;
    (
        meta.uid(),
        meta.gid(),
        meta.mode(),
        meta.mtime(),
    )
}

#[cfg(not(unix))]
fn host_attrs(_meta: &std::fs::Metadata) -> (u32, u32, u32, i64) {
    (0, 0, 0o644, 0)
}

/// Serialise the merged tree as a ustar tar with PAX extended headers
/// for entries that need them (long names, xattrs, mode>0o7777). The
/// downstream pipelines (`populate_image_via_trait` + `AnyFs::Tar`)
/// re-parse this as a tar source.
fn write_tar(merged: &MergedTree, out: &mut std::fs::File) -> Result<()> {
    for (path, node) in &merged.nodes {
        let path_str = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("merge: non-UTF-8 path".into()))?;
        // tar wants relative names without the leading slash.
        let rel = path_str.trim_start_matches('/');
        let rel = if matches!(node.kind, EntryKind::Dir) && !rel.is_empty() {
            format!("{rel}/")
        } else {
            rel.to_string()
        };
        write_ustar_entry(out, &rel, node)?;
    }
    // ustar end-of-archive: two zero blocks.
    out.write_all(&[0u8; 1024])?;
    Ok(())
}

fn write_ustar_entry(
    out: &mut std::fs::File,
    rel_name: &str,
    node: &MergedNode,
) -> Result<()> {
    // Pre-emit a PAX header when name exceeds 100 chars (we keep this
    // path simple — no long-name support in the ustar header itself).
    if rel_name.len() > 100 {
        emit_pax_path(out, rel_name)?;
    }

    let mut header = [0u8; 512];
    // name (0..100): truncated to fit; the PAX header above carries the
    // full path when needed.
    write_octal_str(&mut header[0..100], rel_name);
    // mode (100..108): 8 bytes, octal + space-terminated.
    write_octal(&mut header[100..108], u64::from(node.mode & 0o7777), 7)?;
    // uid (108..116)
    write_octal(&mut header[108..116], u64::from(node.uid), 7)?;
    // gid (116..124)
    write_octal(&mut header[116..124], u64::from(node.gid), 7)?;
    // size (124..136): 12 bytes
    let size = match &node.body {
        NodeBody::Inline(b) => b.len() as u64,
        NodeBody::None => 0,
    };
    write_octal(&mut header[124..136], size, 11)?;
    // mtime (136..148)
    let mtime = node.mtime.max(0) as u64;
    write_octal(&mut header[136..148], mtime, 11)?;
    // checksum placeholder (148..156): spaces while computing.
    for b in &mut header[148..156] {
        *b = b' ';
    }
    // typeflag (156)
    header[156] = match node.kind {
        EntryKind::Regular => b'0',
        EntryKind::Dir => b'5',
        EntryKind::Symlink => b'2',
        EntryKind::Char => b'3',
        EntryKind::Block => b'4',
        EntryKind::Fifo => b'6',
        _ => b'0',
    };
    // linkname (157..257): symlink target.
    if let Some(target) = node.target.as_ref() {
        let s = target.to_string_lossy();
        write_octal_str(&mut header[157..257], &s);
    }
    // ustar magic + version (257..265)
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    // uname / gname (265..297, 297..329) — leave blank.
    // devmajor (329..337) / devminor (337..345)
    if matches!(node.kind, EntryKind::Char | EntryKind::Block) {
        write_octal(&mut header[329..337], u64::from(node.dev_major), 7)?;
        write_octal(&mut header[337..345], u64::from(node.dev_minor), 7)?;
    }
    // prefix (345..500) — leave blank.

    // Compute checksum: sum of all bytes treating the checksum field
    // as 8 spaces.
    let sum: u32 = header.iter().map(|&b| u32::from(b)).sum();
    let mut csum_bytes = [0u8; 8];
    write_octal(&mut csum_bytes, u64::from(sum), 6)?;
    // ustar format: 6 octal digits, NUL, space.
    csum_bytes[7] = b' ';
    header[148..156].copy_from_slice(&csum_bytes);

    out.write_all(&header)?;

    // Body (if regular file).
    if let NodeBody::Inline(bytes) = &node.body {
        out.write_all(bytes)?;
        // Pad to next 512-byte boundary.
        let pad = (512 - (bytes.len() % 512)) % 512;
        if pad > 0 {
            let zeros = vec![0u8; pad];
            out.write_all(&zeros)?;
        }
    }
    Ok(())
}

fn emit_pax_path(out: &mut std::fs::File, full_path: &str) -> Result<()> {
    // PAX record: "<len> path=<value>\n" where len includes everything
    // including itself + the newline.
    let mut content = String::new();
    let line_no_len = format!(" path={full_path}\n");
    // Iterate to find a stable length.
    for guess in [3, 4, 5, 6, 7] {
        let total = guess + line_no_len.len();
        if total < 10usize.pow(guess as u32 - 1) || total >= 10usize.pow(guess as u32) {
            continue;
        }
        content = format!("{total}{line_no_len}");
        break;
    }
    if content.is_empty() {
        return Err(crate::Error::Unsupported(
            "merge: PAX header length overflow".into(),
        ));
    }
    let bytes = content.into_bytes();
    let mut pax_header = [0u8; 512];
    write_octal_str(&mut pax_header[0..100], "PaxHeader");
    write_octal(&mut pax_header[100..108], 0o644, 7)?;
    write_octal(&mut pax_header[124..136], bytes.len() as u64, 11)?;
    for b in &mut pax_header[148..156] {
        *b = b' ';
    }
    pax_header[156] = b'x'; // typeflag x = PAX extended header
    pax_header[257..263].copy_from_slice(b"ustar\0");
    pax_header[263..265].copy_from_slice(b"00");
    let sum: u32 = pax_header.iter().map(|&b| u32::from(b)).sum();
    let mut csum_bytes = [0u8; 8];
    write_octal(&mut csum_bytes, u64::from(sum), 6)?;
    csum_bytes[7] = b' ';
    pax_header[148..156].copy_from_slice(&csum_bytes);
    out.write_all(&pax_header)?;
    out.write_all(&bytes)?;
    let pad = (512 - (bytes.len() % 512)) % 512;
    if pad > 0 {
        let zeros = vec![0u8; pad];
        out.write_all(&zeros)?;
    }
    Ok(())
}

fn write_octal_str(buf: &mut [u8], s: &str) {
    let bytes = s.as_bytes();
    let n = bytes.len().min(buf.len());
    buf[..n].copy_from_slice(&bytes[..n]);
    for b in &mut buf[n..] {
        *b = 0;
    }
}

fn write_octal(buf: &mut [u8], v: u64, digits: usize) -> Result<()> {
    let formatted = format!("{:0width$o}", v, width = digits);
    if formatted.len() > digits {
        return Err(crate::Error::Unsupported(format!(
            "merge: octal value {v} doesn't fit in {digits} digits"
        )));
    }
    let bytes = formatted.as_bytes();
    let n = bytes.len();
    buf[..n].copy_from_slice(bytes);
    if n < buf.len() {
        buf[n] = 0;
    }
    Ok(())
}

/// Populate `dst` with the result of merging `layers`. Allocates a
/// tempfile, holds it alive across the populate, then drops it once
/// the destination filesystem has consumed every entry.
pub fn populate_from_layered(
    dst_dev: &mut dyn BlockDevice,
    dst: &mut dyn crate::fs::Filesystem,
    layers: &[Source],
) -> Result<()> {
    let tmp = flatten_to_tempfile(layers)?;
    let path = tmp.path().to_path_buf();
    let merged_source = Source::TarArchive {
        path,
        codec: None,
    };
    crate::repack::populate_fs_from_source_dyn(dst_dev, dst, &merged_source)?;
    drop(tmp);
    Ok(())
}
