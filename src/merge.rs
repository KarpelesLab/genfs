//! Layered source merging with whiteout / opaque-dir semantics.
//!
//! Folds `Source::Layered(Vec<Source>)` into an **in-memory metadata-only
//! model** (`MergeModel`) and drives the destination directly from it — no
//! temp file, no buffered file bodies. RAM stays bounded by metadata
//! regardless of how large the layered contents are.
//!
//! Two passes:
//!
//! 1. **Pass 1 — build the model.** Walk every layer forward (tar via
//!    [`crate::fs::tar::TarStreamIndex`] / host dirs via `read_dir`)
//!    recording each surviving entry's kind / mode / uid / gid / mtime /
//!    xattrs / symlink target / device numbers and a `BodyRef` pointing at
//!    where the bytes live (a host path, or a (layer, body_offset, size)
//!    tuple inside a tar). Apply whiteouts (`.wh.<name>` → delete subtree)
//!    and opaque-dir markers (`.wh..wh..opq` → drop lower-layer children
//!    at this dir). Synthesize missing ancestor directories at the end.
//!
//! 2. **Pass 2 — emit through a [`crate::repack::RepackSink`].** First every directory
//!    (sorted, so parents precede children); then, for each layer in order,
//!    read its files **forward, in the order they appear in that source**
//!    — tar layers stream through a `TarStreamReader`, matching the winner
//!    entry by body offset; host-layer files are opened directly (no
//!    ordering constraint). Finally symlinks and device nodes from the
//!    model. Each tar layer is decompressed at most twice in total (once
//!    for the index in pass 1, once for body streaming in pass 2).
//!
//! Two tombstone conventions are supported:
//!
//! - **Tar-OCI** (the canonical container-image convention):
//!     - `.wh.<name>` in directory D → delete `D/<name>` (including subtree).
//!     - `.wh..wh..opq` in directory D → drop everything previously
//!       contributed to `D/*`; this layer's own children of D survive.
//! - **OverlayFS native:** char device with major=0, minor=0 → delete;
//!   `trusted.overlay.opaque == "y"` xattr on a dir → opaque (planned).
//!
//! Layering is bottom→top: the first source is the base, later sources
//! override files of the same path and apply tombstones to the running
//! merge. Tombstone entries themselves never reach the destination.
//!
//! Hard links from tar and host-dir layers are preserved: the model
//! records the first occurrence's body and emits subsequent links via
//! `put_hardlink`, falling back to `materialise_copy` on destinations that
//! can't represent links (FAT/exFAT). Image-source layers remain
//! unsupported (use tar or host-dir layers).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::EntryKind;
use crate::repack::Source;

/// One node in the merged tree.
pub(crate) struct Node {
    pub(crate) kind: EntryKind,
    pub(crate) mode: u32,
    pub(crate) uid: u32,
    pub(crate) gid: u32,
    pub(crate) mtime: i64,
    pub(crate) dev_major: u32,
    pub(crate) dev_minor: u32,
    /// Symlink target (only for `EntryKind::Symlink`).
    pub(crate) target: Option<PathBuf>,
    /// Carried through to the destination sink. Tar xattrs are picked up
    /// here so they survive the merge.
    pub(crate) xattrs: Vec<(String, Vec<u8>)>,
    /// Where the file body lives, if applicable.
    pub(crate) body: BodyRef,
}

/// Locator for a regular file's bytes during pass 2.
pub(crate) enum BodyRef {
    /// No body (directories, symlinks, devices).
    None,
    /// Zero-length regular file — emit an empty body.
    Empty,
    /// Host filesystem path; opened directly during pass 2.
    Host(PathBuf),
    /// A tar entry inside `layers[layer]`. Matched in pass 2 by body offset
    /// to disambiguate duplicate paths within a layer (last write wins).
    Tar {
        layer: usize,
        body_offset: u64,
        size: u64,
    },
    /// Hard link to another file in the merged tree. Emitted in pass 2
    /// after every regular file has been written so the target exists.
    /// Falls back to `materialise_copy` when the destination FS can't
    /// represent links.
    HardLink(PathBuf),
}

/// Running state of the merge. Public surface is small: build, analyse,
/// walk into a sink. Construction is private to this module.
pub struct MergeModel {
    nodes: BTreeMap<PathBuf, Node>,
}

impl MergeModel {
    fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
        }
    }

    /// Build the merged model by folding `layers` bottom→top. No file body
    /// is ever read — only metadata. RAM stays bounded by tree size.
    pub fn build(layers: &[Source]) -> Result<Self> {
        let mut model = Self::new();
        let mut layer_idx = 0usize;
        for layer in layers {
            apply_layer(layer, &mut model, &mut layer_idx)?;
        }
        model.synthesize_parents();
        Ok(model)
    }

    /// Drop `path` and every descendant. Used by tombstone application.
    fn remove_subtree(&mut self, path: &Path) {
        let prefix = path.to_path_buf();
        let prefix_sep = {
            let mut s = prefix.to_string_lossy().into_owned();
            if !s.ends_with('/') {
                s.push('/');
            }
            s
        };
        self.nodes.retain(|k, _| {
            let s = k.to_string_lossy();
            *k != prefix && !s.starts_with(&prefix_sep)
        });
    }

    /// Drop every descendant of `path` but keep `path` itself.
    fn make_opaque(&mut self, path: &Path) {
        let prefix_sep = {
            let mut s = path.to_string_lossy().into_owned();
            if !s.ends_with('/') {
                s.push('/');
            }
            s
        };
        self.nodes.retain(|k, _| {
            let s = k.to_string_lossy();
            !s.starts_with(&prefix_sep)
        });
    }

    fn insert(&mut self, path: PathBuf, node: Node) {
        // Replacing a directory with a non-directory wipes the old subtree.
        if let Some(existing) = self.nodes.get(&path)
            && matches!(existing.kind, EntryKind::Dir)
            && !matches!(node.kind, EntryKind::Dir)
        {
            self.remove_subtree(&path);
        }
        self.nodes.insert(path, node);
    }

    /// Ensure every entry's ancestor directories exist in the model as
    /// `Dir` nodes; tar layers often omit intermediate directories.
    fn synthesize_parents(&mut self) {
        let keys: Vec<PathBuf> = self.nodes.keys().cloned().collect();
        for k in keys {
            let mut cur = k.parent().map(|p| p.to_path_buf());
            while let Some(p) = cur {
                if p.as_os_str().is_empty() {
                    break;
                }
                let stop = p == Path::new("/");
                if !self.nodes.contains_key(&p) {
                    self.nodes.insert(
                        p.clone(),
                        Node {
                            kind: EntryKind::Dir,
                            mode: 0o755,
                            uid: 0,
                            gid: 0,
                            mtime: 0,
                            dev_major: 0,
                            dev_minor: 0,
                            target: None,
                            xattrs: Vec::new(),
                            body: BodyRef::None,
                        },
                    );
                }
                if stop {
                    break;
                }
                cur = p.parent().map(|q| q.to_path_buf());
            }
        }
    }

    /// Fill an [`crate::analyze::Analysis`] straight from model metadata —
    /// no body reads, no tar decompression beyond pass-1 index building.
    pub fn analysis(&self, block_size: u32) -> crate::analyze::Analysis {
        use crate::analyze::Analysis;
        use crate::fs::ext::{BuildPlan, FsKind};
        let mut a = Analysis {
            files: 0,
            dirs: 0,
            symlinks: 0,
            devices: 0,
            hardlinks: 0,
            total_file_bytes: 0,
            plan: BuildPlan::new(block_size, FsKind::Ext4),
        };
        for (path, node) in &self.nodes {
            if path == &PathBuf::from("/") {
                continue;
            }
            match node.kind {
                EntryKind::Dir => {
                    a.dirs += 1;
                    a.plan.add_dir();
                }
                EntryKind::Regular => {
                    let size = match &node.body {
                        BodyRef::Tar { size, .. } => *size,
                        BodyRef::Host(p) => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
                        BodyRef::Empty => 0,
                        BodyRef::None => 0,
                        BodyRef::HardLink(_) => 0,
                    };
                    // Hardlinks share the target's inode in any FS that
                    // supports them (ext/xfs/ntfs/squashfs/iso/…), so they
                    // count toward hardlinks rather than fresh files; lossy
                    // sinks materialise them as copies, but the sizing path
                    // doesn't need to charge for that twice.
                    if matches!(node.body, BodyRef::HardLink(_)) {
                        a.hardlinks += 1;
                    } else {
                        a.files += 1;
                        a.plan.add_file(size);
                    }
                    a.total_file_bytes = a.total_file_bytes.saturating_add(size);
                }
                EntryKind::Symlink => {
                    a.symlinks += 1;
                    let len = node
                        .target
                        .as_ref()
                        .map(|t| t.to_string_lossy().len())
                        .unwrap_or(0);
                    a.plan.add_symlink(len);
                }
                EntryKind::Char | EntryKind::Block | EntryKind::Fifo | EntryKind::Socket => {
                    a.devices += 1;
                    a.plan.add_device();
                }
                _ => {}
            }
        }
        a
    }

    /// Drive `sink` from the model:
    ///   1. emit every Dir node (ascending — parents first),
    ///   2. for each tar layer, stream it forward and `put_file` the winner
    ///      regular files (matched by `body_offset`),
    ///   3. for each host-backed regular file, open the host path and emit,
    ///   4. emit symlinks and devices.
    pub fn walk_into_sink(
        &self,
        layers: &[Source],
        sink: &mut dyn crate::repack::RepackSink,
    ) -> Result<()> {
        use crate::fs::XattrPair;
        use crate::repack::RepackMeta;

        let to_meta = |n: &Node| RepackMeta {
            mode: (n.mode & 0o7777) as u16,
            uid: n.uid,
            gid: n.gid,
            mtime: n.mtime.max(0) as u32,
            atime: n.mtime.max(0) as u32,
            ctime: n.mtime.max(0) as u32,
        };
        let to_xattrs = |n: &Node| -> Vec<XattrPair> {
            n.xattrs
                .iter()
                .map(|(k, v)| XattrPair {
                    name: k.clone(),
                    value: v.clone(),
                })
                .collect()
        };
        let path_str = |p: &Path| -> Result<String> {
            p.to_str()
                .map(|s| s.to_string())
                .ok_or_else(|| crate::Error::InvalidArgument("merge: non-UTF-8 path".into()))
        };

        // (1) Directories first, in ascending path order so parents
        //     precede children. The model's BTreeMap iteration is already
        //     sorted; just filter to dirs.
        for (path, node) in &self.nodes {
            if matches!(node.kind, EntryKind::Dir) && path != &PathBuf::from("/") {
                let p = path_str(path)?;
                sink.put_dir(&p, to_meta(node), &to_xattrs(node))?;
                crate::repack::note(&p);
            }
        }

        // (2) Per-layer forward tar walk for regular-file bodies.
        for (idx, layer) in layers.iter().enumerate() {
            if let Source::TarArchive { path, codec } = layer {
                stream_tar_layer_winners(self, idx, path, *codec, sink, to_meta, &to_xattrs)?;
            }
        }

        // (3) Host-backed regular files (random access).
        for (path, node) in &self.nodes {
            if !matches!(node.kind, EntryKind::Regular) {
                continue;
            }
            let p_str = path_str(path)?;
            match &node.body {
                BodyRef::Host(host) => {
                    let mut f = std::fs::File::open(host)?;
                    let len = std::fs::metadata(host)?.len();
                    sink.put_file(&p_str, &mut f, len, to_meta(node), &to_xattrs(node))?;
                    crate::repack::note(&p_str);
                    crate::repack::note_bytes(len);
                }
                BodyRef::Empty => {
                    let mut empty: &[u8] = &[];
                    sink.put_file(&p_str, &mut empty, 0, to_meta(node), &to_xattrs(node))?;
                    crate::repack::note(&p_str);
                }
                BodyRef::Tar { .. } | BodyRef::HardLink(_) | BodyRef::None => {
                    // Tar winners emitted in (2); hardlinks deferred to
                    // (3.5) so their target exists; BodyRef::None on a
                    // Regular shouldn't happen but is a safe no-op.
                }
            }
        }

        // (3.5) Hard links — every regular-file body has been written by
        // now, so the target exists on the destination. `put_hardlink`
        // returns false on destinations that can't represent links
        // (FAT/exFAT); fall back to materialise_copy from the destination.
        for (path, node) in &self.nodes {
            let target = match &node.body {
                BodyRef::HardLink(t) => t,
                _ => continue,
            };
            // Resolve through hardlink chains so the sink sees a real file.
            let resolved = resolve_hardlink_target(&self.nodes, target, 8);
            let Some(resolved_path) = resolved else {
                eprintln!(
                    "merge: hardlink {} → {} skipped (target missing from merged tree)",
                    path.display(),
                    target.display(),
                );
                continue;
            };
            let resolved_str = path_str(&resolved_path)?;
            let p_str = path_str(path)?;
            let linked =
                sink.put_hardlink(&p_str, &resolved_str, to_meta(node), &to_xattrs(node))?;
            if !linked {
                // Destination FS has no hardlink concept — copy the body
                // back out of it under the new path. Re-read from the dest
                // is the only option here since the source body was
                // already streamed past.
                sink.materialise_copy(&p_str, &resolved_str, to_meta(node), &to_xattrs(node))?;
            }
        }

        // (4) Symlinks and device nodes.
        for (path, node) in &self.nodes {
            let p_str = path_str(path)?;
            match node.kind {
                EntryKind::Symlink => {
                    let t = node
                        .target
                        .as_ref()
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    sink.put_symlink(&p_str, &t, to_meta(node), &to_xattrs(node))?;
                    crate::repack::note(&p_str);
                }
                EntryKind::Char => {
                    sink.put_device(
                        &p_str,
                        crate::fs::DeviceKind::Char,
                        node.dev_major,
                        node.dev_minor,
                        to_meta(node),
                        &to_xattrs(node),
                    )?;
                }
                EntryKind::Block => {
                    sink.put_device(
                        &p_str,
                        crate::fs::DeviceKind::Block,
                        node.dev_major,
                        node.dev_minor,
                        to_meta(node),
                        &to_xattrs(node),
                    )?;
                }
                EntryKind::Fifo => {
                    sink.put_device(
                        &p_str,
                        crate::fs::DeviceKind::Fifo,
                        0,
                        0,
                        to_meta(node),
                        &to_xattrs(node),
                    )?;
                }
                EntryKind::Socket => {
                    sink.put_device(
                        &p_str,
                        crate::fs::DeviceKind::Socket,
                        0,
                        0,
                        to_meta(node),
                        &to_xattrs(node),
                    )?;
                }
                _ => {}
            }
            if matches!(
                node.kind,
                EntryKind::Symlink
                    | EntryKind::Char
                    | EntryKind::Block
                    | EntryKind::Fifo
                    | EntryKind::Socket
            ) {
                // Symlink already noted above; this covers the device kinds.
                if !matches!(node.kind, EntryKind::Symlink) {
                    crate::repack::note(&p_str);
                }
            }
        }

        Ok(())
    }
}

fn apply_layer(layer: &Source, model: &mut MergeModel, layer_idx: &mut usize) -> Result<()> {
    match layer {
        Source::HostDir(p) => {
            apply_host_dir(p, model)?;
            *layer_idx += 1;
            Ok(())
        }
        Source::TarArchive { path, codec } => {
            apply_tar_layer(path, *codec, *layer_idx, model)?;
            *layer_idx += 1;
            Ok(())
        }
        Source::Image(_) => Err(crate::Error::Unsupported(
            "merge: FS-image source layers are not yet wired (use tar layers or host dirs)".into(),
        )),
        Source::Layered(nested) => {
            for s in nested {
                apply_layer(s, model, layer_idx)?;
            }
            Ok(())
        }
    }
}

fn apply_host_dir(root: &Path, model: &mut MergeModel) -> Result<()> {
    // Track inodes seen with nlink > 1: the first occurrence emits a real
    // file (BodyRef::Host); subsequent ones become hardlinks pointing back
    // to the first path. Scoped per host-dir layer (inodes only collide
    // within one filesystem).
    #[cfg(unix)]
    let mut link_map: std::collections::HashMap<u64, PathBuf> = std::collections::HashMap::new();
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
                apply_action(action, model);
                continue;
            }
            let (uid, gid, mode, mtime) = host_attrs(&meta);
            if ft.is_dir() {
                model.insert(
                    dest.clone(),
                    Node {
                        kind: EntryKind::Dir,
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: 0,
                        dev_minor: 0,
                        target: None,
                        xattrs: Vec::new(),
                        body: BodyRef::None,
                    },
                );
                stack.push((entry.path(), dest));
            } else if ft.is_symlink() {
                let target = std::fs::read_link(entry.path())?;
                model.insert(
                    dest,
                    Node {
                        kind: EntryKind::Symlink,
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: 0,
                        dev_minor: 0,
                        target: Some(target),
                        xattrs: Vec::new(),
                        body: BodyRef::None,
                    },
                );
            } else if ft.is_file() {
                // On Unix, files with nlink > 1 sharing an inode are hard
                // links; record the first path and emit subsequent ones
                // as BodyRef::HardLink so the destination preserves the
                // shared inode (or copies, on FAT/exFAT).
                #[cfg(unix)]
                let body = {
                    use std::os::unix::fs::MetadataExt;
                    if meta.len() == 0 {
                        BodyRef::Empty
                    } else if meta.nlink() > 1 {
                        let ino = meta.ino();
                        match link_map.get(&ino) {
                            Some(first) => BodyRef::HardLink(first.clone()),
                            None => {
                                link_map.insert(ino, dest.clone());
                                BodyRef::Host(entry.path())
                            }
                        }
                    } else {
                        BodyRef::Host(entry.path())
                    }
                };
                #[cfg(not(unix))]
                let body = if meta.len() == 0 {
                    BodyRef::Empty
                } else {
                    BodyRef::Host(entry.path())
                };
                model.insert(
                    dest,
                    Node {
                        kind: EntryKind::Regular,
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: 0,
                        dev_minor: 0,
                        target: None,
                        xattrs: Vec::new(),
                        body,
                    },
                );
            }
        }
    }
    Ok(())
}

fn apply_tar_layer(
    path: &Path,
    codec: Option<crate::compression::Algo>,
    layer: usize,
    model: &mut MergeModel,
) -> Result<()> {
    use crate::fs::tar::EntryKind as TarKind;

    // Build the metadata index over the (decompressed) stream — no body
    // reads, but `body_offset` is recorded so pass 2 can match the winner
    // entry on a fresh forward walk.
    let spec = path.to_string_lossy().into_owned();
    let index = match codec {
        Some(algo) => crate::repack::open_tar_stream_index(&spec, Some(algo))?,
        None => crate::repack::open_tar_stream_index(&spec, None)?,
    };

    for ix in index.entries() {
        let e = &ix.entry;
        let canon = normalise_tar_path(&e.path);
        if canon.is_empty() {
            continue;
        }
        let parent = parent_of_str(&canon);
        let base = basename_of_str(&canon);
        if let Some(action) = whiteout_action(Path::new(&parent), &base) {
            apply_action(action, model);
            continue;
        }
        let p = PathBuf::from(&canon);
        let xattrs: Vec<(String, Vec<u8>)> = e
            .xattrs
            .iter()
            .map(|x| (x.name.clone(), x.value.clone()))
            .collect();
        let mode = u32::from(e.mode);
        let uid = e.uid;
        let gid = e.gid;
        let mtime = e.mtime as i64;

        match e.kind {
            TarKind::Dir => {
                model.insert(
                    p,
                    Node {
                        kind: EntryKind::Dir,
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: 0,
                        dev_minor: 0,
                        target: None,
                        xattrs,
                        body: BodyRef::None,
                    },
                );
            }
            TarKind::Regular => {
                let body = if e.size == 0 {
                    BodyRef::Empty
                } else {
                    BodyRef::Tar {
                        layer,
                        body_offset: ix.body_offset,
                        size: e.size,
                    }
                };
                model.insert(
                    p,
                    Node {
                        kind: EntryKind::Regular,
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: 0,
                        dev_minor: 0,
                        target: None,
                        xattrs,
                        body,
                    },
                );
            }
            TarKind::Symlink => {
                let target = e.link_target.clone().unwrap_or_default();
                model.insert(
                    p,
                    Node {
                        kind: EntryKind::Symlink,
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: 0,
                        dev_minor: 0,
                        target: Some(PathBuf::from(target)),
                        xattrs,
                        body: BodyRef::None,
                    },
                );
            }
            TarKind::CharDev | TarKind::BlockDev => {
                // OverlayFS deletes are encoded as character device 0/0.
                if matches!(e.kind, TarKind::CharDev) && e.device_major == 0 && e.device_minor == 0
                {
                    model.remove_subtree(&p);
                    continue;
                }
                let kind = if matches!(e.kind, TarKind::CharDev) {
                    EntryKind::Char
                } else {
                    EntryKind::Block
                };
                model.insert(
                    p,
                    Node {
                        kind,
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: e.device_major,
                        dev_minor: e.device_minor,
                        target: None,
                        xattrs,
                        body: BodyRef::None,
                    },
                );
            }
            TarKind::Fifo => {
                model.insert(
                    p,
                    Node {
                        kind: EntryKind::Fifo,
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: 0,
                        dev_minor: 0,
                        target: None,
                        xattrs,
                        body: BodyRef::None,
                    },
                );
            }
            TarKind::HardLink => {
                // `link_target` is the archive-relative path of the file
                // this entry links to. Normalise it to a canonical absolute
                // path so it matches the keys in `model.nodes`.
                let target = e.link_target.clone().unwrap_or_default();
                let target_path = normalise_tar_path(&target);
                if target_path.is_empty() {
                    continue;
                }
                model.insert(
                    p,
                    Node {
                        kind: EntryKind::Regular,
                        mode,
                        uid,
                        gid,
                        mtime,
                        dev_major: 0,
                        dev_minor: 0,
                        target: None,
                        xattrs,
                        body: BodyRef::HardLink(PathBuf::from(target_path)),
                    },
                );
            }
        }
    }
    Ok(())
}

/// Follow a hardlink chain in the model until reaching a real file body
/// or running out of hops. Returns the resolved path, or `None` when the
/// target is missing (e.g. whited out by an upper layer) or the chain
/// dead-ends in another hardlink within the hop budget.
fn resolve_hardlink_target(
    nodes: &BTreeMap<PathBuf, Node>,
    start: &Path,
    mut hops: usize,
) -> Option<PathBuf> {
    let mut cur = start.to_path_buf();
    while hops > 0 {
        let n = nodes.get(&cur)?;
        match &n.body {
            BodyRef::Host(_) | BodyRef::Tar { .. } | BodyRef::Empty => return Some(cur),
            BodyRef::HardLink(next) => {
                cur = next.clone();
                hops -= 1;
            }
            BodyRef::None => return None,
        }
    }
    None
}

fn normalise_tar_path(p: &str) -> String {
    let mut out = String::new();
    for seg in p.split('/').filter(|s| !s.is_empty() && *s != ".") {
        out.push('/');
        out.push_str(seg);
    }
    if out.is_empty() { "/".to_string() } else { out }
}

/// Pass-2 helper: stream tar layer `idx` from disk forward, emitting any
/// regular-file entries whose model winner is this (layer, body_offset).
fn stream_tar_layer_winners<F, G>(
    model: &MergeModel,
    idx: usize,
    path: &Path,
    codec: Option<crate::compression::Algo>,
    sink: &mut dyn crate::repack::RepackSink,
    to_meta: F,
    to_xattrs: &G,
) -> Result<()>
where
    F: Fn(&Node) -> crate::repack::RepackMeta + Copy,
    G: Fn(&Node) -> Vec<crate::fs::XattrPair>,
{
    use crate::fs::tar::EntryKind as TarKind;
    use crate::fs::tar::stream::TarStreamReader;

    let mut reader = crate::repack::open_tar_stream(path, codec)?;
    let mut tsr = TarStreamReader::new(&mut reader);
    while let Some(mut se) = tsr.next_entry()? {
        if !matches!(se.entry.kind, TarKind::Regular) {
            continue;
        }
        // `StreamEntry::body_offset` is captured by the reader when the
        // entry's header finishes decoding — the exact value pass 1
        // recorded via `TarStreamIndex`, so a winner matches byte-for-byte.
        let body_offset = se.body_offset;
        let canon = normalise_tar_path(&se.entry.path);
        if canon.is_empty() {
            continue;
        }
        let p = PathBuf::from(&canon);
        let node = match model.nodes.get(&p) {
            Some(n) => n,
            None => continue,
        };
        let winner = match &node.body {
            BodyRef::Tar {
                layer,
                body_offset: bo,
                ..
            } => *layer == idx && *bo == body_offset,
            _ => false,
        };
        if !winner {
            continue;
        }
        let size = se.entry.size;
        sink.put_file(&canon, &mut se, size, to_meta(node), &to_xattrs(node))?;
        crate::repack::note(&canon);
        crate::repack::note_bytes(size);
    }
    Ok(())
}

/// Populate `dst` with the result of merging `layers`. Builds the model in
/// memory then walks it into an `FsSink` — no temp file, RAM bounded by
/// tree metadata.
pub fn populate_from_layered(
    dst_dev: &mut dyn BlockDevice,
    dst: &mut dyn crate::fs::Filesystem,
    layers: &[Source],
) -> Result<()> {
    let model = MergeModel::build(layers)?;
    let mut sink = crate::repack::FsSink::new(dst, dst_dev).lossy();
    model.walk_into_sink(layers, &mut sink)
}

// ---------------------------------------------------------------- helpers

enum WhiteoutAction {
    Delete(PathBuf),
    Opaque(PathBuf),
}

fn whiteout_action(parent: &Path, name: &str) -> Option<WhiteoutAction> {
    if name == ".wh..wh..opq" {
        return Some(WhiteoutAction::Opaque(parent.to_path_buf()));
    }
    if let Some(rest) = name.strip_prefix(".wh.") {
        return Some(WhiteoutAction::Delete(join_path(parent, rest)));
    }
    None
}

fn apply_action(action: WhiteoutAction, model: &mut MergeModel) {
    match action {
        WhiteoutAction::Delete(p) => model.remove_subtree(&p),
        WhiteoutAction::Opaque(p) => model.make_opaque(&p),
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
    (meta.uid(), meta.gid(), meta.mode(), meta.mtime())
}

#[cfg(not(unix))]
fn host_attrs(_meta: &std::fs::Metadata) -> (u32, u32, u32, i64) {
    (0, 0, 0o644, 0)
}
