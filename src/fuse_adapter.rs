//! FUSE adapter — exposes any opened [`Filesystem`] image as a userspace
//! filesystem on a host mountpoint.
//!
//! Linux mounts via libfuse, macOS via macFUSE; both reached through
//! the [`fuser`] crate. Gated by the `fuse` Cargo feature so the
//! default build doesn't need a C-side FUSE library installed.
//!
//! ## Backend-agnostic
//!
//! The adapter takes a [`Box<dyn Filesystem>`](crate::fs::Filesystem) —
//! ext{2,3,4}, FAT32, SquashFS, ISO 9660, GRF, anything that implements
//! the trait. Write capability follows from
//! [`Filesystem::mutation_capability`] — backends that report
//! [`Streaming`](crate::fs::MutationCapability::Streaming) or
//! [`Immutable`](crate::fs::MutationCapability::Immutable) refuse every
//! mutation callback with `EROFS`.
//!
//! ## Inode mapping
//!
//! FUSE's `FUSE_ROOT_ID` is `1`, and the kernel addresses every
//! subsequent object by an opaque inode number we hand back from
//! `lookup`. Because not every backend has a stable inode space (FAT's
//! cluster ids aren't kept, tar has none at all), the adapter maintains
//! its own `ino ↔ path` map: root is `1`, every other path gets a
//! monotonically increasing id on first `lookup`. We don't recycle ids
//! — even after `unlink`, the same id stays out of circulation for the
//! lifetime of the mount.
//!
//! ## Concurrency
//!
//! [`fuser::mount2`] runs the filesystem on the calling thread; all
//! callbacks come in serially. That suits our single-threaded
//! filesystem handles — when concurrency is added we'll switch to
//! `spawn_mount2` and share state via `Arc<Mutex<…>>`.
//!
//! ## Flush model
//!
//! Writes go through the backend's own buffering (ext stages in
//! in-memory metadata, for example); the disk only sees them after
//! [`Filesystem::flush`]. We flush on every `fsync`, `fsyncdir`, and
//! `flush` callback, plus on `destroy` (unmount). Between those, the
//! FUSE kernel-side page cache may serve subsequent reads from RAM
//! anyway.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fuser::{
    BackgroundSession, FUSE_ROOT_ID, FileAttr, FileType, Filesystem as FuseFilesystem,
    KernelConfig, MountOption, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty,
    ReplyEntry, ReplyOpen, ReplyStatfs, ReplyWrite, ReplyXattr, Request, TimeOrNow,
};

use crate::block::BlockDevice;
use crate::fs::{
    DeviceKind, EntryKind, FileAttrs, FileMeta, Filesystem, MutationCapability, OpenFlags, SetAttrs,
};

const TTL: Duration = Duration::from_secs(1);

/// A FUSE-mountable view of any [`Filesystem`]. Wraps a `Box<dyn
/// Filesystem>` and its backing device, plus an `ino ↔ path` table
/// keyed by FUSE inode number.
///
/// Both inner boxes are `Send` so the adapter can be moved onto a
/// background thread by `fuser::spawn_mount2` (used by the integration
/// tests; the CLI uses the blocking `mount`). Every backend in-tree
/// satisfies that bound — none of them hold `Rc`, and the
/// `RefCell` cells inside `Squashfs` are `Send` because their
/// contents are.
pub struct FstoolFs {
    fs: Box<dyn Filesystem + Send>,
    dev: Box<dyn BlockDevice + Send>,
    fs_name: &'static str,
    /// FUSE ino → absolute path. `1` is always `/`.
    ino_to_path: HashMap<u64, PathBuf>,
    /// Reverse map for cheap re-lookups.
    path_to_ino: HashMap<PathBuf, u64>,
    /// Next ino to hand out. Starts at 2 (1 is reserved for root).
    next_ino: u64,
    /// Whether the underlying FS reports `Mutable` or
    /// `WholeFileOnly`. Captured at construction so per-callback
    /// dispatch is one bool check.
    writable: bool,
    /// True only for `Mutable` — i.e. partial in-place byte writes
    /// are allowed (otherwise `write` is `EROFS`).
    partial_writable: bool,
    /// Opt-in for the `allow_other` mount option, which lets users
    /// other than the mounter see the mount. Requires the system to
    /// have `user_allow_other` in `/etc/fuse.conf`; we leave it off
    /// by default so the integration tests work on stock setups.
    allow_other: bool,
}

impl FstoolFs {
    /// Wrap an already-opened filesystem and its backing device.
    ///
    /// `fs_name` is the string the kernel surfaces as the FS type in
    /// `mount(8)` output — pass the short backend name (`"ext"`,
    /// `"fat32"`, `"squashfs"`, etc.).
    pub fn new(
        fs: Box<dyn Filesystem + Send>,
        dev: Box<dyn BlockDevice + Send>,
        fs_name: &'static str,
    ) -> Self {
        let cap = fs.mutation_capability();
        let writable = cap.supports_add_remove();
        let partial_writable = cap.supports_partial_writes();
        let mut ino_to_path = HashMap::new();
        let mut path_to_ino = HashMap::new();
        ino_to_path.insert(FUSE_ROOT_ID, PathBuf::from("/"));
        path_to_ino.insert(PathBuf::from("/"), FUSE_ROOT_ID);
        Self {
            fs,
            dev,
            fs_name,
            ino_to_path,
            path_to_ino,
            next_ino: 2,
            writable,
            partial_writable,
            allow_other: false,
        }
    }

    /// Turn on the `allow_other` mount option so users other than the
    /// mounter can see the mount. Requires the system to have
    /// `user_allow_other` set in `/etc/fuse.conf`; on hosts without
    /// that, `mount` returns `EPERM`. The CLI's `fstool mount` opts
    /// in via this method; tests leave it off.
    pub fn allow_other(mut self, yes: bool) -> Self {
        self.allow_other = yes;
        self
    }

    /// Assemble the [`MountOption`] vector this adapter wants to
    /// pass to fuser. Shared between [`Self::mount`] and
    /// [`Self::spawn_mount`].
    fn mount_options(&self) -> Vec<MountOption> {
        let mut opts = vec![MountOption::FSName(self.fs_name.to_string())];
        // `AutoUnmount` makes fusermount tear the mount down if the
        // mounting process dies — convenient for the CLI but
        // *requires* `AllowOther` (or `AllowRoot`) per fuser's
        // contract, which in turn requires `user_allow_other` in
        // `/etc/fuse.conf`. We only enable the pair when the
        // operator opted in via `.allow_other(true)`; the test path
        // relies on `BackgroundSession::Drop` for unmount instead.
        if self.allow_other {
            opts.push(MountOption::AllowOther);
            opts.push(MountOption::AutoUnmount);
        }
        if !self.writable {
            opts.push(MountOption::RO);
        }
        opts
    }

    /// Mount under `mountpoint` and pump events on this thread until
    /// `umount` on the mountpoint or a fatal callback returns. Blocks
    /// indefinitely; use [`Self::spawn_mount`] for async-ish behaviour.
    pub fn mount(self, mountpoint: &Path) -> std::io::Result<()> {
        let opts = self.mount_options();
        fuser::mount2(self, mountpoint, &opts)
    }

    /// Mount on a background thread and return the session handle.
    /// Drop the returned [`BackgroundSession`] (or call its
    /// `.join()`) to unmount cleanly. The mount survives only as
    /// long as the session does.
    pub fn spawn_mount(self, mountpoint: &Path) -> std::io::Result<BackgroundSession> {
        let opts = self.mount_options();
        fuser::spawn_mount2(self, mountpoint, &opts)
    }

    /// Look up the path for a FUSE inode. Returns `None` for unknown
    /// inos — the adapter only mints inos on successful `lookup`, so
    /// the kernel handing back one we've never seen is a protocol
    /// error.
    fn path_for(&self, ino: u64) -> Option<PathBuf> {
        self.ino_to_path.get(&ino).cloned()
    }

    /// Mint or recall the inode number for a path. Two `lookup`s of
    /// the same path return the same id — that's a requirement of
    /// the FUSE protocol (the kernel uses the id as a stable handle).
    fn ino_for(&mut self, path: &Path) -> u64 {
        if let Some(&id) = self.path_to_ino.get(path) {
            return id;
        }
        let id = self.next_ino;
        self.next_ino += 1;
        self.ino_to_path.insert(id, path.to_path_buf());
        self.path_to_ino.insert(path.to_path_buf(), id);
        id
    }

    /// Drop the path↔ino mapping for `path`. Called after unlink /
    /// rmdir so subsequent `lookup`s of the same name get a fresh
    /// inode. The id itself stays out of circulation for the lifetime
    /// of the mount (FUSE doesn't permit recycling).
    fn forget_path(&mut self, path: &Path) {
        if let Some(id) = self.path_to_ino.remove(path) {
            self.ino_to_path.remove(&id);
        }
    }

    /// Build a `FileAttr` from a backend [`FileAttrs`] plus the FUSE
    /// inode we want to surface.
    fn make_attr(&self, fuse_ino: u64, a: &FileAttrs) -> FileAttr {
        let kind = entry_kind_to_file_type(a.kind);
        let blksize = self.fs_block_size_hint().unwrap_or(4096);
        FileAttr {
            ino: fuse_ino,
            size: a.size,
            blocks: a.blocks,
            atime: ts(a.atime),
            mtime: ts(a.mtime),
            ctime: ts(a.ctime),
            crtime: ts(a.mtime),
            kind,
            perm: a.mode & 0o7777,
            nlink: a.nlink,
            uid: a.uid,
            gid: a.gid,
            rdev: a.rdev,
            blksize,
            flags: 0,
        }
    }

    /// Best-effort block-size hint for `blksize` on the inode. We
    /// don't want to spam the dev with a `statfs` call on every
    /// `getattr`, so we cache nothing — backends can be cheap or
    /// not. For the time being we just return 4 KiB.
    fn fs_block_size_hint(&self) -> Option<u32> {
        Some(4096)
    }

    fn now_secs() -> u32 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0)
    }

    /// Join `parent_path / name` into an absolute path. Names from
    /// FUSE arrive as `OsStr`; we decode them as UTF-8-lossy since
    /// the underlying filesystem trait is path-typed.
    fn child_path(parent: &Path, name: &OsStr) -> PathBuf {
        let name_str = name.to_string_lossy();
        if parent == Path::new("/") {
            PathBuf::from(format!("/{name_str}"))
        } else {
            let mut p = parent.to_path_buf();
            p.push(name_str.as_ref());
            p
        }
    }
}

fn ts(secs: u32) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(secs as u64)
}

fn time_or_now_secs(t: TimeOrNow) -> u32 {
    match t {
        TimeOrNow::SpecificTime(st) => st
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0),
        TimeOrNow::Now => SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0),
    }
}

fn fs_err_to_errno(e: &crate::Error) -> i32 {
    match e {
        crate::Error::InvalidArgument(_) => libc::ENOENT,
        crate::Error::Unsupported(_) => libc::ENOSYS,
        crate::Error::Immutable { .. } => libc::EROFS,
        crate::Error::Io(_) => libc::EIO,
        _ => libc::EIO,
    }
}

fn entry_kind_to_file_type(k: EntryKind) -> FileType {
    match k {
        EntryKind::Regular => FileType::RegularFile,
        EntryKind::Dir => FileType::Directory,
        EntryKind::Symlink => FileType::Symlink,
        EntryKind::Block => FileType::BlockDevice,
        EntryKind::Char => FileType::CharDevice,
        EntryKind::Fifo => FileType::NamedPipe,
        EntryKind::Socket => FileType::Socket,
        EntryKind::Unknown => FileType::RegularFile,
    }
}

impl FuseFilesystem for FstoolFs {
    fn init(&mut self, _req: &Request<'_>, _config: &mut KernelConfig) -> Result<(), libc::c_int> {
        Ok(())
    }

    fn destroy(&mut self) {
        if self.writable
            && let Err(e) = self.fs.flush(self.dev.as_mut())
        {
            eprintln!("fstool fuse: flush on unmount failed: {e}");
        }
        let _ = self.dev.sync();
    }

    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let parent_path = match self.path_for(parent) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let child = Self::child_path(&parent_path, name);
        let attrs = match self.fs.getattr(self.dev.as_mut(), &child) {
            Ok(a) => a,
            Err(_) => return reply.error(libc::ENOENT),
        };
        let ino = self.ino_for(&child);
        let attr = self.make_attr(ino, &attrs);
        reply.entry(&TTL, &attr, 0);
    }

    fn getattr(&mut self, _req: &Request<'_>, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        let path = match self.path_for(ino) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        match self.fs.getattr(self.dev.as_mut(), &path) {
            Ok(attrs) => {
                let attr = self.make_attr(ino, &attrs);
                reply.attr(&TTL, &attr);
            }
            Err(_) => reply.error(libc::ENOENT),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn setattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        mode: Option<u32>,
        uid: Option<u32>,
        gid: Option<u32>,
        size: Option<u64>,
        atime: Option<TimeOrNow>,
        mtime: Option<TimeOrNow>,
        ctime: Option<SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        if !self.writable {
            return reply.error(libc::EROFS);
        }
        let path = match self.path_for(ino) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let attrs = SetAttrs {
            mode: mode.map(|m| (m & 0o7777) as u16),
            uid,
            gid,
            atime: atime.map(time_or_now_secs),
            mtime: mtime.map(time_or_now_secs),
            ctime: ctime.map(|t| {
                t.duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs() as u32)
                    .unwrap_or(0)
            }),
        };
        // set_attrs ignores Unsupported for the *whole* set when the
        // backend can't change any of these. Surface that as ENOSYS
        // only when the caller actually asked for something.
        let asked_attrs = attrs.mode.is_some()
            || attrs.uid.is_some()
            || attrs.gid.is_some()
            || attrs.atime.is_some()
            || attrs.mtime.is_some()
            || attrs.ctime.is_some();
        if asked_attrs && let Err(e) = self.fs.set_attrs(self.dev.as_mut(), &path, attrs) {
            return reply.error(fs_err_to_errno(&e));
        }
        if let Some(sz) = size
            && let Err(e) = self.fs.truncate(self.dev.as_mut(), &path, sz)
        {
            return reply.error(fs_err_to_errno(&e));
        }
        match self.fs.getattr(self.dev.as_mut(), &path) {
            Ok(a) => {
                let attr = self.make_attr(ino, &a);
                reply.attr(&TTL, &attr);
            }
            Err(_) => reply.error(libc::EIO),
        }
    }

    fn readlink(&mut self, _req: &Request<'_>, ino: u64, reply: ReplyData) {
        let path = match self.path_for(ino) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        match self.fs.read_symlink(self.dev.as_mut(), &path) {
            Ok(target) => reply.data(target.as_os_str().as_encoded_bytes()),
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn mknod(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        rdev: u32,
        reply: ReplyEntry,
    ) {
        if !self.writable {
            return reply.error(libc::EROFS);
        }
        let parent_path = match self.path_for(parent) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let child = Self::child_path(&parent_path, name);
        let mode_type = mode & 0o170000;
        let meta = FileMeta {
            mode: (mode & 0o7777) as u16,
            uid: req.uid(),
            gid: req.gid(),
            mtime: Self::now_secs(),
            atime: Self::now_secs(),
            ctime: Self::now_secs(),
        };
        // S_IFREG and the special-file modes go through different
        // create_* methods in the trait.
        let res = match mode_type {
            0o100000 /* S_IFREG */ => self.fs.create_file(
                self.dev.as_mut(),
                &child,
                crate::fs::FileSource::Zero(0),
                meta,
            ),
            0o020000 /* S_IFCHR */ => {
                let major = (rdev >> 8) & 0xfff;
                let minor = (rdev & 0xff) | ((rdev >> 12) & 0xfff00);
                self.fs.create_device(
                    self.dev.as_mut(),
                    &child,
                    DeviceKind::Char,
                    major,
                    minor,
                    meta,
                )
            }
            0o060000 /* S_IFBLK */ => {
                let major = (rdev >> 8) & 0xfff;
                let minor = (rdev & 0xff) | ((rdev >> 12) & 0xfff00);
                self.fs.create_device(
                    self.dev.as_mut(),
                    &child,
                    DeviceKind::Block,
                    major,
                    minor,
                    meta,
                )
            }
            0o010000 /* S_IFIFO */ => self.fs.create_device(
                self.dev.as_mut(),
                &child,
                DeviceKind::Fifo,
                0,
                0,
                meta,
            ),
            0o140000 /* S_IFSOCK */ => self.fs.create_device(
                self.dev.as_mut(),
                &child,
                DeviceKind::Socket,
                0,
                0,
                meta,
            ),
            _ => return reply.error(libc::ENOSYS),
        };
        match res {
            Ok(()) => self.reply_with_new_entry(&child, reply),
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn mkdir(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        if !self.writable {
            return reply.error(libc::EROFS);
        }
        let parent_path = match self.path_for(parent) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let child = Self::child_path(&parent_path, name);
        let meta = FileMeta {
            mode: (mode & 0o7777) as u16,
            uid: req.uid(),
            gid: req.gid(),
            mtime: Self::now_secs(),
            atime: Self::now_secs(),
            ctime: Self::now_secs(),
        };
        match self.fs.create_dir(self.dev.as_mut(), &child, meta) {
            Ok(()) => self.reply_with_new_entry(&child, reply),
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn unlink(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        if !self.writable {
            return reply.error(libc::EROFS);
        }
        let parent_path = match self.path_for(parent) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let child = Self::child_path(&parent_path, name);
        match self.fs.remove(self.dev.as_mut(), &child) {
            Ok(()) => {
                self.forget_path(&child);
                reply.ok();
            }
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        // unlink + rmdir share the trait-level `remove(path)`.
        self.unlink(_req, parent, name, reply)
    }

    fn symlink(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        link_name: &OsStr,
        target: &Path,
        reply: ReplyEntry,
    ) {
        if !self.writable {
            return reply.error(libc::EROFS);
        }
        let parent_path = match self.path_for(parent) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let child = Self::child_path(&parent_path, link_name);
        let meta = FileMeta {
            mode: 0o777,
            uid: req.uid(),
            gid: req.gid(),
            mtime: Self::now_secs(),
            atime: Self::now_secs(),
            ctime: Self::now_secs(),
        };
        match self
            .fs
            .create_symlink(self.dev.as_mut(), &child, target, meta)
        {
            Ok(()) => self.reply_with_new_entry(&child, reply),
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn rename(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        _flags: u32,
        reply: ReplyEmpty,
    ) {
        if !self.writable {
            return reply.error(libc::EROFS);
        }
        let parent_path = match self.path_for(parent) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let newparent_path = match self.path_for(newparent) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let old = Self::child_path(&parent_path, name);
        let new = Self::child_path(&newparent_path, newname);
        match self.fs.rename(self.dev.as_mut(), &old, &new) {
            Ok(()) => {
                // Keep the same FUSE inode pointing at the new path.
                if let Some(id) = self.path_to_ino.remove(&old) {
                    self.ino_to_path.insert(id, new.clone());
                    self.path_to_ino.insert(new, id);
                }
                reply.ok();
            }
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn link(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        newparent: u64,
        newname: &OsStr,
        reply: ReplyEntry,
    ) {
        if !self.writable {
            return reply.error(libc::EROFS);
        }
        let target = match self.path_for(ino) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let newparent_path = match self.path_for(newparent) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let new = Self::child_path(&newparent_path, newname);
        match self.fs.hardlink(self.dev.as_mut(), &target, &new) {
            Ok(()) => self.reply_with_new_entry(&new, reply),
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn open(&mut self, _req: &Request<'_>, _ino: u64, _flags: i32, reply: ReplyOpen) {
        reply.opened(0, 0);
    }

    fn read(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        let path = match self.path_for(ino) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let mut handle = match self.fs.open_file_ro(self.dev.as_mut(), &path) {
            Ok(h) => h,
            Err(e) => return reply.error(fs_err_to_errno(&e)),
        };
        if handle.seek(SeekFrom::Start(offset as u64)).is_err() {
            return reply.error(libc::EIO);
        }
        let mut buf = vec![0u8; size as usize];
        let n = match handle.read(&mut buf) {
            Ok(n) => n,
            Err(_) => return reply.error(libc::EIO),
        };
        buf.truncate(n);
        reply.data(&buf);
    }

    #[allow(clippy::too_many_arguments)]
    fn write(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyWrite,
    ) {
        if !self.partial_writable {
            return reply.error(libc::EROFS);
        }
        let path = match self.path_for(ino) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let res = (|| -> crate::Result<usize> {
            let mut h =
                self.fs
                    .open_file_rw(self.dev.as_mut(), &path, OpenFlags::default(), None)?;
            h.seek(SeekFrom::Start(offset as u64))
                .map_err(crate::Error::Io)?;
            h.write_all(data).map_err(crate::Error::Io)?;
            Ok(data.len())
        })();
        match res {
            Ok(n) => reply.written(n as u32),
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn flush(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _lock_owner: u64,
        reply: ReplyEmpty,
    ) {
        if !self.writable {
            return reply.ok();
        }
        match self.fs.flush(self.dev.as_mut()) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn fsync(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        if !self.writable {
            return reply.ok();
        }
        match self.fs.flush(self.dev.as_mut()) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn release(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _flags: i32,
        _lock_owner: Option<u64>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    fn opendir(&mut self, _req: &Request<'_>, _ino: u64, _flags: i32, reply: ReplyOpen) {
        reply.opened(0, 0);
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        let path = match self.path_for(ino) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let entries = match self.fs.list(self.dev.as_mut(), &path) {
            Ok(v) => v,
            Err(e) => return reply.error(fs_err_to_errno(&e)),
        };
        // Synthesize "." and ".." — most backends' `list` skips them.
        let parent_path = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| path.clone());
        let parent_ino = if path == Path::new("/") {
            FUSE_ROOT_ID
        } else {
            self.ino_for(&parent_path)
        };
        let mut all: Vec<(u64, FileType, String)> = Vec::with_capacity(entries.len() + 2);
        all.push((ino, FileType::Directory, ".".to_string()));
        all.push((parent_ino, FileType::Directory, "..".to_string()));
        for e in entries {
            if e.name == "." || e.name == ".." {
                continue;
            }
            let child = Self::child_path(&path, OsStr::new(&e.name));
            let child_ino = self.ino_for(&child);
            let kind = entry_kind_to_file_type(e.kind);
            all.push((child_ino, kind, e.name));
        }
        for (i, (child, kind, name)) in all.into_iter().enumerate().skip(offset as usize) {
            if reply.add(child, (i + 1) as i64, kind, name) {
                break;
            }
        }
        reply.ok();
    }

    fn releasedir(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _flags: i32,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    fn statfs(&mut self, _req: &Request<'_>, _ino: u64, reply: ReplyStatfs) {
        match self.fs.statfs(self.dev.as_mut()) {
            Ok(s) => reply.statfs(
                s.blocks,
                s.blocks_free,
                s.blocks_avail,
                s.inodes,
                s.inodes_free,
                s.block_size,
                s.name_max,
                s.block_size,
            ),
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn create(
        &mut self,
        req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        umask: u32,
        _flags: i32,
        reply: ReplyCreate,
    ) {
        if !self.writable {
            return reply.error(libc::EROFS);
        }
        let parent_path = match self.path_for(parent) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let child = Self::child_path(&parent_path, name);
        let meta = FileMeta {
            mode: (mode & 0o7777 & !(umask & 0o7777)) as u16,
            uid: req.uid(),
            gid: req.gid(),
            mtime: Self::now_secs(),
            atime: Self::now_secs(),
            ctime: Self::now_secs(),
        };
        let res = self.fs.create_file(
            self.dev.as_mut(),
            &child,
            crate::fs::FileSource::Zero(0),
            meta,
        );
        match res {
            Ok(()) => match self.fs.getattr(self.dev.as_mut(), &child) {
                Ok(attrs) => {
                    let id = self.ino_for(&child);
                    let attr = self.make_attr(id, &attrs);
                    reply.created(&TTL, &attr, 0, 0, 0);
                }
                Err(_) => reply.error(libc::EIO),
            },
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn access(&mut self, _req: &Request<'_>, _ino: u64, _mask: i32, reply: ReplyEmpty) {
        reply.ok();
    }

    fn getxattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        name: &OsStr,
        size: u32,
        reply: ReplyXattr,
    ) {
        let path = match self.path_for(ino) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let xattrs = match self.fs.list_xattrs(self.dev.as_mut(), &path) {
            Ok(x) => x,
            Err(e) => return reply.error(fs_err_to_errno(&e)),
        };
        let want = name.to_string_lossy();
        let value = xattrs.into_iter().find(|x| x.name == want).map(|x| x.value);
        match value {
            Some(v) => {
                if size == 0 {
                    reply.size(v.len() as u32);
                } else if (size as usize) < v.len() {
                    reply.error(libc::ERANGE);
                } else {
                    reply.data(&v);
                }
            }
            None => reply.error(libc::ENODATA),
        }
    }

    fn listxattr(&mut self, _req: &Request<'_>, ino: u64, size: u32, reply: ReplyXattr) {
        let path = match self.path_for(ino) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let xattrs = match self.fs.list_xattrs(self.dev.as_mut(), &path) {
            Ok(x) => x,
            Err(e) => return reply.error(fs_err_to_errno(&e)),
        };
        let mut payload = Vec::new();
        for x in &xattrs {
            payload.extend_from_slice(x.name.as_bytes());
            payload.push(0);
        }
        if size == 0 {
            reply.size(payload.len() as u32);
        } else if (size as usize) < payload.len() {
            reply.error(libc::ERANGE);
        } else {
            reply.data(&payload);
        }
    }

    fn setxattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        name: &OsStr,
        value: &[u8],
        _flags: i32,
        _position: u32,
        reply: ReplyEmpty,
    ) {
        if !self.writable {
            return reply.error(libc::EROFS);
        }
        let path = match self.path_for(ino) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let n = name.to_string_lossy();
        match self
            .fs
            .set_xattr(self.dev.as_mut(), &path, n.as_ref(), value)
        {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }

    fn removexattr(&mut self, _req: &Request<'_>, ino: u64, name: &OsStr, reply: ReplyEmpty) {
        if !self.writable {
            return reply.error(libc::EROFS);
        }
        let path = match self.path_for(ino) {
            Some(p) => p,
            None => return reply.error(libc::ENOENT),
        };
        let n = name.to_string_lossy();
        match self.fs.remove_xattr(self.dev.as_mut(), &path, n.as_ref()) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(fs_err_to_errno(&e)),
        }
    }
}

impl FstoolFs {
    /// Shared tail for create_file/mkdir/symlink/mknod replies: look
    /// up the new entry's attrs and surface them through the kernel
    /// `entry` reply.
    fn reply_with_new_entry(&mut self, path: &Path, reply: ReplyEntry) {
        match self.fs.getattr(self.dev.as_mut(), path) {
            Ok(attrs) => {
                let id = self.ino_for(path);
                let attr = self.make_attr(id, &attrs);
                reply.entry(&TTL, &attr, 0);
            }
            Err(_) => reply.error(libc::EIO),
        }
    }
}

/// Make sure `MutationCapability` is referenced even when only the
/// `EROFS` paths are exercised, so unused-import warnings don't fire
/// in --no-default-features-without-fuse builds. No-op at runtime.
#[allow(dead_code)]
fn _mutation_cap_touch(m: MutationCapability) -> bool {
    m.supports_add_remove()
}
