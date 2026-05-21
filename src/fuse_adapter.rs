//! FUSE adapter — exposes an opened ext{2,3,4} image as a userspace
//! filesystem on a host mountpoint.
//!
//! Linux mounts via libfuse, macOS via macFUSE; both reached through
//! the [`fuser`] crate. Gated by the `fuse` Cargo feature so the
//! default build doesn't need a C-side FUSE library installed.
//!
//! ## Inode mapping
//!
//! FUSE's `FUSE_ROOT_ID` is `1`; ext's root inode is `2`. We translate
//! at the kernel ↔ ext boundary: FUSE sees the root as ino 1, every
//! other inode is passed through unchanged.
//!
//! ## Concurrency
//!
//! [`fuser::mount2`] runs the filesystem on the calling thread; all
//! callbacks come in serially. That suits our current single-threaded
//! `Ext` handle — when Phase E adds real concurrency we'll switch to
//! `spawn_mount2` and share state via `Arc<Mutex<…>>`.
//!
//! ## Flush model
//!
//! Writes stage changes in `Ext`'s in-memory metadata + dirty-block
//! tracking; the disk only sees them after [`Ext::flush`]. We flush
//! on every `fsync`, `fsyncdir`, and `flush` callback, plus on
//! `destroy` (unmount). Between those, the FUSE kernel-side page
//! cache may serve subsequent reads from RAM anyway.

use std::ffi::OsStr;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fuser::{
    FUSE_ROOT_ID, FileAttr, FileType, Filesystem, KernelConfig, MountOption, ReplyAttr,
    ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyOpen, ReplyStatfs,
    ReplyWrite, ReplyXattr, Request, TimeOrNow,
};

use crate::block::BlockDevice;
use crate::fs::ext::constants;
use crate::fs::ext::{Ext, rw};
use crate::fs::{DeviceKind, FileMeta};

const TTL: Duration = Duration::from_secs(1);

/// A FUSE-mountable view of an ext{2,3,4} image.
pub struct FstoolFs {
    ext: Ext,
    dev: Box<dyn BlockDevice>,
}

impl FstoolFs {
    /// Wrap an already-opened [`Ext`] and its backing device. Caller
    /// is responsible for any required journal replay before
    /// constructing (replay is a no-op on a clean image, so passing
    /// the freshly-opened handle through is fine).
    pub fn new(ext: Ext, dev: Box<dyn BlockDevice>) -> Self {
        Self { ext, dev }
    }

    /// Mount under `mountpoint` and pump events on this thread until
    /// `umount` on the mountpoint or a fatal callback returns. Blocks
    /// indefinitely; spawn a thread if you want async-ish behaviour.
    pub fn mount(self, mountpoint: &Path, fs_name: &str) -> std::io::Result<()> {
        let opts = vec![
            MountOption::FSName(fs_name.to_string()),
            MountOption::AutoUnmount,
            MountOption::AllowOther,
        ];
        fuser::mount2(self, mountpoint, &opts)
    }

    fn ext_ino(&self, fuse_ino: u64) -> u32 {
        if fuse_ino == FUSE_ROOT_ID {
            constants::INO_ROOT_DIR
        } else {
            fuse_ino as u32
        }
    }

    fn fuse_ino(&self, ext_ino: u32) -> u64 {
        if ext_ino == constants::INO_ROOT_DIR {
            FUSE_ROOT_ID
        } else {
            ext_ino as u64
        }
    }

    fn attr_for(&mut self, ino: u32) -> Option<FileAttr> {
        let inode = self.ext.read_inode(self.dev.as_mut(), ino).ok()?;
        let kind = match inode.mode & constants::S_IFMT {
            constants::S_IFREG => FileType::RegularFile,
            constants::S_IFDIR => FileType::Directory,
            constants::S_IFLNK => FileType::Symlink,
            constants::S_IFBLK => FileType::BlockDevice,
            constants::S_IFCHR => FileType::CharDevice,
            constants::S_IFIFO => FileType::NamedPipe,
            constants::S_IFSOCK => FileType::Socket,
            _ => return None,
        };
        // Device numbers live in i_block[0] when the inode is a
        // char/block device (encoded by `add_device_to`).
        let rdev = if matches!(kind, FileType::CharDevice | FileType::BlockDevice) {
            inode.block[0]
        } else {
            0
        };
        Some(FileAttr {
            ino: self.fuse_ino(ino),
            size: inode.size as u64,
            blocks: inode.blocks_512 as u64,
            atime: ts(inode.atime),
            mtime: ts(inode.mtime),
            ctime: ts(inode.ctime),
            crtime: ts(inode.mtime),
            kind,
            perm: (inode.mode & 0o7777) as u16,
            nlink: inode.links_count as u32,
            uid: inode.uid as u32,
            gid: inode.gid as u32,
            rdev,
            blksize: self.ext.layout.block_size,
            flags: 0,
        })
    }

    fn now_secs() -> u32 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0)
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

fn ext_err_to_errno(e: &crate::Error) -> i32 {
    match e {
        crate::Error::InvalidArgument(_) => libc::EINVAL,
        crate::Error::Unsupported(_) => libc::ENOSYS,
        crate::Error::Io(_) => libc::EIO,
        _ => libc::EIO,
    }
}

impl Filesystem for FstoolFs {
    fn init(&mut self, _req: &Request<'_>, _config: &mut KernelConfig) -> Result<(), libc::c_int> {
        Ok(())
    }

    fn destroy(&mut self) {
        // Best-effort flush on unmount. We can't surface errors from
        // destroy — fuser swallows them — so just log via stderr.
        if let Err(e) = self.ext.flush(self.dev.as_mut()) {
            eprintln!("fstool fuse: flush on unmount failed: {e}");
        }
        let _ = self.dev.sync();
    }

    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let parent = self.ext_ino(parent);
        let name_bytes = name.as_encoded_bytes();
        let entries = match self.ext.list_inode(self.dev.as_mut(), parent) {
            Ok(v) => v,
            Err(_) => return reply.error(libc::ENOENT),
        };
        let entry = match entries.iter().find(|e| e.name.as_bytes() == name_bytes) {
            Some(e) => e,
            None => return reply.error(libc::ENOENT),
        };
        match self.attr_for(entry.inode) {
            Some(attr) => reply.entry(&TTL, &attr, 0),
            None => reply.error(libc::EIO),
        }
    }

    fn getattr(&mut self, _req: &Request<'_>, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        let ino = self.ext_ino(ino);
        match self.attr_for(ino) {
            Some(attr) => reply.attr(&TTL, &attr),
            None => reply.error(libc::ENOENT),
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
        let ino = self.ext_ino(ino);
        if let Some(m) = mode {
            if let Err(e) = self.ext.chmod(self.dev.as_mut(), ino, m as u16) {
                return reply.error(ext_err_to_errno(&e));
            }
        }
        if uid.is_some() || gid.is_some() {
            let cur = match self.ext.read_inode(self.dev.as_mut(), ino) {
                Ok(i) => i,
                Err(e) => return reply.error(ext_err_to_errno(&e)),
            };
            let new_uid = uid.unwrap_or(cur.uid as u32);
            let new_gid = gid.unwrap_or(cur.gid as u32);
            if let Err(e) = self.ext.chown(self.dev.as_mut(), ino, new_uid, new_gid) {
                return reply.error(ext_err_to_errno(&e));
            }
        }
        if atime.is_some() || mtime.is_some() || ctime.is_some() {
            let a = atime.map(time_or_now_secs);
            let m = mtime.map(time_or_now_secs);
            let c = ctime.map(|t| {
                t.duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs() as u32)
                    .unwrap_or(0)
            });
            if let Err(e) = self.ext.set_times(self.dev.as_mut(), ino, a, m, c) {
                return reply.error(ext_err_to_errno(&e));
            }
        }
        if let Some(sz) = size {
            if let Err(e) = self.ext.truncate(self.dev.as_mut(), ino, sz) {
                return reply.error(ext_err_to_errno(&e));
            }
        }
        match self.attr_for(ino) {
            Some(attr) => reply.attr(&TTL, &attr),
            None => reply.error(libc::EIO),
        }
    }

    fn readlink(&mut self, _req: &Request<'_>, ino: u64, reply: ReplyData) {
        let ino = self.ext_ino(ino);
        match self.ext.read_symlink_target(self.dev.as_mut(), ino) {
            Ok(target) => reply.data(target.as_bytes()),
            Err(e) => reply.error(ext_err_to_errno(&e)),
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
        let parent = self.ext_ino(parent);
        let mode_type = mode & 0o170000;
        let meta = FileMeta {
            mode: (mode & 0o7777) as u16,
            uid: req.uid(),
            gid: req.gid(),
            mtime: Self::now_secs(),
            atime: Self::now_secs(),
            ctime: Self::now_secs(),
        };
        let res = match mode_type as u16 {
            constants::S_IFREG => {
                use std::io::Cursor;
                let mut empty = Cursor::new(Vec::<u8>::new());
                self.ext.add_file_to_streaming(
                    self.dev.as_mut(),
                    parent,
                    name.as_encoded_bytes(),
                    &mut empty,
                    0,
                    meta,
                )
            }
            constants::S_IFCHR => {
                let major = (rdev >> 8) & 0xfff;
                let minor = (rdev & 0xff) | ((rdev >> 12) & 0xfff00);
                self.ext.add_device_to(
                    self.dev.as_mut(),
                    parent,
                    name.as_encoded_bytes(),
                    DeviceKind::Char,
                    major,
                    minor,
                    meta,
                )
            }
            constants::S_IFBLK => {
                let major = (rdev >> 8) & 0xfff;
                let minor = (rdev & 0xff) | ((rdev >> 12) & 0xfff00);
                self.ext.add_device_to(
                    self.dev.as_mut(),
                    parent,
                    name.as_encoded_bytes(),
                    DeviceKind::Block,
                    major,
                    minor,
                    meta,
                )
            }
            constants::S_IFIFO => self.ext.add_device_to(
                self.dev.as_mut(),
                parent,
                name.as_encoded_bytes(),
                DeviceKind::Fifo,
                0,
                0,
                meta,
            ),
            constants::S_IFSOCK => self.ext.add_device_to(
                self.dev.as_mut(),
                parent,
                name.as_encoded_bytes(),
                DeviceKind::Socket,
                0,
                0,
                meta,
            ),
            _ => return reply.error(libc::ENOSYS),
        };
        match res {
            Ok(new_ino) => match self.attr_for(new_ino) {
                Some(attr) => reply.entry(&TTL, &attr, 0),
                None => reply.error(libc::EIO),
            },
            Err(e) => reply.error(ext_err_to_errno(&e)),
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
        let parent = self.ext_ino(parent);
        let meta = FileMeta {
            mode: (mode & 0o7777) as u16,
            uid: req.uid(),
            gid: req.gid(),
            mtime: Self::now_secs(),
            atime: Self::now_secs(),
            ctime: Self::now_secs(),
        };
        match self
            .ext
            .add_dir_to(self.dev.as_mut(), parent, name.as_encoded_bytes(), meta)
        {
            Ok(new_ino) => match self.attr_for(new_ino) {
                Some(attr) => reply.entry(&TTL, &attr, 0),
                None => reply.error(libc::EIO),
            },
            Err(e) => reply.error(ext_err_to_errno(&e)),
        }
    }

    fn unlink(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let parent = self.ext_ino(parent);
        // The current remove_path takes an absolute path. Resolve a
        // synthetic path from parent + name; we don't have ino → path
        // reverse mapping yet so the underlying API does the work.
        let path = match self.path_of(parent, name.as_encoded_bytes()) {
            Ok(p) => p,
            Err(e) => return reply.error(ext_err_to_errno(&e)),
        };
        match self.ext.remove_path(self.dev.as_mut(), &path) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(ext_err_to_errno(&e)),
        }
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        // unlink + rmdir share remove_path's empty-dir check.
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
        let parent = self.ext_ino(parent);
        let target_bytes = target.as_os_str().as_encoded_bytes();
        let meta = FileMeta {
            mode: 0o777,
            uid: req.uid(),
            gid: req.gid(),
            mtime: Self::now_secs(),
            atime: Self::now_secs(),
            ctime: Self::now_secs(),
        };
        match self.ext.add_symlink_to(
            self.dev.as_mut(),
            parent,
            link_name.as_encoded_bytes(),
            target_bytes,
            meta,
        ) {
            Ok(new_ino) => match self.attr_for(new_ino) {
                Some(attr) => reply.entry(&TTL, &attr, 0),
                None => reply.error(libc::EIO),
            },
            Err(e) => reply.error(ext_err_to_errno(&e)),
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
        let parent = self.ext_ino(parent);
        let newparent = self.ext_ino(newparent);
        match self.ext.rename(
            self.dev.as_mut(),
            parent,
            name.as_encoded_bytes(),
            newparent,
            newname.as_encoded_bytes(),
        ) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(ext_err_to_errno(&e)),
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
        let ino = self.ext_ino(ino);
        let newparent = self.ext_ino(newparent);
        match self.ext.add_link_to(
            self.dev.as_mut(),
            newparent,
            newname.as_encoded_bytes(),
            ino,
        ) {
            Ok(()) => match self.attr_for(ino) {
                Some(attr) => reply.entry(&TTL, &attr, 0),
                None => reply.error(libc::EIO),
            },
            Err(e) => reply.error(ext_err_to_errno(&e)),
        }
    }

    fn open(&mut self, _req: &Request<'_>, _ino: u64, _flags: i32, reply: ReplyOpen) {
        // No per-open state beyond the inode itself; fh = 0 is a
        // valid signal that there's nothing to look up later.
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
        let ino = self.ext_ino(ino);
        let mut reader = match self.ext.open_file_reader(self.dev.as_mut(), ino) {
            Ok(r) => r,
            Err(e) => return reply.error(ext_err_to_errno(&e)),
        };
        if reader.seek(SeekFrom::Start(offset as u64)).is_err() {
            return reply.error(libc::EIO);
        }
        let mut buf = vec![0u8; size as usize];
        let n = match reader.read(&mut buf) {
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
        let ino = self.ext_ino(ino);
        // open_file_rw_ext_by_inode borrows ext+dev mutably; the
        // handle's Drop refreshes blocks_512 on the inode.
        let res = (|| -> std::io::Result<usize> {
            let mut h = rw::open_file_rw_ext_by_inode(&mut self.ext, self.dev.as_mut(), ino)
                .map_err(|e| std::io::Error::other(format!("{e}")))?;
            h.seek(SeekFrom::Start(offset as u64))?;
            h.write_all(data)?;
            Ok(data.len())
        })();
        match res {
            Ok(n) => reply.written(n as u32),
            Err(_) => reply.error(libc::EIO),
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
        match self.ext.flush(self.dev.as_mut()) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(ext_err_to_errno(&e)),
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
        match self.ext.flush(self.dev.as_mut()) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(ext_err_to_errno(&e)),
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
        let ino = self.ext_ino(ino);
        let entries = match self.ext.list_inode(self.dev.as_mut(), ino) {
            Ok(v) => v,
            Err(e) => return reply.error(ext_err_to_errno(&e)),
        };
        // Synthesize "." and "..". list_inode skips them; FUSE
        // expects them at offsets 0 and 1 conventionally, but the
        // kernel doesn't actually mind if we just emit children.
        // We do include them so userspace tools that compare with
        // POSIX behaviour don't trip.
        let parent_ino = self.parent_of(ino).unwrap_or(ino);
        let mut all: Vec<(u32, FileType, String)> = Vec::with_capacity(entries.len() + 2);
        all.push((ino, FileType::Directory, ".".to_string()));
        all.push((parent_ino, FileType::Directory, "..".to_string()));
        for e in &entries {
            if e.name == "." || e.name == ".." {
                continue;
            }
            let kind = match e.kind {
                crate::fs::EntryKind::Regular => FileType::RegularFile,
                crate::fs::EntryKind::Dir => FileType::Directory,
                crate::fs::EntryKind::Symlink => FileType::Symlink,
                crate::fs::EntryKind::Block => FileType::BlockDevice,
                crate::fs::EntryKind::Char => FileType::CharDevice,
                crate::fs::EntryKind::Fifo => FileType::NamedPipe,
                crate::fs::EntryKind::Socket => FileType::Socket,
                _ => FileType::RegularFile,
            };
            all.push((e.inode, kind, e.name.clone()));
        }
        for (i, (child, kind, name)) in all.into_iter().enumerate().skip(offset as usize) {
            if reply.add(self.fuse_ino(child), (i + 1) as i64, kind, name) {
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
        let sb = &self.ext.sb;
        reply.statfs(
            sb.blocks_count as u64,
            sb.free_blocks_count as u64,
            sb.free_blocks_count as u64,
            sb.inodes_count as u64,
            sb.free_inodes_count as u64,
            self.ext.layout.block_size,
            255,
            self.ext.layout.block_size,
        );
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
        // FUSE `create` is mknod (regular file) + open. We make the
        // file then return fh = 0 (same as `open`).
        let parent = self.ext_ino(parent);
        let meta = FileMeta {
            mode: (mode & 0o7777 & !(umask & 0o7777)) as u16,
            uid: req.uid(),
            gid: req.gid(),
            mtime: Self::now_secs(),
            atime: Self::now_secs(),
            ctime: Self::now_secs(),
        };
        use std::io::Cursor;
        let mut empty = Cursor::new(Vec::<u8>::new());
        match self.ext.add_file_to_streaming(
            self.dev.as_mut(),
            parent,
            name.as_encoded_bytes(),
            &mut empty,
            0,
            meta,
        ) {
            Ok(new_ino) => match self.attr_for(new_ino) {
                Some(attr) => reply.created(&TTL, &attr, 0, 0, 0),
                None => reply.error(libc::EIO),
            },
            Err(e) => reply.error(ext_err_to_errno(&e)),
        }
    }

    fn access(&mut self, _req: &Request<'_>, _ino: u64, _mask: i32, reply: ReplyEmpty) {
        // No permission enforcement; the kernel does its own UID/GID
        // checks against the attrs we already returned via getattr.
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
        let ino = self.ext_ino(ino);
        let xattrs = match self.ext.read_xattrs(self.dev.as_mut(), ino) {
            Ok(x) => x,
            Err(e) => return reply.error(ext_err_to_errno(&e)),
        };
        let want = name.as_encoded_bytes();
        let value = xattrs
            .iter()
            .find(|x| x.name.as_bytes() == want)
            .map(|x| x.value.clone());
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
        let ino = self.ext_ino(ino);
        let xattrs = match self.ext.read_xattrs(self.dev.as_mut(), ino) {
            Ok(x) => x,
            Err(e) => return reply.error(ext_err_to_errno(&e)),
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
}

impl FstoolFs {
    /// Build a virtual absolute path "/.../name" for an entry under
    /// `parent_ino`. Walks up the inode tree by following each dir's
    /// `..` entry. Used by unlink/rmdir to bridge to the path-based
    /// `remove_path` API.
    fn path_of(&mut self, parent_ino: u32, name: &[u8]) -> crate::Result<String> {
        let mut comps: Vec<String> = vec![String::from_utf8_lossy(name).into_owned()];
        let mut cur = parent_ino;
        let root = constants::INO_ROOT_DIR;
        while cur != root {
            let entries = self.ext.list_inode(self.dev.as_mut(), cur)?;
            let parent = entries
                .iter()
                .find(|e| e.name == "..")
                .map(|e| e.inode)
                .unwrap_or(root);
            let parent_entries = self.ext.list_inode(self.dev.as_mut(), parent)?;
            let my_name = parent_entries
                .iter()
                .find(|e| e.inode == cur)
                .map(|e| e.name.clone())
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "ext: inode {cur} not found in parent {parent}"
                    ))
                })?;
            comps.push(my_name);
            cur = parent;
            if cur == root {
                break;
            }
        }
        comps.reverse();
        let mut s = String::from("/");
        s.push_str(&comps.join("/"));
        Ok(s)
    }

    /// Resolve a directory inode's `..` to find its parent inode.
    fn parent_of(&mut self, dir_ino: u32) -> Option<u32> {
        let entries = self.ext.list_inode(self.dev.as_mut(), dir_ino).ok()?;
        entries.iter().find(|e| e.name == "..").map(|e| e.inode)
    }
}
