//! Shared-access wrappers for the otherwise-single-threaded `Ext` API.
//!
//! [`SharedExt`] couples an [`Ext`] handle and its backing
//! [`BlockDevice`] under one mutex so they can be shared across
//! threads safely. Every call locks the mutex for the duration of
//! the inner operation — that gives correctness without true
//! parallelism: reads of different files serialise the same way
//! writes do.
//!
//! This is the "safe, not fast" half of the concurrency story. The
//! Linux kernel's per-inode `i_rwsem` + per-extent locking + journal
//! sequencing is what enables real concurrent throughput on ext4;
//! shipping the same here means restructuring nearly every field on
//! `Ext` (the staged inode/block/dir caches in particular) into
//! interior-mutex'd containers. That work is tracked as a separate
//! followup.
//!
//! What you get today:
//!
//! - `SharedExt::new(ext, dev)` → `Arc`-shared handle.
//! - `.with_inner(|ext, dev| …)` → run a closure with mutable access
//!   to both; the lock is held only across the closure.
//! - Convenience wrappers for the most common read paths
//!   (`read_inode`, `list_inode`, `path_to_inode`) so callers don't
//!   have to spell out the closure every time.
//!
//! What you don't get today:
//!
//! - Multiple readers in parallel (single mutex blocks even
//!   read-only ops).
//! - Async / `tokio` integration. Wrap in `tokio::task::spawn_blocking`
//!   if you need it from an async context.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::block::BlockDevice;
use crate::fs::ext::Ext;
use crate::fs::{DirEntry, FileMeta};

/// Thread-safe shared handle on an [`Ext`] + its [`BlockDevice`].
/// Cheap to clone (it's an `Arc` under the hood); each clone shares
/// the same underlying state.
#[derive(Clone)]
pub struct SharedExt {
    inner: Arc<Mutex<Inner>>,
}

struct Inner {
    ext: Ext,
    dev: Box<dyn BlockDevice>,
}

impl SharedExt {
    /// Wrap an opened `Ext` + its device. Subsequent `.clone()`s
    /// share the same state.
    pub fn new(ext: Ext, dev: Box<dyn BlockDevice>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner { ext, dev })),
        }
    }

    /// Lock the inner handle and run `f` with mutable access to both
    /// the `Ext` and the device. Returns whatever `f` returns. Errors
    /// from a poisoned mutex are surfaced as `Err`.
    pub fn with_inner<F, T>(&self, f: F) -> crate::Result<T>
    where
        F: FnOnce(&mut Ext, &mut dyn BlockDevice) -> crate::Result<T>,
    {
        let mut guard: MutexGuard<'_, Inner> = self
            .inner
            .lock()
            .map_err(|_| crate::Error::Unsupported("SharedExt: mutex poisoned".into()))?;
        let inner = &mut *guard;
        f(&mut inner.ext, inner.dev.as_mut())
    }

    // ─── Convenience wrappers for the most common read paths ─────

    /// Read an inode by its number.
    pub fn read_inode(&self, ino: u32) -> crate::Result<crate::fs::ext::inode::Inode> {
        self.with_inner(|ext, dev| ext.read_inode(dev, ino))
    }

    /// List a directory's entries.
    pub fn list_inode(&self, ino: u32) -> crate::Result<Vec<DirEntry>> {
        self.with_inner(|ext, dev| ext.list_inode(dev, ino))
    }

    /// Resolve an absolute path to its inode number.
    pub fn path_to_inode(&self, path: &str) -> crate::Result<u32> {
        self.with_inner(|ext, dev| ext.path_to_inode(dev, path))
    }

    /// Add a regular file under `parent_ino`. Locks for the duration
    /// of the write.
    pub fn add_file_to_streaming(
        &self,
        parent_ino: u32,
        name: &[u8],
        reader: &mut dyn std::io::Read,
        len: u64,
        meta: FileMeta,
    ) -> crate::Result<u32> {
        self.with_inner(|ext, dev| {
            ext.add_file_to_streaming(dev, parent_ino, name, reader, len, meta)
        })
    }

    /// Persist staged metadata to the underlying device.
    pub fn flush(&self) -> crate::Result<()> {
        self.with_inner(|ext, dev| ext.flush(dev))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;
    use crate::fs::ext::{FormatOpts, FsKind};
    use std::sync::Arc;
    use std::thread;

    /// Hammer the same `SharedExt` from many threads with mixed
    /// read + write ops. Verifies the mutex prevents data races and
    /// that the final state is consistent: every named file we added
    /// is enumerable, and every inode resolves to a valid mode.
    #[test]
    fn shared_ext_takes_concurrent_traffic() {
        let opts = FormatOpts {
            kind: FsKind::Ext4,
            block_size: 1024,
            blocks_count: 4096,
            inodes_count: 256,
            journal_blocks: 1024,
            ..FormatOpts::default()
        };
        let total = opts.blocks_count as u64 * opts.block_size as u64;
        let mut dev = MemoryBackend::new(total);
        let ext = Ext::format_with(&mut dev, &opts).unwrap();
        let shared = SharedExt::new(ext, Box::new(dev));

        // Spawn 8 writer threads, each adds 5 files. Plus 4 reader
        // threads that walk the root dir on every iteration.
        let n_writers = 8usize;
        let per_writer = 5u32;
        let mut handles = Vec::new();

        for w in 0..n_writers {
            let s = shared.clone();
            handles.push(thread::spawn(move || -> crate::Result<()> {
                for i in 0..per_writer {
                    let name = format!("w{w}_f{i}");
                    let body = format!("writer {w} file {i}\n");
                    let mut reader = std::io::Cursor::new(body.into_bytes());
                    let len = reader.get_ref().len() as u64;
                    s.add_file_to_streaming(
                        2,
                        name.as_bytes(),
                        &mut reader,
                        len,
                        FileMeta::with_mode(0o644),
                    )?;
                }
                Ok(())
            }));
        }
        for _ in 0..4 {
            let s = shared.clone();
            handles.push(thread::spawn(move || -> crate::Result<()> {
                for _ in 0..20 {
                    let _ = s.list_inode(2)?;
                }
                Ok(())
            }));
        }

        for h in handles {
            h.join().unwrap().unwrap();
        }

        let entries = shared.list_inode(2).unwrap();
        let names: std::collections::HashSet<String> =
            entries.iter().map(|e| e.name.clone()).collect();
        for w in 0..n_writers {
            for i in 0..per_writer {
                let expected = format!("w{w}_f{i}");
                assert!(
                    names.contains(&expected),
                    "writer {w} file {i} missing from final listing: {names:?}"
                );
            }
        }
        // Every inode must resolve cleanly (no torn metadata).
        for e in &entries {
            let inode = shared.read_inode(e.inode).unwrap();
            assert_ne!(inode.mode, 0, "inode {} has zero mode", e.inode);
        }
    }

    /// Send + Sync sanity check: code that won't compile is the
    /// strongest possible assertion. If `SharedExt: Send + Sync`
    /// breaks, this test fails to compile.
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}
    #[allow(dead_code)]
    fn _check_shared_ext_send_sync() {
        assert_send_sync::<SharedExt>();
        assert_send_sync::<Arc<SharedExt>>();
    }
}
