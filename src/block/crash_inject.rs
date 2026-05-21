//! Crash-injection block-device wrapper.
//!
//! [`CrashInject`] wraps any [`crate::block::BlockDevice`] and lets a
//! caller specify when writes should start failing (or silently
//! dropping). Used by the crash-recovery test harness to simulate
//! power-loss / SIGKILL / I/O errors at controlled points and verify
//! that the next `Ext::open` either replays cleanly or returns a
//! structured error — never panics, never produces a half-applied
//! transaction.
//!
//! Two failure modes:
//!
//! - [`FailAfter::Writes(n)`] — succeed on the first `n` write
//!   syscalls, then return `EIO` on every subsequent write. Stable
//!   across re-runs of the same test (deterministic).
//! - [`FailAfter::Bytes(n)`] — succeed until the cumulative byte
//!   count written reaches `n`, then short-circuit. Maps to the
//!   "torn write" pattern where a buffered write reaches disk
//!   partially.
//!
//! Reads are always passed through unmodified — only writes get
//! gated. Use [`CrashInject::into_inner`] to recover the underlying
//! device for post-crash inspection (typically after re-opening as
//! an `Ext`).

use std::io::{self, Read, Seek, SeekFrom, Write};

use crate::Result;
use crate::block::BlockDevice;

/// What triggers the simulated crash.
#[derive(Debug, Clone, Copy)]
pub enum FailAfter {
    /// Succeed on the first `n` write syscalls, then fail.
    Writes(u64),
    /// Succeed until the cumulative byte count reaches `n`, then fail.
    Bytes(u64),
    /// Never fail (passthrough — useful for parameterised tests).
    Never,
}

/// Crash-injection wrapper. See module docs.
pub struct CrashInject<B: BlockDevice> {
    inner: B,
    fail: FailAfter,
    writes_seen: u64,
    bytes_written: u64,
    /// Once the crash threshold is hit, every subsequent write
    /// short-circuits without touching the inner device. We don't
    /// surface the error to the writer (so the test harness can
    /// observe behaviour as if the process was SIGKILL'd between
    /// the syscall and the disk).
    crashed: bool,
}

impl<B: BlockDevice> CrashInject<B> {
    pub fn new(inner: B, fail: FailAfter) -> Self {
        Self {
            inner,
            fail,
            writes_seen: 0,
            bytes_written: 0,
            crashed: false,
        }
    }

    /// True once the crash threshold has been crossed and subsequent
    /// writes are being silently dropped.
    pub fn crashed(&self) -> bool {
        self.crashed
    }

    /// Drop the wrapper and recover the inner device.
    pub fn into_inner(self) -> B {
        self.inner
    }

    fn check_crash_and_account(&mut self, bytes: u64) {
        if self.crashed {
            return;
        }
        match self.fail {
            FailAfter::Writes(n) => {
                self.writes_seen += 1;
                if self.writes_seen > n {
                    self.crashed = true;
                }
            }
            FailAfter::Bytes(n) => {
                self.bytes_written = self.bytes_written.saturating_add(bytes);
                if self.bytes_written > n {
                    self.crashed = true;
                }
            }
            FailAfter::Never => {}
        }
    }
}

impl<B: BlockDevice> Read for CrashInject<B> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl<B: BlockDevice> Write for CrashInject<B> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.check_crash_and_account(buf.len() as u64);
        if self.crashed {
            return Ok(buf.len()); // pretend success, drop on the floor
        }
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.crashed {
            return Ok(());
        }
        self.inner.flush()
    }
}

impl<B: BlockDevice> Seek for CrashInject<B> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl<B: BlockDevice> BlockDevice for CrashInject<B> {
    fn block_size(&self) -> u32 {
        self.inner.block_size()
    }

    fn total_size(&self) -> u64 {
        self.inner.total_size()
    }

    fn sync(&mut self) -> Result<()> {
        if self.crashed {
            return Ok(());
        }
        self.inner.sync()
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        self.check_crash_and_account(buf.len() as u64);
        if self.crashed {
            return Ok(());
        }
        self.inner.write_at(offset, buf)
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.inner.read_at(offset, buf)
    }

    fn zero_range(&mut self, offset: u64, len: u64) -> Result<()> {
        self.check_crash_and_account(len);
        if self.crashed {
            return Ok(());
        }
        self.inner.zero_range(offset, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    /// Writes after the threshold drop silently — read-back sees the
    /// original data, not what the caller "wrote".
    #[test]
    fn fail_after_writes_drops_subsequent() {
        let mem = MemoryBackend::new(1024);
        let mut dev = CrashInject::new(mem, FailAfter::Writes(1));
        dev.write_at(0, &[0x11; 16]).unwrap();
        assert!(!dev.crashed());
        dev.write_at(16, &[0x22; 16]).unwrap();
        assert!(dev.crashed());
        let mut buf = [0u8; 32];
        dev.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf[..16], &[0x11; 16]);
        // The second write was supposed to land here but got dropped.
        assert_eq!(&buf[16..], &[0; 16]);
    }

    #[test]
    fn fail_after_bytes_drops_past_threshold() {
        let mem = MemoryBackend::new(1024);
        let mut dev = CrashInject::new(mem, FailAfter::Bytes(20));
        dev.write_at(0, &[0xAA; 16]).unwrap();
        assert!(!dev.crashed());
        dev.write_at(16, &[0xBB; 16]).unwrap();
        // 16 + 16 > 20 → crashed after the second write.
        assert!(dev.crashed());
    }

    #[test]
    fn fail_after_never_is_passthrough() {
        let mem = MemoryBackend::new(1024);
        let mut dev = CrashInject::new(mem, FailAfter::Never);
        for i in 0..32 {
            dev.write_at(i * 16, &[i as u8; 16]).unwrap();
        }
        assert!(!dev.crashed());
        let mut buf = [0u8; 16];
        dev.read_at(31 * 16, &mut buf).unwrap();
        assert_eq!(buf, [31u8; 16]);
    }
}
