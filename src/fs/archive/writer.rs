//! Shared write helper: a bumping cursor over a pre-sized device with a
//! capacity guard. The format writers (zip / cpio / ar) own one of
//! these and stream headers + bodies through it, exactly like tar's
//! `TarWriter`.

use crate::Result;
use crate::block::BlockDevice;

/// A monotonic write cursor over a device. The device is sized by the
/// factory (`create_image` / `build_bare_via_trait`); we track our own
/// position against `total_size` and refuse to overrun it.
pub struct Cursor {
    pos: u64,
    capacity: u64,
}

impl Cursor {
    pub fn new(dev: &dyn BlockDevice) -> Self {
        Self {
            pos: 0,
            capacity: dev.total_size(),
        }
    }

    /// Current write position (== bytes written so far).
    pub fn position(&self) -> u64 {
        self.pos
    }

    /// Append `buf` at the cursor, advancing it.
    pub fn write(&mut self, dev: &mut dyn BlockDevice, buf: &[u8]) -> Result<()> {
        if self.pos + buf.len() as u64 > self.capacity {
            return Err(crate::Error::OutOfBounds {
                offset: self.pos,
                len: buf.len() as u64,
                size: self.capacity,
            });
        }
        dev.write_at(self.pos, buf)?;
        self.pos += buf.len() as u64;
        Ok(())
    }
}
