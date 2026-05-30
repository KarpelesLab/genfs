//! Apple DiskCopy 4.2 disk image — read-only container.
//!
//! DiskCopy 4.2 (the classic Mac OS disk-imaging format) wraps a raw volume in
//! an 84-byte header followed by the data fork, then optional tag data. The
//! real filesystem (typically classic HFS or MFS) is the data fork at file
//! offset `0x54`. This backend exposes that data fork as a read-only device, so
//! the usual filesystem detection runs against the inner volume transparently.
//!
//! Header layout (big-endian):
//! ```text
//!   0x00  64  disk name (Pascal string: length byte + up to 63 chars)
//!   0x40  u32 data fork size (bytes)
//!   0x44  u32 tag size (bytes)
//!   0x48  u32 data checksum
//!   0x4C  u32 tag checksum
//!   0x50  u8  disk encoding (0=400k, 1=800k, 2=720k, 3=1440k)
//!   0x51  u8  format byte
//!   0x52  u16 0x0100  (private / magic)
//!   0x54  ..  data fork (data-size bytes), then tag data
//! ```

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::BlockDevice;
use crate::Result;

/// Byte length of the fixed DiskCopy 4.2 header preceding the data fork.
const HEADER_LEN: u64 = 0x54;

/// Validate the 84-byte header and return `(data_size, tag_size)`.
fn header_lengths(h: &[u8]) -> Option<(u64, u64)> {
    if h.len() < HEADER_LEN as usize {
        return None;
    }
    // Private magic word 0x0100 at 0x52, plus a plausible floppy encoding byte.
    if u16::from_be_bytes([h[0x52], h[0x53]]) != 0x0100 || h[0x50] > 3 {
        return None;
    }
    let data = u32::from_be_bytes([h[0x40], h[0x41], h[0x42], h[0x43]]) as u64;
    let tag = u32::from_be_bytes([h[0x44], h[0x45], h[0x46], h[0x47]]) as u64;
    Some((data, tag))
}

/// `true` if `path` is a DiskCopy 4.2 image: magic `0x0100` at `0x52`, a floppy
/// encoding byte, and `84 + data_size + tag_size` accounts for the whole file.
pub fn probe(path: &Path) -> Result<bool> {
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };
    let mut h = [0u8; HEADER_LEN as usize];
    if f.read_exact(&mut h).is_err() {
        return Ok(false);
    }
    let file_len = f.metadata().map(|m| m.len()).unwrap_or(0);
    Ok(header_lengths(&h)
        .and_then(|(data, tag)| HEADER_LEN.checked_add(data)?.checked_add(tag))
        .map(|total| total == file_len)
        .unwrap_or(false))
}

/// Read-only view of a DiskCopy 4.2 image's data fork.
pub struct DiskCopy42Backend {
    inner: Box<dyn BlockDevice>,
    data_size: u64,
    cursor: u64,
}

impl DiskCopy42Backend {
    /// Wrap an already-open device as a DiskCopy 4.2 container, exposing its
    /// inner data fork. The device should have passed [`probe`].
    pub fn new(mut inner: Box<dyn BlockDevice>) -> Result<Self> {
        let mut h = [0u8; HEADER_LEN as usize];
        inner.read_at(0, &mut h)?;
        let (data_size, _tag) = header_lengths(&h).ok_or_else(|| {
            crate::Error::InvalidImage("diskcopy: not a DiskCopy 4.2 image".into())
        })?;
        if HEADER_LEN
            .checked_add(data_size)
            .is_none_or(|end| end > inner.total_size())
        {
            return Err(crate::Error::InvalidImage(
                "diskcopy: data fork extends past end of file".into(),
            ));
        }
        Ok(Self {
            inner,
            data_size,
            cursor: 0,
        })
    }
}

impl BlockDevice for DiskCopy42Backend {
    fn block_size(&self) -> u32 {
        512
    }

    fn total_size(&self) -> u64 {
        self.data_size
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let len = buf.len() as u64;
        if offset
            .checked_add(len)
            .is_none_or(|end| end > self.data_size)
        {
            return Err(crate::Error::OutOfBounds {
                offset,
                len,
                size: self.data_size,
            });
        }
        self.inner.read_at(HEADER_LEN + offset, buf)
    }

    fn write_at(&mut self, _offset: u64, _buf: &[u8]) -> Result<()> {
        Err(crate::Error::Unsupported(
            "diskcopy: read-only container; writes are out of scope".into(),
        ))
    }
}

impl Read for DiskCopy42Backend {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.cursor >= self.data_size {
            return Ok(0);
        }
        let take = ((self.data_size - self.cursor) as usize).min(buf.len());
        if take == 0 {
            return Ok(0);
        }
        self.read_at(self.cursor, &mut buf[..take])
            .map_err(|e| io::Error::other(format!("{e}")))?;
        self.cursor += take as u64;
        Ok(take)
    }
}

impl Write for DiskCopy42Backend {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("diskcopy: read-only container"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for DiskCopy42Backend {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new = match pos {
            SeekFrom::Start(n) => n as i128,
            SeekFrom::End(d) => self.data_size as i128 + d as i128,
            SeekFrom::Current(d) => self.cursor as i128 + d as i128,
        };
        if new < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "diskcopy: seek before start",
            ));
        }
        self.cursor = new as u64;
        Ok(self.cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;

    /// Build a minimal DiskCopy 4.2 image wrapping `payload` as the data fork.
    fn wrap(payload: &[u8]) -> Vec<u8> {
        let mut v = vec![0u8; HEADER_LEN as usize];
        v[0] = 4;
        v[1..5].copy_from_slice(b"test");
        v[0x40..0x44].copy_from_slice(&(payload.len() as u32).to_be_bytes());
        // tag size 0
        v[0x50] = 3; // 1440k encoding
        v[0x52] = 0x01;
        v[0x53] = 0x00;
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn unwraps_data_fork() {
        let payload: Vec<u8> = (0..2048u32).map(|i| (i * 7) as u8).collect();
        let img = wrap(&payload);
        let mut dev = MemoryBackend::new(img.len() as u64);
        dev.write_at(0, &img).unwrap();

        let mut dc = DiskCopy42Backend::new(Box::new(dev)).unwrap();
        assert_eq!(dc.total_size(), payload.len() as u64);
        let mut got = vec![0u8; payload.len()];
        dc.read_at(0, &mut got).unwrap();
        assert_eq!(got, payload);

        // A mid-fork read lands at the right offset (header is skipped).
        let mut mid = [0u8; 16];
        dc.read_at(100, &mut mid).unwrap();
        assert_eq!(&mid, &payload[100..116]);

        // Out-of-bounds + writes are rejected.
        assert!(dc.read_at(payload.len() as u64, &mut [0u8; 1]).is_err());
        assert!(dc.write_at(0, &[0u8; 1]).is_err());
    }
}
