//! Streaming adapters for codecs whose underlying crate exposes only a
//! "decompress all" / "compress all" function.
//!
//! `lzma-rs` and `minilzo-rs` both work on slices, not streams. We wrap
//! them by buffering the whole input on the read side (then handing out
//! the decoded payload through a `Cursor`) and by buffering all writes
//! until `flush` (then emitting one compressed payload).
//!
//! That's a streaming API in shape but a buffer-the-whole-thing
//! implementation under the hood. For SquashFS metablocks and tar
//! frames this is fine — the upper-layer caller chunks input below 1
//! MiB so the working set stays small. For multi-GiB single-stream
//! compressed tar archives the gzip / zstd / lz4 adapters (which DO
//! stream natively in their crates) are the right pick anyway.

#[cfg(any(feature = "lzma", feature = "lzo"))]
use std::io::{self, Read, Write};

// =============================== lzma ==============================
//
// `lzma-rs` exposes only slice-based `lzma_decompress` / `lzma_compress`
// (the legacy `.lzma` alone format), so we buffer the whole payload to
// give the streaming API its shape. lzma stays on `lzma-rs` because
// compcol 0.4.0's alone codec isn't liblzma-interoperable yet
// (KarpelesLab/compcol#14); xz/gzip/zlib/zstd are on compcol.

/// Buffering `.lzma` decoder: reads all compressed input on first
/// `read`, decodes it once, then hands out the plaintext via a cursor.
#[cfg(feature = "lzma")]
pub struct LzmaDecoderAdapter<R: Read> {
    inner: R,
    decoded: Option<io::Cursor<Vec<u8>>>,
}

#[cfg(feature = "lzma")]
impl<R: Read> LzmaDecoderAdapter<R> {
    pub fn new(r: R) -> Self {
        Self {
            inner: r,
            decoded: None,
        }
    }
}

#[cfg(feature = "lzma")]
impl<R: Read> Read for LzmaDecoderAdapter<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.decoded.is_none() {
            let mut compressed = Vec::new();
            self.inner.read_to_end(&mut compressed)?;
            let mut out = Vec::new();
            let mut input = std::io::BufReader::new(&compressed[..]);
            lzma_rs::lzma_decompress(&mut input, &mut out).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("lzma decode failed: {e}"),
                )
            })?;
            self.decoded = Some(io::Cursor::new(out));
        }
        self.decoded.as_mut().unwrap().read(buf)
    }
}

/// Buffering `.lzma` encoder: stages every write, then encodes the whole
/// buffer to the inner writer once on `flush` / `Drop`.
#[cfg(feature = "lzma")]
pub struct LzmaEncoderAdapter<W: Write> {
    inner: Option<W>,
    buf: Vec<u8>,
}

#[cfg(feature = "lzma")]
impl<W: Write> LzmaEncoderAdapter<W> {
    pub fn new(w: W) -> Self {
        Self {
            inner: Some(w),
            buf: Vec::new(),
        }
    }

    fn finish_inner(&mut self) -> io::Result<()> {
        let Some(mut w) = self.inner.take() else {
            return Ok(());
        };
        let mut input = std::io::BufReader::new(&self.buf[..]);
        let mut out = Vec::new();
        lzma_rs::lzma_compress(&mut input, &mut out)
            .map_err(|e| io::Error::other(format!("lzma encode failed: {e}")))?;
        w.write_all(&out)?;
        w.flush()?;
        self.buf.clear();
        Ok(())
    }
}

#[cfg(feature = "lzma")]
impl<W: Write> Write for LzmaEncoderAdapter<W> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(b);
        Ok(b.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.finish_inner()
    }
}

#[cfg(feature = "lzma")]
impl<W: Write> Drop for LzmaEncoderAdapter<W> {
    fn drop(&mut self) {
        let _ = self.finish_inner();
    }
}

// =============================== lzo ===============================
//
// LZO1X has no native frame format and the `minilzo-rs` API takes
// whole slices. We invent a tiny frame: a stream of `(u32 LE
// compressed-len, u32 LE uncompressed-len, payload)` records terminated
// by a `(0, 0)` sentinel. Records are emitted in 1 MiB chunks. Files
// produced with this framing are NOT interchangeable with any other
// LZO toolchain — they exist only so `make_reader_lzo`'s `Read` impl
// can produce predictable streaming output.

#[cfg(feature = "lzo")]
const LZO_FRAME_CHUNK: usize = 1 << 20;

/// Stream encoder for the fstool LZO frame format described above.
#[cfg(feature = "lzo")]
pub struct LzoFrameWriter<W: Write> {
    inner: Option<W>,
    pending: Vec<u8>,
}

#[cfg(feature = "lzo")]
impl<W: Write> LzoFrameWriter<W> {
    pub fn new(w: W) -> Self {
        Self {
            inner: Some(w),
            pending: Vec::with_capacity(LZO_FRAME_CHUNK),
        }
    }

    fn emit_chunk(&mut self) -> io::Result<()> {
        if self.pending.is_empty() || self.inner.is_none() {
            return Ok(());
        }
        let mut lzo =
            minilzo_rs::LZO::init().map_err(|e| io::Error::other(format!("lzo init: {e:?}")))?;
        let compressed = lzo
            .compress(&self.pending)
            .map_err(|e| io::Error::other(format!("lzo encode failed: {e:?}")))?;
        let w = self.inner.as_mut().unwrap();
        w.write_all(&(compressed.len() as u32).to_le_bytes())?;
        w.write_all(&(self.pending.len() as u32).to_le_bytes())?;
        w.write_all(&compressed)?;
        self.pending.clear();
        Ok(())
    }
}

#[cfg(feature = "lzo")]
impl<W: Write> Write for LzoFrameWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let space = LZO_FRAME_CHUNK - self.pending.len();
        let take = buf.len().min(space);
        self.pending.extend_from_slice(&buf[..take]);
        if self.pending.len() == LZO_FRAME_CHUNK {
            self.emit_chunk()?;
        }
        Ok(take)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.emit_chunk()?;
        if let Some(w) = self.inner.as_mut() {
            // Sentinel: zero-length chunk header marks end of stream.
            w.write_all(&[0u8; 8])?;
            w.flush()?;
        }
        Ok(())
    }
}

#[cfg(feature = "lzo")]
impl<W: Write> Drop for LzoFrameWriter<W> {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Stream decoder for the fstool LZO frame format.
#[cfg(feature = "lzo")]
pub struct LzoFrameReader<R: Read> {
    inner: R,
    pending: io::Cursor<Vec<u8>>,
    done: bool,
}

#[cfg(feature = "lzo")]
impl<R: Read> LzoFrameReader<R> {
    pub fn new(r: R) -> Self {
        Self {
            inner: r,
            pending: io::Cursor::new(Vec::new()),
            done: false,
        }
    }

    fn fill_next_chunk(&mut self) -> io::Result<()> {
        if self.done {
            return Ok(());
        }
        let mut hdr = [0u8; 8];
        self.inner.read_exact(&mut hdr)?;
        let clen = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
        let ulen = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
        if clen == 0 && ulen == 0 {
            self.done = true;
            return Ok(());
        }
        let mut compressed = vec![0u8; clen];
        self.inner.read_exact(&mut compressed)?;
        let lzo =
            minilzo_rs::LZO::init().map_err(|e| io::Error::other(format!("lzo init: {e:?}")))?;
        let out = lzo.decompress(&compressed, ulen).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("lzo decode failed: {e:?}"),
            )
        })?;
        self.pending = io::Cursor::new(out);
        Ok(())
    }
}

#[cfg(feature = "lzo")]
impl<R: Read> Read for LzoFrameReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            let n = self.pending.read(buf)?;
            if n > 0 {
                return Ok(n);
            }
            if self.done {
                return Ok(0);
            }
            self.fill_next_chunk()?;
            if self.pending.get_ref().is_empty() && self.done {
                return Ok(0);
            }
        }
    }
}
