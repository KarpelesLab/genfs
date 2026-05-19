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

use std::io::{self, Read, Write};

// ============================ xz / lzma ============================

/// Codec selector for [`DecoderAdapter`] / [`EncoderAdapter`].
#[cfg(any(feature = "xz", feature = "lzma"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LzmaFlavour {
    /// `.xz` container.
    Xz,
    /// Raw LZMA1 stream (the legacy `.lzma` format).
    Lzma,
}

/// Buffering decoder shared by xz and lzma.
#[cfg(any(feature = "xz", feature = "lzma"))]
pub struct DecoderAdapter<R: Read> {
    inner: R,
    flavour: LzmaFlavour,
    decoded: Option<io::Cursor<Vec<u8>>>,
}

#[cfg(feature = "xz")]
impl<R: Read> DecoderAdapter<R> {
    pub fn new_xz(r: R) -> Self {
        Self {
            inner: r,
            flavour: LzmaFlavour::Xz,
            decoded: None,
        }
    }
}

#[cfg(feature = "lzma")]
impl<R: Read> DecoderAdapter<R> {
    pub fn new_lzma(r: R) -> Self {
        Self {
            inner: r,
            flavour: LzmaFlavour::Lzma,
            decoded: None,
        }
    }
}

#[cfg(any(feature = "xz", feature = "lzma"))]
impl<R: Read> Read for DecoderAdapter<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.decoded.is_none() {
            let mut compressed = Vec::new();
            self.inner.read_to_end(&mut compressed)?;
            let mut out = Vec::new();
            let mut input = std::io::BufReader::new(&compressed[..]);
            match self.flavour {
                LzmaFlavour::Xz => {
                    #[cfg(feature = "xz")]
                    {
                        lzma_rs::xz_decompress(&mut input, &mut out).map_err(|e| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("xz decode failed: {e}"),
                            )
                        })?;
                    }
                    #[cfg(not(feature = "xz"))]
                    unreachable!("xz flavour built without `xz` feature");
                }
                LzmaFlavour::Lzma => {
                    #[cfg(feature = "lzma")]
                    {
                        lzma_rs::lzma_decompress(&mut input, &mut out).map_err(|e| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("lzma decode failed: {e}"),
                            )
                        })?;
                    }
                    #[cfg(not(feature = "lzma"))]
                    unreachable!("lzma flavour built without `lzma` feature");
                }
            }
            self.decoded = Some(io::Cursor::new(out));
        }
        self.decoded.as_mut().unwrap().read(buf)
    }
}

/// Buffering encoder shared by xz and lzma.
#[cfg(any(feature = "xz", feature = "lzma"))]
pub struct EncoderAdapter<W: Write> {
    inner: Option<W>,
    flavour: LzmaFlavour,
    buf: Vec<u8>,
}

#[cfg(feature = "xz")]
impl<W: Write> EncoderAdapter<W> {
    pub fn new_xz(w: W) -> Self {
        Self {
            inner: Some(w),
            flavour: LzmaFlavour::Xz,
            buf: Vec::new(),
        }
    }
}

#[cfg(feature = "lzma")]
impl<W: Write> EncoderAdapter<W> {
    pub fn new_lzma(w: W) -> Self {
        Self {
            inner: Some(w),
            flavour: LzmaFlavour::Lzma,
            buf: Vec::new(),
        }
    }
}

#[cfg(any(feature = "xz", feature = "lzma"))]
impl<W: Write> EncoderAdapter<W> {
    /// Encode the staged buffer and write it to the inner writer once.
    /// Used by both `flush` and the `Drop` finaliser.
    fn finish_inner(&mut self) -> io::Result<()> {
        if self.inner.is_none() {
            return Ok(());
        }
        let mut input = std::io::BufReader::new(&self.buf[..]);
        let mut out = Vec::new();
        match self.flavour {
            LzmaFlavour::Xz => {
                #[cfg(feature = "xz")]
                lzma_rs::xz_compress(&mut input, &mut out)
                    .map_err(|e| io::Error::other(format!("xz encode failed: {e}")))?;
                #[cfg(not(feature = "xz"))]
                unreachable!("xz encoder built without `xz` feature");
            }
            LzmaFlavour::Lzma => {
                #[cfg(feature = "lzma")]
                lzma_rs::lzma_compress(&mut input, &mut out)
                    .map_err(|e| io::Error::other(format!("lzma encode failed: {e}")))?;
                #[cfg(not(feature = "lzma"))]
                unreachable!("lzma encoder built without `lzma` feature");
            }
        }
        let mut w = self.inner.take().unwrap();
        w.write_all(&out)?;
        w.flush()?;
        self.buf.clear();
        Ok(())
    }
}

#[cfg(any(feature = "xz", feature = "lzma"))]
impl<W: Write> Write for EncoderAdapter<W> {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(b);
        Ok(b.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.finish_inner()
    }
}

#[cfg(any(feature = "xz", feature = "lzma"))]
impl<W: Write> Drop for EncoderAdapter<W> {
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
