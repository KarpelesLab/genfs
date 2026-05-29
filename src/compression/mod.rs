//! Compression / decompression codecs.
//!
//! Each algorithm sits behind its own Cargo feature flag (`gzip`, `xz`,
//! `lzma`, `lz4`, `zstd`, `lzo`). All six are enabled by default; trim
//! the dependency tree by switching `default-features = false` and
//! picking the subset you need.
//!
//! Two API layers:
//!
//! - **Block API** (`decompress` / `compress`): one-shot, in-memory.
//!   Used by SquashFS, whose metablocks are ≤ 8 KiB and whose data
//!   blocks are ≤ 1 MiB. Both sides cap at a configurable maximum so a
//!   corrupt header can't make us allocate gigabytes.
//! - **Stream API** (`make_reader` / `make_writer`): wraps a `Read` or
//!   `Write` in an algorithm-specific codec. Used by streaming tar so a
//!   `.tar.gz` can be walked end-to-end without ever buffering the full
//!   archive.
//!
//! Auto-detection lives in [`detect_magic`]: feed it the first ~6 bytes
//! of an input and it tells you which algorithm the stream uses, or
//! `None` for plain (uncompressed) input.
//!
//! ## Crate choices
//!
//! - **gzip / zlib / xz / zstd**: [`compcol`], a uniform pure-Rust codec
//!   collection behind one `Encoder`/`Decoder` trait. The block API rides
//!   `compcol::vec`, the stream API `compcol::io`, and the output cap
//!   `compcol::limit::LimitedDecoder`.
//! - **lzma**: `lzma-rs` (legacy `.lzma` alone format). Not yet on compcol
//!   because compcol 0.4.0's alone codec isn't liblzma-interoperable
//!   (KarpelesLab/compcol#14); the `.xz` container is on compcol.
//! - **lz4**: `lz4_flex`, pure-Rust. We use the LZ4 frame format for
//!   tar streaming and the LZ4 block format for SquashFS. (compcol can't
//!   yet emit raw blocks / a canonical frame — compcol #9 / #10.)
//! - **lzo**: `minilzo-rs`, wraps the upstream `minilzo` C library
//!   (LZO1X-1 / LZO1X-999). SquashFS encodes raw LZO1X blocks.

use crate::Result;

/// Which compression algorithm a stream / block is encoded with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Algo {
    /// gzip framing: 10-byte header + deflate + 8-byte trailer (crc32 + isize).
    /// Used by `.tar.gz`. The `gzip` feature gates the codec.
    Gzip,
    /// zlib framing: 2-byte header + deflate + 4-byte adler32. SquashFS
    /// labels this "gzip" on disk (compressor id 1) but uses zlib framing.
    /// Same `gzip` feature.
    Zlib,
    Xz,
    Lzma,
    Lz4,
    Zstd,
    Lzo,
}

impl Algo {
    /// Short human-readable name (the one used in error messages and CLI
    /// filename extensions: `.tar.gz`, `.tar.zst`, etc.).
    pub fn name(self) -> &'static str {
        match self {
            Self::Gzip => "gzip",
            Self::Zlib => "zlib",
            Self::Xz => "xz",
            Self::Lzma => "lzma",
            Self::Lz4 => "lz4",
            Self::Zstd => "zstd",
            Self::Lzo => "lzo",
        }
    }

    /// True when the algorithm's feature flag was enabled at build
    /// time. Always check this before calling the block / stream APIs;
    /// otherwise those return [`crate::Error::Unsupported`].
    pub fn enabled(self) -> bool {
        match self {
            Self::Gzip | Self::Zlib => cfg!(feature = "gzip"),
            Self::Xz => cfg!(feature = "xz"),
            Self::Lzma => cfg!(feature = "lzma"),
            Self::Lz4 => cfg!(feature = "lz4"),
            Self::Zstd => cfg!(feature = "zstd"),
            Self::Lzo => cfg!(feature = "lzo"),
        }
    }

    /// Guess the algorithm from a filename's suffix. Returns `None` for
    /// `.tar` or anything we don't recognise. Used by the CLI to pick a
    /// codec when the user types `fstool repack disk.img out.tar.zst`.
    pub fn from_extension(path: &std::path::Path) -> Option<Self> {
        let s = path.to_string_lossy();
        let lower = s.to_ascii_lowercase();
        if lower.ends_with(".gz") || lower.ends_with(".tgz") {
            Some(Self::Gzip)
        } else if lower.ends_with(".xz") || lower.ends_with(".txz") {
            Some(Self::Xz)
        } else if lower.ends_with(".lzma") {
            Some(Self::Lzma)
        } else if lower.ends_with(".lz4") {
            Some(Self::Lz4)
        } else if lower.ends_with(".zst") || lower.ends_with(".zstd") {
            Some(Self::Zstd)
        } else if lower.ends_with(".lzo") {
            Some(Self::Lzo)
        } else {
            None
        }
    }
}

/// Sniff the first few bytes of a stream and decide whether it's
/// compressed and with which algorithm. Returns `None` for plain (no
/// magic match) input. Caller must supply at least 6 bytes for the
/// detection to cover every supported codec.
pub fn detect_magic(prefix: &[u8]) -> Option<Algo> {
    if prefix.len() >= 2 && prefix[0] == 0x1F && prefix[1] == 0x8B {
        return Some(Algo::Gzip);
    }
    if prefix.len() >= 6 && &prefix[0..6] == b"\xfd7zXZ\x00" {
        return Some(Algo::Xz);
    }
    if prefix.len() >= 4 && &prefix[0..4] == b"\x28\xb5\x2f\xfd" {
        return Some(Algo::Zstd);
    }
    if prefix.len() >= 4 && &prefix[0..4] == b"\x04\x22\x4d\x18" {
        return Some(Algo::Lz4);
    }
    // Legacy LZMA1 stream: 0x5D 0x00 0x00 then a little-endian dict
    // size. Heuristic, not strict — adjust if a false positive shows up.
    if prefix.len() >= 3 && prefix[0] == 0x5D && prefix[1] == 0x00 && prefix[2] == 0x00 {
        return Some(Algo::Lzma);
    }
    // LZO has no standard framing magic. We can't auto-detect it; the
    // caller must declare the algorithm out-of-band.
    None
}

/// One-shot block decompression. `compressed` is the encoded payload;
/// `max_out` is the largest output we're willing to allocate (so a
/// corrupt header can't pin gigabytes of RAM). On success returns the
/// decoded bytes.
///
/// Caveat for LZO: minilzo's underlying API treats this argument as the
/// *exact* expected output size, not a soft maximum — pass the size you
/// read from the source's metadata, not a generous cap.
pub fn decompress(algo: Algo, compressed: &[u8], max_out: usize) -> Result<Vec<u8>> {
    match algo {
        Algo::Gzip => decompress_gzip(compressed, max_out),
        Algo::Zlib => decompress_zlib(compressed, max_out),
        Algo::Xz => decompress_xz(compressed, max_out),
        Algo::Lzma => decompress_lzma(compressed, max_out),
        Algo::Lz4 => decompress_lz4(compressed, max_out),
        Algo::Zstd => decompress_zstd(compressed, max_out),
        Algo::Lzo => decompress_lzo(compressed, max_out),
    }
}

/// One-shot block compression. Mostly useful for tests + small fixtures
/// — large producers should reach for [`make_writer`] instead.
pub fn compress(algo: Algo, plain: &[u8]) -> Result<Vec<u8>> {
    match algo {
        Algo::Gzip => compress_gzip(plain),
        Algo::Zlib => compress_zlib(plain),
        Algo::Xz => compress_xz(plain),
        Algo::Lzma => compress_lzma(plain),
        Algo::Lz4 => compress_lz4(plain),
        Algo::Zstd => compress_zstd(plain),
        Algo::Lzo => compress_lzo(plain),
    }
}

/// Wrap a `Read` in an algorithm-specific decompressor.
pub fn make_reader<'a, R: std::io::Read + 'a>(
    algo: Algo,
    reader: R,
) -> Result<Box<dyn std::io::Read + 'a>> {
    match algo {
        Algo::Gzip => make_reader_gzip(reader),
        Algo::Zlib => make_reader_zlib(reader),
        Algo::Xz => make_reader_xz(reader),
        Algo::Lzma => make_reader_lzma(reader),
        Algo::Lz4 => make_reader_lz4(reader),
        Algo::Zstd => make_reader_zstd(reader),
        Algo::Lzo => make_reader_lzo(reader),
    }
}

/// Wrap a `Write` in an algorithm-specific compressor. The returned
/// writer **must** be `flush`-ed before being dropped if the codec
/// produces a trailer (gzip, zstd, etc.); the streaming-tar caller
/// handles this.
pub fn make_writer<'a, W: std::io::Write + 'a>(
    algo: Algo,
    writer: W,
) -> Result<Box<dyn std::io::Write + 'a>> {
    match algo {
        Algo::Gzip => make_writer_gzip(writer),
        Algo::Zlib => make_writer_zlib(writer),
        Algo::Xz => make_writer_xz(writer),
        Algo::Lzma => make_writer_lzma(writer),
        Algo::Lz4 => make_writer_lz4(writer),
        Algo::Zstd => make_writer_zstd(writer),
        Algo::Lzo => make_writer_lzo(writer),
    }
}

#[allow(dead_code)]
fn disabled(algo: Algo) -> crate::Error {
    crate::Error::Unsupported(format!(
        "{} support is disabled — rebuild fstool with `--features {}`",
        algo.name(),
        algo.name()
    ))
}

// Used by the still-slice-based codecs (lzma / lz4 / lzo); the
// compcol-backed arms cap via `LimitedDecoder` instead.
#[cfg(any(feature = "lzma", feature = "lz4", feature = "lzo"))]
fn cap_check(buf: &[u8], max_out: usize) -> Result<()> {
    if buf.len() > max_out {
        return Err(crate::Error::InvalidImage(format!(
            "compression: decoded payload {} > cap {max_out}",
            buf.len()
        )));
    }
    Ok(())
}

// ─────────────────────── compcol-backed codecs ───────────────────────
//
// gzip / zlib / xz / lzma / zstd are served by the `compcol` crate behind
// its uniform `Encoder`/`Decoder` traits. These four generic helpers
// adapt compcol to fstool's block + stream API; each per-algorithm arm
// below just instantiates them with the right `compcol::*` marker type.

/// One-shot decompress with a hard output cap (decompression-bomb guard
/// via [`compcol::limit::LimitedDecoder`], which aborts mid-stream rather
/// than allocating the whole payload first).
#[cfg(any(feature = "gzip", feature = "xz", feature = "lzma", feature = "zstd"))]
fn cc_decompress<A: compcol::Algorithm>(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let dec = compcol::limit::LimitedDecoder::new(A::decoder(), max_out as u64);
    let mut rdr = compcol::io::DecoderReader::new(src, dec);
    let mut out = Vec::new();
    rdr.read_to_end(&mut out)
        .map_err(|e| crate::Error::InvalidImage(format!("{} decode failed: {e}", A::NAME)))?;
    Ok(out)
}

/// One-shot compress with `A`'s default encoder config.
#[cfg(any(feature = "gzip", feature = "xz", feature = "lzma", feature = "zstd"))]
fn cc_compress<A: compcol::Algorithm>(plain: &[u8]) -> Result<Vec<u8>> {
    compcol::vec::compress_to_vec::<A>(plain).map_err(|e| {
        crate::Error::Io(std::io::Error::other(format!(
            "{} encode failed: {e}",
            A::NAME
        )))
    })
}

/// Wrap a `Read` in a streaming decompressor for `A`.
#[cfg(any(feature = "gzip", feature = "xz", feature = "lzma", feature = "zstd"))]
fn cc_reader<'a, A: compcol::Algorithm, R: std::io::Read + 'a>(r: R) -> Box<dyn std::io::Read + 'a>
where
    A::Decoder: 'a,
{
    Box::new(compcol::io::DecoderReader::new(r, A::decoder()))
}

/// Wrap a `Write` in a streaming compressor for `A`. The encoder flushes
/// its trailer when the returned writer is dropped.
#[cfg(any(feature = "gzip", feature = "xz", feature = "lzma", feature = "zstd"))]
fn cc_writer<'a, A: compcol::Algorithm, W: std::io::Write + 'a>(
    w: W,
) -> Box<dyn std::io::Write + 'a>
where
    A::Encoder: 'a,
{
    Box::new(compcol::io::EncoderWriter::new(w, A::encoder()))
}

// =========================== gzip ===========================

#[cfg(feature = "gzip")]
fn decompress_gzip(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    cc_decompress::<compcol::gzip::Gzip>(src, max_out)
}

#[cfg(not(feature = "gzip"))]
fn decompress_gzip(_src: &[u8], _max_out: usize) -> Result<Vec<u8>> {
    Err(disabled(Algo::Gzip))
}

#[cfg(feature = "gzip")]
fn compress_gzip(plain: &[u8]) -> Result<Vec<u8>> {
    cc_compress::<compcol::gzip::Gzip>(plain)
}

#[cfg(not(feature = "gzip"))]
fn compress_gzip(_plain: &[u8]) -> Result<Vec<u8>> {
    Err(disabled(Algo::Gzip))
}

#[cfg(feature = "gzip")]
fn make_reader_gzip<'a, R: std::io::Read + 'a>(r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Ok(cc_reader::<compcol::gzip::Gzip, _>(r))
}

#[cfg(not(feature = "gzip"))]
fn make_reader_gzip<'a, R: std::io::Read + 'a>(_r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Err(disabled(Algo::Gzip))
}

#[cfg(feature = "gzip")]
fn make_writer_gzip<'a, W: std::io::Write + 'a>(w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Ok(cc_writer::<compcol::gzip::Gzip, _>(w))
}

#[cfg(not(feature = "gzip"))]
fn make_writer_gzip<'a, W: std::io::Write + 'a>(_w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Err(disabled(Algo::Gzip))
}

// =========================== zlib ===========================
// Used by SquashFS (compressor id 1 is labeled "gzip" but really uses
// zlib framing per the on-disk spec). Shares the `gzip` Cargo feature.

#[cfg(feature = "gzip")]
fn decompress_zlib(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    cc_decompress::<compcol::zlib::Zlib>(src, max_out)
}

#[cfg(not(feature = "gzip"))]
fn decompress_zlib(_src: &[u8], _max_out: usize) -> Result<Vec<u8>> {
    Err(disabled(Algo::Zlib))
}

#[cfg(feature = "gzip")]
fn compress_zlib(plain: &[u8]) -> Result<Vec<u8>> {
    cc_compress::<compcol::zlib::Zlib>(plain)
}

#[cfg(not(feature = "gzip"))]
fn compress_zlib(_plain: &[u8]) -> Result<Vec<u8>> {
    Err(disabled(Algo::Zlib))
}

#[cfg(feature = "gzip")]
fn make_reader_zlib<'a, R: std::io::Read + 'a>(r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Ok(cc_reader::<compcol::zlib::Zlib, _>(r))
}

#[cfg(not(feature = "gzip"))]
fn make_reader_zlib<'a, R: std::io::Read + 'a>(_r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Err(disabled(Algo::Zlib))
}

#[cfg(feature = "gzip")]
fn make_writer_zlib<'a, W: std::io::Write + 'a>(w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Ok(cc_writer::<compcol::zlib::Zlib, _>(w))
}

#[cfg(not(feature = "gzip"))]
fn make_writer_zlib<'a, W: std::io::Write + 'a>(_w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Err(disabled(Algo::Zlib))
}

// ============================ xz ============================

#[cfg(feature = "xz")]
fn decompress_xz(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    cc_decompress::<compcol::xz::Xz>(src, max_out)
}

#[cfg(not(feature = "xz"))]
fn decompress_xz(_src: &[u8], _max_out: usize) -> Result<Vec<u8>> {
    Err(disabled(Algo::Xz))
}

#[cfg(feature = "xz")]
fn compress_xz(plain: &[u8]) -> Result<Vec<u8>> {
    cc_compress::<compcol::xz::Xz>(plain)
}

#[cfg(not(feature = "xz"))]
fn compress_xz(_plain: &[u8]) -> Result<Vec<u8>> {
    Err(disabled(Algo::Xz))
}

#[cfg(feature = "xz")]
fn make_reader_xz<'a, R: std::io::Read + 'a>(r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Ok(cc_reader::<compcol::xz::Xz, _>(r))
}

#[cfg(not(feature = "xz"))]
fn make_reader_xz<'a, R: std::io::Read + 'a>(_r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Err(disabled(Algo::Xz))
}

#[cfg(feature = "xz")]
fn make_writer_xz<'a, W: std::io::Write + 'a>(w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Ok(cc_writer::<compcol::xz::Xz, _>(w))
}

#[cfg(not(feature = "xz"))]
fn make_writer_xz<'a, W: std::io::Write + 'a>(_w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Err(disabled(Algo::Xz))
}

// =========================== lzma ===========================

// lzma stays on `lzma-rs`: compcol 0.4.0's `.lzma` (alone) codec isn't
// liblzma-interoperable in either direction (KarpelesLab/compcol#14).
// The `.xz` container is on compcol; revisit lzma once #14 lands.

#[cfg(feature = "lzma")]
fn decompress_lzma(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    let mut input = std::io::BufReader::new(src);
    let mut out = Vec::new();
    lzma_rs::lzma_decompress(&mut input, &mut out)
        .map_err(|e| crate::Error::InvalidImage(format!("lzma decode failed: {e}")))?;
    cap_check(&out, max_out)?;
    Ok(out)
}

#[cfg(not(feature = "lzma"))]
fn decompress_lzma(_src: &[u8], _max_out: usize) -> Result<Vec<u8>> {
    Err(disabled(Algo::Lzma))
}

#[cfg(feature = "lzma")]
fn compress_lzma(plain: &[u8]) -> Result<Vec<u8>> {
    let mut input = std::io::BufReader::new(plain);
    let mut out = Vec::new();
    lzma_rs::lzma_compress(&mut input, &mut out)
        .map_err(|e| crate::Error::Io(std::io::Error::other(format!("lzma encode failed: {e}"))))?;
    Ok(out)
}

#[cfg(not(feature = "lzma"))]
fn compress_lzma(_plain: &[u8]) -> Result<Vec<u8>> {
    Err(disabled(Algo::Lzma))
}

#[cfg(feature = "lzma")]
fn make_reader_lzma<'a, R: std::io::Read + 'a>(r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Ok(Box::new(stream::LzmaDecoderAdapter::new(r)))
}

#[cfg(not(feature = "lzma"))]
fn make_reader_lzma<'a, R: std::io::Read + 'a>(_r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Err(disabled(Algo::Lzma))
}

#[cfg(feature = "lzma")]
fn make_writer_lzma<'a, W: std::io::Write + 'a>(w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Ok(Box::new(stream::LzmaEncoderAdapter::new(w)))
}

#[cfg(not(feature = "lzma"))]
fn make_writer_lzma<'a, W: std::io::Write + 'a>(_w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Err(disabled(Algo::Lzma))
}

// =========================== lz4 ============================

#[cfg(feature = "lz4")]
fn decompress_lz4(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    // SquashFS uses raw LZ4 block format (no frame). For tar (frame
    // format), use `make_reader_lz4` instead.
    let out = lz4_flex::block::decompress(src, max_out)
        .map_err(|e| crate::Error::InvalidImage(format!("lz4 decode failed: {e}")))?;
    cap_check(&out, max_out)?;
    Ok(out)
}

#[cfg(not(feature = "lz4"))]
fn decompress_lz4(_src: &[u8], _max_out: usize) -> Result<Vec<u8>> {
    Err(disabled(Algo::Lz4))
}

#[cfg(feature = "lz4")]
fn compress_lz4(plain: &[u8]) -> Result<Vec<u8>> {
    Ok(lz4_flex::block::compress(plain))
}

#[cfg(not(feature = "lz4"))]
fn compress_lz4(_plain: &[u8]) -> Result<Vec<u8>> {
    Err(disabled(Algo::Lz4))
}

#[cfg(feature = "lz4")]
fn make_reader_lz4<'a, R: std::io::Read + 'a>(r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Ok(Box::new(lz4_flex::frame::FrameDecoder::new(r)))
}

#[cfg(not(feature = "lz4"))]
fn make_reader_lz4<'a, R: std::io::Read + 'a>(_r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Err(disabled(Algo::Lz4))
}

#[cfg(feature = "lz4")]
fn make_writer_lz4<'a, W: std::io::Write + 'a>(w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Ok(Box::new(lz4_flex::frame::FrameEncoder::new(w)))
}

#[cfg(not(feature = "lz4"))]
fn make_writer_lz4<'a, W: std::io::Write + 'a>(_w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Err(disabled(Algo::Lz4))
}

// =========================== zstd ===========================

#[cfg(feature = "zstd")]
fn decompress_zstd(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    cc_decompress::<compcol::zstd::Zstd>(src, max_out)
}

#[cfg(not(feature = "zstd"))]
fn decompress_zstd(_src: &[u8], _max_out: usize) -> Result<Vec<u8>> {
    Err(disabled(Algo::Zstd))
}

#[cfg(feature = "zstd")]
fn compress_zstd(plain: &[u8]) -> Result<Vec<u8>> {
    cc_compress::<compcol::zstd::Zstd>(plain)
}

#[cfg(not(feature = "zstd"))]
fn compress_zstd(_plain: &[u8]) -> Result<Vec<u8>> {
    Err(disabled(Algo::Zstd))
}

#[cfg(feature = "zstd")]
fn make_reader_zstd<'a, R: std::io::Read + 'a>(r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Ok(cc_reader::<compcol::zstd::Zstd, _>(r))
}

#[cfg(not(feature = "zstd"))]
fn make_reader_zstd<'a, R: std::io::Read + 'a>(_r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Err(disabled(Algo::Zstd))
}

#[cfg(feature = "zstd")]
fn make_writer_zstd<'a, W: std::io::Write + 'a>(w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Ok(cc_writer::<compcol::zstd::Zstd, _>(w))
}

#[cfg(not(feature = "zstd"))]
fn make_writer_zstd<'a, W: std::io::Write + 'a>(_w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Err(disabled(Algo::Zstd))
}

// =========================== lzo ============================
//
// `minilzo-rs` doesn't have a streaming API — LZO1X is a one-shot codec
// by design. We synthesise streaming as "buffer the whole payload, then
// encode/decode in one go on flush." Fine for SquashFS (≤ 1 MiB blocks);
// the tar streaming path uses a 1 MiB internal frame and emits one LZO
// blob per chunk.

#[cfg(feature = "lzo")]
fn decompress_lzo(src: &[u8], max_out: usize) -> Result<Vec<u8>> {
    let lzo = minilzo_rs::LZO::init()
        .map_err(|e| crate::Error::InvalidImage(format!("lzo init: {e:?}")))?;
    let out = lzo
        .decompress(src, max_out)
        .map_err(|e| crate::Error::InvalidImage(format!("lzo decode failed: {e:?}")))?;
    cap_check(&out, max_out)?;
    Ok(out)
}

#[cfg(not(feature = "lzo"))]
fn decompress_lzo(_src: &[u8], _max_out: usize) -> Result<Vec<u8>> {
    Err(disabled(Algo::Lzo))
}

#[cfg(feature = "lzo")]
fn compress_lzo(plain: &[u8]) -> Result<Vec<u8>> {
    let mut lzo = minilzo_rs::LZO::init()
        .map_err(|e| crate::Error::Io(std::io::Error::other(format!("lzo init: {e:?}"))))?;
    lzo.compress(plain)
        .map_err(|e| crate::Error::Io(std::io::Error::other(format!("lzo encode failed: {e:?}"))))
}

#[cfg(not(feature = "lzo"))]
fn compress_lzo(_plain: &[u8]) -> Result<Vec<u8>> {
    Err(disabled(Algo::Lzo))
}

#[cfg(feature = "lzo")]
fn make_reader_lzo<'a, R: std::io::Read + 'a>(r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Ok(Box::new(stream::LzoFrameReader::new(r)))
}

#[cfg(not(feature = "lzo"))]
fn make_reader_lzo<'a, R: std::io::Read + 'a>(_r: R) -> Result<Box<dyn std::io::Read + 'a>> {
    Err(disabled(Algo::Lzo))
}

#[cfg(feature = "lzo")]
fn make_writer_lzo<'a, W: std::io::Write + 'a>(w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Ok(Box::new(stream::LzoFrameWriter::new(w)))
}

#[cfg(not(feature = "lzo"))]
fn make_writer_lzo<'a, W: std::io::Write + 'a>(_w: W) -> Result<Box<dyn std::io::Write + 'a>> {
    Err(disabled(Algo::Lzo))
}

// ====================== streaming adapters ======================

mod stream;

// ====================== path helpers ======================

/// Inspect `path`'s extension and the first 8 bytes of the file. Return
/// the codec if either says compressed; `None` for a plain file.
pub fn detect_path(path: &std::path::Path) -> Result<Option<Algo>> {
    if let Some(algo) = Algo::from_extension(path) {
        return Ok(Some(algo));
    }
    // No extension hint — sniff the magic bytes.
    if !path.exists() {
        return Ok(None);
    }
    let mut f = std::fs::File::open(path)?;
    let mut head = [0u8; 8];
    use std::io::Read;
    let n = f.read(&mut head)?;
    Ok(detect_magic(&head[..n]))
}

/// Decompress `src` into a new temp file and return both. The temp
/// file's path is returned so callers (e.g. `inspect::with_target_device`)
/// can open it as a `BlockDevice`. The `NamedTempFile` is kept alive by
/// the caller — once it's dropped, the file is deleted.
pub fn decompress_to_tempfile(
    src: &std::path::Path,
    algo: Algo,
) -> Result<tempfile::NamedTempFile> {
    use std::io::Write;
    let f = std::fs::File::open(src)?;
    let mut reader = make_reader(algo, std::io::BufReader::new(f))?;
    let tmp = tempfile::NamedTempFile::new()?;
    {
        let mut out = std::io::BufWriter::new(tmp.as_file());
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
        }
        out.flush()?;
    }
    Ok(tmp)
}

/// Stream-compress `src` (a plain file path) into `dst` (the final
/// output path), applying `algo`. Used to ship a temp tar archive
/// through the codec before writing the final `.tar.<algo>`.
pub fn compress_file_to_file(
    src: &std::path::Path,
    dst: &std::path::Path,
    algo: Algo,
) -> Result<()> {
    use std::io::{Read, Write};
    let f_in = std::fs::File::open(src)?;
    let mut reader = std::io::BufReader::new(f_in);
    let f_out = std::fs::File::create(dst)?;
    let mut writer = make_writer(algo, std::io::BufWriter::new(f_out))?;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_magic_recognises_known_prefixes() {
        assert_eq!(detect_magic(b"\x1f\x8b\x08").unwrap(), Algo::Gzip);
        assert_eq!(detect_magic(b"\xfd7zXZ\x00").unwrap(), Algo::Xz);
        assert_eq!(detect_magic(b"\x28\xb5\x2f\xfd_").unwrap(), Algo::Zstd);
        assert_eq!(detect_magic(b"\x04\x22\x4d\x18_").unwrap(), Algo::Lz4);
        assert_eq!(detect_magic(b"\x5d\x00\x00\x80").unwrap(), Algo::Lzma);
        assert!(detect_magic(b"plain").is_none());
    }

    #[test]
    fn from_extension_picks_codec() {
        let p = std::path::Path::new("/tmp/out.tar.gz");
        assert_eq!(Algo::from_extension(p), Some(Algo::Gzip));
        let p = std::path::Path::new("disk.tar.zst");
        assert_eq!(Algo::from_extension(p), Some(Algo::Zstd));
        let p = std::path::Path::new("plain.tar");
        assert_eq!(Algo::from_extension(p), None);
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn gzip_block_round_trip() {
        let plain: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
        let enc = compress(Algo::Gzip, &plain).unwrap();
        let dec = decompress(Algo::Gzip, &enc, 1 << 16).unwrap();
        assert_eq!(dec, plain);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn lz4_block_round_trip() {
        let plain: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
        let enc = compress(Algo::Lz4, &plain).unwrap();
        let dec = decompress(Algo::Lz4, &enc, 1 << 16).unwrap();
        assert_eq!(dec, plain);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_block_round_trip() {
        let plain: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
        let enc = compress(Algo::Zstd, &plain).unwrap();
        let dec = decompress(Algo::Zstd, &enc, 1 << 16).unwrap();
        assert_eq!(dec, plain);
    }

    #[cfg(feature = "xz")]
    #[test]
    fn xz_block_round_trip() {
        let plain: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
        let enc = compress(Algo::Xz, &plain).unwrap();
        let dec = decompress(Algo::Xz, &enc, 1 << 16).unwrap();
        assert_eq!(dec, plain);
    }

    #[cfg(feature = "lzma")]
    #[test]
    fn lzma_block_round_trip() {
        let plain: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
        let enc = compress(Algo::Lzma, &plain).unwrap();
        let dec = decompress(Algo::Lzma, &enc, 1 << 16).unwrap();
        assert_eq!(dec, plain);
    }

    #[cfg(feature = "lzo")]
    #[test]
    fn lzo_block_round_trip() {
        // minilzo-rs's decompress sizes the output buffer to the caller's
        // hint exactly, not a soft maximum — so callers must know the
        // uncompressed size up front (SquashFS does, from the inode's
        // block-size table). We pass `plain.len()` here.
        let plain: Vec<u8> = (0u8..=255).cycle().take(8192).collect();
        let enc = compress(Algo::Lzo, &plain).unwrap();
        let dec = decompress(Algo::Lzo, &enc, plain.len()).unwrap();
        assert_eq!(dec, plain);
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn gzip_stream_round_trip() {
        use std::io::{Read, Write};
        let plain: Vec<u8> = (0u8..=255).cycle().take(16 * 1024).collect();
        let mut compressed = Vec::new();
        {
            let mut w = make_writer(Algo::Gzip, &mut compressed).unwrap();
            w.write_all(&plain).unwrap();
            w.flush().unwrap();
        }
        let mut r = make_reader(Algo::Gzip, &compressed[..]).unwrap();
        let mut decoded = Vec::new();
        r.read_to_end(&mut decoded).unwrap();
        assert_eq!(decoded, plain);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn zstd_stream_round_trip() {
        use std::io::{Read, Write};
        let plain: Vec<u8> = (0u8..=255).cycle().take(16 * 1024).collect();
        let mut compressed = Vec::new();
        {
            let mut w = make_writer(Algo::Zstd, &mut compressed).unwrap();
            w.write_all(&plain).unwrap();
            w.flush().unwrap();
        }
        let mut r = make_reader(Algo::Zstd, &compressed[..]).unwrap();
        let mut decoded = Vec::new();
        r.read_to_end(&mut decoded).unwrap();
        assert_eq!(decoded, plain);
    }
}
