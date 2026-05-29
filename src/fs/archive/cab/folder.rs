//! Streaming CAB folder decompression via `compcol`.
//!
//! A CAB folder is one logical stream split across CFDATA blocks; a file is
//! a byte slice of that decompressed stream. To keep memory bounded even for
//! folders that decompress to many gigabytes, this exposes the folder as a
//! lazy [`Read`] — the caller skips to the file's offset and `take`s its
//! length — rather than materialising the whole folder.
//!
//! Method support:
//! - **None** — the CFDATA payloads concatenated, read block by block.
//! - **LZX** — the payloads framed with compcol's 5-byte header (window bits
//!   + total length) and run through `compcol::lzx` via `compcol::io`.
//! - **Quantum** — payloads through `compcol::quantum` with the folder's
//!   window size.
//! - **MSZIP** — each CFDATA block is its own `CK` + raw-DEFLATE stream;
//!   decoded block-by-block with `reset_keep_window` between blocks so the
//!   32 KiB cross-block history carries over (compcol 0.4.3).

use std::io::{self, Read};

use compcol::{Algorithm, Decoder, Status};

use super::scan::{CabMethod, CfData, Folder};
use crate::block::BlockDevice;
use crate::{Error, Result};

/// Build a lazy reader over `folder`'s decompressed byte stream. Memory is
/// bounded to the codec's working set (≤ ~a block + window), never the whole
/// folder.
pub fn decode_folder_reader<'a>(
    dev: &'a mut dyn BlockDevice,
    folder: &Folder,
) -> Result<Box<dyn Read + 'a>> {
    let blocks = folder.blocks.clone();
    match folder.method {
        CabMethod::None => Ok(Box::new(PayloadReader::new(dev, blocks))),

        CabMethod::MsZip => Ok(Box::new(MsZipReader {
            payloads: PayloadReader::new(dev, blocks),
            dec: compcol::deflate::Decoder::new(),
            started: false,
            out: Vec::new(),
            pos: 0,
        })),

        CabMethod::Lzx { window_bits } => {
            let total = u32::try_from(folder.total_uncomp)
                .map_err(|_| Error::Unsupported("cab: LZX folder larger than 4 GiB".into()))?;
            // compcol's standalone LZX framing prepended to the bitstream.
            let mut header = Vec::with_capacity(5);
            header.push(window_bits as u8);
            header.extend_from_slice(&total.to_le_bytes());
            let framed = io::Cursor::new(header).chain(PayloadReader::new(dev, blocks));
            Ok(Box::new(compcol::io::DecoderReader::new(
                framed,
                compcol::lzx::Lzx::decoder(),
            )))
        }

        CabMethod::Quantum { window_bits } => {
            let dec = compcol::quantum::Decoder::with_window_bits(window_bits)
                .map_err(|e| Error::InvalidImage(format!("cab: bad Quantum window: {e}")))?;
            Ok(Box::new(compcol::io::DecoderReader::new(
                PayloadReader::new(dev, blocks),
                dec,
            )))
        }

        CabMethod::Unsupported(id) => Err(Error::Unsupported(format!(
            "cab: compression method {id} not supported"
        ))),
    }
}

/// Read and discard exactly `n` bytes from `r` (used to skip a file's
/// folder-relative offset without buffering the prefix).
pub fn skip_exact(r: &mut dyn Read, mut n: u64) -> Result<()> {
    let mut scratch = [0u8; 64 * 1024];
    while n > 0 {
        let want = n.min(scratch.len() as u64) as usize;
        let got = r.read(&mut scratch[..want]).map_err(crate::Error::from)?;
        if got == 0 {
            return Err(Error::InvalidImage(
                "cab: folder stream ended before the file offset".into(),
            ));
        }
        n -= got as u64;
    }
    Ok(())
}

/// `Read` over a folder's CFDATA payloads concatenated, fetched one block at
/// a time from the device.
struct PayloadReader<'a> {
    dev: &'a mut dyn BlockDevice,
    blocks: Vec<CfData>,
    idx: usize,
    buf: Vec<u8>,
    pos: usize,
}

impl<'a> PayloadReader<'a> {
    fn new(dev: &'a mut dyn BlockDevice, blocks: Vec<CfData>) -> Self {
        Self {
            dev,
            blocks,
            idx: 0,
            buf: Vec::new(),
            pos: 0,
        }
    }
}

impl Read for PayloadReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        while self.pos >= self.buf.len() {
            if self.idx >= self.blocks.len() {
                return Ok(0);
            }
            let b = self.blocks[self.idx];
            self.idx += 1;
            self.buf = vec![0u8; b.comp_len as usize];
            self.dev
                .read_at(b.offset, &mut self.buf)
                .map_err(io::Error::other)?;
            self.pos = 0;
        }
        let n = (self.buf.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Streaming MSZIP decoder: one deflate `Decoder` reused across CFDATA
/// blocks via `reset_keep_window`, so block N+1's back-references resolve
/// against block N's last 32 KiB. Holds at most one decoded block in memory.
struct MsZipReader<'a> {
    payloads: PayloadReader<'a>,
    dec: compcol::deflate::Decoder,
    started: bool,
    out: Vec<u8>,
    pos: usize,
}

impl MsZipReader<'_> {
    /// Pull the next CFDATA block's compressed payload (raw, with its `CK`
    /// prefix) using the inner [`PayloadReader`]'s block boundaries.
    fn next_block(&mut self) -> io::Result<Option<Vec<u8>>> {
        let p = &mut self.payloads;
        if p.idx >= p.blocks.len() {
            return Ok(None);
        }
        let b = p.blocks[p.idx];
        p.idx += 1;
        let mut payload = vec![0u8; b.comp_len as usize];
        p.dev
            .read_at(b.offset, &mut payload)
            .map_err(io::Error::other)?;
        Ok(Some(payload))
    }
}

impl Read for MsZipReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        while self.pos >= self.out.len() {
            let Some(payload) = self.next_block()? else {
                return Ok(0);
            };
            if payload.len() < 2 || &payload[0..2] != b"CK" {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "cab: MSZIP block missing 'CK' signature",
                ));
            }
            if self.started {
                self.dec.reset_keep_window();
            }
            self.started = true;
            self.out.clear();
            self.pos = 0;
            decode_deflate_block(&mut self.dec, &payload[2..], &mut self.out)?;
        }
        let n = (self.out.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.out[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Decode one complete raw-DEFLATE stream into `out` (one MSZIP block).
fn decode_deflate_block(
    dec: &mut compcol::deflate::Decoder,
    input: &[u8],
    out: &mut Vec<u8>,
) -> io::Result<()> {
    let mut scratch = vec![0u8; 64 * 1024];
    let mut consumed = 0usize;
    let err = |e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cab: MSZIP decode: {e}"),
        )
    };
    loop {
        let (p, status) = dec.decode(&input[consumed..], &mut scratch).map_err(err)?;
        out.extend_from_slice(&scratch[..p.written]);
        consumed += p.consumed;
        match status {
            Status::StreamEnd => return Ok(()),
            Status::OutputFull => continue,
            Status::InputEmpty => break,
        }
    }
    loop {
        let (p, status) = dec.finish(&mut scratch).map_err(err)?;
        out.extend_from_slice(&scratch[..p.written]);
        if matches!(status, Status::StreamEnd) || p.written == 0 {
            break;
        }
    }
    Ok(())
}
