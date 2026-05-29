//! CAB folder decompression via `compcol`.
//!
//! A CAB folder is a single logical stream split across CFDATA blocks; this
//! decodes the whole folder into one `Vec<u8>` that the reader then slices
//! per file. Method support:
//!
//! - **None**  — concatenate the raw CFDATA payloads.
//! - **LZX**   — concatenate payloads into the LZX bitstream and decode via
//!   `compcol::lzx`, prepending compcol's 5-byte framing (window bits +
//!   total uncompressed length).
//! - **Quantum** — concatenate payloads and decode via `compcol::quantum`
//!   with the folder's window size.
//! - **MSZIP** — each CFDATA block is its own `CK` + raw-DEFLATE stream;
//!   decoded block-by-block, seeding each with the previous block's last
//!   32 KiB as the deflate preset dictionary (compcol 0.4.3) so cross-block
//!   back-references resolve.

use compcol::{Decoder, Status};

use super::scan::{CabMethod, Folder};
use crate::block::BlockDevice;
use crate::{Error, Result};

fn read_payloads(dev: &mut dyn BlockDevice, folder: &Folder) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(folder.blocks.iter().map(|b| b.comp_len as usize).sum());
    for b in &folder.blocks {
        let mut buf = vec![0u8; b.comp_len as usize];
        dev.read_at(b.offset, &mut buf)?;
        out.extend_from_slice(&buf);
    }
    Ok(out)
}

/// Decode `folder` into its full decompressed byte stream.
pub fn decode_folder(dev: &mut dyn BlockDevice, folder: &Folder) -> Result<Vec<u8>> {
    let total = folder.total_uncomp;
    match folder.method {
        CabMethod::None => read_payloads(dev, folder),

        CabMethod::Lzx { window_bits } => {
            let bitstream = read_payloads(dev, folder)?;
            let total_u32 = u32::try_from(total)
                .map_err(|_| Error::Unsupported("cab: LZX folder larger than 4 GiB".into()))?;
            // compcol's standalone LZX framing: window_bits, LE u32 total
            // uncompressed length, then the raw bitstream.
            let mut framed = Vec::with_capacity(bitstream.len() + 5);
            framed.push(window_bits as u8);
            framed.extend_from_slice(&total_u32.to_le_bytes());
            framed.extend_from_slice(&bitstream);
            compcol::vec::decompress_to_vec::<compcol::lzx::Lzx>(&framed)
                .map_err(|e| Error::InvalidImage(format!("cab: LZX decode failed: {e}")))
        }

        CabMethod::Quantum { window_bits } => {
            let bitstream = read_payloads(dev, folder)?;
            let mut dec = compcol::quantum::Decoder::with_window_bits(window_bits)
                .map_err(|e| Error::InvalidImage(format!("cab: bad Quantum window: {e}")))?;
            decode_stream(&mut dec, &bitstream, total, "Quantum")
        }

        CabMethod::MsZip => {
            // Each CFDATA block is its own `CK` + raw-DEFLATE stream, but
            // back-references may reach into the previous block's output, so
            // each block is decoded with the prior block's last 32 KiB as the
            // deflate preset dictionary (compcol 0.4.3, #22).
            const MSZIP_WINDOW: usize = 32 * 1024;
            let mut out: Vec<u8> = Vec::with_capacity(total as usize);
            for blk in &folder.blocks {
                let mut payload = vec![0u8; blk.comp_len as usize];
                dev.read_at(blk.offset, &mut payload)?;
                if payload.len() < 2 || &payload[0..2] != b"CK" {
                    return Err(Error::InvalidImage(
                        "cab: MSZIP block missing 'CK' signature".into(),
                    ));
                }
                let dict_start = out.len().saturating_sub(MSZIP_WINDOW);
                let dictionary = out[dict_start..].to_vec();
                let mut dec =
                    compcol::deflate::Decoder::with_config(compcol::deflate::DecoderConfig {
                        dictionary,
                    });
                let block_out =
                    decode_stream(&mut dec, &payload[2..], MSZIP_WINDOW as u64, "MSZIP")?;
                out.extend_from_slice(&block_out);
            }
            Ok(out)
        }

        CabMethod::Unsupported(id) => Err(Error::Unsupported(format!(
            "cab: compression method {id} not supported"
        ))),
    }
}

/// Drive a `Decoder` over the whole `input` slice, collecting up to `cap`
/// bytes. Mirrors `compression::cc_decompress` but for a pre-built decoder.
fn decode_stream(dec: &mut dyn Decoder, input: &[u8], cap: u64, label: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(cap as usize);
    let mut scratch = vec![0u8; 64 * 1024];
    let mut consumed = 0usize;
    let err = |e| Error::InvalidImage(format!("cab: {label} decode failed: {e}"));
    loop {
        let (p, status) = dec.decode(&input[consumed..], &mut scratch).map_err(err)?;
        out.extend_from_slice(&scratch[..p.written]);
        consumed += p.consumed;
        match status {
            Status::StreamEnd => return Ok(out),
            Status::OutputFull => continue,
            Status::InputEmpty => break,
        }
    }
    loop {
        let (p, status) = dec.finish(&mut scratch).map_err(err)?;
        out.extend_from_slice(&scratch[..p.written]);
        if matches!(status, Status::StreamEnd) {
            break;
        }
        if p.written == 0 {
            return Err(Error::InvalidImage(format!(
                "cab: {label} decode failed: truncated stream"
            )));
        }
    }
    Ok(out)
}
