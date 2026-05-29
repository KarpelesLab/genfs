//! Microsoft Cabinet (`.cab`) container parser.
//!
//! Walks CFHEADER → CFFOLDER[] → CFFILE[] and each folder's CFDATA chain,
//! producing an [`ArchiveIndex`] (the directory tree + file sizes) plus the
//! side tables needed to extract bodies: a [`Folder`] per CFFOLDER (its
//! compression method and the on-disk location of every CFDATA block) and a
//! [`FileSlice`] per file (which folder, and the byte range inside that
//! folder's *decompressed* stream).
//!
//! Reference: [MS-CAB] (Microsoft Cabinet File Format) and libmspack `cabd.c`.

use std::collections::HashMap;

use crate::block::BlockDevice;
use crate::fs::archive::{ArchiveEntry, ArchiveIndex, DataLocator, EntryKind, Method};
use crate::{Error, Result};

/// CAB folder compression methods (the low byte of `typeCompress`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CabMethod {
    None,
    MsZip,
    /// Quantum, with the window size (bits) from the folder header.
    Quantum {
        window_bits: u32,
    },
    /// LZX, with the window size (bits) from the folder header.
    Lzx {
        window_bits: u32,
    },
    /// A method id we recognise the number of but can't decode.
    Unsupported(u16),
}

/// One CFDATA block: where its (compressed) payload lives and its length.
#[derive(Debug, Clone, Copy)]
pub struct CfData {
    /// Absolute device offset of the `cbData` payload bytes.
    pub offset: u64,
    /// Compressed byte count (payload length on disk).
    pub comp_len: u32,
}

/// A CFFOLDER: its compression method and CFDATA block chain.
#[derive(Debug, Clone)]
pub struct Folder {
    pub method: CabMethod,
    pub blocks: Vec<CfData>,
    /// Σ of every block's `uncomp_len` — the folder's decompressed length.
    pub total_uncomp: u64,
}

/// Where a file's bytes live: a slice of folder `folder`'s decompressed
/// stream.
#[derive(Debug, Clone, Copy)]
pub struct FileSlice {
    pub folder: usize,
    pub uncomp_offset: u64,
    pub len: u64,
}

/// Result of scanning a cabinet.
pub struct Parsed {
    pub index: ArchiveIndex,
    pub folders: Vec<Folder>,
    /// Every regular file's normalised path → its extractable slice, or
    /// `None` when the file can't be extracted from this cabinet alone
    /// (spans cabinets, or references a missing folder).
    pub files: HashMap<String, Option<FileSlice>>,
}

// CFHEADER flags.
const FLAG_PREV_CABINET: u16 = 0x0001;
const FLAG_NEXT_CABINET: u16 = 0x0002;
const FLAG_RESERVE_PRESENT: u16 = 0x0004;

// CFFILE attribs.
const ATTR_RDONLY: u16 = 0x01;
const ATTR_NAME_IS_UTF: u16 = 0x80;

// Special iFolder values for files that span cabinets.
const IFOLD_CONTINUED_FROM_PREV: u16 = 0xFFFD;
const IFOLD_CONTINUED_TO_NEXT: u16 = 0xFFFE;
const IFOLD_CONTINUED_PREV_AND_NEXT: u16 = 0xFFFF;

fn read_exact_at(dev: &mut dyn BlockDevice, off: u64, len: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    dev.read_at(off, &mut buf)?;
    Ok(buf)
}

#[inline]
fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn corrupt(msg: impl Into<String>) -> Error {
    Error::InvalidImage(format!("cab: {}", msg.into()))
}

/// Parse the cabinet on `dev` into an [`ArchiveIndex`] plus the folder /
/// file side tables.
pub fn scan(dev: &mut dyn BlockDevice) -> Result<Parsed> {
    // --- CFHEADER (36-byte base) ---
    let hdr = read_exact_at(dev, 0, 36)?;
    if &hdr[0..4] != b"MSCF" {
        return Err(corrupt("bad signature (expected MSCF)"));
    }
    let coff_files = le32(&hdr, 16);
    let c_folders = le16(&hdr, 26);
    let c_files = le16(&hdr, 28);
    let flags = le16(&hdr, 30);

    if flags & (FLAG_PREV_CABINET | FLAG_NEXT_CABINET) != 0 {
        return Err(Error::Unsupported(
            "cab: spanned / multi-cabinet sets are not supported".into(),
        ));
    }

    // Optional reserve sizes (RESERVE_PRESENT): cbCFHeader u16, cbCFFolder
    // u8, cbCFData u8, then cbCFHeader bytes of per-cabinet reserve.
    let mut cursor: u64 = 36;
    let (cb_cffolder, cb_cfdata) = if flags & FLAG_RESERVE_PRESENT != 0 {
        let r = read_exact_at(dev, cursor, 4)?;
        let cb_cfheader = le16(&r, 0);
        let cb_cffolder = r[2] as u64;
        let cb_cfdata = r[3] as u64;
        cursor += 4 + cb_cfheader as u64;
        (cb_cffolder, cb_cfdata)
    } else {
        (0, 0)
    };

    // --- CFFOLDER[c_folders] ---
    let mut folders = Vec::with_capacity(c_folders as usize);
    for _ in 0..c_folders {
        let f = read_exact_at(dev, cursor, 8)?;
        let coff_cab_start = le32(&f, 0) as u64;
        let c_cfdata = le16(&f, 4);
        let type_compress = le16(&f, 6);
        cursor += 8 + cb_cffolder;

        let method = decode_method(type_compress);
        let (blocks, total_uncomp) = walk_cfdata_chain(dev, coff_cab_start, c_cfdata, cb_cfdata)?;
        folders.push(Folder {
            method,
            blocks,
            total_uncomp,
        });
    }

    // --- CFFILE[c_files] at coff_files ---
    let mut index = ArchiveIndex::new("cab");
    let mut files = HashMap::new();
    let mut fpos = coff_files as u64;
    for _ in 0..c_files {
        let fixed = read_exact_at(dev, fpos, 16)?;
        let cb_file = le32(&fixed, 0);
        let uoff_folder_start = le32(&fixed, 4);
        let i_folder = le16(&fixed, 8);
        let date = le16(&fixed, 10);
        let time = le16(&fixed, 12);
        let attribs = le16(&fixed, 14);

        // szName: NUL-terminated, in a window after the fixed part.
        let name_bytes = read_name(dev, fpos + 16)?;
        let name_len = name_bytes.len();
        fpos += 16 + name_len as u64 + 1;

        // Files that continue from/to another cabinet can't be extracted
        // from this one alone — record them as zero-length unsupported.
        let spanned = matches!(
            i_folder,
            IFOLD_CONTINUED_FROM_PREV | IFOLD_CONTINUED_TO_NEXT | IFOLD_CONTINUED_PREV_AND_NEXT
        );

        let name = decode_name(&name_bytes, attribs & ATTR_NAME_IS_UTF != 0);
        let path = crate::fs::archive::tree::normalise_path(&normalise_cab_name(&name));
        if path == "/" {
            continue;
        }

        let mode: u16 = if attribs & ATTR_RDONLY != 0 {
            0o444
        } else {
            0o644
        };
        let mut entry = ArchiveEntry::regular(
            path.clone(),
            // Locator carries only the *size* for getattr/list; the CAB
            // read path is custom (folder decode + slice), so offset /
            // method here are never used for I/O.
            DataLocator {
                offset: 0,
                compressed_len: 0,
                uncompressed_len: cb_file as u64,
                method: Method::Stored,
            },
        );
        entry.kind = EntryKind::Regular;
        entry.mode = mode;
        entry.mtime = crate::fs::archive::zip::dos_to_unix(date, time);
        index.push(entry);

        // Record the extraction slice — or `None` for files that span
        // cabinets / reference a missing folder, so reads of them return a
        // clean Unsupported rather than empty bytes.
        let slice = if spanned || (i_folder as usize) >= folders.len() {
            None
        } else {
            Some(FileSlice {
                folder: i_folder as usize,
                uncomp_offset: uoff_folder_start as u64,
                len: cb_file as u64,
            })
        };
        files.insert(path, slice);
    }

    Ok(Parsed {
        index,
        folders,
        files,
    })
}

/// Decode `typeCompress` into a [`CabMethod`]. Window bits for LZX/Quantum
/// live in bits 8..=12.
fn decode_method(type_compress: u16) -> CabMethod {
    let window_bits = ((type_compress >> 8) & 0x1f) as u32;
    match type_compress & 0x0f {
        0 => CabMethod::None,
        1 => CabMethod::MsZip,
        2 => CabMethod::Quantum { window_bits },
        3 => CabMethod::Lzx { window_bits },
        other => CabMethod::Unsupported(other),
    }
}

/// Walk a folder's `count` CFDATA blocks starting at `start`, recording each
/// block's payload offset + sizes. Returns the blocks and the folder's total
/// uncompressed length.
fn walk_cfdata_chain(
    dev: &mut dyn BlockDevice,
    start: u64,
    count: u16,
    cb_cfdata_reserve: u64,
) -> Result<(Vec<CfData>, u64)> {
    let mut blocks = Vec::with_capacity(count as usize);
    let mut total: u64 = 0;
    let mut pos = start;
    for _ in 0..count {
        // CFDATA header: csum u32, cbData u16, cbUncomp u16, then reserve.
        let h = read_exact_at(dev, pos, 8)?;
        let comp_len = le16(&h, 4) as u32;
        let uncomp_len = le16(&h, 6) as u32;
        let payload = pos + 8 + cb_cfdata_reserve;
        blocks.push(CfData {
            offset: payload,
            comp_len,
        });
        total += uncomp_len as u64;
        pos = payload + comp_len as u64;
    }
    Ok((blocks, total))
}

/// Read a NUL-terminated name starting at `off`, scanning a bounded window.
fn read_name(dev: &mut dyn BlockDevice, off: u64) -> Result<Vec<u8>> {
    // CAB names are ≤ 256 bytes; read a generous window and find the NUL.
    // Clamp the read to the device so a name near EOF doesn't over-read.
    let dev_len = dev.total_size();
    let want = 512u64.min(dev_len.saturating_sub(off)).max(1) as usize;
    let buf = read_exact_at(dev, off, want)?;
    match buf.iter().position(|&b| b == 0) {
        Some(n) => Ok(buf[..n].to_vec()),
        None => Err(corrupt("CFFILE name not NUL-terminated within 512 bytes")),
    }
}

/// Decode a CFFILE name. UTF-8 when the attrib bit is set; otherwise treat
/// the bytes as CP-1252-ish (Latin-1 superset) — good enough for the ASCII
/// names CAB tooling emits, and lossless for the high range.
fn decode_name(bytes: &[u8], utf8: bool) -> String {
    if utf8 {
        String::from_utf8_lossy(bytes).into_owned()
    } else {
        bytes.iter().map(|&b| b as char).collect()
    }
}

/// CAB paths use `\` separators and are relative; produce a normalised
/// absolute slash path. Drops `.`/`..` and empty components defensively.
fn normalise_cab_name(name: &str) -> String {
    let mut out = String::new();
    for comp in name.split(['\\', '/']) {
        if comp.is_empty() || comp == "." || comp == ".." {
            continue;
        }
        out.push('/');
        out.push_str(comp);
    }
    out
}
