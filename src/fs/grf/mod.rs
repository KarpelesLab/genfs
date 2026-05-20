//! GRF (Gravity Ragnarok File) — Korean MMO archive format.
//!
//! GRF is the on-disk archive format used by *Ragnarok Online*'s
//! game client to ship art, maps, scripts, and sounds. The original
//! libgrf implementation (~2003, by the user) is the reference; the
//! port here lives under the MIT-licensed fstool crate with the
//! rights holder's permission.
//!
//! Three versions are in the wild:
//!
//! - **`0x102` / `0x103`**: file table is RAW (not zlib) and each
//!   filename inside it is encrypted with a fixed-key permutation
//!   cipher ([`crypt::decode_filename`]). The on-disk `len` /
//!   `len_aligned` fields carry magic offsets that the v0x102 decoder
//!   in the `table` module strips. File bodies can also be encrypted
//!   per-entry via the `MIXCRYPT` / `DES` flags.
//! - **`0x200`**: file table is zlib-compressed but filenames are
//!   plain CP949. Magic offsets removed. This is what the writer
//!   produces today.
//!
//! Filenames stored on disk are CP949 (Korean MS codepage); the
//! [`crate::fs::Filesystem`] surface exposes UTF-8 strings, with
//! conversion happening once at parse / write time
//! (see [`encoding`]).
//!
//! Layout on disk:
//!
//! ```text
//! offset 0     : 46-byte header
//! offset 46    : file data blocks (each compressed, optionally encrypted)
//! offset N     : file table
//! ```
//!
//! `header.table_offset` is relative to the end of the 46-byte
//! header, so the absolute file position is
//! `header.table_offset + HEADER_SIZE`. Removing files marks their
//! data wasted; flush rewrites the table; repacking compacts.
//! See [`writer`].

pub mod crypt;
pub mod encoding;
pub mod header;
pub mod table;
pub mod writer;

pub use table::{Entry, GRF_FLAG_DES, GRF_FLAG_FILE, GRF_FLAG_MIXCRYPT};

use std::collections::BTreeMap;
use std::io::Read;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::{FileMeta, FileSource, MutationCapability};

pub(crate) const HEADER_SIZE: usize = 0x2e;

/// Public format-side options for
/// [`crate::fs::FilesystemFactory::format`]. The writer always emits
/// version 0x200 today; older versions are readable but not writeable
/// (they're rarely useful outside legacy game clients).
#[derive(Debug, Clone)]
pub struct FormatOpts {
    /// GRF version word to write. Only 0x200 is supported by the
    /// writer right now.
    pub version: u32,
    /// zlib compression level (0..=9). 0 = store, 6 = default.
    pub compression_level: u32,
}

impl Default for FormatOpts {
    fn default() -> Self {
        Self {
            version: 0x200,
            compression_level: 6,
        }
    }
}

/// An opened GRF archive.
pub struct Grf {
    pub version: u32,
    pub table_offset: u32,
    pub seed: u32,
    pub encrypted_header: bool,
    /// Entries keyed by their normalised path (`/` prefix stripped).
    /// On-disk filenames are CP949; this map's keys are UTF-8.
    pub entries: BTreeMap<String, Entry>,
    /// First byte past the last file's data — where new data appends
    /// and where the table will land at flush time.
    data_end: u64,
    /// Bytes inside the data area that no longer back any entry
    /// (accumulated when files are removed). Drives the repack
    /// decision.
    wasted_space: u64,
    /// True if the in-memory state diverges from disk; flush rewrites
    /// the table + header.
    dirty: bool,
    /// `false` until [`Self::format`] or [`Self::open`] finishes. Set
    /// to mark the handle as a fresh writer (no existing file data
    /// to preserve).
    fresh: bool,
}

impl Grf {
    /// Build a `Grf` handle that represents a freshly-formatted empty
    /// archive. The header isn't written until
    /// [`<Self as crate::fs::Filesystem>::flush`](crate::fs::Filesystem::flush).
    pub fn format_with(_dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<Self> {
        if opts.version != 0x200 {
            return Err(crate::Error::Unsupported(format!(
                "grf: writer only emits v0x200 (asked for {:#x})",
                opts.version
            )));
        }
        Ok(Self {
            version: opts.version,
            table_offset: 0,
            seed: 0,
            encrypted_header: false,
            entries: BTreeMap::new(),
            data_end: HEADER_SIZE as u64,
            wasted_space: 0,
            dirty: true,
            fresh: true,
        })
    }

    /// Open an existing GRF on `dev`. Parses the header + file table
    /// fully into memory.
    pub fn open_dev(dev: &mut dyn BlockDevice) -> Result<Self> {
        let mut head_buf = [0u8; HEADER_SIZE];
        dev.read_at(0, &mut head_buf)?;
        let head = header::Header::decode(&head_buf)?;

        let table_abs = head.table_offset as u64 + HEADER_SIZE as u64;
        let entries = read_table(dev, table_abs, head.version, head.filecount)?;

        // data_end = the maximum (pos + len_aligned) across all
        // entries, anchored at HEADER_SIZE so an empty archive lays
        // its first file directly after the header.
        let mut data_end = HEADER_SIZE as u64;
        for e in entries.values() {
            let end = HEADER_SIZE as u64 + e.pos as u64 + e.len_aligned as u64;
            if end > data_end {
                data_end = end;
            }
        }

        // Wasted space: the table starts at `table_abs` and runs to
        // the end of the file. If there's a gap between data_end and
        // table_abs, that gap is wasted (left over from removed
        // files in a previous lifetime of this archive).
        let wasted_space = table_abs.saturating_sub(data_end);

        Ok(Self {
            version: head.version,
            table_offset: head.table_offset,
            seed: head.seed,
            encrypted_header: head.encrypted_header,
            entries,
            data_end,
            wasted_space,
            dirty: false,
            fresh: false,
        })
    }

    /// Read the body of `entry` into a freshly-allocated buffer.
    /// Handles per-file MIXCRYPT/DES decryption and zlib inflation.
    pub fn read_entry(&self, dev: &mut dyn BlockDevice, entry: &Entry) -> Result<Vec<u8>> {
        let abs = HEADER_SIZE as u64 + entry.pos as u64;
        let mut comp = vec![0u8; entry.len_aligned as usize];
        if entry.len_aligned > 0 {
            dev.read_at(abs, &mut comp)?;
        }
        if let Some(cycle) = entry.crypto_cycle() {
            // flag_type is 0 for MIXCRYPT, 1 for DES — see grf.c
            // decode_des_etc(..., (cycle==0), cycle).
            let flag_type = if cycle == 0 { 1 } else { 0 };
            crypt::decode_des_etc(&mut comp, flag_type, cycle);
        }
        let plain = crate::compression::decompress(
            crate::compression::Algo::Zlib,
            &comp[..entry.len as usize],
            entry.size as usize,
        )?;
        Ok(plain)
    }

    /// Total wasted bytes inside the data area. A nonzero value
    /// means a repack would shrink the archive.
    pub fn wasted_space(&self) -> u64 {
        self.wasted_space
    }
}

fn read_table(
    dev: &mut dyn BlockDevice,
    table_abs: u64,
    version: u32,
    filecount: u32,
) -> Result<BTreeMap<String, Entry>> {
    if filecount == 0 {
        return Ok(BTreeMap::new());
    }

    let dev_size = dev.total_size();
    if table_abs >= dev_size {
        return Err(crate::Error::InvalidImage(
            "grf: table offset past end of file".into(),
        ));
    }

    let entries = match version {
        0x102 | 0x103 => {
            // v0x102/0x103: the table layout starts with 8 bytes of
            // posinfo just like v0x200 (libgrf inflates the
            // remainder), followed by a 4-byte legacy-framing word.
            // See grf.c lines 826–910 — the only difference from
            // v0x200 is the extra 4-byte `brokenpos` field after the
            // compressed payload.
            let table = read_compressed_table(dev, table_abs, /* legacy_framing = */ true)?;
            table::decode_v102(&table)?
        }
        0x200 => {
            let table = read_compressed_table(dev, table_abs, /* legacy_framing = */ false)?;
            table::decode_v200(&table)?
        }
        other => {
            return Err(crate::Error::Unsupported(format!(
                "grf: cannot read table for version {other:#x}"
            )));
        }
    };

    let mut map = BTreeMap::new();
    for e in entries {
        map.insert(normalise_path(&e.name), e);
    }
    Ok(map)
}

fn read_compressed_table(
    dev: &mut dyn BlockDevice,
    table_abs: u64,
    legacy_framing: bool,
) -> Result<Vec<u8>> {
    let dev_size = dev.total_size();
    let mut posinfo = [0u8; 8];
    if table_abs + 8 > dev_size {
        return Err(crate::Error::InvalidImage(
            "grf: table header truncated".into(),
        ));
    }
    dev.read_at(table_abs, &mut posinfo)?;
    let comp_size = u32::from_le_bytes(posinfo[0..4].try_into().unwrap()) as usize;
    let uncomp_size = u32::from_le_bytes(posinfo[4..8].try_into().unwrap()) as usize;

    let comp_start = table_abs + 8;
    if comp_start + comp_size as u64 > dev_size {
        return Err(crate::Error::InvalidImage(
            "grf: compressed table payload past end of file".into(),
        ));
    }
    let mut comp = vec![0u8; comp_size];
    dev.read_at(comp_start, &mut comp)?;

    // Legacy framing — there's an additional 4-byte word after the
    // compressed payload in v0x102/0x103. We don't use its value
    // (libgrf calls it `brokenpos` and treats it as opaque).
    let _ = legacy_framing;

    crate::compression::decompress(crate::compression::Algo::Zlib, &comp, uncomp_size)
}

/// Strip a leading `/` from a path string so it lines up with the
/// CP949 names libgrf writes (which never start with `/`).
fn normalise_path(s: &str) -> String {
    s.trim_start_matches('/').to_string()
}

impl crate::fs::FilesystemFactory for Grf {
    type FormatOpts = FormatOpts;

    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format_with(dev, opts)
    }

    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open_dev(dev)
    }
}

impl crate::fs::Filesystem for Grf {
    fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &std::path::Path,
        src: FileSource,
        _meta: FileMeta,
    ) -> Result<()> {
        let key = normalise_path(
            path.to_str()
                .ok_or_else(|| crate::Error::InvalidArgument("grf: non-UTF-8 path".into()))?,
        );
        writer::add_file(self, dev, key, src)
    }

    fn create_dir(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &std::path::Path,
        _meta: FileMeta,
    ) -> Result<()> {
        // GRF has no directory entries — paths' parents are implicit
        // from the slashes. `create_dir` is a no-op so that callers
        // who emit dirs (e.g. the repack walker) don't error out.
        Ok(())
    }

    fn create_symlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &std::path::Path,
        _target: &std::path::Path,
        _meta: FileMeta,
    ) -> Result<()> {
        Err(crate::Error::Unsupported(
            "grf: symlinks are not part of the archive format".into(),
        ))
    }

    fn create_device(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &std::path::Path,
        _kind: crate::fs::DeviceKind,
        _major: u32,
        _minor: u32,
        _meta: FileMeta,
    ) -> Result<()> {
        Err(crate::Error::Unsupported(
            "grf: device nodes are not part of the archive format".into(),
        ))
    }

    fn remove(&mut self, _dev: &mut dyn BlockDevice, path: &std::path::Path) -> Result<()> {
        let key = normalise_path(
            path.to_str()
                .ok_or_else(|| crate::Error::InvalidArgument("grf: non-UTF-8 path".into()))?,
        );
        writer::remove(self, &key)
    }

    fn list(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let prefix = {
            let s = path
                .to_str()
                .ok_or_else(|| crate::Error::InvalidArgument("grf: non-UTF-8 path".into()))?;
            let trimmed = s.trim_start_matches('/').trim_end_matches('/');
            if trimmed.is_empty() {
                String::new()
            } else {
                format!("{trimmed}/")
            }
        };

        // Collect the immediate children of `prefix`: each unique
        // first path component after the prefix. Files appear as
        // Regular, intermediate path components appear as Dir.
        use std::collections::BTreeMap as B;
        let mut children: B<String, crate::fs::EntryKind> = B::new();
        let mut sizes: B<String, u64> = B::new();
        for (name, entry) in &self.entries {
            let Some(tail) = name.strip_prefix(&prefix) else {
                continue;
            };
            if tail.is_empty() {
                continue;
            }
            if let Some((leaf, _)) = tail.split_once('/') {
                children.insert(leaf.to_string(), crate::fs::EntryKind::Dir);
                sizes.insert(leaf.to_string(), 0);
            } else {
                children.insert(tail.to_string(), crate::fs::EntryKind::Regular);
                sizes.insert(tail.to_string(), entry.size as u64);
            }
        }
        Ok(children
            .into_iter()
            .map(|(name, kind)| {
                let size = *sizes.get(&name).unwrap_or(&0);
                crate::fs::DirEntry {
                    name,
                    inode: 0,
                    kind,
                    size,
                }
            })
            .collect())
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &std::path::Path,
    ) -> Result<Box<dyn Read + 'a>> {
        let key = normalise_path(
            path.to_str()
                .ok_or_else(|| crate::Error::InvalidArgument("grf: non-UTF-8 path".into()))?,
        );
        let entry =
            self.entries.get(&key).cloned().ok_or_else(|| {
                crate::Error::InvalidArgument(format!("grf: no entry at {key:?}"))
            })?;
        let bytes = self.read_entry(dev, &entry)?;
        Ok(Box::new(std::io::Cursor::new(bytes)))
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        writer::flush(self, dev)
    }

    fn mutation_capability(&self) -> MutationCapability {
        MutationCapability::Mutable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;
    use crate::fs::{Filesystem, FilesystemFactory};

    #[test]
    fn empty_round_trip() {
        let mut dev = MemoryBackend::new(64 * 1024);
        let mut grf = Grf::format(&mut dev, &FormatOpts::default()).unwrap();
        grf.flush(&mut dev).unwrap();

        let reopen = Grf::open(&mut dev).unwrap();
        assert_eq!(reopen.version, 0x200);
        assert_eq!(reopen.entries.len(), 0);
    }

    #[test]
    fn add_read_round_trip() {
        let mut dev = MemoryBackend::new(64 * 1024);
        let mut grf = Grf::format(&mut dev, &FormatOpts::default()).unwrap();

        let body = b"hello, world!";
        grf.create_file(
            &mut dev,
            std::path::Path::new("/data/info.txt"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(body.to_vec())),
                len: body.len() as u64,
            },
            FileMeta::default(),
        )
        .unwrap();
        grf.flush(&mut dev).unwrap();

        let mut reopen = Grf::open(&mut dev).unwrap();
        assert_eq!(reopen.entries.len(), 1);
        let entries = reopen
            .list(&mut dev, std::path::Path::new("/data"))
            .unwrap();
        assert!(entries.iter().any(|e| e.name == "info.txt"));
        let entry = reopen.entries.get("data/info.txt").cloned().unwrap();
        let bytes = reopen.read_entry(&mut dev, &entry).unwrap();
        assert_eq!(bytes, body);
    }

    #[test]
    fn hangul_filename_round_trip() {
        let mut dev = MemoryBackend::new(64 * 1024);
        let mut grf = Grf::format(&mut dev, &FormatOpts::default()).unwrap();
        grf.create_file(
            &mut dev,
            std::path::Path::new("/data/한글.txt"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(b"hi".to_vec())),
                len: 2,
            },
            FileMeta::default(),
        )
        .unwrap();
        grf.flush(&mut dev).unwrap();

        let reopen = Grf::open(&mut dev).unwrap();
        assert!(reopen.entries.contains_key("data/한글.txt"));
    }

    #[test]
    fn remove_marks_wasted_space() {
        let mut dev = MemoryBackend::new(64 * 1024);
        let mut grf = Grf::format(&mut dev, &FormatOpts::default()).unwrap();
        grf.create_file(
            &mut dev,
            std::path::Path::new("/a.txt"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(vec![0u8; 4096])),
                len: 4096,
            },
            FileMeta::default(),
        )
        .unwrap();
        grf.create_file(
            &mut dev,
            std::path::Path::new("/b.txt"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(vec![0u8; 4096])),
                len: 4096,
            },
            FileMeta::default(),
        )
        .unwrap();
        grf.flush(&mut dev).unwrap();

        let mut reopen = Grf::open(&mut dev).unwrap();
        reopen
            .remove(&mut dev, std::path::Path::new("/a.txt"))
            .unwrap();
        reopen.flush(&mut dev).unwrap();
        assert!(reopen.wasted_space() > 0);
    }
}
