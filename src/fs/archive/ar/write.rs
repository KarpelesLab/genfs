//! `ar` writer: GNU-format, members stored uncompressed.
//!
//! Member bodies are buffered in memory until [`finish`](ArWriter::finish)
//! because GNU's `//` long-name string table must precede the members
//! that reference it. `ar` archives are small by nature (object files,
//! `.deb` control members), so this is an acceptable, documented cost.

use std::io::Read;

use crate::block::BlockDevice;
use crate::fs::archive::ArchiveBuilder;
use crate::fs::archive::tree;
use crate::fs::archive::writer::Cursor;
use crate::fs::{DeviceKind, FileMeta, FileSource};
use crate::{Error, Result};

struct Member {
    name: String,
    mtime: u64,
    uid: u32,
    gid: u32,
    mode: u16,
    data: Vec<u8>,
}

pub struct ArWriter {
    cursor: Cursor,
    members: Vec<Member>,
}

impl ArWriter {
    pub fn new(dev: &dyn BlockDevice) -> Self {
        Self {
            cursor: Cursor::new(dev),
            members: Vec::new(),
        }
    }
}

/// `ar` is flat: reject a path that names a subdirectory.
fn flat_name(path: &str) -> Result<String> {
    let norm = tree::normalise_path(path);
    let leaf = norm.trim_start_matches('/');
    if leaf.contains('/') {
        return Err(Error::Unsupported(format!(
            "ar: {path:?} is inside a subdirectory — `ar` is a flat archive; \
             use tar/zip/cpio for directory trees"
        )));
    }
    Ok(leaf.to_string())
}

/// Left-justify `s` into `field`, space-padding the remainder.
fn put(field: &mut [u8], s: &str) {
    let b = s.as_bytes();
    let n = b.len().min(field.len());
    field[..n].copy_from_slice(&b[..n]);
}

#[allow(clippy::too_many_arguments)]
fn write_member(
    cur: &mut Cursor,
    dev: &mut dyn BlockDevice,
    name_field: &str,
    mtime: u64,
    uid: u32,
    gid: u32,
    mode: u16,
    body: &[u8],
) -> Result<()> {
    let mut hdr = [b' '; 60];
    put(&mut hdr[0..16], name_field);
    put(&mut hdr[16..28], &mtime.to_string());
    put(&mut hdr[28..34], &uid.to_string());
    put(&mut hdr[34..40], &gid.to_string());
    // Full st_mode in octal (regular file | perm bits), e.g. "100644".
    put(
        &mut hdr[40..48],
        &format!("{:o}", 0o100000u32 | u32::from(mode)),
    );
    put(&mut hdr[48..58], &body.len().to_string());
    hdr[58] = b'`';
    hdr[59] = b'\n';
    cur.write(dev, &hdr)?;
    cur.write(dev, body)?;
    if body.len() % 2 == 1 {
        cur.write(dev, b"\n")?;
    }
    Ok(())
}

impl ArchiveBuilder for ArWriter {
    fn add_file(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &str,
        src: FileSource,
        meta: FileMeta,
    ) -> Result<()> {
        let name = flat_name(path)?;
        let (mut reader, len) = src.open()?;
        let mut data = Vec::with_capacity(len.min(1 << 20) as usize);
        reader.read_to_end(&mut data).map_err(Error::from)?;
        self.members.push(Member {
            name,
            mtime: u64::from(meta.mtime),
            uid: meta.uid,
            gid: meta.gid,
            mode: meta.mode,
            data,
        });
        Ok(())
    }

    fn add_dir(&mut self, _dev: &mut dyn BlockDevice, _path: &str, _meta: FileMeta) -> Result<()> {
        // `ar` has no directories; silently drop (files carry no path).
        Ok(())
    }

    fn add_symlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &str,
        _target: &str,
        _meta: FileMeta,
    ) -> Result<()> {
        Err(Error::Unsupported(format!(
            "ar: cannot store symlink {path:?} — `ar` has no symlink records"
        )))
    }

    fn add_device(
        &mut self,
        _dev: &mut dyn BlockDevice,
        path: &str,
        _kind: DeviceKind,
        _major: u32,
        _minor: u32,
        _meta: FileMeta,
    ) -> Result<()> {
        Err(Error::Unsupported(format!(
            "ar: cannot store device/special node {path:?}"
        )))
    }

    fn finish(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        self.cursor.write(dev, super::MAGIC)?;

        // Build the GNU long-name table and per-member name fields.
        let mut table = Vec::new();
        let mut name_fields = Vec::with_capacity(self.members.len());
        for m in &self.members {
            if m.name.len() <= 15 && !m.name.contains(' ') {
                name_fields.push(format!("{}/", m.name));
            } else {
                let off = table.len();
                table.extend_from_slice(m.name.as_bytes());
                table.extend_from_slice(b"/\n");
                name_fields.push(format!("/{off}"));
            }
        }
        if !table.is_empty() {
            write_member(&mut self.cursor, dev, "//", 0, 0, 0, 0, &table)?;
        }

        let members = std::mem::take(&mut self.members);
        for (m, nf) in members.iter().zip(name_fields) {
            write_member(
                &mut self.cursor,
                dev,
                &nf,
                m.mtime,
                m.uid,
                m.gid,
                m.mode,
                &m.data,
            )?;
        }
        dev.sync()?;
        Ok(())
    }

    fn position(&self) -> u64 {
        self.cursor.position()
    }
}
