//! 7-Zip (`.7z`) — detection-only scaffold.
//!
//! Recognised by `detect_fs` via the `37 7A BC AF 27 1C` signature, but
//! the reader (header decode + LZMA/LZMA2 coders) is not implemented
//! yet; all operations return a clean `Unsupported` naming the format.
//! A future pass wires a pure-Rust decoder behind a `sevenz` Cargo
//! feature.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;

/// 7z filesystem handle (scaffold).
pub struct SevenZFs(pub ArchiveFs);

impl SevenZFs {
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self(ArchiveFs::scaffold("7z")))
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "7z: creating archives is not supported".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for SevenZFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

crate::impl_archive_fs_filesystem!(SevenZFs);
