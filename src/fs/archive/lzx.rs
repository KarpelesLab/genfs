//! Amiga LZX (`.lzx`, `LZX\0`) — detection-only scaffold.
//!
//! Recognised by `detect_fs` via the `LZX\0` signature. (This is the
//! Amiga archive format, distinct from the Microsoft LZX *compression
//! method* used inside CAB/CHM.) The decoder is not implemented yet;
//! all operations return a clean `Unsupported`.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;

/// LZX filesystem handle (scaffold).
pub struct LzxFs(pub ArchiveFs);

impl LzxFs {
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self(ArchiveFs::scaffold("lzx")))
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "lzx: creating archives is not supported".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for LzxFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

crate::impl_archive_fs_filesystem!(LzxFs);
