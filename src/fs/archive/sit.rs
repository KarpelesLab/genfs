//! StuffIt (`.sit`, `SIT!` / `StuffIt`) — detection-only scaffold.
//!
//! Recognised by `detect_fs` via the `SIT!` (classic) or `StuffIt`
//! (SIT5) signature. The format is proprietary and poorly documented;
//! the reader is not implemented and all operations return a clean
//! `Unsupported`.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;

/// StuffIt filesystem handle (scaffold).
pub struct SitFs(pub ArchiveFs);

impl SitFs {
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self(ArchiveFs::scaffold("sit")))
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "sit: creating archives is not supported".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for SitFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

crate::impl_archive_fs_filesystem!(SitFs);
