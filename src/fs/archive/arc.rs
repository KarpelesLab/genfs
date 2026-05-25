//! SEA ARC (`.arc`) — detection-only scaffold.
//!
//! Recognised by `detect_fs` via the leading `0x1A` marker followed by
//! a method byte. The (ancient) compression methods are not decoded
//! yet; all operations return a clean `Unsupported`.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;

/// ARC filesystem handle (scaffold).
pub struct ArcFs(pub ArchiveFs);

impl ArcFs {
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self(ArchiveFs::scaffold("arc")))
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "arc: creating archives is not supported".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for ArcFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

crate::impl_archive_fs_filesystem!(ArcFs);
