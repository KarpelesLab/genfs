//! Microsoft Cabinet (`.cab`, `MSCF`) — detection-only scaffold.
//!
//! Recognised by `detect_fs` via the `MSCF` signature. CAB folders use
//! MSZIP (raw DEFLATE), Quantum, or LZX; the reader is not implemented
//! yet, so all operations return a clean `Unsupported`. A future pass
//! can wire a pure-Rust CAB reader behind a `cab` Cargo feature.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;

/// CAB filesystem handle (scaffold).
pub struct CabFs(pub ArchiveFs);

impl CabFs {
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self(ArchiveFs::scaffold("cab")))
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "cab: creating archives is not supported".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for CabFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

crate::impl_archive_fs_filesystem!(CabFs);
