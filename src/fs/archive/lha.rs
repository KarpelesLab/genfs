//! LHA / LZH (`.lzh`, `.lha`) — detection-only scaffold.
//!
//! Recognised by `detect_fs` via the `-lh?-` / `-lz?-` method tag at
//! offset 2. The LZSS/Huffman decoders are not implemented yet; all
//! operations return a clean `Unsupported`. A future pass can wire a
//! pure-Rust decoder behind an `lha` Cargo feature.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;

/// LHA filesystem handle (scaffold).
pub struct LhaFs(pub ArchiveFs);

impl LhaFs {
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self(ArchiveFs::scaffold("lha")))
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "lha: creating archives is not supported".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for LhaFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

crate::impl_archive_fs_filesystem!(LhaFs);
