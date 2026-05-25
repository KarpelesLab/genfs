//! RAR (`.rar`, v4 + v5) — detection-only scaffold.
//!
//! Recognised by `detect_fs` via the `Rar!\x1A\x07` signature (byte 6
//! distinguishes v4 `0x00` from v5 `0x01`). RAR decompression is
//! reverse-engineered and **archive creation is forbidden by the RAR
//! licence**, so this format is read-only-at-best and remains a
//! scaffold until a pure-Rust extractor is wired. All operations return
//! a clean `Unsupported`.

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;

/// RAR filesystem handle (scaffold).
pub struct RarFs(pub ArchiveFs);

impl RarFs {
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self(ArchiveFs::scaffold("rar")))
    }

    pub fn format(_dev: &mut dyn BlockDevice, _opts: &()) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "rar: creating archives is not supported (RAR compression is proprietary)".into(),
        ))
    }
}

impl crate::fs::FilesystemFactory for RarFs {
    type FormatOpts = ();
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

crate::impl_archive_fs_filesystem!(RarFs);
