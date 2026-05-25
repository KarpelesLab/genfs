//! `cpio` archives as an fstool filesystem.
//!
//! Covers the two portable ASCII variants:
//!
//! - **newc** (`070701`) and **newc-crc** (`070702`) — 110-byte
//!   hex-ASCII headers, 4-byte-aligned name + body.
//! - **odc** (`070707`) — 76-byte octal-ASCII headers, no padding.
//!
//! The binary `cpio` variants (`0x71c7` / byte-swapped) are out of
//! scope. Reads cover all file types (regular, dir, symlink, device,
//! fifo); the writer emits **newc**. Hard links are read as independent
//! files (the link relationship is not reconstructed).

mod scan;
mod write;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;

/// newc header magic.
pub const MAGIC_NEWC: &[u8; 6] = b"070701";
/// newc-with-CRC header magic.
pub const MAGIC_NEWC_CRC: &[u8; 6] = b"070702";
/// odc (old portable) header magic.
pub const MAGIC_ODC: &[u8; 6] = b"070707";
/// End-of-archive sentinel name.
pub const TRAILER: &str = "TRAILER!!!";
/// POSIX file-type mask (`S_IFMT`).
pub const S_IFMT: u32 = 0o170000;

/// Format options for `cpio` (none today).
#[derive(Debug, Clone, Default)]
pub struct CpioFormatOpts;

impl CpioFormatOpts {
    pub fn apply_options(&mut self, _bag: &mut crate::format_opts::OptionMap) -> Result<()> {
        Ok(())
    }
}

/// `cpio` filesystem handle.
pub struct CpioFs(pub ArchiveFs);

impl CpioFs {
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self(ArchiveFs::from_index(scan::scan(dev)?)))
    }

    pub fn format(dev: &mut dyn BlockDevice, _opts: &CpioFormatOpts) -> Result<Self> {
        Ok(Self(ArchiveFs::writer(
            "cpio",
            Box::new(write::CpioWriter::new(dev)),
        )))
    }
}

impl crate::fs::FilesystemFactory for CpioFs {
    type FormatOpts = CpioFormatOpts;
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

crate::impl_archive_fs_filesystem!(CpioFs);
