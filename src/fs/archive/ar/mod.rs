//! Unix `ar` archives (`!<arch>\n`) as an fstool filesystem.
//!
//! `ar` is the flat archive behind static libraries (`.a`) and Debian
//! packages (`.deb`). It has **no directories, symlinks, or device
//! nodes** — just a sequence of named members, each a regular file.
//!
//! - **Read:** the common GNU/SysV layout (`name/` short names, a `//`
//!   string table for long names referenced as `/<offset>`, a `/`
//!   symbol table that is skipped) plus the BSD layout (`#1/<len>` with
//!   the name stored inline ahead of the data, `__.SYMDEF` skipped).
//! - **Write:** GNU format. Members are stored uncompressed; long names
//!   go through the `//` table. Because `ar` is flat, writing a nested
//!   tree is refused with a message pointing at tar/zip/cpio.

mod scan;
mod write;

use crate::Result;
use crate::block::BlockDevice;
use crate::fs::archive::ArchiveFs;

/// 8-byte global header.
pub const MAGIC: &[u8; 8] = b"!<arch>\n";

/// Format options for `ar` (none today).
#[derive(Debug, Clone, Default)]
pub struct ArFormatOpts;

impl ArFormatOpts {
    pub fn apply_options(&mut self, _bag: &mut crate::format_opts::OptionMap) -> Result<()> {
        Ok(())
    }
}

/// `ar` filesystem handle.
pub struct ArFs(pub ArchiveFs);

impl ArFs {
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Ok(Self(ArchiveFs::from_index(scan::scan(dev)?)))
    }

    pub fn format(dev: &mut dyn BlockDevice, _opts: &ArFormatOpts) -> Result<Self> {
        Ok(Self(ArchiveFs::writer(
            "ar",
            Box::new(write::ArWriter::new(dev)),
        )))
    }
}

impl crate::fs::FilesystemFactory for ArFs {
    type FormatOpts = ArFormatOpts;
    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }
    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

crate::impl_archive_fs_filesystem!(ArFs);
