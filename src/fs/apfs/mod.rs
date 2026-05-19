//! APFS — Apple's modern macOS / iOS filesystem. Read-only support.
//!
//! Status: **stub** — a parallel agent run will fill this in. APFS is
//! the most complex of the four new filesystems; even read-only support
//! is a substantial undertaking.

use crate::Result;
use crate::block::BlockDevice;

pub struct Apfs {
    _private: (),
}

impl Apfs {
    pub fn open(_dev: &mut dyn BlockDevice) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "apfs: read support not yet implemented".into(),
        ))
    }
}

/// Probe for the APFS container superblock magic `"NXSB"` at offset
/// 32 of LBA 0 (block 0 is the container superblock; its `nx_magic`
/// field lives at offset 32 in the `nx_superblock_t` layout).
pub fn probe(dev: &mut dyn BlockDevice) -> Result<bool> {
    if dev.total_size() < 64 {
        return Ok(false);
    }
    let mut head = [0u8; 64];
    dev.read_at(0, &mut head)?;
    Ok(&head[32..36] == b"NXSB")
}
