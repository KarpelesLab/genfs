//! ext2 / ext3 / ext4 filesystem implementation.
//!
//! v1 writes ext2 (no journal, no extents, no htree). Feature-flag wiring
//! for ext3/ext4 follows in P4; the on-disk format types are intentionally
//! shared so adding the deltas does not duplicate the superblock /
//! group-descriptor / inode encoders.
//!
//! ## Binary-exact compatibility with genext2fs
//!
//! When configured with [`FormatOpts::genext2fs_compat`] the writer aims to
//! produce a file byte-for-byte identical to `genext2fs -d <dir> img.ext2`
//! on the same input. This affects defaults across every layer (block size,
//! UUID, timestamps, dirent ordering, allocation order, padding). See
//! `tests/ext2_genext2fs_compat.rs` for the diff harness.

pub mod constants;
pub mod dir;
pub mod group;
pub mod inode;
pub mod superblock;
