//! fstool — build disk images and filesystems from a directory tree and TOML spec.
//!
//! The crate is organised as a stack of three trait-based layers:
//!
//! - [`block`] — `BlockDevice`: raw seekable byte storage. Backends include
//!   on-disk files, in-memory buffers (for tests), and sub-range slices used to
//!   give each partition an isolated view.
//! - [`part`] — `PartitionTable`: MBR and GPT. (Coming in P2.)
//! - [`fs`] — `Filesystem`: ext2/3/4 in v1; FAT32 deferred. (Coming in P3+.)
//!
//! High-level entry points for building or inspecting an image live at the
//! crate root once P5 lands.

pub mod block;
pub mod error;
pub mod fs;
pub mod part;
pub mod spec;

pub use error::{Error, Result};
