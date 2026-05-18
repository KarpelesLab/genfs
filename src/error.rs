//! Error type and `Result` alias for the crate.
//!
//! All public APIs return `genfs::Result<T>` = `Result<T, genfs::Error>`. The
//! variants are intentionally small at this stage; further variants will be
//! added as later layers (partition tables, filesystems, spec parsing) come
//! online.

use std::io;

use thiserror::Error;

/// Crate-wide error type.
#[derive(Debug, Error)]
pub enum Error {
    /// Underlying I/O failure (file backend, host file source, etc.).
    #[error("io: {0}")]
    Io(#[from] io::Error),

    /// A block-device operation referenced a byte range that lies (partly or
    /// wholly) outside the device's logical extent. Includes slice violations.
    #[error("out of bounds: offset {offset} len {len} exceeds device size {size}")]
    OutOfBounds { offset: u64, len: u64, size: u64 },

    /// On-disk structure failed validation (bad magic, bad checksum, etc.).
    #[error("invalid image: {0}")]
    InvalidImage(String),

    /// The requested feature exists in the format but is not implemented in
    /// this build of genfs. Used for clean "FAT32 not in v1" type messages.
    #[error("unsupported feature: {0}")]
    Unsupported(String),

    /// A user-supplied value was malformed or contradictory (bad spec, etc.).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
