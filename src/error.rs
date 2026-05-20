//! Error type and `Result` alias for the crate.
//!
//! All public APIs return `fstool::Result<T>` = `Result<T, fstool::Error>`. The
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
    /// this build of fstool. Used for clean "FAT32 not in v1" type messages.
    #[error("unsupported feature: {0}")]
    Unsupported(String),

    /// A user-supplied value was malformed or contradictory (bad spec, etc.).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The operation tried to modify a **streaming** filesystem — one
    /// whose writer can't seek backward once bytes have been emitted.
    /// Tar today; any future stream-of-records format that lands in
    /// fstool. Distinct from [`Error::Immutable`] so callers can tell
    /// "the writer fundamentally can't go back" apart from "the
    /// on-disk layout was never designed for in-place edits."
    #[error("{op}: {kind} is a streaming format — use `fstool repack` to produce a new one")]
    Streaming {
        /// The filesystem kind that refused (today: `"tar"`).
        kind: &'static str,
        /// Short verb describing the attempted op (`"add"`, `"rm"`, …).
        /// Free-form; not a stable enum.
        op: &'static str,
    },

    /// The operation tried to modify a **write-once** filesystem whose
    /// on-disk layout has no in-place mutation hooks (no free-block
    /// tracking, no journal). ISO 9660 and SquashFS today. The
    /// writer can seek, but re-opening the image as writable isn't
    /// part of the format's design — modifications go through
    /// `fstool repack` to rebuild the image from scratch.
    #[error("{op}: {kind} is a write-once format — use `fstool repack` to rebuild it")]
    Immutable {
        /// The filesystem kind that refused (today: `"iso9660"`,
        /// `"squashfs"`).
        kind: &'static str,
        /// Short verb describing the attempted op.
        op: &'static str,
    },
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
