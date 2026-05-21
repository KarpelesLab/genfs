//! Feed arbitrary bytes into `Ext::open` and assert: no panics, no
//! unwraps, no infinite loops — every malformed input either parses
//! into a usable handle (the lucky case) or returns a structured
//! `crate::Error`.
//!
//! Run with:
//!   cargo +nightly fuzz run ext_open

#![no_main]

use fstool::block::{BlockDevice, MemoryBackend};
use fstool::fs::ext::Ext;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Minimum size: SB lives at offset 1024 + a few bytes of header.
    if data.len() < 4096 {
        return;
    }
    let mut dev = MemoryBackend::new(data.len() as u64);
    // Seed the device with the fuzzer's bytes.
    let _ = dev.write_at(0, data);
    // Open + minimal walk. Discard the result — we only care that
    // these calls don't panic. Any returned Err is acceptable.
    if let Ok(ext) = Ext::open(&mut dev) {
        let _ = ext.list_inode(&mut dev, 2);
        let _ = ext.path_to_inode(&mut dev, "/");
    }
});
