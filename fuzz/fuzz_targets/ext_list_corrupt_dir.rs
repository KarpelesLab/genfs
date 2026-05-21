//! Build a tiny valid ext4 image, then splatter the fuzzer's bytes
//! into a random offset of the root dir block. `list_inode` on that
//! corrupted dir must either return entries or an `Err` — never
//! panic, never read past the end of the block.
//!
//! Catches off-by-one + integer-overflow regressions in
//! `dir::decode_entry` and the multi-block walker added in the
//! multi-block-dir work.

#![no_main]

use fstool::block::{BlockDevice, MemoryBackend};
use fstool::fs::ext::{Ext, FormatOpts, FsKind};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }
    let opts = FormatOpts {
        kind: FsKind::Ext4,
        block_size: 1024,
        blocks_count: 2048,
        inodes_count: 16,
        journal_blocks: 1024,
        ..FormatOpts::default()
    };
    let total = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = MemoryBackend::new(total);
    let Ok(_) = Ext::format_with(&mut dev, &opts) else {
        return;
    };

    // Use the first 4 bytes as an offset, the rest as the splatter.
    let off = u32::from_le_bytes(data[..4].try_into().unwrap()) as u64;
    let off = off % total;
    let payload = &data[4..];
    let n = (payload.len() as u64).min(total - off);
    let _ = dev.write_at(off, &payload[..n as usize]);

    // Now try to reopen + walk; everything must terminate cleanly.
    if let Ok(ext) = Ext::open(&mut dev) {
        let _ = ext.list_inode(&mut dev, 2);
        for ino in 1..=16 {
            let _ = ext.read_inode(&mut dev, ino);
        }
    }
});
