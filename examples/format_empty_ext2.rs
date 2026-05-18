//! Format a 1 MiB empty ext2 image, then exit. Compare with genext2fs:
//!
//! ```sh
//! cargo run --example format_empty_ext2 -- /tmp/genfs.ext2
//! mkdir -p /tmp/empty
//! genext2fs -d /tmp/empty -f -q -B 1024 -b 1024 /tmp/ref.ext2
//! e2fsck -fn /tmp/genfs.ext2     # must report clean
//! diff <(dumpe2fs -h /tmp/genfs.ext2) <(dumpe2fs -h /tmp/ref.ext2)
//! ```

use std::path::PathBuf;

use genfs::block::{BlockDevice, FileBackend};
use genfs::fs::ext::{Ext, FormatOpts};

fn main() {
    let path: PathBuf = std::env::args_os()
        .nth(1)
        .unwrap_or_else(|| "/tmp/genfs_empty.ext2".into())
        .into();
    let opts = FormatOpts::default();
    let size = opts.blocks_count as u64 * opts.block_size as u64;
    let mut dev = FileBackend::create(&path, size).expect("create image file");
    Ext::format_with(&mut dev, &opts).expect("format ext2");
    dev.sync().expect("sync");
    println!("wrote empty ext2 image to {}", path.display());
}
