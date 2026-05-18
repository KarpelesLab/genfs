//! Build a 64 MiB GPT image with an EFI System Partition and a Linux root
//! partition, then exit. Cross-check the result with system tools, e.g.:
//!
//! ```sh
//! cargo run --example inspect_gpt -- /tmp/out.img
//! sgdisk -p /tmp/out.img
//! sgdisk -v /tmp/out.img
//! ```

use std::path::PathBuf;

use fstool::block::{BlockDevice, FileBackend};
use fstool::part::{Gpt, Partition, PartitionKind, PartitionTable};

fn main() {
    let path: PathBuf = std::env::args_os()
        .nth(1)
        .unwrap_or_else(|| "/tmp/fstool_demo.img".into())
        .into();
    let total: u64 = 64 * 1024 * 1024;

    let mut dev = FileBackend::create(&path, total).expect("create image file");

    let total_lba = total / 512;
    let parts = vec![
        Partition {
            start_lba: 2048,
            size_lba: 2048,
            kind: PartitionKind::EfiSystem,
            name: Some("EFI System".into()),
            ..Partition::new(0, 0, PartitionKind::EfiSystem)
        },
        Partition {
            start_lba: 4096,
            size_lba: total_lba - 4096 - 34,
            kind: PartitionKind::LinuxFilesystem,
            name: Some("root".into()),
            ..Partition::new(0, 0, PartitionKind::LinuxFilesystem)
        },
    ];
    let gpt = Gpt::build(parts).expect("build gpt");
    gpt.write(&mut dev).expect("write gpt");
    dev.sync().expect("sync");

    println!("wrote {} ({} MiB)", path.display(), total / (1024 * 1024));
    println!("inspect with: sgdisk -p {}", path.display());
}
