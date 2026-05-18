# genfs

Build disk images and filesystem images from a directory tree and a TOML
spec — in the spirit of `genext2fs`, but going further:

- **Multiple filesystems** — ext2 / ext3 / ext4 (v1), FAT32 to follow.
- **Whole disk images** — MBR and GPT partition tables, not just bare FS images.
- **Streaming** — file contents are never preloaded in memory regardless of
  size. Generation is a two-pass scan-then-stream.
- **Modular** — block devices, partition tables, and filesystems each go
  through a small trait; adding a new partition scheme or filesystem is a
  drop-in plugin.
- **Inspectable** — the same trait stack reads existing images, so the
  CLI ships `genfs ls`, `genfs cat`, and `genfs info` alongside `build`.
- **In-place modification** — add, remove, and replace whole files inside an
  existing image without rewriting it.

genfs is implemented in Rust and ships as a library (`genfs`) plus a thin CLI
binary (`genfs`).

## Status

Early in development. Public API is **unstable** until v0.5.

| Phase                                    | Status        |
|------------------------------------------|---------------|
| 1. `BlockDevice` foundation              | ✅ done        |
| 2. MBR + GPT partition tables            | ✅ done        |
| 3. ext2 writer                           | 🚧 in progress |
| 4. ext2 reader + ext3/4 feature matrix   | planned       |
| 5. TOML spec + CLI                       | planned       |
| 6. FAT32                                 | post-v1        |

What works today:

- `genfs::block::{FileBackend, MemoryBackend, SlicedBackend}` — sparse
  file-backed devices, in-memory devices for tests, bounds-checked sub-range
  views for carving partitions.
- `genfs::part::{Mbr, Gpt}` — write and read 4-primary MBR and 128-entry GPT
  layouts. GPT includes the protective MBR, primary and backup headers, and
  CRC32 on both the header and the partition entry array. Cross-checked
  against `sgdisk -v`.

## Architecture

```
                ┌────────────────────────────────────────────┐
                │           CLI (clap) — bin/genfs           │
                └────────────────────────────────────────────┘
                                     │
                ┌────────────────────────────────────────────┐
                │  Spec layer (TOML → ImageSpec / FsSpec)    │
                └────────────────────────────────────────────┘
                                     │
                ┌────────────────────────────────────────────┐
                │  Filesystem trait → ext::Ext (2/3/4)       │
                └────────────────────────────────────────────┘
                                     │
                ┌────────────────────────────────────────────┐
                │  PartitionTable trait → Mbr, Gpt           │
                └────────────────────────────────────────────┘
                                     │
                ┌────────────────────────────────────────────┐
                │  BlockDevice trait → FileBackend, Sliced…  │
                └────────────────────────────────────────────┘
```

Each layer is substitutable. A filesystem implementation talks only to a
`BlockDevice`; it doesn't know or care whether the device is a real file, an
in-memory buffer in a test, or a slice carved out of a larger disk by a
partition table.

## Try it

GPT demo (requires `sgdisk` for the validation step):

```sh
cargo run --example inspect_gpt -- /tmp/demo.img
sgdisk -p /tmp/demo.img
sgdisk -v /tmp/demo.img        # "No problems found."
```

Run the test suite:

```sh
cargo test                     # unit tests + external cross-checks if tools present
```

## Roadmap

The streaming invariant is the project's load-bearing constraint: regardless
of image size, no file's contents are ever fully resident in memory. The
writer scans the source twice — once to compute geometry (inode count, block
count, dir-entry sizes), once to stream bytes from each file directly to the
image. This is the difference between genfs and a "build a Vec<u8> and dump
it" approach.

In-place modification is restricted to whole-file granularity in v1 (add,
remove, replace). Partial-file rewrites are explicitly out of scope until
there's a use case that demands them.

## Licence

Dual-licensed under MIT OR Apache-2.0.
