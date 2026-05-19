# fstool

[![CI](https://github.com/KarpelesLab/fstool/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/fstool/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/fstool.svg)](https://crates.io/crates/fstool)
[![docs.rs](https://docs.rs/fstool/badge.svg)](https://docs.rs/fstool)

Build disk images and filesystem images from a directory tree and a TOML
spec — in the spirit of `genext2fs`, but going further:

- **Multiple filesystems** — read+write for ext2/3/4, FAT32, and tar; read-only
  for XFS, exFAT, HFS+, and APFS; detection scaffolds in place for NTFS,
  F2FS, and SquashFS.
- **Whole disk images** — MBR and GPT partition tables, not just bare FS images.
- **Streaming** — file contents are never preloaded in memory regardless of
  size. Generation is a two-pass scan-then-stream.
- **Modular** — block devices, partition tables, and filesystems each go
  through a small trait; adding a new partition scheme or filesystem is a
  drop-in plugin.
- **Inspectable** — the same trait stack reads existing images, so the
  CLI ships `fstool ls`, `fstool cat`, and `fstool info` alongside `build`.
- **In-place modification** — add, remove, and replace whole files inside an
  existing image without rewriting it.

fstool is implemented in Rust and ships as a library (`fstool`) plus a thin CLI
binary (`fstool`).

## Status

Early in development. Public API is **unstable** until v0.5.

| Phase                                    | Status        |
|------------------------------------------|---------------|
| 1. `BlockDevice` foundation              | ✅ done        |
| 2. MBR + GPT partition tables            | ✅ done        |
| 3. ext2 writer                           | ✅ done        |
| 4. ext2/3/4 reader + writer              | ✅ done        |
| 5. TOML spec + CLI                       | ✅ done        |
| 6. FAT32                                 | ✅ done        |

What works today:

**CLI** — `build` (from a TOML spec), `ext-build` (bare ext FS from a
directory), `fat-build` (bare FAT32 from a directory), `ls`, `cat`,
`info`, `add` (copy a host file/tree in), `rm` (unlink a file / symlink /
device / empty directory), `shell` (interactive SFTP-style REPL with
`ls`/`cd`/`pwd`/`cat`/`put`/`rm`/`mkdir`/`info`), `convert` (byte-level
raw↔qcow2 conversion with optional grow), `repack` (walk the source
filesystem and rebuild into a fresh image, with `--shrink` for
auto-min sizing and `--fs-type` to convert between FS types — preserves
symlinks, device nodes, mode and uid/gid for ext → ext via a direct
FS-to-FS copier, no host filesystem involvement). ext xattrs round-trip
through repack — both inline (extended-inode-body) source xattrs and
external `file_acl`-block ones are read; the destination always writes
to an external block with a correctly-computed CRC32C when
`metadata_csum` is on. `debugfs ea_get` confirms identical values
after repack. tar is supported as both a source and destination FS
(ustar + PAX, with `SCHILY.xattr.*` records for xattrs), so
`fstool repack disk.img out.tar` and `fstool repack archive.tar
disk.img` round-trip content + mode + uid/gid + mtime + symlinks +
device nodes + xattrs. All
inspection and in-place modification commands accept a `disk.img:N`
(1-indexed) target to walk into a partition of a GPT or MBR disk
image; `fstool info disk.img` (no suffix) prints the partition table.

**Block layer** — `fstool::block::{FileBackend, MemoryBackend,
SlicedBackend, Qcow2Backend}`: sparse file-backed devices, in-memory
devices for tests, bounds-checked sub-range views for carving
partitions, and qcow2 (QEMU's copy-on-write disk format) for both
reading existing v2/v3 images and writing fresh v3 ones with
allocate-on-write. `FileBackend` also handles real block devices on
Unix (`/dev/sdX`, `/dev/nvme0n1`, loop devices): capacity is queried
via the kernel ioctl (`BLKGETSIZE64` on Linux, `DKIOCGETBLOCK*` on
macOS) and the open uses `O_EXCL` so the kernel refuses if any
partition is mounted. The CLI's `ext-build` / `fat-build` require
`--force` when the output is a block device. Path-based factories
(`block::open_image`, `block::create_image`) auto-dispatch by
qcow2 magic / file extension, so qcow2 output is as simple as
`fstool ext-build src -o out.qcow2`.

**Partition tables** — `fstool::part::{Mbr, Gpt}`: write and read
4-primary MBR and 128-entry GPT (protective MBR, primary + backup
headers, CRC32 on header and entry array). Cross-checked against
`sgdisk -v` and `fdisk -l`.

**ext2 / ext3 / ext4** — `fstool::fs::ext::Ext`:
- Write all three: ext2 (no features), ext3 (+ JBD2 journal), ext4
  (+ extent trees, FILETYPE dirents, CRC32C `metadata_csum`). Every
  produced image is verified `e2fsck -fn` clean.
- Read any conforming image, including a stock `mke2fs -t ext4`
  (64-bit 64-byte group descriptors, flex_bg, metadata_csum) — `ls`,
  `cat`, and `info` work on it.
- Streaming populate API (`add_file_to`, `add_dir_to`, `add_symlink_to`
  with fast-symlink inline storage, `add_device_to`); direct +
  single/double-indirect blocks for ext2/3, extent trees for ext4.
- Modify-in-place: open an existing image, add/remove whole files, flush
  — metadata checksums are re-stamped so the result stays fsck-clean.
- `BuildPlan` auto-sizes a filesystem to fit a source tree exactly
  (genext2fs-style "size to fit").
- `Ext::populate_rootdevs` drops a `Minimal` or `Standard` `/dev/*` tree
  (console, null, zero, ptmx, tty, fuse, random, urandom — plus tty0..15,
  ttyS0..3, kmsg, mem, port, hda..hdd + partitions, sda..sdd + partitions
  for `Standard`), so a non-root user can build a Linux root FS without
  CAP_MKNOD.

**FAT32** — `fstool::fs::fat::Fat32`:
- Format a fresh volume (boot sector + backup, FSInfo + backup, two FAT
  copies, root cluster with a mirrored volume-label entry).
- Build from a host directory in one pass — VFAT LFN entries for any
  name that isn't strictly 8.3, a generated `FTxxxxxx` short name as
  the alias, and `.` / `..` in every subdirectory. Symlinks and device
  nodes in the source are skipped (FAT has no representation).
- Files stream cluster-by-cluster, never resident in memory.
- Every produced image is verified `fsck.vfat -n` clean, and `mdir` /
  `mtype` see the populated tree.

**TOML spec** — `fstool::spec`: declarative image descriptions, either a
bare filesystem (`[filesystem]`, ext or FAT32) or a partitioned disk
(`[image]` + `[[partitions]]`, MBR or GPT, ext or FAT32 per partition).

## Architecture

```
                ┌────────────────────────────────────────────┐
                │           CLI (clap) — bin/fstool           │
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

CLI quick tour — build an ext4 image from a directory, then inspect it:

```sh
cargo install fstool                           # or: cargo install --path .
mkdir -p /tmp/src/etc && echo hi > /tmp/src/greeting.txt
fstool ext-build --kind ext4 /tmp/src -o /tmp/out.img
fstool info /tmp/out.img
fstool ls   /tmp/out.img /
fstool cat  /tmp/out.img /greeting.txt
e2fsck -fn  /tmp/out.img                       # must report clean
```

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

Build a partitioned disk image from a TOML spec:

```sh
cat > disk.toml <<'EOF'
[image]
size = "64MiB"
partition_table = "gpt"

[[partitions]]
name = "EFI"
type = "esp"
size = "16MiB"

[[partitions]]
name = "root"
type = "linux"
size = "remaining"

[partitions.filesystem]
type = "ext4"
source = "./rootfs"
EOF
fstool build disk.toml -o disk.img
sgdisk -v disk.img             # "No problems found."
```

## Roadmap

The streaming invariant is the project's load-bearing constraint: regardless
of image size, no file's contents are ever fully resident in memory. The
writer scans the source twice — once to compute geometry (inode count, block
count, dir-entry sizes), once to stream bytes from each file directly to the
image. This is the difference between fstool and a "build a Vec<u8> and dump
it" approach.

In-place modification is restricted to whole-file granularity in v1 (add,
remove, replace). Partial-file rewrites are explicitly out of scope until
there's a use case that demands them.

Not yet implemented: `flex_bg` on the ext4 *write* path (the reader
handles it).

## Licence

MIT. Copyright © 2026 Karpelès Lab Inc. See [LICENSE](LICENSE).
