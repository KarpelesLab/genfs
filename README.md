# fstool

[![CI](https://github.com/KarpelesLab/fstool/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/fstool/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/fstool.svg)](https://crates.io/crates/fstool)
[![docs.rs](https://docs.rs/fstool/badge.svg)](https://docs.rs/fstool)

Build, inspect, modify, and repack disk images and filesystem images.
In the spirit of `genext2fs`, but covering whole disks, multiple filesystems,
and round-tripping between formats — all from a TOML spec or directly from
the command line.

fstool ships as a Rust library (`fstool`) plus a thin CLI binary (`fstool`).
Public API is **unstable** until v0.5.

```sh
cargo install fstool
fstool ext-build --kind ext4 ./src -o out.img    # build an ext4 image from a dir
fstool info out.img                              # what's inside
fstool ls   out.img /                            # walk it
fstool repack out.img out.tar                    # convert ext4 → tar (and back)
```

## Filesystem support

| Filesystem | Read           | Write | Notes                                              |
|------------|----------------|-------|----------------------------------------------------|
| ext2       | ✅              | ✅     | byte-exact with `genext2fs` on the same input      |
| ext3       | ✅              | ✅     | + JBD2 journal                                     |
| ext4       | ✅              | ✅     | extents, FILETYPE, `metadata_csum`, xattrs         |
| FAT32      | ✅              | ✅     | VFAT LFN entries, 8.3 short-name aliases           |
| exFAT      | ✅              | ✅     | format + create + remove + flush                   |
| tar        | ✅              | ✅     | ustar + PAX, `SCHILY.xattr.*` for xattrs           |
| XFS        | ✅ (shortform + block / leaf / node dirs + BMBT) | —     | + inline + remote symlinks; B-tree dirs deferred   |
| HFS+ / HFSX| ✅              | —     | inline + extents-overflow, symlinks, hard links    |
| APFS       | ✅              | —     | multi-level omap + fs-tree; no snapshots / crypto  |
| NTFS       | ✅              | —     | MFT, attributes, $DATA + ADS, indexes; xattr map   |
| F2FS       | ✅              | —     | CP / NAT / dnodes / inline data + dentries         |
| SquashFS   | ✅ (uncompressed) | —     | gzip/xz/lz4/zstd refused until compression lands   |
| qcow2      | ✅              | ✅     | v2 + v3, allocate-on-write writer                  |

The reader for each FS streams: file contents are never fully resident in
memory regardless of size. The writers do the same, two-pass: scan to size
the geometry, then stream bytes from each source file into the image.

NTFS metadata that has no POSIX analogue (DOS attributes, ADS, security
descriptors, NT-FILETIME timestamps, short names, reparse data) round-trips
through xattrs under `user.ntfs.*` and `system.ntfs_security`.

## CLI commands

| Command       | What it does                                                            |
|---------------|-------------------------------------------------------------------------|
| `build`       | Build from a TOML spec — bare FS or a partitioned disk image.           |
| `ext-build`   | Bare ext2 / ext3 / ext4 image from a host directory tree.               |
| `fat-build`   | Bare FAT32 image from a host directory tree.                            |
| `info`        | Print partition table (whole-disk) or FS summary + root listing.        |
| `ls`          | List a directory inside an image.                                       |
| `cat`         | Stream a file's bytes out of an image to stdout.                        |
| `add`         | Copy a host file / tree into an existing image (ext or FAT).            |
| `rm`          | Unlink a file, symlink, device, or empty directory.                     |
| `shell`       | SFTP-style REPL — `ls cd pwd cat put rm mkdir info`.                    |
| `convert`     | Byte-level raw ↔ qcow2 conversion with optional grow.                   |
| `repack`      | Walk source FS, rebuild into a fresh image, optionally a different FS.  |
| `fstool …`    | Plus `ext-build`, `fat-build`, partition-aware `disk.img:N` targets.    |

All inspection / modification commands accept a `disk.img:N` (1-indexed)
target to walk into a partition of a GPT or MBR disk image. `fstool info
disk.img` without the suffix prints the partition table itself.

## Partitions, block devices, qcow2

- **Partition tables** — MBR (4 primaries) and GPT (128-entry, CRC32 on
  header + entry array, primary + backup, protective MBR). Cross-checked
  against `sgdisk -v` and `fdisk -l`.
- **Block devices** — on Unix, fstool can format and mutate real block
  devices (`/dev/sdX`, `/dev/nvme0n1`, loop devices). Capacity is queried via
  the kernel ioctl (`BLKGETSIZE64` on Linux, `DKIOCGETBLOCK*` on macOS) and
  open uses `O_EXCL` so the kernel refuses if any partition is mounted.
  Build commands require `--force` when the output is a block device.
- **qcow2** — `Qcow2Backend` reads QEMU v2 / v3 images and writes fresh v3
  ones with allocate-on-write. Path-based factories (`block::open_image`,
  `block::create_image`) auto-dispatch by qcow2 magic or file extension, so
  `fstool ext-build src -o out.qcow2` Just Works.

## TOML spec

Declarative image descriptions — either a bare filesystem (`[filesystem]`)
or a partitioned disk (`[image]` + `[[partitions]]`):

```toml
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
```

```sh
fstool build disk.toml -o disk.img
sgdisk -v disk.img             # "No problems found."
```

## Architecture

```
              ┌────────────────────────────────────────────┐
              │           CLI (clap) — bin/fstool          │
              └────────────────────────────────────────────┘
                                  │
              ┌────────────────────────────────────────────┐
              │  Spec layer (TOML → ImageSpec / FsSpec)    │
              └────────────────────────────────────────────┘
                                  │
              ┌────────────────────────────────────────────┐
              │  Filesystem trait → ext, fat, xfs, ntfs, … │
              └────────────────────────────────────────────┘
                                  │
              ┌────────────────────────────────────────────┐
              │  PartitionTable trait → Mbr, Gpt           │
              └────────────────────────────────────────────┘
                                  │
              ┌────────────────────────────────────────────┐
              │  BlockDevice trait → File, Mem, Sliced,    │
              │                       Qcow2                │
              └────────────────────────────────────────────┘
```

Each layer is substitutable. A filesystem implementation talks only to a
`BlockDevice`; it doesn't know or care whether the device is a real file,
an in-memory buffer in a test, a slice carved out of a larger disk by a
partition table, or a qcow2-backed sparse container.

## ext-specific niceties

- `BuildPlan` auto-sizes a filesystem to fit a source tree exactly
  (genext2fs-style "size to fit").
- `Ext::populate_rootdevs` drops a `Minimal` or `Standard` `/dev/*` tree
  (console, null, zero, ptmx, tty, fuse, random, urandom — plus tty0..15,
  ttyS0..3, kmsg, mem, port, hda..hdd, sda..sdd + partitions for
  `Standard`), so a non-root user can build a Linux root FS without
  `CAP_MKNOD`.
- xattrs round-trip through repack: both inline (extended-inode-body) and
  external `file_acl`-block sources are read; the destination writes to an
  external block with a correctly-computed CRC32C when `metadata_csum` is on.
  `debugfs ea_get` confirms identical values after repack.

## Cross-FS repack

`fstool repack` walks the source filesystem and rebuilds the tree into a
fresh image. With `--fs-type` it changes filesystem on the fly; `--shrink`
auto-sizes the output to the minimum that fits the content. The ext → ext
path uses a direct FS-to-FS copier (no host-filesystem intermediation),
preserving symlinks, device nodes, mode, uid/gid, and xattrs. tar in either
direction round-trips content + mode + uid/gid + mtime + symlinks + device
nodes + xattrs.

For the read-only filesystems (XFS, HFS+, APFS, NTFS, F2FS, SquashFS),
repack works **from** them. Repacking **to** them isn't supported until
their writers land.

## Limitations

Things explicitly out of scope today, in rough order of likely-to-change:

- SquashFS compression decode (needs flate2 / zstd / xz / lz4 dependencies).
  Compressed blocks return a clean `Unsupported` naming the algorithm.
- NTFS / F2FS / XFS / APFS / HFS+ writers.
- NTFS compressed and encrypted `$DATA`, `$ATTRIBUTE_LIST` spill, `$Secure`
  security-descriptor indirection — all return `Unsupported`.
- APFS snapshots, encryption, sealed-volume integrity.
- XFS B-tree-format directories (block / leaf / node formats are covered).
- ext4 `flex_bg` on the *write* path (the reader handles it).
- Partial-file rewrites — in-place modification is whole-file granularity.

## Try it

```sh
cargo install fstool                          # or: cargo install --path .
mkdir -p /tmp/src/etc && echo hi > /tmp/src/greeting.txt
fstool ext-build --kind ext4 /tmp/src -o /tmp/out.img
fstool info /tmp/out.img
fstool ls   /tmp/out.img /
fstool cat  /tmp/out.img /greeting.txt
e2fsck -fn  /tmp/out.img                      # must report clean
```

Run the test suite:

```sh
cargo test                    # unit tests + external cross-checks if tools present
```

CI runs the full suite on Linux (with `apt`-installed `e2fsprogs`,
`dosfstools`, `mtools`, `gdisk`, `qemu-utils` for cross-validation) plus a
build + test pass on macOS (Homebrew `qemu`) and Windows.

## Licence

MIT. Copyright © 2026 Karpelès Lab Inc. See [LICENSE](LICENSE).
