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

| Filesystem | Read | Write | Notes                                                                |
|------------|------|-------|----------------------------------------------------------------------|
| ext2       | ✅    | ✅     | byte-exact with `genext2fs` on the same input                        |
| ext3       | ✅    | ✅     | + JBD2 journal                                                       |
| ext4       | ✅    | ✅     | extents, FILETYPE, `metadata_csum`, xattrs                           |
| FAT32      | ✅    | ✅     | VFAT LFN entries, 8.3 short-name aliases                             |
| exFAT      | ✅    | ✅     | format + create + remove + flush                                     |
| tar        | ✅    | ✅     | ustar + PAX, `SCHILY.xattr.*` for xattrs                             |
| XFS        | ✅    | ✅     | shortform + block / leaf / node dirs + BMBT; writer passes `xfs_repair -n` single + multi-AG; B-tree dirs deferred |
| HFS+/HFSX  | ✅    | ✅     | inline + extents-overflow, symlinks, hard links; writer passes `fsck.hfsplus` with optional journal stub |
| APFS       | ✅    | 🚧    | multi-level omap + fs-tree; writer is single-volume + stub spaceman (no snapshots / encryption); not yet `fsck_apfs` clean |
| NTFS       | ✅    | 🚧    | MFT, attributes, $DATA + ADS, indexes; xattr map; writer passes `ntfsfix --no-action` but isn't `ntfs-3g`-mountable yet (system files in root `$I30` pending) |
| F2FS       | ✅    | ✅     | CP / NAT / dnodes / inline data + dentries; writer passes `fsck.f2fs` |
| SquashFS   | ✅    | ✅     | gzip / xz / lz4 / zstd / lzo / lzma via Cargo features; writer round-trips via `unsquashfs` |
| qcow2      | ✅    | ✅     | v2 + v3, allocate-on-write writer                                    |
| dmg        | 🚧   | —     | UDIF v4 trailer parsed; chunk decoder TBD                            |

`🚧` marks writers that exist at the library level but have known gaps
(see Limitations). The high-level CLI only writes ext2/3/4, FAT32, and
tar today — the other writers are reachable via the library API.

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

`fstool repack` writes ext2/3/4, FAT32, and tar destinations today;
the XFS / HFS+ / APFS / NTFS / F2FS / SquashFS writers exist as a
library API (see the support table) but aren't wired into `repack` /
`add` / `rm` yet — call them directly through the crate for now.

## Compression

`fstool` ships with six compression codecs enabled by default. Each has
its own Cargo feature flag so you can trim the binary down:

| Codec | Feature | Used for |
|-------|---------|----------|
| gzip  | `gzip`  | SquashFS, `.tar.gz` / `.tgz` |
| xz    | `xz`    | SquashFS, `.tar.xz` / `.txz` |
| lzma  | `lzma`  | SquashFS, `.tar.lzma` |
| lz4   | `lz4`   | SquashFS, `.tar.lz4` |
| zstd  | `zstd`  | SquashFS, `.tar.zst` |
| lzo   | `lzo`   | SquashFS, `.tar.lzo` |

Compressed tar input / output is detected by filename extension (or by
magic for inputs without a recognisable extension): `fstool ls
disk.tar.zst /` and `fstool repack ext.img out.tar.gz` Just Work.
Internally the codec is streamed through a temp file so the whole
archive is never resident in RAM.

To disable a codec at build time, e.g. to avoid the bundled C `zstd`
build on a constrained system:

```sh
cargo install fstool --no-default-features --features gzip,lz4,xz,lzma
```

## Limitations

Things explicitly out of scope today, in rough order of likely-to-change:

- High-level CLI (`build`, `add`, `rm`, `repack`) only writes ext2/3/4,
  FAT32, and tar. The library writers for XFS / HFS+ / APFS / NTFS /
  F2FS / SquashFS work, just not via the CLI.
- NTFS writer: produced image isn't `ntfs-3g`-mountable — root `$I30`
  doesn't index the system files yet; `ntfsfix --no-action` is clean.
- NTFS reader: compressed and encrypted `$DATA`, `$ATTRIBUTE_LIST`
  spill, and `$Secure` security-descriptor indirection beyond what
  the resident path handles all return `Unsupported`.
- APFS writer: single volume, stub space-manager → `fsck_apfs`
  flags the spaceman; `mount_apfs` typically refuses the image.
- APFS reader: snapshots, encryption, and sealed-volume integrity
  are out of scope.
- XFS reader: B-tree-format directories (block / leaf / node formats
  are covered); writer assumes shortform / extent dirs.
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
