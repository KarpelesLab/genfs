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
fstool create -t ext4 ./src -o out.img           # build an ext4 image from a dir
fstool create -t squashfs ./src -o out.sqsh \
       -O compression=zstd,block_size=128KiB     # FS-specific knobs via -O
fstool info out.img                              # what's inside
fstool ls   out.img /                            # walk it
fstool repack out.img out.tar                    # convert ext4 → tar (and back)
fstool repack base.tar patch.tar flat.tar        # OCI-style layer merge with .wh.* whiteouts
```

## Filesystem support

| Filesystem | Read | Write | In-place edits | Notes                                                                                                              |
|------------|------|-------|----------------|--------------------------------------------------------------------------------------------------------------------|
| ext2       | ✅    | ✅     | ✅              | byte-exact with `genext2fs` on the same input                                                                      |
| ext3       | ✅    | ✅     | ✅              | + JBD2 journal — real transactions on `open_file_rw` (Path A)                                                      |
| ext4       | ✅    | ✅     | ✅              | extents (read + write: any depth), FILETYPE, `metadata_csum`, xattrs, JBD2                                         |
| FAT32      | ✅    | ✅     | ✅              | VFAT LFN entries, 8.3 short-name aliases                                                                           |
| exFAT      | ✅    | ✅     | ✅              | format + create + remove + flush + `open_file_rw`                                                                  |
| tar        | ✅    | ✅     | —              | ustar + PAX, `SCHILY.xattr.*` for xattrs; streaming-only                                                           |
| XFS        | ✅    | ✅     | ✅              | shortform + block / leaf / node + multi-level B-tree dirs + BMBT; leaf-form xattrs; real XLOG transactions (Path A); passes `xfs_repair -n` single + multi-AG |
| HFS+/HFSX  | ✅    | ✅     | ✅              | inline + extents-overflow, symlinks, hard links; decmpfs read (zlib types 3 + 4); real journal (Path A); passes `fsck.hfsplus` |
| APFS       | ✅    | ✅     | 🚧             | multi-level omap + fs-tree; spaceman with IP ring + SFQ free-queues; `open_file_rw` rebuilds a fresh COW checkpoint (whole-file overwrite only — no partial-extent COW yet); not yet `fsck_apfs` clean |
| NTFS       | ✅    | ✅     | ✅              | MFT, attributes, $DATA + ADS, indexes; xattr map; multi-class `$Secure` ($SDS/$SDH/$SII); real `$LogFile` LFS records (Path A) |
| F2FS       | ✅    | ✅     | —              | CP / NAT / dnodes / inline data + dentries; writer passes `fsck.f2fs`; **build-once** — the writer serializes the whole FS from memory at flush, so a re-opened image is read-only (reports `Immutable`) |
| SquashFS   | ✅    | ✅     | —              | gzip / xz / lz4 / zstd / lzo / lzma via Cargo features; writer round-trips via `unsquashfs`; repack-only           |
| ISO 9660   | ✅    | ✅     | —              | PVD + Joliet (UCS-2) + Rock Ridge (PX/NM/SL/TF) + El Torito boot catalog; repack-only                              |
| GRF        | ✅    | ✅     | ✅              | Gravity Ragnarok Online archive — v0x102 / v0x103 / v0x200; permutation cipher (`MIXCRYPT` / `DES`); CP949 filenames |
| zip        | ✅    | ✅     | —              | central-directory index, ZIP64, Stored + Deflate, Unix mode/symlinks, UTF-8/Shift-JIS/EUC-JP filename detection; repack-only writer |
| cpio       | ✅    | ✅     | —              | newc / newc-crc / odc read; newc write; repack-only                                                              |
| ar         | ✅    | ✅     | —              | GNU + BSD long names (read), GNU write; flat (no directories); repack-only                                       |
| cab        | ✅    | —     | —              | Microsoft Cabinet read-only: Store / LZX / Quantum / single-block MSZIP folders decode via `compcol` (cross-checked with `cabextract`). Multi-block MSZIP, spanned cabinets, and creation are unsupported |
| 7z / rar / arc / lha / lzx / sit | 🚧 | — | — | detected by `info`; reader not implemented yet (returns a clear `Unsupported`) — pure-Rust decoders land behind per-format Cargo features |

`🚧` marks writers / mutation paths with known gaps (see Limitations).
All writable filesystems — ext2/3/4, FAT32, exFAT, XFS, HFS+, NTFS,
APFS, F2FS, SquashFS, ISO 9660, GRF — implement a single
`Filesystem` trait, so the CLI (`build`, `repack`, `add`, `rm`) and
the TOML `[filesystem] type = "…"` spec dispatch through one
codepath; pick a target FS by setting `--fs-type` on `repack` or
`type = "hfsplus"` (etc.) in the TOML spec. "In-place edits"
means an already-flushed image can be re-opened for `add` / `rm` /
`open_file_rw` — for filesystems with a journal, that path commits
through a real transaction so a crash mid-write leaves an image the
host's `fsck` can replay.

`qcow2` and `dmg` are **not** in the table above: they aren't
filesystems but *disk-image containers*. They live one layer down, as
`BlockDevice` backends (see the architecture diagram and "Partitions,
block devices, qcow2"), presenting a flat byte-addressable device that
any of the filesystems above is then laid down *inside* — fstool reads
and writes through them transparently. qcow2 is read/write (v2 + v3,
allocate-on-write); dmg is read-only (UDIF v4 mish chunks: zero / raw /
zlib / ADC / bzip2 / LZFSE / LZMA, plus encrypted v2 `encrcdsa`).

The reader for each FS streams: file contents are never fully resident in
memory regardless of size. The writers do the same, two-pass: scan to size
the geometry, then stream bytes from each source file into the image.

NTFS metadata that has no POSIX analogue (DOS attributes, ADS, security
descriptors, NT-FILETIME timestamps, short names, reparse data) round-trips
through xattrs under `user.ntfs.*` and `system.ntfs_security`.

## CLI commands

| Command       | What it does                                                            |
|---------------|-------------------------------------------------------------------------|
| `create`      | Build a bare image of any supported FS (`-t ext4` / `fat32` / `xfs` / `hfs+` / `ntfs` / `f2fs` / `squashfs` / `iso` / `apfs` / `exfat` / `grf` / `zip` / `cpio` / `ar`) from a host directory tree. FS-specific knobs go through `-O key=val,key=val`. |
| `build`       | Build from a TOML spec — bare FS or a partitioned disk image.           |
| `info`        | Print partition table (whole-disk) or FS summary + root listing.        |
| `ls`          | List a directory inside an image.                                       |
| `cat`         | Stream a file's bytes out of an image to stdout.                        |
| `add`         | Copy a host file / tree into an existing image (any mutable FS).        |
| `rm`          | Unlink a file, symlink, device, or empty directory.                     |
| `shell`       | SFTP-style REPL — `ls cd pwd cat put rm mkdir info`.                    |
| `convert`     | Byte-level raw ↔ qcow2 conversion with optional grow.                   |
| `repack`      | Walk one or more source FSes, merge bottom→top with whiteouts, rebuild into a fresh image. |

All commands accept partition-aware `disk.img:N` targets (1-indexed) — see
"Partitions, block devices, qcow2" below.

All inspection / modification commands accept a `disk.img:N` (1-indexed)
target to walk into a partition of a GPT or MBR disk image. `fstool info
disk.img` without the suffix prints the partition table itself.

### FS-specific options (`-O`)

Most filesystems expose tunables (block size, label, compression codec,
volume name, journaling on/off, etc.) through a generic `-O
key=value,key=value` flag that is repeatable, modelled on `mke2fs -O`:

```sh
# 4 KiB blocks + custom label on ext4
fstool create -t ext4 ./rootfs -o out.img -O block_size=4096,volume_label=ROOT

# Pick a SquashFS codec and tighten the block size
fstool create -t squashfs ./rootfs -o out.sqsh \
       -O compression=zstd,block_size=128KiB

# Force a v0x103 GRF with deflate level 9
fstool create -t grf ./rootfs -o out.grf -O version=0x103,compression_level=9
```

Each backend's `apply_options` validates keys; unknown keys are rejected
with a clear error citing the FS type. The same options are available
through the TOML spec — see "[filesystem.options]" below.

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
  `fstool create -t ext4 src -o out.qcow2` Just Works.

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

### `source` — what to populate the FS with

`source` accepts three shapes, auto-detected by what the string points at:

```toml
[partitions.filesystem]
type = "ext4"
source = "./rootfs"            # a host directory — walk it recursively
```

```toml
[partitions.filesystem]
type = "ext4"
source = "./rootfs.tar.gz"     # a tar archive — repack entries into the FS
```

```toml
[partitions.filesystem]
type = "ext4"
source = "./old-disk.img:2"    # an existing image, optional :N partition
                               # — walks the source FS, copies every
                               # entry into the new partition
```

Recognised tar extensions: `.tar`, `.tar.gz`, `.tgz`, `.tar.xz`, `.txz`,
`.tar.zst`, `.tar.lz4`, `.tar.lzma`, `.tar.lzo` (codecs gated on the
matching Cargo feature). For images, the `:N` suffix selects partition
*N* (1-indexed); without it, the source is opened as a bare filesystem.
The source FS may be any readable type — `ext{2,3,4}`, FAT32, exFAT,
XFS, HFS+, APFS, NTFS, F2FS, SquashFS, ISO 9660, tar, or GRF — and the
destination is sized automatically to fit unless `size` is set
explicitly.

### `[filesystem.options]` — FS-specific tunables

The same `-O key=val` knobs the CLI exposes are available in TOML
through a free-form `[filesystem.options]` table:

```toml
[filesystem]
type = "squashfs"
source = "./rootfs"

[filesystem.options]
compression = "zstd"
block_size  = 131072

[partitions.filesystem]
type = "ext4"
source = "./rootfs"

[partitions.filesystem.options]
block_size   = 4096
volume_label = "ROOT"
```

Recognised keys are documented next to each backend's
`FormatOpts::apply_options`; unknown keys are rejected at spec parse
time with a clear error citing the FS type. The existing flat fields
(`block_size`, `volume_label`, `mtime`, …) continue to work for
backward compatibility.

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
              │                       Qcow2, Dmg           │
              └────────────────────────────────────────────┘
```

Each layer is substitutable. A filesystem implementation talks only to a
`BlockDevice`; it doesn't know or care whether the device is a real file,
an in-memory buffer in a test, a slice carved out of a larger disk by a
partition table, or a qcow2-backed sparse container. DMG (`.dmg`) is
treated the same way: open the image, walk the mish table for the
chunk layout, and the rest of the stack reads through it as if it were
a flat block device — including the encrypted (`encrcdsa` v2) variant
when an unlock password is supplied.

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
auto-sizes the output to the minimum that fits the content.

The pipeline is **one generic walker feeding one of two sinks** — a
streaming-tar sink (tar / `.tar.<codec>`) or a block-device `Filesystem`
sink — with no per-`(source,dest)`-type special cases. So **any readable
source repacks into any writable destination** through a single path
(`fstool repack app.zip out.tar`, `fstool repack disk.xfs out.iso`, …).
The walker reads each entry's metadata through the source's trait
`getattr` / `list_xattrs` / `read_symlink`, so mode, uid/gid, mtime,
symlinks, device numbers, xattrs, and hard links round-trip wherever both
ends can represent them. File bodies stream straight from source to
destination (`create_file_streaming`, no per-file tempfile). Hard links
are de-duplicated when the destination supports them (ext) and
materialised as copies otherwise (tar, FAT, …); a destination that can't
hold a symlink/device/xattr (FAT) drops it with a warning.

Every reader surfaces the metadata its format actually stores:
ext, tar, the archive formats, F2FS, XFS, SquashFS, APFS, and HFS+ carry
full POSIX mode/uid/gid + timestamps (HFS+ converts its 1904 epoch);
ISO 9660 does too when Rock Ridge is present (plain/Joliet have none);
NTFS — which has no POSIX ownership — surfaces real timestamps + a mode
synthesised from its DOS attributes, and carries its native metadata
(DOS attrs, ADS, security descriptor, reparse data, …) through repack as
`user.ntfs.*` / `system.ntfs_security` xattrs.

`fstool repack` writes any destination implementing the `Filesystem`
trait — `ext2/3/4`, FAT32, exFAT, tar, XFS, HFS+, APFS, NTFS, F2FS,
SquashFS, ISO 9660, GRF. `add` / `rm` go through the same trait,
which means they work on any FS whose writer can re-open an existing
image; today that's all of the mutable backends — ext, FAT32, exFAT,
F2FS, XFS, HFS+, NTFS, APFS, and GRF. SquashFS, ISO 9660, and tar
are repack-only (their `MutationCapability` is `Immutable` or
`Streaming`, so `add` / `rm` fail fast with an actionable error and
the user is steered to `repack`).

## Layered merge with whiteouts

`repack` takes one or more source positional arguments followed by the
destination. With one source it behaves as before; with two or more
it merges the sources bottom→top before writing — later layers
override files of the same path, and tombstones from the upper
layer remove paths from the lower one. Two tombstone conventions are
auto-detected:

| Convention | Marker | Effect |
|------------|--------|--------|
| tar-OCI    | `.wh.<name>` in directory D | delete `D/<name>` |
| tar-OCI    | `.wh..wh..opq` in directory D | drop all lower-layer children of D before this layer's own land |
| OverlayFS  | character device with major=0, minor=0 | delete this path |
| OverlayFS  | xattr `trusted.overlay.opaque = "y"` on a dir | opaque-dir semantics on that dir |

The tombstones themselves never appear in the output. Sources may be
host directories, tar archives (compressed or plain), or filesystem
images — any mix works.

```sh
# OCI-style: rebuild a stack of layers into a flat tar
fstool repack base.tar layer1.tar layer2.tar flat.tar

# Patch an ISO with a tar of replacement files
fstool repack disc.iso patch.tar updated.iso --fs-type iso

# Shell globs work — last positional is the destination
fstool repack layer*.tar merged.tar
```

Internally the merge folds all layers into a single uncompressed tar
held in a tempfile, then drives the existing single-source repack
pipeline; the destination FS doesn't know it came from multiple
sources.

## ISO 9660

ISO 9660 reads cover the bare ECMA-119 layout plus three of the four
common extensions:

- **Joliet** (Microsoft) — UCS-2 BE long names via the supplementary
  volume descriptor.
- **Rock Ridge** (IEEE P1282) — POSIX mode + uid + gid via `PX`, long
  names via `NM`, symlinks via `SL`, timestamps via `TF`. Continuation
  areas (`CE`) are followed across sector boundaries.
- **El Torito** — boot catalog: validation entry, default entry, and
  section headers (`0x90` / `0x91`); the parsed catalog is surfaced
  in `fstool info`.

The writer is repack-only — ISO is sequential and a single `flush()`
writes the whole image. It emits a PVD plus optional Joliet SVD,
both L-type and M-type path tables, dual directory record trees (one
for PVD, one for Joliet), and Rock Ridge System Use Areas (`NM` /
`PX` / `SL`) attached to the PVD records. The output round-trips
through `isoinfo -lR` and back through fstool's own reader.

```sh
# Build an ISO from a host directory
fstool repack ./rootfs disc.iso --fs-type iso

# Walk an existing ISO
fstool ls   disc.iso /
fstool cat  disc.iso /README.TXT

# Round-trip ISO → tar → ISO
fstool repack disc.iso plain.tar
fstool repack plain.tar disc2.iso --fs-type iso
```

## Archive formats

Archives are treated as filesystems through the same `Filesystem` trait as
tar and GRF, so `info` / `ls` / `cat` / `repack` work on them uniformly. They
share a common core (`src/fs/archive/`): each format supplies a *scanner* that
indexes the archive into an in-memory tree, and — where writable — a *builder*;
the core provides the generic read path and decodes each entry's byte range
through the existing compression codecs.

```sh
fstool create -t zip ./rootfs -o out.zip          # build a zip from a dir
fstool create -t zip ./rootfs -o out.zip -O compression=stored
fstool ls   app.zip /                             # walk any zip/cpio/ar
fstool cat  app.zip /etc/config
fstool repack app.zip out.cpio --fs-type cpio     # convert between archives
```

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| zip    | ✅    | ✅     | ZIP64, Stored + Deflate, Unix mode + symlinks; reads archives from any tool; filenames decoded as UTF-8 (flagged) else auto-detected (Shift-JIS / EUC-JP / Latin-9). On write the UTF-8 flag is set only for non-ASCII names. |
| cpio   | ✅    | ✅     | newc / newc-crc / odc read; newc write. |
| ar     | ✅    | ✅     | GNU + BSD long names on read, GNU on write. Flat — a nested source tree is rejected with a pointer to tar/zip/cpio. |

The writers are repack-only (`MutationCapability::Streaming`, like tar): an
existing archive can't be edited in place — `add` / `rm` steer you to
`repack`, which rebuilds. `cab` is a read-only reader (Store / LZX / Quantum
/ single-block MSZIP via `compcol`, behind the `cab` feature). `7z`, `rar`,
`arc`, `lha`, `lzx`, and `sit` are recognised by `info` today but their
readers are scaffolds that return a clear `Unsupported`; pure-Rust decoders
will land behind per-format Cargo features. (`rar` and `sit` are read-only-at-best — their creation is
proprietary.)

zip's Deflate support rides the existing `gzip` Cargo feature (raw DEFLATE via
`flate2`); a build without it falls back to Stored. `cpio` and `ar` need no
codec. Archive-to-`ext`/`fat`/`tar` repack uses the specialised FS-to-FS
copiers and isn't wired yet (same limitation as XFS/HFS+ sources) — convert
between archives, or to `iso`/`grf`, via the generic trait path.

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

- **ext4 write path**: `flex_bg` on the write path (reader is fine).
- **APFS in-place edits**: `open_file_rw` rebuilds a fresh COW
  checkpoint over the entire file content, so it's whole-file
  granularity — partial-extent COW is not yet implemented, and
  `create_file` / `remove` over the rw path piggyback on the same
  checkpoint. Multiple back-to-back commits are bounded by the
  `xp_desc` ring (the reader doesn't rotate it yet).
- **APFS reader**: snapshots, encryption, and sealed-volume integrity
  are out of scope.
- **APFS / NTFS strict-checker pass**: the spaceman + `$Secure` /
  `$LogFile` structures are now populated, but `fsck_apfs` and
  `ntfs-3g` mount can still flag the images for finer points
  (free-queue B-trees, journal metadata layout). Read + write work
  end-to-end; the host-tool gate is the remaining polish.
- **NTFS reader**: compressed and encrypted `$DATA`, `$ATTRIBUTE_LIST`
  spill, and security-descriptor indirection through `$Secure`
  beyond what the resident path handles all return `Unsupported`.
- **XFS reader**: B-tree-format (`di_format=BTREE`) directories
  deeper than one level above the leaves return `Error::Unsupported`
  (shortform / block / leaf / node and single-level B-tree dirs are
  covered); writer assumes shortform / extent dirs. Node-form
  (multi-leaf dabtree) xattrs are read-only.
- **HFS+ decmpfs**: type 3 (zlib inline) + type 4 (zlib resource
  fork) work. LZVN (types 7/8) and LZFSE (types 11/12) return
  `Unsupported`.
- **DMG**: read-only — no DMG writer / `convert` path. Encrypted v1
  (`cdsaencr` legacy 3DES) chunks return `Unsupported`; v2 is
  covered.
- **Partial-file rewrites** on the trait surface — `open_file_rw`
  exists everywhere it's safe, but a typed "patch this byte range
  on a known-large file" API is not surfaced beyond `Read + Write +
  Seek` on the handle.

## Try it

```sh
cargo install fstool                          # or: cargo install --path .
mkdir -p /tmp/src/etc && echo hi > /tmp/src/greeting.txt
fstool create -t ext4 /tmp/src -o /tmp/out.img
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
