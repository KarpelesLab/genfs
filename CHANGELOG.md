# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.6](https://github.com/KarpelesLab/fstool/compare/v0.4.5...v0.4.6) - 2026-05-27

### Added

- *(merge)* hard links + fix(fat): flush dir batches before read

### Fixed

- *(doc)* resolve merge.rs intra-doc links for `cargo doc -D warnings`
- *(repack)* don't strip Windows drive letters from tar paths

### Other

- *(repack)* unify plain + compressed tar arms in walk_source_into_sink
- *(cli)* stream plain tar sources too — kill the random-access Tar::open
- *(ext)* O(1) data-block allocator via per-group cursor
- *(fat32)* O(1) child_exists via per-parent name index
- *(iso9660)* tree children → BTreeMap, kills O(n²) insert + lookup
- *(f2fs)* lazy `i_addr` Vec — 8× RAM cut on bulk-insert workloads

## [0.4.5](https://github.com/KarpelesLab/fstool/compare/v0.4.4...v0.4.5) - 2026-05-26

### Other

- *(merge)* in-memory model + per-source ordered emission, no tempfile
- *(repack)* stream tar into zip/cpio + tar→tar, drop archive temp files
- *(repack)* stream compressed tar into squashfs/iso/grf, no tempfile
- *(iso9660)* stream file data to the device, no temp file, bounded RAM
- *(grf)* stream body into the archive directly, no temp file
- *(squashfs)* stream file data to the device, no temp files
- *(clone)* buffer small clones in memory instead of a temp file

## [0.4.4](https://github.com/KarpelesLab/fstool/compare/v0.4.3...v0.4.4) - 2026-05-25

### Fixed

- *(ntfs)* size resident $DATA by actual $SI/$FN length (fuzz panic)

### Other

- *(repack)* stop spilling every streamed file to a temp file
- *(hfs+)* bump-cursor allocation — drop O(n²) from large-dir builds
- *(f2fs)* O(1) directory lookups — drop O(n²) from large-dir builds

## [0.4.3](https://github.com/KarpelesLab/fstool/compare/v0.4.2...v0.4.3) - 2026-05-25

### Added

- *(f2fs)* hashed multi-level directories — large dirs pass fsck.f2fs
- *(hfs+)* grow catalog B-tree + correct clump size — 100k files clean
- *(xfs)* 2-level INOBT — 100k+ files in one directory pass xfs_repair
- *(xfs)* leaf + node directories and aligned inode chunks (to ~16k files)
- *(ext4)* incremental depth-2 extent growth for large directories
- *(ext4)* depth-N extent trees + journal/flex_bg sizing for large dirs
- *(analyze)* generic source-analysis API + `fstool analyze` command
- *(repack)* stream compressed-tar sources — no decompress-to-tempfile
- *(repack)* phase markers + wire up the per-file progress counter
- *(shell)* `info <path>` dumps per-file metadata + xattrs

### Fixed

- *(xfs)* escape `bestfree[0]` in doc comment to unbreak cargo doc
- *(xfs)* clean error instead of panic on block-dir overflow
- *(ntfs)* scale directories + $MFT to 100k files (clean ntfs-3g mount)
- *(ext4)* one-shot build path promotes to depth-1 extent tree

### Other

- *(f2fs)* mark large-directory test ignored — known writer limitation
- *(f2fs)* large-directory guard (read-back local, fsck.f2fs in CI)
- *(ntfs)* external scale guard — 4000-file dir mounts ntfsfix-clean
- *(exfat)* batch directory writes via DirBatch + lookup overlay
- *(fat)* batch directory writes via DirBatch + lookup overlay
- *(xfs)* batch directory writes via DirBatch + lookup overlay
- *(ntfs,ext)* batch directory writes; add shared DirBatch cache
- *(squashfs)* multithread block compression by default
- *(repack)* gate compressed-tar stream test to Unix

## [0.4.2](https://github.com/KarpelesLab/fstool/compare/v0.4.1...v0.4.2) - 2026-05-25

### Added

- *(xfs)* refuse open_file_rw on REFLINK files — prevent clone corruption (Phase 3b stage 3)
- *(xfs)* clone_file via shared extents + REFCNTBT records (Phase 3b stage 2)
- *(xfs)* REFLINK feature opt-in + per-AG REFCNTBT root (Phase 3b stage 1)
- *(fs)* clone API — Filesystem::clone_file / clone_range + CloneCapability (Phase 3a)
- *(ntfs)* create_device for char/block via INTX_FILE; sort $I30 entries
- *(hfs+)* create_device — char / block / FIFO / socket nodes
- *(ntfs)* implement remove (file / empty-dir / symlink), the inverse of create
- *(ntfs)* make a reopened image mutable (lazy writer reconstruction)
- *(ntfs)* getattr (times + synthesised mode) and list_xattrs
- *(hfs+)* faithful getattr
- *(iso9660)* faithful getattr from Rock Ridge
- *(apfs)* faithful getattr
- *(archive)* shared archive core + zip/cpio/ar backends, 7 scaffolds

### Fixed

- *(fs)* owned-tempfile FileSource for deferred-write backends; SquashFS getattr

### Other

- fix 5 broken intra-doc links + BSD-ar cross-check on macOS
- *(dmg)* end-to-end against hdiutil on macOS (UDRW / UDZO / UDBZ / ULFO)
- *(fuzz)* NTFS fuzz target + Op::Clone with shares_extents freezing
- F2FS is build-once — correct the in-place-edits column
- cross-backend reopen-mutate sweep; make F2FS advertise build-once
- every repack source reader now surfaces faithful metadata
- move qcow2 / dmg out of the filesystem-support table
- *(repack)* unify pipeline — one walker + sink, no per-pair paths
- lib-level fuzz across 8 mutable backends
- *(ext)* cover multi-open_file_rw write extending file across drops

### Changed

- *(repack)* unified the repack pipeline: one generic source walker feeds
  one of two sinks (a streaming-tar sink or a block-device `Filesystem`
  sink). The per-`(source,dest)`-type copiers are gone — any readable
  source now repacks into any writable destination through a single
  trait-driven path. The only branch is streaming (tar / `.tar.<codec>`)
  vs non-streaming output. Previously-rejected combinations now work
  (e.g. `repack app.zip out.tar`, `repack image.xfs out.tar`).
- *(fs)* `Filesystem` gains `create_file_streaming` (zero-copy body
  streaming, no per-file tempfile; ext/fat32/exfat override it) and a
  batch `set_xattrs`. Faithful `getattr` (real mode/uid/gid/times, and
  xattrs/device numbers where stored) now on tar, f2fs, and XFS sources
  in addition to ext — so repacking from them preserves metadata.

### Added

- *(archive)* shared archive core (`src/fs/archive/`) — an indexed-entry
  model plus a generic read-only `Filesystem` implementation that archive
  formats plug into by supplying a scanner (and, if writable, a builder).
- *(archive)* **zip** — full read (central-directory scan, robust EOCD
  search, ZIP64, Unix mode/symlinks, Shift-JIS/EUC-JP/UTF-8 filename
  detection) and write (Stored + Deflate, CRC-32, ZIP64 when needed).
  Reads archives produced by other tools; output validates with `unzip`.
- *(archive)* **cpio** — read newc/odc + write newc; round-trips through
  system `cpio`.
- *(archive)* **ar** — read GNU + BSD long names, write GNU; round-trips
  through system `ar`. Flat archive (rejects nested paths).
- *(archive)* detection-only scaffolds for **7z, rar, arc, lha, lzx, cab,
  sit** — recognised by `info`, with a clean `Unsupported` on read until
  pure-Rust decoders are wired (per format, behind a future Cargo feature).
- *(cli)* `create -t {zip,cpio,ar}`, `repack --fs-type {zip,cpio,ar}`,
  `build` with `type = "zip"|"cpio"|"ar"`, and `mount` for all archive
  formats; archive output is truncated to its exact length.

## [0.4.1](https://github.com/KarpelesLab/fstool/compare/v0.4.0...v0.4.1) - 2026-05-22

### Added

- *(fuse)* backend-agnostic adapter — mount any Filesystem via FUSE
- *(apfs)* rename, unlink (hardlink-aware), and link()
- *(apfs)* chmod / chown / set_times mutation API
- shared-access wrapper for cross-thread Ext usage (Phase E)
- fuzz harness + crash-injection block device (Phase D)
- *(ext)* inline_data — store small files in the inode
- FUSE adapter — mount ext{2,3,4} images as a userspace filesystem
- *(ext)* post-build mutation API (chmod, chown, set_times, truncate, rename)
- *(ext)* multi-descriptor JBD2 transactions + fix dx_node header
- *(ext)* two-level HTree (dx_node intermediates)
- *(repack)* replay pending JBD2 journal on the source before reading
- *(repack)* preserve sparse files in ext repack
- *(ext)* preserve hard links across repack
- *(ext)* HTree (DIR_INDEX) write-side support for ext4
- *(ext)* multi-block directories, depth-1 extents, repack progress

### Fixed

- *(clippy)* clean up 11 lints exposed by --all-features build
- *(concurrent)* drop unused `FileSource` import from test module
- *(repack)* wire progress sink through tar-output paths

### Other

- *(fuse)* kernel round-trip test via spawn_mount
- fix 7 broken intra-doc links exposed by --all-features doc build
- install libfuse3-dev + pkg-config on Linux for clippy --all-features
- cargo fmt across recent landings

## [0.4.0](https://github.com/KarpelesLab/fstool/compare/v0.3.1...v0.4.0) - 2026-05-21

### Added

- *(cli)* unify create + add -O / [filesystem.options] for FS knobs

### Fixed

- *(spec)* mark FilesystemSpec #[non_exhaustive]

### Other

- *(readme)* refresh FS matrix + limitations for current state

## [0.3.1](https://github.com/KarpelesLab/fstool/compare/v0.3.0...v0.3.1) - 2026-05-21

### Added

- *(ext)* real JBD2 transactions for open_file_rw (Path A)
- *(apfs)* open_file_rw on flushed images via fresh checkpoint COW
- *(xfs)* leaf-form xattrs (read+write) + remove_xattr
- *(hfs+)* decmpfs read support (types 3 + 4 zlib)
- *(ntfs)* real $LogFile LFS records (Path A) for open_file_rw
- *(dmg)* encrcdsa v2 encrypted DMG read support

### Fixed

- *(hfs+)* keep HfsPlusFileReader as struct to preserve public API

### Other

- drop intra-doc links to private items in apfs

## [0.3.0](https://github.com/KarpelesLab/fstool/compare/v0.2.0...v0.3.0) - 2026-05-20

### Added

- *(hfs+)* route flush metadata writes through journal (Path A)
- *(ntfs)* multi-SD $Secure (User + System); defer $LogFile Path A
- *(xfs)* multi-level B-tree dirs + Path A log transactions
- *(ext4)* open_file_rw on depth-1 extent trees
- *(apfs)* populate IP ring, SFQ free-queues, and main-device alloc zone
- *(dmg)* implement ADC, bzip2, LZFSE, and LZMA chunk codecs
- *(hfs+)* real journal transactions (Path A) for open_file_rw
- *(ntfs)* populate $Secure ($SDS/$SDH/$SII) + sort root $I30
- *(xfs)* single-level B-tree directory reader (di_format=BTREE)
- *(ext4)* open_file_rw on depth-0 inline extent trees
- *(dmg)* chunk decoder — zero / raw / zlib over UDIF v4
- *(apfs)* emit a real spaceman bitmap + checkpoint map
- *(fs)* implement open_file_ro for ext/FAT/exFAT/F2FS/HFS+/NTFS/XFS
- *(apfs)* implement Filesystem::open_file_ro
- *(squashfs)* implement Filesystem::open_file_ro
- *(grf)* implement Filesystem::open_file_ro
- *(iso9660)* implement Filesystem::open_file_ro for random-access reads
- *(fs)* add Filesystem::open_file_ro + FileReadHandle
- *(xfs)* implement Filesystem::open_file_rw via clean-unmount bypass
- *(ntfs)* implement Filesystem::open_file_rw for in-place edits
- *(ext3/4)* accept clean-journal images in open_file_rw
- *(hfs+)* implement Filesystem::open_file_rw for in-place edits
- *(f2fs)* implement Filesystem::open_file_rw for in-place edits
- *(ext2)* implement Filesystem::open_file_rw for in-place edits
- *(fat)* implement Filesystem::open_file_rw for in-place edits
- *(exfat)* implement Filesystem::open_file_rw for in-place edits
- *(fs)* add Filesystem::open_file_rw + FileHandle for in-place edits
- *(apfs)* wire library writer through Filesystem trait
- *(hfs+)* make open() return a writable handle for add/rm round-trips
- *(ntfs)* index system files (records 0..=15) in root $I30 on format
- *(exfat)* wire writer into the Filesystem trait
- *(grf)* GRF (Gravity Ragnarok File) read + write + add/rm
- *(fs)* add MutationCapability::WholeFileOnly for future formats
- *(error)* typed Error::RepackOnly for sequential-by-design FSes

### Fixed

- *(hfs+)* clamp VH nextAllocation < totalBlocks for fsck.hfsplus
- *(exfat)* drop unused FileHandle import in open_file_rw tests
- *(iso9660)* emit SUSP SP marker on root's "." dir record
- *(repack)* Source::detect mishandled Windows drive letters

### Other

- replace links to private items with plain backticks
- cargo fmt across drifted files
- *(ext/flex_bg)* tighten leader/follower mapping check + e2fsck-clean
- resume writes from on-disk AGF/AGI/INOBT/BNO after reopen
- *(hfs+)* lock down create_hardlink link-inode invariant
- *(xfs/dir)* cover dahashname, leaf sort, and i8 shortform decode
- *(squashfs)* cover fragment table reader
- *(ext)* end-to-end xattr round-trip through set_xattrs + read_xattrs
- fix broken intra-doc links from public items into pub(crate)
- rustfmt across the tree
- *(error)* split Streaming vs Immutable instead of one RepackOnly
- collapse build-plan walkers through Filesystem::read_symlink

## [0.2.0](https://github.com/KarpelesLab/fstool/compare/v0.1.0...v0.2.0) - 2026-05-20

### Added

- *(inspect)* variant-agnostic public surface — inspect::open + summary
- *(fs)* Filesystem::supports_mutation() gates add/rm cleanly
- *(cli)* repack accepts positional sources — `repack a b … out`
- *(repack)* layered sources with tar-OCI + overlayfs whiteouts
- *(iso9660)* writer + Filesystem trait + repack-to-ISO wiring
- *(iso9660)* read support — PVD + Joliet + Rock Ridge + El Torito
- *(cli,docs)* wire repack to write XFS/HFS+/NTFS/F2FS/SquashFS via the trait
- *(fs)* wire all writable FSes (XFS/HFS+/NTFS/F2FS/SquashFS/FAT32) through one trait

### Other

- collapse sum_*_file_bytes into Filesystem::total_file_bytes
- *(readme)* cover ISO 9660 + layered merge with whiteouts

## [0.1.0](https://github.com/KarpelesLab/fstool/compare/v0.0.5...v0.1.0) - 2026-05-20

### Added

- *(block)* scaffold Apple DMG (UDIF v4) container support
- *(tar)* random-access index + hardlink materialization + tar.<algo>→ext repack
- *(squashfs)* hardlinks + device nodes + multi-fragment + ext-dir promotion
- *(f2fs)* hard links + triple-indirect nodes + multi-block dentry spill
- *(ntfs)* writer — format + create_file/dir/symlink + flush
- *(apfs)* multi-leaf writer + embedded xattrs (read + write)
- *(hfs+)* extents-overflow spill on write + hard links + journal stub
- *(xfs)* journal stub + multi-AG writes + remove + shortform xattrs
- *(ext)* BuildPlan auto-flex_bg + INCOMPAT_64BIT writer + sparse_super2
- *(tar)* TarStreamReader/Writer + CLI streaming integration (no tempfile)
- *(squashfs)* writer + xattr / id-table / export-table coverage
- *(f2fs)* writer (format, create_file/dir/symlink/device, remove, flush)
- *(ntfs)* fill read-side holes (attr-list, $Secure, $UpCase, LZNT1)
- *(apfs)* multi-volume + snapshots (read) + minimal writer
- *(hfs+)* writer (format, create_dir/file/symlink, remove, flush)
- *(xfs)* B+tree directories + write support (format, add_file/dir/symlink/device)
- *(ext)* flex_bg writer (opt-in via FormatOpts)
- *(compression)* codec features for squashfs reads and tar I/O

### Fixed

- *(hfs+)* drop intra-doc link from public to private fold_case
- *(hfs+)* make fsck.hfsplus accept writer output end-to-end
- *(hfs+)* mark Private Data dir invisible in Finder (frFlags |= kIsInvisible)
- *(hfs+)* set HasLinkChain / HasChildLink flags on hardlink records
- *(hfs+)* iNode files need fileType='iNod' / creator='hfs+' + link count
- *(hfs+)* catalog case-folding compare ignores NUL code units
- *(hfs+)* map record fills the rest of the header node
- *(hfs+)* empty B-trees need a header AND one empty leaf node
- *(hfs+)* B-tree forks need clumpSize ≥ nodeSize
- *(f2fs)* populate valid_node/inode/free_segment counts in CP head
- *(f2fs)* SIT valid_map is MSB-first, not LSB-first
- *(f2fs)* I_ADDR_OFFSET must be 0x168 (kernel spec), not 0xD0
- *(f2fs)* inline-dentry INLINE_RESERVED_SIZE is 7 bytes, not 1
- *(f2fs)* inline payload starts at i_addr[1], not i_addr[0]
- *(f2fs)* emit "." and ".." dentries + correct i_blocks
- *(f2fs)* real curseg layout + SIT type bits + node_footer
- *(f2fs)* NAT entries for node_ino / meta_ino + drop bogus NAT/SIT/SSA CRC
- *(f2fs)* write 8-block CP pack + drop bogus reserved-nid NAT entries
- *(f2fs)* SIT segment count must be even + derive bitmap size from geometry
- *(f2fs)* non-zero rsvd / overprov segments + correct user_block_count
- *(f2fs)* write CP footer at end of pack + correct CP flag values
- *(f2fs)* use real crc32_le(F2FS_SUPER_MAGIC, …) + correct CP field offsets
- *(f2fs)* segment0_blkaddr = cp_blkaddr + ignore reverse-read test
- *(f2fs,ci)* correct f2fs SB field offsets + drop deprecated brew ntfs-3g

### Other

- rustfmt insert_journal_entry signature
- *(readme)* update FS support table for current writer coverage
- Revert "fix(hfs+): catalog case-folding compare ignores NUL code units"
- *(hfs+)* diagnostic also tries mkfs.hfsplus (hfsprogs spelling)
- *(hfs+)* add diagnostic test to dump mkfs vs fstool extents header
- rustfmt write.rs after CP-pack restructure
- *(fs)* native-tool external validation for exfat/xfs/hfs+/apfs/ntfs/f2fs/squashfs + codec fixes
- *(release-plz)* fix release-binaries dispatch (tag schema + actions:write)
- cargo fmt --all

## [0.0.5](https://github.com/KarpelesLab/fstool/compare/v0.0.4...v0.0.5) - 2026-05-19

### Added

- *(fs)* fill out xfs/hfs+/apfs/ntfs/f2fs/squashfs read paths + exfat writer
- *(fs)* xfs/exfat/hfs+/apfs read-only + ntfs/f2fs/squashfs scaffolds
- *(tar)* tar as a read/write filesystem — ext↔tar / fat↔tar repack

### Other

- gate Unix-only integration tests for the Windows / macOS matrix
- *(release-plz)* chain release-binaries via workflow_dispatch

## [0.0.4](https://github.com/KarpelesLab/fstool/compare/v0.0.3...v0.0.4) - 2026-05-19

### Added

- *(ext)* xattr support — read inline + block, write block, preserve on repack
- *(cli)* convert + repack — byte-copy and FS-aware resize
- *(block, cli)* qcow2 write + create — Phase B
- *(block)* qcow2 read path — Phase A
- *(cli)* partition-aware target syntax — disk.img:N
- *(block, cli)* real block-device support on Unix
- *(cli)* fstool shell — interactive REPL over any image
- *(ext4)* sparse_super on the write path
- *(fat32, cli)* modify-in-place — add files, add dirs, remove entries
- *(fat32, cli)* read-side parity — FAT32 reader + unified CLI dispatch

### Fixed

- *(cli)* repack as a direct FS-to-FS copy, no host tempdir

### Other

- release-binaries workflow — five archives per release

## [0.0.3](https://github.com/KarpelesLab/fstool/compare/v0.0.2...v0.0.3) - 2026-05-19

### Added

- *(fat32)* write-path FAT32 filesystem + spec/CLI/CI integration
- *(ext)* automatic sparse files — all-zero blocks become holes
- *(cli)* fstool rm — remove a file / symlink / device / empty directory
- *(cli)* fstool add — copy a host file or directory into an image
- *(ext4)* full metadata_csum write path — ext4 emits checksummed images

### Other

- bring README up to date with phases 4-5 + ext4 features
- metadata_csum foundation — csum module + superblock checksum

## [0.0.2](https://github.com/KarpelesLab/fstool/compare/v0.0.1...v0.0.2) - 2026-05-19

### Added

- *(ext4)* read INCOMPAT_64BIT images — 64-byte group descriptors
- *(spec)* partitioned disk-image build + multi-group ext allocation
- *(spec)* TOML image spec + `fstool build` (bare-filesystem mode)
- *(cli)* add fstool subcommands — ext-build / ls / cat / info
- *(ext4)* write extent-tree inodes (INCOMPAT_EXTENTS) + read them back

### Other

- lazy-stage parent inode + dir block on add_*, enabling modify-after-open
- add release-plz workflow for automated releases
- add CI / crates.io / docs.rs badges to README
