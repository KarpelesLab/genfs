# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
