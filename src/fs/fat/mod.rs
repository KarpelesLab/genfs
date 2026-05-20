//! FAT32 filesystem writer.
//!
//! Produces a FAT32 image from a host directory tree. FAT32 has no concept
//! of symlinks, device nodes, or Unix ownership/permissions, so those are
//! silently skipped when copying a tree.
//!
//! v1 scope: write path only (`format` + `build_from_host_dir`). Long file
//! names use VFAT LFN entries; short 8.3 names are generated where needed.
//!
//! Reference: the public Microsoft FAT specification.

pub mod boot;
pub mod dir;
pub mod fsinfo;
pub mod handle;
pub mod mutate;
pub mod table;

use std::io::Read;
use std::path::Path;

use boot::BootSector;
use fsinfo::FsInfo;
use table::Fat;

use crate::Result;
use crate::block::BlockDevice;

/// FAT32 requires at least this many data clusters; fewer makes it a
/// FAT12/FAT16 volume, which fsck.vfat rejects as "not FAT32".
pub const MIN_FAT32_CLUSTERS: u32 = 65525;

/// Logical sector size. FAT32 supports others; genfs fixes it at 512.
pub const SECTOR: u32 = 512;

/// Options for [`Fat32::format`].
#[derive(Debug, Clone)]
pub struct FatFormatOpts {
    /// Total volume size in 512-byte sectors.
    pub total_sectors: u32,
    /// Volume ID (serial number).
    pub volume_id: u32,
    /// Volume label — up to 11 bytes, space-padded.
    pub volume_label: [u8; 11],
}

impl Default for FatFormatOpts {
    fn default() -> Self {
        Self {
            total_sectors: 0,
            volume_id: 0,
            volume_label: *b"NO NAME    ",
        }
    }
}

/// An under-construction FAT32 filesystem.
#[derive(Debug)]
pub struct Fat32 {
    boot: BootSector,
    fat: Fat,
    /// Cluster to hand out next from the free pool.
    next_free: u32,
}

impl Fat32 {
    /// Pick `sectors_per_cluster` for a volume of `total_sectors`, mirroring
    /// the conventional mkfs.vfat size thresholds.
    fn pick_spc(total_sectors: u32) -> u8 {
        match total_sectors {
            0..=532_480 => 1,          // ≤ 260 MiB
            532_481..=16_777_216 => 8, // ≤ 8 GiB
            16_777_217..=33_554_432 => 16,
            33_554_433..=67_108_864 => 32,
            _ => 64,
        }
    }

    /// Compute `(sectors_per_cluster, fat_size_sectors, cluster_count)` for a
    /// volume of `total_sectors`. Errors if the volume is too small to be a
    /// valid FAT32.
    fn geometry(total_sectors: u32) -> Result<(u8, u32, u32)> {
        let spc = Self::pick_spc(total_sectors);
        let reserved = 32u32;
        let num_fats = 2u32;
        let entries_per_fat_sector = SECTOR / 4; // 128

        // Converge fat_size upward until the FAT is big enough to map every
        // data cluster it leaves room for.
        let mut fat_size = 1u32;
        loop {
            let meta = reserved + num_fats * fat_size;
            if meta >= total_sectors {
                return Err(crate::Error::InvalidArgument(
                    "fat32: volume too small to hold the FAT metadata".into(),
                ));
            }
            let clusters = (total_sectors - meta) / spc as u32;
            let needed = (clusters + 2).div_ceil(entries_per_fat_sector);
            if needed <= fat_size {
                if clusters < MIN_FAT32_CLUSTERS {
                    return Err(crate::Error::InvalidArgument(format!(
                        "fat32: {clusters} clusters is below the FAT32 minimum of \
                         {MIN_FAT32_CLUSTERS} — use a volume of at least ~33 MiB"
                    )));
                }
                return Ok((spc, fat_size, clusters));
            }
            fat_size = needed;
        }
    }

    /// Format a fresh, empty FAT32 onto `dev`. Writes the boot sector and
    /// its backup, the FSInfo sector and its backup, both FAT copies, and
    /// the (empty) root-directory cluster.
    pub fn format(dev: &mut dyn BlockDevice, opts: &FatFormatOpts) -> Result<Self> {
        let total = opts.total_sectors;
        let need = total as u64 * SECTOR as u64;
        if dev.total_size() < need {
            return Err(crate::Error::InvalidArgument(format!(
                "fat32: device has {} bytes, need {need}",
                dev.total_size()
            )));
        }
        let (spc, fat_size, clusters) = Self::geometry(total)?;

        let mut boot = BootSector::fat32_default();
        boot.sectors_per_cluster = spc;
        boot.total_sectors = total;
        boot.fat_size = fat_size;
        boot.volume_id = opts.volume_id;
        boot.volume_label = opts.volume_label;

        // FAT has one entry per cluster (+ the 2 reserved); size the table
        // to the full on-disk FAT so encode() produces exactly fat_size
        // sectors.
        let fat_entries = (fat_size * (SECTOR / 4)) as usize;
        let mut fat = Fat::new(fat_entries, boot.media);
        // Root directory occupies cluster 2, a one-cluster chain.
        fat.set(boot.root_cluster, table::EOC);

        let fs = Self {
            boot,
            fat,
            next_free: 3,
        };
        // Zero the whole volume up front so unwritten clusters read clean.
        dev.zero_range(0, need)?;

        // Mirror the boot-sector volume label as the first root-dir entry;
        // without this fsck.vfat treats the boot label as stale and would
        // "auto-remove" it (-n exit 1).
        dev.write_at(
            fs.cluster_offset(fs.boot.root_cluster),
            &fs.volume_label_entry(),
        )?;

        let _ = clusters;
        fs.flush(dev)?;
        Ok(fs)
    }

    /// Encode the volume-label directory entry that mirrors the boot
    /// sector's `volume_label` field.
    fn volume_label_entry(&self) -> [u8; dir::ENTRY_SIZE] {
        dir::DirEntry {
            name_83: self.boot.volume_label,
            attr: dir::ATTR_VOLUME_ID,
            first_cluster: 0,
            file_size: 0,
        }
        .encode()
    }

    /// Absolute byte offset of a cluster's first sector.
    fn cluster_offset(&self, cluster: u32) -> u64 {
        let sector =
            self.boot.data_start_sector() + (cluster - 2) * self.boot.sectors_per_cluster as u32;
        sector as u64 * SECTOR as u64
    }

    /// Bytes per cluster.
    fn cluster_bytes(&self) -> u64 {
        self.boot.sectors_per_cluster as u64 * SECTOR as u64
    }

    /// Allocate `n` clusters, linking them into one chain, and return the
    /// chain. The last cluster's FAT entry is the end-of-chain marker.
    fn alloc_chain(&mut self, n: u32) -> Result<Vec<u32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut chain = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let c = self.next_free;
            if c as usize >= self.fat.capacity() {
                return Err(crate::Error::Unsupported("fat32: out of clusters".into()));
            }
            chain.push(c);
            self.next_free += 1;
        }
        for w in chain.windows(2) {
            self.fat.set(w[0], w[1]);
        }
        self.fat.set(*chain.last().unwrap(), table::EOC);
        Ok(chain)
    }

    /// Write `data` across the cluster `chain` (the chain must be large
    /// enough). The final cluster's slack is left zero.
    fn write_chain(&self, dev: &mut dyn BlockDevice, chain: &[u32], data: &[u8]) -> Result<()> {
        let cb = self.cluster_bytes() as usize;
        for (i, &c) in chain.iter().enumerate() {
            let start = i * cb;
            if start >= data.len() {
                break;
            }
            let end = (start + cb).min(data.len());
            dev.write_at(self.cluster_offset(c), &data[start..end])?;
        }
        Ok(())
    }

    /// Persist the boot sector (+ backup), FSInfo (+ backup) and both FAT
    /// copies. Free-cluster accounting is derived from the current FAT, so
    /// this works for both fresh-format and modify-in-place flows.
    pub fn flush(&self, dev: &mut dyn BlockDevice) -> Result<()> {
        let boot_bytes = self.boot.encode();
        dev.write_at(0, &boot_bytes)?;
        dev.write_at(
            self.boot.backup_boot_sector as u64 * SECTOR as u64,
            &boot_bytes,
        )?;

        let clusters = self.boot.cluster_count();
        let free_count = self.count_free_clusters();
        let next_hint = if self.next_free >= 2 && self.next_free < clusters + 2 {
            self.next_free
        } else {
            2
        };
        let fsinfo = FsInfo {
            free_count,
            next_free: next_hint,
        };
        let fsinfo_bytes = fsinfo.encode();
        dev.write_at(
            self.boot.fs_info_sector as u64 * SECTOR as u64,
            &fsinfo_bytes,
        )?;
        // The backup boot region also carries a backup FSInfo at +1.
        dev.write_at(
            (self.boot.backup_boot_sector as u64 + 1) * SECTOR as u64,
            &fsinfo_bytes,
        )?;

        let fat_bytes = self.fat.encode();
        for i in 0..self.boot.num_fats as u32 {
            let off = (self.boot.reserved_sector_count as u64
                + i as u64 * self.boot.fat_size as u64)
                * SECTOR as u64;
            dev.write_at(off, &fat_bytes)?;
        }
        Ok(())
    }

    /// Count clusters whose FAT entry is FREE, across the data-cluster
    /// range `[2, cluster_count + 2)`.
    fn count_free_clusters(&self) -> u32 {
        let clusters = self.boot.cluster_count();
        let mut n = 0u32;
        for c in 2..(2 + clusters) {
            if self.fat.get(c) == table::FREE {
                n += 1;
            }
        }
        n
    }

    /// One-shot: format `dev` to `total_sectors` and copy a host directory
    /// tree into the root. Symlinks and device nodes in the source are
    /// skipped (FAT has no representation for them).
    pub fn build_from_host_dir(
        dev: &mut dyn BlockDevice,
        total_sectors: u32,
        src: &Path,
        volume_id: u32,
        volume_label: [u8; 11],
    ) -> Result<()> {
        let opts = FatFormatOpts {
            total_sectors,
            volume_id,
            volume_label,
        };
        let mut fs = Self::format(dev, &opts)?;
        fs.populate_from_host_dir(dev, src)?;
        fs.flush(dev)?;
        dev.sync()?;
        Ok(())
    }

    /// Populate an already-formatted FAT32 root with the contents of
    /// `src`. The volume label set at format time stays in place;
    /// callers that want to re-set it should re-format. Used by the
    /// repack flow where the destination has been formatted already.
    pub fn populate_from_host_dir(&mut self, dev: &mut dyn BlockDevice, src: &Path) -> Result<()> {
        let root_cluster = self.boot.root_cluster;
        // Root is its own "parent" placeholder; parent_cluster is unused when
        // is_root = true.
        self.write_dir_tree(dev, src, root_cluster, true, root_cluster)
    }

    /// Recursively populate the directory whose data starts at `dir_cluster`
    /// from the host directory `src`. `is_root` suppresses the "." / ".."
    /// entries (the FAT32 root has none).
    ///
    /// `dir_cluster` must already be a one-cluster chain; the directory is
    /// extended if its entries overflow one cluster.
    fn write_dir_tree(
        &mut self,
        dev: &mut dyn BlockDevice,
        src: &Path,
        dir_cluster: u32,
        is_root: bool,
        parent_cluster: u32,
    ) -> Result<()> {
        // Assemble the directory's 32-byte entries in memory.
        let mut entries: Vec<u8> = Vec::new();
        if is_root {
            // Mirror the boot-sector volume label as a root-dir entry; without
            // this fsck.vfat treats the boot label as stale.
            entries.extend_from_slice(&self.volume_label_entry());
        } else {
            entries.extend_from_slice(&dot_entry(b".          ", dir_cluster));
            // ".." points at the parent; a parent that is the root is
            // recorded as cluster 0 by convention.
            let pc = if parent_cluster == self.boot.root_cluster {
                0
            } else {
                parent_cluster
            };
            entries.extend_from_slice(&dot_entry(b"..         ", pc));
        }

        let mut short_seq: u32 = 0;
        let mut children: Vec<(std::path::PathBuf, std::fs::Metadata)> = Vec::new();
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let meta = entry.metadata()?;
            children.push((entry.path(), meta));
        }
        // Deterministic order.
        children.sort_by(|a, b| a.0.file_name().cmp(&b.0.file_name()));

        for (path, meta) in children {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| crate::Error::InvalidArgument("fat32: non-UTF-8 file name".into()))?
                .to_string();
            let ft = meta.file_type();
            if ft.is_symlink() {
                continue; // FAT has no symlinks
            }
            if ft.is_file() {
                let size = meta.len();
                let cb = self.cluster_bytes();
                let n_clusters = size.div_ceil(cb).max(1) as u32;
                let chain = self.alloc_chain(n_clusters)?;
                self.stream_file(dev, &path, &chain, size)?;
                let first = if size == 0 { 0 } else { chain[0] };
                self.push_entry(
                    &mut entries,
                    &name,
                    dir::ATTR_ARCHIVE,
                    first,
                    size as u32,
                    &mut short_seq,
                );
                // A zero-length file keeps no clusters.
                if size == 0 {
                    self.free_unused_chain(&chain);
                }
            } else if ft.is_dir() {
                // Each subdirectory starts as a one-cluster chain.
                let chain = self.alloc_chain(1)?;
                let child_cluster = chain[0];
                self.write_dir_tree(dev, &path, child_cluster, false, dir_cluster)?;
                self.push_entry(
                    &mut entries,
                    &name,
                    dir::ATTR_DIRECTORY,
                    child_cluster,
                    0,
                    &mut short_seq,
                );
            }
            // Other types (devices, fifos, sockets) are skipped.
        }

        self.write_dir_entries(dev, dir_cluster, &entries)?;
        Ok(())
    }

    /// Append a directory entry for `name` to `entries`, emitting LFN
    /// fragments first when the name isn't a plain 8.3 name.
    fn push_entry(
        &self,
        entries: &mut Vec<u8>,
        name: &str,
        attr: u8,
        first_cluster: u32,
        file_size: u32,
        short_seq: &mut u32,
    ) {
        let upper = name.to_ascii_uppercase();
        // An LFN run is needed when the on-disk 8.3 name can't reproduce the
        // original verbatim — either because the original isn't a valid 8.3
        // name (too long, lower-case, weird chars) or because case was lost.
        let (name_83, need_lfn) = if dir::is_valid_83(&upper) {
            (dir::pack_83(&upper), upper != name)
        } else {
            let s = dir::generate_83(name, *short_seq);
            *short_seq += 1;
            (s, true)
        };
        if need_lfn {
            let csum = dir::lfn_checksum(&name_83);
            for frag in dir::encode_lfn_run(name, csum) {
                entries.extend_from_slice(&frag);
            }
        }
        let entry = dir::DirEntry {
            name_83,
            attr,
            first_cluster,
            file_size,
        };
        entries.extend_from_slice(&entry.encode());
    }

    /// Write a directory's assembled entry bytes into its cluster chain,
    /// extending the chain if the entries overflow `dir_cluster`'s single
    /// cluster.
    fn write_dir_entries(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_cluster: u32,
        entries: &[u8],
    ) -> Result<()> {
        let cb = self.cluster_bytes() as usize;
        let need_clusters = entries.len().div_ceil(cb).max(1) as u32;
        let mut chain = vec![dir_cluster];
        // Extend if more than one cluster of entries.
        if need_clusters > 1 {
            let extra = self.alloc_chain(need_clusters - 1)?;
            // Link dir_cluster -> extra[0] -> ... -> EOC.
            self.fat.set(dir_cluster, extra[0]);
            chain.extend_from_slice(&extra);
        }
        // Pad to a whole number of clusters with zero (free entries).
        let mut buf = entries.to_vec();
        buf.resize(need_clusters as usize * cb, 0);
        self.write_chain(dev, &chain, &buf)?;
        Ok(())
    }

    /// Stream a host file's bytes into its cluster chain. The file is read
    /// one cluster at a time — never fully resident in memory.
    fn stream_file(
        &self,
        dev: &mut dyn BlockDevice,
        host: &Path,
        chain: &[u32],
        size: u64,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        let cb = self.cluster_bytes() as usize;
        let mut file = std::fs::File::open(host)?;
        let mut buf = vec![0u8; cb];
        let mut remaining = size;
        for &c in chain {
            let want = remaining.min(cb as u64) as usize;
            buf[..want].fill(0);
            file.read_exact(&mut buf[..want])?;
            dev.write_at(self.cluster_offset(c), &buf[..want])?;
            remaining -= want as u64;
            if remaining == 0 {
                break;
            }
        }
        Ok(())
    }

    /// Return clusters allocated for a zero-length file to the free pool.
    /// Only valid for the most-recently-allocated chain (we just rewind
    /// `next_free`); used right after `alloc_chain` for empty files.
    fn free_unused_chain(&mut self, chain: &[u32]) {
        for &c in chain {
            self.fat.set(c, table::FREE);
        }
        // The chain was the tail of the free pool — rewind.
        if let Some(&first) = chain.first()
            && first + chain.len() as u32 == self.next_free
        {
            self.next_free = first;
        }
    }

    // -- read path --------------------------------------------------------

    /// Open an existing FAT32 volume from `dev`: decode the boot sector,
    /// validate the FAT32 fs_type signature, and load the primary FAT into
    /// memory.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let mut bs = [0u8; 512];
        dev.read_at(0, &mut bs)?;
        let boot = BootSector::decode(&bs)?;
        if boot.bytes_per_sector as u32 != SECTOR {
            return Err(crate::Error::Unsupported(format!(
                "fat32: only 512-byte sectors are supported (got {})",
                boot.bytes_per_sector
            )));
        }
        // Read the first FAT copy.
        let fat_bytes_len = boot.fat_size as u64 * SECTOR as u64;
        let mut fat_bytes = vec![0u8; fat_bytes_len as usize];
        let fat_off = boot.reserved_sector_count as u64 * SECTOR as u64;
        dev.read_at(fat_off, &mut fat_bytes)?;
        let fat = Fat::decode(&fat_bytes);
        // For an opened volume we don't track a free-pool cursor; set it
        // past the end so accidental allocation needs an explicit reset.
        let next_free = fat.capacity() as u32;
        Ok(Self {
            boot,
            fat,
            next_free,
        })
    }

    /// The boot sector — exposed read-only for callers (e.g. `fstool info`).
    pub fn boot_sector(&self) -> &BootSector {
        &self.boot
    }

    /// In-memory FAT — exposed read-only for diagnostics.
    pub fn fat(&self) -> &Fat {
        &self.fat
    }

    /// Mutable access to the in-memory FAT — used by the modify-in-place
    /// file handle to grow / shrink cluster chains.
    pub(super) fn fat_mut(&mut self) -> &mut Fat {
        &mut self.fat
    }

    /// Hint the free-cluster scanner to consider `cluster` next. Used when
    /// the file handle frees a tail of clusters during a shrink so the
    /// allocator can hand them out again.
    pub(super) fn hint_next_free(&mut self, cluster: u32) {
        if cluster >= 2 && cluster < self.boot.cluster_count() + 2 {
            self.next_free = cluster;
        }
    }

    /// Walk the cluster chain starting at `start`, collecting every cluster
    /// in order.
    pub fn chain_of(&self, start: u32) -> Result<Vec<u32>> {
        self.fat.chain(start)
    }

    /// List the entries of a directory by absolute path. `/` resolves to
    /// the root directory. Returns one [`crate::fs::DirEntry`] per visible
    /// entry, with `inode` set to the entry's `first_cluster` (FAT has no
    /// inode numbers, but the cluster number is a stable per-entry id).
    /// Volume-label entries and `.` / `..` are skipped.
    pub fn list_path(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let cluster = self.resolve_dir(dev, path)?;
        self.list_cluster(dev, cluster)
    }

    /// Open a regular file by absolute path for streaming reads. The
    /// returned reader holds an in-memory copy of the cluster chain and
    /// borrows `dev` for the actual block reads.
    pub fn open_file_reader<'a>(
        &self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<FatFileReader<'a>> {
        let (entry, dir_cluster) = self.resolve_entry(dev, path)?;
        if entry.attr & dir::ATTR_DIRECTORY != 0 {
            return Err(crate::Error::InvalidArgument(format!(
                "fat32: {path:?} is a directory, not a file"
            )));
        }
        let _ = dir_cluster; // unused once we have the leaf
        let chain = if entry.first_cluster < 2 {
            Vec::new() // zero-length file
        } else {
            self.chain_of(entry.first_cluster)?
        };
        let cluster_bytes = self.cluster_bytes();
        let data_start = self.boot.data_start_sector() as u64 * SECTOR as u64;
        let spc = self.boot.sectors_per_cluster;
        Ok(FatFileReader {
            dev,
            chain,
            cluster_bytes,
            data_start,
            spc,
            remaining: entry.file_size as u64,
            cluster_idx: 0,
            cluster_off: 0,
        })
    }

    /// Resolve `path` to the cluster number of the named directory, or the
    /// root cluster for `/` / "".
    pub fn resolve_dir(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<u32> {
        let parts = split_path(path);
        let mut cluster = self.boot.root_cluster;
        for part in parts {
            let entries = self.list_cluster_raw(dev, cluster)?;
            let next = entries
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(part))
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "fat32: no such entry {part:?} under {path:?}"
                    ))
                })?;
            if next.1.attr & dir::ATTR_DIRECTORY == 0 {
                return Err(crate::Error::InvalidArgument(format!(
                    "fat32: {part:?} is not a directory"
                )));
            }
            // For ".." pointing at the root, the on-disk first_cluster is 0.
            cluster = if next.1.first_cluster == 0 {
                self.boot.root_cluster
            } else {
                next.1.first_cluster
            };
        }
        Ok(cluster)
    }

    /// Resolve `path` to its 8.3 entry plus the cluster of the containing
    /// directory. Errors if the path is `/` (root has no entry).
    pub fn resolve_entry(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<(dir::DirEntry, u32)> {
        let parts = split_path(path);
        if parts.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "fat32: cannot resolve root \"/\" as a file entry".into(),
            ));
        }
        let mut cluster = self.boot.root_cluster;
        let (last, prefix) = parts.split_last().unwrap();
        for part in prefix {
            let entries = self.list_cluster_raw(dev, cluster)?;
            let next = entries
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(part))
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "fat32: no such entry {part:?} under {path:?}"
                    ))
                })?;
            cluster = if next.1.first_cluster == 0 {
                self.boot.root_cluster
            } else {
                next.1.first_cluster
            };
        }
        let entries = self.list_cluster_raw(dev, cluster)?;
        let found = entries
            .into_iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(last))
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "fat32: no such entry {last:?} under {path:?}"
                ))
            })?;
        Ok((found.1, cluster))
    }

    /// Read every 32-byte slot of the directory at `dir_cluster`, walking
    /// the cluster chain to the end. Returns the raw bytes concatenated.
    fn read_dir_bytes(&self, dev: &mut dyn BlockDevice, dir_cluster: u32) -> Result<Vec<u8>> {
        let chain = self.chain_of(dir_cluster)?;
        let cb = self.cluster_bytes() as usize;
        let mut buf = vec![0u8; chain.len() * cb];
        for (i, &c) in chain.iter().enumerate() {
            dev.read_at(self.cluster_offset(c), &mut buf[i * cb..(i + 1) * cb])?;
        }
        Ok(buf)
    }

    /// Walk a directory's slots, reassembling LFN runs into long names.
    /// Returns `(long-or-short-name, entry)` pairs in on-disk order,
    /// excluding the volume-label entry and `.` / `..`.
    fn list_cluster_raw(
        &self,
        dev: &mut dyn BlockDevice,
        dir_cluster: u32,
    ) -> Result<Vec<(String, dir::DirEntry)>> {
        let bytes = self.read_dir_bytes(dev, dir_cluster)?;
        let mut out = Vec::new();
        let mut lfn_run: Vec<dir::LfnFragment> = Vec::new();
        for chunk in bytes.chunks_exact(dir::ENTRY_SIZE) {
            let slot: &[u8; dir::ENTRY_SIZE] = chunk.try_into().unwrap();
            match dir::classify_slot(slot) {
                dir::RawSlot::End => break,
                dir::RawSlot::Deleted => {
                    lfn_run.clear();
                }
                dir::RawSlot::Lfn(frag) => {
                    lfn_run.push(frag);
                }
                dir::RawSlot::ShortEntry(entry) => {
                    if entry.attr & dir::ATTR_VOLUME_ID != 0
                        && entry.attr & dir::ATTR_DIRECTORY == 0
                    {
                        // Volume label entry.
                        lfn_run.clear();
                        continue;
                    }
                    let short_name = entry.short_name_string();
                    if short_name == "." || short_name == ".." {
                        lfn_run.clear();
                        continue;
                    }
                    let name = dir::assemble_lfn(&lfn_run, &entry.name_83)
                        .unwrap_or_else(|| short_name.clone());
                    lfn_run.clear();
                    out.push((name, entry));
                }
            }
        }
        Ok(out)
    }

    /// List the entries of `dir_cluster` as generic [`crate::fs::DirEntry`]s.
    fn list_cluster(
        &self,
        dev: &mut dyn BlockDevice,
        dir_cluster: u32,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        use crate::fs::{DirEntry as FsDirEntry, EntryKind};
        let entries = self.list_cluster_raw(dev, dir_cluster)?;
        Ok(entries
            .into_iter()
            .map(|(name, e)| {
                let is_dir = e.attr & dir::ATTR_DIRECTORY != 0;
                FsDirEntry {
                    name,
                    inode: e.first_cluster,
                    kind: if is_dir {
                        EntryKind::Dir
                    } else {
                        EntryKind::Regular
                    },
                    size: if is_dir { 0 } else { u64::from(e.file_size) },
                }
            })
            .collect())
    }
}

/// Split an absolute or relative FAT path into its non-empty components.
/// `/`, `""`, and `.` all yield an empty vec (= "the root").
fn split_path(path: &str) -> Vec<&str> {
    path.split(['/', '\\'])
        .filter(|p| !p.is_empty() && *p != ".")
        .collect()
}

/// Streaming reader for a FAT32 file. Walks the cluster chain on demand;
/// the file's bytes are never buffered beyond one [`std::io::Read::read`]
/// call's destination buffer.
pub struct FatFileReader<'a> {
    dev: &'a mut dyn BlockDevice,
    chain: Vec<u32>,
    cluster_bytes: u64,
    data_start: u64,
    spc: u8,
    /// Bytes of the file still to be returned.
    remaining: u64,
    /// Index into `chain` of the cluster currently being read from.
    cluster_idx: usize,
    /// Byte offset into the current cluster.
    cluster_off: u64,
}

impl<'a> std::io::Read for FatFileReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 || self.cluster_idx >= self.chain.len() {
            return Ok(0);
        }
        let avail_in_cluster = self.cluster_bytes - self.cluster_off;
        let want = (buf.len() as u64).min(avail_in_cluster).min(self.remaining) as usize;
        let cluster = self.chain[self.cluster_idx];
        let cluster_start =
            self.data_start + (cluster as u64 - 2) * self.spc as u64 * SECTOR as u64;
        let off = cluster_start + self.cluster_off;
        self.dev
            .read_at(off, &mut buf[..want])
            .map_err(std::io::Error::other)?;
        self.cluster_off += want as u64;
        self.remaining -= want as u64;
        if self.cluster_off == self.cluster_bytes {
            self.cluster_idx += 1;
            self.cluster_off = 0;
        }
        Ok(want)
    }
}

/// Build a "." or ".." directory entry (11-byte raw name, directory attr).
fn dot_entry(name_83: &[u8; 11], cluster: u32) -> [u8; dir::ENTRY_SIZE] {
    dir::DirEntry {
        name_83: *name_83,
        attr: dir::ATTR_DIRECTORY,
        first_cluster: cluster,
        file_size: 0,
    }
    .encode()
}

// ----------------------------------------------------------------------
// `crate::fs::Filesystem` trait impl — lets `Fat32` be driven by the
// generic walker in `crate::repack` alongside the other writable FSes.
// ----------------------------------------------------------------------

impl crate::fs::FilesystemFactory for Fat32 {
    type FormatOpts = FatFormatOpts;

    fn format(dev: &mut dyn BlockDevice, opts: &Self::FormatOpts) -> Result<Self> {
        Self::format(dev, opts)
    }

    fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        Self::open(dev)
    }
}

impl crate::fs::Filesystem for Fat32 {
    fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &Path,
        src: crate::fs::FileSource,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("fat32: non-UTF-8 path".into()))?;
        let (mut reader, len) = src.open()?;
        self.add_file_from_reader(dev, s, &mut reader, len)
    }

    fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &Path,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("fat32: non-UTF-8 path".into()))?;
        self.add_dir(dev, s)
    }

    fn create_symlink(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &Path,
        _target: &Path,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(crate::Error::Unsupported(
            "fat32: filesystem does not support symbolic links".into(),
        ))
    }

    fn create_device(
        &mut self,
        _dev: &mut dyn BlockDevice,
        _path: &Path,
        _kind: crate::fs::DeviceKind,
        _major: u32,
        _minor: u32,
        _meta: crate::fs::FileMeta,
    ) -> Result<()> {
        Err(crate::Error::Unsupported(
            "fat32: filesystem does not support device / FIFO / socket nodes".into(),
        ))
    }

    fn remove(&mut self, dev: &mut dyn BlockDevice, path: &Path) -> Result<()> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("fat32: non-UTF-8 path".into()))?;
        self.remove(dev, s)
    }

    fn list(&mut self, dev: &mut dyn BlockDevice, path: &Path) -> Result<Vec<crate::fs::DirEntry>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("fat32: non-UTF-8 path".into()))?;
        self.list_path(dev, s)
    }

    fn read_file<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &Path,
    ) -> Result<Box<dyn Read + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("fat32: non-UTF-8 path".into()))?;
        let r = self.open_file_reader(dev, s)?;
        Ok(Box::new(r))
    }

    fn open_file_rw<'a>(
        &'a mut self,
        dev: &'a mut dyn BlockDevice,
        path: &Path,
        flags: crate::fs::OpenFlags,
        meta: Option<crate::fs::FileMeta>,
    ) -> Result<Box<dyn crate::fs::FileHandle + 'a>> {
        let s = path
            .to_str()
            .ok_or_else(|| crate::Error::InvalidArgument("fat32: non-UTF-8 path".into()))?;
        // Resolve the parent + leaf. We do this once up front so the
        // create-then-reopen branch shares the result.
        let (parent_cluster, leaf) = self.resolve_parent(dev, s)?;
        let existing = self.find_entry(dev, parent_cluster, &leaf)?;
        let found = match existing {
            Some(f) => {
                if f.entry.attr & dir::ATTR_DIRECTORY != 0 {
                    return Err(crate::Error::InvalidArgument(format!(
                        "fat32: {s:?} is a directory, not a file"
                    )));
                }
                f
            }
            None => {
                if !flags.create {
                    return Err(crate::Error::InvalidArgument(format!(
                        "fat32: {s:?} not found and `create` is false"
                    )));
                }
                if meta.is_none() {
                    return Err(crate::Error::InvalidArgument(
                        "fat32: open_file_rw with create=true requires meta".into(),
                    ));
                }
                // Create an empty file via the existing modify-in-place
                // path, then re-find its entry.
                self.add_file_from_reader(dev, s, &mut std::io::empty(), 0)?;
                self.find_entry(dev, parent_cluster, &leaf)?
                    .ok_or_else(|| {
                        crate::Error::InvalidImage(
                            "fat32: created file disappeared before open".into(),
                        )
                    })?
            }
        };
        let mutate::FoundEntry {
            chain: dir_chain,
            entry_pos,
            entry,
            ..
        } = found;
        let mut handle = handle::FatFileHandle::open_existing(self, dev, dir_chain, entry_pos, entry)?;
        if flags.truncate {
            crate::fs::FileHandle::set_len(&mut handle, 0)?;
        }
        if flags.append {
            // Position at end so the first write appends.
            use std::io::Seek as _;
            let len = crate::fs::FileHandle::len(&handle);
            handle
                .seek(std::io::SeekFrom::Start(len))
                .map_err(crate::Error::Io)?;
        }
        Ok(Box::new(handle))
    }

    fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        Self::flush(self, dev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::MemoryBackend;
    use crate::fs::{FileMeta, FileSource, Filesystem, OpenFlags};
    use std::io::{Seek as _, SeekFrom, Write as _};

    /// Format a fresh 48 MiB FAT32 volume in memory; return (dev, fs).
    fn fresh_volume() -> (MemoryBackend, Fat32) {
        let mut dev = MemoryBackend::new(48 * 1024 * 1024);
        let opts = FatFormatOpts {
            total_sectors: 48 * 1024 * 1024 / 512,
            volume_id: 0xCAFE_F00D,
            volume_label: *b"OPENRWTEST ",
        };
        let fs = Fat32::format(&mut dev, &opts).unwrap();
        (dev, fs)
    }

    /// Read a whole file by path into a Vec via the streaming reader.
    fn read_all(fs: &mut Fat32, dev: &mut dyn BlockDevice, path: &str) -> Vec<u8> {
        let mut r = fs
            .open_file_reader(dev, path)
            .expect("open_file_reader for read_all");
        let mut out = Vec::new();
        r.read_to_end(&mut out).expect("read_to_end");
        out
    }

    #[test]
    fn geometry_small_volume() {
        // 64 MiB volume = 131072 sectors.
        let (spc, fat_size, clusters) = Fat32::geometry(131072).unwrap();
        assert_eq!(spc, 1);
        assert!(fat_size > 0);
        assert!(clusters >= MIN_FAT32_CLUSTERS);
        // Consistency: reserved + 2*fat + clusters*spc <= total.
        assert!(32 + 2 * fat_size + clusters * spc as u32 <= 131072);
        // The FAT must map every cluster.
        assert!(fat_size * (SECTOR / 4) >= clusters + 2);
    }

    #[test]
    fn geometry_rejects_tiny_volume() {
        // 4 MiB is far below the FAT32 minimum.
        assert!(Fat32::geometry(8192).is_err());
    }

    #[test]
    fn format_empty_volume() {
        let mut dev = MemoryBackend::new(48 * 1024 * 1024);
        let opts = FatFormatOpts {
            total_sectors: 48 * 1024 * 1024 / 512,
            volume_id: 0xCAFE_F00D,
            volume_label: *b"TESTVOL    ",
        };
        let fs = Fat32::format(&mut dev, &opts).unwrap();
        // Boot sector round-trips.
        let mut bs = [0u8; 512];
        dev.read_at(0, &mut bs).unwrap();
        let decoded = BootSector::decode(&bs).unwrap();
        assert_eq!(decoded.total_sectors, opts.total_sectors);
        assert_eq!(decoded.root_cluster, 2);
        assert_eq!(decoded.volume_id, 0xCAFE_F00D);
        // Backup boot sector matches.
        let mut backup = [0u8; 512];
        dev.read_at(6 * 512, &mut backup).unwrap();
        assert_eq!(bs, backup);
        // Root cluster's FAT entry is an end-of-chain marker.
        assert!(Fat::is_eoc(fs.fat.get(2)));
    }

    #[test]
    fn open_file_rw_partial_write_round_trip() {
        let (mut dev, mut fs) = fresh_volume();
        // Initial contents: 200 bytes of 0xAA.
        let initial = vec![0xAAu8; 200];
        fs.create_file(
            &mut dev,
            Path::new("hello.bin"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(initial.clone())),
                len: 200,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        // Reopen rw and patch 16 bytes at offset 100.
        let patch = [0x55u8; 16];
        {
            let mut h = fs
                .open_file_rw(
                    &mut dev,
                    Path::new("hello.bin"),
                    OpenFlags::default(),
                    None,
                )
                .unwrap();
            h.seek(SeekFrom::Start(100)).unwrap();
            h.write_all(&patch).unwrap();
            h.sync().unwrap();
        }

        let got = read_all(&mut fs, &mut dev, "hello.bin");
        assert_eq!(got.len(), 200);
        // 0..100 unchanged.
        assert!(got[..100].iter().all(|&b| b == 0xAA));
        // 100..116 patched.
        assert_eq!(&got[100..116], &patch);
        // 116..200 unchanged.
        assert!(got[116..].iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn open_file_rw_extends_file() {
        let (mut dev, mut fs) = fresh_volume();
        let initial = vec![0x11u8; 50];
        fs.create_file(
            &mut dev,
            Path::new("grow.bin"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(initial)),
                len: 50,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        // Seek past EOF and write 1 KiB of pattern.
        let pattern: Vec<u8> = (0..1024u32).map(|i| (i & 0xFF) as u8).collect();
        {
            let mut h = fs
                .open_file_rw(&mut dev, Path::new("grow.bin"), OpenFlags::default(), None)
                .unwrap();
            assert_eq!(h.len(), 50);
            h.seek(SeekFrom::Start(2000)).unwrap();
            h.write_all(&pattern).unwrap();
            // len() = 2000 + 1024 = 3024.
            assert_eq!(h.len(), 2000 + 1024);
            h.sync().unwrap();
        }

        let got = read_all(&mut fs, &mut dev, "grow.bin");
        assert_eq!(got.len(), 3024);
        // First 50 bytes preserved.
        assert!(got[..50].iter().all(|&b| b == 0x11));
        // Gap 50..2000 is zero.
        assert!(got[50..2000].iter().all(|&b| b == 0));
        // Patched range matches.
        assert_eq!(&got[2000..], &pattern[..]);
    }

    #[test]
    fn open_file_rw_set_len_grow_and_shrink() {
        let (mut dev, mut fs) = fresh_volume();
        let initial = vec![0x77u8; 128];
        fs.create_file(
            &mut dev,
            Path::new("resize.bin"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(initial)),
                len: 128,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        // Grow to 4096 bytes — added bytes must read as zero.
        {
            let mut h = fs
                .open_file_rw(
                    &mut dev,
                    Path::new("resize.bin"),
                    OpenFlags::default(),
                    None,
                )
                .unwrap();
            h.set_len(4096).unwrap();
            assert_eq!(h.len(), 4096);
            h.sync().unwrap();
        }
        let after_grow = read_all(&mut fs, &mut dev, "resize.bin");
        assert_eq!(after_grow.len(), 4096);
        assert!(after_grow[..128].iter().all(|&b| b == 0x77));
        assert!(after_grow[128..].iter().all(|&b| b == 0));

        // Shrink back to 64 — truncation discards trailing bytes.
        {
            let mut h = fs
                .open_file_rw(
                    &mut dev,
                    Path::new("resize.bin"),
                    OpenFlags::default(),
                    None,
                )
                .unwrap();
            h.set_len(64).unwrap();
            assert_eq!(h.len(), 64);
            h.sync().unwrap();
        }
        let after_shrink = read_all(&mut fs, &mut dev, "resize.bin");
        assert_eq!(after_shrink.len(), 64);
        assert!(after_shrink.iter().all(|&b| b == 0x77));
    }

    #[test]
    fn open_file_rw_append() {
        let (mut dev, mut fs) = fresh_volume();
        let initial = b"head".to_vec();
        fs.create_file(
            &mut dev,
            Path::new("app.txt"),
            FileSource::Reader {
                reader: Box::new(std::io::Cursor::new(initial.clone())),
                len: initial.len() as u64,
            },
            FileMeta::default(),
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();

        {
            let mut h = fs
                .open_file_rw(
                    &mut dev,
                    Path::new("app.txt"),
                    OpenFlags {
                        append: true,
                        ..OpenFlags::default()
                    },
                    None,
                )
                .unwrap();
            h.write_all(b"-tail").unwrap();
            h.sync().unwrap();
        }
        let got = read_all(&mut fs, &mut dev, "app.txt");
        assert_eq!(got, b"head-tail");
    }

    #[test]
    fn open_file_rw_create_new() {
        let (mut dev, mut fs) = fresh_volume();
        // The path doesn't exist yet — `create: true` should make it.
        {
            let mut h = fs
                .open_file_rw(
                    &mut dev,
                    Path::new("brand-new.dat"),
                    OpenFlags {
                        create: true,
                        ..OpenFlags::default()
                    },
                    Some(FileMeta::default()),
                )
                .unwrap();
            assert_eq!(h.len(), 0);
            h.write_all(b"hello from rw create").unwrap();
            h.sync().unwrap();
        }
        let got = read_all(&mut fs, &mut dev, "brand-new.dat");
        assert_eq!(got, b"hello from rw create");

        // Without `create`, a non-existent path is an error.
        match fs.open_file_rw(
            &mut dev,
            Path::new("never.bin"),
            OpenFlags::default(),
            None,
        ) {
            Ok(_) => panic!("expected error for non-existent path with create=false"),
            Err(crate::Error::InvalidArgument(_)) => {}
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }
}
