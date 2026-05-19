//! exFAT — Microsoft's flash-friendly FAT successor. Read-only support.
//!
//! v1 scope: open + list + read on a well-formed exFAT image. Reference
//! is the publicly-published Microsoft exFAT specification (2019); no
//! GPL-licensed code was consulted.
//!
//! ## High-level layout
//!
//! ```text
//!   sector 0          Main Boot Sector (this is what `probe` looks at)
//!   sector 1..=8      Extended Boot Sectors (ignored)
//!   sector 9          OEM Parameters (ignored)
//!   sector 10         Reserved (ignored)
//!   sector 11         Main Boot Checksum (ignored)
//!   sector 12..=23    Backup of sectors 0..=11 (ignored)
//!   FatOffset         First FAT
//!   ClusterHeapOffset First data cluster (cluster 2)
//! ```
//!
//! The root directory lives at `FirstClusterOfRootDirectory` and is a
//! normal directory — i.e. the same 32-byte entry-set scheme as every
//! other directory, with the addition of three "metadata" entries:
//! AllocationBitmap (0x81), UpcaseTable (0x82), and VolumeLabel (0x83).

pub mod boot;
pub mod dir;
pub mod fat;
pub mod format;
pub mod upcase;

use boot::BootSector;
use dir::{ENTRY_SIZE, FileEntrySet, RawSlot};
use fat::Fat;
use upcase::Upcase;

use crate::Result;
use crate::block::BlockDevice;

pub use format::FormatOpts;

/// Streaming buffer size used when copying host-file bytes through the
/// writer. Per project rule: never load a whole file in memory.
const SCRATCH_BUF_BYTES: usize = 64 * 1024;

/// What [`Exfat::scan_root_metadata`] returns: the volume label, the
/// up-case table read from the root directory, and the
/// `(first_cluster, data_length)` pair of the allocation bitmap if one
/// was found.
struct RootMetadata {
    volume_label: String,
    upcase: Upcase,
    bitmap_info: Option<(u32, u64)>,
}

/// An opened exFAT volume. Supports both read and write — call
/// [`Exfat::flush`] before dropping the volume to ensure the FAT,
/// allocation bitmap and boot-region copies are persisted.
pub struct Exfat {
    boot: BootSector,
    fat: Fat,
    upcase: Upcase,
    volume_label: String,
    /// Allocation bitmap: one bit per data cluster, bit N → cluster N+2.
    /// `bitmap[i]` is the byte covering clusters `2 + 8*i..2 + 8*(i+1)`.
    bitmap: Vec<u8>,
    /// First cluster of the allocation bitmap. 0 → bitmap unknown
    /// (read-only image we couldn't fully introspect).
    bitmap_first_cluster: u32,
    /// DataLength of the bitmap in bytes.
    bitmap_data_length: u64,
    /// Next-cluster hint for [`Exfat::alloc_cluster`] — the allocator
    /// starts scanning here, bounding cost on a near-empty volume.
    next_free_hint: u32,
    /// True if the FAT has unwritten changes.
    fat_dirty: bool,
    /// True if the bitmap has unwritten changes.
    bitmap_dirty: bool,
}

impl Exfat {
    /// Open the volume on `dev`: decode the boot sector, load the
    /// primary FAT into memory, scan the root directory for the
    /// AllocationBitmap / UpcaseTable / VolumeLabel metadata entries,
    /// and read the up-case table.
    pub fn open(dev: &mut dyn BlockDevice) -> Result<Self> {
        let mut bs = [0u8; boot::BOOT_SECTOR_PARSE_SIZE];
        dev.read_at(0, &mut bs)?;
        let boot = BootSector::decode(&bs)?;

        // Read the primary FAT. exFAT may have NumberOfFats == 2 (TexFAT);
        // we only need the primary copy for read.
        let fat_off = boot.fat_byte_offset();
        let fat_len = boot.fat_byte_length() as usize;
        if fat_len == 0 {
            return Err(crate::Error::InvalidImage(
                "exfat: FatLength is zero".into(),
            ));
        }
        let mut fat_bytes = vec![0u8; fat_len];
        dev.read_at(fat_off, &mut fat_bytes)?;
        let fat = Fat::decode(&fat_bytes);

        // Walk the root directory to find the volume metadata. We do this
        // with a temporary `Self` lacking an upcase table — case-
        // insensitive matching isn't required for metadata-entry
        // discovery, which is keyed by entry type.
        let mut tmp = Self {
            boot,
            fat,
            upcase: Upcase::default(),
            volume_label: String::new(),
            bitmap: Vec::new(),
            bitmap_first_cluster: 0,
            bitmap_data_length: 0,
            next_free_hint: 2,
            fat_dirty: false,
            bitmap_dirty: false,
        };
        let root_bytes = tmp.read_chain_bytes(
            dev,
            tmp.boot.first_cluster_of_root_directory,
            /* no_fat_chain */ false,
            /* hint_byte_len */ None,
        )?;

        let RootMetadata {
            volume_label,
            upcase,
            bitmap_info,
        } = tmp.scan_root_metadata(dev, &root_bytes)?;
        tmp.upcase = upcase;
        tmp.volume_label = volume_label;
        if let Some((first_cluster, data_length)) = bitmap_info {
            // Read the bitmap into memory so the writer can flip bits.
            let bm = tmp.read_chain_bytes(dev, first_cluster, false, Some(data_length))?;
            tmp.bitmap_first_cluster = first_cluster;
            tmp.bitmap_data_length = data_length;
            tmp.bitmap = bm;
        }
        Ok(tmp)
    }

    /// Format a fresh exFAT volume on `dev`. Overwrites the boot region,
    /// FAT, and root directory cluster. Returns a writable [`Exfat`].
    pub fn format(dev: &mut dyn BlockDevice, opts: &FormatOpts) -> Result<Self> {
        let geom = format::compute_geometry(dev.total_size(), opts)?;
        let ss = geom.bytes_per_sector as usize;

        // --- 1. Boot region (main + backup). ----------------------------
        format::write_boot_region(dev, &geom, 0)?;
        format::write_boot_region(dev, &geom, 12 * ss as u64)?;

        // --- 2. FAT. Initialise reserved entries and chain entries for
        //        bitmap (2), upcase (3), root (4) as one-cluster EOC.
        let mut fat = Fat::new_blank(geom.cluster_count as usize + 2);
        const CL_BITMAP: u32 = 2;
        const CL_UPCASE: u32 = 3;
        const CL_ROOT: u32 = 4;
        fat.set_raw(CL_BITMAP, fat::EOC);
        fat.set_raw(CL_UPCASE, fat::EOC);
        fat.set_raw(CL_ROOT, fat::EOC);

        // --- 3. Allocation bitmap. -------------------------------------
        let bitmap_byte_len = (geom.cluster_count as u64).div_ceil(8);
        let mut bitmap = vec![0u8; bitmap_byte_len as usize];
        set_bitmap_bit(&mut bitmap, CL_BITMAP, true);
        set_bitmap_bit(&mut bitmap, CL_UPCASE, true);
        set_bitmap_bit(&mut bitmap, CL_ROOT, true);

        // --- 4. Up-case table. -----------------------------------------
        let (upcase_bytes, upcase_csum) = format::make_ascii_upcase_table();
        let upcase = Upcase::decode(&upcase_bytes, upcase_bytes.len() as u64)?;

        // --- 5. Root directory: VolumeLabel + Bitmap + UpcaseTable. ----
        let mut root = Vec::new();
        if !opts.volume_label.is_empty() {
            root.extend_from_slice(&format::make_volume_label_entry(&opts.volume_label));
        }
        root.extend_from_slice(&format::make_bitmap_entry(CL_BITMAP, bitmap_byte_len));
        root.extend_from_slice(&format::make_upcase_entry(
            upcase_csum,
            CL_UPCASE,
            upcase_bytes.len() as u64,
        ));

        // --- 6. Write everything. --------------------------------------
        // Zero the boot-region tail clusters that fall within the FAT or
        // first three data clusters so reads see clean data.
        let fat_bytes = fat.encode();
        // Ensure the FAT region in the file matches what we built.
        // (The FAT length covers the whole table including unused tail
        // entries, which are all-zero FREE.)
        let fat_byte_off = geom.fat_byte_offset();
        let fat_byte_len = geom.fat_byte_length() as usize;
        let mut fat_image = vec![0u8; fat_byte_len];
        let n_copy = fat_bytes.len().min(fat_byte_len);
        fat_image[..n_copy].copy_from_slice(&fat_bytes[..n_copy]);
        dev.write_at(fat_byte_off, &fat_image)?;

        // Zero the bitmap and upcase clusters first, then overwrite the
        // populated prefix.
        let bpc = geom.bytes_per_cluster as usize;
        let bm_off = geom.cluster_byte_offset(CL_BITMAP);
        let up_off = geom.cluster_byte_offset(CL_UPCASE);
        let root_off = geom.cluster_byte_offset(CL_ROOT);
        let zero_cluster = vec![0u8; bpc];
        dev.write_at(bm_off, &zero_cluster)?;
        dev.write_at(up_off, &zero_cluster)?;
        dev.write_at(root_off, &zero_cluster)?;

        dev.write_at(bm_off, &bitmap)?;
        dev.write_at(up_off, &upcase_bytes)?;
        dev.write_at(root_off, &root)?;

        // --- 7. Build the in-memory BootSector mirror. -----------------
        let mut bs_buf = [0u8; boot::BOOT_SECTOR_PARSE_SIZE];
        let mb = format::make_main_boot_sector(&geom, boot::BOOT_SECTOR_PARSE_SIZE);
        bs_buf.copy_from_slice(&mb[..boot::BOOT_SECTOR_PARSE_SIZE]);
        let boot = BootSector::decode(&bs_buf)?;

        Ok(Self {
            boot,
            fat,
            upcase,
            volume_label: opts.volume_label.clone(),
            bitmap,
            bitmap_first_cluster: CL_BITMAP,
            bitmap_data_length: bitmap_byte_len,
            next_free_hint: 5, // first free cluster after metadata
            fat_dirty: false,
            bitmap_dirty: false,
        })
    }

    /// Total volume size in bytes (per `VolumeLength` in the boot sector).
    pub fn total_bytes(&self) -> u64 {
        self.boot.volume_length * self.boot.bytes_per_sector() as u64
    }

    /// Cluster size in bytes.
    pub fn cluster_size(&self) -> u32 {
        self.boot.bytes_per_cluster()
    }

    /// Sectors per cluster (the cluster size in sector units).
    pub fn sectors_per_cluster(&self) -> u32 {
        self.boot.sectors_per_cluster()
    }

    /// First cluster of the root directory.
    pub fn root_directory_cluster(&self) -> u32 {
        self.boot.first_cluster_of_root_directory
    }

    /// Decoded volume label, or empty string if none was set.
    pub fn volume_label(&self) -> &str {
        &self.volume_label
    }

    /// List the entries of a directory by absolute path. `/`, `""`, and
    /// "." all resolve to the root.
    pub fn list_path(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        let dir_cluster = self.resolve_dir(dev, path)?;
        // For directories we don't know the byte length up-front; read
        // the whole cluster chain (directories are not marked NoFatChain
        // in practice — they grow via the FAT).
        let bytes = self.read_chain_bytes(dev, dir_cluster, false, None)?;
        let mut out = Vec::new();
        for entry in iter_file_sets(&bytes)? {
            let kind = if entry.is_directory {
                crate::fs::EntryKind::Dir
            } else {
                crate::fs::EntryKind::Regular
            };
            out.push(crate::fs::DirEntry {
                name: entry.name,
                inode: entry.first_cluster,
                kind,
            });
        }
        Ok(out)
    }

    /// Open a streaming reader for a regular file at `path`.
    pub fn open_file_reader<'a>(
        &self,
        dev: &'a mut dyn BlockDevice,
        path: &str,
    ) -> Result<ExfatFileReader<'a>> {
        let (entry, _parent_cluster) = self.resolve_entry(dev, path)?;
        if entry.is_directory {
            return Err(crate::Error::InvalidArgument(format!(
                "exfat: {path:?} is a directory, not a file"
            )));
        }
        let cluster_bytes = self.boot.bytes_per_cluster() as u64;
        let chain =
            self.build_data_chain(entry.first_cluster, entry.no_fat_chain(), entry.data_length)?;
        // ValidDataLength is what the file system reports as the logical
        // file size; bytes beyond it but within DataLength are nominally
        // zero. Cap reads to ValidDataLength.
        let remaining = entry.valid_data_length;
        Ok(ExfatFileReader {
            dev,
            chain,
            cluster_heap_offset: self.boot.cluster_heap_byte_offset(),
            cluster_bytes,
            remaining,
            cluster_idx: 0,
            cluster_off: 0,
        })
    }

    // -- internals --------------------------------------------------------

    /// Resolve `path` to the cluster number of the named directory.
    fn resolve_dir(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<u32> {
        let mut cluster = self.boot.first_cluster_of_root_directory;
        for part in split_path(path) {
            let bytes = self.read_chain_bytes(dev, cluster, false, None)?;
            let entries = iter_file_sets(&bytes)?;
            let next = entries
                .into_iter()
                .find(|e| self.name_matches(&e.name_utf16, part))
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "exfat: no such entry {part:?} under {path:?}"
                    ))
                })?;
            if !next.is_directory {
                return Err(crate::Error::InvalidArgument(format!(
                    "exfat: {part:?} is not a directory"
                )));
            }
            cluster = next.first_cluster;
        }
        Ok(cluster)
    }

    /// Resolve `path` to its file entry set plus the cluster of the
    /// containing directory. Errors if `path` is the root.
    fn resolve_entry(&self, dev: &mut dyn BlockDevice, path: &str) -> Result<(FileEntrySet, u32)> {
        let parts = split_path(path);
        if parts.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "exfat: cannot resolve root as a file entry".into(),
            ));
        }
        let mut cluster = self.boot.first_cluster_of_root_directory;
        let (last, prefix) = parts.split_last().unwrap();
        for part in prefix {
            let bytes = self.read_chain_bytes(dev, cluster, false, None)?;
            let entries = iter_file_sets(&bytes)?;
            let next = entries
                .into_iter()
                .find(|e| self.name_matches(&e.name_utf16, part))
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "exfat: no such entry {part:?} under {path:?}"
                    ))
                })?;
            if !next.is_directory {
                return Err(crate::Error::InvalidArgument(format!(
                    "exfat: {part:?} is not a directory"
                )));
            }
            cluster = next.first_cluster;
        }
        let bytes = self.read_chain_bytes(dev, cluster, false, None)?;
        let entries = iter_file_sets(&bytes)?;
        let found = entries
            .into_iter()
            .find(|e| self.name_matches(&e.name_utf16, last))
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "exfat: no such entry {last:?} under {path:?}"
                ))
            })?;
        Ok((found, cluster))
    }

    /// Case-insensitive name comparison via the volume's up-case table.
    fn name_matches(&self, on_disk: &[u16], query: &str) -> bool {
        let q: Vec<u16> = query.encode_utf16().collect();
        self.upcase.eq_ignore_case(on_disk, &q)
    }

    /// Build the ordered list of clusters that hold a file's data.
    ///
    /// `no_fat_chain == true` means the file is contiguous starting at
    /// `first_cluster` and the FAT entries are not required to be valid
    /// — the cluster count is derived from `data_length`.
    fn build_data_chain(
        &self,
        first_cluster: u32,
        no_fat_chain: bool,
        data_length: u64,
    ) -> Result<Vec<u32>> {
        if data_length == 0 {
            return Ok(Vec::new());
        }
        let cb = self.boot.bytes_per_cluster() as u64;
        let n = data_length.div_ceil(cb) as u32;
        if no_fat_chain {
            // Contiguous run.
            return Ok((0..n).map(|i| first_cluster + i).collect());
        }
        // Otherwise walk the FAT chain — but only return as many clusters
        // as the file actually needs. (The FAT chain may, in principle,
        // run longer; in well-formed images it does not.)
        let chain = self.fat.chain(first_cluster)?;
        if (chain.len() as u32) < n {
            return Err(crate::Error::InvalidImage(format!(
                "exfat: FAT chain has {} clusters but file needs {n}",
                chain.len()
            )));
        }
        Ok(chain.into_iter().take(n as usize).collect())
    }

    /// Read every byte of the data referenced by a cluster chain.
    /// `hint_byte_len` truncates the result; `None` reads the full
    /// chain.
    fn read_chain_bytes(
        &self,
        dev: &mut dyn BlockDevice,
        first_cluster: u32,
        no_fat_chain: bool,
        hint_byte_len: Option<u64>,
    ) -> Result<Vec<u8>> {
        if first_cluster == 0 {
            return Ok(Vec::new());
        }
        let cb = self.boot.bytes_per_cluster() as u64;
        let chain = if no_fat_chain {
            let n = match hint_byte_len {
                Some(b) if b > 0 => b.div_ceil(cb) as u32,
                _ => 1,
            };
            (0..n).map(|i| first_cluster + i).collect()
        } else {
            self.fat.chain(first_cluster)?
        };
        let total = chain.len() as u64 * cb;
        let out_len = match hint_byte_len {
            Some(b) => b.min(total),
            None => total,
        };
        let mut out = vec![0u8; out_len as usize];
        let mut pos = 0usize;
        for &c in &chain {
            if pos >= out.len() {
                break;
            }
            let take = (out.len() - pos).min(cb as usize);
            let off = self.boot.cluster_byte_offset(c);
            dev.read_at(off, &mut out[pos..pos + take])?;
            pos += take;
        }
        Ok(out)
    }

    /// Walk the root directory's slots looking for the three metadata
    /// entries we care about (AllocationBitmap, UpcaseTable,
    /// VolumeLabel). Returns the parsed volume label, up-case table, and
    /// the AllocationBitmap's `(first_cluster, data_length)` pair if found.
    fn scan_root_metadata(
        &self,
        dev: &mut dyn BlockDevice,
        root_bytes: &[u8],
    ) -> Result<RootMetadata> {
        let mut volume_label = String::new();
        let mut upcase = Upcase::ascii();
        let mut bitmap_info: Option<(u32, u64)> = None;

        let mut i = 0;
        while i + ENTRY_SIZE <= root_bytes.len() {
            let slot: &[u8; ENTRY_SIZE] = (&root_bytes[i..i + ENTRY_SIZE]).try_into().unwrap();
            match dir::classify_slot(slot) {
                RawSlot::EndOfDirectory => break,
                RawSlot::Unused => {
                    i += ENTRY_SIZE;
                }
                RawSlot::VolumeLabel(units) => {
                    volume_label = dir::decode_volume_label(&units);
                    i += ENTRY_SIZE;
                }
                RawSlot::UpcaseTable {
                    first_cluster,
                    data_length,
                    ..
                } => {
                    // Read DataLength bytes from the up-case table's
                    // cluster chain. The up-case table is typically a
                    // single FAT chain (not NoFatChain), but the spec
                    // permits either; we always walk the FAT.
                    let raw = self.read_chain_bytes(
                        dev,
                        first_cluster,
                        /* no_fat_chain */ false,
                        Some(data_length),
                    )?;
                    upcase = match Upcase::decode(&raw, data_length) {
                        Ok(u) => u,
                        Err(_) => Upcase::ascii(),
                    };
                    i += ENTRY_SIZE;
                }
                RawSlot::AllocationBitmap {
                    first_cluster,
                    data_length,
                    ..
                } => {
                    bitmap_info = Some((first_cluster, data_length));
                    i += ENTRY_SIZE;
                }
                RawSlot::File {
                    secondary_count, ..
                } => {
                    // Skip the whole set — we're only metadata-mining
                    // here.
                    i += (1 + secondary_count as usize) * ENTRY_SIZE;
                }
                RawSlot::Other { .. } => {
                    i += ENTRY_SIZE;
                }
            }
        }
        Ok(RootMetadata {
            volume_label,
            upcase,
            bitmap_info,
        })
    }

    // ===================================================================
    // Writer API — section below covers cluster allocation, FAT updates,
    // directory-entry mutation, file/dir create + remove, and flush.
    // ===================================================================

    /// Allocate one free cluster, mark it used in the FAT (as a one-cluster
    /// EOC chain) and in the allocation bitmap. Returns the cluster number.
    fn alloc_cluster(&mut self) -> Result<u32> {
        let max = (self.boot.cluster_count + 2) as usize;
        let start = self.next_free_hint.max(2) as usize;
        for cluster in start..max {
            if self.fat.raw(cluster as u32) == Some(fat::FREE) {
                self.fat.set_raw(cluster as u32, fat::EOC);
                self.fat_dirty = true;
                set_bitmap_bit(&mut self.bitmap, cluster as u32, true);
                self.bitmap_dirty = true;
                self.next_free_hint = (cluster as u32).saturating_add(1);
                return Ok(cluster as u32);
            }
        }
        // Wrap and retry from cluster 2.
        for cluster in 2..start {
            if self.fat.raw(cluster as u32) == Some(fat::FREE) {
                self.fat.set_raw(cluster as u32, fat::EOC);
                self.fat_dirty = true;
                set_bitmap_bit(&mut self.bitmap, cluster as u32, true);
                self.bitmap_dirty = true;
                self.next_free_hint = (cluster as u32).saturating_add(1);
                return Ok(cluster as u32);
            }
        }
        Err(crate::Error::InvalidArgument(
            "exfat: out of clusters".into(),
        ))
    }

    /// Allocate `n` clusters and link them into a single FAT chain
    /// (terminated with EOC). Returns the first cluster of the chain.
    fn alloc_chain(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Err(crate::Error::InvalidArgument(
                "exfat: alloc_chain(0)".into(),
            ));
        }
        let first = self.alloc_cluster()?;
        let mut prev = first;
        for _ in 1..n {
            let next = self.alloc_cluster()?;
            self.fat.set_raw(prev, next);
            prev = next;
        }
        // Last cluster stays EOC.
        Ok(first)
    }

    /// Free the FAT chain starting at `first_cluster` (no-op if already
    /// free / cluster is zero). Clears the matching bitmap bits.
    fn free_chain(&mut self, first_cluster: u32) -> Result<()> {
        if first_cluster < 2 {
            return Ok(());
        }
        let max = self.boot.cluster_count + 2;
        let mut cur = first_cluster;
        let mut steps = 0u64;
        loop {
            if cur < 2 || cur >= max {
                break;
            }
            steps += 1;
            if steps > self.boot.cluster_count as u64 + 2 {
                return Err(crate::Error::InvalidImage(
                    "exfat: cluster chain loops while freeing".into(),
                ));
            }
            let raw = self.fat.raw(cur).unwrap_or(fat::FREE);
            self.fat.set_raw(cur, fat::FREE);
            set_bitmap_bit(&mut self.bitmap, cur, false);
            if self.next_free_hint > cur {
                self.next_free_hint = cur;
            }
            match fat::classify(raw) {
                fat::FatEntry::Eoc | fat::FatEntry::Free | fat::FatEntry::Bad => break,
                fat::FatEntry::Next(n) => cur = n,
            }
        }
        self.fat_dirty = true;
        self.bitmap_dirty = true;
        Ok(())
    }

    /// Walk the FAT chain of a directory and collect every cluster.
    fn dir_chain(&self, first_cluster: u32) -> Result<Vec<u32>> {
        self.fat.chain(first_cluster)
    }

    /// Read the entire byte content of a directory (all its clusters in
    /// FAT order).
    fn read_dir_bytes(&self, dev: &mut dyn BlockDevice, first_cluster: u32) -> Result<Vec<u8>> {
        self.read_chain_bytes(dev, first_cluster, false, None)
    }

    /// Resolve a path to the cluster of the *parent directory* of the
    /// terminal component, plus the terminal component name. Errors on
    /// the root path.
    fn split_path_for_create<'p>(
        &self,
        dev: &mut dyn BlockDevice,
        path: &'p str,
    ) -> Result<(u32, &'p str)> {
        let parts = split_path(path);
        if parts.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "exfat: cannot create root".into(),
            ));
        }
        let (last, prefix) = parts.split_last().unwrap();
        let mut cluster = self.boot.first_cluster_of_root_directory;
        for part in prefix {
            let bytes = self.read_dir_bytes(dev, cluster)?;
            let next = iter_file_sets(&bytes)?
                .into_iter()
                .find(|e| self.name_matches(&e.name_utf16, part))
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "exfat: no such entry {part:?} under {path:?}"
                    ))
                })?;
            if !next.is_directory {
                return Err(crate::Error::InvalidArgument(format!(
                    "exfat: {part:?} is not a directory"
                )));
            }
            cluster = next.first_cluster;
        }
        Ok((cluster, last))
    }

    /// Append `entry_set_bytes` (a 32n-byte file entry set) to the
    /// directory whose chain begins at `first_cluster`, allocating a new
    /// cluster on overflow and writing the bytes through `dev`.
    ///
    /// Strategy: walk the chain, scan each cluster for a run of
    /// `entry_set_bytes.len() / 32` consecutive free slots (`0x00` or
    /// high-bit-clear), then write into that run. If no run exists,
    /// extend the chain by one cluster and place the bytes at the start
    /// of the new cluster.
    fn append_to_directory(
        &mut self,
        dev: &mut dyn BlockDevice,
        first_cluster: u32,
        entry_set_bytes: &[u8],
    ) -> Result<()> {
        let cb = self.boot.bytes_per_cluster() as usize;
        let need_slots = entry_set_bytes.len() / ENTRY_SIZE;
        if entry_set_bytes.len() % ENTRY_SIZE != 0 || need_slots == 0 {
            return Err(crate::Error::InvalidArgument(
                "exfat: entry set must be a non-empty multiple of 32 bytes".into(),
            ));
        }
        let chain = self.dir_chain(first_cluster)?;
        for &cluster in &chain {
            let off = self.boot.cluster_byte_offset(cluster);
            let mut buf = vec![0u8; cb];
            dev.read_at(off, &mut buf)?;
            // Scan slot-by-slot for a run of free slots large enough.
            let mut i = 0;
            while i + need_slots * ENTRY_SIZE <= cb {
                let mut all_free = true;
                for j in 0..need_slots {
                    let slot_off = i + j * ENTRY_SIZE;
                    let t = buf[slot_off];
                    if t != 0x00 && t & dir::ENTRY_INUSE != 0 {
                        all_free = false;
                        break;
                    }
                }
                if all_free {
                    buf[i..i + entry_set_bytes.len()].copy_from_slice(entry_set_bytes);
                    dev.write_at(off, &buf)?;
                    return Ok(());
                }
                // Skip past a known in-use set (or one slot).
                let t = buf[i];
                if t & dir::ENTRY_INUSE != 0 && t == dir::ENTRY_FILE {
                    let sec = buf[i + 1] as usize;
                    i += (1 + sec) * ENTRY_SIZE;
                } else {
                    i += ENTRY_SIZE;
                }
            }
        }
        // No room in any existing cluster — extend the chain.
        //
        // Before linking a new cluster, we must ensure no `0x00` byte at
        // slot[0] terminates the directory scan inside any earlier
        // cluster. Reader semantics: `t == 0x00` at slot[0] is end-of-
        // directory and stops iteration. So we walk every cluster of
        // the existing chain and rewrite each slot whose type byte is
        // 0x00 to 0x05 (an Unused slot — high bit clear, non-zero).
        // This is safe: readers classify high-bit-clear as Unused and
        // continue past it.
        for &cluster in &chain {
            let off = self.boot.cluster_byte_offset(cluster);
            let mut buf = vec![0u8; cb];
            dev.read_at(off, &mut buf)?;
            let mut changed = false;
            let mut i = 0;
            while i + ENTRY_SIZE <= cb {
                let t = buf[i];
                if t == 0x00 {
                    buf[i] = 0x05; // any value with high bit clear works
                    changed = true;
                    i += ENTRY_SIZE;
                } else if t & dir::ENTRY_INUSE != 0 && t == dir::ENTRY_FILE {
                    let sec = buf[i + 1] as usize;
                    i += (1 + sec) * ENTRY_SIZE;
                } else {
                    i += ENTRY_SIZE;
                }
            }
            if changed {
                dev.write_at(off, &buf)?;
            }
        }
        let new_cluster = self.alloc_cluster()?;
        let last = *chain.last().unwrap();
        self.fat.set_raw(last, new_cluster);
        self.fat_dirty = true;
        let mut buf = vec![0u8; cb];
        buf[..entry_set_bytes.len()].copy_from_slice(entry_set_bytes);
        let off = self.boot.cluster_byte_offset(new_cluster);
        dev.write_at(off, &buf)?;
        Ok(())
    }

    /// Find an existing entry by name in the directory at `first_cluster`,
    /// returning its byte offset within the directory bytes plus the
    /// parsed file entry set. Used by `remove`.
    fn find_entry_in_dir(
        &self,
        dev: &mut dyn BlockDevice,
        first_cluster: u32,
        name: &str,
    ) -> Result<Option<(u64, FileEntrySet, usize)>> {
        let bytes = self.read_dir_bytes(dev, first_cluster)?;
        let mut i = 0;
        while i + ENTRY_SIZE <= bytes.len() {
            let slot: &[u8; ENTRY_SIZE] = (&bytes[i..i + ENTRY_SIZE]).try_into().unwrap();
            match dir::classify_slot(slot) {
                RawSlot::EndOfDirectory => break,
                RawSlot::Unused => {
                    i += ENTRY_SIZE;
                }
                RawSlot::File {
                    secondary_count, ..
                } => {
                    let total = (1 + secondary_count as usize) * ENTRY_SIZE;
                    if i + total > bytes.len() {
                        break;
                    }
                    let set = dir::parse_file_set(&bytes[i..i + total])?;
                    if self.name_matches(&set.name_utf16, name) {
                        return Ok(Some((i as u64, set, total)));
                    }
                    i += total;
                }
                _ => {
                    i += ENTRY_SIZE;
                }
            }
        }
        Ok(None)
    }

    /// Compute the on-disk byte offset of a position within a directory
    /// whose chain begins at `first_cluster`. `pos_in_dir` is the byte
    /// offset within the directory's logical bytes (all clusters
    /// concatenated).
    fn dir_pos_to_disk_offset(&self, first_cluster: u32, pos_in_dir: u64) -> Result<u64> {
        let cb = self.boot.bytes_per_cluster() as u64;
        let chain = self.dir_chain(first_cluster)?;
        let cluster_idx = (pos_in_dir / cb) as usize;
        let cluster_off = pos_in_dir % cb;
        if cluster_idx >= chain.len() {
            return Err(crate::Error::InvalidImage(
                "exfat: dir position past chain".into(),
            ));
        }
        Ok(self.boot.cluster_byte_offset(chain[cluster_idx]) + cluster_off)
    }

    /// Mark an entry set's slots as deleted (clear the InUse high bit on
    /// every entry-type byte). Writes back through `dev`.
    fn clear_entry_set(
        &mut self,
        dev: &mut dyn BlockDevice,
        parent_first_cluster: u32,
        pos_in_dir: u64,
        total_bytes: usize,
    ) -> Result<()> {
        let cb = self.boot.bytes_per_cluster() as u64;
        let n_slots = total_bytes / ENTRY_SIZE;
        let chain = self.dir_chain(parent_first_cluster)?;
        for k in 0..n_slots {
            let p = pos_in_dir + (k as u64) * ENTRY_SIZE as u64;
            let cluster_idx = (p / cb) as usize;
            let cluster_off = p % cb;
            let cluster = chain[cluster_idx];
            let disk_off = self.boot.cluster_byte_offset(cluster) + cluster_off;
            // Read 1 byte, clear high bit, write 1 byte.
            let mut byte = [0u8; 1];
            dev.read_at(disk_off, &mut byte)?;
            byte[0] &= !dir::ENTRY_INUSE;
            dev.write_at(disk_off, &byte)?;
        }
        Ok(())
    }

    /// Stream the bytes of `reader` into the cluster chain starting at
    /// `first_cluster`. Reads up to `total_len` bytes from the reader in
    /// 64 KiB chunks. Trailing bytes of the last cluster are zeroed.
    fn stream_into_chain(
        &self,
        dev: &mut dyn BlockDevice,
        chain: &[u32],
        reader: &mut dyn std::io::Read,
        total_len: u64,
    ) -> Result<()> {
        let cb = self.boot.bytes_per_cluster() as u64;
        let mut scratch = vec![0u8; SCRATCH_BUF_BYTES];
        let mut remaining = total_len;
        let mut cluster_idx = 0usize;
        let mut cluster_off: u64 = 0;
        while remaining > 0 {
            if cluster_idx >= chain.len() {
                return Err(crate::Error::InvalidImage(
                    "exfat: writer exhausted cluster chain".into(),
                ));
            }
            let want = (remaining as usize)
                .min(scratch.len())
                .min((cb - cluster_off) as usize);
            let mut got = 0;
            while got < want {
                let n = reader
                    .read(&mut scratch[got..want])
                    .map_err(crate::Error::Io)?;
                if n == 0 {
                    return Err(crate::Error::InvalidArgument(
                        "exfat: reader produced fewer bytes than declared length".into(),
                    ));
                }
                got += n;
            }
            let disk_off = self.boot.cluster_byte_offset(chain[cluster_idx]) + cluster_off;
            dev.write_at(disk_off, &scratch[..got])?;
            remaining -= got as u64;
            cluster_off += got as u64;
            if cluster_off == cb {
                cluster_idx += 1;
                cluster_off = 0;
            }
        }
        // Zero the tail of the last cluster (if we ended mid-cluster).
        if !chain.is_empty() && cluster_off > 0 && cluster_off < cb {
            let zeros = vec![0u8; (cb - cluster_off) as usize];
            let disk_off = self.boot.cluster_byte_offset(chain[cluster_idx]) + cluster_off;
            dev.write_at(disk_off, &zeros)?;
        }
        Ok(())
    }

    /// Compute the exFAT NameHash over a file name, using the volume's
    /// up-case table. The hash is computed over the up-cased name as a
    /// little-endian byte stream.
    fn name_hash_for(&self, name: &str) -> u16 {
        let units: Vec<u16> = name.encode_utf16().collect();
        let upcased = self.upcase.up_slice(&units);
        let mut bytes = Vec::with_capacity(upcased.len() * 2);
        for u in &upcased {
            bytes.extend_from_slice(&u.to_le_bytes());
        }
        dir::name_hash(&bytes)
    }

    /// Create a new empty regular file under `dir_cluster` named `name`.
    /// Returns the new entry's first cluster (or 0 if the file is empty).
    /// `data_length` clusters are allocated up-front; bytes are then
    /// streamed through `reader`. If `data_length == 0` no clusters are
    /// allocated and the entry's `first_cluster` is left zero.
    pub fn create_file_in(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_cluster: u32,
        name: &str,
        reader: &mut dyn std::io::Read,
        data_length: u64,
        timestamp: u32,
    ) -> Result<u32> {
        // Reject names that contain a path separator.
        if name.is_empty() || name.contains('/') || name.contains('\\') {
            return Err(crate::Error::InvalidArgument(format!(
                "exfat: invalid file name {name:?}"
            )));
        }
        let cb = self.boot.bytes_per_cluster() as u64;
        let (first_cluster, chain) = if data_length > 0 {
            let n_clusters = data_length.div_ceil(cb) as u32;
            let first = self.alloc_chain(n_clusters)?;
            let chain = self.dir_chain(first)?;
            (first, chain)
        } else {
            (0, Vec::new())
        };
        if data_length > 0 {
            self.stream_into_chain(dev, &chain, reader, data_length)?;
        }

        // Per the spec, when DataLength == 0 the FirstCluster must be 0
        // and the AllocationPossible flag must be clear.
        let secondary_flags = if data_length == 0 {
            0
        } else {
            dir::SECFLAG_ALLOC_POSSIBLE
        };
        let entry = format::make_file_entry_set(
            name,
            /* is_directory */ false,
            secondary_flags,
            first_cluster,
            data_length,
            data_length,
            timestamp,
            self.name_hash_for(name),
        );
        self.append_to_directory(dev, dir_cluster, &entry)?;
        Ok(first_cluster)
    }

    /// Public helper: create a regular file at `path` with bytes from
    /// `reader`. `data_length` must equal the number of bytes the
    /// reader will produce. Use `0` for an empty file (reader is not
    /// consulted).
    pub fn create_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        reader: &mut dyn std::io::Read,
        data_length: u64,
        timestamp: u32,
    ) -> Result<u32> {
        let (parent_cluster, name) = {
            let (c, n) = self.split_path_for_create(dev, path)?;
            (c, n.to_string())
        };
        self.create_file_in(dev, parent_cluster, &name, reader, data_length, timestamp)
    }

    /// Create a new directory under `dir_cluster` named `name`. Returns
    /// the first cluster of the new directory.
    pub fn create_dir_in(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_cluster: u32,
        name: &str,
        timestamp: u32,
    ) -> Result<u32> {
        if name.is_empty() || name.contains('/') || name.contains('\\') {
            return Err(crate::Error::InvalidArgument(format!(
                "exfat: invalid directory name {name:?}"
            )));
        }
        let cb = self.boot.bytes_per_cluster();
        let new_cluster = self.alloc_cluster()?;
        // Zero the new directory cluster.
        let zeros = vec![0u8; cb as usize];
        dev.write_at(self.boot.cluster_byte_offset(new_cluster), &zeros)?;

        let entry = format::make_file_entry_set(
            name,
            /* is_directory */ true,
            dir::SECFLAG_ALLOC_POSSIBLE,
            new_cluster,
            cb as u64,
            cb as u64,
            timestamp,
            self.name_hash_for(name),
        );
        self.append_to_directory(dev, dir_cluster, &entry)?;
        Ok(new_cluster)
    }

    /// Public helper: create a directory at `path`.
    pub fn create_dir(
        &mut self,
        dev: &mut dyn BlockDevice,
        path: &str,
        timestamp: u32,
    ) -> Result<u32> {
        let (parent_cluster, name) = {
            let (c, n) = self.split_path_for_create(dev, path)?;
            (c, n.to_string())
        };
        self.create_dir_in(dev, parent_cluster, &name, timestamp)
    }

    /// Remove the entry at `path`. For directories, fails if non-empty.
    pub fn remove(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<()> {
        let parts = split_path(path);
        if parts.is_empty() {
            return Err(crate::Error::InvalidArgument(
                "exfat: cannot remove root".into(),
            ));
        }
        let (last, prefix) = parts.split_last().unwrap();
        let mut parent_cluster = self.boot.first_cluster_of_root_directory;
        for part in prefix {
            let bytes = self.read_dir_bytes(dev, parent_cluster)?;
            let next = iter_file_sets(&bytes)?
                .into_iter()
                .find(|e| self.name_matches(&e.name_utf16, part))
                .ok_or_else(|| {
                    crate::Error::InvalidArgument(format!(
                        "exfat: no such entry {part:?} under {path:?}"
                    ))
                })?;
            if !next.is_directory {
                return Err(crate::Error::InvalidArgument(format!(
                    "exfat: {part:?} is not a directory"
                )));
            }
            parent_cluster = next.first_cluster;
        }
        let (pos, set, total) = self
            .find_entry_in_dir(dev, parent_cluster, last)?
            .ok_or_else(|| {
                crate::Error::InvalidArgument(format!(
                    "exfat: no such entry {last:?} under {path:?}"
                ))
            })?;
        if set.is_directory {
            // Verify the directory is empty (only EndOfDir / unused slots).
            let bytes = self.read_dir_bytes(dev, set.first_cluster)?;
            let mut i = 0;
            let mut has_entries = false;
            while i + ENTRY_SIZE <= bytes.len() {
                let slot: &[u8; ENTRY_SIZE] = (&bytes[i..i + ENTRY_SIZE]).try_into().unwrap();
                match dir::classify_slot(slot) {
                    RawSlot::EndOfDirectory => break,
                    RawSlot::Unused => {
                        i += ENTRY_SIZE;
                    }
                    RawSlot::File { .. } => {
                        has_entries = true;
                        break;
                    }
                    _ => i += ENTRY_SIZE,
                }
            }
            if has_entries {
                return Err(crate::Error::InvalidArgument(format!(
                    "exfat: directory {last:?} is not empty"
                )));
            }
        }
        // Clear the in-use bits on each slot of the entry set.
        let _ = self.dir_pos_to_disk_offset(parent_cluster, pos)?; // sanity
        self.clear_entry_set(dev, parent_cluster, pos, total)?;
        // Free the data cluster chain (only if first_cluster > 0).
        if set.first_cluster >= 2 {
            self.free_chain(set.first_cluster)?;
        }
        Ok(())
    }

    /// Flush all dirty state back to disk: rewrite the FAT image (both
    /// copies if NumberOfFats == 2 — currently we only emit one), rewrite
    /// the allocation bitmap, and `sync()` the device.
    pub fn flush(&mut self, dev: &mut dyn BlockDevice) -> Result<()> {
        if self.fat_dirty {
            let fat_off = self.boot.fat_byte_offset();
            let fat_byte_len = self.boot.fat_byte_length() as usize;
            let mut buf = vec![0u8; fat_byte_len];
            let enc = self.fat.encode();
            let n = enc.len().min(fat_byte_len);
            buf[..n].copy_from_slice(&enc[..n]);
            dev.write_at(fat_off, &buf)?;
            // If a backup FAT exists, mirror to it.
            if self.boot.number_of_fats == 2 {
                let backup_off = fat_off + fat_byte_len as u64;
                if backup_off + fat_byte_len as u64
                    <= self.boot.volume_length * self.boot.bytes_per_sector() as u64
                {
                    dev.write_at(backup_off, &buf)?;
                }
            }
            self.fat_dirty = false;
        }
        if self.bitmap_dirty && self.bitmap_first_cluster >= 2 {
            // Walk the bitmap's cluster chain and write the bitmap bytes
            // across them.
            let chain = self.dir_chain(self.bitmap_first_cluster)?;
            let cb = self.boot.bytes_per_cluster() as usize;
            let bm = &self.bitmap;
            let mut pos = 0usize;
            for cluster in chain {
                if pos >= bm.len() {
                    break;
                }
                let take = (bm.len() - pos).min(cb);
                let mut chunk = vec![0u8; cb];
                chunk[..take].copy_from_slice(&bm[pos..pos + take]);
                dev.write_at(self.boot.cluster_byte_offset(cluster), &chunk)?;
                pos += take;
            }
            self.bitmap_dirty = false;
        }
        dev.sync()?;
        Ok(())
    }
}

/// Set or clear bit `cluster - 2` in the allocation bitmap. No-op if the
/// cluster index is outside the bitmap.
fn set_bitmap_bit(bitmap: &mut [u8], cluster: u32, used: bool) {
    if cluster < 2 {
        return;
    }
    let bit = (cluster - 2) as usize;
    let byte = bit / 8;
    let mask = 1u8 << (bit % 8);
    if byte >= bitmap.len() {
        return;
    }
    if used {
        bitmap[byte] |= mask;
    } else {
        bitmap[byte] &= !mask;
    }
}

/// Walk `bytes` slot-by-slot and assemble every file entry set found.
/// Sets that fail checksum validation are propagated as
/// [`crate::Error::InvalidImage`].
fn iter_file_sets(bytes: &[u8]) -> Result<Vec<FileEntrySet>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + ENTRY_SIZE <= bytes.len() {
        let slot: &[u8; ENTRY_SIZE] = (&bytes[i..i + ENTRY_SIZE]).try_into().unwrap();
        match dir::classify_slot(slot) {
            RawSlot::EndOfDirectory => break,
            RawSlot::Unused => {
                i += ENTRY_SIZE;
            }
            RawSlot::File {
                secondary_count, ..
            } => {
                let total = (1 + secondary_count as usize) * ENTRY_SIZE;
                if i + total > bytes.len() {
                    return Err(crate::Error::InvalidImage(
                        "exfat: file entry set runs past directory end".into(),
                    ));
                }
                let set = dir::parse_file_set(&bytes[i..i + total])?;
                out.push(set);
                i += total;
            }
            _ => {
                // Skip metadata entries (AllocationBitmap, UpcaseTable,
                // VolumeLabel) and anything unrecognised.
                i += ENTRY_SIZE;
            }
        }
    }
    Ok(out)
}

/// Split an absolute or relative path into its non-empty components.
/// `/`, `""`, and `.` all yield an empty vec (= "the root").
fn split_path(path: &str) -> Vec<&str> {
    path.split(['/', '\\'])
        .filter(|p| !p.is_empty() && *p != ".")
        .collect()
}

/// Streaming reader for an exFAT file. Walks the file's cluster vector
/// on demand; the file is never buffered beyond the destination of one
/// `read` call.
pub struct ExfatFileReader<'a> {
    dev: &'a mut dyn BlockDevice,
    chain: Vec<u32>,
    cluster_heap_offset: u64,
    cluster_bytes: u64,
    /// Bytes of the file still to be returned (capped to ValidDataLength).
    remaining: u64,
    cluster_idx: usize,
    cluster_off: u64,
}

impl<'a> std::io::Read for ExfatFileReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 || self.cluster_idx >= self.chain.len() {
            return Ok(0);
        }
        let avail_in_cluster = self.cluster_bytes - self.cluster_off;
        let want = (buf.len() as u64).min(avail_in_cluster).min(self.remaining) as usize;
        let cluster = self.chain[self.cluster_idx];
        let cluster_start = self.cluster_heap_offset + (cluster as u64 - 2) * self.cluster_bytes;
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

/// Probe for the exFAT boot-sector signature: `"EXFAT   "` at offset 3
/// of LBA 0 (after the 3-byte jump instruction).
pub fn probe(dev: &mut dyn BlockDevice) -> Result<bool> {
    if dev.total_size() < 512 {
        return Ok(false);
    }
    let mut head = [0u8; 16];
    dev.read_at(0, &mut head)?;
    Ok(&head[3..11] == b"EXFAT   ")
}

#[cfg(test)]
mod tests {
    //! End-to-end test on a synthetic in-memory exFAT image we build
    //! ourselves. The image has:
    //!
    //! - 512-byte sectors, 8 sectors/cluster (4 KiB clusters)
    //! - 1 FAT
    //! - A single regular file "hello.txt" with "Hello, exFAT!\n"
    //! - A subdirectory "sub" containing "x.bin" (12 bytes)
    //! - An ASCII-only up-case table
    //! - Volume label "MYVOL"
    //!
    //! The image is just big enough to fit metadata + a few clusters of
    //! data; total ≈ 1 MiB.
    use super::*;
    use crate::block::{BlockDevice, MemoryBackend};

    const BPS_SHIFT: u8 = 9; // 512 bytes/sector
    const SPC_SHIFT: u8 = 3; // 8 sectors/cluster = 4 KiB clusters
    const BPS: u32 = 1 << BPS_SHIFT;
    const BPC: u32 = BPS << SPC_SHIFT;

    /// Cluster layout, all picked to keep things simple:
    ///   2 = AllocationBitmap (we just zero it; it isn't checked on read)
    ///   3 = UpcaseTable     (ASCII, identity 0..0x80)
    ///   4 = root directory  (a single cluster of entries)
    ///   5 = "hello.txt"     (one cluster of data, 14 bytes used)
    ///   6 = "sub" directory (one cluster)
    ///   7 = "x.bin"         (one cluster, 12 bytes used)
    const CL_BITMAP: u32 = 2;
    const CL_UPCASE: u32 = 3;
    const CL_ROOT: u32 = 4;
    const CL_HELLO: u32 = 5;
    const CL_SUB: u32 = 6;
    const CL_XBIN: u32 = 7;

    /// Build a minimal exFAT image in memory and return the backend.
    fn build_test_image() -> MemoryBackend {
        // Volume geometry — keep tiny.
        const FAT_OFFSET_SECTORS: u32 = 64; // 32 KiB in
        const FAT_LENGTH_SECTORS: u32 = 8; // 8 sectors = 4 KiB → 1024 entries
        const CLUSTER_HEAP_OFFSET_SECTORS: u32 = 128; // 64 KiB in
        const CLUSTER_COUNT: u32 = 32;
        const VOLUME_LENGTH_SECTORS: u64 =
            CLUSTER_HEAP_OFFSET_SECTORS as u64 + (CLUSTER_COUNT as u64 * (1u64 << SPC_SHIFT));
        let total_bytes = VOLUME_LENGTH_SECTORS * BPS as u64;
        let mut dev = MemoryBackend::new(total_bytes);

        // Boot sector.
        let mut bs = [0u8; 512];
        bs[0..3].copy_from_slice(&[0xEB, 0x76, 0x90]);
        bs[3..11].copy_from_slice(b"EXFAT   ");
        // 11..64 MustBeZero
        bs[64..72].copy_from_slice(&0u64.to_le_bytes());
        bs[72..80].copy_from_slice(&VOLUME_LENGTH_SECTORS.to_le_bytes());
        bs[80..84].copy_from_slice(&FAT_OFFSET_SECTORS.to_le_bytes());
        bs[84..88].copy_from_slice(&FAT_LENGTH_SECTORS.to_le_bytes());
        bs[88..92].copy_from_slice(&CLUSTER_HEAP_OFFSET_SECTORS.to_le_bytes());
        bs[92..96].copy_from_slice(&CLUSTER_COUNT.to_le_bytes());
        bs[96..100].copy_from_slice(&CL_ROOT.to_le_bytes());
        bs[100..104].copy_from_slice(&0xCAFE_F00Du32.to_le_bytes());
        bs[104..106].copy_from_slice(&0x0100u16.to_le_bytes());
        bs[106..108].copy_from_slice(&0u16.to_le_bytes());
        bs[108] = BPS_SHIFT;
        bs[109] = SPC_SHIFT;
        bs[110] = 1; // NumberOfFats
        bs[111] = 0x80;
        bs[112] = 0;
        bs[510] = 0x55;
        bs[511] = 0xAA;
        dev.write_at(0, &bs).unwrap();

        // FAT — initialise reserved entries + chains.
        let fat_bytes_len = FAT_LENGTH_SECTORS as usize * BPS as usize;
        let mut fat = vec![0u8; fat_bytes_len];
        let write_entry = |fat: &mut [u8], cluster: u32, value: u32| {
            let off = cluster as usize * 4;
            fat[off..off + 4].copy_from_slice(&value.to_le_bytes());
        };
        write_entry(&mut fat, 0, 0xFFFFFFF8);
        write_entry(&mut fat, 1, 0xFFFFFFFF);
        // Each used cluster is a one-cluster chain (EOC).
        for c in [CL_BITMAP, CL_UPCASE, CL_ROOT, CL_HELLO, CL_SUB, CL_XBIN] {
            write_entry(&mut fat, c, 0xFFFFFFFF);
        }
        let fat_off = FAT_OFFSET_SECTORS as u64 * BPS as u64;
        dev.write_at(fat_off, &fat).unwrap();

        // Helper: byte offset of cluster N.
        let cluster_off = |c: u32| -> u64 {
            CLUSTER_HEAP_OFFSET_SECTORS as u64 * BPS as u64 + (c as u64 - 2) * BPC as u64
        };

        // Up-case table (ASCII identity 0..0x80, with a..z → A..Z).
        let mut upcase = Vec::new();
        for i in 0..0x80u16 {
            let c = i as u8;
            let v = if c.is_ascii_lowercase() {
                (c - b'a' + b'A') as u16
            } else {
                i
            };
            upcase.extend_from_slice(&v.to_le_bytes());
        }
        let upcase_len = upcase.len() as u64;
        let upcase_checksum = super::upcase::table_checksum(&upcase);
        let mut upcase_cluster = vec![0u8; BPC as usize];
        upcase_cluster[..upcase.len()].copy_from_slice(&upcase);
        dev.write_at(cluster_off(CL_UPCASE), &upcase_cluster)
            .unwrap();

        // File: "hello.txt" contents.
        let hello_text = b"Hello, exFAT!\n";
        let mut hello_cluster = vec![0u8; BPC as usize];
        hello_cluster[..hello_text.len()].copy_from_slice(hello_text);
        dev.write_at(cluster_off(CL_HELLO), &hello_cluster).unwrap();

        // File: "x.bin" contents.
        let xbin_data: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let mut xbin_cluster = vec![0u8; BPC as usize];
        xbin_cluster[..xbin_data.len()].copy_from_slice(&xbin_data);
        dev.write_at(cluster_off(CL_XBIN), &xbin_cluster).unwrap();

        // Sub directory: holds entry-set for "x.bin".
        let sub_entries = build_dir_entries(&[("x.bin", false, CL_XBIN, xbin_data.len() as u64)]);
        let mut sub_cluster = vec![0u8; BPC as usize];
        sub_cluster[..sub_entries.len()].copy_from_slice(&sub_entries);
        dev.write_at(cluster_off(CL_SUB), &sub_cluster).unwrap();

        // Root directory: volume label + bitmap + upcase + file +
        // directory entry sets.
        let mut root = Vec::new();

        // VolumeLabel "MYVOL"
        {
            let label_units: Vec<u16> = "MYVOL".encode_utf16().collect();
            let mut e = [0u8; ENTRY_SIZE];
            e[0] = dir::ENTRY_VOLUME_LABEL;
            e[1] = label_units.len() as u8;
            for (i, &u) in label_units.iter().enumerate() {
                let off = 2 + i * 2;
                e[off..off + 2].copy_from_slice(&u.to_le_bytes());
            }
            root.extend_from_slice(&e);
        }

        // AllocationBitmap — minimum-viable: flags=0, first_cluster=2, data_length=ceil(cluster_count/8)
        {
            let bitmap_bytes = (CLUSTER_COUNT as u64).div_ceil(8);
            let mut e = [0u8; ENTRY_SIZE];
            e[0] = dir::ENTRY_ALLOCATION_BITMAP;
            e[1] = 0; // bitmap flags
            e[20..24].copy_from_slice(&CL_BITMAP.to_le_bytes());
            e[24..32].copy_from_slice(&bitmap_bytes.to_le_bytes());
            root.extend_from_slice(&e);
        }

        // UpcaseTable
        {
            let mut e = [0u8; ENTRY_SIZE];
            e[0] = dir::ENTRY_UPCASE_TABLE;
            e[4..8].copy_from_slice(&upcase_checksum.to_le_bytes());
            e[20..24].copy_from_slice(&CL_UPCASE.to_le_bytes());
            e[24..32].copy_from_slice(&upcase_len.to_le_bytes());
            root.extend_from_slice(&e);
        }

        // File "hello.txt" (regular file, 14 bytes).
        root.extend_from_slice(&build_dir_entries(&[("hello.txt", false, CL_HELLO, 14)]));

        // Directory "sub" (size = one cluster — the spec records data_length
        // as the directory's total byte length).
        root.extend_from_slice(&build_dir_entries(&[("sub", true, CL_SUB, BPC as u64)]));

        let mut root_cluster = vec![0u8; BPC as usize];
        root_cluster[..root.len()].copy_from_slice(&root);
        dev.write_at(cluster_off(CL_ROOT), &root_cluster).unwrap();

        dev
    }

    /// Build a sequence of file entry sets, one per `(name, is_dir,
    /// first_cluster, data_length)`. Each set uses the NoFatChain flag
    /// off (we provide proper FAT chains).
    fn build_dir_entries(items: &[(&str, bool, u32, u64)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, is_dir, first_cluster, data_length) in items {
            let name_units: Vec<u16> = name.encode_utf16().collect();
            let n_name_entries = name_units.len().div_ceil(15).max(1);
            let secondary_count = (1 + n_name_entries) as u8;
            let attr = if *is_dir { dir::ATTR_DIRECTORY } else { 0 };

            // Primary.
            let mut primary = [0u8; ENTRY_SIZE];
            primary[0] = dir::ENTRY_FILE;
            primary[1] = secondary_count;
            // checksum filled below
            primary[4..6].copy_from_slice(&attr.to_le_bytes());

            // StreamExtension.
            let mut stream = [0u8; ENTRY_SIZE];
            stream[0] = dir::ENTRY_STREAM_EXTENSION;
            stream[1] = dir::SECFLAG_ALLOC_POSSIBLE; // FAT chain present
            stream[3] = name_units.len() as u8;
            // name_hash skipped; we don't enforce it on read.
            stream[8..16].copy_from_slice(&data_length.to_le_bytes()); // ValidDataLength
            stream[20..24].copy_from_slice(&first_cluster.to_le_bytes());
            stream[24..32].copy_from_slice(&data_length.to_le_bytes());

            // FileName slots.
            let mut names: Vec<[u8; ENTRY_SIZE]> = Vec::new();
            for chunk in name_units.chunks(15) {
                let mut e = [0u8; ENTRY_SIZE];
                e[0] = dir::ENTRY_FILE_NAME;
                for (i, &u) in chunk.iter().enumerate() {
                    let off = 2 + i * 2;
                    e[off..off + 2].copy_from_slice(&u.to_le_bytes());
                }
                names.push(e);
            }

            // Assemble in a buffer to compute the checksum.
            let mut set = Vec::new();
            set.extend_from_slice(&primary);
            set.extend_from_slice(&stream);
            for n in &names {
                set.extend_from_slice(n);
            }
            let csum = dir::set_checksum(&set);
            // Write checksum back into primary, then concat.
            set[2..4].copy_from_slice(&csum.to_le_bytes());
            out.extend_from_slice(&set);
        }
        out
    }

    #[test]
    fn open_decodes_boot_and_metadata() {
        let mut dev = build_test_image();
        let fs = Exfat::open(&mut dev).unwrap();
        assert_eq!(fs.cluster_size(), BPC);
        assert_eq!(fs.sectors_per_cluster(), 8);
        assert_eq!(fs.root_directory_cluster(), CL_ROOT);
        assert_eq!(fs.volume_label(), "MYVOL");
        assert!(fs.total_bytes() > 0);
    }

    #[test]
    fn list_root_returns_files_and_dirs() {
        let mut dev = build_test_image();
        let fs = Exfat::open(&mut dev).unwrap();
        let entries = fs.list_path(&mut dev, "/").unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"hello.txt"));
        assert!(names.contains(&"sub"));
        let sub = entries.iter().find(|e| e.name == "sub").unwrap();
        assert_eq!(sub.kind, crate::fs::EntryKind::Dir);
        let hello = entries.iter().find(|e| e.name == "hello.txt").unwrap();
        assert_eq!(hello.kind, crate::fs::EntryKind::Regular);
    }

    #[test]
    fn list_subdirectory() {
        let mut dev = build_test_image();
        let fs = Exfat::open(&mut dev).unwrap();
        let entries = fs.list_path(&mut dev, "/sub").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "x.bin");
    }

    #[test]
    fn case_insensitive_lookup() {
        let mut dev = build_test_image();
        let fs = Exfat::open(&mut dev).unwrap();
        // The on-disk name is "hello.txt"; query in mixed case must match.
        let entries = fs.list_path(&mut dev, "/SUB").unwrap();
        assert_eq!(entries[0].name, "x.bin");
    }

    #[test]
    fn read_file_returns_contents() {
        use std::io::Read;
        let mut dev = build_test_image();
        let fs = Exfat::open(&mut dev).unwrap();
        let mut r = fs.open_file_reader(&mut dev, "/hello.txt").unwrap();
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"Hello, exFAT!\n");
    }

    #[test]
    fn read_nested_file() {
        use std::io::Read;
        let mut dev = build_test_image();
        let fs = Exfat::open(&mut dev).unwrap();
        let mut r = fs.open_file_reader(&mut dev, "/sub/x.bin").unwrap();
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, &[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn probe_recognises_image() {
        let mut dev = build_test_image();
        assert!(probe(&mut dev).unwrap());
    }

    #[test]
    fn probe_rejects_non_exfat() {
        let mut dev = MemoryBackend::new(4096);
        assert!(!probe(&mut dev).unwrap());
    }

    // ===================================================================
    // Writer tests — format a fresh volume, create files / dirs, remove
    // them, re-open and inspect.
    // ===================================================================

    use crate::fs::exfat::format::FormatOpts;

    fn fresh_volume(label: &str) -> (MemoryBackend, Exfat) {
        // 4 MiB volume: 512 B sectors, 4 KiB clusters → ~1000 clusters.
        let mut dev = MemoryBackend::new(4 * 1024 * 1024);
        let opts = FormatOpts {
            bytes_per_sector_shift: 9,
            sectors_per_cluster_shift: 3,
            volume_serial_number: 0xCAFE_BABE,
            volume_label: label.to_string(),
        };
        let fs = Exfat::format(&mut dev, &opts).unwrap();
        (dev, fs)
    }

    #[test]
    fn format_produces_openable_volume() {
        let (mut dev, _fs) = fresh_volume("WRTEST");
        assert!(probe(&mut dev).unwrap());
        let fs2 = Exfat::open(&mut dev).unwrap();
        assert_eq!(fs2.volume_label(), "WRTEST");
        assert!(fs2.cluster_size() >= 512);
        let root = fs2.list_path(&mut dev, "/").unwrap();
        assert!(root.is_empty(), "fresh root should be empty, got {root:?}");
    }

    #[test]
    fn format_no_label_omits_volume_entry() {
        let (mut dev, _fs) = fresh_volume("");
        let fs2 = Exfat::open(&mut dev).unwrap();
        assert_eq!(fs2.volume_label(), "");
    }

    #[test]
    fn create_file_then_list() {
        let (mut dev, mut fs) = fresh_volume("CRT");
        let payload = b"hello, exfat writer!\n";
        let mut reader: &[u8] = payload;
        fs.create_file(&mut dev, "/hello.txt", &mut reader, payload.len() as u64, 0)
            .unwrap();
        fs.flush(&mut dev).unwrap();
        // Re-open and verify listing + content.
        let fs2 = Exfat::open(&mut dev).unwrap();
        let entries = fs2.list_path(&mut dev, "/").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "hello.txt");
        assert_eq!(entries[0].kind, crate::fs::EntryKind::Regular);
        use std::io::Read;
        let mut r = fs2.open_file_reader(&mut dev, "/hello.txt").unwrap();
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, payload);
    }

    #[test]
    fn create_empty_file() {
        let (mut dev, mut fs) = fresh_volume("EMPTY");
        let mut empty: &[u8] = &[];
        fs.create_file(&mut dev, "/zero.bin", &mut empty, 0, 0)
            .unwrap();
        fs.flush(&mut dev).unwrap();
        let fs2 = Exfat::open(&mut dev).unwrap();
        let entries = fs2.list_path(&mut dev, "/").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "zero.bin");
    }

    #[test]
    fn create_directory_and_nested_file() {
        let (mut dev, mut fs) = fresh_volume("DIRS");
        fs.create_dir(&mut dev, "/sub", 0).unwrap();
        let payload = b"nested";
        let mut reader: &[u8] = payload;
        fs.create_file(&mut dev, "/sub/x.bin", &mut reader, payload.len() as u64, 0)
            .unwrap();
        fs.flush(&mut dev).unwrap();

        let fs2 = Exfat::open(&mut dev).unwrap();
        let root = fs2.list_path(&mut dev, "/").unwrap();
        assert_eq!(root.len(), 1);
        assert_eq!(root[0].name, "sub");
        assert_eq!(root[0].kind, crate::fs::EntryKind::Dir);
        let sub = fs2.list_path(&mut dev, "/sub").unwrap();
        assert_eq!(sub.len(), 1);
        assert_eq!(sub[0].name, "x.bin");

        use std::io::Read;
        let mut r = fs2.open_file_reader(&mut dev, "/sub/x.bin").unwrap();
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, payload);
    }

    #[test]
    fn multi_cluster_file_streams_correctly() {
        // 4 KiB clusters; write a 20 KiB file → 5 clusters.
        let (mut dev, mut fs) = fresh_volume("BIG");
        let mut payload = Vec::with_capacity(20 * 1024);
        for i in 0..(20 * 1024) {
            payload.push((i % 251) as u8);
        }
        let mut reader: &[u8] = &payload;
        fs.create_file(&mut dev, "/big.bin", &mut reader, payload.len() as u64, 0)
            .unwrap();
        fs.flush(&mut dev).unwrap();

        let fs2 = Exfat::open(&mut dev).unwrap();
        use std::io::Read;
        let mut r = fs2.open_file_reader(&mut dev, "/big.bin").unwrap();
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).unwrap();
        assert_eq!(buf.len(), payload.len());
        assert_eq!(buf, payload);
    }

    #[test]
    fn remove_file_frees_clusters() {
        let (mut dev, mut fs) = fresh_volume("RM");
        let payload = b"to be deleted";
        let mut reader: &[u8] = payload;
        fs.create_file(
            &mut dev,
            "/doomed.txt",
            &mut reader,
            payload.len() as u64,
            0,
        )
        .unwrap();
        fs.flush(&mut dev).unwrap();
        // Capture cluster count snapshot via bitmap byte count (sanity).
        let used_before: u32 = fs.bitmap.iter().map(|b| b.count_ones()).sum();

        fs.remove(&mut dev, "/doomed.txt").unwrap();
        fs.flush(&mut dev).unwrap();
        let used_after: u32 = fs.bitmap.iter().map(|b| b.count_ones()).sum();
        assert!(
            used_after < used_before,
            "remove should free at least one cluster (before={used_before}, after={used_after})"
        );

        let fs2 = Exfat::open(&mut dev).unwrap();
        let root = fs2.list_path(&mut dev, "/").unwrap();
        assert!(
            root.is_empty(),
            "root must be empty after remove, got {root:?}"
        );
    }

    #[test]
    fn remove_empty_directory_succeeds() {
        let (mut dev, mut fs) = fresh_volume("RMD");
        fs.create_dir(&mut dev, "/empty", 0).unwrap();
        fs.flush(&mut dev).unwrap();
        fs.remove(&mut dev, "/empty").unwrap();
        fs.flush(&mut dev).unwrap();
        let fs2 = Exfat::open(&mut dev).unwrap();
        assert!(fs2.list_path(&mut dev, "/").unwrap().is_empty());
    }

    #[test]
    fn remove_non_empty_directory_fails() {
        let (mut dev, mut fs) = fresh_volume("RMNE");
        fs.create_dir(&mut dev, "/sub", 0).unwrap();
        let mut empty: &[u8] = &[];
        fs.create_file(&mut dev, "/sub/x", &mut empty, 0, 0)
            .unwrap();
        fs.flush(&mut dev).unwrap();
        let err = fs.remove(&mut dev, "/sub").unwrap_err();
        match err {
            crate::Error::InvalidArgument(msg) => assert!(msg.contains("not empty")),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn case_insensitive_lookup_on_writer() {
        let (mut dev, mut fs) = fresh_volume("CASE");
        let mut reader: &[u8] = b"x";
        fs.create_file(&mut dev, "/Hello.TXT", &mut reader, 1, 0)
            .unwrap();
        fs.flush(&mut dev).unwrap();
        let fs2 = Exfat::open(&mut dev).unwrap();
        // Different case, same file.
        use std::io::Read;
        let mut r = fs2.open_file_reader(&mut dev, "/HELLO.txt").unwrap();
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"x");
    }

    #[test]
    fn many_files_stress_directory_expansion() {
        // Cluster = 4 KiB → 128 entries; create 60 files of varying name
        // lengths (each ≥ 3 entries) so the root spills into a second
        // cluster, exercising `append_to_directory`'s chain-extend path.
        let (mut dev, mut fs) = fresh_volume("MANY");
        for i in 0..60u32 {
            let name = format!("file_{i:04}.bin");
            let mut reader: &[u8] = b"";
            fs.create_file(&mut dev, &format!("/{name}"), &mut reader, 0, 0)
                .unwrap();
        }
        fs.flush(&mut dev).unwrap();
        let fs2 = Exfat::open(&mut dev).unwrap();
        let entries = fs2.list_path(&mut dev, "/").unwrap();
        assert_eq!(entries.len(), 60);
    }

    #[test]
    fn flush_persists_fat_and_bitmap() {
        let (mut dev, mut fs) = fresh_volume("FLUSH");
        let mut reader: &[u8] = b"persistence test";
        fs.create_file(&mut dev, "/p.txt", &mut reader, 16, 0)
            .unwrap();
        // Without flush, re-open should still see writes for file content
        // (those go straight to disk), but bitmap/FAT would be stale. We
        // call flush() to commit.
        fs.flush(&mut dev).unwrap();

        // Re-open and verify everything is consistent.
        let fs2 = Exfat::open(&mut dev).unwrap();
        // Bitmap bits for the file's cluster should be set.
        let entries = fs2.list_path(&mut dev, "/").unwrap();
        assert_eq!(entries.len(), 1);
        let cluster = entries[0].inode;
        let bit = (cluster - 2) as usize;
        let byte = bit / 8;
        let mask = 1u8 << (bit % 8);
        assert!(
            fs2.bitmap[byte] & mask != 0,
            "bitmap bit for cluster {cluster} must be set"
        );
    }
}
