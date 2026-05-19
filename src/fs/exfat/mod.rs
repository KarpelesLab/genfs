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
pub mod upcase;

use boot::BootSector;
use dir::{ENTRY_SIZE, FileEntrySet, RawSlot};
use fat::Fat;
use upcase::Upcase;

use crate::Result;
use crate::block::BlockDevice;

/// An opened exFAT volume — read-only.
pub struct Exfat {
    boot: BootSector,
    fat: Fat,
    upcase: Upcase,
    volume_label: String,
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
        };
        let root_bytes = tmp.read_chain_bytes(
            dev,
            tmp.boot.first_cluster_of_root_directory,
            /* no_fat_chain */ false,
            /* hint_byte_len */ None,
        )?;

        let (volume_label, upcase) = tmp.scan_root_metadata(dev, &root_bytes)?;
        tmp.upcase = upcase;
        tmp.volume_label = volume_label;
        Ok(tmp)
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
        let chain = self.build_data_chain(
            entry.first_cluster,
            entry.no_fat_chain(),
            entry.data_length,
        )?;
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
    fn resolve_entry(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<(FileEntrySet, u32)> {
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
    /// VolumeLabel). Returns the parsed volume label and up-case table.
    fn scan_root_metadata(
        &self,
        dev: &mut dyn BlockDevice,
        root_bytes: &[u8],
    ) -> Result<(String, Upcase)> {
        let mut volume_label = String::new();
        let mut upcase = Upcase::ascii();

        let mut i = 0;
        while i + ENTRY_SIZE <= root_bytes.len() {
            let slot: &[u8; ENTRY_SIZE] =
                (&root_bytes[i..i + ENTRY_SIZE]).try_into().unwrap();
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
                RawSlot::File {
                    secondary_count, ..
                } => {
                    // Skip the whole set — we're only metadata-mining
                    // here.
                    i += (1 + secondary_count as usize) * ENTRY_SIZE;
                }
                RawSlot::AllocationBitmap { .. } | RawSlot::Other { .. } => {
                    i += ENTRY_SIZE;
                }
            }
        }
        Ok((volume_label, upcase))
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
        let want = (buf.len() as u64)
            .min(avail_in_cluster)
            .min(self.remaining) as usize;
        let cluster = self.chain[self.cluster_idx];
        let cluster_start =
            self.cluster_heap_offset + (cluster as u64 - 2) * self.cluster_bytes;
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
            CLUSTER_HEAP_OFFSET_SECTORS as u64 * BPS as u64
                + (c as u64 - 2) * BPC as u64
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
        dev.write_at(cluster_off(CL_UPCASE), &upcase_cluster).unwrap();

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
        let sub_entries = build_dir_entries(&[(
            "x.bin",
            false,
            CL_XBIN,
            xbin_data.len() as u64,
        )]);
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
        root.extend_from_slice(&build_dir_entries(&[(
            "hello.txt",
            false,
            CL_HELLO,
            14,
        )]));

        // Directory "sub" (size = one cluster — the spec records data_length
        // as the directory's total byte length).
        root.extend_from_slice(&build_dir_entries(&[(
            "sub",
            true,
            CL_SUB,
            BPC as u64,
        )]));

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
}
