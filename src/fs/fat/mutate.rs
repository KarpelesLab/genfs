//! FAT32 modify-in-place: allocate clusters, insert/remove directory
//! entries, and prune cluster chains.
//!
//! Unlike the format-time writer in `mod.rs`, these operations work on an
//! [`Fat32`] that was opened from an existing image — they scan the FAT
//! for free clusters rather than handing them out from a sequential
//! counter, and they preserve directory-entry slack so an LFN run plus
//! its 8.3 entry land in consecutive slots as the spec requires.

use std::path::Path;

use super::{Fat32, SECTOR, dir, table};
use crate::Result;
use crate::block::BlockDevice;

/// Result of [`Fat32::find_entry`] — a directory's loaded byte buffer plus
/// the position of the named entry (and any preceding LFN run) within it.
pub(super) struct FoundEntry {
    /// The directory's cluster chain.
    pub(super) chain: Vec<u32>,
    /// Every cluster of the directory, concatenated.
    pub(super) bytes: Vec<u8>,
    /// Byte offset of the first slot of the LFN run (== entry_pos if none).
    pub(super) run_start: usize,
    /// Byte offset of the 8.3 entry itself.
    pub(super) entry_pos: usize,
    /// Decoded 8.3 entry.
    pub(super) entry: dir::DirEntry,
}

impl Fat32 {
    // -- cluster allocation on opened volumes ----------------------------

    /// Allocate `n` free clusters by scanning the FAT, link them into one
    /// chain (last → EOC), and return the chain. Starts at the FSInfo
    /// `next_free` hint and wraps to cluster 2 if it hits the end.
    pub(super) fn alloc_free_clusters(&mut self, n: u32) -> Result<Vec<u32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let clusters = self.boot.cluster_count();
        let mut found: Vec<u32> = Vec::with_capacity(n as usize);
        let mut cur = self.next_free.clamp(2, clusters + 1);
        // Scan up to two full sweeps before giving up.
        for _ in 0..(2 * clusters) {
            if cur >= clusters + 2 {
                cur = 2;
            }
            if self.fat.get(cur) == table::FREE {
                found.push(cur);
                if found.len() as u32 == n {
                    break;
                }
            }
            cur += 1;
        }
        if (found.len() as u32) < n {
            return Err(crate::Error::Unsupported(format!(
                "fat32: only {} free clusters available, need {n}",
                found.len()
            )));
        }
        for w in found.windows(2) {
            self.fat.set(w[0], w[1]);
        }
        self.fat.set(*found.last().unwrap(), table::EOC);
        // Hint the next search past the last cluster we just took.
        self.next_free = found.last().unwrap().saturating_add(1);
        Ok(found)
    }

    /// Free a whole cluster chain by setting every entry to FREE.
    pub(super) fn free_chain(&mut self, start: u32) -> Result<()> {
        if start < 2 {
            return Ok(()); // zero-length file
        }
        let chain = self.fat.chain(start)?;
        for c in chain {
            self.fat.set(c, table::FREE);
        }
        Ok(())
    }

    // -- directory-entry insertion / deletion ----------------------------

    /// Read the directory at `dir_cluster` as a flat byte buffer (every
    /// cluster in its chain concatenated).
    fn read_dir_with_chain(
        &self,
        dev: &mut dyn BlockDevice,
        dir_cluster: u32,
    ) -> Result<(Vec<u32>, Vec<u8>)> {
        let chain = self.fat.chain(dir_cluster)?;
        let cb = self.cluster_bytes() as usize;
        let mut buf = vec![0u8; chain.len() * cb];
        for (i, &c) in chain.iter().enumerate() {
            dev.read_at(self.cluster_offset(c), &mut buf[i * cb..(i + 1) * cb])?;
        }
        Ok((chain, buf))
    }

    /// Write back the byte range `[start, end)` of a directory's flat
    /// buffer, cluster-aligned.
    fn write_dir_range(
        &self,
        dev: &mut dyn BlockDevice,
        chain: &[u32],
        bytes: &[u8],
        start: usize,
        end: usize,
    ) -> Result<()> {
        let cb = self.cluster_bytes() as usize;
        let first = start / cb;
        let last = (end - 1) / cb;
        for i in first..=last {
            dev.write_at(self.cluster_offset(chain[i]), &bytes[i * cb..(i + 1) * cb])?;
        }
        Ok(())
    }

    /// Insert `new_entries` (a run of 32-byte directory entries, ending
    /// with the 8.3 entry) into the directory at `dir_cluster`. Extends
    /// the directory's cluster chain if necessary.
    fn append_dir_entries(
        &mut self,
        dev: &mut dyn BlockDevice,
        dir_cluster: u32,
        new_entries: &[u8],
    ) -> Result<()> {
        let (chain, mut bytes) = self.read_dir_with_chain(dev, dir_cluster)?;
        let cb = self.cluster_bytes() as usize;
        // Find the first end-of-directory slot (byte 0 == 0x00).
        let mut end_pos = bytes.len();
        for i in (0..bytes.len()).step_by(dir::ENTRY_SIZE) {
            if bytes[i] == 0x00 {
                end_pos = i;
                break;
            }
        }
        let need = new_entries.len();
        let avail = bytes.len() - end_pos;
        if avail >= need {
            bytes[end_pos..end_pos + need].copy_from_slice(new_entries);
            self.write_dir_range(dev, &chain, &bytes, end_pos, end_pos + need)?;
        } else {
            // Grow the chain.
            let deficit = need - avail;
            let extra_clusters = deficit.div_ceil(cb) as u32;
            let extra = self.alloc_free_clusters(extra_clusters)?;
            self.fat.set(*chain.last().unwrap(), extra[0]);
            let mut full_chain = chain.to_vec();
            full_chain.extend_from_slice(&extra);
            // Combined byte buffer: original + zero pad for new clusters.
            bytes.resize(full_chain.len() * cb, 0);
            bytes[end_pos..end_pos + need].copy_from_slice(new_entries);
            self.write_dir_range(dev, &full_chain, &bytes, end_pos, end_pos + need)?;
        }
        Ok(())
    }

    /// Locate the 8.3 entry whose long-or-short name equals `name` in the
    /// directory at `dir_cluster`. Returns the *byte position* of the 8.3
    /// entry within the directory's flat buffer, along with the position
    /// of the first slot of its preceding LFN run (== same position when
    /// there is no LFN run).
    pub(super) fn find_entry(
        &self,
        dev: &mut dyn BlockDevice,
        dir_cluster: u32,
        name: &str,
    ) -> Result<Option<FoundEntry>> {
        let (chain, bytes) = self.read_dir_with_chain(dev, dir_cluster)?;
        let mut lfn_start: Option<usize> = None;
        let mut lfn_run: Vec<dir::LfnFragment> = Vec::new();
        let mut i = 0;
        while i + dir::ENTRY_SIZE <= bytes.len() {
            let slot: &[u8; dir::ENTRY_SIZE] = (&bytes[i..i + dir::ENTRY_SIZE]).try_into().unwrap();
            match dir::classify_slot(slot) {
                dir::RawSlot::End => break,
                dir::RawSlot::Deleted => {
                    lfn_start = None;
                    lfn_run.clear();
                }
                dir::RawSlot::Lfn(frag) => {
                    if lfn_start.is_none() {
                        lfn_start = Some(i);
                    }
                    lfn_run.push(frag);
                }
                dir::RawSlot::ShortEntry(entry) => {
                    if entry.attr & dir::ATTR_VOLUME_ID != 0
                        && entry.attr & dir::ATTR_DIRECTORY == 0
                    {
                        lfn_start = None;
                        lfn_run.clear();
                        i += dir::ENTRY_SIZE;
                        continue;
                    }
                    let short = entry.short_name_string();
                    let long = dir::assemble_lfn(&lfn_run, &entry.name_83);
                    let matches_short = short.eq_ignore_ascii_case(name);
                    let matches_long = long
                        .as_deref()
                        .map(|l| l.eq_ignore_ascii_case(name))
                        .unwrap_or(false);
                    if matches_short || matches_long {
                        let run_start = lfn_start.unwrap_or(i);
                        return Ok(Some(FoundEntry {
                            chain,
                            bytes,
                            run_start,
                            entry_pos: i,
                            entry,
                        }));
                    }
                    lfn_start = None;
                    lfn_run.clear();
                }
            }
            i += dir::ENTRY_SIZE;
        }
        Ok(None)
    }

    /// Mark every slot in `[start_pos, end_pos_inclusive + ENTRY_SIZE)`
    /// as deleted (first byte 0xE5).
    fn mark_entries_deleted(
        &self,
        dev: &mut dyn BlockDevice,
        chain: &[u32],
        bytes: &mut [u8],
        start_pos: usize,
        end_pos: usize,
    ) -> Result<()> {
        let mut i = start_pos;
        while i <= end_pos {
            bytes[i] = 0xE5;
            i += dir::ENTRY_SIZE;
        }
        self.write_dir_range(dev, chain, bytes, start_pos, end_pos + dir::ENTRY_SIZE)?;
        Ok(())
    }

    // -- public modify-in-place API --------------------------------------

    /// Add a regular file at `dest_path` populated from a host file. The
    /// parent directory must already exist; an existing entry at the same
    /// destination is an error.
    pub fn add_file(
        &mut self,
        dev: &mut dyn BlockDevice,
        dest_path: &str,
        host_src: &Path,
    ) -> Result<()> {
        let size = std::fs::metadata(host_src)?.len();
        let mut file = std::fs::File::open(host_src)?;
        self.add_file_from_reader(dev, dest_path, &mut file, size)
    }

    /// Like [`Self::add_file`] but pulls bytes from any [`std::io::Read`]
    /// instead of a host filesystem path. The reader must produce exactly
    /// `size` bytes; used by the repack path to stream content directly
    /// from another opened filesystem.
    pub fn add_file_from_reader(
        &mut self,
        dev: &mut dyn BlockDevice,
        dest_path: &str,
        reader: &mut dyn std::io::Read,
        size: u64,
    ) -> Result<()> {
        let (parent_cluster, leaf) = self.resolve_parent(dev, dest_path)?;
        if self.find_entry(dev, parent_cluster, &leaf)?.is_some() {
            return Err(crate::Error::InvalidArgument(format!(
                "fat32: {dest_path:?} already exists"
            )));
        }
        let cb = self.cluster_bytes();
        let chain = if size == 0 {
            Vec::new()
        } else {
            let n = size.div_ceil(cb) as u32;
            self.alloc_free_clusters(n)?
        };
        self.stream_reader_chain(dev, reader, &chain, size)?;
        let first = chain.first().copied().unwrap_or(0);
        let mut entries: Vec<u8> = Vec::new();
        self.push_dir_entry(&mut entries, &leaf, dir::ATTR_ARCHIVE, first, size as u32);
        self.append_dir_entries(dev, parent_cluster, &entries)?;
        Ok(())
    }

    fn stream_reader_chain(
        &self,
        dev: &mut dyn BlockDevice,
        reader: &mut dyn std::io::Read,
        chain: &[u32],
        size: u64,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        let cb = self.cluster_bytes() as usize;
        let mut buf = vec![0u8; cb];
        let mut remaining = size;
        for &c in chain {
            let want = remaining.min(cb as u64) as usize;
            buf[..want].fill(0);
            reader.read_exact(&mut buf[..want])?;
            dev.write_at(self.cluster_offset(c), &buf[..want])?;
            remaining -= want as u64;
            if remaining == 0 {
                break;
            }
        }
        Ok(())
    }

    /// Create a directory at `dest_path`. The parent must already exist.
    pub fn add_dir(&mut self, dev: &mut dyn BlockDevice, dest_path: &str) -> Result<()> {
        let (parent_cluster, leaf) = self.resolve_parent(dev, dest_path)?;
        if self.find_entry(dev, parent_cluster, &leaf)?.is_some() {
            return Err(crate::Error::InvalidArgument(format!(
                "fat32: {dest_path:?} already exists"
            )));
        }
        let chain = self.alloc_free_clusters(1)?;
        let child_cluster = chain[0];
        // Initialise the new directory with "." and ".." entries.
        let mut dir_body: Vec<u8> = Vec::with_capacity(2 * dir::ENTRY_SIZE);
        dir_body.extend_from_slice(&dot_entry(b".          ", child_cluster));
        let parent_in_dotdot = if parent_cluster == self.boot.root_cluster {
            0
        } else {
            parent_cluster
        };
        dir_body.extend_from_slice(&dot_entry(b"..         ", parent_in_dotdot));
        let cb = self.cluster_bytes() as usize;
        dir_body.resize(cb, 0);
        dev.write_at(self.cluster_offset(child_cluster), &dir_body)?;

        let mut entries: Vec<u8> = Vec::new();
        self.push_dir_entry(&mut entries, &leaf, dir::ATTR_DIRECTORY, child_cluster, 0);
        self.append_dir_entries(dev, parent_cluster, &entries)?;
        Ok(())
    }

    /// Remove a file, or an empty directory, at `path`. Returns
    /// [`crate::Error::InvalidArgument`] for a non-empty directory.
    pub fn remove(&mut self, dev: &mut dyn BlockDevice, path: &str) -> Result<()> {
        let (parent_cluster, leaf) = self.resolve_parent(dev, path)?;
        let Some(FoundEntry {
            chain,
            mut bytes,
            run_start,
            entry_pos,
            entry,
        }) = self.find_entry(dev, parent_cluster, &leaf)?
        else {
            return Err(crate::Error::InvalidArgument(format!(
                "fat32: {path:?} not found"
            )));
        };
        // Disallow removing "." / "..".
        let short = entry.short_name_string();
        if short == "." || short == ".." {
            return Err(crate::Error::InvalidArgument(
                "fat32: cannot remove . or ..".into(),
            ));
        }
        if entry.attr & dir::ATTR_DIRECTORY != 0 {
            // Empty-dir check: only "." and ".." entries beyond which all
            // slots are free or end-of-dir.
            if entry.first_cluster < 2 {
                return Err(crate::Error::InvalidImage(
                    "fat32: directory entry has no cluster".into(),
                ));
            }
            let listed = self.list_path_by_cluster(dev, entry.first_cluster)?;
            if !listed.is_empty() {
                return Err(crate::Error::InvalidArgument(format!(
                    "fat32: directory {path:?} is not empty"
                )));
            }
        }
        if entry.first_cluster >= 2 {
            self.free_chain(entry.first_cluster)?;
        }
        self.mark_entries_deleted(dev, &chain, &mut bytes, run_start, entry_pos)?;
        Ok(())
    }

    /// Split `path` into (parent_dir_cluster, leaf_name). Errors if the
    /// path is `/` or if the parent doesn't exist.
    pub(super) fn resolve_parent(
        &self,
        dev: &mut dyn BlockDevice,
        path: &str,
    ) -> Result<(u32, String)> {
        let parts: Vec<&str> = path
            .split(['/', '\\'])
            .filter(|p| !p.is_empty() && *p != ".")
            .collect();
        let (last, prefix) = parts.split_last().ok_or_else(|| {
            crate::Error::InvalidArgument("fat32: cannot operate on root \"/\"".into())
        })?;
        let mut cluster = self.boot.root_cluster;
        for part in prefix {
            let entry = match self.find_entry(dev, cluster, part)? {
                Some(found) => found.entry,
                None => {
                    return Err(crate::Error::InvalidArgument(format!(
                        "fat32: parent {part:?} of {path:?} does not exist"
                    )));
                }
            };
            if entry.attr & dir::ATTR_DIRECTORY == 0 {
                return Err(crate::Error::InvalidArgument(format!(
                    "fat32: {part:?} is not a directory"
                )));
            }
            cluster = if entry.first_cluster == 0 {
                self.boot.root_cluster
            } else {
                entry.first_cluster
            };
        }
        Ok((cluster, (*last).to_string()))
    }

    /// List a directory by its starting cluster, returning generic entries.
    /// Wraps [`Fat32::list_cluster`](super::Fat32) so [`remove`] can run an
    /// empty-dir check without going through the path resolver.
    fn list_path_by_cluster(
        &self,
        dev: &mut dyn BlockDevice,
        dir_cluster: u32,
    ) -> Result<Vec<crate::fs::DirEntry>> {
        self.list_cluster(dev, dir_cluster)
    }

    /// Append a "long-name-aware" directory entry run (LFN fragments + 8.3
    /// entry) for `name` into `entries`. Wraps the existing private helper
    /// without the short-name uniqueness counter — modify-in-place callers
    /// pick a fresh seq via the cluster number, which is itself unique.
    fn push_dir_entry(
        &self,
        entries: &mut Vec<u8>,
        name: &str,
        attr: u8,
        first_cluster: u32,
        file_size: u32,
    ) {
        let upper = name.to_ascii_uppercase();
        let (name_83, need_lfn) = if dir::is_valid_83(&upper) {
            (dir::pack_83(&upper), upper != name)
        } else {
            (dir::generate_83(name, first_cluster), true)
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
}

fn dot_entry(name_83: &[u8; 11], cluster: u32) -> [u8; dir::ENTRY_SIZE] {
    dir::DirEntry {
        name_83: *name_83,
        attr: dir::ATTR_DIRECTORY,
        first_cluster: cluster,
        file_size: 0,
    }
    .encode()
}

// `cluster_offset` and `cluster_bytes` are on the parent impl; export them
// here via re-import so the helpers above can call them through `self`.
#[allow(dead_code)]
const _: u32 = SECTOR; // keep the import used
