//! Pre-defined sets of `/dev/*` device nodes for building Linux root
//! filesystem images without root privileges.
//!
//! Each set is a list of `(name, kind, major, minor, mode)` entries that a
//! filesystem implementation creates under `/dev/`. Filesystems that don't
//! support device nodes (FAT32, vfat) MUST return
//! [`Error::Unsupported`](crate::Error::Unsupported) when asked to populate
//! a non-empty set.
//!
//! Device numbers are the conventional Linux `Documentation/admin-guide/devices.txt`
//! assignments.

use crate::fs::DeviceKind;

/// Which preset to materialise under `/dev`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootDevs {
    /// Do not create `/dev` at all.
    None,
    /// Bare essentials for a Linux userspace to boot and use stdin/stdout/stderr,
    /// pseudoterminals, fuse, and the random sources: `console`, `null`,
    /// `zero`, `ptmx`, `tty`, `fuse`, `random`, `urandom`.
    Minimal,
    /// `Minimal` plus virtual consoles (tty0..15), kernel logging, raw memory
    /// access, the I/O port window, and IDE/SCSI disk + partition nodes
    /// (`hda-d` + `[a-d][1-4]`, `sda-d` + `[a-d][1-4]`). Suitable for a
    /// general-purpose root filesystem on a workstation.
    Standard,
}

/// One device-node entry to be created under `/dev`.
#[derive(Debug, Clone)]
pub struct DeviceEntry {
    pub name: String,
    pub kind: DeviceKind,
    pub major: u32,
    pub minor: u32,
    pub mode: u16,
}

/// Return the full ordered list of device-node entries for `kind`. Returns
/// an empty Vec for [`RootDevs::None`].
pub fn device_table(kind: RootDevs) -> Vec<DeviceEntry> {
    use DeviceKind::*;
    let mut out: Vec<DeviceEntry> = Vec::new();

    if matches!(kind, RootDevs::None) {
        return out;
    }

    // Minimal — common to both presets, in /dev creation order.
    for (name, dk, major, minor, mode) in [
        ("console", Char, 5, 1, 0o600u16),
        ("null", Char, 1, 3, 0o666),
        ("zero", Char, 1, 5, 0o666),
        ("ptmx", Char, 5, 2, 0o666),
        ("tty", Char, 5, 0, 0o666),
        ("fuse", Char, 10, 229, 0o666),
        ("random", Char, 1, 8, 0o666),
        ("urandom", Char, 1, 9, 0o666),
    ] {
        out.push(DeviceEntry {
            name: name.to_string(),
            kind: dk,
            major,
            minor,
            mode,
        });
    }

    if matches!(kind, RootDevs::Minimal) {
        return out;
    }

    // Standard additions on top of Minimal.

    // Virtual consoles tty0..tty15.
    for i in 0..16u32 {
        out.push(DeviceEntry {
            name: format!("tty{i}"),
            kind: Char,
            major: 4,
            minor: i,
            mode: 0o620,
        });
    }

    // Serial ports ttyS0..ttyS3 (major 4, minor 64+).
    for i in 0..4u32 {
        out.push(DeviceEntry {
            name: format!("ttyS{i}"),
            kind: Char,
            major: 4,
            minor: 64 + i,
            mode: 0o660,
        });
    }

    // Kernel logging, raw memory, I/O port window.
    for (name, major, minor, mode) in [
        ("kmsg", 1u32, 11u32, 0o644u16),
        ("mem", 1, 1, 0o640),
        ("port", 1, 4, 0o640),
    ] {
        out.push(DeviceEntry {
            name: name.to_string(),
            kind: Char,
            major,
            minor,
            mode,
        });
    }

    // IDE disks hda..hdd plus 4 partitions each.
    // hda/hdb are on major 3; hdc/hdd on major 22. Minor stride 64 per disk
    // within a major.
    let ide_disks = [
        ("hda", 3u32, 0u32),
        ("hdb", 3, 64),
        ("hdc", 22, 0),
        ("hdd", 22, 64),
    ];
    for (base, major, disk_minor) in ide_disks {
        out.push(DeviceEntry {
            name: base.to_string(),
            kind: Block,
            major,
            minor: disk_minor,
            mode: 0o660,
        });
        for p in 1..=4u32 {
            out.push(DeviceEntry {
                name: format!("{base}{p}"),
                kind: Block,
                major,
                minor: disk_minor + p,
                mode: 0o660,
            });
        }
    }

    // SCSI disks sda..sdd plus 4 partitions each. Major 8, minor stride 16.
    for (i, base) in ["sda", "sdb", "sdc", "sdd"].iter().enumerate() {
        let disk_minor = (i as u32) * 16;
        out.push(DeviceEntry {
            name: (*base).to_string(),
            kind: Block,
            major: 8,
            minor: disk_minor,
            mode: 0o660,
        });
        for p in 1..=4u32 {
            out.push(DeviceEntry {
                name: format!("{base}{p}"),
                kind: Block,
                major: 8,
                minor: disk_minor + p,
                mode: 0o660,
            });
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_has_expected_entries() {
        let t = device_table(RootDevs::Minimal);
        let names: Vec<&str> = t.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "console", "null", "zero", "ptmx", "tty", "fuse", "random", "urandom"
            ]
        );
    }

    #[test]
    fn standard_extends_minimal() {
        let m: std::collections::HashSet<_> = device_table(RootDevs::Minimal)
            .into_iter()
            .map(|e| e.name)
            .collect();
        let s: std::collections::HashSet<_> = device_table(RootDevs::Standard)
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(m.is_subset(&s));
        // Spot-checks
        assert!(s.contains("tty0"));
        assert!(s.contains("tty15"));
        assert!(!s.contains("tty16"));
        assert!(s.contains("hda"));
        assert!(s.contains("hda4"));
        assert!(!s.contains("hda5"));
        assert!(s.contains("sda"));
        assert!(s.contains("sdd4"));
        assert!(s.contains("kmsg"));
        assert!(s.contains("mem"));
        assert!(s.contains("port"));
    }

    #[test]
    fn standard_entry_count_is_predictable() {
        let s = device_table(RootDevs::Standard);
        // 8 minimal + 16 ttys + 4 ttyS + 3 (kmsg/mem/port)
        //   + 4 IDE disks × 5 (disk+4parts) + 4 SCSI disks × 5
        // = 8 + 16 + 4 + 3 + 20 + 20 = 71
        assert_eq!(s.len(), 71);
    }

    #[test]
    fn standard_has_serial_ports() {
        let s: std::collections::HashMap<String, (u32, u32)> = device_table(RootDevs::Standard)
            .into_iter()
            .map(|e| (e.name, (e.major, e.minor)))
            .collect();
        assert_eq!(s.get("ttyS0"), Some(&(4, 64)));
        assert_eq!(s.get("ttyS3"), Some(&(4, 67)));
    }

    #[test]
    fn none_is_empty() {
        assert!(device_table(RootDevs::None).is_empty());
    }
}
