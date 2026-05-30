//! Path-spelling styles for the CLI and interactive shell.
//!
//! Filesystem readers in this crate all speak one **canonical** path form:
//! components separated by `/`, and — for classic HFS and HFS+, whose on-disk
//! separator is `:` so a literal `/` is a legal filename character — any real
//! `/` inside a name is stored swapped to `:`. (A real `:` can never appear in
//! an HFS/HFS+ name, so that swap is a bijection.)
//!
//! Users, however, may want to spell paths the way the filesystem's own OS did.
//! [`PathStyle`] selects between:
//!
//! * [`PathStyle::Unix`] (default) — `/` separates everywhere; an HFS/HFS+
//!   name's literal `/` shows as `:` (the convention macOS itself uses when
//!   surfacing HFS names to the BSD layer). Identical to the canonical form, so
//!   translation is a no-op.
//! * [`PathStyle::Native`] — the filesystem's own separator: `:` for HFS/HFS+,
//!   `\` for FAT/exFAT/NTFS, `/` for everything else. Real filenames are shown
//!   verbatim (an HFS name keeps its literal `/`). The root is a bare leading
//!   separator (`:` or `\`).
//!
//! Translation lives here, at the CLI boundary, so the readers never need to
//! know which style the user picked: [`to_canonical`] turns user input into the
//! canonical form the readers consume, and [`display_name`]/[`display_path`]
//! turn canonical names/paths back into the chosen style for output.

use crate::inspect::FsKind;

/// How paths are spelled on the command line and in the shell.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, clap::ValueEnum)]
pub enum PathStyle {
    /// `/` separator everywhere; HFS/HFS+ literal `/` shown as `:`.
    #[default]
    Unix,
    /// The filesystem's native separator (`:` HFS, `\` FAT/NTFS, `/` else),
    /// real filenames preserved.
    Native,
}

/// The native on-disk path separator for `kind`: `:` for HFS/HFS+, `\` for the
/// DOS/Windows filesystems, `/` for everything else (and unknown kinds).
fn native_separator(kind: FsKind) -> char {
    match kind {
        FsKind::Hfs | FsKind::HfsPlus => ':',
        FsKind::Fat32 | FsKind::Exfat | FsKind::Ntfs => '\\',
        _ => '/',
    }
}

/// Whether `kind`'s canonical names carry a `/`→`:` swap (i.e. a real `/` is a
/// legal filename byte because the native separator is `:`). Strictly HFS/HFS+:
/// only there is a real `:` provably impossible in a name, which is what makes
/// the swap reversible.
fn in_name_swap(kind: FsKind) -> bool {
    matches!(kind, FsKind::Hfs | FsKind::HfsPlus)
}

/// Translate a user-supplied path (written in `style`) into the canonical form
/// the filesystem readers consume.
///
/// [`PathStyle::Unix`], and any filesystem whose native separator is already
/// `/`, pass through unchanged. For [`PathStyle::Native`] on a `:`/`\`-separated
/// filesystem, the path is re-split on the native separator and rejoined with
/// `/`; HFS/HFS+ additionally swap each component's literal `/` to `:`. A
/// leading native separator marks an absolute path; a lone separator (or `/`,
/// the CLI's default) is the root.
pub fn to_canonical(user: &str, kind: FsKind, style: PathStyle) -> String {
    if style == PathStyle::Unix {
        return user.to_string();
    }
    let sep = native_separator(kind);
    if sep == '/' {
        // Native == unix for these filesystems.
        return user.to_string();
    }
    // Accept the canonical/CLI-default root marker regardless of style.
    if user.is_empty() || user == "/" {
        return "/".to_string();
    }
    let absolute = user.starts_with(sep);
    let swap = in_name_swap(kind);
    let comps: Vec<String> = user
        .split(sep)
        .filter(|s| !s.is_empty())
        .map(|s| {
            if swap {
                s.replace('/', ":")
            } else {
                s.to_string()
            }
        })
        .collect();
    if comps.is_empty() {
        return "/".to_string();
    }
    let joined = comps.join("/");
    if absolute {
        format!("/{joined}")
    } else {
        joined
    }
}

/// Translate a single canonical leaf name into display form for `style` — used
/// for each entry of an `ls`. Only HFS/HFS+ under [`PathStyle::Native`] change:
/// the canonical `:` (a swapped-in `/`) is shown as the real `/`.
pub fn display_name(name: &str, kind: FsKind, style: PathStyle) -> String {
    if style == PathStyle::Native && in_name_swap(kind) {
        name.replace(':', "/")
    } else {
        name.to_string()
    }
}

/// Translate a full canonical path into display form for `style` — used for
/// `ls -R` directory headers and the shell prompt. [`PathStyle::Unix`] (and
/// `/`-native filesystems) pass through; native rejoins components with the
/// native separator (prefixed, so the root is a bare separator) and reverses
/// the HFS/HFS+ in-name `:`→`/` swap.
pub fn display_path(path: &str, kind: FsKind, style: PathStyle) -> String {
    if style == PathStyle::Unix {
        return path.to_string();
    }
    let sep = native_separator(kind);
    if sep == '/' {
        return path.to_string();
    }
    let swap = in_name_swap(kind);
    let comps: Vec<String> = path
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|s| {
            if swap {
                s.replace(':', "/")
            } else {
                s.to_string()
            }
        })
        .collect();
    format!("{sep}{}", comps.join(&sep.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // The motivating real-world name: a classic-HFS directory literally named
    // "A/ROSE Includes" — canonical form swaps the slash to a colon.
    const CANON: &str = "/Apps/A:ROSE Includes";
    const NATIVE_HFS: &str = ":Apps:A/ROSE Includes";

    #[test]
    fn unix_style_is_identity_for_every_kind() {
        for kind in [FsKind::Hfs, FsKind::HfsPlus, FsKind::Fat32, FsKind::Ext] {
            assert_eq!(to_canonical(CANON, kind, PathStyle::Unix), CANON);
            assert_eq!(display_path(CANON, kind, PathStyle::Unix), CANON);
            assert_eq!(
                display_name("A:ROSE Includes", kind, PathStyle::Unix),
                "A:ROSE Includes"
            );
        }
    }

    #[test]
    fn hfs_native_round_trip() {
        assert_eq!(
            to_canonical(NATIVE_HFS, FsKind::Hfs, PathStyle::Native),
            CANON
        );
        assert_eq!(
            display_path(CANON, FsKind::Hfs, PathStyle::Native),
            NATIVE_HFS
        );
        // Round-trip both directions.
        assert_eq!(
            to_canonical(
                &display_path(CANON, FsKind::Hfs, PathStyle::Native),
                FsKind::Hfs,
                PathStyle::Native
            ),
            CANON
        );
        // Leaf display un-swaps the slash.
        assert_eq!(
            display_name("A:ROSE Includes", FsKind::Hfs, PathStyle::Native),
            "A/ROSE Includes"
        );
    }

    #[test]
    fn hfs_plus_behaves_like_hfs() {
        assert_eq!(
            to_canonical(NATIVE_HFS, FsKind::HfsPlus, PathStyle::Native),
            CANON
        );
        assert_eq!(
            display_path(CANON, FsKind::HfsPlus, PathStyle::Native),
            NATIVE_HFS
        );
    }

    #[test]
    fn root_maps_to_bare_separator() {
        // CLI default "/" and the native root marker both mean root.
        assert_eq!(to_canonical("/", FsKind::Hfs, PathStyle::Native), "/");
        assert_eq!(to_canonical(":", FsKind::Hfs, PathStyle::Native), "/");
        assert_eq!(to_canonical("", FsKind::Hfs, PathStyle::Native), "/");
        assert_eq!(display_path("/", FsKind::Hfs, PathStyle::Native), ":");
        assert_eq!(display_path("/", FsKind::Fat32, PathStyle::Native), "\\");
    }

    #[test]
    fn relative_paths_stay_relative() {
        assert_eq!(
            to_canonical("System Folder", FsKind::Hfs, PathStyle::Native),
            "System Folder"
        );
        assert_eq!(
            to_canonical("Apps:Tool", FsKind::Hfs, PathStyle::Native),
            "Apps/Tool"
        );
    }

    #[test]
    fn fat_native_uses_backslash_without_in_name_swap() {
        assert_eq!(
            to_canonical("\\Windows\\System32", FsKind::Fat32, PathStyle::Native),
            "/Windows/System32"
        );
        assert_eq!(
            display_path("/Windows/System32", FsKind::Fat32, PathStyle::Native),
            "\\Windows\\System32"
        );
        // No `:`→`/` swap for FAT: a `:` in a name is left intact.
        assert_eq!(display_name("a:b", FsKind::Fat32, PathStyle::Native), "a:b");
    }

    #[test]
    fn non_hfs_unix_separator_kinds_are_identity_in_native() {
        // ext/xfs/tar already use `/`; native is a no-op.
        for kind in [FsKind::Ext, FsKind::Xfs, FsKind::Tar, FsKind::Iso9660] {
            assert_eq!(to_canonical("/a/b", kind, PathStyle::Native), "/a/b");
            assert_eq!(display_path("/a/b", kind, PathStyle::Native), "/a/b");
            assert_eq!(display_name("b", kind, PathStyle::Native), "b");
        }
    }
}
