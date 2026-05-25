//! Path helpers shared by the archive core. Archives store flat path
//! lists; [`ArchiveIndex`](super::ArchiveIndex) synthesises the
//! directory tree from these.

/// Normalise a path to start with `/` and not end with `/` (root is
/// `/`). Collapses repeated slashes and strips `.` components, matching
/// how tar normalises entry names.
pub fn normalise_path(p: &str) -> String {
    let mut out = String::new();
    for comp in p.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        out.push('/');
        out.push_str(comp);
    }
    if out.is_empty() {
        out.push('/');
    }
    out
}

/// Split a normalised path into `(parent, leaf)`. The input must start
/// with `/` and not end with `/`.
pub fn split_path(p: &str) -> (&str, &str) {
    match p.rfind('/') {
        Some(0) => ("/", &p[1..]),
        Some(i) => (&p[..i], &p[i + 1..]),
        None => ("/", p),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_handles_edges() {
        assert_eq!(normalise_path(""), "/");
        assert_eq!(normalise_path("/"), "/");
        assert_eq!(normalise_path("a"), "/a");
        assert_eq!(normalise_path("/a/"), "/a");
        assert_eq!(normalise_path("a/b/"), "/a/b");
        assert_eq!(normalise_path("./a//b/./c"), "/a/b/c");
        assert_eq!(normalise_path("///"), "/");
    }

    #[test]
    fn split_parent_and_leaf() {
        assert_eq!(split_path("/a"), ("/", "a"));
        assert_eq!(split_path("/a/b"), ("/a", "b"));
        assert_eq!(split_path("/a/b/c.txt"), ("/a/b", "c.txt"));
    }
}
