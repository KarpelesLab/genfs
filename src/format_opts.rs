//! Filesystem-format options bag.
//!
//! Every backend's `FormatOpts` struct has its own shape — `ext` cares
//! about block size and journal length, `fat32` about volume id and
//! cluster shift, `squashfs` about compression. The CLI and the TOML
//! spec layer both need a way to ferry those settings through a single
//! type-erased channel:
//!
//! - CLI: `fstool create --type ext4 -O block_size=4096,sparse=true ./src out.img`
//! - TOML: `[filesystem.options]` table with the same key names
//!
//! [`OptionMap`] is that channel. The caller (a `parse_format_opts`
//! function on each FS) consumes recognised keys with [`take_u32`] /
//! [`take_bool`] / [`take_str`] etc., then calls
//! [`OptionMap::check_empty`] to surface a clear error if the user
//! passed a key the backend doesn't recognise.
//!
//! [`take_u32`]: OptionMap::take_u32
//! [`take_bool`]: OptionMap::take_bool
//! [`take_str`]: OptionMap::take_str

use std::collections::BTreeMap;

use crate::{Error, Result};

/// A type-erased bag of `key=value` strings consumed by a backend's
/// format-options parser. Built from CLI `-O` flags, a TOML
/// `[filesystem.options]` table, or both at once.
///
/// Keys are case-sensitive; values are normalised to strings (booleans
/// arrive as `"true"`/`"false"`, ints as their decimal repr).
///
/// Once a backend has finished consuming the keys it recognises, call
/// [`Self::check_empty`] to reject leftover keys with a helpful error
/// citing the FS type and the unknown key names.
#[derive(Debug, Default, Clone)]
pub struct OptionMap {
    map: BTreeMap<String, String>,
}

impl OptionMap {
    /// Empty bag.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a single CLI fragment of the form `key=value[,key=value]…`.
    /// Multiple fragments (from `-O` repeated) can be merged with
    /// [`Self::merge_cli`].
    pub fn from_cli(s: &str) -> Result<Self> {
        let mut out = Self::new();
        out.merge_cli(s)?;
        Ok(out)
    }

    /// Parse and merge a CLI fragment into this map. Later values
    /// shadow earlier ones (so `-O block_size=1024 -O block_size=4096`
    /// keeps 4096).
    pub fn merge_cli(&mut self, s: &str) -> Result<()> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(());
        }
        for piece in s.split(',') {
            let piece = piece.trim();
            if piece.is_empty() {
                continue;
            }
            let (k, v) = piece.split_once('=').ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "options: expected `key=value`, got {piece:?} \
                     (use `-O key=val,key=val` or repeat `-O`)"
                ))
            })?;
            let k = k.trim();
            if k.is_empty() {
                return Err(Error::InvalidArgument(
                    "options: empty key in `=value`".into(),
                ));
            }
            self.map.insert(k.to_string(), v.trim().to_string());
        }
        Ok(())
    }

    /// Merge values from a TOML table. Each entry's value is rendered
    /// to its conventional string form: booleans → `"true"`/`"false"`,
    /// integers / floats → decimal, strings → their raw text. Nested
    /// arrays / tables aren't accepted — pass scalar values only.
    pub fn merge_toml(&mut self, table: &toml::Table) -> Result<()> {
        for (k, v) in table.iter() {
            let s = match v {
                toml::Value::String(s) => s.clone(),
                toml::Value::Integer(i) => i.to_string(),
                toml::Value::Float(f) => f.to_string(),
                toml::Value::Boolean(b) => b.to_string(),
                _ => {
                    return Err(Error::InvalidArgument(format!(
                        "spec: option {k:?} must be a scalar (string / int / float / bool), \
                         not {kind}",
                        kind = v.type_str()
                    )));
                }
            };
            self.map.insert(k.clone(), s);
        }
        Ok(())
    }

    /// Insert a single value. Used by the spec layer to pre-load
    /// legacy flat TOML fields (`block_size`, `journal_blocks`, …)
    /// into the map before applying the explicit `options` table.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.map.insert(key.into(), value.into());
    }

    /// Whether the map currently has no keys.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Remove and return the raw string for `key`, if present.
    pub fn take_str(&mut self, key: &str) -> Option<String> {
        self.map.remove(key)
    }

    /// Remove and parse a `u32` for `key`. Returns `Ok(None)` if
    /// absent. Decimal, hex (`0x…`), or binary (`0b…`) accepted.
    pub fn take_u32(&mut self, key: &str) -> Result<Option<u32>> {
        match self.map.remove(key) {
            None => Ok(None),
            Some(v) => parse_integer::<u32>(key, &v).map(Some),
        }
    }

    /// Remove and parse a `u64` for `key`. Returns `Ok(None)` if
    /// absent. Decimal, hex, or binary accepted.
    pub fn take_u64(&mut self, key: &str) -> Result<Option<u64>> {
        match self.map.remove(key) {
            None => Ok(None),
            Some(v) => parse_integer::<u64>(key, &v).map(Some),
        }
    }

    /// Remove and parse a `u16` for `key`.
    pub fn take_u16(&mut self, key: &str) -> Result<Option<u16>> {
        match self.map.remove(key) {
            None => Ok(None),
            Some(v) => parse_integer::<u16>(key, &v).map(Some),
        }
    }

    /// Remove and parse a `u8` for `key`.
    pub fn take_u8(&mut self, key: &str) -> Result<Option<u8>> {
        match self.map.remove(key) {
            None => Ok(None),
            Some(v) => parse_integer::<u8>(key, &v).map(Some),
        }
    }

    /// Remove and parse a boolean for `key`. Accepts `true`/`false`,
    /// `yes`/`no`, `on`/`off`, `1`/`0` — case-insensitive.
    pub fn take_bool(&mut self, key: &str) -> Result<Option<bool>> {
        let Some(v) = self.map.remove(key) else {
            return Ok(None);
        };
        match v.to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" | "1" => Ok(Some(true)),
            "false" | "no" | "off" | "0" => Ok(Some(false)),
            _ => Err(Error::InvalidArgument(format!(
                "options: {key}={v:?} is not a boolean (try true/false/yes/no/on/off/0/1)"
            ))),
        }
    }

    /// Remove and parse a human-friendly size for `key`. Accepts the
    /// same `64MiB` / `1GiB` / bare-byte forms as
    /// [`crate::spec::parse_size`].
    pub fn take_size(&mut self, key: &str) -> Result<Option<u64>> {
        match self.map.remove(key) {
            None => Ok(None),
            Some(v) => crate::spec::parse_size(&v).map(Some).map_err(|e| {
                Error::InvalidArgument(format!("options: {key}={v:?} is not a valid size: {e}"))
            }),
        }
    }

    /// Remove and copy a UTF-8 string for `key` into a fixed-size
    /// volume-label byte array, padding with the given byte. Returns
    /// `Err` if the string is too long to fit in `N` bytes.
    pub fn take_label<const N: usize>(
        &mut self,
        key: &str,
        pad: u8,
    ) -> Result<Option<[u8; N]>> {
        let Some(v) = self.map.remove(key) else {
            return Ok(None);
        };
        let bytes = v.as_bytes();
        if bytes.len() > N {
            return Err(Error::InvalidArgument(format!(
                "options: {key}={v:?} is {len} bytes but the maximum is {N}",
                len = bytes.len()
            )));
        }
        let mut out = [pad; N];
        out[..bytes.len()].copy_from_slice(bytes);
        Ok(Some(out))
    }

    /// Reject any remaining keys with an error that names the FS type
    /// and lists the unknown keys. Call this after the backend has
    /// taken everything it recognises.
    pub fn check_empty(self, fs_type: &str) -> Result<()> {
        if self.map.is_empty() {
            return Ok(());
        }
        let names: Vec<&str> = self.map.keys().map(String::as_str).collect();
        Err(Error::InvalidArgument(format!(
            "{fs_type}: unrecognised option(s): {}",
            names.join(", ")
        )))
    }
}

fn parse_integer<T>(key: &str, raw: &str) -> Result<T>
where
    T: TryFrom<u64> + std::str::FromStr,
{
    // Try a base prefix first (0x… / 0b…); fall back to decimal.
    let trimmed = raw.trim();
    let parsed: u64 = if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).map_err(|e| {
            Error::InvalidArgument(format!(
                "options: {key}={raw:?} is not a valid hex integer ({e})"
            ))
        })?
    } else if let Some(bin) = trimmed
        .strip_prefix("0b")
        .or_else(|| trimmed.strip_prefix("0B"))
    {
        u64::from_str_radix(bin, 2).map_err(|e| {
            Error::InvalidArgument(format!(
                "options: {key}={raw:?} is not a valid binary integer ({e})"
            ))
        })?
    } else {
        trimmed.parse::<u64>().map_err(|e| {
            Error::InvalidArgument(format!(
                "options: {key}={raw:?} is not a valid integer ({e})"
            ))
        })?
    };
    T::try_from(parsed).map_err(|_| {
        Error::InvalidArgument(format!(
            "options: {key}={raw:?} doesn't fit in a {}",
            std::any::type_name::<T>()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_comma_pairs() {
        let mut m = OptionMap::from_cli("block_size=4096,sparse=true").unwrap();
        assert_eq!(m.take_u32("block_size").unwrap(), Some(4096));
        assert_eq!(m.take_bool("sparse").unwrap(), Some(true));
        m.check_empty("ext4").unwrap();
    }

    #[test]
    fn cli_rejects_bareword() {
        let err = OptionMap::from_cli("block_size4096").unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn cli_supports_hex_and_bool_synonyms() {
        let mut m =
            OptionMap::from_cli("volume_id=0xCAFEBABE,journaled=yes,trim=off").unwrap();
        assert_eq!(m.take_u32("volume_id").unwrap(), Some(0xCAFE_BABE));
        assert_eq!(m.take_bool("journaled").unwrap(), Some(true));
        assert_eq!(m.take_bool("trim").unwrap(), Some(false));
    }

    #[test]
    fn check_empty_reports_unknown_keys() {
        let mut m = OptionMap::from_cli("block_size=1024,journal_blocks=512").unwrap();
        let _ = m.take_u32("block_size").unwrap();
        let err = m.check_empty("ext2").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("ext2"), "msg: {msg}");
        assert!(msg.contains("journal_blocks"), "msg: {msg}");
    }

    #[test]
    fn toml_table_round_trips() {
        let table: toml::Table = toml::from_str(
            r#"
            block_size = 4096
            sparse = true
            volume_label = "rootfs"
            "#,
        )
        .unwrap();
        let mut m = OptionMap::new();
        m.merge_toml(&table).unwrap();
        assert_eq!(m.take_u32("block_size").unwrap(), Some(4096));
        assert_eq!(m.take_bool("sparse").unwrap(), Some(true));
        assert_eq!(m.take_str("volume_label"), Some("rootfs".into()));
    }

    #[test]
    fn label_pads_to_fixed_width() {
        let mut m = OptionMap::from_cli("volume_label=ROOT").unwrap();
        let label = m.take_label::<11>("volume_label", b' ').unwrap().unwrap();
        assert_eq!(&label, b"ROOT       ");
    }

    #[test]
    fn label_rejects_overlong_string() {
        let mut m = OptionMap::from_cli("volume_label=THISIS12LONGER").unwrap();
        let err = m.take_label::<11>("volume_label", b' ').unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[test]
    fn size_parses_units() {
        let mut m = OptionMap::from_cli("journal_size=4MiB").unwrap();
        assert_eq!(m.take_size("journal_size").unwrap(), Some(4 * 1024 * 1024));
    }
}
