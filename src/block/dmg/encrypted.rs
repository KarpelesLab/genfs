//! Encrypted DMG (`encrcdsa` v2) read-only backend.
//!
//! ## Status
//!
//! - Detection: probe for the 8-byte magic `b"encrcdsa"` at offset 0.
//! - Header parse: full v2 layout — PBKDF2 parameters, 3DES-wrapped
//!   keyblob, chunk layout. Always available, regardless of the
//!   `dmg-encrypted` feature.
//! - Decryption (`dmg-encrypted` feature): PBKDF2-SHA1 → 3DES-CBC unwrap
//!   of the keyblob → per-chunk AES-CBC decryption with chunk-indexed
//!   HMAC-SHA1 IVs.
//!
//! ## Format recap (v2)
//!
//! Apple's encrypted disk images carry an `encrcdsa` v2 header at offset
//! 0. The data fork that follows is split into fixed-size *chunks*
//! (`block_size` bytes each, typically 4096); each chunk is encrypted
//! independently in AES-CBC with a per-chunk IV derived from the
//! chunk index plus an HMAC-SHA1 key. A separate *chunk encryption key*
//! (CEK) and IV-derivation key are wrapped under a 3DES key derived
//! from the user passphrase via PBKDF2-SHA1.
//!
//! All multi-byte fields are big-endian on disk.
//!
//! ```text
//!   0x00  8 bytes  magic  "encrcdsa"
//!   0x08  u32 BE   version  (= 2)
//!   0x0C  u32 BE   enc_iv_size (= 32; only 16 used for AES)
//!   0x10  u32 BE   encryption_mode (0 = AES-128, 1 = AES-256)
//!   0x14  u32 BE   encryption_algorithm (1 = AES_CBC)
//!   0x18  u32 BE   pbkdf2_prng_algorithm
//!   0x1C  u32 BE   pbkdf2_iteration_count
//!   0x20  u32 BE   pbkdf2_salt_length
//!   0x24  32 bytes salt buffer (first salt_length bytes are live)
//!   0x44  u32 BE   blob_enc_iv_size
//!   0x48  32 bytes IV buffer (first IV-size bytes are live)
//!   0x68  u32 BE   blob_enc_key_bits (= 192 for 3DES_EDE3)
//!   0x6C  u32 BE   blob_enc_algorithm (3 = 3DES_EDE3_CBC)
//!   0x70  u32 BE   blob_enc_padding
//!   0x74  u32 BE   blob_enc_mode
//!   0x78  u32 BE   encrypted_keyblob_size
//!   0x7C  ~48 B    encrypted_keyblob (3DES-CBC; PKCS#7 padded)
//!   ...
//!   0xBC  u32 BE   block_size
//!   0xC0  u64 BE   n_chunks
//!   0xC8  u64 BE   data_offset
//!   0xD0  u64 BE   data_size
//! ```
//!
//! After PBKDF2-SHA1 derives a 24-byte KEK from `(password, salt,
//! iter_count)`, the keyblob is decrypted with 3DES_EDE3_CBC using
//! `(KEK, IV)`. PKCS#7 padding is removed; the resulting plaintext is
//! the concatenation of the AES key (16 or 32 bytes) and the HMAC-SHA1
//! key (20 bytes). The chunk IV is the first 16 bytes of
//! `HMAC-SHA1(hmac_key, chunk_index_as_u32_be)`.
//!
//! References (public reverse-engineering write-ups):
//!
//! - Jonathan Levin, *DMG file structure* (newosxbook.com).
//! - Public PKCS#5 / RFC 2898 (PBKDF2).
//! - Apple CDSA / CSSM algorithm-identifier documentation
//!   (PKCS5_PBKDF2 = 0x67, 3DES_3KEY_EDE = 0x11).
//!
//! No Apple source / SDK and no GPL-licensed reference implementation
//! was consulted while writing this module.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[cfg(feature = "dmg-encrypted")]
use std::io::{self, Write};

use crate::Result;
#[cfg(feature = "dmg-encrypted")]
use crate::block::BlockDevice;

/// Eight-byte v2 magic at file offset 0.
pub const ENCRCDSA_MAGIC: &[u8; 8] = b"encrcdsa";

/// Total size of the fixed-layout header up to and including
/// `data_size`. The encrypted keyblob lives inside this range starting
/// at offset 0x7C. We capture this as a constant so the reader can
/// validate the file is at least this big before chasing offsets.
pub const ENCRCDSA_V2_HEADER_MIN_BYTES: usize = 0xD8;

/// Cheap detector — peeks at the first 8 bytes of `path` and returns
/// `Ok(true)` when they match [`ENCRCDSA_MAGIC`]. Any I/O failure or
/// short read returns `Ok(false)` so callers can fall through to other
/// backends.
pub fn probe(path: &Path) -> Result<bool> {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };
    let mut head = [0u8; 8];
    if f.read_exact(&mut head).is_err() {
        return Ok(false);
    }
    Ok(&head == ENCRCDSA_MAGIC)
}

/// Decoded fixed-layout header for an `encrcdsa` v2 image.
///
/// All sized buffers are kept as raw 32-byte (or whatever-the-field-says)
/// arrays plus a "live length" so the reader can pass exactly the bytes
/// that matter to PBKDF2 / 3DES while still letting a curious caller
/// inspect the trailing zeros.
#[derive(Debug, Clone)]
pub struct EncryptedDmgHeader {
    /// Format version — must be 2.
    pub version: u32,
    /// IV-buffer size, in bytes. Per the spec this is 32; only the
    /// first 16 bytes are used for AES-CBC.
    pub enc_iv_size: u32,
    /// Encryption mode: 0 = AES-128, 1 = AES-256.
    pub encryption_mode: u32,
    /// Encryption algorithm: 1 = AES_CBC.
    pub encryption_algorithm: u32,
    /// PRNG used inside PBKDF2 — Apple's keystore only ever picks
    /// HMAC-SHA1 in shipped images; we accept any value and let the
    /// decryption path assume SHA-1.
    pub pbkdf2_prng_algorithm: u32,
    /// Number of PBKDF2 iterations. Typically 1 000–250 000.
    pub pbkdf2_iteration_count: u32,
    /// Number of live bytes in `pbkdf2_salt`.
    pub pbkdf2_salt_length: u32,
    /// Salt buffer (32 bytes on disk; first `pbkdf2_salt_length` are live).
    pub pbkdf2_salt: [u8; 32],
    /// Number of live bytes in `blob_enc_iv`.
    pub blob_enc_iv_size: u32,
    /// IV buffer used to 3DES-decrypt the keyblob (32 bytes on disk;
    /// first `blob_enc_iv_size` are live).
    pub blob_enc_iv: [u8; 32],
    /// Bit-length of the KEK; 192 for 3DES_EDE3 (24 bytes).
    pub blob_enc_key_bits: u32,
    /// Blob-wrap algorithm: 3 = 3DES_EDE3_CBC.
    pub blob_enc_algorithm: u32,
    /// CSSM padding mode for the keyblob. PKCS#7 in shipped images.
    pub blob_enc_padding: u32,
    /// CSSM block-mode parameter; we don't act on it.
    pub blob_enc_mode: u32,
    /// Live length of the encrypted keyblob (≥ 48).
    pub encrypted_keyblob_size: u32,
    /// Encrypted keyblob bytes (heap-allocated; size = `encrypted_keyblob_size`).
    pub encrypted_keyblob: Vec<u8>,
    /// Chunk size, in bytes. The data fork is split into `n_chunks`
    /// non-overlapping chunks of this size, each independently AES-CBC
    /// encrypted.
    pub block_size: u32,
    /// Number of chunks in the data fork.
    pub n_chunks: u64,
    /// Absolute file offset of the first chunk's ciphertext.
    pub data_offset: u64,
    /// Length of the encrypted data fork in bytes. Equal to
    /// `n_chunks * block_size` for v2 images.
    pub data_size: u64,
}

impl EncryptedDmgHeader {
    /// Decode an `encrcdsa` v2 header from `buf`.
    ///
    /// `buf` must start at file offset 0 and contain at least the full
    /// fixed-layout region plus the encrypted keyblob.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < ENCRCDSA_V2_HEADER_MIN_BYTES {
            return Err(crate::Error::InvalidImage(format!(
                "encrcdsa: header slice shorter than {ENCRCDSA_V2_HEADER_MIN_BYTES} bytes"
            )));
        }
        if &buf[0..8] != ENCRCDSA_MAGIC {
            return Err(crate::Error::InvalidImage(
                "encrcdsa: magic mismatch (expected \"encrcdsa\")".into(),
            ));
        }
        let version = u32::from_be_bytes(buf[0x08..0x0C].try_into().unwrap());
        if version != 2 {
            return Err(crate::Error::Unsupported(format!(
                "encrcdsa: version {version} not supported (only v2)"
            )));
        }

        let enc_iv_size = u32::from_be_bytes(buf[0x0C..0x10].try_into().unwrap());
        let encryption_mode = u32::from_be_bytes(buf[0x10..0x14].try_into().unwrap());
        let encryption_algorithm = u32::from_be_bytes(buf[0x14..0x18].try_into().unwrap());
        let pbkdf2_prng_algorithm = u32::from_be_bytes(buf[0x18..0x1C].try_into().unwrap());
        let pbkdf2_iteration_count = u32::from_be_bytes(buf[0x1C..0x20].try_into().unwrap());
        let pbkdf2_salt_length = u32::from_be_bytes(buf[0x20..0x24].try_into().unwrap());
        let mut pbkdf2_salt = [0u8; 32];
        pbkdf2_salt.copy_from_slice(&buf[0x24..0x44]);
        let blob_enc_iv_size = u32::from_be_bytes(buf[0x44..0x48].try_into().unwrap());
        let mut blob_enc_iv = [0u8; 32];
        blob_enc_iv.copy_from_slice(&buf[0x48..0x68]);
        let blob_enc_key_bits = u32::from_be_bytes(buf[0x68..0x6C].try_into().unwrap());
        let blob_enc_algorithm = u32::from_be_bytes(buf[0x6C..0x70].try_into().unwrap());
        let blob_enc_padding = u32::from_be_bytes(buf[0x70..0x74].try_into().unwrap());
        let blob_enc_mode = u32::from_be_bytes(buf[0x74..0x78].try_into().unwrap());
        let encrypted_keyblob_size = u32::from_be_bytes(buf[0x78..0x7C].try_into().unwrap());

        // Bounds-check the keyblob length against the buffer we have.
        let blob_start = 0x7C;
        let blob_end = blob_start + encrypted_keyblob_size as usize;
        if blob_end > buf.len() {
            return Err(crate::Error::InvalidImage(format!(
                "encrcdsa: keyblob ({encrypted_keyblob_size} bytes) overruns provided header buffer"
            )));
        }
        let encrypted_keyblob = buf[blob_start..blob_end].to_vec();

        // Chunk-layout fields live at a fixed offset, regardless of how
        // big the keyblob was — the keyblob slot is sized for the
        // maximum (64 bytes) and zero-padded.
        let block_size = u32::from_be_bytes(buf[0xBC..0xC0].try_into().unwrap());
        let n_chunks = u64::from_be_bytes(buf[0xC0..0xC8].try_into().unwrap());
        let data_offset = u64::from_be_bytes(buf[0xC8..0xD0].try_into().unwrap());
        let data_size = u64::from_be_bytes(buf[0xD0..0xD8].try_into().unwrap());

        Ok(Self {
            version,
            enc_iv_size,
            encryption_mode,
            encryption_algorithm,
            pbkdf2_prng_algorithm,
            pbkdf2_iteration_count,
            pbkdf2_salt_length,
            pbkdf2_salt,
            blob_enc_iv_size,
            blob_enc_iv,
            blob_enc_key_bits,
            blob_enc_algorithm,
            blob_enc_padding,
            blob_enc_mode,
            encrypted_keyblob_size,
            encrypted_keyblob,
            block_size,
            n_chunks,
            data_offset,
            data_size,
        })
    }

    /// Convenience accessor for the live salt slice.
    pub fn salt(&self) -> &[u8] {
        &self.pbkdf2_salt[..self.pbkdf2_salt_length as usize]
    }

    /// Convenience accessor for the live blob-IV slice.
    pub fn blob_iv(&self) -> &[u8] {
        &self.blob_enc_iv[..self.blob_enc_iv_size as usize]
    }

    /// AES key length in bytes, derived from `encryption_mode`.
    /// Returns `Err(Unsupported)` for modes other than 0 (AES-128) and
    /// 1 (AES-256).
    pub fn aes_key_len(&self) -> Result<usize> {
        match self.encryption_mode {
            0 => Ok(16),
            1 => Ok(32),
            other => Err(crate::Error::Unsupported(format!(
                "encrcdsa: unknown encryption_mode {other} (expected 0 = AES-128, 1 = AES-256)"
            ))),
        }
    }
}

/// Read at least the fixed-layout header off `file`, decode it, and
/// return the parsed [`EncryptedDmgHeader`]. The file cursor is left
/// at an unspecified position; callers should seek explicitly before
/// the next read.
pub fn read_header(file: &mut File) -> Result<EncryptedDmgHeader> {
    // The fixed layout up through `data_size` is 0xD8 bytes. The
    // keyblob slot is up to 64 bytes (largest published value), and
    // chunk-layout fields *follow* the keyblob slot — so a header read
    // of 0xD8 bytes is always enough to decode both regions.
    let mut buf = vec![0u8; ENCRCDSA_V2_HEADER_MIN_BYTES];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut buf)?;
    EncryptedDmgHeader::decode(&buf)
}

/// Read-only backend for password-protected DMGs (`encrcdsa` v2). Open
/// with [`EncryptedDmgBackend::open_with_password`].
///
/// The decrypted plaintext stream is `n_chunks * block_size` bytes
/// long. Reads slice into that virtual range; each chunk is decrypted
/// on demand using AES-CBC + a per-chunk IV derived from HMAC-SHA1.
///
/// The decrypted stream is what would normally be a *plain* DMG (or
/// raw filesystem image). Higher layers can hand this backend straight
/// to [`crate::block::dmg::DmgBackend`] if they want to read the
/// koly-trailer payload that lives inside, but we don't do that here —
/// scope of this module is the encryption layer alone.
#[cfg(feature = "dmg-encrypted")]
#[derive(Debug)]
pub struct EncryptedDmgBackend {
    file: File,
    header: EncryptedDmgHeader,
    /// AES key recovered from the keyblob — 16 bytes (AES-128) or 32
    /// bytes (AES-256).
    aes_key: Vec<u8>,
    /// HMAC-SHA1 key recovered from the keyblob — 20 bytes.
    hmac_key: [u8; 20],
    /// Cached plaintext size: `n_chunks * block_size`.
    virtual_size: u64,
    /// Implicit `Seek` cursor for the `Read` / `Seek` impls.
    cursor: u64,
}

#[cfg(feature = "dmg-encrypted")]
impl EncryptedDmgBackend {
    /// Open `path` as an encrypted DMG, authenticating with `password`.
    ///
    /// Fails with [`crate::Error::Unsupported`] when the password
    /// produces a keyblob whose PKCS#7 padding is invalid — that's how
    /// 3DES-CBC fails when the KEK is wrong, so the error variant
    /// doubles as a "wrong password" signal.
    pub fn open_with_password(path: &Path, password: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let header = read_header(&mut file)?;

        // Reject anything we don't actually implement yet.
        if header.encryption_algorithm != 1 {
            return Err(crate::Error::Unsupported(format!(
                "encrcdsa: encryption_algorithm {} not supported (only 1 = AES_CBC)",
                header.encryption_algorithm
            )));
        }
        let aes_key_len = header.aes_key_len()?;
        if header.blob_enc_algorithm != 3 {
            return Err(crate::Error::Unsupported(format!(
                "encrcdsa: blob_enc_algorithm {} not supported (only 3 = 3DES_EDE3_CBC)",
                header.blob_enc_algorithm
            )));
        }
        if header.blob_enc_key_bits != 192 {
            return Err(crate::Error::Unsupported(format!(
                "encrcdsa: blob_enc_key_bits {} not supported (only 192 = 3DES)",
                header.blob_enc_key_bits
            )));
        }
        if header.block_size == 0 {
            return Err(crate::Error::InvalidImage(
                "encrcdsa: block_size is zero".into(),
            ));
        }

        // Derive the KEK with PBKDF2-SHA1. The output is 24 bytes (=
        // 3DES key length).
        let mut kek = [0u8; 24];
        pbkdf2::pbkdf2_hmac::<sha1::Sha1>(
            password.as_bytes(),
            header.salt(),
            header.pbkdf2_iteration_count,
            &mut kek,
        );

        // 3DES-CBC decrypt the keyblob with PKCS#7 padding stripped.
        let keyblob_plain = decrypt_keyblob(&kek, header.blob_iv(), &header.encrypted_keyblob)?;

        // The plaintext is `aes_key || hmac_sha1_key`. Apple sometimes
        // ships images where the two halves are identical, but we don't
        // care — we just split.
        let needed = aes_key_len + 20;
        if keyblob_plain.len() < needed {
            return Err(crate::Error::InvalidImage(format!(
                "encrcdsa: unwrapped keyblob too short ({} bytes, need >= {})",
                keyblob_plain.len(),
                needed
            )));
        }
        let aes_key = keyblob_plain[..aes_key_len].to_vec();
        let mut hmac_key = [0u8; 20];
        hmac_key.copy_from_slice(&keyblob_plain[aes_key_len..aes_key_len + 20]);

        let virtual_size = header
            .n_chunks
            .checked_mul(header.block_size as u64)
            .ok_or_else(|| {
                crate::Error::InvalidImage("encrcdsa: n_chunks * block_size overflows u64".into())
            })?;

        Ok(Self {
            file,
            header,
            aes_key,
            hmac_key,
            virtual_size,
            cursor: 0,
        })
    }

    /// Borrow the decoded header for diagnostics.
    pub fn header(&self) -> &EncryptedDmgHeader {
        &self.header
    }

    /// Decrypt the `chunk_index`-th chunk into a fresh `Vec<u8>` of
    /// length `block_size`. Used internally by [`read_at`].
    ///
    /// [`read_at`]: BlockDevice::read_at
    fn decrypt_chunk(&mut self, chunk_index: u64) -> Result<Vec<u8>> {
        let block_size = self.header.block_size as usize;
        // Read the chunk's ciphertext.
        let abs_offset = self
            .header
            .data_offset
            .checked_add(chunk_index * self.header.block_size as u64)
            .ok_or_else(|| {
                crate::Error::InvalidImage(
                    "encrcdsa: chunk absolute offset overflows the data fork".into(),
                )
            })?;
        self.file.seek(SeekFrom::Start(abs_offset))?;
        let mut ciphertext = vec![0u8; block_size];
        self.file.read_exact(&mut ciphertext)?;

        // IV = first 16 bytes of HMAC-SHA1(hmac_key, chunk_index_as_u32_be).
        let iv = chunk_iv(&self.hmac_key, chunk_index as u32);

        // AES-CBC decrypt in place. No padding — the chunk's ciphertext
        // is always a multiple of the AES block size (16 bytes), and the
        // plaintext is the chunk's literal contents.
        if ciphertext.len() % 16 != 0 {
            return Err(crate::Error::InvalidImage(format!(
                "encrcdsa: chunk ciphertext length {} is not a multiple of 16",
                ciphertext.len()
            )));
        }
        match self.aes_key.len() {
            16 => decrypt_aes128_cbc(&self.aes_key, &iv, &mut ciphertext)?,
            32 => decrypt_aes256_cbc(&self.aes_key, &iv, &mut ciphertext)?,
            other => {
                return Err(crate::Error::InvalidImage(format!(
                    "encrcdsa: unwrapped AES key has unexpected length {other}"
                )));
            }
        }
        Ok(ciphertext)
    }
}

/// AES-128-CBC decrypt `buf` in place. `buf.len()` MUST be a multiple
/// of 16; `iv` and `key` MUST be 16 bytes each.
#[cfg(feature = "dmg-encrypted")]
fn decrypt_aes128_cbc(key: &[u8], iv: &[u8; 16], buf: &mut [u8]) -> Result<()> {
    use cipher::{BlockDecryptMut, KeyIvInit, generic_array::GenericArray};

    let mut dec = cbc::Decryptor::<aes::Aes128>::new_from_slices(key, iv).map_err(|e| {
        crate::Error::InvalidImage(format!("encrcdsa: AES-128-CBC init failed: {e}"))
    })?;
    for block in buf.chunks_exact_mut(16) {
        let g: &mut GenericArray<u8, _> = GenericArray::from_mut_slice(block);
        dec.decrypt_block_mut(g);
    }
    Ok(())
}

/// AES-256-CBC decrypt `buf` in place. `buf.len()` MUST be a multiple
/// of 16; `iv` is 16 bytes, `key` is 32 bytes.
#[cfg(feature = "dmg-encrypted")]
fn decrypt_aes256_cbc(key: &[u8], iv: &[u8; 16], buf: &mut [u8]) -> Result<()> {
    use cipher::{BlockDecryptMut, KeyIvInit, generic_array::GenericArray};

    let mut dec = cbc::Decryptor::<aes::Aes256>::new_from_slices(key, iv).map_err(|e| {
        crate::Error::InvalidImage(format!("encrcdsa: AES-256-CBC init failed: {e}"))
    })?;
    for block in buf.chunks_exact_mut(16) {
        let g: &mut GenericArray<u8, _> = GenericArray::from_mut_slice(block);
        dec.decrypt_block_mut(g);
    }
    Ok(())
}

/// 3DES-CBC decrypt `ciphertext` with `(kek, iv)`, then strip PKCS#7
/// padding. Returns the plaintext keyblob (typically 36 or 52 bytes).
#[cfg(feature = "dmg-encrypted")]
fn decrypt_keyblob(kek: &[u8], iv: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>> {
    use cipher::{BlockDecryptMut, KeyIvInit, block_padding::Pkcs7};

    if kek.len() != 24 {
        return Err(crate::Error::InvalidImage(format!(
            "encrcdsa: KEK has wrong length {} (expected 24)",
            kek.len()
        )));
    }
    if iv.len() < 8 {
        return Err(crate::Error::InvalidImage(format!(
            "encrcdsa: 3DES IV slice too short ({} bytes)",
            iv.len()
        )));
    }
    // 3DES uses an 8-byte IV; the on-disk IV slot is 32 bytes but only
    // the first 8 are live.
    let iv8: [u8; 8] = iv[..8].try_into().unwrap();
    let dec = cbc::Decryptor::<des::TdesEde3>::new_from_slices(kek, &iv8)
        .map_err(|e| crate::Error::InvalidImage(format!("encrcdsa: 3DES-CBC init failed: {e}")))?;
    if ciphertext.len() % 8 != 0 || ciphertext.is_empty() {
        return Err(crate::Error::InvalidImage(format!(
            "encrcdsa: keyblob ciphertext length {} is not a positive multiple of 8",
            ciphertext.len()
        )));
    }
    let mut buf = ciphertext.to_vec();
    let plain_slice = dec.decrypt_padded_mut::<Pkcs7>(&mut buf).map_err(|_| {
        crate::Error::Unsupported(
            "encrcdsa: keyblob unwrap failed — wrong password, or unsupported padding".into(),
        )
    })?;
    let plain_len = plain_slice.len();
    buf.truncate(plain_len);
    Ok(buf)
}

/// Compute the AES-CBC IV for `chunk_index`: first 16 bytes of
/// `HMAC-SHA1(hmac_key, chunk_index_as_u32_be)`.
#[cfg(feature = "dmg-encrypted")]
fn chunk_iv(hmac_key: &[u8; 20], chunk_index: u32) -> [u8; 16] {
    use hmac::{Hmac, Mac};

    let mut mac =
        Hmac::<sha1::Sha1>::new_from_slice(hmac_key).expect("HMAC-SHA1 accepts any key length");
    mac.update(&chunk_index.to_be_bytes());
    let tag = mac.finalize().into_bytes();
    let mut iv = [0u8; 16];
    iv.copy_from_slice(&tag[..16]);
    iv
}

#[cfg(feature = "dmg-encrypted")]
impl BlockDevice for EncryptedDmgBackend {
    fn block_size(&self) -> u32 {
        // Logical sector hint — surface 512 for parity with the rest
        // of the stack; the AES chunk size (typically 4096) is a
        // separate concept.
        512
    }

    fn total_size(&self) -> u64 {
        self.virtual_size
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let size = self.virtual_size;
        let end = offset
            .checked_add(buf.len() as u64)
            .ok_or(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size,
            })?;
        if end > size {
            return Err(crate::Error::OutOfBounds {
                offset,
                len: buf.len() as u64,
                size,
            });
        }
        if buf.is_empty() {
            return Ok(());
        }

        let bs = self.header.block_size as u64;
        let mut filled = 0usize;
        let mut cursor = offset;
        while filled < buf.len() {
            let chunk_index = cursor / bs;
            let chunk_base = chunk_index * bs;
            let plain = self.decrypt_chunk(chunk_index)?;
            debug_assert_eq!(plain.len() as u64, bs);
            let local_start = (cursor - chunk_base) as usize;
            let available = (bs - (cursor - chunk_base)) as usize;
            let want = (buf.len() - filled).min(available);
            buf[filled..filled + want].copy_from_slice(&plain[local_start..local_start + want]);
            filled += want;
            cursor += want as u64;
        }
        Ok(())
    }

    fn write_at(&mut self, _offset: u64, _buf: &[u8]) -> Result<()> {
        Err(crate::Error::Unsupported(
            "encrcdsa: read-only container; writes are out of scope".into(),
        ))
    }
}

#[cfg(feature = "dmg-encrypted")]
impl Read for EncryptedDmgBackend {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.cursor >= self.virtual_size {
            return Ok(0);
        }
        let remaining = self.virtual_size - self.cursor;
        let take = (buf.len() as u64).min(remaining) as usize;
        if take == 0 {
            return Ok(0);
        }
        self.read_at(self.cursor, &mut buf[..take])
            .map_err(|e| io::Error::other(format!("{e}")))?;
        self.cursor += take as u64;
        Ok(take)
    }
}

#[cfg(feature = "dmg-encrypted")]
impl Write for EncryptedDmgBackend {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("encrcdsa: read-only container"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "dmg-encrypted")]
impl Seek for EncryptedDmgBackend {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let total = self.virtual_size;
        let new = match pos {
            SeekFrom::Start(o) => o,
            SeekFrom::Current(d) => (self.cursor as i64).saturating_add(d).max(0) as u64,
            SeekFrom::End(d) => (total as i64).saturating_add(d).max(0) as u64,
        };
        self.cursor = new;
        Ok(new)
    }
}

/// Fallback when the crate is built without `dmg-encrypted`. The header
/// still parses (so `fstool inspect` can recognise an encrypted DMG and
/// say something useful), but the `open_with_password` constructor
/// returns `Unsupported`. We expose a zero-sized type so the public
/// surface is identical between feature flags.
#[cfg(not(feature = "dmg-encrypted"))]
#[derive(Debug)]
pub struct EncryptedDmgBackend {
    _never: std::convert::Infallible,
}

#[cfg(not(feature = "dmg-encrypted"))]
impl EncryptedDmgBackend {
    /// Stub for `--no-default-features` builds. Always returns
    /// [`crate::Error::Unsupported`] with a message pointing at the
    /// `dmg-encrypted` feature flag.
    pub fn open_with_password(_path: &Path, _password: &str) -> Result<Self> {
        Err(crate::Error::Unsupported(
            "encrcdsa: encrypted DMG support requires the `dmg-encrypted` Cargo feature".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic v2 header buffer with the supplied fields and
    /// a heap-allocated keyblob. Returns a fresh `Vec<u8>` whose layout
    /// matches what an encrypted DMG would carry at file offset 0.
    #[allow(clippy::too_many_arguments)]
    fn build_header_bytes(
        iter_count: u32,
        salt: &[u8],
        blob_iv: &[u8],
        keyblob: &[u8],
        encryption_mode: u32,
        block_size: u32,
        n_chunks: u64,
        data_offset: u64,
    ) -> Vec<u8> {
        let mut buf = vec![0u8; ENCRCDSA_V2_HEADER_MIN_BYTES];
        buf[0..8].copy_from_slice(ENCRCDSA_MAGIC);
        buf[0x08..0x0C].copy_from_slice(&2u32.to_be_bytes());
        buf[0x0C..0x10].copy_from_slice(&32u32.to_be_bytes());
        buf[0x10..0x14].copy_from_slice(&encryption_mode.to_be_bytes());
        buf[0x14..0x18].copy_from_slice(&1u32.to_be_bytes()); // AES_CBC
        buf[0x18..0x1C].copy_from_slice(&0u32.to_be_bytes()); // PRNG = irrelevant for our parser
        buf[0x1C..0x20].copy_from_slice(&iter_count.to_be_bytes());
        buf[0x20..0x24].copy_from_slice(&(salt.len() as u32).to_be_bytes());
        buf[0x24..0x24 + salt.len()].copy_from_slice(salt);
        buf[0x44..0x48].copy_from_slice(&(blob_iv.len() as u32).to_be_bytes());
        buf[0x48..0x48 + blob_iv.len()].copy_from_slice(blob_iv);
        buf[0x68..0x6C].copy_from_slice(&192u32.to_be_bytes());
        buf[0x6C..0x70].copy_from_slice(&3u32.to_be_bytes()); // 3DES_EDE3_CBC
        buf[0x70..0x74].copy_from_slice(&7u32.to_be_bytes()); // PKCS#7
        buf[0x74..0x78].copy_from_slice(&6u32.to_be_bytes()); // CBC-pad-IV8
        buf[0x78..0x7C].copy_from_slice(&(keyblob.len() as u32).to_be_bytes());

        // Some real images carry a keyblob larger than the 64-byte slot
        // we naively assumed; we extend the buffer instead of cramping
        // it. Chunk-layout fields then live at 0xBC regardless — copy
        // the keyblob into [0x7C..0x7C+keyblob.len()] and pad with zeros
        // up to 0xBC.
        let blob_start = 0x7C;
        let blob_end = blob_start + keyblob.len();
        if blob_end > 0xBC {
            // Test inputs are constructed under our control; keep the
            // assertion loud rather than silently extending.
            panic!("keyblob too large for fixed slot");
        }
        buf[blob_start..blob_end].copy_from_slice(keyblob);

        buf[0xBC..0xC0].copy_from_slice(&block_size.to_be_bytes());
        buf[0xC0..0xC8].copy_from_slice(&n_chunks.to_be_bytes());
        buf[0xC8..0xD0].copy_from_slice(&data_offset.to_be_bytes());
        buf[0xD0..0xD8].copy_from_slice(&(n_chunks * block_size as u64).to_be_bytes());
        buf
    }

    #[test]
    fn header_decodes_minimal_v2() {
        let salt = b"saltsaltsaltsaltsalt"; // 20 bytes
        let blob_iv = b"iv8iv8iv"; // 8 bytes
        let keyblob = vec![0u8; 48];
        let buf = build_header_bytes(1000, salt, blob_iv, &keyblob, 0, 4096, 4, 0x1000);
        let h = EncryptedDmgHeader::decode(&buf).unwrap();
        assert_eq!(h.version, 2);
        assert_eq!(h.encryption_mode, 0);
        assert_eq!(h.encryption_algorithm, 1);
        assert_eq!(h.pbkdf2_iteration_count, 1000);
        assert_eq!(h.salt(), salt);
        assert_eq!(h.blob_iv(), blob_iv);
        assert_eq!(h.block_size, 4096);
        assert_eq!(h.n_chunks, 4);
        assert_eq!(h.data_offset, 0x1000);
        assert_eq!(h.aes_key_len().unwrap(), 16);
    }

    #[test]
    fn header_rejects_wrong_magic() {
        let mut buf =
            build_header_bytes(1000, b"salt", b"iv8iv8iv", &[0u8; 48], 0, 4096, 1, 0x1000);
        buf[0] = b'X';
        let err = EncryptedDmgHeader::decode(&buf).unwrap_err();
        match err {
            crate::Error::InvalidImage(_) => {}
            _ => panic!("expected InvalidImage, got {err:?}"),
        }
    }

    #[test]
    fn header_rejects_v1() {
        let mut buf =
            build_header_bytes(1000, b"salt", b"iv8iv8iv", &[0u8; 48], 0, 4096, 1, 0x1000);
        buf[0x08..0x0C].copy_from_slice(&1u32.to_be_bytes());
        let err = EncryptedDmgHeader::decode(&buf).unwrap_err();
        match err {
            crate::Error::Unsupported(_) => {}
            _ => panic!("expected Unsupported, got {err:?}"),
        }
    }

    #[test]
    fn probe_recognises_v2_magic() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("enc.dmg");
        let mut content = vec![0u8; 128];
        content[..8].copy_from_slice(ENCRCDSA_MAGIC);
        std::fs::write(&p, &content).unwrap();
        assert!(probe(&p).unwrap());
    }

    #[test]
    fn probe_misses_unrelated_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("not-encrypted.dmg");
        std::fs::write(&p, b"random bytes").unwrap();
        assert!(!probe(&p).unwrap());
    }

    #[cfg(not(feature = "dmg-encrypted"))]
    #[test]
    fn open_returns_unsupported_without_feature() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("enc.dmg");
        std::fs::write(&p, ENCRCDSA_MAGIC).unwrap();
        let err = EncryptedDmgBackend::open_with_password(&p, "irrelevant").unwrap_err();
        match err {
            crate::Error::Unsupported(_) => {}
            _ => panic!("expected Unsupported, got {err:?}"),
        }
    }

    /// End-to-end synthesise + decrypt round trip for AES-128. We build
    /// a one-chunk image by:
    ///
    ///   1. Picking a known passphrase + salt + iter count.
    ///   2. Deriving the KEK ourselves with PBKDF2-SHA1.
    ///   3. Concatenating an AES key + HMAC key and PKCS#7-padding the result.
    ///   4. Encrypting the keyblob with 3DES-EDE3-CBC under the KEK.
    ///   5. AES-CBC encrypting a 4096-byte plaintext chunk with the AES key
    ///      and a chunk-zero IV derived from HMAC-SHA1(hmac_key, 0u32).
    ///   6. Writing header + ciphertext to a temp file and reading it back
    ///      through `EncryptedDmgBackend::open_with_password`.
    #[cfg(feature = "dmg-encrypted")]
    #[test]
    fn round_trip_synthetic_aes128() {
        use cipher::{BlockEncryptMut, KeyIvInit, block_padding::Pkcs7};
        use hmac::{Hmac, Mac};

        let password = "correct horse battery staple";
        // Iteration count kept tiny on purpose — we don't want the test
        // suite to take seconds. Real images use 100k+.
        let iter_count: u32 = 100;
        let salt: &[u8] = b"saltsaltsaltsaltsalt"; // 20 bytes
        let blob_iv8: [u8; 8] = *b"ivivivIV";
        let aes_key: [u8; 16] = *b"AESKEY-128-BIT!!";
        let hmac_key: [u8; 20] = *b"HMACKEY-20-BYTES!!??";

        // 1) Derive the KEK the same way the open path does.
        let mut kek = [0u8; 24];
        pbkdf2::pbkdf2_hmac::<sha1::Sha1>(password.as_bytes(), salt, iter_count, &mut kek);

        // 2) Build + 3DES-CBC encrypt the keyblob.
        let mut keyblob_plain = Vec::new();
        keyblob_plain.extend_from_slice(&aes_key);
        keyblob_plain.extend_from_slice(&hmac_key);
        let mut keyblob_pad = keyblob_plain.clone();
        // PKCS#7 padding to the next multiple of 8.
        let unpadded_len = keyblob_pad.len();
        let pad_len = 8 - (unpadded_len % 8);
        keyblob_pad.resize(unpadded_len + 16, 0); // buffer space for encrypt
        let enc = cbc::Encryptor::<des::TdesEde3>::new_from_slices(&kek, &blob_iv8).unwrap();
        let ct = enc
            .encrypt_padded_mut::<Pkcs7>(&mut keyblob_pad, unpadded_len)
            .unwrap();
        let keyblob_ciphertext = ct.to_vec();
        assert_eq!(keyblob_ciphertext.len(), unpadded_len + pad_len);

        // 3) Build a 4096-byte plaintext chunk and encrypt it. The IV
        //    is HMAC-SHA1(hmac_key, 0u32 BE)[..16].
        let mut plaintext = vec![0u8; 4096];
        for (i, b) in plaintext.iter_mut().enumerate() {
            *b = ((i * 31 + 7) & 0xFF) as u8;
        }
        let mut mac = Hmac::<sha1::Sha1>::new_from_slice(&hmac_key).unwrap();
        mac.update(&0u32.to_be_bytes());
        let tag = mac.finalize().into_bytes();
        let mut iv16 = [0u8; 16];
        iv16.copy_from_slice(&tag[..16]);
        // Encrypt the chunk in place. No padding — chunk length is an
        // exact multiple of 16. The CBC state advances block-by-block.
        let mut aes_enc = cbc::Encryptor::<aes::Aes128>::new_from_slices(&aes_key, &iv16).unwrap();
        let mut chunk_ct = plaintext.clone();
        for block in chunk_ct.chunks_exact_mut(16) {
            let g = cipher::generic_array::GenericArray::from_mut_slice(block);
            aes_enc.encrypt_block_mut(g);
        }

        // 4) Lay out the file: header(0xD8 bytes) + chunk ciphertext.
        let data_offset = 0xD8u64;
        let mut file_bytes = build_header_bytes(
            iter_count,
            salt,
            &blob_iv8,
            &keyblob_ciphertext,
            0,
            4096,
            1,
            data_offset,
        );
        // Sanity: build_header_bytes always emits exactly 0xD8 bytes.
        assert_eq!(file_bytes.len(), ENCRCDSA_V2_HEADER_MIN_BYTES);
        file_bytes.extend_from_slice(&chunk_ct);

        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("enc.dmg");
        std::fs::write(&p, &file_bytes).unwrap();

        // 5) Read it back.
        let mut be = EncryptedDmgBackend::open_with_password(&p, password).unwrap();
        assert_eq!(be.total_size(), 4096);
        let mut out = vec![0u8; 4096];
        be.read_at(0, &mut out).unwrap();
        assert_eq!(out, plaintext);

        // Mid-chunk slice.
        let mut mid = vec![0u8; 16];
        be.read_at(100, &mut mid).unwrap();
        assert_eq!(mid, &plaintext[100..116]);
    }

    /// Same as `round_trip_synthetic_aes128` but with a 32-byte AES key
    /// (`encryption_mode = 1`). Cross-checks the AES-256-CBC dispatch.
    #[cfg(feature = "dmg-encrypted")]
    #[test]
    fn round_trip_synthetic_aes256() {
        use cipher::{BlockEncryptMut, KeyIvInit, block_padding::Pkcs7};
        use hmac::{Hmac, Mac};

        let password = "another-password";
        let iter_count: u32 = 64;
        let salt: &[u8] = b"sodium_chloride_xx"; // 18 bytes
        let blob_iv8: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let aes_key: [u8; 32] = *b"AES256-KEY-MATERIAL-32-BYTES---!";
        let hmac_key: [u8; 20] = *b"hmac-key-20-bytes-OK";

        let mut kek = [0u8; 24];
        pbkdf2::pbkdf2_hmac::<sha1::Sha1>(password.as_bytes(), salt, iter_count, &mut kek);

        let mut keyblob_plain = Vec::new();
        keyblob_plain.extend_from_slice(&aes_key);
        keyblob_plain.extend_from_slice(&hmac_key);
        let unpadded_len = keyblob_plain.len();
        let mut keyblob_pad = keyblob_plain.clone();
        keyblob_pad.resize(unpadded_len + 16, 0);
        let enc = cbc::Encryptor::<des::TdesEde3>::new_from_slices(&kek, &blob_iv8).unwrap();
        let ct = enc
            .encrypt_padded_mut::<Pkcs7>(&mut keyblob_pad, unpadded_len)
            .unwrap();
        let keyblob_ciphertext = ct.to_vec();

        // Two-chunk plaintext to also exercise the second-chunk IV path.
        let mut plain = vec![0u8; 8192];
        for (i, b) in plain.iter_mut().enumerate() {
            *b = ((i ^ (i >> 4)) & 0xFF) as u8;
        }
        let mut chunks_ct = Vec::with_capacity(8192);
        for chunk_idx in 0u32..2 {
            let mut mac = Hmac::<sha1::Sha1>::new_from_slice(&hmac_key).unwrap();
            mac.update(&chunk_idx.to_be_bytes());
            let tag = mac.finalize().into_bytes();
            let mut iv16 = [0u8; 16];
            iv16.copy_from_slice(&tag[..16]);
            let mut aes_enc =
                cbc::Encryptor::<aes::Aes256>::new_from_slices(&aes_key, &iv16).unwrap();
            let start = (chunk_idx as usize) * 4096;
            let mut buf = plain[start..start + 4096].to_vec();
            for block in buf.chunks_exact_mut(16) {
                let g = cipher::generic_array::GenericArray::from_mut_slice(block);
                aes_enc.encrypt_block_mut(g);
            }
            chunks_ct.extend_from_slice(&buf);
        }

        let data_offset = 0xD8u64;
        let mut file_bytes = build_header_bytes(
            iter_count,
            salt,
            &blob_iv8,
            &keyblob_ciphertext,
            1, // encryption_mode = AES-256
            4096,
            2,
            data_offset,
        );
        file_bytes.extend_from_slice(&chunks_ct);

        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("enc256.dmg");
        std::fs::write(&p, &file_bytes).unwrap();

        let mut be = EncryptedDmgBackend::open_with_password(&p, password).unwrap();
        assert_eq!(be.total_size(), 8192);
        let mut out = vec![0u8; 8192];
        be.read_at(0, &mut out).unwrap();
        assert_eq!(out, plain);

        // Cross-chunk slice that straddles the boundary.
        let mut cross = vec![0u8; 64];
        be.read_at(4096 - 32, &mut cross).unwrap();
        assert_eq!(&cross[..32], &plain[4096 - 32..4096]);
        assert_eq!(&cross[32..], &plain[4096..4096 + 32]);
    }

    /// Wrong password produces `Unsupported` from `decrypt_keyblob`
    /// (the PKCS#7 unpad fails). Confirms the error variant used as
    /// the "bad password" signal.
    #[cfg(feature = "dmg-encrypted")]
    #[test]
    fn wrong_password_rejected() {
        use cipher::{BlockEncryptMut, KeyIvInit, block_padding::Pkcs7};

        let password = "supersecret";
        let iter_count: u32 = 100;
        let salt: &[u8] = b"saltsaltsaltsaltsalt";
        let blob_iv8: [u8; 8] = *b"ivivivIV";
        let aes_key: [u8; 16] = *b"AESKEY-128-BIT!!";
        let hmac_key: [u8; 20] = *b"HMACKEY-20-BYTES!!??";

        let mut kek = [0u8; 24];
        pbkdf2::pbkdf2_hmac::<sha1::Sha1>(password.as_bytes(), salt, iter_count, &mut kek);

        let mut keyblob_plain = Vec::new();
        keyblob_plain.extend_from_slice(&aes_key);
        keyblob_plain.extend_from_slice(&hmac_key);
        let unpadded_len = keyblob_plain.len();
        let mut keyblob_pad = keyblob_plain.clone();
        keyblob_pad.resize(unpadded_len + 16, 0);
        let enc = cbc::Encryptor::<des::TdesEde3>::new_from_slices(&kek, &blob_iv8).unwrap();
        let ct = enc
            .encrypt_padded_mut::<Pkcs7>(&mut keyblob_pad, unpadded_len)
            .unwrap();
        let keyblob_ciphertext = ct.to_vec();

        let data_offset = 0xD8u64;
        let mut file_bytes = build_header_bytes(
            iter_count,
            salt,
            &blob_iv8,
            &keyblob_ciphertext,
            0,
            4096,
            1,
            data_offset,
        );
        file_bytes.extend_from_slice(&[0u8; 4096]);

        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("enc.dmg");
        std::fs::write(&p, &file_bytes).unwrap();

        let err = EncryptedDmgBackend::open_with_password(&p, "wrong-password").unwrap_err();
        match err {
            crate::Error::Unsupported(msg) => {
                assert!(msg.contains("wrong password") || msg.contains("padding"));
            }
            _ => panic!("expected Unsupported, got {err:?}"),
        }
    }

    /// Out-of-bounds read returns `OutOfBounds`.
    #[cfg(feature = "dmg-encrypted")]
    #[test]
    fn read_at_rejects_out_of_bounds() {
        use cipher::{BlockEncryptMut, KeyIvInit, block_padding::Pkcs7};

        let password = "pw";
        let iter_count: u32 = 50;
        let salt: &[u8] = b"saltsaltsaltsaltsalt";
        let blob_iv8: [u8; 8] = *b"ivivivIV";
        let aes_key: [u8; 16] = *b"AESKEY-128-BIT!!";
        let hmac_key: [u8; 20] = *b"HMACKEY-20-BYTES!!??";

        let mut kek = [0u8; 24];
        pbkdf2::pbkdf2_hmac::<sha1::Sha1>(password.as_bytes(), salt, iter_count, &mut kek);

        let mut keyblob_plain = Vec::new();
        keyblob_plain.extend_from_slice(&aes_key);
        keyblob_plain.extend_from_slice(&hmac_key);
        let unpadded_len = keyblob_plain.len();
        let mut keyblob_pad = keyblob_plain.clone();
        keyblob_pad.resize(unpadded_len + 16, 0);
        let enc = cbc::Encryptor::<des::TdesEde3>::new_from_slices(&kek, &blob_iv8).unwrap();
        let ct = enc
            .encrypt_padded_mut::<Pkcs7>(&mut keyblob_pad, unpadded_len)
            .unwrap();
        let keyblob_ciphertext = ct.to_vec();

        // Two zero chunks of ciphertext (decryption result will be
        // garbage but won't trip OOB checks since we only test that
        // path).
        let mut file_bytes = build_header_bytes(
            iter_count,
            salt,
            &blob_iv8,
            &keyblob_ciphertext,
            0,
            4096,
            2,
            0xD8,
        );
        file_bytes.extend_from_slice(&[0u8; 8192]);
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("enc.dmg");
        std::fs::write(&p, &file_bytes).unwrap();

        let mut be = EncryptedDmgBackend::open_with_password(&p, password).unwrap();
        assert_eq!(be.total_size(), 8192);
        let mut out = [0u8; 16];
        let err = be.read_at(8192, &mut out).unwrap_err();
        match err {
            crate::Error::OutOfBounds { .. } => {}
            _ => panic!("expected OutOfBounds, got {err:?}"),
        }
    }
}
