# Example `fstool build` specs

Each `.toml` here is a self-contained image specification. Build one with:

```sh
fstool build examples/<name>.toml -o /tmp/out.img
fstool info /tmp/out.img
```

The specs reference host directories via commented-out `source = "…"` lines, so
they build an empty-but-correctly-laid-out image out of the box. Uncomment and
point `source` at your own tree to populate the filesystems.

| Spec | Layout |
|------|--------|
| [`bare-ext4.toml`](bare-ext4.toml) | A single ext4 filesystem, no partition table (the `genext2fs` replacement). |
| [`bare-fat32.toml`](bare-fat32.toml) | A single FAT32 filesystem, no partition table. |
| [`raspberry-pi.toml`](raspberry-pi.toml) | Raspberry Pi SD card — MBR with a FAT32 boot partition + ext4 root. |
| [`efi-disk.toml`](efi-disk.toml) | UEFI-bootable GPT disk (also a UEFI USB image) — ESP (FAT32) + ext4 root. |
| [`bios-legacy-disk.toml`](bios-legacy-disk.toml) | Legacy-BIOS MBR disk — ext2 `/boot` + ext4 root. |

## What `build` does and doesn't do

`fstool build` creates the partition table and formats + populates the
filesystems. It does **not**:

- **install a bootloader** — run `grub-install` / `syslinux` (BIOS) afterwards,
  or place a UEFI application at `EFI/BOOT/BOOTX64.EFI` in the ESP source tree
  (UEFI). fstool just provides the layout and filesystems.
- **set the MBR active/boot flag** — the bootloader installer sets it; the spec
  has no per-partition bootable flag yet.

A bootable **USB** image is just the GPT + ESP layout (`efi-disk.toml`) written
to the stick. A bootable hybrid **ISO** (El Torito / isohybrid) is not yet
expressible through `build` and is planned for a later iteration.
