//! ONLY synthetic task fixtures. Not part of product storage configuration.
#![allow(dead_code)]
use std::path::{Component, Path, PathBuf};

// Tomas manually relocated these agent-owned task files. Never generalize this
// test-only root to user projects/media or substitute it for product policy.
pub const ROOT: &str = "/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01";
pub const MOUNT: &str = "/home";
// mountinfo uses the block device; findmnt's [/@home] is a subvolume annotation.
pub const SOURCE: &str = "/dev/mapper/cryptroot";
pub const FILESYSTEM: &str = "btrfs";

pub fn is_owned(path: &Path) -> bool {
    path.is_absolute()
        && path.starts_with(ROOT)
        && path
            .components()
            .all(|part| matches!(part, Component::RootDir | Component::Normal(_)))
}

/// Existing, explicit test directory only; resolve before creating fixture files.
pub fn env_root(name: &str) -> PathBuf {
    let path = PathBuf::from(
        std::env::var_os(name).unwrap_or_else(|| panic!("explicit own fixture {name} required")),
    );
    assert!(
        is_owned(&path),
        "{name} must be inside the approved synthetic task root"
    );
    let resolved = path
        .canonicalize()
        .unwrap_or_else(|error| panic!("cannot resolve own fixture {name}: {error}"));
    assert!(
        is_owned(&resolved),
        "{name} resolves outside the approved synthetic task root"
    );
    assert!(
        resolved.is_dir(),
        "{name} must name an own fixture directory"
    );
    resolved
}
