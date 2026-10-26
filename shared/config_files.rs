//! Editable templates, profiles and guides under one configuration root
//! (`<project>/config`). Each file is read once per process, so a restart applies
//! an edit. Without `init`, development builds and tests use the source tree's root.
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

static ROOT: OnceLock<PathBuf> = OnceLock::new();
static TEXTS: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();

/// Sets the root before any file is read; returns false if one was already chosen.
pub fn init(root: PathBuf) -> bool {
    ROOT.set(root).is_ok()
}

/// The configuration root.
pub fn root() -> &'static Path {
    ROOT.get_or_init(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .map(|dir| dir.join("config"))
            .find(|dir| dir.join("mcp").is_dir())
            .unwrap_or_else(|| PathBuf::from("config"))
    })
}

/// A path below the root, such as `video/prompt-profiles`.
pub fn path(relative: &str) -> PathBuf {
    root().join(relative)
}

/// A text file below the root, read on first use and kept for the process.
pub fn text(relative: &str) -> std::io::Result<&'static str> {
    let texts = TEXTS.get_or_init(Default::default);
    if let Some(text) = texts.lock().unwrap().get(relative) {
        return Ok(text);
    }
    let path = path(relative);
    if std::fs::metadata(&path)?.len() > 4 * 1024 * 1024 {
        return Err(std::io::Error::other(format!(
            "{} is larger than 4 MiB",
            path.display()
        )));
    }
    let text: &'static str = std::fs::read_to_string(&path)
        .map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", path.display())))?
        .leak();
    Ok(*texts
        .lock()
        .unwrap()
        .entry(relative.to_owned())
        .or_insert(text))
}

/// Reads every listed file at startup, so a missing or unreadable one stops the process
/// before it serves anything; `required` then cannot fail for them.
pub fn preload(files: &[&str]) -> std::io::Result<()> {
    files.iter().try_for_each(|file| text(file).map(|_| ()))
}

/// A file that `preload` read at startup.
pub fn required(relative: &str) -> &'static str {
    text(relative).unwrap_or_else(|e| panic!("configuration file not preloaded: {e}"))
}
