//! Local, optional LoRA discovery. No inference validation or remote lookup occurs here.
#[path = "../../shared/lora_catalog.rs"]
mod wire;
pub use wire::{Catalog, Entry, Info};

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

// A header is parsed in memory, tensors never are. Match safetensors' 100 MB header bound.
const MAX_HEADER_BYTES: u64 = 100_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    CatalogUnavailable,
    InfoMissing,
    InfoUnreadable,
    InfoFormat,
}
impl Error {
    pub fn code(self) -> &'static str {
        match self {
            Self::CatalogUnavailable => "lora_catalog_unavailable",
            Self::InfoMissing => "lora_info_missing",
            Self::InfoUnreadable => "lora_info_unreadable",
            Self::InfoFormat => "lora_info_format",
        }
    }
}

/// Enumerate all regular files without selecting models by name or extension.
/// Canonical directory identities prevent symlink loops; sorted traversal is deterministic.
pub fn scan(root: &Path) -> Result<Catalog, Error> {
    let metadata = fs::metadata(root).map_err(|_| Error::CatalogUnavailable)?;
    if !metadata.is_dir() {
        return Err(Error::CatalogUnavailable);
    }
    let mut catalog = Catalog::default();
    let mut pending = vec![(root.to_path_buf(), PathBuf::new())];
    let mut visited = BTreeSet::new();
    while let Some((directory, relative)) = pending.pop() {
        let canonical = match fs::canonicalize(&directory) {
            Ok(path) => path,
            Err(_) if relative.as_os_str().is_empty() => return Err(Error::CatalogUnavailable),
            Err(_) => {
                catalog
                    .warnings
                    .push("lora_catalog_directory_unreadable".into());
                continue;
            }
        };
        if !visited.insert(canonical) {
            catalog
                .warnings
                .push("lora_catalog_directory_cycle_or_alias".into());
            continue;
        }
        let children = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(_) if relative.as_os_str().is_empty() => return Err(Error::CatalogUnavailable),
            Err(_) => {
                catalog
                    .warnings
                    .push("lora_catalog_directory_unreadable".into());
                continue;
            }
        };
        let mut children = children
            .filter_map(|entry| match entry {
                Ok(entry) => Some(entry),
                Err(_) => {
                    catalog
                        .warnings
                        .push("lora_catalog_entry_unreadable".into());
                    None
                }
            })
            .collect::<Vec<_>>();
        children.sort_by_key(|entry| entry.file_name());
        let mut directories = Vec::new();
        for child in children {
            let path = child.path();
            let relative = relative.join(child.file_name());
            let Some(name) = relative.to_str() else {
                catalog
                    .warnings
                    .push("lora_catalog_name_unrepresentable".into());
                continue;
            };
            match fs::metadata(&path) {
                Ok(metadata) if metadata.is_dir() => directories.push((path, relative)),
                Ok(metadata) if metadata.is_file() => catalog.files.push(Entry {
                    file: name.replace(std::path::MAIN_SEPARATOR, "/"),
                    size: metadata.len(),
                }),
                Ok(_) => {}
                Err(_) => catalog
                    .warnings
                    .push("lora_catalog_entry_unreadable".into()),
            }
        }
        // Stack order preserves lexical traversal (also determines a symlink alias' chosen name).
        pending.extend(directories.into_iter().rev());
    }
    catalog.files.sort_by(|a, b| a.file.cmp(&b.file));
    catalog.warnings.sort();
    catalog.warnings.dedup();
    catalog.root = Some(root.to_string_lossy().into_owned());
    Ok(catalog)
}

/// Inspect the selected file directly, including compatible manual paths outside the catalog.
/// No sidecar is written and neither tensors nor pickle objects are loaded.
pub fn info(root: &Path, choice: &str) -> Result<Info, Error> {
    if choice.trim().is_empty() {
        return Err(Error::InfoMissing);
    }
    let selected = Path::new(choice);
    let path = if selected.is_absolute() {
        selected.to_path_buf()
    } else {
        root.join(selected)
    };
    if !fs::metadata(&path).map_err(info_io)?.is_file() {
        return Err(Error::InfoFormat);
    }
    let mut file = File::open(&path).map_err(info_io)?;
    let metadata = file.metadata().map_err(info_io)?;
    if !metadata.is_file() {
        return Err(Error::InfoFormat);
    }
    let length = metadata.len();
    let mut prefix = [0u8; 8];
    let count = length.min(8) as usize;
    file.read_exact(&mut prefix[..count]).map_err(info_io)?;
    let (format, attributes) = if count >= 4 && &prefix[..4] == b"GGUF" {
        // Version is part of the container, not a model allow-list. Inspect only the fixed header.
        if count < 8
            || !matches!(u32::from_le_bytes(prefix[4..8].try_into().unwrap()), 1..=3)
            || length < 24
        {
            return Err(Error::InfoFormat);
        }
        ("GGUF", Map::new())
    } else if count >= 4 && &prefix[..4] == b"PK\x03\x04" {
        // A ZIP prefix alone does not prove PyTorch; report the container without unpickling it.
        ("ZIP (possible PyTorch)", Map::new())
    } else if count >= 2 && prefix[0] == 0x80 && (2..=5).contains(&prefix[1]) {
        ("Pickle (possible PyTorch)", Map::new())
    } else {
        if count != 8 {
            return Err(Error::InfoFormat);
        }
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_HEADER_BYTES || header_len > length.saturating_sub(8) {
            return Err(Error::InfoFormat);
        }
        let header_len = usize::try_from(header_len).map_err(|_| Error::InfoFormat)?;
        let mut bytes = vec![0; header_len];
        file.read_exact(&mut bytes).map_err(info_io)?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|_| Error::InfoFormat)?;
        let header = value.as_object().ok_or(Error::InfoFormat)?;
        validate_header(header, length - 8 - header_len as u64)?;
        let attributes = match header.get("__metadata__") {
            None => Map::new(),
            Some(Value::Object(value)) if value.values().all(Value::is_string) => value.clone(),
            _ => return Err(Error::InfoFormat),
        };
        ("safetensors", attributes)
    };
    let mut result = Info {
        file: choice.into(),
        name: path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or(choice)
            .into(),
        format: format.into(),
        ..Info::default()
    };
    if let Some(name) = text(&attributes, &["ss_output_name", "modelspec.title"]) {
        result.name = name;
    }
    result.base_model = text(
        &attributes,
        &[
            "ss_base_model_version",
            "ss_sd_model_name",
            "modelspec.architecture",
        ],
    )
    .unwrap_or_default();
    result.model_type = text(
        &attributes,
        &[
            "ss_network_module",
            "ss_network_type",
            "modelspec.implementation",
        ],
    )
    .unwrap_or_default();
    result.trained_words = trained_words(&attributes);
    // Hash only for explicit information requests, streaming the same opened file.
    file.seek(SeekFrom::Start(0)).map_err(info_io)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(info_io)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    result.sha256 = format!("{:x}", digest.finalize());
    Ok(result)
}

fn info_io(error: std::io::Error) -> Error {
    if error.kind() == std::io::ErrorKind::NotFound {
        Error::InfoMissing
    } else {
        Error::InfoUnreadable
    }
}

fn validate_header(header: &Map<String, Value>, data_len: u64) -> Result<(), Error> {
    let mut intervals = Vec::new();
    for (name, tensor) in header {
        if name == "__metadata__" {
            continue;
        }
        let tensor = tensor.as_object().ok_or(Error::InfoFormat)?;
        if tensor
            .get("dtype")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err(Error::InfoFormat);
        }
        let shape = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or(Error::InfoFormat)?;
        if !shape.iter().all(|v| v.as_u64().is_some()) {
            return Err(Error::InfoFormat);
        }
        let offsets = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or(Error::InfoFormat)?;
        if offsets.len() != 2 {
            return Err(Error::InfoFormat);
        }
        let start = offsets[0].as_u64().ok_or(Error::InfoFormat)?;
        let end = offsets[1].as_u64().ok_or(Error::InfoFormat)?;
        if start > end || end > data_len {
            return Err(Error::InfoFormat);
        }
        intervals.push((start, end));
    }
    intervals.sort_unstable();
    let mut end = 0;
    for (start, next) in intervals {
        if start != end {
            return Err(Error::InfoFormat);
        }
        end = next;
    }
    if end != data_len {
        return Err(Error::InfoFormat);
    }
    Ok(())
}

fn text(attributes: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| attributes.get(*key)?.as_str())
        .find(|v| !v.trim().is_empty())
        .map(str::to_owned)
}

fn trained_words(attributes: &Map<String, Value>) -> Vec<String> {
    let Some(value) = attributes.get("ss_tag_frequency") else {
        return Vec::new();
    };
    let decoded;
    let value = if let Some(text) = value.as_str() {
        decoded = match serde_json::from_str::<Value>(text) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        &decoded
    } else {
        value
    };
    let mut counts = BTreeMap::<String, f64>::new();
    if let Some(buckets) = value.as_object() {
        for bucket in buckets.values() {
            if let Some(words) = bucket.as_object() {
                for (word, frequency) in words {
                    let frequency = frequency
                        .as_f64()
                        .or_else(|| frequency.as_str()?.parse().ok());
                    if let Some(frequency) =
                        frequency.filter(|count| count.is_finite() && *count >= 0.0)
                    {
                        *counts.entry(word.clone()).or_default() += frequency;
                    }
                }
            }
        }
    }
    let mut counts = counts.into_iter().collect::<Vec<_>>();
    counts.sort_by(|(a, ac), (b, bc)| bc.total_cmp(ac).then(a.cmp(b)));
    counts.into_iter().map(|(word, _)| word).collect()
}
