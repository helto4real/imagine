//! Original cache-invalidating environment normalization, separate from knobs.
use crate::native::Error;
use std::collections::BTreeMap;
include!("triton_environment_names.rs");
pub fn normalize(values: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    NAMES
        .iter()
        .filter_map(|name| {
            let value = values.get(*name)?;
            if value.is_empty() {
                return None;
            }
            let normalized = match value.to_ascii_lowercase().as_str() {
                "on" | "true" | "1" => "true",
                "off" | "false" | "0" => "false",
                _ => value,
            };
            Some((name.to_string(), normalized.to_string()))
        })
        .collect()
}
pub fn from_process() -> Result<BTreeMap<String, String>, Error> {
    use std::os::unix::ffi::OsStrExt;
    let mut values = BTreeMap::new();
    for name in NAMES {
        if let Some(value) = std::env::var_os(name) {
            values.insert(
                name.to_string(),
                crate::triton_compiler::utf8(value.as_bytes())?,
            );
        }
    }
    Ok(normalize(&values))
}
/// The Python knob getter also accepts y/yes, unlike cache normalization and
/// the separate native compiler getBoolEnv helper.
pub fn boolean_knob(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "1" | "y" | "yes" | "on" | "true"
    )
}
