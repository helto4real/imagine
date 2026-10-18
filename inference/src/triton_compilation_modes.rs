//! Source compiler modes, separate from the JIT specialization memory cache.
use crate::{
    native::Error,
    triton_file_cache::{CacheText, JsonPath},
};
use std::{path::PathBuf, sync::Arc};
#[derive(Clone, Default)]
pub struct Modes {
    /// Explicit original triton.compile _env_vars; None reads the native list.
    pub environment: Option<std::collections::BTreeMap<String, String>>,
    pub always_compile: bool,
    pub store_binary_only: bool,
    pub override_directory: Option<PathBuf>,
    pub dump_directory: Option<PathBuf>,
    pub listener: Option<Listener>,
}
pub struct Compilation<'a> {
    pub ast_digest: &'a str,
    pub metadata: &'a serde_json::Value,
    pub metadata_group: &'a [(CacheText, JsonPath)],
    pub cache_hit: bool,
    pub lowering_stages: &'a [&'static str],
}
pub type Listener = Arc<dyn Fn(&Compilation<'_>) -> Result<(), Error> + Send + Sync>;
/// Port of triton.tools.disasm.extract/processSassLines/parseCtrl.
pub(crate) fn sass(text: &str) -> Option<String> {
    let lines = text.lines().collect::<Vec<_>>();
    let (header, name) = lines.iter().enumerate().find_map(|(i, line)| {
        let name = line.trim().strip_prefix("Function : ")?.trim();
        name.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
            .then_some((i, name))
    })?;
    let mut i = header + 2;
    let mut instructions = vec![];
    let mut labels = std::collections::BTreeMap::new();
    while let Some(line) = lines.get(i) {
        let line = line.trim();
        if !line.starts_with("/*")
            || line
                .get(2..6)
                .is_none_or(|s| !s.chars().all(|c| c.is_ascii_hexdigit()))
            || line.get(6..8) != Some("*/")
        {
            break;
        }
        let asm = line.split_once("*/")?.1.trim();
        let mut asm = asm[..=asm.find(';')?].trim().to_string();
        if asm.ends_with(" ;") {
            asm.truncate(asm.len() - 2);
            asm.push(';');
        }
        let enc = lines
            .get(i + 1)?
            .split_once("0x")?
            .1
            .split_whitespace()
            .next()?;
        let enc = u64::from_str_radix(enc, 16).ok()?;
        let target = branch(&asm).and_then(|(_, address)| u64::from_str_radix(address, 16).ok());
        if let Some(target) = target {
            let next = labels.len();
            labels.entry(target).or_insert(next);
        }
        let field = |v: u64| {
            if v == 7 {
                "-".to_string()
            } else {
                v.to_string()
            }
        };
        let wait = (enc >> 52) & 0x3f;
        let ctrl = format!(
            "{}:{}:{}:{}:{:x}",
            if wait == 0 {
                "--".into()
            } else {
                format!("{wait:02}")
            },
            field((enc >> 49) & 7),
            field((enc >> 46) & 7),
            if (enc >> 45) & 1 == 0 { "Y" } else { "-" },
            (enc >> 41) & 15
        );
        instructions.push((ctrl, asm, target));
        i += 2;
    }
    let mut result = format!("Function:{name}\n");
    for (i, (ctrl, mut asm, target)) in instructions.into_iter().enumerate() {
        if let Some(label) = labels.get(&(i as u64 * 16)) {
            result.push_str(&format!("LBB{label}:\n"));
        }
        if let Some(target) = target {
            let (prefix, _) = branch(&asm)?;
            asm = format!("{prefix}LBB{};", labels[&target]);
        }
        result.push_str(&format!("{ctrl}\t{asm}\n"));
    }
    result.push('\n');
    Some(result)
}
fn branch(asm: &str) -> Option<(&str, &str)> {
    let i = asm
        .rfind("BRA ")
        .map(|i| i + 4)
        .or_else(|| asm.rfind("BRA.U ").map(|i| i + 6))?;
    let address = asm[i..].strip_prefix("0x")?.strip_suffix(';')?;
    Some((&asm[..i], address))
}
