//! LLM backends from `proxy/backends.json` under the editable configuration
//! root. An adapter is compiled protocol code; a backend is a configured
//! instance of one, so backends and models change without a rebuild.
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The repository's file, used only when the configuration root has none.
pub const BUILTIN: &str = include_str!("../../config/proxy/backends.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Adapter {
    /// KoboldCpp, in Router mode with model administration.
    Koboldcpp,
    Ollama,
    Tabbyapi,
    /// llama.cpp's router server with model presets (Prism).
    LlamaRouter,
    /// Strata servers with lazy loading and explicit load/unload.
    Strata,
}
impl Adapter {
    pub fn name(self) -> &'static str {
        match self {
            Self::Koboldcpp => "koboldcpp",
            Self::Ollama => "ollama",
            Self::Tabbyapi => "tabbyapi",
            Self::LlamaRouter => "llama_router",
            Self::Strata => "strata",
        }
    }
}

/// How the stack supervisor starts and recognizes the backend's process.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    /// Working directory; a leading `~/` is the home directory.
    pub dir: String,
    /// Executable and arguments; `{port}` is the origin's port.
    pub command: Vec<String>,
    /// Lowercase text in the process command line that identifies it in `dir`.
    #[serde(rename = "match")]
    pub process: String,
    /// Arguments added only when every backend runs together.
    #[serde(default)]
    pub all_args: Vec<String>,
    /// Color the backend's terminal output with st-strata-logging.
    #[serde(default)]
    pub terminal_filter: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub id: String,
    pub label: String,
    pub adapter: Adapter,
    #[serde(default = "yes")]
    pub enabled: bool,
    /// Whether "all" includes this backend.
    #[serde(default = "yes")]
    pub in_all: bool,
    /// Whether the stack stops when the backend cannot start, instead of
    /// continuing without it.
    #[serde(default)]
    pub required: bool,
    pub origin: String,
    /// Default preset (llama_router) or model (tabbyapi); the global option
    /// applies when unset.
    #[serde(default)]
    pub default_model: Option<String>,
    /// Paths the backend needs, such as its weights; a missing one skips it.
    #[serde(default)]
    pub requires: Vec<String>,
    #[serde(default)]
    pub launch: Option<Launch>,
}
fn yes() -> bool {
    true
}
impl Spec {
    /// Why the backend cannot start here, if it cannot.
    pub fn missing(&self) -> Option<String> {
        self.requires
            .iter()
            .map(|p| expand(p))
            .find(|p| !p.exists())
            .map(|p| format!("missing {}", p.display()))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    schema: u32,
    backends: Vec<Spec>,
}

pub fn expand(path: &str) -> PathBuf {
    let rest = if path == "~" {
        Some("")
    } else {
        path.strip_prefix("~/")
    };
    match (rest, std::env::var_os("HOME")) {
        (Some(""), Some(home)) => PathBuf::from(home),
        (Some(rest), Some(home)) => Path::new(&home).join(rest),
        _ => PathBuf::from(path),
    }
}

/// The backends file: an explicit path, else `<config>/proxy/backends.json`,
/// else the repository's copy.
pub fn load(explicit: Option<&str>, config_root: &str) -> Result<Vec<Spec>> {
    let path = match explicit {
        Some(p) => Some(PathBuf::from(p)),
        None => Some(Path::new(config_root).join("proxy/backends.json")).filter(|p| p.is_file()),
    };
    let text = match &path {
        Some(p) => std::fs::read_to_string(p)
            .with_context(|| format!("could not read backends file {}", p.display()))?,
        None => BUILTIN.to_owned(),
    };
    parse(&text).with_context(|| match &path {
        Some(p) => format!("invalid backends file {}", p.display()),
        None => "invalid built-in backends".into(),
    })
}

pub fn parse(text: &str) -> Result<Vec<Spec>> {
    let file: File = serde_json::from_str(text)?;
    if file.schema != 1 {
        bail!("unsupported backends schema {}", file.schema)
    }
    let mut ids = std::collections::HashSet::new();
    for spec in &file.backends {
        let id = &spec.id;
        if id.is_empty()
            || id == "all"
            || !id
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        {
            bail!("backend id {id:?} must be lowercase letters, digits and _, and not \"all\"")
        }
        if !ids.insert(id.as_str()) {
            bail!("duplicate backend id {id:?}")
        }
        if spec.label.trim().is_empty() {
            bail!("backend {id}: label cannot be empty")
        }
        if let Some(launch) = &spec.launch
            && (launch.dir.is_empty() || launch.command.is_empty() || launch.process.is_empty())
        {
            bail!("backend {id}: launch needs dir, command and match")
        }
    }
    Ok(file.backends)
}

fn quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// zsh assignments for the stack supervisor: every enabled backend, its
/// launch command, and why an unavailable one is skipped. The supervisor
/// replaces `{port}` with the port of the origin it uses.
pub fn shell(specs: &[Spec]) -> String {
    let mut out = String::from(
        "typeset -ga BACKEND_IDS=()\ntypeset -gA BACKEND_LABEL BACKEND_ADAPTER BACKEND_ORIGIN \
         BACKEND_DIR BACKEND_MATCH BACKEND_FILTER BACKEND_REQUIRED BACKEND_IN_ALL BACKEND_SKIP\n",
    );
    for spec in specs.iter().filter(|s| s.enabled) {
        let id = &spec.id;
        out += &format!("BACKEND_IDS+=({})\n", quote(id));
        let launch = spec.launch.as_ref();
        let dir = launch.map_or(PathBuf::new(), |l| expand(&l.dir));
        for (name, value) in [
            ("LABEL", spec.label.clone()),
            ("ADAPTER", spec.adapter.name().into()),
            ("ORIGIN", spec.origin.clone()),
            ("DIR", dir.to_string_lossy().into_owned()),
            (
                "MATCH",
                launch.map_or(String::new(), |l| l.process.to_lowercase()),
            ),
            (
                "FILTER",
                u8::from(launch.is_some_and(|l| l.terminal_filter)).to_string(),
            ),
            ("REQUIRED", u8::from(spec.required).to_string()),
            ("IN_ALL", u8::from(spec.in_all).to_string()),
            ("SKIP", spec.missing().unwrap_or_default()),
        ] {
            out += &format!("BACKEND_{name}[{id}]={}\n", quote(&value));
        }
        for (name, args) in [
            ("COMMAND", launch.map(|l| &l.command)),
            ("ALL_ARGS", launch.map(|l| &l.all_args)),
        ] {
            let args = args
                .into_iter()
                .flatten()
                .map(|a| quote(a))
                .collect::<Vec<_>>()
                .join(" ");
            out += &format!("typeset -ga BACKEND_{name}_{id}=({args})\n");
        }
    }
    out
}

/// One line per configured backend, for people.
pub fn table(specs: &[Spec]) -> String {
    specs
        .iter()
        .map(|s| {
            let state = if !s.enabled {
                "disabled".to_owned()
            } else if let Some(reason) = s.missing() {
                format!("skipped: {reason}")
            } else {
                "available".to_owned()
            };
            format!(
                "{:<14} {:<12} {:<26} {:<24} {state}\n",
                s.id,
                s.adapter.name(),
                s.label,
                s.origin
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_repository_file_is_valid_and_disables_removed_models() {
        let specs = parse(BUILTIN).unwrap();
        let orca = specs.iter().find(|s| s.id == "orca").unwrap();
        assert!(!orca.enabled);
        assert!(specs.iter().any(|s| s.id == "swift_128k" && s.enabled));
        assert!(
            specs
                .iter()
                .any(|s| s.adapter == Adapter::Ollama && !s.in_all)
        );
    }
    #[test]
    fn ids_are_checked_and_missing_paths_skip_a_backend() {
        let one = |id: &str, requires: &str| {
            format!(
                r#"{{"schema":1,"backends":[{{"id":"{id}","label":"Own","adapter":"strata","origin":"http://127.0.0.1:1","requires":["{requires}"]}}]}}"#
            )
        };
        assert!(parse(&one("all", "/")).is_err());
        assert!(parse(&one("Own-Model", "/")).is_err());
        let spec = &parse(&one("own_1", "/nonexistent/own-weights")).unwrap()[0];
        assert_eq!(
            spec.missing().as_deref(),
            Some("missing /nonexistent/own-weights")
        );
        assert!(parse(&one("own_1", "/")).unwrap()[0].missing().is_none());
        let twice = r#"{"schema":1,"backends":[
            {"id":"a","label":"A","adapter":"strata","origin":"http://127.0.0.1:1"},
            {"id":"a","label":"B","adapter":"strata","origin":"http://127.0.0.1:2"}]}"#;
        assert!(parse(twice).is_err());
    }
    #[test]
    fn shell_output_quotes_values_and_lists_enabled_backends() {
        let specs = parse(
            r#"{"schema":1,"backends":[{"id":"own","label":"Own's model","adapter":"tabbyapi",
            "origin":"http://127.0.0.1:5003","launch":{"dir":"/srv/own","command":["./start.sh","--port","{port}"],
            "match":"Start.sh","all_args":["--model-name",""]}},
            {"id":"off","label":"Off","adapter":"strata","enabled":false,"origin":"http://127.0.0.1:1"}]}"#,
        )
        .unwrap();
        let out = shell(&specs);
        assert!(out.contains("BACKEND_LABEL[own]='Own'\\''s model'"));
        assert!(out.contains("typeset -ga BACKEND_COMMAND_own=('./start.sh' '--port' '{port}')"));
        assert!(out.contains("typeset -ga BACKEND_ALL_ARGS_own=('--model-name' '')"));
        assert!(out.contains("BACKEND_MATCH[own]='start.sh'"));
        assert!(!out.contains("off"));
    }
}
