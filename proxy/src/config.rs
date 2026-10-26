use crate::registry::{self, Adapter, Spec};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Serialize, Deserialize)]
pub struct Config {
    pub coordinator_socket: Option<String>,
    pub imagine_url: String,
    pub listen_host: String,
    pub chat_port: usize,
    pub image_port: usize,
    pub mcp_port: Option<u16>,
    pub mcp_workflows_dir: String,
    /// The editable configuration root; the MCP texts are under mcp/.
    pub config: String,
    pub mcp_output_dir: Option<String>,
    pub mcp_token: Option<String>,
    pub model_profiles: Option<String>,
    /// A backend id from the backends file, or "all".
    pub llm_backend: String,
    /// The selected backend's origin; in "all" mode, KoboldCpp's.
    pub llm_url: String,
    /// The backends file; default `<config>/proxy/backends.json`.
    pub backends: Option<String>,
    /// Comma-separated backend ids that "all" leaves out.
    pub exclude_backends: String,
    pub comfy_url: String,
    pub comfy_enabled: bool,
    pub kobold_admin_password: Option<String>,
    pub kobold_router_mode: bool,
    pub kobold_model_cache: Option<String>,
    pub kobold_default_model: String,
    pub prism_model: String,
    pub tabby_model: Option<String>,
    pub tabby_max_seq_len: usize,
    pub max_chat_body_bytes: usize,
    pub connect_timeout: f64,
    pub request_timeout: f64,
    pub image_timeout: f64,
    pub chat_drain_timeout: f64,
    pub unload_timeout: f64,
    pub reload_timeout: f64,
    pub cleanup_timeout: f64,
    pub idle_timeout: f64,
    pub restore_llm_on_idle: bool,
    pub poll_interval: f64,
    pub comfy_poll_failure_limit: usize,
    pub max_workflow_body_bytes: usize,
    pub max_queued_images: usize,
    pub max_queued_workflow_bytes: usize,
    pub allow_unknown_comfy_routes: bool,
    /// The configured backends, with origin overrides applied.
    #[serde(skip)]
    pub registry: Vec<Spec>,
    /// Backend origin overrides from parse: (id, origin, given on the command line).
    #[serde(skip)]
    pub origins: Vec<(String, String, bool)>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            coordinator_socket: None,
            imagine_url: "http://127.0.0.1:8080".into(),
            listen_host: "127.0.0.1".into(),
            chat_port: 5001,
            image_port: 8188,
            mcp_port: None,
            mcp_workflows_dir: concat!(env!("CARGO_MANIFEST_DIR"), "/workflows").into(),
            config: concat!(env!("CARGO_MANIFEST_DIR"), "/../config").into(),
            mcp_output_dir: None,
            mcp_token: None,
            model_profiles: None,
            llm_backend: "koboldcpp".into(),
            llm_url: "".into(),
            backends: None,
            exclude_backends: "".into(),
            comfy_url: "http://127.0.0.1:8189".into(),
            comfy_enabled: true,
            kobold_admin_password: None,
            kobold_router_mode: false,
            kobold_model_cache: None,
            kobold_default_model: "initial_model".into(),
            prism_model: "prism-bonsai-2-27b-abliterated".into(),
            tabby_model: None,
            tabby_max_seq_len: 32768,
            max_chat_body_bytes: 33554432,
            connect_timeout: 30.0,
            request_timeout: 3900.0,
            image_timeout: 1800.0,
            chat_drain_timeout: 3900.0,
            unload_timeout: 180.0,
            reload_timeout: 600.0,
            cleanup_timeout: 60.0,
            idle_timeout: 60.0,
            restore_llm_on_idle: false,
            poll_interval: 0.5,
            comfy_poll_failure_limit: 6,
            max_workflow_body_bytes: 67108864,
            max_queued_images: 32,
            max_queued_workflow_bytes: 268435456,
            allow_unknown_comfy_routes: true,
            registry: registry::parse(registry::BUILTIN).expect("built-in backends"),
            origins: Vec::new(),
        }
    }
}
/// A backend-specific origin option, `--<id>-url` or `ST_PROXY_<ID>_URL`.
fn origin_option(key: &str) -> Option<&str> {
    let id = key.strip_suffix("_url")?;
    (!id.is_empty()
        && !matches!(id, "llm" | "kobold" | "comfy" | "imagine" | "backend")
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'))
    .then_some(id)
}
pub fn normalize_origin(s: &str) -> Result<String> {
    let mut url = reqwest::Url::parse(s).map_err(|_| {
        anyhow::anyhow!("upstream URLs must include http(s), a host, and an explicit port")
    })?;
    let authority = s
        .split_once("://")
        .map(|(_, s)| s.split(['/', '?', '#']).next().unwrap())
        .unwrap_or("");
    let explicit = authority
        .rsplit_once(':')
        .is_some_and(|(_, p)| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() || !explicit {
        bail!("upstream URLs must include http(s), a host, and an explicit port")
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("credentials are not allowed in upstream URLs")
    }
    url.set_query(None);
    url.set_fragment(None);
    // Preserve the explicit default port, which Url's canonical display omits.
    Ok(format!(
        "{}://{}{}",
        url.scheme(),
        authority,
        url.path().trim_end_matches('/')
    ))
}
fn boolean(name: &str, s: &str) -> Result<bool> {
    match s.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("ST_PROXY_{name} must be a boolean"),
    }
}
#[derive(Default)]
pub struct Cli {
    pub check_backend: bool,
    pub backend_check_timeout: f64,
    pub write_kobold_model_cache: Option<String>,
    pub log_level: String,
    pub help: bool,
    /// List the backends ("table" or "shell") instead of starting.
    pub print_backends: Option<String>,
}
impl Config {
    pub fn resolve_option(flag: &str) -> Result<String> {
        let mut options: Vec<String> = serde_json::to_value(Self::default())
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| k.as_str() != "mcp_token")
            .map(|k| format!("--{}", k.replace('_', "-")))
            .collect();
        options.extend(
            [
                "--help",
                "--kobold-url",
                "--check-backend",
                "--strict-comfy-routes",
                "--log-level",
                "--backend-check-timeout",
                "--write-kobold-model-cache",
                "--backend-url",
                "--print-backends",
            ]
            .into_iter()
            .map(str::to_string),
        );
        if flag == "-h" {
            return Ok("--help".into());
        }
        if options.iter().any(|s| s == flag) {
            return Ok(flag.into());
        }
        // Backend ids come from the backends file, so their options are exact.
        if flag
            .strip_prefix("--")
            .is_some_and(|f| origin_option(&f.replace('-', "_")).is_some())
        {
            return Ok(flag.into());
        }
        let matches: Vec<_> = options.iter().filter(|s| s.starts_with(flag)).collect();
        match matches.as_slice() {
            [only] => Ok((*only).clone()),
            [] => bail!("unrecognized arguments: {flag}"),
            _ => bail!(
                "ambiguous option: {flag} could match {}",
                matches
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
    pub fn validate(&mut self) -> Result<()> {
        if let Some(port) = self.mcp_port {
            let _ = port;
            if self.mcp_workflows_dir.is_empty() {
                bail!("MCP requires workflow directory")
            }
        }
        if self.kobold_default_model.trim().is_empty() {
            bail!("kobold_default_model cannot be empty")
        }
        if self.prism_model.trim().is_empty() {
            bail!("prism_model cannot be empty")
        }
        if self.llm_backend.is_empty() {
            bail!("llm_backend cannot be empty")
        }
        for spec in &mut self.registry {
            spec.origin = normalize_origin(&spec.origin)?;
        }
        for id in self.excluded() {
            if !self.registry.iter().any(|s| s.id == id) {
                bail!("unknown backend {id:?} in exclude_backends")
            }
        }
        self.check_selection()?;
        self.comfy_url = normalize_origin(&self.comfy_url)?;
        self.imagine_url = normalize_origin(&self.imagine_url)?;
        if self.chat_port > 65535 || self.image_port > 65535 {
            bail!("listen ports must be between 0 and 65535")
        }
        let val = serde_json::to_value(&*self)?;
        for key in [
            "connect_timeout",
            "request_timeout",
            "image_timeout",
            "chat_drain_timeout",
            "unload_timeout",
            "reload_timeout",
            "cleanup_timeout",
            "idle_timeout",
            "poll_interval",
            "tabby_max_seq_len",
            "max_chat_body_bytes",
            "comfy_poll_failure_limit",
            "max_workflow_body_bytes",
            "max_queued_images",
            "max_queued_workflow_bytes",
        ] {
            if val[key].as_f64().unwrap_or(0.0) <= 0.0 {
                bail!("{key} must be greater than zero")
            }
        }
        if self.max_queued_workflow_bytes < self.max_workflow_body_bytes {
            bail!("max_queued_workflow_bytes must be at least max_workflow_body_bytes")
        }
        Ok(())
    }
    /// Loads the backends file and applies origin overrides.
    fn resolve_backends(&mut self) -> Result<()> {
        self.registry = registry::load(self.backends.as_deref(), &self.config)?;
        for (id, origin, explicit) in std::mem::take(&mut self.origins) {
            match self.registry.iter_mut().find(|s| s.id == id) {
                Some(spec) => spec.origin = origin,
                None if explicit => {
                    bail!("unknown backend {id:?} in --{}-url", id.replace('_', "-"))
                }
                // A stale environment variable for a removed backend is harmless.
                None => {}
            }
        }
        Ok(())
    }
    /// Checks that the selected backend, or every backend of "all", can run.
    fn check_selection(&mut self) -> Result<()> {
        if self.all() {
            if self.kobold_router_mode {
                bail!("KoboldCpp Router mode requires the koboldcpp backend")
            }
            if !self.llm_url.is_empty() {
                // The historical KoboldCpp option also applies to "all".
                self.llm_url = normalize_origin(&self.llm_url)?;
                if let Some(spec) = self
                    .registry
                    .iter_mut()
                    .find(|s| s.adapter == Adapter::Koboldcpp)
                {
                    spec.origin = self.llm_url.clone();
                }
            }
            let active = self.active();
            if active.is_empty() {
                bail!("no LLM backend is available; check the backends file")
            }
            let origins: std::collections::HashSet<_> = active.iter().map(|s| &s.origin).collect();
            if origins.len() != active.len() {
                bail!("all backends require distinct upstream origins")
            }
            if active
                .iter()
                .filter(|s| s.adapter == Adapter::Koboldcpp)
                .count()
                > 1
            {
                bail!("all backends allow one KoboldCpp backend")
            }
            return Ok(());
        }
        let choices = || {
            std::iter::once("all")
                .chain(
                    self.registry
                        .iter()
                        .filter(|s| s.enabled)
                        .map(|s| s.id.as_str()),
                )
                .collect::<Vec<_>>()
                .join(", ")
        };
        let Some(spec) = self
            .registry
            .iter()
            .find(|s| s.id == self.llm_backend && s.enabled)
        else {
            bail!(
                "unsupported LLM backend {:?}; choose one of: {}",
                self.llm_backend,
                choices()
            )
        };
        if self.excluded().contains(&spec.id.as_str()) {
            bail!("{} is excluded", spec.label)
        }
        if let Some(reason) = spec.missing() {
            bail!("{} cannot start: {reason}", spec.label)
        }
        if self.kobold_router_mode && spec.adapter != Adapter::Koboldcpp {
            bail!("KoboldCpp Router mode requires the koboldcpp backend")
        }
        if self.llm_url.is_empty() {
            self.llm_url = spec.origin.clone();
        }
        self.llm_url = normalize_origin(&self.llm_url)?;
        Ok(())
    }
    pub fn all(&self) -> bool {
        self.llm_backend == "all"
    }
    fn excluded(&self) -> Vec<&str> {
        self.exclude_backends
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    }
    /// The backends this proxy serves: every available one of "all", or the
    /// selected one at `llm_url`.
    pub fn active(&self) -> Vec<Spec> {
        if self.all() {
            let excluded = self.excluded();
            return self
                .registry
                .iter()
                .filter(|s| {
                    s.enabled
                        && s.in_all
                        && !excluded.contains(&s.id.as_str())
                        && s.missing().is_none()
                })
                .cloned()
                .collect();
        }
        self.spec(&self.llm_backend)
            .map(|s| {
                let mut s = s.clone();
                if !self.llm_url.is_empty() {
                    s.origin = self.llm_url.clone();
                }
                s
            })
            .into_iter()
            .collect()
    }
    /// Backends "all" leaves out, with the reason.
    pub fn skipped(&self) -> Vec<(String, String)> {
        if !self.all() {
            return Vec::new();
        }
        let excluded = self.excluded();
        self.registry
            .iter()
            .filter(|s| s.enabled && s.in_all)
            .filter_map(|s| {
                if excluded.contains(&s.id.as_str()) {
                    Some((s.label.clone(), "excluded".into()))
                } else {
                    s.missing().map(|r| (s.label.clone(), r))
                }
            })
            .collect()
    }
    pub fn spec(&self, id: &str) -> Option<&Spec> {
        self.registry.iter().find(|s| s.id == id)
    }
    /// The selected backend's adapter; None for "all".
    pub fn adapter(&self) -> Option<Adapter> {
        if self.all() {
            return None;
        }
        self.spec(&self.llm_backend).map(|s| s.adapter)
    }
    pub fn label(&self) -> String {
        if self.all() {
            return "LLM backends".into();
        }
        self.spec(&self.llm_backend)
            .map_or_else(|| self.llm_backend.clone(), |s| s.label.clone())
    }
    pub fn parse(
        args: impl IntoIterator<Item = String>,
        env: &std::collections::HashMap<String, String>,
    ) -> Result<(Self, Cli)> {
        let mut v = serde_json::to_value(Self::default())?;
        let mut cli = Cli {
            backend_check_timeout: 2.0,
            log_level: "INFO".into(),
            ..Cli::default()
        };
        if env
            .get("ST_PROXY_TEST_MODE")
            .is_some_and(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        {
            bail!("test mode is available only to the in-process test harness")
        }
        for (key, val) in v.as_object_mut().unwrap() {
            let ename = key.to_uppercase();
            if let Some(raw) = env.get(&format!("ST_PROXY_{ename}")) {
                *val = parse_value(key, raw, val)?;
            }
        }
        let mut origins = Vec::new();
        for (name, value) in env {
            if let Some(key) = name.strip_prefix("ST_PROXY_").map(str::to_lowercase)
                && v.get(&key).is_none()
                && let Some(id) = origin_option(&key)
                && !value.is_empty()
            {
                origins.push((id.to_owned(), value.clone(), false));
            }
        }
        if v["llm_url"].as_str().is_none_or(str::is_empty)
            && let Some(x) = env.get("ST_PROXY_KOBOLD_URL")
        {
            v["llm_url"] = x.clone().into();
        }
        if let Some(raw) = env.get("ST_PROXY_STRICT_COMFY_ROUTES") {
            v["allow_unknown_comfy_routes"] = (!boolean("STRICT_COMFY_ROUTES", raw)?).into();
        }
        if let Some(raw) = env.get("ST_PROXY_BACKEND_CHECK_TIMEOUT") {
            cli.backend_check_timeout = raw.parse()?;
        }
        if let Some(raw) = env.get("ST_PROXY_LOG_LEVEL") {
            cli.log_level = raw.clone();
        }
        let mut args = args.into_iter().peekable();
        let mut strict = false;
        let mut transparent = false;
        while let Some(arg) = args.next() {
            let (flag, inline) = arg
                .split_once('=')
                .map_or((arg.as_str(), None), |(a, b)| (a, Some(b.to_owned())));
            let resolved = Self::resolve_option(flag)?;
            let flag = resolved.as_str();
            if let Some(value) = &inline
                && matches!(
                    flag,
                    "--help"
                        | "-h"
                        | "--check-backend"
                        | "--strict-comfy-routes"
                        | "--allow-unknown-comfy-routes"
                        | "--kobold-router-mode"
                        | "--restore-llm-on-idle"
                )
            {
                bail!("argument {flag}: ignored explicit argument {:?}", value);
            }
            match flag {
                "--help" | "-h" => {
                    cli.help = true;
                    return Ok((Self::default(), cli));
                }
                "--check-backend" => {
                    cli.check_backend = true;
                    continue;
                }
                _ if flag.starts_with("--")
                    && origin_option(&flag[2..].replace('-', "_")).is_some()
                    && v.get(flag[2..].replace('-', "_")).is_none() =>
                {
                    let id = flag[2..].strip_suffix("-url").unwrap().replace('-', "_");
                    let raw = inline
                        .clone()
                        .or_else(|| args.next())
                        .ok_or_else(|| anyhow::anyhow!("argument {flag}: expected one argument"))?;
                    origins.push((id, raw, true));
                    continue;
                }
                "--strict-comfy-routes" => {
                    strict = true;
                    v["allow_unknown_comfy_routes"] = false.into();
                    continue;
                }
                "--allow-unknown-comfy-routes" => {
                    transparent = true;
                    v["allow_unknown_comfy_routes"] = true.into();
                    continue;
                }
                "--kobold-router-mode" | "--restore-llm-on-idle" => {
                    v[flag[2..].replace('-', "_")] = true.into();
                    continue;
                }
                _ => {}
            }
            if inline.is_none()
                && args
                    .peek()
                    .is_some_and(|s| s.starts_with("--") || s == "-h")
            {
                bail!("argument {flag}: expected one argument")
            }
            let raw = inline
                .or_else(|| args.next())
                .ok_or_else(|| anyhow::anyhow!("argument {flag}: expected one argument"))?;
            match flag {
                "--log-level" => cli.log_level = raw,
                "--backend-check-timeout" => cli.backend_check_timeout = raw.parse()?,
                "--write-kobold-model-cache" => cli.write_kobold_model_cache = Some(raw),
                "--print-backends" => {
                    if !matches!(raw.as_str(), "table" | "shell") {
                        bail!("argument --print-backends: choose table or shell")
                    }
                    cli.print_backends = Some(raw)
                }
                "--backend-url" => {
                    let (id, origin) = raw.split_once('=').ok_or_else(|| {
                        anyhow::anyhow!("argument --backend-url: expected ID=URL")
                    })?;
                    origins.push((id.trim().to_owned(), origin.trim().to_owned(), true));
                }
                _ => {
                    let key = if flag == "--kobold-url" {
                        "llm_url".to_string()
                    } else {
                        flag.strip_prefix("--").unwrap_or(flag).replace('-', "_")
                    };
                    if key == "mcp_token" || v.get(&key).is_none() {
                        bail!("unrecognized arguments: {flag}")
                    }
                    v[&key] = parse_value(&key, &raw, &v[&key])?;
                }
            }
        }
        if strict && transparent {
            bail!(
                "argument --allow-unknown-comfy-routes: not allowed with argument --strict-comfy-routes"
            )
        }
        if cli.write_kobold_model_cache.is_some() && !cli.check_backend {
            bail!("--write-kobold-model-cache requires --check-backend")
        }
        for key in ["model_profiles", "backends"] {
            if v[key] == "" {
                v[key] = Value::Null;
            }
        }
        for key in ["chat_port", "image_port", "mcp_port"] {
            if v[key].as_i64().is_some_and(|n| !(0..=65535).contains(&n)) {
                bail!(
                    "{}",
                    if key == "mcp_port" {
                        "mcp_port must be between 0 and 65535"
                    } else {
                        "listen ports must be between 0 and 65535"
                    }
                )
            }
        }
        for key in [
            "tabby_max_seq_len",
            "max_chat_body_bytes",
            "comfy_poll_failure_limit",
            "max_workflow_body_bytes",
            "max_queued_images",
            "max_queued_workflow_bytes",
        ] {
            if v[key].as_i64().is_some_and(|v| v <= 0) {
                bail!("{key} must be greater than zero")
            }
        }
        let mut config: Self = serde_json::from_value(v)?;
        config.origins = origins;
        config.resolve_backends()?;
        if cli.print_backends.is_some() {
            // Listing works even when the selected backend cannot start.
            for spec in &mut config.registry {
                spec.origin = normalize_origin(&spec.origin)?;
            }
        } else {
            config.validate()?;
        }
        Ok((config, cli))
    }
}
fn parse_value(key: &str, raw: &str, default: &Value) -> Result<Value> {
    if default.is_boolean() {
        return Ok(boolean(&key.to_uppercase(), raw)?.into());
    }
    if default.is_number() || key == "mcp_port" {
        return if key.ends_with("timeout") || key == "poll_interval" {
            Ok(serde_json::json!(raw.parse::<f64>()?))
        } else {
            Ok(serde_json::json!(raw.parse::<i64>()?))
        };
    }
    Ok(raw.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    /// A backends file with three Strata backends; `orca` needs a missing file.
    fn env(extra: &[(&str, &str)]) -> HashMap<String, String> {
        // Each call has its own file: tests run in parallel.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("st-config-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("backends-{n}.json"));
        std::fs::write(
            &path,
            r#"{"schema":1,"backends":[
            {"id":"sc117","label":"Qwen SC117","adapter":"strata","origin":"http://127.0.0.1:48640"},
            {"id":"orca","label":"Qwen Orca","adapter":"strata","origin":"http://127.0.0.1:48642",
             "requires":["/nonexistent/own-orca-weights"]},
            {"id":"swift","label":"Qwen SC117 Swift","adapter":"strata","origin":"http://127.0.0.1:48648"},
            {"id":"old","label":"Old","adapter":"strata","enabled":false,"origin":"http://127.0.0.1:48600"}]}"#,
        )
        .unwrap();
        let mut env = HashMap::from([(
            "ST_PROXY_BACKENDS".to_owned(),
            path.to_string_lossy().into_owned(),
        )]);
        env.extend(extra.iter().map(|(k, v)| (k.to_string(), v.to_string())));
        env
    }
    fn parse(args: &[&str], env: &HashMap<String, String>) -> Result<Config> {
        Config::parse(args.iter().map(|s| s.to_string()), env).map(|(c, _)| c)
    }
    fn ids(config: &Config) -> Vec<String> {
        config.active().into_iter().map(|s| s.id).collect()
    }
    #[test]
    fn all_skips_unavailable_and_excluded_backends() {
        let config = parse(&["--llm-backend", "all"], &env(&[])).unwrap();
        assert_eq!(ids(&config), ["sc117", "swift"]);
        assert_eq!(
            config.skipped(),
            [(
                "Qwen Orca".to_owned(),
                "missing /nonexistent/own-orca-weights".to_owned()
            )]
        );
        let config = parse(
            &["--llm-backend", "all", "--exclude-backends", "swift"],
            &env(&[]),
        )
        .unwrap();
        assert_eq!(ids(&config), ["sc117"]);
        assert!(
            parse(
                &["--llm-backend", "all", "--exclude-backends", "sc117,swift"],
                &env(&[])
            )
            .is_err()
        );
        assert!(
            parse(
                &["--llm-backend", "all", "--exclude-backends", "nonexistent"],
                &env(&[])
            )
            .is_err()
        );
    }
    #[test]
    fn one_backend_must_be_enabled_and_able_to_start() {
        let config = parse(&["--llm-backend", "swift"], &env(&[])).unwrap();
        assert_eq!(config.llm_url, "http://127.0.0.1:48648");
        assert_eq!(config.adapter(), Some(Adapter::Strata));
        assert_eq!(config.label(), "Qwen SC117 Swift");
        let error = parse(&["--llm-backend", "orca"], &env(&[]))
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("missing /nonexistent/own-orca-weights"),
            "{error}"
        );
        assert!(parse(&["--llm-backend", "old"], &env(&[])).is_err());
        assert!(parse(&["--llm-backend", "nonexistent"], &env(&[])).is_err());
    }
    #[test]
    fn origins_follow_options_and_stale_variables_are_ignored() {
        let env = env(&[
            ("ST_PROXY_SWIFT_URL", "http://127.0.0.1:12001"),
            ("ST_PROXY_REMOVED_URL", "http://127.0.0.1:12002"),
        ]);
        let config = parse(
            &[
                "--llm-backend",
                "all",
                "--backend-url",
                "sc117=http://127.0.0.1:12003/",
            ],
            &env,
        )
        .unwrap();
        assert_eq!(
            config.spec("swift").unwrap().origin,
            "http://127.0.0.1:12001"
        );
        assert_eq!(
            config.spec("sc117").unwrap().origin,
            "http://127.0.0.1:12003"
        );
        let config = parse(
            &[
                "--llm-backend",
                "swift",
                "--swift-url",
                "http://127.0.0.1:12004",
            ],
            &env,
        )
        .unwrap();
        assert_eq!(config.llm_url, "http://127.0.0.1:12004");
        assert!(parse(&["--removed-url", "http://127.0.0.1:1"], &env).is_err());
        assert!(parse(&["--backend-url", "sc117"], &env).is_err());
        // Listing works even though the selected backend cannot start.
        let (config, cli) = Config::parse(
            ["--llm-backend", "orca", "--print-backends", "table"].map(str::to_owned),
            &env,
        )
        .unwrap();
        assert_eq!(cli.print_backends.as_deref(), Some("table"));
        assert!(
            registry::table(&config.registry)
                .contains("skipped: missing /nonexistent/own-orca-weights")
        );
    }
}
