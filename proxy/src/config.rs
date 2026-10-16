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
    pub mcp_output_dir: Option<String>,
    pub mcp_input_dir: String,
    pub mcp_token: Option<String>,
    pub model_profiles: Option<String>,
    pub llm_backend: String,
    pub llm_url: String,
    pub tabby_url: String,
    pub prism_url: String,
    pub sc117_url: String,
    pub orca_url: String,
    pub sc117_128k_url: String,
    pub orca_128k_url: String,
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
            mcp_output_dir: None,
            mcp_input_dir: "/run/media/veracrypt1/mcp-inputs".into(),
            mcp_token: None,
            model_profiles: None,
            llm_backend: "koboldcpp".into(),
            llm_url: "".into(),
            tabby_url: "http://127.0.0.1:5003".into(),
            prism_url: "http://127.0.0.1:5006".into(),
            sc117_url: "http://127.0.0.1:48640".into(),
            orca_url: "http://127.0.0.1:48642".into(),
            sc117_128k_url: "http://127.0.0.1:48644".into(),
            orca_128k_url: "http://127.0.0.1:48646".into(),
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
        }
    }
}
pub const KINDS: &[&str] = &[
    "koboldcpp",
    "ollama",
    "sc117",
    "orca",
    "sc117_128k",
    "orca_128k",
    "prism",
    "tabbyapi",
    "all",
];
pub fn label(kind: &str) -> &str {
    match kind {
        "koboldcpp" => "KoboldCpp",
        "ollama" => "Ollama",
        "tabbyapi" => "TabbyAPI",
        "prism" => "Prism",
        "sc117" => "Qwen SC117",
        "orca" => "Qwen Orca",
        "sc117_128k" => "Qwen SC117 128k",
        "orca_128k" => "Qwen Orca 128k",
        "all" => "LLM backends",
        _ => kind,
    }
}
pub fn default_origin(kind: &str) -> Result<String> {
    let port = match kind {
        "koboldcpp" | "all" => 5002,
        "ollama" => 11434,
        "tabbyapi" => 5003,
        "prism" => 5006,
        "sc117" => 48640,
        "orca" => 48642,
        "sc117_128k" => 48644,
        "orca_128k" => 48646,
        _ => bail!(
            "unsupported LLM backend {kind:?}; choose one of: {}",
            KINDS.join(", ")
        ),
    };
    Ok(format!("http://127.0.0.1:{port}"))
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
            if self.mcp_workflows_dir.is_empty()
                || self.mcp_output_dir.as_deref().is_none_or(str::is_empty)
            {
                bail!("MCP requires workflow/output directories")
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
        if self.kobold_router_mode && self.llm_backend != "koboldcpp" {
            bail!("KoboldCpp Router mode requires the koboldcpp backend")
        }
        if self.llm_url.is_empty() {
            self.llm_url = default_origin(&self.llm_backend)?
        }
        self.llm_url = normalize_origin(&self.llm_url)?;
        let mut origins = std::collections::HashSet::from([self.llm_url.clone()]);
        for url in [
            &mut self.tabby_url,
            &mut self.prism_url,
            &mut self.sc117_url,
            &mut self.orca_url,
            &mut self.sc117_128k_url,
            &mut self.orca_128k_url,
        ] {
            *url = normalize_origin(url)?;
            origins.insert(url.clone());
        }
        if self.llm_backend == "all" && origins.len() != 7 {
            bail!("all backends require distinct upstream origins")
        }
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
                _ => {
                    let key = if flag == "--kobold-url" {
                        "llm_url".to_string()
                    } else {
                        flag.strip_prefix("--").unwrap_or(flag).replace('-', "_")
                    };
                    if key == "mcp_token" || v.get(&key).is_none() {
                        bail!("unrecognized arguments: {flag}")
                    }
                    if key == "llm_backend" && !KINDS.contains(&raw.as_str()) {
                        bail!(
                            "argument --llm-backend: invalid choice: {raw:?} (choose from {})",
                            KINDS.join(", ")
                        );
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
        if v["model_profiles"] == "" {
            v["model_profiles"] = Value::Null;
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
        config.validate()?;
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
