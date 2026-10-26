use crate::{
    config::{Config, label},
    logging::log,
};
use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::time::{sleep, timeout};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Target {
    pub engine: String,
    pub model: String,
    #[serde(default)]
    pub parameters: Map<String, Value>,
}
impl Target {
    pub fn new(engine: &str, model: &str) -> Self {
        Self {
            engine: engine.into(),
            model: model.into(),
            parameters: Map::new(),
        }
    }
}
#[derive(Clone)]
pub struct Backend {
    pub kind: String,
    pub origin: String,
    pub config: Config,
    pub client: reqwest::Client,
    pub limit: crate::connection_limit::Limit,
    uncertain: bool,
    pending: Option<Target>,
    load_failed: bool,
}
fn seconds(n: f64) -> Duration {
    Duration::from_secs_f64(n)
}
impl Backend {
    pub fn new(kind: &str, origin: &str, config: &Config, client: reqwest::Client) -> Self {
        Self {
            kind: kind.into(),
            origin: origin.into(),
            config: config.clone(),
            client,
            limit: crate::connection_limit::Limit::new(16, config.connect_timeout),
            uncertain: false,
            pending: None,
            load_failed: false,
        }
    }
    pub fn label(&self) -> &str {
        label(&self.kind)
    }
    pub fn strata(&self) -> bool {
        matches!(
            self.kind.as_str(),
            "sc117" | "orca" | "sc117_128k" | "orca_128k" | "swift" | "swift_128k"
        )
    }
    fn err(&self, action: &str, detail: &str) -> anyhow::Error {
        anyhow!("{} {action} failed: {detail}", self.label())
    }
    pub async fn json(&self, path: &str, payload: Option<Value>) -> Result<Value> {
        let _permit = self
            .limit
            .acquire()
            .await
            .map_err(|_| self.err(path, "TimeoutError"))?;
        let call = if let Some(payload) = payload {
            self.client
                .post(format!("{}{path}", self.origin))
                .json(&payload)
        } else {
            self.client.get(format!("{}{path}", self.origin))
        };
        let response = call
            .timeout(seconds(self.config.request_timeout))
            .send()
            .await
            .map_err(|e| {
                self.err(
                    path,
                    if e.is_timeout() {
                        "TimeoutError"
                    } else {
                        "ClientConnectorError"
                    },
                )
            })?;
        let status = response.status().as_u16();
        if if ["koboldcpp", "ollama"].contains(&self.kind.as_str()) {
            status >= 400
        } else {
            status != 200
        } {
            bail!("{} {path} failed: HTTP {status}", self.label())
        }
        response
            .json()
            .await
            .map_err(|_| self.err(path, "ValueError"))
    }
    fn certain(&self) -> Result<()> {
        if self.uncertain {
            return Err(self.err(
                "lifecycle confirmation",
                if self.strata() {
                    "previous load/unload did not complete; resolve backend state"
                } else {
                    "previous load/unload did not complete; resolve backend state before restart"
                },
            ));
        }
        Ok(())
    }
    async fn health(&self) -> Result<Value> {
        let v = self.json("/health", None).await?;
        if v["service"] != "strata"
            || v["status"] != "ok"
            || !v["loaded"].is_boolean()
            || v["model"].as_str().is_none_or(str::is_empty)
        {
            return Err(self.err("model observation", "invalid Strata health"));
        }
        Ok(v)
    }
    async fn states(&self) -> Result<Vec<(String, Value)>> {
        let v = self.json("/models", None).await?;
        let cards = v["data"]
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| self.err("model observation", "expected router presets"))?;
        let mut states = Vec::new();
        for card in cards {
            let id = card["id"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| self.err("model observation", "invalid preset ID"))?;
            if states.iter().any(|(name, _)| name == id) {
                return Err(self.err("model observation", "invalid preset ID"));
            }
            if !matches!(
                card["status"]["value"].as_str(),
                Some("loaded" | "loading" | "unloaded" | "sleeping")
            ) {
                return Err(self.err("model observation", "invalid router state"));
            }
            states.push((id.into(), card["status"].clone()));
        }
        if !states
            .iter()
            .any(|(name, _)| name == &self.config.prism_model)
        {
            return Err(self.err("model observation", "default preset missing"));
        }
        if states
            .iter()
            .filter(|(_, s)| s["value"] != "unloaded")
            .count()
            > 1
        {
            return Err(self.err("model observation", "multiple active workers"));
        }
        Ok(states)
    }
    async fn kobold_name(&self) -> Result<String> {
        let v = self.json("/api/v1/model", None).await?;
        let val = if v["result"].is_array() {
            &v["result"][0]
        } else {
            &v["result"]
        };
        if val.is_null() {
            return Err(self.err("model-state observation", "invalid API response"));
        }
        Ok(val
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| val.to_string())
            .trim()
            .into())
    }
    async fn version_ready(&self) -> bool {
        let Ok(_permit) = self.limit.acquire().await else {
            return false;
        };
        self.client
            .get(format!("{}/api/v1/info/version", self.origin))
            .send()
            .await
            .is_ok_and(|r| r.status().as_u16() < 400)
    }
    async fn running(&self) -> Result<Vec<String>> {
        let v = self.json("/api/ps", None).await?;
        v["models"]
            .as_array()
            .ok_or_else(|| self.err("running-model check", "invalid API response"))?
            .iter()
            .map(|v| {
                v.get("model")
                    .filter(|v| !v.is_null())
                    .or(v.get("name"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
                    .map(str::to_string)
                    .ok_or_else(|| self.err("running-model check", "model identity missing"))
            })
            .collect()
    }
    pub async fn validate(&self) -> Result<()> {
        match self.kind.as_str() {
            "koboldcpp" => {
                let cap = self.json("/api/extra/version", None).await?;
                if (self.config.kobold_router_mode || self.config.llm_backend == "all")
                    && cap["router"] != true
                {
                    return Err(self.err(
                        "router readiness check",
                        "restart KoboldCpp with --admin, --admindir and --routermode",
                    ));
                }
                let guidance = "enable Model Administration, configure its required directory, and verify the admin password";
                if cap["admin"].as_i64().unwrap_or(0) < 1 {
                    return Err(self.err(
                        "model administration check",
                        &format!("unavailable; {guidance}"),
                    ));
                }
                let options = self
                    .admin_request("/api/admin/list_options", None, self.config.request_timeout)
                    .await?;
                if !["initial_model", "unload_model"]
                    .iter()
                    .all(|s| options.as_array().is_some_and(|v| v.iter().any(|v| v == s)))
                {
                    return Err(self.err(
                        "admin model options check",
                        &format!("required options missing; {guidance}"),
                    ));
                }
                log(
                    20,
                    "st_proxy.llm.koboldcpp",
                    format!(
                        "KoboldCpp model administration confirmed: admin_level={}",
                        cap["admin"]
                    ),
                );
            }
            "tabbyapi" => {
                let h = self.json("/health", None).await?;
                let m = self.json("/v1/models", None).await?;
                if h["status"] != "healthy" || !m["data"].is_array() {
                    return Err(self.err("API check", "ValueError"));
                }
            }
            "ollama" => {
                let v = self.json("/api/version", None).await?;
                if v["version"].as_str().is_none_or(|v| v.trim().is_empty()) {
                    return Err(self.err("readiness check", "invalid version response"));
                }
                self.running().await?;
                log(20, "st_proxy.llm.ollama", "Ollama lifecycle API confirmed");
            }
            "prism" => {
                self.certain()?;
                self.states().await?;
            }
            _ => {
                self.certain()?;
                self.health().await?;
                self.catalog().await?;
            }
        }
        Ok(())
    }
    pub async fn catalog(&self) -> Result<Value> {
        if self.kind == "prism" {
            return Ok(
                json!({"object":"list","data":self.states().await?.iter().map(|(m,_)|json!({"id":m,"object":"model"})).collect::<Vec<_>>()}),
            );
        }
        let v = self
            .json(
                if self.kind == "tabbyapi" {
                    "/v1/model/list"
                } else {
                    "/v1/models"
                },
                None,
            )
            .await?;
        if self.kind == "koboldcpp" {
            return kobold_models(&v);
        }
        if v["data"].as_array().is_none_or(|cards| {
            (self.strata() && cards.is_empty())
                || cards
                    .iter()
                    .any(|v| v["id"].as_str().is_none_or(str::is_empty))
        }) {
            return Err(self.err("model catalog", "expected native model IDs"));
        }
        Ok(v)
    }
    pub async fn target(&self, model: Option<&str>) -> Result<Target> {
        if self.strata() {
            let h = self.health().await?;
            let m = model.unwrap_or(h["model"].as_str().unwrap());
            if !self.catalog().await?["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v["id"] == m)
            {
                bail!("unknown Strata model")
            };
            return Ok(Target::new(&self.kind, h["model"].as_str().unwrap()));
        }
        if self.kind == "prism" {
            let m = model.unwrap_or(&self.config.prism_model);
            if m.trim().is_empty() {
                bail!("model must be a nonempty Prism preset ID")
            };
            if !self.states().await?.iter().any(|(id, _)| id == m) {
                bail!("unknown Prism model")
            };
            return Ok(Target::new(&self.kind, m));
        }
        Ok(Target::new(&self.kind, model.unwrap_or("")))
    }
    async fn tabby_observe(&self) -> Result<Option<Target>> {
        let _permit = self.limit.acquire().await?;
        let response = self
            .client
            .get(format!("{}/v1/model", self.origin))
            .timeout(seconds(self.config.request_timeout))
            .send()
            .await?;
        let status = response.status().as_u16();
        let v: Value = response.json().await?;
        if status == 503 && v["detail"] == "No models are currently loaded." {
            return Ok(None);
        }
        if status != 200 {
            return Err(self.err("/v1/model", &format!("HTTP {status}")));
        }
        let id = v["id"]
            .as_str()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| self.err("model observation", "ValueError"))?;
        let params = v["parameters"]
            .as_object()
            .ok_or_else(|| self.err("model observation", "ValueError"))?;
        if params
            .get("max_seq_len")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            return Err(self.err("model observation", "ValueError"));
        }
        let mut t = Target::new(&self.kind, id);
        for k in [
            "max_seq_len",
            "cache_size",
            "cache_mode",
            "chunk_size",
            "use_vision",
        ] {
            if let Some(v) = params.get(k).filter(|v| !v.is_null()) {
                t.parameters.insert(
                    if k == "use_vision" { "vision" } else { k }.into(),
                    v.clone(),
                );
            }
        }
        Ok(Some(t))
    }
    async fn tabby_confirm(&mut self, retry: bool) -> Result<Option<Target>> {
        let current = self.tabby_observe().await?;
        if self.uncertain {
            let confirmed = match &current {
                None => self.pending.is_none() || retry && self.load_failed,
                Some(c) => self.pending.as_ref().is_some_and(|p| {
                    p.model == c.model
                        && p.parameters
                            .iter()
                            .all(|(k, v)| c.parameters.get(k) == Some(v))
                }),
            };
            if !confirmed {
                return Err(self.err(
                    "lifecycle confirmation",
                    "load/unload is still unconfirmed; status will be checked on the next request",
                ));
            }
            self.uncertain = false;
            self.pending = None;
            self.load_failed = false;
        }
        Ok(current)
    }
    pub async fn observe(&mut self) -> Result<Option<Target>> {
        match self.kind.as_str() {
            "tabbyapi" => self.tabby_confirm(false).await,
            "koboldcpp" => {
                let m = self.kobold_name().await?;
                if !self.version_ready().await {
                    return Err(self.err("ready-state observation", "unavailable"));
                }
                Ok(if inactive(&m) {
                    None
                } else {
                    Some(Target::new(&self.kind, &m))
                })
            }
            "ollama" => {
                let m = self.running().await?;
                if m.len() > 1 {
                    return Err(self.err(
                        "ready-state observation",
                        "exactly one loaded model is required",
                    ));
                }
                Ok(m.first().map(|m| Target::new(&self.kind, m)))
            }
            "prism" => {
                self.certain()?;
                for (m, s) in self.states().await? {
                    if s["value"] != "unloaded" {
                        if s["value"] != "loaded" {
                            return Err(
                                self.err("model observation", "model not in a stable state")
                            );
                        }
                        return Ok(Some(Target::new(&self.kind, &m)));
                    }
                }
                Ok(None)
            }
            _ => {
                self.certain()?;
                let h = self.health().await?;
                Ok(if h["loaded"] == true {
                    Some(Target::new(&self.kind, h["model"].as_str().unwrap()))
                } else {
                    None
                })
            }
        }
    }
    pub async fn snapshot(&mut self) -> Result<Target> {
        let wait = matches!(self.kind.as_str(), "tabbyapi" | "koboldcpp" | "prism");
        let duration = self.config.reload_timeout;
        timeout(seconds(duration), async {
            loop {
                if let Some(t) = self.observe().await? {
                    return Ok(t);
                }
                if !wait {
                    return Err(self.err("ready-state snapshot", "no model is loaded"));
                }
                sleep(seconds(self.config.poll_interval)).await;
            }
        })
        .await
        .map_err(|_| self.err("readiness", "TimeoutError"))?
    }
    async fn admin_request(&self, path: &str, body: Option<Value>, duration: f64) -> Result<Value> {
        let action = body
            .as_ref()
            .and_then(|v| v["filename"].as_str())
            .unwrap_or("admin options check")
            .to_string();
        let _permit = self
            .limit
            .acquire()
            .await
            .map_err(|_| self.err(&action, "TimeoutError"))?;
        let call = if let Some(v) = body {
            self.client.post(format!("{}{path}", self.origin)).json(&v)
        } else {
            self.client.get(format!("{}{path}", self.origin))
        };
        let call = call;
        let call = if let Some(p) = &self.config.kobold_admin_password {
            call.bearer_auth(p)
        } else {
            call
        };
        let r = call.timeout(seconds(duration)).send().await.map_err(|e| {
            self.err(
                &action,
                if e.is_timeout() {
                    "TimeoutError"
                } else {
                    "ClientConnectorError"
                },
            )
        })?;
        let s = r.status().as_u16();
        if s >= 400 {
            return Err(self.err(&action, &format!("HTTP {s}")));
        }
        r.json()
            .await
            .map_err(|_| self.err(&action, "invalid admin response"))
    }
    async fn kobold_reload(&self, filename: &str, duration: f64) -> Result<()> {
        log(
            20,
            "st_proxy.llm.koboldcpp",
            format!("KoboldCpp admin operation started: config={filename}"),
        );
        let v = self
            .admin_request(
                "/api/admin/reload_config",
                Some(json!({"filename":filename})),
                duration,
            )
            .await?;
        if v["success"] != true {
            return Err(self.err(filename, "admin request rejected"));
        }
        log(
            20,
            "st_proxy.llm.koboldcpp",
            format!("KoboldCpp admin operation completed: config={filename} status=200"),
        );
        Ok(())
    }
    async fn kobold_wait(
        &self,
        loaded: bool,
        expected: Option<&str>,
        duration: f64,
    ) -> Result<String> {
        let start = Instant::now();
        let state = if loaded { "loaded" } else { "unloaded" };
        log(
            20,
            "st_proxy.llm.koboldcpp",
            format!("waiting for KoboldCpp model state: expected={state} timeout={duration:.1}s"),
        );
        timeout(seconds(duration), async {
            loop {
                let m = self.kobold_name().await.ok();
                let version = self.version_ready().await;
                if version
                    && m.as_ref().is_some_and(|m| !inactive(m)) == loaded
                    && (!loaded || expected.is_none() || m.as_deref() == expected)
                {
                    log(
                        20,
                        "st_proxy.llm.koboldcpp",
                        format!(
                            "KoboldCpp model state confirmed: state={state} duration={:.3}s",
                            start.elapsed().as_secs_f64()
                        ),
                    );
                    return Ok(m.unwrap_or_default());
                }
                sleep(seconds(self.config.poll_interval)).await;
            }
        })
        .await
        .map_err(|_| self.err(&format!("confirmation of {state}"), "timed out"))?
    }
    async fn change(&mut self, t: &Target, loaded: bool) -> Result<()> {
        self.certain()?;
        self.uncertain = true;
        let action = if loaded { "load" } else { "unload" };
        let duration = if loaded {
            self.config.reload_timeout
        } else {
            self.config.unload_timeout
        };
        timeout(seconds(duration), async {
            let v = self
                .json(
                    &format!("/{}/{action}", if self.strata() { "v1" } else { "models" }),
                    Some(json!({"model":t.model})),
                )
                .await?;
            if self.strata() {
                if (loaded && v["status"] != "loaded")
                    || (!loaded && v["status"] != "unloaded" && v["status"] != "not loaded")
                {
                    return Err(self.err(action, "operation not confirmed"));
                }
                let h = self.health().await?;
                if h["loaded"] != loaded || h["model"] != t.model {
                    return Err(self.err(action, "state not confirmed"));
                }
            } else {
                if v["success"] != true {
                    return Err(self.err(action, "operation not accepted"));
                }
                loop {
                    let states = self.states().await?;
                    let s = states
                        .iter()
                        .find(|(m, _)| m == &t.model)
                        .ok_or_else(|| self.err("confirmation", "preset disappeared"))?;
                    if s.1["value"] == if loaded { "loaded" } else { "unloaded" } {
                        if !loaded && states.iter().any(|(_, s)| s["value"] != "unloaded") {
                            return Err(self.err("unload", "another worker is active"));
                        }
                        break;
                    }
                    if loaded && s.1["failed"] == true {
                        return Err(self.err("model load", "worker failed"));
                    }
                    sleep(seconds(self.config.poll_interval)).await;
                }
            }
            Ok::<_, anyhow::Error>(())
        })
        .await
        .map_err(|_| self.err(action, "timed out"))??;
        self.uncertain = false;
        Ok(())
    }
    async fn ollama_change(&self, t: &Target, loaded: bool) -> Result<()> {
        let duration = if loaded {
            self.config.reload_timeout
        } else {
            self.config.unload_timeout
        };
        let state = if loaded { "loaded" } else { "unloaded" };
        log(
            20,
            "st_proxy.llm.ollama",
            format!(
                "Ollama GPU {} requested",
                if loaded { "acquisition" } else { "release" }
            ),
        );
        let permit = self.limit.acquire().await?;
        let r = self
            .client
            .post(format!("{}/api/generate", self.origin))
            .json(&json!({"model":t.model,"keep_alive":if loaded{-1}else{0},"stream":false}))
            .timeout(seconds(duration))
            .send()
            .await?;
        let status = r.status();
        let _ = r.bytes().await?;
        drop(permit);
        if status.as_u16() >= 400 {
            return Err(self.err(
                if loaded { "model load" } else { "model unload" },
                &format!("HTTP {}", status.as_u16()),
            ));
        }
        let start = Instant::now();
        timeout(seconds(duration), async {
            loop {
                let m = self.running().await?;
                if (!loaded && m.is_empty()) || (loaded && m == [t.model.clone()]) {
                    log(
                        20,
                        "st_proxy.llm.ollama",
                        format!(
                            "Ollama model state confirmed: state={state} duration={:.3}s",
                            start.elapsed().as_secs_f64()
                        ),
                    );
                    return Ok(());
                }
                sleep(seconds(self.config.poll_interval)).await;
            }
        })
        .await
        .map_err(|_| self.err(&format!("confirmation of {state}"), "timed out"))?
    }
    pub async fn release(&mut self, t: &mut Target) -> Result<()> {
        match self.kind.as_str() {
            "koboldcpp" => {
                self.kobold_reload("unload_model", self.config.unload_timeout)
                    .await?;
                self.kobold_wait(false, None, self.config.unload_timeout)
                    .await?;
            }
            "ollama" => self.ollama_change(t, false).await?,
            "tabbyapi" => {
                let duration = self.config.unload_timeout;
                timeout(seconds(duration), async {
                    let Some(current) = self.tabby_confirm(false).await? else {
                        return Ok(());
                    };
                    *t = current;
                    self.uncertain = true;
                    self.pending = None;
                    let permit = self.limit.acquire().await?;
                    let r = self
                        .client
                        .post(format!("{}/v1/model/unload", self.origin))
                        .send()
                        .await?;
                    let s = r.status().as_u16();
                    let _ = r.bytes().await?;
                    drop(permit);
                    if s != 200 {
                        return Err(self.err("unload", &format!("HTTP {s}")));
                    }
                    if self.tabby_observe().await?.is_some() {
                        return Err(self.err("unload", "model is still loaded"));
                    }
                    self.uncertain = false;
                    Ok(())
                })
                .await
                .map_err(|_| self.err("unload", "TimeoutError"))??;
            }
            "prism" => {
                self.certain()?;
                let states = self.states().await?;
                if !states.iter().any(|(m, _)| m == &t.model) {
                    return Err(self.err("restore target", "preset missing"));
                }
                if let Some((m, _)) = states.iter().find(|(_, s)| s["value"] != "unloaded") {
                    if m != &t.model {
                        return Err(self.err("unload", "active model differs from restore point"));
                    }
                    self.change(t, false).await?;
                }
            }
            _ => {
                self.certain()?;
                if self.health().await?["model"] != t.model {
                    return Err(self.err("restore target", "model mismatch"));
                }
                self.change(t, false).await?;
            }
        }
        Ok(())
    }
    async fn tabby_load(&mut self, t: &Target) -> Result<()> {
        let _permit = self.limit.acquire().await?;
        let mut v = t.parameters.clone();
        v.insert("model_name".into(), t.model.clone().into());
        v.insert("backend".into(), "exllamav3".into());
        v.insert("skip_queue".into(), false.into());
        let mut r = self
            .client
            .post(format!("{}/v1/model/load", self.origin))
            .json(&v)
            .send()
            .await?;
        if r.status() != 200 {
            return Err(self.err("load", &format!("HTTP {}", r.status().as_u16())));
        }
        if !r
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("text/event-stream"))
        {
            return Err(self.err("load", "ValueError"));
        }
        let mut buffer = Vec::new();
        let mut data = Vec::new();
        let (mut finished, mut failed, mut oom) = (false, false, false);
        while let Some(bytes) = r.chunk().await? {
            buffer.extend_from_slice(&bytes);
            while let Some(end) = buffer.iter().position(|b| *b == b'\n') {
                let line = String::from_utf8(buffer.drain(..=end).collect())?;
                let line = line.trim_end_matches(['\r', '\n']);
                if line
                    .strip_prefix("event:")
                    .is_some_and(|v| v.trim() == "error")
                {
                    return Err(self.err("load", "SSE error"));
                }
                if let Some(s) = line.strip_prefix("data:") {
                    data.push(s.trim_start_matches(' ').to_string())
                } else if line.is_empty() && !data.is_empty() {
                    let event: Value = serde_json::from_str(&data.join("\n"))?;
                    data.clear();
                    if !event.is_object() {
                        return Err(self.err("load", "invalid SSE event"));
                    }
                    if let Some(error) = event.get("error") {
                        let message = error["message"]
                            .as_str()
                            .ok_or_else(|| self.err("load", "invalid SSE error"))?
                            .to_lowercase();
                        failed = true;
                        oom |= [
                            "cuda out of memory",
                            "cuda error: out of memory",
                            "insufficient vram",
                            "torch.cuda.outofmemoryerror",
                        ]
                        .iter()
                        .any(|v| message.contains(v));
                        log(
                            40,
                            "st_proxy.llm.tabbyapi",
                            format!(
                                "TabbyAPI load error: {}; waiting for load stream to end",
                                if oom {
                                    "insufficient_vram (not enough GPU memory for model/cache)"
                                } else {
                                    "backend reported a load failure"
                                }
                            ),
                        );
                    } else if failed {
                        return Err(self.err("load", "progress after SSE error"));
                    } else if event["model_type"] == "model" && event["status"] == "finished" {
                        finished = true
                    }
                }
            }
        }
        if !data.is_empty() || !buffer.is_empty() || !(finished || failed) {
            return Err(self.err("load", "incomplete SSE response"));
        }
        if failed {
            self.load_failed = true;
            if oom {
                bail!(
                    "TabbyAPI load failed: insufficient_vram (not enough GPU memory for model/cache); model is not ready; next active LLM request will retry"
                )
            };
            return Err(self.err(
                "load",
                "backend reported a completed load failure; next LLM request may retry",
            ));
        }
        Ok(())
    }
    pub async fn acquire(&mut self, target: Option<Target>) -> Result<Target> {
        match self.kind.as_str() {
            "koboldcpp" => {
                if self.config.kobold_router_mode || self.config.llm_backend == "all" {
                    if !self.version_ready().await {
                        return Err(self.err("router readiness", "unavailable"));
                    }
                    return Ok(Target::new(&self.kind, ""));
                }
                self.kobold_reload("initial_model", self.config.reload_timeout)
                    .await?;
                let m = self
                    .kobold_wait(
                        true,
                        target.as_ref().map(|t| t.model.as_str()),
                        self.config.reload_timeout,
                    )
                    .await?;
                Ok(target.unwrap_or_else(|| Target::new(&self.kind, &m)))
            }
            "ollama" => {
                let t =
                    target.ok_or_else(|| self.err("model load", "a restore target is required"))?;
                self.ollama_change(&t, true).await?;
                Ok(t)
            }
            "tabbyapi" => {
                let duration = self.config.reload_timeout;
                timeout(seconds(duration),async{
                let expected=target.as_ref().map(|t|t.parameters.clone()).unwrap_or_default();
                let current=self.tabby_confirm(true).await?;
                let t=if let Some(t)=target{t}else{if let Some(c)=current{return Ok(c)}let model=self.config.tabby_model.as_ref().ok_or_else(||self.err("load","no loaded model; configure --tabby-model"))?;let mut t=Target::new(&self.kind,model);t.parameters.insert("max_seq_len".into(),json!(self.config.tabby_max_seq_len));t};
                if let Some(mut c)=current {if c.model==t.model && expected.iter().all(|(k,v)|c.parameters.get(k)==Some(v)){return Ok(c)}self.release(&mut c).await?;}
                self.uncertain=true;self.pending=Some(Target{parameters:expected.clone(),..t.clone()});self.load_failed=false;self.tabby_load(&t).await?;
                loop{if let Some(c)=self.tabby_observe().await?{if c.model!=t.model || expected.iter().any(|(k,v)|c.parameters.get(k)!=Some(v)){return Err(self.err("load verification","model/settings mismatch"))}self.uncertain=false;self.pending=None;return Ok(c)}sleep(seconds(self.config.poll_interval)).await;}
            }).await.map_err(|_|self.err("load","TimeoutError"))?
            }
            "prism" => {
                self.certain()?;
                let t = target.unwrap_or_else(|| Target::new(&self.kind, &self.config.prism_model));
                let states = self.states().await?;
                if !states.iter().any(|(m, _)| m == &t.model) {
                    return Err(self.err("restore target", "preset missing"));
                }
                if let Some((m, s)) = states.iter().find(|(_, s)| s["value"] != "unloaded") {
                    if s["value"] != "loaded" {
                        return Err(self.err("model observation", "model not in a stable state"));
                    }
                    if m == &t.model {
                        return Ok(t);
                    }
                    self.change(&Target::new(&self.kind, m), false).await?;
                }
                self.change(&t, true).await?;
                Ok(t)
            }
            _ => {
                self.certain()?;
                let t = match target {
                    Some(t) => t,
                    None => self.target(None).await?,
                };
                if self.health().await?["model"] != t.model {
                    return Err(self.err("restore target", "model mismatch"));
                }
                if self.observe().await?.is_none() {
                    self.change(&t, true).await?
                }
                Ok(t)
            }
        }
    }
}
fn inactive(s: &str) -> bool {
    matches!(
        s.to_lowercase().as_str(),
        "" | "inactive" | "unloaded" | "none" | "no model"
    )
}
pub fn kobold_models(v: &Value) -> Result<Value> {
    let cards = v["data"]
        .as_array()
        .ok_or_else(|| anyhow!("invalid KoboldCpp model list"))?;
    let mut names: Vec<_> = cards
        .iter()
        .filter_map(|v| v["id"].as_str())
        .filter(|s| *s == "initial_model" || s.ends_with(".kcpps"))
        .collect();
    names.sort_by_key(|s| (*s != "initial_model", *s));
    names.dedup();
    if !names.contains(&"initial_model") {
        bail!("KoboldCpp Router mode model list is unavailable")
    }
    Ok(
        json!({"object":"list","data":names.into_iter().map(|id|json!({"id":id,"object":"model","owned_by":"koboldcpp"})).collect::<Vec<_>>()}),
    )
}

pub struct Backends {
    pub config: Config,
    pub entries: Vec<Backend>,
    pub active: Option<Target>,
    saved: HashMap<(String, String), Target>,
    pending: Option<Target>,
    router_failed: bool,
    kobold_released: bool,
}
impl Backends {
    pub fn new(config: &Config, client: reqwest::Client) -> Self {
        Self::limited(
            config,
            client,
            crate::connection_limit::Limit::new(16, config.connect_timeout),
        )
    }
    pub fn limited(
        config: &Config,
        client: reqwest::Client,
        limit: crate::connection_limit::Limit,
    ) -> Self {
        let entries = if config.llm_backend == "all" {
            vec![
                ("koboldcpp", &config.llm_url),
                ("tabbyapi", &config.tabby_url),
                ("sc117", &config.sc117_url),
                ("orca", &config.orca_url),
                ("sc117_128k", &config.sc117_128k_url),
                ("orca_128k", &config.orca_128k_url),
                ("swift", &config.swift_url),
                ("swift_128k", &config.swift_128k_url),
                ("prism", &config.prism_url),
            ]
        } else {
            vec![(config.llm_backend.as_str(), &config.llm_url)]
        };
        Self {
            config: config.clone(),
            entries: entries
                .into_iter()
                .map(|(k, o)| {
                    let mut backend = Backend::new(k, o, config, client.clone());
                    backend.limit = limit.clone();
                    backend
                })
                .collect(),
            active: None,
            saved: HashMap::new(),
            pending: None,
            router_failed: false,
            kobold_released: false,
        }
    }
    pub async fn validate(&self) -> Result<()> {
        for b in &self.entries {
            b.validate().await?
        }
        Ok(())
    }
    pub async fn catalog(&self, engine: Option<&str>) -> Result<Value> {
        let mut cards = Vec::new();
        for b in &self.entries {
            if engine.is_some_and(|e| e != b.kind) {
                continue;
            }
            let v = b.catalog().await?;
            for c in v["data"].as_array().unwrap() {
                let mut c = c.clone();
                if self.config.llm_backend == "all" {
                    c["id"] = format!("{}/{}", b.kind, c["id"].as_str().unwrap()).into();
                    c["owned_by"] = b.kind.clone().into();
                }
                cards.push(c)
            }
        }
        Ok(json!({"object":"list","data":cards}))
    }
    pub async fn target(&self, model: Option<&str>) -> Result<Target> {
        if self.config.llm_backend != "all" {
            return self.entries[0].target(model).await;
        }
        let model =
            model.ok_or_else(|| anyhow!("select a model ID from /v1/models (backend/model)"))?;
        let (e, m) = model
            .split_once('/')
            .ok_or_else(|| anyhow!("select a model ID from /v1/models (backend/model)"))?;
        if !self.entries.iter().any(|b| b.kind == e) || m.is_empty() {
            bail!("unknown backend/model")
        }
        if !self.catalog(Some(e)).await?["data"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["id"] == model)
        {
            bail!("unknown backend/model")
        }
        Ok(Target::new(e, m))
    }
    pub async fn observe(&mut self) -> Result<Option<Target>> {
        if self.config.llm_backend != "all" {
            return self.entries[0].observe().await;
        }
        let mut occupied = self.pending.is_some();
        for b in &mut self.entries {
            occupied |= b.observe().await?.is_some()
        }
        Ok(if occupied {
            Some(self.active.clone().unwrap_or_else(|| Target::new("", "")))
        } else {
            None
        })
    }
    async fn release_one(&mut self, i: usize) -> Result<()> {
        let b = &mut self.entries[i];
        if b.kind == "koboldcpp" {
            if !self.kobold_released {
                b.release(&mut Target::new("koboldcpp", "")).await?;
                self.kobold_released = true;
            }
            self.router_failed = false;
        } else if let Some(mut observed) = b.observe().await? {
            if let Some(active) = self.active.as_ref().filter(|a| a.engine == b.kind) {
                self.saved
                    .insert((b.kind.clone(), active.model.clone()), observed.clone());
            }
            b.release(&mut observed).await?;
        }
        if self.active.as_ref().is_some_and(|a| a.engine == b.kind) {
            self.active = None;
        }
        if self.pending.as_ref().is_some_and(|a| a.engine == b.kind) {
            self.pending = None;
        }
        Ok(())
    }
    pub async fn release(&mut self, t: &mut Target) -> Result<()> {
        if self.config.llm_backend != "all" {
            return self.entries[0].release(t).await;
        }
        for i in 0..self.entries.len() {
            self.release_one(i).await?
        }
        Ok(())
    }
    pub fn failed(&mut self, t: Option<&Target>) {
        if t.is_some_and(|t| t.engine == "koboldcpp") {
            self.router_failed = true;
            self.kobold_released = false;
        }
    }
    pub async fn acquire(&mut self, target: Option<Target>) -> Result<Target> {
        if self.config.llm_backend != "all" {
            return self.entries[0].acquire(target).await;
        }
        let t = target.ok_or_else(|| {
            anyhow!("LLM backends model selection failed: explicit model required")
        })?;
        let i = self
            .entries
            .iter()
            .position(|b| b.kind == t.engine)
            .ok_or_else(|| {
                anyhow!("LLM backends model selection failed: explicit model required")
            })?;
        for n in 0..self.entries.len() {
            if n != i {
                self.release_one(n).await?
            }
        }
        if t.engine == "koboldcpp" && (self.router_failed || self.active.as_ref() == Some(&t)) {
            self.release_one(i).await?
        }
        if let Some(active) = self
            .active
            .as_ref()
            .filter(|a| a.engine == t.engine && **a != t)
            && let Some(o) = self.entries[i].observe().await?
        {
            self.saved
                .insert((active.engine.clone(), active.model.clone()), o);
        }
        let native = if let Some(t) = self.saved.get(&(t.engine.clone(), t.model.clone())) {
            t.clone()
        } else if self.entries[i].strata() {
            self.entries[i].target(Some(&t.model)).await?
        } else {
            t.clone()
        };
        self.pending = Some(t.clone());
        self.active = None;
        if t.engine == "koboldcpp" {
            self.kobold_released = false;
        }
        let restored = self.entries[i].acquire(Some(native)).await?;
        self.saved
            .insert((t.engine.clone(), t.model.clone()), restored);
        self.active = Some(t.clone());
        Ok(t)
    }
}
