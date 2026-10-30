//! Saved configuration and ephemeral Chat Completions operations. No inference coupling.
use crate::{enhancement_types::*, scheduler::Fault};
use anyhow::{Context, Result, bail};
use axum::{
    Json, Router,
    extract::{Path, State},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::Write,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Serialize, Deserialize)]
struct Saved {
    profiles: Vec<Profile>,
}
struct Job {
    status: Status,
    cancel: CancellationToken,
    created: Instant,
}
struct StateData {
    saved: Saved,
    jobs: BTreeMap<String, Job>,
}
pub struct Service {
    path: PathBuf,
    private: Option<(crate::private_storage::Root, PathBuf)>,
    state: Mutex<StateData>,
    http: reqwest::Client,
    slots: Arc<Semaphore>,
}
/// Captured provider configuration; external API credentials are not supported.
#[derive(Clone)]
pub(crate) struct ProviderSnapshot {
    pub profile: Profile,
}
#[derive(Clone)]
pub(crate) struct ProviderImage {
    pub mime: String,
    pub base64: String,
}
impl Service {
    pub(crate) fn snapshot(&self, id: &str) -> std::result::Result<ProviderSnapshot, Fault> {
        let state = self.state.lock().unwrap();
        let profile = state
            .saved
            .profiles
            .iter()
            .find(|p| p.id == id)
            .cloned()
            .ok_or(Fault::new(404, "enhancement_profile_not_found"))?;
        Ok(ProviderSnapshot { profile })
    }
    pub(crate) async fn authoring_complete(
        &self,
        snapshot: &ProviderSnapshot,
        provider: &str,
        system: &str,
        prompt: &str,
        parameters: &serde_json::Map<String, Value>,
        images: &[ProviderImage],
    ) -> std::result::Result<String, &'static str> {
        let p = &snapshot.profile;
        let (url, body) = if provider == "ollama" {
            let mut options = json!({"temperature":parameters["temperature"],"top_p":parameters["top_p"],"top_k":parameters["top_k"],"repeat_penalty":parameters["repeat_penalty"],"num_predict":parameters["max_tokens"]});
            if parameters["context_tokens"].as_u64().is_some_and(|n| n > 0) {
                options["num_ctx"] = parameters["context_tokens"].clone();
            }
            (
                format!("{}/api/generate", p.base_url.trim_end_matches('/')),
                {
                    let mut body = json!({"model":p.model,"system":system,"prompt":prompt,"stream":false,"keep_alive":0,"options":options});
                    if !images.is_empty() {
                        body["images"] =
                            json!(images.iter().map(|i| &i.base64).collect::<Vec<_>>());
                    }
                    body
                },
            )
        } else if provider == "koboldcpp" {
            let reasoning = &parameters["reasoning"];
            let mut body = json!({"model":p.model,"messages":[{"role":"system","content":system},{"role":"user","content":prompt}],"temperature":parameters["temperature"],"top_p":parameters["top_p"],"top_k":parameters["top_k"],"min_p":parameters["min_p"],"repetition_penalty":parameters["repeat_penalty"],"max_tokens":parameters["max_tokens"],"include_reasoning":reasoning["include"],"chat_template_kwargs":{"enable_thinking":reasoning["enable_thinking"]},"stream":false,"genkey":format!("imagine-{}",uuid::Uuid::new_v4().simple())});
            if !images.is_empty() {
                let mut content = vec![json!({"type":"text","text":prompt})];
                content.extend(images.iter().map(|i|json!({"type":"image_url","image_url":{"url":format!("data:{};base64,{}",i.mime,i.base64)}})));
                body["messages"][1]["content"] = json!(content);
            }
            if reasoning["effort"].as_str().is_some_and(|s| !s.is_empty()) {
                body["reasoning_effort"] = reasoning["effort"].clone();
            }
            if !reasoning["budget_tokens"].is_null() {
                body["reasoning_budget_tokens"] = reasoning["budget_tokens"].clone();
            }
            (
                format!("{}/chat/completions", p.base_url.trim_end_matches('/')),
                body,
            )
        } else {
            return Err("unsupported_authoring_provider");
        };
        let value = self
            .receive(
                self.http
                    .post(url)
                    .timeout(Duration::from_secs(p.timeout_seconds.into()))
                    .json(&body),
            )
            .await?;
        if provider == "ollama" {
            if value["done"] != true {
                return Err("enhancement_invalid_response");
            }
            if value["done_reason"] == "length" {
                return Err("enhancement_truncated");
            }
            value["response"]
                .as_str()
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().to_owned())
                .ok_or("enhancement_empty_response")
        } else {
            completion_text(&value)
        }
    }
    /// The selected model's context window as the provider's `/models`
    /// reports it (`meta.n_ctx`), as Director's outline budget reads it.
    pub(crate) async fn context_window(
        &self,
        snapshot: &ProviderSnapshot,
        provider: &str,
    ) -> std::result::Result<u64, &'static str> {
        const UNKNOWN: &str = "authoring_context_window_unknown";
        if provider != "koboldcpp" {
            return Err(UNKNOWN);
        }
        let p = &snapshot.profile;
        let value = self
            .receive(
                self.http
                    .get(format!("{}/models", p.base_url.trim_end_matches('/')))
                    .timeout(Duration::from_secs(20)),
            )
            .await?;
        let rows = value["data"].as_array().ok_or(UNKNOWN)?;
        let mut matches = rows.iter().filter(|row| row["id"] == p.model.as_str());
        match (matches.next(), matches.next()) {
            (Some(row), None) => row["meta"]["n_ctx"]
                .as_u64()
                .filter(|n| *n > 0)
                .ok_or(UNKNOWN),
            _ => Err(UNKNOWN),
        }
    }
    /// Explicit fixture/developer path; normal startup uses open_private.
    pub fn open(path: PathBuf) -> Result<Arc<Self>> {
        let saved: Saved = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("Invalid enhancement profiles")?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Saved {
                profiles: vec![Profile::default()],
            },
            Err(e) => return Err(e.into()),
        };
        Self::from_saved(path, None, saved)
    }
    pub fn open_private(
        root: crate::private_storage::Root,
        relative: PathBuf,
    ) -> Result<Arc<Self>> {
        let saved = match root.read(&relative).map_err(anyhow::Error::msg)? {
            Some(bytes) => {
                serde_json::from_slice(&bytes).context("Invalid enhancement profiles")?
            }
            None => Saved {
                profiles: vec![Profile::default()],
            },
        };
        Self::from_saved(relative.clone(), Some((root, relative)), saved)
    }
    fn from_saved(
        path: PathBuf,
        private: Option<(crate::private_storage::Root, PathBuf)>,
        saved: Saved,
    ) -> Result<Arc<Self>> {
        let mut ids = std::collections::BTreeSet::new();
        for profile in &saved.profiles {
            validate(profile)?;
            if !ids.insert(&profile.id) {
                bail!("Duplicate enhancement profile ID");
            }
        }
        Ok(Arc::new(Self {
            path,
            private,
            state: Mutex::new(StateData {
                saved,
                jobs: BTreeMap::new(),
            }),
            http: reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECONDS.into()))
                .build()?,
            slots: Arc::new(Semaphore::new(4)),
        }))
    }
    pub fn profiles(&self) -> Vec<Profile> {
        let state = self.state.lock().unwrap();
        public_profiles(&state.saved)
    }
    pub async fn models(&self, lookup: ModelsRequest) -> std::result::Result<Vec<String>, Fault> {
        validate_url(&lookup.base_url).map_err(|_| Fault::new(400, "invalid_enhancement_url"))?;
        let request = self
            .http
            .get(format!("{}/models", lookup.base_url.trim_end_matches('/')))
            .timeout(Duration::from_secs(20));
        let mut response = request
            .send()
            .await
            .map_err(|_| Fault::new(502, "enhancement_unreachable"))?;
        if !response.status().is_success() {
            return Err(Fault::new(
                502,
                match response.status().as_u16() {
                    401 | 403 => "enhancement_auth_failed",
                    404 | 405 => "enhancement_models_unsupported",
                    429 => "enhancement_rate_limited",
                    _ => "enhancement_provider_failed",
                },
            ));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| Fault::new(502, "enhancement_invalid_models"))?
        {
            if bytes.len() + chunk.len() > 1024 * 1024 {
                return Err(Fault::new(502, "enhancement_invalid_models"));
            }
            bytes.extend_from_slice(&chunk);
        }
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|_| Fault::new(502, "enhancement_invalid_models"))?;
        let data = value["data"]
            .as_array()
            .ok_or(Fault::new(502, "enhancement_invalid_models"))?;
        let mut models = data
            .iter()
            .map(|model| {
                model["id"]
                    .as_str()
                    .filter(|id| !id.trim().is_empty() && id.len() <= 1024)
                    .map(str::to_owned)
                    .ok_or(Fault::new(502, "enhancement_invalid_models"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        models.sort();
        models.dedup();
        Ok(models)
    }
    fn persist(&self, saved: &Saved) -> Result<()> {
        if let Some((root, relative)) = &self.private {
            return root
                .write(relative, &serde_json::to_vec_pretty(saved)?)
                .map_err(anyhow::Error::msg);
        }
        let directory = self.path.parent().context("Profile directory missing")?;
        std::fs::create_dir_all(directory)?;
        std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;
        let temp = directory.join(format!("enhancement-{}.tmp", uuid::Uuid::new_v4()));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temp)?;
            file.write_all(&serde_json::to_vec_pretty(saved)?)?;
            file.sync_all()?;
            std::fs::rename(&temp, &self.path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(temp);
        }
        result
    }
    pub fn save(&self, update: ProfileUpdate) -> std::result::Result<Vec<Profile>, Fault> {
        validate(&update.profile).map_err(|_| Fault::new(400, "invalid_enhancement_profile"))?;
        let mut state = self.state.lock().unwrap();
        let mut saved = state.saved.clone();
        let profile = update.profile;
        if let Some(existing) = saved.profiles.iter_mut().find(|p| p.id == profile.id) {
            *existing = profile;
        } else {
            if saved.profiles.len() >= 128 {
                return Err(Fault::new(409, "enhancement_profile_limit"));
            }
            saved.profiles.push(profile);
        }
        self.commit_saved(&mut state, saved)?;
        Ok(public_profiles(&state.saved))
    }
    pub fn delete(&self, id: &str) -> std::result::Result<Vec<Profile>, Fault> {
        let mut state = self.state.lock().unwrap();
        let mut saved = state.saved.clone();
        if !saved.profiles.iter().any(|p| p.id == id) {
            return Err(Fault::new(404, "enhancement_profile_not_found"));
        }
        saved.profiles.retain(|p| p.id != id);
        self.commit_saved(&mut state, saved)?;
        Ok(public_profiles(&state.saved))
    }
    fn commit_saved(&self, state: &mut StateData, saved: Saved) -> std::result::Result<(), Fault> {
        if let Err(error) = self.persist(&saved) {
            // This exact private backend code means rename succeeded but the
            // directory sync was uncertain. Match committed disk state in RAM,
            // while reporting failure rather than claiming durable success.
            if self.private.is_some() && error.to_string() == "private_storage_write_unconfirmed" {
                state.saved = saved;
            }
            return Err(Fault::new(500, "enhancement_save_failed"));
        }
        state.saved = saved;
        Ok(())
    }
    pub fn start(self: &Arc<Self>, request: Request) -> std::result::Result<(), Fault> {
        if uuid::Uuid::parse_str(&request.id).is_err()
            || request.prompt.trim().is_empty()
            || request.prompt.len() > 65536
        {
            return Err(Fault::new(400, "invalid_enhancement_request"));
        }
        let permit = self
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| Fault::new(429, "enhancement_busy"))?;
        let mut state = self.state.lock().unwrap();
        state.jobs.retain(|_, job| {
            if job.created.elapsed() > Duration::from_secs(600) {
                job.cancel.cancel();
                false
            } else {
                true
            }
        });
        if state.jobs.contains_key(&request.id) {
            return Err(Fault::new(409, "enhancement_id_exists"));
        }
        let profile = state
            .saved
            .profiles
            .iter()
            .find(|p| p.id == request.profile_id)
            .cloned()
            .ok_or(Fault::new(404, "enhancement_profile_not_found"))?;
        let cancel = CancellationToken::new();
        state.jobs.insert(
            request.id.clone(),
            Job {
                status: Status::Running,
                cancel: cancel.clone(),
                created: Instant::now(),
            },
        );
        drop(state);
        let service = self.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let status = tokio::select! {
                biased;
                _ = cancel.cancelled() => Status::Cancelled,
                result = service.complete(profile, request.prompt) => match result {
                    Ok(text) => Status::Succeeded { text },
                    Err(code) => Status::Failed { code: code.into() },
                },
            };
            if let Some(job) = service.state.lock().unwrap().jobs.get_mut(&request.id) {
                job.status = status;
            }
        });
        Ok(())
    }
    pub fn status(&self, id: &str) -> std::result::Result<Status, Fault> {
        self.state
            .lock()
            .unwrap()
            .jobs
            .get(id)
            .map(|j| j.status.clone())
            .ok_or(Fault::new(404, "enhancement_not_found"))
    }
    pub fn remove(&self, id: &str) {
        if let Some(job) = self.state.lock().unwrap().jobs.remove(id) {
            job.cancel.cancel();
        }
    }
    async fn complete(
        &self,
        profile: Profile,
        prompt: String,
    ) -> std::result::Result<String, &'static str> {
        let mut body = json!({"model": profile.model, "messages": [
            {"role":"system", "content":profile.system_prompt},
            {"role":"user", "content":prompt}
        ], "stream":false, "n":1});
        body[profile.token_parameter.name()] = json!(profile.max_tokens);
        if let Some(temperature) = profile.temperature {
            body["temperature"] = json!(temperature);
        }
        body.as_object_mut().unwrap().extend(profile.parameters);
        let request = self
            .http
            .post(format!(
                "{}/chat/completions",
                profile.base_url.trim_end_matches('/')
            ))
            .timeout(Duration::from_secs(profile.timeout_seconds.into()))
            .json(&body);
        let value = self.receive(request).await?;
        completion_text(&value)
    }
    async fn receive(
        &self,
        request: reqwest::RequestBuilder,
    ) -> std::result::Result<Value, &'static str> {
        let mut response = request.send().await.map_err(|e| {
            if e.is_timeout() {
                "enhancement_timeout"
            } else {
                "enhancement_unreachable"
            }
        })?;
        if !response.status().is_success() {
            return Err(match response.status().as_u16() {
                401 | 403 => "enhancement_auth_failed",
                429 => "enhancement_rate_limited",
                400 | 404 | 422 => "enhancement_request_rejected",
                _ => "enhancement_provider_failed",
            });
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| "enhancement_invalid_response")?
        {
            if bytes.len() + chunk.len() > 1024 * 1024 {
                return Err("enhancement_invalid_response");
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes).map_err(|_| "enhancement_invalid_response")
    }
}
fn completion_text(value: &Value) -> std::result::Result<String, &'static str> {
    let choice = &value["choices"][0];
    if choice["finish_reason"] == "length" {
        return Err("enhancement_truncated");
    }
    if choice["message"]["refusal"]
        .as_str()
        .is_some_and(|s| !s.is_empty())
        || choice["finish_reason"] == "content_filter"
    {
        return Err("enhancement_refused");
    }
    if choice["finish_reason"] != "stop" {
        return Err("enhancement_invalid_response");
    }
    let text = choice["message"]["content"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or("enhancement_empty_response")?;
    Ok(text.trim().to_owned())
}
fn public_profiles(saved: &Saved) -> Vec<Profile> {
    saved.profiles.clone()
}
pub fn validate(p: &Profile) -> Result<()> {
    validate_url(&p.base_url)?;
    validate_parameters(&p.parameters)?;
    if p.id.is_empty()
        || p.id.len() > 64
        || !p
            .id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        || p.name.trim().is_empty()
        || p.name.len() > 128
        || p.model.trim().is_empty()
        || p.model.len() > 1024
        || p.system_prompt.trim().is_empty()
        || !(1..=131072).contains(&p.max_tokens)
        || !(1..=3600).contains(&p.timeout_seconds)
        || p.temperature
            .is_some_and(|t| !t.is_finite() || !(0.0..=2.0).contains(&t))
    {
        bail!("Invalid enhancement profile");
    }
    Ok(())
}
fn validate_url(base_url: &str) -> Result<()> {
    let url = reqwest::Url::parse(base_url)?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        bail!("Invalid enhancement URL");
    }
    Ok(())
}
fn validate_parameters(parameters: &serde_json::Map<String, Value>) -> Result<()> {
    for (name, value) in parameters {
        if reserved_parameter(name) || name.trim().is_empty() {
            bail!("Reserved enhancement parameter");
        }
        let range = match name.as_str() {
            "top_p" | "min_p" | "typical_p" => Some(0.0..=1.0),
            "frequency_penalty" | "presence_penalty" => Some(-2.0..=2.0),
            "repetition_penalty" => Some(0.0..=f64::MAX),
            _ => None,
        };
        if let Some(range) = range
            && !value
                .as_f64()
                .is_some_and(|v| v.is_finite() && range.contains(&v))
        {
            bail!("Invalid sampling parameter");
        }
        if name == "top_k" && value.as_u64().is_none()
            || name == "seed" && value.as_i64().is_none()
            || name == "stop"
                && !(value.is_string()
                    || value
                        .as_array()
                        .is_some_and(|v| v.iter().all(Value::is_string)))
        {
            bail!("Invalid sampling parameter");
        }
    }
    Ok(())
}
pub fn routes(service: Arc<Service>) -> Router {
    Router::new()
        .route("/v1/enhancement/profiles", get(list))
        .route("/v1/enhancement/models", post(models))
        .route(
            "/v1/enhancement/profiles/{id}",
            axum::routing::put(save).delete(delete),
        )
        .route("/v1/enhancements", post(start))
        .route("/v1/enhancements/{id}", get(status).delete(remove))
        .with_state(service)
}
async fn models(
    State(service): State<Arc<Service>>,
    Json(lookup): Json<ModelsRequest>,
) -> std::result::Result<Json<Vec<String>>, Fault> {
    service.models(lookup).await.map(Json)
}
async fn list(State(service): State<Arc<Service>>) -> Json<Vec<Profile>> {
    Json(service.profiles())
}
async fn save(
    State(service): State<Arc<Service>>,
    Path(id): Path<String>,
    Json(update): Json<ProfileUpdate>,
) -> std::result::Result<Json<Vec<Profile>>, Fault> {
    if id != update.profile.id {
        return Err(Fault::new(400, "invalid_enhancement_profile"));
    }
    service.save(update).map(Json)
}
async fn delete(
    State(service): State<Arc<Service>>,
    Path(id): Path<String>,
) -> std::result::Result<Json<Vec<Profile>>, Fault> {
    service.delete(&id).map(Json)
}
async fn start(
    State(service): State<Arc<Service>>,
    Json(request): Json<Request>,
) -> std::result::Result<axum::http::StatusCode, Fault> {
    service.start(request)?;
    Ok(axum::http::StatusCode::ACCEPTED)
}
async fn status(
    State(service): State<Arc<Service>>,
    Path(id): Path<String>,
) -> std::result::Result<Json<Status>, Fault> {
    service.status(&id).map(Json)
}
async fn remove(
    State(service): State<Arc<Service>>,
    Path(id): Path<String>,
) -> axum::http::StatusCode {
    service.remove(&id);
    axum::http::StatusCode::NO_CONTENT
}

#[cfg(test)]
#[path = "enhancement_tests.rs"]
mod tests;
