//! MCP transport and process-owned media jobs; HTTP disconnect never cancels generation.
use crate::{
    config::Config,
    imagine::Client,
    logging::log,
    references::{self, Reference},
};
use anyhow::{Context, Result, anyhow};
use axum::{
    Router,
    body::Body,
    extract::State,
    http::{Request, Response, StatusCode},
    routing::any,
};
use hmac::{Hmac, Mac};
use serde_json::{Value, json};
use sha2::Sha256;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
#[derive(Clone)]
pub struct Mcp {
    inner: Arc<Inner>,
    client: Client,
}
struct Inner {
    config: Config,
    client: Client,
    references: Mutex<HashMap<String, Reference>>,
    jobs: Mutex<HashMap<String, Job>>,
    key: [u8; 32],
    stop: CancellationToken,
    storage: crate::mcp_storage::Storage,
    gate: tokio::sync::Mutex<()>,
    session: Mutex<Option<String>>,
    writes: Mutex<()>,
    epoch: std::sync::atomic::AtomicU64,
}
struct Job {
    fingerprint: Vec<u8>,
    result: watch::Sender<Value>,
    task: Option<tokio::task::JoinHandle<()>>,
    count: usize,
    /// The workspace whose key started the job; only it can read the job.
    workspace: String,
}
#[derive(Debug, serde::Serialize)]
struct Fault {
    category: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}
impl Fault {
    fn new(category: &str, message: impl ToString) -> Self {
        Self {
            category: category.into(),
            message: message.to_string(),
            code: None,
            at: None,
            detail: None,
        }
    }
    fn generation(job: &Value) -> Self {
        // Forward only Imagine's stable code, never native logs, prompts or paths.
        let code = job["error"]["code"].as_str().filter(|code| {
            !code.is_empty()
                && code.len() <= 128
                && code.bytes().all(|c| c.is_ascii_lowercase() || c == b'_')
        });
        Self {
            category: "generation".into(),
            message: code.map_or_else(
                || "image generation did not complete successfully".into(),
                |code| format!("image generation failed: {code}"),
            ),
            code: code.map(str::to_owned),
            at: None,
            detail: None,
        }
    }
}
fn validation(e: impl ToString) -> Fault {
    Fault::new("validation", e)
}
fn error(f: Fault, id: Option<&str>) -> Value {
    json!({"job_id":id,"error":f})
}
fn tool_result(payload: Value) -> Value {
    tool_result_with(payload, vec![])
}
/// Extra content blocks (images) follow the text block; structuredContent stays plain JSON.
fn tool_result_with(payload: Value, blocks: Vec<Value>) -> Value {
    let mut content = vec![json!({"type":"text","text":python_json(&payload)})];
    content.extend(blocks);
    json!({"content":content,"structuredContent":payload,"isError":payload.get("error").is_some_and(|v|!v.is_null())})
}
pub fn python_json(value: &Value) -> String {
    crate::json_wire::dumps(value, true, false)
}

fn canonical(v: &Value) -> Value {
    match v {
        Value::Object(v) => {
            let sorted: std::collections::BTreeMap<_, _> =
                v.iter().map(|(k, v)| (k, canonical(v))).collect();
            json!(sorted)
        }
        Value::Array(v) => json!(v.iter().map(canonical).collect::<Vec<_>>()),
        _ => v.clone(),
    }
}
impl Mcp {
    pub async fn new(config: Config) -> Result<Self> {
        if config.coordinator_socket.is_none() {
            return Err(anyhow!(
                "MCP Imagine integration requires --coordinator-socket"
            ));
        }
        let client = Client::new(&config).await?;
        let proof = client
            .json(reqwest::Method::GET, "/v1/media-destination", None, None)
            .await
            .context("MCP needs a reachable Imagine with a verified encrypted media destination")?;
        let proof: crate::media_destination::Proof = serde_json::from_value(proof)?;
        let storage = crate::mcp_storage::Storage::open(proof)?;
        let mut key = [0u8; 32];
        use std::io::Read;
        std::fs::File::open("/dev/urandom")?.read_exact(&mut key)?;
        let this = Self {
            client: client.clone(),
            inner: Arc::new(Inner {
                config,
                client,
                references: Default::default(),
                jobs: Default::default(),
                key,
                stop: CancellationToken::new(),
                storage,
                gate: tokio::sync::Mutex::new(()),
                session: Mutex::new(None),
                writes: Mutex::new(()),
                epoch: std::sync::atomic::AtomicU64::new(0),
            }),
        };
        let weak = Arc::downgrade(&this.inner);
        let stop = this.inner.stop.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = stop.cancelled() => break,
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {},
                }
                let Some(inner) = weak.upgrade() else {
                    break;
                };
                let mcp = Self {
                    client: inner.client.clone(),
                    inner,
                };
                let _guard = mcp.inner.gate.lock().await;
                let _ = mcp.refresh_session().await;
            }
        });
        Ok(this)
    }
    pub fn router(&self) -> Router {
        Router::new()
            .route("/mcp", any(handle))
            .route("/mcp/", any(handle))
            .route(
                "/v1/workspace/files",
                axum::routing::get(workspace::list_files).put(workspace::upload_file),
            )
            .route(
                "/v1/workspace/files/{file}",
                axum::routing::get(workspace::download_file),
            )
            .with_state(self.clone())
    }
    pub async fn close(&self) {
        self.inner.stop.cancel();
        let _guard = self.inner.gate.lock().await;
        if self.clear_session().await.is_err() {
            log(40, "st_proxy.mcp", "MCP session cleanup failed");
        }
    }
    async fn clear_session(&self) -> Result<()> {
        self.inner
            .epoch
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let tasks: Vec<_> = self
            .inner
            .jobs
            .lock()
            .unwrap()
            .values_mut()
            .filter_map(|job| job.task.take())
            .collect();
        // Join every owned task before discarding session-only references/results.
        for task in tasks {
            task.abort();
            let _ = task.await;
        }
        self.inner.jobs.lock().unwrap().clear();
        self.inner.references.lock().unwrap().clear();
        *self.inner.session.lock().unwrap() = None;
        let _writes = self.inner.writes.lock().unwrap();
        Ok(())
    }
    async fn refresh_session(&self) -> Result<()> {
        if self.inner.stop.is_cancelled() {
            anyhow::bail!("MCP stopped");
        }
        self.inner.storage.preflight(0)?;
        let current = self.client.session().await.ok();
        let changed = *self.inner.session.lock().unwrap() != current;
        if changed || current.is_none() {
            self.clear_session().await?;
            *self.inner.session.lock().unwrap() = current.clone();
        }
        if current.is_none() {
            anyhow::bail!("Imagine is unavailable");
        }
        Ok(())
    }
    pub async fn call(&self, name: &str, args: &Value) -> Value {
        let guard = self.inner.gate.lock().await;
        if let Err(failure) = self.refresh_session().await {
            return tool_result(error(Fault::new("privacy", failure.to_string()), None));
        }
        let id = args["job_id"].as_str();
        let active = self.active();
        // Video tools keep no proxy state, so their longer calls do not hold the session gate.
        let limit = video::limit(name, args);
        let guard = if limit.is_some() {
            drop(guard);
            None
        } else {
            Some(guard)
        };
        let mut blocks = vec![];
        let operation = async {
            match name {
                "list_profiles" => active
                    .client
                    .json(reqwest::Method::GET, "/v1/profiles", None, None)
                    .await
                    .map_err(validation),
                "describe_profile" => active.profile(args).await,
                "imagine_guide" => Ok(json!({"guide":GUIDE})),
                "register_image" => active.register(args).await,
                "generate" | "generate_batch" => {
                    active.generate(args, name == "generate_batch").await
                }
                "get_job" => match active.bound(args).await {
                    Ok(workspace) => active.get(id.unwrap_or(""), false, &workspace.id).await,
                    Err(fault) => Err(fault),
                },
                "workspace_create" | "workspace_open" | "workspace_files" | "workspace_upload"
                | "workspace_view" => {
                    active
                        .workspace_tool(name, args)
                        .await
                        .map(|(value, images)| {
                            blocks = images;
                            value
                        })
                }
                _ if limit.is_some() => active.video(name, args).await.map(|(value, images)| {
                    blocks = images;
                    value
                }),
                _ => Err(validation(format!("Unknown tool: {name}"))),
            }
        };
        let result = tokio::select! {
            _ = self.inner.stop.cancelled() => Err(Fault::new("shutdown", "broker is shutting down")),
            result = tokio::time::timeout(limit.unwrap_or(Duration::from_secs(5)), operation) =>
                result.unwrap_or_else(|_| Err(Fault::new("connection", "Imagine did not respond; retry within the same session"))),
        };
        drop(guard);
        let waits = result.is_ok()
            && (matches!(name, "generate" | "generate_batch")
                || (name == "get_job" && args["wait"].as_bool().unwrap_or(true)));
        let result = if waits {
            match self.job_workspace(id.unwrap_or("")) {
                Some(workspace) => self.get(id.unwrap_or(""), true, &workspace).await,
                None => result,
            }
        } else {
            result
        };
        match result {
            Ok(mut v) => {
                // No host path ever reaches the agent.
                self.scrub(&mut v, None);
                tool_result_with(v, blocks)
            }
            Err(e) => tool_result(error(e, id)),
        }
    }
    /// This MCP handle bound to the current Imagine session.
    fn active(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            client: self.inner.client.with_session(
                self.inner
                    .session
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_default(),
            ),
        }
    }
    fn job_workspace(&self, id: &str) -> Option<String> {
        self.inner
            .jobs
            .lock()
            .unwrap()
            .get(id)
            .map(|job| job.workspace.clone())
    }
    async fn profile(&self, args: &Value) -> std::result::Result<Value, Fault> {
        let id = args["profile"]
            .as_str()
            .filter(|s| {
                !s.is_empty()
                    && s.len() <= 128
                    && s.bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
            })
            .ok_or_else(|| validation("invalid profile"))?;
        self.client
            .json(
                reqwest::Method::GET,
                &format!("/v1/profiles/{id}"),
                None,
                None,
            )
            .await
            .map_err(validation)
    }
    async fn register(&self, args: &Value) -> std::result::Result<Value, Fault> {
        if self.inner.stop.is_cancelled() {
            return Err(Fault::new("shutdown", "broker is shutting down"));
        }
        let profile = self.profile(args).await?;
        if profile["references"]["max"].as_u64().unwrap_or(0) == 0 {
            return Err(validation("profile does not accept references"));
        }
        let workspace = self.bound(args).await?;
        let file = args["file_id"]
            .as_str()
            .ok_or_else(|| validation("file_id must be a file ID from this workspace"))?;
        let (path, _) = self.file_path(&workspace, file).await?;
        let roots = vec![workspace.folder.clone()];
        let epoch = self.inner.epoch.load(std::sync::atomic::Ordering::SeqCst);
        let prepared = tokio::task::spawn_blocking(move || {
            let path = path.canonicalize()?;
            let original = references::read_image(&path, &roots)?;
            let digest = references::hash(&original);
            let bytes = original.len() as u64;
            let (preview, width, height) = references::normalize(&original)?;
            Ok::<_, anyhow::Error>((path, digest, bytes, preview, width, height))
        })
        .await
        .map_err(|_| validation("reference could not be read or decoded"))?
        .map_err(|e| {
            if e.downcast_ref::<std::io::Error>().is_some()
                || e.downcast_ref::<image::ImageError>().is_some()
            {
                validation("reference could not be read or decoded")
            } else {
                validation(e)
            }
        })?;
        let (path, digest, bytes, preview, width, height) = prepared;
        let linked = self.client.link(&path).await.map_err(validation)?;
        if linked["sha256"].as_str() != Some(digest.as_str())
            || linked["bytes"].as_u64() != Some(bytes)
        {
            return Err(validation(
                "registered reference has changed; register the image again",
            ));
        }
        let asset = linked["id"]
            .as_str()
            .ok_or_else(|| validation("invalid reference storage response"))?
            .to_owned();
        let _writes = self.inner.writes.lock().unwrap();
        if epoch != self.inner.epoch.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(Fault::new("privacy", "Imagine session ended"));
        }
        let r = Reference {
            workflow: args["profile"].as_str().unwrap().into(),
            workspace: workspace.id.clone(),
            asset,
            path,
            width,
            height,
            digest,
            mime_type: linked["mime_type"]
                .as_str()
                .unwrap_or("image/png")
                .to_owned(),
            preview_png: Arc::new(preview),
        };
        let id = uuid::Uuid::new_v4().simple().to_string();
        let result = r.result(&id);
        self.inner.references.lock().unwrap().insert(id, r);
        Ok(result)
    }
    async fn get(
        &self,
        id: &str,
        wait: bool,
        workspace: &str,
    ) -> std::result::Result<Value, Fault> {
        let mut rx = self
            .inner
            .jobs
            .lock()
            .unwrap()
            .get(id)
            .filter(|job| job.workspace == workspace)
            .ok_or_else(|| {
                Fault::new(
                    "not_found",
                    "job is unknown in this workspace and broker process",
                )
            })?
            .result
            .subscribe();
        loop {
            let value = rx.borrow_and_update().clone();
            if !wait
                || matches!(
                    value["status"].as_str(),
                    Some("succeeded" | "failed" | "partial")
                )
            {
                return Ok(value);
            }
            if rx.changed().await.is_err() {
                return Err(Fault::new(
                    "privacy",
                    "Imagine session ended; temporary history was cleared",
                ));
            }
        }
    }
    fn fingerprint(&self, batch: bool, args: &Value) -> Vec<u8> {
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&self.inner.key).unwrap();
        mac.update(python_json(&canonical(&json!([batch, args]))).as_bytes());
        mac.finalize().into_bytes().to_vec()
    }
    async fn generate(&self, args: &Value, batch: bool) -> std::result::Result<Value, Fault> {
        if self.inner.stop.is_cancelled() {
            return Err(Fault::new("shutdown", "broker is shutting down"));
        }
        let id = args["job_id"]
            .as_str()
            .filter(|s| {
                !s.is_empty()
                    && s.len() <= 128
                    && s.bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
            })
            .ok_or_else(|| validation("job_id must be 1-128 letters, digits, _ or -"))?;
        let workspace = self.bound(args).await?;
        let fingerprint = self.fingerprint(batch, args);
        let existing = {
            let jobs = self.inner.jobs.lock().unwrap();
            jobs.get(id).map(|job| {
                use subtle::ConstantTimeEq;
                bool::from(job.fingerprint.ct_eq(&fingerprint))
            })
        };
        match existing {
            Some(true) => return self.get(id, false, &workspace.id).await,
            Some(false) => {
                return Err(Fault::new(
                    "conflict",
                    "job_id already belongs to another request",
                ));
            }
            None => {}
        }
        let profile = self.profile(args).await?;
        let wid = args["profile"].as_str().unwrap();
        // Images are saved in the workspace's own image folder.
        let destination = self
            .client
            .json(
                reqwest::Method::GET,
                &format!("/v1/workspaces/{}/destination", workspace.id),
                None,
                None,
            )
            .await
            .map_err(|_| Fault::new("storage", "the workspace image folder is unavailable"))?;
        let items = if batch {
            args["items"]
                .as_array()
                .filter(|items| {
                    !items.is_empty() && items.len() <= self.inner.config.max_queued_images
                })
                .cloned()
                .ok_or_else(|| {
                    validation("items must be non-empty and within the image queue limit")
                })?
        } else {
            vec![
                json!({"parameters":args["parameters"],"references":args.get("references").filter(|v|!v.is_null()).cloned().unwrap_or(json!([]))}),
            ]
        };
        let mut values = Vec::new();
        let mut public = Vec::new();
        for item in &items {
            let (p, r) = if item["parameters"].is_object() {
                if item
                    .as_object()
                    .unwrap()
                    .keys()
                    .any(|k| k != "parameters" && k != "references")
                {
                    return Err(validation("unknown batch item field"));
                };
                (
                    &item["parameters"],
                    item.get("references").cloned().unwrap_or(json!([])),
                )
            } else {
                (item, json!([]))
            };
            let mut request = p
                .as_object()
                .cloned()
                .ok_or_else(|| validation("parameters must be an object"))?;
            if ["profile", "profile_version", "resolved_profile", "inputs"]
                .iter()
                .any(|k| request.contains_key(*k))
            {
                return Err(validation(
                    "choose the profile and registered references outside parameters",
                ));
            }
            request.insert("profile".into(), json!(wid));
            request.insert("profile_version".into(), profile["version"].clone());
            let ids = r
                .as_array()
                .filter(|ids| ids.iter().all(Value::is_string))
                .ok_or_else(|| validation("invalid reference list for this profile"))?;
            if ids.len() < profile["references"]["min"].as_u64().unwrap_or(0) as usize
                || ids.len() > profile["references"]["max"].as_u64().unwrap_or(0) as usize
            {
                return Err(validation("invalid reference count"));
            }
            let mut resolved = Vec::new();
            for id in ids {
                let image = self
                    .inner
                    .references
                    .lock()
                    .unwrap()
                    .get(id.as_str().unwrap())
                    .filter(|r| r.workflow == wid && r.workspace == workspace.id)
                    .cloned()
                    .ok_or_else(|| {
                        validation("reference ID is unknown for this profile; register it again")
                    })?;
                let verify = image.clone();
                let roots = vec![workspace.folder.clone()];
                tokio::task::spawn_blocking(move || verify.verify(&roots))
                    .await
                    .map_err(|_| {
                        validation("registered reference is unavailable; register it again")
                    })?
                    .map_err(|e| {
                        if e.downcast_ref::<std::io::Error>().is_some() {
                            validation("registered reference is unavailable; register it again")
                        } else {
                            validation(e)
                        }
                    })?;
                resolved.push(image)
            }
            request.insert(
                "inputs".into(),
                json!({"reference_images":resolved.iter().map(|r| &r.asset).collect::<Vec<_>>()}),
            );
            request.insert(
                "client_metadata".into(),
                json!({"video_destination":destination}),
            );
            let request = Value::Object(request);
            let validated = self
                .client
                .json(
                    reqwest::Method::POST,
                    "/v1/jobs/validate",
                    Some(&request),
                    None,
                )
                .await
                .map_err(validation)?;
            public.push(validated["effective"].clone());
            values.push(request);
        }
        {
            let mut jobs = self.inner.jobs.lock().unwrap();
            if let Some(job) = jobs.get(id) {
                use subtle::ConstantTimeEq;
                if !bool::from(job.fingerprint.ct_eq(&fingerprint)) {
                    return Err(Fault::new(
                        "conflict",
                        "job_id already belongs to another request",
                    ));
                }
            } else {
                let active: usize = jobs
                    .values()
                    .filter(|j| {
                        !matches!(
                            j.result.borrow()["status"].as_str(),
                            Some("succeeded" | "failed" | "partial")
                        )
                    })
                    .map(|j| j.count)
                    .sum();
                if active + items.len() > self.inner.config.max_queued_images {
                    return Err(Fault::new("queue_full", "MCP image capacity is exhausted"));
                }
                let result = if batch {
                    json!({"job_id":id,"profile":wid,"status":"preparing","items":public.iter().enumerate().map(|(i,p)|json!({"index":i,"status":"preparing","parameters":p,"images":[],"error":null})).collect::<Vec<_>>(),"error":null})
                } else {
                    json!({"job_id":id,"profile":wid,"status":"preparing","parameters":public[0],"images":[],"error":null})
                };
                let (tx, _) = watch::channel(result);
                let this = self.clone();
                let job_id = id.to_string();
                let sender = tx.clone();
                let epoch = self.inner.epoch.load(std::sync::atomic::Ordering::SeqCst);
                let owner = workspace.clone();
                let task = tokio::spawn(async move {
                    this.run(job_id, values, batch, sender, epoch, owner).await;
                });
                jobs.insert(
                    id.into(),
                    Job {
                        fingerprint,
                        result: tx,
                        task: Some(task),
                        count: items.len(),
                        workspace: workspace.id.clone(),
                    },
                );
            }
        }
        self.get(id, false, &workspace.id).await
    }
    async fn run(
        &self,
        id: String,
        values: Vec<Value>,
        batch: bool,
        tx: watch::Sender<Value>,
        epoch: u64,
        workspace: Bound,
    ) {
        let result = async {
            let mut requests = Vec::new();
            let mut total = 0;
            for request in &values {
                let size = serde_json::to_vec(&request).unwrap().len();
                if size > self.inner.config.max_workflow_body_bytes {
                    return Err(validation("profile request body exceeds the size limit"));
                };
                total += size;
                if total > self.inner.config.max_queued_workflow_bytes {
                    return Err(Fault::new(
                        "queue_full",
                        "profile batch exceeds the queue memory limit",
                    ));
                };
                requests.push(request.clone());
            }
            tx.send_modify(|v| v["status"] = json!("queued"));
            let key = format!(
                "mcp-{}-{}",
                hex(&self.inner.key[..8]),
                references::hash(id.as_bytes())
            );
            let body = if batch { json!({"jobs":requests}) } else { requests[0].clone() };
            let response = loop {
                let response = self
                .client
                .json(
                    reqwest::Method::POST,
                    if batch { "/v1/jobs/batch" } else { "/v1/jobs" },
                    Some(&body),
                    Some(&key),
                )
                .await;
                match response {
                    Ok(response) => break response,
                    Err(error) if crate::imagine::retryable(&error) && !self.inner.stop.is_cancelled() => {
                        // The request may already have been admitted. Retry the exact
                        // same body/key to recover its ID, never create a second job.
                        tx.send_modify(|v| {
                            v["status"] = json!("recovering");
                            v["error"] = json!({"category":"connection","message":"Imagine is unavailable; recovering the same job"});
                        });
                        tokio::time::sleep(Duration::from_secs_f64(self.inner.config.poll_interval.max(0.1))).await;
                    }
                    Err(_) => return Err(Fault::new("submission", "image job failed during submission")),
                }
            };
            tx.send_modify(|v| { v["status"] = json!("queued"); v["error"] = Value::Null; });
            let ids: Vec<String> = if batch {
                response["jobs"]
                    .as_array()
                    .ok_or_else(|| Fault::new("submission", "image batch could not complete"))?
                    .iter()
                    .filter_map(|v| v["id"].as_str().map(str::to_string))
                    .collect()
            } else {
                vec![
                    response["id"]
                        .as_str()
                        .ok_or_else(|| {
                            Fault::new("submission", "image job failed during submission")
                        })?
                        .into(),
                ]
            };
            if ids.len() != values.len() {
                return Err(Fault::new("submission", "image batch could not complete"));
            }
            let mut futures = Vec::new();
            for (index, native_id) in ids.into_iter().enumerate() {
                let this = self.clone();
                let tx = tx.clone();
                let workspace = workspace.clone();
                futures.push(async move {
                    this.finish_item(&native_id, index, batch, &tx, epoch, &workspace).await;
                });
            }
            futures_util::future::join_all(futures).await;
            if batch {
                tx.send_modify(|v| {
                    let succeeded = v["items"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|i| i["status"] == "succeeded")
                        .count();
                    v["status"] = json!(if succeeded == values.len() {
                        "succeeded"
                    } else if succeeded > 0 {
                        "partial"
                    } else {
                        "failed"
                    });
                    if succeeded != values.len() {
                        v["error"] =
                            json!({"category":"batch","message":"one or more images failed"});
                    }
                })
            }
            Ok::<_, Fault>(())
        }
        .await;
        if let Err(e) = result {
            log(
                40,
                "st_proxy.media_jobs",
                if batch {
                    format!("MCP image batch failed: category={}", e.category)
                } else {
                    format!(
                        "MCP image job failed: stage=submission category={}",
                        e.category
                    )
                },
            );
            tx.send_modify(|v| {
                v["status"] = json!("failed");
                v["error"] = json!({"category":e.category,"message":e.message});
                if batch {
                    let err = v["error"].clone();
                    for item in v["items"].as_array_mut().unwrap() {
                        if item["status"] != "succeeded" {
                            item["status"] = json!("failed");
                            item["error"] = err.clone()
                        }
                    }
                }
            });
        }
    }
    async fn finish_item(
        &self,
        id: &str,
        index: usize,
        batch: bool,
        tx: &watch::Sender<Value>,
        epoch: u64,
        workspace: &Bound,
    ) {
        let update = |status: &str, images: Option<Vec<Value>>, error: Option<Value>| {
            tx.send_modify(|v| {
                if batch && status == "generating" {
                    v["status"] = json!("generating")
                };
                let item = if batch { &mut v["items"][index] } else { v };
                item["status"] = json!(status);
                if let Some(images) = images {
                    item["images"] = json!(images)
                }
                item["error"] = error.unwrap_or(Value::Null);
            });
        };
        let mut stage = "generation";
        let mut started = None;
        let mut cancelling = false;
        let mut timed_out = false;
        let result = async {
            loop {
                if self.inner.stop.is_cancelled() || cancelling {
                    let _ = self
                        .client
                        .json(
                            reqwest::Method::POST,
                            &format!("/v1/jobs/{id}/cancel"),
                            None,
                            None,
                        )
                        .await;
                    cancelling = true;
                }
                let job = match self
                    .client
                    .json(reqwest::Method::GET, &format!("/v1/jobs/{id}"), None, None)
                    .await {
                        Ok(job) => job,
                        Err(_) => {
                            // An observation failure is not a failed generation. Keep
                            // owning and monitoring this job, including cancellation.
                            update("recovering", None, Some(json!({"category":"connection","message":"Imagine is unavailable; recovering the same job"})));
                            tokio::time::sleep(Duration::from_secs_f64(self.inner.config.poll_interval.max(0.1))).await;
                            continue;
                        }
                    };
                if job["effective"].is_object() {
                    let mut effective = job["effective"].clone();
                    for key in ["prompt", "negative_prompt", "client_metadata"] {
                        effective.as_object_mut().unwrap().remove(key);
                    }
                    if let Some(sheet) = effective.get_mut("character_sheet").and_then(Value::as_object_mut) {
                        sheet.remove("character_description");
                        sheet.remove("clothing_description");
                    }
                    tx.send_modify(|v| {
                        let item = if batch { &mut v["items"][index] } else { v };
                        item["parameters"] = effective;
                    });
                }
                match job["status"].as_str() {
                    Some("queued") => update("queued", None, None),
                    Some("running" | "cancelling") => {
                        started.get_or_insert_with(std::time::Instant::now);
                        update("generating", None, None)
                    }
                    Some("succeeded") => {
                        if cancelling {
                            return Err(if timed_out {
                                Fault::new("generation", "image job failed during generation")
                            } else {
                                Fault::new("shutdown", "broker is shutting down")
                            });
                        };
                        update("saving", None, None);
                        stage = "result";
                        let asset = job["output"]["assets"][0].clone();
                        let inner = self.inner.clone();
                        let workspace = workspace.clone();
                        let urls = self.clone();
                        let images = tokio::task::spawn_blocking(move || {
                            let _writes = inner.writes.lock().unwrap();
                            if epoch != inner.epoch.load(std::sync::atomic::Ordering::SeqCst) {
                                anyhow::bail!("session ended");
                            }
                            references::workspace_result(&workspace, &asset, |id| urls.download_url(id))
                        }).await
                        .map_err(|_| Fault::new("result", "image job failed during result"))?
                        .map_err(|e| Fault::new("result", e))?;
                        return Ok(images);
                    }
                    Some("failed" | "cancelled" | "interrupted") => {
                        return Err(if timed_out {
                            Fault::new("generation", "image job failed during generation")
                        } else if cancelling {
                            Fault::new("shutdown", "broker is shutting down")
                        } else {
                            Fault::generation(&job)
                        });
                    }
                    _ => {
                        return Err(Fault::new(
                            "generation",
                            "image generation did not complete successfully",
                        ));
                    }
                }
                if !cancelling
                    && started.is_some_and(|t: std::time::Instant| {
                        t.elapsed().as_secs_f64() > self.inner.config.image_timeout
                    })
                {
                    let _ = self
                        .client
                        .json(
                            reqwest::Method::POST,
                            &format!("/v1/jobs/{id}/cancel"),
                            None,
                            None,
                        )
                        .await;
                    cancelling = true;
                    timed_out = true;
                }
                tokio::time::sleep(Duration::from_secs_f64(self.inner.config.poll_interval)).await;
            }
        }
        .await;
        match result {
            Ok(images) => update("succeeded", Some(images), None),
            Err(e) => {
                log(
                    40,
                    "st_proxy.media_jobs",
                    format!(
                        "MCP image job failed: stage={stage} category={}",
                        e.category
                    ),
                );
                update("failed", None, Some(json!(e)));
            }
        }
    }
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn plain(status: u16, text: impl Into<String>) -> Response<Body> {
    Response::builder()
        .status(status)
        .body(Body::from(text.into()))
        .unwrap()
}
fn rpc_error(status: u16, code: i32, message: impl Into<String>) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(
            json!({"jsonrpc":"2.0","id":null,"error":{"code":code,"message":message.into()}})
                .to_string(),
        ))
        .unwrap()
}
fn sse(body: Body) -> Response<Body> {
    Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache, no-transform")
        .header("Connection", "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(body)
        .unwrap()
}
fn legacy_ping() -> axum::body::Bytes {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let seconds = now.as_secs() as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe {
        libc::gmtime_r(&seconds, &mut tm);
    }
    format!(
        ": ping - {:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:06}+00:00\r\n\r\n",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        now.subsec_micros()
    )
    .into()
}
async fn handle(State(mcp): State<Mcp>, req: Request<Body>) -> Response<Body> {
    use subtle::ConstantTimeEq;
    if let Some(token) = &mcp.inner.config.mcp_token {
        let expected = format!("Bearer {token}");
        let actual = req
            .headers()
            .get("authorization")
            .map(|h| h.as_bytes())
            .unwrap_or_default();
        if !bool::from(expected.as_bytes().ct_eq(actual)) {
            return Response::builder()
                .status(401)
                .header("WWW-Authenticate", "Bearer")
                .body(Body::from("Unauthorized"))
                .unwrap();
        }
    }
    let modern = req
        .headers()
        .get("mcp-protocol-version")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| !crate::mcp_protocol::HANDSHAKE.contains(&v));
    let request_headers = req.headers().clone();
    let method = req.method().as_str();
    let content_type = req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if method == "POST"
        && !content_type
            .to_ascii_lowercase()
            .starts_with("application/json")
    {
        return plain(400, "Invalid Content-Type header");
    }
    let port = mcp.inner.config.mcp_port.unwrap();
    let hosts = [format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    if !req
        .headers()
        .get("host")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|h| hosts.iter().any(|v| v == h))
    {
        return plain(421, "Invalid Host header");
    }
    if req.headers().get("origin").is_some_and(|v| {
        !v.to_str()
            .ok()
            .is_some_and(|s| s.is_empty() || hosts.iter().any(|host| s == format!("http://{host}")))
    }) {
        return plain(403, "Invalid Origin header");
    }
    let accept = req
        .headers()
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let accepts = |mime: &str| {
        accept
            .split(',')
            .any(|v| v.split(';').next().unwrap().trim() == mime)
    };
    if modern && method != "POST" {
        return Response::builder()
            .status(405)
            .header("Allow", "POST")
            .body(Body::empty())
            .unwrap();
    }
    if method == "GET" {
        if !accepts("text/event-stream") {
            return rpc_error(
                406,
                -32600,
                "Not Acceptable: Client must accept text/event-stream",
            );
        }
        let stop = mcp.inner.stop.clone();
        let stream = futures_util::stream::unfold(stop, |stop| async move {
            tokio::select! {
                _ = stop.cancelled() => None,
                _ = tokio::time::sleep(Duration::from_secs(15)) => Some((Ok::<_,std::io::Error>(legacy_ping()),stop)),
            }
        });
        return sse(Body::from_stream(stream));
    }

    if method == "DELETE" {
        return rpc_error(
            405,
            -32600,
            "Method Not Allowed: Session termination not supported",
        );
    }
    if method != "POST" {
        let mut r = rpc_error(405, -32600, "Method Not Allowed");
        r.headers_mut()
            .insert("Allow", "GET, POST, DELETE".parse().unwrap());
        return r;
    }
    if !accepts("application/json") || !accepts("text/event-stream") {
        if modern {
            return plain(406, "");
        }
        return rpc_error(
            406,
            -32600,
            "Not Acceptable: Client must accept both application/json and text/event-stream",
        );
    }
    if !content_type
        .split(';')
        .next()
        .unwrap()
        .split(',')
        .any(|p| p.trim() == "application/json")
    {
        return rpc_error(
            415,
            -32600,
            "Unsupported Media Type: Content-Type must be application/json",
        );
    }
    let bytes = match axum::body::to_bytes(req.into_body(), 4 * 1024 * 1024).await {
        Ok(v) => v,
        Err(_) => return plain(413, "Request body too large"),
    };
    let v: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return rpc_error(
                400,
                -32700,
                if modern {
                    "Parse error".into()
                } else {
                    format!("Parse error: {e}")
                },
            );
        }
    };
    if modern && (crate::mcp_protocol::envelope(&v).is_err() || !v["method"].is_string()) {
        return rpc_error(
            400,
            -32600,
            "Body must be a single JSON-RPC request or notification object",
        );
    }
    if modern && let Err(error) = crate::mcp_protocol::modern(&v, &request_headers) {
        return modern_reply(
            json!({"jsonrpc":"2.0","id":v.get("id").unwrap_or(&Value::Null),"error":error}),
        );
    }
    if let Err(e) = crate::mcp_protocol::envelope(&v) {
        return rpc_error(400, -32602, e);
    }
    if modern && v.get("id").is_none() {
        return plain(202, "");
    }
    if v.get("id").is_none() || !v["method"].is_string() {
        return Response::builder()
            .status(StatusCode::ACCEPTED)
            .header("Content-Type", "application/json")
            .body(Body::empty())
            .unwrap();
    }
    let mut future = Box::pin(dispatch_rpc(mcp, v, modern));
    if !modern {
        return sse(Body::from_stream(async_stream_reply(future, false)));
    }
    tokio::select! {
        value = &mut future => modern_reply(value),
        _ = tokio::time::sleep(std::time::Duration::from_secs(15)) => {
            // Long-running calls retain a live transport while the process-owned job runs.
            use futures_util::StreamExt;
            let ping = futures_util::stream::once(async { Ok::<_, std::io::Error>(axum::body::Bytes::from_static(b": ping\r\n\r\n")) });
            let stream = ping.chain(async_stream_reply(future, true));
            Response::builder().header("Content-Type", "text/event-stream")
                .header("Cache-Control", "no-cache, no-transform").header("Connection", "keep-alive")
                .header("X-Accel-Buffering", "no").body(Body::from_stream(stream)).unwrap()
        }
    }
}
fn async_stream_reply(
    future: impl std::future::Future<Output = Value> + Send + 'static,
    modern: bool,
) -> impl futures_util::Stream<Item = Result<axum::body::Bytes, std::io::Error>> {
    futures_util::stream::unfold(
        (Box::pin(future), false),
        move |(mut future, done)| async move {
            if done {
                return None;
            }
            tokio::select! {
                value = &mut future => Some((Ok(format!("event: message\r\ndata: {value}\r\n\r\n").into()), (future,true))),
                _ = tokio::time::sleep(std::time::Duration::from_secs(15)) => Some((Ok(if modern { ": ping\r\n\r\n".into() } else { legacy_ping() }), (future,false))),
            }
        },
    )
}
fn modern_reply(value: Value) -> Response<Body> {
    let status = match value["error"]["code"].as_i64() {
        Some(-32601) => 404,
        Some(-32603) => 500,
        Some(_) => 400,
        None => 200,
    };
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(value.to_string()))
        .unwrap()
}
async fn dispatch_rpc(mcp: Mcp, v: Value, modern: bool) -> Value {
    let id = &v["id"];
    let missing = || json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":"Method not found","data":v["method"]}});
    if modern && matches!(v["method"].as_str(), Some("initialize" | "ping")) {
        return missing();
    }
    let mut result = match v["method"].as_str() {
        Some("server/discover") if modern => {
            json!({"cacheScope":"private","capabilities":{"prompts":{"listChanged":true},"resources":{"listChanged":true,"subscribe":true},"tools":{"listChanged":true}},"instructions":include_str!("../mcp/instructions.txt"),"supportedVersions":[crate::mcp_protocol::MODERN],"ttlMs":0})
        }
        Some("initialize") => {
            let version = v["params"]["protocolVersion"]
                .as_str()
                .filter(|s| crate::mcp_protocol::HANDSHAKE.contains(s))
                .unwrap_or("2025-11-25");
            json!({"protocolVersion":version,"capabilities":{"prompts":{"listChanged":false},"resources":{"listChanged":false,"subscribe":false},"tools":{"listChanged":false}},"serverInfo":{"name":"st-proxy","version":""},"instructions":include_str!("../mcp/instructions.txt")})
        }
        Some("ping") => json!({}),
        Some("resources/list") => json!({"resources":[]}),
        Some("resources/templates/list") => json!({"resourceTemplates":[]}),
        Some("prompts/list") => {
            json!({"prompts":[{"name":"video_interactive","title":"Interactive video",
            "description":"Make a video step by step: settings, story, outline, characters and render, with your answer at each step.",
            "arguments":[{"name":"idea","description":"Your idea for the video, if you have one.","required":false}]}]})
        }
        Some("prompts/get") => match prompt(&v["params"]) {
            Some(result) => result,
            None => {
                return json!({"jsonrpc":"2.0","id":id,"error":{"code":-32602,"message":"Unknown prompt or invalid arguments"}});
            }
        },
        Some("tools/list") => {
            json!({"tools":serde_json::from_str::<Value>(include_str!("../mcp/tools.json")).unwrap()})
        }
        Some("tools/call") => {
            let name = v["params"]["name"].as_str().unwrap_or("");
            match crate::mcp_protocol::tool_arguments(name, &v["params"]["arguments"]) {
                Ok(args) => mcp.call(name, &args).await,
                Err(message) => json!({"content":[{"type":"text","text":message}],"isError":true}),
            }
        }
        _ => return missing(),
    };
    if modern {
        result["resultType"] = json!("complete");
        result["_meta"] =
            json!({"io.modelcontextprotocol/serverInfo":{"name":"st-proxy","version":""}});
        if matches!(
            v["method"].as_str(),
            Some("tools/list" | "resources/list" | "resources/templates/list" | "prompts/list")
        ) {
            result["cacheScope"] = json!("private");
            result["ttlMs"] = json!(0);
        }
    }
    json!({"jsonrpc":"2.0","id":id,"result":result})
}

#[cfg(test)]
mod profile_tests {
    use super::*;
    #[tokio::test]
    async fn offline_imagine_clears_history_and_rejects_recovery() {
        let dir = tempfile::tempdir().unwrap();

        let proof = test_destination(dir.path());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let router = Router::new().route(
            "/v1/media-destination",
            axum::routing::get(move || {
                let p = proof.clone();
                async move { axum::Json(p) }
            }),
        );
        let mock = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let mcp = Mcp::new(Config {
            coordinator_socket: Some(dir.path().join("gpu.sock").to_string_lossy().into()),
            imagine_url: url,
            mcp_output_dir: Some(dir.path().join("output").to_string_lossy().into()),
            ..Default::default()
        })
        .await
        .unwrap();
        mock.abort();
        let _ = mock.await;
        let args = json!({"job_id":"retry","profile":"qwen","parameters":{"prompt":"synthetic"}});
        let expected = json!({"job_id":"retry","status":"succeeded","parameters":{"seed":42},"images":[],"error":null});
        let (result, _) = watch::channel(expected.clone());
        mcp.inner.jobs.lock().unwrap().insert(
            "retry".into(),
            Job {
                fingerprint: mcp.fingerprint(false, &args),
                result,
                task: None,
                count: 1,
                workspace: "ws_own".into(),
            },
        );
        assert_eq!(
            mcp.call("generate", &args).await["structuredContent"]["error"]["category"],
            "privacy"
        );
        assert!(mcp.inner.jobs.lock().unwrap().is_empty());
    }
}

/// An MCP prompt with its arguments filled in.
fn prompt(params: &Value) -> Option<Value> {
    if params["name"] != "video_interactive" {
        return None;
    }
    let idea = match &params["arguments"]["idea"] {
        Value::Null => "",
        Value::String(text) if text.chars().count() <= 4000 => text.trim(),
        _ => return None,
    };
    let idea = if idea.is_empty() {
        "I have no idea yet: start with the settings, then propose premises.".to_owned()
    } else {
        format!("My idea: {idea}")
    };
    let text = include_str!("../mcp/prompts/video_interactive.txt").replace("{idea}", &idea);
    Some(
        json!({"description":"Interactive video","messages":[{"role":"user","content":{"type":"text","text":text.trim_end()}}]}),
    )
}
/// The complete agent guidance. Clients cap server instructions (Claude Code
/// near 4,000 characters), so the instructions stay short and point here.
pub(crate) const GUIDE: &str = include_str!("../mcp/guide.txt");
#[path = "mcp_video.rs"]
mod video;
#[path = "mcp_workspace.rs"]
mod workspace;
pub(crate) use workspace::Bound;

#[cfg(test)]
#[path = "mcp_workspace_tests.rs"]
mod workspace_tests;

#[cfg(test)]
use crate::task_fixtures;

#[cfg(test)]
pub(crate) fn test_destination(root: &std::path::Path) -> Value {
    use crate::media_destination as d;
    assert!(task_fixtures::is_owned(root));
    let config = d::Config {
        root: root.into(),
        mount: task_fixtures::MOUNT.into(),
        source: task_fixtures::SOURCE.into(),
        filesystem: task_fixtures::FILESYSTEM.into(),
        fsid: d::fsid(&d::directory(root).unwrap()).unwrap(),
    };
    json!(d::inspect(&config, &d::System).unwrap().0)
}
