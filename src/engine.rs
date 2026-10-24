use crate::{
    config::{Config, baseline_dir},
    generation::ImagePass,
    progress::{SamplingProgress, read_steps},
    storage::hash,
};
use anyhow::{Context, Result, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, future::Future, pin::Pin, process::Stdio, time::Duration};
use tokio::{
    process::{Child, Command},
    sync::watch,
};

#[derive(Debug)]
pub struct EngineError(pub &'static str);
impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for EngineError {}

pub type EngineFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;
pub trait Backend: Send {
    fn reference_cache(
        &self,
    ) -> Option<std::sync::Arc<dyn crate::video::reference_cache::Control>> {
        None
    }
    /// Where the H3 worker resolves model files, when H3 is deployed.
    fn h3_model_root(&self) -> Option<std::path::PathBuf> {
        None
    }
    fn generate<'a>(&'a mut self, request: &'a ImagePass) -> EngineFuture<'a, Vec<u8>>;
    fn generate_video<'a>(
        &'a mut self,
        _request: &'a crate::h3_protocol::Request,
    ) -> EngineFuture<'a, crate::short_av::Av> {
        Box::pin(async { anyhow::bail!(EngineError("h3_runtime_unavailable")) })
    }
    fn stage(&mut self, _stage: Option<&str>) {}
    fn mark_idle(&mut self) {}
    fn begin_job(&mut self, _id: &str) {}
    fn stop(&mut self) -> EngineFuture<'_, ()>;
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineState {
    #[serde(skip)]
    pub sampling_preview: std::sync::Arc<crate::sampling_preview::Hub>,
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub h3_available: bool,
    pub phase: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    pub pid: Option<u32>,
    pub job_id: Option<String>,
    pub progress: Option<SamplingProgress>,
    #[serde(skip)]
    pub requested_steps: u32,
}
pub struct Native {
    config: Config,
    child: Option<Child>,
    base: String,
    http: reqwest::Client,
    state: watch::Sender<EngineState>,
    output_reader: Option<tokio::task::JoinHandle<()>>,
    loaded_weights: Option<String>,
    loaded_profile: Option<NativeProfile>,
}
// Selection belongs to inference passes, independent of the public operation.
// Other model families keep their own native profile until separately validated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeProfile {
    Frozen,
    ReferenceCache,
    Workflow,
}
impl NativeProfile {
    fn for_pass(request: &ImagePass, cache_enabled: bool) -> Self {
        match request.model.as_str() {
            crate::contract::TURBO => Self::Workflow,
            crate::contract::MODEL if cache_enabled && !request.references.is_empty() => {
                Self::ReferenceCache
            }
            _ => Self::Frozen,
        }
    }
}
fn reference_cache_enabled(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("auto") => Ok(true),
        Some("off") => Ok(false),
        _ => bail!(EngineError("invalid_reference_cache_mode")),
    }
}
async fn reference_cache_binary(project: &std::path::Path) -> Option<std::path::PathBuf> {
    let binary = project.join("native/reference-cache/bin/sd-server");
    tokio::fs::metadata(&binary)
        .await
        .ok()
        .filter(|m| m.is_file())
        .map(|_| binary)
}
async fn qwen_default_binary(project: &std::path::Path) -> Option<std::path::PathBuf> {
    let binary = project.join("native/qwen-default/bin/sd-server");
    tokio::fs::metadata(&binary)
        .await
        .ok()
        .filter(|m| m.is_file())
        .map(|_| binary)
}
struct NativeExecutable {
    // None explicitly selects the frozen control from its manifest.
    binary: Option<std::path::PathBuf>,
    qwen_default: bool,
    reference_cache: bool,
}
async fn select_native_executable(
    project: &std::path::Path,
    model: &str,
    profile: NativeProfile,
) -> Result<NativeExecutable> {
    if profile == NativeProfile::Workflow {
        let binary = project.join(".imagine/workflow-native/build/bin/sd-server");
        if !tokio::fs::metadata(&binary)
            .await
            .is_ok_and(|m| m.is_file())
        {
            bail!(EngineError("workflow_binary_unavailable"));
        }
        return Ok(NativeExecutable {
            binary: Some(binary),
            qwen_default: false,
            reference_cache: false,
        });
    }
    let qwen = if model == crate::contract::MODEL {
        qwen_default_binary(project).await
    } else {
        None
    };
    let qwen_default = qwen.is_some();
    let binary = if profile == NativeProfile::ReferenceCache && qwen.is_none() {
        reference_cache_binary(project).await
    } else {
        qwen
    };
    let reference_cache = profile == NativeProfile::ReferenceCache && binary.is_some();
    Ok(NativeExecutable {
        binary,
        qwen_default,
        reference_cache,
    })
}
async fn frozen_binary(project: &std::path::Path, manifest: &Value) -> Result<std::path::PathBuf> {
    let binary = project.join(manifest["binary_path"].as_str().context("binary path")?);
    let expected = manifest["binary_sha256"].as_str().context("binary hash")?;
    if hash(
        &tokio::fs::read(&binary)
            .await
            .context(EngineError("frozen_binary_unavailable"))?,
    ) != expected
    {
        bail!(EngineError("frozen_binary_hash_mismatch"));
    }
    Ok(binary)
}
impl Native {
    pub fn new(config: Config, state: watch::Sender<EngineState>) -> Result<Self> {
        Ok(Self {
            config,
            child: None,
            base: String::new(),
            http: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(30))
                .build()?,
            state,
            output_reader: None,
            loaded_weights: None,
            loaded_profile: None,
        })
    }
    fn phase(&self, phase: &str) {
        self.state.send_modify(|state| {
            state.phase = phase.into();
            state.pid = self.child.as_ref().and_then(Child::id);
            state.progress = None;
        });
    }
    async fn ready(&mut self, request: &ImagePass) -> Result<()> {
        if let Some(resolved) = &request.resolved_models {
            resolved
                .verify()
                .map_err(|_| EngineError("model_files_changed"))?;
        }
        let model_root = request
            .resolved_models
            .as_ref()
            .map(|r| &r.root)
            .unwrap_or(&self.config.model_root)
            .clone();
        let cache_enabled = reference_cache_enabled(
            std::env::var("IMAGINE_QWEN_REFERENCE_CACHE")
                .ok()
                .as_deref(),
        )?;
        let profile = NativeProfile::for_pass(request, cache_enabled);
        let workflow = request.model_options.qwen_image_2_1.workflow.as_ref();
        let weights = workflow
            .map(|w| {
                serde_json::json!([
                    model_root,
                    w.diffusion_model,
                    w.text_encoder,
                    w.vae,
                    request.resolved_models
                ])
                .to_string()
            })
            .unwrap_or_else(|| {
                request
                    .resolved_models
                    .as_ref()
                    .map(|r| serde_json::to_string(r).unwrap())
                    .unwrap_or_default()
            });
        if self
            .loaded_profile
            .is_some_and(|current| current != profile)
            || self
                .loaded_weights
                .as_ref()
                .is_some_and(|current| current != &weights)
        {
            self.stop_process().await?;
        }
        if let Some(child) = &mut self.child {
            if child.try_wait()?.is_none() {
                return Ok(());
            }
            self.child = None;
        }
        self.phase("verifying_profile");
        if ["LD_PRELOAD", "QWEN_ACTIVITY_TRACE", "QWEN_COPY_STACKS"]
            .iter()
            .any(|key| std::env::var_os(key).is_some_and(|v| !v.is_empty()))
        {
            bail!(EngineError("profiling_environment_rejected"));
        }
        let selected =
            select_native_executable(&self.config.project, &request.model, profile).await?;
        let dir = baseline_dir(&self.config.project);
        // The manifest still supplies the unchanged model/environment configuration.
        let manifest: Value =
            serde_json::from_slice(&tokio::fs::read(dir.join("manifest.json")).await?)?;
        let frozen = selected.binary.is_none();
        let binary = match &selected.binary {
            Some(binary) => binary.clone(),
            None => frozen_binary(&self.config.project, &manifest).await?,
        };
        let env_bytes = tokio::fs::read(dir.join("baseline-environment.json")).await?;
        if frozen
            && hash(&env_bytes)
                != manifest["files_sha256"]["baseline-environment.json"]
                    .as_str()
                    .unwrap_or("")
        {
            bail!(EngineError("frozen_profile_hash_mismatch"));
        }
        let env: BTreeMap<String, String> = serde_json::from_slice(&env_bytes)?;
        let socket = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = socket.local_addr()?.port();
        let mut command = Command::new(binary);
        let models = if let Some(w) = workflow {
            json!({"--diffusion-model":w.diffusion_model,"--llm":w.text_encoder,"--vae":w.vae})
        } else {
            manifest["models"].clone()
        };
        for (key, rel) in models.as_object().context("model paths")? {
            let path = model_root.join(rel.as_str().context("model filename")?);
            if !path.is_file() {
                bail!(EngineError("model_weights_missing"));
            }
            command.arg(key).arg(path);
        }
        if selected.reference_cache {
            command.args([
                "--model-args",
                "qwen_image_2_1_prefix_cache_device=hybrid,qwen_image_2_1_prefix_cache_prefetch=true",
            ]);
        }
        if workflow.is_some() {
            command
                .arg("--lora-model-dir")
                .arg(model_root.join("loras"));
        }
        command.args([
            "--width",
            "1600",
            "--height",
            "896",
            "--steps",
            "40",
            "--cfg-scale",
            "2.5",
            "--sampling-method",
            "er_sde",
            "--scheduler",
            "beta",
            "--seed",
            "42",
            "--rng",
            "cpu",
            "--backend",
            "cuda0",
            "--auto-fit",
            "on",
            "--sage-attn",
            "--log-level",
            "error",
            "--threads",
            "11",
            "--listen-ip",
            "127.0.0.1",
            "--listen-port",
            &port.to_string(),
            "--conditioning-cache-size",
            "4",
            "--disable-image-metadata",
        ]);
        for (key, _) in std::env::vars_os() {
            let s = key.to_string_lossy();
            if s.starts_with("SD_") || s.starts_with("GGML_") {
                command.env_remove(key);
            }
        }
        command.envs(env);
        self.config
            .private_worker_environment(command.as_std_mut(), "rust")?;
        if selected.qwen_default {
            command.env("SD_QWEN21_MERGE_LORA_INT8", "auto");
            command.env("GGML_CUDA_FUSED_LORA_OUTPUT", "1");
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        // This Linux engine belongs to this server: no surviving CUDA process on parent death.
        let parent = std::process::id() as libc::pid_t;
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(std::io::Error::other("parent exited"));
                }
                Ok(())
            });
        }
        drop(socket);
        self.child = Some(
            command
                .spawn()
                .context(EngineError("native_spawn_failed"))?,
        );
        if let Some(reader) = self.output_reader.take() {
            reader.abort();
        }
        let stdout = self
            .child
            .as_mut()
            .unwrap()
            .stdout
            .take()
            .context("native stdout")?;
        let state = self.state.clone();
        self.output_reader = Some(tokio::spawn(async move {
            read_steps(stdout, |step, total, seconds| {
                state.send_if_modified(|state| update_progress(state, step, total, seconds));
            })
            .await;
        }));
        self.loaded_weights = Some(weights);
        self.loaded_profile = Some(profile);
        self.base = format!("http://127.0.0.1:{port}");
        self.phase("loading_model");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(300);
        loop {
            if self.child.as_mut().unwrap().try_wait()?.is_some() {
                bail!(EngineError("native_startup_failed"));
            }
            if let Ok(r) = self
                .http
                .get(format!("{}/sdcpp/v1/capabilities", self.base))
                .send()
                .await
                && r.status().is_success()
            {
                return Ok(());
            }
            if tokio::time::Instant::now() > deadline {
                bail!(EngineError("native_startup_timeout"));
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
    async fn run(&mut self, req: &ImagePass) -> Result<Vec<u8>> {
        self.state
            .send_modify(|state| state.requested_steps = req.sampling.steps);
        self.ready(req).await?;
        self.submit(req).await
    }
    async fn submit(&mut self, req: &ImagePass) -> Result<Vec<u8>> {
        self.phase("preparing_inputs");
        let mut body = native_body(req)?;
        let mut preview = {
            let state = self.state.borrow();
            state.job_id.as_ref().map(|job| {
                crate::image_sampling_preview::Controller::new(
                    state.sampling_preview.clone(),
                    job.clone(),
                )
            })
        };
        if preview.as_ref().is_some_and(|preview| preview.enabled()) {
            body["preview"] = json!("fast");
            body["preview_interval"] = json!(1);
        }
        self.phase("inference");
        let response: Value = self
            .http
            .post(format!("{}/sdcpp/v1/img_gen", self.base))
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let native_id = response["id"].as_str().context("native job id")?;
        if !native_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            bail!("invalid native job id");
        }
        loop {
            if let Some(child) = &mut self.child
                && child.try_wait()?.is_some()
            {
                bail!(EngineError("native_process_exited"));
            }
            let status: Value = self
                .http
                .get(format!("{}/sdcpp/v1/jobs/{native_id}", self.base))
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            if let Some(preview) = &mut preview {
                preview.update(&status).await;
            }
            match status["status"].as_str() {
                Some("completed") => {
                    self.phase("saving_output");
                    let bytes = STANDARD.decode(
                        status["result"]["images"][0]["b64_json"]
                            .as_str()
                            .context("native output")?,
                    )?;
                    if let Some(w) = &req.model_options.qwen_image_2_1.workflow
                        && (w.color_match_strength > 0.
                            || w.brightness_strength > 0.
                            || w.contrast_strength > 0.)
                    {
                        let reference = req.references.first().context("colour reference")?.clone();
                        let (chroma, brightness, contrast) = (
                            w.color_match_strength,
                            w.brightness_strength,
                            w.contrast_strength,
                        );
                        return tokio::task::spawn_blocking(move || {
                            crate::color_match::apply(
                                &bytes, &reference, chroma, brightness, contrast,
                            )
                        })
                        .await?;
                    }
                    return Ok(bytes);
                }
                Some("failed" | "cancelled") => bail!(EngineError("native_generation_failed")),
                Some("queued" | "generating") => {}
                _ => bail!("invalid native status"),
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    async fn stop_process(&mut self) -> Result<()> {
        self.phase("unloading");
        if let Some(child) = &mut self.child {
            child.kill().await?;
            child.wait().await?;
        }
        self.child = None;
        if let Some(reader) = self.output_reader.take() {
            reader.abort();
        }
        self.phase("unloaded");
        Ok(())
    }
}
impl Backend for Native {
    fn stage(&mut self, stage: Option<&str>) {
        self.state.send_modify(|state| {
            state.stage = stage.map(str::to_owned);
            state.progress = None;
        });
        if stage == Some("compose") {
            self.phase("composing");
        }
    }
    fn begin_job(&mut self, id: &str) {
        self.state.send_modify(|state| {
            state.sampling_preview.begin(id);
            state.job_id = Some(id.into());
            state.stage = None;
            state.phase = "preparing".into();
            state.progress = None;
            state.requested_steps = 0;
        });
    }
    fn mark_idle(&mut self) {
        self.state.send_modify(|state| {
            if let Some(id) = &state.job_id {
                state.sampling_preview.end(id);
            }
            state.job_id = None;
            state.stage = None;
        });
        self.phase(if self.child.is_some() {
            "ready"
        } else {
            "unloaded"
        });
    }
    fn generate<'a>(&'a mut self, r: &'a ImagePass) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async move {
            tokio::time::timeout(Duration::from_secs(3600), self.run(r))
                .await
                .context(EngineError("generation_timeout"))?
        })
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async move {
            if let Some(id) = self.state.borrow().job_id.clone() {
                self.state.borrow().sampling_preview.end(&id);
            }
            self.stop_process().await
        })
    }
}

fn native_body(req: &ImagePass) -> Result<Value> {
    let mut refs = Vec::new();
    for bytes in &req.references {
        let bytes = crate::workflow_image::prepare(
            bytes,
            req.model_options
                .qwen_image_2_1
                .reference_max_pixels_per_image,
        )?;
        refs.push(STANDARD.encode(bytes));
    }
    let mut body = json!({"prompt":req.prompt,"negative_prompt":req.negative_prompt,"width":req.width,"height":req.height,
            "seed":req.seed,"batch_count":1,"sample_params":{"sample_method":req.sampling.sampler,
            "scheduler":req.sampling.scheduler,"sample_steps":req.sampling.steps,"guidance":{"txt_cfg":req.sampling.cfg_scale}},
            "preview":"none","output_format":"png","embed_image_metadata":false});
    if !req.sampling.custom_sigmas.is_empty() {
        body["sample_params"]["custom_sigmas"] = json!(req.sampling.custom_sigmas);
    }
    if let Some(w) = &req.model_options.qwen_image_2_1.workflow {
        body["lora"] = json!(
            w.loras
                .iter()
                .map(|l| json!({"path":l.path,"multiplier":l.strength}))
                .collect::<Vec<_>>()
        );
    }
    if !refs.is_empty() {
        body["ref_images"] = json!(refs);
        body["ref_image_args"] = json!("resize_before_vae=false");
    }
    Ok(body)
}

fn update_progress(state: &mut EngineState, step: u32, total: u32, seconds: f64) -> bool {
    // This profile has VAE tiling disabled, batch=1, and one sampling pass.
    // Only its sequential iteration records qualify; byte-loading progress does not.
    if state.job_id.is_none()
        || !matches!(state.phase.as_str(), "inference" | "sampling")
        || total != state.requested_steps
        || step == 0
        || step > total
        || !seconds.is_finite()
        || seconds <= 0.
        || Some(step)
            != state
                .progress
                .as_ref()
                .map_or(Some(1), |p| p.completed.checked_add(1))
    {
        return false;
    }
    let elapsed = state.progress.as_ref().map_or(0., |p| p.elapsed_seconds) + seconds;
    state.progress = Some(SamplingProgress {
        completed: step,
        total,
        elapsed_seconds: elapsed,
        remaining_seconds: elapsed / step as f64 * (total - step) as f64,
        updated_at_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
    });
    state.phase = if step == total {
        "finishing_image"
    } else {
        "sampling"
    }
    .into();
    true
}

/// Backend-specific admission prevents an unimplemented sampler from falling
/// through to another algorithm. Existing native/Comfy contracts are preserved.
pub fn validate_selection(
    backend: &str,
    model: &str,
    sampling: &crate::contract::Sampling,
) -> Result<()> {
    if matches!(model, crate::contract::BF16 | crate::contract::MOSS) && backend != "rust" {
        bail!(EngineError("unsupported_model"))
    }
    if backend == "rust"
        && (!matches!(sampling.sampler.as_str(), "euler" | "er_sde")
            || (model == crate::contract::MOSS
                && (sampling.sampler != "euler"
                    || !matches!(sampling.scheduler.as_str(), "normal" | "sgm_uniform"))))
    {
        bail!(EngineError("unsupported_sampling"))
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn native_image_preview_cpu_http_preserves_final_png_and_public_job_lifecycle() {
        use axum::{
            Json, Router,
            extract::State,
            routing::{get, post},
        };
        use std::sync::{
            Arc, Mutex,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        };
        struct Fixture {
            polls: AtomicUsize,
            requests: Mutex<Vec<Value>>,
            final_png: String,
            preview_png: String,
            hub: Arc<crate::sampling_preview::Hub>,
            warning_observed: AtomicBool,
        }
        async fn submit(State(f): State<Arc<Fixture>>, Json(body): Json<Value>) -> Json<Value> {
            f.requests.lock().unwrap().push(body);
            f.polls.store(0, Ordering::SeqCst);
            Json(json!({"id":"native-request-not-public-job"}))
        }
        async fn poll(State(f): State<Arc<Fixture>>) -> Json<Value> {
            let poll = f.polls.fetch_add(1, Ordering::SeqCst);
            if f.hub.status().warning.is_some() {
                f.warning_observed.store(true, Ordering::SeqCst);
            }
            Json(match poll {
                0 => json!({"status":"queued"}),
                1 => {
                    json!({"status":"generating","preview":{"pass":0,"step":1,"total_steps":20,"b64_json":"invalid-base64"}})
                }
                2 => {
                    json!({"status":"generating","preview":{"pass":0,"step":2,"total_steps":20,"b64_json":f.preview_png}})
                }
                _ => json!({"status":"completed","result":{"images":[{"b64_json":f.final_png}]}}),
            })
        }
        let final_png = crate::workflow_image::encode(&image::DynamicImage::ImageRgb8(
            image::RgbImage::from_pixel(64, 32, image::Rgb([140, 80, 30])),
        ))
        .unwrap();
        let preview_png = crate::workflow_image::encode(&image::DynamicImage::ImageRgb8(
            image::RgbImage::from_pixel(80, 40, image::Rgb([30, 80, 150])),
        ))
        .unwrap();
        let (state, _receiver) = watch::channel(EngineState::default());
        let hub = state.borrow().sampling_preview.clone();
        let fixture = Arc::new(Fixture {
            polls: AtomicUsize::new(0),
            requests: Mutex::new(vec![]),
            final_png: STANDARD.encode(&final_png),
            preview_png: STANDARD.encode(preview_png),
            hub: hub.clone(),
            warning_observed: AtomicBool::new(false),
        });
        let routes = Router::new()
            .route("/sdcpp/v1/img_gen", post(submit))
            .route("/sdcpp/v1/jobs/{id}", get(poll))
            .with_state(fixture.clone());
        let socket = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = socket.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(socket, routes)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        let config = Config { project:env!("CARGO_MANIFEST_DIR").into(),
            data:"/home/thhel/tmp/.codex-video-night-20261020-a6428d9e/ui-groups-01/synthetic-native-preview-no-disk".into(),
            bind:address,model_root:"/own/unused-synthetic-model-root".into(),private_storage:None };
        let mut native = Native::new(config, state).unwrap();
        native.base = format!("http://{address}");
        // submit() exercises real request/poll/final-output code, without ready()
        // starting a model/native process. Every image remains a RAM fixture.
        for enabled in [false, true] {
            hub.configure(crate::h3_protocol::preview::Settings {
                enabled,
                interval_seconds: 0.001,
                max_resolution: 32,
                ..Default::default()
            })
            .unwrap();
            native.begin_job("public-scheduler-job");
            assert_eq!(native.submit(&synthetic_pass(0)).await.unwrap(), final_png);
            if enabled {
                let clip = hub.snapshot("public-scheduler-job", 0).unwrap();
                assert_eq!(
                    (
                        clip.metadata.width,
                        clip.metadata.height,
                        clip.metadata.frames
                    ),
                    (32, 16, 1)
                );
                assert_eq!(&clip.bytes[..3], [30, 80, 150]);
                assert!(hub.snapshot("native-request-not-public-job", 0).is_none());
                assert!(fixture.warning_observed.load(Ordering::SeqCst));
                assert!(hub.status().warning.is_none());
                assert_eq!(native.submit(&synthetic_pass(0)).await.unwrap(), final_png);
                assert_eq!(
                    hub.snapshot("public-scheduler-job", 1)
                        .unwrap()
                        .metadata
                        .sequence,
                    2
                );
            } else {
                assert!(hub.snapshot("public-scheduler-job", 0).is_none());
            }
            native.mark_idle();
            assert!(hub.status().active_job.is_none());
        }
        let requests = fixture.requests.lock().unwrap().clone();
        assert_eq!(requests[0]["preview"], "none");
        assert_eq!(requests[1]["preview"], "fast");
        assert_eq!(requests[1]["preview_interval"], 1);
        let strip = |mut value: Value| {
            value.as_object_mut().unwrap().remove("preview");
            value.as_object_mut().unwrap().remove("preview_interval");
            value
        };
        assert_eq!(strip(requests[0].clone()), strip(requests[1].clone()));
        native.begin_job("stop-job");
        native.stop().await.unwrap();
        assert!(hub.status().active_job.is_none());
        stop.send(()).unwrap();
        server.await.unwrap();
    }
    #[tokio::test]
    async fn selected_ordinary_executables_do_not_read_a_missing_frozen_control() {
        let root = tempfile::tempdir().unwrap();
        for (relative, model, profile, cached, merged) in [
            (
                "native/qwen-default/bin/sd-server",
                crate::contract::MODEL,
                NativeProfile::ReferenceCache,
                true,
                true,
            ),
            (
                "native/reference-cache/bin/sd-server",
                crate::contract::MODEL,
                NativeProfile::ReferenceCache,
                true,
                false,
            ),
            (
                ".imagine/workflow-native/build/bin/sd-server",
                crate::contract::TURBO,
                NativeProfile::Workflow,
                false,
                false,
            ),
        ] {
            // Each case has its own source root and no historical manifest/binary.
            let case = root.path().join(if merged {
                "default"
            } else if cached {
                "cache"
            } else {
                "workflow"
            });
            let binary = case.join(relative);
            std::fs::create_dir_all(binary.parent().unwrap()).unwrap();
            std::fs::write(&binary, b"own synthetic compatible candidate").unwrap();
            let selected = select_native_executable(&case, model, profile)
                .await
                .unwrap();
            assert_eq!(selected.binary.as_ref(), Some(&binary));
            assert_eq!(selected.reference_cache, cached);
            assert_eq!(selected.qwen_default, merged);
        }
        let selected =
            select_native_executable(root.path(), crate::contract::MODEL, NativeProfile::Frozen)
                .await
                .unwrap();
        assert!(selected.binary.is_none());
    }
    #[tokio::test]
    async fn actually_selected_frozen_control_still_requires_exact_bytes() {
        let root = tempfile::tempdir().unwrap();
        let binary = root.path().join("control");
        let manifest =
            json!({"binary_path":"control", "binary_sha256":hash(b"own frozen control")});
        assert!(frozen_binary(root.path(), &manifest).await.is_err());
        std::fs::write(&binary, b"own frozen control").unwrap();
        assert_eq!(frozen_binary(root.path(), &manifest).await.unwrap(), binary);
        std::fs::write(&binary, b"own different executable").unwrap();
        let error = frozen_binary(root.path(), &manifest).await.unwrap_err();
        assert_eq!(
            error.downcast_ref::<EngineError>().unwrap().0,
            "frozen_binary_hash_mismatch"
        );
    }
    #[test]
    fn progress_accepts_130_steps_in_order_and_still_rejects_wrong_counts() {
        let mut state = EngineState {
            phase: "sampling".into(),
            job_id: Some("own-k2".into()),
            requested_steps: 130,
            ..Default::default()
        };
        assert!(!update_progress(&mut state, 1, 131, 1.));
        for step in 1..=130 {
            assert!(update_progress(&mut state, step, 130, 0.5));
        }
        assert_eq!(state.progress.as_ref().unwrap().completed, 130);
        assert_eq!(state.phase, "finishing_image");
        assert!(!update_progress(&mut state, 130, 130, 0.5));
    }

    use super::*;
    fn synthetic_pass(reference_count: usize) -> ImagePass {
        ImagePass {
            resolved_models: None,
            model: crate::contract::MODEL.into(),
            model_options: Default::default(),
            sampling: Default::default(),
            prompt: "synthetic".into(),
            negative_prompt: String::new(),
            width: 1600,
            height: 896,
            seed: 42,
            references: (0..reference_count).map(|_| vec![0u8].into()).collect(),
        }
    }
    #[test]
    fn reference_cache_selection_uses_model_and_actual_pass_inputs() {
        assert!(reference_cache_enabled(None).unwrap());
        assert!(reference_cache_enabled(Some("auto")).unwrap());
        assert!(!reference_cache_enabled(Some("off")).unwrap());
        assert!(reference_cache_enabled(Some("typo")).is_err());
        assert_eq!(
            NativeProfile::for_pass(&synthetic_pass(0), true),
            NativeProfile::Frozen
        );
        for count in [1, 3, 10] {
            let mut pass = synthetic_pass(count);
            assert_eq!(
                NativeProfile::for_pass(&pass, true),
                NativeProfile::ReferenceCache
            );
            assert_eq!(NativeProfile::for_pass(&pass, false), NativeProfile::Frozen);
            pass.model = crate::contract::TURBO.into();
            assert_eq!(
                NativeProfile::for_pass(&pass, true),
                NativeProfile::Workflow
            );
            pass.model = "future-model".into();
            assert_eq!(NativeProfile::for_pass(&pass, true), NativeProfile::Frozen);
        }
    }
    #[tokio::test]
    async fn ordinary_reference_executable_does_not_need_a_historic_digest() {
        let root = tempfile::tempdir().unwrap();
        assert!(reference_cache_binary(root.path()).await.is_none());
        let binary = root.path().join("native/reference-cache/bin/sd-server");
        std::fs::create_dir_all(binary.parent().unwrap()).unwrap();
        std::fs::write(binary, b"synthetic rebuilt executable").unwrap();
        assert!(reference_cache_binary(root.path()).await.is_some());
    }
    #[tokio::test]
    async fn ordinary_default_executable_does_not_need_a_historic_digest() {
        let root = tempfile::tempdir().unwrap();
        assert!(qwen_default_binary(root.path()).await.is_none());
        let binary = root.path().join("native/qwen-default/bin/sd-server");
        std::fs::create_dir_all(binary.parent().unwrap()).unwrap();
        std::fs::write(binary, b"synthetic rebuilt executable").unwrap();
        assert!(qwen_default_binary(root.path()).await.is_some());
    }
    #[tokio::test]
    async fn changing_reference_profile_stops_the_previous_native_process() {
        // A harmless child exercises the actual reuse/stop path without a GPU.
        for (loaded, count) in [
            (NativeProfile::Frozen, 1),
            (NativeProfile::ReferenceCache, 0),
        ] {
            let root = tempfile::tempdir().unwrap();
            let (config, _lock) = Config::load(
                root.path().into(),
                root.path().join("data"),
                "127.0.0.1:0".parse().unwrap(),
            )
            .unwrap();
            let (state, _) = watch::channel(EngineState::default());
            let mut native = Native::new(config, state).unwrap();
            native.child = Some(
                Command::new("sleep")
                    .arg("60")
                    .kill_on_drop(true)
                    .spawn()
                    .unwrap(),
            );
            native.loaded_profile = Some(loaded);
            native.loaded_weights = Some(String::new());
            // Same profile keeps the running process, even without model files.
            let current_count = if loaded == NativeProfile::Frozen {
                0
            } else {
                1
            };
            native.ready(&synthetic_pass(current_count)).await.unwrap();
            assert!(native.child.is_some());
            // Switching must kill/wait before trying to verify the new profile.
            assert!(native.ready(&synthetic_pass(count)).await.is_err());
            assert!(native.child.is_none());
        }
    }
    #[tokio::test]
    #[ignore = "requires the accepted Qwen default binary and an idle CUDA GPU"]
    async fn qwen_default_live_reference_and_fallback() {
        let project = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        assert!(qwen_default_binary(&project).await.is_some());
        let root = tempfile::tempdir().unwrap();
        let (mut config, _lock) = Config::load(
            project.clone(),
            root.path().join("data"),
            "127.0.0.1:0".parse().unwrap(),
        )
        .unwrap();
        config.model_root = std::env::var("IMAGINE_TEST_MODEL_ROOT")
            .expect("explicit test model root")
            .into();
        let lora = std::env::var("IMAGINE_TEST_LORA").expect("explicit test LoRA");
        let (state, _) = watch::channel(EngineState::default());
        let mut native = Native::new(config, state).unwrap();
        let mut pass = synthetic_pass(0);
        pass.prompt = "A photographic portrait of an adult woman wearing a blue cotton shirt, neutral expression, looking at the camera, soft natural window light, realistic skin texture, simple light gray background, waist-up composition.".into();
        pass.sampling.cfg_scale = 2.5;
        pass.model_options.qwen_image_2_1.workflow = Some(serde_json::from_value(json!({
            "diffusion_model":"diffusion_models/qwen-image2.1/qwen_image_2.1_int8_convrot.safetensors",
            "text_encoder":"text_encoders/qwen-image2.1/qwen3vl_8b_int8_convrot.safetensors",
            "vae":"vae/qwen-image2.1/qwen_image_2.1_vae_bf16.safetensors",
            "loras":[{"path":lora,"strength":1.0}]
        })).unwrap());
        pass.references = vec![
            crate::workflow_image::encode(&image::DynamicImage::ImageRgb8(
                image::RgbImage::from_pixel(256, 256, image::Rgb([80, 120, 160])),
            ))
            .unwrap()
            .into(),
        ];
        for case in [
            "reference_lora_first",
            "reference_lora_warm",
            "reference_removed",
            "runtime_rank257",
            "merged_return",
        ] {
            if case == "reference_removed" {
                pass.model_options
                    .qwen_image_2_1
                    .workflow
                    .as_mut()
                    .unwrap()
                    .loras
                    .clear();
            }
            if case == "runtime_rank257" {
                pass.references.clear();
                pass.sampling.cfg_scale = 1.0;
                pass.model_options.qwen_image_2_1.workflow.as_mut().unwrap().loras = serde_json::from_value(json!([
                    {"path":project.join(".imagine/qwen-default/synthetic-lora.safetensors.rank257"),"strength":1.0}
                ])).unwrap();
            }
            if case == "merged_return" {
                pass.model_options
                    .qwen_image_2_1
                    .workflow
                    .as_mut()
                    .unwrap()
                    .loras = serde_json::from_value(json!([
                    {"path":lora,"strength":1.0}
                ]))
                .unwrap();
            }
            let start = std::time::Instant::now();
            let result = native.generate(&pass).await;
            let ok = result.is_ok();
            drop(result);
            eprintln!(
                "QWEN_ACCEPTANCE {case} seconds={:.6} success={ok}",
                start.elapsed().as_secs_f64()
            );
            if !ok {
                native.stop().await.unwrap();
            }
            assert!(ok, "controlled Qwen acceptance case failed");
        }
        native.stop().await.unwrap();
    }
    #[test]
    fn native_pass_prepares_every_reference_in_order_without_touching_output_size() {
        use crate::contract::{ModelOptions, Sampling};
        let references = [
            ([255, 0, 0], (96, 64)),
            ([0, 255, 0], (224, 288)),
            ([0, 0, 255], (288, 288)),
        ]
        .into_iter()
        .map(|(color, (w, h))| {
            crate::workflow_image::encode(&image::DynamicImage::ImageRgb8(
                image::RgbImage::from_pixel(w, h, image::Rgb(color)),
            ))
            .unwrap()
            .into()
        })
        .collect();
        let mut options = ModelOptions::default();
        options.qwen_image_2_1.reference_max_pixels_per_image = 4096;
        let pass = ImagePass {
            resolved_models: None,
            model: crate::contract::MODEL.into(),
            model_options: options,
            sampling: Sampling::default(),
            prompt: "synthetic".into(),
            negative_prompt: "synthetic negative".into(),
            width: 288,
            height: 288,
            seed: 43,
            references,
        };
        let body = native_body(&pass).unwrap();
        assert_eq!(body["width"], 288);
        assert_eq!(body["seed"], 43);
        assert_eq!(body["ref_image_args"], "resize_before_vae=false");
        assert_eq!(body["embed_image_metadata"], false);
        for (encoded, color) in body["ref_images"].as_array().unwrap().iter().zip([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ]) {
            let bytes = STANDARD.decode(encoded.as_str().unwrap()).unwrap();
            let image = image::load_from_memory(&bytes).unwrap().to_rgb8();
            assert_eq!(image.width() % 32, 0);
            assert_eq!(image.height() % 32, 0);
            assert!(image.width() * image.height() <= 4096);
            assert_eq!(image.get_pixel(0, 0).0, color);
        }
        // The original portrait remains full resolution for the compositor.
        assert_eq!(
            image::load_from_memory(&pass.references[1])
                .unwrap()
                .width(),
            224
        );
    }
    #[test]
    fn counts_real_steps_and_estimates_only_sampling() {
        let mut state = EngineState {
            job_id: Some("job_a".into()),
            phase: "inference".into(),
            requested_steps: 3,
            ..Default::default()
        };
        assert!(!update_progress(&mut state, 2, 3, 1.));
        assert!(!update_progress(&mut state, 1, 40, 1.));
        assert!(update_progress(&mut state, 1, 3, 2.));
        assert_eq!(state.progress.as_ref().unwrap().remaining_seconds, 4.);
        assert!(!update_progress(&mut state, 1, 3, 2.));
        assert!(update_progress(&mut state, 2, 3, 1.));
        assert_eq!(state.progress.as_ref().unwrap().remaining_seconds, 1.5);
        assert!(update_progress(&mut state, 3, 3, 1.));
        assert_eq!(state.phase, "finishing_image");
        assert_eq!(state.progress.as_ref().unwrap().remaining_seconds, 0.);
        assert!(!update_progress(&mut state, 1, 3, 1.));
        state.phase = "loading_model".into();
        state.progress = None;
        assert!(!update_progress(&mut state, 1, 3, 1.));
    }
}
