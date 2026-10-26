//! Shared private worker supervision for Rust inference and optional Comfy core.
use crate::{
    component_error::{ComponentLoadError, LoadDiagnostic},
    config::Config,
    contract::{self, WorkflowOptions},
    engine::{Backend, EngineError, EngineFuture, EngineState},
    generation::ImagePass,
    progress::SamplingProgress,
};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    path::{Component, Path, PathBuf},
    process::Stdio,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::watch,
};

mod h3;
mod image_preview;

const MAX_HEADER: usize = 1024 * 1024;
const MAX_PAYLOAD: usize = 256 * 1024 * 1024;
const ATTENTION: &str = "comfy_kitchen_int8";

/// Startup-only selection. Neither backend nor attention can change inside a job.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EngineKind {
    #[default]
    Rust,
    Native,
    Comfy,
}
impl EngineKind {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "rust" => Ok(Self::Rust),
            "native" => Ok(Self::Native),
            "comfy" => Ok(Self::Comfy),
            _ => bail!("--backend must be rust, native or comfy"),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Native => "native",
            Self::Comfy => "comfy",
        }
    }
}

pub struct PipeBackend {
    reference_cache: crate::video::reference_cache_owner::Owner,
    config: Config,
    python: PathBuf,
    root: PathBuf,
    script: PathBuf,
    rust_runtime: Option<crate::rust_runtime::Runtime>,
    child: Option<Child>,
    h3_active: bool,
    input: Option<ChildStdin>,
    output: Option<BufReader<ChildStdout>>,
    state: watch::Sender<EngineState>,
}
pub type Comfy = PipeBackend;
impl PipeBackend {
    pub fn new(
        config: Config,
        state: watch::Sender<EngineState>,
        python: PathBuf,
        root: PathBuf,
    ) -> Result<Self> {
        // Validate source/runtime locations only; do not import Python or touch CUDA here.
        // Preserve the venv executable path: resolving its symlink loses pyvenv.cfg.
        let python = std::path::absolute(python).context("Comfy Python executable")?;
        let root = root.canonicalize().context("Comfy source root")?;
        if !python.is_file() || !root.join("comfy/sd.py").is_file() {
            bail!("Invalid Comfy Python executable or source root");
        }
        // A selected immutable package carries its matching Python worker too;
        // never mix a private-storage server with an older checkout bootstrap.
        let packaged = std::env::current_exe()?.with_file_name("comfy-worker/worker.py");
        let script = if packaged.is_file() {
            packaged
        } else {
            config.project.join("workers/comfy/worker.py")
        };
        if !script.is_file() {
            bail!("Comfy worker source missing");
        }
        Ok(Self {
            reference_cache: crate::video::reference_cache_owner::new(),
            config,
            python,
            root,
            script,
            rust_runtime: None,
            child: None,
            h3_active: false,
            input: None,
            output: None,
            state,
        })
    }
    pub fn new_rust(
        config: Config,
        state: watch::Sender<EngineState>,
        manifest: &Path,
    ) -> Result<Self> {
        let runtime = crate::rust_runtime::Runtime::load(manifest)?;
        state.send_modify(|s| s.h3_available = runtime.h3.is_some());
        Ok(Self {
            reference_cache: crate::video::reference_cache_owner::new(),
            config,
            python: runtime.binary.clone(),
            root: PathBuf::new(),
            script: PathBuf::new(),
            rust_runtime: Some(runtime),
            child: None,
            h3_active: false,
            input: None,
            output: None,
            state,
        })
    }
    fn error(&self, comfy: &'static str) -> EngineError {
        EngineError(if self.rust_runtime.is_some() {
            match comfy {
                "comfy_startup_failed" => "rust_startup_failed",
                "comfy_startup_timeout" => "rust_startup_timeout",
                "comfy_protocol_error" => "rust_protocol_error",
                "comfy_invalid_output" => "rust_invalid_output",
                _ => "rust_generation_failed",
            }
        } else {
            comfy
        })
    }
    fn phase(&self, phase: &str) {
        self.state.send_modify(|s| {
            s.phase = phase.into();
            s.pid = self.child.as_ref().and_then(Child::id);
        });
    }
    async fn launch(&mut self, mut command: Command, cache_family: &str) -> Result<()> {
        self.config
            .private_worker_environment(command.as_std_mut(), cache_family)?;
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        let parent = std::process::id() as libc::pid_t;
        // Match the native worker's ownership, including abrupt server death.
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
        let child = command
            .spawn()
            .context(self.error("comfy_startup_failed"))?;
        self.child = Some(child);
        let child = self.child.as_mut().unwrap();
        self.input = child.stdin.take();
        self.output = child.stdout.take().map(BufReader::new);
        Ok(())
    }
    async fn ready(&mut self) -> Result<()> {
        if self.h3_active {
            self.stop().await?;
        }
        if let Some(child) = self.child.as_mut() {
            if child.try_wait()?.is_none() {
                return Ok(());
            }
            self.stop().await?;
        }
        if let Some(runtime) = &self.rust_runtime {
            runtime
                .verify()
                .context(self.error("comfy_startup_failed"))?;
        }
        let mut command = Command::new(&self.python);
        if let Some(runtime) = &self.rust_runtime {
            let cache = self.config.data.join("rust-cache");
            command
                .envs(&runtime.environment)
                .env("IMAGINE_WORKER_CACHE", &cache)
                .env("CUDA_CACHE_PATH", cache.join("cuda"));
        } else {
            command
                .args(["-I", "-u"])
                .arg(&self.script)
                .arg("--comfy-root")
                .arg(&self.root)
                .current_dir(&self.root);
        }
        self.launch(command, "rust").await?;
        self.phase("loading_model");
        let (event, body) = tokio::time::timeout(
            Duration::from_secs(300),
            read_frame(self.output.as_mut().unwrap()),
        )
        .await
        .context(self.error("comfy_startup_timeout"))??;
        if !body.is_empty() {
            bail!(EngineError("comfy_protocol_error"));
        }
        match event {
            Event::Ready {
                protocol: 1,
                attention,
            } if attention
                == if self.rust_runtime.is_some() {
                    "checkpoint_preferences"
                } else {
                    ATTENTION
                } =>
            {
                Ok(())
            }
            Event::Error { code, .. } => bail!(self.error(worker_error(&code))),
            _ => bail!(EngineError("comfy_protocol_error")),
        }
    }
    async fn run(&mut self, request: &ImagePass) -> Result<Vec<u8>> {
        if !(matches!(request.model.as_str(), contract::MODEL | contract::TURBO)
            || self.rust_runtime.is_some()
                && matches!(
                    request.model.as_str(),
                    contract::BF16 | contract::MOSS | contract::MOSS_BF16
                ))
        {
            bail!(EngineError("unsupported_model"));
        }
        if let Some(models) = &request.resolved_models {
            models
                .verify()
                .context(EngineError("model_files_changed"))?;
        }
        let (mut header, mut payload) = request_frame(request, &self.config.model_root)?;
        if self.rust_runtime.is_some() {
            crate::engine::validate_selection("rust", &request.model, &request.sampling)?;
            header["model"] = json!(request.model);
            let mut raw = Vec::new();
            let mut shapes = Vec::new();
            let mut lengths = Vec::new();
            let mut offset = 0;
            for n in header["reference_lengths"].as_array().unwrap() {
                let end = offset + n.as_u64().unwrap() as usize;
                let image = image::load_from_memory(&payload[offset..end])?.to_rgba8();
                shapes.push([image.height(), image.width()]);
                lengths.push(image.as_raw().len());
                raw.extend(image.into_raw());
                offset = end;
                if raw.len() > MAX_PAYLOAD {
                    bail!(EngineError("rust_input_too_large"))
                }
            }
            header["reference_shapes"] = json!(shapes);
            header["reference_lengths"] = json!(lengths);
            payload = raw;
        }
        self.state.send_modify(|s| {
            s.requested_steps = request.sampling.steps;
            s.progress = None;
        });
        self.ready().await?;
        let mut preview = image_preview::Context::new(&self.state.borrow());
        if let Some(preview) = &preview {
            header["sampling_preview"] = json!(preview.settings);
        }
        let id = header["id"].as_str().unwrap();
        write_frame(self.input.as_mut().unwrap(), &header, &payload).await?;
        let mut completed = 0;
        loop {
            let (event, body) = read_frame(self.output.as_mut().unwrap()).await?;
            if !matches!(event, Event::Result { .. } | Event::Preview { .. }) && !body.is_empty() {
                bail!(EngineError("comfy_protocol_error"));
            }
            match event {
                Event::Preview {
                    id: event_id,
                    metadata,
                } if event_id == id => {
                    if let Some(preview) = &mut preview {
                        preview.receive(metadata, body, request.sampling.steps);
                    }
                }
                Event::PreviewWarning { id: event_id } if event_id == id => {
                    if let Some(preview) = &preview {
                        preview.failed();
                    }
                }
                Event::Phase {
                    id: event_id,
                    phase,
                } if event_id == id
                    && matches!(
                        phase.as_str(),
                        "loading_model"
                            | "preparing_inputs"
                            | "encoding"
                            | "inference"
                            | "saving_output"
                    ) =>
                {
                    self.phase(&phase)
                }
                Event::Progress {
                    id: event_id,
                    completed: step,
                    total,
                    elapsed_seconds,
                } if event_id == id
                    && total == request.sampling.steps
                    && step > completed
                    && step <= total
                    && elapsed_seconds.is_finite()
                    && elapsed_seconds >= 0. =>
                {
                    completed = step;
                    self.state.send_modify(|s| {
                        s.progress = Some(SamplingProgress {
                            completed: step,
                            total,
                            elapsed_seconds,
                            remaining_seconds: elapsed_seconds / f64::from(step)
                                * f64::from(total - step),
                            updated_at_ms: SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64,
                        })
                    });
                }
                Event::Result { id: event_id } if event_id == id => {
                    let body = if self.rust_runtime.is_some() {
                        let channels = if crate::model_spec::is_moss(&request.model) {
                            3
                        } else {
                            4
                        };
                        if body.len() != request.width as usize * request.height as usize * channels
                        {
                            bail!(EngineError("rust_invalid_output"))
                        }
                        let mut png = std::io::Cursor::new(Vec::new());
                        image::write_buffer_with_format(
                            &mut png,
                            &body,
                            request.width,
                            request.height,
                            if channels == 3 {
                                image::ColorType::Rgb8
                            } else {
                                image::ColorType::Rgba8
                            },
                            image::ImageFormat::Png,
                        )?;
                        png.into_inner()
                    } else {
                        body
                    };
                    let info = crate::storage::inspect_image(body.clone())
                        .context(self.error("comfy_invalid_output"))?;
                    if (info.width, info.height) != (request.width, request.height)
                        || !body.starts_with(b"\x89PNG\r\n\x1a\n")
                    {
                        bail!(EngineError("unexpected_output_dimensions"));
                    }
                    if let Some(w) = &request.model_options.qwen_image_2_1.workflow
                        && (w.color_match_strength > 0.
                            || w.brightness_strength > 0.
                            || w.contrast_strength > 0.)
                    {
                        let reference = request
                            .references
                            .first()
                            .context("colour reference")?
                            .clone();
                        let (chroma, brightness, contrast) = (
                            w.color_match_strength,
                            w.brightness_strength,
                            w.contrast_strength,
                        );
                        return tokio::task::spawn_blocking(move || {
                            crate::color_match::apply(
                                &body, &reference, chroma, brightness, contrast,
                            )
                        })
                        .await?;
                    }
                    return Ok(body);
                }
                Event::Error {
                    id: Some(event_id),
                    code,
                    diagnostic,
                } if event_id == id => {
                    if code == "model_component_load_failed"
                        && let Some(diagnostic) = diagnostic.filter(LoadDiagnostic::valid)
                    {
                        let root = request
                            .resolved_models
                            .as_ref()
                            .map_or(self.config.model_root.as_path(), |m| m.root.as_path());
                        if let Some(file) = header[diagnostic.component.field()]
                            .as_str()
                            .and_then(|path| Path::new(path).strip_prefix(root).ok())
                            .and_then(Path::to_str)
                        {
                            let failure = ComponentLoadError {
                                diagnostic,
                                file: file.to_owned(),
                            };
                            if failure.valid() {
                                return Err(failure.into());
                            }
                        }
                    }
                    bail!(self.error(worker_error(&code)));
                }
                _ => bail!(EngineError("comfy_protocol_error")),
            }
        }
    }
}
impl Backend for PipeBackend {
    fn h3_model_root(&self) -> Option<std::path::PathBuf> {
        self.rust_runtime
            .as_ref()
            .and_then(|r| r.h3.as_ref())
            .map(|h3| h3.model_root.clone())
    }
    fn reference_cache(
        &self,
    ) -> Option<std::sync::Arc<dyn crate::video::reference_cache::Control>> {
        self.rust_runtime
            .as_ref()
            .and_then(|r| r.h3.as_ref())
            .map(|_| {
                self.reference_cache.clone()
                    as std::sync::Arc<dyn crate::video::reference_cache::Control>
            })
    }
    fn generate_video<'a>(
        &'a mut self,
        request: &'a crate::h3_protocol::Request,
    ) -> EngineFuture<'a, crate::short_av::Av> {
        Box::pin(async move {
            tokio::time::timeout(Duration::from_secs(3600), self.run_video(request))
                .await
                .map_err(|_| EngineError("generation_timeout"))?
        })
    }
    fn generate<'a>(&'a mut self, request: &'a ImagePass) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async move {
            tokio::time::timeout(Duration::from_secs(3600), self.run(request))
                .await
                .context(EngineError("generation_timeout"))?
                .map_err(|error| {
                    if self.rust_runtime.is_some()
                        && let Some(EngineError(code)) = error.downcast_ref::<EngineError>()
                        && code.starts_with("comfy_")
                    {
                        self.error(code).into()
                    } else {
                        error
                    }
                })
        })
    }
    fn stage(&mut self, stage: Option<&str>) {
        self.state.send_modify(|s| {
            s.stage = stage.map(str::to_owned);
            s.progress = None;
        });
        if stage == Some("compose") {
            self.phase("composing");
        }
    }
    fn begin_job(&mut self, id: &str) {
        self.state.borrow().sampling_preview.begin(id);
        self.state.send_modify(|s| {
            s.job_id = Some(id.into());
            s.stage = None;
            s.progress = None;
            s.requested_steps = 0;
        });
        self.phase("preparing");
    }
    fn mark_idle(&mut self) {
        if let Some(id) = &self.state.borrow().job_id {
            self.state.borrow().sampling_preview.end(id);
        }
        self.state.send_modify(|s| {
            s.job_id = None;
            s.stage = None;
        });
        self.phase(if self.child.is_some() {
            "ready"
        } else {
            "unloaded"
        });
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async move {
            self.state.borrow().sampling_preview.clear();
            self.phase("unloading");
            if let Some(child) = &mut self.child {
                if child.try_wait()?.is_none() {
                    child.start_kill()?;
                }
                child.wait().await?;
            }
            self.child = None;
            self.h3_active = false;
            self.input = None;
            self.output = None;
            self.state.send_modify(|s| s.progress = None);
            self.phase("unloaded");
            Ok(())
        })
    }
}

fn worker_error(code: &str) -> &'static str {
    match code {
        "comfy_kitchen_unavailable" => "comfy_kitchen_unavailable",
        "comfy_protocol_error" | "rust_protocol_error" => "comfy_protocol_error",
        "comfy_startup_failed" | "rust_startup_failed" => "comfy_startup_failed",
        _ => "comfy_generation_failed",
    }
}
fn relative_file(root: &Path, value: &str) -> Result<PathBuf> {
    let path = Path::new(value);
    if value.is_empty()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        bail!(EngineError("invalid_model_path"));
    }
    Ok(root.join(path))
}
fn request_frame(request: &ImagePass, fallback_root: &Path) -> Result<(Value, Vec<u8>)> {
    let root = request
        .resolved_models
        .as_ref()
        .map_or(fallback_root, |m| m.root.as_path());
    let defaults;
    let options = if let Some(w) = request.model_options.generation(&request.model) {
        w
    } else {
        let profile: Value =
            serde_yaml::from_str(crate::config_files::text("image/profiles/qwen.yaml")?)?;
        defaults = serde_json::from_value::<WorkflowOptions>(
            profile["defaults"]["model_options"]["qwen_image_2_1"]["generation"].clone(),
        )?;
        &defaults
    };
    let mut payload = Vec::new();
    let mut lengths = Vec::new();
    for reference in &request.references {
        let prepared = crate::workflow_image::prepare(
            reference,
            request
                .model_options
                .qwen_image_2_1
                .reference_max_pixels_per_image,
        )?;
        if payload.len() + prepared.len() > MAX_PAYLOAD {
            bail!(EngineError("comfy_input_too_large"));
        }
        lengths.push(prepared.len());
        payload.extend(prepared);
    }
    let mut loras = Vec::new();
    for lora in &options.loras {
        // Library/native LoRA paths are relative to the dedicated loras directory.
        loras.push(
            json!({"path":relative_file(&root.join("loras"),&lora.path)?,"strength":lora.strength}),
        );
    }
    Ok((
        json!({"protocol":1,"type":"generate","id":uuid::Uuid::new_v4().to_string(),
        "diffusion_model":relative_file(root,&options.diffusion_model)?,
        "text_encoder":relative_file(root,&options.text_encoder)?,"vae":relative_file(root,&options.vae)?,
        "loras":loras,"zero_negative_conditioning":options.zero_negative_conditioning,
        "prompt":request.prompt,"negative_prompt":request.negative_prompt,
        "width":request.width,"height":request.height,"seed":request.seed,"sampling":request.sampling,
        "reference_lengths":lengths}),
        payload,
    ))
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Event {
    Ready {
        protocol: u32,
        attention: String,
    },
    Phase {
        id: String,
        phase: String,
    },
    Progress {
        id: String,
        completed: u32,
        total: u32,
        elapsed_seconds: f64,
    },
    Preview {
        id: String,
        #[serde(default)]
        metadata: Value,
    },
    PreviewWarning {
        id: String,
    },
    Result {
        id: String,
    },
    Error {
        id: Option<String>,
        code: String,
        #[serde(default)]
        diagnostic: Option<LoadDiagnostic>,
    },
}
async fn read_frame(reader: &mut (impl AsyncRead + Unpin)) -> Result<(Event, Vec<u8>)> {
    async fn read(reader: &mut (impl AsyncRead + Unpin), max: usize) -> Result<Vec<u8>> {
        let size = reader.read_u32().await? as usize;
        if size > max {
            bail!(EngineError("comfy_protocol_error"));
        }
        let mut bytes = vec![0; size];
        reader.read_exact(&mut bytes).await?;
        Ok(bytes)
    }
    let header = read(reader, MAX_HEADER)
        .await
        .context(EngineError("comfy_protocol_error"))?;
    let event = serde_json::from_slice(&header).context(EngineError("comfy_protocol_error"))?;
    let body = read(reader, MAX_PAYLOAD)
        .await
        .context(EngineError("comfy_protocol_error"))?;
    Ok((event, body))
}
async fn write_frame(
    writer: &mut (impl AsyncWrite + Unpin),
    header: &Value,
    payload: &[u8],
) -> Result<()> {
    let header = serde_json::to_vec(header)?;
    if header.len() > MAX_HEADER || payload.len() > MAX_PAYLOAD {
        bail!(EngineError("comfy_input_too_large"));
    }
    writer.write_u32(header.len() as u32).await?;
    writer.write_all(&header).await?;
    writer.write_u32(payload.len() as u32).await?;
    writer.write_all(payload).await?;
    writer.flush().await?;
    Ok(())
}

/// Native's measured profile must never be advertised as Comfy validation.
pub fn model_metadata(mut value: Value, backend: &str) -> Value {
    let owned_model = value["model"]
        .as_str()
        .or_else(|| value["id"].as_str())
        .map(str::to_owned);
    let model = owned_model.as_deref();
    if backend == "comfy" && matches!(model, Some(contract::MODEL | contract::TURBO)) {
        value["engine"] = json!("ComfyUI");
        value["attention"] = json!(ATTENTION);
        value["availability"] = json!("requires_local_comfy_runtime_and_weights");
        value["live_validated"] = json!(false);
        if let Some(constraints) = value.get_mut("constraints") {
            constraints["validated_resolutions"] = json!([]);
            constraints["reference_images"]["locally_validated_count"] = json!(0);
            constraints["resolution_note"] =
                json!("Comfy backend has not been locally model-validated");
        }
    }
    if backend == "rust"
        && matches!(
            model,
            Some(
                contract::MODEL
                    | contract::BF16
                    | contract::TURBO
                    | contract::MOSS
                    | contract::MOSS_BF16
            )
        )
    {
        value["engine"] = json!("Imagine Rust");
        value["attention"] = json!("checkpoint_preferences");
        value["availability"] = json!("requires_local_rust_runtime_and_weights");
        value["live_validated"] = json!(false);
        let spec = crate::model_spec::model(model.unwrap()).unwrap();
        value["constraints"]["samplers"] = json!(spec.architecture.samplers());
        value["constraints"]["schedulers"] = json!(spec.architecture.schedulers());
        value["constraints"]["validated_resolutions"] = json!([]);
        value["constraints"]["resolution_note"] = json!(
            "Selected inference components validated; server integration validation is separate"
        );
    }
    if backend != "rust"
        && matches!(
            model,
            Some(contract::BF16 | contract::MOSS | contract::MOSS_BF16)
        )
    {
        value["availability"] = json!("requires_rust_backend");
        value["available"] = json!(false);
        value["problem"] = json!("requires_rust_backend");
        value["operations"] = json!([]);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ModelOptions, Sampling};

    fn pass(prompt: &str) -> ImagePass {
        ImagePass {
            resolved_models: None,
            model: contract::MODEL.into(),
            model_options: ModelOptions::default(),
            sampling: Sampling::default(),
            prompt: prompt.into(),
            negative_prompt: String::new(),
            width: 64,
            height: 32,
            seed: 42,
            references: vec![],
        }
    }
    #[test]
    fn rust_is_default_and_metadata_does_not_claim_native_validation() {
        assert_eq!(EngineKind::default(), EngineKind::Rust);
        assert_eq!(EngineKind::parse("comfy").unwrap(), EngineKind::Comfy);
        assert!(EngineKind::parse("auto").is_err());
        let native = contract::model(contract::MODEL).unwrap();
        assert_eq!(model_metadata(native.clone(), "native"), native);
        let comfy = model_metadata(native, "comfy");
        assert_eq!(comfy["attention"], ATTENTION);
        assert_eq!(comfy["constraints"]["validated_resolutions"], json!([]));
        assert_eq!(comfy["live_validated"], false);
        let mut library_model = contract::model(contract::MODEL).unwrap();
        library_model["id"] = json!("file-specific-id");
        library_model["model"] = json!(contract::MODEL);
        assert_eq!(model_metadata(library_model, "comfy")["engine"], "ComfyUI");
        let h3 = contract::model("minimax-h3").unwrap();
        assert_eq!(model_metadata(h3.clone(), "comfy"), h3);
    }
    #[test]
    fn lora_paths_use_the_selected_library_loras_directory() {
        let mut request = pass("synthetic");
        let profile: Value =
            serde_yaml::from_str(include_str!("../config/image/profiles/qwen.yaml")).unwrap();
        let mut workflow: WorkflowOptions = serde_json::from_value(
            profile["defaults"]["model_options"]["qwen_image_2_1"]["generation"].clone(),
        )
        .unwrap();
        workflow.loras = vec![
            contract::Lora {
                path: "qwen-image-21/xxx/qwen21_fullbody_pass2_fp16.safetensors".into(),
                strength: 1.0,
            },
            contract::Lora {
                path: "nested folder/another.safetensors".into(),
                strength: 0.5,
            },
        ];
        request.model_options.qwen_image_2_1.workflow = Some(workflow);
        request.resolved_models = Some(crate::model_library::ResolvedModels {
            root: "/selected/library".into(),
            files: vec![],
        });
        let (header, _) = request_frame(&request, Path::new("/unused/default")).unwrap();
        assert_eq!(
            header["loras"],
            json!([
                {"path":"/selected/library/loras/qwen-image-21/xxx/qwen21_fullbody_pass2_fp16.safetensors","strength":1.0},
                {"path":"/selected/library/loras/nested folder/another.safetensors","strength":0.5},
            ])
        );
        request
            .model_options
            .qwen_image_2_1
            .workflow
            .as_mut()
            .unwrap()
            .loras[0]
            .path = "../outside.safetensors".into();
        assert!(request_frame(&request, Path::new("/unused/default")).is_err());
    }
    #[test]
    fn ordered_refs_use_shared_geometry_and_turbo_sigmas_are_unchanged() {
        let mut request = pass("synthetic");
        for (width, height, color) in [(67, 69, [10, 20, 30, 80]), (35, 99, [40, 50, 60, 200])] {
            request.references.push(
                crate::workflow_image::encode(&image::DynamicImage::ImageRgba8(
                    image::RgbaImage::from_pixel(width, height, image::Rgba(color)),
                ))
                .unwrap()
                .into(),
            );
        }
        let profile: Value =
            serde_yaml::from_str(include_str!("../config/image/profiles/quick-draft.yaml"))
                .unwrap();
        request.model = contract::TURBO.into();
        request.sampling = serde_json::from_value(profile["defaults"]["sampling"].clone()).unwrap();
        request.model_options.qwen_image_2_1.workflow = Some(
            serde_json::from_value(
                profile["defaults"]["model_options"]["qwen_image_2_1"]["generation"].clone(),
            )
            .unwrap(),
        );
        let (header, payload) = request_frame(&request, Path::new("/synthetic/models")).unwrap();
        let expected: Vec<f32> = vec![
            1., 0.978453, 0.954180, 0.926626, 0.895080, 0.845148, 0.704534, 0.414568, 0.,
        ];
        assert_eq!(
            serde_json::from_value::<Vec<f32>>(header["sampling"]["custom_sigmas"].clone())
                .unwrap(),
            expected
        );
        assert_eq!(header["sampling"]["steps"], 8);
        assert!(
            header["diffusion_model"]
                .as_str()
                .unwrap()
                .ends_with("qwen_image_2.1_turbo_bf16.safetensors")
        );
        assert_eq!(
            (header["width"].as_u64(), header["height"].as_u64()),
            (Some(64), Some(32))
        );
        let mut offset = 0;
        for (index, (width, height, rgba)) in
            [(64, 64, [10, 20, 30, 80]), (32, 96, [40, 50, 60, 200])]
                .into_iter()
                .enumerate()
        {
            let count = header["reference_lengths"][index].as_u64().unwrap() as usize;
            let image = image::load_from_memory(&payload[offset..offset + count])
                .unwrap()
                .to_rgba8();
            assert_eq!(image.dimensions(), (width, height));
            assert_eq!(image.get_pixel(0, 0).0, rgba);
            offset += count;
        }
        assert_eq!(offset, payload.len());
        for path in ["../outside", "/absolute", ""] {
            assert!(relative_file(Path::new("/root"), path).is_err());
        }
    }
    #[tokio::test]
    async fn bounded_protocol_rejects_oversize_truncated_and_unknown_events() {
        for bytes in [
            ((MAX_HEADER + 1) as u32).to_be_bytes().to_vec(),
            vec![0, 0],
            {
                let mut b = Vec::new();
                write_frame(&mut b,&json!({"type":"ready","protocol":1,"attention":ATTENTION,"unexpected":"private"}),&[]).await.unwrap();
                b
            },
        ] {
            assert!(read_frame(&mut bytes.as_slice()).await.is_err());
        }
        let mut raw = Vec::new();
        write_frame(&mut raw, &json!({"type":"result","id":"synthetic"}), &[])
            .await
            .unwrap();
        raw.truncate(raw.len() - 4);
        raw.extend(((MAX_PAYLOAD + 1) as u32).to_be_bytes());
        assert!(read_frame(&mut raw.as_slice()).await.is_err());
    }
}
