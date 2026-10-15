use crate::{
    config::{Config, baseline_dir},
    contract::JobRequest,
    progress::{SamplingProgress, read_steps},
    storage::{Store, hash},
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
    fn generate<'a>(
        &'a mut self,
        request: &'a JobRequest,
        owner: &'a str,
        store: &'a Store,
    ) -> EngineFuture<'a, Vec<u8>>;
    fn mark_idle(&mut self) {}
    fn begin_job(&mut self, _id: &str) {}
    fn stop(&mut self) -> EngineFuture<'_, ()>;
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineState {
    pub phase: String,
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
        })
    }
    fn phase(&self, phase: &str) {
        self.state.send_modify(|state| {
            state.phase = phase.into();
            state.pid = self.child.as_ref().and_then(Child::id);
            state.progress = None;
        });
    }
    async fn ready(&mut self, request: &JobRequest) -> Result<()> {
        let workflow = request.model_options.qwen_image_2_1.workflow.as_ref();
        let weights = workflow
            .map(|w| serde_json::json!([w.diffusion_model, w.text_encoder, w.vae]).to_string())
            .unwrap_or_default();
        if self
            .loaded_weights
            .as_ref()
            .is_some_and(|current| current != &weights)
        {
            self.stop().await?;
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
        let dir = baseline_dir(&self.config.project);
        let manifest: Value =
            serde_json::from_slice(&tokio::fs::read(dir.join("manifest.json")).await?)?;
        let mut binary = self
            .config
            .project
            .join(manifest["binary_path"].as_str().context("binary path")?);
        let expected = manifest["binary_sha256"].as_str().context("binary hash")?;
        if hash(
            &tokio::fs::read(&binary)
                .await
                .context(EngineError("frozen_binary_unavailable"))?,
        ) != expected
        {
            bail!(EngineError("frozen_binary_hash_mismatch"));
        }
        if request.model == crate::contract::TURBO {
            let compatibility: Value = serde_json::from_slice(
                &tokio::fs::read(
                    self.config
                        .project
                        .join(".imagine/workflow-native/manifest.json"),
                )
                .await
                .context(EngineError("workflow_binary_unavailable"))?,
            )?;
            for (path, key) in [
                ("native/upstream-lock.json", "base_lock_sha256"),
                ("native/workflow/compat.patch", "compat_patch_sha256"),
            ] {
                if hash(&tokio::fs::read(self.config.project.join(path)).await?)
                    != compatibility[key].as_str().unwrap_or("")
                {
                    bail!(EngineError("workflow_profile_hash_mismatch"));
                }
            }
            binary = self
                .config
                .project
                .join(".imagine/workflow-native/build/bin/sd-server");
            if hash(
                &tokio::fs::read(&binary)
                    .await
                    .context(EngineError("workflow_binary_unavailable"))?,
            ) != compatibility["binary_sha256"].as_str().unwrap_or("")
            {
                bail!(EngineError("workflow_binary_hash_mismatch"));
            }
        }
        let env_bytes = tokio::fs::read(dir.join("baseline-environment.json")).await?;
        if hash(&env_bytes)
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
            let path = self
                .config
                .model_root
                .join(rel.as_str().context("model filename")?);
            if !path.is_file() {
                bail!(EngineError("model_weights_missing"));
            }
            command.arg(key).arg(path);
        }
        if workflow.is_some() {
            command
                .arg("--lora-model-dir")
                .arg(self.config.model_root.join("loras"));
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
        command
            .envs(env)
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
    async fn run(&mut self, req: &JobRequest, owner: &str, store: &Store) -> Result<Vec<u8>> {
        self.state
            .send_modify(|state| state.requested_steps = req.sampling.steps);
        self.ready(req).await?;
        self.phase("preparing_inputs");
        let mut refs = Vec::new();
        for id in &req.inputs.reference_images {
            let bytes = store.read_asset(id, owner).await?;
            let bytes = if req.model_options.qwen_image_2_1.workflow.is_some() {
                crate::workflow_image::prepare(
                    &bytes,
                    req.model_options
                        .qwen_image_2_1
                        .reference_max_pixels_per_image,
                )?
            } else {
                bytes
            };
            refs.push(STANDARD.encode(bytes));
        }
        let mut body = json!({"prompt":req.prompt,"negative_prompt":req.negative_prompt,"width":req.width,"height":req.height,
            "seed":req.seed.context("resolved seed")?,"batch_count":1,"sample_params":{"sample_method":req.sampling.sampler,
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
            body["ref_image_args"] = if req.model_options.qwen_image_2_1.workflow.is_some() {
                json!("resize_before_vae=false")
            } else {
                json!(format!(
                    "vae_input_max_pixels={}",
                    req.model_options
                        .qwen_image_2_1
                        .reference_max_pixels_per_image
                ))
            };
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
            if self.child.as_mut().unwrap().try_wait()?.is_some() {
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
                        let reference = store
                            .read_asset(&req.inputs.reference_images[0], owner)
                            .await?;
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
}
impl Backend for Native {
    fn begin_job(&mut self, id: &str) {
        self.state.send_modify(|state| {
            state.job_id = Some(id.into());
            state.phase = "preparing".into();
            state.progress = None;
            state.requested_steps = 0;
        });
    }
    fn mark_idle(&mut self) {
        self.state.send_modify(|state| state.job_id = None);
        self.phase(if self.child.is_some() {
            "ready"
        } else {
            "unloaded"
        });
    }
    fn generate<'a>(
        &'a mut self,
        r: &'a JobRequest,
        o: &'a str,
        s: &'a Store,
    ) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async move {
            tokio::time::timeout(Duration::from_secs(3600), self.run(r, o, s))
                .await
                .context(EngineError("generation_timeout"))?
        })
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async move {
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
        })
    }
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
        || step != state.progress.as_ref().map_or(1, |p| p.completed + 1)
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

#[cfg(test)]
mod tests {
    use super::*;
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
