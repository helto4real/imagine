use crate::settings::{Connection, GenerationSettings, UpscaleTarget};
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::{
    io::{Cursor, Read},
    path::{Path, PathBuf},
    time::Duration,
};

pub const MODEL: &str = "qwen-image-2.1-int8";
#[path = "enhancement_client.rs"]
mod enhancement;
#[path = "sampling_preview_client.rs"]
pub(crate) mod sampling_preview;
#[path = "timeline_client.rs"]
mod timeline;
pub use timeline::{TimelineFailure, TimelineSession};
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base: String,
}
#[derive(Debug)]
pub struct Failure {
    pub message: String,
    pub retryable: bool,
}
impl Failure {
    fn retry(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: true,
        }
    }
    fn terminal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            retryable: false,
        }
    }
}
#[derive(Clone)]
pub struct Submission {
    /// Original UI result context. Server derives trusted upscale naming from the owned asset.
    pub upscale_source: Option<PathBuf>,
    pub request: Value,
    pub key: String,
    pub folder: PathBuf,
    pub job: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputRole {
    Image,
    Sheet,
    Portrait,
    Views,
}
impl OutputRole {
    pub fn name(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Sheet => "character_sheet",
            Self::Portrait => "portrait",
            Self::Views => "turnaround",
        }
    }
}
pub struct CompletedAsset {
    pub asset_id: String,
    pub role: OutputRole,
    pub file: PathBuf,
    pub preview: Preview,
}
pub struct Completed {
    pub assets: Vec<CompletedAsset>,
    pub seed: Option<u64>,
}
pub struct Preview {
    pub pixels: Vec<u8>,
    pub width: usize,
    pub height: usize,
}
pub struct Reference {
    pub id: String,
    pub name: String,
    pub preview: Preview,
}
pub enum Progress {
    Submitted(String),
    Status {
        label: String,
        sampling: Option<SamplingProgress>,
    },
}
#[derive(Clone, Debug, serde::Deserialize)]
pub struct SamplingProgress {
    pub completed: u32,
    pub total: u32,
    pub remaining_seconds: f64,
    pub updated_at_ms: u64,
}
impl SamplingProgress {
    pub fn finish_time(&self) -> Option<std::time::SystemTime> {
        if self.completed >= self.total
            || !self.remaining_seconds.is_finite()
            || self.remaining_seconds < 0.
            || self.remaining_seconds > 86_400.
        {
            return None;
        }
        std::time::UNIX_EPOCH
            .checked_add(Duration::from_millis(self.updated_at_ms))?
            .checked_add(Duration::from_secs_f64(self.remaining_seconds))
    }
}
pub(crate) fn job_progress(value: &Value) -> Progress {
    let status = value["status"].as_str().unwrap_or("");
    let sampling = if matches!(status, "running" | "cancelling") {
        serde_json::from_value::<SamplingProgress>(value["progress"].clone())
            .ok()
            .filter(|p| p.total > 0 && p.completed <= p.total)
    } else {
        None
    };
    let label = match status {
        "queued" => "Queued",
        "cancelling" => "Cancelling",
        _ => match value["phase"].as_str() {
            Some("verifying_profile") => "Verifying model",
            Some("loading_model") => "Loading model",
            Some("loading_text") => "Loading text encoder",
            Some("encoding_text") => "Encoding prompt",
            Some("loading_dit") => "Loading video model",
            Some("loading_upscaler") => "Loading latent upscaler",
            Some("upscaling_latents") => "Upscaling video latents",
            Some("loading_video_vae") => "Loading video decoder",
            Some("decoding_video") => "Decoding video",
            Some("loading_audio_vae") => "Loading audio decoder",
            Some("decoding_audio") => "Decoding audio",
            Some("preparing" | "preparing_inputs") => "Preparing inputs",
            Some("sampling") => "Sampling",
            Some("finishing_image") => "Finishing image",
            Some("saving_output") => "Saving output",
            _ => "Generating",
        },
    };
    let label = if status == "running" {
        match value["stage"].as_str() {
            Some("portrait") => format!("Portrait · {label}"),
            Some("turnaround") => format!("Views · {label}"),
            Some("compose") => "Composing".into(),
            Some("encoding_video") => "Encoding video".into(),
            _ => label.into(),
        }
    } else {
        label.into()
    };
    Progress::Status {
        label,
        sampling: if value["stage"] == "compose" {
            None
        } else {
            sampling
        },
    }
}
impl Client {
    async fn prepare_destination(
        &self,
        submission: &mut Submission,
    ) -> std::result::Result<(), Failure> {
        if submission.job.is_none() {
            let proof: crate::video_destination::Proof = serde_json::from_value(
                Self::json(
                    self.get(if submission.request["model"] == "minimax-h3" {
                        "/v1/video-destination"
                    } else {
                        "/v1/media-destination"
                    })
                    .send()
                    .await,
                )
                .await?,
            )
            .map_err(|_| Failure::terminal("Media destination is not configured on the server."))?;
            if !submission.folder.as_os_str().is_empty() && proof.config.root != submission.folder {
                return Err(Failure::terminal(
                    "The UI output folder differs from the server's verified media destination. Configure the same mounted destination before generating.",
                ));
            }
            submission.folder = proof.config.root.clone();
            let (local,fd)=crate::video_destination::inspect(&proof.config,&crate::video_destination::System).map_err(|_|Failure::terminal("The verified media volume is unavailable locally. Remote media storage cannot be verified."))?;
            if local != proof {
                return Err(Failure::terminal(
                    "UI and server do not share the same host and mounted media destination.",
                ));
            }
            crate::video_destination::writable(&fd, 0)
                .map_err(|_| Failure::terminal("The media destination is not writable."))?;
            let value = serde_json::to_value(proof).unwrap();
            if let Some(existing) = submission.request["client_metadata"].get("video_destination")
                && existing != &value
            {
                return Err(Failure::terminal(
                    "Media destination changed. Start a new request.",
                ));
            }
            submission.request["client_metadata"]["video_destination"] = value;
        }
        Ok(())
    }

    /// Admit the whole series before watching individual jobs. Retrying admission
    /// with the same key recovers the original jobs after a lost response.
    pub async fn submit_batch(
        &self,
        submissions: &[Submission],
        key: &str,
    ) -> std::result::Result<Vec<String>, Failure> {
        let mut prepared = submissions.to_vec();
        for submission in &mut prepared {
            self.prepare_destination(submission).await?;
        }
        let body = Self::json(
            self.post("/v1/jobs/batch")
                .header("Idempotency-Key", key)
                .json(&json!({"jobs": prepared.iter().map(|s| &s.request).collect::<Vec<_>>()}))
                .send()
                .await,
        )
        .await?;
        let jobs = body["jobs"]
            .as_array()
            .filter(|jobs| jobs.len() == submissions.len())
            .ok_or_else(|| Failure::retry("Invalid batch response. Retry to retrieve the jobs."))?;
        jobs.iter().map(|job| identifier(&job["id"])).collect()
    }

    pub fn connect(c: &Connection) -> Result<Self> {
        c.validate()?;
        Ok(Self {
            http: reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(30))
                .build()?,
            base: c.url.trim_end_matches('/').into(),
        })
    }
    fn get(&self, path: &str) -> reqwest::RequestBuilder {
        self.http.get(format!("{}{path}", self.base))
    }
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        self.http.post(format!("{}{path}", self.base))
    }
    pub async fn library(
        &self,
        settings: Option<&crate::model_catalog::LibrarySettings>,
        refresh: bool,
    ) -> std::result::Result<crate::model_catalog::LibrarySnapshot, Failure> {
        let response = if let Some(settings) = settings {
            self.http
                .put(format!("{}/v1/model-library", self.base))
                .json(settings)
                .send()
                .await
        } else if refresh {
            self.post("/v1/model-library/refresh").send().await
        } else {
            self.get("/v1/model-library").send().await
        };
        serde_json::from_value(Self::json(response).await?)
            .map_err(|_| Failure::retry("Invalid model catalog from server."))
    }
    pub async fn metadata(&self) -> std::result::Result<Value, Failure> {
        Self::json(self.get(&format!("/v1/models/{MODEL}")).send().await).await
    }
    async fn json(
        response: Result<reqwest::Response, reqwest::Error>,
    ) -> std::result::Result<Value, Failure> {
        let response = response.map_err(|_| {
            Failure::retry("Cannot reach the proxy/API. Start ./start-server.sh and ./start-proxy.sh, then retry.")
        })?;
        let status = response.status();
        let body: Value = response
            .json()
            .await
            .map_err(|_| Failure::retry("Could not read the server response. Retry."))?;
        if !status.is_success() {
            return Err(Failure {
                message: job_error_message(&body["error"]),
                retryable: status.is_server_error(),
            });
        }
        Ok(body)
    }
    pub async fn upload(&self, path: &Path) -> std::result::Result<Reference, Failure> {
        let (_bytes, _mime, preview) =
            read_reference(path).map_err(|e| Failure::terminal(e.to_string()))?;
        let path = std::path::absolute(path)
            .map_err(|_| Failure::terminal("Could not resolve the selected image."))?;
        let value = Self::json(
            self.post("/v1/assets/link")
                .json(&json!({"path":path}))
                .send()
                .await,
        )
        .await?;
        Ok(Reference {
            id: identifier(&value["id"])?,
            name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into(),
            preview,
        })
    }
    pub async fn cancel(&self, job: &str) -> std::result::Result<(), Failure> {
        Self::json(self.post(&format!("/v1/jobs/{job}/cancel")).send().await).await?;
        Ok(())
    }
    async fn wait_for_job(
        &self,
        submission: &mut Submission,
        progress: &mut impl FnMut(Progress),
    ) -> std::result::Result<Value, Failure> {
        let job = match &submission.job {
            Some(id) => id.clone(),
            None => {
                let body = Self::json(
                    self.post("/v1/jobs")
                        .header("Idempotency-Key", &submission.key)
                        .json(&submission.request)
                        .send()
                        .await,
                )
                .await?;
                identifier(&body["id"])?
            }
        };
        submission.job = Some(job.clone());
        progress(Progress::Submitted(job.clone()));
        let result = loop {
            let value = Self::json(self.get(&format!("/v1/jobs/{job}")).send().await).await?;
            match value["status"].as_str() {
                Some("succeeded") => break value,
                Some("failed") => {
                    return Err(Failure::terminal(job_error_message(&value["error"])));
                }
                Some("cancelled") => return Err(Failure::terminal("Generation cancelled.")),
                Some("interrupted") => {
                    return Err(Failure::terminal(
                        "The server stopped during generation. Start a new job.",
                    ));
                }
                Some("queued" | "running" | "cancelling") => progress(job_progress(&value)),
                _ => {
                    return Err(Failure::retry(
                        "Unknown job status. Retry to retrieve the job.",
                    ));
                }
            }
            tokio::time::sleep(Duration::from_millis(400)).await;
        };
        Ok(result)
    }
    pub async fn generate(
        &self,
        mut submission: Submission,
        mut progress: impl FnMut(Progress),
    ) -> std::result::Result<Completed, Failure> {
        self.prepare_destination(&mut submission).await?;
        let result = self.wait_for_job(&mut submission, &mut progress).await?;
        progress(Progress::Status {
            label: "Loading saved preview".into(),
            sampling: None,
        });
        let roles = if submission.request["operation"] == "character_sheet" {
            sheet_roles(&result)?
        } else {
            vec![OutputRole::Image]
        };
        let ids = result_assets(&result, &roles)?;
        // Preview bytes stay in memory; publication and naming already completed on the server.
        let mut downloads = Vec::new();
        for (role, asset) in ids {
            let response = self
                .get(&format!("/v1/assets/{asset}"))
                .send()
                .await
                .map_err(|_| {
                    Failure::retry("The result is ready but could not be downloaded. Retry.")
                })?;
            if !response.status().is_success() {
                return Err(Failure::retry(
                    "Could not download the result. Check the connection and retry.",
                ));
            }
            let bytes = response
                .bytes()
                .await
                .map_err(|_| Failure::retry("The image download was interrupted. Retry."))?;
            let preview = decode(&bytes, false).map_err(|_| {
                Failure::retry("Could not decode the image from the server. Retry.")
            })?;
            downloads.push((role, bytes, preview, asset));
        }
        let files = saved_image_paths(
            &result,
            &downloads.iter().map(|v| v.3.as_str()).collect::<Vec<_>>(),
            &submission.folder,
        )?;
        Ok(Completed {
            assets: downloads
                .into_iter()
                .zip(files)
                .map(|((role, _, preview, asset_id), file)| CompletedAsset {
                    asset_id,
                    role,
                    file,
                    preview,
                })
                .collect(),
            seed: result["effective"]["seed"].as_u64(),
        })
    }
}
fn saved_image_paths(
    result: &Value,
    ids: &[&str],
    root: &Path,
) -> std::result::Result<Vec<PathBuf>, Failure> {
    if result["output"]["persistent"] != true {
        return Err(Failure::terminal(
            "The server did not confirm persistent image storage.",
        ));
    }
    let assets = result["output"]["assets"]
        .as_array()
        .ok_or_else(|| Failure::terminal("Missing saved image paths."))?;
    ids.iter()
        .map(|id| {
            let asset = assets
                .iter()
                .find(|a| a["id"].as_str() == Some(id))
                .ok_or_else(|| Failure::terminal("Missing saved image path."))?;
            let path = asset["saved_path"]
                .as_str()
                .map(PathBuf::from)
                .filter(|p| {
                    p.is_absolute()
                        && p.starts_with(root)
                        && p.components().all(|c| {
                            matches!(
                                c,
                                std::path::Component::RootDir | std::path::Component::Normal(_)
                            )
                        })
                })
                .ok_or_else(|| {
                    Failure::terminal("The saved image is outside the verified destination.")
                })?;
            if asset["persistent"] != true {
                return Err(Failure::terminal("The image was not saved persistently."));
            }
            Ok(path)
        })
        .collect()
}

/// The images a sheet job publishes, as its execution plan names them. Jobs
/// from before plans named them are Qwen sheets with all three roles.
fn sheet_roles(result: &Value) -> std::result::Result<Vec<OutputRole>, Failure> {
    let Some(names) = result["effective"]["execution_plan"]["output_roles"].as_array() else {
        return Ok(vec![
            OutputRole::Sheet,
            OutputRole::Portrait,
            OutputRole::Views,
        ]);
    };
    names
        .iter()
        .map(|name| {
            [OutputRole::Sheet, OutputRole::Portrait, OutputRole::Views]
                .into_iter()
                .find(|role| name == role.name())
                .ok_or_else(|| Failure::retry("Unexpected result roles from the server. Retry."))
        })
        .collect()
}
fn result_assets(
    result: &Value,
    roles: &[OutputRole],
) -> std::result::Result<Vec<(OutputRole, String)>, Failure> {
    let assets = result["output"]["assets"]
        .as_array()
        .ok_or_else(|| Failure::retry("The result is incomplete. Retry to retrieve it."))?;
    if assets.len() != roles.len() {
        return Err(Failure::retry(
            "The result is incomplete. Retry to retrieve it.",
        ));
    }
    roles
        .iter()
        .map(|&role| {
            let matches: Vec<_> = assets
                .iter()
                .filter(|a| {
                    a["role"] == role.name() || (role == OutputRole::Image && a["role"].is_null())
                })
                .collect();
            if matches.len() != 1 {
                return Err(Failure::retry(
                    "Unexpected result roles from the server. Retry.",
                ));
            }
            Ok((role, identifier(&matches[0]["id"])?))
        })
        .collect()
}

pub fn make_upscale_request(
    asset: &str,
    target: UpscaleTarget,
    ratio: &str,
    settings: &GenerationSettings,
) -> Result<Value> {
    target.dimensions(ratio)?;
    if identifier(&json!(asset)).is_err() {
        bail!("Generate an image before upscaling.");
    }
    Ok(json!({
        "profile":"image_upscale", "max_side":target.max_side(), "aspect_ratio":ratio,
        "inputs":{"reference_images":[asset]},
        "seed":if settings.random_seed { Value::Null } else { json!(settings.seed) },
    }))
}

/// Whether the model's sheets include this image; unknown models list all three.
pub fn sheet_has_role(metadata: Option<&Value>, role: OutputRole) -> bool {
    metadata
        .and_then(|m| m["character_sheet"]["output_roles"].as_array())
        .is_none_or(|roles| roles.iter().any(|r| r == role.name()))
}
pub fn supports_character_sheets(metadata: &Value) -> bool {
    metadata["operations"]
        .as_array()
        .is_some_and(|ops| ops.iter().any(|op| op == "character_sheet"))
}

pub fn make_sheet_request(
    character: &str,
    clothing: &str,
    width: u32,
    negative: &str,
    refs: &[String],
    metadata: &Value,
    settings: &GenerationSettings,
) -> Result<Value> {
    if !supports_character_sheets(metadata) {
        bail!("This server does not support character sheets. Update the server and reconnect.");
    }
    let max_refs = metadata["character_sheet"]["max_identity_references"]
        .as_u64()
        .unwrap_or(8) as usize;
    if max_refs == 0 && !refs.is_empty() {
        bail!("This model draws sheets from the description alone. Remove the reference images.");
    }
    if character.trim().is_empty() && refs.is_empty() {
        bail!(if max_refs == 0 {
            "Describe the character."
        } else {
            "Describe the character or add a reference image."
        });
    }
    if character.len() > 12000 || clothing.len() > 12000 {
        bail!("Each description must be at most 12,000 UTF-8 bytes.");
    }
    if refs.len() > max_refs {
        bail!("Up to {max_refs} identity references.");
    }
    if !(512..=4096).contains(&width) || !width.is_multiple_of(512) {
        bail!("Width must be 512–4096 px in steps of 512.");
    }
    // Only common sampling/seed settings are shared; the image tab's geometry is independent.
    let settings = GenerationSettings {
        aspect_ratio: "16:9".into(),
        max_side: width,
        ..settings.clone()
    };
    let mut request = make_request("character sheet", negative, refs, metadata, &settings)?;
    request["operation"] = json!("character_sheet");
    for field in ["prompt", "width", "height"] {
        request.as_object_mut().unwrap().remove(field);
    }
    request["character_sheet"] = json!({"character_description":character,"clothing_description":clothing,"layout":{"output_width":width,"portrait_fraction":0.45}});
    Ok(request)
}

fn identifier(value: &Value) -> std::result::Result<String, Failure> {
    let id = value.as_str().filter(|s| {
        !s.is_empty()
            && s.len() < 128
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    });
    id.map(str::to_owned)
        .ok_or_else(|| Failure::retry("Invalid ID from the server."))
}
pub fn make_request(
    prompt: &str,
    negative: &str,
    refs: &[String],
    metadata: &Value,
    settings: &GenerationSettings,
) -> Result<Value> {
    settings.validate()?;
    if prompt.trim().is_empty() {
        bail!("Enter a prompt.");
    }
    if let Some(max) = metadata["constraints"]["reference_images"]["max_count"].as_u64()
        && refs.len() as u64 > max
    {
        if max == 0 {
            bail!("This model takes no reference images. Remove them.");
        }
        bail!("This model takes up to {max} reference images. Remove some.");
    }
    if refs.len() > 10 {
        bail!("Up to ten reference images.");
    }
    let defaults = metadata
        .get("defaults")
        .and_then(Value::as_object)
        .context("The server did not provide default settings.")?;
    if let Some(sigmas) = defaults
        .get("sampling")
        .and_then(|s| s["custom_sigmas"].as_array())
        && !sigmas.is_empty()
        && usize::try_from(settings.steps)
            .ok()
            .and_then(|n| n.checked_add(1))
            != Some(sigmas.len())
    {
        bail!(
            "This model requires {} steps. Update Steps in Settings.",
            sigmas.len() - 1
        );
    }
    if metadata["model"] == "qwen-image-2.1-turbo" && settings.cfg_scale != 1. {
        bail!("Qwen Image 2.1 Turbo requires CFG 1. Update CFG in Settings.");
    }
    let mut request = Value::Object(defaults.clone());
    request["model"] = metadata
        .get("model")
        .or_else(|| metadata.get("id"))
        .cloned()
        .context("Missing model ID.")?;
    request["operation"] = json!(if refs.is_empty() {
        "text_to_image"
    } else {
        "image_edit"
    });
    request["prompt"] = json!(prompt);
    request["negative_prompt"] = json!(negative);
    request["inputs"] = json!({"reference_images":refs});
    let (width, height) = settings.dimensions()?;
    request["width"] = json!(width);
    request["height"] = json!(height);
    request["sampling"]["steps"] = json!(settings.steps);
    request["sampling"]["cfg_scale"] = json!(settings.cfg_scale);
    for (field, constraint, selected) in [
        ("sampler", "samplers", &settings.sampler),
        ("scheduler", "schedulers", &settings.scheduler),
    ] {
        // Manual sigmas replace the scheduler; retain the model's nominal default.
        if field == "scheduler"
            && request["sampling"]["custom_sigmas"]
                .as_array()
                .is_some_and(|s| !s.is_empty())
        {
            continue;
        }
        if let Some(selected) = selected {
            if !metadata["constraints"][constraint]
                .as_array()
                .is_some_and(|choices| choices.iter().any(|c| c.as_str() == Some(selected)))
            {
                bail!("The selected {field} is unavailable for this model. Update Settings.");
            }
            request["sampling"][field] = json!(selected);
        }
    }
    request["model_options"]["qwen_image_2_1"]["reference_max_pixels_per_image"] =
        json!(settings.reference_max_pixels_per_image);
    request["seed"] = if settings.random_seed {
        Value::Null
    } else {
        json!(settings.seed)
    };
    Ok(request)
}
/// Whether the model offers MOSS's modes: text to image, edit and control maps.
pub fn has_moss_modes(metadata: &Value) -> bool {
    metadata["constraints"]["control_maps"].is_object()
}
/// MOSS control maps by asset ID.
#[derive(Clone, Copy, Default)]
pub struct ControlMaps<'a> {
    pub color: Option<&'a str>,
    pub depth: Option<&'a str>,
}
/// A MOSS request in the selected mode: references only when editing (the server
/// adds the edit LoRA), maps and their control only in control mode (the server
/// adds the color/depth adapter).
pub fn make_moss_request(
    prompt: &str,
    negative: &str,
    refs: &[String],
    maps: ControlMaps<'_>,
    metadata: &Value,
    settings: &GenerationSettings,
) -> Result<Value> {
    use crate::settings::MossMode;
    match settings.moss_mode {
        MossMode::TextToImage => make_request(prompt, negative, &[], metadata, settings),
        MossMode::Edit => {
            if refs.is_empty() {
                bail!("Add a reference image to edit.");
            }
            make_request(prompt, negative, refs, metadata, settings)
        }
        MossMode::Control => {
            if maps.color.is_none() && maps.depth.is_none() {
                bail!("Add a depth or region map.");
            }
            let (strength, end) = (settings.control_strength, settings.control_end);
            if !strength.is_finite() || !(0.0..=10.0).contains(&strength) {
                bail!("Control strength must be 0–10.");
            }
            if !end.is_finite() || !(0.0..=1.0).contains(&end) {
                bail!("The control must end between 0 and 100%.");
            }
            let mut request = make_request(prompt, negative, &[], metadata, settings)?;
            if let Some(id) = maps.color {
                request["inputs"]["color_map"] = json!(id);
            }
            if let Some(id) = maps.depth {
                request["inputs"]["depth_map"] = json!(id);
            }
            request["model_options"]["moss_control"] =
                json!({"strength": strength, "start": 0., "end": end});
            Ok(request)
        }
    }
}
pub struct EditSource<'a> {
    pub asset_id: &'a str,
    pub width: u32,
    pub height: u32,
}

#[cfg(test)]
#[path = "edit_request_tests.rs"]
mod edit_request_tests;

/// Bind stable UI reference numbers to Qwen's positional image labels without
/// rewriting the user's instruction. The source always occupies image1.
pub fn make_edit_request(
    prompt: &str,
    negative: &str,
    source: EditSource<'_>,
    references: &[(usize, String)],
    metadata: &Value,
    settings: &GenerationSettings,
) -> Result<Value> {
    if source.asset_id.is_empty() {
        bail!("The source image is missing. Select an available result to edit.");
    }
    if references.len() > 9 {
        bail!("Up to nine extra references; the source image uses one image slot.");
    }
    if !(128..=4096).contains(&source.width)
        || !(128..=4096).contains(&source.height)
        || source.width.max(source.height) < 256
        || !source.width.is_multiple_of(32)
        || !source.height.is_multiple_of(32)
    {
        bail!("The source dimensions are not supported for editing.");
    }
    let mut numbers = std::collections::HashSet::new();
    for (number, asset) in references {
        if *number == 0 || !numbers.insert(*number) || asset.is_empty() {
            bail!("Invalid reference binding. Remove the reference and add it again.");
        }
    }
    let words: Vec<_> = prompt
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|word| !word.is_empty())
        .collect();
    for pair in words.windows(2) {
        if (pair[0].eq_ignore_ascii_case("reference") || pair[0].eq_ignore_ascii_case("referens"))
            && pair[1].bytes().all(|c| c.is_ascii_digit())
            && !pair[1]
                .parse::<usize>()
                .ok()
                .is_some_and(|n| numbers.contains(&n))
        {
            bail!(
                "Reference {} is missing. Update the prompt or add the reference.",
                pair[1]
            );
        }
    }
    let mut ids = vec![source.asset_id.to_owned()];
    ids.extend(references.iter().map(|(_, asset)| asset.clone()));
    let mut request = make_request(prompt, negative, &ids, metadata, settings)?;
    let mut binding = String::from(
        "Edit <image1>, the source image, according to the instruction below. Other images are references whose use is defined by that instruction.\n",
    );
    for (index, (number, _)) in references.iter().enumerate() {
        binding.push_str(&format!(
            "Reference {number} / referens {number} = <image{}>.\n",
            index + 2
        ));
    }
    binding.push_str("\nInstruction:\n");
    binding.push_str(prompt);
    request["prompt"] = json!(binding);
    request["width"] = json!(source.width);
    request["height"] = json!(source.height);
    Ok(request)
}

fn read_reference(path: &Path) -> Result<(Vec<u8>, &'static str, Preview)> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .context("Could not open the reference image.")?
        .take(32 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 32 * 1024 * 1024 {
        bail!("The reference image exceeds 32 MiB.");
    }
    let mime = match image::guess_format(&bytes)? {
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::Jpeg => "image/jpeg",
        image::ImageFormat::WebP => "image/webp",
        _ => bail!("Choose PNG, JPEG or WebP."),
    };
    let preview =
        decode(&bytes, true).context("The reference image is unreadable or too large.")?;
    Ok((bytes, mime, preview))
}
pub(crate) fn load_preview(path: &Path) -> Result<Preview> {
    decode(&std::fs::read(path)?, false)
}

pub(crate) fn thumbnail(preview: &Preview) -> Preview {
    let image = image::RgbaImage::from_raw(
        preview.width as u32,
        preview.height as u32,
        preview.pixels.clone(),
    )
    .expect("decoded RGBA image");
    let rgba = image::DynamicImage::ImageRgba8(image)
        .thumbnail(160, 112)
        .into_rgba8();
    Preview {
        width: rgba.width() as usize,
        height: rgba.height() as usize,
        pixels: rgba.into_raw(),
    }
}

pub(crate) fn decode(bytes: &[u8], thumbnail: bool) -> Result<Preview> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let image = reader.decode()?;
    if u64::from(image.width()) * u64::from(image.height()) > 16_777_216 {
        bail!("Maximum 16 megapixels.");
    }
    let image = if thumbnail {
        image.thumbnail(160, 112)
    } else {
        image
    };
    let rgba = image.to_rgba8();
    Ok(Preview {
        width: rgba.width() as usize,
        height: rgba.height() as usize,
        pixels: rgba.into_raw(),
    })
}
fn job_error_message(error: &Value) -> String {
    let code = error["code"].as_str().unwrap_or("unknown");
    if code == "model_component_load_failed" {
        let details = serde_json::json!({"component":error["component"],"file":error["file"],"reason":error["reason"],"tensor":error["tensor"]});
        if let Ok(details) =
            serde_json::from_value::<crate::component_error::ComponentLoadError>(details)
            && details.valid()
        {
            return details.to_string();
        }
    }
    let mut message = error_message(code);
    // H3 jobs may carry bounded diagnostics: where it failed and what kind of failure.
    let fields = ["stage", "kind", "exception", "accelerator_code", "sizes"];
    let failure = serde_json::Value::Object(
        fields
            .iter()
            .filter(|f| !error[**f].is_null())
            .map(|f| ((*f).to_owned(), error[*f].clone()))
            .collect(),
    );
    if let Ok(failure) = serde_json::from_value::<crate::h3_protocol::Failure>(failure)
        && failure.valid()
        && !failure.is_empty()
    {
        let name = |value: &serde_json::Value| value.as_str().unwrap_or("").replace('_', " ");
        let failure = serde_json::to_value(&failure).unwrap();
        if failure["stage"].is_string() {
            message.push_str(&format!(" Stage: {}.", name(&failure["stage"])));
        }
        if failure["kind"].is_string() {
            message.push_str(&format!(" Cause: {}.", name(&failure["kind"])));
        }
        let mut extra = vec![];
        if let Some(class) = failure["exception"].as_str() {
            extra.push(class.to_owned());
        }
        if let Some(code) = failure["accelerator_code"].as_i64() {
            extra.push(format!("accelerator code {code}"));
        }
        if let Some(sizes) = failure["sizes"].as_array() {
            extra.push(format!("sizes {}", serde_json::Value::Array(sizes.clone())));
        }
        if !extra.is_empty() {
            message.push_str(&format!(" ({})", extra.join(", ")));
        }
    }
    message
}
pub fn error_message(code: &str) -> String {
    if let Some(message) = enhancement::message(code) {
        return message.into();
    }
    match code {
        "source_missing"=>"The original reference file is missing. Use Check Media and Relink to select its current location.",
        "scene_runtime_unsupported"=>"The scene's inputs cannot run on its model. H3 I2V needs exactly one First Frame image and no other inputs.",
        "model_settings_required"=>"Choose generation settings for this scene or the project.",
        "h3_i2v_audio_reference_unavailable"=>"H3 I2V does not take audio references. Use H3 Reference or remove the audio references.",
        "continuity_first_frame_conflict"=>"Last Frame → First Frame continuity already provides the first frame. Remove the First Frame image.",
        "continuity_source_not_accepted"=>"Accept a Take in the source scene before using it for continuity.",
        "continuity_endpoint_incompatible"=>"This continuity method cannot carry audio. Turn off audio continuity.",
        "scene_profile_mismatch"=>"The scene's H3 profile does not match its variant. Choose the model again.",
        "invalid_sampling"=>"The scene's sampling settings are invalid. Review them in Generation settings.",
        "unsupported_h3_geometry"=>"H3 cannot render this canvas size and duration. Change the size or duration.",
        "source_changed"=>"The original reference file changed. Use Check Media and Relink to accept the current file explicitly.",
        "older_format_external_reference_not_representable"=>"The older project format cannot store original-file links. Save the current format; original media will not be copied.",
        "source_unavailable"|"invalid_source_path"|"invalid_source_file"=>"The selected original reference file cannot be opened. Check its location and permissions.",
        "take_requires_video"=>"Choose a file containing a video stream to attach it as a Take.",
        "take_generation_recipe_unknown"=>"This imported Take has no generation recipe. Choose generation settings for a new Take.",
        "older_format_imported_take_not_representable"=>"The older project format cannot represent an imported Take without a generation recipe. The original project is unchanged.",
        "capture_source_missing"=>"The imported Take's source is missing from this project. Choose the video again.",
        "h3_model_file_missing"=>"A model file selected in the generation settings is not in the model folder. Choose an existing file, or clear the field to use the default.",
        "h3_component_load_failed"|"component_load_failed"=>"H3 could not load a selected model component. Check the selected model, text encoder and VAE files (component_load_failed).",
        "h3_out_of_memory"|"out_of_memory"=>"H3 ran out of memory while loading or running the model (out_of_memory).",
        "h3_generation_failed"=>"H3 inference failed (h3_generation_failed). The worker did not identify a more specific component or cause.",
        "imported_scene_not_generatable"=>"This scene is Imported. Choose a source clip, or change its type to generate a Take.",
        "latent_upscale_failed"|"h3_latent_upscale_failed"=>"Trained latent upscaling failed. Check the selected upscaler model and the available memory.",
        "lora_load_failed"|"h3_lora_load_failed"=>"Could not load or apply a selected H3 LoRA. Check its file, adapter format and model compatibility.",
        "upscale_context_unavailable"|"h3_upscale_context_unavailable"=>"Exp upscaling needs its original high-resolution prefix. This worker transport does not provide it yet.",
        "assembly_imported_clip_missing"=>"Choose a video clip for each Imported scene before exporting.",
        "output_save_failed"=>"The result could not be saved. Restore the destination and retry. A completed montage is retained in this server session for saving again.",
        "h3_section_reference_unavailable"=>"The section image bindings are retained, but this H3 worker cannot apply these roles yet. Review the bindings or use a runtime that supports them.",
        "story_dialogue_cast_required"=>"Select at least two project characters to plan dialogue.",
        "story_image_pair_incomplete"=>"Choose a directed start frame and its end frame in order.",
        "story_image_missing"|"cast_reference_missing"=>"A selected Story source is no longer in the project. Select its current reference or image.",
        "assembly_scene_without_take"=>"Choose an Accepted Take or imported video for each scene before assembling.",
        "assembly_transition_take_required"=>"The selected transition has not been generated. Final can join the existing Takes with an explicit warning.",
        "assembly_blend_exceeds_clip"=>"The blend overlap exceeds a neighbouring clip. Shorten the overlap or lengthen the clips.",
        "assembly_empty_source_range"=>"The selected source trim contains no video frames. Adjust Source in / out and freeze a new plan.",
        "assembly_decode_memory_limit"|"assembly_audio_memory_limit"=>"The montage exceeds the processor's resident memory buffer. Use shorter clips or a smaller delivery canvas.",
        "assembly_plan_not_found"=>"This frozen plan belonged to an earlier server session. Freeze the current montage again.",
        "media_decode_failed"=>"The selected media could not be decoded. Check the source file and its codec.",
        "video_destination_unconfigured" => "Media storage is not configured on the server. Configure the encrypted destination before generating.",
        "video_volume_not_mounted" | "video_volume_mismatch" | "video_destination_unavailable" => "The configured media volume is unavailable or does not match. Mount the expected encrypted volume before generating.",
        "video_destination_mismatch" | "video_destination_invalid" => "The UI and server must use the same verified media destination on this computer. Check the output folder.",
        "video_destination_read_only" | "video_destination_not_writable" => "The media destination is read-only or not writable. Check the mounted volume before generating.",
        "video_destination_full" => "The encrypted media destination has insufficient free space. Free space there before retrying.",
        "video_destination_write_failed" => "The output could not be saved on the encrypted destination. No fallback folder was used.",
        "model_component_load_failed" => "A model component could not be loaded. Restart the updated backend and check the selected encoder and VAE files.",
        "older_format_seed_out_of_range" => "The older project format cannot represent one of the exact Take seeds. The original is unchanged; use the current Imagine version for this project.",
        "older_format_boundary_not_representable" => "The older project format cannot represent these boundary settings or the Take's applied conditioning receipt. Save with the current Imagine version; the original is unchanged.",
        "older_format_generation_not_representable" => "The older project format cannot represent this LoRA or trained-upscaling recipe and its Take provenance. Save with the current Imagine format; the original is unchanged.",
        "model_library_update_failed" => "Could not save the model folder or assignments. Check the folder, compatibility and server write permissions.",
        "model_library_scan_failed" => "Could not scan the model folder. Check permissions and folder depth, then refresh.",
        "model_file_missing"|"model_component_missing" => "A model or LoRA file is missing. Refresh Models in Settings and update your selection.",
        "choose_model_components" => "Select the text encoder and VAE in Settings.",
        "incompatible_component"|"incompatible_lora" => "The selected component or LoRA does not belong to this model group.",
        "unsupported_model_file" => "The file does not match a supported model. Select another model or check the files in Settings.",
        "model_catalog_stale" => "A model file changed. Refresh Models in Settings and try again.",
        "invalid_sigma_schedule" => "This model has a fixed sampling schedule. Restore the required step count in Settings.",
        "unsupported_zero_negative_conditioning" => "This model requires CFG 1. Update CFG in Settings.",
        "model_files_changed" => "A selected model file changed after submission. Refresh Models and submit a new job.",
        "coordinator_unavailable" => "The GPU queue is unavailable. Check that Imagine and the proxy use the same coordinator socket." ,
        "imagine_proxy_not_configured" => "Restart the proxy with ./start-proxy.sh to enable Imagine coordination.",
        "imagine_upstream_unavailable" => "The proxy cannot reach Imagine. Start ./start-server.sh and retry.",
        "unsupported_resolution"=>"This server does not accept the selected image size. Restart the updated server or choose a supported size.",
        "unsupported_sampling"=>"Choose a sampler and scheduler supported by this backend, positive steps and finite CFG.",
        "unsupported_moss_configuration"=>"MOSS supports Euler with normal or sgm_uniform and up to four reference images.",
        "unsupported_control_maps"=>"Control maps work with MOSS text to image only. Remove the references or the maps.",
        "invalid_control_options"|"unexpected_control_options"=>"Check the control strength and end, and add a depth or region map.",
        "moss_adapter_missing"=>"The MOSS edit LoRA or color/depth adapter is missing from the model library. Add it under loras/moss and refresh Models.",
        "runtime_suspended"=>"The server is suspended. Resume it through the API before generating.",
        "model_weights_missing"=>"Model files are missing. Check QWEN_MODEL_ROOT on the server.",
        "frozen_binary_unavailable"|"frozen_binary_hash_mismatch"=>"The frozen inference engine is missing or has changed on the server.",
        "native_startup_failed"=>"The inference engine could not start. Check the GPU and model files on the server.",
        "rust_startup_failed"=>"Rust inference could not start. Check the prepared runtime and available GPU memory.",
        "rust_startup_timeout"=>"Rust inference startup timed out.",
        "rust_protocol_error"=>"Rust worker communication failed. Rebuild the Rust runtime and restart Imagine.",
        "rust_generation_failed"=>"Rust inference failed. Check available RAM/VRAM and the selected model files.",
        "rust_invalid_output"=>"Rust inference returned invalid image data.",
        "rust_input_too_large"=>"Reference images exceed the Rust transfer limit. Use fewer or smaller references.",
        "requires_rust_backend"=>"Start Imagine with --backend rust to use MOSS.",
        "comfy_startup_failed"=>"Comfy worker could not start (comfy_startup_failed). Check the selected Python environment and Comfy version.",
        "comfy_startup_timeout"=>"Comfy worker startup timed out (comfy_startup_timeout).",
        "comfy_kitchen_unavailable"=>"Kitchen CUDA attention is unavailable (comfy_kitchen_unavailable). Check Kitchen support in the selected Comfy Python environment.",
        "comfy_protocol_error"=>"Comfy worker communication failed (comfy_protocol_error). Restart Imagine with the updated worker files.",
        "comfy_generation_failed"=>"Comfy inference failed (comfy_generation_failed).",
        "comfy_invalid_output"=>"Comfy returned an unreadable image (comfy_invalid_output).",
        "comfy_input_too_large"=>"Comfy input exceeds the transfer limit (comfy_input_too_large). Use fewer or smaller references.",
        "unexpected_output_dimensions"=>"The inference engine returned the wrong image dimensions (unexpected_output_dimensions).",
        "asset_not_found"=>"A source or reference image is missing on the server. Select a current result or upload the reference again.",
        "invalid_image"=>"The image is unreadable. Choose PNG, JPEG or WebP within the server size limits.",
        "upload_busy"=>"The server is handling other uploads. Retry.",
        "invalid_prompt_length"=>"Text exceeds the allowed length: 12,000 UTF-8 bytes per sheet description or 32,768 per prompt.",
        "character_description_or_reference_required"=>"Describe the character or add a reference image.",
        "invalid_character_sheet_layout"|"conflicting_dimensions"=>"Choose a sheet width of 512–4096 px in steps of 512.",
        "invalid_reference_count"=>"Use at most eight identity references for a character sheet, or ten for an image.",
        "private_prompt_unavailable"=>"Unlock the Helto session used by the upscale profile, then try again.",
        "unknown_profile"=>"The requested generation profile is unavailable. Check the server profiles and reconnect.",
        "unsupported_character_sheet_model"=>"This model does not support character sheets.",
        "generation_failed"|"native_generation_failed"=>"Generation failed. Check available GPU memory and try a smaller image or fewer references.",
        "job_not_found"=>"The job is missing on the server. Start a new generation.",
        "preset_unknown"=>"This preset is no longer in the presets file. Reload the presets.",
        "preset_file_missing"=>"A model or LoRA file this preset names is missing.",
        "preset_key_invalid"=>"The preset key must be 1–64 characters without control characters.",
        "preset_key_duplicate"=>"Another preset already uses this key.",
        "preset_variant_invalid"=>"The preset variant must be reference or i2v.",
        "preset_settings_invalid"=>"The preset's settings are invalid for its variant.",
        "preset_lora_invalid"=>"A preset LoRA needs a relative file name and finite strengths.",
        "video_presets_unreadable"=>"The presets file cannot be read. Only the default presets are available.",
        "video_presets_invalid"=>"The presets file is invalid. Only the default presets are available.",
        "video_presets_unsupported_schema"=>"The presets file uses an unsupported schema_version. Only the default presets are available.",
        _=>"The server could not complete the request. Check the connection and settings."
    }.into()
}

#[cfg(test)]
mod tests {
    #[test]
    fn h3_failures_keep_specific_error_category_and_do_not_guess_generic_cause() {
        for (code, word) in [
            ("h3_component_load_failed", "component_load_failed"),
            ("component_load_failed", "component_load_failed"),
            ("h3_out_of_memory", "out_of_memory"),
            ("out_of_memory", "out_of_memory"),
            ("h3_generation_failed", "h3_generation_failed"),
        ] {
            let message = super::error_message(code);
            assert!(message.contains(word));
            assert!(!message.contains("The server could not complete"));
        }
        assert!(super::error_message("h3_generation_failed").contains("did not identify"));
    }
    #[test]
    fn h3_job_errors_show_bounded_stage_and_cause_but_never_other_text() {
        let message = super::job_error_message(&serde_json::json!({"code":"h3_generation_failed",
            "stage":"preparing_motion_context","kind":"shape_mismatch","exception":"RuntimeError","sizes":[7,8]}));
        assert!(
            message.contains("Stage: preparing motion context."),
            "{message}"
        );
        assert!(message.contains("Cause: shape mismatch."), "{message}");
        assert!(message.contains("(RuntimeError, sizes [7,8])"), "{message}");
        // Unknown or malformed fields drop the diagnostics instead of showing them.
        let message = super::job_error_message(&serde_json::json!({"code":"h3_generation_failed",
            "stage":"sampling","exception":"Not a class /home/x"}));
        assert!(
            !message.contains("Stage") && !message.contains("/home"),
            "{message}"
        );
    }
    #[test]
    fn larger_image_sampling_reaches_the_server_and_keeps_130_step_progress() {
        let settings = GenerationSettings {
            steps: 130,
            cfg_scale: 20.1,
            ..Default::default()
        };
        let metadata = imagine_server::contract::model(imagine_server::contract::MODEL).unwrap();
        let request = make_request("own K2 fixture", "", &[], &metadata, &settings).unwrap();
        let typed: imagine_server::contract::JobRequest = serde_json::from_value(request).unwrap();
        typed.validate().unwrap();
        assert_eq!(typed.sampling.steps, 130);
        assert_eq!(typed.sampling.cfg_scale, 20.1);
        let mut value = json!({"status":"running","phase":"sampling","progress":{"completed":101,"total":130,"remaining_seconds":14.5,"updated_at_ms":100}});
        let Progress::Status {
            sampling: Some(progress),
            ..
        } = job_progress(&value)
        else {
            panic!("130-step progress missing")
        };
        assert_eq!((progress.completed, progress.total), (101, 130));
        value["progress"]["completed"] = json!(131);
        assert!(matches!(
            job_progress(&value),
            Progress::Status { sampling: None, .. }
        ));
    }

    use super::*;
    #[test]
    fn upscale_request_uses_profile_and_one_output_asset_without_generation_overrides() {
        let settings = GenerationSettings {
            steps: 17,
            cfg_scale: 4.,
            aspect_ratio: "1:1".into(),
            max_side: 512,
            random_seed: false,
            seed: 7,
            ..Default::default()
        };
        for target in UpscaleTarget::ALL {
            let request = make_upscale_request("asset_output", target, "3:4", &settings).unwrap();
            assert_eq!(
                request,
                json!({"profile":"image_upscale","max_side":target.max_side(),"aspect_ratio":"3:4","seed":7,"inputs":{"reference_images":["asset_output"]}})
            );
        }
        assert!(make_upscale_request("", UpscaleTarget::K2, "16:9", &settings).is_err());
        assert!(make_upscale_request("asset", UpscaleTarget::K2, "invalid", &settings).is_err());
    }
    #[test]
    fn sheet_request_preserves_descriptions_and_validates_against_server_contract() {
        let meta = imagine_server::contract::model(MODEL).unwrap();
        let settings = GenerationSettings {
            aspect_ratio: "3:4".into(),
            max_side: 1024,
            random_seed: false,
            seed: 17,
            ..Default::default()
        };
        for refs in [vec![], vec!["b".into(), "a".into()], vec!["ref".into(); 8]] {
            for width in [512, 1536, 4096] {
                let request = make_sheet_request(
                    "  Character\nexact  ",
                    "  Outfit  ",
                    width,
                    "negative",
                    &refs,
                    &meta,
                    &settings,
                )
                .unwrap();
                for field in ["prompt", "width", "height"] {
                    assert!(request.get(field).is_none());
                }
                assert_eq!(
                    request["character_sheet"]["character_description"],
                    "  Character\nexact  "
                );
                assert_eq!(
                    request["character_sheet"]["clothing_description"],
                    "  Outfit  "
                );
                assert_eq!(request["inputs"]["reference_images"], json!(refs));
                assert_eq!(request["seed"], 17);
                assert_eq!(
                    request["sampling"]["sampler"],
                    meta["defaults"]["sampling"]["sampler"]
                );
                let request: imagine_server::contract::JobRequest =
                    serde_json::from_value(request).unwrap();
                request.validate().unwrap();
            }
        }
        assert!(make_sheet_request("", "outfit", 1536, "", &[], &meta, &settings).is_err());
        assert!(
            make_sheet_request("", "outfit", 1536, "", &["ref".into()], &meta, &settings).is_ok()
        );
        for width in [0, 511, 1600, 4097] {
            assert!(make_sheet_request("x", "", width, "", &[], &meta, &settings).is_err());
        }
        assert!(
            make_sheet_request("x", "", 1536, "", &vec!["ref".into(); 9], &meta, &settings)
                .is_err()
        );
        assert!(
            make_sheet_request(&"x".repeat(12001), "", 1536, "", &[], &meta, &settings).is_err()
        );
        assert!(make_sheet_request("x", "", 1536, "", &[], &json!({}), &settings).is_err());
    }
    #[test]
    fn moss_sheets_need_a_description_take_no_references_and_have_one_image() {
        let meta = imagine_server::contract::model(imagine_server::contract::MOSS).unwrap();
        let settings = GenerationSettings::default();
        assert!(make_sheet_request("", "outfit", 1536, "", &[], &meta, &settings).is_err());
        assert!(make_sheet_request("x", "", 1536, "", &["ref".into()], &meta, &settings).is_err());
        let request = make_sheet_request("x", "", 1536, "", &[], &meta, &settings).unwrap();
        assert_eq!(request["model"], imagine_server::contract::MOSS);
        let request: imagine_server::contract::JobRequest =
            serde_json::from_value(request).unwrap();
        request.validate().unwrap();
        let plan = request.effective()["execution_plan"].clone();
        assert_eq!(
            sheet_roles(&json!({"effective":{"execution_plan":plan}})).unwrap(),
            [OutputRole::Sheet]
        );
        assert!(sheet_has_role(Some(&meta), OutputRole::Sheet));
        assert!(!sheet_has_role(Some(&meta), OutputRole::Portrait));
        assert!(sheet_has_role(None, OutputRole::Views));
        // Jobs from before plans named their roles are three-image Qwen sheets.
        assert_eq!(
            sheet_roles(&json!({"effective":{}})).unwrap(),
            [OutputRole::Sheet, OutputRole::Portrait, OutputRole::Views]
        );
        let unknown = json!({"effective":{"execution_plan":{"output_roles":["image"]}}});
        assert!(sheet_roles(&unknown).unwrap_err().retryable);
    }
    #[test]
    fn sheet_roles_are_order_independent_and_partial_results_are_retryable() {
        let roles = [OutputRole::Sheet, OutputRole::Portrait, OutputRole::Views];
        let mut result = json!({"output":{"assets":[{"id":"v","role":"turnaround"},{"id":"s","role":"character_sheet"},{"id":"p","role":"portrait"}]}});
        let assets = result_assets(&result, &roles).unwrap();
        assert_eq!(
            assets,
            vec![
                (OutputRole::Sheet, "s".into()),
                (OutputRole::Portrait, "p".into()),
                (OutputRole::Views, "v".into())
            ]
        );
        result["output"]["assets"][0]["role"] = json!("portrait");
        assert!(result_assets(&result, &roles).unwrap_err().retryable);
        result["output"]["assets"].as_array_mut().unwrap().pop();
        assert!(result_assets(&result, &roles).unwrap_err().retryable);
    }
    #[test]
    fn sheet_progress_labels_current_pass_and_clears_sampling_for_composition() {
        for (stage, expected) in [
            ("portrait", "Portrait · Sampling"),
            ("turnaround", "Views · Sampling"),
            ("compose", "Composing"),
        ] {
            let value = json!({"status":"running","phase":"sampling","stage":stage,"progress":{"completed":12,"total":40,"remaining_seconds":20.,"updated_at_ms":100}});
            let Progress::Status { label, sampling } = job_progress(&value) else {
                panic!()
            };
            assert_eq!(label, expected);
            assert_eq!(sampling.is_some(), stage != "compose");
        }
    }
    #[test]
    fn progress_does_not_invent_counts_or_full_job_eta() {
        let value = json!({"status":"running","phase":"loading_model"});
        assert!(matches!(
            job_progress(&value),
            Progress::Status { sampling: None, .. }
        ));
        let mut value = json!({"status":"running","phase":"sampling","progress":{"completed":10,"total":40,"remaining_seconds":24.,"updated_at_ms":100000}});
        let Progress::Status {
            sampling: Some(p), ..
        } = job_progress(&value)
        else {
            panic!("missing progress")
        };
        assert_eq!(
            p.finish_time()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            124
        );
        value["progress"]["completed"] = json!(40);
        let Progress::Status {
            sampling: Some(p), ..
        } = job_progress(&value)
        else {
            panic!("missing progress")
        };
        assert!(p.finish_time().is_none());
        value["status"] = json!("queued");
        assert!(matches!(
            job_progress(&value),
            Progress::Status { sampling: None, .. }
        ));
        value["status"] = json!("running");
        value["progress"]["total"] = json!(0);
        assert!(matches!(
            job_progress(&value),
            Progress::Status { sampling: None, .. }
        ));
    }

    #[test]
    fn moss_modes_send_references_only_to_edit_and_maps_only_to_control() {
        use crate::settings::MossMode;
        let meta = imagine_server::contract::model(imagine_server::contract::MOSS).unwrap();
        assert!(has_moss_modes(&meta));
        assert!(!has_moss_modes(
            &imagine_server::contract::model(MODEL).unwrap()
        ));
        let refs = vec!["a".to_string(), "b".to_string()];
        let maps = ControlMaps {
            color: Some("regions"),
            depth: Some("depth"),
        };
        let mut settings = GenerationSettings {
            max_side: 1024,
            ..Default::default()
        };
        let request = |settings: &GenerationSettings, maps| {
            make_moss_request("test", "", &refs, maps, &meta, settings)
        };
        let check = |request: Value| {
            let request: imagine_server::contract::JobRequest =
                serde_json::from_value(request).unwrap();
            assert_eq!(request.validate(), Ok(()));
            request
        };
        let text = check(request(&settings, maps).unwrap());
        assert_eq!(
            text.operation,
            imagine_server::contract::Operation::TextToImage
        );
        assert!(text.inputs.reference_images.is_empty() && !text.inputs.has_control_maps());
        settings.moss_mode = MossMode::Edit;
        let edit = check(request(&settings, maps).unwrap());
        assert_eq!(
            edit.operation,
            imagine_server::contract::Operation::ImageEdit
        );
        assert_eq!(edit.inputs.reference_images, refs);
        assert!(!edit.inputs.has_control_maps());
        assert!(make_moss_request("test", "", &[], maps, &meta, &settings).is_err());
        assert!(
            make_moss_request("test", "", &vec!["a".into(); 5], maps, &meta, &settings).is_err()
        );
        settings.moss_mode = MossMode::Control;
        settings.control_strength = 0.75;
        settings.control_end = 0.5;
        let control = check(request(&settings, maps).unwrap());
        assert!(control.inputs.reference_images.is_empty());
        assert_eq!(control.inputs.color_map.as_deref(), Some("regions"));
        assert_eq!(control.inputs.depth_map.as_deref(), Some("depth"));
        assert_eq!(
            control.model_options.moss_control,
            Some(imagine_server::contract::ControlOptions {
                strength: 0.75,
                start: 0.,
                end: 0.5,
            })
        );
        assert!(request(&settings, ControlMaps::default()).is_err());
        settings.control_end = 1.5;
        assert!(request(&settings, maps).is_err());
    }

    #[test]
    fn reference_size_reaches_the_contract_and_rejects_invalid_limits() {
        let meta = imagine_server::contract::model(MODEL).unwrap();
        for limit in [262_144, 500_000, 1_000_000, 2_500_000, 4_000_000] {
            let settings = GenerationSettings {
                reference_max_pixels_per_image: limit,
                ..Default::default()
            };
            let request =
                make_request("test", "", &["reference".into()], &meta, &settings).unwrap();
            let request: imagine_server::contract::JobRequest =
                serde_json::from_value(request).unwrap();
            assert_eq!(
                request
                    .model_options
                    .qwen_image_2_1
                    .reference_max_pixels_per_image,
                limit
            );
            assert!(request.validate().is_ok());
        }
        for limit in [0, 262_143, 4_000_001, u32::MAX] {
            let settings = GenerationSettings {
                reference_max_pixels_per_image: limit,
                ..Default::default()
            };
            assert!(make_request("test", "", &[], &meta, &settings).is_err());
        }
    }

    #[test]
    fn request_uses_saved_controls_and_preserves_other_model_defaults() {
        let meta = imagine_server::contract::model(MODEL).unwrap();
        let mut settings = GenerationSettings::default();
        // Explicitly clear any server default seed in random-each-time mode.
        let mut seeded_meta = meta.clone();
        seeded_meta["defaults"]["seed"] = json!(42);
        let request = make_request("test", "", &[], &seeded_meta, &settings).unwrap();
        assert!(request["seed"].is_null());
        assert_eq!(
            (request["width"].as_u64(), request["height"].as_u64()),
            (Some(1600), Some(896))
        );
        assert_eq!(request["sampling"]["steps"], 40);
        assert_eq!(request["sampling"]["cfg_scale"], 2.5);
        settings.random_seed = false;
        settings.steps = 17;
        settings.cfg_scale = 4.;
        for seed in [0, u32::MAX] {
            settings.seed = seed;
            for ratio in crate::settings::ASPECT_RATIOS {
                settings.aspect_ratio = ratio.into();
                let request = make_request("test", "negative", &[], &meta, &settings).unwrap();
                assert_eq!(request["seed"], seed);
                assert_eq!(request["sampling"]["steps"], 17);
                assert_eq!(request["sampling"]["cfg_scale"], 4.);
                assert_eq!(request["sampling"]["sampler"], "er_sde");
                assert_eq!(request["sampling"]["scheduler"], "beta");
                assert_eq!(request["model_options"], meta["defaults"]["model_options"]);
                let request: imagine_server::contract::JobRequest =
                    serde_json::from_value(request).unwrap();
                assert!(request.validate().is_ok());
            }
        }
        settings.steps = 0;
        assert!(make_request("test", "", &[], &meta, &settings).is_err());
        settings.steps = 40;
        settings.cfg_scale = f32::NAN;
        assert!(make_request("test", "", &[], &meta, &settings).is_err());
    }

    #[test]
    fn sampling_choices_follow_server_capabilities_and_old_preferences_inherit_defaults() {
        let meta = imagine_server::contract::model(MODEL).unwrap();
        let old: GenerationSettings =
            serde_json::from_str(r#"{"steps":40,"cfg_scale":2.5}"#).unwrap();
        let request = make_request("Synthetic", "", &[], &meta, &old).unwrap();
        assert_eq!(request["sampling"], meta["defaults"]["sampling"]);
        let settings = GenerationSettings {
            sampler: Some("euler".into()),
            ..old
        };
        assert!(make_request("Synthetic", "", &[], &meta, &settings).is_err());
        let library_meta = imagine_server::model_library::model_metadata(MODEL);
        assert!(make_request("Synthetic", "", &[], &library_meta, &settings).is_ok());
        let settings = GenerationSettings {
            scheduler: Some("unknown".into()),
            ..settings
        };
        assert!(make_request("Synthetic", "", &[], &library_meta, &settings).is_err());
    }
}

#[path = "video_client.rs"]
mod video;
pub use video::{VideoCompleted, i2v_request, video_request, video_request_with_selection};

#[cfg(test)]
mod persistent_image_tests {
    use super::*;
    #[test]
    fn persistent_paths_are_returned_without_copying_or_renaming_on_retry() {
        let result = json!({"output":{"persistent":true,"assets":[{"id":"own_asset","persistent":true,"saved_path":"/synthetic/mounted/2026-10-20/image_0001.png"}]}});
        let a =
            saved_image_paths(&result, &["own_asset"], Path::new("/synthetic/mounted")).unwrap();
        let b =
            saved_image_paths(&result, &["own_asset"], Path::new("/synthetic/mounted")).unwrap();
        assert_eq!(a, b);
        assert!(
            saved_image_paths(
                &result,
                &["other_owner_asset"],
                Path::new("/synthetic/mounted")
            )
            .is_err()
        );
        let mut bad = result.clone();
        bad["output"]["persistent"] = false.into();
        assert!(saved_image_paths(&bad, &["own_asset"], Path::new("/synthetic/mounted")).is_err());
        for path in [
            "/tmp/output.png",
            "/synthetic/mounted/../outside.png",
            "relative.png",
        ] {
            let mut bad = result.clone();
            bad["output"]["assets"][0]["saved_path"] = path.into();
            assert!(
                saved_image_paths(&bad, &["own_asset"], Path::new("/synthetic/mounted")).is_err()
            );
        }
    }
}
