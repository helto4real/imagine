use crate::settings::{
    Connection, GenerationSettings, UpscaleTarget, check_output_folder, save_image, save_sheet,
    save_upscaled,
};
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::{
    io::{Cursor, Read},
    path::{Path, PathBuf},
    time::Duration,
};

pub const MODEL: &str = "qwen-image-2.1-int8";
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
    /// Local naming context only; never included in the API request.
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
fn job_progress(value: &Value) -> Progress {
    let status = value["status"].as_str().unwrap_or("");
    let sampling = if matches!(status, "running" | "cancelling") {
        serde_json::from_value::<SamplingProgress>(value["progress"].clone())
            .ok()
            .filter(|p| p.total > 0 && p.total <= 100 && p.completed <= p.total)
    } else {
        None
    };
    let label = match status {
        "queued" => "Queued",
        "cancelling" => "Cancelling",
        _ => match value["phase"].as_str() {
            Some("verifying_profile") => "Verifying model",
            Some("loading_model") => "Loading model",
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
            let code = body["error"]["code"].as_str().unwrap_or("unknown");
            return Err(Failure {
                message: error_message(code),
                retryable: status.is_server_error(),
            });
        }
        Ok(body)
    }
    pub async fn upload(&self, path: &Path) -> std::result::Result<Reference, Failure> {
        let (bytes, mime, preview) =
            read_reference(path).map_err(|e| Failure::terminal(e.to_string()))?;
        let value = Self::json(
            self.post("/v1/assets")
                .header("Content-Type", mime)
                .body(bytes)
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
    pub async fn generate(
        &self,
        submission: Submission,
        mut progress: impl FnMut(Progress),
    ) -> std::result::Result<Completed, Failure> {
        let destination = submission
            .upscale_source
            .as_deref()
            .and_then(Path::parent)
            .unwrap_or(&submission.folder);
        check_output_folder(destination).map_err(|e| {
            Failure::retry(if submission.upscale_source.is_some() {
                "Cannot write beside the original image. Restore access to that folder and retry."
                    .into()
            } else {
                e.to_string()
            })
        })?;
        let job = match submission.job {
            Some(id) => id,
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
        progress(Progress::Submitted(job.clone()));
        let result = loop {
            let value = Self::json(self.get(&format!("/v1/jobs/{job}")).send().await).await?;
            match value["status"].as_str() {
                Some("succeeded") => break value,
                Some("failed") => {
                    return Err(Failure::terminal(error_message(
                        value["error"]["code"].as_str().unwrap_or("unknown"),
                    )));
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
        progress(Progress::Status {
            label: "Saving".into(),
            sampling: None,
        });
        let roles: &[OutputRole] = if submission.request["operation"] == "character_sheet" {
            &[OutputRole::Sheet, OutputRole::Portrait, OutputRole::Views]
        } else {
            &[OutputRole::Image]
        };
        let ids = result_assets(&result, roles)?;
        // Download and decode the complete set before touching the output folder.
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
        let files = if roles == [OutputRole::Image] {
            match &submission.upscale_source {
                Some(source) => save_upscaled(source, &downloads[0].1),
                None => save_image(&submission.folder, &downloads[0].1),
            }
            .map(|file| vec![file])
        } else {
            save_sheet(
                &submission.folder,
                &downloads
                    .iter()
                    .map(|(role, bytes, _, _)| (role.name(), bytes.as_ref()))
                    .collect::<Vec<_>>(),
            )
        }
        .map_err(|e| Failure::retry(e.to_string()))?;
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
    if character.trim().is_empty() && refs.is_empty() {
        bail!("Describe the character or add a reference image.");
    }
    if character.len() > 12000 || clothing.len() > 12000 {
        bail!("Each description must be at most 12,000 UTF-8 bytes.");
    }
    if refs.len() > 8 {
        bail!("Up to eight identity references.");
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
    if metadata["model"] == "moss-image-sft-int8" && !refs.is_empty() {
        bail!("MOSS currently supports text-to-image only. Remove reference images.");
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
        && settings.steps as usize + 1 != sigmas.len()
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
fn decode(bytes: &[u8], thumbnail: bool) -> Result<Preview> {
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
pub fn error_message(code: &str) -> String {
    match code {
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
        "unsupported_sampling"=>"Choose a sampler and scheduler supported by this backend, steps between 1 and 100, and CFG between 0 and 20.",
        "unsupported_moss_configuration"=>"MOSS currently supports text-to-image with Euler and normal or sgm_uniform, without references or LoRAs.",
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
        _=>"The server could not complete the request. Check the connection and settings."
    }.into()
}

#[cfg(test)]
mod tests {
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
