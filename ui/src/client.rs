use crate::settings::{Connection, GenerationSettings, check_output_folder, save_image};
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
    token: String,
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
    pub request: Value,
    pub key: String,
    pub folder: PathBuf,
    pub job: Option<String>,
}
pub struct Completed {
    pub file: PathBuf,
    pub preview: Preview,
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
    Progress::Status {
        label: label.into(),
        sampling,
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
            token: c.token()?,
        })
    }
    fn get(&self, path: &str) -> reqwest::RequestBuilder {
        self.http
            .get(format!("{}{path}", self.base))
            .bearer_auth(&self.token)
    }
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        self.http
            .post(format!("{}{path}", self.base))
            .bearer_auth(&self.token)
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
        check_output_folder(&submission.folder).map_err(|e| Failure::retry(e.to_string()))?;
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
        let asset = identifier(&result["output"]["assets"][0]["id"])?;
        let response = self
            .get(&format!("/v1/assets/{asset}"))
            .send()
            .await
            .map_err(|_| {
                Failure::retry("The image is ready but could not be downloaded. Retry.")
            })?;
        if !response.status().is_success() {
            return Err(Failure::retry(
                "Could not download the image. Check the connection and retry.",
            ));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|_| Failure::retry("The image download was interrupted. Retry."))?;
        let preview = decode(&bytes, false)
            .map_err(|_| Failure::retry("Could not decode the image from the server. Retry."))?;
        let file =
            save_image(&submission.folder, &bytes).map_err(|e| Failure::retry(e.to_string()))?;
        Ok(Completed {
            file,
            preview,
            seed: result["effective"]["seed"].as_u64(),
        })
    }
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
    if refs.len() > 10 {
        bail!("Up to ten reference images.");
    }
    let defaults = metadata
        .get("defaults")
        .and_then(Value::as_object)
        .context("The server did not provide default settings.")?;
    let mut request = Value::Object(defaults.clone());
    request["model"] = json!(MODEL);
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
    request["model_options"]["qwen_image_2_1"]["reference_max_pixels_per_image"] =
        json!(settings.reference_max_pixels_per_image);
    request["seed"] = if settings.random_seed {
        Value::Null
    } else {
        json!(settings.seed)
    };
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
        "coordinator_unavailable" => "The GPU queue is unavailable. Check that Imagine and the proxy use the same coordinator socket." ,
        "imagine_proxy_not_configured" => "Restart the proxy with ./start-proxy.sh to enable Imagine coordination.",
        "imagine_upstream_unavailable" => "The proxy cannot reach Imagine. Start ./start-server.sh and retry.",
        "unsupported_resolution"=>"This server does not accept the selected image size. Restart the updated server or choose a supported size.",
        "unsupported_sampling"=>"Choose steps between 1 and 100 and CFG between 0 and 20.",
        "unauthorized"=>"The API key was rejected. Check access.json and reconnect.",
        "runtime_suspended"=>"The server is suspended. Resume it through the API before generating.",
        "model_weights_missing"=>"Model files are missing. Check QWEN_MODEL_ROOT on the server.",
        "frozen_binary_unavailable"|"frozen_binary_hash_mismatch"=>"The frozen inference engine is missing or has changed on the server.",
        "native_startup_failed"=>"The inference engine could not start. Check the GPU and model files on the server.",
        "asset_not_found"=>"A reference is missing on the server. Remove it and upload it again.",
        "invalid_image"=>"The image is unreadable. Choose PNG, JPEG or WebP within the server size limits.",
        "upload_busy"=>"The server is handling other uploads. Retry.",
        "invalid_prompt_length"=>"The prompt must contain 1–32,768 UTF-8 bytes.",
        "generation_failed"|"native_generation_failed"=>"Generation failed. Check available GPU memory and try a smaller image or fewer references.",
        "job_not_found"=>"The job is missing on the server. Start a new generation.",
        _=>"The server could not complete the request. Check the connection and settings."
    }.into()
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
