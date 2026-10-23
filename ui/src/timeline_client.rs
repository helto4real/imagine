//! Session-bound project API; private document snapshots never enter job progress events.
use super::*;
#[derive(Debug)]
pub struct TimelineFailure {
    pub code: String,
    pub message: String,
    pub snapshot: Option<Value>,
    pub retryable: bool,
}
impl TimelineFailure {
    fn network() -> Self {
        Self {
            code: "network_error".into(),
            message: "Could not reach the video API. Retry.".into(),
            snapshot: None,
            retryable: true,
        }
    }
    fn invalid() -> Self {
        Self {
            code: "invalid_response".into(),
            message: "The video API returned an invalid response.".into(),
            snapshot: None,
            retryable: true,
        }
    }
}
impl Client {
    pub async fn timeline_start_run(
        &self,
        session: &str,
        body: &Value,
        key: &str,
    ) -> std::result::Result<Value, TimelineFailure> {
        timeline_json(
            self.post("/v1/video/runs")
                .header("X-Imagine-Session", session)
                .header("Idempotency-Key", key)
                .json(body)
                .send()
                .await,
        )
        .await
    }
    pub async fn timeline_session(&self) -> std::result::Result<String, TimelineFailure> {
        let value = self
            .timeline_request("", false, "/v1/session", None)
            .await?;
        identifier(&value["id"]).map_err(|_| TimelineFailure::invalid())
    }
    pub async fn timeline_request(
        &self,
        session: &str,
        post: bool,
        path: &str,
        body: Option<&Value>,
    ) -> std::result::Result<Value, TimelineFailure> {
        let mut request = if post {
            self.post(path)
        } else {
            self.get(path)
        };
        if !session.is_empty() {
            request = request.header("X-Imagine-Session", session);
        }
        if let Some(body) = body {
            request = request.json(body);
        }
        timeline_json(request.send().await).await
    }
    pub async fn timeline_submit(
        &self,
        session: &str,
        body: &Value,
        key: &str,
    ) -> std::result::Result<Value, TimelineFailure> {
        let path = if body["kind"] == "video_assembly" {
            format!(
                "/v1/video/assemblies/{}/jobs",
                identifier(&body["assembly_plan"]["id"]).map_err(|_| TimelineFailure::invalid())?
            )
        } else {
            "/v1/jobs".into()
        };
        timeline_json(
            self.post(&path)
                .header("X-Imagine-Session", session)
                .header("Idempotency-Key", key)
                .json(body)
                .send()
                .await,
        )
        .await
    }
    pub async fn upload_media(
        &self,
        session: &str,
        path: &Path,
    ) -> std::result::Result<Value, TimelineFailure> {
        let mime = match path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str()
        {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "webp" => "image/webp",
            "mp4" | "mov" => "video/mp4",
            "mkv" => "video/x-matroska",
            "webm" => "video/webm",
            "wav" => "audio/wav",
            "flac" => "audio/flac",
            "mp3" => "audio/mpeg",
            "ogg" => "audio/ogg",
            _ => {
                return Err(TimelineFailure {
                    code: "unsupported_media_type".into(),
                    message: "Choose an image, video or audio file.".into(),
                    snapshot: None,
                    retryable: false,
                });
            }
        };
        let source = tokio::fs::File::open(path)
            .await
            .map_err(|_| TimelineFailure {
                code: "media_unavailable".into(),
                message: "Could not open the selected file.".into(),
                snapshot: None,
                retryable: false,
            })?;
        let size = source
            .metadata()
            .await
            .map_err(|_| TimelineFailure::invalid())?
            .len();
        if size == 0 || size > 2 * 1024 * 1024 * 1024 {
            return Err(TimelineFailure {
                code: "invalid_media_size".into(),
                message: "Choose a non-empty file smaller than 2 GiB.".into(),
                snapshot: None,
                retryable: false,
            });
        }
        timeline_json(
            self.post("/v1/video/assets")
                .header("X-Imagine-Session", session)
                .header("Content-Type", mime)
                .header("Content-Length", size)
                .timeout(Duration::from_secs(180))
                .body(reqwest::Body::wrap_stream(
                    tokio_util::io::ReaderStream::new(source),
                ))
                .send()
                .await,
        )
        .await
    }
    pub async fn timeline_media(
        &self,
        session: &str,
        asset: &str,
    ) -> std::result::Result<Vec<u8>, TimelineFailure> {
        identifier(&json!(asset)).map_err(|_| TimelineFailure::invalid())?;
        let mut response = self
            .get(&format!("/v1/assets/{asset}"))
            .header("X-Imagine-Session", session)
            .timeout(Duration::from_secs(180))
            .send()
            .await
            .map_err(|_| TimelineFailure::network())?;
        if !response.status().is_success() {
            return Err(TimelineFailure {
                code: "media_unavailable".into(),
                message: "Media unavailable. Check media or reopen the project.".into(),
                snapshot: None,
                retryable: false,
            });
        }
        let mut bytes = vec![];
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| TimelineFailure::network())?
        {
            if bytes.len() + chunk.len() > 2 * 1024 * 1024 * 1024 {
                return Err(TimelineFailure::invalid());
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }
}
async fn timeline_json(
    response: Result<reqwest::Response, reqwest::Error>,
) -> std::result::Result<Value, TimelineFailure> {
    let response = response.map_err(|_| TimelineFailure::network())?;
    let status = response.status();
    let value: Value = response
        .json()
        .await
        .map_err(|_| TimelineFailure::invalid())?;
    if status.is_success() {
        return Ok(value);
    }
    let code = value["error"]["code"]
        .as_str()
        .unwrap_or("invalid_response")
        .to_owned();
    let message = timeline_error_message(&code);
    Err(TimelineFailure {
        code,
        message,
        snapshot: value.get("snapshot").cloned(),
        retryable: status.is_server_error(),
    })
}
fn timeline_error_message(code: &str) -> String {
    match code {
        "revision_conflict" => "The project changed. Refresh before applying this edit.",
        "authoring_revision_conflict"=>"The project changed. The proposal is retained; start a new proposal before applying.",
        "authoring_visual_context_unavailable"=>"Image-grounded enhancement is not connected yet. Edit this scene manually.",
        "authoring_video_context_unavailable"=>"Video-grounded enhancement is not connected to this provider path yet. The source binding is retained.",
        "authoring_busy"=>"Another proposal is generating. Wait or cancel that proposal.",
        "story_scene_duration_capacity"=>"Add more scenes or reduce Story duration. Each Story scene can be at most 15 seconds.",
        "authoring_not_editable"=>"Wait for generation to finish or cancel it before editing this proposal.",
        "session_changed" | "session_required" => {
            "The server session changed. Open the project again."
        }
        "storage_root_changed" => {
            "The selected project folder changed or is unavailable. Restore it and retry Save."
        }
        "storage_conflict" => {
            "The project file changed outside this session. Save a copy or reopen it."
        }
        "storage_io_failed" | "storage_error" => {
            "Could not save the project. Check the destination and retry Save."
        }
        "resource_integrity_failed" | "asset_content_changed" => {
            "Media changed. Use Check media before continuing."
        }
        "scene_runtime_unsupported" => {
            "This scene requires H3 capabilities that are not available yet."
        }
        "unsupported_h3_sampling" => "The selected H3 sampling settings cannot run in this worker.",
        "h3_trained_upscaler_unavailable" => {
            "The trained H3 latent upscaler is not connected to this worker yet."
        }
        "project_already_open" => "This project is already open in this session.",
        "continuity_method_unbound" => "The selected continuity method is not connected to this worker yet.",
        "continuity_source_not_accepted" => "Accept a Take in the source scene before continuing.",
        "continuity_endpoint_missing_or_ambiguous" => "The source Take has no unique endpoint artifact. Generate and accept a new source Take.",
        "continuity_endpoint_incompatible" | "invalid_continuity_endpoint" => "The source endpoint is incompatible or damaged. Generate a new source Take.",
        "continuity_first_frame_conflict" => "Choose either the continuity endpoint or an explicit First Frame binding.",
        "invalid_media" | "media_type_mismatch" | "unsupported_media_type" => {
            "The selected media could not be decoded."
        }
        code if code.starts_with("assembly_") || code.starts_with("story_image_") || code=="story_dialogue_cast_required" || code=="cast_reference_missing" || code=="h3_section_reference_unavailable"
            || code.starts_with("video_destination_")
            || matches!(code, "media_decode_failed" | "output_save_failed") => return error_message(code),
        _ => "The video operation failed. Refresh and retry.",
    }
    .to_owned()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn montage_dialog_preserves_specific_shared_errors() {
        for code in [
            "assembly_transition_take_required",
            "assembly_empty_source_range",
            "output_save_failed",
            "video_destination_write_failed",
        ] {
            assert_eq!(timeline_error_message(code), error_message(code));
        }
        assert!(
            timeline_error_message("assembly_transition_take_required").contains("transition Take")
        );
        assert!(timeline_error_message("revision_conflict").contains("project changed"));
    }
}
