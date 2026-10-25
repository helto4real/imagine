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
    fn network(error: reqwest::Error) -> Self {
        let (code, message) = if error.is_timeout() {
            (
                "network_timeout",
                "The video API request timed out. Retry to retrieve the result.",
            )
        } else if error.is_connect() {
            (
                "network_connect",
                "Could not connect to the video API. Check the server connection and retry.",
            )
        } else if error.is_body() {
            (
                "network_body",
                "The video API response was interrupted. Retry to retrieve the result.",
            )
        } else if error.is_request() {
            (
                "network_request",
                "The video API request was interrupted. Retry to retrieve the result.",
            )
        } else {
            (
                "network_error",
                "The video API connection was interrupted. Retry to retrieve the result.",
            )
        };
        Self {
            code: code.into(),
            message: message.into(),
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
#[derive(Debug)]
pub struct TimelineSession {
    pub id: String,
    pub private_storage: bool,
}
impl From<String> for TimelineSession {
    fn from(id: String) -> Self {
        Self {
            id,
            private_storage: false,
        }
    }
}
impl From<&str> for TimelineSession {
    fn from(id: &str) -> Self {
        id.to_owned().into()
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
    pub async fn timeline_session(&self) -> std::result::Result<TimelineSession, TimelineFailure> {
        let value = self
            .timeline_request("", false, "/v1/session", None)
            .await?;
        Ok(TimelineSession {
            id: identifier(&value["id"]).map_err(|_| TimelineFailure::invalid())?,
            private_storage: value["encrypted"].as_bool().unwrap_or(false),
        })
    }
    pub async fn timeline_request(
        &self,
        session: &str,
        post: bool,
        path: &str,
        body: Option<&Value>,
    ) -> std::result::Result<Value, TimelineFailure> {
        let mut request = if resource_io(post, path) {
            let http = media_http().map_err(TimelineFailure::network)?;
            http.request(
                if post {
                    reqwest::Method::POST
                } else {
                    reqwest::Method::GET
                },
                format!("{}{path}", self.base),
            )
        } else if post {
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
        self.upload_media_cancellable(
            session,
            path,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )
        .await
    }
    /// Local reference registration sends only the explicitly selected path.
    /// Original bytes stay in place; cancelling never binds a late response.
    pub async fn upload_media_cancellable(
        &self,
        session: &str,
        path: &Path,
        cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> std::result::Result<Value, TimelineFailure> {
        use std::sync::atomic::Ordering;
        let cancelled = || TimelineFailure {
            code: "media_import_cancelled".into(),
            message: "Media import cancelled.".into(),
            snapshot: None,
            retryable: false,
        };
        if cancel.load(Ordering::Acquire) {
            return Err(cancelled());
        }
        let path = std::path::absolute(path).map_err(|_| TimelineFailure::invalid())?;
        let http = media_http().map_err(TimelineFailure::network)?;
        let request = async {
            let response = http
                .post(format!("{}/v1/video/assets/link", self.base))
                .header("X-Imagine-Session", session)
                .json(&json!({"path":path}))
                .send()
                .await;
            timeline_json(response).await
        };
        tokio::select! {
            biased;
            _=async {while !cancel.load(Ordering::Acquire){tokio::time::sleep(Duration::from_millis(20)).await;}}=>Err(cancelled()),
            result=request=>result,
        }
    }
    pub async fn timeline_media(
        &self,
        session: &str,
        asset: &str,
    ) -> std::result::Result<Vec<u8>, TimelineFailure> {
        identifier(&json!(asset)).map_err(|_| TimelineFailure::invalid())?;
        self.timeline_bytes_path(session, &format!("/v1/assets/{asset}"))
            .await
    }
    pub async fn timeline_bytes_path(
        &self,
        session: &str,
        path: &str,
    ) -> std::result::Result<Vec<u8>, TimelineFailure> {
        let mut response = self
            .get(path)
            .header("X-Imagine-Session", session)
            .timeout(Duration::from_secs(180))
            .send()
            .await
            .map_err(TimelineFailure::network)?;
        if !response.status().is_success() {
            return Err(TimelineFailure {
                code: "media_unavailable".into(),
                message: "Media unavailable. Check media or reopen the project.".into(),
                snapshot: None,
                retryable: false,
            });
        }
        let mut bytes = vec![];
        while let Some(chunk) = response.chunk().await.map_err(TimelineFailure::network)? {
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
    let response = response.map_err(TimelineFailure::network)?;
    let status = response.status();
    let value: Value = response.json().await.map_err(|error| {
        if error.is_timeout() || error.is_body() || error.is_request() {
            TimelineFailure::network(error)
        } else {
            TimelineFailure::invalid()
        }
    })?;
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
        "source_missing" | "source_changed" | "source_unavailable" | "invalid_source_path" | "invalid_source_file" => return error_message(code),
        "invalid_character" => "The library character needs a name and valid character metadata.",
        "character_resource_missing" => "A selected character image or voice is unavailable. Reselect that media.",
        "character_media_type_mismatch" => "Select an image for an image field or audio for a voice field.",
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
        "media_import_storage_full" => "Not enough free space in the media destination. Free space and retry.",
        "media_import_write_failed" => "Could not write the media. Check the destination storage and retry.",
        "media_import_cancelled" => "Media import cancelled.",
        "upload_timeout" | "upload_interrupted" => "The media transfer was interrupted. Retry the import.",
        "upload_length_mismatch" => "The source changed or the transfer was incomplete. Choose the media again.",
        "invalid_media_size" => "Choose a non-empty media file.",
        "media_size_unrepresentable" => "The selected file exceeds the project format's size range.",
        "invalid_media" | "media_type_mismatch" | "unsupported_media_type" => {
            "The selected media could not be decoded."
        }
        code if code.starts_with("older_format_") || code.starts_with("assembly_") || code.starts_with("story_image_") || code=="story_dialogue_cast_required" || code=="cast_reference_missing" || code=="h3_section_reference_unavailable"
            || code.starts_with("video_destination_")
            || code.starts_with("preset_")
            || matches!(code, "media_decode_failed" | "output_save_failed" | "older_format_seed_out_of_range" | "imported_scene_not_generatable" | "take_requires_video" | "take_generation_recipe_unknown" | "older_format_imported_take_not_representable" | "capture_source_missing" | "h3_model_file_missing") => return error_message(code),
        _ => "The video operation failed. Refresh and retry.",
    }
    .to_owned()
}
// These operations verify/copy owned media and can legitimately exceed a wall-clock
// request deadline. Ordinary polling/status calls retain the client's default timeout.
fn resource_io(post: bool, path: &str) -> bool {
    if !post {
        return path.starts_with("/v1/video/assets/");
    }
    if !path.starts_with("/v1/video/") {
        return false;
    }
    if path.ends_with("/captures/import") {
        return true;
    }
    matches!(
        path.rsplit('/').next(),
        Some(
            "open"
                | "assets"
                | "save"
                | "save-as"
                | "export-compatible"
                | "health"
                | "relink-preview"
                | "relink"
                | "captures"
                | "media-copy"
                | "libraries"
        )
    )
}
fn media_http() -> Result<reqwest::Client, reqwest::Error> {
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(30))
        .build()
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
            "older_format_seed_out_of_range",
            "older_format_boundary_not_representable",
            "h3_model_file_missing",
        ] {
            assert_eq!(timeline_error_message(code), error_message(code));
        }
        assert!(
            timeline_error_message("assembly_transition_take_required")
                .contains("join the existing Takes with an explicit warning")
        );
        assert!(timeline_error_message("revision_conflict").contains("project changed"));
    }
}

#[cfg(test)]
mod media_transport_tests {
    use super::*;
    #[test]
    fn streaming_resource_io_has_no_arbitrary_total_deadline_but_polling_does() {
        for path in [
            "/v1/video/projects/open",
            "/v1/video/projects/p/assets",
            "/v1/video/projects/p/media/relink-preview",
            "/v1/video/projects/p/media/relink",
            "/v1/video/projects/p/media/health",
            "/v1/video/projects/p/save",
            "/v1/video/projects/p/captures",
            "/v1/video/projects/p/captures/import",
        ] {
            assert!(resource_io(true, path));
        }
        assert!(resource_io(false, "/v1/video/assets/a"));
        assert!(!resource_io(false, "/v1/video/projects/p"));
        assert!(!resource_io(false, "/v1/jobs/j"));
        assert!(!resource_io(true, "/v1/jobs"));
        assert!(!resource_io(true, "/v1/video/projects/p/runs"));
        assert!(timeline_error_message("media_import_storage_full").contains("Free space"));
        assert!(
            timeline_error_message("upload_length_mismatch").contains("Choose the media again")
        );
    }
}
