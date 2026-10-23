//! Merge as a child of video::api to retain its existing session authorization boundary.
use super::super::media_health::{self, Relink};
use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
pub(super) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/video/projects/{id}/media/health", post(check))
        .route(
            "/v1/video/projects/{id}/media/relink-preview",
            post(preview),
        )
        .route("/v1/video/projects/{id}/media/relink", post(relink))
}
struct Cancel(Arc<AtomicBool>);
impl Drop for Cancel {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Check {
    schema_version: u32,
    expected_project_revision: u64,
}
async fn check(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Check>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    schema(request.schema_version)?;
    let captured = {
        let service = app.video.lock().await;
        media_health::Check::capture(
            service.projects.session(&ProjectId(id))?,
            request.expected_project_revision,
        )?
    };
    let flag = Arc::new(AtomicBool::new(false));
    let _cancel = Cancel(flag.clone());
    let report = tokio::task::spawn_blocking(move || captured.run(&flag))
        .await
        .map_err(internal)??;
    Ok(Json(json!(report)))
}
async fn replacement(app: &App, request: &Relink) -> Result<(std::fs::File, media::Info), Error> {
    schema(request.schema_version)?;
    let descriptor = app
        .store
        .asset(&request.replacement_asset_id, "local")
        .await
        .map_err(internal)?
        .ok_or_else(|| Error::new(StatusCode::NOT_FOUND, "asset_not_found"))?;
    let staged = staging(app)?;
    let mut file = if let Some(file) = app
        .store
        .open_media(&request.replacement_asset_id)
        .map_err(internal)?
    {
        file
    } else {
        let bytes = app
            .store
            .upload_bytes(&request.replacement_asset_id)
            .map_err(internal)?
            .ok_or_else(|| media_error("asset_not_found"))?;
        let mut file = staged.temporary()?;
        file.write_all(&bytes).map_err(internal)?;
        file
    };
    let kind = match descriptor["mime_type"]
        .as_str()
        .unwrap_or("")
        .split('/')
        .next()
    {
        Some("image") => MediaKind::Image,
        Some("video") => MediaKind::Video,
        Some("audio") => MediaKind::Audio,
        _ => return Err(media_error("unsupported_media_type")),
    };
    let flag = Arc::new(AtomicBool::new(false));
    let _cancel = Cancel(flag.clone());
    let (returned, info) = tokio::task::spawn_blocking(move || {
        media_health::validate(&file, &kind, &flag).map(|info| (file, info))
    })
    .await
    .map_err(internal)??;
    file = returned;
    let resource = Resource::inspect(
        "check_replacement".into(),
        ResourceKind::Reference {
            asset_id: request.asset_id.clone(),
            revision: request.expected_asset_revision,
        },
        &mut file,
    )?;
    if Some(resource.bytes) != descriptor["bytes"].as_u64()
        || Some(resource.sha256.as_str()) != descriptor["sha256"].as_str()
    {
        return Err(media_error("asset_content_changed"));
    }
    Ok((file, info))
}
async fn preview(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Relink>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    let (mut file, info) = replacement(&app, &request).await?;
    let service = app.video.lock().await;
    let source = app
        .store
        .original_path(&request.replacement_asset_id)
        .map_err(internal)?;
    let (_, _, impacts) = media_health::prepare_relink_source(
        service.projects.session(&ProjectId(id))?,
        &request,
        &mut file,
        &info.kind,
        source.as_deref(),
    )?;
    Ok(Json(
        json!({"schema_version":1,"request":request,"impacts":impacts}),
    ))
}
async fn relink(
    State(app): State<App>,
    Path(id): Path<String>,
    body: Result<Json<Relink>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let request = input(body)?;
    let (file, info) = replacement(&app, &request).await?;
    let staged = staging(&app)?;
    let id = ProjectId(id);
    let mut service = app.video.lock().await;
    let source = app
        .store
        .original_path(&request.replacement_asset_id)
        .map_err(internal)?;
    let result = media_health::apply_relink_source(
        service.projects.session_mut(&id)?,
        &request,
        file,
        &info.kind,
        &staged,
        source,
    );
    let impacts = match result {
        Ok(impacts) => impacts,
        Err(error) => {
            let mut error = Error::from(error);
            error.snapshot = service.snapshot(&id).ok().map(Box::new);
            return Err(error);
        }
    };
    service
        .runtime_assets
        .entry(id.clone())
        .or_default()
        .insert(request.asset_id, request.replacement_asset_id);
    Ok(Json(
        json!({"snapshot":service.snapshot(&id)?,"impacts":impacts}),
    ))
}
